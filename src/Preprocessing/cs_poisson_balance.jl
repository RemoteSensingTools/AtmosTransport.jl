# ---------------------------------------------------------------------------
# Global multi-panel Poisson mass-flux balance for cubed-sphere grids.
#
# Line-for-line port from the legacy preprocessing runner (git commit
# ec2d2c0, path scripts_legacy/preprocessing/cs_global_poisson_balance.jl)
# into the modern src/Preprocessing/ pipeline.
#
# Unlike the per-panel FFT approach (which treats each panel as doubly-
# periodic and ignores cross-panel continuity), this solver operates on
# a GLOBAL face table that includes all cross-panel boundary faces.
# It uses Jacobi-preconditioned CG on the global graph Laplacian.
#
# On a 6-panel CS with Nc cells per edge:
#   - 6 × Nc² total cells
#   - 12 × Nc² total faces (degree = 4 everywhere on the closed sphere)
#   - Graph Laplacian L = D - A has a 1-D constant null space
#   - Solver uses mean-zero projection (same as the RG path)
#
# References:
#   - ring_poisson_balance.jl: reduced-Gaussian CG balance
#   - PanelConnectivity.jl: default_panel_connectivity()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Global cell and face indexing
# ---------------------------------------------------------------------------

"""
    _cs_global_cell(i, j, p, Nc) -> Int

Map panel-local cell `(i, j)` on panel `p` to a global cell index in `1:6Nc²`.
Column-major within each panel: cell `(i, j, p)` → `(p-1)*Nc² + (j-1)*Nc + i`.
"""
@inline _cs_global_cell(i::Int, j::Int, p::Int, Nc::Int) =
    (p - 1) * Nc * Nc + (j - 1) * Nc + i

"""
    CSGlobalFaceTable

Global face table for a 6-panel cubed sphere.

# Fields
- `face_left  :: Vector{Int32}` — global cell index on the "left" side of each face
- `face_right :: Vector{Int32}` — global cell index on the "right" side of each face
- `nf :: Int` — total number of faces (= 12 × Nc²)
- `nc :: Int` — total number of cells (= 6 × Nc²)
- `Nc :: Int` — cells per panel edge

## Per-face back-mapping to per-panel am/bm arrays

- `face_panel :: Vector{Int32}` — which panel owns this face (1-6)
- `face_dir   :: Vector{Int32}` — direction: 1 = x-face (am), 2 = y-face (bm)
- `face_idx_i :: Vector{Int32}` — the `i` index into `am[i, j, k]` or `bm[i, j, k]`
- `face_idx_j :: Vector{Int32}` — the `j` index

For cross-panel faces, there's also a **mirror** entry on the neighbor panel:

- `mirror_panel :: Vector{Int32}` — 0 for interior faces; neighbor panel for cross-panel
- `mirror_dir   :: Vector{Int32}` — direction of the mirror entry
- `mirror_idx_i :: Vector{Int32}` — the `i` index of the mirror entry
- `mirror_idx_j :: Vector{Int32}` — the `j` index of the mirror entry

Sign convention: flux positive → mass flows from `face_left` to `face_right`.
"""
struct CSGlobalFaceTable
    face_left    :: Vector{Int32}
    face_right   :: Vector{Int32}
    face_panel   :: Vector{Int32}
    face_dir     :: Vector{Int32}
    face_idx_i   :: Vector{Int32}
    face_idx_j   :: Vector{Int32}
    mirror_panel :: Vector{Int32}
    mirror_dir   :: Vector{Int32}
    mirror_idx_i :: Vector{Int32}
    mirror_idx_j :: Vector{Int32}
    mirror_sign  :: Vector{Int32}   # +1 or -1: sign flip when writing mirror entry
    nf :: Int
    nc :: Int
    Nc :: Int
end

"""
    _cs_edge_cell(edge, s, Nc) -> (i, j)

Return the cell indices `(i, j)` at position `s` along edge `edge`.
Edge 1=north, 2=south, 3=east, 4=west.
"""
@inline function _cs_edge_cell(edge::Int, s::Int, Nc::Int)
    if edge == EDGE_NORTH       # j = Nc, i = s
        return (s, Nc)
    elseif edge == EDGE_SOUTH   # j = 1, i = s
        return (s, 1)
    elseif edge == EDGE_EAST    # i = Nc, j = s
        return (Nc, s)
    else                        # i = 1, j = s
        return (1, s)
    end
end

"""
    _cs_edge_face_location(edge, s, Nc) -> (dir, i, j)

Return the per-panel am/bm index for the boundary face at position `s` along
edge `edge`.

| Edge    | Face array   | Index             |
|---------|-------------|-------------------|
| north   | bm[s, Nc+1] | dir=2, i=s, j=Nc+1 |
| south   | bm[s, 1]    | dir=2, i=s, j=1     |
| east    | am[Nc+1, s] | dir=1, i=Nc+1, j=s  |
| west    | am[1, s]    | dir=1, i=1, j=s      |
"""
@inline function _cs_edge_face_location(edge::Int, s::Int, Nc::Int)
    if edge == EDGE_NORTH       # north boundary
        return (2, s, Nc + 1)
    elseif edge == EDGE_SOUTH   # south boundary
        return (2, s, 1)
    elseif edge == EDGE_EAST    # east boundary
        return (1, Nc + 1, s)
    else                        # west boundary
        return (1, 1, s)
    end
end

"""
    build_cs_global_face_table(Nc, conn) -> CSGlobalFaceTable

Build the global face table for a C`Nc` cubed sphere with panel connectivity
`conn`. Enumerates all `12Nc²` unique faces (interior + cross-panel).

Cross-panel faces are created only from outgoing edges (north, east) to avoid
double-counting. The canonical entry is on the outgoing panel, the mirror on
the incoming panel.
"""
function build_cs_global_face_table(Nc::Int, conn::PanelConnectivity)
    max_nf = 12 * Nc^2
    fl = Vector{Int32}(undef, max_nf)
    fr = Vector{Int32}(undef, max_nf)
    fp = Vector{Int32}(undef, max_nf)
    fd = Vector{Int32}(undef, max_nf)
    fi = Vector{Int32}(undef, max_nf)
    fj = Vector{Int32}(undef, max_nf)
    mp = zeros(Int32, max_nf)
    md = zeros(Int32, max_nf)
    mi = zeros(Int32, max_nf)
    mj = zeros(Int32, max_nf)
    ms = ones(Int32, max_nf)    # mirror_sign: default +1

    nf = 0

    # --- Phase 1: Interior faces (within each panel) ---
    for p in 1:6
        # Interior x-faces: am[i, j] for i ∈ 2:Nc, j ∈ 1:Nc
        for j in 1:Nc, i in 2:Nc
            nf += 1
            fl[nf] = _cs_global_cell(i - 1, j, p, Nc)
            fr[nf] = _cs_global_cell(i,     j, p, Nc)
            fp[nf] = Int32(p)
            fd[nf] = Int32(1)   # x-face → am
            fi[nf] = Int32(i)
            fj[nf] = Int32(j)
        end
        # Interior y-faces: bm[i, j] for i ∈ 1:Nc, j ∈ 2:Nc
        for j in 2:Nc, i in 1:Nc
            nf += 1
            fl[nf] = _cs_global_cell(i, j - 1, p, Nc)
            fr[nf] = _cs_global_cell(i, j,     p, Nc)
            fp[nf] = Int32(p)
            fd[nf] = Int32(2)   # y-face → bm
            fi[nf] = Int32(i)
            fj[nf] = Int32(j)
        end
    end

    n_interior = nf

    # --- Phase 2: Cross-panel faces ---
    # Iterate ALL 4 edges of each panel.  De-duplicate by keeping only the
    # (p, e) side where p < q (the lower-numbered panel is canonical).
    # This correctly enumerates all 12 physical edges regardless of which
    # combination of NORTH/SOUTH/EAST/WEST they land on.
    for p in 1:6
        for e in (EDGE_NORTH, EDGE_SOUTH, EDGE_EAST, EDGE_WEST)
            q   = conn.neighbors[p][e].panel
            ori = conn.neighbors[p][e].orientation
            eq  = reciprocal_edge(conn, p, e)

            # De-duplicate: each physical edge is shared by (p, q).
            # Keep only the p < q side.
            p > q && continue

            for s in 1:Nc
                t = ori == 0 ? s : Nc + 1 - s

                cp = _cs_edge_cell(e, s, Nc)
                cq = _cs_edge_cell(eq, t, Nc)

                gp = _cs_global_cell(cp[1], cp[2], p, Nc)
                gq = _cs_global_cell(cq[1], cq[2], q, Nc)

                can_dir, can_i, can_j = _cs_edge_face_location(e, s, Nc)
                mir_dir, mir_i, mir_j = _cs_edge_face_location(eq, t, Nc)

                # Face-table convention: fl = cell on the lower-index side of
                # the face, fr = cell on the higher-index side.
                # For NORTH/EAST canonicals (outflow at Nc+1): fl = P-cell (inside),
                #   fr = Q-cell (outside).
                # For SOUTH/WEST canonicals (inflow at 1): fl = Q-cell (outside),
                #   fr = P-cell (inside) — swap because the P-cell is on the
                #   higher-index side of the boundary face.
                can_at_outflow = (can_dir == 1 && can_i == Nc + 1) ||
                                 (can_dir == 2 && can_j == Nc + 1)

                nf += 1
                if can_at_outflow
                    fl[nf] = gp
                    fr[nf] = gq
                else
                    fl[nf] = gq
                    fr[nf] = gp
                end
                fp[nf] = Int32(p)
                fd[nf] = Int32(can_dir)
                fi[nf] = Int32(can_i)
                fj[nf] = Int32(can_j)
                mp[nf] = Int32(q)
                md[nf] = Int32(mir_dir)
                mi[nf] = Int32(mir_i)
                mj[nf] = Int32(mir_j)

                # Mirror sign: determines whether the mirror value must be
                # negated so that per-panel flux telescoping conserves mass.
                #
                # The advection kernel computes:
                #   m_new = m + flux_in(j=1 or i=1) - flux_out(j=Nc+1 or i=Nc+1)
                # For global mass conservation, the canonical and mirror must
                # cancel.  When both are at the SAME position type (both inflow
                # or both outflow), negate; when at OPPOSITE types, keep as-is.
                mir_at_outflow = (mir_dir == 1 && mir_i == Nc + 1) ||
                                 (mir_dir == 2 && mir_j == Nc + 1)
                ms[nf] = (can_at_outflow == mir_at_outflow) ? Int32(-1) : Int32(1)
            end
        end
    end

    n_cross = nf - n_interior
    nc = 6 * Nc^2

    @assert nf == 12 * Nc^2 "Expected $(12*Nc^2) faces, got $nf " *
        "(interior=$n_interior, cross=$n_cross)"

    return CSGlobalFaceTable(
        fl[1:nf], fr[1:nf], fp[1:nf], fd[1:nf], fi[1:nf], fj[1:nf],
        mp[1:nf], md[1:nf], mi[1:nf], mj[1:nf], ms[1:nf],
        nf, nc, Nc,
    )
end

# ---------------------------------------------------------------------------
# Graph Laplacian and CG solver
# ---------------------------------------------------------------------------

"""
    cs_cell_face_degree(ft::CSGlobalFaceTable) -> Vector{Int}

Compute face degree for each global cell. On a closed CS, every cell has
degree 4.
"""
function cs_cell_face_degree(ft::CSGlobalFaceTable)
    degree = zeros(Int, ft.nc)
    @inbounds for f in 1:ft.nf
        degree[ft.face_left[f]]  += 1
        degree[ft.face_right[f]] += 1
    end
    for c in 1:ft.nc
        degree[c] == 4 || @warn "Cell $c has degree $(degree[c]) (expected 4)"
    end
    return degree
end

"""
    CSPoissonScratch

Pre-allocated work buffers for the CS global Poisson balance.
"""
struct CSPoissonScratch
    psi :: Vector{Float64}
    rhs :: Vector{Float64}
    r   :: Vector{Float64}
    p   :: Vector{Float64}
    Ap  :: Vector{Float64}
    z   :: Vector{Float64}
    div :: Vector{Float64}
end

function CSPoissonScratch(nc::Int)
    return CSPoissonScratch(
        zeros(Float64, nc), zeros(Float64, nc),
        zeros(Float64, nc), zeros(Float64, nc),
        zeros(Float64, nc), zeros(Float64, nc),
        zeros(Float64, nc),
    )
end

"""
    _cs_graph_laplacian_mul!(out, psi, ft, degree)

Compute `out = L · psi` for the global CS graph Laplacian.
"""
function _cs_graph_laplacian_mul!(out::AbstractVector{Float64},
                                  psi::AbstractVector{Float64},
                                  ft::CSGlobalFaceTable,
                                  degree::Vector{Int})
    @inbounds for c in eachindex(out)
        out[c] = degree[c] * psi[c]
    end
    @inbounds for f in 1:ft.nf
        left  = Int(ft.face_left[f])
        right = Int(ft.face_right[f])
        out[left]  -= psi[right]
        out[right] -= psi[left]
    end
    return out
end

@inline function _cs_project_mean_zero!(v::AbstractVector{Float64})
    s = sum(v) / length(v)
    @. v -= s
    return v
end

@inline function _cs_dot(a::AbstractVector{Float64}, b::AbstractVector{Float64})
    s = 0.0
    @inbounds @simd for i in eachindex(a, b)
        s += a[i] * b[i]
    end
    return s
end

@inline function _cs_linf(a::AbstractVector{Float64})
    m = 0.0
    @inbounds @simd for i in eachindex(a)
        v = abs(a[i])
        m = ifelse(v > m, v, m)
    end
    return m
end

@inline function _cs_project_mean_zero_linf!(v::AbstractVector{Float64})
    s = 0.0
    @inbounds @simd for i in eachindex(v)
        s += v[i]
    end
    μ = s / length(v)
    m = 0.0
    @inbounds @simd for i in eachindex(v)
        vi = v[i] - μ
        v[i] = vi
        av = abs(vi)
        m = ifelse(av > m, av, m)
    end
    return m
end

@inline function _cs_update_psi_r_linf!(psi::AbstractVector{Float64},
                                        r::AbstractVector{Float64},
                                        p::AbstractVector{Float64},
                                        Ap::AbstractVector{Float64},
                                        alpha::Float64)
    m = 0.0
    @inbounds @simd for i in eachindex(r)
        psi[i] += alpha * p[i]
        ri = r[i] - alpha * Ap[i]
        r[i] = ri
        ar = abs(ri)
        m = ifelse(ar > m, ar, m)
    end
    return m
end

@inline function _cs_precondition!(z::AbstractVector{Float64},
                                  r::AbstractVector{Float64},
                                  degree::Vector{Int})
    @inbounds @simd for c in eachindex(r)
        z[c] = degree[c] > 0 ? r[c] / degree[c] : r[c]
    end
    return z
end

@inline function _cs_update_direction!(p::AbstractVector{Float64},
                                      z::AbstractVector{Float64},
                                      beta::Float64)
    @inbounds @simd for c in eachindex(p)
        p[c] = z[c] + beta * p[c]
    end
    return p
end

"""
    solve_cs_poisson_pcg!(psi, rhs, ft, degree, scratch; tol=1e-14, max_iter=20000,
                           project_every=50)

Jacobi-Preconditioned CG for the global CS graph Laplacian `L · psi = rhs`.

L has a 1-D constant null space (closed surface), so we project `rhs`
and iterates to mean-zero. The graph Laplacian preserves the mean-zero
subspace, so the residual/preconditioned residual projection is applied
periodically rather than every iteration by default; set `project_every=1`
to recover the fully projected legacy path. Returns
`(residual_linfty, iterations)`.
"""
function solve_cs_poisson_pcg!(psi::AbstractVector{Float64},
                                rhs::AbstractVector{Float64},
                                ft::CSGlobalFaceTable,
                                degree::Vector{Int},
                                scratch;
                                tol::Float64=1e-14,
                                max_iter::Int=20000,
                                project_every::Int=50)
    r  = scratch.r
    p  = scratch.p
    Ap = scratch.Ap
    z  = scratch.z

    _cs_project_mean_zero!(rhs)
    fill!(psi, 0.0)
    copyto!(r, rhs)

    # Jacobi preconditioner: z = M⁻¹ r, M = diag(degree)
    _cs_precondition!(z, r, degree)
    copyto!(p, z)

    rz_old = _cs_dot(r, z)
    rhs_linf = _cs_linf(rhs) + eps()

    iter = 0
    r_linf = rhs_linf
    while iter < max_iter
        _cs_graph_laplacian_mul!(Ap, p, ft, degree)
        pAp = _cs_dot(p, Ap)
        pAp <= 0 && break
        alpha = rz_old / pAp
        r_linf = _cs_update_psi_r_linf!(psi, r, p, Ap, alpha)

        do_project = project_every <= 1 ||
                     (project_every > 1 && (iter + 1) % project_every == 0)
        do_project && (r_linf = _cs_project_mean_zero_linf!(r))
        r_linf / rhs_linf < tol && break

        _cs_precondition!(z, r, degree)
        do_project && _cs_project_mean_zero!(z)
        rz_new = _cs_dot(r, z)
        rz_old == 0.0 && break
        beta = rz_new / rz_old
        _cs_update_direction!(p, z, beta)
        rz_old = rz_new
        iter += 1
    end

    _cs_project_mean_zero!(psi)
    return r_linf, iter
end

# ---------------------------------------------------------------------------
# Correction application: map global ψ to per-panel am/bm
# ---------------------------------------------------------------------------

"""
    apply_cs_flux_correction!(panels_am, panels_bm, psi, ft, k)

Apply the Poisson correction to per-panel `am` and `bm` arrays at level `k`.

For each face, the correction is `δflux = psi[face_right] - psi[face_left]`.
Cross-panel mirror entries are set equal to the corrected canonical value
(same sign — both entries describe "mass flows from p to q" as positive).
"""
function apply_cs_flux_correction!(panels_am::NTuple{6, Array{FT, 3}},
                                    panels_bm::NTuple{6, Array{FT, 3}},
                                    psi::AbstractVector{Float64},
                                    ft::CSGlobalFaceTable,
                                    k::Int) where FT
    @inbounds for f in 1:ft.nf
        left  = Int(ft.face_left[f])
        right = Int(ft.face_right[f])
        delta = psi[right] - psi[left]

        p   = Int(ft.face_panel[f])
        dir = Int(ft.face_dir[f])
        i   = Int(ft.face_idx_i[f])
        j   = Int(ft.face_idx_j[f])

        if dir == 1
            panels_am[p][i, j, k] += FT(delta)
        else
            panels_bm[p][i, j, k] += FT(delta)
        end

        mq = Int(ft.mirror_panel[f])
        mq == 0 && continue   # interior face

        mdir  = Int(ft.mirror_dir[f])
        mi    = Int(ft.mirror_idx_i[f])
        mj    = Int(ft.mirror_idx_j[f])
        msign = Int(ft.mirror_sign[f])

        canonical_val = dir == 1 ? panels_am[p][i, j, k] : panels_bm[p][i, j, k]
        mirror_val = FT(msign) * canonical_val
        if mdir == 1
            panels_am[mq][mi, mj, k] = mirror_val
        else
            panels_bm[mq][mi, mj, k] = mirror_val
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Mirror synchronization
# ---------------------------------------------------------------------------

"""
    _sync_cs_mirrors!(panels_am, panels_bm, ft, Nz)

Copy canonical boundary flux values to their cross-panel mirror entries
for all levels.
"""
function _sync_cs_mirrors!(panels_am::NTuple{6, Array{FT, 3}},
                            panels_bm::NTuple{6, Array{FT, 3}},
                            ft::CSGlobalFaceTable,
                            Nz::Int) where FT
    @inbounds for f in 1:ft.nf
        mq = Int(ft.mirror_panel[f])
        mq == 0 && continue

        p     = Int(ft.face_panel[f])
        dir   = Int(ft.face_dir[f])
        i     = Int(ft.face_idx_i[f])
        j     = Int(ft.face_idx_j[f])
        mdir  = Int(ft.mirror_dir[f])
        mi    = Int(ft.mirror_idx_i[f])
        mj    = Int(ft.mirror_idx_j[f])
        msign = Int(ft.mirror_sign[f])

        for k in 1:Nz
            canonical_val = dir == 1 ? panels_am[p][i, j, k] : panels_bm[p][i, j, k]
            mirror_val = FT(msign) * canonical_val
            if mdir == 1
                panels_am[mq][mi, mj, k] = mirror_val
            else
                panels_bm[mq][mi, mj, k] = mirror_val
            end
        end
    end
    return nothing
end

"""
    sync_all_cs_boundary_mirrors!(panels_am, panels_bm, conn, Nc, Nz)

Comprehensive boundary flux mirror sync using panel connectivity directly.

Iterates all 12 physical edges of the cubed sphere (all 4 directions per panel,
de-duplicated by p < q). For each edge, syncs both directions: canonical panel's
boundary → neighbor's mirror, AND neighbor's boundary → canonical panel's mirror.

Sign convention: when canonical and mirror are both at inflow positions (i=1 or
j=1) or both at outflow positions (i=Nc+1 or j=Nc+1), negate so that per-panel
flux telescoping conserves mass.

Must be called after horizontal flux balancing to propagate balanced canonical
values to all mirror positions.
"""
function sync_all_cs_boundary_mirrors!(panels_am::NTuple{6, Array{FT, 3}},
                                        panels_bm::NTuple{6, Array{FT, 3}},
                                        conn::PanelConnectivity,
                                        Nc::Int, Nz::Int) where FT
    # Iterate all 4 edges of each panel with p < q dedup — same canonical
    # ordering as the face table.  For each canonical edge (p, e) → (q, eq),
    # sync both directions: p's boundary → q's mirror AND q's boundary → p's
    # mirror.
    for p in 1:6
        for e in (EDGE_NORTH, EDGE_SOUTH, EDGE_EAST, EDGE_WEST)
            q   = conn.neighbors[p][e].panel
            ori = conn.neighbors[p][e].orientation
            eq  = reciprocal_edge(conn, p, e)

            p > q && continue   # other side is canonical

            # --- Forward: canonical on p → mirror on q ---
            for s in 1:Nc
                t = ori == 0 ? s : Nc + 1 - s

                can_dir, can_i, can_j = _cs_edge_face_location(e, s, Nc)
                mir_dir, mir_i, mir_j = _cs_edge_face_location(eq, t, Nc)

                can_at_outflow = (can_dir == 1 && can_i == Nc + 1) ||
                                 (can_dir == 2 && can_j == Nc + 1)
                mir_at_outflow = (mir_dir == 1 && mir_i == Nc + 1) ||
                                 (mir_dir == 2 && mir_j == Nc + 1)
                msign = (can_at_outflow == mir_at_outflow) ? FT(-1) : FT(1)

                @inbounds for k in 1:Nz
                    canonical = can_dir == 1 ? panels_am[p][can_i, can_j, k] :
                                               panels_bm[p][can_i, can_j, k]
                    mirror_val = msign * canonical
                    if mir_dir == 1
                        panels_am[q][mir_i, mir_j, k] = mirror_val
                    else
                        panels_bm[q][mir_i, mir_j, k] = mirror_val
                    end
                end
            end

            # --- Reverse: q's boundary → p's mirror ---
            for s in 1:Nc
                t = ori == 0 ? s : Nc + 1 - s

                can_dir, can_i, can_j = _cs_edge_face_location(eq, t, Nc)
                mir_dir, mir_i, mir_j = _cs_edge_face_location(e, s, Nc)

                can_at_outflow = (can_dir == 1 && can_i == Nc + 1) ||
                                 (can_dir == 2 && can_j == Nc + 1)
                mir_at_outflow = (mir_dir == 1 && mir_i == Nc + 1) ||
                                 (mir_dir == 2 && mir_j == Nc + 1)
                msign = (can_at_outflow == mir_at_outflow) ? FT(-1) : FT(1)

                @inbounds for k in 1:Nz
                    canonical = can_dir == 1 ? panels_am[q][can_i, can_j, k] :
                                               panels_bm[q][can_i, can_j, k]
                    mirror_val = msign * canonical
                    if mir_dir == 1
                        panels_am[p][mir_i, mir_j, k] = mirror_val
                    else
                        panels_bm[p][mir_i, mir_j, k] = mirror_val
                    end
                end
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# High-level balance entry point
# ---------------------------------------------------------------------------

function _balance_cs_level!(
    k::Int,
    panels_am::NTuple{6, Array{FT, 3}},
    panels_bm::NTuple{6, Array{FT, 3}},
    panels_m::NTuple{6, Array{FT, 3}},
    panels_m_next::NTuple{6, Array{FT, 3}},
    ft::CSGlobalFaceTable,
    degree::Vector{Int},
    scratch::CSPoissonScratch,
    inv_scale::Float64;
    tol::Float64,
    max_iter::Int,
    project_every::Int,
) where FT
    Nc = ft.Nc
    nc = ft.nc
    div = scratch.div
    rhs = scratch.rhs
    psi = scratch.psi
    cg_scratch = (r = scratch.r, p = scratch.p, Ap = scratch.Ap, z = scratch.z)

    # 1. Compute current horizontal divergence.
    fill!(div, 0.0)
    @inbounds for f in 1:ft.nf
        panel = Int(ft.face_panel[f])
        dir = Int(ft.face_dir[f])
        i = Int(ft.face_idx_i[f])
        j = Int(ft.face_idx_j[f])
        flux = dir == 1 ? Float64(panels_am[panel][i, j, k]) :
                          Float64(panels_bm[panel][i, j, k])
        left = Int(ft.face_left[f])
        right = Int(ft.face_right[f])
        div[left] += flux
        div[right] -= flux
    end

    # 2. RHS = divergence - target mass tendency.
    rhs_sum = 0.0
    @inbounds for c in 1:nc
        p_idx = (c - 1) ÷ (Nc * Nc) + 1
        local_idx = (c - 1) % (Nc * Nc)
        j_local = local_idx ÷ Nc + 1
        i_local = local_idx % Nc + 1
        target = (Float64(panels_m[p_idx][i_local, j_local, k]) -
                  Float64(panels_m_next[p_idx][i_local, j_local, k])) * inv_scale
        rc = div[c] - target
        rhs[c] = rc
        rhs_sum += rc
    end

    rhs_raw_linf = _cs_linf(rhs)
    rhs_mean = rhs_sum / nc
    pre_proj = 0.0
    @inbounds @simd for c in 1:nc
        a = abs(rhs[c] - rhs_mean)
        pre_proj = ifelse(a > pre_proj, a, pre_proj)
    end

    if pre_proj < tol
        return (pre = rhs_raw_linf, post = 0.0, iter = 0,
                rhs_mean = abs(rhs_mean), pre_proj = pre_proj, post_proj = 0.0)
    end

    # 3. Solve L * psi = rhs.
    _, it = solve_cs_poisson_pcg!(psi, rhs, ft, degree, cg_scratch;
                                  tol=tol, max_iter=max_iter,
                                  project_every=project_every)

    # Diagnostic: post-solve projected residual.
    Lpsi = scratch.Ap
    _cs_graph_laplacian_mul!(Lpsi, psi, ft, degree)
    fill!(div, 0.0)
    @inbounds for f in 1:ft.nf
        panel = Int(ft.face_panel[f])
        dir = Int(ft.face_dir[f])
        i = Int(ft.face_idx_i[f])
        j = Int(ft.face_idx_j[f])
        flux = dir == 1 ? Float64(panels_am[panel][i, j, k]) :
                          Float64(panels_bm[panel][i, j, k])
        left = Int(ft.face_left[f])
        right = Int(ft.face_right[f])
        div[left] += flux
        div[right] -= flux
    end
    @inbounds for c in 1:nc
        p_idx = (c - 1) ÷ (Nc * Nc) + 1
        local_idx = (c - 1) % (Nc * Nc)
        j_local = local_idx ÷ Nc + 1
        i_local = local_idx % Nc + 1
        target = (Float64(panels_m[p_idx][i_local, j_local, k]) -
                  Float64(panels_m_next[p_idx][i_local, j_local, k])) * inv_scale
        rhs[c] = (div[c] - target) - rhs_mean
    end
    post_proj = 0.0
    @inbounds @simd for c in 1:nc
        a = abs(Lpsi[c] - rhs[c])
        post_proj = ifelse(a > post_proj, a, post_proj)
    end

    # 4. Apply correction to all faces at this level.
    apply_cs_flux_correction!(panels_am, panels_bm, psi, ft, k)

    # 5. Post-balance raw residual.
    fill!(div, 0.0)
    @inbounds for f in 1:ft.nf
        panel = Int(ft.face_panel[f])
        dir = Int(ft.face_dir[f])
        i = Int(ft.face_idx_i[f])
        j = Int(ft.face_idx_j[f])
        flux = dir == 1 ? Float64(panels_am[panel][i, j, k]) :
                          Float64(panels_bm[panel][i, j, k])
        left = Int(ft.face_left[f])
        right = Int(ft.face_right[f])
        div[left] += flux
        div[right] -= flux
    end
    post_raw = 0.0
    @inbounds for c in 1:nc
        p_idx = (c - 1) ÷ (Nc * Nc) + 1
        local_idx = (c - 1) % (Nc * Nc)
        j_local = local_idx ÷ Nc + 1
        i_local = local_idx % Nc + 1
        target = (Float64(panels_m[p_idx][i_local, j_local, k]) -
                  Float64(panels_m_next[p_idx][i_local, j_local, k])) * inv_scale
        r = abs(div[c] - target)
        post_raw = ifelse(r > post_raw, r, post_raw)
    end

    return (pre = rhs_raw_linf, post = post_raw, iter = it,
            rhs_mean = abs(rhs_mean), pre_proj = pre_proj, post_proj = post_proj)
end

"""
    balance_cs_global_mass_fluxes!(panels_am, panels_bm, panels_m, panels_m_next,
                                    ft, degree, steps_per_window, scratch;
                                    tol=1e-14, max_iter=20000)

TM5-style global Poisson mass-flux balance for a 6-panel cubed sphere.

Corrects `panels_am[p]` and `panels_bm[p]` so that horizontal flux
convergence at every cell matches the prescribed mass tendency:

    dm_dt[c, k] = (m_next[c, k] - m_cur[c, k]) / (2 × steps_per_window)

Returns a diagnostic NamedTuple with pre/post residuals and CG iteration counts.
"""
function balance_cs_global_mass_fluxes!(
    panels_am::NTuple{6, Array{FT, 3}},
    panels_bm::NTuple{6, Array{FT, 3}},
    panels_m::NTuple{6, Array{FT, 3}},
    panels_m_next::NTuple{6, Array{FT, 3}},
    ft::CSGlobalFaceTable,
    degree::Vector{Int},
    steps_per_window::Int,
    scratch::CSPoissonScratch;
    tol::Float64=1e-14,
    max_iter::Int=20000,
    project_every::Int=50,
) where FT

    Nc = ft.Nc
    Nz = size(panels_am[1], 3)
    nc = ft.nc
    inv_scale = 1.0 / (2.0 * steps_per_window)

    nthread = Threads.maxthreadid()
    scratches = Vector{CSPoissonScratch}(undef, nthread)
    scratches[1] = scratch
    for t in 2:nthread
        scratches[t] = CSPoissonScratch(nc)
    end

    pre_by_level = zeros(Float64, Nz)
    post_by_level = zeros(Float64, Nz)
    rhs_mean_by_level = zeros(Float64, Nz)
    pre_proj_by_level = zeros(Float64, Nz)
    post_proj_by_level = zeros(Float64, Nz)
    iter_by_level = zeros(Int, Nz)

    Threads.@threads :static for k in 1:Nz
        diag = _balance_cs_level!(
            k, panels_am, panels_bm, panels_m, panels_m_next,
            ft, degree, scratches[Threads.threadid()], inv_scale;
            tol=tol, max_iter=max_iter, project_every=project_every)
        pre_by_level[k] = diag.pre
        post_by_level[k] = diag.post
        rhs_mean_by_level[k] = diag.rhs_mean
        pre_proj_by_level[k] = diag.pre_proj
        post_proj_by_level[k] = diag.post_proj
        iter_by_level[k] = diag.iter
    end

    # 6. Synchronize ALL cross-panel mirror entries at ALL levels.
    _sync_cs_mirrors!(panels_am, panels_bm, ft, Nz)

    return (;
        max_pre_residual = maximum(pre_by_level),
        max_post_residual = maximum(post_by_level),
        max_rhs_mean = maximum(rhs_mean_by_level),
        max_pre_projected = maximum(pre_proj_by_level),
        max_post_projected = maximum(post_proj_by_level),
        max_cg_iter = maximum(iter_by_level),
    )
end

function _fill_cs_column_buffers!(col_am::NTuple{6, Array{FT, 3}},
                                  col_bm::NTuple{6, Array{FT, 3}},
                                  col_m::NTuple{6, Array{FT, 3}},
                                  col_m_next::NTuple{6, Array{FT, 3}},
                                  panels_am::NTuple{6, Array{FT, 3}},
                                  panels_bm::NTuple{6, Array{FT, 3}},
                                  panels_m::NTuple{6, Array{FT, 3}},
                                  panels_m_next::NTuple{6, Array{FT, 3}},
                                  Nc::Int, Nz::Int) where FT
    for p in 1:6
        fill!(col_am[p], zero(FT))
        fill!(col_bm[p], zero(FT))
        fill!(col_m[p], zero(FT))
        fill!(col_m_next[p], zero(FT))
        @inbounds for k in 1:Nz
            for j in 1:Nc, i in 1:Nc + 1
                col_am[p][i, j, 1] += panels_am[p][i, j, k]
            end
            for j in 1:Nc + 1, i in 1:Nc
                col_bm[p][i, j, 1] += panels_bm[p][i, j, k]
            end
            for j in 1:Nc, i in 1:Nc
                col_m[p][i, j, 1] += panels_m[p][i, j, k]
                col_m_next[p][i, j, 1] += panels_m_next[p][i, j, k]
            end
        end
    end
    return nothing
end

function _cs_column_balance_projected_linf(col_am::NTuple{6, Array{FT, 3}},
                                           col_bm::NTuple{6, Array{FT, 3}},
                                           col_m::NTuple{6, Array{FT, 3}},
                                           col_m_next::NTuple{6, Array{FT, 3}},
                                           ft::CSGlobalFaceTable,
                                           steps_per_window::Int,
                                           scratch::CSPoissonScratch) where FT
    Nc = ft.Nc
    inv_scale = 1.0 / (2.0 * steps_per_window)
    div = scratch.div
    fill!(div, 0.0)
    @inbounds for f in 1:ft.nf
        panel = Int(ft.face_panel[f])
        dir = Int(ft.face_dir[f])
        i = Int(ft.face_idx_i[f])
        j = Int(ft.face_idx_j[f])
        flux = dir == 1 ? Float64(col_am[panel][i, j, 1]) :
                          Float64(col_bm[panel][i, j, 1])
        div[Int(ft.face_left[f])] += flux
        div[Int(ft.face_right[f])] -= flux
    end

    raw_linf = 0.0
    mean = 0.0
    @inbounds for c in 1:ft.nc
        p_idx = (c - 1) ÷ (Nc * Nc) + 1
        local_idx = (c - 1) % (Nc * Nc)
        j_local = local_idx ÷ Nc + 1
        i_local = local_idx % Nc + 1
        target = (Float64(col_m[p_idx][i_local, j_local, 1]) -
                  Float64(col_m_next[p_idx][i_local, j_local, 1])) * inv_scale
        r = div[c] - target
        div[c] = r
        mean += r
        raw_linf = max(raw_linf, abs(r))
    end
    mean /= ft.nc
    projected_linf = 0.0
    @inbounds @simd for c in 1:ft.nc
        projected_linf = max(projected_linf, abs(div[c] - mean))
    end
    return (raw_linf = raw_linf, projected_linf = projected_linf,
            mean_abs = abs(mean))
end

"""
How a column mass-budget correction is spread over the levels of a column.

- `MassWeightedColumn()`: in proportion to layer air mass (default).
- `HybridBWeightedColumn(B)`: in proportion to `ΔB_k`, the layer's share of a
  surface-pressure change, as in TM5. Pure-pressure layers (`ΔB = 0`, the
  stratosphere) receive none of it, so a column mismatch cannot appear in
  their vertical mass flux as a coherent mode proportional to pressure.
- `HybridMassWeightedColumn(B)`: in proportion to layer air mass, but only in
  layers with `ΔB > 0`; pure-pressure layers receive none of it.
"""
abstract type AbstractColumnWeights end
struct MassWeightedColumn <: AbstractColumnWeights end

# Layer thicknesses in B from hybrid `B` at the `Nz + 1` interfaces, top first.
function _hybrid_layer_dB(B::AbstractVector{<:AbstractFloat})
    dB = diff(Float64.(B))
    all(>=(0), dB) && sum(dB) > 0 ||
        throw(ArgumentError("hybrid B must be non-decreasing from top to surface and not constant"))
    return dB
end

"""
    HybridBWeightedColumn(B) -> weights

`ΔB_k` weights from hybrid `B` at the `Nz + 1` interfaces, top first.
"""
struct HybridBWeightedColumn <: AbstractColumnWeights
    dB :: Vector{Float64}          # per layer, top first; sums to 1
    function HybridBWeightedColumn(B::AbstractVector{<:AbstractFloat})
        dB = _hybrid_layer_dB(B)
        return new(dB ./ sum(dB))
    end
end

"""
    HybridMassWeightedColumn(B) -> weights

Air-mass weights restricted to the hybrid layers (`ΔB_k > 0`) of interface
`B`, top first.
"""
struct HybridMassWeightedColumn <: AbstractColumnWeights
    hybrid :: BitVector            # layers with ΔB > 0, top first
    HybridMassWeightedColumn(B::AbstractVector{<:AbstractFloat}) = new(_hybrid_layer_dB(B) .> 0)
end

# Weight of level `k` for a face between cells of air mass `m_a`, `m_b`,
# and for a single cell of air mass `m`.
@inline _layer_weight(::MassWeightedColumn, m_a, m_b, k) =
    max(0.0, Float64(m_a)) + max(0.0, Float64(m_b))
@inline _layer_weight(w::HybridBWeightedColumn, m_a, m_b, k) = w.dB[k]
@inline _layer_weight(w::HybridMassWeightedColumn, m_a, m_b, k) =
    w.hybrid[k] ? _layer_weight(MassWeightedColumn(), m_a, m_b, k) : 0.0
@inline _cell_weight(::MassWeightedColumn, m, k) = m
@inline _cell_weight(w::HybridBWeightedColumn, m, k) = oftype(m, w.dB[k])
@inline _cell_weight(w::HybridMassWeightedColumn, m, k) = w.hybrid[k] ? m : zero(m)

# Number of layers the weights are defined for (`nothing`: any).
_weight_levels(::MassWeightedColumn) = nothing
_weight_levels(w::HybridBWeightedColumn) = length(w.dB)
_weight_levels(w::HybridMassWeightedColumn) = length(w.hybrid)

function _check_weight_levels(w::AbstractColumnWeights, Nz)
    n = _weight_levels(w)
    n === nothing || n == Nz || throw(DimensionMismatch(
        "column balance weights are defined for $n layers, the fluxes have $Nz"))
    return nothing
end

"""
    COLUMN_WEIGHT_KINDS

TOML names of the column-balance weightings: `mass`, `hybrid_b`, `hybrid_mass`.
"""
const COLUMN_WEIGHT_KINDS = (mass = (B -> MassWeightedColumn()),
                             hybrid_b = HybridBWeightedColumn,
                             hybrid_mass = HybridMassWeightedColumn)

"""
    column_weights(kind::Symbol, B) -> AbstractColumnWeights

Weights named `kind` (a key of `COLUMN_WEIGHT_KINDS`) for hybrid interfaces
`B`, top first.
"""
function column_weights(kind::Symbol, B)
    haskey(COLUMN_WEIGHT_KINDS, kind) || throw(ArgumentError(
        "column balance weights must be one of $(keys(COLUMN_WEIGHT_KINDS)); got :$kind"))
    return COLUMN_WEIGHT_KINDS[kind](B)
end

function _distribute_cs_column_delta!(panels_am::NTuple{6, Array{FT, 3}},
                                      panels_bm::NTuple{6, Array{FT, 3}},
                                      panels_m::NTuple{6, Array{FT, 3}},
                                      col_am::NTuple{6, Array{FT, 3}},
                                      col_bm::NTuple{6, Array{FT, 3}},
                                      col_am_before::NTuple{6, Array{FT, 3}},
                                      col_bm_before::NTuple{6, Array{FT, 3}},
                                      Nc::Int, Nz::Int,
                                      weights::AbstractColumnWeights = MassWeightedColumn()) where FT
    _check_weight_levels(weights, Nz)
    max_face_delta = 0.0
    for p in 1:6
        @inbounds for j in 1:Nc, i in 1:Nc + 1
            delta = Float64(col_am[p][i, j, 1] - col_am_before[p][i, j, 1])
            max_face_delta = max(max_face_delta, abs(delta))
            delta == 0.0 && continue

            i_l = max(i - 1, 1)
            i_r = min(i, Nc)
            denom = 0.0
            for k in 1:Nz
                denom += _layer_weight(weights, panels_m[p][i_l, j, k], panels_m[p][i_r, j, k], k)
            end
            if denom > 0.0
                applied = 0.0
                for k in 1:Nz-1
                    w = _layer_weight(weights, panels_m[p][i_l, j, k], panels_m[p][i_r, j, k], k) / denom
                    inc = FT(delta * w)
                    panels_am[p][i, j, k] += inc
                    applied += Float64(inc)
                end
                panels_am[p][i, j, Nz] += FT(delta - applied)
            else
                applied = 0.0
                even = delta / Nz
                for k in 1:Nz-1
                    inc = FT(even)
                    panels_am[p][i, j, k] += inc
                    applied += Float64(inc)
                end
                panels_am[p][i, j, Nz] += FT(delta - applied)
            end
        end

        @inbounds for j in 1:Nc + 1, i in 1:Nc
            delta = Float64(col_bm[p][i, j, 1] - col_bm_before[p][i, j, 1])
            max_face_delta = max(max_face_delta, abs(delta))
            delta == 0.0 && continue

            j_s = max(j - 1, 1)
            j_n = min(j, Nc)
            denom = 0.0
            for k in 1:Nz
                denom += _layer_weight(weights, panels_m[p][i, j_s, k], panels_m[p][i, j_n, k], k)
            end
            if denom > 0.0
                applied = 0.0
                for k in 1:Nz-1
                    w = _layer_weight(weights, panels_m[p][i, j_s, k], panels_m[p][i, j_n, k], k) / denom
                    inc = FT(delta * w)
                    panels_bm[p][i, j, k] += inc
                    applied += Float64(inc)
                end
                panels_bm[p][i, j, Nz] += FT(delta - applied)
            else
                applied = 0.0
                even = delta / Nz
                for k in 1:Nz-1
                    inc = FT(even)
                    panels_bm[p][i, j, k] += inc
                    applied += Float64(inc)
                end
                panels_bm[p][i, j, Nz] += FT(delta - applied)
            end
        end
    end
    return max_face_delta
end

"""
    balance_cs_column_mass_fluxes!(panels_am, panels_bm, panels_m, panels_m_next,
                                   ft, degree, steps_per_window, scratch; ...)

Apply a single vertically integrated CS Poisson correction, then distribute the
face correction over levels with `weights` (`MassWeightedColumn()`, local air
mass, by default; `HybridBWeightedColumn` follows TM5).

This is the ERA CS default. It enforces the column mass budget required by zero
top/bottom `cm` while avoiding the legacy per-layer correction that can rewrite
real vertical wind shear.
"""
function balance_cs_column_mass_fluxes!(
    panels_am::NTuple{6, Array{FT, 3}},
    panels_bm::NTuple{6, Array{FT, 3}},
    panels_m::NTuple{6, Array{FT, 3}},
    panels_m_next::NTuple{6, Array{FT, 3}},
    ft::CSGlobalFaceTable,
    degree::Vector{Int},
    steps_per_window::Int,
    scratch::CSPoissonScratch;
    tol::Float64=1e-14,
    max_iter::Int=20000,
    project_every::Int=50,
    closure_passes::Int=1,
    closure_tol::Float64=10.0,
    weights::AbstractColumnWeights=MassWeightedColumn(),
) where FT
    Nc = ft.Nc
    Nz = size(panels_am[1], 3)

    col_am = ntuple(_ -> zeros(FT, Nc + 1, Nc, 1), 6)
    col_bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, 1), 6)
    col_m = ntuple(_ -> zeros(FT, Nc, Nc, 1), 6)
    col_m_next = ntuple(_ -> zeros(FT, Nc, Nc, 1), 6)
    col_am_before = ntuple(_ -> zeros(FT, Nc + 1, Nc, 1), 6)
    col_bm_before = ntuple(_ -> zeros(FT, Nc, Nc + 1, 1), 6)

    max_face_delta = 0.0
    diag = nothing
    final_stats = nothing
    passes = max(1, closure_passes)
    for pass in 1:passes
        _fill_cs_column_buffers!(col_am, col_bm, col_m, col_m_next,
                                 panels_am, panels_bm, panels_m, panels_m_next,
                                 Nc, Nz)
        for p in 1:6
            copyto!(col_am_before[p], col_am[p])
            copyto!(col_bm_before[p], col_bm[p])
        end
        diag = balance_cs_global_mass_fluxes!(
            col_am, col_bm, col_m, col_m_next, ft, degree, steps_per_window, scratch;
            tol, max_iter, project_every)

        max_face_delta = max(max_face_delta,
            _distribute_cs_column_delta!(panels_am, panels_bm, panels_m,
                                         col_am, col_bm,
                                         col_am_before, col_bm_before,
                                         Nc, Nz, weights))
        _sync_cs_mirrors!(panels_am, panels_bm, ft, Nz)

        _fill_cs_column_buffers!(col_am, col_bm, col_m, col_m_next,
                                 panels_am, panels_bm, panels_m, panels_m_next,
                                 Nc, Nz)
        final_stats = _cs_column_balance_projected_linf(
            col_am, col_bm, col_m, col_m_next, ft, steps_per_window, scratch)
        final_stats.projected_linf <= closure_tol && break
    end

    return (;
        max_pre_residual = diag.max_pre_residual,
        max_post_residual = diag.max_post_residual,
        max_rhs_mean = diag.max_rhs_mean,
        max_pre_projected = diag.max_pre_projected,
        max_post_projected = diag.max_post_projected,
        final_column_raw_residual = final_stats.raw_linf,
        final_column_projected_residual = final_stats.projected_linf,
        final_column_mean_residual = final_stats.mean_abs,
        max_cg_iter = diag.max_cg_iter,
        max_face_delta = max_face_delta,
    )
end

# ---------------------------------------------------------------------------
# Vertical mass flux diagnosis
# ---------------------------------------------------------------------------

"""
    diagnose_cs_cm!(panels_cm, panels_am, panels_bm, panels_dm, panels_m, Nc, Nz[, weights])

Diagnose vertical mass flux `cm` from column-balanced horizontal flux divergence
and mass tendency for all 6 panels. A remaining column residual is spread with
`weights` (layer air mass by default).
"""
function diagnose_cs_cm!(panels_cm::NTuple{6, Array{FT, 3}},
                          panels_am::NTuple{6, Array{FT, 3}},
                          panels_bm::NTuple{6, Array{FT, 3}},
                          panels_dm::NTuple{6, Array{FT, 3}},
                          panels_m::NTuple{6, Array{FT, 3}},
                          Nc::Int, Nz::Int,
                          weights::AbstractColumnWeights = MassWeightedColumn()) where FT
    _check_weight_levels(weights, Nz)
    for p in 1:6
        am = panels_am[p]
        bm = panels_bm[p]
        cm = panels_cm[p]
        dm = panels_dm[p]
        m  = panels_m[p]

        @inbounds for j in 1:Nc, i in 1:Nc
            cm[i, j, 1] = zero(FT)

            for k in 1:Nz
                div_h = (am[i, j, k] - am[i + 1, j, k]) +
                        (bm[i, j, k] - bm[i, j + 1, k])
                cm[i, j, k + 1] = cm[i, j, k] + div_h - dm[i, j, k]
            end

            # Redistribute any remaining residual with the column weights
            residual = cm[i, j, Nz + 1]
            if abs(residual) > eps(FT)
                total_m = zero(FT)
                for k in 1:Nz
                    total_m += _cell_weight(weights, m[i, j, k], k)
                end
                if total_m > zero(FT)
                    cum_fix = zero(FT)
                    for k in 1:Nz
                        frac = _cell_weight(weights, m[i, j, k], k) / total_m
                        cum_fix += frac * residual
                        cm[i, j, k + 1] -= cum_fix
                    end
                end
            end
        end
    end
    return nothing
end
