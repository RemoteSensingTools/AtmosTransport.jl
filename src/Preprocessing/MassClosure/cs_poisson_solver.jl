# Cubed-sphere Poisson balance: graph Laplacian, CG solver, correction mapping, mirror synchronization.
# Split from cs_poisson_balance.jl (refactor phase 4); included by Preprocessing.jl in this order.

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

Pre-allocated work buffers for the CS global Poisson balance. `tol` holds the
per-cell closure tolerance of the last column balance
(`balance_cs_column_mass_fluxes!`), which the `cm` diagnosis that follows it
checks the column residuals against.
"""
struct CSPoissonScratch
    psi :: Vector{Float64}
    rhs :: Vector{Float64}
    r   :: Vector{Float64}
    p   :: Vector{Float64}
    Ap  :: Vector{Float64}
    z   :: Vector{Float64}
    div :: Vector{Float64}
    tol :: Vector{Float64}
end

function CSPoissonScratch(nc::Int)
    return CSPoissonScratch(
        zeros(Float64, nc), zeros(Float64, nc),
        zeros(Float64, nc), zeros(Float64, nc),
        zeros(Float64, nc), zeros(Float64, nc),
        zeros(Float64, nc), zeros(Float64, nc),
    )
end

"""
    cs_graph_laplacian_mul!(out, psi, ft, degree)

Compute `out = L · psi` for the global CS graph Laplacian.
"""
function cs_graph_laplacian_mul!(out::AbstractVector{Float64},
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

@inline function _cs_dot(a::AbstractVector{Float64}, b::AbstractVector{Float64})
    s = 0.0
    @inbounds @simd for i in eachindex(a, b)
        s += a[i] * b[i]
    end
    return s
end

@inline function cs_linf(a::AbstractVector{Float64})
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
    solve_cs_poisson_pcg!(psi, rhs, ft, degree, scratch; tol=DEFAULT_POISSON_BALANCE_TOL,
                           max_iter=DEFAULT_POISSON_BALANCE_MAX_ITER,
                           project_every=DEFAULT_POISSON_PROJECT_EVERY)

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
                                tol::Float64=DEFAULT_POISSON_BALANCE_TOL,
                                max_iter::Int=DEFAULT_POISSON_BALANCE_MAX_ITER,
                                project_every::Int=DEFAULT_POISSON_PROJECT_EVERY)
    r  = scratch.r
    p  = scratch.p
    Ap = scratch.Ap
    z  = scratch.z

    project_mean_zero!(rhs)
    fill!(psi, 0.0)
    copyto!(r, rhs)

    # Jacobi preconditioner: z = M⁻¹ r, M = diag(degree)
    _cs_precondition!(z, r, degree)
    copyto!(p, z)

    rz_old = _cs_dot(r, z)
    rhs_linf = cs_linf(rhs) + eps()

    iter = 0
    r_linf = rhs_linf
    while iter < max_iter
        cs_graph_laplacian_mul!(Ap, p, ft, degree)
        pAp = _cs_dot(p, Ap)
        pAp <= 0 && break
        alpha = rz_old / pAp
        r_linf = _cs_update_psi_r_linf!(psi, r, p, Ap, alpha)

        do_project = project_every <= 1 ||
                     (project_every > 1 && (iter + 1) % project_every == 0)
        do_project && (r_linf = _cs_project_mean_zero_linf!(r))
        r_linf / rhs_linf < tol && break

        _cs_precondition!(z, r, degree)
        do_project && project_mean_zero!(z)
        rz_new = _cs_dot(r, z)
        rz_old == 0.0 && break
        beta = rz_new / rz_old
        _cs_update_direction!(p, z, beta)
        rz_old = rz_new
        iter += 1
    end

    project_mean_zero!(psi)
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
    sync_cs_mirrors!(panels_am, panels_bm, ft, Nz)

Copy canonical boundary flux values to their cross-panel mirror entries
for all levels.
"""
function sync_cs_mirrors!(panels_am::NTuple{6, Array{FT, 3}},
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
de-duplicated by p < q). For each edge, copy the lower-numbered canonical
panel's boundary flux to its neighbor's mirror slot.

Sign convention: when canonical and mirror are both at inflow positions (i=1 or
j=1) or both at outflow positions (i=Nc+1 or j=Nc+1), negate so that per-panel
flux telescoping conserves mass.

Call after modifying canonical boundary fluxes to propagate them to all mirror
positions. Column balance already synchronizes through its face table.
"""
function sync_all_cs_boundary_mirrors!(panels_am::NTuple{6, Array{FT, 3}},
                                        panels_bm::NTuple{6, Array{FT, 3}},
                                        conn::PanelConnectivity,
                                        Nc::Int, Nz::Int) where FT
    # Iterate all 4 edges of each panel with p < q dedup — same canonical
    # ordering as the face table.  For each canonical edge (p, e) → (q, eq),
    # copy p's canonical boundary to q's mirror.
    for p in 1:6
        for e in (EDGE_NORTH, EDGE_SOUTH, EDGE_EAST, EDGE_WEST)
            q   = conn.neighbors[p][e].panel
            ori = conn.neighbors[p][e].orientation
            eq  = reciprocal_edge(conn, p, e)

            p > q && continue   # other side is canonical

            # --- Forward: canonical on p → mirror on q ---
            for s in 1:Nc
                t = ori == 0 ? s : Nc + 1 - s

                can_dir, can_i, can_j = cs_edge_face_location(e, s, Nc)
                mir_dir, mir_i, mir_j = cs_edge_face_location(eq, t, Nc)

                msign = seam_mirror_sign(e, eq, FT)

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

        end
    end
    return nothing
end
