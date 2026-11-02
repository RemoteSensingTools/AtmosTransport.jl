# =========================================================================
# Compressed-Laplacian Poisson balance for reduced-Gaussian meshes.
#
# The LCM boundary segmentation in ReducedGaussianMesh creates millions
# of faces (O90: 2.25M for 35K cells) because lcm(nlon_j, nlon_{j+1})
# explodes for adjacent octahedral rings.  However, each cell has only
# ~4-6 UNIQUE neighbors — the LCM duplicates are redundant for the
# graph Laplacian.
#
# This module precomputes a compressed cell-to-cell adjacency with
# integer multiplicity weights, mathematically equivalent to the full
# face-indexed graph Laplacian.  The CG MatVec cost drops from
# O(nfaces) to O(4 × ncells), giving 10-30× speedup.
#
# Measured: O90 222s → 19s/window (11×).
#
# The solver uses Jacobi-preconditioned CG on the exact multiplicity-
# weighted discrete operator.  Future speedup paths (not yet implemented):
#   - Normalized-overlap-weight Laplacian as PCG preconditioner
#   - Ring-block smoother (zonal FFT) as preconditioning component
#   - Graph AMG on the compressed operator
# These would reduce the ~1300 CG iterations without changing the answer.
# =========================================================================

"""
    CompressedLaplacian

Sparse cell-to-cell adjacency with multiplicity weights, equivalent to
the graph Laplacian on the face-indexed mesh but stored in CSR-like format
with O(ncells) entries instead of O(nfaces).

For octahedral reduced-Gaussian grids, adjacent rings differ by only 4 cells,
so each cell has at most ~6 unique neighbors (2 zonal + 2-4 meridional).
The LCM boundary segmentation creates up to ~100 faces per cell for the same
neighbors — the compressed form collapses these into weighted edges.
"""
struct CompressedLaplacian
    # CSR-like storage: cell c's neighbors are at indices row_ptr[c]:(row_ptr[c+1]-1)
    row_ptr   :: Vector{Int}       # length ncells + 1
    col_idx   :: Vector{Int32}     # unique neighbor cell indices
    weights   :: Vector{Int32}     # multiplicity (number of shared faces per pair)
    degree    :: Vector{Int}       # total interior face count per cell (sum of weights)
end

"""
    build_compressed_laplacian(face_left, face_right, nc) -> CompressedLaplacian

Compress the face-indexed graph Laplacian into cell-to-cell adjacency.
Iterates over faces once to build the compressed representation.
"""
function build_compressed_laplacian(face_left::Vector{Int32},
                                    face_right::Vector{Int32},
                                    nc::Int)
    # Pass 1: count unique neighbors per cell using a temporary Dict
    # (We use a Dict{Int32, Int32} per cell — small since each cell has ~4-6 neighbors)
    neighbor_maps = [Dict{Int32, Int32}() for _ in 1:nc]
    degree = zeros(Int, nc)

    @inbounds for f in eachindex(face_left)
        left  = face_left[f]
        right = face_right[f]
        left > 0 && right > 0 || continue

        degree[left]  += 1
        degree[right] += 1

        # Accumulate multiplicity for the (left, right) pair
        d_left = neighbor_maps[left]
        d_left[right] = get(d_left, right, Int32(0)) + Int32(1)

        d_right = neighbor_maps[right]
        d_right[left] = get(d_right, left, Int32(0)) + Int32(1)
    end

    # Pass 2: flatten into CSR
    total_entries = sum(length(d) for d in neighbor_maps)
    row_ptr = Vector{Int}(undef, nc + 1)
    col_idx = Vector{Int32}(undef, total_entries)
    weights = Vector{Int32}(undef, total_entries)

    offset = 1
    for c in 1:nc
        row_ptr[c] = offset
        d = neighbor_maps[c]
        for (neighbor, w) in d
            col_idx[offset] = neighbor
            weights[offset] = w
            offset += 1
        end
    end
    row_ptr[nc + 1] = offset

    return CompressedLaplacian(row_ptr, col_idx, weights, degree)
end

"""
    _compressed_laplacian_mul!(out, psi, L::CompressedLaplacian)

Compute `out = L · psi` using the compressed cell-to-cell adjacency.
Cost: O(ncells × avg_neighbors) ≈ O(4 × ncells), vs O(nfaces) for the
face-indexed version.
"""
function _compressed_laplacian_mul!(out::AbstractVector{Float64},
                                    psi::AbstractVector{Float64},
                                    L::CompressedLaplacian)
    @inbounds for c in eachindex(out)
        out[c] = L.degree[c] * psi[c]
    end
    @inbounds for c in eachindex(out)
        for idx in L.row_ptr[c]:(L.row_ptr[c + 1] - 1)
            out[c] -= Float64(L.weights[idx]) * psi[L.col_idx[idx]]
        end
    end
    return out
end

"""
    solve_compressed_poisson_pcg!(psi, rhs, L::CompressedLaplacian, scratch;
                                  tol=DEFAULT_POISSON_BALANCE_TOL,
                                  max_iter=DEFAULT_POISSON_BALANCE_MAX_ITER)

Solve the singular system `L · psi = rhs` via Jacobi-Preconditioned CG
using the compressed Laplacian. Mathematically identical to
the face-indexed graph-Laplacian solver but ~16-27× faster due to compressed MatVec.

The compressed Laplacian has the same 1-D constant null space as the
face-indexed version, so the same mean-zero projection is applied.
"""
function solve_compressed_poisson_pcg!(psi::AbstractVector{Float64},
                                       rhs::AbstractVector{Float64},
                                       L::CompressedLaplacian,
                                       scratch;
                                       tol::Float64 = DEFAULT_POISSON_BALANCE_TOL,
                                       max_iter::Int = DEFAULT_POISSON_BALANCE_MAX_ITER,
                                       projection::Val = Val(false))
    r  = scratch.r
    p  = scratch.p
    Ap = scratch.Ap
    z  = scratch.z

    # Enforce solvability: project rhs to range(L) (mean-zero).
    project_mean_zero!(rhs, projection)

    fill!(psi, 0.0)
    copyto!(r, rhs)

    # Jacobi preconditioner: z = M^{-1} r, M = diag(degree).
    @inbounds for c in eachindex(r)
        z[c] = L.degree[c] > 0 ? r[c] / L.degree[c] : r[c]
    end
    copyto!(p, z)

    rz_old = dot(r, z)
    rhs_linf = maximum(abs, rhs) + eps()

    iter = 0
    r_linf = rhs_linf
    while iter < max_iter
        _compressed_laplacian_mul!(Ap, p, L)
        pAp = dot(p, Ap)
        pAp <= 0 && break
        alpha = rz_old / pAp
        @. psi += alpha * p
        @. r   -= alpha * Ap

        project_mean_zero!(r, projection)

        r_linf = maximum(abs, r)
        r_linf / rhs_linf < tol && break

        @inbounds for c in eachindex(r)
            z[c] = L.degree[c] > 0 ? r[c] / L.degree[c] : r[c]
        end
        project_mean_zero!(z, projection)
        rz_new = dot(r, z)
        rz_old == 0.0 && break
        beta = rz_new / rz_old
        @. p = z + beta * p
        rz_old = rz_new
        iter += 1
    end

    project_mean_zero!(psi, projection)
    return r_linf, iter
end

"""
    _target_outflow(m_cur, m_next, c, k, inv_scale)

Outflow divergence of cell `c` at level `k` that closes continuity over one
window: a cell exports what it loses, `(m_cur − m_next) / (2 · steps)`, since
the palindrome applies each window flux twice per substep. Same target as the
cubed-sphere balance (`cs_poisson_balance.jl`).
"""
@inline _target_outflow(m_cur, m_next, c, k, inv_scale) = (m_cur[c, k] - m_next[c, k]) * inv_scale

# Column residual of the continuity target, `D_c(Σ_k F_k) − Σ_k T_{c,k}`, into
# `res` (length `nc`). `D_c(F) = Σ_{left(f) = c} F_f − Σ_{right(f) = c} F_f` is
# the outflow divergence; polar stub faces (a zero neighbour) count for their
# real cell only, as in the runtime face-indexed sweep.
function _column_balance_residual!(res::AbstractVector{Float64},
                                   hflux::AbstractMatrix{Float64},
                                   m_cur::AbstractMatrix{Float64},
                                   m_next::AbstractMatrix{Float64},
                                   face_left::Vector{Int32},
                                   face_right::Vector{Int32},
                                   inv_scale::Float64)
    fill!(res, 0.0)
    @inbounds for k in axes(hflux, 2), f in eachindex(face_left)
        flux = hflux[f, k]
        left  = Int(face_left[f])
        right = Int(face_right[f])
        left > 0 && (res[left]  += flux)
        right > 0 && (res[right] -= flux)
    end
    @inbounds for k in axes(m_cur, 2), c in axes(m_cur, 1)
        res[c] -= _target_outflow(m_cur, m_next, c, k, inv_scale)
    end
    return res
end

"""
    balance_compressed_column_fluxes!(hflux, m_cur, m_next, face_left, face_right,
                                      L::CompressedLaplacian, steps_per_window, scratch;
                                      weights = MassWeightedColumn(),
                                      tol = DEFAULT_POISSON_BALANCE_TOL,
                                      max_iter = DEFAULT_POISSON_BALANCE_MAX_ITER)

Column Poisson balance of the face-indexed horizontal fluxes `hflux[f, k]` of
a reduced-Gaussian window (kg per application, twice per substep; `m_cur` and
`m_next` are the window's endpoint air masses, `(ncells, Nz)`, top first).

One window of `n = steps_per_window` substeps, whose palindrome applies each
window flux twice, must take `m_cur` to `m_next`:

```math
m^{\\mathrm{next}}_{c,k} = m^{\\mathrm{cur}}_{c,k}
    - 2n \\left[ D_c(F_k) + C_{c,k+1} - C_{c,k} \\right],
\\qquad
D_c(F) = \\sum_{f:\\,\\mathrm{left}(f) = c} F_f - \\sum_{f:\\,\\mathrm{right}(f) = c} F_f ,
```

with the vertical mass flux `C` (`cm`) zero at the top and at the surface.
Summed over a column the `C` terms cancel, so the horizontal fluxes must
satisfy the column budget

```math
D_c\\Big(\\sum_k F_k\\Big) = T_c \\equiv \\sum_k \\frac{m^{\\mathrm{cur}}_{c,k} - m^{\\mathrm{next}}_{c,k}}{2n}.
```

One Jacobi-preconditioned CG solve of `L ψ = D(Σ_k F_k) − T` (`L` the graph
Laplacian with face multiplicities, `CompressedLaplacian`; the right-hand side
removes the global mode in proportion to current column mass only with
`fixed_global_mass=true`, after checking the independent fixed-target global
budget; otherwise it retains the uniform mode. It is then accurately projected
to zero mean) gives every
interior face the column correction `δ_f = ψ_{right(f)} − ψ_{left(f)}`, which
changes the column divergence by `−Lψ`. Each `δ_f` is spread over the layers
by `spread_column_correction!` (air mass of the two cells for
`MassWeightedColumn()`). The vertical mass flux then follows from the explicit
tendency, `recompute_cm_from_dm_target!`, and carries the vertical motion. The
lat-lon (`balance_column_mass_fluxes!`) and cubed-sphere
(`balance_cs_column_mass_fluxes!`) paths balance the same way.

Balancing every level to its own tendency instead, `D_c(F_k) = T_{c,k}`,
leaves `cm` only the column residual: about 1e-4 of the vertical mass flux on
the O24 test day (issue #11). The CG works on the compressed Laplacian; the
corrections go to every face-indexed flux, LCM duplicates included, which is
equivalent because `ψ` is cell-valued.

Returns the column residuals before and after (`max_pre_raw_residual`,
`max_post_raw_residual`, kg per application), the residual mean the solve
cannot remove (`max_rhs_mean`), the projected residuals (`max_pre_projected`,
`max_post_projected`, the latter `|Lψ − rhs|` of the solve), the CG iterations
and the largest face correction (`max_face_delta`).
"""
function balance_compressed_column_fluxes!(hflux::AbstractMatrix{Float64},
                                           m_cur::AbstractMatrix{Float64},
                                           m_next::AbstractMatrix{Float64},
                                           face_left::Vector{Int32},
                                           face_right::Vector{Int32},
                                           L::CompressedLaplacian,
                                           steps_per_window::Int,
                                           scratch;
                                           weights::AbstractColumnWeights = MassWeightedColumn(),
                                           tol::Float64 = DEFAULT_POISSON_BALANCE_TOL,
                                           max_iter::Int = DEFAULT_POISSON_BALANCE_MAX_ITER,
                                           direct_solver = nothing,
                                           fixed_global_mass::Bool = false,
                                           mass_precision::Type = eltype(m_cur))
    nc, Nz = size(m_cur)
    size(m_next) == size(m_cur) ||
        throw(DimensionMismatch("m_next $(size(m_next)) ≠ m_cur $(size(m_cur))"))
    size(hflux) == (length(face_left), Nz) ||
        throw(DimensionMismatch("hflux $(size(hflux)) ≠ (nfaces, Nz) = $((length(face_left), Nz))"))
    check_column_weight_levels(weights, Nz)
    fixed_global_mass && check_fixed_global_mass(m_cur, m_next; mass_type=mass_precision)
    inv_scale = 1.0 / (2.0 * steps_per_window)
    psi = scratch.psi
    rhs = scratch.rhs

    # 1. Column residual of the continuity target.
    _column_balance_residual!(rhs, hflux, m_cur, m_next, face_left, face_right, inv_scale)
    pre_raw = maximum(abs, rhs)
    rhs_mean = +(_neumaier_sum(c -> rhs[c], 1:nc, Float64)...) / nc
    col_mass = [ +(_neumaier_sum(k -> m_cur[c, k], 1:Nz, Float64)...) for c in 1:nc ]
    mean_mass = +(_neumaier_sum(c -> col_mass[c], 1:nc, Float64)...) / nc
    # Shift only the Poisson target; cm still uses the original endpoints.
    if fixed_global_mass
        @inbounds for c in 1:nc
            rhs[c] -= rhs_mean * (col_mass[c] / mean_mass - 1.0)
        end
    end
    pre_proj = 0.0
    @inbounds for c in 1:nc
        pre_proj = max(pre_proj, abs(rhs[c] - rhs_mean))
    end
    pre_proj < tol && return (; max_pre_raw_residual = pre_raw, max_rhs_mean = abs(rhs_mean),
                              max_pre_projected = pre_proj, max_post_projected = 0.0,
                              max_post_raw_residual = pre_raw, max_cg_iter = 0,
                              max_face_delta = 0.0)

    # 2. Solve L ψ = rhs (the solver projects rhs to zero mean in place).
    iter = if direct_solver === nothing
        _, niter = solve_compressed_poisson_pcg!(psi, rhs, L, scratch; tol = tol, max_iter = max_iter, projection=Val(true))
        niter
    else
        solve_compressed_poisson_direct!(psi, rhs, direct_solver; projection=Val(true))
        1
    end
    Lpsi = scratch.Ap
    _compressed_laplacian_mul!(Lpsi, psi, L)
    post_proj = 0.0
    @inbounds for c in 1:nc
        post_proj = max(post_proj, abs(Lpsi[c] - rhs[c]))
    end

    # 3. Spread each face's column correction over its layers.
    max_face_delta = 0.0
    @inbounds for f in eachindex(face_left)
        left  = Int(face_left[f])
        right = Int(face_right[f])
        (left > 0 && right > 0) || continue
        delta = psi[right] - psi[left]
        max_face_delta = max(max_face_delta, abs(delta))
        delta == 0.0 && continue
        spread_column_correction!(view(hflux, f, :), delta, weights,
                                  view(m_cur, left, :), view(m_cur, right, :))
    end

    # 4. Column residual after the correction (its mean stays).
    post = _column_balance_residual!(scratch.r, hflux, m_cur, m_next, face_left, face_right,
                                     inv_scale)
    return (; max_pre_raw_residual = pre_raw, max_rhs_mean = abs(rhs_mean),
            max_pre_projected = pre_proj, max_post_projected = post_proj,
            max_post_raw_residual = maximum(abs, post), max_cg_iter = iter,
            max_face_delta = max_face_delta)
end
