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
                                  tol=1e-14, max_iter=20000)

Solve the singular system `L · psi = rhs` via Jacobi-Preconditioned CG
using the compressed Laplacian. Mathematically identical to
`solve_graph_poisson_pcg!` but ~16-27× faster due to compressed MatVec.

The compressed Laplacian has the same 1-D constant null space as the
face-indexed version, so the same mean-zero projection is applied.
"""
function solve_compressed_poisson_pcg!(psi::AbstractVector{Float64},
                                       rhs::AbstractVector{Float64},
                                       L::CompressedLaplacian,
                                       scratch;
                                       tol::Float64 = 1e-14,
                                       max_iter::Int = 20000)
    r  = scratch.r
    p  = scratch.p
    Ap = scratch.Ap
    z  = scratch.z

    # Enforce solvability: project rhs to range(L) (mean-zero).
    _project_mean_zero!(rhs)

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

        _project_mean_zero!(r)

        r_linf = maximum(abs, r)
        r_linf / rhs_linf < tol && break

        @inbounds for c in eachindex(r)
            z[c] = L.degree[c] > 0 ? r[c] / L.degree[c] : r[c]
        end
        _project_mean_zero!(z)
        rz_new = dot(r, z)
        rz_old == 0.0 && break
        beta = rz_new / rz_old
        @. p = z + beta * p
        rz_old = rz_new
        iter += 1
    end

    _project_mean_zero!(psi)
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

"""
    balance_compressed_horizontal_fluxes!(hflux, m_cur, m_next,
                                          face_left, face_right,
                                          L::CompressedLaplacian,
                                          steps_per_window, scratch;
                                          tol, max_iter)

Drop-in replacement for `balance_reduced_horizontal_fluxes!` that uses
the compressed Laplacian for the CG solver while still applying flux
corrections to the full face-indexed `hflux` array.

The compressed Laplacian is mathematically equivalent to the face-indexed
graph Laplacian — the flux corrections `hflux[f] += ψ[right] - ψ[left]`
produce identical results because ψ is cell-valued.
"""
function balance_compressed_horizontal_fluxes!(hflux::AbstractMatrix{Float64},
                                               m_cur::AbstractMatrix{Float64},
                                               m_next::AbstractMatrix{Float64},
                                               face_left::Vector{Int32},
                                               face_right::Vector{Int32},
                                               L::CompressedLaplacian,
                                               steps_per_window::Int,
                                               scratch;
                                               tol::Float64 = 1e-14,
                                               max_iter::Int = 20000)
    nc, Nz = size(m_cur)
    size(hflux, 2) == Nz || error("hflux Nz mismatch with m")
    inv_scale = 1.0 / (2.0 * steps_per_window)

    max_pre_raw = 0.0
    max_rhs_mean = 0.0
    max_pre_proj = 0.0
    max_post_proj = 0.0
    max_post_raw = 0.0
    max_it = 0

    psi = scratch.psi
    rhs = scratch.rhs
    div = scratch.r   # reuse CG residual buffer for divergence

    for k in 1:Nz
        # 1. Compute current horizontal divergence at level k.
        fill!(div, 0.0)
        @inbounds for f in eachindex(face_left)
            flux = hflux[f, k]
            left  = Int(face_left[f])
            right = Int(face_right[f])
            left > 0 && (div[left]  += flux)
            right > 0 && (div[right] -= flux)
        end

        # 2. Residual of the continuity target: outflow = mass loss per substep.
        @inbounds for c in 1:nc
            rhs[c] = div[c] - _target_outflow(m_cur, m_next, c, k, inv_scale)
        end

        # Diagnostics: pre-balance residual
        rhs_sum = 0.0
        rhs_raw_linf = 0.0
        @inbounds for c in 1:nc
            rhs_sum += rhs[c]
            a = abs(rhs[c])
            a > rhs_raw_linf && (rhs_raw_linf = a)
        end
        rhs_mean = rhs_sum / nc
        rhs_raw_linf > max_pre_raw && (max_pre_raw = rhs_raw_linf)
        a_mean = abs(rhs_mean)
        a_mean > max_rhs_mean && (max_rhs_mean = a_mean)

        pre_proj = 0.0
        @inbounds for c in 1:nc
            a = abs(rhs[c] - rhs_mean)
            a > pre_proj && (pre_proj = a)
        end
        pre_proj > max_pre_proj && (max_pre_proj = pre_proj)

        pre_proj < tol && continue

        # 3. Solve L · ψ = rhs using compressed Laplacian CG.
        _, it = solve_compressed_poisson_pcg!(psi, rhs, L, scratch;
                                              tol=tol, max_iter=max_iter)
        it > max_it && (max_it = it)

        # 4. Post-solve residual check on range(L).
        Lpsi = scratch.Ap
        _compressed_laplacian_mul!(Lpsi, psi, L)
        fill!(div, 0.0)
        @inbounds for f in eachindex(face_left)
            flux = hflux[f, k]
            left  = Int(face_left[f])
            right = Int(face_right[f])
            left > 0 && (div[left]  += flux)
            right > 0 && (div[right] -= flux)
        end
        @inbounds for c in 1:nc
            rhs[c] = (div[c] - _target_outflow(m_cur, m_next, c, k, inv_scale)) - rhs_mean
        end
        post_proj = 0.0
        @inbounds for c in 1:nc
            a = abs(Lpsi[c] - rhs[c])
            a > post_proj && (post_proj = a)
        end
        post_proj > max_post_proj && (max_post_proj = post_proj)

        # 5. Apply flux correction to ALL faces (including LCM duplicates).
        @inbounds for f in eachindex(face_left)
            left  = Int(face_left[f])
            right = Int(face_right[f])
            if left > 0 && right > 0
                hflux[f, k] += psi[right] - psi[left]
            end
        end

        # 6. Post-balance raw residual.
        fill!(div, 0.0)
        @inbounds for f in eachindex(face_left)
            flux = hflux[f, k]
            left  = Int(face_left[f])
            right = Int(face_right[f])
            left > 0 && (div[left]  += flux)
            right > 0 && (div[right] -= flux)
        end
        post_raw = 0.0
        @inbounds for c in 1:nc
            r = abs(div[c] - _target_outflow(m_cur, m_next, c, k, inv_scale))
            r > post_raw && (post_raw = r)
        end
        post_raw > max_post_raw && (max_post_raw = post_raw)
    end

    return (;
        max_pre_raw_residual = max_pre_raw,
        max_rhs_mean = max_rhs_mean,
        max_pre_projected = max_pre_proj,
        max_post_projected = max_post_proj,
        max_post_raw_residual = max_post_raw,
        max_cg_iter = max_it,
        max_pre_residual = max_pre_raw,
        max_post_residual = max_post_raw,
    )
end
