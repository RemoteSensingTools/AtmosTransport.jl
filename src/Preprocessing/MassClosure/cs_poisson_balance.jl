# Canonical face order is part of the numerical contract.
function cs_face_divergence!(div::AbstractVector{Float64}, panels_am, panels_bm,
                             ft::CSGlobalFaceTable, k::Int)
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
    return div
end

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
# High-level balance entry point
# ---------------------------------------------------------------------------

function balance_cs_level!(
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
    mass_weighted_global_mode::Bool = false,
) where FT
    Nc = ft.Nc
    nc = ft.nc
    div = scratch.div
    rhs = scratch.rhs
    psi = scratch.psi
    cg_scratch = (r = scratch.r, p = scratch.p, Ap = scratch.Ap, z = scratch.z)

    # 1. Compute current horizontal divergence.
    cs_face_divergence!(div, panels_am, panels_bm, ft, k)

    # 2. RHS = divergence - target mass tendency.
    rhs_sum = 0.0
    @inbounds for c in 1:nc
        i_local, j_local, p_idx = cs_panel_ij(c, Nc)
        target = (Float64(panels_m[p_idx][i_local, j_local, k]) -
                  Float64(panels_m_next[p_idx][i_local, j_local, k])) * inv_scale
        rc = div[c] - target
        rhs[c] = rc
        rhs_sum += rc
    end

    rhs_raw_linf = cs_linf(rhs)
    rhs_mean = mass_weighted_global_mode ? +(_neumaier_sum(c -> rhs[c], 1:nc, Float64)...) / nc : rhs_sum / nc
    mean_mass = 1.0
    if mass_weighted_global_mode
        mass_sum = _neumaier_sum(1:nc, Float64) do c
            i, j, p = cs_panel_ij(c, Nc)
            Float64(panels_m[p][i, j, k])
        end
        mean_mass = +(mass_sum...) / nc
    end
    if mass_weighted_global_mode
        @inbounds for c in 1:nc
            i, j, p = cs_panel_ij(c, Nc)
            q = Float64(panels_m[p][i, j, k]) / mean_mass
            rhs[c] -= rhs_mean * (q - 1.0)
        end
    end
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
                                  project_every=project_every, projection=Val(mass_weighted_global_mode))

    # Diagnostic: post-solve projected residual.
    Lpsi = scratch.Ap
    cs_graph_laplacian_mul!(Lpsi, psi, ft, degree)
    cs_face_divergence!(div, panels_am, panels_bm, ft, k)
    @inbounds for c in 1:nc
        i_local, j_local, p_idx = cs_panel_ij(c, Nc)
        target = (Float64(panels_m[p_idx][i_local, j_local, k]) -
                  Float64(panels_m_next[p_idx][i_local, j_local, k])) * inv_scale
        q = mass_weighted_global_mode ? Float64(panels_m[p_idx][i_local, j_local, k]) / mean_mass : 1.0
        rhs[c] = (div[c] - target) - rhs_mean * q
    end
    post_proj = 0.0
    @inbounds @simd for c in 1:nc
        a = abs(Lpsi[c] - rhs[c])
        post_proj = ifelse(a > post_proj, a, post_proj)
    end

    # 4. Apply correction to all faces at this level.
    apply_cs_flux_correction!(panels_am, panels_bm, psi, ft, k)

    # 5. Post-balance raw residual.
    cs_face_divergence!(div, panels_am, panels_bm, ft, k)
    post_raw = 0.0
    @inbounds for c in 1:nc
        i_local, j_local, p_idx = cs_panel_ij(c, Nc)
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
                                    tol=DEFAULT_POISSON_BALANCE_TOL,
                                    max_iter=DEFAULT_POISSON_BALANCE_MAX_ITER)

TM5-style global Poisson mass-flux balance for a 6-panel cubed sphere.

Corrects `panels_am[p]` and `panels_bm[p]` so that horizontal flux
convergence at every cell matches the prescribed mass tendency:

    dm_dt[c, k] = (m_next[c, k] - m_cur[c, k]) / (2 × steps_per_window)

Returns a diagnostic NamedTuple with pre/post residuals and CG iteration counts.
`mass_weighted_global_mode=true` is used on the single-level column buffers by
`balance_cs_column_mass_fluxes!`; the standalone per-level balance defaults to
its original uniform global mode.
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
    tol::Float64=DEFAULT_POISSON_BALANCE_TOL,
    max_iter::Int=DEFAULT_POISSON_BALANCE_MAX_ITER,
    project_every::Int=DEFAULT_POISSON_PROJECT_EVERY,
    mass_weighted_global_mode::Bool=false,
) where FT

    Nc = ft.Nc
    Nz = size(panels_am[1], 3)
    nc = ft.nc
    inv_scale = 1.0 / (2.0 * steps_per_window)

    scratches = cs_level_scratches(scratch, nc)

    pre_by_level = zeros(Float64, Nz)
    post_by_level = zeros(Float64, Nz)
    rhs_mean_by_level = zeros(Float64, Nz)
    pre_proj_by_level = zeros(Float64, Nz)
    post_proj_by_level = zeros(Float64, Nz)
    iter_by_level = zeros(Int, Nz)

    Threads.@threads :static for k in 1:Nz
        diag = balance_cs_level!(
            k, panels_am, panels_bm, panels_m, panels_m_next,
            ft, degree, scratches[Threads.threadid()], inv_scale;
            tol=tol, max_iter=max_iter, project_every=project_every,
            mass_weighted_global_mode)
        pre_by_level[k] = diag.pre
        post_by_level[k] = diag.post
        rhs_mean_by_level[k] = diag.rhs_mean
        pre_proj_by_level[k] = diag.pre_proj
        post_proj_by_level[k] = diag.post_proj
        iter_by_level[k] = diag.iter
    end

    # 6. Synchronize ALL cross-panel mirror entries at ALL levels.
    sync_cs_mirrors!(panels_am, panels_bm, ft, Nz)

    return (;
        max_pre_residual = maximum(pre_by_level),
        max_post_residual = maximum(post_by_level),
        max_rhs_mean = maximum(rhs_mean_by_level),
        max_pre_projected = maximum(pre_proj_by_level),
        max_post_projected = maximum(post_proj_by_level),
        max_cg_iter = maximum(iter_by_level),
    )
end

function _fill_cs_column_buffers!(col_am::NTuple{6, Array{BT, 3}},
                                  col_bm::NTuple{6, Array{BT, 3}},
                                  col_m::NTuple{6, Array{BT, 3}},
                                  col_m_next::NTuple{6, Array{BT, 3}},
                                  panels_am::NTuple{6, Array{FT, 3}},
                                  panels_bm::NTuple{6, Array{FT, 3}},
                                  panels_m::NTuple{6, Array{FT, 3}},
                                  panels_m_next::NTuple{6, Array{FT, 3}},
                                  Nc::Int, Nz::Int) where {BT, FT}
    for p in 1:6
        fill!(col_am[p], zero(BT))
        fill!(col_bm[p], zero(BT))
        fill!(col_m[p], zero(BT))
        fill!(col_m_next[p], zero(BT))
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

# |r| / τ, with an exactly closed cell (0/0, possible for closure_tol = 0 and
# a cell without flux) counted as closed.
@inline _residual_ratio(r::Float64, tol::Float64) = r == 0 ? 0.0 : r / tol

# Column residuals of the stored fluxes after a closure pass: the raw and
# mass-weighted-global-mode-free maxima, the mean, and the largest ratio of a cell's
# projected residual to its own tolerance `scratch.tol[c]`.
function cs_column_balance_projected_linf(col_am::NTuple{6, Array{FT, 3}},
                                           col_bm::NTuple{6, Array{FT, 3}},
                                           col_m::NTuple{6, Array{FT, 3}},
                                           col_m_next::NTuple{6, Array{FT, 3}},
                                           ft::CSGlobalFaceTable,
                                           steps_per_window::Int,
                                           scratch::CSPoissonScratch;
                                           mass_weighted_global_mode::Bool=false) where FT
    Nc = ft.Nc
    inv_scale = 1.0 / (2.0 * steps_per_window)
    div = scratch.div
    cs_face_divergence!(div, col_am, col_bm, ft, 1)

    raw_linf = 0.0
    @inbounds for c in 1:ft.nc
        i_local, j_local, p_idx = cs_panel_ij(c, Nc)
        target = (Float64(col_m[p_idx][i_local, j_local, 1]) -
                  Float64(col_m_next[p_idx][i_local, j_local, 1])) * inv_scale
        r = div[c] - target
        div[c] = r
        raw_linf = max(raw_linf, abs(r))
    end
    mean = +(_neumaier_sum(c -> div[c], 1:ft.nc, Float64)...) / ft.nc
    mass_sum = _neumaier_sum(1:ft.nc, Float64) do c
        i, j, p = cs_panel_ij(c, Nc)
        Float64(col_m[p][i, j, 1])
    end
    mean_mass = +(mass_sum...) / ft.nc
    projected_linf = 0.0
    worst_ratio = 0.0
    tol = scratch.tol
    @inbounds for c in 1:ft.nc
        i, j, p = cs_panel_ij(c, Nc)
        q = mass_weighted_global_mode ? Float64(col_m[p][i, j, 1]) / mean_mass : 1.0
        r = abs(div[c] - mean * q)
        projected_linf = max(projected_linf, r)
        worst_ratio = max(worst_ratio, _residual_ratio(r, tol[c]))
    end
    return (raw_linf = raw_linf, projected_linf = projected_linf,
            mean_abs = abs(mean), worst_ratio = worst_ratio)
end



function _distribute_cs_column_delta!(panels_am::NTuple{6, Array{FT, 3}},
                                      panels_bm::NTuple{6, Array{FT, 3}},
                                      panels_m::NTuple{6, Array{FT, 3}},
                                      col_am::NTuple{6, <:Array},
                                      col_bm::NTuple{6, <:Array},
                                      col_am_before::NTuple{6, <:Array},
                                      col_bm_before::NTuple{6, <:Array},
                                      ft::CSGlobalFaceTable, Nz::Int,
                                      weights::AbstractColumnWeights = MassWeightedColumn()) where FT
    check_column_weight_levels(weights, Nz)
    Nc = ft.Nc
    max_face_delta = 0.0
    @inbounds for f in 1:ft.nf
        p = Int(ft.face_panel[f])
        dir = Int(ft.face_dir[f])
        i = Int(ft.face_idx_i[f])
        j = Int(ft.face_idx_j[f])
        col = dir == 1 ? col_am[p] : col_bm[p]
        col_before = dir == 1 ? col_am_before[p] : col_bm_before[p]
        delta = Float64(col[i, j, 1] - col_before[i, j, 1])
        max_face_delta = max(max_face_delta, abs(delta))
        delta == 0.0 && continue

        left = Int(ft.face_left[f])
        right = Int(ft.face_right[f])
        li, lj, lp = cs_panel_ij(left, Nc)
        ri, rj, rp = cs_panel_ij(right, Nc)
        flux = dir == 1 ? view(panels_am[p], i, j, :) : view(panels_bm[p], i, j, :)
        spread_column_correction!(flux, delta, weights,
                                  view(panels_m[lp], li, lj, :),
                                  view(panels_m[rp], ri, rj, :))
    end
    return max_face_delta
end

# Per-cell closure tolerance, into `scratch.tol`:
#     τ_c = max(closure_tol, 8 ε_FT Σ_{f∂c} Σ_k |F_{f,k}|),
# with the stored values `F` multiplied by `flux_scale` (per-application units).
# The Float64 column solve itself has the configured absolute tolerance;
# storing a correction in FT can round both the increment and the addition at
# every level of the cell's own faces. Eight epsilons per incident stored face
# value bounds those operations with margin. The bound is per cell: a global
# maximum would let a cell with small fluxes keep a residual set by the
# largest-flux cell of the sphere.
function cs_column_roundoff_tolerance!(scratch::CSPoissonScratch,
                                        panels_am::NTuple{6, Array{FT, 3}},
                                        panels_bm::NTuple{6, Array{FT, 3}},
                                        ft::CSGlobalFaceTable,
                                        closure_tol::Float64,
                                        flux_scale::Float64 = 1.0) where FT
    budget = scratch.tol
    fill!(budget, 0.0)
    factor = 8.0 * eps(FT)
    @inbounds for f in 1:ft.nf
        p = Int(ft.face_panel[f])
        dir = Int(ft.face_dir[f])
        i = Int(ft.face_idx_i[f])
        j = Int(ft.face_idx_j[f])
        flux = dir == 1 ? view(panels_am[p], i, j, :) : view(panels_bm[p], i, j, :)
        scale = 0.0
        for value in flux
            isfinite(value) || throw(DomainError(value,
                "column-balanced fluxes must be finite"))
            scale += abs(Float64(value))
        end
        face_budget = factor * scale * flux_scale
        budget[Int(ft.face_left[f])] += face_budget
        budget[Int(ft.face_right[f])] += face_budget
    end
    @inbounds @simd for c in eachindex(budget)
        budget[c] = max(closure_tol, budget[c])
    end
    return budget
end

function require_cs_column_balance(diag, context::AbstractString)
    ratio = diag.final_column_residual_ratio
    isfinite(ratio) && ratio <= 1 || error(
        "$context: final projected column-balance residual " *
        "$(diag.final_column_projected_residual) kg exceeds its cell's typed closure " *
        "tolerance (largest residual/tolerance ratio $ratio; per-cell tolerances " *
        "$(diag.final_column_tolerance_min)..$(diag.final_column_tolerance_max) kg)")
    return diag
end

"""
    balance_cs_column_mass_fluxes!(panels_am, panels_bm, panels_m, panels_m_next,
                                   ft, degree, steps_per_window, scratch; ...)

Apply vertically integrated CS Poisson corrections, then distribute every
physical face correction over levels with `weights` (`MassWeightedColumn()`,
local air mass, by default; `HybridBWeightedColumn` follows TM5). Seam faces
use the two cells in `ft.face_left/right`, independent of which panel owns the
canonical stored face.

With `fixed_global_mass=true`, the independent global dry-mass budget guard
runs before any flux correction. The projected column equation is

```math
r_c = ∑_{f∂c} s_{cf}∑_k F_{f,k}
      - \\frac{∑_k(m_{c,k}-m'_{c,k})}{2N_s}, \\qquad
q_c = M_c / \\operatorname{mean}(M), \\qquad
r_c^\\perp = r_c - q_c \\frac{1}{N_c}∑_d r_d .
```

For moist or unpinned windows, leave `fixed_global_mass=false` (the default):
`q_c = 1`, with no mass-proportional shift or zero-global-change assertion.

Corrections repeat until every cell has `|r_c^⊥| ≤ τ_c` or `closure_passes`
is exhausted, in which case this function raises. The typed tolerance is per
cell,

```math
τ_c = \\max\\!\\left(\\mathtt{closure\\_tol},
  8\\,\\epsilon_{FT}∑_{f∂c}∑_k |F_{f,k}|\\right),
```

the requested absolute tolerance (`COLUMN_CLOSURE_TOL_KG`, 10 kg, by default)
for Float64 and a conservative
representability bound of the cell's own stored faces for Float32
redistribution (a global maximum over cells would let a small-flux cell keep a
residual set by the largest-flux cell). The returned
`final_column_residual_ratio = max_c |r_c^⊥|/τ_c` must still be checked by
production callers with `require_cs_column_balance`; `τ_c` stays in
`scratch.tol` (`final_column_tolerance_by_cell`) for the `cm` diagnosis.

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
    tol::Float64=DEFAULT_POISSON_BALANCE_TOL,
    max_iter::Int=DEFAULT_POISSON_BALANCE_MAX_ITER,
    project_every::Int=DEFAULT_POISSON_PROJECT_EVERY,
    closure_passes::Int=8,
    closure_tol::Float64=COLUMN_CLOSURE_TOL_KG,
    weights::AbstractColumnWeights=MassWeightedColumn(),
    fixed_global_mass::Bool=false,
) where FT
    Nc = ft.Nc
    Nz = size(panels_am[1], 3)

    fixed_global_mass && check_fixed_global_mass(panels_m, panels_m_next)
    closure_passes > 0 || throw(ArgumentError("closure_passes must be positive"))
    closure_tol >= 0.0 || throw(ArgumentError("closure_tol must be non-negative"))

    # Accumulate the stored values exactly enough for the Poisson target even
    # when the payload itself is Float32; only the redistributed faces are FT.
    col_am = ntuple(_ -> zeros(Float64, Nc + 1, Nc, 1), 6)
    col_bm = ntuple(_ -> zeros(Float64, Nc, Nc + 1, 1), 6)
    col_m = ntuple(_ -> zeros(Float64, Nc, Nc, 1), 6)
    col_m_next = ntuple(_ -> zeros(Float64, Nc, Nc, 1), 6)
    col_am_before = ntuple(_ -> zeros(Float64, Nc + 1, Nc, 1), 6)
    col_bm_before = ntuple(_ -> zeros(Float64, Nc, Nc + 1, 1), 6)

    max_face_delta = 0.0
    diag = nothing
    final_stats = nothing
    passes_used = 0
    converged = false
    for pass in 1:closure_passes
        passes_used = pass
        _fill_cs_column_buffers!(col_am, col_bm, col_m, col_m_next,
                                 panels_am, panels_bm, panels_m, panels_m_next,
                                 Nc, Nz)
        for p in 1:6
            copyto!(col_am_before[p], col_am[p])
            copyto!(col_bm_before[p], col_bm[p])
        end
        diag = balance_cs_global_mass_fluxes!(
            col_am, col_bm, col_m, col_m_next, ft, degree, steps_per_window, scratch;
            tol, max_iter, project_every, mass_weighted_global_mode=fixed_global_mass)

        max_face_delta = max(max_face_delta,
            _distribute_cs_column_delta!(panels_am, panels_bm, panels_m,
                                         col_am, col_bm,
                                         col_am_before, col_bm_before,
                                         ft, Nz, weights))
        sync_cs_mirrors!(panels_am, panels_bm, ft, Nz)

        _fill_cs_column_buffers!(col_am, col_bm, col_m, col_m_next,
                                 panels_am, panels_bm, panels_m, panels_m_next,
                                 Nc, Nz)
        cs_column_roundoff_tolerance!(scratch, panels_am, panels_bm, ft, closure_tol)
        final_stats = cs_column_balance_projected_linf(
            col_am, col_bm, col_m, col_m_next, ft, steps_per_window, scratch;
            mass_weighted_global_mode=fixed_global_mass)
        if final_stats.worst_ratio <= 1
            converged = true
            break
        end
    end

    converged || error(
        "cubed-sphere column balance failed after $closure_passes passes: " *
        "final projected residual $(final_stats.projected_linf) kg exceeds its " *
        "cell's typed closure tolerance (largest residual/tolerance ratio " *
        "$(final_stats.worst_ratio))")

    return (;
        max_pre_residual = diag.max_pre_residual,
        max_post_residual = diag.max_post_residual,
        max_rhs_mean = diag.max_rhs_mean,
        max_pre_projected = diag.max_pre_projected,
        max_post_projected = diag.max_post_projected,
        mass_weighted_global_mode = fixed_global_mass,
        final_column_raw_residual = final_stats.raw_linf,
        final_column_projected_residual = final_stats.projected_linf,
        final_column_mean_residual = final_stats.mean_abs,
        final_column_residual_ratio = final_stats.worst_ratio,
        final_column_tolerance_min = minimum(scratch.tol),
        final_column_tolerance_max = maximum(scratch.tol),
        final_column_tolerance_by_cell = scratch.tol,
        closure_passes = passes_used,
        max_cg_iter = diag.max_cg_iter,
        max_face_delta = max_face_delta,
    )
end

"""
    recheck_cs_column_closure(panels_am, panels_bm, panels_m, panels_m_next, ft,
                              steps_per_window, scratch, balance_diag;
                              closure_tol = COLUMN_CLOSURE_TOL_KG,
                              flux_scale = 1.0) -> balance_diag′

Recompute the column-closure statistics and per-cell tolerances of
`balance_cs_column_mass_fluxes!` for fluxes changed after the balance (the GEOS
OMEGA reconstruction moves flux between levels of a column; GEOS stores
full-window amounts), so the `cm` diagnosis and the recorded closure describe
the fluxes that are stored. `flux_scale` converts the given amounts to
amounts per flux application before the check (`1/(2 steps_per_window)` for
full-window amounts); residuals and tolerances are per application. Returns
`balance_diag` with its `final_column_*` fields replaced.
"""
function recheck_cs_column_closure(panels_am::NTuple{6, Array{FT, 3}},
                                   panels_bm::NTuple{6, Array{FT, 3}},
                                   panels_m::NTuple{6, Array{FT, 3}},
                                   panels_m_next::NTuple{6, Array{FT, 3}},
                                   ft::CSGlobalFaceTable, steps_per_window::Int,
                                   scratch::CSPoissonScratch, balance_diag;
                                   closure_tol::Float64 = COLUMN_CLOSURE_TOL_KG,
                                   flux_scale::Float64 = 1.0) where FT
    isfinite(flux_scale) && flux_scale > 0 || throw(ArgumentError(
        "flux_scale must be finite and positive; got $flux_scale"))
    Nc, Nz = ft.Nc, size(panels_am[1], 3)
    col_am = ntuple(_ -> zeros(Float64, Nc + 1, Nc, 1), 6)
    col_bm = ntuple(_ -> zeros(Float64, Nc, Nc + 1, 1), 6)
    col_m = ntuple(_ -> zeros(Float64, Nc, Nc, 1), 6)
    col_m_next = ntuple(_ -> zeros(Float64, Nc, Nc, 1), 6)
    _fill_cs_column_buffers!(col_am, col_bm, col_m, col_m_next,
                             panels_am, panels_bm, panels_m, panels_m_next, Nc, Nz)
    for p in 1:6
        col_am[p] .*= flux_scale
        col_bm[p] .*= flux_scale
    end
    cs_column_roundoff_tolerance!(scratch, panels_am, panels_bm, ft, closure_tol, flux_scale)
    stats = cs_column_balance_projected_linf(col_am, col_bm, col_m, col_m_next, ft,
                                              steps_per_window, scratch;
                                              mass_weighted_global_mode=get(balance_diag, :mass_weighted_global_mode, false))
    return merge(balance_diag, (
        final_column_raw_residual = stats.raw_linf,
        final_column_projected_residual = stats.projected_linf,
        final_column_mean_residual = stats.mean_abs,
        final_column_residual_ratio = stats.worst_ratio,
        final_column_tolerance_min = minimum(scratch.tol),
        final_column_tolerance_max = maximum(scratch.tol),
        final_column_tolerance_by_cell = scratch.tol))
end

# ---------------------------------------------------------------------------
# Vertical mass flux diagnosis
# ---------------------------------------------------------------------------

"""
    diagnose_cs_cm!(panels_cm, panels_am, panels_bm, panels_dm, panels_m, Nc, Nz[, weights];
                    residual_tol=COLUMN_CLOSURE_TOL_KG, context="cm diagnosis")

Diagnose vertical mass flux `cm` from column-balanced horizontal flux divergence
and mass tendency for all 6 panels:

```math
C_{k+1} = C_k + (F^x_{i}-F^x_{i+1})
                    + (F^y_{j}-F^y_{j+1}) - \\Delta m_k,
\\qquad C_1=0.
```

The raw surface residual `R_c = C_{N_z+1}` of column `c` is never folded
silently. It must satisfy

```math
|R_c| ≤ τ_c + 2\\,ε_{FT} ∑_k \\big(|C_k| + |F^x_{i,k}| + |F^x_{i+1,k}|
        + |F^y_{j,k}| + |F^y_{j+1,k}| + |Δm_k|\\big),
```

where `τ_c` is `residual_tol` (a number, or a vector over the `6 N_c^2` cells
in face-table order) and the sum bounds the rounding of the recursion above in
the storage type `FT`; otherwise this function raises before redistribution,
naming `[mass_fix]` when the global-mean residual (a change of the global air
mass that no horizontal flux can remove) dominates. Accepted residuals are
booked in the returned diagnostics and folded back as

```math
C_{k+1} \\leftarrow C_{k+1} - R
  \\frac{∑_{j=1}^{k}w_j}{∑_{j=1}^{N_z}w_j}.
```

Weights must be finite and have a positive sum. Production callers pass the
per-cell closure tolerance of `balance_cs_column_mass_fluxes!`
(`diagnose_balanced_cs_cm!`).
"""
function diagnose_cs_cm!(panels_cm::NTuple{6, Array{FT, 3}},
                          panels_am::NTuple{6, Array{FT, 3}},
                          panels_bm::NTuple{6, Array{FT, 3}},
                          panels_dm::NTuple{6, Array{FT, 3}},
                          panels_m::NTuple{6, Array{FT, 3}},
                          Nc::Int, Nz::Int,
                          weights::AbstractColumnWeights = MassWeightedColumn();
                          residual_tol::Union{Real, AbstractVector{<:Real}} = COLUMN_CLOSURE_TOL_KG,
                          context::AbstractString = "cm diagnosis") where FT
    check_column_weight_levels(weights, Nz)
    _check_column_tolerance(residual_tol, 6 * Nc * Nc)
    max_raw_residual = 0.0
    max_ratio = 0.0
    worst = (0, 0, 0)
    worst_residual = 0.0
    worst_tolerance = 0.0
    sum_residual = 0.0
    epsFT = Float64(eps(FT))
    for p in 1:6
        am = panels_am[p]
        bm = panels_bm[p]
        cm = panels_cm[p]
        dm = panels_dm[p]

        @inbounds for j in 1:Nc, i in 1:Nc
            cm[i, j, 1] = zero(FT)
            rounding = 0.0

            for k in 1:Nz
                div_h = (am[i, j, k] - am[i + 1, j, k]) +
                        (bm[i, j, k] - bm[i, j + 1, k])
                cm[i, j, k + 1] = cm[i, j, k] + div_h - dm[i, j, k]
                rounding += abs(Float64(cm[i, j, k])) +
                            abs(Float64(am[i, j, k])) + abs(Float64(am[i + 1, j, k])) +
                            abs(Float64(bm[i, j, k])) + abs(Float64(bm[i, j + 1, k])) +
                            abs(Float64(dm[i, j, k]))
            end

            raw_residual = Float64(cm[i, j, Nz + 1])
            isfinite(raw_residual) || throw(DomainError(raw_residual,
                "vertical mass-flux diagnosis residual must be finite"))
            c = cs_flat_index(i, j, p, Nc)
            tolerance = _column_tolerance(residual_tol, c) + 2epsFT * rounding
            ratio = _residual_ratio(abs(raw_residual), tolerance)
            sum_residual += raw_residual
            max_raw_residual = max(max_raw_residual, abs(raw_residual))
            if ratio > max_ratio
                max_ratio, worst = ratio, (p, i, j)
                worst_residual, worst_tolerance = raw_residual, tolerance
            end
        end
    end
    mean_residual = sum_residual / (6 * Nc * Nc)

    if !(max_ratio <= 1)
        hint = abs(mean_residual) >= 0.5 * abs(worst_residual) ?
            " The mean column residual over the sphere is $(mean_residual) kg: a " *
            "global-mean residual is a change of the global air mass that no " *
            "horizontal flux can remove. A run without the dry-mass pin keeps it " *
            "in every column; enable [mass_fix] (enable = true) so the window " *
            "endpoints share one global dry mass." :
            " The mean column residual over the sphere is $(mean_residual) kg."
        error("$context: vertical mass-flux diagnosis residual $(worst_residual) kg " *
              "in column (panel, i, j) = $(worst) exceeds its tolerance " *
              "$(worst_tolerance) kg." * hint)
    end

    max_booked_residual = 0.0
    for p in 1:6
        cm = panels_cm[p]
        m = panels_m[p]
        @inbounds for j in 1:Nc, i in 1:Nc
            residual = Float64(cm[i, j, Nz + 1])
            if abs(residual) > eps(FT)
                total_m = 0.0
                for k in 1:Nz
                    total_m += _cell_weight(weights, m[i, j, k], k)
                end
                total_m > 0.0 || throw(DomainError(total_m,
                    "column-balance cell weights sum to zero; cannot book cm residual"))
                cum_fix = 0.0
                for k in 1:Nz
                    frac = _cell_weight(weights, m[i, j, k], k) / total_m
                    cum_fix += frac * residual
                    cm[i, j, k + 1] -= FT(cum_fix)
                end
                max_booked_residual = max(max_booked_residual, abs(residual))
            end
        end
    end
    return (; max_raw_residual, max_raw_residual_ratio = max_ratio,
            max_booked_residual, mean_raw_residual = mean_residual)
end

# A scalar column tolerance, or one per cell of the 6-panel face table.
_check_column_tolerance(tol::Real, _nc) =
    (tol >= 0 || throw(ArgumentError("residual_tol must be non-negative")); nothing)
function _check_column_tolerance(tol::AbstractVector{<:Real}, nc)
    length(tol) == nc || throw(DimensionMismatch(
        "per-cell residual_tol has $(length(tol)) entries for $nc cells"))
    all(x -> x >= 0, tol) || throw(ArgumentError("residual_tol must be non-negative"))
    return nothing
end
@inline _column_tolerance(tol::Real, _c) = Float64(tol)
@inline _column_tolerance(tol::AbstractVector{<:Real}, c) = Float64(@inbounds tol[c])

"""
    diagnose_balanced_cs_cm!(..., balance_diag, context[, weights])

Production bridge between column balance and vertical-flux diagnosis. It
explicitly checks `balance_diag.final_column_residual_ratio ≤ 1` (every cell's
projected residual within its own closure tolerance), then passes the same
per-cell tolerances (`final_column_tolerance_by_cell`) to `diagnose_cs_cm!`,
whose returned diagnostic books any residual folded into `cm`.
"""
function diagnose_balanced_cs_cm!(panels_cm::NTuple{6, Array{FT, 3}},
                                   panels_am::NTuple{6, Array{FT, 3}},
                                   panels_bm::NTuple{6, Array{FT, 3}},
                                   panels_dm::NTuple{6, Array{FT, 3}},
                                   panels_m::NTuple{6, Array{FT, 3}},
                                   Nc::Int, Nz::Int,
                                   balance_diag, context::AbstractString,
                                   weights::AbstractColumnWeights = MassWeightedColumn()) where FT
    require_cs_column_balance(balance_diag, context)
    return diagnose_cs_cm!(panels_cm, panels_am, panels_bm, panels_dm,
                           panels_m, Nc, Nz, weights;
                           residual_tol = balance_diag.final_column_tolerance_by_cell,
                           context)
end

"""Allocate independent per-thread CS solve scratch, reusing the supplied first slot."""
function cs_level_scratches(scratch::CSPoissonScratch, nc::Int)
    nthread = Threads.maxthreadid()
    scratches = Vector{CSPoissonScratch}(undef, nthread)
    scratches[1] = scratch
    for t in 2:nthread
        scratches[t] = CSPoissonScratch(nc)
    end

    return scratches
end
