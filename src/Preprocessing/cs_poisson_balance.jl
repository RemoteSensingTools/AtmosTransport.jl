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
