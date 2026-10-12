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
                                    tol=DEFAULT_POISSON_BALANCE_TOL,
                                    max_iter=DEFAULT_POISSON_BALANCE_MAX_ITER)

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
    tol::Float64=DEFAULT_POISSON_BALANCE_TOL,
    max_iter::Int=DEFAULT_POISSON_BALANCE_MAX_ITER,
    project_every::Int=DEFAULT_POISSON_PROJECT_EVERY,
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
@inline function _finite_nonnegative_weight(m)
    x = Float64(m)
    isfinite(x) || throw(DomainError(m, "column-balance weights must be finite"))
    return max(0.0, x)
end

@inline _layer_weight(::MassWeightedColumn, m_a, m_b, k) =
    _finite_nonnegative_weight(m_a) + _finite_nonnegative_weight(m_b)
@inline _layer_weight(w::HybridBWeightedColumn, m_a, m_b, k) = w.dB[k]
@inline _layer_weight(w::HybridMassWeightedColumn, m_a, m_b, k) =
    w.hybrid[k] ? _layer_weight(MassWeightedColumn(), m_a, m_b, k) : 0.0
@inline _cell_weight(::MassWeightedColumn, m, k) = _finite_nonnegative_weight(m)
@inline _cell_weight(w::HybridBWeightedColumn, m, k) = w.dB[k]
@inline _cell_weight(w::HybridMassWeightedColumn, m, k) =
    w.hybrid[k] ? _finite_nonnegative_weight(m) : 0.0

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

"""
    spread_column_correction!(flux, delta, weights, m_a, m_b)

Add the column correction `delta` (kg per flux application) of one face to the
face's layer fluxes `flux[k]`, `k = 1 … Nz` (top first), with `m_a[k]` and
`m_b[k]` the air masses of the two cells that share the face:

```math
\\Delta F_k = \\delta \\, \\frac{w_k}{\\sum_{j=1}^{N_z} w_j},
```

where the finite, non-negative layer weight `w_k` comes from `weights` (see
`AbstractColumnWeights`): `w_k = max(m_{a,k},0) + max(m_{b,k},0)` for
`MassWeightedColumn()`. Each increment is rounded to the flux type. The last
level with positive weight receives `δ` minus the *realized* changes of the
other levels, so a hybrid mask can never leak a correction into an excluded
level. A non-finite weight or `Σ_j w_j = 0` is an error: no physically defined
vertical distribution exists in that case.

This is the vertical distribution step of every column Poisson balance (the
lat-lon `balance_column_mass_fluxes!`, the reduced-Gaussian
`balance_compressed_column_fluxes!` and the cubed-sphere
`balance_cs_column_mass_fluxes!`).
"""
@inline function spread_column_correction!(flux::AbstractVector{FT}, delta::Float64,
                                           weights::AbstractColumnWeights,
                                           m_a::AbstractVector, m_b::AbstractVector) where FT
    Nz = length(flux)
    denom = 0.0
    anchor = 0
    @inbounds for k in 1:Nz
        w = _layer_weight(weights, m_a[k], m_b[k], k)
        denom += w
        w > 0.0 && (anchor = k)
    end
    denom > 0.0 || throw(DomainError(denom,
        "column-balance face weights sum to zero; cannot distribute a correction"))
    applied = 0.0
    @inbounds for k in 1:Nz
        k == anchor && continue
        w = _layer_weight(weights, m_a[k], m_b[k], k)
        w == 0.0 && continue
        old = flux[k]
        flux[k] = old + FT(delta * w / denom)
        applied += Float64(flux[k]) - Float64(old)
    end
    @inbounds flux[anchor] += FT(delta - applied)
    return flux
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
    _check_weight_levels(weights, Nz)
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
        lp = (left - 1) ÷ (Nc * Nc) + 1
        rp = (right - 1) ÷ (Nc * Nc) + 1
        ll = (left - 1) % (Nc * Nc)
        rl = (right - 1) % (Nc * Nc)
        li, lj = ll % Nc + 1, ll ÷ Nc + 1
        ri, rj = rl % Nc + 1, rl ÷ Nc + 1
        flux = dir == 1 ? view(panels_am[p], i, j, :) : view(panels_bm[p], i, j, :)
        spread_column_correction!(flux, delta, weights,
                                  view(panels_m[lp], li, lj, :),
                                  view(panels_m[rp], ri, rj, :))
    end
    return max_face_delta
end

# Conservative typed-redistribution roundoff bound. The Float64 column solve
# itself has its configured absolute tolerance; storing a correction in FT can
# round both the increment and the addition at every level. Eight epsilons per
# incident stored face value bounds those operations with margin.
function _cs_column_roundoff_tolerance(panels_am::NTuple{6, Array{FT, 3}},
                                       panels_bm::NTuple{6, Array{FT, 3}},
                                       ft::CSGlobalFaceTable,
                                       scratch::CSPoissonScratch) where FT
    budget = scratch.div
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
        face_budget = factor * scale
        budget[Int(ft.face_left[f])] += face_budget
        budget[Int(ft.face_right[f])] += face_budget
    end
    return maximum(budget)
end

function _require_cs_column_balance(diag, context::AbstractString)
    residual = diag.final_column_projected_residual
    tolerance = diag.final_column_tolerance
    isfinite(residual) && residual <= tolerance || error(
        "$context: final projected column-balance residual $residual kg exceeds " *
        "the typed closure tolerance $tolerance kg")
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

The column equation is

```math
r_c = ∑_{f∂c} s_{cf}∑_k F_{f,k}
      - \\frac{∑_k(m_{c,k}-m'_{c,k})}{2N_s}, \\qquad
r_c^\\perp = r_c - \\frac{1}{N_c}∑_d r_d .
```

Corrections repeat until `max_c |r_c^⊥| ≤ τ` or `closure_passes` is
exhausted, in which case this function raises. The typed tolerance is

```math
τ = \\max\\!\\left(\\mathtt{closure\\_tol},
  8\\,\\epsilon_{FT}\\max_c∑_{f∂c}∑_k |F_{f,k}|\\right),
```

which is the requested absolute tolerance for Float64 and a conservative
representability bound for Float32 redistribution. The returned
`final_column_projected_residual` and `final_column_tolerance` must still be
checked by production callers with `_require_cs_column_balance`.

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
    closure_tol::Float64=10.0,
    weights::AbstractColumnWeights=MassWeightedColumn(),
) where FT
    Nc = ft.Nc
    Nz = size(panels_am[1], 3)

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
    final_tolerance = closure_tol
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
            tol, max_iter, project_every)

        max_face_delta = max(max_face_delta,
            _distribute_cs_column_delta!(panels_am, panels_bm, panels_m,
                                         col_am, col_bm,
                                         col_am_before, col_bm_before,
                                         ft, Nz, weights))
        _sync_cs_mirrors!(panels_am, panels_bm, ft, Nz)

        _fill_cs_column_buffers!(col_am, col_bm, col_m, col_m_next,
                                 panels_am, panels_bm, panels_m, panels_m_next,
                                 Nc, Nz)
        final_stats = _cs_column_balance_projected_linf(
            col_am, col_bm, col_m, col_m_next, ft, steps_per_window, scratch)
        final_tolerance = max(closure_tol,
            _cs_column_roundoff_tolerance(panels_am, panels_bm, ft, scratch))
        if final_stats.projected_linf <= final_tolerance
            converged = true
            break
        end
    end

    converged || error(
        "cubed-sphere column balance failed after $closure_passes passes: " *
        "final projected residual $(final_stats.projected_linf) kg exceeds " *
        "typed closure tolerance $final_tolerance kg")

    return (;
        max_pre_residual = diag.max_pre_residual,
        max_post_residual = diag.max_post_residual,
        max_rhs_mean = diag.max_rhs_mean,
        max_pre_projected = diag.max_pre_projected,
        max_post_projected = diag.max_post_projected,
        final_column_raw_residual = final_stats.raw_linf,
        final_column_projected_residual = final_stats.projected_linf,
        final_column_mean_residual = final_stats.mean_abs,
        final_column_tolerance = final_tolerance,
        closure_passes = passes_used,
        max_cg_iter = diag.max_cg_iter,
        max_face_delta = max_face_delta,
    )
end

# ---------------------------------------------------------------------------
# Vertical mass flux diagnosis
# ---------------------------------------------------------------------------

"""
    diagnose_cs_cm!(panels_cm, panels_am, panels_bm, panels_dm, panels_m, Nc, Nz[, weights];
                    residual_tol=10.0)

Diagnose vertical mass flux `cm` from column-balanced horizontal flux divergence
and mass tendency for all 6 panels:

```math
C_{k+1} = C_k + (F^x_{i}-F^x_{i+1})
                    + (F^y_{j}-F^y_{j+1}) - \\Delta m_k,
\\qquad C_1=0.
```

The raw surface residual `R = C_{N_z+1}` is never folded silently. If
`|R| > residual_tol`, this function raises before redistribution. Otherwise it
books `R` in the returned diagnostics and applies

```math
C_{k+1} \\leftarrow C_{k+1} - R
  \\frac{∑_{j=1}^{k}w_j}{∑_{j=1}^{N_z}w_j}.
```

Weights must be finite and have a positive sum. Production callers pass the
checked `final_column_tolerance` from `balance_cs_column_mass_fluxes!`.
"""
function diagnose_cs_cm!(panels_cm::NTuple{6, Array{FT, 3}},
                          panels_am::NTuple{6, Array{FT, 3}},
                          panels_bm::NTuple{6, Array{FT, 3}},
                          panels_dm::NTuple{6, Array{FT, 3}},
                          panels_m::NTuple{6, Array{FT, 3}},
                          Nc::Int, Nz::Int,
                          weights::AbstractColumnWeights = MassWeightedColumn();
                          residual_tol::Float64 = 10.0) where FT
    _check_weight_levels(weights, Nz)
    residual_tol >= 0.0 || throw(ArgumentError("residual_tol must be non-negative"))
    max_raw_residual = 0.0
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

            raw_residual = Float64(cm[i, j, Nz + 1])
            isfinite(raw_residual) || throw(DomainError(raw_residual,
                "vertical mass-flux diagnosis residual must be finite"))
            max_raw_residual = max(max_raw_residual, abs(raw_residual))
        end
    end

    max_raw_residual <= residual_tol || error(
        "vertical mass-flux diagnosis residual $max_raw_residual kg exceeds " *
        "the booked tolerance $residual_tol kg")

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
    return (; max_raw_residual, max_booked_residual,
            residual_tolerance = residual_tol)
end

"""
    _diagnose_balanced_cs_cm!(..., balance_diag, context[, weights])

Production bridge between column balance and vertical-flux diagnosis. It
explicitly checks `balance_diag.final_column_projected_residual` against
`balance_diag.final_column_tolerance`, then passes that same typed tolerance to
`diagnose_cs_cm!`, whose returned diagnostic books any residual folded into
`cm`.
"""
function _diagnose_balanced_cs_cm!(panels_cm::NTuple{6, Array{FT, 3}},
                                   panels_am::NTuple{6, Array{FT, 3}},
                                   panels_bm::NTuple{6, Array{FT, 3}},
                                   panels_dm::NTuple{6, Array{FT, 3}},
                                   panels_m::NTuple{6, Array{FT, 3}},
                                   Nc::Int, Nz::Int,
                                   balance_diag, context::AbstractString,
                                   weights::AbstractColumnWeights = MassWeightedColumn()) where FT
    _require_cs_column_balance(balance_diag, context)
    cm_diag = diagnose_cs_cm!(panels_cm, panels_am, panels_bm, panels_dm,
                              panels_m, Nc, Nz, weights;
                              residual_tol = balance_diag.final_column_tolerance)
    cm_diag.max_raw_residual <= cm_diag.residual_tolerance || error(
        "$context: diagnosed cm residual $(cm_diag.max_raw_residual) kg exceeds " *
        "the booked tolerance $(cm_diag.residual_tolerance) kg")
    return cm_diag
end
