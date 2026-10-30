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

Air-mass weights restricted to the hybrid layers of interface `B` (top first):
layer `k` is hybrid when `ΔB_k = B_{k+1} − B_k ≥ HYBRID_LAYER_MIN_DB` (1e-6),
i.e. when a 1000 hPa change of surface pressure changes its thickness by at
least 0.1 Pa. Smaller positive increments are rounding residues of a
pure-pressure layer (GEOS/MERRA-2 L72 layer 41, 150–177 hPa: `ΔB = 8.2e-9`,
8.2e-4 Pa per 1000 hPa, 3e-7 of its thickness); the smallest genuine increment
of the shipped tables is ERA5 L137 layer 55 (`ΔB = 7e-6`, 0.7 Pa per
1000 hPa). A coordinate with no hybrid layer is an error.
"""
struct HybridMassWeightedColumn <: AbstractColumnWeights
    hybrid :: BitVector            # layers with ΔB ≥ HYBRID_LAYER_MIN_DB, top first
    function HybridMassWeightedColumn(B::AbstractVector{<:AbstractFloat})
        hybrid = _hybrid_layer_dB(B) .>= HYBRID_LAYER_MIN_DB
        any(hybrid) || throw(ArgumentError(
            "hybrid B has no layer with ΔB ≥ $(HYBRID_LAYER_MIN_DB); " *
            "hybrid_mass column weights need at least one hybrid layer"))
        return new(hybrid)
    end
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

# The column weights as the residual weight of the explicit-tendency closure
# `recompute_cm_from_dm_target!` (lat-lon and reduced Gaussian), so a column's
# residual is folded into the same levels as its horizontal correction.
struct ColumnResidualWeight{W<:AbstractColumnWeights} <: Function
    weights :: W
end
(r::ColumnResidualWeight)(m, k) = _cell_weight(r.weights, m, k)

# Number of layers the weights are defined for (`nothing`: any).
_weight_levels(::MassWeightedColumn) = nothing
_weight_levels(w::HybridBWeightedColumn) = length(w.dB)
_weight_levels(w::HybridMassWeightedColumn) = length(w.hybrid)

function check_column_weight_levels(w::AbstractColumnWeights, Nz)
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
