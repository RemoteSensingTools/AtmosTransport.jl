# ---------------------------------------------------------------------------
# Averaging-kernel (retrieval) column operator.
#
# A retrieved column-averaged dry mole fraction (an OCO-2, GOSAT or TROPOMI
# sounding) is compared with the model through the retrieval's own vertical
# description: its pressures, pressure weights h, column averaging kernel a
# and prior profile x_a (`RetrievalColumn`). The operator is
#
#   1. stretch the model interface pressures linearly so that the model surface
#      and top fall on the retrieval's (`stretch_model_interfaces!`, TM5-4DVar's
#      stretch: deps/tm5-cy3-4dvar/base/f2py/src/averaging_kernels.F90, the
#      OCO-2 routine at lines 162-270);
#   2. map the model layers onto the retrieval's values with a projection
#      matrix P whose rows sum to 1, chosen by the column's mapping
#      (`AbstractRetrievalMapping`):
#        * `LayerRebin`: values are means over N layers between N + 1
#          interfaces; P = R, the mass-conserving rebin by pressure overlap
#          (TM5's `inv_rebin`, averaging_kernels.F90:60-146);
#        * `HatLevelProjection`: values sit on N levels and the profile is
#          linear in pressure between them (OCO-2/OCO-3); P is the L2
#          (Galerkin) projection of the model profile onto the levels' hat
#          functions in dry air, P = M⁻¹ B with the tridiagonal hat mass matrix;
#   3. X = Σᵢ hᵢ aᵢ (P x)ᵢ + Σᵢ hᵢ (1 − aᵢ) x_a,ᵢ.
#
# X is linear in the model profile x: X = Σₖ Hₖ xₖ + c with
# Hₖ = Σᵢ hᵢ aᵢ Pᵢₖ (`retrieval_column_weights!`) and c = Σᵢ hᵢ(1 − aᵢ) x_a,ᵢ
# (`retrieval_prior_offset`). These are pure host Float64 functions of one
# column, shared by the 4D-Var observation operator
# (`CSRetrievalColumnObjective`, src/Inversion/) and the forward sampler of
# soundings (`[output.observations]`, OCO-2 sources with averaging kernels).
#
# Orderings: model interfaces top first (k = 1 at the model top, as every
# model column here); retrieval pressures surface first (index 1 is the
# retrieval surface, pressures strictly decrease), as TM5 and the observation
# files store them.
# ---------------------------------------------------------------------------

# The pressure weights of a retrieval sum to 1 within this tolerance: a deficit
# of δ biases the simulated column by about δ·X (1e-4 of 400 ppm is 0.04 ppm),
# while weights stored as Float32 sum to 1 within ~1e-6.
const RETRIEVAL_WEIGHT_SUM_TOLERANCE = 1e-4

# The stated retrieval surface pressure and the first (surface) interface agree
# within this relative tolerance.
const _RETRIEVAL_SURFACE_TOLERANCE = 1e-6

# Units accepted on a retrieval prior profile (and its uncertainty), as mol/mol
# per unit; matched case-insensitively with blanks removed (CF writes "1" for a
# mole fraction). A missing or unknown unit is an error: a prior in ppm read as
# mol/mol would be off by 10⁶.
const _MOLE_FRACTION_UNITS = Dict("mol/mol" => 1.0, "molmol-1" => 1.0, "molmol^-1" => 1.0, "1" => 1.0,
                                  "ppm" => 1e-6, "ppmv" => 1e-6, "umol/mol" => 1e-6, "µmol/mol" => 1e-6,
                                  "umolmol-1" => 1e-6, "µmolmol-1" => 1e-6,
                                  "ppb" => 1e-9, "ppbv" => 1e-9, "nmol/mol" => 1e-9, "nmolmol-1" => 1e-9)

# The mol/mol factor of a mole-fraction unit, `nothing` for anything else.
mole_fraction_scale(units::AbstractString) = get(_MOLE_FRACTION_UNITS, replace(lowercase(units), r"\s" => ""),
                                                  nothing)
mole_fraction_scale(units) = nothing

"""
    mole_fraction_unit_scale(units, label) -> Float64

The factor that converts a dry mole fraction stated in `units` (`"mol mol-1"`,
`"1"`, `"ppm"`, `"ppb"`, …; case and blanks ignored) to mol/mol. Throws for a
missing or unknown unit, naming `label`.
"""
function mole_fraction_unit_scale(units, label::AbstractString)
    scale = mole_fraction_scale(units)
    scale === nothing && throw(ArgumentError(
        "$(label) has units $(repr(units)); a dry mole fraction must state its unit, one of " *
        "$(join(sort!(collect(keys(_MOLE_FRACTION_UNITS))), ", "))"))
    return scale
end

"""
    AbstractRetrievalMapping

How the profile values of a retrieval relate to its pressures, and so how the
model column is mapped onto them (the projection matrix ``P`` of
[`retrieval_column_weights!`](@ref)): [`LayerRebin`](@ref) (values are layer
means between interfaces) or [`HatLevelProjection`](@ref) (values on levels,
the profile linear in pressure between them). A [`RetrievalColumn`](@ref)
carries one; the operator dispatches on it.
"""
abstract type AbstractRetrievalMapping end

"""
    LayerRebin()

Retrieval values are means over `N` layers between `N + 1` interfaces (a
TCCON or GOSAT layer grid, TM5-4DVar's 19-layer form of OCO-2,
[`tm5_layer_column`](@ref)). The model column, stretched onto the retrieval
column, is rebinned by pressure overlap: ``P = R``, TM5-4DVar's
mass-conserving `inv_rebin` (pressure-overlap rebin). Stored as
`retrieval_mapping = "layer_rebin"`.
"""
struct LayerRebin <: AbstractRetrievalMapping end

"""
    HatLevelProjection()

Retrieval values sit on `N` levels and the profile is linear in pressure
between them: the OCO-2 and OCO-3 retrievals (20 levels, "defined on layer
boundaries", the first at ``σ = 10^{-4}``), whose pressure weights are the
integrals of the levels' hat functions (the trapezoid rule with half-thickness
end levels, with a dry-air correction). The model column, stretched onto
the retrieval levels (top level to bottom level), is projected onto the
retrieval's basis, the hat functions ``φ_j`` of the levels (``φ_j(p_j) = 1``,
linear to 0 at the neighbouring levels): the level values ``c`` are the
L2 (Galerkin) projection in dry air,

```math
\\sum_i M_{ji}\\, c_i = b_j, \\qquad
M_{ji} = \\int φ_j φ_i\\, \\mathrm{d}p_\\mathrm{dry}, \\qquad
b_j = \\int φ_j\\, x\\, \\mathrm{d}p_\\mathrm{dry} = \\sum_k B_{jk}\\, x_k, \\qquad
B_{jk} = \\frac{m_k}{Δs_k} \\int_{s_{k-1/2}}^{s_{k+1/2}} φ_j(p)\\, \\mathrm{d}p,
```

so ``P = M^{-1} B``, with each stretched model layer ``k`` (air mass
``m_k``, the run's dry air mass by default) spread evenly over its pressure
thickness ``Δs_k``; ``M`` is tridiagonal. The hats sum to 1 over the
stretched column, so the rows of ``P`` sum to 1 and, with the hat integrals
``h_j = \\sum_k B_{jk} / \\sum_k m_k`` as weights, ``\\sum_j h_j c_j`` is the
model's air-mass-weighted column mean: the projection conserves the column.
It reproduces every profile the retrieval can represent (piecewise linear on
the levels), to the discretization of the model's layer means. ``P``, the
inverse of the consistent mass matrix times ``B``, has negative entries in
general (about −0.2 in every row on a GEOS L72 column); the weights
``H = (h \\circ a)^T P`` of OCO-2 retrievals stayed positive in the cases
measured. Stored as `retrieval_mapping = "hat_levels"`.
"""
struct HatLevelProjection <: AbstractRetrievalMapping end

"The label of a mapping in observation files (`retrieval_mapping`)."
retrieval_mapping_label(::LayerRebin) = "layer_rebin"
retrieval_mapping_label(::HatLevelProjection) = "hat_levels"
const _RETRIEVAL_MAPPINGS = (layer_rebin = LayerRebin(), hat_levels = HatLevelProjection())

# The mapping named by a file or config label.
function parse_retrieval_mapping(label::AbstractString, context::AbstractString)
    key = Symbol(label)
    haskey(_RETRIEVAL_MAPPINGS, key) || throw(ArgumentError(
        "$(context): retrieval mapping $(repr(label)) is not one of " *
        "$(join(repr.(String.(keys(_RETRIEVAL_MAPPINGS))), ", "))"))
    return _RETRIEVAL_MAPPINGS[key]
end

# The number of pressures a mapping needs for `n` values, and their name.
retrieval_pressure_count(::LayerRebin, n::Integer) = n + 1
retrieval_pressure_count(::HatLevelProjection, n::Integer) = n
_pressure_noun(::LayerRebin) = "interface pressures"
_pressure_noun(::HatLevelProjection) = "level pressures"
_value_noun(::LayerRebin) = "layers"
_value_noun(::HatLevelProjection) = "levels"
# A hat needs a neighbouring level; a layer needs nothing more.
_min_value_count(::LayerRebin) = 1
_min_value_count(::HatLevelProjection) = 2

"""
    RetrievalColumn(pressure_pa, pressure_weight, averaging_kernel, prior_profile,
                    profile_sigma, surface_pressure_pa[, mapping = LayerRebin()[, prior_column = NaN]])
    RetrievalColumn(; pressure_pa, pressure_weight, averaging_kernel, prior_profile,
                      profile_sigma = Float64[], surface_pressure_pa = first(pressure_pa),
                      mapping = LayerRebin(), prior_column = NaN)

The vertical description of one retrieved column average (a satellite or
TCCON sounding) with `N` retrieval values, ordered from the surface up. The
`mapping` ([`AbstractRetrievalMapping`](@ref)) says what the values are:

- [`LayerRebin`](@ref): means over `N` layers; `pressure_pa` holds the
  `N + 1` layer interfaces;
- [`HatLevelProjection`](@ref): values on `N ≥ 2` levels, the profile linear
  in pressure between them (OCO-2/OCO-3); `pressure_pa` holds the `N` level
  pressures.

Fields:

- `pressure_pa`: the interfaces or levels (Pa, total pressure), surface
  first, strictly decreasing; the last is the retrieval top (`≥ 0`);
- `pressure_weight`: the pressure weights ``h_i`` (`N`, `≥ 0`, summing to 1
  within `RETRIEVAL_WEIGHT_SUM_TOLERANCE`);
- `averaging_kernel`: the column averaging kernel ``a_i`` (`N`, normalized,
  dimensionless: ``∂X/∂x_i = h_i a_i``);
- `prior_profile`: the retrieval prior ``x_{a,i}`` (`N`, dry mole fraction,
  mol mol⁻¹, in `[0, 1]`);
- `profile_sigma`: optional 1-σ uncertainty ``σ_i`` of the *model* profile on
  the retrieval layers or levels (`N`, mol mol⁻¹, in `[0, 1]`; empty when
  absent), the input of the model-error term
  [`retrieval_model_error_variance`](@ref). It is not the retrieval's
  a-priori uncertainty: the smoothing error is already represented by the
  averaging kernel;
- `surface_pressure_pa`: the retrieval surface pressure (Pa), equal to the
  first pressure (relative tolerance 1e-6);
- `mapping`: the [`AbstractRetrievalMapping`](@ref);
- `prior_column`: optional, the retrieval's prior column ``X_a`` (mol mol⁻¹,
  in `[0, 1]`; `NaN` when absent), e.g. the OCO-2 `xco2_apriori`. When given,
  the column is ``X = X_a + \\sum_i h_i a_i ((P x)_i - x_{a,i})`` (the MIP
  protocol's eq. 2); without it ``X_a = \\sum_i h_i x_{a,i}``. The two differ
  when the weights and the prior were averaged separately (OCO-2 10-s
  averages: a few thousandths of a ppm).

Every value but an absent `prior_column` must be finite. See [`retrieval_column_weights!`](@ref) for the
operator.
"""
struct RetrievalColumn{M <: AbstractRetrievalMapping}
    pressure_pa::Vector{Float64}
    pressure_weight::Vector{Float64}
    averaging_kernel::Vector{Float64}
    prior_profile::Vector{Float64}
    profile_sigma::Vector{Float64}
    surface_pressure_pa::Float64
    mapping::M
    prior_column::Float64
    function RetrievalColumn(pressure_pa::AbstractVector{<:Real}, pressure_weight::AbstractVector{<:Real},
                             averaging_kernel::AbstractVector{<:Real}, prior_profile::AbstractVector{<:Real},
                             profile_sigma::AbstractVector{<:Real}, surface_pressure_pa::Real,
                             mapping::M = LayerRebin(), prior_column::Real = NaN) where {M <: AbstractRetrievalMapping}
        p = collect(Float64, pressure_pa)
        h = collect(Float64, pressure_weight)
        a = collect(Float64, averaging_kernel)
        xa = collect(Float64, prior_profile)
        σ = collect(Float64, profile_sigma)
        ps = Float64(surface_pressure_pa)
        _check_retrieval_column(p, h, a, xa, σ, ps, mapping)
        Xa = Float64(prior_column)
        isnan(Xa) || 0 <= Xa <= 1 || throw(ArgumentError(
            "the retrieval prior column is a dry mole fraction in mol/mol, within [0, 1], or NaN (absent); got $(Xa)"))
        return new{M}(p, h, a, xa, σ, ps, mapping, Xa)
    end
end

RetrievalColumn(; pressure_pa, pressure_weight, averaging_kernel, prior_profile,
                  profile_sigma = Float64[], surface_pressure_pa = first(pressure_pa),
                  mapping::AbstractRetrievalMapping = LayerRebin(), prior_column::Real = NaN) =
    RetrievalColumn(pressure_pa, pressure_weight, averaging_kernel, prior_profile, profile_sigma,
                    surface_pressure_pa, mapping, prior_column)

function _check_retrieval_column(p, h, a, xa, σ, ps, mapping::AbstractRetrievalMapping)
    n = length(h)
    values = _value_noun(mapping)
    n >= _min_value_count(mapping) || throw(ArgumentError(
        "a retrieval column on $(values) needs at least $(_min_value_count(mapping)) (pressure_weight has $(n))"))
    np = retrieval_pressure_count(mapping, n)
    length(p) == np || throw(DimensionMismatch(
        "a retrieval column of $(n) $(values) needs $(np) $(_pressure_noun(mapping)); got $(length(p))"))
    for (name, v) in (("averaging_kernel", a), ("prior_profile", xa))
        length(v) == n || throw(DimensionMismatch(
            "retrieval column $(name) has $(length(v)) entries; the column has $(n) $(values)"))
    end
    isempty(σ) || length(σ) == n || throw(DimensionMismatch(
        "retrieval column profile_sigma has $(length(σ)) entries; the column has $(n) $(values) (or none)"))
    for (name, v) in (("pressure_pa", p), ("pressure_weight", h), ("averaging_kernel", a),
                      ("prior_profile", xa), ("profile_sigma", σ))
        all(isfinite, v) || throw(ArgumentError("retrieval column $(name) must be finite; got $(v)"))
    end
    all(>(0), -diff(p)) || throw(ArgumentError(
        "retrieval column $(_pressure_noun(mapping)) must be ordered from the surface up (strictly " *
        "decreasing, index 1 = the retrieval surface); got $(p)"))
    last(p) >= 0 || throw(ArgumentError("the retrieval top pressure must be >= 0 Pa; got $(last(p))"))
    all(>=(0), h) || throw(ArgumentError("retrieval pressure weights must be >= 0; got $(h)"))
    total = sum(h)
    abs(total - 1) <= RETRIEVAL_WEIGHT_SUM_TOLERANCE || throw(ArgumentError(
        "retrieval pressure weights must sum to 1 (within $(RETRIEVAL_WEIGHT_SUM_TOLERANCE)); they sum to " *
        "$(total). Weights h = Δp/p_surface leave out the column above a retrieval top above 0 Pa; " *
        "normalize them to the column the retrieval describes"))
    all(x -> 0 <= x <= 1, xa) || throw(ArgumentError(
        "the retrieval prior profile is a dry mole fraction in mol/mol, within [0, 1]; got $(xa) " *
        "(a profile in ppm or ppb must be converted)"))
    all(x -> 0 <= x <= 1, σ) || throw(ArgumentError(
        "retrieval profile_sigma is a mole-fraction uncertainty in mol/mol, within [0, 1]; got $(σ)"))
    isfinite(ps) && ps > 0 || throw(ArgumentError(
        "the retrieval surface pressure must be finite and positive; got $(ps)"))
    abs(ps - first(p)) <= _RETRIEVAL_SURFACE_TOLERANCE * ps || throw(ArgumentError(
        "the retrieval surface pressure $(ps) Pa differs from the first (surface) pressure " *
        "$(first(p)) Pa; pressures are ordered from the surface up"))
    return nothing
end

Base.:(==)(a::RetrievalColumn, b::RetrievalColumn) =
    a.pressure_pa == b.pressure_pa && a.pressure_weight == b.pressure_weight &&
    a.averaging_kernel == b.averaging_kernel && a.prior_profile == b.prior_profile &&
    a.profile_sigma == b.profile_sigma && a.surface_pressure_pa == b.surface_pressure_pa &&
    a.mapping == b.mapping && isequal(a.prior_column, b.prior_column)
Base.hash(c::RetrievalColumn, h::UInt) =
    hash((c.pressure_pa, c.pressure_weight, c.averaging_kernel, c.prior_profile, c.profile_sigma,
          c.surface_pressure_pa, c.mapping, c.prior_column), hash(:RetrievalColumn, h))

# The two concrete column types, a small union for fields that hold either.
const AnyRetrievalColumn = Union{RetrievalColumn{LayerRebin}, RetrievalColumn{HatLevelProjection}}

"""
    retrieval_layer_count(column::RetrievalColumn) -> Int

Number of retrieval values of a [`RetrievalColumn`](@ref): its layers
([`LayerRebin`](@ref)) or levels ([`HatLevelProjection`](@ref)).
"""
retrieval_layer_count(c::RetrievalColumn) = length(c.pressure_weight)

"""
    stretch_model_interfaces!(s, p_half, surface_pa, top_pa) -> s

TM5-4DVar's stretch of the model column onto a retrieval column
(`averaging_kernels.F90`, lines 213-219): the model interface pressures `p_half` (Pa,
top first, `p_half[end]` the model surface pressure ``p_s``, `p_half[1]` the
model top ``p_t``) mapped linearly so that the model surface falls on the
retrieval surface ``p^r_s`` (`surface_pa`) and the model top on the
retrieval top ``p^r_t`` (`top_pa`),

```math
s_k = \\mu\\, p_k + c, \\qquad
\\mu = \\frac{p^r_s - p^r_t}{p_s - p_t}, \\qquad
c = \\tfrac12\\big[(p^r_s - \\mu p_s) + (p^r_t - \\mu p_t)\\big].
```

Each model layer keeps its share ``Δp_k/(p_s - p_t)`` of the column, so the
pressure-weighted mean of any profile over the stretched layers is its mean
over the model layers: the stretch moves the model column onto the
retrieval's pressure range without adding or removing air.
"""
function stretch_model_interfaces!(s::AbstractVector{Float64}, p_half::AbstractVector, surface_pa::Real,
                                   top_pa::Real)
    n = length(p_half)
    n >= 2 || throw(ArgumentError("a model column needs at least one layer (two interfaces)"))
    length(s) == n || throw(DimensionMismatch(
        "the stretched interfaces need $(n) entries; got $(length(s))"))
    p_top = Float64(p_half[1])
    p_surface = Float64(p_half[n])
    rs = Float64(surface_pa)
    rt = Float64(top_pa)
    # The operations and their order are TM5's, so the results agree to rounding.
    μ = (rs - rt) / (p_surface - p_top)
    c1 = rs - μ * p_surface
    c2 = rt - μ * p_top
    c = 0.5 * (c1 + c2)
    @inbounds for k in 1:n
        s[k] = μ * Float64(p_half[k]) + c
    end
    return s
end

# Model interfaces (top first) strictly increasing and finite.
function _check_model_interfaces(p_half::AbstractVector)
    all(isfinite, p_half) && first(p_half) >= 0 && all(>(0), diff(p_half)) || throw(ArgumentError(
        "model interface pressures must be finite, >= 0 and strictly increasing from the top; got $(p_half)"))
    return nothing
end

# TM5's validity check of `inv_rebin` (averaging_kernels.F90:80-99) on the
# stretched model interfaces `s` (top first) and the retrieval interfaces `r`
# (surface first): the interior retrieval interfaces lie inside the stretched
# column. TM5 drops an observation that fails it; here it is an error, since a
# validated retrieval column and model column can fail it only when a
# retrieval layer is thinner than the stretch's rounding.
function _check_rebin(s::AbstractVector{Float64}, r::AbstractVector)
    all(>(0), diff(s)) || throw(ArgumentError(
        "the stretched model interfaces are not strictly increasing: $(s)"))
    n = length(r) - 1
    n >= 2 || return nothing
    Float64(r[2]) < s[end] && Float64(r[n]) > s[1] || throw(ArgumentError(
        "the interior retrieval interfaces ($(r[2]) … $(r[n]) Pa) do not lie inside the stretched model " *
        "column ($(s[1]) … $(s[end]) Pa)"))
    return nothing
end

# TM5's mass-conserving rebin (`inv_rebin`, averaging_kernels.F90:124-142):
# `f(i, k, R)` for every retrieval layer `i` (surface first) and model layer
# `k` (top first) that overlap, with R = overlap / thickness of retrieval
# layer i. Retrieval layer i spans [r_{i+1}, r_i], except that the outermost
# interfaces are those of the stretched model column (TM5's out_x(1) = in_x(1)
# and out_x(N+1) = in_x(Nz+1)); stretched model layer k spans [s_k, s_{k+1}].
function _foreach_rebin_overlap(f, s::AbstractVector{Float64}, r::AbstractVector)
    nz = length(s) - 1
    n = length(r) - 1
    @inbounds for i in 1:n
        hi = i == 1 ? s[nz + 1] : Float64(r[i])
        lo = i == n ? s[1] : Float64(r[i + 1])
        thickness = hi - lo
        for k in nz:-1:1                     # TM5's model layer j = 1 (surface) … Nz
            s_lo, s_hi = s[k], s[k + 1]
            (s_lo >= hi || lo >= s_hi) && continue
            f(i, k, (min(hi, s_hi) - max(s_lo, lo)) / thickness)
        end
    end
    return nothing
end

# The pieces of the hat projection on the stretched model column `s` (top
# first), the model air masses `air` and the retrieval levels `r` (surface
# first): `B[j, k]` = ∫ φ_j dp_dry over stretched layer k, and the tridiagonal
# mass matrix `M[j, i]` = ∫ φ_j φ_i dp_dry, the air of each layer spread evenly
# over its pressure thickness. Between neighbouring levels j and j + 1
# (pressures r_j > r_{j+1}) the hats are φ_j = (p − r_{j+1})/(r_j − r_{j+1}) and
# φ_{j+1} = 1 − φ_j. The outermost levels are those of the stretched model
# column, as the outer interfaces of the rebin (they equal the retrieval
# surface and top to rounding), so the hats sum to 1 over every stretched
# layer. On each piece [u, v] of a layer between two levels the hats are linear:
# the midpoint rule is exact for them and Simpson's rule for their products.
function _hat_projection_pieces(s::AbstractVector{Float64}, air::AbstractVector, r::AbstractVector)
    nz = length(s) - 1
    n = length(r)
    B = zeros(n, nz)
    diagonal = zeros(n)
    offdiagonal = zeros(n - 1)
    @inbounds for k in 1:nz
        s_lo, s_hi = s[k], s[k + 1]
        density = Float64(air[k]) / (s_hi - s_lo)
        for j in 1:(n - 1)
            hi = j == 1 ? s[nz + 1] : Float64(r[j])
            lo = j == n - 1 ? s[1] : Float64(r[j + 1])
            (s_lo >= hi || lo >= s_hi) && continue
            u = max(s_lo, lo)
            v = min(s_hi, hi)
            mass = density * (v - u)
            φu, φv = (u - lo) / (hi - lo), (v - lo) / (hi - lo)       # φ_j at u and v
            φm = (φu + φv) / 2
            B[j, k] += mass * φm
            B[j + 1, k] += mass * (1 - φm)
            diagonal[j] += mass * (φu^2 + 4φm^2 + φv^2) / 6
            diagonal[j + 1] += mass * ((1 - φu)^2 + 4(1 - φm)^2 + (1 - φv)^2) / 6
            offdiagonal[j] += mass * (φu * (1 - φu) + 4φm * (1 - φm) + φv * (1 - φv)) / 6
        end
    end
    all(>(0), diagonal) || throw(ArgumentError(
        "a retrieval level has no model air under its hat (hat mass-matrix diagonal: $(diagonal))"))
    return B, LinearAlgebra.SymTridiagonal(diagonal, offdiagonal)
end

# The entries `f(i, k, P)` of the projection matrix of a mapping, on the
# stretched model interfaces `s`, the model air mass `air` (kg per cell, top
# first; read by the hat projection only) and the retrieval pressures `r`.
_foreach_projection_entry(f, ::LayerRebin, s::AbstractVector{Float64}, air, r::AbstractVector) =
    _foreach_rebin_overlap(f, s, r)

function _foreach_projection_entry(f, ::HatLevelProjection, s::AbstractVector{Float64}, air::AbstractVector,
                                   r::AbstractVector)
    B, M = _hat_projection_pieces(s, air, r)
    P = M \ B                                   # the L2 projection, P = M⁻¹ B
    @inbounds for k in axes(P, 2), j in axes(P, 1)
        f(j, k, P[j, k])
    end
    return nothing
end

# The air-mass column a mapping reads: none for the rebin; positive finite
# masses, one per model layer, for the hat projection.
_check_air_column(::LayerRebin, air, nz::Int) = nothing
_check_air_column(::HatLevelProjection, ::Nothing, nz::Int) = throw(ArgumentError(
    "the hat projection (HatLevelProjection) weights each model layer by its air mass; pass the model's " *
    "air-mass column (kg per cell, top first) to the operator"))
function _check_air_column(::HatLevelProjection, air::AbstractVector, nz::Int)
    length(air) == nz || throw(DimensionMismatch(
        "the air-mass column has $(length(air)) layers; the model column has $(nz)"))
    all(m -> isfinite(m) && m > 0, air) || throw(ArgumentError(
        "the air masses of the hat projection must be finite and positive; got $(air)"))
    return nothing
end

# Stretch the model interfaces onto the retrieval column and check that the
# projection can be built: the model interfaces, the air column, and the
# interior retrieval pressures inside the stretched column.
function _stretch_onto!(s::AbstractVector{Float64}, p_half::AbstractVector, air, column::RetrievalColumn)
    nz = length(p_half) - 1
    length(s) == nz + 1 || throw(DimensionMismatch(
        "the stretched-interface scratch needs $(nz + 1) entries; got $(length(s))"))
    _check_model_interfaces(p_half)
    _check_air_column(column.mapping, air, nz)
    r = column.pressure_pa
    stretch_model_interfaces!(s, p_half, first(r), last(r))
    _check_rebin(s, r)
    return s
end

"""
    retrieval_column_weights!(H, s, p_half, air, column::RetrievalColumn) -> H

Weights of the averaging-kernel column on the model layers: with the model
profile ``x_k`` (dry mole fraction, `k = 1` at the top),

```math
X = \\sum_k H_k\\, x_k + c, \\qquad
H_k = \\sum_i h_i\\, a_i\\, P_{ik}, \\qquad
c = X_a - \\sum_i h_i\\, a_i\\, x_{a,i}
```

([`retrieval_prior_offset`](@ref)) with the column's prior column ``X_a``
(`prior_column`) when it gives one, ``X_a = \\sum_i h_i x_{a,i}`` (so
``c = \\sum_i h_i (1 - a_i) x_{a,i}``) when not. ``P`` maps the model column, stretched
onto the retrieval column ([`stretch_model_interfaces!`](@ref)), onto the
retrieval's values as its `mapping` says: the rebin ``R`` of
[`LayerRebin`](@ref) (the column of
TM5-4DVar's OCO-2 operator, `averaging_kernels.F90`, lines 228-235 and 245),
or the hat projection of [`HatLevelProjection`](@ref), which weights each
model layer by its air mass `air` (kg per cell, top first; the run's dry air
mass on a dry basis). Every row of ``P`` sums to 1
(dry-air projection). `p_half` are the model interface
pressures (Pa, top first; total pressures, so that they compare with the
retrieval's), `s` a scratch vector of the same length that receives the
stretched interfaces. A layer column ignores `air`; pass `nothing` for it.
"""
function retrieval_column_weights!(H::AbstractVector{Float64}, s::AbstractVector{Float64}, p_half::AbstractVector,
                                   air, column::RetrievalColumn)
    nz = length(p_half) - 1
    length(H) == nz || throw(DimensionMismatch("H needs $(nz) entries; got $(length(H))"))
    _stretch_onto!(s, p_half, air, column)
    h = column.pressure_weight
    a = column.averaging_kernel
    fill!(H, 0.0)
    # W = h·a, then H = W P (TM5: W = pres_weights * avg_ker; H = matmul(W, R)).
    _foreach_projection_entry(column.mapping, s, air, column.pressure_pa) do i, k, P
        @inbounds H[k] += h[i] * a[i] * P
    end
    return H
end

"""
    retrieval_prior_offset(column::RetrievalColumn) -> Float64
    retrieval_prior_offset(h, a, prior) -> Float64

The prior term of the averaging-kernel column, which does not depend on the
model: ``c = X_a - \\sum_i h_i a_i\\, x_{a,i}`` with the column's
`prior_column` ``X_a`` when it gives one (``X = X_a + \\sum_i h_i a_i
((P x)_i - x_{a,i})``, the MIP protocol's form), else
``c = \\sum_i (h_i - h_i a_i)\\, x_{a,i}`` (TM5: `dot_product(pres_weights -
W, prior_profile)`; the raw form always).
"""
retrieval_prior_offset(column::RetrievalColumn) =
    _prior_offset(column.prior_column, column.pressure_weight, column.averaging_kernel, column.prior_profile)

_prior_offset(Xa::Float64, h, a, prior) =
    isnan(Xa) ? retrieval_prior_offset(h, a, prior) : Xa - sum(h[i] * a[i] * prior[i] for i in eachindex(h))

function retrieval_prior_offset(h::AbstractVector, a::AbstractVector, prior::AbstractVector)
    length(h) == length(a) == length(prior) || throw(DimensionMismatch(
        "weights, kernel and prior must have equal lengths; got $(length(h)), $(length(a)), $(length(prior))"))
    c = 0.0
    @inbounds for i in eachindex(h)
        hi = Float64(h[i])
        c += (hi - hi * Float64(a[i])) * Float64(prior[i])
    end
    return c
end

"""
    retrieval_profile!(xr, s, p_half, air, column::RetrievalColumn, x) -> xr

The model profile `x` (top first) on the retrieval layers or levels (surface
first), ``(P x)_i``: for a layer column the pressure-weighted mean over each
retrieval layer, for a level column the hat projection, of the model column
stretched onto the retrieval column (`air` as in
[`retrieval_column_weights!`](@ref); pass `nothing` for a `LayerRebin`
column, which does not read it). `s` is scratch for the stretched
interfaces. A diagnostic: the column itself is
``\\sum_i h_i a_i (P x)_i + c``.
"""
function retrieval_profile!(xr::AbstractVector{Float64}, s::AbstractVector{Float64}, p_half::AbstractVector,
                            air, column::RetrievalColumn, x::AbstractVector)
    nz = length(p_half) - 1
    length(x) == nz || throw(DimensionMismatch("the model profile has $(length(x)) layers; the column $(nz)"))
    length(xr) == retrieval_layer_count(column) || throw(DimensionMismatch(
        "the retrieval profile needs $(retrieval_layer_count(column)) entries; got $(length(xr))"))
    _stretch_onto!(s, p_half, air, column)
    fill!(xr, 0.0)
    _foreach_projection_entry(column.mapping, s, air, column.pressure_pa) do i, k, P
        @inbounds xr[i] += P * Float64(x[k])
    end
    return xr
end


"""
    retrieval_column_value(p_half, air, column::RetrievalColumn, x) -> Float64

The simulated column ``X = \\sum_k H_k x_k + c`` of the model profile `x`
(dry mole fraction, top first) on the model interface pressures `p_half`
(Pa, top first, total pressures) and air masses `air` (read by the hat
projection only; pass `nothing` for a `LayerRebin` column): [`retrieval_column_weights!`](@ref) and
[`retrieval_prior_offset`](@ref) in one call.
"""
function retrieval_column_value(p_half::AbstractVector, air, column::RetrievalColumn, x::AbstractVector)
    nz = length(p_half) - 1
    length(x) == nz || throw(DimensionMismatch("the model profile has $(length(x)) layers; the column $(nz)"))
    H = retrieval_column_weights!(Vector{Float64}(undef, nz), Vector{Float64}(undef, nz + 1), p_half, air, column)
    value = retrieval_prior_offset(column)
    @inbounds for k in 1:nz
        value += H[k] * Float64(x[k])
    end
    return value
end


"""
    tm5_layer_column(column::RetrievalColumn{HatLevelProjection}) -> RetrievalColumn{LayerRebin}

TM5-4DVar's layer form of a level retrieval (OCO-2/OCO-3 10-s files in
`proj/tracer/CO2/py/Observations.py`, lines 7016-7052, used by
`Satellite.py`, lines 707-717): the `N` levels become the interfaces of
`N − 1` layers, the averaging kernel and the prior are interpolated linearly
in pressure to the layer midpoints (the mean of the two levels), and the
pressure weights are the layer thicknesses ``Δp_i``, normalized to the column
between the bottom and top levels. TM5 divides by the surface pressure
instead, so its weights sum to ``1 - σ_\\mathrm{top}`` (0.9999 for OCO-2)
and its column is that fraction of this one. A `profile_sigma` is averaged
like the kernel. The prior column is TM5's, ``\\sum_i h_i x_{a,i}`` on its
layers (a `prior_column` of the record is not carried). For comparisons with
TM5-4DVAR; the hat projection ([`HatLevelProjection`](@ref)) is the
consistent mapping of level retrievals.
"""
function tm5_layer_column(column::RetrievalColumn{HatLevelProjection})
    p = column.pressure_pa
    n = length(p)
    midpoint(v) = [(v[i] + v[i + 1]) / 2 for i in 1:(n - 1)]
    thickness = [p[i] - p[i + 1] for i in 1:(n - 1)]
    return RetrievalColumn(p, thickness ./ (p[1] - p[n]), midpoint(column.averaging_kernel),
                           midpoint(column.prior_profile),
                           isempty(column.profile_sigma) ? Float64[] : midpoint(column.profile_sigma),
                           column.surface_pressure_pa, LayerRebin())
end

"""
    oco2_level_form(mapping::AbstractRetrievalMapping, column::RetrievalColumn{HatLevelProjection})

A level retrieval (OCO-2/OCO-3) in the form the operator uses under
`mapping`: unchanged for [`HatLevelProjection`](@ref) (the levels as given,
the default), TM5-4DVar's 19 layers for [`LayerRebin`](@ref)
([`tm5_layer_column`](@ref), the "tm5" mode).
"""
oco2_level_form(::HatLevelProjection, column::RetrievalColumn{HatLevelProjection}) = column
oco2_level_form(::LayerRebin, column::RetrievalColumn{HatLevelProjection}) = tm5_layer_column(column)

# The level forms by their config and command-line names.
const OCO2_LEVEL_FORMS = (hat = HatLevelProjection(), tm5 = LayerRebin())

"""
    oco2_level_form_from_label(label, context) -> AbstractRetrievalMapping

The mapping named `"hat"` ([`HatLevelProjection`](@ref)) or `"tm5"`
([`LayerRebin`](@ref) on TM5's 19 layers) in configs and on the command line.
"""
function oco2_level_form_from_label(label::AbstractString, context::AbstractString)
    key = Symbol(label)
    haskey(OCO2_LEVEL_FORMS, key) || throw(ArgumentError(
        "$(context) must be one of $(join(repr.(String.(keys(OCO2_LEVEL_FORMS))), ", ")); got $(repr(label))"))
    return OCO2_LEVEL_FORMS[key]
end

"""
    retrieval_model_error_variance(column::RetrievalColumn) -> Float64

The model-error variance of the averaging-kernel column from the uncertainty
``σ_i`` of the model profile on the retrieval layers (`profile_sigma`), the
layers taken independent: ``σ^2_\\mathrm{mod} = \\sum_i (h_i a_i σ_i)^2``,
the operator's weights on the retrieval layers, squared, times the profile
variance. Zero when the column gives no `profile_sigma`. It depends on the
record only, so it is fixed for an inversion and the cost stays quadratic.

This is analogous to, not the same as, TM5-4DVar's model error: its CH4
TROPOMI routine (`averaging_kernels.F90`, line 669, and `Satellite.py` of the
CH4 project) evaluates ``\\sum_k H_k^2 σ_k^2`` with the weights
``H = (h a) R`` on the model layers and the spread of the modelled profiles
``σ_k`` from its track file, and its OCO-2 routine adds a column model error
instead. `profile_sigma` is not the retrieval's a-priori uncertainty, whose
effect (the smoothing error) the averaging kernel already represents.
"""
function retrieval_model_error_variance(column::RetrievalColumn)
    isempty(column.profile_sigma) && return 0.0
    v = 0.0
    @inbounds for i in eachindex(column.profile_sigma)
        v += (column.pressure_weight[i] * column.averaging_kernel[i] * column.profile_sigma[i])^2
    end
    return v
end
