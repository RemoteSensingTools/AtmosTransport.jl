# ---------------------------------------------------------------------------
# Averaging-kernel (retrieval) column operator.
#
# A retrieved column-averaged dry mole fraction (an OCO-2, GOSAT or TROPOMI
# sounding) is compared with the model through the retrieval's own vertical
# description: its layer interfaces, pressure weights h, column averaging
# kernel a and prior profile x_a (`RetrievalColumn`). The operator is the one
# of TM5-4DVar (deps/tm5-cy3-4dvar/base/f2py/src/averaging_kernels.F90, the
# OCO-2 routine at lines 162-270, the rebin `inv_rebin` at lines 60-146):
#
#   1. stretch the model interface pressures linearly so that the model surface
#      and top fall on the retrieval's (`stretch_model_interfaces!`);
#   2. redistribute the model layers onto the retrieval layers by pressure
#      overlap, a mass-conserving rebin R (`retrieval_rebin_matrix`);
#   3. X = Σᵢ hᵢ aᵢ (R x)ᵢ + Σᵢ hᵢ (1 − aᵢ) x_a,ᵢ.
#
# X is linear in the model profile x: X = Σₖ Hₖ xₖ + c with
# Hₖ = Σᵢ hᵢ aᵢ Rᵢₖ (`retrieval_column_weights!`) and c = Σᵢ hᵢ(1 − aᵢ) x_a,ᵢ
# (`retrieval_prior_offset`). These are pure host Float64 functions of one
# column, shared by the 4D-Var observation operator
# (`CSRetrievalColumnObjective`, src/Inversion/) and any forward sampler of
# soundings.
#
# Orderings: model interfaces top first (k = 1 at the model top, as every
# model column here); retrieval interfaces surface first (index 1 is the
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
_mole_fraction_scale(units::AbstractString) = get(_MOLE_FRACTION_UNITS, replace(lowercase(units), r"\s" => ""),
                                                  nothing)
_mole_fraction_scale(units) = nothing

"""
    mole_fraction_unit_scale(units, label) -> Float64

The factor that converts a dry mole fraction stated in `units` (`"mol mol-1"`,
`"1"`, `"ppm"`, `"ppb"`, …; case and blanks ignored) to mol/mol. Throws for a
missing or unknown unit, naming `label`.
"""
function mole_fraction_unit_scale(units, label::AbstractString)
    scale = _mole_fraction_scale(units)
    scale === nothing && throw(ArgumentError(
        "$(label) has units $(repr(units)); a dry mole fraction must state its unit, one of " *
        "$(join(sort!(collect(keys(_MOLE_FRACTION_UNITS))), ", "))"))
    return scale
end

"""
    RetrievalColumn(pressure_pa, pressure_weight, averaging_kernel, prior_profile,
                    profile_sigma, surface_pressure_pa)
    RetrievalColumn(; pressure_pa, pressure_weight, averaging_kernel, prior_profile,
                      profile_sigma = Float64[], surface_pressure_pa = first(pressure_pa))

The vertical description of one retrieved column average (a satellite or
TCCON sounding) on its `N` retrieval layers, ordered from the surface up:

- `pressure_pa`: the `N + 1` layer interfaces (Pa, total pressure), surface
  first, strictly decreasing; the last is the retrieval top (`≥ 0`);
- `pressure_weight`: the pressure weights ``h_i`` (`N`, `≥ 0`, summing to 1
  within `RETRIEVAL_WEIGHT_SUM_TOLERANCE`);
- `averaging_kernel`: the column averaging kernel ``a_i`` (`N`, normalized,
  dimensionless: ``∂X/∂x_i = h_i a_i``);
- `prior_profile`: the retrieval prior ``x_{a,i}`` (`N`, dry mole fraction,
  mol mol⁻¹, in `[0, 1]`);
- `profile_sigma`: optional 1-σ uncertainty ``σ_i`` of the *model* profile on
  the retrieval layers (`N`, mol mol⁻¹, in `[0, 1]`; empty when absent), the
  input of the model-error term [`retrieval_model_error_variance`](@ref). It
  is not the retrieval's a-priori uncertainty: the smoothing error is already
  represented by the averaging kernel;
- `surface_pressure_pa`: the retrieval surface pressure (Pa), equal to the
  first interface (relative tolerance 1e-6).

Every value must be finite. See [`retrieval_column_weights!`](@ref) for the
operator.
"""
struct RetrievalColumn
    pressure_pa::Vector{Float64}
    pressure_weight::Vector{Float64}
    averaging_kernel::Vector{Float64}
    prior_profile::Vector{Float64}
    profile_sigma::Vector{Float64}
    surface_pressure_pa::Float64
    function RetrievalColumn(pressure_pa::AbstractVector{<:Real}, pressure_weight::AbstractVector{<:Real},
                             averaging_kernel::AbstractVector{<:Real}, prior_profile::AbstractVector{<:Real},
                             profile_sigma::AbstractVector{<:Real}, surface_pressure_pa::Real)
        p = collect(Float64, pressure_pa)
        h = collect(Float64, pressure_weight)
        a = collect(Float64, averaging_kernel)
        xa = collect(Float64, prior_profile)
        σ = collect(Float64, profile_sigma)
        ps = Float64(surface_pressure_pa)
        _check_retrieval_column(p, h, a, xa, σ, ps)
        return new(p, h, a, xa, σ, ps)
    end
end

RetrievalColumn(; pressure_pa, pressure_weight, averaging_kernel, prior_profile,
                  profile_sigma = Float64[], surface_pressure_pa = first(pressure_pa)) =
    RetrievalColumn(pressure_pa, pressure_weight, averaging_kernel, prior_profile, profile_sigma,
                    surface_pressure_pa)

function _check_retrieval_column(p, h, a, xa, σ, ps)
    n = length(h)
    n >= 1 || throw(ArgumentError("a retrieval column needs at least one layer (pressure_weight is empty)"))
    length(p) == n + 1 || throw(DimensionMismatch(
        "a retrieval column of $(n) layers needs $(n + 1) interface pressures; got $(length(p))"))
    for (name, v) in (("averaging_kernel", a), ("prior_profile", xa))
        length(v) == n || throw(DimensionMismatch(
            "retrieval column $(name) has $(length(v)) entries; the column has $(n) layers"))
    end
    isempty(σ) || length(σ) == n || throw(DimensionMismatch(
        "retrieval column profile_sigma has $(length(σ)) entries; the column has $(n) layers (or none)"))
    for (name, v) in (("pressure_pa", p), ("pressure_weight", h), ("averaging_kernel", a),
                      ("prior_profile", xa), ("profile_sigma", σ))
        all(isfinite, v) || throw(ArgumentError("retrieval column $(name) must be finite; got $(v)"))
    end
    all(>(0), -diff(p)) || throw(ArgumentError(
        "retrieval column interface pressures must be ordered from the surface up (strictly decreasing, " *
        "index 1 = the retrieval surface); got $(p)"))
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
        "the retrieval surface pressure $(ps) Pa differs from the first (surface) interface pressure " *
        "$(first(p)) Pa; interfaces are ordered from the surface up"))
    return nothing
end

Base.:(==)(a::RetrievalColumn, b::RetrievalColumn) =
    a.pressure_pa == b.pressure_pa && a.pressure_weight == b.pressure_weight &&
    a.averaging_kernel == b.averaging_kernel && a.prior_profile == b.prior_profile &&
    a.profile_sigma == b.profile_sigma && a.surface_pressure_pa == b.surface_pressure_pa
Base.hash(c::RetrievalColumn, h::UInt) =
    hash((c.pressure_pa, c.pressure_weight, c.averaging_kernel, c.prior_profile, c.profile_sigma,
          c.surface_pressure_pa), hash(:RetrievalColumn, h))

"Number of retrieval layers of a [`RetrievalColumn`](@ref)."
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

function _check_operator_lengths(nz::Int, s, r, h, a)
    length(s) == nz + 1 || throw(DimensionMismatch(
        "the stretched-interface scratch needs $(nz + 1) entries; got $(length(s))"))
    n = length(r) - 1
    n >= 1 && length(h) == n && length(a) == n || throw(DimensionMismatch(
        "a retrieval column of $(length(r)) interfaces needs $(length(r) - 1) weights and kernel values; " *
        "got $(length(h)) and $(length(a))"))
    _check_retrieval_interfaces(r)
    return nothing
end

# Retrieval interfaces of the raw forms: at least two, finite, surface first
# (strictly decreasing), top >= 0 (`RetrievalColumn` checks the same).
function _check_retrieval_interfaces(r::AbstractVector)
    length(r) >= 2 && all(isfinite, r) && all(<(0), diff(r)) && last(r) >= 0 || throw(ArgumentError(
        "retrieval interface pressures must be finite, ordered from the surface up (strictly decreasing) " *
        "and end at a top >= 0 Pa; got $(r)"))
    return nothing
end

"""
    retrieval_column_weights!(H, s, p_half, column::RetrievalColumn) -> H
    retrieval_column_weights!(H, s, p_half, r, h, a) -> H

Weights of the averaging-kernel column on the model layers: with the model
profile ``x_k`` (dry mole fraction, `k = 1` at the top),

```math
X = \\sum_k H_k\\, x_k + c, \\qquad
H_k = \\sum_i h_i\\, a_i\\, R_{ik}, \\qquad
c = \\sum_i h_i\\,(1 - a_i)\\, x_{a,i}
```

([`retrieval_prior_offset`](@ref)), the column of TM5-4DVar's OCO-2
operator (`averaging_kernels.F90`, lines 228-235 and 245). ``R`` is the mass-conserving
rebin of the stretched model layers ([`stretch_model_interfaces!`](@ref))
onto the retrieval layers ([`retrieval_rebin_matrix`](@ref)). `p_half` are
the model interface pressures (Pa, top first; total pressures, so that they
compare with the retrieval's), `s` a scratch vector of the same length that
receives the stretched interfaces. The second form takes the retrieval
interfaces `r` (surface first), pressure weights `h` and kernel `a` as they
are, without the checks of [`RetrievalColumn`](@ref) (weights that do not
sum to 1, as TM5's ``h_i = Δp_i/p_s`` with a retrieval top above 0 Pa).
"""
retrieval_column_weights!(H::AbstractVector{Float64}, s::AbstractVector{Float64}, p_half::AbstractVector,
                          column::RetrievalColumn) =
    retrieval_column_weights!(H, s, p_half, column.pressure_pa, column.pressure_weight,
                              column.averaging_kernel)

function retrieval_column_weights!(H::AbstractVector{Float64}, s::AbstractVector{Float64}, p_half::AbstractVector,
                                   r::AbstractVector, h::AbstractVector, a::AbstractVector)
    nz = length(p_half) - 1
    length(H) == nz || throw(DimensionMismatch("H needs $(nz) entries; got $(length(H))"))
    _check_operator_lengths(nz, s, r, h, a)
    _check_model_interfaces(p_half)
    stretch_model_interfaces!(s, p_half, first(r), last(r))
    _check_rebin(s, r)
    fill!(H, 0.0)
    # W = h·a, then H = W R (TM5: W = pres_weights * avg_ker; H = matmul(W, R)).
    _foreach_rebin_overlap(s, r) do i, k, R
        @inbounds H[k] += Float64(h[i]) * Float64(a[i]) * R
    end
    return H
end

"""
    retrieval_prior_offset(column::RetrievalColumn) -> Float64
    retrieval_prior_offset(h, a, prior) -> Float64

The prior term of the averaging-kernel column,
``c = \\sum_i (h_i - h_i a_i)\\, x_{a,i}``, which does not depend on the model
(TM5: `dot_product(pres_weights - W, prior_profile)`).
"""
retrieval_prior_offset(column::RetrievalColumn) =
    retrieval_prior_offset(column.pressure_weight, column.averaging_kernel, column.prior_profile)

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
    retrieval_rebin_matrix(p_half, column::RetrievalColumn) -> Matrix{Float64}
    retrieval_rebin_matrix(p_half, r) -> Matrix{Float64}

The redistribution matrix ``R`` (`N × Nz`, retrieval layers surface first,
model layers top first) of TM5-4DVar's `inv_rebin`
(`averaging_kernels.F90`, lines 60-146) on the model column stretched onto the
retrieval column: ``R_{ik}`` is the pressure overlap of stretched model layer
``k`` with retrieval layer ``i`` divided by the thickness of layer ``i``,
so ``(R x)_i`` is the pressure-weighted mean of ``x`` over retrieval layer
``i``. The outermost retrieval interfaces are those of the stretched model
column (TM5's choice; they equal the retrieval surface and top to rounding).
Every row sums to 1 and every column ``k`` distributes all of the stretched
layer: ``\\sum_i R_{ik}\\,Δp^r_i = Δs_k`` (the rebin conserves the column
integral of ``x``).
"""
retrieval_rebin_matrix(p_half::AbstractVector, column::RetrievalColumn) =
    retrieval_rebin_matrix(p_half, column.pressure_pa)

function retrieval_rebin_matrix(p_half::AbstractVector, r::AbstractVector)
    nz = length(p_half) - 1
    _check_retrieval_interfaces(r)
    _check_model_interfaces(p_half)
    s = stretch_model_interfaces!(Vector{Float64}(undef, nz + 1), p_half, first(r), last(r))
    _check_rebin(s, r)
    R = zeros(length(r) - 1, nz)
    _foreach_rebin_overlap((i, k, w) -> (@inbounds R[i, k] = w), s, r)
    return R
end

"""
    retrieval_profile!(xr, s, p_half, column::RetrievalColumn, x) -> xr

The model profile `x` (top first) on the retrieval layers (surface first),
``(R x)_i``: the pressure-weighted mean over each retrieval layer of the
model column stretched onto the retrieval column. `s` is scratch for the
stretched interfaces. A diagnostic: the column itself is
``\\sum_i h_i a_i (R x)_i + c``.
"""
function retrieval_profile!(xr::AbstractVector{Float64}, s::AbstractVector{Float64}, p_half::AbstractVector,
                            column::RetrievalColumn, x::AbstractVector)
    nz = length(p_half) - 1
    length(x) == nz || throw(DimensionMismatch("the model profile has $(length(x)) layers; the column $(nz)"))
    length(xr) == retrieval_layer_count(column) || throw(DimensionMismatch(
        "the retrieval profile needs $(retrieval_layer_count(column)) entries; got $(length(xr))"))
    length(s) == nz + 1 || throw(DimensionMismatch(
        "the stretched-interface scratch needs $(nz + 1) entries; got $(length(s))"))
    _check_model_interfaces(p_half)
    r = column.pressure_pa
    stretch_model_interfaces!(s, p_half, first(r), last(r))
    _check_rebin(s, r)
    fill!(xr, 0.0)
    _foreach_rebin_overlap(s, r) do i, k, R
        @inbounds xr[i] += R * Float64(x[k])
    end
    return xr
end

"""
    retrieval_column_value(p_half, column::RetrievalColumn, x) -> Float64

The simulated column ``X = \\sum_k H_k x_k + c`` of the model profile `x`
(dry mole fraction, top first) on the model interface pressures `p_half`
(Pa, top first, total pressures): [`retrieval_column_weights!`](@ref) and
[`retrieval_prior_offset`](@ref) in one call.
"""
function retrieval_column_value(p_half::AbstractVector, column::RetrievalColumn, x::AbstractVector)
    nz = length(p_half) - 1
    length(x) == nz || throw(DimensionMismatch("the model profile has $(length(x)) layers; the column $(nz)"))
    H = retrieval_column_weights!(Vector{Float64}(undef, nz), Vector{Float64}(undef, nz + 1), p_half, column)
    value = retrieval_prior_offset(column)
    @inbounds for k in 1:nz
        value += H[k] * Float64(x[k])
    end
    return value
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
