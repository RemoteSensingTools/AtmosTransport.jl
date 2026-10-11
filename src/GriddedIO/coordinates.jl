# ---------------------------------------------------------------------------
# Coordinates, lat-lon orientation and fill policies of gridded input files
#
# Coordinates are found by variable name only: their attributes cannot be
# trusted (FLUXCOM-X labels its `lon` variable `units = degrees_north`,
# `long_name = latitude`; GFED5 NRT `lat`/`lon` carry no attributes at all).
# Fill values are honoured as the file declares them, through `_FillValue` and
# `missing_value` (VISIT, VISIT-UT and iMAPLE mask only through
# `missing_value = -99999`), except where an attribute is known to be wrong
# (IBIS land fraction: `missing_value = 0` marks a valid value).
# ---------------------------------------------------------------------------

"""
    AbstractAxis

A coordinate axis of a gridded input file: [`LongitudeAxis`](@ref),
[`LatitudeAxis`](@ref), [`TimeAxis`](@ref) or [`VerticalAxis`](@ref).
[`find_coordinate`](@ref) dispatches on it.
"""
abstract type AbstractAxis end

"Longitude axis; coordinate variable names `lon`, `longitude`, `x` (in that order)."
struct LongitudeAxis <: AbstractAxis end

"Latitude axis; coordinate variable names `lat`, `latitude`, `y` (in that order)."
struct LatitudeAxis <: AbstractAxis end

"Time axis; coordinate variable names `time`, `t` (in that order)."
struct TimeAxis <: AbstractAxis end

"""
Vertical axis of initial-condition files; coordinate variable names `lev`,
`level`, `plev`, `z`, `hybrid`, `nhym` (in that order).
"""
struct VerticalAxis <: AbstractAxis end

# Candidate names, tried in order. The single-letter names are those the
# runtime loaders have always accepted.
coordinate_names(::LongitudeAxis) = ("lon", "longitude", "x")
coordinate_names(::LatitudeAxis)  = ("lat", "latitude", "y")
coordinate_names(::TimeAxis)      = ("time", "t")
coordinate_names(::VerticalAxis)  = ("lev", "level", "plev", "z", "hybrid", "nhym")

axis_label(::LongitudeAxis) = "longitude"
axis_label(::LatitudeAxis)  = "latitude"
axis_label(::TimeAxis)      = "time"
axis_label(::VerticalAxis)  = "vertical"

_dataset_path(ds) = try
    NCDatasets.path(ds)
catch
    "the dataset"
end

"""
    find_coordinate(ds, axis::AbstractAxis; length = nothing, required = true) -> String

Name of the coordinate variable of `axis` in the NetCDF dataset `ds`: the first
of the axis's candidate names (`coordinate_names(axis)`) that is a variable of
`ds`. Coordinates are identified **by variable name only**; `units`,
`long_name`, `standard_name` and `axis` attributes are never consulted, because
real files mislabel them (FLUXCOM-X: `lon` with `units = degrees_north`) or
omit them (GFED5 NRT).

The found variable must be one-dimensional and, when `length` is given, have
`length` values (the size of the matching dimension of the data variable);
otherwise an `ArgumentError` (not 1-D) or `DimensionMismatch` (wrong length)
is thrown. Bounds variables (`lat_bnds`, `lon_bnds`, `time_bnds`) are never
returned. With `required = false` a missing coordinate returns `nothing`
instead of throwing.
"""
function find_coordinate(ds, axis::AbstractAxis;
                         length::Union{Nothing, Integer} = nothing, required::Bool = true)
    names = coordinate_names(axis)
    for name in names
        haskey(ds, name) || continue
        v = NCDatasets.variable(ds, name)
        ndims(v) == 1 || throw(ArgumentError(
            "$(axis_label(axis)) coordinate '$(name)' in $(_dataset_path(ds)) has " *
            "$(ndims(v)) dimensions; a coordinate variable must be one-dimensional"))
        length === nothing || Base.length(v) == length || throw(DimensionMismatch(
            "$(axis_label(axis)) coordinate '$(name)' in $(_dataset_path(ds)) has " *
            "$(Base.length(v)) values; the data need $(length)"))
        return String(name)
    end
    required && throw(ArgumentError(
        "no $(axis_label(axis)) coordinate in $(_dataset_path(ds)); looked for the " *
        "variables $(join(repr.(names), ", "))"))
    return nothing
end

# ---------------------------------------------------------------------------
# Lat-lon orientation
# ---------------------------------------------------------------------------

"""
    LatLonOrientation(flip_lat, lon_shift, wrap_lon)

How [`orient`](@ref) reorders a latitude-longitude field `(lon, lat, ...)`:

- `flip_lat`: reverse the latitude axis (the file stores north to south);
- `lon_shift`: move the first `lon_shift` longitude columns to the end
  (a circular shift by `-lon_shift`);
- `wrap_lon`: map the reordered longitudes into `[0°, 360°)`
  ([`orient_coordinates`](@ref)).

Built by [`latlon_orientation`](@ref).
"""
struct LatLonOrientation
    flip_lat  :: Bool
    lon_shift :: Int
    wrap_lon  :: Bool
end

"""
    latlon_orientation(lon, lat) -> LatLonOrientation

Orientation that puts a file's latitude-longitude field into the runtime's
convention: latitude ascending (south to north), and, when any longitude is
negative, the longitude axis rolled to start at its first value `≥ 0` and the
coordinates taken modulo 360°. A longitude axis without negative values, or
with no value `≥ 0`, is left as it is.

Examples: centres `-179.5 … 179.5` become `0.5 … 359.5` (shift 180); the
wrapped CLM-FATES axis `0 … 177.5, -180 … -2.5` keeps its order (shift 0) and
becomes `0 … 357.5`; latitudes `89.75 … -89.75` (VISIT) are flipped.
"""
function latlon_orientation(lon::AbstractVector{<:Real}, lat::AbstractVector{<:Real})
    isempty(lon) && throw(ArgumentError("longitude coordinate is empty"))
    isempty(lat) && throw(ArgumentError("latitude coordinate is empty"))
    flip_lat = length(lat) > 1 && lat[1] > lat[end]
    if minimum(lon) < 0
        split = findfirst(>=(0), lon)
        split === nothing || return LatLonOrientation(flip_lat, split - 1, true)
    end
    return LatLonOrientation(flip_lat, 0, false)
end

_longitude_order(o::LatLonOrientation, Nx::Integer) =
    o.lon_shift == 0 ? (1:Nx) : vcat((o.lon_shift + 1):Nx, 1:o.lon_shift)
_latitude_order(o::LatLonOrientation, Ny::Integer) = o.flip_lat ? (Ny:-1:1) : (1:Ny)

function _check_orientation(o::LatLonOrientation, Nx::Integer)
    0 <= o.lon_shift < max(Nx, 1) || throw(ArgumentError(
        "orientation shifts $(o.lon_shift) longitude columns, but the field has $(Nx)"))
    return nothing
end

"""
    orient(A::AbstractArray, o::LatLonOrientation) -> Array

A new array with the two leading dimensions `(lon, lat)` of `A` reordered by
`o` (longitude rolled, latitude flipped); trailing dimensions (time, level,
...) are untouched. Values are copied, never computed, so the result is
bitwise the input in the new order.
"""
function orient(A::AbstractArray, o::LatLonOrientation)
    ndims(A) >= 2 || throw(ArgumentError(
        "orient needs a field with (lon, lat) leading dimensions; got ndims = $(ndims(A))"))
    Nx, Ny = size(A, 1), size(A, 2)
    _check_orientation(o, Nx)
    trailing = ntuple(_ -> Colon(), ndims(A) - 2)
    return A[_longitude_order(o, Nx), _latitude_order(o, Ny), trailing...]
end

"""
    orient_coordinates(lon, lat, o::LatLonOrientation) -> (lon, lat)

The coordinates that match [`orient`](@ref)`(A, o)`, as `Vector{Float64}`:
latitude reversed when `o.flip_lat`; longitude rolled by `o.lon_shift` and,
when `o.wrap_lon`, taken modulo 360° (`mod(λ, 360)`).
"""
function orient_coordinates(lon::AbstractVector{<:Real}, lat::AbstractVector{<:Real},
                            o::LatLonOrientation)
    _check_orientation(o, length(lon))
    lon64 = Float64.(lon)
    lat64 = Float64.(lat)
    lat_out = o.flip_lat ? reverse(lat64) : lat64
    lon_out = o.wrap_lon ? mod.(lon64[_longitude_order(o, length(lon64))], 360.0) : lon64
    return lon_out, lat_out
end

# ---------------------------------------------------------------------------
# Fill policies
# ---------------------------------------------------------------------------

"""
    AbstractFillPolicy

Which attributes of a variable mark fill values, and what [`read_raw`](@ref)
puts in their place:

| Policy | Fill marked by | Fill becomes | Use |
|---|---|---|---|
| [`MaskedAsZero`](@ref) | `_FillValue` and `missing_value` | `0.0` | runtime flux loaders; FLUXCOM-X land fraction |
| [`MaskedAsNaN`](@ref) | `_FillValue` and `missing_value` | `NaN` | flux sources whose statistics count only valid cells |
| [`FillValueOnly`](@ref) | `_FillValue` only | `NaN` | variables whose `missing_value` is wrong (IBIS land fraction) |

A `_FillValue` of NaN matches NaN values. `scale_factor` and `add_offset` are
applied as NCDatasets applies them (no flux variable of the measured inputs
has them). `valid_min`, `valid_max` and `valid_range` are not applied.
"""
abstract type AbstractFillPolicy end

"""
    MaskedAsZero()

Fill policy: values equal to `_FillValue` or to any `missing_value` become
`0.0`. This is the runtime convention (`nomissing(ds[var][...], 0)`).
"""
struct MaskedAsZero <: AbstractFillPolicy end

"""
    MaskedAsNaN()

Fill policy: values equal to `_FillValue` or to any `missing_value` become
`NaN`, so a caller can count valid values (`isfinite`) before replacing them.
VISIT, VISIT-UT and iMAPLE mark fill only through `missing_value = -99999`.
"""
struct MaskedAsNaN <: AbstractFillPolicy end

"""
    FillValueOnly()

Fill policy: only `_FillValue` marks fill (it becomes `NaN`); `missing_value`
is ignored. For variables whose `missing_value` attribute is wrong: the IBIS
land-fraction file declares `missing_value = 0`, which is a valid fraction.
"""
struct FillValueOnly <: AbstractFillPolicy end

fill_replacement(::MaskedAsZero)  = 0.0
fill_replacement(::MaskedAsNaN)   = NaN
fill_replacement(::FillValueOnly) = NaN

# A CF view of the variable with the policy's fill attributes, no time
# decoding, and `missing` as the masking value whatever the dataset's default.
_cf_view(ds, name, ::Union{MaskedAsZero, MaskedAsNaN}) =
    NCDatasets.cfvariable(ds, name; units = nothing, calendar = nothing, maskingvalue = missing)
_cf_view(ds, name, ::FillValueOnly) =
    NCDatasets.cfvariable(ds, name; missing_value = Float64[], units = nothing,
                          calendar = nothing, maskingvalue = missing)

# Each stored value converted once, straight to `T` (`T(x)`, as
# `T.(nomissing(ds[name][idx...], zero(T)))` did); fill becomes `T(replacement)`.
function _replace_fill(::Type{T}, data::AbstractArray, replacement::Float64) where T
    out = Array{T}(undef, size(data))
    r = T(replacement)
    @inbounds for i in eachindex(data, out)
        x = data[i]
        out[i] = x === missing ? r : T(x)
    end
    return out
end
_replace_fill(::Type{T}, x, replacement::Float64) where T = fill(x === missing ? T(replacement) : T(x))

"""
    read_raw([T = Float64,] ds, name, policy::AbstractFillPolicy, idx...) -> Array{T}

Values of variable `name` of the NetCDF dataset `ds` at indices `idx` (all
values when `idx` is empty), as element type `T` (`Float64` by default), with
fill values replaced by the rule of `policy` ([`MaskedAsZero`](@ref),
[`MaskedAsNaN`](@ref), [`FillValueOnly`](@ref)). Indices follow the Julia
(reversed CF) dimension order, as `ds[name][idx...]`. All-integer indices give
a 0-dimensional array.

Each stored value is converted once, straight to `T`, so a Float32 run reads
into Float32 without a Float64 copy, and `MaskedAsZero` reproduces
`T.(nomissing(ds[name][idx...], zero(T)))` bit for bit.
"""
read_raw(ds, name::AbstractString, policy::AbstractFillPolicy, idx...) =
    read_raw(Float64, ds, name, policy, idx...)

function read_raw(::Type{T}, ds, name::AbstractString, policy::AbstractFillPolicy,
                  idx...) where {T <: AbstractFloat}
    haskey(ds, name) || throw(ArgumentError(
        "variable '$(name)' not found in $(_dataset_path(ds))"))
    v = _cf_view(ds, name, policy)
    data = isempty(idx) ? Array(v) : v[idx...]
    return _replace_fill(T, data, fill_replacement(policy))
end
