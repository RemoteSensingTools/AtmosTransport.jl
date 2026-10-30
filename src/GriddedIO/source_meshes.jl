# ---------------------------------------------------------------------------
# Source lat-lon meshes: the cell faces a file's coordinates imply
#
# A gridded input file stores cell centres, not faces. The faces are derived
# from the centres by the rules below (measured on GFED5, FLUXCOM-X and the
# 23 usable TRENDY v14 S3 models, P0 facts memo Sections 2 and 5) and handed to
# the explicit-face constructor `Grids.LatLonMesh(λᶠ, φᶠ; radius)`.
# Bounds variables (`lat_bnds`, `lon_bnds`) are never used: FLUXCOM-X
# `lon_bnds` run from -179.99999999999997 to 180.00000000000003, and ELM's
# pole-row `lat_bnds` reach ±90.471°.
# ---------------------------------------------------------------------------

"""
Largest residual, in cells, of a stored centre of a regular global axis from
its snapped position (snapping rule, step 2). Measured residuals: 0 for GFED5,
at most 1.1e-13 for FLUXCOM-X; a half-cell or wrong-resolution error is
0.5 or more.
"""
const SNAP_RESIDUAL_TOLERANCE = 1e-6

# Precision in which a coordinate was stored: Float32 for a Float32 vector or
# for Float64 values that are all Float32 numbers (GridFED stores its Float32
# centres as Float64, e.g. -179.9499969482422), else Float64. A longitude
# (`period = 360`) also counts as a Float32 number when it is one after
# subtracting 360°: the runtime's orientation maps negative Float32 longitudes
# to `λ + 360`, exactly. The snapping and uniformity tolerances allow for the
# rounding of that precision.
_is_float32_number(x::Real) = Float64(Float32(x)) == Float64(x)
_coordinate_precision(::AbstractVector{Float32}; period::Real = 0) = Float32
_coordinate_precision(x::AbstractVector{<:Real}; period::Real = 0) =
    all(v -> _is_float32_number(v) || (period != 0 && _is_float32_number(v - period)), x) ?
        Float32 : Float64

# Allowance, in cells of width `res`, for the storage rounding of centres of
# magnitude up to `maxabs`: two units of `eps(T)` (four of the unit roundoff).
_rounding_allowance(::Type{T}, maxabs::Real, res::Real) where T =
    2 * Float64(eps(T)) * Float64(maxabs) / Float64(res)

function _check_coordinate_centres(x::AbstractVector{<:Real}, name::AbstractString)
    length(x) >= 2 || throw(ArgumentError(
        "a source grid needs at least two $(name) centres; got $(length(x))"))
    all(isfinite, x) || throw(ArgumentError("$(name) centres must be finite"))
    return nothing
end

# ---- Latitude ----

# Faces of a global, regular, cell-centred latitude axis (snapping rule):
# n centres φ_k = x₀ + s (k + ½) res with res = 180/n, x₀ = -90 (ascending,
# s = +1) or +90 (descending, s = -1). Faces x₀ + s j res, j = 0 … n, each
# the correctly rounded value of the integer ratio (x₀ n + s j 180) / n, so
# the outer faces are exactly ±90.
function _snapped_latitude_faces(φ::Vector{Float64}, ::Type{T}) where T
    n = length(φ)
    s = φ[end] > φ[1] ? 1 : -1
    x₀ = -90 * s
    res = 180 / n
    tol = SNAP_RESIDUAL_TOLERANCE + _rounding_allowance(T, maximum(abs, φ), res)
    for k in 0:(n - 1)
        q = (φ[k + 1] - x₀) / (s * res) - 0.5
        k̂ = round(Int, q)
        (k̂ == k && abs(q - k̂) <= tol) || throw(ArgumentError(
            "latitude centres span a global grid of $(n) rows but are not cell-centred " *
            "on it: centre $(k + 1) is $(φ[k + 1])°, $(q - k) cells from " *
            "$(x₀ + s * (k + 0.5) * res)° (tolerance $(tol) cells); a half-cell offset " *
            "or a wrong resolution"))
    end
    return [Float64(x₀ * n + s * j * 180) / n for j in 0:n]
end

# The outer face beyond centre `c`, whose neighbour is `c_adj`: the pole when
# the grid reaches it (the pole is closer to `c` than the adjacent spacing:
# pole-centred rows, regular and Gaussian grids), otherwise `c` minus half
# the adjacent spacing (a grid truncated there, such as the FATES grids south
# of -59.684°, or a regional grid).
function _outer_latitude_face(c::Float64, c_adj::Float64)
    pole = c > c_adj ? 90.0 : -90.0
    return abs(pole - c) < abs(c - c_adj) ? pole : c + (c - c_adj) / 2
end

# Faces of any other latitude axis: midpoints of adjacent stored centres, and
# the outer faces of `_outer_latitude_face`.
function _midpoint_latitude_faces(φ::Vector{Float64})
    n = length(φ)
    faces = Vector{Float64}(undef, n + 1)
    for k in 2:n
        faces[k] = (φ[k - 1] + φ[k]) / 2
    end
    faces[1] = _outer_latitude_face(φ[1], φ[2])
    faces[n + 1] = _outer_latitude_face(φ[n], φ[n - 1])
    return faces
end

function _latitude_faces(lat::AbstractVector{<:Real})
    _check_coordinate_centres(lat, "latitude")
    T = _coordinate_precision(lat)
    φ = Float64.(lat)
    maximum(abs, φ) <= 90 || throw(ArgumentError(
        "latitude centres must lie in [-90, 90]; got [$(minimum(φ)), $(maximum(φ))]"))
    ascending = φ[2] > φ[1]
    for k in 2:length(φ)
        (ascending ? φ[k] > φ[k - 1] : φ[k] < φ[k - 1]) || throw(ArgumentError(
            "latitude centres must be strictly monotone; centres $(k - 1) and $(k) are " *
            "$(φ[k - 1])° and $(φ[k])°"))
    end
    n = length(φ)
    Δ = abs(φ[end] - φ[1]) / (n - 1)
    # n cells of the mean spacing span 180°: a global cell-centred grid, which
    # must snap. (Pole-centred grids span 180° with n - 1 spacings.)
    abs(n * Δ - 180) < Δ / 2 && return _snapped_latitude_faces(φ, T)
    return _midpoint_latitude_faces(φ)
end

# ---- Longitude ----

# Centres unwrapped through one -360° wrap (CLM-FATES: 0 … 177.5, then
# -180 … -2.5): every centre after the negative step is shifted by +360°.
function _unwrapped_longitudes(λ::Vector{Float64})
    u = similar(λ)
    wraps = 0
    for k in eachindex(λ)
        k > firstindex(λ) && λ[k] < λ[k - 1] && (wraps += 1)
        u[k] = λ[k] + 360 * wraps
    end
    wraps <= 1 || throw(ArgumentError(
        "longitude centres must increase eastward with at most one wrap by -360°; " *
        "they decrease $(wraps) times"))
    return u
end

# Faces of a global, regular longitude axis of n centres, res = 360/n, origin
# x₀ = -180 (some centre < 0) or 0. Either the centres are cell-centred,
# x₀ + (m + ½) res (snapping rule), or a centre sits on a multiple of res from
# x₀ (on 0° or ±180°, CLASSIC, CLM, ELM, JSBACH, the FATES grids): then the
# faces are the centres ± res/2 and the first face is not ±180 or 0. Faces are
# multiples h of res/2, the correctly rounded integer ratios
# (h 180 + x₀ n) / n; they increase monotonically (an unwrapped axis).
function _snapped_longitude_faces(λ::Vector{Float64}, u::Vector{Float64}, ::Type{T}) where T
    n = length(λ)
    res = 360 / n
    x₀ = minimum(λ) >= 0 ? 0 : -180
    tol = SNAP_RESIDUAL_TOLERANCE + _rounding_allowance(T, maximum(abs, λ), res)
    q₀ = (u[1] - x₀) / res
    f = q₀ - floor(q₀)
    h₀, q_first = if abs(f - 0.5) <= tol                 # cell-centred
        m = round(Int, q₀ - 0.5)
        2m, m + 0.5
    elseif min(f, 1 - f) <= tol                         # centre on a multiple of res
        m = round(Int, q₀)
        2m - 1, Float64(m)
    else
        throw(ArgumentError(
            "longitude centres span a global grid of $(n) columns but are neither " *
            "cell-centred nor centred on multiples of the spacing from $(x₀)°: the first " *
            "centre $(λ[1])° is $(f) cells from a face (tolerance $(tol) cells)"))
    end
    for k in 1:n
        q = (u[k] - x₀) / res
        r = q - (q_first + (k - 1))
        abs(r) <= tol || throw(ArgumentError(
            "longitude centres are not regular on a global grid of $(n) columns: centre " *
            "$(k) is $(λ[k])°, $(r) cells from its regular position (tolerance $(tol) cells)"))
    end
    return [Float64((h₀ + 2j) * 180 + x₀ * n) / n for j in 0:n]
end

# Faces of a regional regular longitude axis (spanning less than 360°): the
# first and last centres ± half the mean spacing, uniform in between.
function _regional_longitude_faces(u::Vector{Float64}, Δ::Float64, ::Type{T}) where T
    n = length(u)
    tol = SNAP_RESIDUAL_TOLERANCE * Δ + 2 * Float64(eps(T)) * maximum(abs, u)
    for k in 1:(n - 1)
        step = u[k + 1] - u[k]
        abs(step - Δ) <= tol || throw(ArgumentError(
            "longitude centres must be uniformly spaced: step $(k) is $(step)°, the mean " *
            "spacing is $(Δ)°"))
    end
    return collect(range(u[1] - Δ / 2, u[end] + Δ / 2, length = n + 1))
end

function _longitude_faces(lon::AbstractVector{<:Real})
    _check_coordinate_centres(lon, "longitude")
    T = _coordinate_precision(lon; period = 360)
    λ = Float64.(lon)
    u = _unwrapped_longitudes(λ)
    n = length(u)
    Δ = (u[end] - u[1]) / (n - 1)
    Δ > 0 || throw(ArgumentError("longitude centres must increase eastward"))
    abs(n * Δ - 360) < Δ / 2 && return _snapped_longitude_faces(λ, u, T)
    n * Δ < 360 || throw(ArgumentError(
        "longitude centres of mean spacing $(Δ)° span more than 360° ($(n) centres)"))
    return _regional_longitude_faces(u, Δ, T)
end

# ---- Face rules ----

"""
    AbstractFaceRule

How [`source_latlon_mesh`](@ref) derives cell faces from a file's cell
centres: [`SnappedFaces`](@ref) (the default).
"""
abstract type AbstractFaceRule end

"""
    SnappedFaces()

Face rule of [`source_latlon_mesh`](@ref), its default: regular global axes
snap to the grid their resolution implies, with faces built from integers;
pole-centred, truncated, Gaussian and regional latitudes get midpoint faces
with the pole and truncation rules of the `source_latlon_mesh` docstring.
"""
struct SnappedFaces <: AbstractFaceRule end

# ---- Public constructor ----

"""
    source_latlon_mesh(lon, lat, rule = SnappedFaces(); radius) -> LatLonMesh{Float64}

The latitude-longitude mesh of a gridded input file whose cell centres are
`lon` and `lat` [degrees], with faces derived by the face rule `rule`
([`SnappedFaces`](@ref), described below). Read coordinate vectors with
[`find_coordinate`](@ref). `radius` [m] has no default: the source mesh must
sit on the sphere of the mesh it is regridded to.

Under `SnappedFaces`, the mesh is built on the explicit-face constructor
`LatLonMesh(λᶠ, φᶠ; radius)`; latitude rows of a descending axis are stored
ascending (`Grids.latitude_row_order` gives the data permutation), and
longitude columns keep their input order (a wrapped axis is unwrapped, not
rolled).

## Faces under `SnappedFaces`

**Regular global axes: the snapping rule.** For `n` stored centres `x_k`,
`k = 0 … n-1`, with `res = 180/n` (latitude) or `360/n` (longitude) and the
origin `x₀ = -90` (ascending latitude), `+90` (descending latitude), `-180`
(longitude with a negative centre) or `0` (longitude without one), direction
`s = ±1`:

1. `k̂ = round((x_k − x₀)/(s·res) − ½)`; require `k̂ = k` and
   `|(x_k − x₀)/(s·res) − ½ − k̂| ≤ 1e-6` cells plus the storage rounding of
   the centres, `2 eps(T) max|x| / res` cells for storage precision `T`
   (Float64 values that are all Float32 numbers, for longitudes also after
   subtracting 360°, count as Float32: GridFED's Float32 0.1° centres sit up to
   3e-5 cells off). Otherwise an `ArgumentError`: a half-cell offset or a wrong
   resolution.
2. The faces are `x₀ + s·j·res`, `j = 0 … n`, built from integers (the
   correctly rounded ratio `(x₀ n + s j 180)/n`), so the outer faces are exactly
   `±90` and `-180/180` (or `0/360`).

A longitude axis may start anywhere on that grid (after a roll or a wrap); a
global axis whose centres sit on multiples of `res` (a centre on 0° or ±180°)
gets faces at the centres ± `res/2`, so its first face is not ±180 or 0
(CLASSIC -180.5°, CLM -0.625°, CLM-FATES and ELM-FATES -1.25°,
ELM -180.625°, JSBACH -180.9375°). Any other registration throws.

**Other latitude axes** (pole-centred, truncated, Gaussian, regional): faces
at the midpoints of adjacent centres. An outer face is the pole when the outer
centre is closer to the pole than the adjacent spacing (pole-centred rows of
CLM, ELM and iMAPLE, whose polar cells are half-width caps; the Gaussian
JSBACH rows, whose midpoint faces with polar outer faces reproduce the Python
audit); otherwise it is the outer centre minus half the adjacent spacing (the
FATES grids, truncated south of their row at -59.684°: southern face
-60.6316°). Such meshes have non-uniform latitude faces (`Δφ = NaN`), so
`dx`, `dy` and other `Δφ`-based metrics throw.

**Regional longitude axes** (spanning less than 360°): uniform faces from the
first centre minus to the last centre plus half the mean spacing.

Faces are never taken from `lat_bnds`/`lon_bnds`.
"""
function source_latlon_mesh(lon::AbstractVector{<:Real}, lat::AbstractVector{<:Real},
                            rule::AbstractFaceRule = SnappedFaces(); radius::Real)
    return _source_latlon_mesh(rule, lon, lat, radius)
end

_source_latlon_mesh(::SnappedFaces, lon, lat, radius) =
    LatLonMesh(_longitude_faces(lon), _latitude_faces(lat); FT = Float64, radius)
