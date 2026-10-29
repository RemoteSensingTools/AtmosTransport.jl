# ---------------------------------------------------------------------------
# LatLonMesh — latitude-longitude horizontal mesh
#
# Implements the AbstractStructuredMesh interface with efficient (i,j) indexing.
# Longitude spacing Δλ is always uniform. Latitude faces are either uniform
# (the regular transport mesh, constructed from intervals) or explicit and
# non-uniform (source grids of input files: pole-centred, truncated, Gaussian),
# in which case Δφ is NaN and the Δφ-based metrics throw. Cell areas always
# come from the faces.
# ---------------------------------------------------------------------------

"""
    LatLonMesh{FT} <: AbstractStructuredMesh

Latitude-longitude mesh on the sphere with uniform longitude spacing.

Two constructors build it: `LatLonMesh(; Nx, Ny, longitude, latitude, ...)`
gives the regular transport mesh, and `LatLonMesh(λᶠ, φᶠ; FT, radius)` gives a
mesh from explicit cell faces (source grids of input files, whose latitude
faces may be non-uniform). The regular constructor builds its faces and then
calls the explicit one, so both share one construction path.

# Fields
- `Nx`, `Ny` — number of cells in longitude and latitude
- `Δλ` — uniform longitude spacing [degrees]
- `Δφ` — uniform latitude spacing [degrees], or `NaN` when the latitude faces
  are not uniform (then `dx`, `dy` and the x-face lengths throw)
- `λᶜ`, `λᶠ` — cell-center and cell-face longitudes [degrees]; the faces
  increase eastward, `λᶠ[i+1] = λᶠ[i] + Δλ`, from any first face
- `φᶜ`, `φᶠ` — cell-center and cell-face latitudes [degrees]; the faces are
  strictly ascending (south to north) in `[-90, 90]`
- `radius`   — planet radius [m]

Cell centres are the face midpoints, `λᶜ[i] = (λᶠ[i] + λᶠ[i+1]) / 2` and
`φᶜ[j] = (φᶠ[j] + φᶠ[j+1]) / 2`.
"""
struct LatLonMesh{FT <: AbstractFloat} <: AbstractStructuredMesh
    Nx :: Int
    Ny :: Int
    Δλ :: FT
    Δφ :: FT
    λᶜ :: Vector{FT}
    λᶠ :: Vector{FT}
    φᶜ :: Vector{FT}
    φᶠ :: Vector{FT}
    radius :: FT
end

Base.eltype(::LatLonMesh{FT}) where FT = FT
Base.eltype(::Type{<:LatLonMesh{FT}}) where FT = FT

function _validate_latlon_size(Nx, Ny, size)
    if isnothing(size)
        isnothing(Nx) && throw(ArgumentError("LatLonMesh requires either Nx/Ny or size=(Nx, Ny)"))
        isnothing(Ny) && throw(ArgumentError("LatLonMesh requires either Nx/Ny or size=(Nx, Ny)"))
        Nx = Int(Nx)
        Ny = Int(Ny)
    else
        length(size) == 2 || throw(ArgumentError("size must be a 2-tuple (Nx, Ny)"))
        Nx_size, Ny_size = Int.(size)
        if !isnothing(Nx) && Nx != Nx_size
            throw(ArgumentError("Nx=$Nx conflicts with size[1]=$Nx_size"))
        end
        if !isnothing(Ny) && Ny != Ny_size
            throw(ArgumentError("Ny=$Ny conflicts with size[2]=$Ny_size"))
        end
        Nx, Ny = Nx_size, Ny_size
    end

    Nx > 0 || throw(ArgumentError("Nx must be positive"))
    Ny > 0 || throw(ArgumentError("Ny must be positive"))

    return Nx, Ny
end

function _validate_interval(name::AbstractString, interval, ::Type{FT};
                            lower = -Inf, upper = Inf,
                            max_extent = Inf) where FT <: AbstractFloat
    length(interval) == 2 || throw(ArgumentError("$name must be a 2-tuple"))
    left, right = FT.(interval)
    left < right || throw(ArgumentError("$name must be strictly increasing"))
    left >= FT(lower) || throw(ArgumentError("$name lower bound must be >= $lower"))
    right <= FT(upper) || throw(ArgumentError("$name upper bound must be <= $upper"))
    right - left <= FT(max_extent) || throw(ArgumentError("$name extent must be <= $max_extent"))
    return left, right
end

@inline _regular_faces(left, right, N, ::Type{FT}) where FT = collect(range(left, right, length=N+1) .|> FT)
@inline _regular_centers(faces::AbstractVector{FT}) where FT = FT[(faces[i] + faces[i+1]) / 2 for i in 1:length(faces)-1]

"""
    LatLonMesh(; Nx, Ny, longitude=(-180, 180), latitude=(-90, 90), ...)

Construct a regular latitude-longitude mesh: `Nx` × `Ny` cells of uniform
spacing over the `longitude` and `latitude` intervals [degrees]. The faces are
built here and passed to the explicit-face constructor
`LatLonMesh(λᶠ, φᶠ; FT, radius)`, the one construction path of the type.

## Coordinate conventions

- **Longitude**: cell faces at `λᶠ[1..Nx+1]` spanning `[λ_west, λ_east]`,
  cell centers at midpoints `λᶜ[i] = (λᶠ[i] + λᶠ[i+1]) / 2`.
  Default range `(-180, 180)` → `λᶜ[1] = -178.125°` for `Nx=96`.
  **IMPORTANT**: the spectral preprocessor's FFT synthesis
  (`spectral_to_grid!`) must use `lon_shift_rad = deg2rad(λᶜ[1])` to align
  its output with this mesh. Using `dlon/2` instead causes a 180° shift
  for `(-180, 180)` meshes (see 2026-04-11 bug fix in spectral_synthesis.jl).

- **Latitude**: cell faces at `φᶠ[1..Ny+1]` spanning `[φ_south, φ_north]`,
  cell centers at `φᶜ[j]`. Default `(-90, 90)` → `φᶜ[1] = -88.125°` for
  `Ny=48`. Latitude is NOT periodic (closed pole boundaries).

## Indexing (column-major)

3D arrays on this mesh are `(Nx, Ny, Nz)` where:
- `i` (lon) is the **fastest-varying** index (column-major, Julia/Fortran style)
- `j` (lat) is the middle index
- `k` (level) is the slowest-varying index; `k=1` is TOA, `k=Nz` is surface

The flat cell index is `c = i + (j-1) × Nx` (1-based). Face arrays for
the Arakawa C-grid are `am[Nx+1, Ny, Nz]` (zonal, at west edges) and
`bm[Nx, Ny+1, Nz]` (meridional, at south edges).
"""
function LatLonMesh(;
        FT::Type{<:AbstractFloat} = Float64,
        Nx::Union{Nothing, Int} = nothing,
        Ny::Union{Nothing, Int} = nothing,
        size = nothing,
        longitude = (-180, 180),   # [degrees] — determines λᶜ and λᶠ
        latitude  = (-90, 90),     # [degrees] — determines φᶜ and φᶠ
        radius    = FT(EARTH_RADIUS))   # [m] — planet radius

    Nx, Ny = _validate_latlon_size(Nx, Ny, size)
    λ_west, λ_east = _validate_interval("longitude", longitude, FT; max_extent=360)
    φ_south, φ_north = _validate_interval("latitude", latitude, FT; lower=-90, upper=90)

    # Range endpoints are exact, so the explicit-face constructor's
    # Δλ = (λᶠ[end] - λᶠ[1]) / Nx equals (λ_east - λ_west) / Nx, and the
    # correctly rounded range steps pass its uniformity test (Δφ finite).
    λᶠ = _regular_faces(λ_west, λ_east, Nx, FT)   # cell-edge longitudes [Nx+1]
    φᶠ = _regular_faces(φ_south, φ_north, Ny, FT)  # cell-edge latitudes [Ny+1]
    return LatLonMesh(λᶠ, φᶠ; FT, radius)
end

# ---- Explicit cell faces ----

# Relative tolerance of the face-spacing uniformity test (design 2.3).
const _FACE_UNIFORMITY_RTOL = 1e-10

# Precision of the face values a caller passed. The uniformity test allows for
# their own rounding: Float32 faces of a regular mesh step unevenly by an ulp.
_face_input_float_type(faces::AbstractVector) =
    eltype(faces) <: AbstractFloat ? eltype(faces) : Float64

# Largest admissible |step - Δ| of one face step around the mean step Δ:
# the relative tolerance plus two units of roundoff of the largest face value
# (each face is within half an ulp of the uniform value, so a step is within
# one ulp of the largest face).
function _face_step_tolerance(Δ, faces::AbstractVector{Float64}, ::Type{T}) where T
    return _FACE_UNIFORMITY_RTOL * abs(Δ) + 2 * Float64(eps(T)) * maximum(abs, faces)
end

# Widen a face value before adding the 360° unwrapping shift, so the shifted
# face is rounded to the mesh precision once.
_widen_face(x::Real) = convert(promote_type(float(typeof(x)), Float64), x)

function _check_face_count(faces::AbstractVector, name::AbstractString)
    length(faces) >= 2 || throw(ArgumentError(
        "LatLonMesh needs at least two $name faces (one cell); got $(length(faces))"))
    all(isfinite, faces) || throw(ArgumentError("$name faces must be finite"))
    return length(faces) - 1
end

# Longitude faces → (stored faces::Vector{FT}, Δλ::FT). A negative step is a
# wrap through the date line or the prime meridian: that face and every later
# one are shifted by +360°, so the stored faces increase monotonically and the
# data columns keep their file order.
function _explicit_longitude_faces(λᶠ::AbstractVector{<:Real}, ::Type{FT}) where FT
    Nx = _check_face_count(λᶠ, "longitude")
    wraps = 0
    unwrapped = Vector{Float64}(undef, Nx + 1)
    faces = Vector{FT}(undef, Nx + 1)
    @inbounds for i in 1:(Nx + 1)
        i > 1 && λᶠ[i] < λᶠ[i - 1] && (wraps += 1)
        unwrapped[i] = Float64(λᶠ[i]) + 360.0 * wraps
        faces[i] = wraps == 0 ? FT(λᶠ[i]) : FT(_widen_face(λᶠ[i]) + 360 * wraps)
    end
    T = _face_input_float_type(λᶠ)
    span = unwrapped[end] - unwrapped[1]
    Δ = span / Nx
    tol = _face_step_tolerance(Δ, unwrapped, T)
    span <= 360 * (1 + _FACE_UNIFORMITY_RTOL) + tol || throw(ArgumentError(
        "longitude faces must increase eastward (a wrap by -360° is allowed) and " *
        "span at most 360°; after unwrapping they span $(span)°"))
    @inbounds for i in 1:Nx
        step = unwrapped[i + 1] - unwrapped[i]
        step > 0 && abs(step - Δ) <= tol || throw(ArgumentError(
            "longitude faces must be uniformly spaced: step $i is $(step)°, " *
            "the mean step is $(Δ)° (relative tolerance $(_FACE_UNIFORMITY_RTOL))"))
    end
    return faces, (faces[end] - faces[1]) / Nx
end

# `true` for strictly ascending faces, `false` for strictly descending ones;
# anything else throws.
function _ring_latitude_faces_ascending(φᶠ::AbstractVector{<:Real})
    _check_face_count(φᶠ, "latitude")
    ascending = φᶠ[2] > φᶠ[1]
    @inbounds for j in 2:length(φᶠ)
        (ascending ? φᶠ[j] > φᶠ[j - 1] : φᶠ[j] < φᶠ[j - 1]) || throw(ArgumentError(
            "latitude faces must be strictly monotone; faces $(j - 1) and $j are " *
            "$(φᶠ[j - 1])° and $(φᶠ[j])°"))
    end
    return ascending
end

"""
    latitude_row_order(φᶠ) -> StepRange{Int, Int}

Order in which [`LatLonMesh`](@ref) built from latitude faces `φᶠ` stores the
latitude rows: mesh row `j` is input row `latitude_row_order(φᶠ)[j]`.

The mesh always stores ascending (south-to-north) faces, the orientation the
transport code, the conservative regridder (counter-clockwise cell polygons),
the observation locator and the output writers expect. The order is `1:1:Ny`
for ascending faces and `Ny:-1:1` for descending ones (files stored north to
south, such as VISIT), so `view(data, :, latitude_row_order(φᶠ))` puts an
`(Nx, Ny)` input field on the mesh. Longitude needs no reordering: wrapped
longitude faces are unwrapped, not rolled, so mesh column `i` is input column
`i`. Throws an `ArgumentError` unless `φᶠ` is finite and strictly monotone.
"""
function latitude_row_order(φᶠ::AbstractVector{<:Real})
    Ny = length(φᶠ) - 1
    return _ring_latitude_faces_ascending(φᶠ) ? (1:1:Ny) : (Ny:-1:1)
end

# Latitude faces → (ascending faces::Vector{FT}, Δφ::FT), Δφ = NaN when the
# faces are not uniform.
function _explicit_ring_latitude_faces(φᶠ::AbstractVector{<:Real}, ::Type{FT}) where FT
    Ny = length(φᶠ) - 1
    order = _ring_latitude_faces_ascending(φᶠ) ? (1:1:(Ny + 1)) : ((Ny + 1):-1:1)   # face order
    faces64 = Float64[φᶠ[r] for r in order]
    first(faces64) >= -90 && last(faces64) <= 90 || throw(ArgumentError(
        "latitude faces must lie in [-90, 90]; got [$(first(faces64)), $(last(faces64))]"))
    Δ = (faces64[end] - faces64[1]) / Ny
    tol = _face_step_tolerance(Δ, faces64, _face_input_float_type(φᶠ))
    uniform = all(j -> abs((faces64[j + 1] - faces64[j]) - Δ) <= tol, 1:Ny)
    faces = FT[φᶠ[r] for r in order]
    return faces, uniform ? (faces[end] - faces[1]) / Ny : FT(NaN)
end

"""
    LatLonMesh(λᶠ, φᶠ; FT = Float64, radius)

Latitude-longitude mesh from explicit cell faces [degrees]: `λᶠ` holds the
`Nx + 1` longitude faces and `φᶠ` the `Ny + 1` latitude faces. This is the
constructor for the source grids of input files; the regular
`LatLonMesh(; Nx, Ny, ...)` builds its faces and calls it.

`radius` [m] has no default: a source mesh must sit on the sphere of the mesh
it is regridded to (pass the target mesh's `radius`), so conservative weights
compare areas on one sphere.

## Longitude faces

- Spacing is uniform: every step `λᶠ[i+1] - λᶠ[i]` equals
  `Δλ = (λᶠ[end] - λᶠ[1]) / Nx` within a relative tolerance of `1e-10`, plus
  two units of roundoff of the input precision times `max |λᶠ|` (so Float32
  faces of a regular mesh pass). Non-uniform longitudes throw.
- The first face may be any value. A centre on 0° or ±180° puts the date line
  inside a cell (first faces -180.5°, -0.625°, -1.25°, -180.625°, -180.9375°
  in TRENDY v14 grids); `0 … 360` grids are accepted as they are.
- A sequence that wraps by -360° (centres `0 … 177.5`, then `-180 … -2.5`,
  faces `-1.25 … 178.75, -178.75 … -1.25`) is unwrapped: every face after a
  negative step is shifted by +360°, so the stored faces increase
  (`-1.25 … 358.75`). Mesh column `i` stays input column `i`.
- The span `λᶠ[end] - λᶠ[1]` (after unwrapping) is at most 360° within the
  same tolerance.

## Latitude faces

- Strictly monotone and within `[-90, 90]`. Descending faces (files stored
  north to south) are stored ascending; [`latitude_row_order`](@ref) returns
  the row permutation the caller applies to its data.
- Uniform faces (same tolerance) give `Δφ = (φᶠ[end] - φᶠ[1]) / Ny`.
  Non-uniform faces (pole-centred rows with half-width polar caps, a truncated
  first row, Gaussian latitudes) give `Δφ = NaN`; `dx`, `dy` and the x-face
  lengths (`face_length` of a meridional face) then throw an `ArgumentError`,
  so such a mesh cannot enter Δφ-based transport metrics silently.

## Geometry

The area of cell `(i, j)` is the exact spherical area between two meridians and
two parallels,

    A_ij = R² Δλ (sin φᶠ[j+1] - sin φᶠ[j])      (Δλ in radians),

so a mesh spanning 360° × [-90°, 90°] sums to `4πR²`. Centres are face
midpoints in degrees, `λᶜ[i] = (λᶠ[i] + λᶠ[i+1]) / 2`,
`φᶜ[j] = (φᶠ[j] + φᶠ[j+1]) / 2`. For pole-centred rows (whose file coordinate
is the pole) and Gaussian rows they differ from the file's coordinate: the
faces define the cell, the file coordinate only labels it.

Faces of a regular mesh reproduce `LatLonMesh(; ...)` bit for bit.
"""
function LatLonMesh(λᶠ::AbstractVector{<:Real}, φᶠ::AbstractVector{<:Real};
                    FT::Type{<:AbstractFloat} = Float64, radius::Real)
    λ_faces, Δλ = _explicit_longitude_faces(λᶠ, FT)
    φ_faces, Δφ = _explicit_ring_latitude_faces(φᶠ, FT)
    Nx = length(λ_faces) - 1
    Ny = length(φ_faces) - 1
    return LatLonMesh{FT}(Nx, Ny, Δλ, Δφ, _regular_centers(λ_faces), λ_faces,
                          _regular_centers(φ_faces), φ_faces, FT(radius))
end

# ---- AbstractStructuredMesh interface ----

nx(m::LatLonMesh) = m.Nx
ny(m::LatLonMesh) = m.Ny
ncells(m::LatLonMesh) = m.Nx * m.Ny
nfaces(m::LatLonMesh) = (m.Nx + 1) * m.Ny + m.Nx * (m.Ny + 1)

function Base.summary(m::LatLonMesh)
    FT = eltype(m)
    return string(m.Nx, "×", m.Ny, " LatLonMesh{", FT, "}")
end

function Base.show(io::IO, m::LatLonMesh)
    latitude_spacing = isnan(m.Δφ) ? "non-uniform faces" : string("Δφ=", m.Δφ)
    print(io, summary(m), "\n",
          "├── longitude: [", first(m.λᶠ), ", ", last(m.λᶠ), "] degrees with Δλ=", m.Δλ, "\n",
          "├── latitude:  [", first(m.φᶠ), ", ", last(m.φᶠ), "] degrees with ", latitude_spacing, "\n",
          "└── radius:    ", m.radius, " m")
end

@inline function _check_latlon_ij(m::LatLonMesh, i::Integer, j::Integer)
    1 <= i <= m.Nx || throw(BoundsError(1:m.Nx, i))
    1 <= j <= m.Ny || throw(BoundsError(1:m.Ny, j))
    return i, j
end

@inline function _check_latlon_cell(m::LatLonMesh, c::Integer)
    1 <= c <= ncells(m) || throw(BoundsError(1:ncells(m), c))
    return c
end

@inline function _check_latlon_face(m::LatLonMesh, f::Integer)
    1 <= f <= nfaces(m) || throw(BoundsError(1:nfaces(m), f))
    return f
end

# ---- Geometry: cell metrics ----

"""
Cell area [m²] at structured index (i, j): the exact spherical area between
the cell's meridians and parallels, `R² Δλ |sin φᶠ[j+1] - sin φᶠ[j]|` with
`Δλ` in radians. Valid for uniform and non-uniform latitude faces.
"""
function cell_area(m::LatLonMesh{FT}, i::Integer, j::Integer) where FT
    _check_latlon_ij(m, i, j)
    R = m.radius
    return R^2 * deg2rad(m.Δλ) * abs(sind(m.φᶠ[j+1]) - sind(m.φᶠ[j]))
end
cell_area(m::LatLonMesh, c::Tuple{<:Integer, <:Integer}) = cell_area(m, c[1], c[2])

"""Cell area [m²] from flat cell index `c` (column-major: `c = i + (j-1)*Nx`)."""
function cell_area(m::LatLonMesh, c::Integer)
    _check_latlon_cell(m, c)
    j = div(c - 1, m.Nx) + 1
    return cell_area(m, 1, j)
end

"""Cell area vector (precomputed, one per latitude band)."""
function cell_areas_by_latitude(m::LatLonMesh{FT}) where FT
    return FT[cell_area(m, 1, j) for j in 1:m.Ny]
end

# ---- Internal structured-only metric helpers (not exported) ----

# Uniform latitude spacing Δφ [degrees] of `m`. A mesh built from non-uniform
# latitude faces (Δφ = NaN) has no single meridional spacing, so every Δφ-based
# metric refuses it instead of returning NaN.
function _uniform_latitude_spacing(m::LatLonMesh, metric::Symbol)
    isnan(m.Δφ) && throw(ArgumentError(
        "$(metric) needs uniform latitude faces, but this $(summary(m)) has non-uniform " *
        "latitude faces (Δφ = NaN). Use cell_area and the faces φᶠ for its geometry."))
    return m.Δφ
end

"""x-spacing [m] at latitude j. Clamped to Δy near poles. Throws for non-uniform latitude faces."""
function dx(m::LatLonMesh{FT}, j::Integer) where FT
    1 <= j <= m.Ny || throw(BoundsError(1:m.Ny, j))
    R = m.radius
    dx_val = R * cosd(m.φᶜ[j]) * deg2rad(m.Δλ)
    dy_val = R * deg2rad(_uniform_latitude_spacing(m, :dx))
    return max(dx_val, dy_val)
end

"""y-spacing [m] — uniform. Throws for non-uniform latitude faces."""
function dy(m::LatLonMesh{FT}) where FT
    return m.radius * deg2rad(_uniform_latitude_spacing(m, :dy))
end

"""x-face length [m] (meridional edges) at latitude band j. Throws for non-uniform latitude faces."""
_face_length_x(m::LatLonMesh{FT}, j::Integer) where FT =
    m.radius * deg2rad(_uniform_latitude_spacing(m, :_face_length_x))

"""y-face length [m] (zonal edges) at face latitude j (1:Ny+1)."""
function _face_length_y(m::LatLonMesh{FT}, j::Integer) where FT
    return m.radius * cosd(m.φᶠ[j]) * deg2rad(m.Δλ)
end

# ---- Universal face_length(mesh, f) implementation ----

"""
    face_length(m::LatLonMesh, f) -> FT

Length [m] of face `f` using the universal face-indexed API.

X-faces (f ∈ 1:(Nx+1)*Ny) return the meridional edge length.
Y-faces (f ∈ (Nx+1)*Ny+1 : nfaces) return the zonal edge length.
"""
function face_length(m::LatLonMesh{FT}, f::Integer) where FT
    _check_latlon_face(m, f)
    n_xfaces = (m.Nx + 1) * m.Ny
    if f <= n_xfaces
        j = div(f - 1, m.Nx + 1) + 1
        return _face_length_x(m, j)
    else
        fi = f - n_xfaces
        j = div(fi - 1, m.Nx) + 1
        return _face_length_y(m, j)
    end
end

"""
    face_normal(m::LatLonMesh, f) -> (nx, ny)

Unit normal of face `f`. X-faces → (1, 0), Y-faces → (0, 1).
"""
function face_normal(m::LatLonMesh{FT}, f::Integer) where FT
    _check_latlon_face(m, f)
    n_xfaces = (m.Nx + 1) * m.Ny
    return ifelse(f <= n_xfaces, (one(FT), zero(FT)), (zero(FT), one(FT)))
end

# ---- Face connectivity (structured shortcut) ----

"""
For structured LatLon, faces are implicitly indexed:
- X-faces: index f in 1:(Nx+1)*Ny  → face between (i-1,j) and (i,j), periodic in x
- Y-faces: index f in 1:Nx*(Ny+1)  → face between (i,j-1) and (i,j), bounded in y
Direct (i,j,k) array access is preferred over these for performance.
"""
function face_cells(m::LatLonMesh, f::Integer)
    _check_latlon_face(m, f)
    Nx = m.Nx
    n_xfaces = (Nx + 1) * m.Ny
    if f <= n_xfaces
        j = div(f - 1, Nx + 1) + 1
        i = mod(f - 1, Nx + 1) + 1
        left  = i == 1      ? Nx : i - 1
        right = i == Nx + 1 ? 1  : i
        return (left + (j-1)*Nx, right + (j-1)*Nx)
    else
        fi = f - n_xfaces
        j = div(fi - 1, Nx) + 1
        i = mod(fi - 1, Nx) + 1
        below = j == 1        ? 0 : i + (j-2)*Nx
        above = j == m.Ny + 1 ? 0 : i + (j-1)*Nx
        return (below, above)
    end
end

"""
    cell_faces(m::LatLonMesh, c) -> (west, east, south, north)

Indices of the 4 faces bounding cell `c`. Accepts either a flat cell index
(column-major: `c = i + (j-1)*Nx`) or a tuple `(i, j)`.

Returns face indices consistent with the face_length/face_normal/face_cells
numbering: X-faces 1:(Nx+1)*Ny, then Y-faces (Nx+1)*Ny+1 : nfaces.
"""
function cell_faces(m::LatLonMesh, c::Integer)
    _check_latlon_cell(m, c)
    Nx = m.Nx
    i = mod(c - 1, Nx) + 1
    j = div(c - 1, Nx) + 1
    return _cell_faces_ij(m, i, j)
end

function cell_faces(m::LatLonMesh, c::Tuple{<:Integer, <:Integer})
    return _cell_faces_ij(m, c[1], c[2])
end

function _cell_faces_ij(m::LatLonMesh, i::Integer, j::Integer)
    _check_latlon_ij(m, i, j)
    Nx = m.Nx
    n_xfaces = (Nx + 1) * m.Ny
    west  = (j - 1) * (Nx + 1) + i
    east  = (j - 1) * (Nx + 1) + i + 1
    south = n_xfaces + (j - 1) * Nx + i
    north = n_xfaces + j * Nx + i
    return (west, east, south, north)
end

export LatLonMesh, cell_areas_by_latitude, latitude_row_order
