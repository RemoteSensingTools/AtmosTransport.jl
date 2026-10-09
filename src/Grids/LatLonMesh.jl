# ---------------------------------------------------------------------------
# LatLonMesh — regular latitude-longitude horizontal mesh
#
# Implements the AbstractStructuredMesh interface with efficient (i,j) indexing.
# Geometry is derived from uniform Δλ, Δφ spacing and planet radius.
# ---------------------------------------------------------------------------

"""
    LatLonMesh{FT} <: AbstractStructuredMesh

Regular latitude-longitude mesh on the sphere.

# Fields
- `Nx`, `Ny` — number of cells in longitude and latitude
- `Δλ`, `Δφ` — uniform spacing [degrees]
- `λᶜ`, `λᶠ` — cell-center and cell-face longitudes [degrees]
- `φᶜ`, `φᶠ` — cell-center and cell-face latitudes [degrees]
- `radius`   — planet radius [m]
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

Construct a regular latitude-longitude mesh.

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

    Δλ = (λ_east - λ_west) / Nx    # [degrees] — longitudinal cell width
    Δφ = (φ_north - φ_south) / Ny   # [degrees] — latitudinal cell width

    λᶠ = _regular_faces(λ_west, λ_east, Nx, FT)   # cell-edge longitudes [Nx+1]
    λᶜ = _regular_centers(λᶠ)                      # cell-center longitudes [Nx]
    φᶠ = _regular_faces(φ_south, φ_north, Ny, FT)  # cell-edge latitudes [Ny+1]
    φᶜ = _regular_centers(φᶠ)                      # cell-center latitudes [Ny]

    return LatLonMesh{FT}(Nx, Ny, Δλ, Δφ, λᶜ, λᶠ, φᶜ, φᶠ, FT(radius))
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
    print(io, summary(m), "\n",
          "├── longitude: [", first(m.λᶠ), ", ", last(m.λᶠ), "] degrees with Δλ=", m.Δλ, "\n",
          "├── latitude:  [", first(m.φᶠ), ", ", last(m.φᶠ), "] degrees with Δφ=", m.Δφ, "\n",
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

"""Cell area [m²] at structured index (i, j)."""
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
    R = m.radius
    return FT[R^2 * deg2rad(m.Δλ) * abs(sind(m.φᶠ[j+1]) - sind(m.φᶠ[j])) for j in 1:m.Ny]
end

# ---- Internal structured-only metric helpers (not exported) ----

"""x-spacing [m] at latitude j. Clamped to Δy near poles."""
function dx(m::LatLonMesh{FT}, j::Integer) where FT
    1 <= j <= m.Ny || throw(BoundsError(1:m.Ny, j))
    R = m.radius
    dx_val = R * cosd(m.φᶜ[j]) * deg2rad(m.Δλ)
    dy_val = R * deg2rad(m.Δφ)
    return max(dx_val, dy_val)
end

"""y-spacing [m] — uniform."""
function dy(m::LatLonMesh{FT}) where FT
    return m.radius * deg2rad(m.Δφ)
end

"""x-face length [m] (meridional edges) at latitude band j."""
_face_length_x(m::LatLonMesh{FT}, j::Integer) where FT = m.radius * deg2rad(m.Δφ)

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

export LatLonMesh, cell_areas_by_latitude
