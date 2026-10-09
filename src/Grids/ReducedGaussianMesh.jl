# ---------------------------------------------------------------------------
# ReducedGaussianMesh -- native ERA5 / IFS reduced Gaussian mesh
#
# The mesh is stored as latitude rings with variable `nlon` per ring. This is
# a true unstructured mesh in the transport sense: north/south connectivity is
# defined by overlap between adjacent rings rather than by a globally uniform
# `(i, j)` indexing convention.
#
# The geometry below uses midpoint ring boundaries on the sphere. That is good
# enough for topology, indexing, and first-order area/face metrics. Exact
# Gaussian quadrature cell metrics can be added later without changing the
# public mesh contract.
# ---------------------------------------------------------------------------

"""
    ReducedGaussianMesh{FT} <: AbstractHorizontalMesh

Reduced Gaussian mesh with variable longitude count per latitude ring.

# Fields
- `latitudes`        -- ring-center latitudes [degrees], ordered south -> north
- `nlon_per_ring`    -- number of cells on each latitude ring
- `ring_offsets`     -- 1-based start indices into the flattened cell array
- `lat_faces`        -- latitude boundaries between rings [degrees]
- `boundary_counts`  -- number of meridional face segments on each ring boundary
- `boundary_offsets` -- 1-based start offsets within the meridional-face block
- `radius`           -- planet radius [m]

# Flattening convention
Cells are flattened ring-by-ring from south to north. Within a ring, cell `i`
is ordered west to east on a periodic longitude partition `[0, 360)`.

# Geometry note
The ring boundaries are defined by midpoint latitudes with pole caps at
`-90` and `+90` degrees. This preserves total surface area exactly while the
exact Gaussian control-volume boundaries are still being wired in.
"""
struct ReducedGaussianMesh{FT <: AbstractFloat} <: AbstractHorizontalMesh
    latitudes        :: Vector{FT}
    nlon_per_ring    :: Vector{Int}
    ring_offsets     :: Vector{Int}
    lat_faces        :: Vector{FT}
    boundary_counts  :: Vector{Int}
    boundary_offsets :: Vector{Int}
    radius           :: FT
    _ncells          :: Int
    _nfaces          :: Int
end

Base.eltype(::ReducedGaussianMesh{FT}) where FT = FT
Base.eltype(::Type{<:ReducedGaussianMesh{FT}}) where FT = FT

function _validate_reduced_gaussian_rings(latitudes_in::AbstractVector{<:Real},
                                          nlon_in::AbstractVector{<:Integer},
                                          ::Type{FT}) where FT <: AbstractFloat
    length(latitudes_in) == length(nlon_in) ||
        throw(DimensionMismatch("latitudes and nlon_per_ring must have the same length"))
    isempty(latitudes_in) &&
        throw(ArgumentError("reduced Gaussian mesh needs at least one ring"))

    latitudes = FT.(latitudes_in)
    nlon_per_ring = Int.(nlon_in)
    all(nlon_per_ring .> 0) ||
        throw(ArgumentError("every ring must have at least one longitude cell"))

    if latitudes[1] > latitudes[end]
        reverse!(latitudes)
        reverse!(nlon_per_ring)
    end
    all(diff(latitudes) .> zero(FT)) ||
        throw(ArgumentError("ring latitudes must be strictly increasing south -> north"))

    return latitudes, nlon_per_ring
end

function _latitude_faces(latitudes::AbstractVector{FT}) where FT <: AbstractFloat
    nr = length(latitudes)

    lat_faces = Vector{FT}(undef, nr + 1)
    lat_faces[1] = FT(-90)
    for j in 1:nr-1
        lat_faces[j + 1] = (latitudes[j] + latitudes[j + 1]) / FT(2)
    end
    lat_faces[end] = FT(90)

    return lat_faces
end

function _ring_offsets(nlon_per_ring::AbstractVector{Int})
    nr = length(nlon_per_ring)
    ring_offsets = Vector{Int}(undef, nr + 1)
    ring_offsets[1] = 1
    for j in 1:nr
        ring_offsets[j + 1] = ring_offsets[j] + nlon_per_ring[j]
    end

    return ring_offsets
end

"""
    _boundary_counts(nlon_per_ring) -> Vector{Int}

Compute the number of meridional-face segments on each latitude boundary.

Between ring `j` (nlon_j cells) and ring `j+1` (nlon_{j+1} cells), the
shared boundary is divided into `lcm(nlon_j, nlon_{j+1})` segments so
that each segment aligns with exactly one cell on each ring. This
ensures conservative face-based flux accumulation: each cell on ring `j`
contributes `lcm/nlon_j` boundary segments, and each cell on ring `j+1`
contributes `lcm/nlon_{j+1}` segments.

Boundary 1 (south pole cap) and boundary `nr+1` (north pole cap) each
have `nlon_per_ring[1]` and `nlon_per_ring[end]` segments respectively
(one per cell on the pole-adjacent ring; the other side is the pole
singularity, represented by face_left=0 or face_right=0).

**Performance note**: for octahedral grids (nlon = 4k+16), adjacent
rings can have large LCMs (e.g. lcm(108, 112) = 3024), leading to
high face counts. This is correct but expensive for runtime solvers.
"""
function _boundary_counts(nlon_per_ring::AbstractVector{Int})
    nr = length(nlon_per_ring)
    boundary_counts = Vector{Int}(undef, nr + 1)
    boundary_counts[1] = nlon_per_ring[1]       # south pole cap: one segment per cell on ring 1
    for j in 1:nr-1
        boundary_counts[j + 1] = lcm(nlon_per_ring[j], nlon_per_ring[j + 1])
    end
    boundary_counts[end] = nlon_per_ring[end]   # north pole cap: one segment per cell on last ring

    return boundary_counts
end

function _boundary_offsets(boundary_counts::AbstractVector{Int})
    nb = length(boundary_counts)
    boundary_offsets = Vector{Int}(undef, nb + 1)
    boundary_offsets[1] = 1
    for b in 1:nb
        boundary_offsets[b + 1] = boundary_offsets[b] + boundary_counts[b]
    end

    return boundary_offsets
end

function ReducedGaussianMesh(latitudes_in::AbstractVector{<:Real},
                             nlon_in::AbstractVector{<:Integer};
                             FT::Type{<:AbstractFloat} = Float64,
                             radius = FT(EARTH_RADIUS))
    latitudes, nlon_per_ring = _validate_reduced_gaussian_rings(latitudes_in, nlon_in, FT)
    lat_faces = _latitude_faces(latitudes)
    ring_offsets = _ring_offsets(nlon_per_ring)
    boundary_counts = _boundary_counts(nlon_per_ring)
    boundary_offsets = _boundary_offsets(boundary_counts)

    ncells_total = ring_offsets[end] - 1

    nfaces_total = ncells_total + (boundary_offsets[end] - 1)

    return ReducedGaussianMesh{FT}(latitudes, nlon_per_ring, ring_offsets,
                                   lat_faces, boundary_counts, boundary_offsets,
                                   FT(radius), ncells_total, nfaces_total)
end

nrings(m::ReducedGaussianMesh) = length(m.nlon_per_ring)
nboundaries(m::ReducedGaussianMesh) = length(m.boundary_counts)
ncells(m::ReducedGaussianMesh) = m._ncells
nfaces(m::ReducedGaussianMesh) = m._nfaces

function Base.summary(m::ReducedGaussianMesh)
    FT = eltype(m)
    return string(nrings(m), "-ring ReducedGaussianMesh{", FT, "} with ",
                  ncells(m), " cells and ", nfaces(m), " faces")
end

function Base.show(io::IO, m::ReducedGaussianMesh)
    print(io, summary(m), "\n",
          "├── latitude:  [", first(m.latitudes), ", ", last(m.latitudes), "] degrees\n",
          "├── nlon/ring: min=", minimum(m.nlon_per_ring), ", max=", maximum(m.nlon_per_ring), "\n",
          "└── radius:    ", m.radius, " m")
end

@inline function _check_ring_index(m::ReducedGaussianMesh, j::Integer)
    1 <= j <= nrings(m) || throw(BoundsError(m.nlon_per_ring, j))
    return j
end

@inline function _check_boundary_index(m::ReducedGaussianMesh, b::Integer)
    1 <= b <= nboundaries(m) || throw(BoundsError(m.boundary_counts, b))
    return b
end

@inline function _check_cell_index(m::ReducedGaussianMesh, c::Integer)
    1 <= c <= ncells(m) || throw(BoundsError(1:ncells(m), c))
    return c
end

@inline function _check_face_index(m::ReducedGaussianMesh, f::Integer)
    1 <= f <= nfaces(m) || throw(BoundsError(1:nfaces(m), f))
    return f
end

@inline function ring_cell_count(m::ReducedGaussianMesh, j::Integer)
    return m.nlon_per_ring[_check_ring_index(m, j)]
end

"""
    ring_longitudes(m, j) -> Vector{FT}

Return the cell-center longitudes for ring `j`, in [0°, 360°) convention
with half-cell offset: `lon[i] = (i − 0.5) × (360° / nlon)`.

For `nlon = 96`: `[1.875, 5.625, ..., 358.125]`.

This convention matches the spectral synthesis `spectral_to_ring!` which
uses `lon_shift_rad = π / nlon` to place FFT output at cell centers.
"""
function ring_longitudes(m::ReducedGaussianMesh{FT}, j::Integer) where FT
    nlon = ring_cell_count(m, j)
    dlon = FT(360 / nlon)
    return FT[(i - FT(0.5)) * dlon for i in 1:nlon]  # cell centers in [0, 360)
end

@inline function boundary_face_count(m::ReducedGaussianMesh, b::Integer)
    return m.boundary_counts[_check_boundary_index(m, b)]
end

@inline function boundary_face_offset(m::ReducedGaussianMesh, b::Integer)
    return m._ncells + m.boundary_offsets[_check_boundary_index(m, b)]
end

@inline function boundary_face_range(m::ReducedGaussianMesh, b::Integer)
    offset = boundary_face_offset(m, b)
    count = boundary_face_count(m, b)
    return offset:(offset + count - 1)
end

@inline function _block_index(offsets::AbstractVector{Int}, idx::Int)
    return searchsortedlast(offsets, idx)
end

@inline function _cell_ij(m::ReducedGaussianMesh, c::Integer)
    _check_cell_index(m, c)
    j = _block_index(m.ring_offsets, c)
    i = c - m.ring_offsets[j] + 1
    return i, j
end

@inline function cell_index(m::ReducedGaussianMesh, i::Integer, j::Integer)
    _check_ring_index(m, j)
    1 <= i <= m.nlon_per_ring[j] || throw(BoundsError(1:m.nlon_per_ring[j], i))
    return m.ring_offsets[j] + i - 1
end

@inline function _segment_range(nseg::Int, ncell::Int, i::Int)
    per_cell = nseg ÷ ncell
    first_seg = (i - 1) * per_cell + 1
    last_seg = i * per_cell
    return first_seg:last_seg
end

function cell_area(m::ReducedGaussianMesh{FT}, c::Integer) where FT
    _, j = _cell_ij(m, c)
    dlon = FT(360 / m.nlon_per_ring[j])
    return m.radius^2 * deg2rad(dlon) *
           abs(sind(m.lat_faces[j + 1]) - sind(m.lat_faces[j]))
end

cell_area(m::ReducedGaussianMesh, c::Tuple{<:Integer, <:Integer}) =
    cell_area(m, cell_index(m, c[1], c[2]))

function face_length(m::ReducedGaussianMesh{FT}, f::Integer) where FT
    _check_face_index(m, f)
    if f <= m._ncells
        j = _block_index(m.ring_offsets, f)
        return m.radius * deg2rad(m.lat_faces[j + 1] - m.lat_faces[j])
    else
        yf = f - m._ncells
        b = _block_index(m.boundary_offsets, yf)
        dlon = FT(360 / m.boundary_counts[b])
        return m.radius * cosd(m.lat_faces[b]) * deg2rad(dlon)
    end
end

function face_normal(m::ReducedGaussianMesh{FT}, f::Integer) where FT
    _check_face_index(m, f)
    return f <= m._ncells ? (one(FT), zero(FT)) : (zero(FT), one(FT))
end

"""
    face_cells(m, f) -> (left, right)

Return the two cell indices adjacent to face `f`. For the face-indexed
transport operator, flux through face `f` goes from `left` to `right`
(positive flux = transfer from left to right).

## Face types

Faces `1..ncells` are **zonal (X) faces** within each ring:
- Face `f` sits between the western cell `left_i = i-1` (periodic wrap)
  and the eastern cell `right_i = i` in the same ring `j`.
- These faces are always interior (both indices > 0).

Faces `ncells+1..nfaces` are **meridional (Y) boundary faces** between
adjacent latitude rings (or at the poles):

- **South pole cap** (boundary 1): `left = 0` (pole singularity),
  `right = cell on ring 1`. A face_left=0 tells the transport kernel
  to treat this as a boundary — no flux accumulation on the left side.
- **North pole cap** (boundary `nrings+1`): `left = cell on last ring`,
  `right = 0` (pole singularity).
- **Interior boundaries** (boundary `b`, between ring `b-1` and ring `b`):
  the boundary has `lcm(nlon[b-1], nlon[b])` segments. Segment `seg`
  maps to cell `south_i` in ring `b-1` and cell `north_i` in ring `b`
  via integer division: `cell_i = ((seg-1) × nlon_ring) ÷ nseg + 1`.
  This distributes boundary segments evenly among cells.
"""
function face_cells(m::ReducedGaussianMesh, f::Integer)
    _check_face_index(m, f)
    nr = nrings(m)

    # Zonal (X) faces: f ∈ 1..ncells — between left and right cells in same ring
    if f <= m._ncells
        j = _block_index(m.ring_offsets, f)
        i = f - m.ring_offsets[j] + 1
        nlon = m.nlon_per_ring[j]
        left_i = i == 1 ? nlon : i - 1   # periodic wrap within ring
        return cell_index(m, left_i, j), cell_index(m, i, j)
    end

    # Meridional (Y) boundary faces: between adjacent rings or at poles
    yf = f - m._ncells
    b = _block_index(m.boundary_offsets, yf)
    seg = yf - m.boundary_offsets[b] + 1

    if b == 1
        # South pole cap: left = 0 (pole singularity), right = cell on ring 1
        return 0, cell_index(m, seg, 1)
    elseif b == nr + 1
        # North pole cap: left = cell on last ring, right = 0 (pole singularity)
        return cell_index(m, seg, nr), 0
    else
        # Interior boundary between ring (b-1) and ring b:
        # distribute `nseg = lcm(nlon[b-1], nlon[b])` segments across cells
        south_ring = b - 1
        north_ring = b
        nseg = m.boundary_counts[b]
        # Integer division maps segment index to the owning cell on each ring
        south_i = ((seg - 1) * m.nlon_per_ring[south_ring]) ÷ nseg + 1
        north_i = ((seg - 1) * m.nlon_per_ring[north_ring]) ÷ nseg + 1
        return cell_index(m, south_i, south_ring), cell_index(m, north_i, north_ring)
    end
end

function cell_faces(m::ReducedGaussianMesh, c::Integer)
    i, j = _cell_ij(m, c)
    faces = Int[]

    west = m.ring_offsets[j] + i - 1
    east = m.ring_offsets[j] + (i == m.nlon_per_ring[j] ? 0 : i)
    push!(faces, west, east)

    south_band = j
    for seg in _segment_range(m.boundary_counts[south_band], m.nlon_per_ring[j], i)
        push!(faces, m._ncells + m.boundary_offsets[south_band] + seg - 1)
    end

    north_band = j + 1
    for seg in _segment_range(m.boundary_counts[north_band], m.nlon_per_ring[j], i)
        push!(faces, m._ncells + m.boundary_offsets[north_band] + seg - 1)
    end

    return faces
end

export ReducedGaussianMesh, nrings, nboundaries
export ring_cell_count, ring_longitudes, cell_index
export boundary_face_count, boundary_face_offset, boundary_face_range
