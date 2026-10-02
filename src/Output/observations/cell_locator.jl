# ---------------------------------------------------------------------------
# Containing-cell lookup for observation points, one locator per topology.
#
# Observation sampling reads the model cell that contains a point (cell
# values are cell means and satellite footprints are far smaller than cells),
# so the locator only has to answer "which cell" and "where in device
# storage". Lat-lon and reduced-Gaussian lookups are closed-form on the face
# arrays; the cubed sphere uses the analytic inverse projection.
# ---------------------------------------------------------------------------

"""
    CellLocation

Containing model cell for an observation point. `cell` is the topology's
native flat cell index (lat-lon `i + (j-1)Nx`, reduced Gaussian `cell_index`,
cubed sphere `i + (j-1)Nc` within `panel`). `column` is the linear column
index into the device slab holding this topology's state, including the
cubed-sphere halo offset, so a gather can address `storage[column + (k-1) *
ncolumns]`. Indices are plain `Int` so they can be passed straight to Grids
helpers. Device index buffers may narrow `column` to `Int32` (every supported
resolution fits) but must widen again before multiplying by `ncolumns`.
`lon`/`lat` are the cell-centre coordinates in the mesh's own longitude
convention (lat-lon follows the mesh interval; reduced Gaussian and cubed
sphere use [0, 360)).
"""
struct CellLocation
    panel::Int
    i::Int
    j::Int
    cell::Int
    column::Int
    lon::Float64
    lat::Float64
    area::Float64
end

"""
    AbstractCellLocator

Per-topology containing-cell lookup. The interface is `locate(loc, lon, lat)`
and `ncolumns(loc)`; locators are immutable and `locate` is pure, so they may
be shared across threads.
"""
abstract type AbstractCellLocator end

struct LatLonCellLocator{FT} <: AbstractCellLocator
    mesh::LatLonMesh{FT}
end

struct ReducedGaussianCellLocator{FT} <: AbstractCellLocator
    mesh::ReducedGaussianMesh{FT}
end

struct CubedSphereCellLocator{M <: CubedSphereMesh} <: AbstractCellLocator
    mesh::M
    halo_width::Int
    center_lons::NTuple{6, Matrix{Float64}}
    center_lats::NTuple{6, Matrix{Float64}}
end

function _require_no_halo(halo_width::Integer, topology::AbstractString)
    halo_width == 0 || throw(ArgumentError(
        "$(topology) state is unpadded; halo_width must be 0, got $(halo_width)"))
    return nothing
end

"""
    cell_locator(mesh; halo_width) -> AbstractCellLocator

Build the containing-cell locator for `mesh`. `halo_width` is the padding of
the state arrays that `CellLocation.column` addresses: lat-lon and
reduced-Gaussian state is unpadded; cubed-sphere state defaults to the mesh
halo `Hp`.
"""
function cell_locator(mesh::LatLonMesh; halo_width::Integer = 0)
    _require_no_halo(halo_width, "lat-lon")
    return LatLonCellLocator(mesh)
end

function cell_locator(mesh::ReducedGaussianMesh; halo_width::Integer = 0)
    _require_no_halo(halo_width, "reduced-Gaussian")
    return ReducedGaussianCellLocator(mesh)
end

function cell_locator(mesh::CubedSphereMesh; halo_width::Integer = mesh.Hp)
    halo_width >= 0 || throw(ArgumentError("halo_width must be non-negative; got $(halo_width)"))
    # Cached once per run (C720: ~50 MB); `convert` is a no-op for Float64 meshes.
    centers = ntuple(p -> panel_cell_center_lonlat(mesh, p), 6)
    return CubedSphereCellLocator(mesh, Int(halo_width),
                                  ntuple(p -> convert(Matrix{Float64}, centers[p][1]), 6),
                                  ntuple(p -> convert(Matrix{Float64}, centers[p][2]), 6))
end

"Number of storage columns per device slab (per panel on the cubed sphere)."
ncolumns(loc::LatLonCellLocator) = ncells(loc.mesh)
ncolumns(loc::ReducedGaussianCellLocator) = ncells(loc.mesh)
ncolumns(loc::CubedSphereCellLocator) = (loc.mesh.Nc + 2 * loc.halo_width)^2

"""
    isvalid_lonlat(lon, lat) -> Bool

Finite coordinates with `|lat| ≤ 90` and `|lon| ≤ 720`: fill values such as
-999999 are rejected while every ordinary longitude convention passes. Readers
filter with this predicate; `locate` enforces it.
"""
isvalid_lonlat(lon::Real, lat::Real) =
    isfinite(lon) && isfinite(lat) && abs(lat) <= 90 && abs(lon) <= 720

function _check_point(lon::Real, lat::Real)
    isvalid_lonlat(lon, lat) || throw(ArgumentError(
        "observation location must be finite with |lat| <= 90 and |lon| <= 720; " *
        "got lon=$(lon), lat=$(lat)"))
    return nothing
end

"""
    locate(loc::AbstractCellLocator, lon, lat) -> Union{CellLocation, Nothing}

Containing cell of a point given in degrees, in any longitude convention.
Returns `nothing` only when the mesh does not cover the point (regional
lat-lon meshes); global meshes always return a cell. Faces follow the
half-open convention: a point exactly on a face belongs to the cell to its
east or north, except on the mesh's outermost northern and eastern faces.
Invalid coordinates (see [`isvalid_lonlat`](@ref)) throw an `ArgumentError`.
"""
function locate end

function locate(loc::LatLonCellLocator, lon::Real, lat::Real)
    _check_point(lon, lat)
    mesh = loc.mesh
    φ = Float64(lat)
    west = Float64(mesh.λᶠ[1])
    # Wrap into [west, west + 360): exact for Float32 faces too, because the
    # face span of a global mesh is exactly 360 in either precision. Regional
    # meshes then only need the eastern bound.
    λ = mod(Float64(lon) - west, 360.0) + west
    λ <= Float64(mesh.λᶠ[end]) || return nothing
    Float64(mesh.φᶠ[1]) <= φ <= Float64(mesh.φᶠ[end]) || return nothing
    i = clamp(searchsortedlast(mesh.λᶠ, λ), 1, mesh.Nx)
    j = clamp(searchsortedlast(mesh.φᶠ, φ), 1, mesh.Ny)
    cell = i + (j - 1) * mesh.Nx
    return CellLocation(1, i, j, cell, cell, Float64(mesh.λᶜ[i]), Float64(mesh.φᶜ[j]),
                        Float64(cell_area(mesh, i, j)))
end

function locate(loc::ReducedGaussianCellLocator, lon::Real, lat::Real)
    _check_point(lon, lat)
    mesh = loc.mesh
    j = clamp(searchsortedlast(mesh.lat_faces, Float64(lat)), 1, nrings(mesh))
    nlon = mesh.nlon_per_ring[j]
    dlon = 360.0 / nlon
    i = clamp(floor(Int, mod(Float64(lon), 360.0) / dlon) + 1, 1, nlon)
    cell = cell_index(mesh, i, j)
    return CellLocation(1, i, j, cell, cell, (i - 0.5) * dlon, Float64(mesh.latitudes[j]),
                        Float64(cell_area(mesh, cell)))
end

function locate(loc::CubedSphereCellLocator, lon::Real, lat::Real)
    _check_point(lon, lat)
    mesh = loc.mesh
    # Float64 inverse regardless of the mesh precision; floor lands in 1..Nc.
    panel, s, t = lonlat_to_panel_xy(cs_definition(mesh), mesh.Nc, Float64(lon), Float64(lat),
                                     Float64)
    i = floor(Int, s)
    j = floor(Int, t)
    Hp = loc.halo_width
    column = (Hp + i) + (Hp + j - 1) * (mesh.Nc + 2 * Hp)
    return CellLocation(panel, i, j, i + (j - 1) * mesh.Nc, column,
                        loc.center_lons[panel][i, j], loc.center_lats[panel][i, j],
                        Float64(cell_area(mesh, i, j)))
end
