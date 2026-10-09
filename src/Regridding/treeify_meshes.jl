# ---------------------------------------------------------------------------
# Trees.treeify methods for mesh types
#
# Each mesh type is converted into a ConservativeRegridding spatial tree that
# implements the SpatialTreeInterface. The tree is consumed by CR.jl's dual
# DFS during Regridder construction.
# ---------------------------------------------------------------------------

# --- best_manifold overloads --------------------------------------------------
#
# GeometryOpsCore.best_manifold tells CR.jl which manifold (Planar or
# Spherical) a grid lives on, and at what radius. All AtmosTransport
# meshes are spherical with radius = mesh.radius (default 6.371e6 m). The
# manifold is Float64 for every mesh precision: regridding geometry is
# evaluated once, Float32 and Float64 runs must share weights, and CR.jl
# requires source and destination manifolds of the same type.

"""Return `Spherical(radius = Float64(mesh.radius))` for an AtmosTransport mesh."""
GOCore.best_manifold(mesh::Union{LatLonMesh, CubedSphereMesh, ReducedGaussianMesh}) =
    GO.Spherical(; radius = Float64(mesh.radius))

# ---------------------------------------------------------------------------
# LatLonMesh → CellBasedGrid(UnitSphericalPoint) → TopDownQuadtreeCursor
#
# Uses the cell-face vectors λᶠ, φᶠ to build an (Nx+1) × (Ny+1) matrix of
# corner points on the unit sphere. If the mesh covers the full sphere we
# skip extent computation via KnownFullSphereExtentWrapper.
# ---------------------------------------------------------------------------

"""
    _latlon_full_sphere(mesh::LatLonMesh) -> Bool

Return `true` when the mesh's face vectors span the full sphere
(longitude covers 360°, latitude from −90 to +90).
"""
function _latlon_full_sphere(mesh::LatLonMesh)
    Δλ = last(mesh.λᶠ) - first(mesh.λᶠ)
    return isapprox(Δλ, 360; atol = 1e-6) &&
           isapprox(first(mesh.φᶠ), -90; atol = 1e-6) &&
           isapprox(last(mesh.φᶠ),   90; atol = 1e-6)
end

"""
    _latlon_corner_matrix(mesh::LatLonMesh) -> Matrix{UnitSphericalPoint}

Build the `(Nx+1) × (Ny+1)` matrix of cell corners, mapped to unit-sphere
Cartesian coordinates. This matches the Oceananigans pattern in
`ConservativeRegriddingOceananigansExt.jl`.
"""
function _latlon_corner_matrix(mesh::LatLonMesh)
    Nx1 = length(mesh.λᶠ)
    Ny1 = length(mesh.φᶠ)
    pts = Matrix{UnitSphericalPoint{Float64}}(undef, Nx1, Ny1)
    # GO.UnitSphereFromGeographic() expects (lon, lat) degrees
    to_sphere = GO.UnitSphereFromGeographic()
    @inbounds for j in 1:Ny1, i in 1:Nx1
        pts[i, j] = to_sphere((Float64(mesh.λᶠ[i]), Float64(mesh.φᶠ[j])))
    end
    return pts
end

"""
    Trees.treeify(::Spherical, mesh::LatLonMesh)

Build a spatial tree for a `LatLonMesh` on the sphere.

## Algorithm

1. Build an `(Nx+1) × (Ny+1)` matrix of cell-face corner points as
   `UnitSphericalPoint`s, from the mesh's face vectors `λᶠ` (longitude)
   and `φᶠ` (latitude). Each corner `(i, j)` is at geographic coordinate
   `(λᶠ[i], φᶠ[j])`, converted to Cartesian `(x, y, z)` on the unit
   sphere via `GO.UnitSphereFromGeographic()`.

2. Wrap in `CellBasedGrid{Spherical}`, which builds cell polygons on the
   fly as quadrilaterals from 4 adjacent corner points. Cell `(i, j)` has
   vertices SW=`(i,j)`, SE=`(i+1,j)`, NE=`(i+1,j+1)`, NW=`(i,j+1)`.

3. Wrap in `TopDownQuadtreeCursor` for O(log(Nx·Ny)) spatial pruning.

4. If the mesh covers the full sphere (360° longitude, ±90° latitude),
   wrap in `KnownFullSphereExtentWrapper` to skip expensive extent
   computation at the root.

## Longitude convention

The mesh's `λᶠ` vector defines cell-face longitudes. The default
`LatLonMesh()` uses `[-180, 180]`. The UnitSphericalPoint representation
is the same physical location regardless of whether the source convention
is `[-180, 180]` or `[0, 360]` — the Cartesian `(x, y, z)` coordinates
are unambiguous.

## Cell indexing

CR.jl's `CellBasedGrid` uses column-major linear indexing:
`linear_idx = i + (j-1) * Nx`, where `i` is the longitude index (1:Nx)
and `j` is the latitude index (1:Ny). This matches Julia's default
`vec(reshape(field, Nx, Ny))` memory layout.
"""
function Trees.treeify(manifold::GOCore.Spherical, mesh::LatLonMesh)
    corners = _latlon_corner_matrix(mesh)
    grid    = Trees.CellBasedGrid(manifold, corners)
    tree    = Trees.TopDownQuadtreeCursor(grid)
    return _latlon_full_sphere(mesh) ? Trees.KnownFullSphereExtentWrapper(tree) : tree
end

Trees.treeify(mesh::LatLonMesh) = Trees.treeify(GOCore.best_manifold(mesh), mesh)

# ---------------------------------------------------------------------------
# CubedSphereMesh → CubedSphereToplevelTree of 6 CellBasedGrids
#
# CubedSphereMesh does not store face-corner coordinates; we regenerate
# them via the convention-aware coordinate helpers in Grids. Panels whose
# native file axes are left-handed are flipped internally for polygon winding
# while retaining file-order linear indices. This keeps regridding aligned
# with diagnostic NetCDF output for both gnomonic and GEOS-native panel
# conventions.
# ---------------------------------------------------------------------------

"""
    cubed_sphere_face_corners(mesh::CubedSphereMesh)
        -> NTuple{6, Matrix{UnitSphericalPoint{Float64}}}

Return a 6-tuple of `(Nc+1) × (Nc+1)` corner-point matrices, one per panel,
with corners on the unit sphere. Panel ordering follows `mesh.convention`;
for each user-visible panel `p`, corner `(i, j)` matches
`panel_cell_corner_lonlat(mesh, p, Float64)`. Corners are evaluated in
Float64 for every mesh precision, so a Float32 run regrids with the same
weights as a Float64 run (the cache key does not include the precision).

Used by `Trees.treeify(::Spherical, ::CubedSphereMesh)` to assemble a
`CubedSphereToplevelTree`. Cached externally (the regridder cache includes
the mesh hash) so this is only called once per `(Nc, convention)`.
"""
function cubed_sphere_face_corners(mesh::CubedSphereMesh)
    Nc  = mesh.Nc
    Np  = Nc + 1
    to_sphere = GO.UnitSphereFromGeographic()
    panels = ntuple(6) do user_panel
        lons, lats = panel_cell_corner_lonlat(mesh, user_panel, Float64)
        m = Matrix{UnitSphericalPoint{Float64}}(undef, Np, Np)
        @inbounds for j in 1:Np, i in 1:Np
            m[i, j] = to_sphere((lons[i, j], lats[i, j]))
        end
        m
    end
    return panels
end

"""
    YReversedCellBasedGrid(manifold, points)

`CellBasedGrid` variant for cubed-sphere panels whose file-space `(i, j)`
indices are left-handed. The stored `points` matrix is flipped in the tree's
Y direction so each cell polygon has the winding expected by
ConservativeRegridding, while `cartesian_to_linear_idx` and
`linear_to_cartesian_idx` preserve the original file-order linear index
`i + (j - 1) * Nc`.

This keeps GEOS-native panels 4/5 conservative without materializing one
polygon per cell or reordering tracer arrays.
"""
struct YReversedCellBasedGrid{M <: GOCore.Manifold, PointMatrixType <: AbstractMatrix} <: Trees.AbstractCurvilinearGrid
    manifold::M
    points::PointMatrixType
end

GOCore.manifold(grid::YReversedCellBasedGrid) = grid.manifold
Trees.ncells(grid::YReversedCellBasedGrid, dim::Int) = size(grid.points, dim) - 1

Base.@propagate_inbounds function Trees.getcell(grid::YReversedCellBasedGrid, i::Int, j::Int)
    @boundscheck begin
        if i < 1 || i >= size(grid.points, 1) || j < 1 || j >= size(grid.points, 2)
            error("Invalid index for YReversedCellBasedGrid; got ($i, $j), but the matrix has $(size(grid.points) .- 1) polygons.")
        end
    end
    return GI.Polygon(SA[GI.LinearRing(SA[
        grid.points[i, j],
        grid.points[i + 1, j],
        grid.points[i + 1, j + 1],
        grid.points[i, j + 1],
        grid.points[i, j],
    ])])
end

function Trees.cartesian_to_linear_idx(grid::YReversedCellBasedGrid, idx::CartesianIndex{2})
    nx = Trees.ncells(grid, 1)
    ny = Trees.ncells(grid, 2)
    i, j_tree = idx.I
    j_data = ny - j_tree + 1
    return i + (j_data - 1) * nx
end

function Trees.linear_to_cartesian_idx(grid::YReversedCellBasedGrid, idx::Integer)
    nx = Trees.ncells(grid, 1)
    ny = Trees.ncells(grid, 2)
    i = mod(Int(idx) - 1, nx) + 1
    j_data = div(Int(idx) - 1, nx) + 1
    return CartesianIndex(i, ny - j_data + 1)
end

function Trees.cell_range_extent(grid::YReversedCellBasedGrid{<:GO.Spherical},
                                 irange::UnitRange{Int}, jrange::UnitRange{Int})
    return Trees.cell_range_extent(Trees.CellBasedGrid(grid.manifold, grid.points),
                                   irange, jrange)
end

function _corner_winding_sign(corners::AbstractMatrix, i::Int=1, j::Int=1)
    verts = (corners[i, j],
             corners[i + 1, j],
             corners[i + 1, j + 1],
             corners[i, j + 1])
    normal = zeros(Float64, 3)
    centroid = zeros(Float64, 3)
    for n in 1:4
        a = verts[n]
        b = verts[mod1(n + 1, 4)]
        av = SA[Float64(a[1]), Float64(a[2]), Float64(a[3])]
        bv = SA[Float64(b[1]), Float64(b[2]), Float64(b[3])]
        normal .+= cross(av, bv)
        centroid .+= av
    end
    return dot(normal, centroid)
end

function _cs_panel_grid(manifold::GOCore.Spherical, corners::AbstractMatrix)
    if _corner_winding_sign(corners) > 0
        return Trees.CellBasedGrid(manifold, corners)
    else
        return YReversedCellBasedGrid(manifold, reverse(corners; dims=2))
    end
end

"""
    Trees.treeify(::Spherical, mesh::CubedSphereMesh)

Build a spatial tree for a `CubedSphereMesh` on the sphere.

## Algorithm

1. Call [`cubed_sphere_face_corners`](@ref) to produce 6 matrices of
   `(Nc+1) × (Nc+1)` `UnitSphericalPoint` corner coordinates, one per
   panel. The panel ordering follows `mesh.convention` (gnomonic or
   GEOS-native).

2. For each panel `p`, build a quadtree grid from the corner matrix, correcting
   left-handed panel winding where needed, and wrap it in an
   `IndexOffsetQuadtreeCursor` with offset `(p-1) × Nc²`. This ensures the
   global cell index is:

       global_idx = (panel - 1) × Nc² + local_linear_idx

   where `local_linear_idx = i + (j-1) × Nc` (column-major within
   the panel, `i` = local ξ-index, `j` = local η-index).

3. Assemble all 6 per-panel cursors into a `CubedSphereToplevelTree`,
   whose top-level extent is the full sphere.

## Cell indexing

The global flat index space is `1 : 6·Nc²`. Panel `p` occupies indices
`(p-1)·Nc² + 1 : p·Nc²`. Within a panel, cells are column-major in
`(i, j)` gnomonic indices: `i` varies fastest (ξ direction), `j` second
(η direction). This layout matches `vec(panel_data[:, :])` in Julia.

## GEOS-native support

`GEOSNativePanelConvention` uses the GEOS-FP/GEOS-IT panel order and
orientation exposed by NCDatasets as `(Xdim, Ydim, nf, ...)`, including the
global `-10°` longitude offset and rotated north-pole/equatorial panels.
Panels 4 and 5 are left-handed in file index space; treeify flips their
corner matrices only inside the quadtree grid and maps tree indices back to
the original GEOS file-order linear indices.
"""
function Trees.treeify(manifold::GOCore.Spherical, mesh::CubedSphereMesh)
    corners = cubed_sphere_face_corners(mesh)
    Nc = mesh.Nc
    N_per_panel = Nc * Nc
    quadtrees = [
        Trees.IndexOffsetQuadtreeCursor(
            _cs_panel_grid(manifold, corners[p]),
            (p - 1) * N_per_panel,
        )
        for p in 1:6
    ]
    return Trees.CubedSphereToplevelTree(quadtrees)
end

Trees.treeify(mesh::CubedSphereMesh) = Trees.treeify(GOCore.best_manifold(mesh), mesh)

# ---------------------------------------------------------------------------
# ReducedGaussianMesh → per-ring CellBasedGrid trees → MultiTreeWrapper
#
# A reduced Gaussian mesh has variable `nlon` per latitude ring, so it
# cannot be directly represented as a single rectangular
# AbstractCurvilinearGrid. Instead, we treat each ring as an independent
# `CellBasedGrid{Spherical}` with shape `(nlon+1, 2)` — the +1 in the
# longitude dimension gives face edges, and the 2 latitudes are the
# ring's south and north boundaries.
#
# Each ring tree is an `IndexOffsetQuadtreeCursor` that maps the ring's
# local cell indices (1:nlon) to the mesh's global flat index:
#   global_idx = ring_offsets[j] + local_idx - 1
# via offset = ring_offsets[j] - 1.
#
# The per-ring trees are combined into a `MultiTreeWrapper` (from
# CR.jl's wrappers.jl), which uses cumulative cell offsets for O(1)
# ring lookup given a global index. The whole thing is wrapped in
# `KnownFullSphereExtentWrapper` since a reduced Gaussian mesh
# (ERA5 / IFS native) always covers the full sphere.
#
# This approach:
#   - Reuses CellBasedGrid{Spherical}'s `cell_range_extent`, which
#     correctly produces SphericalCap extents (avoids the
#     SphericalCap vs Extent{(:X,:Y,:Z)} type mismatch that crashes
#     FlatNoTree on spherical manifolds)
#   - Gives O(log nlon) spatial pruning within each ring via
#     IndexOffsetQuadtreeCursor's recursive subdivision
#   - Gives O(nrings) lookup at the top level, acceptable for ERA5
#     (nrings ~ 320)
#   - Uses only existing CR.jl infrastructure: CellBasedGrid,
#     IndexOffsetQuadtreeCursor, MultiTreeWrapper,
#     KnownFullSphereExtentWrapper
#
# Cell index convention:
#   Cells are flattened ring-by-ring south→north, west→east within
#   each ring, starting at longitude 0° with uniform spacing
#   Δlon = 360° / nlon. This matches ReducedGaussianMesh's own
#   `cell_index(mesh, i, j) = ring_offsets[j] + i - 1`.
#
# Corner-point construction:
#   For ring j with nlon cells:
#     - Longitude face edges: lon_k = (k-1) * 360/nlon for k = 1:nlon+1
#       (first face at 0°, last at 360° — numerically equal as
#       UnitSphericalPoint to ~1e-15)
#     - Latitude face edges: lat_faces[j] (south), lat_faces[j+1] (north)
#     - Corner matrix pts[k, ℓ] where k = 1:nlon+1 (lon faces),
#       ℓ = 1 (south lat) or 2 (north lat)
#     - All corners are UnitSphericalPoint via GO.UnitSphereFromGeographic()
#   CellBasedGrid.getcell(grid, i, 1) builds the polygon:
#     pts[i,1] (SW) → pts[i+1,1] (SE) → pts[i+1,2] (NE) → pts[i,2] (NW)
#   Winding: counter-clockwise when viewed from outside the sphere. ✓
# ---------------------------------------------------------------------------

"""
    Trees.treeify(::Spherical, mesh::ReducedGaussianMesh) -> KnownFullSphereExtentWrapper{MultiTreeWrapper}

Build a spatial tree for a `ReducedGaussianMesh` by dividing each latitude
ring into longitude sectors and combining the sector trees in a
`MultiTreeWrapper`.

Each sector becomes a corner-point grid wrapped in an
`IndexOffsetQuadtreeCursor`, so its global cell indices match the mesh's flat
cell indexing:

    global_cell_index = ring_offsets[j] + in_ring_index - 1

No sector spans a hemisphere. This is important because a full 360-degree
equatorial ring has a degenerate root `SphericalCap`, which can incorrectly
prune every cell in the ring during a dual-tree intersection search.

## Longitude convention

Within each ring, the `nlon` cells span `[0°, 360°)` uniformly:
- Cell `i` covers longitude `[(i-1) × 360/nlon, i × 360/nlon]`
- Cell centers are at `(i - 0.5) × 360/nlon` (matching
  `ReducedGaussianMesh.ring_longitudes`)

## Latitude convention

Ring boundaries come from `mesh.lat_faces` (south to north, with
`lat_faces[1] = -90°` and `lat_faces[end] = +90°`).
"""
function Trees.treeify(manifold::GOCore.Spherical, mesh::ReducedGaussianMesh)
    to_sphere = GO.UnitSphereFromGeographic()
    nr = nrings(mesh)
    minimum(mesh.nlon_per_ring) >= 3 ||
        throw(ArgumentError("regridding requires at least three cells per reduced-Gaussian ring"))

    sector_specs = Tuple{Int, Int, Int}[]
    for j in 1:nr
        nlon = mesh.nlon_per_ring[j]
        nsectors = min(4, nlon)
        sector_edges = [fld(k * nlon, nsectors) for k in 0:nsectors]
        for sector in 1:nsectors
            push!(sector_specs,
                  (j, sector_edges[sector] + 1, sector_edges[sector + 1]))
        end
    end

    sector_trees = map(sector_specs) do (j, first_cell, last_cell)
        nlon = mesh.nlon_per_ring[j]
        dlon = 360.0 / nlon
        φ_s = Float64(mesh.lat_faces[j])
        φ_n = Float64(mesh.lat_faces[j + 1])

        # Epsilon-clamp polar latitudes to avoid degenerate polygons.
        #
        # At exactly ±90°, all longitude face edges map to the same
        # UnitSphericalPoint (the pole). This makes the polygon a
        # degenerate quadrilateral (two vertices coincide), and
        # GO.area(Spherical(), polygon) can underestimate the area by
        # up to ~6% for large polar cells. Clamping by ε = 0.001°
        # (~111 m at the pole) creates a tiny but non-degenerate
        # quadrilateral. The omitted polar cap area is
        # π × (R × ε_rad)² ≈ 4e4 m² — negligible relative to the
        # full sphere (5.1e14 m²), ratio ~8e-11.
        φ_s = max(φ_s, -90.0 + 0.001)
        φ_n = min(φ_n,  90.0 - 0.001)

        nsector = last_cell - first_cell + 1
        points = Matrix{UnitSphericalPoint{Float64}}(undef, nsector + 1, 2)
        @inbounds for local_face in 1:(nsector + 1)
            lon = (first_cell + local_face - 2) * dlon
            points[local_face, 1] = to_sphere((lon, φ_s))
            points[local_face, 2] = to_sphere((lon, φ_n))
        end

        grid = Trees.CellBasedGrid(manifold, points)
        index_offset = mesh.ring_offsets[j] + first_cell - 2
        Trees.IndexOffsetQuadtreeCursor(grid, index_offset)
    end
    offsets = [mesh.ring_offsets[j] + last_cell - 1
               for (j, _, last_cell) in sector_specs]

    tree = Trees.MultiTreeWrapper(sector_trees, offsets)
    return Trees.KnownFullSphereExtentWrapper(tree)
end

Trees.treeify(mesh::ReducedGaussianMesh) = Trees.treeify(GOCore.best_manifold(mesh), mesh)
