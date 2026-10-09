"""
    Regridding (src)

Offline conservative regridding between mesh types, built on
[ConservativeRegridding.jl](https://github.com/JuliaGeo/ConservativeRegridding.jl).

Build a sparse weights matrix once per `(source_mesh, target_mesh)` pair,
optionally cache it to disk, and reuse it. Callers are `Preprocessing` (met
data to target grids), `Models.InitialConditionIO` (initial conditions and
surface-flux inventories at run setup) and `Visualization` (cubed-sphere to
lon-lat rasters). The operators and the stepping loop never call into this
module — `TransportBinaryReader` consumes binaries that are already on the
target grid.

## Workflow

```julia
using AtmosTransport: LatLonMesh, CubedSphereMesh
using AtmosTransport.Regridding

src = LatLonMesh(Nx=1440, Ny=721)
dst = CubedSphereMesh(Nc=90)

# Build weights (expensive — minutes for C90, cached to disk)
r = build_regridder(src, dst; cache_dir="/tmp/atmos_regrid_cache")

# Apply to a 2D field
src_field = zeros(src.Nx * src.Ny)
dst_field = zeros(6 * dst.Nc * dst.Nc)
apply_regridder!(dst_field, r, src_field)

# Persist / export
save_regridder("weights.jld2", r)
save_esmf_weights("weights_esmf.nc", r)
```

## Supported mesh types

| Source/Destination | Tree strategy | Spatial acceleration |
|--------------------|---------------|----------------------|
| `LatLonMesh`       | `CellBasedGrid` from `(Nx+1)×(Ny+1)` face corners → `TopDownQuadtreeCursor` | O(log(Nx·Ny)) |
| `CubedSphereMesh`  | 6 per-panel convention-aware grids from CS corners → `CubedSphereToplevelTree` | O(log Nc²) per panel |
| `ReducedGaussianMesh` | Sectorized per-ring `CellBasedGrid` trees → `MultiTreeWrapper` | O(nrings · log nlon) |

All three produce `SphericalCap` extents at every tree level, which is
required by CR.jl's spherical dual-DFS intersection search.

## Known limitations

- `CubedSphereMesh` uses analytical coordinates from
  `src/Grids/cs_mesh_coordinates.jl`. `GEOSNativePanelConvention` includes the
  GEOS-FP/GEOS-IT panel order, native orientation, and global `-10°`
  longitude offset used by GEOS grid files. Left-handed GEOS panels are
  wound correctly for tree traversal while preserving file-order indices.
- `ReducedGaussianMesh` clamps polar face latitudes by 0.001° to avoid
  degenerate polygons at the poles. Also, its cell polygons use great-circle
  edges to approximate curved latitude boundaries. This approximation is
  visible on unusually coarse synthetic rings and converges as ring
  resolution increases.

## Architecture reference

See `docs/src/preprocessing/regridding.md` for the public API, the weights
cache, and the mass-consistency correction applied after regridding, and
`src/Regridding/README.md` for the file map and invariants.
"""
module Regridding

using ..Grids: AbstractHorizontalMesh, AbstractStructuredMesh,
               LatLonMesh, CubedSphereMesh, ReducedGaussianMesh,
               AbstractCubedSpherePanelConvention,
               GnomonicPanelConvention, GEOSNativePanelConvention,
               nrings, nboundaries, ring_cell_count, cell_index,
               nx, ny, ncells, cell_area, panel_cell_corner_lonlat,
               cs_definition, coordinate_law, center_law, longitude_offset_deg,
               cs_definition_tag, coordinate_law_tag, center_law_tag

using ConservativeRegridding
using ConservativeRegridding: Regridder
using ConservativeRegridding.Trees

import GeometryOps as GO
import GeometryOpsCore as GOCore
import GeoInterface as GI
import GeometryOps: SpatialTreeInterface as STI
using GeometryOps.UnitSpherical: UnitSphericalPoint

using JLD2
using NCDatasets
using SparseArrays
using LinearAlgebra
using Dates
using SHA
using StaticArrays: SA

export build_regridder, save_regridder, load_regridder
export save_esmf_weights, apply_regridder!
export cubed_sphere_face_corners
export IdentityRegrid, meshes_equivalent

include("treeify_meshes.jl")
include("identity_regrid.jl")
include("weights_io.jl")

end # module Regridding
