"""
    Grids

Geometry layer for the dry-mass transport architecture.

Provides:
- `AbstractHorizontalMesh` hierarchy with face/cell oriented API
- `AbstractVerticalCoordinate` with `HybridSigmaPressure`
- `AtmosGrid` composite type (mesh + vertical + architecture)
- Concrete meshes: `LatLonMesh` (structured fast path), `ReducedGaussianMesh`
  (variable-ring native ERA5 / IFS path), and `CubedSphereMesh`
  (structured GEOS/FV3 metadata with explicit panel conventions)
"""
module Grids

using ..Parameters: PlanetParameters, earth_parameters, EARTH_RADIUS, GMAO_CS_LONGITUDE_OFFSET_DEG

include("AbstractMeshes.jl")
include("VerticalCoordinates.jl")
include("GeometryOps.jl")
include("LatLonMesh.jl")
include("PanelConnectivity.jl")
include("PanelIndices.jl")
include("CSGlobalFaceTable.jl")
include("PanelPacking.jl")
include("CubedSphereMesh.jl")
include("SphereVectors.jl")
include("cs_mesh_coordinates.jl")
include("cs_mesh_locate.jl")
include("ReducedGaussianMesh.jl")
include("CellAreas.jl")
include("BilinearBrackets.jl")

end # module Grids
