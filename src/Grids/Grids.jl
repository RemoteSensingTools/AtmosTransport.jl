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

using ..Architectures: CPU
import ..Architectures: architecture  # extended with a method on AtmosGrid below
using ..Parameters: PlanetParameters, earth_parameters, EARTH_RADIUS

include("AbstractMeshes.jl")
include("VerticalCoordinates.jl")
include("GeometryOps.jl")
include("LatLonMesh.jl")
include("PanelConnectivity.jl")
include("CubedSphereMesh.jl")
include("cs_mesh_coordinates.jl")
include("cs_mesh_locate.jl")
include("ReducedGaussianMesh.jl")

end # module Grids
