# Grids

Geometry and topology layer for the transport runtime.

This folder defines the horizontal meshes, vertical coordinates, grid
composition object, and geometry accessors that the rest of the runtime
dispatches on. If the code is deciding between structured, face-indexed,
or cubed-sphere behavior, the types from this folder are usually what
make that decision possible.

## Entry Points

- Grid module assembly:
  [`Grids.jl`](Grids.jl)
- Core mesh/topology abstractions:
  [`AbstractMeshes.jl`](AbstractMeshes.jl)
  defines `AbstractHorizontalMesh`, `AbstractStructuredMesh`,
  `AbstractFluxTopology`, and `AtmosGrid`
- Generic geometry accessors:
  [`GeometryOps.jl`](GeometryOps.jl)
- Structured lat-lon mesh:
  [`LatLonMesh.jl`](LatLonMesh.jl)
- Reduced-Gaussian face-connected mesh:
  [`ReducedGaussianMesh.jl`](ReducedGaussianMesh.jl)
- Cubed-sphere mesh definitions and panel conventions:
  [`CubedSphereMesh.jl`](CubedSphereMesh.jl)
- Cubed-sphere panel connectivity:
  [`PanelConnectivity.jl`](PanelConnectivity.jl)
- Vertical coordinate:
  [`VerticalCoordinates.jl`](VerticalCoordinates.jl)
  defines `HybridSigmaPressure`

## What This Folder Owns

- The topology tags that dispatch storage and operator behavior
- The `AtmosGrid` composite used across drivers, models, and operators
- Mesh-native geometry helpers like cell counts, face connectivity,
  normals, and lengths
- Cubed-sphere coordinate law, center law, panel convention, and
  edge-connectivity metadata

## File Map

- [`Grids.jl`](Grids.jl) — submodule assembly
- [`AbstractMeshes.jl`](AbstractMeshes.jl) — mesh hierarchy, flux
  topology tags, `AtmosGrid`
- [`VerticalCoordinates.jl`](VerticalCoordinates.jl) — hybrid-sigma
  vertical coordinate and pressure helpers
- [`GeometryOps.jl`](GeometryOps.jl) — generic geometry accessor API
- [`LatLonMesh.jl`](LatLonMesh.jl) — logically rectangular lat-lon mesh
- [`ReducedGaussianMesh.jl`](ReducedGaussianMesh.jl) — face-connected
  reduced-Gaussian mesh with ring-aware metadata
- [`PanelConnectivity.jl`](PanelConnectivity.jl) — cubed-sphere edge
  graph and reciprocal-edge helpers
- [`CubedSphereMesh.jl`](CubedSphereMesh.jl) — cubed-sphere mesh,
  GMAO/equiangular definitions, panel conventions
- [`cs_mesh_coordinates.jl`](cs_mesh_coordinates.jl) — forward geometry:
  panel → xyz/lon-lat projection, cell centers and corners, face edge lengths,
  local tangent bases (used by the constructor)
- [`cs_mesh_locate.jl`](cs_mesh_locate.jl) — inverse projection
  (lon, lat) → (panel, s, t)

## Common Tasks

- Adding a new mesh topology:
  start in [`AbstractMeshes.jl`](AbstractMeshes.jl), then implement the
  geometry accessors the operators and state allocators expect
- Debugging topology dispatch:
  check `flux_topology(mesh)` and the concrete mesh type before touching
  operator code
- Working on cubed-sphere edge behavior:
  read [`PanelConnectivity.jl`](PanelConnectivity.jl) and
  [`CubedSphereMesh.jl`](CubedSphereMesh.jl) together
- Working on vertical-coordinate-dependent logic:
  use [`VerticalCoordinates.jl`](VerticalCoordinates.jl) before
  duplicating pressure helper math elsewhere

## Cross-Dependencies

- [`../State/README.md`](../State/README.md) uses these mesh/topology
  tags to choose storage layouts and flux allocators
- [`../Operators/README.md`](../Operators/README.md) dispatches heavily
  on mesh type and topology
- [`../MetDrivers/README.md`](../MetDrivers/README.md) constructs
  runtime grids in `load_grid`
- [`../Models/README.md`](../Models/README.md) carries `AtmosGrid`
  through the runtime

## Related Docs And Tests

- User guides:
  - [`../../docs/src/concepts/grids.md`](../../docs/src/concepts/grids.md)
  - [`../../docs/src/preprocessing/conventions.md`](../../docs/src/preprocessing/conventions.md)
- Tests:
  - [`../../test/core/test_structured_mesh_metadata.jl`](../../test/core/test_structured_mesh_metadata.jl)
  - [`../../test/core/test_reduced_gaussian_mesh.jl`](../../test/core/test_reduced_gaussian_mesh.jl)
  - [`../../test/core/test_basis_explicit_core.jl`](../../test/core/test_basis_explicit_core.jl)
  - [`../../test/orphan/test_cubed_sphere_runtime.jl`](../../test/orphan/test_cubed_sphere_runtime.jl)
