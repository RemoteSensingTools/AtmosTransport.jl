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
  [`LatLonMesh.jl`](LatLonMesh.jl); `LatLonMesh(; Nx, Ny, longitude, latitude)`
  for the regular transport mesh and `LatLonMesh(λᶠ, φᶠ; FT, radius)` for
  meshes from explicit cell faces (source grids of input files)
- Reduced-Gaussian face-connected mesh:
  [`ReducedGaussianMesh.jl`](ReducedGaussianMesh.jl)
- Cubed-sphere mesh definitions and panel conventions:
  [`CubedSphereMesh.jl`](CubedSphereMesh.jl)
  The exported `_default_cs_definition(convention)` returns the convention's
  standard definition for consumers such as the non-core snapshot reader.
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
- [`LatLonMesh.jl`](LatLonMesh.jl) — logically rectangular lat-lon mesh,
  regular or from explicit faces, and `latitude_row_order`
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

## Lat-lon meshes from explicit faces

`LatLonMesh(λᶠ, φᶠ; FT, radius)` is the one construction path of the type; the
regular keyword constructor builds its faces with `range` and calls it, so a
regular mesh is reproduced bit for bit. `radius` is required: a source mesh
must sit on the sphere of the mesh it is regridded to. The contract:

- Longitude faces are uniform (relative tolerance `1e-10`, plus two units of
  roundoff of the input precision) and span at most 360°. The first face may be
  any value (TRENDY grids start at -180.5°, -0.625°, -1.25°, -180.625° or
  -180.9375°; 0…360 grids are kept as they are). A sequence that wraps by -360°
  is unwrapped, not rolled: every face after a negative step is shifted by
  +360°, so the stored faces increase and mesh column `i` is input column `i`.
- Latitude faces are strictly monotone in `[-90, 90]`. The mesh always stores
  them ascending (south to north), which the transport code, the conservative
  regridder's cell polygons, the observation locator and the output writers
  expect. For descending input (files stored north to south),
  `latitude_row_order(φᶠ)` returns the row permutation: mesh row `j` is input
  row `latitude_row_order(φᶠ)[j]`.
- Uniform latitude faces give a finite `Δφ`. Non-uniform faces (pole-centred
  rows with half-width polar caps, a truncated first row, Gaussian latitudes)
  give `Δφ = NaN`; `dx`, `dy` and the x-face lengths then throw an
  `ArgumentError`, so such a mesh cannot reach Δφ-based metrics silently. The
  lat-lon `write_transport_binary` refuses it too, because the binary header
  records only the latitude interval.
- Cell areas always come from the faces,
  `A_ij = R² Δλ (sin φᶠ[j+1] - sin φᶠ[j])` (Δλ in radians), and sum to `4πR²`
  for a global mesh. Centres are face midpoints; for pole-centred and Gaussian
  rows they differ from the file's coordinate, which only labels the cell.

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
  - [`../../test/core/test_latlon_explicit_faces.jl`](../../test/core/test_latlon_explicit_faces.jl)
  - [`../../test/core/test_reduced_gaussian_mesh.jl`](../../test/core/test_reduced_gaussian_mesh.jl)
  - [`../../test/core/test_basis_explicit_core.jl`](../../test/core/test_basis_explicit_core.jl)
  - [`../../test/core/test_cubed_sphere_runtime.jl`](../../test/core/test_cubed_sphere_runtime.jl)
