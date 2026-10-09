# State

Runtime state containers and field abstractions.

This folder owns the data structures that the transport runtime mutates
or reads directly: cell-centered air mass and tracer state, face-flux
state, meteorological snapshots, tracer accessors, basis tags, and the
time-varying field contracts consumed by operators.

## Entry Points

- State module assembly:
  [`State.jl`](State.jl)
- Prognostic cell-centered state:
  [`CellState.jl`](CellState.jl)
  defines `CellState`
- Panel-native cubed-sphere prognostic state:
  [`CubedSphereState.jl`](CubedSphereState.jl)
  defines `CubedSphereState`
- Face-flux state hierarchy and allocators:
  [`FaceFluxState.jl`](FaceFluxState.jl)
  defines `StructuredFaceFluxState`, `FaceIndexedFluxState`,
  `CubedSphereFaceFluxState`, and `allocate_face_fluxes`
- Tracer allocation and access helpers:
  [`Tracers.jl`](Tracers.jl)
- Basis tags:
  [`Basis.jl`](Basis.jl)
- Time-varying field models:
  [`Fields/README.md`](Fields/README.md)

## What This Folder Owns

- Basis-safe storage for air mass and fluxes
- Topology-specific runtime containers:
  - structured / face-indexed via `CellState`
  - cubed sphere via `CubedSphereState`
- Flux-state containers that match the mesh topology:
  - directional structured storage
  - face-indexed reduced-Gaussian storage
  - panel-native cubed-sphere storage
- Tracer naming and packed storage conventions
- Time-varying field contracts used by diffusion, chemistry, and surface
  flux operators

## File Map

- [`State.jl`](State.jl) — submodule assembly and re-exports
- [`Basis.jl`](Basis.jl) — dry/moist basis tags for state and fluxes
- [`CellState.jl`](CellState.jl) — structured and face-indexed
  cell-centered prognostic state
- [`CubedSphereState.jl`](CubedSphereState.jl) — panel-native
  cubed-sphere prognostic state
- [`FaceFluxState.jl`](FaceFluxState.jl) — flux-state hierarchy,
  directional/face-indexed/panel-native storage, allocators
- [`MetState.jl`](MetState.jl) — legacy container for upstream meteorological
  fields; deprecated, no caller in the package
- [`Tracers.jl`](Tracers.jl) — tracer allocation, lookup, iteration, and
  mutation helpers
- [`Fields/`](Fields/README.md) — time-varying field abstraction and
  concrete field types

## Storage Conventions

- `CellState.air_mass` matches the active grid topology:
  - structured: `(Nx, Ny, Nz)`
  - face-indexed: `(ncells, Nz)`
- `CellState.tracers_raw` is packed with tracer axis last:
  - structured: `(Nx, Ny, Nz, Nt)`
  - face-indexed: `(ncells, Nz, Nt)`
- `CubedSphereState` is panel-native:
  - `air_mass :: NTuple{6}` of halo-padded `(Nc + 2Hp, Nc + 2Hp, Nz)`
  - `tracers_raw :: NTuple{6}` of halo-padded
    `(Nc + 2Hp, Nc + 2Hp, Nz, Nt)`
- Flux states follow mesh-native storage rather than forcing one global
  layout

## Common Tasks

- Adding a new tracer-aware runtime path:
  start in [`Tracers.jl`](Tracers.jl) and the relevant state container
- Adding a new topology-specific state:
  keep the public access pattern aligned with `ntracers`, `get_tracer`,
  `eachtracer`, and `allocate_face_fluxes`
- Debugging basis mismatches:
  read [`Basis.jl`](Basis.jl) and basis checks in
  [`../Models/DrivenSimulation.jl`](../Models/DrivenSimulation.jl)
- Debugging cubed-sphere storage:
  start with [`CubedSphereState.jl`](CubedSphereState.jl) and
  [`FaceFluxState.jl`](FaceFluxState.jl), not the operators
- Adding a new operator input field:
  use [`Fields/README.md`](Fields/README.md)

## Cross-Dependencies

- [`../Grids/README.md`](../Grids/README.md) defines the mesh/topology
  contracts that choose the concrete storage layout
- [`../Operators/README.md`](../Operators/README.md) consumes these state
  and flux containers directly
- [`../MetDrivers/README.md`](../MetDrivers/README.md) fills air mass and
  flux storage from transport windows
- [`../Models/README.md`](../Models/README.md) composes these containers
  into runtime model objects

## Related Docs And Tests

- Runtime flow:
  [`../../docs/20_RUNTIME_FLOW.md`](../../docs/20_RUNTIME_FLOW.md)
- Topology status:
  [`../Operators/TOPOLOGY_SUPPORT.md`](../Operators/TOPOLOGY_SUPPORT.md)
- Tests:
  - [`../../test/core/test_basis_explicit_core.jl`](../../test/core/test_basis_explicit_core.jl)
  - [`../../test/orphan/test_fields.jl`](../../test/orphan/test_fields.jl)
  - [`../../test/core/test_reduced_gaussian_mesh.jl`](../../test/core/test_reduced_gaussian_mesh.jl)
  - [`../../test/orphan/test_cubed_sphere_runtime.jl`](../../test/orphan/test_cubed_sphere_runtime.jl)
