# Operators

Physics operators for the transport runtime.

This folder is where the model-level operator contract becomes concrete:
advection, diffusion, surface flux, convection, and chemistry all expose
compatible `apply!` entry points. Topology-specific implementations remain
separate where their storage and numerical kernels genuinely differ.

## What This Tree Owns

- The shared operator entrypoint contract in [`AbstractOperators.jl`](AbstractOperators.jl)
- The operator module assembly and public exports in [`Operators.jl`](Operators.jl)
- The concrete submodules:
  - [`Advection/`](Advection/README.md)
  - [`Diffusion/`](Diffusion/README.md)
  - [`SurfaceFlux/`](SurfaceFlux/README.md)
  - [`Convection/`](Convection/README.md)
  - [`Chemistry/`](Chemistry/README.md)

## Runtime Composition Today

- [`TransportModel.step!`](../Models/TransportModel.jl) currently runs:
  - transport block: advection, with diffusion and surface flux embedded
    at the Strang midpoint
  - convection block, executed when the model carries a non-`NoConvection`
    operator
  - chemistry block

## Topology Coverage

See [`TOPOLOGY_SUPPORT.md`](TOPOLOGY_SUPPORT.md) for the canonical
operator × topology matrix.

## File Map

- [`AbstractOperators.jl`](AbstractOperators.jl) — root `apply!` contract
  and abstract operator families
- [`Operators.jl`](Operators.jl) — include order, submodule assembly,
  re-exports, and the public surface seen by `AtmosTransport`
- [`Advection/`](Advection/README.md) — finite-volume transport core,
  Strang palindromes, cubed-sphere panel transport, Lin-Rood utilities
- [`Diffusion/`](Diffusion/README.md) — implicit vertical diffusion,
  Thomas solve, topology-specific runtime adapters
- [`SurfaceFlux/`](SurfaceFlux/README.md) — per-tracer surface sources,
  source-to-tracer mapping, topology-specific application kernels
- [`Convection/`](Convection/README.md) — convection operator hierarchy,
  CMFMC kernels, workspaces, forcing contract
- [`Chemistry/`](Chemistry/README.md) — source/sink operators applied
  after transport

## Include-Order Dependencies

[`Operators.jl`](Operators.jl) is not just a manifest. Its include order
encodes real dependencies:

- `Diffusion` is loaded before `Advection` because the transport
  palindrome imports diffusion operator types and `apply_vertical_diffusion_vmr!`
- `SurfaceFlux` is loaded before `Advection` for the same reason
- `Convection` is loaded before `Advection` for consistency, but is not
  consumed inside the current palindrome
- `Chemistry` is loaded after `Advection` because it is a separate
  post-transport block

If you refactor include order here, re-check `using` imports inside the
submodules before assuming it is cosmetic.

## Common Tasks

- Adding a new operator family:
  start in [`AbstractOperators.jl`](AbstractOperators.jl), then add a
  submodule and wire exports in [`Operators.jl`](Operators.jl)
- Extending topology support:
  read the submodule README first, then check the corresponding
  `State/`, `Grids/`, and `Models/TransportModel.jl` runtime boundary
- Tracing a live runtime path:
  start at [`../Models/TransportModel.jl`](../Models/TransportModel.jl),
  then follow the relevant operator `apply!`
- Understanding what is actually production-live:
  consult [`TOPOLOGY_SUPPORT.md`](TOPOLOGY_SUPPORT.md) (canonical
  per-operator × per-topology matrix) and submodule-specific tests

## Cross-Dependencies

- [`../State/`](../State/) provides `CellState`, `CubedSphereState`,
  tracer accessors, flux-state types, and time-varying field contracts
- [`../Grids/`](../Grids/) provides mesh/topology types and geometry
  accessors used by topology-specific kernels
- [`../MetDrivers/`](../MetDrivers/) provides transport-window timing,
  `current_time`, and convection forcing containers
- [`../Models/TransportModel.jl`](../Models/TransportModel.jl) decides
  block ordering and which operator families are actually executed
- [`../Models/driven_stepping.jl`](../Models/driven_stepping.jl) (the
  `DrivenSimulation` constructor) installs operator configuration into the
  model runtime

## Related Docs And Tests

- Runtime/block ordering:
  [`../Models/TransportModel.jl`](../Models/TransportModel.jl) and
  [`../../docs/20_RUNTIME_FLOW.md`](../../docs/20_RUNTIME_FLOW.md)
- Runtime walkthrough:
  [`../../docs/20_RUNTIME_FLOW.md`](../../docs/20_RUNTIME_FLOW.md)
- Topology coverage and current status:
  [`TOPOLOGY_SUPPORT.md`](TOPOLOGY_SUPPORT.md)
- Test entrypoints:
  - [`../../test/core/test_basis_explicit_core.jl`](../../test/core/test_basis_explicit_core.jl)
  - [`../../test/core/test_driven_simulation.jl`](../../test/core/test_driven_simulation.jl)
  - submodule-specific tests listed in each subfolder README
