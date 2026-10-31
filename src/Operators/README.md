# Operators

Physics operators for the transport runtime.

This folder is where the model-level operator contract becomes concrete:
advection, diffusion, surface flux, convection, and chemistry all expose
compatible `apply!` entry points. Topology-specific implementations remain
separate where their storage and numerical kernels genuinely differ.

## What This Tree Owns

- The shared operator entrypoint contract in [`AbstractOperators.jl`](AbstractOperators.jl)
- The step recorders in [`StepRecording.jl`](StepRecording.jl): the
  observation hooks the production step calls (no-ops for `NoStepRecorder`),
  through which the cubed-sphere adjoint records its tape
- The model step as data, [`StepPlan/`](StepPlan/README.md): the stages a run
  executes at window advance, window start, every step and window end
- The operator module assembly and public exports in [`Operators.jl`](Operators.jl)
- The concrete submodules:
  - [`Advection/`](Advection/README.md)
  - [`Diffusion/`](Diffusion/README.md)
  - [`SurfaceFlux/`](SurfaceFlux/README.md)
  - [`Convection/`](Convection/README.md)
  - [`Chemistry/`](Chemistry/README.md)

## Runtime Composition Today

- A [`StepPlan`](StepPlan/README.md) holds the composition.
  [`TransportModel.step!`](../Models/Core/TransportModel.jl) executes the `step`
  stages of the substep plan:
  - first constant-rate decay half-step
  - transport block: StepPlan supplies diffusion and surface flux to the
    abstract hook at the advection Strang midpoint
  - convection block, executed when the model carries a non-`NoConvection`
    operator
  - closing chemistry block
- `DrivenSimulation` executes all four phases of its plan; under a binary
  window contract with `physics_cadence = :window` convection and chemistry
  run once per met window. The cubed-sphere adjoint replays the same plan.

## Topology Coverage

See [`TOPOLOGY_SUPPORT.md`](TOPOLOGY_SUPPORT.md) for the canonical
operator × topology matrix.

## File Map

- [`AbstractOperators.jl`](AbstractOperators.jl) — root `apply!` contract
  and abstract operator families
- [`OperatorState.jl`](OperatorState.jl) — persistent-state interface and storage copying
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
- [`StepRecording.jl`](StepRecording.jl) — `AbstractStepRecorder`,
  `NoStepRecorder` and the recorder hooks (`record_sweep!`, `record_halo!`,
  `record_center!`, `record_diffusion!`, `record_emission!`,
  `record_linrood_horizontal!`, `record_convection!`,
  `record_air_mass_reset!`, `record_chemistry!`)
- [`StepPlan/`](StepPlan/README.md) — `StepPlan`, its stages, `StepOperators`,
  the stage executors, the air-mass reset, and the persistent-operator-state
  snapshots the adjoint's checkpoints use

## Include-Order Dependencies

[`Operators.jl`](Operators.jl) is not just a manifest. Its include order
encodes real dependencies:

- `Diffusion`, `SurfaceFlux` and `Convection` load before `Advection` in the
  current assembly. Advection imports none of them; its midpoint is abstract.
- `StepPlan/` composes diffusion/emission with advection after all families load.
- `Chemistry` is loaded after `Advection` because it is a separate
  post-transport block
- `StepRecording.jl` is included first (the palindromes call its hooks) and
  `StepPlan/` last (its stages call every operator family)

If you refactor include order here, re-check `using` imports inside the
submodules before assuming it is cosmetic.

## Common Tasks

- Adding a new operator family:
  start in [`AbstractOperators.jl`](AbstractOperators.jl), then add a
  submodule and wire exports in [`Operators.jl`](Operators.jl)
- Extending topology support:
  read the submodule README first, then check the corresponding
  `State/`, `Grids/`, and `Models/Core/TransportModel.jl` runtime boundary
- Tracing a live runtime path:
  start at [`../Models/Core/TransportModel.jl`](../Models/Core/TransportModel.jl),
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
- [`../Models/Core/TransportModel.jl`](../Models/Core/TransportModel.jl) decides
  block ordering and which operator families are actually executed
- [`../Models/Core/driven_stepping.jl`](../Models/Core/driven_stepping.jl) (the
  `DrivenSimulation` constructor) installs operator configuration into the
  model runtime

## Related Docs And Tests

- Runtime/block ordering:
  [`../Models/Core/TransportModel.jl`](../Models/Core/TransportModel.jl) and
  [`../../docs/20_RUNTIME_FLOW.md`](../../docs/20_RUNTIME_FLOW.md)
- Runtime walkthrough:
  [`../../docs/20_RUNTIME_FLOW.md`](../../docs/20_RUNTIME_FLOW.md)
- Topology coverage and current status:
  [`TOPOLOGY_SUPPORT.md`](TOPOLOGY_SUPPORT.md)
- Test entrypoints:
  - [`../../test/core/test_basis_explicit_core.jl`](../../test/core/test_basis_explicit_core.jl)
  - [`../../test/core/test_driven_simulation.jl`](../../test/core/test_driven_simulation.jl)
  - submodule-specific tests listed in each subfolder README

## Depends on

State, Grids, MetDrivers and foundation modules. Each physics submodule imports
its own lower dependencies. Diffusion loads before SurfaceFlux and Advection.

## Used by

Models and Adjoints compose the public operator APIs; extensions provide backend hooks.

## Shared helpers owned here

Operator roots, `apply!`, recorder hooks and re-exports from physics submodules.
Diffusion owns PBL physics, diffusion workspace validation and staged geometry APIs.
`operator_state`, `restore_operator_state!` and `reset_operator_state!` define the
checkpoint interface shared by Convection and StepPlans. Storage copying and
zeroing stay with that interface.

# shared: operator_state, restore_operator_state!, reset_operator_state!, _copy_state, _copy_state!, _zero_state!

## Forward<->adjoint mirrors

Diffusion transpose kernels live beside their forward code. Tape/objective wiring
stays in Adjoints. Other operator mirrors are listed in `test/structure/mirrors.toml`.
