# Model Core

The `Models.Core` submodule owns model construction, fixed-step simulation,
callbacks and the `step!`/`run!` generics. It has no dependency on Runner or
its concrete DrivenSimulation type. Its public names are Models facade aliases.

## File Map

- `Core.jl`: module imports, the `step!`/`run!` declarations shared with
  Runner, and include order.
- `TransportModel.jl`: concrete model/workspace, constructors, operator
  installers, adaptation, `step_operators` and the unchanged substep plan.
- `Simulation.jl`: fixed-step `Simulation`, `step!`, `run!` methods.
- `Callbacks.jl`: schedules, `Callback`, `NaNChecker`; the existing
  `NonfiniteScan` point operation and launch are unchanged.
- `adjoint_support.jl`: model-only `require_adjoint_support`; driven
  scheduling and forcing capture live in `../Runner/adjoint_schedule.jl`.

## Depends on

State, Grids, Architectures, Operators and StepPlans, MetDrivers, Adjoints
and SectionTimer. No sibling or parent-facade dependency.

## Used by

Runner, package exports, runtime and adjoint tests, direct model callers.

## Shared helpers owned here

Public `TransportModel`, `Simulation`, `step!`, `run!`, installers, schedules
and callbacks. No underscore helper is shared; Runner owns `run_window!`,
`cs_adjoint_inputs` and the prefetch drain. Construction funnels through
`_construct_model` after each constructor's original preflight/workspace
allocation; defaults stay local.

## Forward<->adjoint mirrors

Runner stepping and capture share Core model operators and the same
Operators.StepPlans plan. `test_cs_adjoint_step_plan_parity.jl` checks their
numerical parity; Core contains no driven forcing or clock implementation.
