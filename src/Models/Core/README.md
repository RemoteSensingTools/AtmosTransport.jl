# Model Core

Plain subfolder of Models; see the parent README for the shared-type cycle and
include order. `TransportModel` and `DrivenSimulation` keep their old paths.

## File Map

- `TransportModel.jl`: concrete model/workspace, constructors, operator
  installers, adaptation, `step_operators` and the unchanged substep plan.
- `Simulation.jl`: fixed-step `Simulation`, `step!`, `run!`.
- `Callbacks.jl`: interval schedules, `Callback`, `NaNChecker`; the existing
  `NonfiniteScan` point operation and launch are unchanged.
- `driven_stepping.jl`: driven constructor, window advance, surface-flux
  staging, plan phases, `run_window!` and `run!`.
- `adjoint_schedule.jl`: adjoint support, `CSStepSchedule(sim)`,
  `cs_adjoint_inputs`, fresh persistent-state checks and capture (one copy
  of the face fluxes each time production rewrites them, shared by the steps
  of a met window).

## Depends on

State, Grids, Architectures, Operators and StepPlans, MetDrivers, Adjoints;
Runner's driven type/storage and Physics's forcing refresh in the shared module.

## Used by

Runner, package exports, runtime and adjoint tests, direct model callers.

## Shared helpers owned here

Public `TransportModel`, `Simulation`, `step!`, `run!`, installers, schedules,
callbacks and `cs_adjoint_inputs`. DrivenRunner also shares the prefetch drain
owned by Runner. Construction funnels through `_construct_model` after each
constructor's original preflight/workspace allocation; defaults stay local.

## Forward<->adjoint mirrors

`driven_stepping.jl` and `adjoint_schedule.jl` retain their forcing, plan and
clock order. Clocks come from `Operators.StepPlans`. No capture/step phase is
reordered; `test_cs_adjoint_step_plan_parity.jl` checks their numerical parity.
