# Driven Runtime

The `Models.Runner` submodule owns DrivenSimulation, window storage, forcing
refresh, driven stepping and adjoint capture. It loads after Physics, Core and
InitialConditionIO. Core declares the `step!` and `run!` generics; Runner
imports and extends them on its concrete driven type and owns `run_window!`.
No lower sibling imports Runner.

## File Map

- `Runner.jl`: imports, child modules and runtime include order.
- `DrivenSimulation.jl`: concrete driven type, grid/basis compatibility and
  storage adaptation helpers.
- `driven_window_state.jl`: window storage, copies, staging and prefetch.
- `driven_physics_refresh.jl`: per-window dz/Kz, convection forcing and midpoint
  refresh; expressions and the one-step-window rule are unchanged.
  `_refreshes_fluxes` says at which steps `_refresh_forcing!` rewrites the face
  fluxes (a window's first step, or every step when interpolated).
- `driven_stepping.jl`: driven constructor, plan phases, window advance,
  source staging, clock transition, `step!`, `run_window!` and `run!`; imports
  the StepPlans reset/halo helpers and SurfaceFlux staging helpers it calls.
- `adjoint_schedule.jl`: `CSStepSchedule(sim)`, `cs_adjoint_inputs`, fresh-state
  checks and production forcing capture (one copy of the face fluxes each time
  production rewrites them, shared by the steps of a met window).
- `BinaryPathExpander.jl`: existing binary-path child module.
- `InputStaging.jl`: existing input-staging child module.
- [`DrivenRunner/`](DrivenRunner/README.md): configured runner, model setup,
  output/observation glue, resource ownership and run layout.

`Models.DrivenSimulation` and the public stepping APIs remain facade aliases.
`Models.DrivenRunner`, `Models.BinaryPathExpander` and `Models.InputStaging`
still refer to the same nested modules. Underscore helpers are private to
Runner; tests reach them as `Models.Runner._name`.

## Depends on

Core stepping/model APIs; Physics and InitialConditionIO through DrivenRunner;
Architectures, SectionTimer, State, Grids, Operators and StepPlans, Adjoints,
MetDrivers, ConfigChecks and AtomicFiles. Child DrivenRunner additionally uses
Output, Observations, GriddedIO, Parameters and LogLevels.

## Used by

The Models/package facades; scripts/run_transport.jl; Diagnostics and its
Analysis facade; diagnostic/inversion/flux-building scripts; runtime tests.

## Shared helpers owned here

Public driven stepping, adjoint capture, path/staging and configured-run APIs
retain their facade bindings. The internal prefetch drain is shared with the
configured runner (DrivenRunner closes inputs after draining it):

# shared: _finish_window_prefetch!

## Forward<->adjoint mirrors

`driven_stepping.jl` and `adjoint_schedule.jl` share the production plan,
refresh functions and clock transition; the capture copies the fluxes exactly
when `_refreshes_fluxes` says production rewrote them. Operators.StepPlans owns clock/dt
arithmetic. Resource cleanup drains prefetch before closing the input binary.

The live `TM5CalcKzzField` refresh reads runtime carrier mass and stored-window
face amounts after the per-window mass reset. Its direct exchange path skips
dz reconstruction; `driven_window_state.jl` copies the new input group on device
handoffs, through the existing optional NamedTuple storage copier.

Storage allocation delegates to Architectures.similar_arrays with the existing
`Base.invokelatest(similar, array)` allocator at every leaf. Flux scaling uses
`scale_arrays!`, retaining one broadcast per panel in the original order.
