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
  refresh. With midpoint forcing enabled, interpolated fluxes and humidity
  use `(substep - 0.5) / steps_per_window`, including `0.5` for one-step windows.
  `_refreshes_fluxes` says at which steps `_refresh_forcing!` rewrites the face
  fluxes (a window's first step, or every step when interpolated).
- `driven_stepping.jl`: driven constructor, plan phases, window advance,
  source staging, clock transition, `step!`, `run_window!` and `run!`; imports
  the StepPlans reset/halo helpers and SurfaceFlux staging helpers it calls.
  The constructor resolves `air_mass_update` from the model's advection (a
  per-step air-mass reset for `NoAdvection`); `air_mass_reset_target` supplies
  the window-start, window-end and step-end (`m + λ dm`) targets.
- `adjoint_schedule.jl`: `CSStepSchedule(sim)`, `cs_adjoint_inputs`, fresh-state
  checks and production forcing capture (one copy of the face fluxes each time
  production rewrites them, shared by the steps of a met window; per window
  one convection forcing, diffusion operator and layer thickness, no diffusion
  workspace). Reset targets load each window once; a window's start target is
  the previous window's end target, one array, when the two are bitwise equal
  (`_window_reset_targets`, `_bitwise_equal`). `residency` selects where the
  capture keeps the met: `:device` (`_DeviceCapture`: per-step vectors, every
  window on the model backend) or `:host` (`_HostCapture`: each completed
  window copied to the host and its device copies freed, targets kept on the
  host, projected host bytes checked against the budget after every window;
  the inputs carry a `HostCSWindowSource` as `kwargs.met_source`).
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

TM5 live CS diffusion captures a frozen copy of D per window, without scratch
or metrics, and no layer thickness (`diffusion_layer_thickness = nothing`)
because D needs no dz. The default capture for existing Kz operators retains
the deepcopy behavior and copies the window's layer thickness (none for Dkg
fields); the replay owns the diffusion workspace and reverse scratch.

Sharing reset targets is exact: the replay writes a target only when a reset
refreshes its halos (`fill_panel_halos!(…; dir = 1)` rewrites every halo cell
from the interior), so either reset of a shared target installs what separate
copies would; the interior is never written.
