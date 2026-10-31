# Models

Runtime composition of state, met drivers, operators and output. The entry
point for `scripts/run_transport.jl` remains `run_driven_simulation(cfg)`.
All existing `AtmosTransport.Models.<name>` and package exports retain their
module paths, including `InitialConditionIO`, `DrivenRunner`,
`BinaryPathExpander` and `InputStaging`.

## File Map

- [`Models.jl`](Models.jl): imports, include order and facade exports.
- [`Core/`](Core/README.md): model construction, fixed-step simulation,
  callbacks, driven stepping and the adjoint schedule/capture bridge.
- [`Physics/`](Physics/README.md): runtime specs, styles, recipes, metadata
  and per-window forcing refresh.
- [`InitialConditions/`](InitialConditions/README.md): `InitialConditionIO`,
  analytic/file IC builders, conservative hybrid remapping and flux loading.
- [`Runner/`](Runner/README.md): `DrivenSimulation`, window storage,
  `DrivenRunner`, binary-path expansion, input staging and runtime layout API.

## Assembly and dependency order

These are plain subfolders of the existing Models module. Independent modules
would be cyclic with the requested ownership: Core stepping and adjoint
schedule dispatch on Runner's `DrivenSimulation`, Runner constructs Core's
`TransportModel`, and Physics refresh also dispatches on `DrivenSimulation`.
The existing child modules stay nested at their original qualified paths.

`Models.jl` keeps the original include order: Core model type; Physics styles,
config specs, recipes and metadata; InitialConditionIO; Runner path expansion
and staging; Core fixed-step simulation and callbacks; Runner driven type and
window storage; Physics refresh; Core stepping and adjoint schedule; DrivenRunner.
This defines types before their dispatched methods without moving numerical
execution or introducing callbacks across new module boundaries. The structure
checker records the four folders and retains the shared Models wiring explicitly;
external dependencies and child-module ordering remain checked.

## Runtime contract

`TransportModel.step!` executes the existing `Operators.StepPlans` plan.
`DrivenSimulation` executes its window advance/start/step/end phases. Transport
retains its Strang midpoint diffusion/emission calls; this PR does not change
the palindrome. Window clocks and step lengths remain owned by
`Operators.StepPlans.window_clock` and `window_step_dt` (T1).

GPU world-age trampolines, allocation types, kernels, launch configurations,
copy order, resource cleanup and prefetch order are preserved. Conservative
IC remapping keeps Float64 pressure and cumulative scratch for either tracer
precision. Static and time-varying flux remapping retain their different totals.

## Depends on

Architectures, ConfigChecks, SectionTimer, Parameters, State, Grids,
MetDrivers, Operators (including StepPlans), Adjoints, Regridding, GriddedIO,
Output and Observations. The runtime does not import Preprocessing or Extras.

## Used by

The package facade; `scripts/run_transport.jl`; core Diagnostics (re-exported
by Analysis); diagnostic, inversion and flux-building scripts; runtime tests.
Extras/Analysis, Extras/Visualization and extensions need no import changes.

## Shared helpers owned here

Public model installers, schedules, callbacks, simulation and recipe APIs keep
their existing exports. The facade additionally exposes the runner's shared
layout API: `run_tracer_specs`, `runtime_output_origin`, `run_reference_time`,
`run_time_origin`, `run_time_span`, `run_window_index`, `runtime_window_layout`,
`layout_run_hours`, `window_end_hours`, `check_multifile_window_count`,
`binary_date_label`, `configured_float_type`, `parse_tracer_specs`,
`structured_tracer_specs`, `observation_terrain`, and IC `uniform_background`.
`ADVECTION_SPEC_KEYS` is the public schema table used by the inversion CLI.
Public aliases share the original generic; legacy qualified bindings remain.
See each folder for the explicitly shared internal assembly seams.

## Forward<->adjoint mirrors

Core's `adjoint_schedule.jl` builds `CSStepSchedule` and captures production
forcing for Adjoints.Footprint. The `_has_initial_operator_state` predicate
retains T1's name. `test_cs_adjoint_step_plan_parity.jl` guards forward/capture
agreement; CPU/GPU golden and adjoint parity runs belong to the main session.
