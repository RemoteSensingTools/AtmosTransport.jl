# Models

Runtime composition of state, met drivers, operators and output. The entry
point for `scripts/run_transport.jl` remains `run_driven_simulation(cfg)`.
All existing `AtmosTransport.Models.<name>` and package exports retain their
module paths, including `InitialConditionIO`, `DrivenRunner`,
`BinaryPathExpander` and `InputStaging`.

## File Map

- [`Models.jl`](Models.jl): imports, include order and facade exports.
- [`Core/`](Core/README.md): model construction, fixed-step simulation,
  callbacks and model adjoint support.
- [`Physics/`](Physics/README.md): runtime specs, styles, recipes, metadata
  and provenance.
- [`InitialConditions/`](InitialConditions/README.md): the `InitialConditionIO`
  submodule: analytic/file IC builders, conservative hybrid remapping and flux
  loading.
- [`Runner/`](Runner/README.md): `DrivenSimulation`, window storage,
  `DrivenRunner`, forcing refresh, driven stepping, adjoint capture, binary-path
  expansion, input staging and runtime layout API.

## Assembly and dependency order

Each folder is one submodule, loaded in the order
`Physics → Core → InitialConditionIO → Runner` (the InitialConditions folder's
module keeps its existing name `InitialConditionIO`). Dependencies between
siblings point to earlier modules: Physics owns specs/recipes without model
types; Core owns the concrete model, callbacks and the `step!`/`run!` generics;
Runner owns DrivenSimulation, `run_window!`, and its stepping, forcing refresh
and adjoint capture methods, and extends Core's `step!` and `run!`. Core's
model-only adjoint support stays with TransportModel.

The facade re-exports and aliases only public names. Underscore helpers stay
private to their owning submodule; tests reach them as `Models.Physics._name`,
`Models.Runner._name` or `Operators.StepPlans._name`. `DrivenRunner`,
`BinaryPathExpander` and `InputStaging` remain reachable at their old Models
paths as aliases of their nested modules.
Children import their dependencies directly through package-relative paths,
including `...Models.Core`, so sandboxed package copies and JET resolve the
same owners. There is no sibling back-reference or special checker exception.
Runtime execution order, staging and module initialization effects are unchanged.

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
Public aliases share the original generic. See each folder for the few
underscore seams shared across submodules.

## Forward<->adjoint mirrors

Runner's `adjoint_schedule.jl` builds `CSStepSchedule` and captures production
forcing for Adjoints.Footprint. The `_has_initial_operator_state` predicate
retains T1's name. `test_cs_adjoint_step_plan_parity.jl` guards forward/capture
agreement; CPU/GPU golden and adjoint parity runs belong to the main session.
