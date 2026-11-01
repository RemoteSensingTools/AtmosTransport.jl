# Adjoints

`AtmosTransport.Adjoints` owns objective seeding, sparse linear functionals,
step forcing, support checks, panel allocation helpers, and operator-adjoint
wiring. Its children are `Footprint` and `Inversion`; their public names
remain available through Adjoints and as qualified AtmosTransport names.

## File Map

- `Adjoints.jl`: imports, supported scheme unions, child modules and re-exports.
- `ObjectiveSeeding.jl`: layer, column, placed, retrieval and cell-average
  objectives, evaluation, final seeds and surface readout.
- `StepForcing.jl`: `CSLinearFunctional`, `CSStepForcing`, `NoStepForcing`,
  assembly and deterministic injection, shared by objective seeds and 4D-Var.
- `PanelArrays.jl`: copy and zero allocation for six-panel tuples and surfaces.
- `AdjointSupport.jl`: scheme, stage and operator support policy; the
  `tape_dependence` trait (`AirMassDependence`, `TracerDependence`): whether a
  scheme's reverse reads the tracer, which decides `AirMassTape` legality.
- `DiffusionAdjoint.jl`, `ConvectionAdjoint.jl`: validation and launch wiring.
- `LinRoodTape.jl`: Lin–Rood horizontal record and reverse composition (fenced).
- `Footprint/`: production-step replay, recording, reverse walk, checkpointing.
- `Inversion/`: observation binding/sampling/IO and surface-flux solvers.

## Depends on

Architectures, Grids, State, Parameters, MetDrivers, Operators (including StepPlans), Tape,
and Observations (placement and retrieval types). External: KernelAbstractions.
Child includes and re-exports assemble the facade; core implementation never
calls Footprint or Inversion. Numerical transposes live beside the forward
operators; Lin–Rood composition remains here until its readiness work.

## Used by

Footprint, Inversion, Models (support checks and the re-exported
`CSStepSchedule`), inversion scripts, and adjoint/finite-difference tests.

## Shared helpers owned here

Objective types, `evaluate_objective`, `adjoint_support`,
`require_adjoint_support`, `tape_dependence`, scheme unions and
`CSAdjointWorkspace`.
The following internal interfaces serve the child modules: objective
validation/seeding, fixed sparse functionals and forcing injection, panel
allocation, diffusion/convection validation, and operator reverse launchers.
The Lin–Rood record and storage gate are shared without changing that file.

## Forward<->adjoint mirrors

The reverse launchers transpose the production advection, diffusion and
convection operators. Objective evaluation and seeding use the same mass
floor. Forcing assembly adds entries in a fixed order; one work item owns
one cell. The step-plan parity, kernel identities, gradient identities and
finite-difference tests are the numerical gate.

# shared: AbstractCSFootprintObjective, AbstractStepForcing, AirMassDependence, CSAdjointLinRoodScheme, CSAdjointSupportedScheme, CSCellAverageObjective, CSColumnMeanObjective, CSLayerMeanObjective, CSLinearFunctional, CSPlacedPointObjective, CSRetrievalColumnObjective, CSSeedObjective, NoStepForcing, TracerDependence, _CSLinRoodHorizRecord, _accumulate_surface_footprint!, _apply_cs_convection_adjoint!, _apply_cs_diffusion_adjoint!, _apply_cs_linrood_horizontal_adjoint!, _assemble_step_forcing, _cell_average_needs_4dvar, _copy_panel_tuple, _inject_step_forcing!, _linear_functional, _linrood_validate_tape_storage, _objective_mass_floor, _placed_objective_needs_met, _require_cs_convection_workspace, _retrieval_objective_needs_met, _seed_objective!, _validate_cs_diffusion_inputs, _validate_objective, _validate_step_forcing, _zero_surface_rates, evaluate_objective, require_adjoint_support, tape_dependence

Architectures.zeros_like owns non-fenced panel zero allocation, shared with
Footprint and Inversion.Solver. PanelArrays.jl retains dimensioned surface
allocation and panel copies. LinRoodTape.jl remains untouched.

Reverse-pass entry points default to `AutoTape()`: supplied fields constrain
one concrete tape, sized and reserved before recording. An explicit
`AdjointTape` is reserved without replacement. A runtime (one per optimization)
caches CFL counts and pool reservations across evaluations; resident-tail
and pinned-host-snapshot Stride choices preserve the same reverse order.
