# StepPlan

The model step as data: which operators run, in which order, with which time
step, at every advection step and at the met-window boundaries. One
description of the step serves production and the cubed-sphere adjoint
(owner decision D31): `TransportModel.step!` and `DrivenSimulation` execute
it, and the adjoint replays and records the same plan, so the forward model
it differentiates is the production step by construction.

## Entry Points

- `StepPlan(; physics_cadence = nothing, window_contract = false,
  air_mass_reset = :none)` — the plan of a run configuration, with four
  phases of stages:

  | phase | runs | stages |
  | --- | --- | --- |
  | `window_advance` | on entering a met window after the first | `AirMassResetStage(mode, WindowStartAirMass())` |
  | `window_start` | before the first step of each window | `ChemistryBeforeStage(WindowDt())` (`:window` cadence) |
  | `step` | every advection step | `TransportStage()`; with `:substep` cadence (or no window contract) also `ChemistryBeforeStage(StepDt())`, `ConvectionStage(StepDt())`, `ChemistryAfterStage(StepDt())` |
  | `window_end` | after the last step of each window | `AirMassResetStage(mode, WindowEndAirMass())` (window contract), then `ConvectionStage(WindowDt())`, `ChemistryAfterStage(WindowDt())` (`:window` cadence) |

- `StepOperators(; state, fluxes, grid, advection, advection_workspace, ...)`
  — what the stages act on (`step_operators(model)` for a `TransportModel`).
- `execute_stages!`, `execute_window_advance!`, `execute_window_start!`,
  `execute_step_stages!`, `execute_window_end!` — apply stages with a
  `StepTiming(dt, window_dt)`, the run context `meteo` (`current_time`, the
  air-mass reset targets through `air_mass_reset_target`) and a recorder
  (`NoStepRecorder()` in production).
- `operator_state`, `restore_operator_state!`, `reset_operator_state!` — the
  persistent state some operators carry between calls (surface-source Kahan
  compensation, convection and Dkg ledger carries, the CMFMC CFL cache), for
  the adjoint's checkpoints. Production never calls them.

## File Map

- [`StepPlan.jl`](StepPlan.jl) — module `Operators.StepPlans`: the cadence and
  reset vocabulary (`_resolve_physics_cadence`, `_normalize_air_mass_reset_mode`),
  stage types and `StepPlan`, `StepOperators`, the stage executors, the
  air-mass reset (`_reset_air_mass!`, `_refresh_state_halos!`), the
  cubed-sphere advection workspace constructor shared with the adjoint
  replay, and the persistent operator state.

## Common Tasks

- Changing what a run does in a step: change the plan constructor or a stage
  executor here; production and the adjoint follow. A new tracer-changing
  stage also needs an `adjoint_support` method, a recorder hook and a tape
  record (`../../Adjoints/AdjointSupport.jl`, `../StepRecording.jl`,
  `../../Tape/TapeRecords.jl`); until it has them the adjoint's setup guard
  rejects every plan containing it.
- A new operator with state that persists between calls: add
  `operator_state` / `restore_operator_state!` / `reset_operator_state!`
  methods, or adjoint checkpoint restarts will not reproduce the production
  trajectory (`test/core/test_cs_adjoint_step_plan_parity.jl` fails).

## Invariants

- The plan reproduces the pre-plan production order exactly; production
  results are bit-identical to the hard-coded composition it replaced.
- The plan is a pure function of the configuration: the stages of absolute
  step `s` follow from the window structure alone, so an adjoint checkpoint
  restart inside a met window knows that the window's convection (a
  `window_end` stage) has not run yet.
- Stages call the production operator entry points (`apply!`); a recorder
  only reads.

The checkpoint-state generic interface lives in `../OperatorState.jl`;
Convection owns its workspace snapshot methods in `../Convection/workspace_state.jl`.
