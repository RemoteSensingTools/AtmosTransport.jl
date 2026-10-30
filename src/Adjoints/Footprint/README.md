# Footprint

`AtmosTransport.Adjoints.Footprint` records and reverses the production
cubed-sphere step. Public footprint and tape-estimate names are re-exported
through Adjoints. Meteorology is fixed; only tracer mass is differentiated.

## File Map

- `Footprint.jl`: module imports, includes and exports.
- `FootprintResult.jl`: `CSFootprintResult` and `CSTapeByteEstimate`.
- `StepReplay.jl`: `CSStepSchedule`, resolved windows, replay setup, stepping,
  snapshots and restore. Clock arithmetic is shared with Models via StepPlans.
- `ReplaySupport.jl`: replay-specific extension of `require_adjoint_support`.
- `TapeRecorder.jl`: production recorder hooks and counting recorder.
- `TapeRecording.jl`: sweep records, recording and `cs_tape_byte_estimate`.
- `ReverseLoop.jl`: adjoint start states, reverse walk and objective forward replay.
- `StrideCheckpoint.jl`, `revolve_checkpoint.jl`: Full/Stride/Revolve drivers.
- `FootprintAPI.jl`: footprint and seeded-footprint entry points.

## Depends on

Adjoints core, Tape, Grids, State, MetDrivers, Parameters and Operators,
including StepPlans. External: KernelAbstractions. There is no dependency on
Inversion; observation-specific replay sampling belongs to Inversion.Observations.

## Used by

Inversion.Observations (sampled forward replay), Inversion.Solver
(objective footprints and forced reverse pass), Models (`CSStepSchedule`),
scripts and tests.

## Shared helpers owned here

`CSFootprintResult`, `CSTapeByteEstimate`, `CSStepSchedule`,
`run_cs_footprint_forward`, `cs_tape_byte_estimate`,
`cs_surface_emission_footprint`, `cs_surface_emission_footprint_from_seed`.
Inversion uses the replay accessors, call setup/validation, objective-footprint
batch and zero-start forced reverse driver listed below.

## Forward<->adjoint mirrors

Replay executes production StepPlan stages with production operators and
recorder hooks. Checkpoints save tracer/air-mass halos and persistent operator
state (emission Kahan compensation, convection/diffusion carries and caches).
Each sweep stores its entering air mass and, for limited schemes, tracer mass.
The reverse reads them with separate Tape roles. Every step ends with a
forcing marker; one emission record per step stores the surface footprint.
Full/Stride/Revolve preserve the existing arithmetic and restart behavior.
Lin–Rood accepts device tape storage only. Level Nz is the surface.

# shared: CSFootprintResult, _ZeroStart, _cs_objective_footprints, _cs_replay_for_call, _cs_surface_footprints, _replay_m, _replay_rm, _replay_steps!, _require_tape_path_supported, _require_tape_storage_supported, _validate_step_sequences

# shared: _validate_step_sequences
