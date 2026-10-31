# Footprint

`AtmosTransport.Adjoints.Footprint` records and reverses the production
cubed-sphere step. Public footprint and tape-estimate names are re-exported
through Adjoints. Meteorology is fixed; only tracer mass is differentiated.

## File Map

- `Footprint.jl`: module imports, includes and exports.
- `FootprintResult.jl`: `CSFootprintResult` and `CSTapeByteEstimate`.
- `StepReplay.jl`: `CSStepSchedule`, resolved windows, replay setup, stepping,
  snapshots and restore. Clock arithmetic is shared with Models via StepPlans.
  Consecutive steps given the same flux arrays (a met window's, as
  `cs_adjoint_inputs` shares them) share one flux state, so `flux_scale ≠ 1`
  scales each set once.
- `ReplaySupport.jl`: replay-specific extension of `require_adjoint_support`.
- `TapeRecorder.jl`: production recorder hooks and counting recorder (also
  the per-step byte count an mmap tape reserves before recording).
- `TapeRecording.jl`: sweep records, recording and `cs_tape_byte_estimate`.
- `ReverseLoop.jl`: adjoint start states, reverse walk and objective forward replay.
- `StrideCheckpoint.jl`, `revolve_checkpoint.jl`: Full/Stride/Revolve drivers
  in four phases on one replay (`_begin_taped_run`, the forward run
  `_run_forward!`, `_reverse_taped_run`, `_end_taped_run!`), the 4D-Var's
  zero-start forced reverse (`_forced_reverse`), and the tape-storage checks
  made before the forward run.
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
batch, the taped forward run (which samples the observations while it records)
and the zero-start forced reverse driver listed below.

## Forward<->adjoint mirrors

Replay executes production StepPlan stages with production operators and
recorder hooks. Checkpoints save tracer/air-mass halos and persistent operator
state (emission Kahan compensation, convection/diffusion carries and caches).
Each sweep stores its entering air mass and, for limited schemes, tracer mass.
The reverse reads them with separate Tape roles. Every step ends with a
forcing marker; one emission record per step stores the surface footprint.
Full/Stride/Revolve give the same gradients bit for bit on one CPU thread
(`test/core/test_cs_adjoint_tape_bitwise.jl`). Each call opens one tape
storage, rewinds it before each re-recorded window and releases it at the
end; a storage that reserves its extent (`MmapCSTapeStorage`) is sized by
counting the largest tape on the replay first. A call runs one forward run of
its replay, which records what the schedule's reverse pass needs (Full: the
tape; Stride: the K-step checkpoints; Revolve: the initial state) through its
taped steps; the reverse pass may stop earlier (the 4D-Var's last assimilated
step, `_tape_through`) and runs from the state the forward run ended in. The
4D-Var samples its observations in that forward run, so one run gives the
simulated values and the tape. Lin–Rood accepts device tape storage only.
Level Nz is the surface.

# shared: CSFootprintResult, _UntapedRun, _begin_taped_run, _cs_objective_footprints, _cs_replay_for_call, _end_taped_run!, _forced_reverse, _replay_m, _replay_nsteps, _replay_rm, _require_tape_storage_supported, _run_forward!, _validate_step_sequences

# shared: _validate_step_sequences

The replay's transport stage uses StepPlans `transport_step!` and its concrete
`TransportMidpoint`. Advection sees only the abstract hook and current buffers.
Center/diffusion/emission records retain their order; ReverseLoop visits them
backwards, including both half solves for split coupling. Numerical transposes
beside the forward sweeps and the fenced Lin–Rood tape code are unchanged.
