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
- `TapeRecorder.jl`: the recorder hooks, shared by the tape recorder
  (staging into the call's pool) and the counting recorder (the per-step
  byte count an mmap tape reserves before recording, and
  `cs_tape_byte_estimate`): what each record stages, which records share a
  slot (split sweeps: the palindrome center and the next Z sweep; a
  convection record and the next air mass with its interior), the limiter
  codes of limited PPM sweeps.
- `TapeRecording.jl`: recording a range of steps (`_record_cs_tape!`) and
  `cs_tape_byte_estimate`.
- `ReverseLoop.jl`: adjoint start states, reverse walk and objective forward replay.
- `TapeRuntime.jl`: the runtime of an `AdjointTape` for one call or one
  optimization (`_TapeRuntime`, `_with_tape_runtime`): its pool, opened
  once, rewound for each further tape and released explicitly; the counted
  mmap bytes (once per runtime); the `AirMassTape` recording it holds and
  its identity (`_TapeIdentity`, `_held_air_mass_tape`); explicit release of
  checkpoints and Lin–Rood records.
- `StrideCheckpoint.jl`: Full/Stride drivers in four phases on one replay
  (`_begin_taped_run`, the forward run `_run_forward!`, `_reverse_taped_run`,
  `_end_taped_run!`; Full records, or with `AirMassTape` reuses the tape its
  runtime holds), the 4D-Var's zero-start forced reverse
  (`_forced_reverse`), and the tape checks made before the forward run
  (`_require_tape_supported`: Lin–Rood storage, `AirMassTape` legality).
- `FootprintAPI.jl`: footprint and seeded-footprint entry points.

## Depends on

Adjoints core, Tape, Grids, State, MetDrivers, Parameters and Operators,
including StepPlans. External: KernelAbstractions. There is no dependency on
Inversion; observation-specific replay sampling belongs to Inversion.ObservationBinding.

## Used by

Inversion.ObservationBinding (sampled forward replay), Inversion.Solver
(objective footprints and forced reverse pass), Models (`CSStepSchedule`),
scripts and tests.

## Shared helpers owned here

`CSFootprintResult`, `CSTapeByteEstimate`, `CSStepSchedule`,
`run_cs_footprint_forward`, `cs_tape_byte_estimate`,
`cs_surface_emission_footprint`, `cs_surface_emission_footprint_from_seed`.
Inversion uses the replay accessors, call setup/validation, objective-footprint
batch, the tape runtime (one per evaluation, or one shared by an
optimization's evaluations), the taped forward run (which samples the
observations while it records) and the zero-start forced reverse driver
listed below.

## Forward<->adjoint mirrors

Replay executes production StepPlan stages with production operators and
recorder hooks. Checkpoints save tracer/air-mass halos and persistent operator
state (emission Kahan compensation, convection/diffusion carries and caches).
Each sweep stores its entering air mass and, for limited PPM, the limiter
branch code of every cell (computed by `record_ppm_limiter_codes!` from the
entering state; the FV3-profile sweeps of Lin–Rood store the tracer mass).
The reverse reads them with separate Tape roles. The center's air mass and a
convection call's are not staged again: their records share the slot of the
sweep (or `:preserve_vmr` reset) that stages the same air mass, and a
recording stages a convection air mass still waiting for a slot at its end
(`_finish_recording!`). With `AirMassTape` (linear schemes) the first forward
run of a runtime records the tape and later ones reuse it after comparing
their identity (grid, taped steps, windows, scheme and operators, met
checksums) and the pool generation with the recording's. Every step ends with a
forcing marker; one emission record per step stores the surface footprint.
Full and Stride give the same gradients bit for bit on one CPU thread
(`test/core/test_cs_adjoint_tape_bitwise.jl`). Each call opens one tape pool
from its `AdjointTape`, rewinds it before each re-recorded window and
releases it at the end (`cs_surface_flux_4dvar_optimize` keeps one pool for
all its evaluations); a storage that reserves its extent (`MmapCSTapeStorage`)
is sized by counting the largest tape on the replay first, once per runtime.
A call runs one forward run of its replay, which records what the schedule's
reverse pass needs (Full: the tape; Stride: the K-step checkpoints) through
its taped steps; the reverse pass may stop earlier (the 4D-Var's last
assimilated step, `_tape_through`) and runs from the state the forward run
ended in. Stride records each window once and walks it for every start. The
4D-Var samples its observations in that forward run, so one run gives the
simulated values and the tape. Lin–Rood accepts device tape storage only.
Level Nz is the surface.

# shared: CSFootprintResult, _TapeRuntime, _UntapedRun, _begin_taped_run, _cs_objective_footprints, _cs_replay_for_call, _end_taped_run!, _forced_reverse, _replay_m, _replay_nsteps, _replay_rm, _require_tape_supported, _run_forward!, _validate_step_sequences, _with_tape_runtime

# shared: _validate_step_sequences

The replay's transport stage uses StepPlans `transport_step!` and its concrete
`TransportMidpoint`. Advection sees only the abstract hook and current buffers.
Center/diffusion/emission records retain their order; ReverseLoop visits them
backwards, including both half solves for split coupling. Numerical transposes
beside the forward sweeps and the fenced Lin–Rood tape code are unchanged.
