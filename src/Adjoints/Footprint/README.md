# Footprint

`AtmosTransport.Adjoints.Footprint` records and reverses the production
cubed-sphere step. Public footprint and tape-estimate names are re-exported
through Adjoints. Meteorology is fixed; only tracer mass is differentiated.

## File Map

- `Footprint.jl`: module imports, includes and exports.
- `FootprintResult.jl`: `CSFootprintResult` and `CSTapeByteEstimate`.
- `WindowSource.jl`: the met windows the replay reads. The interface
  (`AbstractCSWindowSource`, `AbstractStreamedCSWindowSource`,
  `CSWindowMet`, `CSWindowLease` with its device slot and fill count,
  `window_count`, `window_steps`, `window_layout`, `acquire_window!`,
  `release_window!`, `prefetch_window!` (a hint; no-op while loads are
  synchronous), `require_current` / `StaleWindowSlotError`, `close`,
  `streaming`), and two sources:
  - `ResidentCSWindowSource` groups the per-step inputs of the entry points by
    met window once: one flux state per window (one per step when the fluxes
    change within the window), its convection forcing, diffusion operator,
    layer thickness and reset targets, each validated once (constant within
    the window; shapes, element types and backend). Consecutive steps given
    the same flux arrays (a met window's, as `cs_adjoint_inputs` shares them)
    share one flux state, so `flux_scale ≠ 1` scales each set once.
    Releasing a resident window does nothing.
  - `HostCSWindowSource` keeps every window in host memory (`CSWindowMet`s
    of host arrays, from `cs_adjoint_inputs(sim; residency = :host)`) and
    `device_slots` windows on the replay's backend. `acquire_window!` returns
    the slot holding the window or refills the least recently used slot
    without a lease, in place (one `copyto!` per array; `generation` counts
    the fills); `release_window!` synchronizes the backend when a slot's last
    lease ends (the version-1 fence). Every window must have the first
    window's layout (piece types, array shapes), checked with its diffusion
    operator when the source is built, and its host bytes may be bounded by
    `host_budget_bytes`. Pieces are copied through their `Adapt` methods
    (`_host_copy`, a backend copy, `_array_leaves`), so a fill copies exactly
    the arrays a model's move to the GPU would.
- `StepReplay.jl`: `CSStepSchedule`, resolved windows, replay setup, stepping,
  snapshots and restore. Clock arithmetic is shared with Models via StepPlans.
  The replay holds its window source and the windows it has acquired
  (`leases`); every met input of a step is read through the step's window
  (`_load_step!`, `_replay_operators`, the recorder's layer thickness). An
  untaped forward run (and a counting run, whose layouts keep array
  templates only) releases the windows it acquired when it acquires the next
  and hints the next window to the source; a recording run retains the
  windows it reads, which the drivers release after the records
  (`_release_replay_windows!`, also in their `finally` blocks). The replay is
  built on the caller's `met_source` or on a resident source of the per-step
  inputs (`_window_source`); a given source is checked against the run
  without acquiring a window (`_check_window_steps`, its `window_layout`
  against the replay's diffusion workspace and initial air mass), and the
  replay meteorology's reset-target fields take the layout's types. A plan
  that resets the air mass needs the targets in the schedule or in the
  source's windows (`_check_reset_targets`); a met source excludes schedule
  targets. The caller passes the diffusion layer thickness per step
  (`diffusion_layer_thickness`, read only); the replay owns its diffusion
  workspace and one reverse-solve scratch (`diffusion_scratch`), which the
  reverse walk passes to every diffusion record.
- `ReplaySupport.jl`: replay-specific extension of `require_adjoint_support`:
  one operator set per distinct diffusion operator of the windows, or, for a
  streamed source, of its window layout (no window is acquired).
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
- `AutoTape.jl`: closed-form binary schedule sizes, cached CFL/Lin–Rood
  counts (`_TapeSizing` also keeps the met window of every sized step, from
  which `MetWindowCheckpoint`'s segments are sized; with a streamed source a
  Lin–Rood tape is sized from window 1 and the longest later window by step
  class, `_class_tape_size`, falling back to counting every step when a
  class is missing), constrained candidate
  selection (a concrete schedule, `StrideSearch`), memory budgets,
  reserve-before-run setup and preallocated snapshot capture. Convection owns the lazy carry
  shapes through `convection_checkpoint_state`.
  CPU budgets subtract `min(10% of host/cgroup capacity, 2 GiB)` plus reverse
  workspace from available headroom; `Tape._CPU_TAPE_MARGIN_MAX_BYTES` owns
  the cap. Tape and snapshots share that budget. GPU margins are unchanged
  (device: `max(10% of capacity, 4 GiB)`; pinned host:
  `max(10% of limit, 8 GiB)`). This changes candidate selection when memory is
  tight and preserves bitwise forward/adjoint results on one CPU thread.
- `TapeRuntime.jl`: the runtime of an `AdjointTape` for one call or one
  optimization (`_TapeRuntime`, `_with_tape_runtime`): its pool, opened
  once, rewound for each further tape and released explicitly; the counted
  mmap bytes (once per runtime); the `AirMassTape` recording it holds and
  its identity (`_TapeIdentity`, `_held_air_mass_tape`); explicit release of
  checkpoints and Lin–Rood records.
- `StrideCheckpoint.jl`: Full and segmented drivers in four phases on one
  replay (`_begin_taped_run`, the forward run `_run_forward!`,
  `_reverse_taped_run`, `_end_taped_run!`; Full records, or with
  `AirMassTape` reuses the tape its runtime holds; `StrideCheckpoint` and
  `MetWindowCheckpoint` share one driver, `_StrideTapedRun`, which resolves
  the segment list of its taped steps once, `_run_segments`: K-step windows,
  or the met windows of `replay.steps.window`), the 4D-Var's zero-start
  forced reverse
  (`_forced_reverse`), and the tape checks made before the forward run
  (`_require_tape_supported`: Lin–Rood storage, `AirMassTape` legality, and
  with a streamed source `MetWindowCheckpoint` with `StateTape` only). The
  segmented driver checks that the windows a re-recorded segment references
  are current (`_require_current_windows`) before it walks the segment, and
  hints the previous segment's window to the source.
- `FootprintAPI.jl`: footprint and seeded-footprint entry points; each takes
  `met_source` in place of the per-step met arguments (`_call_nsteps`,
  `_require_source_keywords` in `ReverseLoop.jl`).

## Depends on

Architectures, Adjoints core, Tape, Grids, State, MetDrivers, Parameters and Operators,
including StepPlans. External: KernelAbstractions. There is no dependency on
Inversion; observation-specific replay sampling belongs to Inversion.ObservationBinding.

## Used by

Inversion.ObservationBinding (sampled forward replay), Inversion.Solver
(objective footprints and forced reverse pass), Models (`CSStepSchedule`),
scripts and tests.

## Shared helpers owned here

`CSFootprintResult`, `CSTapeByteEstimate`, `CSStepSchedule`,
`run_cs_footprint_forward`, `cs_tape_byte_estimate`,
`cs_surface_emission_footprint`, `cs_surface_emission_footprint_from_seed`,
`AbstractCSWindowSource`, `HostCSWindowSource`, `StaleWindowSlotError`.
Inversion uses the replay accessors, call setup/validation (including the
step count of a call's met, `_call_nsteps`), the release of the replay's met
windows, objective-footprint batch, the tape runtime (one per evaluation, or one shared by an
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
checksums) and the pool generation with the recording's. Checksums weight bit
patterns by cell position; device reductions feed one result vector, copied
to the host once per evaluation (reset targets cover interiors only). Holding
a tape also rejects tracer payloads in sweep or Lin–Rood horizontal records.
Device code slots use air-mass shapes directly; only host staging needs code
scratch in the reserved limiter read cache, released with the pool. With `flux_scale ≠ 1`, a held tape
retains the first evaluation's scaled flux copies; later evaluations also
allocate their own copies, so peak scaled-flux memory is twice one set.
Every step ends with a
forcing marker; one emission record per step stores the surface footprint.
Full and Stride give the same gradients bit for bit on one CPU thread
(`test/core/test_cs_adjoint_tape_bitwise.jl`). Each call defaults to `AutoTape()`, resolves a concrete `AdjointTape` and
reserves its tape and checkpoint pools before recording. Explicit descriptors
also reserve and are never overridden. The driver rewinds the tape pool
before each re-recorded window and releases the pools at the end (`cs_surface_flux_4dvar_optimize` keeps one pool for
all its evaluations). Binary schedules use the closed-form size; CFL
schedules and Lin–Rood use one cached counting replay per runtime.
Segment sizing omits a leading convection flush inherited from the preceding
step: a fresh segment recorder has no pending convection there. It includes
the final pending convection slot. Head pools reserve the maximum count for
each type and shape across their windows; a resident tail reserves its exact
segment counts.
A call runs one forward run of its replay, which records what the schedule's
reverse pass needs (Full: the tape; Stride: the K-step checkpoints) through
its taped steps; the reverse pass may stop earlier (the 4D-Var's last
assimilated step, `_tape_through`) and runs from the state the forward run
ended in. Stride records each window once and walks it for every start. The
4D-Var samples its observations in that forward run, so one run gives the
simulated values and the tape. Lin–Rood accepts device tape storage only.
Stride may record a resident tail during the forward run and recompute
only the head. Its snapshots' tracer and air masses may use pinned host
storage; operator state retains backend types. `MetWindowCheckpoint`
checkpoints at met-window boundaries (optionally split into pieces of at most
`split_steps`) with the Stride driver, no tail and pinned-host snapshots by
default; unconstrained `AutoTape()` does not consider it,
`AutoTape(; checkpoint = MetWindowCheckpoint(...))` sizes and reserves it.
With a streamed met source (`HostCSWindowSource`) it is the only schedule
accepted (plain, split, or as the `AutoTape` constraint), and streamed
results equal the resident inputs' bit for bit on one CPU thread
(`test/core/test_cs_window_source.jl`).
Full, plain Stride, resident tails, host snapshots and MetWindow (plain,
split, AutoTape-constrained) are checked against recorded digests and the
reference variant by the CPU bitwise harness.
Lin–Rood's horizontal records are counted toward the device peak and remain
outside the pools. Level Nz is the surface.

# shared: CSFootprintResult, _TapeRuntime, _UntapedRun, _begin_taped_run, _call_nsteps, _cs_objective_footprints, _cs_replay_for_call, _end_taped_run!, _forced_reverse, _release_replay_windows!, _replay_m, _replay_nsteps, _replay_rm, _require_tape_supported, _run_forward!, _validate_step_sequences, _with_tape_runtime

Models.Runner's host capture (`cs_adjoint_inputs(sim; residency = :host)`)
uses the host copies, window met, freeing and budget of `WindowSource.jl`:

# shared: _host_copy, _host_window_met, _met_pieces, _free_met_arrays!, _new_array_bytes!, _host_window_budget, _require_host_budget

# shared: _validate_step_sequences

The replay's transport stage uses StepPlans `transport_step!` and its concrete
`TransportMidpoint`. Advection sees only the abstract hook and current buffers.
Center/diffusion/emission records retain their order; ReverseLoop visits them
backwards, including both half solves for split coupling. Numerical transposes
beside the forward sweeps and the fenced Lin–Rood tape code are unchanged.

Non-fenced reverse-start and air-mass adjoint zero allocation uses
Architectures.zeros_like, preserving allocation/fill order and element types.


# shared: _auto_memory_budgets
