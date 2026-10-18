# Footprint

Surface-emission footprints for the cubed sphere: replay the production step
plan with a tape recorder, then walk the tape backwards.

This folder turns a final-time objective, an explicit final adjoint seed, or
a set of observations sampled at earlier steps, into `dJ/dE_t`: the
sensitivity to the surface emission rate the surface-flux operator applies at
the palindrome center of each model step, and `dJ/drm_0`, the sensitivity to
the initial tracer mass. The forward model is the production `StepPlan`
(`../Operators/StepPlan/`), replayed with the production operators on a
single-tracer state; its tape is written by the recorder hooks of the
production step and walked in reverse with the kernels in
[`../Adjoints/`](../Adjoints/), adding observation forcing at end-of-step
markers. It also runs the strided and bisection checkpoint schedules.

## How It Is Loaded

`Footprint/` is **not a module**. [`../Adjoints/Adjoints.jl`](../Adjoints/Adjoints.jl)
`include`s these files into `module Adjoints`, so every name here is an
`Adjoints` name, and public ones are exported from the `Adjoints.jl` export
block. Include order:
`FootprintResult.jl` and `StepForcing.jl` (early, right after
`ObjectiveSeeding.jl`), then, after all `Adjoints` kernel files and
`LinRoodTape.jl`: `StepReplay.jl` -> (`../Adjoints/AdjointSupport.jl`) ->
`TapeRecorder.jl` -> `TapeRecording.jl` -> `ReverseLoop.jl` ->
`StrideCheckpoint.jl` -> `revolve_checkpoint.jl` -> `FootprintAPI.jl`.
Storage types, record types, and schedules come from [`../Tape/`](../Tape/).

## Entry Points

- [`FootprintAPI.jl`](FootprintAPI.jl):
  - `cs_surface_emission_footprint(panels_rm0, panels_m0, panels_am_steps,
    panels_bm_steps, panels_cm_steps, mesh, objective; scheme, dt, schedule,
    flux_scale, base_emission_rates, diffusion_op, ..., convection_op, ...,
    tape_storage, tape_path, checkpoint)` returns a `CSFootprintResult`.
  - `cs_surface_emission_footprint_from_seed(final_adjoint_rm, panels_m0, ...;
    surface_flux = true)` is the same pass with a caller-supplied
    `dJ/drm_final`. `base_panels_rm0` sets the tracer state the tape is
    recorded around (zero by default; it only matters for limited schemes).
    `surface_flux = false` replays a configuration without surface sources
    (center `V(dt)`): no footprints, the initial-condition sensitivity in
    `initial_adjoint`.
  - `run_cs_footprint_forward(...)` is forward only and returns the objective
    value. Use it for finite-difference checks.
  - `_cs_surface_footprints(tape_rm0, panels_m0, ..., starts; scheme,
    checkpoint, schedule, nsteps, ..., forcing)` is the one internal
    checkpoint dispatch behind both entry points, `cs_surface_flux_jacobian`
    (several objectives on one recording) and `cs_surface_flux_4dvar` (a zero
    start plus the observation forcing). It returns one `CSFootprintResult`
    per adjoint start.
- [`StepReplay.jl`](StepReplay.jl): `CSStepSchedule` — the production step the
  adjoint differentiates (the `StepPlan`, steps per met window, window length,
  start time, binary subcycle contract, air-mass reset targets);
  `CSStepSchedule(sim)` in `../Models/adjoint_schedule.jl` builds it for a
  `DrivenSimulation`, and `cs_adjoint_inputs(sim; tracer)` there captures every
  adjoint input of a driven configuration from production (per-step face
  fluxes after window loading and flux scaling, per-window convection forcing,
  diffusion layer thickness and Kz, the tracer's static emission rates, fresh
  workspaces; nonzero initial compensation/carries or a valid CMFMC CFL
  cache are rejected). Schedule construction validates production operators,
  retains chemistry, and places reset targets on the model backend.
- [`StepForcing.jl`](StepForcing.jl): `CSLinearFunctional` (an observation as
  `offset + Σ weight · rm_step[panel][i, j, k]`, each entry at one step: one
  step for an instantaneous observation, several for a time window; built by
  `_linear_functional` from the replayed air mass of a step and assembled per
  observation by `_observe_cs_forward`), `CSStepForcing`
  (scaled functionals grouped by step and panel, on the device),
  `NoStepForcing`, `_assemble_step_forcing` and `_inject_step_forcing!`.
- [`TapeRecording.jl`](TapeRecording.jl): `cs_tape_byte_estimate(...)` returns
  a `CSTapeByteEstimate`: it replays the transport with a counting recorder
  and counts the records of the plan's convection and diffusion.
- [`FootprintResult.jl`](FootprintResult.jl): `CSFootprintResult` (with
  `footprints[t]::NTuple{6}` of `(Nc, Nc)` arrays, `lag_steps[t] == nsteps -
  t` and `initial_adjoint`) and `CSTapeByteEstimate`.

## File Map

- [`FootprintResult.jl`](FootprintResult.jl) — result and tape-size types
- [`StepForcing.jl`](StepForcing.jl) — observation operators as linear
  functionals, the step forcing of the reverse walk and its injection kernel
  (one work item per cell, entries in a fixed order, no atomics)
- [`StepReplay.jl`](StepReplay.jl) — `CSStepSchedule`, the replay meteorology
  (`_CSReplayMeteo`: clock, binary subcycle contract, reset targets), the
  replay `_CSReplay` (single-tracer `CubedSphereState`, per-step face fluxes,
  the production operators and a refillable surface source), `_build_cs_replay`,
  `_replay_steps!` (executes each step's plan phases in production order and
  calls the recorder), `_replay_snapshot` / `_replay_restore!` (checkpoints)
- [`TapeRecorder.jl`](TapeRecorder.jl) — `_CSTapeRecorder`, the recorder hook
  methods that stage inputs and push records (sweeps, halos, center, diffusion,
  emission, Lin-Rood phases, convection with its sub-step count, air-mass
  resets, end-of-step markers), and `_CSCountingRecorder` for the estimate
- [`TapeRecording.jl`](TapeRecording.jl) — `_record_sweep!`,
  `_center_diffusion_records`, `_record_cs_tape!`, `_record_cs_adjoint_tape`
  (a whole run, for tests), `cs_tape_byte_estimate`
- [`ReverseLoop.jl`](ReverseLoop.jl) — the adjoint starts (`_ObjectiveStart`,
  `_SeedStart`, `_ZeroStart`, `_initial_adjoint_state`),
  `_walk_window_reverse!` (dispatch on record type; step forcing at
  `_CSStepEndRecord`), the air-mass reset transpose, `_cs_replay_for_call`
  (input validation, the guard, the replay), and the forward replays
  `_run_cs_footprint_forward` and `_observe_cs_forward` (each observation
  sampled after every step of its `CSTimeSampling`: simulated values,
  stencils and functionals; placed point observations choose their layer from
  `column_met` at each sampled state; `_run_cs_observations_forward` wraps it)
- [`StrideCheckpoint.jl`](StrideCheckpoint.jl) — `FullCheckpoint` and
  `StrideCheckpoint` drivers (`_collect_footprints`), the shared reverse-state
  helpers, and the `_require_checkpoint_supported`, tape-storage and
  `tape_path` guards; [`revolve_checkpoint.jl`](revolve_checkpoint.jl) — the
  `RevolveCheckpoint` driver. One driver per schedule serves every scheme
  family.
- [`FootprintAPI.jl`](FootprintAPI.jl) — public entry points;
  `_cs_objective_footprints` (several objectives, used by the Jacobian);
  `_cs_surface_footprints`, the one input validation and dispatch to Full,
  Stride, or Revolve

## Common Tasks

- Changing what production runs in a step: change the `StepPlan`
  (`../Operators/StepPlan/`) or an operator; the replay follows. If the new
  piece transforms the tracer, add a recorder hook to production, a record
  and its transpose (`../Tape/TapeRecords.jl`, `_walk_window_reverse!`), a
  `_CSTapeRecorder` method, and an `adjoint_support` method;
  `../../test/core/test_cs_adjoint_step_plan_parity.jl` checks replay ==
  production bit for bit.
- Adding a tape record type: add the record in `../Tape/TapeRecords.jl`,
  then add a branch in `_walk_window_reverse!`.
- Adding an observation operator: give its objective a `_linear_functional`
  method in [`StepForcing.jl`](StepForcing.jl) (weights from the air mass at
  the sampled step) and an `evaluate_objective` method; the 4D-Var gradient
  and the final-time seed then use it unchanged. An operator that needs more
  than the air mass at the sampled state (as `CSPlacedPointObjective` needs
  the column meteorology) adds a `_sample_objective` method in
  `../Inversion/ObservationColumns.jl` instead and is a 4D-Var observation
  only.
- Changing schedule behaviour: [`StrideCheckpoint.jl`](StrideCheckpoint.jl),
  [`revolve_checkpoint.jl`](revolve_checkpoint.jl). Propagation passes run
  the replay with `NoStepRecorder()` (the production call path); every
  propagation and recording starts from a restored snapshot.
- Sizing a run before launching it: `cs_tape_byte_estimate`.

## Invariants

- The forward model is the production step plan of the schedule
  (`CSStepSchedule`; default one step per met window, the full operator suite
  every step, as `TransportModel.step!`). The replay runs the production
  stage functions and operators (packed-tracer palindrome, the
  Kahan-compensated surface source, convection and diffusion with their
  ledger carries and the CMFMC CFL cache) on a single-tracer
  `CubedSphereState`, so the forward replay, the tape recording and every
  checkpoint restart equal production bit for bit.
- Every entry point runs `require_adjoint_support(replay)` before recording:
  every plan stage and operator needs an `adjoint_support` method returning
  `nothing`.
- Per-step tape order for split-sweep schemes: `halo(1) (X Y Z)^n <center>
  (Z Y X)^n`, then the plan's convection (every step, or after the window's
  last step and the window-end reset), then `end(step)`. `n` is the
  production subcycle count (CFL budget, or 1 under a binary window
  contract). Lin-Rood steps record `H Z <center> Z H` (its step-start fills
  are never read and have no record). Z is the scheme's vertical sweep.
- Every step closes with `_CSStepEndRecord(step)` (absolute step, after the
  window-end stages): the state that runtime output and the forward
  observation sampler see. The reverse walk adds the step forcing there,
  before it transposes the step's last operator, as TM5-4DVAR adds its forcing
  just before the adjoint of the operator after which it sampled. With
  `NoStepForcing` the marker does nothing.
- An observation operator is a fixed linear functional of the tracer mass at
  the steps it samples, with weights from the replayed air mass (meteorology
  is a fixed tape input) times its time-sampling weights. Its gradient is its
  weight vector. The 4D-Var gradient starts from zero (`_ZeroStart`) and adds
  every observation's scaled weights at each sampled step: one tape (to the
  last sampled step of the same schedule) and one reverse walk per gradient.
- The injection kernel gives each cell of a (step, panel) pair one work
  item, which adds the cell's entries in input order: deterministic, no
  atomics. (The halo and Lin-Rood face adjoint kernels do use atomics, so a
  multithreaded CPU or GPU reverse pass is reproducible to rounding only,
  about 1e-22 of the largest gradient entry between two identical calls;
  single-threaded it is bit for bit. Tests compare gradients with
  `gradients_match` in `test/helpers/cs_observation_columns.jl`: `==`
  single-threaded, 1e-13 relative otherwise.)
- `<center>` is production's `palindrome_center!`: `D(dt/2) | emission |
  D(dt/2)` with `SplitSurfaceFluxCoupling`, `emission | D(dt)` with
  `DiffusiveSurfaceFluxBoundary`, `emission` alone without diffusion, and
  `D(dt)` without a surface-flux operator. Footprints and 4D-Var always have
  a surface-flux operator (zero rates allowed).
- Convection records carry the interval they integrate (the step or the met
  window), the forcing they read and the CFL sub-step count of the forward
  call; the reverse transposes the same sub-steps.
- Footprint units: `E` is a per-cell emission rate in model-storage units,
  i.e. dry mixing ratio × carrier air mass, per second. It is not species
  kg/s and not per unit area. The footprint already includes the `dt`
  factor.
- Air mass and fluxes are fixed tape inputs. The Lin-Rood branch computes a
  `lambda_m` for each step and then discards it.
- The column ledger carries of convection and Dkg diffusion and the surface
  source's Kahan compensation are part of the forward (so the replay equals
  production bit for bit) but are rounding-level terms the reverse does not
  transpose: each reverse kernel is the transpose of the exact operator.
- The replay resets the caller's convection workspace to a fresh run's
  persistent state and uses the diffusion workspaces' `factors` as scratch:
  pass fresh workspaces, not a live production model's.
- Diffusion records carry the clock of the forward solve; the reverse
  evaluates a time-varying Kz field at that clock. The replay clock is
  production's (`CSStepSchedule.start_time` plus the elapsed windows) unless
  `diffusion_meteo` is given.
- Monotone PPM and the FV3 vertical profile record the tracer branch state
  of the base trajectory. Pass `base_emission_rates` when you differentiate
  around non-zero emissions.
- `tape_path` requires `tape_storage = :mmap`. Stride writes
  `window_NNNNN/` directories and Revolve writes `step_NNNNN/` directories.
  Stride and Revolve reject a pre-built storage object. With
  `FullCheckpoint`, a pre-built storage object is used as is, and the
  caller is responsible for `finalize_tape!`.
- `FT` is `eltype(panels_rm0[1])` (or `panels_m0` for the seed entry point).
  Footprint arrays stay on the input backend.

## Related Docs And Tests

- [`../../docs/src/for_tm5_gchp_users/adjoints.md`](../../docs/src/for_tm5_gchp_users/adjoints.md),
  [`../../docs/src/theory/adjoint_status.md`](../../docs/src/theory/adjoint_status.md)
- [`../../test/core/test_cs_adjoint_step_plan_parity.jl`](../../test/core/test_cs_adjoint_step_plan_parity.jl):
  the D31 parity matrix — replay, tape recording and checkpoint restarts ==
  production `DrivenSimulation` bit for bit across schemes, cadences,
  convection, diffusion couplings, no-emission diffusion and reset modes;
  OF-114 and OF-115; the setup guard; gradients vs. production finite
  differences
- [`../../test/core/test_cs_ppm_adjoint_footprint.jl`](../../test/core/test_cs_ppm_adjoint_footprint.jl):
  footprint vs. finite differences for every scheme, diffusion, TM5, CMFMC,
  seed entry, GPU smoke test (including limited-PPM `:pinned_host` and
  `:mmap` tapes against `:device`)
- [`../../test/core/test_cs_adjoint_production_parity.jl`](../../test/core/test_cs_adjoint_production_parity.jl):
  replay vs. `TransportModel.step!` for both surface-flux couplings with TM5
  and CMFMC convection, footprints vs. production finite differences,
  checkpoint parity with convection, tape layout vs. `cs_tape_byte_estimate`,
  and distinct air-mass / tracer-mass reads from host-side tape storage
- [`../../test/core/test_cs_linrood_footprint_fd.jl`](../../test/core/test_cs_linrood_footprint_fd.jl):
  Lin-Rood footprint vs. finite differences with upward vertical flux
- [`../../test/core/test_cs_linrood_fv3_adjoint.jl`](../../test/core/test_cs_linrood_fv3_adjoint.jl):
  the FV3 vertical sweep's transpose and Lin-Rood footprints with it vs.
  finite differences, checkpoint parity
- [`../../test/core/test_cs_4dvar_single_sweep.jl`](../../test/core/test_cs_4dvar_single_sweep.jl):
  the single-sweep 4D-Var gradient vs. one footprint per observation for
  every scheme family, diffusion, TM5 and CMFMC, and the three schedules;
  one recording and one reverse pass per gradient; the reused-tape Jacobian;
  the step-forcing injection
- [`../../test/core/test_cs_stride_checkpoint.jl`](../../test/core/test_cs_stride_checkpoint.jl),
  [`../../test/core/test_cs_tape_path.jl`](../../test/core/test_cs_tape_path.jl),
  [`../../test/core/test_cs_tape_mmap_roundtrip.jl`](../../test/core/test_cs_tape_mmap_roundtrip.jl)
- Opt-in GPU reset parity:
  [`../../test/diagnostic/test_cs_adjoint_reset_gpu.jl`](../../test/diagnostic/test_cs_adjoint_reset_gpu.jl)
  checks captured `:preserve_vmr` targets, untaped/taped replay and checkpoint
  gradients on CUDA (load CUDA before AtmosTransport in the root project).
- Opt-in GPU and Lin-Rood checks:
  [`../../test/diagnostic/test_cs_transport_adjoint_gpu.jl`](../../test/diagnostic/test_cs_transport_adjoint_gpu.jl),
  [`../../test/diagnostic/test_linrood_adjoint_integration.jl`](../../test/diagnostic/test_linrood_adjoint_integration.jl)
- Example callers: [`../../scripts/diagnostics/plot_cs_ppm_adjoint_footprint.jl`](../../scripts/diagnostics/plot_cs_ppm_adjoint_footprint.jl),
  [`../../scripts/diagnostics/linrood_la_footprint.jl`](../../scripts/diagnostics/linrood_la_footprint.jl)
