# Footprint

Surface-emission footprints for the cubed sphere: record the forward run on
a tape, then walk it backwards.

This folder turns a final-time objective, an explicit final adjoint seed, or
a set of observations sampled at earlier steps, into `dJ/dE_t`: the
sensitivity to the surface emission rate at the midpoint of each model step.
It records the forward palindrome on a tape and walks the tape in reverse
using the kernels in [`../Adjoints/`](../Adjoints/), adding observation
forcing at end-of-step markers. It also runs the strided and bisection
checkpoint schedules.

## How It Is Loaded

`Footprint/` is **not a module**. [`../Adjoints/Adjoints.jl`](../Adjoints/Adjoints.jl)
`include`s these files into `module Adjoints`, so every name here is an
`Adjoints` name, and public ones are exported from the `Adjoints.jl` export
block. Include order:
`FootprintResult.jl` and `StepForcing.jl` (early, right after
`ObjectiveSeeding.jl`), then, after
all `Adjoints` kernel files and `LinRoodTape.jl`: `TapeRecording.jl` ->
`ReverseLoop.jl` -> `StrideCheckpoint.jl` -> `stride_checkpoint_ppm.jl` ->
`stride_checkpoint_linrood.jl` -> `revolve_checkpoint.jl` -> `FootprintAPI.jl`. Storage types,
record types, and schedules come from [`../Tape/`](../Tape/).

## Entry Points

- [`FootprintAPI.jl`](FootprintAPI.jl):
  - `cs_surface_emission_footprint(panels_rm0, panels_m0, panels_am_steps,
    panels_bm_steps, panels_cm_steps, mesh, objective; scheme, dt,
    flux_scale, cfl_limit, base_emission_rates, diffusion_op, ...,
    convection_op, ..., tape_storage, tape_path, checkpoint)` returns a
    `CSFootprintResult`.
  - `cs_surface_emission_footprint_from_seed(final_adjoint_rm, panels_m0, ...)`
    is the same pass with a caller-supplied `dJ/drm_final`. `base_panels_rm0`
    sets the tracer state the tape is recorded around. It defaults to zero
    and only matters for limited schemes (monotone PPM, Lin-Rood).
  - `run_cs_footprint_forward(...)` is forward only and returns the objective
    value. Use it for finite-difference checks.
  - `_cs_surface_footprints(tape_rm0, panels_m0, ..., starts; scheme,
    checkpoint, ..., forcing)` is the one internal checkpoint dispatch behind
    both entry points, `cs_surface_flux_jacobian` (several objectives on one
    tape) and `cs_surface_flux_4dvar` (a zero start plus the observation
    forcing). It returns one `CSFootprintResult` per adjoint start.
- [`StepForcing.jl`](StepForcing.jl): `CSLinearFunctional` (an observation as
  `offset + Σ weight · rm_step[panel][i, j, k]`, each entry at one step: one
  step for an instantaneous observation, several for a time window; built by
  `_linear_functional` from the replayed air mass of a step and assembled per
  observation by `_observe_cs_forward`), `CSStepForcing`
  (scaled functionals grouped by step and panel, on the device),
  `NoStepForcing`, `_assemble_step_forcing` and `_inject_step_forcing!`.
- [`TapeRecording.jl`](TapeRecording.jl): `cs_tape_byte_estimate(...)` returns
  a `CSTapeByteEstimate` without recording anything. It counts both the
  split-sweep layouts (linear and limited PPM) and Lin-Rood's larger layout;
  each Lin-Rood step retains ten (upwind vertical) or twelve (FV3 vertical)
  full-panel states and six face-panel tuples before optional diffusion and
  convection records.
- [`FootprintResult.jl`](FootprintResult.jl): `CSFootprintResult` (with
  `footprints[t]::NTuple{6}` of `(Nc, Nc)` arrays and
  `lag_steps[t] == nsteps - t`) and `CSTapeByteEstimate`.

## File Map

- [`FootprintResult.jl`](FootprintResult.jl) — result and tape-size types
- [`StepForcing.jl`](StepForcing.jl) — observation operators as linear
  functionals, the step forcing of the reverse walk and its injection kernel
  (one work item per cell, entries in a fixed order, no atomics)
- [`TapeRecording.jl`](TapeRecording.jl) — `_tape_byte_estimate`,
  `_record_sweep!`, `_record_and_sweep_z!` (records, then applies, one Z
  sweep; used by all three tapes, including Lin-Rood), the per-step pieces
  the recorders share (`_record_step_start_halos!`,
  `_record_and_apply_center!` / `_record_center!` around
  `surface_flux_center!`, `_record_and_apply_convection!`,
  `_record_step_end!`),
  `_record_cs_mass_tape` (linear schemes: stages `m`
  only), `_record_cs_tracer_tape` (monotone PPM: stages `m` and `rm`; the
  FV3 vertical sweeps of the Lin-Rood tape stage both as well), and
  the `_record_cs_adjoint_tape` dispatcher (Lin-Rood goes to
  `_record_cs_linrood_tape`)
- [`ReverseLoop.jl`](ReverseLoop.jl) — the adjoint starts (`_ObjectiveStart`,
  `_SeedStart`, `_ZeroStart`, `_initial_adjoint_state`),
  `_collect_surface_footprints`, `_walk_window_reverse!` (dispatch on record
  type; step forcing at `_CSStepEndRecord`), `_replay_cs_step!` (one
  production step: step-start halo refresh, palindrome with the coupled
  center, convection), and the tape-free forward replays
  `_run_cs_footprint_forward` and `_observe_cs_forward` (each observation
  sampled after every step of its `CSTimeSampling`: simulated values,
  stencils and functionals; placed point observations choose their layer from
  `column_met` at each sampled state; `_run_cs_observations_forward` wraps it)
  built on it
- [`StrideCheckpoint.jl`](StrideCheckpoint.jl) (linear mass tape and the
  shared helpers), [`stride_checkpoint_ppm.jl`](stride_checkpoint_ppm.jl)
  (monotone-PPM tracer tape), [`stride_checkpoint_linrood.jl`](stride_checkpoint_linrood.jl)
  (Lin-Rood tape), [`revolve_checkpoint.jl`](revolve_checkpoint.jl) — the
  `StrideCheckpoint` and `RevolveCheckpoint` drivers for the mass, tracer, and Lin-Rood tapes
  (`_collect_surface_footprints_stride`,
  `_collect_surface_footprints_revolve`, `_propagate_*_checkpoints`), plus
  the `_require_checkpoint_supported` and `_require_tape_path_supported`
  guards
- [`FootprintAPI.jl`](FootprintAPI.jl) — public entry points;
  `_cs_objective_footprints` (several objectives, used by the Jacobian);
  `_cs_surface_footprints`, the one input validation and dispatch to Full,
  Stride, or Revolve (`_collect_footprints`)

## Common Tasks

- Changing what the tape records for a step: edit the recorder for that
  scheme family in [`TapeRecording.jl`](TapeRecording.jl), or
  `_record_cs_linrood_tape` in `../Adjoints/LinRoodTape.jl`. Keep the
  forward replay `_replay_cs_step!` in [`ReverseLoop.jl`](ReverseLoop.jl)
  and production `strang_split!` consistent with it;
  `../../test/core/test_cs_adjoint_production_parity.jl` compares the replay
  with `TransportModel.step!`, the footprints with production finite
  differences, and the tape layout with `cs_tape_byte_estimate`.
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
- Changing schedule behaviour: [`StrideCheckpoint.jl`](StrideCheckpoint.jl) and the
  scheme-specific `stride_checkpoint_*.jl`, `revolve_checkpoint.jl`.
  The forward propagation passes call the same recorders with
  `record_ops = false`, and `step_offset` shifts midpoint indices for each
  window.
- Sizing a run before launching it: `cs_tape_byte_estimate`.

## Invariants

- The replays and tapes follow the production step of a run with a
  surface-flux operator (production `strang_split!` followed by the
  `TransportModel.step!` convection block). Per-step tape order for
  split-sweep schemes: `halo(1) (X Y Z)^n <center> (Z Y X)^n`, then optional
  convection, then `end(step)`. `n` is the subcycle count from
  `_cs_static_palindrome_subcycle_count`, the production CFL schedule.
  Lin-Rood steps record `H Z <center> Z H`, then optional convection and
  `end(step)`. There is no subcycling (`cfl_limit` is ignored), and Z is the
  scheme's vertical sweep (upwind or the FV3 profile).
- Every recorder closes each step with `_CSStepEndRecord(step)` (absolute
  step, after convection): the state that runtime output and the forward
  observation sampler see. The reverse walk adds the step forcing there,
  before it transposes the step's convection, as TM5-4DVAR adds its forcing
  just before the adjoint of the operator after which it sampled. With
  `NoStepForcing` (every footprint path) the marker does nothing, so
  footprints are bit-identical to a tape without it.
- An observation operator is a fixed linear functional of the tracer mass at
  the steps it samples, with weights from the replayed air mass (meteorology
  is a fixed tape input) times its time-sampling weights. Its gradient is its
  weight vector. `_seed_objective!` adds the functional at the final air mass
  to a zeroed adjoint state, which reproduces the earlier seeds bit for bit.
  The 4D-Var gradient starts from zero (`_ZeroStart`) and adds every
  observation's scaled weights at each sampled step: one tape (to the last
  sampled step) and one reverse walk per gradient. A placed point
  observation's level is resolved per sampled step from the replayed air
  mass, so its stencil may change from step to step; the forcing carries the
  stencil of the forward run.
- The injection kernel gives each cell of a (step, panel) pair one work
  item, which adds the cell's entries in input order. The injection is
  deterministic and needs no atomics. (The halo and Lin-Rood face adjoint
  kernels do use atomics, so a multithreaded CPU or GPU reverse pass is
  reproducible to rounding only; single-threaded it is bit for bit.)
- `_seed_objective!` computes the seed weights on the host (`1 / max(m, eps)`,
  `1 / max(Σm, eps)`) where the old seed kernels divided on the device; IEEE division
  gives the same bits on CPU and CUDA (not checked on Metal).
- `<center>` is `surface_flux_center!` with the diffusion operator's
  coupling: `D(dt/2) | midpoint | D(dt/2)` with `SplitSurfaceFluxCoupling`,
  `midpoint | D(dt)` with `DiffusiveSurfaceFluxBoundary` (one full-step
  diffusion record after the midpoint), and just `midpoint` without
  diffusion.
- In the split-sweep tapes, the forward halo fills after initialization are
  recorded as `_CSHaloRecord(dir)` (1 = X, 2 = Y): the step-start refresh
  (`fill_cs_step_start_halos!`, edges plus X-orientation corners, as production; the first X sweep would
  otherwise read tracer halos from before the previous step's convection),
  one after each horizontal sweep, one before the X sweep of each later
  forward pass (the Z sweep before it changed the interiors), and one before
  each second-half Y and X sweep (Lin-Rood keeps its halo work inside
  `_CSLinRoodHorizRecord`; its horizontal operator refills the halos it
  reads, so it has no step-start record). The reverse pass applies
  `_adjoint_fill_panel_halos!` with the same `dir`.
- The reverse walk reads the per-step footprint when it reaches each
  `_CSMidpointRecord`. Emissions are injected where the production center
  injects them: between the two diffusion half-steps (split coupling) or
  before the full-step solve (diffusive boundary).
- Footprint units: `E` is a per-cell emission rate in model-storage units,
  i.e. dry mixing ratio × carrier air mass, per second. It is not species
  kg/s and not per unit area. The footprint already includes the `dt`
  factor.
- Air mass and fluxes are fixed tape inputs. The Lin-Rood branch computes a
  `lambda_m` for each step and then discards it.
- Monotone PPM and the FV3 vertical profile record the tracer branch state
  of the base trajectory. Pass `base_emission_rates` when you differentiate
  around non-zero emissions.
- `StrideCheckpoint` and `RevolveCheckpoint` replay the same kernels as
  `FullCheckpoint`; because every step starts with the step-start halo
  refresh, a window restarted at a step boundary records the full tape's
  states. Tests require agreement within `atol = 1e-12` for every scheme
  family, with diffusion and convection (see
  `../Tape/CheckpointSchedule.jl`).
- `tape_path` requires `tape_storage = :mmap`. Stride writes
  `window_NNNNN/` directories and Revolve writes `step_NNNNN/` directories.
  Stride and Revolve reject a pre-built storage object. With
  `FullCheckpoint`, a pre-built storage object is used as is, and the
  caller is responsible for `finalize_tape!`.
- `FT` is `eltype(panels_rm0[1])` (or `panels_m0` for the seed entry point).
  Footprint arrays stay on the input backend.

## Related Docs And Tests

- [`../../docs/src/getting_started/adjoints.md`](../../docs/src/getting_started/adjoints.md),
  [`../../docs/src/theory/adjoint_status.md`](../../docs/src/theory/adjoint_status.md)
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
- Opt-in GPU and Lin-Rood checks:
  [`../../test/diagnostic/test_cs_transport_adjoint_gpu.jl`](../../test/diagnostic/test_cs_transport_adjoint_gpu.jl),
  [`../../test/diagnostic/test_linrood_adjoint_integration.jl`](../../test/diagnostic/test_linrood_adjoint_integration.jl)
- Example callers: [`../../scripts/diagnostics/plot_cs_ppm_adjoint_footprint.jl`](../../scripts/diagnostics/plot_cs_ppm_adjoint_footprint.jl),
  [`../../scripts/diagnostics/linrood_la_footprint.jl`](../../scripts/diagnostics/linrood_la_footprint.jl)
