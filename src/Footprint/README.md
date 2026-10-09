# Footprint

Surface-emission footprints for the cubed sphere: record the forward run on
a tape, then walk it backwards.

This folder turns a final-time objective, or an explicit final adjoint seed,
into `dJ/dE_t`: the sensitivity to the surface emission rate at the midpoint
of each earlier model step. It records the forward palindrome on a tape and
walks the tape in reverse using the kernels in [`../Adjoints/`](../Adjoints/).
It also runs the strided and bisection checkpoint schedules.

## How It Is Loaded

`Footprint/` is **not a module**. [`../Adjoints/Adjoints.jl`](../Adjoints/Adjoints.jl)
`include`s these files into `module Adjoints`, so every name here is an
`Adjoints` name, and public ones are exported from the `Adjoints.jl` export
block. Include order:
`FootprintResult.jl` (early, right after `ObjectiveSeeding.jl`), then, after
all `Adjoints` kernel files and `LinRoodTape.jl`: `TapeRecording.jl` ->
`ReverseLoop.jl` -> `StrideCheckpoint.jl` -> `FootprintAPI.jl`. Storage types,
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
- [`TapeRecording.jl`](TapeRecording.jl): `cs_tape_byte_estimate(...)` returns
  a `CSTapeByteEstimate` without recording anything. It counts the split-sweep
  record layout (tested for the linear and monotone-PPM schemes); Lin-Rood
  records hold more panel states.
- [`FootprintResult.jl`](FootprintResult.jl): `CSFootprintResult` (with
  `footprints[t]::NTuple{6}` of `(Nc, Nc)` arrays and
  `lag_steps[t] == nsteps - t`) and `CSTapeByteEstimate`.

## File Map

- [`FootprintResult.jl`](FootprintResult.jl) — result and tape-size types
- [`TapeRecording.jl`](TapeRecording.jl) — `_tape_byte_estimate`,
  `_record_sweep!`, `_record_cs_mass_tape` (linear schemes: stages `m`
  only), `_record_cs_tracer_tape` (monotone PPM: stages `m` and `rm`), and
  the `_record_cs_adjoint_tape` dispatcher (Lin-Rood goes to
  `_record_cs_linrood_tape`)
- [`ReverseLoop.jl`](ReverseLoop.jl) — `_collect_surface_footprints`,
  `_walk_window_reverse!` (dispatch on record type), and the tape-free
  forward replays `_run_cs_footprint_forward` and
  `_run_cs_observations_forward`
- [`StrideCheckpoint.jl`](StrideCheckpoint.jl) — `StrideCheckpoint` and
  `RevolveCheckpoint` drivers for the mass, tracer, and Lin-Rood tapes
  (`_collect_surface_footprints_stride`,
  `_collect_surface_footprints_revolve`, `_propagate_*_checkpoints`), plus
  the `_require_checkpoint_supported` and `_require_tape_path_supported`
  guards
- [`FootprintAPI.jl`](FootprintAPI.jl) — public entry points; input
  validation; dispatch to Full, Stride, or Revolve

## Common Tasks

- Changing what the tape records for a step: edit the recorder for that
  scheme family in [`TapeRecording.jl`](TapeRecording.jl), or
  `_record_cs_linrood_tape` in `../Adjoints/LinRoodTape.jl`. Keep the
  forward replay in [`ReverseLoop.jl`](ReverseLoop.jl) consistent with it.
- Adding a tape record type: add the record in `../Tape/TapeRecords.jl`,
  then add a branch in `_walk_window_reverse!`.
- Changing schedule behaviour: [`StrideCheckpoint.jl`](StrideCheckpoint.jl).
  The forward propagation passes call the same recorders with
  `record_ops = false`, and `step_offset` shifts midpoint indices for each
  window.
- Sizing a run before launching it: `cs_tape_byte_estimate`.

## Invariants

- Per-step tape order for split-sweep schemes:
  `X(n) Y(n) Z(n) [D(dt/2) | midpoint | D(dt/2)] Z(n) Y(n) X(n)`, then
  optional convection. `n` is the subcycle count from
  `_cs_static_palindrome_subcycle_count`, the production CFL schedule.
  Lin-Rood steps record `H Z [midpoint] Z H`. There is no subcycling
  (`cfl_limit` is ignored), and Z is `UpwindScheme`.
- In the split-sweep tapes, the forward halo fills after initialization are
  recorded as `_CSHaloRecord(dir)` (1 = X, 2 = Y): one after each horizontal
  sweep, and one before each of the second-half Y and X blocks (Lin-Rood keeps
  its halo work inside `_CSLinRoodHorizRecord`). The reverse pass applies
  `_adjoint_fill_panel_halos!` with the same `dir`.
- The reverse walk reads the per-step footprint when it reaches each
  `_CSMidpointRecord`. Emissions are injected at the palindrome midpoint,
  between the two diffusion half-steps, as in the runtime surface-source
  path.
- Footprint units: `E` is a per-cell emission rate in model-storage units,
  i.e. dry mixing ratio × carrier air mass, per second. It is not species
  kg/s and not per unit area. The footprint already includes the `dt`
  factor.
- Air mass and fluxes are fixed tape inputs. The Lin-Rood branch computes a
  `lambda_m` for each step and then discards it.
- Monotone PPM records the tracer branch state of the base trajectory. Pass
  `base_emission_rates` when you differentiate around non-zero emissions.
- `StrideCheckpoint` replays the same kernels as `FullCheckpoint`, plus halo
  refreshes at segment boundaries. Tests require agreement within
  `atol = 1e-12` for every scheme family. `RevolveCheckpoint` can drift by about 1e-7 for monotone
  PPM combined with diffusion or convection (see
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
  seed entry, GPU smoke test
- [`../../test/core/test_cs_stride_checkpoint.jl`](../../test/core/test_cs_stride_checkpoint.jl),
  [`../../test/core/test_cs_tape_path.jl`](../../test/core/test_cs_tape_path.jl),
  [`../../test/core/test_cs_tape_mmap_roundtrip.jl`](../../test/core/test_cs_tape_mmap_roundtrip.jl)
- Opt-in GPU and Lin-Rood checks:
  [`../../test/diagnostic/test_cs_transport_adjoint_gpu.jl`](../../test/diagnostic/test_cs_transport_adjoint_gpu.jl),
  [`../../test/diagnostic/test_linrood_adjoint_integration.jl`](../../test/diagnostic/test_linrood_adjoint_integration.jl)
- Example callers: [`../../scripts/diagnostics/plot_cs_ppm_adjoint_footprint.jl`](../../scripts/diagnostics/plot_cs_ppm_adjoint_footprint.jl),
  [`../../scripts/diagnostics/linrood_la_footprint.jl`](../../scripts/diagnostics/linrood_la_footprint.jl)
