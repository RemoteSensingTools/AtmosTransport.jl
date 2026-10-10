# Split advection: paired seam transfers

> The reproducer scripts `check_outputs.jl`, `check_profile_outputs.jl`, `check_totals.jl`, `full_day.jl` and `profile.jl` were retired from the tree; recover with `git show 7c515038:scripts/benchmarks/results/main_split_mass_seams_v100_20260905/<file>`.

The paired physical-seam update removes the split schemes' Float64 tracer
imbalance and reduces full-day Float32 drift without normalizing totals.
The [implementation report](../../../../docs/memos/2026-09-05_main_split_mass_seams.md)
explains grouping, forward/adjoint consistency, independent field validation,
and remaining accuracy limitations.

## Full-day conservation

Six pressure-layer tracers, C90 L66, PPM, 24 hourly windows, TM5 convection and
exact Dkg diffusion, no emissions, `preserve_tracer_mass` air resets:

| Precision | Maximum absolute final relative drift before | After |
|---|---:|---:|
| Float32 | 2.2388955649787152e-5 | 8.168654720603056e-7 |
| Float64 | 2.2805808015179457e-5 | 9.914172177906698e-16 |

All initial compensated totals match exactly. Float32 improves about 27-fold;
Float64 reaches roundoff. The baseline hourly totals are retained in
[`../main_mass_seams_v100_20260905/ablation/`](../main_mass_seams_v100_20260905/ablation/).
New totals are `after/all.toml` and `after/all_float64.toml`.
`check_totals.jl` passes 52 hourly-total assertions and writes `drift_summary.csv`.
`check_outputs.jl` passes 64 completeness/finite-field checks against the remote
NetCDFs, each containing all 25 requested snapshots.

Positive initial layers still produce negative column means, down to about
-4.27e-11 mol/mol in the corrected PPM runs. Extrema are in `output_checks.txt`.
Conservation alone does not establish positivity, temporal order, or agreement
with a reference atmospheric simulation.

## Original 32-tracer performance workload

This repeats the [full-day launch-layout workload](../main_ppm_day_v100_20260905/README.md):
32 pressure layers, Float32, 255 transport substeps, column means and compensated
totals at hours 0 and 24, one warmup followed by two measured repetitions.

| Measure | Before paired seams | After |
|---|---:|---:|
| Median whole-run time | 32.598 s | 34.574 s |
| Measured time range | 32.596–32.599 s | 34.383–34.765 s |
| Cumulative host allocation | 6.397 GB | 6.419 GB |
| Maximum absolute final relative tracer drift | 2.565585e-5 | 8.203781e-7 |

The new exchange costs about 6.1% in these measurements and reduces maximum
drift about 31-fold. The seam cache adds 9,408,960 bytes of device workspace at
this resolution and tracer count. Peak host/device memory was not measured.
Convection retains its existing shared-memory batching and takes about 3.67 s.
`check_profile_outputs.jl` passes 432 checks of matching initial totals, finite
fields, snapshot completion, and final drift below 1e-6. The transported fields
are expected to change, so this check does not demand baseline bit equality.

The baseline is the earlier `b7e850a3` launch-layout export; split PPM arithmetic
remained unchanged through the immediate parent `bf9ae4cb`. Candidate sources
are the paired-seam change on `revamp/current-main`. Runs are sequential, with
warm caches and two measured samples per source, not an interleaved statistical
study. Results do not establish a universal runtime cost across archives or GPUs.

Profile TOMLs and timing CSVs are in `after/`; baseline files remain in the
earlier experiment directory. `final_totals` in profile TOMLs are ordinary state
reductions. Conservation uses the compensated NetCDF totals reported in
`profile_checks.txt`. Section timers may nest and include waits for prior GPU
work; their durations cannot be summed into wall time. CSV allocation columns
are unmeasured zeroes, and TOML host allocation is cumulative, not peak RAM.

## Independent field check

`solid_rotation.jl` constructs tilted solid-body rotation from a corner
streamfunction on a unit sphere. It compares a smooth Gaussian plus background
against analytic rotated values at cell centers after quarter and full
rotations, on C8/C16/C32 in both panel conventions. Flux divergence is near
roundoff. The baseline scalar palindrome is extracted from `bf9ae4cb` with
unchanged panel reconstruction kernels; this requires a Git checkout containing
that commit. Both methods use identical initialization, fluxes, and timesteps.

All 12 candidate field L2 errors are slightly lower than the corresponding
baseline; mass drift stays below 6.1e-15 versus up to 1.68e-3 before. Relative
L2 is area-weighted and normalized by the Gaussian component, excluding the
background. `solid_rotation.toml` contains all 24 rows; `check_totals.jl` passes
37 checks of those results. These use center samples rather than analytic
cell averages and do not prove universal order or monotonicity.

The earlier fixed-grid seam refinement experiment remains in
[`../main_mass_seams_v100_20260905/`](../main_mass_seams_v100_20260905/README.md).
Both original and grouped split PPM have first-order time refinement on that
fixture. The production adoption follows the additional field and adjoint
validation here; it does not claim second-order time accuracy from symmetry alone.

## Reproduction and environment

- tofu GPU 0: Tesla V100-PCIE-16GB,
  UUID `GPU-59ee7ef9-898f-524a-dde1-7a0426a41ecb`.
- Julia 1.12.6, CUDA.jl 5.11.3, CUDA runtime 12.6, four Julia threads and one
  OpenBLAS thread. GPU scalar indexing disabled.
- Same 3,114,263,552-byte ERA5 C90 L66 dry format-4, explicit-dm archive as the
  [preceding investigation](../main_mass_seams_v100_20260905/README.md).
  First interior level-33 air mass is 2.829608e12 kg in Float32 `(96,96,66)`
  panels. Surface pressure is 100869.164 Pa.
- The forcing uses experimental 1-degree three-hour convection held against
  hourly transport. This measures numerical behavior, not forcing fidelity.
- Each six-tracer layer targets 1e35 molecules at pressure fractions 0.2–0.9.
  Float32 uses collaborative TM5 LU; Float64 uses the legacy solve. Archive
  arrays remain Float32 in both cases.

The isolated candidate export is `/tmp/atmos-split-mass-seams` on tofu. Existing
CUDA environment pins were retained; no production checkout was changed.
Run GPU scripts from that configured export, with explicit device selection:

```bash
CUDA_VISIBLE_DEVICES=GPU-59ee7ef9-898f-524a-dde1-7a0426a41ecb \
JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
julia --startup-file=no --project=. /path/to/full_day.jl after

ATMOSTR_TIMERS=1 \
CUDA_VISIBLE_DEVICES=GPU-59ee7ef9-898f-524a-dde1-7a0426a41ecb \
JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
julia --startup-file=no --project=. /path/to/profile.jl after
```

`full_day.jl` writes `/tmp/atmos-split-drift-after/`; `ATMOSTR_MASS_INPUT` may
override its archive path. `profile.jl` writes `/tmp/atmos-split-day-profile-after/`.
The lightweight output validators run with `CUDA_VISIBLE_DEVICES=`; large
NetCDF files remain on tofu. Run `solid_rotation.jl` with GPUs hidden from a
candidate Git checkout and copy its `/tmp/atmos-cs-solid-rotation.toml` into this
directory before rechecking. `check_totals.jl` runs locally using stored TOMLs.

The opt-in maintained GPU test is `test/diagnostic/test_cs_seam_exchange_gpu.jl`:
set `ATMOSTR_RUN_CS_SEAMS_GPU_TESTS=1`, `ATMOSTR_CS_SEAMS_GPU_NAME=V100`, and the
explicit UUID above. It passes 2,552 forward checks through 65 tracers plus
24 seam-adjoint comparisons. The V100-enabled footprint test passes 91 checks,
including device, pinned-host, and mmap tape paths (`footprint_gpu_checks.txt`).
Other focused check logs are stored here; the implementation report records the
complete CPU suite and JET qualification.
