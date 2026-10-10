# Full-day PPM launch comparison and initialization profile

> The reproducer scripts `compare.jl`, `initialization_phases.jl` and `profile.jl` were retired from the tree; recover with `git show 7c515038:scripts/benchmarks/results/main_ppm_day_v100_20260905/<file>`.

This follow-up extends the [two-hour V100 experiment](../main_ppm_tiles_v100_20260905/README.md)
to all 24 windows of the same C90 L66 Float32 ERA5 format-4 archive, with
32 pressure-layer tracers, PPM advection, full-column TM5 matrix convection,
and exact Dkg diffusion. Column means and compensated totals are saved at
hours 0 and 24. The run uses 255 transport substeps because the existing CFL
schedule selects 10 or 11 substeps per window.

The source exports are `9a6d14f2` (before) and `b7e850a3` (after), with identical
dependency environments and CUDA runtime preferences. Their only source
differences are the PPM launch policy, its six wrapper call sites, and the
advection README. Staging is disabled. The archive's experimental convection
cadence and provenance are described in the [original input baseline](../main_real_input_v100_20260905/README.md).
This experiment evaluates performance and before/after equivalence, not that
forcing's scientific validity.

## Results

| Measure, 32 tracers and 24 model hours | Before | 32×2 tile |
|---|---:|---:|
| Median whole-run time | 41.404 s | 32.598 s |
| Cumulative host allocation | 6.397 GB | 6.397 GB |

The measured median improves 21.3%. Before samples take 41.292–41.517 s;
after samples take 32.596–32.599 s. All 150 output arrays are exactly equal
across the two measured repetitions, and all 368 comparison checks pass.
Maximum relative compensated-total drift is 2.565584820714352e-5 (0.002566%),
identical before and after. This exceeds the two-hour workload's drift of
2.5718011894571768e-6; equivalence to the baseline does not establish that this
accumulated full-day error meets a particular scientific accuracy requirement.

A subsequent [seam investigation](../main_mass_seams_v100_20260905/README.md)
finds drift in Float64 as well and isolates inconsistent horizontal tracer
exchange. The launch-layout speedup remains valid, but baseline equivalence
must not be read as evidence of tracer conservation.

The median advection section falls from 34.127 to 25.240 s; it includes
midpoint physics and waiting at synchronization boundaries. Matrix convection
remains about 3.675 s per day. The short-run launch improvement therefore
persists when transport contributes a larger share of whole-run time.

## Method

Julia 1.12.6, four Julia threads, one OpenBLAS thread, CUDA.jl 5.11.3 / runtime
12.6, tofu GPU 0 (Tesla V100-PCIE-16GB). Each export runs sample 0 for warmup,
then samples 1 and 2 for measurement. The before export completes all samples
before the after export starts. These warm-cache runs include initialization,
transport, and output; they are longer than the original workload but are not
a pure steady-state kernel benchmark. Two repetitions do not establish a
general speedup across grids or hardware.

Run `profile.jl before` from the configured before export, then `profile.jl
after` from the configured after export:

```bash
ATMOSTR_TIMERS=1 ATMOSTR_PROFILE_GPU=0 \
CUDA_VISIBLE_DEVICES=GPU-59ee7ef9-898f-524a-dde1-7a0426a41ecb \
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4 \
julia --startup-file=no --project=. /path/to/profile.jl before
```

The copied GPU test environment needed the current package's `FileWatching`
stdlib dependency added to its project. It was already present in the
manifest; the resolved package versions and CUDA runtime were retained. No
production checkout on tofu was modified.

Raw profile TOMLs, configurations, and timing CSVs are in `before/` and
`after/`; large NetCDF outputs remain in the `/tmp` directories named by the
scripts. `compare.jl` requires exact equality and finite values for every
output array and reports the full-day compensated-total drift. It checks that
the drift matches the baseline, rather than applying the shorter experiment's
absolute threshold to a different duration. This comparison does not replace
the existing two-hour conservation checks or establish a full-day error budget.

CSV section times can nest, overlap, and wait for earlier queued GPU work.
They cannot be summed as a partition of wall time. Allocation columns are
unmeasured zeroes; profile TOML allocation is cumulative host allocation, not
peak RAM or device memory. TOML `final_totals` are ordinary state reductions;
conservation reporting uses the compensated NetCDF totals.

## Remaining initialization cost

`initialization_phases.jl` is a separate CPU probe on wurst (Intel Xeon
Platinum 8462Y+), using the first window of the same archive. It probes the
Float32 `(96,96,66)` air-mass panels, whose first interior cell at level 33
contains 2.829608e12 kg; corresponding surface pressure is 100869.164 Pa.
It allocates final packed storage outside the timers, then separately measures
each tracer's interior VMR builder and conversion into its final mass slot.
There is one warmup and five measured repetitions per tracer count, with
garbage collection before each whole initialization. Julia and thread settings
match the GPU experiment, with GPUs hidden. These CPU measurements are not
timings of initialization on tofu.

| Tracers | Phase | Median time | Cumulative host allocation |
|---|---|---:|---:|
| 6 | Build interior VMR | 0.053022 s | 79,326,112 B |
| 6 | Pack mass into final slots | 0.018505 s | 0 B |
| 32 | Build interior VMR | 0.329920 s | 423,073,120 B |
| 32 | Pack mass into final slots | 0.126607 s | 0 B |

Accurate host reduction of the final tracer gives 1.0000000304144257e35
molecules for the configured 1e35, reflecting Float32 initialization rounding.
The remaining temporary storage belongs to VMR construction and target-layer
selection. Packing into existing storage allocates no temporary memory in this
probe. No additional initialization optimization is included in this follow-up.
