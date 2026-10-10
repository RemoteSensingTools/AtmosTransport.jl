# Selected and streaming output on current main: V100

> The reproducer scripts `compare.jl` and `profile.jl` were retired from the tree; recover with `git show 7c515038:scripts/benchmarks/results/main_output_v100_20260905/<file>`.

This compares `77f36bbe` (the previous main-based runtime) with its selected
capture and streaming NetCDF port. The same C90 L66 Float32 ERA5 workload,
CUDA.jl 5.11.3 / CUDA runtime 12.6, Julia 1.12.6, four threads, and tofu GPU 0
(V100) were used. The input is the archived experimental hourly transport with
1-degree, three-hour convection fields described in the
[baseline](../main_real_input_v100_20260905/README.md).

`profile.jl` differs from the baseline script only in its output directory.
Each case runs two model hours (20 substeps), with column means requested at
hours 0 and 2. Sample 0 warms each tracer count; samples 1 and 2 are measured.
Run with `ATMOSTR_TIMERS=1`, an explicitly selected V100, and
`julia --startup-file=no --project=.` in the measured checkout. The export's
CUDA runtime must be set to 12.6 before launching a new Julia process.

## Measured results

Medians of the two warm repetitions; allocation units are decimal GB:

| Tracers | Measure | Before | After | Reduction |
|---|---|---:|---:|---:|
| 6 | Whole-run time | 4.149 s | 3.865 s | 6.8% |
| 6 | Total host allocation | 2.497 GB | 2.326 GB | 6.9% |
| 32 | Whole-run time | 16.396 s | 15.524 s | 5.3% |
| 32 | Total host allocation | 8.855 GB | 8.052 GB | 9.1% |
| 32 | Garbage collection | 2.665 s | 2.438 s | 8.5% |

These are short warm-cache measurements, not cold-NAS performance or statistical
confidence intervals. Allocated bytes are cumulative, not peak live memory.
Section times nest and overlap; their fractions cannot be added as wall-time
shares. Per-section allocation counters were not enabled.

Most whole-run allocation remains outside snapshot capture. The next targets
are startup loading and CPU workspace allocation before GPU adaptation.

## Retained snapshot storage

The C90 L66, 32-tracer full-frame field payload is 423.4 MB:
`90*90*66*6*(32+1)*sizeof(Float32)`. The equivalent column-only frame stores
12.83 MB of Float64 column reductions plus small metadata and totals. This is
an array-size calculation, not a process peak-memory measurement.

Single-file NetCDF output now appends and flushes each frame, retaining no
previous frames. Daily output retains selected frames for the current day and
at most one owned background write. The CPU stream regression checks that the
stream object's retained size does not grow as records are appended.

## Correctness

- All 196 written arrays across the four measured before/after file pairs are
  exactly equal, including the compensated Float64 totals: 280 comparison and
  conservation checks pass. Maximum relative tracer-total drift remains
  2.572e-6 over two model hours. See `compare.jl` and `comparison.txt`.
- V100 output tests pass 375 checks, including cancellation across reduction
  lanes and panels, Float32/Float64 inputs, selected layers, omitted layers,
  signed totals, and streamed payloads for all three topologies.
- The compact device result keeps each partial sum and correction separate
  until the compensated host reduction. CUDA does not transfer full tracer
  volumes solely to obtain accurate global totals. Metal retains a bounded
  slab-copy path for CPU Float64 accumulation; Metal hardware was not tested.
- CPU tests compare full/selected and batch/streamed output, preserve main's
  signed ATMSNAP totals, reject missing levels, and drain owned background
  writes on both successful and exceptional exits.

`gpu_checks_summary.txt` contains the device test summaries. The full initial
log also records harmless optional Git-provenance probe errors from the source
export; those probes were made quiet before the real-input profile.
