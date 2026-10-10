# Parallel CUDA tracer solves for conservative Dkg

> The reproducer scripts `check_outputs.jl`, `launch_layout.jl`, `parallel_tracers.jl`, `profile.jl` and `small_batches.jl` were retired from the tree; recover with `git show 7c515038:scripts/benchmarks/results/main_dkg_parallel_v100_20260906/<file>`.

CUDA now factors each column once, then distributes independent tracer solves
across threads. The arithmetic and factor storage are unchanged. The original
32-tracer full-day workload improves from 38.635 to 29.543 s median (23.5%),
with identical output arrays and no new persistent workspace.

## Selection measurements

`launch_layout.jl` compares seven two-dimensional thread layouts. `parallel_tracers.jl`
then compares fused column loops against separate factor/solve kernels; the
latter use the same production column helpers. `small_batches.jl` verifies the
choice for two, three, and four tracers. Each uses one warmup and seven GPU-event
samples, excludes input resets from timing, and checks exact output equality.
The final production wrapper retains the fused kernel for one tracer.

| Precision / tracers | Original 8×8 fused call | Separate factors + 32×1×2 tracer solve |
|---|---:|---:|
| Float32 / 2 | 0.511 ms | 0.294 ms |
| Float32 / 6 | 1.430 ms | 0.360 ms |
| Float32 / 32 | 7.690 ms | 1.214 ms |
| Float32 / 65 | 14.856 ms | 2.255 ms |
| Float64 / 6 | 1.980 ms | 0.468 ms |
| Float64 / 32 | 10.798 ms | 1.723 ms |
| Float64 / 65 | 21.046 ms | 3.628 ms |

Measurements use one actual C90 L66 meteorological panel, fixed initial tracer
layers, and dt=360 s. Its halo-padded air array is `(96,96,66)` Float32 with
level-33 first-interior mass 2.829608e12 kg; `(90,90,66)` Dkg peaks at
1.207936e11 kg/s on that panel. Float64 casts the same archive values. This
kernel microbenchmark does not estimate whole-run speedup or hardware occupancy.
All raw samples and scripts are archived beside this file.

## Full-day production comparison

The baseline is `cee9fed3`, the conservative diffusion fix. Candidate sources
change the CUDA launch decomposition and add two kernels; the conservative
column helper, adjoint, and state/workspace layout remain unchanged.
The production workload uses 32 tracers, C90 L66, PPM, TM5 convection,
precomputed Dkg, 24 hourly windows, and 255 transport substeps. It has no
emissions and preserves tracer mass on air-mass resets.

| Measure | Fused baseline | Parallel CUDA |
|---|---:|---:|
| Median whole-run time | 38.6349346815 s | 29.5433961335 s |
| Measured range | 38.549–38.721 s | 29.323–29.764 s |
| Median cumulative host allocation | 6.413687048 GB | 6.414270756 GB |
| Maximum final compensated-total drift | 3.138139946426091e-7 | identical |

One warmup precedes two measured repetitions. The NetCDF comparison passes
454 assertions, including exact equality of all 150 arrays across both samples,
finite output, and completion of both requested snapshots. No normalization,
limiter, weak-exchange cutoff, or lower precision is introduced. The earlier
1e-7 exploratory daily drift target remains unmet. These are sequential
warm-cache runs, not an interleaved statistical study. Peak memory was not
measured here; cumulative allocation and nested section timings have their
usual limitations described in the preceding experiment.

## Validation and implementation review

- V100: 36 whole-state comparisons to the old serial launch and 6,989,800
  independent diffusion/transpose/weak-exchange assertions through 65 tracers.
- CPU: 12,087 diffusion assertions, Aqua 10, and JET 152 against unchanged 152.
- Strict documentation build and 160 local documentation/link/map checks pass.

Critical Codex self-review checked that the factor and solve launches use the
same caller stream, each factor element has one writer, each tracer element
has one solver, and factors remain read-only during all tracer solves. The
existing panel synchronization completes both launches. Scalar and one-tracer
paths retain a fused solve; CPU and Metal keep the prior fused column loop.
No additional agent was used. The full CPU suite at `cee9fed3` predates this
CUDA launch change; focused CPU and device checks above validate this update.

## Reproduction and environment

Only tofu GPU 0 was used: Tesla V100-PCIE-16GB,
UUID `GPU-59ee7ef9-898f-524a-dde1-7a0426a41ecb`. Julia 1.12.6,
CUDA.jl 5.11.3, runtime 12.6; four Julia threads, one OpenBLAS thread,
GPU scalar indexing disabled. Preserve the runtime pin when using V100.
The input is the same 3,114,263,552-byte dry format-4, explicit-dm ERA5 archive
as the [conservative diffusion experiment](../main_dkg_mass_v100_20260905/README.md).
Experimental three-hour convection held against hourly transport means these
are numerical/performance tests, not forcing validation.

The isolated candidate export is `/tmp/atmos-dkg-parallel` on tofu; the baseline
is `/tmp/atmos-conservative-dkg`. Run `profile.jl after` in the configured candidate:

```bash
ATMOSTR_TIMERS=1 \
CUDA_VISIBLE_DEVICES=GPU-59ee7ef9-898f-524a-dde1-7a0426a41ecb \
JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
julia --startup-file=no --project=. /path/to/profile.jl after
```

Results go to `/tmp/atmos-dkg-parallel-day-after/`. Run `check_outputs.jl` with
`CUDA_VISIBLE_DEVICES=`; it also needs baseline NetCDFs in
`/tmp/atmos-dkg-day-profile-after/`. Large NetCDFs stay on tofu; `after/` contains
small TOMLs and CSVs. GPU regression command:

```bash
ATMOSTR_RUN_DKG_GPU_TESTS=1 ATMOSTR_DKG_GPU_NAME=V100 \
CUDA_VISIBLE_DEVICES=GPU-59ee7ef9-898f-524a-dde1-7a0426a41ecb \
JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
julia --startup-file=no --project=. test/diagnostic/test_conservative_dkg_gpu.jl
```

Performance on other CUDA architectures and Metal was not measured.
