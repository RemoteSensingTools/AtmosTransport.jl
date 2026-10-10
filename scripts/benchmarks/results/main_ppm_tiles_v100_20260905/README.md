# CUDA PPM launch tiles on V100

> The reproducer scripts `compare.jl`, `profile.jl`, `sweep_blocks.jl`, `sweep_tiles.jl`, `sweep_trial.jl` and `trial_kernels.jl` were retired from the tree; recover with `git show 7c515038:scripts/benchmarks/results/main_ppm_tiles_v100_20260905/<file>`.

The cubed-sphere packed Float32 PPM sweeps now use a `(32, 2)` CUDA workgroup.
The per-cell kernel, tracer loop, air-mass update, reconstruction, and limiter
are unchanged. CPU, Metal, Float64 CUDA, and other schemes retain their launch
defaults. Performance measurements cover V100 only; other NVIDIA architectures
have not been benchmarked.

## Whole-run comparison

Baseline: [direct packing](../main_direct_packing_v100_20260905/README.md),
commit `9a6d14f2`. Both runs use the same experimental C90 L66 Float32 ERA5
format-4 archive, PPM, full-column TM5 matrix convection, exact Dkg diffusion,
and two model hours. See the [original baseline](../main_real_input_v100_20260905/README.md)
for the archive's experimental convection cadence and provenance.

Environment: Julia 1.12.6, four Julia threads, one OpenBLAS thread, CUDA.jl
5.11.3, CUDA runtime 12.6, tofu GPU 0 (Tesla V100-PCIE-16GB). Sample 0 warms
each tracer count; the table reports medians of samples 1 and 2.

| Tracers | Measure | Before | 32×2 tile |
|---|---|---:|---:|
| 6 | Whole-run time | 1.753 s | 1.591 s |
| 6 | Cumulative host allocation | 0.954 GB | 0.954 GB |
| 32 | Whole-run time | 5.030 s | 4.466 s |
| 32 | Cumulative host allocation | 1.808 GB | 1.808 GB |

The 32-tracer median improves 11.2%; the six-tracer median improves 9.2%,
with individual new samples spanning 1.427–1.755 s. These are short,
startup-heavy, warm-cache runs. Two samples do not establish a general speedup
across workloads. Host allocation is cumulative, not peak memory. CSV sections
nest and may wait for earlier asynchronous work; do not sum them or interpret
them as isolated GPU kernel times. Allocation columns are unmeasured zeroes.

All 196 saved output arrays remain exactly equal, including compensated
Float64 totals. All 280 comparison/conservation checks pass; maximum relative
mass drift remains 2.5718011894571768e-6. The TOML `final_totals` use ordinary
state reductions; conservation checks use the compensated NetCDF totals.

Reproduce from a configured source export containing the before/after code,
with the input path in `profile.jl` available:

```bash
ATMOSTR_TIMERS=1 ATMOSTR_PROFILE_GPU=0 \
CUDA_VISIBLE_DEVICES=GPU-59ee7ef9-898f-524a-dde1-7a0426a41ecb \
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4 \
julia --startup-file=no --project=. scripts/benchmarks/results/main_ppm_tiles_v100_20260905/profile.jl
```

`compare.jl` reads the before and after NetCDF files in the named `/tmp`
directories. Those large files are not committed. Configuration, raw profile
TOMLs, timing CSVs, and the comparison report are included here.

## Choosing the launch layout

The preceding synchronized profile (`synchronized_baseline_32.csv`) enabled
`ATMOSTR_PROFILE_GPU=1` at the direct-packing checkpoint. It separated sweep
execution from waits attributed to later halo exchanges. Its 240 panel calls
per direction averaged 4.107 ms (X), 1.879 ms (Y), and 2.241 ms (Z). Isolated
tracer halo exchanges averaged 1.62 ms, consistent with the earlier halo probe.
This diagnostic adds synchronization and is not the whole-run timing baseline.
It can be reproduced with the same workload and the profiling flag enabled at
the preceding commit, using a separate output directory.

KernelAbstractions 0.9.41 pads a scalar workgroup of 256 to `(256, 1, 1)` for
these three-dimensional launches. At C90, only 90 of each row's 256 threads
are active (35.2%). The `(32, 2)` tile rounds the first dimension to 96,
giving 93.75% active threads and contiguous `i` cells within each warp. These
are launch-padding fractions, not measured hardware occupancy.

The microbenchmarks read air mass and face fluxes from the archive's first
window and first panel, fill mass halos on the CPU, and construct smooth signed
tracer fields. The mass shape is `(96, 96, 66)`, Float32; first interior cell
mass at level 33 is about 2.829608e12 kg. Each launch reads fixed initial inputs
and writes an independent result. CUDA event timing follows an idle stream,
with one warmup and five measured samples per case. Medians from `sweep_tiles.csv`:

| Tracers | Direction | 256-thread row | 32×2 tile |
|---|---|---:|---:|
| 1 | X | 0.213 ms | 0.110 ms |
| 1 | Y | 0.153 ms | 0.081 ms |
| 1 | Z | 0.157 ms | 0.083 ms |
| 6 | X | 0.966 ms | 0.498 ms |
| 6 | Y | 0.441 ms | 0.223 ms |
| 6 | Z | 0.395 ms | 0.244 ms |
| 32 | X | 4.847 ms | 2.430 ms |
| 32 | Y | 1.683 ms | 0.969 ms |
| 32 | Z | 1.713 ms | 1.117 ms |

`sweep_trial.jl` tests a rejected fourth launch dimension assigning tracers to
independent threads; it was slower at both tested sizes, 128 and 256. For
example, 32-tracer X at 128 threads took 4.001 ms versus 2.533 ms with the
existing serial tracer loop. `trial_kernels.jl` contains that experimental
implementation only; production retains the original kernels.

`sweep_blocks.jl` compares scalar sizes 32/64/128/256; `sweep_tiles.jl`
compares 32, 32×2, 32×4, 16×8, 16×4, 8×8, and 256. The selected 32×2 tile is
near the best measured layout across directions and tracer counts. Those
experiments passed 54, 108, and 189 exact-equivalence/finite-value checks,
respectively. The scripts retain their measured algorithm and `/tmp` TOML
destinations; only the experimental-kernel include path is made relative here.
Raw measurement rows, including warmups, are transcribed losslessly into CSV.

## Regression checks

`test/diagnostic/test_cs_ppm_launch_gpu.jl` passes 540 checks on the V100:
Float32/Float64, panel widths 5/35, 1/6/7/32/65 tracers, three directions,
signed and zero storage, bidirectional fluxes, untouched inputs and halos,
ping-pong and copy-back wrappers, exact comparison to the original GPU launch,
and an independent CPU backend reference within floating-point tolerance.

Run it with `ATMOSTR_RUN_CS_PPM_LAUNCH_GPU_TESTS=1`,
`ATMOSTR_PPM_GPU_NAME=V100`, and the same authorized UUID above.
Focused CPU advection, Aqua, and JET checks pass 619 assertions; JET remains
142 against the unchanged allowance of 144. See `cpu_checks.txt` and
`gpu_checks.txt`. The preceding direct-packing checkpoint passed the complete
83,556-check CPU suite; that full suite was not repeated for this CUDA launch
policy change.
