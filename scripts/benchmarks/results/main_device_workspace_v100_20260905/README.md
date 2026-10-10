# Construct cubed-sphere workspaces on the V100

> The reproducer scripts `compare.jl`, `construction.jl` and `profile.jl` were retired from the tree; recover with `git show 7c515038:scripts/benchmarks/results/main_device_workspace_v100_20260905/<file>`.

The runner transfers state and fluxes before constructing `TransportModel`.
Workspace constructors follow the state backend through `similar`, so they no
longer allocate temporary host scratch and then copy its unused contents to
the GPU. Final adaptation still transfers operator/geometry metadata.

Measurements use Julia 1.12.6, four threads, CUDA.jl 5.11.3 / runtime 12.6,
and tofu GPU 0 (Tesla V100-PCIE-16GB). This follows the
[input checkpoint](../main_input_v100_20260905/README.md).

## Isolated model construction

`construction.jl` compares both construction orders in one process, alternates
the order each repetition, synchronizes CUDA inside the timed region, and
collects/reclaims before each measurement. Sample 0 warms both paths; samples
1–5 are measured. It uses the real C90 L66 grid and physics recipe with
synthetic, already allocated host state, so initial-condition creation, input
loading, stepping, and output are excluded. Device memory is still allocated
in both paths; the bytes below are cumulative Julia host allocation.

| Tracers | Construction | Median time | Median host allocation |
|---|---|---:|---:|
| 6 | Host workspace then adapt | 54.30 ms | 148.71 MB |
| 6 | Device workspace | 32.96 ms | 0.22 MB |
| 32 | Host workspace then adapt | 165.46 ms | 596.61 MB |
| 32 | Device workspace | 104.70 ms | 0.23 MB |

The 32-tracer constructor is 36.7% faster in this experiment. Raw repetitions
are in `construction.toml` and `construction.txt`. Run from the same configured
source export as `profile.jl`; the constructor script reads the prior profile's
`/tmp/atmos-main-real-input-after/tracers32.toml` and the archived grid.

## Whole-run check

`profile.jl` repeats the same archived experimental ERA5 C90 L66 workload as
the prior checkpoint: PPM, TM5 convection, exact Dkg diffusion, two model hours,
column snapshots at hours 0 and 2. Sample 0 is warmup; medians use samples 1–2.

| Tracers | Measure | Before | Direct device allocation |
|---|---|---:|---:|
| 6 | Whole-run time | 3.595 s | 3.625 s |
| 6 | Cumulative host allocation | 2.123 GB | 1.974 GB |
| 32 | Whole-run time | 15.148 s | 15.556 s |
| 32 | Cumulative host allocation | 7.848 GB | 7.252 GB |

This short whole-run comparison does **not** establish an additional speedup:
the medians are 0.8% and 2.7% slower despite the isolated constructor saving.
We retain the change for measured constructor speed and allocation savings.
These are warm-cache runs, not cold-NAS measurements or peak-memory estimates.
The archived forcing's experimental convection cadence is documented in the
[original real-input baseline](../main_real_input_v100_20260905/README.md).

## Correctness

All 196 output arrays are exactly equal before and after this change, including
compensated Float64 tracer totals; 280 comparison/conservation checks pass.
Maximum relative mass drift remains 2.572e-6 over two model hours. Conservation
checks use those NetCDF totals; the raw profile TOMLs' `final_totals` use the
ordinary state reduction and are not the compensated diagnostics.

The V100 split-file physics suite also passes all 75 checks, covering
Upwind/CMFMC, PPM/matrix CMFMC, and LinRood/TM5 with diffusion and tracer decay,
against continuous GPU runs and CPU references. See `gpu_checks.txt`.
Metal hardware was not tested.

The final complete CPU suite passes 82,059 checks across 116 test files, with
22 existing skips or expected failures. Aqua passes; JET is 142 against the
unchanged limit of 144. This includes the output, input-lifetime, and staging
ports as well as the workspace change.
