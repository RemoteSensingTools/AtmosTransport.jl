# Direct packing of cubed-sphere initial tracer state

> The reproducer scripts `compare.jl`, `halo_probe.jl`, `initialization.jl` and `profile.jl` were retired from the tree; recover with `git show 7c515038:scripts/benchmarks/results/main_direct_packing_v100_20260905/<file>`.

The runner now allocates final packed storage once and converts each tracer's
interior VMR directly into its slot. It avoids retaining six temporary mass
panels per tracer and then copying them into state. The shared conversion keeps
signed values, zeros initial halos, and preserves dry/moist arithmetic. The
runner still rejects moist CS binaries without humidity. The packed tracer
order is preserved, including dictionary resize cases at 17 and 64 tracers.

## Real-input V100 comparison

The baseline is the [pressure-layer typing checkpoint](../main_pressure_init_v100_20260905/README.md)
(`e50aecaf`). This uses the same local C90 L66 experimental ERA5 format-4 archive,
PPM advection, TM5 convection, exact Dkg diffusion, and two model hours. Column
means and compensated totals are captured at hours 0 and 2. See the
[original baseline](../main_real_input_v100_20260905/README.md) for the archive's
experimental convection cadence and input provenance.

Julia 1.12.6, four Julia threads, one OpenBLAS thread, CUDA.jl 5.11.3 / runtime
12.6, tofu GPU 0 (Tesla V100-PCIE-16GB). Run `profile.jl` from the configured
source export with `--startup-file=no --project=.`, `ATMOSTR_TIMERS=1`, and the
authorized device in `CUDA_VISIBLE_DEVICES`. Sample 0 warms each tracer count;
samples 1–2 are measured. The table reports their medians.

| Tracers | Measure | Before | Direct packing |
|---|---|---:|---:|
| 6 | Whole-run time | 1.940 s | 1.753 s |
| 6 | Cumulative host allocation | 1.041 GB | 0.954 GB |
| 32 | Whole-run time | 5.675 s | 5.030 s |
| 32 | Cumulative host allocation | 2.275 GB | 1.808 GB |

The 32-tracer median improves another 11.4% and cumulative host allocation
falls by 467 MB. These are short startup-heavy warm-cache runs, not estimates
of peak memory, cold-NAS throughput, or steady-state kernel speed. The two
six-tracer samples span 1.590–1.916 s. Per-section allocation timers are disabled,
so CSV allocation zeroes mean unmeasured. Timing sections nest; do not sum them
as a partition of wall time. GPU halo section times can include waiting for
previously launched sweeps at their synchronization boundary.

All 196 output arrays are exactly equal before and after, including compensated
Float64 tracer totals. All 280 comparison/conservation checks pass, with maximum
relative mass drift unchanged at 2.5718011894571768e-6. Raw profile TOML
`final_totals` use the ordinary state reduction; conservation checks use the
compensated NetCDF totals.

## Local initialization comparison

`initialization.jl` compares the preceding construction algorithm with direct
packing in one CPU process, using the shared current packer. It alternates
measurement order, runs a warmup per tracer count, and collects garbage before
each measurement. Medians use five measured repetitions. The synthetic Float32
C90 L66 case has 100000 Pa surface pressure, 1e16 kg cell masses, and pressure-layer
tracers normalized to 1e35 molecules. Final state shape is `(96,96,66,Nt)` on each
of six panels. Accurate host reduction of the final tracer gives
9.999999507942511e34 molecules in both paths (Float32 initialization rounding).

| Tracers | Initialization | Median time | Cumulative host allocation |
|---|---|---:|---:|
| 6 | Previous construction | 0.128910 s | 254,515,936 B |
| 6 | Direct packing | 0.081558 s | 166,917,600 B |
| 32 | Previous construction | 0.720873 s | 1,357,432,616 B |
| 32 | Direct packing | 0.496270 s | 890,236,048 B |

The new focused CPU suite passes 558 checks for Float32/Float64 signed dry/moist
packing, zero/nonzero halos, independent slots and inputs, invalid shapes, and
exact state/tracer-order equivalence at 1/6/7/17/32/64 tracers. The V100
file-handoff suite passes 75 checks: Upwind/CMFMC, PPM/matrix CMFMC, and
LinRood/TM5, each with diffusion and decay, agree with continuous GPU runs and
CPU references within the existing tolerances. See `gpu_checks.txt`.

The complete CPU suite passes 83,556 checks across 117 core files and the
regridding runner, with 22 existing skips/expected-broken checks and no failures.
This includes the prior CLI and pressure-layer changes. Aqua passes. The full
run reported 144 JET findings within the unchanged allowance; inspection traced
the two additional findings to analysis considering LL arrays for the new
CS-only helper. Its signature now explicitly requires a CS grid and six 3D
panels. After that restriction, all 558 focused packing checks pass again and
JET returns to 142 findings. The whole-run measurements above precede this
signature restriction, which changes no arithmetic or storage layout.

## Separating halo work from queued sweeps

`halo_probe.jl` reads the actual archive's C90 GEOS-native GMAO mesh and uses
Float32 panel-marker fields with 66 levels. It synchronizes the V100 before and
after each halo call, includes directional corner filling, warms each case once,
and measures ten repetitions. Each final GPU panel agrees exactly with the CPU
reference (24 checks). These are isolated synthetic fields on the archived mesh.

| Tracers | Direction | Median isolated halo call |
|---|---|---:|
| 6 | X | 0.304 ms |
| 6 | Y | 0.308 ms |
| 32 | X | 1.683 ms |
| 32 | Y | 1.675 ms |

In the real 32-tracer run, host halo sections average about 18 ms (X) and 14 ms
(Y), because the halo synchronization boundary also waits for previously queued
sweeps. Those section times are therefore not isolated halo-kernel timings. The
next transport investigation should separate sweep execution and launch costs;
this probe does not support prioritizing halo-copy changes from the aggregate
section totals alone. No advection or halo kernel changes are included here.
