# Pressure-layer initialization: concrete configuration scalars

> The reproducer scripts `compare.jl`, `equivalence.jl`, `initialization_probe.jl` and `profile.jl` were retired from the tree; recover with `git show 7c515038:scripts/benchmarks/results/main_pressure_init_v100_20260905/<file>`.

The pressure-layer initializer now asserts the types of the already converted
`psurf_fraction` and `total_molecules` before its column loops. Julia 1.12.6
previously inferred both as `Any` when loaded from `Dict{String,Any}`. That also
made the target pressure, midpoint distance, best distance, and normalized VMR
untyped. The fix retains the same conversions, arithmetic, loop order, and
physical normalization.

## Whole-run V100 measurements

Same setup as the [preceding device-workspace checkpoint](../main_device_workspace_v100_20260905/README.md):
tofu GPU 0, Tesla V100-PCIE-16GB, Julia 1.12.6, four Julia threads, one OpenBLAS
thread, CUDA.jl 5.11.3 / runtime 12.6. The C90 L66 experimental ERA5 format-4
archive runs two model hours with PPM, TM5 convection, and exact Dkg diffusion.
Each pressure-layer tracer contains 1e35 molecules; output contains column means
and compensated Float64 totals at hours 0 and 2.

`profile.jl` records sample 0 as warmup and samples 1–2 as measured repetitions.
Run with `ATMOSTR_TIMERS=1` and the authorized GPU UUID in
`CUDA_VISIBLE_DEVICES`, matching the preceding baseline. Per-section allocation
timers are disabled: CSV allocation zeroes mean unmeasured, and nested timing
sections must not be summed as a partition of wall time.
It uses the same configured `/tmp` source export as the prior checkpoint, with
only `src/Models/initial_conditions/cubed_sphere.jl` replaced. The maintained CLI
import change is outside this direct runner profile. Remote staging is disabled;
the archive is already local. Do not interpret these as cold-NAS or peak-memory
measurements. The [original baseline](../main_real_input_v100_20260905/README.md)
documents the archive's experimental convection cadence.

| Tracers | Measure | Before | Typed configuration |
|---|---|---:|---:|
| 6 | Median whole-run time | 3.625 s | 1.940 s |
| 6 | Median cumulative host allocation | 1.974 GB | 1.041 GB |
| 32 | Median whole-run time | 15.556 s | 5.675 s |
| 32 | Median cumulative host allocation | 7.252 GB | 2.275 GB |

The 32-tracer run is 63.5% faster with 68.6% less cumulative host allocation in
this short workload. Initialization contributes once per run; these percentages
are not steady-state transport-kernel speedups. The six-tracer measured times
span 1.761–2.118 s, so its median has appreciable run-to-run variability.

All 196 output arrays are exactly equal to the preceding checkpoint, including
compensated totals. All 280 comparison/conservation checks pass; maximum relative
mass drift remains 2.5718011894571768e-6. Raw TOML `final_totals` use the ordinary
state reduction; conservation checks use the NetCDF compensated totals.

## Local allocation probe

`initialization_probe.jl` runs on CPU with a synthetic Float32 C90 L66 grid,
96×96 halo-padded panels, uniform 100000 Pa surface pressure and 1e16 kg cell
mass. It separates VMR creation, mass packing, and state construction. It warms
each tracer count once and prints the second pass; raw logs retain both passes.
For 32 tracers the final packed shape is `(96,96,66,32)` per panel.

| 32-tracer operation | Before | After |
|---|---:|---:|
| VMR construction cumulative allocation | 5,399,718,720 B | 423,076,704 B |
| Mass packing cumulative allocation | 467,161,600 B | 467,161,600 B |
| State packing cumulative allocation | 467,178,416 B | 467,178,416 B |

These local timings were collected alongside other checks and are retained as
probe output, not a controlled timing comparison. The local `total` field uses
the ordinary Float32 state reduction and is not an accurate molecule diagnostic.
The scientific tests separately verify the requested molecule count.

## Scientific and package checks

- 934 new CPU checks cover Float32/Float64 layer choices, varying surface
  pressure and hybrid coefficients, molecule normalization, ties, ignored
  pressure fractions in lowest-layer mode, invalid inputs, and excluded halos.
- `equivalence.jl` loads the preserved method from git commit `9544a3d3` under a
  different name. All 80 cases compare exactly (both precisions, L4/L66, two
  halo widths, pressure fractions, and lowest-layer mode). Run from the repo
  root with `--project=.`.
- Existing initial-condition I/O tests pass 133 checks, including signed native
  initialization and dry/moist packing. Aqua passes 10 checks and JET passes
  with 142 reports against the unchanged allowance of 144.

The full 82,059-check CPU suite passed at the preceding checkpoint. This small
configuration-boundary change was validated with the focused tests above and
actual V100 runs; that full suite was not repeated for this checkpoint.
