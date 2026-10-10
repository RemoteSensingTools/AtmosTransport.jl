# First real-input profile after integration onto main

> The reproducer scripts `profile.jl` and `verify.jl` were retired from the tree; recover with `git show 7c515038:scripts/benchmarks/results/main_real_input_v100_20260905/<file>`.

Source: main `a0698dde` plus the matrix ports through `63904499`, exported to
`/tmp/atmos-revamp-main` on tofu. GPU 0 is a Tesla V100-PCIE-16GB; Julia 1.12.6,
CUDA.jl 5.11.3, CUDA runtime 12.6, four Julia threads, scalar GPU indexing off.
No production checkout or source binary was changed.

The source is the existing format-4 C90 L66 Float32 archive:

```
~/data/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour/era5_n320_transport_20181201_float32.bin
```

This archived experimental forcing combines hourly transport with 1-degree,
three-hour convection fields held over the intervening hourly windows. It is a
representative runtime workload, not an independent scientific validation of
that forcing preparation. The binary has explicit `dm`, TM5 rates, and exact
TM5 `dkg`. The current reader loaded it directly without conversion.

`profile.jl` runs the first two hourly windows (20 substeps), PPM advection,
exact `tm5_dkg` diffusion, and collaborative TM5 convection on all 66 levels
with no layer aggregation. Tracers start in different pressure layers. Output
requests column means at hours 0 and 2. Sample 0 warms each tracer count;
samples 1 and 2 are the measured repetitions. The machine-specific paths are
explicit in the script. To reproduce, select the authorized GPU with
`CUDA_VISIBLE_DEVICES`, set `ATMOSTR_TIMERS=1`, and run the script with
`julia --startup-file=no --project=.` in the measured export.

## Results

Medians of the two measured repetitions (a preliminary warm-cache profile):

| Measure | 6 tracers | 32 tracers |
|---|---:|---:|
| Whole run | 4.149 s | 16.396 s |
| Cumulative host allocations | 2.497 GB | 8.855 GB |
| Garbage collection | 0.973 s | 2.665 s |
| Advection section | 0.595 s | 2.758 s |
| Diffusion section | 0.148 s | 0.753 s |
| Convection section | 0.169 s | 0.306 s |
| Prefetch fetch wait | about 1 µs | about 1 µs |
| NetCDF file | 4.345 MB | 14.568 MB |

Allocation counts are total allocated host bytes, **not peak live memory**.
Section timers can overlap or nest; their fractions must not be added as a
partition of wall time. Per-section allocations were not enabled, so zeroes in
those CSV columns mean unmeasured. No cold-NAS or peak GPU-memory measurement
was made. With only two measured repetitions, these are baseline observations,
not confidence intervals or a before/after speedup claim.

`verify.jl` checks all six output files: 456 tracer-field and total-mass checks
plus six time-axis checks pass. Every emitted tracer field is finite; maximum
relative total-storage drift is 2.572e-6 over two model hours (threshold 1e-5).
This tests short-run conservation, not equivalence to an external model.

Convection is now a small part of this run's wall time. The large host allocation
count and garbage-collection cost motivate porting selected snapshot capture,
streaming output, and one-load startup next, then repeating this same profile.
