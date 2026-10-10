# Float64 packed PPM launch geometry on V100

> The reproducer scripts `check_outputs.jl`, `profile.jl`, `summarize.jl` and `sweep_tiles.jl` were retired from the tree; recover with `git show 7c515038:scripts/benchmarks/results/main_f64_ppm_tiles_v100_20260906/<file>`.

Cubed-sphere CUDA Float64 PPM sweeps now use a 32-thread row. Each thread
still evaluates the same reconstruction, limiter, mass update, and tracer loop.
The change adds only a precision-specific launch-policy method; it adds no
workspace and changes no arithmetic. Float32 retains its 32×2 layout.

## Complete model days

The workload is C90 L66, one complete December 2018 ERA5 day, six or 32
pressure-layer tracers, full-column collaborative TM5 convection, conservative
Dkg diffusion, no emissions, and `preserve_tracer_mass` window resets. The
input and science settings match the preceding
[Float64 convection comparison](../main_f64_collab_v100_20260906/README.md).
Only the packed advection workgroup size changes between these runs.

| Float64 tracers | 256-thread baseline | 32-thread row | 32×2 alternative |
|---|---:|---:|---:|
| 6 | 21.590 s | 17.455 s | 17.237 s |
| 32 | 63.061 s | 47.482 s | 48.474 s |

The selected layout reduces measured wall time by 19.2% with six tracers and
24.7% with 32. It also gives the lower isolated Y/Z sweep times at 32 and 65
tracers. The small difference between the two narrow layouts does not establish
a universal ranking: each table entry is the median of two measured samples,
after sample 0 warms that tracer count. Layouts run sequentially, in separate
Julia processes, in the order 256, 32, 32×2. No cold-I/O claim follows from these
cached runs. The six-tracer 32-thread samples span 17.003–17.907 s.

Cumulative host allocation remains approximately 10.781 GB for six tracers and
11.719 GB for 32; differences are below 0.02%. These are allocation totals, not
peak RAM. Nested timing CSV sections include asynchronous waits and must not
be summed. Their allocation columns are unmeasured zeroes.

The initial trial exhausted tofu's 10 GiB `/tmp` filesystem while saving large
benchmark artifacts. That incomplete run is excluded. All layouts above were
rerun with outputs on the same larger data filesystem, including a fresh
256-thread reference. Native state writes occur outside the timed region.

## Conservation and field checks

All 1,788 saved-output checks pass across both measured samples, both tracer
counts, and all three layouts. Every saved array is exactly equal to the
preceding collaborative-convection reference, including compensated tracer
totals. The NetCDF snapshot fields are Float32 even for a Float64 model, so
sample 1 also saves every final packed panel in native Float64 precision.
All 12 file-size/hash checks pass: the complete tracer storage is byte-identical,
including all 66 layers, tracers, and halo cells.

The isolated sweep experiment passes another 240 identity/finite checks on
actual C90 air mass and fluxes with synthetic smooth signed tracer fields.
It covers 1, 6, 32, and 65 tracers, all three directions, and seven launch
layouts. The input panel is `(96,96,66)` Float64 with three halo cells;
first-interior level-33 air mass is 2.829608026112e12 kg. CUDA scalar indexing
is disabled.

For C90, a 256-thread row launches 166 inactive threads per panel row. A
32-thread row pads 90 cells to 96, so 93.75% of launched threads address the
interior, versus 35.2%. These fractions describe launch padding. The event
microbenchmark warms each case and records five samples on an idle stream:

| Tracers | Direction | 256-thread row | 32-thread row |
|---|---|---:|---:|
| 6 | X | 1.233 ms | 0.702 ms |
| 6 | Y | 1.050 ms | 0.432 ms |
| 6 | Z | 0.971 ms | 0.475 ms |
| 32 | X | 6.627 ms | 3.466 ms |
| 32 | Y | 4.089 ms | 2.004 ms |
| 32 | Z | 3.968 ms | 2.286 ms |
| 65 | X | 12.985 ms | 6.361 ms |
| 65 | Y | 7.537 ms | 3.997 ms |
| 65 | Z | 7.942 ms | 4.690 ms |

No new normalization, truncation, aggregation, or precision conversion is
introduced. The measured conservation behavior therefore stays at the preceding
checkpoint. Field positivity and time-discretization accuracy remain separate
scientific questions.

## Reproduction and review

The device is tofu GPU 0, Tesla V100-PCIE-16GB, explicitly selected by
`CUDA_VISIBLE_DEVICES=GPU-59ee7ef9-898f-524a-dde1-7a0426a41ecb`. The environment
is Julia 1.12.6, CUDA.jl 5.11.3, CUDA runtime 12.6, four Julia threads, and one
OpenBLAS thread. Other NVIDIA architectures are unmeasured.

From the configured `/tmp/atmos-f64-collab` export, set `ATMOSTR_TIMERS=1` and
`ATMOSTR_BENCHMARK_OUTPUT_ROOT=/home/cfranken/data/AtmosTransport/benchmarks/revamp-20260906`,
then run `profile.jl 256`, `profile.jl 32`, and `profile.jl 32x2` sequentially.
The prototype overrides only the CUDA Float64 PPM launch trait. This export
has the numerical implementation committed in `48b91344`, before its final
prose and capability-API integration. The final production export is based on
`48b91344` plus the launch-policy method.

Run `check_outputs.jl` with GPUs hidden, the same output-root environment
variable, and `ATMOSTR_BENCHMARK_REFERENCE` pointing to the preceding
`atmos-f64-profile-Float64` results. `sweep_tiles.jl` writes its event samples
to `/tmp/atmos-f64-sweep-tiles.toml`. `summarize.jl` recomputes the day table from
the small archived samples. Large NetCDF/native-state artifacts remain on tofu.

Critical Codex self-review checked all six packed sweep wrappers, disjoint
Float32/Float64 dispatch, partial rows, unchanged CPU/Metal/other-scheme
selection, and the absence of shared state between workgroups. The change does
not alter the adjoint kernels. The maintained GPU diagnostics exercise the
actual production wrappers in both precisions, including copy-back ownership,
signed and zero tracers, halos, partial blocks, and CPU reference comparisons.

On the production export, all 540 launch-wrapper checks, 2,552 paired-seam
forward checks, and 24 adjoint comparisons pass through 65 tracers and both
panel conventions. Local documentation links, module maps, and public
docstrings pass 191 checks; the strict documentation build succeeds.
