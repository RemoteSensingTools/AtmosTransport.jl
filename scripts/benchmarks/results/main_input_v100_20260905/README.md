# Input and file-handoff validation on current main

> The reproducer scripts `compare.jl` and `profile.jl` were retired from the tree; recover with `git show 7c515038:scripts/benchmarks/results/main_input_v100_20260905/<file>`.

Julia 1.12.6, four threads; CUDA.jl 5.11.3 / runtime 12.6 on tofu GPU 0,
Tesla V100-PCIE-16GB. Only the explicitly selected V100 was used.

- CPU: 337 focused checks pass, plus Aqua (10) and JET (141 findings,
  unchanged allowance 144).
- GPU: 223 checks pass (140 startup, 8 failure consumption, 75 split-file
  physics / CPU reference). See the raw log and summaries.
- File-handoff physics uses Upwind/CMFMC, PPM/matrix CMFMC, and LinRood/TM5,
  each with vertical diffusion and tracer decay. GPU comparisons use Float32;
  CPU continuous/split-file checks use Float64. The small fixture tests
  correctness, not representative throughput.

Reproduce the GPU checks with `test/diagnostic/test_window_prefetch_gpu.jl`,
`test_window_prefetch_failure_gpu.jl`, and `test_cs_multifile_gpu.jl`. Their
headers list the opt-in variables. Select the authorized device with
`CUDA_VISIBLE_DEVICES` and set both expected-device variables to `V100`.
Do not use CUDA runtime 13 for this Volta GPU.

## Real ERA5 C90 L66 measurements

The single-file workload is the same experimental archived forcing used by
[the output checkpoint](../main_output_v100_20260905/README.md), with PPM,
TM5 convection, exact Dkg diffusion, and column snapshots at 0 and 2 hours.
This comparison isolates the input port after `36c23f8a`; it does not exercise
workspace reuse across files. See `profile.jl`, warm sample 0, and measured
samples 1 and 2. All 196 before/after arrays are exactly equal, including signed
Float64 totals; 280 comparison/conservation checks pass.

| Tracers | Measure | Output checkpoint | Input port |
|---|---|---:|---:|
| 6 | Median whole-run time | 3.865 s | 3.595 s |
| 6 | Cumulative host allocation | 2.326 GB | 2.123 GB |
| 32 | Median whole-run time | 15.524 s | 15.148 s |
| 32 | Cumulative host allocation | 8.052 GB | 7.848 GB |

These are medians of two warm repetitions, not cold-NAS timings or confidence
intervals. Host allocation is cumulative, not peak resident memory. Startup
avoids about 203 MB of redundant host forcing allocation in this workload.
The archived convection fields have the experimental cadence described in the
baseline README; these runs test runtime performance and equivalence.
