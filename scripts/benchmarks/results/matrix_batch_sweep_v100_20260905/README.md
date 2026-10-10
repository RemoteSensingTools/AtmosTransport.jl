# V100 tracer-batch sweep

Measured on tofu GPU 0, Tesla V100-PCIE-16GB, CUDA.jl 5.11.3 with CUDA runtime
12.6 and Julia 1.12.6. Each case contains nine warmed single-step samples,
restoring its input outside timing. All 4,096 synthetic Float32 columns convect;
this experiment excludes I/O and transfer time.

`batchsix.toml` uses the production six-tracer batch. `batch32.toml` uses a local
prototype changing `_TM5_COLLAB_TRACER_BATCH` from 6 to 32. Both start from
`db9bb0cd` on the old review branch, with benchmark environment controls for
level/tracer sweeps. These are historical measurements, not timings of the
integrated main branch. Reference and conservation errors are recorded per case.

Reproduce the sweep on an explicitly selected authorized GPU using:

```bash
CUDA_VISIBLE_DEVICES=0 ATMOSTR_MATRIX_GPU_NAME=V100 \
ATMOSTR_MATRIX_BENCH_LEVELS=60,66,85 \
ATMOSTR_MATRIX_BENCH_TRACERS=6,16,24,31,32,33,48,65 \
ATMOSTR_MATRIX_BENCH_COLUMNS=4096 \
julia --project=. scripts/benchmarks/heritage/bench_matrix_convection_gpu.jl results.toml native
```

Choose a separate output path for each variant and set
`ATMOSTR_MATRIX_BENCH_REVISION` to record the measured source. The benchmark
checks the selected device name before launching kernels.

## Observations

At 85 levels, batch 32 reduces 65-tracer kernel time from 19.803 to 15.774 ms
without downdrafts and 38.910 to 31.062 ms with downdrafts (about 20%). For six
tracers it increases time from 6.815 to 9.886 ms and 15.503 to 22.608 ms (about
45%). The 33-tracer boundary provides only a 2–4% improvement at this depth,
while 31/32 and 48 tracers benefit more. Level count also changes the crossover.

The portable production batch remains six. Before selecting larger batches at
runtime, check other column counts, signed tracers, backend shared-memory
limits, and a real forcing profile. A global switch to 32 is not justified by
these results.
