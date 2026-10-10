# Heritage benchmarks

Scripts in `heritage/` are not part of a maintained workflow. They are kept
for reference, as the method or evidence of a finished study, until they are
trimmed. They may hard-code the paths, dates and data of that study, and they
are not tested; check imports and inputs before running one. Retired scripts
are listed under "Retired scripts" in [`scripts/README.md`](../../README.md).

| Script | Purpose | Why it is here |
|---|---|---|
| `bench_chemistry_overhead.jl` | Plan 15: `step!` cost with vs without `ExponentialDecay` chemistry (LatLon) | One-off overhead check; ran on CPU 2026-10-09. Shares a duplicated problem builder with the diffusion and emissions benches |
| `bench_diffusion_overhead.jl` | Plan 16b: `step!` cost across Kz-field types vs `NoDiffusion` (LatLon) | LatLon overhead recipe; `benchmarking/` covers cubed-sphere diffusion cost. Ran on CPU 2026-10-09 |
| `bench_emissions_overhead.jl` | Plan 17: `step!` cost with vs without `SurfaceFluxOperator` (LatLon) | One-off overhead check; ran on CPU 2026-10-09. Shares the duplicated problem builder |
| `bench_matrix_convection_gpu.jl` | GPU benchmark of matrix (TM5-style) convection solves: native batched vs split launches | Tool of the finished Sept-2026 matrix-convection study (results in memos and `../results/`); worth rerunning when the convection kernels change. Needs CUDA |
| `bench_strang_sweep.jl` | Plan 14 synthetic LatLon Strang-split per-step benchmark (per-tracer vs multi-tracer, CFL caps) | Only LatLon advection microbenchmark (`benchmarking/` covers the cubed sphere only); the plan-14 decision is settled. Ran on CPU 2026-10-09 |
| `multigpu_dispatch_benchmark.jl` | Synthetic CUDA benchmark of strategies for dispatching 6 CS panels across 1-2 GPUs (C180/C720) | Only exploration of multi-GPU dispatch, which `src/` does not implement; may matter for C720. Uses no package API |
| `probe_matrix_convection_cpu.jl` | CPU comparison of dense vs structured (Hessenberg/bidiagonal) TM5 convection factorizations and solves | Results are in three memos and `../results/matrix_convection_*_cpu_*.toml`; useful when touching the convection solvers. Ran on CPU 2026-10-09 |
| `run_cs_transport.jl` | Low-level CS advection-only runner on real binaries, with its own snapshot writer | Duplicates `scripts/run_transport.jl` with an advection-only config and `bench_cs_advection_gpu.jl`; its example config is in `config/runs/completed_experiments/` |
