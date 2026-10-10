# Release adjoint checks, 2026-09-06

> The reproducer scripts `fresh-resolve.jl`, `gpu-checks.jl` and `gpu-transport.jl` were retired from the tree; recover with `git show 7c515038:scripts/benchmarks/results/main_release_adjoint_v100_20260906/<file>`.

The resumed release check starts from source commit
`1a89772e970d7df04394832669b32c28f9f0c283` (AtmosTransport 0.3.0).
The local release fix changes `test/Project.toml`'s AtmosTransport compatibility
from `0.2` to `0.3`: the old constraint rejects the package in a fresh test
environment. The corrected environment resolves on Julia 1.10.12 and 1.12.6;
`fresh-resolve-1.10.log` and `fresh-resolve-1.12.log` retain those checks.

## Existing adjoint suites

`gpu-checks.log` records **640 passing assertions**, with exit status zero.
The runner preloads CUDA, requires exactly one visible V100, disables scalar
GPU indexing, and verifies that the footprint suite enabled its GPU branch.
The roster includes split-sweep footprints, CMFMC transpose identity,
model-space and preconditioned identities, checkpoint schedules, mmap tape,
Lin-Rood kernel adjoints, and Lin-Rood ORD=5/7 finite-difference integration.
Several suites use CPU arrays even in this CUDA-enabled process; the total is
not a count of GPU-only assertions. The integration file includes the footprint
suite again, as in the committed diagnostic.

The runner emitted one Julia 1.12 world-age warning while reading the freshly
included module's `HAS_GPU` binding. That check passed. This warning belongs to
the archived runner's module inspection, not a transport kernel.

## Lin-Rood GPU repair

The existing suite passed, but the added transporting probe found scalar GPU
indexing in Lin-Rood tape copy-back and reverse halo processing. All four
Lin-Rood cases (ORD=5/7, Float32/Float64) errored before the fix; the four PPM
cases passed. `gpu-transport-before.log` retains that failure.

Copy-back now uses the production `_copy_interior!` kernel. Halo chain-rule
and carry-over additions use backend kernels; interior clearing and final
accumulation use backend array operations. The same repair applies to the
single-panel recording and multi-substep reverse wrappers. The CPU arithmetic
and the forward/reverse ordering are preserved.

After the fix, `gpu-transport-after.log` records **580 passing assertions**.
The Float64 directional relative errors were:

| Scheme | Relative error |
| --- | ---: |
| Unlimited PPM | 9.23e-12 |
| Monotone PPM | 1.66e-11 |
| Lin-Rood ORD=5 | 5.13e-8 |
| Lin-Rood ORD=7 | 5.13e-8 |

The maintained regression,
`test/diagnostic/test_cs_transport_adjoint_gpu.jl`, passed **703 assertions**:
91 existing footprint checks, 580 transporting-gradient/replay checks, and
32 single-panel wrapper checks with nonzero halo seeds. `gpu-regression.log`
records the final run and its successful summaries; the process exited zero.
It runs only when explicitly enabled with
`ATMOSTR_RUN_TRANSPORT_ADJOINT_GPU_TESTS=1`. `source.patch` records the numerical
source changes relative to the starting commit.

## Transporting GPU probe

`gpu-transport.jl` reuses the committed fixture definitions, suppressing their
top-level testsets because those ran in the first stage. It checks a smooth
nonzero tracer on six C4 panels, four vertical levels, and three transport
steps. The observation is a column mean at panel 1's corner cell, so the probe
exercises panel-edge transport. Cases cover unlimited PPM, monotone PPM, and
Lin-Rood ORD=5/7 in Float64 and Float32.

Every footprint panel is checked for device residency and CPU/GPU agreement.
Float64 additionally compares the GPU gradient with centered differences of
the GPU forward model, using emission perturbations of ±2e-6. Both precisions
compare `StrideCheckpoint(2)` and `RevolveCheckpoint()` with full recording.
These are synthetic gradient and replay checks; they do not close the
real-meteorology TM5-4DVAR parity or campaign-validation gaps.

## CPU and documentation checks

Full `Pkg.test()` runs passed on Julia 1.12.6 and 1.10.12, on the starting
source with the test compatibility repair (`pkg-test-1.12.log` and
`pkg-test-1.10-clean.log`). Both ran the core and regridding tiers with bounds
checking. Julia 1.10 used a clean export with a freshly resolved manifest,
avoiding the workspace's Julia 1.12 manifest. The Lin-Rood repair was made
while those baseline runs were active; it is verified separately by the
post-fix checks below.

`linrood-fixed-cpu.log` records passing Lin-Rood kernel, finite-difference
integration, and checkpoint tests on the repaired source. `health-fixed.log`
records passing Aqua and JET checks, and `docs-fixed.log` records a successful
build with deployment disabled. JET's Julia 1.12 snapshot changes from 152 to
154 solely for the two added GPU kernel-dispatch reports; `jet-delta.txt`
records their complete messages from the before/after comparison. The Julia
1.10 allowance is unchanged.

`linrood-fixed-cpu-1.10.log` records passing post-fix Aqua, JET, kernel,
finite-difference integration, and checkpoint tests on Julia 1.10.12, with
exit status zero. JET reports zero items against its existing allowance of
130. The CPU-only footprint fixture's optional CUDA probe encountered a
precompilation error in the user's shared Julia 1.10 CUDA environment; that
probe is caught and the CPU assertions all passed. CUDA validation uses the
explicit Julia 1.12/V100 environment above.

## Environment and reproduction

Only tofu GPU 0 was exposed: Tesla V100-PCIE-16GB,
`GPU-59ee7ef9-898f-524a-dde1-7a0426a41ecb`. The environment uses Julia 1.12.6,
CUDA.jl 5.11.3, CUDA runtime 12.6, four Julia threads, and one OpenBLAS thread.
The isolated export remains at `/tmp/atmos-release-resume-20260906` on tofu.
Checksum comparisons confirmed the initial export matched the local `src/`
and `ext/`. The three repaired Julia source files were then copied into that
export for the post-fix probes. No production checkout on tofu was modified.

The archived `gpu-Project.toml` preserves the earlier benchmark environment:
relative to the source commit's Project it adds `CUDA_Runtime_jll` to extras
for the runtime preference and omits the `FileWatching` compatibility entry.
The post-fix runs include the Lin-Rood repair above. CUDA loads from tofu's
Julia 1.12 shared environment; the project's `LocalPreferences.toml` selects runtime `12.6`.

To repeat on a configured isolated export, copy `gpu-checks.jl` and
`gpu-transport.jl` from this directory into that export's root, then run there:

```bash
export CUDA_VISIBLE_DEVICES=GPU-59ee7ef9-898f-524a-dde1-7a0426a41ecb
export JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
julia --startup-file=no --project=. gpu-checks.jl
julia --startup-file=no --project=. gpu-transport.jl
```

The fresh-resolution probe runs from the repository root with either Julia
version and creates an independent temporary environment:

```bash
julia --startup-file=no --project=. scripts/benchmarks/results/main_release_adjoint_v100_20260906/fresh-resolve.jl
```
