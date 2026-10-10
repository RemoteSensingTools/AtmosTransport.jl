# Release review follow-up, 2026-09-06

> The reproducer scripts `benchmark-smoke.jl` and `hdf5-compat.jl` were retired from the tree; recover with `git show 7c515038:scripts/benchmarks/results/main_release_output_review_20260906/<file>`.

This follows commit `74ae835a36d4d09f8f55afbdf0ee97be1c988a4e`, which repaired
Lin-Rood GPU recording and reverse propagation. The prior
[adjoint evidence](../main_release_adjoint_v100_20260906/README.md) retains the
703-assertion V100 regression and distinguishes baseline CPU runs from focused
post-fix checks. This follow-up addresses the supplied output/dependency
review and checks the complete final source. The production fixes are
committed as `ad50e9b9` (output/dependencies) and `7030c1f7` (benchmark setup).

## Review disposition

| Finding | Disposition |
| --- | --- |
| Test environment still required AtmosTransport 0.2 | The previous commit corrected it to 0.3; this follow-up removes the redundant self-compat bound entirely. `Pkg.test()` injects the checkout. Direct Julia 1.10 invocations explicitly develop it, as documented in `test/README.md`. |
| HDF5_jll restricted to 2.1 | Widened root and test bounds to `1.14, 2`; confirmed coexistence with HDF5.jl 0.17.3 and the older NetCDF stack. The error-handler comment now correctly attributes thread locality to thread-safe builds. |
| Dynamic stream metadata fields | Retained. These are schema/file-handle slots in snapshot I/O; the review found no measured cost. No claim is made that these fields are concrete. |
| Metal column sums reduced in Float32 | Backends without device Float64 now transfer at most 16 vertical levels per slab and accumulate each column in host Float64, preserving CPU model-level order across slab boundaries. CUDA retains its device Float64 reduction. |
| Closing a stream could replace an asynchronous write failure | `RunSnapshotOutput.close` now uses the existing resource helper to retain both errors in a `CompositeException`. A regression injects both failures without corrupting a real dataset. |
| Benchmark environment also required AtmosTransport 0.2 | Independently found and reproduced after checking the failed main benchmark workflow. Removed this second self-compat bound and documented checkout setup for Julia 1.10. |

The HDF5 automatic-error APIs predate either supported dependency series; see
[the HDF5 error API reference](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html).
Per-thread error handling is a property of the
[thread-safe build](https://support.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html).
The direct dependency remains so the imported JLL is declared explicitly.

## Completed focused checks

- `output-focused.log`: selected/full output parity, 19 slab-cancellation
  assertions, stream schema/lifetime suites, and 11 assertions for simultaneous
  asynchronous-write/stream-close failures all pass.
- `hdf5-114.log`: HDF5.jl **0.17.3**, HDF5_jll **1.14.6+0**, NCDatasets
  **0.14.15**, NetCDF_jll **401.900.300+0** resolve together. The snapshot,
  selected-output, stream, and asynchronous lifetime suites pass. The probe
  asserts `NCDatasets.NetCDF_jll.HDF5_jll === HDF5_jll` and prints the shared
  library path. The default environment checks the HDF5 2.1 stack separately.
- `output-gpu.log`: **389 assertions pass** on tofu's Tesla V100-PCIE-16GB,
  with CUDA scalar indexing disabled. These comprise 14 column checks, 33
  compensated-total checks, and 342 selected/full/stream parity checks across
  latitude-longitude, reduced-Gaussian and cubed-sphere grids in Float32/64.
- `docs.log`: the manual builds successfully with deployment disabled.

The cancellation probes use `Float32(2^24), 1, -Float32(2^24)` within a slab,
across the 16-level boundary, and across separated slabs. They also exercise a
non-contiguous cubed-sphere interior view on the GPU. The expected column sum
is exactly one, as with the CPU Float64 diagnostic. This is ordered Float64
column accumulation; the separate global-total algorithm is compensated.

**Metal hardware was unavailable.** Its host-slab policy is tested directly on
CPU and real CUDA arrays, including device-to-host view transfers. This is not
an executed Metal-backend test or a Metal performance measurement.

## Full release checks

`pkg-test-1.12.log`: Julia **1.12.6** passes the complete `Pkg.test()` suite
(120 core files plus regridding) with bounds checking on the final source.
Aqua passes; JET remains at 154 reports against the unchanged 154 allowance.

`pkg-test-1.10.log`: Julia **1.10.12** also passes all 120 core files and
regridding with bounds checking on the final source. Aqua passes; JET reports
zero items against its unchanged allowance of 130. Both complete processes
exit zero. These runs include the Lin-Rood GPU repair and the output fixes,
unlike the baseline-only full suites in the earlier adjoint record.

`benchmark-before.log` reproduces the old self-compat failure. With the bound
removed, `benchmark-smoke.log` records **24 successful CPU smoke cases**:
C4, 32 levels, six operator modes, one/four tracers, and Float32/Float64.
All 24 raw timings and 120 dashboard records are finite and positive. These
single-repeat runs establish harness functionality, not a performance baseline.
The JSON metadata names the base commit; the run includes this follow-up
working-tree patch. The corrected benchmark environment also resolves on
Julia 1.10.12 (`benchmark-110-resolve.log`).

`fresh-test-1.10.log` and `fresh-test-1.12.log` confirm that the archived
resolution probe handles the omitted self-compat bound and resolves a fresh
test environment on both versions.

## Reproduction and source identity

`tested-source-sha256.json` records all source, extension, root Project and test
files exported before the complete CPU matrix. Both Julia versions use separate
clean exports with independently resolved manifests, four Julia threads, one
OpenBLAS thread, and `JULIA_LOAD_PATH='@:@stdlib'` to exclude optional packages
from the user's shared environment. The command in each export is:

```bash
JULIA_LOAD_PATH='@:@stdlib' JULIA_PKG_PRECOMPILE_AUTO=0 \
JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Only tofu GPU 0 was exposed:
`GPU-59ee7ef9-898f-524a-dde1-7a0426a41ecb`. The isolated export was
`/tmp/atmos-release-output-review-20260906`; it uses the same Julia 1.12.6,
CUDA.jl 5.11.3 and CUDA runtime 12.6 setup recorded in the prior adjoint evidence.
Checksum comparisons confirm that the complete tested `src/` and `ext/`
trees and the output diagnostic match this checkout. Run on
an explicitly selected, configured device with:

```bash
CUDA_VISIBLE_DEVICES=GPU-59ee7ef9-898f-524a-dde1-7a0426a41ecb \
JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
ATMOSTR_RUN_SNAPSHOT_GPU_TESTS=1 ATMOSTR_SNAPSHOT_GPU_NAME=V100 \
julia --startup-file=no --project=. test/diagnostic/test_snapshot_totals_gpu.jl
```

The archived `hdf5-compat.jl` and `benchmark-smoke.jl` probes run from the
repository root with `julia --startup-file=no --project=. <probe-path>`. They
create temporary dependency environments; the benchmark probe writes its JSON
outputs beside the script. Set the CPU environment variables above (use two
threads for the benchmark smoke to match CI).
