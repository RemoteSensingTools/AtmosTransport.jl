# Test layout

Tests are organized by tier. The orchestrator is [`runtests.jl`](runtests.jl);
it runs `core/` and `regridding/` by default and accepts opt-in flags to
include other tiers.

| Tier | Folder | When it runs | What's here |
|---|---|---|---|
| **Core** | [`core/`](core/) | Default (CI green-bar) | API/health gates, kernel correctness, runtime composition, adjoint identity, preprocessor contracts, mesh / regridding / vertical-transform invariants, type and aliasing safety. Everything that catches regressions of the production code paths. |
| **Real data** | [`real_data/`](real_data/) | `--all` or `--real-data` | Tests that need preprocessed transport binaries or ERA5 GRIB files on disk in `~/data/AtmosTransport/`. Skipped on CI runners. |
| **Diagnostic** | [`diagnostic/`](diagnostic/) | `--all` or `--diagnostic` | Large numerical sweeps useful for adjoint / inversion debugging (Taylor-sweep, integration footprints). Run occasionally, not every PR. |
| **Orphan** | [`orphan/`](orphan/) | `--orphan` only | Promotion candidates — tests that exist but were not in the CI roster as of 2026-05-29. Each should be reviewed and either promoted to `core/` (with a short note here) or moved to `archived/`. |
| **Archived** | [`archived/`](archived/) | Never (kept for reference) | Tests against deleted preprocessing wrappers and one-off plan-decision "studies". See [`archived/legacy_README.md`](archived/legacy_README.md). |
| **Regridding** | [`regridding/`](regridding/) | Default (CI green-bar) | Conservative-remapping geometry, conservation, direction, and persistence checks. Its `runtests.jl` is included as one isolated suite by the top-level orchestrator. |
| **Golden** | [`golden/`](golden/) | By hand, before committing a refactor | End-to-end preprocessing and runtime cases compared bit for bit with a recorded reference (`golden/run_goldens.jl`); needs the group's data and a GPU. Not run by `runtests.jl`. |

## Usage

For direct `--project=test` invocations, prepare the test environment from the
repository root first (also required on Julia 1.10, which does not use the
`[sources]` entry):

```bash
julia --project=test -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

The test environment intentionally has no version bound for AtmosTransport
itself: `Pkg.test()` supplies the checkout, and direct invocations develop it
with the command above. Release version bumps need no matching test bound.

```bash
julia --project=. -e 'using Pkg; Pkg.test()'         # CI-equivalent default
julia --project=test test/runtests.jl                # core + regridding
julia --project=test test/runtests.jl --all          # default + real data + diagnostic
julia --project=test test/runtests.jl --diagnostic   # default + diagnostic
julia --project=test test/runtests.jl --real-data    # default + real data
julia --project=test test/runtests.jl --orphan       # default + orphan watchlist
julia --project=test test/runtests.jl --tiers=core,orphan # only listed tiers
```

A failing file does not stop the run: every selected file runs (an interrupt
still stops it), and at the end the runner prints the 15 slowest files and then
fails with the list of failed files, if any.

## Adding a new test

1. Decide which tier the test belongs in. Default to `core/` if the test
   doesn't need external data and is part of a production code path.
2. Drop the file into the matching folder. The orchestrator picks up
   anything matching `test_*.jl` automatically — no manual roster edit.
3. Import the cached package with `using AtmosTransport` or
   `import AtmosTransport`. Each test file runs in its own module to isolate
   helpers and constants without creating new copies of package types.
4. Include shared test fixtures with paths relative to `@__DIR__`.

## Promotion / retirement workflow

- An `orphan/` file becomes promotable when it loads cleanly under the
  current `src/`, asserts something the team wants to keep regressing,
  and is not redundant with an existing core test. Move it into `core/`
  in the same commit that fixes any path adjustments.
- A `core/` file becomes archivable when the code path it tests is
  deleted or has been fully subsumed by a newer test. Move it into
  `archived/` with a note in `archived/legacy_README.md`.

Core tests load the installed development package with `using AtmosTransport`
or `import AtmosTransport`. The runner isolates each file's test helpers in its
own module while reusing Julia's package cache. Avoid including the package
source separately in each core test: that repeats compilation and gives each
copy distinct type identities.

The snapshot contract suite was promoted from `orphan/` to
`core/test_output_snapshots.jl` during the current-main output port. It retains
main's signed-total and ATMSNAP-header checks alongside the new selected-capture,
streaming, and asynchronous write-lifetime suites. The opt-in
`diagnostic/test_snapshot_totals_gpu.jl` checks signed cancellation on the
explicitly selected CUDA device (including V100 with CUDA runtime 12.6).

Shared synthetic runtime inputs live in `test/fixtures/`. The window-prefetch
and cubed-sphere file-handoff fixtures are used by both CPU core tests and
explicitly enabled GPU diagnostics. They need no external meteorology files.

The transporting adjoint GPU diagnostic checks unlimited/monotone PPM and
Lin-Rood ORD=5/7 footprints in both precisions, including CPU/GPU agreement,
Float64 directional finite differences, and full/stride/revolve replay. It
also checks single-panel recording and reverse propagation with nonzero halo
seeds. Run it on an explicitly selected device, with scalar indexing disabled:

```bash
ATMOSTR_RUN_TRANSPORT_ADJOINT_GPU_TESTS=1 ATMOSTR_ADJOINT_GPU_NAME=V100 \
CUDA_VISIBLE_DEVICES=<authorized UUID> julia --project=. test/diagnostic/test_cs_transport_adjoint_gpu.jl
```
