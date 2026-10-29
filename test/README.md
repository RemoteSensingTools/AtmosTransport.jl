# Test layout

Tests are organized by tier. The orchestrator is [`runtests.jl`](runtests.jl);
it runs `core/` and `regridding/` by default and accepts opt-in flags to
include other tiers.

| Tier | Folder | When it runs | What's here |
|---|---|---|---|
| **Core** | [`core/`](core/) | Default (CI green-bar) | API/health gates, kernel correctness, runtime composition, adjoint identity, preprocessor contracts, mesh / regridding / vertical-transform invariants, type and aliasing safety. Everything that catches regressions of the production code paths. |
| **Real data** | [`real_data/`](real_data/) | `--all` or `--real-data` | Tests that need preprocessed transport binaries or ERA5 GRIB files on disk in `~/data/AtmosTransport/`. Skipped on CI runners. |
| **Diagnostic** | [`diagnostic/`](diagnostic/) | `--all` or `--diagnostic` | Large numerical sweeps useful for adjoint / inversion debugging (Taylor-sweep, integration footprints). Run occasionally, not every PR. |
| **Regridding** | [`regridding/`](regridding/) | Default (CI green-bar) | Conservative-remapping geometry, conservation, direction, and persistence checks. Its `runtests.jl` is included as one isolated suite by the top-level orchestrator. |
| **Golden** | [`golden/`](golden/) | By hand, before committing a refactor | End-to-end preprocessing and runtime cases compared bit for bit with a recorded reference (`golden/run_goldens.jl`); needs the group's data and a GPU. Not run by `runtests.jl`. |
| **Python** | [`diagnostics/`](diagnostics/) | By hand: `python3 test/diagnostics/test_transcom_partition.py` | `unittest` of the land-weight normalization in `scripts/preprocessing/prepare_transcom_c30.py` (3 cases). Not run by `runtests.jl` or CI. The Python flux port plan ([`docs/memos/2026-10-10_python_flux_port_design.md`](../docs/memos/2026-10-10_python_flux_port_design.md), step P7) ports it to `core/test_surface_flux_regions.jl` and deletes it with the script. |

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
julia --project=test test/runtests.jl --tiers=core,real_data # only listed tiers
```

A failing file does not stop the run: every selected file runs (an interrupt
still stops it), and at the end the runner prints the 15 slowest files and then
fails with the list of failed files, if any.

## Quick set, topics, and the quick gate

The full default run takes 30 to 70 minutes on one process, depending on the
machine and its load. For fast checks while developing,
[`groups.toml`](groups.toml) sorts every `core/` file and the regridding suite
into exactly one topic, and names a quick set:

```bash
julia --project=test test/runtests.jl --quick                     # crucial fast set
julia --project=test test/runtests.jl --topics=advection,diffusion
julia --project=test test/runtests.jl --changed                   # topics of your changes vs origin/main
julia --project=test test/runtests.jl --changed=<git-ref> --quick
julia --project=test test/runtests.jl --quick --list              # print the selection, run nothing
julia --project=. -e 'using Pkg; Pkg.test(test_args=["--quick"])'
```

- `--quick` runs the quick set: the package health gates (Aqua and JET, still
  first), the metadata gates (`test_physical_constants`, `test_readme_current`,
  `test_public_api_surface`, `test_public_docstrings`, `test_public_contracts`,
  `test_environment_variables`, `test_documentation_links`,
  `test_config_schema_agreement`, `test_config_key_checks`), the groups
  meta-test, the source-structure ratchet (`test_structure`), the
  required-keyword call check (`test_keyword_calls`), and
  physics smokes: mass conservation and uniform-tracer invariance of the
  advection kernels, convection and diffusion, short driven lat-lon runs, a
  cubed-sphere run with every operator family across a file handoff, binary
  write/read round trips, and the replay-consistency gates. It takes about
  4 minutes in one process on a lab machine, where the full default run took
  32 minutes.
- `--topics=a,b` runs the files of the named topics (the topic names and
  descriptions are in `groups.toml`).
- `--changed[=<ref>]` (default `origin/main`) maps the files changed on the
  branch since its merge base with `<ref>` (`git diff --name-only <ref>...HEAD`),
  plus uncommitted and untracked files, to topics through each topic's
  `paths` globs. A changed test file selects itself; a changed helper or
  fixture that no topic maps selects every test file that mentions its name
  (the runner and `groups.toml` map to the `health` topic). A changed `src/`
  file that no topic maps runs the full `core/` and `regridding/` tiers, with
  a message naming the file.
- Selectors combine with each other and with the tier flags; the union runs
  (`--quick --diagnostic` adds the diagnostic tier). Without flags the runner
  runs all of `core/` and `regridding/`, as before.

[`core/test_test_groups.jl`](core/test_test_groups.jl) (in the quick set)
fails when a `core/` file is in no topic or in several, a quick or topic
entry does not exist, no topic maps a file of some `src/` folder, or a `paths`
glob matches nothing.

[`scripts/dev/quick_gate.sh`](../scripts/dev/quick_gate.sh) is the check to
run before committing or handing work back:

```bash
scripts/dev/quick_gate.sh               # against origin/main
scripts/dev/quick_gate.sh <git-ref>     # e.g. a stacked branch's base
```

It runs `git diff --check` on the branch's commits and on the uncommitted and
untracked files,
[`scripts/checks/check_keyword_calls.jl`](../scripts/checks/check_keyword_calls.jl)
(calls in `src/`, `ext/`, `test/`, `scripts/`, `benchmarking/`, `examples/`,
`docs/literate/` and `docs/scripts/` that omit a keyword every candidate
method requires, such as `surface_pressure_basis` of the transport-binary
writers), the test-only JuliaSyntax
[`scripts/checks/check_structure.jl`](../scripts/checks/check_structure.jl)
source-structure ratchet (see [`structure/README.md`](structure/README.md)),
and then
`--quick --changed=<ref>` in one Julia process. It prints a summary and exits
non-zero if any step failed. `JULIA="julia +1.10"` selects another Julia.

CI runs `--quick` as a separate fast job for early feedback; the full suite on
Julia 1.10 and the current release stays the merge requirement.

The repository structure test reads `ATMOSTR_STRUCTURE_BASE` (default
`origin/main`). The quick gate uses the same default and exports its selected
ref to the tests; an explicit `<git-ref>` argument takes precedence. CI quick
and full tests, docs, and CPU benchmarks fetch `github.base_ref` for pull
requests, or `main` otherwise, and set `ATMOSTR_STRUCTURE_BASE=origin/<branch>`.
A depth-one fetch suffices for the structure snapshots (`git show <base>:<path>`).
For example, prepare the default base in a shallow checkout with:

```bash
git fetch --no-tags --depth=1 origin '+refs/heads/main:refs/remotes/origin/main'
```

A missing requested ref in a checkout fails the test with fetch instructions.
For package archives without `.git` metadata, the test logs an `@info` and
skips only the comparison against the base. Today's allowlists, folder budgets,
layers, and generated code map are still checked. The quick gate also needs
enough branch history for its merge-base diff and changed-topic selection.

## Adding a new test

1. Decide which tier the test belongs in. Default to `core/` if the test
   doesn't need external data and is part of a production code path.
2. Drop the file into the matching folder. The orchestrator picks up
   anything matching `test_*.jl` automatically — no manual roster edit.
3. Add a `core/` file to exactly one topic in [`groups.toml`](groups.toml)
   (`core/test_test_groups.jl` fails otherwise). If the code it tests lives in
   a file no topic's `paths` cover yet, add the glob. Add it to the quick set
   only if it is a fast, crucial gate.
4. Import the cached package with `using AtmosTransport` or
   `import AtmosTransport`. Each test file runs in its own module to isolate
   helpers and constants without creating new copies of package types.
5. Include shared test fixtures with paths relative to `@__DIR__`.

## Retirement workflow

- Every test file belongs to a tier that some command runs; there is no
  holding area for tests nothing runs. The former `orphan/` tier (tests that
  were not in the CI roster as of 2026-05-29) was emptied on 2026-10-12: of
  its 18 files, 14 moved to `core/` (two after their stale assertions were
  updated) and the unique checks of the other four were folded into core
  files.
- A test file is deleted, not archived, when the code path it tests is
  deleted or a newer test fully subsumes it; git history keeps it. Remove it
  from `groups.toml` in the same commit. The former `archived/` folder (two
  tests of preprocessing scripts deleted in `cc63b87f`) was removed this way.

Core tests load the installed development package with `using AtmosTransport`
or `import AtmosTransport`. The runner isolates each file's test helpers in its
own module while reusing Julia's package cache. Avoid including the package
source separately in each core test: that repeats compilation and gives each
copy distinct type identities.

The snapshot contract suite was promoted from the former `orphan/` tier to
`core/test_output_snapshots.jl` during the current-main output port. It retains
main's signed-total and ATMSNAP-header checks alongside the new selected-capture,
streaming, and asynchronous write-lifetime suites. The opt-in
`diagnostic/test_snapshot_totals_gpu.jl` checks signed cancellation on the
explicitly selected CUDA device (including V100 with CUDA runtime 12.6).

Shared synthetic runtime inputs live in `test/fixtures/`. The window-prefetch
and cubed-sphere file-handoff fixtures are used by both CPU core tests and
explicitly enabled GPU diagnostics. They need no external meteorology files.

The time-varying surface-flux fixtures in
`helpers/surface_flux_window_buffer.jl` (the former full-series operator as a
reference, a small cubed-sphere state on any array type, and a window driver)
serve `core/test_surface_flux_window_buffer.jl` and the opt-in
`diagnostic/test_surface_flux_window_buffer_gpu.jl`, which checks on CUDA that
the host series with its device window buffer gives the full device series'
tracers and Kahan compensation bit for bit:

```bash
ATMOSTR_RUN_SURFACE_FLUX_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=<id> \
    julia --project=. test/diagnostic/test_surface_flux_window_buffer_gpu.jl
```

The transporting adjoint GPU diagnostic checks unlimited/monotone PPM and
Lin-Rood ORD=5/7 footprints in both precisions, including CPU/GPU agreement,
Float64 directional finite differences, and full/stride/revolve replay. It
also checks single-panel recording and reverse propagation with nonzero halo
seeds. Run it on an explicitly selected device, with scalar indexing disabled:

```bash
ATMOSTR_RUN_TRANSPORT_ADJOINT_GPU_TESTS=1 ATMOSTR_ADJOINT_GPU_NAME=V100 \
CUDA_VISIBLE_DEVICES=<authorized UUID> julia --project=. test/diagnostic/test_cs_transport_adjoint_gpu.jl
```

The opt-in CUDA reset diagnostic uses the root project, where CUDA is available:
`ATMOSTR_RUN_CS_ADJOINT_RESET_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=1 julia --project=. test/diagnostic/test_cs_adjoint_reset_gpu.jl`.
It checks that captured `:preserve_vmr` targets stay on the GPU, replay matches
production, and Full/Stride gradients agree. Acquire the assigned GPU lock before
running it.

`core/test_golden_compare.jl` tests the golden comparator with tiny synthetic
NetCDF and transport-binary files, including NaN payloads and signed zero.
It calls the comparison/report functions without running any golden cases.
`core/test_operator_defaults.jl` checks all four optional operator defaults
on lat-lon, reduced-Gaussian and cubed-sphere models and in `StepOperators`.
JuliaSyntax uses the same exact version pins in the root test extras and
this folder's project; it is not a runtime dependency.
