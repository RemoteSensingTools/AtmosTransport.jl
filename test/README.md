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

Visualization, Analysis post-processing, DataDownloads and the Makie extension
are non-core (`src/Extras/`, plus `ext/AtmosTransportMakieExt.jl`) and have no
tests. The `extras` topic has an empty `files` list to map those paths without
selecting core tests. Module-folder globs cover both sides of a source move,
so deleted old paths do not trigger the full-suite fallback; moved budget
helpers still select core diagnostics. The `diagnostics` topic keeps the core mass-budget,
snapshot-reader, mass-drift and decay-closure tests in `src/Diagnostics/`.

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
ref to the tests; an explicit `<git-ref>` argument takes precedence. The CI
quick and full test jobs fetch `github.base_ref` for pull requests and set
`ATMOSTR_STRUCTURE_BASE=origin/<branch>`. On a push to `main` they use the
commit before the push (`github.event.before`, or `HEAD^1` when that commit is
all zeros or cannot be fetched), so a merge validates the migration rows it adds. The docs and benchmark workflows
also set the variable but do not run the structure check.
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

The structure test's historical-ratchet fixture reads existing repository
history through a temporary Git pointer. It does not initialize a repository,
stage files, create commit objects, or update refs. In a package archive it
visibly skips that fixture; synthetic allowance-growth checks still run.
The runner selector fixture supplies changed paths through a read-only Git
stub; it also needs no staging, commits or ref updates. The quick gate itself
uses real read-only diffs to select this checkout's tests.

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

The inversion topic keeps algebraic adjoint identities, gradient Taylor sweeps,
preconditioned finite differences and control-shape checks together in
`core/test_cs_gradient_checks.jl`, using one nonzero-flow problem. The mixed
placed/window/retrieval-column reference and cost finite differences run in
`core/test_cs_4dvar_retrieval_columns.jl`; their independent TM5 rebin and hat
projection live in `helpers/cs_inversion_reference.jl`. Upward Upwind vertical
flux and FV3 Lin–Rood footprint differences share the matrix in
`core/test_cs_linrood_fv3_adjoint.jl`. Tape storage values, rejections and the
mmap tier's space reservation are in `core/test_cs_tape_storage.jl`.

The adjoint tape's bitwise gate, `core/test_cs_adjoint_tape_bitwise.jl` (about
2.5 minutes, adjoint footprint topic), runs `helpers/cs_adjoint_tape_harness.jl`
in a `--threads=1` subprocess: every tape storage and checkpoint schedule must
reproduce the device `FullCheckpoint` gradients bit for bit on rough fields
that reach every limiter branch, and on the reference host the reference
digests must equal `fixtures/cs_adjoint_tape_digests.toml`. A change that is
meant to keep gradients bit for bit must pass it unchanged. The full matrix is
the opt-in `diagnostic/test_cs_adjoint_tape_bitwise_full.jl` (about 10
minutes). To A/B a branch against main on any host (any tier), or to
re-record the core fixture on the reference host:

```bash
F="--startup-file=no --threads=1 --check-bounds=auto -O2 --inline=yes --code-coverage=none"
H=test/helpers/cs_adjoint_tape_harness.jl
# in a main checkout (with this harness file):
julia $F --project=test $H --tier=full --reference-only --record=main.toml
# on the branch:
julia $F --project=test $H --tier=full --check=main.toml
# re-record the fixture (core tier) on the reference host:
julia $F --project=test $H --tier=core --reference-only --record=test/fixtures/cs_adjoint_tape_digests.toml
```

Only main's numerics may be recorded: record from a checkout of current main
(for example after a dependency or Julia update on the reference host), never
from a branch whose gradients differ from main without being meant to. A PR
that changes gradients on purpose (a convection, diffusion or advection
change, which also moves these digests) first runs `--check` against the
current fixture, lists the quantities that change in its description as its
delta (the harness prints one line per changed case, entry point and
quantity), and then re-records the fixture in its last commit from the PR
rebased on current main, which is what main will be after the merge. The
fixture's header names the commit it was recorded from; update it with the
recording.

Models IC checks are in `core/test_initial_condition_io.jl`; surface-inventory
calendar, coverage and species checks are in `core/test_surface_flux_ingestion.jl`,
unit conversion in `core/test_flux_units.jl`, native storage/time encoding in
`core/test_native_cs_surface_flux.jl`, and CS packing in
`core/test_cs_initial_state_packing.jl`. The C90 CATRINE near-surface remap
invariant is opt-in at `real_data/test_catrine_initial_condition.jl`.
