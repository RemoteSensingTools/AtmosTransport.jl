# AtmosTransport Benchmarks

This directory contains the production benchmark harness for tracking
AtmosTransport performance over time. It follows the same broad pattern as the
Oceananigans benchmark dashboard: Julia produces raw benchmark records plus a
`github-action-benchmark` compatible JSON file, and one GitHub Actions
publisher appends them to the dashboard history on the `gh-pages` branch
(`benchmarks/data.js`, rendered by
[`dashboard/render_grouped_dashboard.py`](dashboard/render_grouped_dashboard.py)).

Two lanes feed the dashboard:

- **CI lane**: tiny CPU smokes on GitHub-hosted runners for every pull request
  and push to `main` (`.github/workflows/Benchmarks.yml`).
- **Lab lane**: production-scale GPU numbers measured on the lab machines
  (wurst, curry, the owner's MacBook) by
  [`lab/run_lab_benchmarks.jl`](lab/run_lab_benchmarks.jl), run by hand from
  wurst and published through the same workflow. This is how GPU numbers are
  produced; see [Lab lane](#lab-lane).

## Environment

Prepare the environment from the repository root, including on Julia 1.10
where `[sources]` is ignored:

```bash
julia --project=benchmarking -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

AtmosTransport is sourced from this checkout without a duplicated release
version bound, so the benchmark environment stays usable after version bumps.
On Julia 1.11 and later `Pkg.instantiate()` alone is enough. The lab lane
relies on that: every lab machine runs the exact Julia version pinned in
`lab/machines.toml`, 1.11 or later (see [Lab lane](#lab-lane)).

## Running the harness

```bash
julia --project=benchmarking benchmarking/run_benchmarks.jl \
  --backend=cpu \
  --float-type=Float32,Float64 \
  --grid=C24,C48 \
  --levels=32 \
  --tracers=1,4 \
  --operator=advection,diffusion,convection,full,io,adjoint \
  --steps=5 \
  --repeats=5 \
  --output=benchmark_results.json
```

The run measures every combination of float type, grid, levels, tracers,
operator, and advection scheme on synthetic cubed-sphere fields (no
meteorology input), with two exceptions so that every series name says what
it measured
(`case_matrix` in [`src/case_options.jl`](src/case_options.jl)):

- `adjoint` runs at one tracer whatever `--tracers` lists: it times the
  reverse-mode footprint of one scalar objective of one tracer, whose cost does
  not depend on the tracer count (`SINGLE_TRACER_OPERATORS`).
- `diffusion`, `convection`, `io`, and `adjoint` never run the `--scheme`
  advection scheme, so they run once, under the default `ppm`, whatever
  `--scheme` lists (`SCHEME_FREE_OPERATORS`).

The `io` payload carries every requested tracer. Any grid `C<N>` with
`N >= 3` (the meshes have a 3-cell halo) and any level count are accepted,
including the production sizes C90 to C720 with L72/L137; a single level has
no vertical flux.

The synthetic fields ([`src/synthetic_fields.jl`](src/synthetic_fields.jl))
are smooth air-mass and tracer fields and static face mass fluxes that leave
every cell's air mass unchanged by a step: on each panel a closed circulation
(the discrete curl of a stream function that vanishes on the panel edges, up
to 5 % of a cell's mass per sweep) plus a weaker divergent part that changes
sign with height, with vertical fluxes that close continuity. Every step of a
case therefore does the same work at the same palindrome Courant number,
about 0.1 on every grid (0.10 to 0.12 from C4 to C180, L32 to L137), and no
case subcycles. The harness checks this: a case whose Courant number exceeds
0.5 (`MAX_SYNTHETIC_COURANT`) when built, after the warmup, or after the timed
steps, or whose air mass drifts by more than 0.1 %, fails with a message
naming the case. The fields are evaluated face by face from 1-D factors, so
building them needs no memory beyond the field arrays themselves.

| Option | Meaning |
|---|---|
| `--backend` | `cpu`, `cuda`, or `metal`; Metal runs are `Float32` only |
| `--preset` | named case matrix from [`presets/`](presets/) (e.g. `lab`) or a preset `.toml` path |
| `--float-type`, `--grid`, `--levels`, `--tracers`, `--operator`, `--scheme` | comma-separated case lists |
| `--steps`, `--warmup-steps`, `--repeats` | timing: steps per timed repeat, untimed warmup steps, timed repeats |
| `--convection-depth` | bottom levels in the TM5 convection matrix (`lmax_conv`), clipped to each case's levels; `0` = full column |
| `--group` | dashboard group, the first field of every series name |
| `--device-label` | GPU device label for series names (overrides the normalised device name) |
| `--device-memory-gib` | device memory available to each case, overriding the device query (see [Device memory](#device-memory)) |
| `--commit` | commit recorded in each result; needed when running from a `git archive` without `.git` |
| `--output` | raw JSON path; the dashboard JSON is written next to it |

Case options resolve in this order: command line, then the preset, then the
built-in defaults in [`src/case_options.jl`](src/case_options.jl)
(`DEFAULT_CASE_OPTIONS`). `julia --project=benchmarking
benchmarking/run_benchmarks.jl --help` lists them with their defaults.

Convection cases use the collaborative TM5 LU solve (`use_collab_lu = true`).
On Float32 GPUs that solve requires a matrix depth of at most 85 levels, so
L137 convection needs a `--convection-depth` of at most 85; the `lab` preset
uses 85, as the production ERA5 L137 run does. Float64 CUDA collaborative
solves support at most 73 levels; deeper Float64 cases fall back to the
per-thread solve with a warning, as a production run with that setting would.

### Presets

A preset fixes a case matrix so runs stay comparable over time.
[`presets/lab.toml`](presets/lab.toml) is the lab preset:

| Setting | Value |
|---|---|
| grids | C90, C180, C360, C720 |
| levels | 72 (GEOS/MERRA-2), 137 (ERA5) |
| tracers | 2 (the adjoint runs at 1) |
| operators | advection, diffusion, convection, full, io, adjoint |
| limits | io and adjoint up to C180 |
| scheme | ppm |
| timing | 5 steps × 5 repeats after 2 warmup steps |
| convection depth | 85 bottom levels (L72 keeps the full column) |
| group | `Lab Synthetic CS Sweep` |

The float types come from the machine (see
[`lab/machines.toml`](lab/machines.toml)): Float32 on wurst (L40S) and the Mac
(Metal), Float32 and Float64 on curry (A100). Per float type that is 4
model operators × 4 grids × 2 level counts + (`io` + `adjoint`) × 2 grids ×
2 level counts = 40 cases: 40 on wurst and the Mac and 80 on curry.

The preset's `[limits]` table sets the largest grid an operator runs on
(`[limits.io] max_grid = "C180"`); `case_matrix` leaves that operator's cases
on larger grids out, so the case list is fixed by the preset. At C720 L137
`io` would serialize 10.4 GB (Float32) or 20.8 GB (Float64) per repeat, and
the adjoint tape grows with the grid. Limits are typed: an unknown operator,
an unknown key, or an invalid grid is an error.

Changing the grids, levels, tracers, operators, or limits adds or ends
series. Changing steps, repeats, scheme, or convection depth changes what an
existing series measures; do that together with a new `group`.

The `io` case serializes the air mass, every tracer, and the three face-flux
arrays into `TMPDIR` once per repeat: up to 0.68 GB in Float32 and 1.37 GB in
Float64 (C180 L137, 2 tracers), removed after each repeat. Host arrays are
serialized in place; device arrays are first copied to the host.

### Device memory

Before it builds a case, the harness estimates the case's device memory
(`estimate_case_bytes` in [`src/memory_model.jl`](src/memory_model.jl)) and
compares it with the memory the device has free: CUDA's available memory,
Metal's recommended working set minus its current allocations, or free host
memory on the CPU (`--device-memory-gib` overrides the query). A case that
does not fit is not run: it is recorded with `status = "skipped_memory"`,
a `skip_reason`, and `required_device_gib` and `available_device_gib`, and
the run continues. An out-of-memory error while a case is built or run
(Julia's `OutOfMemoryError`, CUDA's `OutOfGPUMemoryError`, a Metal allocation
failure) frees the memory (`GC.gc`, `CUDA.reclaim`) and records the case as
`skipped_memory` too, so one large case cannot end the run. Skipped cases are
never published; the lab lane accepts and lists them.

Metal reports a failed kernel differently. Metal.jl 1.9 (the version in the
benchmark Manifest) runs each kernel launch in its own command buffer and
reports a failed buffer only by printing `ERROR: Failed to submit command
buffer: <description>` from the buffer's completion handler. Nothing is
thrown, so a kernel that never ran would leave an `ok` record. While a Metal
case runs, the harness therefore redirects standard output to a temporary
file ([`src/device_errors.jl`](src/device_errors.jl)). After the case it
commits an empty sentinel command buffer on the queue the case's kernels used,
with a completion handler (`MTL.on_completed`) that sets a flag, and waits for
the flag. The wait times out after 600 s and then fails the case. Then it
reads and forwards the text. If every failed buffer ran out of memory
(`Insufficient Memory (...ErrorOutOfMemory)`), the case is recorded as
`skipped_memory`; any other failure records it as `failed`, with all the
reported errors and their counts.

The barrier relies on Metal behaviour that Apple does not document
explicitly: a queue completes its command buffers in commit order and runs
their completion handlers one at a time in that order. When the sentinel's
handler runs, the handlers of the case's buffers have therefore returned.
Work committed on another queue or from another task would not be covered;
the benchmark cases launch all their kernels from the harness task.

The estimate is a linear model of the arrays a case allocates, fitted on CPU:
the summed size of every array reachable from the model after construction
and one step (`Base.summarysize`), for C8-C64, L16 and L40, and 1 and 3
tracers. Counted in elements of the float type, with `V = 6 (Nc+6)² Nz`
haloed cells, `Vi = 6 Nc² Nz` interior cells, `A = 6 (Nc+6)²` haloed columns,
and `Nt` tracers:

| Operator | per `V` | per `V × Nt` | per `Vi` | per `A` |
|---|---|---|---|---|
| advection | 5.91 | 2.20 | 0 | 1.5 |
| diffusion | 4.11 | 1.03 | 1.85 | 3.1 |
| convection | 4.17 | 1.00 | 3.84 | 2.3 |
| full | 5.82 | 2.22 | 5.43 | 4.0 |

The coefficients are the fitted values rounded up, and they are the same in
Float32 and Float64. Convection and full add the TM5 column tile (at most
1 GiB, `tm5_legacy_tile_bytes`) when the collaborative GPU solve does not
engage: on the CPU and for Float64 deeper than 73 levels. `io` needs exactly
its payload on a GPU. On the CPU, where the device is the host, it needs the
payload twice, because the read-back copy is allocated while the payload is
live. `adjoint` needs its payload plus `(3 + 8.5 × steps) V` elements,
which bounds its measured allocations. Every case adds 16 MiB. Checked against
the measured sizes, the estimate is never lower and is 1-17 % higher at C90 and
C180 L72 (2 tracers). A case is skipped when `1.05 × estimate + 0.5 GiB`
exceeds the free memory. The margin covers kernel temporaries and pool
fragmentation. The model was fitted on CPU; the GPU allocations of the
lab cases are not yet measured.

Required memory of the largest lab cases (GiB, with the margin) and the
expected outcome:

| Case | Float32 | Float64 |
|---|---|---|
| C720 L72 full | 14.5 | 28.4 |
| C720 L137 advection | 18.0 | 35.5 |
| C720 L137 convection | 17.4 | 35.3 (TM5 tile included) |
| C720 L137 full | 27.0 | 54.5 (TM5 tile included) |

The first lab run (commit `2c56189e`) found 44.0 GiB free on the L40S,
38.6 GiB on the A100 40 GB, and 51.8 GiB of Metal working set on the 64 GB
M5 Pro, so every Float32 case fits. On curry, Float64 C720 L137 `full` was
skipped, so 79 of 80 cases ran, as the estimate predicted. Float64 C720 L137 `advection` and `convection` are within 4 GiB of
the A100's free memory. The harness builds a case's fields on the host and
then copies them to the device. On the Mac's unified memory that host copy
(about 10 GB for C720 L137 Float32) shares the 64 GB with the device arrays.
The copy is freed before the timed steps but is not part of the check.

## Series names

Each case is one dashboard series named

```text
<group> / C<N> L<levels> <operator> <tracers>tr[ <scheme>] / <backend label> / <float type>
```

for example `Lab Synthetic CS Sweep / C180 L137 full 12tr / GPU-CUDA A100 / Float64`.
The scheme appears only when it is not `ppm`, so existing series keep their
names. The group may not contain `/`. Operator phases are separate series
named `Operator Breakdown / <case name> / <phase>`, for example
`Operator Breakdown / Lab Synthetic CS Sweep / C90 L72 full 12tr / GPU-CUDA L40S / Float32 / convection`,
so every case field (group, grid, levels, operator, tracers, scheme, backend,
float type) separates phase series too.

Backend labels:

- CPU: the hardware class alone, `CPU-AMD`, `CPU-Intel`, or `CPU-Apple`, so the
  CI history continues.
- GPU: the class plus the device, so devices are separate series:
  `GPU-CUDA L40S`, `GPU-CUDA A100`, `GPU-Metal M5 Pro`.

The GPU device label is derived from the driver's device name
(`normalize_device_label` in [`src/labels.jl`](src/labels.jl)):

1. drop the vendor prefix (`NVIDIA`, `Apple`, `AMD`, `Intel`) and a product
   line prefix (`GeForce`, `Tesla`, `Quadro`);
2. drop a data-centre board suffix (`-PCIE`, ` PCIe`, `-SXM4`, `-NVL`, with an
   optional `-40GB`) and a trailing memory size (`80GB`, `80GB HBM3`);
3. replace `/` (the series separator) with `-` and collapse whitespace.

So `NVIDIA L40S` → `L40S`, `NVIDIA A100-PCIE-40GB` → `A100`, `Apple M5 Pro` →
`M5 Pro`. Board variants of one chip collapse to one label; to keep them apart
pass `--device-label` (for example `--device-label="A100 80GB"`, or
`device_label` in `machines.toml`). The full device name is kept in each raw
record's `device_name` and in the dashboard record's `extra.device`.

## Outputs

`--output=benchmark_results.json` writes two files:

- `benchmark_results.json`: one raw record per case with `status` (`ok`;
  `failed` with `error`; or `skipped_memory` with `skip_reason`), the
  estimated `required_device_gib` and the `available_device_gib`, the case
  parameters, `backend_class`,
  `backend_label`, `device_name`, the effective `convection_depth`, timings
  (median `time_per_step_seconds`, `mad_seconds`, throughput), the timing
  protocol (`steps_per_repeat`, `repeats`, `warmup_steps`),
  `phase_times_seconds`, and `metadata` (`commit`, `preset`, Julia and
  AtmosTransport versions, threads, architecture, UTC timestamp).
  `steps_per_repeat` is the number of steps one timed repeat covers:
  `--steps` model steps, one adjoint run over `--steps` steps, or for `io`
  one write-and-read round trip (`timed_steps_per_repeat` in
  [`src/records.jl`](src/records.jl)). `phase_times_seconds` are totals over
  all timed repeats (section-timer sections for the model operators, the
  whole round trip or adjoint run for `io` and `adjoint`).
- `benchmark_results.github-action-benchmark.json`: the dashboard records
  (`name`, `unit = "s/step"`, `value`, `extra`) built by `dashboard_records`
  in [`src/records.jl`](src/records.jl). A phase record's value is its total
  divided by `steps_per_repeat × repeats`. `extra` is a JSON string with
  `commit`, `device`, throughput, and the environment fingerprint;
  github-action-benchmark stores `extra` as text.

Every raw record (`metadata`) and every dashboard `extra` carries the
**environment fingerprint**: `julia_version` and `manifest_sha256`, the
sha256 of the `benchmarking/Manifest.toml` the run resolved. Equal
fingerprints mean the same Julia and the same package versions. A Julia
upgrade or a new dependency resolution changes the fingerprint and can change
timings: on the dashboard such a change is a discontinuity, not a regression
of the code, and the grouped dashboard shows each point's fingerprint in its
tooltip and in the latest-value table.

A case that throws is logged, kept in the raw JSON as `failed`, left out of the
dashboard JSON, and makes the run exit with an error after both files are
written, so one failed case does not discard an hour of lab results. A case
that does not fit in device memory is `skipped_memory` and does not fail the
run (see [Device memory](#device-memory)).

## Lab lane

The lab lane benchmarks one commit on the lab machines and publishes the
result. It is the only source of GPU numbers: self-hosted GitHub runners are
deliberately not used, because on this public repository a pull request,
including one from a fork, can change the workflow a runner executes. The lab
lane runs only commits already on `main`, from an operator's shell, and sends
GitHub nothing but a JSON file on the `benchmark-inbox` branch and a workflow
dispatch. Run it on wurst from a clean checkout whose benchmark environment is
instantiated (see [Environment](#environment)):

```bash
# Preview every command (no ssh, no GPU work; read-only git queries still run)
julia --project=benchmarking benchmarking/lab/run_lab_benchmarks.jl --dry-run

# Benchmark origin/main HEAD on all machines and publish the results
julia --project=benchmarking benchmarking/lab/run_lab_benchmarks.jl \
  --machines=wurst,curry,mac --publish
```

The lab lane needs only Base, TOML, JSON3, and Dates (it never loads
AtmosTransport locally); `--help` lists all options.

What it does:

1. Resolve the commit: `origin/main` HEAD after `git fetch` unless `--commit`
   is given. It refuses a dirty working tree (tracked files) and a commit that
   is not on `origin/main`, unless `--allow-unpublished`, which keeps results
   local and cannot be combined with `--publish`. `--publish` additionally
   requires the commit to be the head of `origin/main`.
2. For each selected machine, in order (`--machines=wurst,curry,mac`; default:
   all, alphabetically):
   - deliver the code as a `git archive` of the commit into
     `~/atmostr_bench/<commit>/`: extracted once on wurst for machines that
     share the NFS home (wurst, curry), streamed over ssh for the Mac;
   - check that the machine's Julia is exactly its pinned `julia_version`
     and instantiate the benchmark environment there (`Pkg.instantiate()`).
     The archive has no `benchmarking/Manifest.toml` (it is gitignored), so
     the first instantiation of a commit's work directory resolves one with
     the machine's current registry (and `AtmosTransport` through the
     `[sources]` entry, which Julia 1.10 ignores); later runs of the same
     commit reuse it. wurst and curry share their work directory and so its
     Manifest;
   - run `run_benchmarks.jl --preset=lab` with the machine's backend, float
     types, threads, heap-size hint, environment, and `--commit`;
   - copy `benchmark_results.json` and the dashboard JSON back into
     `benchmarking/lab/results/<commit>/<machine>/` (gitignored). This also
     happens when the run failed: the harness writes both files before it
     exits with an error for a failed case, so the partial results come back
     for inspection, and the run's failure is reported afterwards;
   - validate them: the preset's case count, each case `ok` or
     `skipped_memory` with a reason (skipped cases are listed and must not be
     in the dashboard JSON) and none failed, every record and dashboard entry
     attributed to the commit, the machine's backend and float types, one
     device label of the right class, and one environment fingerprint whose
     Julia version is the machine's pin.

   A machine that fails (unreachable, failed case, invalid results) is
   reported and the remaining machines still run; nothing is published.
3. Merge all dashboard records into
   `results/<commit>/lab-<utc>.github-action-benchmark.json`; two machines
   producing the same series name (same device label) is an error, and so is
   a difference between the machines' environment fingerprints: nothing is
   merged or published. The Mac resolves its own Manifest, so a registry
   change between its resolution and wurst's shows up here; remove
   `~/atmostr_bench/<commit>/benchmarking/Manifest.toml` on wurst (shared
   with curry) and on the Mac and rerun, so both resolve afresh.
4. With `--publish`: re-check that `origin/main` still points at the commit,
   commit the merged file as `lab/<commit>-<utc>.json` on the orphan branch
   `benchmark-inbox` (git plumbing with a temporary index; the working tree is
   untouched), push it, and dispatch the Benchmarks workflow on `main` with
   `/usr/bin/gh workflow run Benchmarks.yml -f publish=true -f run_cpu=false
   -f lab_results=lab/<commit>-<utc>.json`.

The workflow's publish job fetches that file, checks that every record was
measured on the workflow's commit (`github.sha`) and in one environment
(fingerprint), and publishes it with the
same github-action-benchmark step as CI results. If `main` moved during a long
lab run the results cannot be attributed and stay local; benchmark the new head
instead. If the dispatched run is cancelled by the workflow's concurrency group
(a newer `main` run queued behind it), re-dispatch with the same `lab_results`
while `main` still points at the commit.

### Machines

[`lab/machines.toml`](lab/machines.toml) has one table per machine; unknown
keys are errors. Keys: `access` (`local` or `ssh` with `ssh_host`), `julia`
(absolute or `~/` path, since ssh shells do not load the interactive `PATH`),
`julia_version` (the exact Julia version `julia` must report, 1.11 or later;
currently 1.12.6 on all three machines),
`backend`, `float_types`, `threads`, optional `heap_size_hint`, `tmpdir`,
`device_label`, `caffeinate` (wrap long commands in `caffeinate -i` on macOS),
`code = { delivery, root }` (`shared` for an NFS path seen identically by the
machine, `copy` to stream the archive over ssh; the root's last component must
start with `atmostr_`), and `env` (e.g. `CUDA_VISIBLE_DEVICES`).

| Machine | Access | GPU | Float types | Code |
|---|---|---|---|---|
| wurst | local | NVIDIA L40S (48 GB), `CUDA_VISIBLE_DEVICES=0` | Float32 | shared `~/atmostr_bench` |
| curry | `ssh curry` | NVIDIA A100-PCIE-40GB, `CUDA_VISIBLE_DEVICES=0` | Float32, Float64 | shared `~/atmostr_bench` (NFS home) |
| mac | `ssh` to the owner's MacBook | Apple M5 Pro (Metal) | Float32 | copied to `~/atmostr_bench` there |

The Julia pin is deliberate: upgrading Julia on a lab machine makes the lab
lane stop with a version error until `julia_version` is updated. Update it
on all machines together, in its own commit; the new fingerprint marks the
change on the dashboard.

On the Mac everything the lab lane writes stays under `~/atmostr_bench`: the
copied code, the io case's `TMPDIR` (`~/atmostr_bench/tmp`), and the Julia
depot (`JULIA_DEPOT_PATH = "~/atmostr_bench/julia_depot:"`, so packages,
registries, and compile caches do not go to `~/.julia`; the first
instantiation there downloads and precompiles everything once). Long commands
run under `caffeinate -i`.

Per-commit work directories under `~/atmostr_bench/` (and the Mac's copy) are
not removed automatically; delete old ones by hand.

## CI workflow

`.github/workflows/Benchmarks.yml`:

- pull requests: C4 CPU smoke in Float32, not published;
- pushes to `main`: C4 CPU smoke in Float32 and Float64, published;
- manual dispatch: CPU sweep over the `grid`, `operators`, and
  `cpu_float_types` inputs (`run_cpu`, default `true`) and optional lab
  results (`lab_results`), published when `publish=true`.

All jobs run on GitHub-hosted runners. The CPU job, which runs pull-request
code, has a read-only token. The publish job, which never runs on pull
requests, has `contents: write` (push the dashboard data to `gh-pages`; the
`comment-on-alert` commit comment) and `deployments: write`. It merges the
CPU and lab records into one input for github-action-benchmark, then
re-renders the grouped dashboard page.
