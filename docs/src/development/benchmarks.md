# [Performance benchmarks](@id performance-benchmarks)

AtmosTransport times its operators with a synthetic cubed-sphere benchmark
suite in the
[`benchmarking`](https://github.com/RemoteSensingTools/AtmosTransport.jl/tree/main/benchmarking)
directory and publishes the history on the
[performance dashboard](https://remotesensingtools.github.io/AtmosTransport.jl/benchmarks/).
The layout follows the Oceananigans.jl benchmark dashboard: a Julia harness
writes raw benchmark records plus a file in the format of
[github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark),
and a GitHub workflow appends those records to the dashboard data on the
`gh-pages` branch.

## What is measured

A benchmark case builds a [`TransportModel`](@ref) on a synthetic cubed
sphere and times its step; the `io` and `adjoint` cases time the same fields
without a model (see [Operators](@ref benchmark-operators)). No meteorology
files are read, so the timings isolate the operators from data loading:

- **Grid.** A [`CubedSphereMesh`](@ref CubedSphereMesh) in the GEOS-native
  panel convention at resolution `C<N>` (`--grid`, comma-separated), with
  uniform sigma levels (`--levels`).
- **State and fluxes.** Dry-basis air mass and `--tracers` tracers with smooth
  analytic patterns, and static face mass fluxes that leave every cell's air
  mass unchanged by a step, so every step of a case does the same work at the
  same Courant number. On each panel the horizontal fluxes are a closed
  circulation (the discrete curl of a stream function that vanishes on the
  panel edges, moving up to 5 % of a cell's mass per sweep) plus a weaker
  divergent part that changes sign with height; the vertical fluxes close
  continuity with it. The resulting palindrome Courant number is about 0.1 on
  every grid, so no case subcycles. The harness refuses a case whose Courant
  number exceeds 0.5 or whose air mass drifts by more than 0.1 % over its
  steps.
- **Backend and precision.** `--backend=cpu`, `cuda`, `rocm`, or `metal`, and
  `--float-type=Float32,Float64`. Metal runs only `Float32`.

The exact setup is in
[`benchmarking/src/AtmosTransportBenchmarks.jl`](https://github.com/RemoteSensingTools/AtmosTransport.jl/blob/main/benchmarking/src/AtmosTransportBenchmarks.jl).

### [Operators](@id benchmark-operators)

| `--operator` | One timed step runs | Throughput metric |
|---|---|---|
| `advection` | Cubed-sphere advection with the scheme given by `--scheme`: the split-sweep schemes `ppm` (default, `PPMScheme(MonotoneLimiter())`), `upwind`, and `slopes`, or the Lin–Rood schemes `linrood5` and `linrood7` | cell-tracer updates per second |
| `diffusion` | Implicit vertical diffusion with a constant diffusivity, no advection | cell-tracer updates per second |
| `convection` | `TM5Convection` driven by constant updraft entrainment and detrainment in the middle third of the convective column; no advection. The collaborative LU solve is requested as in production runs (`use_collab_lu = true`) and used on supported GPUs: `Float32` on CUDA, ROCm, and Metal for convection depths up to 85 levels, `Float64` on CUDA up to 73. The CPU, and `Float64` cases deeper than 73 levels (the L137 lab cases with depth 85), use the per-thread solve. `--convection-depth` restricts the convection matrix to the bottom levels, as production ERA5 L137 runs do. | cell-tracer updates per second |
| `full` | Advection, vertical diffusion, and convection together | cell-tracer updates per second |
| `io` | Serializing and reading back a host copy of the air mass, all `--tracers` tracers, and the three face-flux arrays | bytes per second |
| `adjoint` | One forward and reverse run of `cs_surface_emission_footprint` over `--steps` steps (PPM without limiter, tape on the device) | adjoint runs per second |

Every case is named after what it measured. The adjoint footprint belongs
to one scalar objective of one tracer, so its cost does not depend on the
tracer count: `adjoint` runs as a single one-tracer case whatever
`--tracers` lists. `diffusion`, `convection`, `io`, and `adjoint` never run
the `--scheme` advection scheme, so they run once, under the default `ppm`.

### Timing protocol

1. Build the case and run `--warmup-steps` untimed steps. The first step
   compiles the kernels, so keep at least one.
2. Run `--repeats` timed repeats of `--steps` steps each. The backend is
   synchronized before and after every repeat, so GPU work is fully counted.
   The time per step of a repeat is its wall time divided by `--steps`.
3. Report the median over repeats and the median absolute deviation (MAD).

For `io`, one repeat writes and reads the payload once, and that round trip
is the case's step. For `adjoint`, one repeat is a full forward and reverse
run, and the time per step is that time divided by `--steps`.

While the repeats run, the package's section timer (`SectionTimer`) records
the time spent in each instrumented part of the step, such as the transport
block, its x, y, and z sweeps, halo exchanges, and convection. These become
the per-operator breakdown series: a section's total over all timed repeats
divided by the number of steps they covered. The `io` and `adjoint` cases
have one phase each, the whole round trip or the whole forward and reverse
run, divided the same way.

### Metrics

| Metric | Definition |
|---|---|
| Time per step (`time_per_step_seconds`, s/step) | Median over repeats of the wall time of one step. This is the dashboard value; smaller is better. |
| `mad_seconds` | Median absolute deviation of the per-step times, a measure of run-to-run noise. |
| Steps per second | ``1 /`` time per step. |
| Cell-tracer updates per second | ``6 N_c^2 N_z N_\mathrm{tr}`` divided by the time per step: the number of tracer values (all six panels, all levels, all tracers) advanced per second of wall time. Larger is better; it compares grids and tracer counts on one scale. |
| Bytes per second (`io`) | Size of the serialized payload divided by the median write-and-read time. |
| Adjoint runs per second (`adjoint`) | ``1 /`` median time of one forward and reverse run. |

A `C90` grid with 72 levels has ``6 \times 90^2 \times 72 = 3\,499\,200``
cells, so one step with four tracers advances about ``1.4 \times 10^7``
cell-tracer values.

### Records

`run_benchmarks.jl` writes two files. The raw JSON (`--output`) holds every
case's status, settings, timings, section times, backend label, and device
name, plus metadata: UTC timestamp, Julia and AtmosTransport versions, git
commit, preset, and the sha256 of the resolved benchmark Manifest. The
second file, with
the suffix `.github-action-benchmark.json`, holds the dashboard records. Their
series names have the form

```text
<group> / C<N> L<levels> <operator> <tracers>tr / <backend class> / <float type>
```

for example `CPU Synthetic CS Sweep / C24 L32 advection 4tr / CPU-AMD / Float32`.
The case field names the advection scheme only when it is not the default
`ppm`. The backend field of a CPU series is its hardware class (`CPU-AMD`,
`CPU-Intel`, or `CPU-Apple`). The backend field of a GPU series also names
the device, for example `GPU-CUDA L40S`, `GPU-CUDA A100`, or
`GPU-Metal M5 Pro`, so every GPU keeps its own history. The full processor or
GPU name is stored with each record. Series named
`Operator Breakdown / <case name> / <phase>` hold the section times of that
case. The naming rules are in
[`benchmarking/src/labels.jl`](https://github.com/RemoteSensingTools/AtmosTransport.jl/blob/main/benchmarking/src/labels.jl).

Every record also stores its **environment fingerprint**: the Julia version
and the sha256 of the resolved `benchmarking/Manifest.toml`, which together
fix the package versions. A change of fingerprint between two points of a
series, such as a Julia upgrade or a newer dependency, can move the timings
by itself; read it as a discontinuity, not as a code regression. The
dashboard shows each point's fingerprint.

## Lab machines

Hosted CI runners have no GPUs, so GPU performance is measured on three lab
machines. A benchmark case uses one GPU.

| Machine | Hardware | Device memory | Backend | Float types | Role |
|---|---|---|---|---|---|
| wurst | 2 × NVIDIA L40S | 48 GB per GPU | CUDA | `Float32` | Float32 gate. Production transport runs in `Float32` on this GPU class, which executes `Float64` at a small fraction of its `Float32` rate (see [Tracer Conservation in Single Precision](@ref)). |
| curry | 2 × NVIDIA A100-PCIE-40GB | 40 GB per GPU | CUDA | `Float32`, `Float64` | Data-centre GPU with fast double precision: the `Float64` series and a second `Float32` device. |
| MacBook | Apple M5 Pro | 64 GB unified | Metal | `Float32` | The Metal backend, a preview (see the status tables in the repository `README.md`). Metal has no `Float64`. |

The lab preset runs C90, C180, C360, and C720 grids with 72 and 137 levels and
two tracers. `io` and `adjoint` run only up to C180, which gives 40 cases per
float type. Before a case is built, the harness estimates its device memory
from a linear model of the model's arrays, fitted on CPU. A case that would
not fit, or that runs out of memory, is recorded as skipped rather than
failing the run, and is not published. With the current estimate, every
`Float32` case fits on all three machines. On the A100, the `Float64` C720
L137 `full` case (about 55 GiB) is skipped. The `benchmarking/README.md`
section *Device memory* gives the model and the memory of the largest cases.

## How results reach the dashboard

Results arrive by two routes; neither runs a GitHub Actions runner on lab
hardware.

**CI smokes.** The `Benchmarks` workflow
([`.github/workflows/Benchmarks.yml`](https://github.com/RemoteSensingTools/AtmosTransport.jl/blob/main/.github/workflows/Benchmarks.yml))
runs tiny CPU cases of every operator on GitHub-hosted runners for each pull
request and each push to `main`; pushes to `main` publish their records. The
smokes show that every operator still builds, runs, and produces valid
records. Hosted runners are shared machines whose processor varies between
runs (the dashboard has both `CPU-AMD` and `CPU-Intel` series), so their
timings are not evidence of a performance change.

**Lab lane.** A maintainer starts the lab lane by hand on wurst, for example
after a performance change or before a release:

```bash
julia --project=benchmarking benchmarking/lab/run_lab_benchmarks.jl --dry-run   # print the plan
julia --project=benchmarking benchmarking/lab/run_lab_benchmarks.jl --publish
```

The driver benchmarks one commit, by default the head of `main`. It runs
`run_benchmarks.jl` on wurst itself and over SSH on curry and on the MacBook
(which shares no file system with wurst, so the checkout is copied there
first), then checks and merges the result files. With `--publish` it pushes
the merged records to the `benchmark-inbox` branch and dispatches the
`Benchmarks` workflow, whose publish job appends them to the dashboard data
with github-action-benchmark and re-renders the grouped dashboard page.

The machines, with their backends, float types, and pinned Julia version,
are listed in `benchmarking/lab/machines.toml`. A machine whose Julia is not
the pinned version is refused, and the driver publishes only when all
machines resolved the same environment fingerprint, so a lab entry never
mixes Julia versions or package versions. When a machine's run fails, its
result files are still copied back for inspection. The cases (production-scale grids, level
counts, and tracer counts, the same on every machine) are fixed by the lab
preset `benchmarking/presets/lab.toml`, so lab series stay comparable over
time. `--help` lists the driver's options, and
[`benchmarking/README.md`](https://github.com/RemoteSensingTools/AtmosTransport.jl/blob/main/benchmarking/README.md)
describes the machine setup.

The lab machines are not registered as self-hosted runners because the
repository is public: a pull request, including one from a fork, can change
the workflow it runs, and GitHub advises against self-hosted runners for
public repositories for that reason.

## Running a benchmark locally

Prepare the benchmark environment once from the repository root (this also
works on Julia 1.10, which ignores `[sources]`; the lab machines need Julia
1.11 or later because the lab lane runs `Pkg.instantiate()` alone):

```bash
julia --project=benchmarking -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

A CPU run of advection and the full step on two grids:

```bash
julia --threads=2 --project=benchmarking benchmarking/run_benchmarks.jl \
  --backend=cpu --float-type=Float32,Float64 \
  --grid=C24,C48 --levels=32 --tracers=1,4 \
  --operator=advection,full \
  --steps=5 --warmup-steps=1 --repeats=5 \
  --output=benchmark_results.json
```

`--backend=cuda`, `--backend=rocm`, or `--backend=metal` loads CUDA.jl, AMDGPU.jl, or Metal.jl before the
package and stops if the device is not functional; Metal needs
`--float-type=Float32`. ROCm needs the isolated runtime first on `JULIA_LOAD_PATH`;
see the [ROCm environment setup](rocm.md). `--preset=lab` runs the lab case matrix of
`benchmarking/presets/lab.toml`, and options given on the command line
override the preset. `--group` sets the first part of the series names;
`julia --project=benchmarking benchmarking/run_benchmarks.jl --help` lists
every option and its default. Each case prints its time per step and
cell-tracer updates per second; the two JSON files are written at the end.

To compare two commits, run the same command on both checkouts on the same
idle machine and compare `time_per_step_seconds` against `mad_seconds`.

## Reading the dashboard

The [dashboard](https://remotesensingtools.github.io/AtmosTransport.jl/benchmarks/)
selects an operator, a tracer count, a float type, and a metric (time per
step, or cell-tracer updates per second). It shows CPU and GPU panels for
grids up to `C180` and above `C180`, each with the history over commits (one
line per group, grid, level count, advection scheme, and device) and a table
of the latest value per case and device. The raw history is in
[`data.js`](https://remotesensingtools.github.io/AtmosTransport.jl/benchmarks/data.js).

- Compare like with like: the same device, float type, grid, levels, tracer
  count, and operator. Between devices, the same case on the same commit is a
  hardware comparison.
- Judge a change against the noise: a difference within a few MADs of either
  run is not evidence of a regression.
- When a value worsens beyond the alert threshold set in the workflow, the
  publish job comments on the commit; it does not fail the build.
- The cases are synthetic. A production run also reads meteorological
  windows, writes output, and subcycles where the Courant number requires it,
  so its wall time per simulated day does not follow from the benchmark time
  per step alone. Measured production runs are listed in
  [Validation status](@ref).

## Latest lab results

Validation runs of the lab lane, not yet published to the dashboard. The
values are median milliseconds per step with 2 tracers
(`time_per_step_seconds` in the raw `benchmark_results.json` files).

1. L40S and A100 columns: commit `2c56189e`. wurst ran alongside other jobs
   during this run, so its numbers are indicative. On curry, `Float64` C720
   L137 `full` did not fit in the A100's 40 GB and was skipped, as the
   memory estimate predicted.
2. M5 Pro column: commit `0810f1d6`, measured with Metal command-buffer
   failure detection (captured completion-handler reports behind a sentinel
   barrier); no command buffer failed.

Full step (advection, vertical diffusion, and convection), ms per step:

| Case | L40S `Float32` ¹ | A100 `Float32` ¹ | M5 Pro `Float32` ² | A100 `Float64` ¹ |
|---|---|---|---|---|
| C90 L72 | 19.3 | 29.2 | 220 | 55.8 |
| C90 L137 | 30.7 | 43.7 | 372 | 329 |
| C180 L72 | 69.3 | 107 | 807 | 216 |
| C180 L137 | 113 | 163 | 1,382 | 987 |
| C360 L72 | 271 | 413 | 3,200 | 854 |
| C360 L137 | 456 | 632 | 5,487 | 3,908 |
| C720 L72 | 1,085 | 1,644 | 12,589 | 3,414 |
| C720 L137 | 1,816 | 2,519 | 22,038 | skipped (needs 54.5 GiB, 38.6 free) |

Advection only, ms per step:

| Case | L40S `Float32` ¹ | A100 `Float32` ¹ | M5 Pro `Float32` ² | A100 `Float64` ¹ |
|---|---|---|---|---|
| C90 L72 | 5.0 | 10.4 | 43.8 | 11.1 |
| C90 L137 | 7.7 | 11.5 | 73.5 | 15.1 |
| C180 L72 | 11.6 | 20.4 | 105 | 28.1 |
| C180 L137 | 20.4 | 36.8 | 188 | 52.0 |
| C360 L72 | 38.3 | 73.2 | 389 | 106 |
| C360 L137 | 73.9 | 138 | 734 | 202 |
| C720 L72 | 145 | 287 | 1,389 | 417 |
| C720 L137 | 282 | 546 | 2,622 | 796 |

The dashboard has the full history once results are published.
