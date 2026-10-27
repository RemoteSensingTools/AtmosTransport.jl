# Contributing to AtmosTransport.jl

Thank you for your interest in contributing! This document provides guidelines
for contributing to AtmosTransport.jl.

## Getting Started

### Prerequisites

- Julia 1.10 or later (install via [juliaup](https://github.com/JuliaLang/juliaup))
- Git
- (Optional) an NVIDIA GPU with CUDA 12+ drivers, or an Apple-silicon Mac
  (Metal, Float32 only), for GPU testing

### Development Setup

```bash
git clone https://github.com/RemoteSensingTools/AtmosTransport.jl.git
cd AtmosTransport.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Running Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The full suite takes 30 to 70 minutes on one process. While developing, run the
quick set and the tests of the topics your change touches, or the quick gate,
which does both and two static checks:

```bash
julia --project=test -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'  # once
julia --project=test test/runtests.jl --quick                  # crucial fast set, a few minutes
julia --project=test test/runtests.jl --topics=advection       # one or more topics
julia --project=test test/runtests.jl --changed                # topics of your changes vs origin/main
scripts/dev/quick_gate.sh                                      # before committing
```

The quick gate runs `git diff --check`, a check that every call passes the
keywords its function requires (`scripts/checks/check_keyword_calls.jl`, which
also covers scripts and the literate docs that no test runs), and
`--quick --changed` in one Julia process; it exits non-zero on any failure.
CI runs the quick set as an early-feedback job, but a pull request merges only
after the full suite passes on Julia 1.10 and the current release. The test
tiers, opt-in flags, topics (`test/groups.toml`), and how to add a test file
are described in [`test/README.md`](test/README.md).

### Coverage

CI records line coverage on every pull request and push to `main`; the
Julia 1 job uploads the report for `src/` and `ext/` to
[Codecov](https://codecov.io/gh/RemoteSensingTools/AtmosTransport.jl), which
comments on pull requests and sets a status check. The thresholds and the
excluded folders are in [`codecov.yml`](codecov.yml). The hosted job runs the
CPU tiers only, so code in the CUDA and Metal extensions shows as uncovered
there; the GPU diagnostics in [`test/README.md`](test/README.md) exercise it
on GPU hardware.

To see coverage locally, run the tests with coverage on, summarize the `.cov`
files they leave next to the source files, and delete them:

```bash
julia --project=. -e 'using Pkg; Pkg.test(coverage=true)'
julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.add("Coverage"); using Coverage
          covered, total = get_summary(process_folder("src"))
          println(covered, " of ", total, " lines in src/ covered")
          foreach(clean_folder, ("src", "ext", "test"))'
```

### Running Benchmarks

The synthetic cubed-sphere benchmarks live in
[`benchmarking/`](benchmarking/README.md). Prepare the benchmark environment
once, then time the operators on the CPU:

```bash
julia --project=benchmarking -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --threads=2 --project=benchmarking benchmarking/run_benchmarks.jl \
  --backend=cpu --float-type=Float32 --grid=C24 --levels=32 --tracers=1,4 \
  --operator=advection,diffusion,convection,full --output=benchmark_results.json
```

Use `--backend=cuda`, or `--backend=metal` with `Float32`, on a GPU. To
measure a change, run the same command on the base and on the changed
checkout, on the same idle machine, and compare `time_per_step_seconds`
against `mad_seconds` in the JSON output. Pull requests run tiny CPU
benchmark smokes in CI on GitHub-hosted runners; the GPU results on the
[per-commit dashboard](https://remotesensingtools.github.io/AtmosTransport.jl/benchmarks/ci/)
come only from the maintainers' lab machines through the lab lane (the
repository deliberately has no self-hosted runners). The
[benchmarks page](https://RemoteSensingTools.github.io/AtmosTransport.jl/dev/development/benchmarks)
of the manual defines the metrics and explains the dashboard.

### Golden Outputs

A refactor must leave every result unchanged. The golden harness reruns 27
end-to-end cases (5 preprocessing one day each, 22 runtime simulating one day,
one of them two) and compares their outputs bit for bit with a reference
recorded on the same machine; it needs the group's data and a GPU:

```bash
julia --project=. test/golden/run_goldens.jl check <ref> <new>
```

The cases run in three worker sessions (preprocessing, CPU runs, GPU runs), so
package loading and compilation happen once per session; a full check takes
about 41 minutes on an L40S (about 70 with one process per case). A case that
crashes or hangs (stopped after `--timeout` minutes, default 45) fails, and
the other cases continue in a new worker. See
[`test/golden/README.md`](test/golden/README.md) for selecting cases,
recording a reference and what is compared.

### Building Documentation Locally

```bash
ATMOSTR_DOCS_BUILD_ONLY=true julia docs/build.jl
```

This runs the same checks as the CI docs build (doctests, exported docstrings,
cross-references) without deploying; [`docs/README.md`](docs/README.md)
explains the docs layout and how to preview the site.

## Code Style

- Follow standard Julia conventions: `snake_case` for functions and variables,
  `CamelCase` for types
- Use multiple dispatch rather than if-else chains on type tags or grid kinds
- Keep functions short and focused; prefer composing small functions
- Add docstrings to all exported functions and types
- Kernels use KernelAbstractions so one implementation runs on CPU, CUDA, and
  Metal; keep array fields of GPU-aware structs parametric

## Architecture Overview

The model is built from abstract type hierarchies and multiple dispatch,
inspired by Oceananigans.jl. The main operator families (all subtypes of
`AbstractOperator` in `src/Operators/`):

```
AbstractAdvectionScheme      →  UpwindScheme, SlopesScheme, PPMScheme, LinRoodPPMScheme
AbstractConvection           →  NoConvection, CMFMCConvection, CMFMCMatrixConvection, TM5Convection
AbstractDiffusion            →  NoDiffusion, ImplicitVerticalDiffusion
AbstractSurfaceFluxOperator  →  NoSurfaceFlux, SurfaceFluxOperator
AbstractChemistryOperator    →  NoChemistry, ExponentialDecay, CompositeChemistry
```

Grids are `AtmosGrid`s that combine a horizontal mesh (`LatLonMesh`,
`ReducedGaussianMesh`, or `CubedSphereMesh`, all subtypes of
`AbstractHorizontalMesh`) with a vertical coordinate and an architecture.
Meteorology enters through an `AbstractMetDriver`, normally the
`TransportBinaryDriver` that reads preprocessed transport binaries. See
[`docs/src/concepts/architecture.md`](docs/src/concepts/architecture.md) for
the full picture.

## Adding a New Physics Operator

- **New advection scheme:** add the type in
  `src/Operators/Advection/schemes.jl`, wire its reconstruction in
  `reconstruction.jl`, and follow the sweep path from the `apply!` methods in
  `strang_apply.jl`; [`src/Operators/Advection/README.md`](src/Operators/Advection/README.md)
  maps the files.
- **New operator family member** (convection, diffusion, surface flux,
  chemistry): subtype the family's abstract type and implement
  `apply!(state, forcing, grid, op, dt; workspace)`, where the forcing
  argument is family-specific (for example `ConvectionForcing` for
  convection). A new family ships a `No<Operator>` default and is wired
  through `TransportModel`.
- **Adjoint** (for 4D-Var and footprints): the per-operator reverse kernels
  live in `src/Adjoints/`; see [`src/Adjoints/README.md`](src/Adjoints/README.md).

Test what applies to the operator: transport operators keep a uniform mixing
ratio uniform and conserve mass; sources and sinks (emissions, decay) match
their analytic budget; operators with a reverse kernel pass the adjoint
identity test; and CPU and GPU results agree. For a new family, also test
that the default path is bit-identical to the explicit no-op path.

## Submitting Changes

1. Fork the repository and create a feature branch
2. Make your changes with clear, focused commits
3. Run `scripts/dev/quick_gate.sh` while you work, and make sure the full
   suite passes before review: `julia --project=. -e 'using Pkg; Pkg.test()'`
   (CI runs it on every pull request and requires it for merging)
4. Update the relevant `README.md` or reference docs when the change affects
   public behavior, scripts, configuration, or setup
5. Open a pull request with a clear description of what changed and why

## Reporting Issues

Please open an issue on GitHub with:
- A clear description of the problem
- Minimal reproducible example (if applicable)
- Julia version and OS information (`versioninfo()`)
