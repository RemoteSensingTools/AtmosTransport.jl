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

The test tiers, opt-in flags, and how to add a test file are described in
[`test/README.md`](test/README.md).

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
3. Ensure all tests pass: `julia --project=. -e 'using Pkg; Pkg.test()'`
4. Update the relevant `README.md` or reference docs when the change affects
   public behavior, scripts, configuration, or setup
5. Open a pull request with a clear description of what changed and why

## Reporting Issues

Please open an issue on GitHub with:
- A clear description of the problem
- Minimal reproducible example (if applicable)
- Julia version and OS information (`versioninfo()`)
