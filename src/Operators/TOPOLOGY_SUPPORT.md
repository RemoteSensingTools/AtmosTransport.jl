# Topology Support Matrix

Canonical source of truth for which operators support which
topologies. Module READMEs reference this file rather than
duplicating coverage claims.

**Last verified:** 2026-07-12

## Topologies

| Symbol | State type | Storage | Flux state |
|--------|-----------|---------|------------|
| **LatLon** | `CellState` | rank-4 `tracers_raw::Array{FT, 4}` | `StructuredFaceFluxState{Basis}` |
| **RG** (reduced Gaussian) | `CellState` | rank-3 face-indexed `tracers_raw::Array{FT, 3}` | `FaceIndexedFluxState{Basis}` |
| **CS** (cubed sphere) | `CubedSphereState` | `NTuple{6, Array{FT, 4}}` panel-native | `CubedSphereFaceFluxState{Basis}` |

## Matrix

| Operator | LatLon | RG | CS |
|----------|:------:|:--:|:--:|
| `UpwindScheme` | ✅ | ✅ | ✅ |
| `SlopesScheme` / `PPMScheme` | ✅ | ❌ | ✅ |
| `LinRoodPPMScheme` | ❌ | ❌ | ✅ |
| `ImplicitVerticalDiffusion` | ✅ | ✅ | ✅ |
| `SurfaceFluxOperator` | ✅ | ✅ | ✅ |
| `CMFMCConvection` | ✅ | ✅ | ✅ |
| `TM5Convection` | ✅ | ✅ | ✅ |
| `CMFMCMatrixConvection` | ✅ | ✅ | ✅ |
| `ExponentialDecay` / `CompositeChemistry` | ✅ | ✅ | ✅ |

✅ = dedicated `apply!` or `apply_*!` dispatch exists, tested and live through `TransportModel.step!`.
❌ = no dispatch; the operator rejects that topology.

## Evidence anchors

For each ✅ combination, the authoritative dispatch method:

### Advection

- **LatLon** — rank-4 Strang palindrome `X→Y→Z→midpoint→Z→Y→X` in
  [`Advection/Structured/StrangSplitting.jl`](Advection/Structured/StrangSplitting.jl)
- **RG** — face-indexed `H→Z→midpoint→Z→H` with `UpwindScheme` in
  [`Advection/Structured/strang_apply.jl`](Advection/Structured/strang_apply.jl);
  higher-order schemes are rejected
- **CS** — panel-oriented packed-tracer
  [`strang_split_cs_mt!`](Advection/CubedSphere/CubedSphereStrang.jl) for split-sweep
  schemes. Lin–Rood uses its own H/Z palindrome and retains the single-tracer
  vertical sweep helpers in Advection/CubedSphere

The midpoint physics is supplied by [`StepPlan/midpoint.jl`](StepPlan/midpoint.jl)
through Advection’s `AbstractMidpointHook`; the default `NoMidpoint` does no work.

### Diffusion

The mass-conserving `apply_vertical_diffusion_vmr!` dispatches in
[`Diffusion/operators.jl`](Diffusion/operators.jl) cover:

- rank-4 (LatLon)
- rank-3 (RG face-indexed, multi-tracer)
- rank-2 (RG face-indexed, single-tracer)
- `NTuple{6, Array{FT, 4}}` (CS packed production path)
- `NTuple{6, Array{FT, 3}}` (CS single-tracer reference path)

### Surface flux

- [`SurfaceFlux/operators.jl`](SurfaceFlux/operators.jl) —
  topology-dispatched `apply!` for all three flux-state types
- Kernels live in [`SurfaceFlux/surface_flux_kernels.jl`](SurfaceFlux/surface_flux_kernels.jl)

### Convection

Three valid `apply!` methods + one rejection in
[`Convection/CMFMCConvection.jl`](Convection/CMFMCConvection.jl):

- `apply!(::CellState, ::ConvectionForcing, ::AtmosGrid{<:LatLonMesh}, ::CMFMCConvection, dt)`
- `apply!(::CellState, ::ConvectionForcing, ::AtmosGrid{<:ReducedGaussianMesh}, ::CMFMCConvection, dt)`
- `apply!(::CubedSphereState, ::ConvectionForcing, ::AtmosGrid{<:CubedSphereMesh}, ::CMFMCConvection, dt)`
- A fourth dispatch rejects face-indexed state on non-RG grids to
  catch configuration mistakes.

`TM5Convection.jl` and `CMFMCMatrixConvection.jl` provide the same three
topology-specific state/grid dispatches. The matrix variant derives TM5-style
rates from CMFMC forcing before delegating to the implicit column solve.

### Chemistry

Three valid `apply!` dispatches in
[`Chemistry/Chemistry.jl`](Chemistry/Chemistry.jl) per state type:

- `apply!(::CellState, ..., ::NoChemistry, dt)`
- `apply!(::CellState, ..., ::ExponentialDecay, dt)`
- `apply!(::CellState, ..., ::CompositeChemistry, dt)`

And three corresponding `CubedSphereState` dispatches. CS chemistry loops over
the six panels and launches the same rank-agnostic decay kernel per panel.

## Known gaps

- Reduced Gaussian advection is first-order only.
- `LinRoodPPMScheme` is cubed-sphere only.

## How to update this file

When topology support changes:

1. Update the matrix above.
2. Update the "Evidence anchors" list to point at the new `apply!`
   method.
3. Update the "Last verified" date.
4. In the corresponding submodule README, reference this file
   rather than restating coverage.

Keep topology-specific tests beside each operator's core tests so every ✅ claim
continues to map to an executable `apply!` path.
