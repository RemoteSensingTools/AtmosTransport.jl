# Surface Flux

Per-tracer surface source operators.

This folder owns the source containers, the tracer-name lookup map, and
the topology-specific kernels that inject mass into the surface layer.

## Entry Points

- Source container:
  [`sources.jl`](sources.jl)
  defines `SurfaceFluxSource`
- Tracer lookup map:
  [`PerTracerFluxMap.jl`](PerTracerFluxMap.jl)
  defines `PerTracerFluxMap` and `flux_for`
- Operator types:
  [`operators.jl`](operators.jl)
  defines `AbstractSurfaceFluxOperator`, `NoSurfaceFlux`, and
  `SurfaceFluxOperator`
- Model-facing runtime entrypoint:
  [`operators.jl`](operators.jl)
  provides `apply!(state, meteo, grid, op, dt; workspace)`
- Array-level runtime entrypoint:
  [`operators.jl`](operators.jl)
  provides `apply_surface_flux!`

## Layout Conventions

- The surface is always `k = Nz`
- Supported source-rate layouts match the runtime topology:
  - structured: `(Nx, Ny)`
  - face-indexed reduced Gaussian: `(ncells,)`
  - cubed sphere: `NTuple{6}` of `(Nc, Nc)` interior panel rates

## File Map

- [`SurfaceFlux.jl`](SurfaceFlux.jl) — submodule assembly and public exports
- [`sources.jl`](sources.jl) — `SurfaceFluxSource`, compatibility checks,
  and low-level source helpers
- [`PerTracerFluxMap.jl`](PerTracerFluxMap.jl) — tuple-backed source map
  keyed by tracer name
- [`surface_flux_kernels.jl`](surface_flux_kernels.jl) — structured,
  face-indexed, and cubed-sphere surface-application kernels; on the cube the
  emitted mass is placed by `SurfaceLayerDeposit` or `ProfileDeposit` (GEOS-Chem
  non-local VDIFF spreads fresh emissions over the PBL, see `emission_deposit`)
- [`operators.jl`](operators.jl) — operator hierarchy, source-to-tracer
  lookup, state-level `apply!`, array-level `apply_surface_flux!`

## Common Tasks

- Adding a new source layout:
  start in [`sources.jl`](sources.jl) and keep compatibility checks close
  to the source type
- Debugging "nothing happened" cases:
  check `flux_for(map, tracer_name)` in
  [`PerTracerFluxMap.jl`](PerTracerFluxMap.jl) and the tracer-name tuple
  passed into [`operators.jl`](operators.jl)
- Debugging cubed-sphere sources:
  verify that panel rates are interior-only `(Nc, Nc)` arrays and not
  halo-padded copies
- Tracing runtime installation:
  start from [`../../Models/DrivenSimulation.jl`](../../Models/DrivenSimulation.jl),
  which turns `surface_sources` into a `SurfaceFluxOperator`

## Cross-Dependencies

- [`../../State/`](../../State/) provides tracer access and `eachtracer`
- [`../Advection/StrangSplitting.jl`](../Advection/StrangSplitting.jl),
  [`../Advection/multitracer_strang.jl`](../Advection/multitracer_strang.jl) and
  [`../Advection/strang_apply.jl`](../Advection/strang_apply.jl)
  embed surface flux at the transport midpoint
- [`../../Models/DrivenSimulation.jl`](../../Models/DrivenSimulation.jl)
  validates and installs sim-level surface sources
- [`../../Grids/`](../../Grids/) determines the active topology and
  source-shape expectations

## Related Docs And Tests

- Topology coverage:
  [`../TOPOLOGY_SUPPORT.md`](../TOPOLOGY_SUPPORT.md)
- Runtime/block ordering:
  [`../../Models/TransportModel.jl`](../../Models/TransportModel.jl) and
  [`../../../docs/20_RUNTIME_FLOW.md`](../../../docs/20_RUNTIME_FLOW.md)
- Tests:
  - [`../../../test/orphan/test_surface_flux_operator.jl`](../../../test/orphan/test_surface_flux_operator.jl)
  - [`../../../test/orphan/test_transport_model_emissions.jl`](../../../test/orphan/test_transport_model_emissions.jl)
  - [`../../../test/orphan/test_emissions_palindrome.jl`](../../../test/orphan/test_emissions_palindrome.jl)
  - [`../../../test/core/test_driven_simulation.jl`](../../../test/core/test_driven_simulation.jl)
  - [`../../../test/orphan/test_cubed_sphere_runtime.jl`](../../../test/orphan/test_cubed_sphere_runtime.jl)
