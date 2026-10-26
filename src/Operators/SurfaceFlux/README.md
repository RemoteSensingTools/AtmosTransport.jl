# Surface Flux

Per-tracer surface source operators.

This folder owns the source containers, the tracer-name lookup map, and
the topology-specific kernels that inject mass into the surface layer.

## Entry Points

- Source containers:
  [`sources.jl`](sources.jl)
  defines `SurfaceFluxSource` and `TimeVaryingSurfaceFluxSource` (with its
  device `FluxWindowBuffer`)
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
- Emission totals for logs and budgets:
  [`integrals.jl`](integrals.jl)
  provides `total_emission_rate(src, t)`, `emission_segments(src, t, dt)` and
  `integrated_emission(src, t0, t1, dt)`

## Time-Varying Series In Host Memory

A `TimeVaryingSurfaceFluxSource` keeps its rate series (`cell_mass_rate_series`,
`NTuple{6}` of `(Nc, Nc, ntime)` on the cube) in host memory on every backend.
Its `Adapt` hook moves only the Kahan compensation and a `FluxWindowBuffer`: a
ring of `capacity` slots on the compute backend that holds the slices the
current met window reads (slice `k` in slot `mod1(k, capacity)`). A C90
2014-2024 daily series (4018 slices) takes 0.78 GB per tracer in Float32, so
93 such tracers (73 GB) did not fit one GPU when the whole series was moved to
the device; now each tracer holds a few 0.19 MB slices there.

- `_flux_window_slices(scheme, times, t0, t1)` (next to
  `_flux_temporal_segments`) is the contiguous range of slices that the steps
  of the window `[t0, t1]` reference: the slices in effect for `StepwiseFlux`,
  from the left bracket at `t0` to the right bracket at the window end for
  `LinearInterpFlux` and `ConservativeMeanFlux`. The window end is padded for
  a Float32 step that ends slightly after it.
- `stage_surface_flux_window!(op, t0, t1)` copies the slices of that range that
  are not resident (one contiguous `copyto!` per panel and slice).
  `DrivenSimulation` calls it on the model's surface-flux operator when a
  window becomes current: at construction and in `_maybe_advance_window!`
  after the next window's met data are taken from the prefetch. It runs on the
  task that launches the kernels; the prefetch task never touches the buffer.
- The operator makes the slices of each step's segments resident before its
  kernels (`_stage_flux_slices!`), so it copies nothing after the window was
  staged and still works when called directly (tests, budget scripts). The
  kernels read `w0 * slots[slot(i0)] + w1 * slots[slot(i1)]`, the same values
  as the full series: results are bit-identical.
- The capacity grows to the longest staged range (2 slices for stepwise, 3 for
  linear or conservative sources with daily or hourly series and hourly met
  windows) and does not depend on the run length.
- The emission totals (`integrals.jl`) sum the host series in place.

## Cubed-Sphere Launches

`apply_surface_flux!` on the cube resolves each source's step on the host
(`_cs_surface_flux_plan`: target tracer, rate arrays, segments) and launches
one `Fused` set of `CSSurfaceFluxSegment` operations per segment index for up
to four sources at a time (`_CS_SURFACE_FLUX_SOURCES_PER_LAUNCH`, which keeps a
launch's arguments within the classic 4 kB limit). The sources of a map target
distinct tracers and own their compensation arrays, so they write disjoint
cells; each source's segments follow in stream order, so its Kahan and TwoSum
sums see the same operations in the same order as one launch per source and
segment (bit-identical). The operation runs as a kernel on the CPU backend too
(`host_parallel`), since a `ProfileDeposit` walks the whole column per index.
On an NVIDIA L40S with 72 levels and `ProfileDeposit`, a C90 step with 12
tracers takes 0.77 ms instead of 1.14 ms (12 launches), C24 with 12 tracers
0.13 ms instead of 0.39 ms (host time included).

## Layout Conventions

- The surface is always `k = Nz`
- Supported source-rate layouts match the runtime topology:
  - structured: `(Nx, Ny)`
  - face-indexed reduced Gaussian: `(ncells,)`
  - cubed sphere: `NTuple{6}` of `(Nc, Nc)` interior panel rates

## File Map

- [`SurfaceFlux.jl`](SurfaceFlux.jl) — submodule assembly and public exports
- [`sources.jl`](sources.jl) — `SurfaceFluxSource`,
  `TimeVaryingSurfaceFluxSource`, the temporal schemes
  (`_flux_temporal_weights`, `_flux_temporal_segments`), the slices a window
  reads (`_flux_window_slices`), the device window buffer (`FluxWindowBuffer`,
  `stage_surface_flux_window!`), compatibility checks, and low-level source
  helpers
- [`PerTracerFluxMap.jl`](PerTracerFluxMap.jl) — tuple-backed source map
  keyed by tracer name
- [`surface_flux_kernels.jl`](surface_flux_kernels.jl) — structured and
  face-indexed surface-application kernels, and the cubed-sphere
  `CSSurfaceFluxSegment` point operation; on the cube the emitted mass is
  placed by `SurfaceLayerDeposit` or `ProfileDeposit` (GEOS-Chem non-local
  VDIFF spreads fresh emissions over the PBL, see `emission_deposit`)
- [`operators.jl`](operators.jl) — operator hierarchy, source-to-tracer
  lookup, state-level `apply!`, array-level `apply_surface_flux!`,
  `stage_surface_flux_window!` for an operator, and the deposit choice of
  the cubed-sphere palindrome center (`emission_deposit`, and
  `center_emission_deposit`: the surface layer with the split coupling, the
  diffusion scheme's own placement with `DiffusiveSurfaceFluxBoundary`)
- [`integrals.jl`](integrals.jl) — global emission totals from the same
  temporal reconstruction the operator applies: the rate at an instant, the
  pieces of one step, and the integral over a run's steps (Float64 sums)

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
- Checking an emission budget:
  `integrated_emission(src, t0, t1, dt)` is the mass the operator adds over
  the run's steps; for stepwise and conservative-mean sources it does not
  depend on `dt`. Scripts call these functions instead of
  `_flux_temporal_segments`
- Tracing runtime installation:
  start from the `DrivenSimulation` constructor in
  [`../../Models/driven_stepping.jl`](../../Models/driven_stepping.jl),
  which turns `surface_sources` into a `SurfaceFluxOperator`; the same file
  stages each window's flux slices (`_stage_surface_flux_window!`)
- Checking device memory or upload counts of a time-varying source:
  `src.window_buffer` holds the slots, the resident slice of each slot and
  the number of slices staged over the logical buffer's lifetime (`uploads`;
  preserved when a populated buffer is adapted to another backend)

## Cross-Dependencies

- [`../../State/`](../../State/) provides tracer access and `eachtracer`
- [`../Advection/StrangSplitting.jl`](../Advection/StrangSplitting.jl),
  [`../Advection/multitracer_strang.jl`](../Advection/multitracer_strang.jl) and
  [`../Advection/strang_apply.jl`](../Advection/strang_apply.jl)
  embed surface flux at the transport midpoint
- [`../../Models/driven_stepping.jl`](../../Models/driven_stepping.jl) (the
  `DrivenSimulation` constructor) validates and installs sim-level surface sources
- [`../../Grids/`](../../Grids/) determines the active topology and
  source-shape expectations

## Related Docs And Tests

- Topology coverage:
  [`../TOPOLOGY_SUPPORT.md`](../TOPOLOGY_SUPPORT.md)
- Runtime/block ordering:
  [`../../Models/TransportModel.jl`](../../Models/TransportModel.jl) and
  [`../../../docs/20_RUNTIME_FLOW.md`](../../../docs/20_RUNTIME_FLOW.md)
- Tests:
  - [`../../../test/core/test_per_tracer_flux_map.jl`](../../../test/core/test_per_tracer_flux_map.jl)
  - [`../../../test/core/test_transport_model_emissions.jl`](../../../test/core/test_transport_model_emissions.jl)
  - [`../../../test/core/test_emissions_palindrome.jl`](../../../test/core/test_emissions_palindrome.jl)
  - [`../../../test/core/test_driven_simulation.jl`](../../../test/core/test_driven_simulation.jl)
  - [`../../../test/core/test_cubed_sphere_runtime.jl`](../../../test/core/test_cubed_sphere_runtime.jl)
  - [`../../../test/core/test_timevarying_surface_flux.jl`](../../../test/core/test_timevarying_surface_flux.jl)
  - [`../../../test/core/test_surface_flux_integrals.jl`](../../../test/core/test_surface_flux_integrals.jl)
  - [`../../../test/core/test_surface_flux_window_buffer.jl`](../../../test/core/test_surface_flux_window_buffer.jl)
  - [`../../../test/diagnostic/test_surface_flux_window_buffer_gpu.jl`](../../../test/diagnostic/test_surface_flux_window_buffer_gpu.jl)
    (opt-in CUDA check, `ATMOSTR_RUN_SURFACE_FLUX_GPU_TESTS=1`)
