# Runtime Flow

This contributor note follows one configured transport step from the binary
driver to topology-specific operator kernels. The newcomer-facing version is
the [architecture tour](src/concepts/architecture.md).

## Ownership

| Concern | Owner |
| --- | --- |
| Open drivers, staging, and output task lifetime | `DrivenRunner` |
| Binary header, window loading, timing semantics | `TransportBinaryDriver` |
| Current host/device forcing window and simulation clock | `DrivenSimulation` |
| Air mass, conservative tracer storage, operators, reusable workspaces | `TransportModel` |
| Topology-specific numerics | concrete `apply!` methods |

Tracer state is deliberately independent of file-I/O policy. The driver
provides typed `TransportWindow`s; the model only consumes prepared fields.

Cubed-sphere startup allocates packed tracer storage once and converts one
initial VMR field at a time into that storage. The shared initial-condition
packer retains signed values and zeros the halos; a small name-to-index map
preserves the existing packed tracer order. This runner path requires dry-basis
binaries because CS windows do not carry the humidity needed for moist packing.

The runner retains numerical state and workspaces across input files. Each
new simulation refreshes forcing and diffusion geometry, invalidates derived
convection caches, and continues the accumulated clock. GPU startup reads the
first window once and initializes independent active and prefetched buffers.
Before closing a driver, the runner drains prefetch so it cannot read a closed
input. Completed prefetch failures are observed once without swapping buffers.

NetCDF capture selects requested layers and column diagnostics while preserving
compensated tracer totals. Single-file output appends records without a frame
history. Daily output owns at most one background write and drains it on exit.
See `src/Models/runner/resources.jl` and `runner/output.jl` for resource lifetime;
numerical stepping remains in `DrivenSimulation` and `TransportModel`.

## One `DrivenSimulation.step!`

```text
step!(sim)
├── load/prefetch the next window at a window boundary
├── find the active substep and refresh forcing
├── run transport at the binary's stored substep cadence
├── advance time and iteration
├── at a met-window boundary, run convection and chemistry once
└── invoke callbacks
```

The implementation lives in `src/Models/driven_stepping.jl`. In simplified
form:

```julia
_maybe_advance_window!(sim)
substep = substep_index(sim)
_refresh_forcing!(sim, substep)

if _uses_binary_transport_schedule(sim)
    transport_step!(sim.model, sim.Δt; meteo = sim)
else
    step!(sim.model, sim.Δt; meteo = sim)
end

sim.iteration += 1
sim.time = _clock_time(sim)     # Float64, from window and step counters

if _uses_binary_transport_schedule(sim) &&
   sim.iteration == sim.current_window_end_iteration
    _maybe_reset_to_window_endpoint!(sim)
    convection_chemistry_step!(sim.model, sim.window_dt; meteo = sim)
end
```

A version-4 transport binary carries the advection substep schedule. It does
not request convection and chemistry at every stored advection substep; those
blocks run once per meteorological window on the canonical path.

## Window advance and forcing refresh

At a boundary, `_maybe_advance_window!`:

- advances the window index and installs its stored substep count;
- computes `Δt = window_dt / steps_per_window`;
- takes a prefetched window or loads one synchronously;
- applies the configured air-mass reset policy;
- validates convection forcing against the selected operator;
- refreshes diffusion geometry/Kz state and invalidates convection caches; and
- starts prefetch for the following window when enabled.

`_refresh_forcing!` then computes the current substep fraction. Depending on
the binary's declared sampling semantics, it either interpolates fluxes or
copies window-constant values. It also refreshes expected air mass, humidity,
and the selected operator's `ConvectionForcing` buffers.

The `meteo` object passed to operators is `sim`, not `sim.driver`. This gives
time-varying fields access to `current_time(sim)` while keeping the driver
available as `sim.driver` when a capability check is needed.

## Model blocks

`src/Models/TransportModel.jl` separates the runtime into two blocks:

```julia
function transport_step!(model, dt; meteo = nothing)
    apply!(model.state, model.fluxes, model.grid, model.advection, dt;
           workspace = model.workspace.advection_ws,
           diffusion_workspace = model.workspace.diffusion_ws,
           diffusion_op = model.diffusion,
           emissions_op = model.emissions,
           meteo)
end

function convection_chemistry_step!(model, dt; meteo = nothing)
    # dispatches away when convection is NoConvection()
    _convection_block!(model.convection, model, dt)
    apply!(model.state, meteo, model.grid, model.chemistry, dt)
end
```

`step!(model, dt)` calls both in order. `NoDiffusion`, `NoSurfaceFlux`,
`NoConvection`, and `NoChemistry` provide explicit identity methods; callers
do not need `nothing` checks or string branches in the operator loop.

## Topology dispatch

The transport block dispatches on state/flux/mesh types:

- lat-lon uses packed rank-4 tracer storage and an
  `X → Y → Z → midpoint → Z → Y → X` split;
- reduced Gaussian uses face-indexed storage and an `H → midpoint → H`
  split; and
- cubed sphere uses six halo-padded panels with either packed split sweeps or
  the Lin-Rood path.

The midpoint composes vertical diffusion and surface flux according to the
diffusion operator's `AbstractSurfaceFluxCoupling` policy. Convection is a
separate block and consumes a `ConvectionForcing` whose populated fields are
validated for `CMFMCConvection`, `TM5Convection`, or
`CMFMCMatrixConvection`.

The canonical operator/topology matrix is
[`src/Operators/TOPOLOGY_SUPPORT.md`](../src/Operators/TOPOLOGY_SUPPORT.md).

## Timing and debugging

`SectionTimer` labels window load, backend copy, forcing refresh, advection,
convection, chemistry, and output separately. Run a configured simulation with

```bash
ATMOSTR_TIMERS=1 julia --project=. scripts/run_transport.jl my_run.toml
```

to print the breakdown and write a sibling timing CSV when output is enabled.

## Related contracts

- [`10_CORE_CONTRACTS.md`](10_CORE_CONTRACTS.md) — state, basis, and kernel contracts
- [`30_BINARY_AND_DRIVERS.md`](30_BINARY_AND_DRIVERS.md) — timing and payload semantics
- [`35_RUNTIME_STABILITY_AND_SUBCYCLING.md`](35_RUNTIME_STABILITY_AND_SUBCYCLING.md) — numerical subcycling history
- [`src/concepts/architecture.md`](src/concepts/architecture.md) — user-facing architecture tour
