# Curated public API

This page is the small top-level surface exposed by `using AtmosTransport`.
Everything else remains reachable through its owning submodule, for example
`AtmosTransport.Preprocessing.process_day` or
`AtmosTransport.Adjoints.cs_surface_emission_footprint`.

## Run a configured simulation

| Symbol | Use |
|---|---|
| `run_driven_simulation(cfg)` | Canonical high-level run entry point for a TOML-derived config dictionary. |
| `validate_config(cfg)` | Pre-flight common config mistakes before model allocation. |
| `expand_data_path(path)` | Resolve `~/...` and environment-variable paths such as `$ATMOSTRANSPORT_DATA_ROOT/...`. |
| `expand_binary_paths(input_cfg)` | Resolve `[input]` `binary_paths` or `folder + start_date + end_date`. |

```julia
using TOML, AtmosTransport

# First run examples/generate_synthetic_quickstart.jl from the terminal.
cfg = TOML.parsefile("config/examples/minimal_template.toml")
ok, errors = validate_config(cfg)
ok || error(join(errors, "\n"))
model = run_driven_simulation(cfg)
```

## Inspect transport binaries

| Symbol | Use |
|---|---|
| `inspect_binary(path)` | Print header/capability information and return capability flags. |
| `binary_capabilities(reader)` | Summarize operator payload support for an open reader. |
| `has_surface(reader_or_settings)` | Query PBL surface payloads in a binary reader, or configured surface-field support in preprocessing settings. |
| `has_vdiff_fields(reader_or_settings)` | Query GCHP vertical-diffusion payloads in a reader, or configured support in preprocessing settings. |
| `TransportBinaryReader`, `TransportBinaryHeader` | Read version-4 and version-5 binaries on any supported topology. |
| `TransportBinaryDriver` | Runtime met driver shared by lat-lon, reduced-Gaussian, and cubed-sphere binaries. |
| `load_transport_window(driver, i)` | Load a runtime forcing window. |
| `write_transport_binary(...)` | Write synthetic or preprocessed LL/RG transport binaries. |
| `total_windows`, `window_dt`, `steps_per_window`, `steps_per_window_schedule` | Inspect driver timing. |
| `grid_type`, `horizontal_topology` | Inspect binary topology metadata. |
| `surface_pressure_basis(reader_or_driver)` | Typed meaning of the binary's `:ps` (`TotalSurfacePressure()` or `DrySurfacePressure()`); inferred once, with a warning, for legacy files without the header key. |
| `supports_diffusion`, `supports_convection` | Check whether a driver can support requested physics. |

All three topologies share this reader, driver, and runtime-window API.
Cubed-sphere preprocessing uses panel-aware streaming writer helpers; those
helpers are advanced construction APIs rather than a second reader hierarchy.

## Work with grids and state

| Symbol | Use |
|---|---|
| `CPU`, `GPU` | Architecture tags for grids and drivers. |
| `earth_parameters` | Default planetary constants. |
| `AtmosGrid` | Combine horizontal mesh, vertical coordinate, and architecture. |
| `LatLonMesh`, `ReducedGaussianMesh`, `CubedSphereMesh` | Horizontal mesh constructors. |
| `HybridSigmaPressure` | Hybrid sigma-pressure vertical coordinate. |
| `nx`, `ny`, `ncells`, `nfaces`, `nlevels`, `nrings`, `cell_index` | Grid dimensions and indexing. |
| `cell_area`, `cell_faces`, `floattype` | Grid geometry and scalar type helpers. |
| `radius`, `gravity`, `reference_pressure` | Planet constants stored on a grid. |
| `pressure_at_interface`, `pressure_at_level`, `level_thickness` | Vertical coordinate helpers. |
| `CellState`, `CubedSphereState` | Store air mass and conservative `mixing ratio × carrier-air-mass` tracer fields. |
| `DryBasis`, `MoistBasis` | Make the air-mass basis explicit in state containers. |
| `AbstractSurfacePressureBasis`, `TotalSurfacePressure`, `DrySurfacePressure` | Surface-pressure basis of a transport binary; every writer requires one. |
| `allocate_face_fluxes`, `allocate_tracers` | Allocate state-compatible transport storage. |
| `get_tracer`, `mixing_ratio`, `total_mass`, `total_air_mass` | Common state diagnostics. |
| `tracer_names`, `ntracers` | Inspect tracer layout. |

## Configure physics and custom loops

| Symbol | Use |
|---|---|
| `AdvectionWorkspace`, `DiffusionWorkspace` | Independent scratch storage for low-level advection and diffusion calls. Model constructors allocate them automatically. |
| `UpwindScheme`, `SlopesScheme`, `PPMScheme`, `LinRoodPPMScheme` | Common advection schemes. |
| `NoDiffusion`, `ImplicitVerticalDiffusion` | Diffusion operator choices. |
| `NoSurfaceFlux`, `SurfaceFluxOperator`, `SurfaceFluxSource` | Surface-flux operator stack. |
| `PerTracerFluxMap`, `flux_for` | Map emissions to named tracers. |
| `NoConvection`, `CMFMCConvection`, `TM5Convection` | Convection operator choices. |
| `AbstractCloudBase`, `CMFMCEdgeCloudBase`, `ArchivedCloudBase` | Cloud-base rule of `CMFMCConvection`: lowest layer with updraft inflow (default) or the archived GEOS-Chem base (`cloud_base = "dqrcu"`). |
| `Callback`, `IterationInterval`, `WindowInterval`, `WallTimeInterval`, `NaNChecker` | Runtime callbacks and their schedules (see *Callbacks and runtime checks*). |
| `NoChemistry`, `ExponentialDecay`, `CompositeChemistry` | Chemistry operator choices. |
| `ConstantField`, `PreComputedKzField` | Simple time-varying fields used by physics operators. |
| `apply!` | Generic low-level operator application hook. |
| `TransportModel` | Low-level bundle of state, fluxes, grid, and operators. |
| `Simulation`, `DrivenSimulation` | Low-level harnesses for custom in-memory and window-driven loops. |
| `step!`, `run!`, `run_window!` | Low-level loop hooks for custom experiments. |
| `with_chemistry`, `with_diffusion`, `with_emissions` | Functional model-update helpers. |
| `build_runtime_physics_recipe`, `validate_runtime_physics_recipe` | Build and check physics recipes from config. |
| `build_initial_mixing_ratio`, `pack_initial_tracer_mass` | Initial-condition helpers. |
| `TransportTracerSpec` | Runtime tracer specification record. |

### Manual `DrivenSimulation` Assembly

Use this pattern when you want the runtime loop but need to own the model
construction. It is the advanced path behind `run_driven_simulation`, without
the wrapper's multi-file output, GPU adaptation, and progress plumbing.

```julia
using TOML, AtmosTransport
using AtmosTransport.MetDrivers: air_mass_basis, driver_grid

# Requires the binary generated by examples/generate_synthetic_quickstart.jl.
cfg = TOML.parsefile("config/examples/minimal_template.toml")
paths = expand_binary_paths(cfg["input"])
FT = Float64

driver = TransportBinaryDriver(first(paths); FT = FT, arch = CPU())
recipe = build_runtime_physics_recipe(cfg, driver, FT)
validate_runtime_physics_recipe(recipe, driver)

grid = driver_grid(driver)
window1 = load_transport_window(driver, 1)
Basis = typeof(mass_basis(air_mass_basis(driver)))
air = copy(window1.air_mass)

vmr = build_initial_mixing_ratio(
    air, grid, Dict("kind" => "uniform", "background" => 400e-6);
    surface_pressure = window1.surface_pressure,
)
co2 = pack_initial_tracer_mass(grid, air, vmr; mass_basis = Basis())

state = CellState(Basis, air; CO2 = co2)
fluxes = allocate_face_fluxes(grid.horizontal, nlevels(grid);
                              FT = FT, basis = Basis)
model = TransportModel(state, fluxes, grid, recipe.advection;
                       diffusion = recipe.diffusion,
                       convection = recipe.convection)

sim = DrivenSimulation(model, driver;
                       stop_window = min(total_windows(driver), 24),
                       chemistry = recipe.chemistry)
run_window!(sim)
```

The constructor validates grid and mass-basis compatibility before stepping.
`DrivenSimulation.step!` then refreshes forcing from the driver and delegates
the actual operator ordering to `TransportModel.step!`.

### Callbacks and runtime checks

As in Oceananigans, a [`Callback`](@ref) pairs a function of the simulation
with a schedule that decides after which steps it runs:
[`IterationInterval`](@ref)`(n)` (every n-th step), [`WindowInterval`](@ref)`(n)`
(the end of every n-th met window) or [`WallTimeInterval`](@ref)`(seconds)`.
[`NaNChecker`](@ref) stops the run with an error when the air mass or a tracer
holds a NaN or Inf; the configured runner installs it from `[checks]`.

```julia
callbacks = (nan_checker = Callback(NaNChecker(), WindowInterval(1)),
             daily = Callback(sim -> @info("window $(sim.current_window_index) done"),
                              WindowInterval(24)))
sim = DrivenSimulation(model, driver; callbacks)
run!(sim)
```

Callbacks run in the order given, after the step and, at a window end, after
the window's convection and chemistry. A plain function in `callbacks` runs
after every step.

## Output, regridding, and visualization

| Symbol | Use |
|---|---|
| `capture_snapshot`, `write_snapshot_netcdf` | Capture and write diagnostic frames. |
| `SnapshotFrame` | CPU-resident tracer and air-mass snapshot, with independent signed Float64 tracer totals. |
| `SnapshotWriteOptions` | Select spatial output precision, compression and shuffle settings for the snapshot writer. |
| `runtime_output_spec`, `observation_output_spec` | Parse `[output]` and `[output.observations]` into typed output specifications. |
| `build_regridder`, `apply_regridder!` | Build and apply offline conservative regridders. |
| `open_snapshot` | Open a written snapshot NetCDF. |
| `fieldview` | Extract a plottable field view from a snapshot. |
| `mapplot`, `movie` | Plot or animate snapshot fields when a Makie backend is loaded. |

## Advanced names

Internal tape records, checkpoint schedules, payload-section loaders, kernel
variants, and preprocessing drivers are intentionally not exported at the top
level. Advanced workflows can use explicit qualification through the owning
module:

```julia
AtmosTransport.MetDrivers.load_qv_pair_window!(...)
AtmosTransport.Preprocessing.process_day(...)
AtmosTransport.Adjoints.cs_surface_emission_footprint(...)
```

## Top-level docstrings

```@autodocs
Modules = [AtmosTransport]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```

Use `MetDrivers.binary_mass_basis(reader)` for the header Symbol, and
`State.mass_basis(tag)` / `State.basis_symbol(basis)` for tag conversion.
Observation placement coordinates subtype `Observations.AbstractPlacementCoordinate`;
model vertical grids subtype `Grids.AbstractVerticalCoordinate`.

Observation APIs live in `AtmosTransport.Observations` (loaded after Output,
before Adjoints), with the `Retrievals`, `Sources`, and `Sampling` submodules.
Snapshot capture, NetCDF schema and streams, and ATMSNAP IO remain in Output.
`Observations.pressure_unit_scale` and `parse_iso_utc` are shared public reader
helpers; inversion and sampling use
`Parameters.DEFAULT_OBSERVATION_LAYER_TEMPERATURE_KELVIN` (280 K) when no layer
temperature is supplied. CF time decoding is unchanged and remains separate
from the later-loaded GriddedIO module.

Boundary-layer field types and refresh methods are owned by
`AtmosTransport.Operators.Diffusion`. Its public staged pressure/dz APIs reuse
`HybridCoefficientWorkspace`; allocating geometry wrappers and the test-only
VMR matrix-form solver were removed.
