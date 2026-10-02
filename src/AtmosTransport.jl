"""
    AtmosTransport

Offline atmospheric transport on lat-lon, reduced-Gaussian, and cubed-sphere
grids.

## Quick start

```julia
using TOML
using AtmosTransport

# First run examples/generate_synthetic_quickstart.jl from the terminal.
cfg = TOML.parsefile("config/examples/minimal_template.toml")
ok, errors = validate_config(cfg)
ok || error(join(errors, "\n"))
run_driven_simulation(cfg)
```

From the shell, use the same library entry point through the canonical runner:

```bash
julia --project=. examples/generate_synthetic_quickstart.jl
julia --project=. scripts/run_transport.jl config/examples/minimal_template.toml
```

## Common entry points

- [`run_driven_simulation`](@ref): load a run config, dispatch on the first
  transport binary's topology, run the simulation, and write snapshots.
- [`validate_config`](@ref): catch common run-config mistakes before opening
  binary readers or allocating model state.
- [`inspect_binary`](@ref): inspect a preprocessed transport binary header and
  capability flags.
- [`open_snapshot`](@ref), [`mapplot`](@ref), [`movie`](@ref): inspect and plot
  written NetCDF snapshots.
- [`write_transport_binary`](@ref): build a transport binary from in-memory
  test or preprocessing windows.

See the rendered getting-started docs under `docs/src/getting_started/`, the
architecture tour in `docs/src/concepts/architecture.md`, and the curated API
map in `docs/src/api/public_api.md`.
"""
module AtmosTransport

using KernelAbstractions

# ---------------------------------------------------------------------------
# Free-choice data root.
#
# Configs and CLI scripts express paths as `~/...`, `$ENV_VAR/...`, or
# `${ENV_VAR}/...`. `expand_data_path` resolves ordinary environment variables
# plus the package data-root fallback:
#
#     export ATMOSTRANSPORT_DATA_ROOT=/scratch/$USER/atmostransport
#
# When this variable is unset, the fallback is `~/data/AtmosTransport`.
# Trailing `/` on environment-variable values is tolerated.
# ---------------------------------------------------------------------------
const _DATA_ROOT_ENV                = "ATMOSTRANSPORT_DATA_ROOT"
const _DATA_ROOT_FALLBACK           = "~/data/AtmosTransport"
const _PATH_ENVVAR_RE = r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)"

"""
    expand_data_path(p::AbstractString) -> String

Resolve a TOML/CLI path string by substituting `\$ATMOSTRANSPORT_DATA_ROOT`
(or `\${ATMOSTRANSPORT_DATA_ROOT}`) and any environment variable that is set
in `ENV`, then running `expanduser` for any leading `~`. Returns a plain
`String`.
"""
function expand_data_path(p::AbstractString)
    s = String(p)
    io = IOBuffer()
    pos = firstindex(s)
    for m in eachmatch(_PATH_ENVVAR_RE, s)
        start = m.offset
        if start > pos
            print(io, s[pos:prevind(s, start)])
        end
        name = something(m.captures[1], m.captures[2])
        value = if haskey(ENV, name)
            ENV[name]
        elseif name == _DATA_ROOT_ENV
            _DATA_ROOT_FALLBACK
        else
            m.match
        end
        print(io, rstrip(String(value), '/'))
        pos = nextind(s, start + ncodeunits(m.match) - 1)
    end
    if pos <= lastindex(s)
        print(io, s[pos:end])
    end
    return expanduser(String(take!(io)))
end
expand_data_path(p) = expand_data_path(String(p))

export expand_data_path

# ---- Architecture and planetary constants ----
include("Architectures.jl")
using .Architectures
const AbstractArchitecture = Architectures.AbstractArchitecture
const CPU = Architectures.CPU
const GPU = Architectures.GPU

# ---- Section timer (host-side wall-clock; off unless ATMOSTR_TIMERS=1) ----
include("Diagnostics/SectionTimer.jl")
using .SectionTimer

# ---- Quantity-kind dispatch traits ----
# Tiny trait module loaded early so any downstream module (Preprocessing,
# Operators, Output) can dispatch on extensive vs intensive vs vector vs flux
# field semantics without circular dependencies. See Quantities.jl for the
# four-type taxonomy and their regrid-handling contracts.
include("Quantities/Quantities.jl")
using .Quantities

include("Parameters/Parameters.jl")
using .Parameters

# ---- Geometry ----
include("Grids/Grids.jl")
using .Grids

# ---- State containers ----
include("State/State.jl")
using .State

# ---- Diagnostic NetCDF output ----
include("Output/Output.jl")
using .Output

# ---- Met-data adapters ----
include("MetDrivers/MetDrivers.jl")
using .MetDrivers: AbstractMetDriver,
                   current_time,
                   AbstractTransportBinaryGeometry, LatLonBinaryGeometry,
                   ReducedGaussianBinaryGeometry, CubedSphereBinaryGeometry,
                   TransportBinaryReader, TransportBinaryHeader, binary_geometry,
                   write_transport_binary,
                   TransportBinaryDriver, TransportWindow,
                   StructuredFluxDeltas, FaceIndexedFluxDeltas,
                   CubedSphereFluxDeltas,
                   load_window!, load_flux_delta_window!,
                   load_qv_pair_window!, load_grid, load_transport_window,
                   driver_grid, air_mass_basis, has_humidity_endpoints,
                   interpolate_fluxes!, expected_air_mass!, interpolate_qv!, copy_fluxes!,
                   load_surface_window!,
                   ConvectionForcing, has_convection_forcing,
                   copy_convection_forcing!, allocate_convection_forcing_like,
                   PBLSurfaceForcing, has_pbl_surface_forcing,
                   window_count, has_qv_endpoints, has_flux_delta, has_cmfmc,
                   has_surface, has_vdiff_fields, has_tm5_convection,
                   grid_type, horizontal_topology,
                   source_flux_sampling, air_mass_sampling, flux_sampling, flux_kind, humidity_sampling, delta_semantics,
                   A_ifc, B_ifc,
                   diagnose_cm_from_continuity!, diagnose_cm_from_continuity_vc!,
                   diagnose_cm_from_continuity_ka!,
                   ERA5ReducedGaussianGeometry,
                   read_era5_reduced_gaussian_geometry, read_era5_reduced_gaussian_mesh,
                   total_windows, window_dt, steps_per_window, steps_per_window_schedule,
                   supports_diffusion, supports_convection,
                   mesh_convention, mesh_definition,
                   TRANSPORT_BINARY_FORMAT_VERSION,
                   StreamingTransportBinaryWriter,
                   open_streaming_transport_binary, write_streaming_window!,
                   close_streaming_transport_binary!, set_streaming_steps_per_window_schedule!,
                   binary_capabilities, inspect_binary

# ---- Physics operators ----
include("Operators/Operators.jl")
using .Operators

# ---- Adjoint tape storage + records ----
# Loaded BEFORE `Adjoints/` so the kernels module can `using ..Tape: ...`
# for the relocated storage policies and record types. The reverse-loop
# driver that DISPATCHES on these record types still lives in Adjoints
# (eventual move to `Footprint/`).
include("Tape/Tape.jl")
using .Tape

# ---- Prototype adjoint / footprint utilities ----
include("Adjoints/Adjoints.jl")
using .Adjoints

# ---- Offline regridding glue (CR.jl + JLD2) ----
# Loaded before Models so `Models.InitialConditionIO` can
# directly `using ..Regridding` and `using ..Preprocessing.CSHelpers` for the
# CS file-based IC path. Regridding and Preprocessing have no back-references
# to Models (verified by grep), so reordering is safe.
include("Regridding/Regridding.jl")
using .Regridding

# ---- Preprocessing pipeline (spectral/gridded → transport binary) ----
include("Preprocessing/Preprocessing.jl")
using .Preprocessing

# ---- Topology-aware snapshot visualization data layer ----
include("Visualization/Visualization.jl")
using .Visualization

# ---- Minimal runtime/model layer ----
include("Models/Models.jl")
using .Models

# ---- Download pipeline (TOML-driven met/emissions data download) ----
include("Downloads/Downloads.jl")
using .DataDownloads

# ---- Curated top-level API ----
#
# Deep internals remain reachable as `AtmosTransport.Submodule.symbol`. The
# top-level exports are intentionally kept to the symbols a scientist is likely
# to call directly while building, running, or inspecting a transport workflow.

export expand_data_path

# Run configs and binary inspection
export run_driven_simulation, validate_config, expand_binary_paths
export inspect_binary, binary_capabilities

# Grids, backends, and geometry
export CPU, GPU, earth_parameters
export AtmosGrid, LatLonMesh, ReducedGaussianMesh, CubedSphereMesh,
       HybridSigmaPressure
export nx, ny, ncells, nfaces, nlevels, nrings, cell_index
export cell_area, cell_faces, floattype
export radius, gravity, reference_pressure
export pressure_at_interface, pressure_at_level, level_thickness
export lonlat_to_panel_xy

# State and diagnostics
export CellState, CubedSphereState, DryBasis, MoistBasis
export allocate_face_fluxes, allocate_tracers
export get_tracer, mixing_ratio, total_mass, total_air_mass
export tracer_names, ntracers
export capture_snapshot, write_snapshot_netcdf, SnapshotFrame, SnapshotWriteOptions
export runtime_output_spec, observation_output_spec
export GnomonicPanelConvention, GEOSNativePanelConvention

# Common operator configuration
export AdvectionWorkspace, DiffusionWorkspace
export UpwindScheme, SlopesScheme, PPMScheme, LinRoodPPMScheme, NoAdvection
export NoDiffusion, ImplicitVerticalDiffusion
export NoSurfaceFlux, SurfaceFluxOperator, SurfaceFluxSource,
       AbstractSurfaceFluxSource, TimeVaryingSurfaceFluxSource,
       AbstractFluxTemporalScheme, StepwiseFlux, LinearInterpFlux, ConservativeMeanFlux,
       flux_temporal_scheme, PerTracerFluxMap, flux_for
export AbstractConvection, NoConvection, CMFMCConvection, TM5Convection,
       CMFMCMatrixConvection, CMFMCWorkspace
export ConvectionForcing, apply_convection!, has_convection_forcing
export AbstractMetDriver, TransportWindow, current_time
export AbstractChemistryOperator, NoChemistry, ExponentialDecay, CompositeChemistry
export ConstantField, ProfileKzField
export apply!

# Low-level model hooks for custom loops
export TransportModel, Simulation, DrivenSimulation
export step!, run!, run_window!, window_index
export with_chemistry, with_diffusion, with_emissions, with_convection, with_convection_forcing
export build_runtime_physics_recipe, validate_runtime_physics_recipe
export build_initial_mixing_ratio, pack_initial_tracer_mass
export TransportTracerSpec

# Transport binaries and met-driver summaries
export write_transport_binary
export AbstractTransportBinaryGeometry, LatLonBinaryGeometry
export ReducedGaussianBinaryGeometry, CubedSphereBinaryGeometry, binary_geometry
export TransportBinaryReader, TransportBinaryHeader, TransportBinaryDriver
export load_transport_window, driver_grid
export total_windows, window_dt, steps_per_window, steps_per_window_schedule
export window_count, load_grid, mass_basis, air_mass_basis
export has_qv_endpoints, has_flux_delta, has_cmfmc
export has_surface, has_vdiff_fields, has_tm5_convection, has_humidity_endpoints
export delta_semantics
export grid_type, horizontal_topology
export supports_diffusion, supports_convection

# Regridding and visualization
export build_regridder, apply_regridder!
export open_snapshot, fieldview, mapplot, movie

end # module AtmosTransport
