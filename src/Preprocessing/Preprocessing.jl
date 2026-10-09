"""
    Preprocessing

Transport binary generation from ERA5 spectral meteorological data.

Provides the complete pipeline from raw spectral GRIB (VO, D, LNSP) to
Poisson-balanced transport binaries ready for the runtime `TransportBinaryDriver`.

## Architecture

The pipeline dispatches on `AbstractTargetGeometry` subtypes:

    AbstractTargetGeometry
    ├── LatLonTargetGeometry              — regular lat-lon (any Nx × Ny)
    ├── ReducedGaussianTargetGeometry     — native ERA5 RG (O90, O160, N320, …)
    └── CubedSphereTargetGeometry         — convention-aware CS (C24, C90, C180, …)

Each target geometry has a dedicated `process_day` method:

    process_day(date, grid::LatLonTargetGeometry, settings, vertical; …)
    process_day(date, grid::ReducedGaussianTargetGeometry, settings, vertical; …)
    process_day(date, grid::CubedSphereTargetGeometry, settings, vertical; …)

## Pipeline (per day)

1. Read ERA5 spectral GRIB (VO, D, LNSP) — `spectral_io.jl`
2. Spectral synthesis (Legendre + FFT → gridpoint) — `spectral_synthesis.jl`
3. Merge native 137L → transport levels — `vertical_coordinates.jl`
4. Pin global mean ps (mass fix) — `transport_binary/latlon_workspaces.jl`
5. Topology-specific transport-binary workflow:
   - LL: FFT on circulant Laplacian — `mass_support.jl`
   - RG: compressed-Laplacian CG — `ring_poisson_balance.jl`
   - CS: global 6-panel graph-Laplacian CG — `cs_poisson_balance.jl`
6. Diagnose `cm` from explicit endpoint mass targets — replay continuity
7. Write transport binary with declared payload semantics and replay checks

The high-level transport-binary workflows are split under
`transport_binary/`. Each topology owns a small `process_day` method while
shared binary metadata, endpoint-delta, and replay helpers stay in common
files.

## Usage

```bash
julia --project=. scripts/preprocessing/preprocess_transport_binary.jl config.toml --day 2021-12-01
```

Advanced users can call `process_day` directly from Julia.
"""
module Preprocessing

using Dates
using Logging
using FFTW
using Printf
using JSON3
using JLD2
using SHA
using TOML
using LinearAlgebra: mul!, dot
using SparseArrays: SparseMatrixCSC, sparse
using NCDatasets

function _config_bool(value, path::AbstractString)
    value isa Bool || throw(ArgumentError("$(path) must be true or false; got $(repr(value))"))
    return value
end

_config_bool(cfg::AbstractDict, key::AbstractString, default::Bool, path::AbstractString) =
    _config_bool(get(cfg, key, default), path)
using GRIB
using SHA
using FastGaussQuadrature: gausslegendre

# Re-export parent module types we need
import ..expand_data_path
import ..source_revision
# Source settings and binary readers extend the same public capability queries.
import ..MetDrivers: has_surface, has_vdiff_fields, has_pbl_eflux, has_cmfmc_cloud_base
using ..Architectures: CPU
using ..Parameters: IFS_EARTH_RADIUS, STANDARD_GRAVITY, STANDARD_PRESSURE, TM5_CONSTANTS, R_DRY_AIR,
                    VIRTUAL_TEMPERATURE_FACTOR
using ..Grids: LatLonMesh, ReducedGaussianMesh, CubedSphereMesh, cs_face_edge_lengths, cs_corner_xyz,
               HybridSigmaPressure, PanelConnectivity,
               AbstractCubedSpherePanelConvention,
               AtmosGrid, ncells, nfaces, nrings, nlevels, face_cells, cell_area,
               ring_longitudes, ring_cell_count, cell_areas_by_latitude,
               n_levels, pressure_at_interface, level_thickness, floattype,
               panel_convention, panel_connectivity_for,
               panel_cell_center_lonlat,
               panel_cell_local_tangent_basis,
               cs_definition, coordinate_law, center_law, longitude_offset_deg,
               cs_definition_tag, coordinate_law_tag, center_law_tag,
               EquiangularCubedSphereDefinition, GMAOCubedSphereDefinition,
               default_panel_connectivity, gnomonic_panel_connectivity, reciprocal_edge,
               GnomonicPanelConvention, GEOSNativePanelConvention,
               EDGE_NORTH, EDGE_SOUTH, EDGE_EAST, EDGE_WEST
using ..State: AbstractMassBasis, DryBasis, MoistBasis
using ..Regridding: build_regridder, apply_regridder!
using ..Quantities: QuantityKind, IntensiveCellField, ExtensiveCellField,
                    HorizontalVectorField, HorizontalFluxField
using ..MetDrivers: LatLonBinaryGeometry, CubedSphereBinaryGeometry, TransportBinaryReader,
                    mesh_definition,
                    TransportBinaryHeader, binary_geometry, grid_type,
                    write_transport_binary,
                    TRANSPORT_BINARY_FORMAT_VERSION,
                    StreamingTransportBinaryWriter,
                    open_streaming_transport_binary, write_streaming_window!,
                    close_streaming_transport_binary!,
                    set_streaming_steps_per_window_schedule!,
                    set_transport_header_steps_per_window_schedule!,
                    open_streaming_cs_transport_binary, write_streaming_cs_window!,
                    read_era5_reduced_gaussian_geometry, read_era5_reduced_gaussian_mesh,
                    load_window!, load_flux_delta_window!,
                    has_tm5_convection, load_tm5_convection_window!,
                    load_surface_window!,
                    TransportBinaryContract, canonical_window_constant_contract,
                    validate_transport_contract!,
                    recompute_cm_from_dm_target!, recompute_faceindexed_cm_from_dm_target!,
                    verify_window_continuity, verify_window_continuity_ll,
                    verify_window_continuity_rg, verify_window_continuity_cs,
                    delta_semantics,
                    replay_tolerance, run_replay_gate,
                    structured_replay_layout, faceindexed_replay_layout

# Re-export the CS preprocessor contract surface. `cubed_sphere_contracts.jl`
# is the single source of truth for the per-window write-time gates that every
# CS-producing preprocessor (spectral, regrid, GEOS-native) must call; making
# these public lets external auditors (the binary inspector, focused tests,
# `scripts/diagnostics/*`) gate a binary against the same contract without
# duplicating logic.
export verify_write_replay_cs!,
       verify_substep_positivity_cs!,
       verify_cs_window_contract!,
       init_cs_positivity_accumulator,
       update_cs_positivity_accumulator,
       summarize_cs_positivity_status

# Typed window-contract surface. The abstract types
# (`AbstractWindowContract` / `AbstractWindowWorkspace` /
# `AbstractBinaryWriter`) plus the concrete LL / CS / RG contract
# structs and per-topology positivity kernels. Mass-basis tags
# (`AbstractMassBasis` / `DryBasis` / `MoistBasis`) are reused from
# `State.Basis` — `AbstractBinaryWriter{G, FT, Basis}` parametrizes
# directly on those existing nominals so a writer↔reader pairing
# mismatch is a compile-time `MethodError`. Exported so external
# auditors (focused tests, the binary inspector,
# `scripts/diagnostics/*`) can gate a binary against the same contract
# without duplicating logic.
export mass_basis_symbol, mass_basis_from_symbol
export AbstractWindowContract, AbstractWindowWorkspace, AbstractBinaryWriter
export SubstepSchedulePolicy, clamp_substeps, initial_substeps,
       required_substeps, next_substeps, rescale_substep_amounts!,
       contract_steps_for_window, set_contract_steps_schedule!
export ReadyWindow, PreverifiedWindow, PreprocessorRunCache
export verify_window!, update_accumulator!, summarize_status!
export contract_replay_tolerance, contract_cfl_limit, contract_require_positivity
export allocate_window_workspace, reset_workspace!,
       ingest_window!, drain_ready_windows!, flush_final_windows!
export LatLonBinaryWriter, ReducedGaussianBinaryWriter, CubedSphereBinaryWriter
export write_window!, close_streaming_binary!, promote_streaming_binary!,
       quarantine_streaming_binary!, writer_staging_path, writer_final_path
export UnifiedPreprocessorDay, run_unified_preprocessor_day!
export driver_windows_per_day, driver_ingest_window!,
       driver_drain_ready_windows!, driver_flush_final_windows!,
       driver_after_write_window!
export CubedSphereContract, LatLonContract, ReducedGaussianContract
export verify_substep_positivity_ll!, verify_ll_window_contract!,
       init_ll_positivity_accumulator, update_ll_positivity_accumulator,
       summarize_ll_positivity_status
export verify_substep_positivity_rg!, verify_rg_window_contract!,
       verify_boundary_stub_flux_rg,
       init_rg_positivity_accumulator, update_rg_positivity_accumulator,
       summarize_rg_positivity_status

# Met source abstraction (AbstractMetSettings + RawWindow)
include("met_sources.jl")

# Physical constants

# Statistics dependency for GEOS reader (level-orientation auto-detect)
using Statistics: mean

# Logging utilities
include("logging.jl")

# Vertical coordinate handling and level merging
include("vertical_coordinates.jl")

# Typed vertical-transform surface (vertical axis of the unified
# preprocessor). Wraps `merge_thin_levels` and
# `select_levels_echlevs` into first-class transform types so layer
# merging is available to every preprocessor pathway (LL/RG/CS ×
# spectral/native), not just the ERA5 spectral path.
include("vertical_transforms.jl")

# Global 6-panel Poisson balance for cubed-sphere grids
# (must precede target_geometry.jl which uses CSGlobalFaceTable)
include("cs_face_table.jl")
include("cs_poisson_solver.jl")
include("cs_poisson_balance.jl")

# Target grid geometry (LL, RG, and CS)
include("target_geometry.jl")

# GRIB spectral IO
include("spectral_io.jl")

# Spectral synthesis (Legendre + FFT → gridpoint)
include("spectral_synthesis.jl")

# Poisson balance (LL FFT + RG conjugate gradient)
include("mass_support.jl")

# Compressed-Laplacian Poisson balance for RG (replaces slow CG on LCM faces)
include("ring_poisson_balance.jl")

# Reduced Gaussian helpers (RG synthesis, RG balance, RG cm diagnosis)
include("reduced_transport_helpers.jl")
include("reduced_window_buffer.jl")
include("reduced_spectral_day.jl")
include("reduced_spectral_synthesis.jl")

# Cubed-sphere transport helpers (regrid, wind recovery, flux reconstruction)
include("cs_transport_helpers.jl")
include("cs_flux_reconstruction.jl")
include("cs_wind_rotation.jl")
include("cs_native_fluxes.jl")
# Flux construction options shared by the wind-derived paths (MERRA-2, ERA5 N320).
include("face_line_integrals.jl")
include("flux_construction.jl")

# Configuration parsing
include("configuration.jl")

# TM5 convection ec2tm conversion
# (Loaded before binary_pipeline so the LL process_day hook can
# reference TM5PreprocessingWorkspace / TM5CleanupStats by type.)
include("tm5_convection_conversion.jl")

# ERA5 physics NC → BIN converter + mmap reader
include("era5_physics_binary.jl")

# ERA5 single-level surface reader for raw PBL diffusion fields.
include("era5_surface_reader.jl")

# TM5 boundary-layer diffusion (Holtslag-Boville non-local PBL) column kernel.
include("tm5_bldiff.jl")

# TM5 convection preprocessor pipeline wiring
include("tm5_convection_pipeline.jl")

# Transport-binary workflows and shared preprocessing contracts.
include("binary_pipeline.jl")

# Native GEOS NetCDF reader
include("sources/geos.jl")
include("sources/geos_panels.jl")
include("sources/geos_read_window.jl")

# ERA5 native-GRIB reader: N320 settings + day handles, spectral synthesis
# (VO+D → U/V, LNSP → PS, reduced_gg Q reader), dry-basis layer-mass
# derivation, convection forecast (UDMF/DDMF/UDRF/DDRF), conservative
# regrid to a cubed-sphere target, and the per-window pipeline that
# wires them all together.
include("sources/era5.jl")
include("sources/era5_n320_window.jl")
include("sources/era5_n320_mass_convection.jl")
include("sources/era5_n320_to_cs.jl")

# Native MERRA-2 reader: MERRA2Settings + day handles + per-window LL field
# reader (PS/QV from inst3, U/V from tavg3). Drives the wind-derived → CS
# path that reproduces the GEOS-Chem CO₂ transport input. Included after the
# ERA5 sources (reader before the preprocessor that uses it).
include("sources/merra2.jl")

# TOML-driven met-source factory
include("sources/loader.jl")

# Typed met-reader surface (source axis of the unified preprocessor;
# additive facade over `AbstractMetSettings` machinery above). Includes
# here so the abstract types are visible to
# `transport_binary/cubed_sphere_geos.jl` even though the existing
# `process_day` orchestrators are not yet ported.
include("met_readers.jl")

# GEOS → CS passthrough orchestrator
include("transport_binary/geos_cs_mass_helpers.jl")
include("transport_binary/geos_cs_omega.jl")
include("transport_binary/geos_cs_resolution.jl")
include("transport_binary/geos_cs_window.jl")
include("transport_binary/cubed_sphere_geos.jl")

# Experimental nested CS-binary operator restriction (for example C90 → C30).
# Kept after the GEOS workflow so it reuses the same audited block-sum rules.
include("transport_binary/cubed_sphere_coarsen.jl")

# ERA5 N320 → CS transport-binary writer. Drives one UTC day end-to-end
# through the per-window pipeline in `sources/era5*.jl` plus the
# C180 dry-mass re-derivation, wind rotation, face flux reconstruction,
# Poisson balance, and v4 writer.
include("transport_binary/era5_n320_regrid.jl")

# MERRA-2 wind-derived → CS transport-binary writer. Near-clone of the ERA5
# N320 writer: direct MERRA-2 NetCDF read + conservative regrid to C180,
# wind-derived flux reconstruction + Cameron-Smith column pressure-fix
# (Poisson balance), 8 windows/day. Reproduces the validated GEOS-Chem CO₂
# transport input path; purely additive. Included after the ERA5 N320 writer
# so `_fill_cs_mass_delta_payload!` is in scope.
include("transport_binary/merra2_latlon_regrid.jl")

# Met source abstraction
export AbstractMetSettings, RawWindow
export read_window!, source_grid, windows_per_day
export has_convection, has_surface, has_vdiff_fields, has_pbl_eflux, has_cmfmc_cloud_base
export open_day, close_day!, allocate_raw_window

# GEOS native NetCDF reader
export AbstractGEOSSettings, GEOSSettings, GEOSITSettings, GEOSFPSettings
export GEOSDayHandles, open_geos_day, close_geos_day!
export GEOSFPNativeDayHandles, geosfp_native_hourly_ctm_path
export geos_collection_path, detect_level_orientation
export endpoint_dry_mass, endpoint_dry_mass!

# ERA5 native-GRIB reader: settings, day handles, and the
# per-window spectral-synthesis surface for the N320 source grid.
export AbstractERA5GRIBSettings, ERA5GRIBSettings, ERA5N320Settings
export ERA5GRIBDayHandles, open_era5_day, close_era5_day!
export era5_grib_path
export ERA5N320SpectralWorkspace, ERA5N320WindowFields
export allocate_era5_n320_spectral_workspace, allocate_era5_n320_window_fields
export discover_era5_n320_source_grid, discover_era5_spectral_truncation
export read_era5_n320_window_fields!
export ERA5N320DryMassFields, allocate_era5_n320_dry_mass_fields,
       derive_n320_dry_mass!, derive_c180_dry_mass!, n320_cell_areas
export ERA5C180RegridFields, ERA5C180RegridWorkspace,
       allocate_era5_c180_regrid_fields, allocate_era5_c180_regrid_workspace,
       regrid_n320_to_c180!
export ERA5N320ConvectionFields, allocate_era5_n320_convection_fields,
       read_era5_n320_convection_window!, era5_convection_hour_address
export ERA5C180RawConvectionFields, ERA5C180TM5ConvectionFields,
       allocate_era5_c180_raw_convection_fields,
       regrid_n320_raw_convection_to_c180!, derive_c180_tm5_convection!
export ERA5N320ToC180Pipeline, allocate_era5_n320_to_c180_pipeline,
       process_era5_n320_window!
export process_era5_n320_to_cs_day

# Native MERRA-2 reader + wind-derived → CS writer (GEOS-Chem CO₂ path).
export MERRA2Settings, MERRA2DayHandles
export open_merra2_day, close_merra2_day!, merra2_path, merra2_stream_code
export read_merra2_window_fields, read_merra2_next_day_endpoint, read_merra2_physics_window
export MERRA2Archive, NASAArchive, GEOSChemArchive, merra2_archive
export MERRA2LevelOrder, TopDown, SurfaceFirst, detect_merra2_level_order
export validate_merra2_settings, merra2_window_physics!,
       allocate_merra2_window_physics, convective_cloud_base!
export MERRA2ToC180Pipeline, allocate_merra2_to_c180_pipeline,
       process_merra2_window!, process_merra2_to_cs_day

# Met-source TOML factory + vertical-coordinate helper used by GEOS CLI
export load_met_settings, load_hybrid_coefficients

# Typed met-reader surface (source axis)
export AbstractMetReader, AbstractChainPolicy, NoChain, ChainedMass
export GEOSNativeReader, ERA5SpectralReader, ERA5SpectralSettings
export open_reader, close_reader!, end_of_day_seed, set_end_of_day_seed!
export native_vertical, window_metadata
# Note: `windows_per_day` and `read_window!` are already exported above
# (the existing `AbstractMetSettings` trait surface). The reader surface
# adds typed methods on the same generic functions.

# Typed vertical-transform surface (vertical axis)
export AbstractVerticalTransform, VerticalPlan
export IdentityVertical, MergeByIndex, MergeLayersThinnerThan,
       MergeAbovePressure, LevelSelection, PressureOverlap
export AbstractFieldKind
export MassField, TracerMassField, MassFluxField, PressureFluxField,
       ConvectionInterfaceFlux, ConvectionTendencyField,
       IntensiveCenterField, SurfaceField
export plan_vertical, apply_vertical!

# Exports for the CLI script and advanced users
export build_target_geometry, target_summary
export process_day, regrid_transport_binary, regrid_ll_binary_to_cs
export ec2tm!
export ec2tm_from_rates!, TM5CleanupStats
export dz_hydrostatic_virtual!, dz_hydrostatic_constT!
export convert_era5_physics_nc_to_bin
export ERA5PhysicsBinaryReader, ERA5PhysicsBinaryHeader
export open_era5_physics_binary, close_era5_physics_binary, get_era5_physics_field
export tm5_native_fields_for_hour!, merge_tm5_field_3d!
export TM5PreprocessingWorkspace, allocate_tm5_workspace
export compute_tm5_merged_hour_on_source!, log_tm5_cleanup_stats
export tm5_copy_or_regrid_ll!
export resolve_tm5_convection_settings

end # module Preprocessing
