"""
    Preprocessing

Transport binary generation from meteorological input.

Converts raw meteorology into mass-balanced transport binaries ready for the
runtime `TransportBinaryDriver`. Supported sources and targets:

- ERA5 spectral GRIB (VO, D, LNSP) → lat-lon, reduced-Gaussian, or cubed sphere
- GEOS native cubed-sphere NetCDF (GEOS-IT, GEOS-FP) → cubed sphere (same
  resolution or nested block coarsening)
- ERA5 N320 native GRIB → cubed sphere (conservative regrid)
- MERRA-2 lat-lon NetCDF → cubed sphere (conservative regrid)
- an existing lat-lon transport binary → cubed sphere (`regrid_transport_binary`)

## Architecture

The pipeline dispatches on `AbstractTargetGeometry` subtypes:

    AbstractTargetGeometry
    ├── LatLonTargetGeometry              — regular lat-lon (any Nx × Ny)
    ├── ReducedGaussianTargetGeometry     — native ERA5 RG (O90, O160, N320, …)
    └── CubedSphereTargetGeometry         — convention-aware CS (C24, C90, C180, …)

and on the source settings type. Each supported pair has a `process_day` method:

    process_day(date, grid::LatLonTargetGeometry,          settings::ERA5SpectralSettings,     vertical; …)
    process_day(date, grid::ReducedGaussianTargetGeometry, settings::ERA5SpectralSettings,     vertical; …)
    process_day(date, grid::CubedSphereTargetGeometry,     settings::ERA5SpectralSettings,     vertical; …)
    process_day(date, grid::CubedSphereTargetGeometry,     settings::AbstractGEOSSettings,     vertical; …)
    process_day(date, grid::CubedSphereTargetGeometry,     settings::AbstractERA5GRIBSettings, vertical; …)
    process_day(date, grid::CubedSphereTargetGeometry,     settings::MERRA2Settings,           vertical; …)

## ERA5 spectral pipeline (per day)

1. Read ERA5 spectral GRIB (VO, D, LNSP) — `spectral_io.jl`
2. Spectral synthesis (Legendre + FFT → gridpoint) — `spectral_synthesis.jl`
3. Merge native 137L → transport levels — `vertical_coordinates.jl`
4. Pin global mean ps (mass fix) — `pin_global_mean_ps!` /
   `pin_global_mean_ps_using_qv!` in `mass_support.jl`
5. Column Poisson balance of the horizontal mass fluxes (header key
   `horizontal_balance = "column"`): one Poisson solve for the
   column-integrated divergence against the column mass tendency, each face's
   correction spread over the layers by the column weights
   (`spread_column_correction!`; air mass in hybrid layers by default),
   so `cm` keeps the vertical motion:
   - LL: FFT Poisson solve — `balance_column_mass_fluxes!`, `mass_support.jl`
   - RG: compressed-Laplacian CG — `balance_compressed_column_fluxes!`,
     `ring_poisson_balance.jl`
   - CS: global 6-panel graph-Laplacian CG — `balance_cs_column_mass_fluxes!`,
     `cs_poisson_balance.jl`
6. Diagnose `cm` from explicit endpoint mass targets — replay continuity
7. Write transport binary with declared payload semantics and replay checks

The GEOS, ERA5 N320, MERRA-2, and binary-regrid paths implement their own
mass construction or regridding, flux balancing, and `cm` diagnosis (with an
optional dry-mass pin where the source has one); every path ends in the same
write-time replay and positivity gates.

The high-level transport-binary workflows are split under
`transport_binary/`. Each meteorological source/target pair owns a
`process_day` method (binary regridding has its own entry point) while
shared binary metadata, endpoint-delta, and replay helpers stay in common
files.

## Usage

```bash
julia --project=. scripts/preprocessing/preprocess_transport_binary.jl config.toml --day 2021-12-01
```

Advanced users can call `process_day` directly from Julia.
"""
module Preprocessing

import ..Grids
import ..Grids: panel_convention_tag

using Dates
using Logging
using FFTW
using Printf
using JSON3
using SHA
using TOML
using LinearAlgebra: mul!, dot
using SparseArrays: SparseMatrixCSC, sparse
using NCDatasets

using ..ConfigChecks: config_bool as _config_bool, check_known_keys, parse_float_type
using ..LogLevels: log_level, with_log_level
using ..AtomicFiles: atomic_publish, open_atomic_staging,
                     commit_atomic_staging, abandon_atomic_staging
using GRIB

import ..expand_data_path
import ..source_revision

# Source settings and binary readers extend the same public capability queries.
import ..MetDrivers: has_surface, has_vdiff_fields, has_pbl_eflux, has_cmfmc_cloud_base
using ..Architectures: CPU
# Physical constants, numerical defaults and met-source constants
# (`src/Parameters`) used by the files below.
using ..Parameters: EARTH_RADIUS, IFS_EARTH_RADIUS, STANDARD_GRAVITY, STANDARD_PRESSURE,
    TM5_CONSTANTS, R_DRY_AIR,
                    VIRTUAL_TEMPERATURE_FACTOR, HYDROSTATIC_REFERENCE_TEMPERATURE,
                    GLOBAL_MEAN_DRY_SURFACE_PRESSURE, GLOBAL_MEAN_SPECIFIC_HUMIDITY,
                    DEFAULT_POSITIVITY_CFL_LIMIT, DEFAULT_SUBSTEP_CFL_TARGET,
                    ERA5_N320_SUBSTEP_CFL_TARGET, COARSEN_SUBSTEP_CFL_TARGET,
                    DEFAULT_MAX_STEPS_PER_WINDOW, ADAPTIVE_SUBSTEP_MAX_REFINEMENTS,
                    DEFAULT_MET_INTERVAL_SECONDS,
                    DEFAULT_POISSON_BALANCE_TOL, DEFAULT_POISSON_BALANCE_MAX_ITER,
                    DEFAULT_POISSON_PROJECT_EVERY, HORIZONTAL_BALANCE_TAG,
                    HYBRID_LAYER_MIN_DB, COLUMN_CLOSURE_TOL_KG, OMEGA_FIDELITY_NRMSE_LIMIT,
                    OMEGA_FIDELITY_CORRELATION_MIN, OMEGA_FIDELITY_SLOPE_MIN,
                    OMEGA_FIDELITY_SLOPE_MAX,
                    DEFAULT_GEOS_SMOOTHING_SWEEPS, DEFAULT_GEOS_SMOOTHING_WEIGHT,
                    ERA5_NATIVE_LEVEL_COUNT, ERA5_MET_INTERVAL_SECONDS,
                    GEOS_MASS_FLUX_DT_SECONDS, GEOS_MET_INTERVAL_SECONDS,
                    GEOS_3HOURLY_INTERVAL_SECONDS,
                    SURFACE_AIR_TEMPERATURE_RANGE, SURFACE_HEAT_FLUX_MAX_ABS,
                    ERA5_L137_ECHLEVS_PRESETS,
                    TM5_CONVECTION_SOURCE_KEY, TM5_EC2TM_CONVENTION_KEY, TM5_EC2TM_CONVENTION
using ..Grids: cell_centers, cell_area_matrix, unit3, arc3, slerp3, unit_xyz_from_radians,
               seam_mirror_sign, face_connectivity, cell_areas, cs_total_area, dot3, cross3,
               CS_PANEL_COUNT, cs_panel_flat_range, unpack_flat_to_panels_2d!,
               unpack_flat_to_panels_3d!, CSGlobalFaceTable, build_cs_global_face_table,
               cs_edge_cell, cs_edge_face_location, cs_flat_index, cs_panel_ij,
               default_cs_definition, parse_panel_convention, parse_cs_definition,
               cs_geometry_tags
using ..Grids: LatLonMesh, ReducedGaussianMesh, CubedSphereMesh, cs_face_edge_lengths,
    cs_corner_xyz,
               HybridSigmaPressure, PanelConnectivity,
               AbstractCubedSpherePanelConvention,
               AtmosGrid, ncells, nfaces, nrings, nlevels, face_cells, cell_area,
               ring_longitudes, cell_areas_by_latitude,
               n_levels, pressure_at_interface, level_thickness, panel_convention, panel_cell_center_lonlat,
               panel_cell_local_tangent_basis,
               cs_definition, coordinate_law, center_law, longitude_offset_deg,
               cs_definition_tag, coordinate_law_tag, center_law_tag,
               EquiangularCubedSphereDefinition, GMAOCubedSphereDefinition,
               default_panel_connectivity, reciprocal_edge,
               GnomonicPanelConvention, GEOSNativePanelConvention,
               EDGE_NORTH, EDGE_SOUTH, EDGE_EAST, EDGE_WEST
import ..State
using ..State: AbstractMassBasis, DryBasis, MoistBasis,
               TotalSurfacePressure, DrySurfacePressure
using ..Regridding: build_regridder, apply_regridder!, coarsen_sum!, coarsen_area_weighted!,
                    default_regrid_cache_dir
using ..Quantities: QuantityKind, IntensiveCellField, ExtensiveCellField
using ..GriddedIO: source_latlon_mesh, SnappedFaces, find_coordinate, LongitudeAxis, LatitudeAxis
using ..MetDrivers: poisson_balance_target, write_padded_header!, section_elements,
                    LatLonBinaryGeometry, CubedSphereBinaryGeometry, TransportBinaryReader,
                    mesh_definition,
                    binary_geometry, grid_type,
                    binary_dry_mass_target, dry_mass_pin_enabled,
                    TRANSPORT_BINARY_FORMAT_VERSION,
                    StreamingTransportBinaryWriter,
                    open_streaming_transport_binary, write_streaming_window!,
                    close_streaming_transport_binary!, abandon_streaming_transport_binary!,
                    FLUX_CERTIFICATE_KEY_PREFIXES, is_certificate_key,
                    COLUMN_CLOSURE_RULE, FLUX_FIDELITY_SCHEMA,
                    set_streaming_steps_per_window_schedule!,
                    set_transport_header_steps_per_window_schedule!,
                    open_streaming_cs_transport_binary, write_streaming_cs_window!,
                    load_window!, load_flux_delta_window!,
                    has_tm5_convection, load_tm5_convection_window!,
                    TM5_CONVECTION_MASS_BASIS_KEY, tm5_convection_mass_basis,
                    load_surface_window!,
                    TransportBinaryContract, canonical_window_constant_contract,
                    validate_transport_contract!,
                    recompute_cm_from_dm_target!,
                    verify_window_continuity,
                    verify_window_continuity_rg, verify_window_continuity_cs,
                    delta_semantics, flux_kind, flux_application_seconds,
                    surface_pressure_basis,
                    replay_tolerance, run_replay_gate, ReplayScratch,
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
export AbstractWindowContract, AbstractWindowWorkspace, AbstractBinaryWriter
export SubstepSchedulePolicy, clamp_substeps, initial_substeps,
       required_substeps, next_substeps, rescale_substep_amounts!,
       set_contract_steps_schedule!
export ReadyWindow, PreverifiedWindow, PreprocessorRunCache
export AbstractFluxFidelityEvidence, AbstractOmegaFidelityEvidence,
       WindDerivedFluxEvidence, NativeFluxEvidence, NoFluxFidelityEvidence,
       SourceOmegaEvidence, NoSourceOmegaEvidence, FluxFidelityEvidenceBundle,
       FluxFidelitySample, OmegaFidelitySample, FidelitySampleMap,
       FluxFidelityAccumulator, verify_flux_fidelity!, flux_fidelity_header!
export verify_window!, update_accumulator!, summarize_status!
export allocate_window_workspace,
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

# Pipeline stages load in dependency order. The facade preserves script names.
include("Core/Core.jl")
import .Core: AbstractChainPolicy, AbstractColumnWeights, AbstractERA5GRIBSettings,
    AbstractGEOSSettings, AbstractMetReader, AbstractMetSettings, AbstractTargetGeometry,
    COLUMN_WEIGHT_NAMES, ChainedMass, CubedSphereTargetGeometry, ERA5SpectralSettings,
    FlushingLogger, GEOSNativeReader, LatLonTargetGeometry, MAX_SPECIFIC_HUMIDITY,
    MassWeightedColumn, NoChain, PACKAGE_ROOT, RawWindow, ReducedGaussianTargetGeometry,
    source_grid, allocate_raw_window, build_target_geometry, check_column_weight_levels,
    check_spectral_column_weights, checked_mass_fix_config, close_day!, close_reader!,
    column_weights, day_threading_enabled, ensure_supported_target, grid_kind, has_cmfmc_cloud_base,
    has_convection, has_pbl_eflux, has_surface, has_vdiff_fields, log_preprocessor_configuration,
    nlat, nlon, open_day, open_preprocessing_staging, open_reader, package_path,
    panel_convention_tag, parse_column_balance_weights, parse_flux_construction,
    preprocessor_pair_supported, read_era5_reduced_gaussian_geometry,
    read_era5_reduced_gaussian_mesh, read_window!, reject_removed_balance_mode,
    reject_removed_horizontal_balance_keyword, reject_removed_tm5_convection_route,
    require_configured_terminal_day, resolve_balance_settings, resolve_column_balance_weights,
    resolve_mass_basis, resolve_mass_fix_settings, resolve_output_float_type,
    resolve_preprocessing_cache_settings, resolve_runtime_settings, resolve_spectral_flux_settings,
    resolve_surface_settings, spectral_column_weights, spread_column_correction!,
    supports_day_threading, supports_initial_endpoint_mass_pin,
    supports_spectral_massflux_preprocessing, target_balance_geometry, target_definition_tag,
    target_header_metadata, target_spectral_truncation, target_summary,
    validate_column_balance_weights, windows_per_day, with_column_weights_record, with_replay_record
include("Vertical/Vertical.jl")
import .Vertical: AbstractFieldKind, AbstractVerticalTransform, ConvectionInterfaceFlux,
    ConvectionTendencyField, IdentityVertical, IntensiveCenterField, LevelSelection, MassField,
    MassFluxField, MergeAbovePressure, MergeLayersThinnerThan, VerticalPlan,
    ab_coefficients, apply_vertical!, build_vertical_setup, echlevs_preset,
    load_era5_vertical_coordinate, load_hybrid_coefficients, merge_thin_levels, plan_vertical,
    resolve_coefficients_path, select_levels_echlevs, sum_merged_layers!
include("Spectral/Spectral.jl")
import .Spectral: LLPoissonWorkspace, ReducedSpectralSynthesis, SPECTRAL_DAY_CACHE_VERSION,
    apply_dry_basis_native!, available_spectral_dates, balance_column_mass_fluxes!,
    balance_mass_fluxes!, compute_air_mass!, compute_dp!, compute_legendre_column!,
    compute_mass_fluxes!, era5_spectral_paths, fill_ring_spectrum!, fill_xface_surface_pressure!,
    fill_yface_surface_pressure!, load_spectral_day_cache, mass_fix_target_ps_dry_record,
    mass_target_offset, maybe_preload_qv_day, merge_cell_field!, merge_qv!, next_day_hour0,
    normalize_lon_to_centered, pin_global_mean_ps!, pin_global_mean_ps_using_qv!,
    read_daily_qv_from_thermo, read_day_spectral, read_day_spectral_streaming, read_hour0_spectral,
    read_qv_from_thermo, read_spectral_coeffs!, read_thermo_coordinates,
    recompute_cm_from_divergence!, recompute_native_mass_fields!,
    remap_extensive_column_pressure_overlap!, remap_extensive_field_pressure_overlap!,
    remap_qv_pressure_overlap!, ring_max_wavenumber, spectral_day_cache_path, spectral_to_grid!,
    spectral_to_native_fields!, stagger_winds!, synthesize_reduced!, validate_spectral_coverage,
    validate_spectral_header, vertical_mapping_method, vod2uv!, write_spectral_day_cache,
    xface_surface_pressure, yface_surface_pressure

using Statistics: mean

# Deferred mass closure and native GRIB geometry register Core interfaces.
include("cs_poisson_solver.jl")
target_balance_geometry(ft::CSGlobalFaceTable) =
    (cs_cell_face_degree(ft), CSPoissonScratch(ft.nc))
include("cs_poisson_balance.jl")
include("sources/era5_grib_geometry.jl")

# Compressed-Laplacian Poisson balance for RG (replaces slow CG on LCM faces)
include("ring_poisson_balance.jl")

# Reduced Gaussian helpers (RG synthesis, RG balance, RG cm diagnosis)
include("reduced_transport_helpers.jl")
include("reduced_window_buffer.jl")
include("reduced_spectral_day.jl")

# Cubed-sphere transport helpers (regrid, wind recovery, flux reconstruction)
include("cs_transport_helpers.jl")
include("cs_flux_reconstruction.jl")
include("cs_wind_rotation.jl")
include("cs_native_fluxes.jl")
# Flux construction options shared by the wind-derived paths (MERRA-2, ERA5 N320).
include("face_line_integrals.jl")
include("flux_construction.jl")

# Convective mass fluxes on the binary's air-mass basis (shared with GEOS).
include("convection_mass_basis.jl")

# TM5 convection ec2tm conversion (used by the ERA5 N320 → CS pipeline).
include("tm5_convection_conversion.jl")

# ERA5 single-level surface reader for raw PBL diffusion fields.
include("era5_surface_reader.jl")

# TM5 boundary-layer diffusion (Holtslag-Boville non-local PBL) column kernel.
include("tm5_bldiff.jl")

# Transport-binary workflows and shared preprocessing contracts.
include("transport_binary/core.jl")
include("transport_binary/flux_fidelity.jl")
include("transport_binary/window_contracts.jl")
include("transport_binary/latlon_workspaces.jl")
include("transport_binary/latlon_contracts.jl")
include("transport_binary/cubed_sphere_contracts.jl")
include("transport_binary/reduced_gaussian_contracts.jl")
include("transport_binary/writer_adapters.jl")
include("transport_binary/driver.jl")
include("transport_binary/topology_dispatch.jl")
include("transport_binary/cubed_sphere_spectral.jl")
include("transport_binary/cubed_sphere_regrid.jl")
include("transport_binary/latlon_spectral.jl")
include("transport_binary/entrypoint.jl")

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
# target-grid dry-mass re-derivation, wind rotation, face flux reconstruction,
# Poisson balance, and v4 writer.
include("transport_binary/era5_n320_regrid.jl")

# MERRA-2 wind-derived → CS transport-binary writer. Near-clone of the ERA5
# N320 writer: direct MERRA-2 NetCDF read + conservative regrid to the CS
# target, wind-derived flux reconstruction + Cameron-Smith column
# pressure-fix (Poisson balance), 3-hourly source blocks written as 8 or 24
# windows/day. Reproduces the validated GEOS-Chem CO₂ transport input path.
# Included after the ERA5 N320 writer so `_fill_cs_mass_delta_payload!` is in
# scope.
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
export read_era5_n320_window_fields!, read_era5_n320_humidity!
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
export ERA5HumidityRemap, open_era5_humidity_remap, remap_era5_humidity!
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
export GEOSNativeReader, ERA5SpectralSettings
export open_reader, close_reader!
# Note: `windows_per_day` and `read_window!` are already exported above
# (the existing `AbstractMetSettings` trait surface). The reader surface
# adds typed methods on the same generic functions.

# Typed vertical-transform surface (vertical axis)
export AbstractVerticalTransform, VerticalPlan
export IdentityVertical, MergeLayersThinnerThan,
       MergeAbovePressure, LevelSelection
export AbstractFieldKind
export MassField, MassFluxField,
       ConvectionInterfaceFlux, ConvectionTendencyField,
       IntensiveCenterField
export plan_vertical, apply_vertical!

# Exports for the CLI script and advanced users
export build_target_geometry, target_summary, FlushingLogger
export process_day, regrid_transport_binary, regrid_ll_binary_to_cs
export ec2tm_from_rates!, TM5CleanupStats
export interface_specific_humidity, ecmwf_convection_to_basis!, tm5_convection_to_basis!
export dz_hydrostatic_virtual!

end # module Preprocessing
