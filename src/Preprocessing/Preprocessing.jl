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

Target workflows are the submodules `LatLonTarget`, `ReducedGaussianTarget` and
`CubedSphereTarget` (folders `Targets/LatLon`, `Targets/ReducedGaussian`,
`Targets/CubedSphere`); shared binary APIs live in `TransportBinary`. Native
readers and day drivers live in the independent `GEOS`, `ERA5` and `MERRA2`
submodules under `Sources/`. The parent owns only entrypoint dispatch.

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

using ..ConfigChecks: config_bool as _config_bool, config_int, config_float, config_date, parse_choice, check_known_keys, parse_float_type
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
# Experimental nested CS-binary coarsener (`Targets/CubedSphere`); the facade
# exported it before PP3 and `scripts/preprocessing/coarsen_cs_transport_binary.jl`
# and the binary-format manual name it.
export coarsen_nested_cs_transport_binary

# Pipeline stages load in dependency order. The facade preserves script names.
include("Core/Core.jl")
import .Common: check_native_closure_config, AbstractChainPolicy, AbstractColumnWeights, AbstractERA5GRIBSettings,
    AbstractGEOSSettings, AbstractMetReader, AbstractMetSettings, AbstractTargetGeometry,
    COLUMN_WEIGHT_NAMES, ChainedMass, CubedSphereTargetGeometry, ERA5SpectralSettings,
    FlushingLogger, LatLonTargetGeometry, MAX_SPECIFIC_HUMIDITY,
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
    target_header_metadata, target_spectral_truncation, target_summary, validate_tm5_pbl_target,
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

include("MassClosure/MassClosure.jl")
import .MassClosure: AbstractFaceLengths,
    AbstractWindRegrid,
    AlongFaceFilter,
    COLUMN_WEIGHT_KINDS,
    CSPoissonScratch,
    CSVectorFaceGeometry,
    CartesianWindRegrid,
    CellCenterlineLengths,
    ColumnResidualWeight,
    CompressedLaplacian,
    DryMassFluxThickness,
    EdgeLengths,
    FACE_FLUX_KINDS,
    FACE_INTERPOLATION_KINDS,
    FACE_LENGTH_KINDS,
    FLUX_THICKNESS_KINDS,
    HybridBWeightedColumn,
    HybridMassWeightedColumn,
    LineIntegralFaceFluxes,
    MoistFluxThickness,
    OuterCellPair,
    PanelAverageFluxes,
    ScalarWindRegrid,
    VectorFaceFluxes,
    WIND_REGRID_KINDS,
    vector_face_value,
    xface_length,
    yface_length,
    apply_cs_flux_correction!,
    balance_compressed_column_fluxes!,
    balance_cs_column_mass_fluxes!,
    balance_cs_global_mass_fluxes!,
    build_compressed_laplacian,
    check_column_weight_levels,
    column_weights,
    compute_cs_cm_pressure_fixer!,
    cs_cell_bases,
    cs_cell_face_degree,
    cs_column_balance_projected_linf,
    cs_column_roundoff_tolerance!,
    cs_face_corners,
    cs_face_divergence!,
    cs_face_fluxes!,
    cs_graph_laplacian_mul!,
    cs_linf,
    cs_vector_face_fluxes!,
    diagnose_balanced_cs_cm!,
    diagnose_cs_cm!,
    east_north_to_cartesian,
    face_flux_method,
    face_fluxes!,
    fill_cs_layer_thickness!,
    flux_header_face_interpolation,
    flux_header_wind_regrid,
    flux_thickness_method,
    geos_native_to_face_flux!,
    has_source_line_integrals,
    line_integral_face_fluxes!,
    point_weights,
    prepare_cell_winds!,
    project_mean_zero!,
    require_cs_column_balance,
    recheck_cs_column_closure,
    reconstruct_cs_fluxes!,
    recover_ll_cell_center_winds!,
    regrid_winds!,
    rotate_winds_to_panel_local!,
    solve_compressed_poisson_pcg!,
    solve_cs_poisson_pcg!,
    source_cell_centers,
    spread_column_correction!,
    sync_all_cs_boundary_mirrors!,
    sync_cs_mirrors!,
    validate_flux_construction,
    validate_flux_thickness,
    validate_flux_time_sampling,
    wind_basis_trig,
    build_wind_regrid
include("PhysicsInputs/PhysicsInputs.jl")
import .PhysicsInputs: BLDiffColumnScratch,
    BLDiffConstants,
    BLDiffDiag,
    ERA5SurfaceReader,
    ERA5_SURFACE_MAX_MASKED_FRACTION,
    ERA5_SURFACE_TIME_DIMS,
    ERA5_SURFACE_VAR_CANDIDATES,
    ERA5_USTAR_DRAG_COEFFICIENT,
    TM5CleanupStats,
    fill_masked_surface,
    resolve_era5_surface_path,
    cbrt_pos,
    close_era5_surface_reader,
    dz_hydrostatic_virtual!,
    ec2tm_from_rates!,
    ecmwf_convection_to_basis!,
    era5_surface_source_mesh,
    era5_tm5_convection_header,
    interface_specific_humidity,
    load_era5_surface_window,
    mix2_at,
    open_era5_surface_reader,
    reset_bldiff_diag!,
    tm5_bldiff_center_kz_column!,
    tm5_bldiff_dkg_column!,
    tm5_bldiff_kvh_column!,
    tm5_convection_to_basis!

# Shared binary APIs load below the target implementations that extend them.
include("TransportBinary/TransportBinary.jl")
import .TransportBinary: cached_or_build!, ranked_fidelity_ids
import .TransportBinary: AbstractBinaryWriter,
    AbstractColumnClosureRecord,
    AbstractFluxFidelityEvidence,
    AbstractFluxFidelityReason,
    AbstractOmegaFidelityEvidence,
    AbstractWindowContract,
    AbstractWindowWorkspace,
    CapturedNativeFluxSamples,
    CapturedWindFluxSamples,
    ColumnClosureRecord,
    CubedSphereBinaryWriter,
    EvidenceNotAttached,
    FidelitySampleMap,
    FluxBalanceSample,
    FluxFidelityAccumulator,
    FluxFidelityEvidenceBundle,
    FluxFidelitySample,
    GEOS_NATIVE_OMEGA_BASELINE,
    HEADER_SIZE,
    HybridLayerThickness,
    LatLonBinaryWriter,
    LineIntegralFidelityStencil,
    MERRA2_WIND_OMEGA_BASELINE,
    NativeFluxEvidence,
    NativeFluxesNotRederivable,
    NoColumnClosureRecord,
    NoFluxFidelityEvidence,
    NoSourceOmegaEvidence,
    OMEGA_FIDELITY_PRESSURE_BANDS,
    OUTPUT_REUSE_CONTRACT_KEYS,
    OmegaFidelityBaseline,
    OmegaFidelityCells,
    OmegaFidelitySample,
    POLE_ROW_ZONAL_FLUX_KEY,
    PreprocessorRunCache,
    PreverifiedWindow,
    ReadyWindow,
    ReducedGaussianBinaryWriter,
    SourceHasNoOmega,
    SourceNotOpened,
    SourceOmegaEvidence,
    SourcePayloadOnly,
    StructuredFidelitySampler,
    SubstepSchedulePolicy,
    UnifiedPreprocessorDay,
    WindDerivedFluxEvidence,
    allocate_window_workspace,
    build_v4_header,
    capture_cs_wind_flux_samples,
    capture_faceindexed_flux_samples,
    capture_native_flux_samples,
    capture_structured_wind_flux_samples,
    clamp_substeps,
    close_streaming_binary!,
    column_closure_record,
    combine_captured_flux_samples,
    drain_ready_windows!,
    driver_after_write_window!,
    driver_before_close_writer!,
    driver_drain_ready_windows!,
    driver_flush_final_windows!,
    driver_ingest_window!,
    driver_windows_per_day,
    ensure_preprocessor_pair_supported,
    exact_steps_per_window,
    existing_output_schema_matches,
    expected_payload_sections,
    faceindexed_wind_evidence,
    fidelity_reason,
    fidelity_sample_rule,
    fidelity_status_label,
    flush_final_windows!,
    flux_fidelity_header!,
    flux_fidelity_method,
    full_window_column_closure_record,
    generation_fingerprint,
    ingest_window!,
    init_positivity_accumulator,
    initial_substeps,
    merge_captured_flux_samples,
    native_flux_evidence,
    next_substeps,
    output_binary_path,
    payload_only_fidelity_header!,
    process_day,
    promote_streaming_binary!,
    quarantine_streaming_binary!,
    required_substeps,
    rescale_substep_amounts!,
    rescaled_column_closure_record,
    resolve_qv_requirements,
    run_unified_preprocessor_day!,
    scale_captured_flux_samples,
    script_provenance,
    set_contract_steps_schedule!,
    settings_include_surface,
    source_omega_evidence_cs,
    splitmix64,
    structured_wind_evidence,
    summarize_positivity_status,
    summarize_status!,
    ulp_distance,
    update_accumulator!,
    update_positivity_accumulator,
    validate_cfl_limit,
    validate_replay_tol,
    validate_staged_binary!,
    validate_staged_size,
    verify_column_closure!,
    verify_flux_fidelity!,
    verify_omega_fidelity!,
    verify_substep_positivity_cs!,
    verify_window!,
    wind_derived_evidence,
    window_byte_sizes,
    window_element_counts,
    write_array!,
    write_window!,
    writer_final_path,
    writer_staging_path
include("Targets/LatLon/LatLon.jl")
import .LatLonTarget: AbstractQVWorkspace,
    LLNextDayFields,
    LLSpectralUnifiedDriverContext,
    LLWorst,
    LatLonBilinearWeights,
    LatLonContract,
    LatLonDeferredBinaryWriter,
    LatLonSpectralWindowWorkspace,
    MergeWorkspace,
    NativeQVWorkspace,
    NoQVWorkspace,
    SpectralTransformWorkspace,
    WindowStorage,
    allocate_merge_workspace,
    allocate_qv_workspace,
    allocate_transform_workspace,
    allocate_window_storage,
    apply_dry_basis_if_needed!,
    apply_mass_fix_if_needed!,
    apply_poisson_balance!,
    close_pole_faces!,
    compute_window_deltas!,
    fill_qv_endpoints!,
    fill_window_mass_tendency!,
    has_daily_qv,
    init_ll_positivity_accumulator,
    interpolate_ll_qv!,
    log_mass_fix_configuration,
    merge_native_window!,
    merge_qv_output!,
    next_day_merged_fields,
    process_window!,
    read_next_day_qv!,
    read_window_qv!,
    should_log_window,
    store_qv_output!,
    store_window_fields!,
    summarize_ll_positivity_status,
    update_ll_positivity_accumulator,
    verify_ll_window_contract!,
    verify_substep_positivity_ll!
include("Targets/ReducedGaussian/ReducedGaussian.jl")
import .ReducedGaussianTarget: RGSpectralUnifiedDriverContext,
    RGWorst,
    ReducedFidelitySampler,
    ReducedGaussianContract,
    ReducedGaussianSpectralWindowWorkspace,
    ReducedMergeWorkspace,
    ReducedQVWorkspace,
    ReducedSpectralThreadCache,
    ReducedTransformWorkspace,
    SlidingWindowBuffer,
    allocate_reduced_merge_workspace,
    allocate_reduced_qv_workspace,
    allocate_reduced_transform_workspace,
    allocate_sliding_window_buffer,
    apply_dry_basis_reduced!,
    apply_dry_basis_reduced_samples!,
    balance_window!,
    compute_reduced_dp_and_mass!,
    compute_reduced_horizontal_fluxes!,
    fill_buffer_slot!,
    init_rg_positivity_accumulator,
    load_rg_qv!,
    merge_field_2d!,
    merge_reduced_window!,
    nsamples,
    pin_convert_merge_window!,
    rescale_reduced_fluxes_to_ps!,
    rescale_reduced_samples_to_ps!,
    sample_reduced_horizontal_fluxes!,
    spectral_to_reduced_boundary!,
    spectral_to_reduced_scalar!,
    spectral_to_ring!,
    summarize_rg_positivity_status,
    synthesize_and_merge_window!,
    synthesize_next_day_hour0!,
    update_rg_positivity_accumulator,
    verify_boundary_stub_flux_rg,
    verify_rg_window_contract!,
    verify_substep_positivity_rg!
include("Targets/CubedSphere/CubedSphere.jl")
import .CubedSphereTarget: CSSpectralUnifiedDriverContext,
    CSWorst,
    CubedSphereContract,
    CubedSpherePreprocessWorkspace,
    CubedSphereSpectralWindowWorkspace,
    allocate_cs_preprocess_workspace,
    carried_tm5_convection_record,
    coarsen_nested_cs_transport_binary,
    convert_cs_mass_target_to_delta!,
    copy_cs_panels!,
    cs_spectral_contract_diag!,
    fill_cs_window_mass_tendency!,
    init_cs_positivity_accumulator,
    rebuild_spectral_window_fluxes!,
    regrid_2d_to_cs_panels!,
    regrid_3d_to_cs_panels!,
    regrid_ll_binary_to_cs,
    regrid_transport_binary,
    summarize_cs_positivity_status,
    update_cs_positivity_accumulator,
    verify_cs_window_contract!,
    verify_write_replay_cs!

include("Sources/GEOS/GEOS.jl")
import .GEOS: uses_omega, validate_omega_regularization, set_omega_level_parallel!, GEOSDayHandles,
    GEOSFPNativeDayHandles,
    GEOSFPSettings,
    GEOSITSettings,
    GEOSNativeReader,
    GEOSSettings,
    OmegaRegularization,
    close_geos_day!,
    detect_level_orientation,
    endpoint_dry_mass!,
    geos_collection_path,
    geosfp_native_hourly_ctm_path,
    open_geos_day

include("Sources/ERA5/ERA5.jl")
import .ERA5: ERA5C180RawConvectionFields,
    ERA5C180RegridFields,
    ERA5C180RegridWorkspace,
    ERA5C180TM5ConvectionFields,
    ERA5GRIBDayHandles,
    ERA5GRIBSettings,
    ERA5HumidityRemap,
    ERA5N320ConvectionFields,
    ERA5N320DryMassFields,
    ERA5N320Settings,
    ERA5N320SpectralWorkspace,
    ERA5N320ToC180Pipeline,
    ERA5N320WindowFields,
    ERA5ReducedGaussianGeometry,
    allocate_era5_c180_raw_convection_fields,
    allocate_era5_c180_regrid_fields,
    allocate_era5_c180_regrid_workspace,
    allocate_era5_n320_convection_fields,
    allocate_era5_n320_dry_mass_fields,
    allocate_era5_n320_spectral_workspace,
    allocate_era5_n320_to_c180_pipeline,
    allocate_era5_n320_window_fields,
    close_era5_day!,
    derive_c180_tm5_convection!,
    derive_n320_dry_mass!,
    discover_era5_n320_source_grid,
    discover_era5_spectral_truncation,
    era5_convection_hour_address,
    era5_grib_path,
    n320_cell_areas,
    open_era5_day,
    open_era5_humidity_remap,
    process_era5_n320_to_cs_day,
    process_era5_n320_window!,
    read_era5_n320_convection_window!,
    read_era5_n320_humidity!,
    read_era5_n320_window_fields!,
    regrid_n320_raw_convection_to_c180!,
    regrid_n320_to_c180!,
    remap_era5_humidity!

include("Sources/MERRA2/MERRA2.jl")
import .MERRA2: GEOSChemArchive,
    MERRA2Archive,
    MERRA2DayHandles,
    MERRA2LevelOrder,
    MERRA2Settings,
    MERRA2ToC180Pipeline,
    NASAArchive,
    SurfaceFirst,
    TopDown,
    allocate_merra2_to_c180_pipeline,
    allocate_merra2_window_physics,
    close_merra2_day!,
    convective_cloud_base!,
    detect_merra2_level_order,
    merra2_archive,
    merra2_path,
    merra2_stream_code,
    merra2_window_physics!,
    open_merra2_day,
    process_merra2_to_cs_day,
    process_merra2_window!,
    read_merra2_next_day_endpoint,
    read_merra2_physics_window,
    read_merra2_window_fields,
    validate_merra2_settings

import .Common: ALL_FLUX_KEYS, FLUX_CONSTRUCTION_KEYS, build_met_settings, column_balance_weights_option, flux_construction_kwargs, load_met_settings, native_output_filename, reject_unsupported_flux_keys, settings_constructor, supported_flux_keys
import .CubedSphereTarget: cs_total_air_mass, derive_c180_dry_mass!, fill_cs_mass_delta_payload!, pin_cs_global_air_mass!, pin_cs_global_air_mass_from_sensitivity!, ps_from_air_mass!
import .PhysicsInputs: TM5ConvectionColumnScratch, allocate_tm5_convection_column_scratch, moist_to_dry_cmfmc!, moist_to_dry_dtrain!
include("entrypoint.jl")

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
export endpoint_dry_mass!

# ERA5 native-GRIB reader: settings, day handles, and the
# per-window spectral-synthesis surface for the N320 source grid.
export AbstractERA5GRIBSettings, ERA5GRIBSettings, ERA5N320Settings
export ERA5ReducedGaussianGeometry, read_era5_reduced_gaussian_geometry,
       read_era5_reduced_gaussian_mesh
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
