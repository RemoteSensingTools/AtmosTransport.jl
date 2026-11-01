"""GEOS meteorological input readers and transport-binary source workflows."""
module GEOS
using ...ConfigChecks: env_flag

import Printf: @sprintf

import ..MassClosure: cs_face_divergence!,
    cs_level_scratches,
    cs_column_convergence

import ...expand_data_path

import ...ConfigChecks: config_bool as _config_bool

import ...Grids: CSGlobalFaceTable,
    CS_PANEL_COUNT,
    CubedSphereMesh,
    GEOSNativePanelConvention,
    HybridSigmaPressure,
    cell_areas,
    cs_geometry_tags,
    cs_panel_ij,
    cs_total_area,
    panel_cell_center_lonlat,
    panel_convention

import ...MetDrivers: flux_kind,
    replay_tolerance,
    surface_pressure_basis

import ...MetDrivers.BinaryFormat: close_streaming_transport_binary!,
    open_streaming_cs_transport_binary,
    set_streaming_steps_per_window_schedule!

import ...MetDrivers.BinaryReaders: has_pbl_eflux,
    has_surface,
    has_vdiff_fields

import ...Parameters: ADAPTIVE_SUBSTEP_MAX_REFINEMENTS,
    DEFAULT_GEOS_SMOOTHING_SWEEPS,
    DEFAULT_GEOS_SMOOTHING_WEIGHT,
    DEFAULT_POISSON_PROJECT_EVERY,
    DEFAULT_POSITIVITY_CFL_LIMIT,
    DEFAULT_SUBSTEP_CFL_TARGET,
    EARTH_RADIUS,
    GEOS_3HOURLY_INTERVAL_SECONDS,
    GEOS_MASS_FLUX_DT_SECONDS,
    GEOS_MET_INTERVAL_SECONDS,
    HORIZONTAL_BALANCE_TAG,
    STANDARD_GRAVITY,
    SURFACE_AIR_TEMPERATURE_RANGE,
    SURFACE_HEAT_FLUX_MAX_ABS

import ..Common: check_native_closure_config,
    AbstractChainPolicy,
    AbstractGEOSSettings,
    AbstractMetReader,
    ChainedMass,
    CubedSphereTargetGeometry,
    MAX_SPECIFIC_HUMIDITY,
    NoChain,
    RawWindow,
    allocate_raw_window,
    build_met_settings,
    close_day!,
    close_reader!,
    column_balance_weights_option,
    column_weights,
    has_convection,
    native_output_prefix,
    open_day,
    open_preprocessing_staging,
    open_reader,
    read_window!,
    settings_constructor,
    source_grid,
    supported_flux_keys,
    validate_column_balance_weights,
    windows_per_day,
    with_column_weights_record,
    with_replay_record

import ..CubedSphereTarget: CubedSphereContract,
    convert_cs_mass_target_to_delta!,
    cs_total_air_mass,
    fill_cs_window_mass_tendency!,
    pin_cs_global_air_mass_from_sensitivity!,
    ps_from_air_mass!

import ..MassClosure: CSPoissonScratch,
    apply_cs_flux_correction!,
    balance_cs_column_mass_fluxes!,
    compute_cs_cm_pressure_fixer!,
    diagnose_balanced_cs_cm!,
    geos_native_to_face_flux!,
    recheck_cs_column_closure,
    solve_cs_poisson_pcg!,
    sync_cs_mirrors!

import ..PhysicsInputs: surface_flux_to_upward_wm2,
    moist_to_dry_cmfmc!,
    moist_to_dry_dtrain!

import ..Spectral: mass_fix_target_ps_dry_record,
    next_day_hour0,
    vertical_mapping_method

import ..TransportBinary: AbstractColumnClosureRecord,
    AbstractWindowWorkspace,
    CubedSphereBinaryWriter,
    FidelitySampleMap,
    FluxFidelityEvidenceBundle,
    GEOS_NATIVE_OMEGA_BASELINE,
    NoColumnClosureRecord,
    NoSourceOmegaEvidence,
    OmegaFidelityCells,
    PreverifiedWindow,
    ReadyWindow,
    SourceNotOpened,
    SubstepSchedulePolicy,
    UnifiedPreprocessorDay,
    allocate_window_workspace,
    capture_native_flux_samples,
    column_closure_record,
    drain_ready_windows!,
    driver_after_write_window!,
    driver_before_close_writer!,
    driver_drain_ready_windows!,
    driver_flush_final_windows!,
    driver_ingest_window!,
    exact_steps_per_window,
    full_window_column_closure_record,
    ingest_window!,
    initial_substeps,
    native_flux_evidence,
    next_substeps,
    process_day,
    run_unified_preprocessor_day!,
    scale_captured_flux_samples,
    source_omega_evidence_cs,
    verify_substep_positivity_cs!,
    verify_window!

import ..Vertical: ConvectionInterfaceFlux,
    ConvectionTendencyField,
    IdentityVertical,
    IntensiveCenterField,
    MassField,
    MassFluxField,
    apply_vertical!,
    load_hybrid_coefficients,
    plan_vertical

import ...Regridding: coarsen_area_weighted!,
    coarsen_sum!,
    coarsen_xface_sum!,
    coarsen_yface_sum!

import ...State: DryBasis,
    DrySurfacePressure

import Dates: Date,
    Dates,
    Day,
    hour,
    minute,
    today

import NCDatasets: NCDataset,
    bounds,
    dimnames,
    name,
    path

import Statistics: mean

export uses_omega, validate_omega_regularization, GEOSDayHandles,
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
    open_geos_day,
    set_omega_level_parallel!

include("geos.jl")
include("geos_panels.jl")
include("geos_read_window.jl")
include("reader.jl")
include("configuration.jl")
include("geos_cs_mass_helpers.jl")
include("geos_cs_omega.jl")
include("geos_cs_resolution.jl")
include("geos_cs_window.jl")
include("cubed_sphere_geos.jl")

end # module GEOS
