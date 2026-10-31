"""MERRA2 meteorological input readers and transport-binary source workflows."""
module MERRA2

import Printf: @sprintf

import AtmosTransport: State,
    expand_data_path

import AtmosTransport.AtomicFiles: abandon_atomic_staging

import AtmosTransport.ConfigChecks: config_bool as _config_bool

import AtmosTransport.Grids: LatLonMesh,
    cell_areas,
    cs_geometry_tags,
    ncells,
    unpack_flat_to_panels_3d!

import AtmosTransport.MetDrivers: replay_tolerance,
    surface_pressure_basis

import AtmosTransport.MetDrivers.BinaryFormat: open_streaming_cs_transport_binary,
    set_streaming_steps_per_window_schedule!

import AtmosTransport.MetDrivers.BinaryReaders: has_cmfmc_cloud_base,
    has_pbl_eflux,
    has_surface,
    has_vdiff_fields

import AtmosTransport.Parameters: ADAPTIVE_SUBSTEP_MAX_REFINEMENTS,
    DEFAULT_POISSON_BALANCE_MAX_ITER,
    DEFAULT_POISSON_BALANCE_TOL,
    DEFAULT_POISSON_PROJECT_EVERY,
    DEFAULT_POSITIVITY_CFL_LIMIT,
    DEFAULT_SUBSTEP_CFL_TARGET,
    EARTH_RADIUS,
    GEOS_3HOURLY_INTERVAL_SECONDS,
    HORIZONTAL_BALANCE_TAG,
    STANDARD_GRAVITY

import AtmosTransport.Preprocessing.Core: AbstractMetSettings,
    CubedSphereTargetGeometry,
    FLUX_CONSTRUCTION_KEYS,
    build_met_settings,
    close_day!,
    column_weights,
    flux_construction_kwargs,
    has_convection,
    native_output_prefix,
    open_day,
    open_preprocessing_staging,
    preprocessor_pair_supported,
    reject_removed_horizontal_balance_keyword,
    require_configured_terminal_day,
    settings_constructor,
    source_grid,
    supported_flux_keys,
    windows_per_day,
    with_replay_record

import AtmosTransport.Preprocessing.CubedSphere: CSWorst,
    CubedSpherePreprocessWorkspace,
    allocate_cs_preprocess_workspace,
    derive_c180_dry_mass!,
    fill_cs_mass_delta_payload!,
    fill_cs_window_mass_tendency!,
    init_cs_positivity_accumulator,
    pin_cs_global_air_mass!,
    ps_from_air_mass!,
    regrid_2d_to_cs_panels!,
    regrid_3d_to_cs_panels!,
    summarize_cs_positivity_status,
    update_cs_positivity_accumulator,
    verify_cs_window_contract!

import AtmosTransport.Preprocessing.MassClosure: AbstractWindRegrid,
    DryMassFluxThickness,
    MoistFluxThickness,
    balance_cs_column_mass_fluxes!,
    build_wind_regrid,
    diagnose_balanced_cs_cm!,
    face_flux_method,
    face_fluxes!,
    fill_cs_layer_thickness!,
    flux_thickness_method,
    prepare_cell_winds!,
    regrid_winds!,
    validate_flux_construction

import AtmosTransport.Preprocessing.PhysicsInputs: moist_to_dry_cmfmc!,
    moist_to_dry_dtrain!

import AtmosTransport.Preprocessing.Spectral: mass_fix_target_ps_dry_record

import AtmosTransport.Preprocessing.TransportBinary: AbstractColumnClosureRecord,
    CubedSphereBinaryWriter,
    FidelitySampleMap,
    FluxFidelityEvidenceBundle,
    MERRA2_WIND_OMEGA_BASELINE,
    NoColumnClosureRecord,
    NoSourceOmegaEvidence,
    OmegaFidelityCells,
    ReadyWindow,
    SourceNotOpened,
    SubstepSchedulePolicy,
    capture_cs_wind_flux_samples,
    column_closure_record,
    flux_fidelity_method,
    next_substeps,
    process_day,
    promote_streaming_binary!,
    quarantine_streaming_binary!,
    source_omega_evidence_cs,
    verify_substep_positivity_cs!,
    wind_derived_evidence,
    write_window!,
    writer_staging_path

import AtmosTransport.Preprocessing.Vertical: load_hybrid_coefficients

import AtmosTransport.Quantities: IntensiveCellField

import AtmosTransport.Regridding: apply_regridder!,
    build_regridder

import AtmosTransport.State: DrySurfacePressure

import Dates: Date,
    Dates,
    Day,
    day,
    year

import NCDatasets: NCDataset,
    name,
    path

export GEOSChemArchive,
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

include("merra2.jl")
include("configuration.jl")
include("merra2_latlon_regrid.jl")

end # module MERRA2
