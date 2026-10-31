"""ERA5 meteorological input readers and transport-binary source workflows."""
module ERA5

import Printf: @sprintf

import ..ReducedGaussianTarget: allocate_reduced_spectral_thread_cache

import ...State

import ...AtomicFiles: abandon_atomic_staging

import ...ConfigChecks: config_bool as _config_bool

import ...Grids: HybridSigmaPressure,
    ReducedGaussianMesh,
    cell_areas,
    cs_geometry_tags,
    cs_total_area,
    n_levels,
    ncells,
    nrings,
    ring_longitudes,
    unpack_flat_to_panels_2d!,
    unpack_flat_to_panels_3d!

import ...MetDrivers: replay_tolerance,
    surface_pressure_basis

import ...MetDrivers.BinaryFormat: grid_type,
    open_streaming_cs_transport_binary,
    set_streaming_steps_per_window_schedule!

import ...Parameters: ADAPTIVE_SUBSTEP_MAX_REFINEMENTS,
    DEFAULT_POISSON_BALANCE_MAX_ITER,
    DEFAULT_POISSON_BALANCE_TOL,
    DEFAULT_POISSON_PROJECT_EVERY,
    DEFAULT_POSITIVITY_CFL_LIMIT,
    EARTH_RADIUS,
    ERA5_MET_INTERVAL_SECONDS,
    ERA5_N320_SUBSTEP_CFL_TARGET,
    ERA5_NATIVE_LEVEL_COUNT,
    HORIZONTAL_BALANCE_TAG,
    IFS_EARTH_RADIUS,
    STANDARD_GRAVITY

import ..Common: AbstractERA5GRIBSettings,
    CubedSphereTargetGeometry,
    FLUX_CONSTRUCTION_KEYS,
    MAX_SPECIFIC_HUMIDITY,
    ReducedGaussianTargetGeometry,
    build_met_settings,
    close_day!,
    column_weights,
    flux_construction_kwargs,
    native_output_prefix,
    nlat,
    nlon,
    open_day,
    open_preprocessing_staging,
    read_era5_reduced_gaussian_geometry,
    read_era5_reduced_gaussian_mesh,
    reject_removed_horizontal_balance_keyword,
    require_configured_terminal_day,
    settings_constructor,
    source_grid,
    supported_flux_keys,
    supports_day_threading,
    windows_per_day,
    with_replay_record

import ..CubedSphereTarget: allocate_cs_preprocess_workspace,
    derive_c180_dry_mass!,
    fill_cs_mass_delta_payload!,
    fill_cs_window_mass_tendency!,
    init_cs_positivity_accumulator,
    pin_cs_global_air_mass!,
    pin_cs_global_air_mass_from_sensitivity!,
    ps_from_air_mass!,
    regrid_2d_to_cs_panels!,
    summarize_cs_positivity_status,
    update_cs_positivity_accumulator,
    verify_cs_window_contract!

import ..MassClosure: AbstractWindRegrid,
    DryMassFluxThickness,
    LineIntegralFaceFluxes,
    MoistFluxThickness,
    PanelAverageFluxes,
    ScalarWindRegrid,
    VectorFaceFluxes,
    balance_cs_column_mass_fluxes!,
    build_wind_regrid,
    diagnose_balanced_cs_cm!,
    face_flux_method,
    face_fluxes!,
    fill_cs_layer_thickness!,
    flux_header_face_interpolation,
    flux_header_wind_regrid,
    flux_thickness_method,
    has_source_line_integrals,
    line_integral_face_fluxes!,
    prepare_cell_winds!,
    regrid_winds!,
    validate_flux_construction,
    validate_flux_thickness,
    validate_flux_time_sampling

import ..PhysicsInputs: BLDiffColumnScratch,
    BLDiffConstants,
    TM5CleanupStats,
    TM5ConvectionColumnScratch,
    allocate_tm5_convection_column_scratch,
    close_era5_surface_reader,
    dz_hydrostatic_virtual!,
    ec2tm_from_rates!,
    ecmwf_convection_to_basis!,
    era5_surface_source_mesh,
    era5_tm5_convection_header,
    load_era5_surface_window,
    open_era5_surface_reader,
    reset_bldiff_diag!,
    tm5_bldiff_dkg_column!

import ..ReducedGaussianTarget: ReducedSpectralThreadCache,
    spectral_to_reduced_scalar!

import ..Spectral: ReducedSpectralSynthesis,
    mass_fix_target_ps_dry_record,
    read_spectral_coeffs!,
    synthesize_reduced!,
    vod2uv!

import ..TransportBinary: AbstractColumnClosureRecord,
    CubedSphereBinaryWriter,
    FidelitySampleMap,
    FluxFidelityEvidenceBundle,
    HybridLayerThickness,
    LineIntegralFidelityStencil,
    NoColumnClosureRecord,
    NoSourceOmegaEvidence,
    ReadyWindow,
    SourceHasNoOmega,
    SubstepSchedulePolicy,
    capture_cs_wind_flux_samples,
    column_closure_record,
    combine_captured_flux_samples,
    flux_fidelity_method,
    merge_captured_flux_samples,
    next_substeps,
    process_day,
    promote_streaming_binary!,
    quarantine_streaming_binary!,
    verify_substep_positivity_cs!,
    wind_derived_evidence,
    write_window!,
    writer_staging_path

import ..Vertical: sum_merged_layers!,
    AbstractFieldKind,
    ConvectionTendencyField,
    IdentityVertical,
    MassField,
    MassFluxField,
    VerticalPlan,
    apply_vertical!,
    load_hybrid_coefficients,
    plan_vertical

import ...Quantities: IntensiveCellField

import ...Regridding: apply_regridder!,
    build_regridder

import ...State: AbstractMassBasis,
    DryBasis,
    DrySurfacePressure

import Dates: Date,
    DateTime,
    Dates,
    Day,
    Hour,
    hour

import GRIB: GRIB,
    GribFile,
    Index,
    destroy,
    select!

import NCDatasets: NCDataset,
    name,
    path

export ERA5C180RawConvectionFields,
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

include("era5_grib_geometry.jl")
include("era5.jl")
include("configuration.jl")
include("era5_n320_window.jl")
include("era5_n320_mass_convection.jl")
include("era5_n320_to_cs.jl")
include("era5_n320_regrid.jl")

end # module ERA5
