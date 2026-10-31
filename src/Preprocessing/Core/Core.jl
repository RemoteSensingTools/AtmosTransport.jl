module Core

import TOML

using Dates
using Logging
using Printf
import ...Grids: panel_convention_tag
using ...ConfigChecks: config_bool as _config_bool, check_known_keys, parse_float_type
using ...AtomicFiles: open_atomic_staging
using ...Parameters: EARTH_RADIUS, GLOBAL_MEAN_DRY_SURFACE_PRESSURE, GLOBAL_MEAN_SPECIFIC_HUMIDITY,
    DEFAULT_POISSON_BALANCE_TOL, DEFAULT_POISSON_PROJECT_EVERY, HORIZONTAL_BALANCE_TAG,
    ERA5_NATIVE_LEVEL_COUNT
using ...Grids: cell_area_matrix, CSGlobalFaceTable, build_cs_global_face_table,
    parse_panel_convention, parse_cs_definition
using ...Grids: LatLonMesh, ReducedGaussianMesh, CubedSphereMesh, nrings, ring_longitudes,
    panel_convention, panel_cell_local_tangent_basis, cs_definition, coordinate_law, center_law,
    longitude_offset_deg, cs_definition_tag, coordinate_law_tag, center_law_tag
using ...Regridding: default_regrid_cache_dir
import ...expand_data_path
import ...MetDrivers: has_surface, has_vdiff_fields, has_pbl_eflux, has_cmfmc_cloud_base
using FastGaussQuadrature: gausslegendre

export AbstractChainPolicy, AbstractColumnWeights, AbstractERA5GRIBSettings, AbstractGEOSSettings,
    AbstractMetReader, AbstractMetSettings, AbstractTargetGeometry, COLUMN_WEIGHT_NAMES,
    ChainedMass, CubedSphereTargetGeometry, ERA5SpectralSettings, FlushingLogger, LatLonTargetGeometry, MAX_SPECIFIC_HUMIDITY, MassWeightedColumn, NoChain, PACKAGE_ROOT,
    RawWindow, ReducedGaussianTargetGeometry, source_grid, allocate_raw_window,
    build_target_geometry, check_column_weight_levels, check_spectral_column_weights,
    checked_mass_fix_config, close_day!, close_reader!, column_weights, day_threading_enabled,
    ensure_supported_target, grid_kind, has_cmfmc_cloud_base, has_convection, has_pbl_eflux,
    has_surface, has_vdiff_fields, log_preprocessor_configuration, nlat, nlon, open_day,
    open_preprocessing_staging, open_reader, package_path, panel_convention_tag,
    parse_column_balance_weights, parse_flux_construction, preprocessor_pair_supported,
    read_era5_reduced_gaussian_geometry, read_era5_reduced_gaussian_mesh, read_window!,
    reject_removed_balance_mode, reject_removed_horizontal_balance_keyword,
    reject_removed_tm5_convection_route, require_configured_terminal_day, resolve_balance_settings,
    resolve_column_balance_weights, resolve_mass_basis, resolve_mass_fix_settings,
    resolve_output_float_type, resolve_preprocessing_cache_settings, resolve_runtime_settings,
    resolve_spectral_flux_settings, resolve_surface_settings, spectral_column_weights,
    spread_column_correction!, supports_day_threading, supports_initial_endpoint_mass_pin,
    supports_spectral_massflux_preprocessing, target_balance_geometry, target_definition_tag,
    target_header_metadata, target_spectral_truncation, target_summary,
    validate_column_balance_weights, windows_per_day, with_column_weights_record, with_replay_record

include("binary_pipeline.jl")
include("met_sources.jl")
include("logging.jl")
include("target_geometry.jl")
include("configuration.jl")
include("met_readers.jl")

export check_native_closure_config, native_output_prefix, ALL_FLUX_KEYS, FLUX_CONSTRUCTION_KEYS, build_met_settings, column_balance_weights_option, flux_construction_kwargs, load_met_settings, native_output_filename, reject_unsupported_flux_keys, settings_constructor, supported_flux_keys
include("source_factory.jl")

end # module Core
