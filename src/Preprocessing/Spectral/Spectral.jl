module Spectral

export era5_thermo_path

using Dates
using Logging
using Printf
using FFTW
using GRIB
using SHA: sha1
using JLD2
using NCDatasets
using ...AtomicFiles: atomic_publish
using ...Parameters: IFS_EARTH_RADIUS, STANDARD_GRAVITY, GLOBAL_MEAN_DRY_SURFACE_PRESSURE,
    GLOBAL_MEAN_SPECIFIC_HUMIDITY, ERA5_NATIVE_LEVEL_COUNT, ERA5_MET_INTERVAL_SECONDS
using ...Grids: HybridSigmaPressure, ncells, nrings, nlevels, n_levels
using ...GriddedIO: find_coordinate, LongitudeAxis, LatitudeAxis
using ..Core: check_column_weight_levels, spread_column_correction!, AbstractColumnWeights,
    LatLonTargetGeometry, MAX_SPECIFIC_HUMIDITY, MassWeightedColumn, ReducedGaussianTargetGeometry,
    nlat, nlon, target_spectral_truncation
using LinearAlgebra: mul!
using ..Vertical: sum_merged_layers!

export LLPoissonWorkspace, ReducedSpectralSynthesis, SPECTRAL_DAY_CACHE_VERSION,
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

include("spectral_io.jl")
include("spectral_synthesis.jl")
include("mass_support.jl")
include("reduced_spectral_synthesis.jl")
include("discovery.jl")

end
