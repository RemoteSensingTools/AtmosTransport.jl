"""Reduced-Gaussian spectral staging, window contracts and daily preprocessing."""
module ReducedGaussianTarget

import ..Common: GPUSynthesis, AutoSynthesisPrecision, PointSampledSpectralFluxes, TM5SpectralFluxes, spectral_construction, validate_spectral_construction, balance_window_layers!
import ..Spectral: spectral_synthesis_metadata, SpectralIntegralWorkspace, construct_tm5_endpoint, TM5EndpointPlan, with_spectral_hours
import ..LatLonTarget: tm5_horizontal_convergence, balance_tm5_layers!, dry_target_fluxes!
import ..MassClosure: DirectRingPoisson, balance_compressed_level!
import ..Common: validate_spectral_vertical
import ..Spectral: read_day_spectral_streaming
import ..TransportBinary: tm5_construction_metadata, settings_include_surface
import ...MetDrivers.BinaryFormat: TM5SpectralFluxesHeader, write_streaming_window!, close_streaming_transport_binary!, abandon_streaming_transport_binary!

import ..Spectral: era5_thermo_path

import ...Grids, ...State
import ...Architectures: CPU
import ...Grids: AtmosGrid,
    cell_areas,
    face_connectivity,
    ncells,
    nfaces,
    nrings
import ...MetDrivers: flux_application_seconds, ReplayScratch,
    faceindexed_replay_layout,
    flux_kind,
    recompute_cm_from_dm_target!,
    replay_tolerance,
    surface_pressure_basis,
    verify_window_continuity_rg
import ...MetDrivers.BinaryFormat: FLUX_FIDELITY_SAMPLE_SEED,
    PointSampledSpectralFluxesHeader, NoFluxConstructionHeader,
    canonical_window_constant_contract,
    open_streaming_transport_binary,
    poisson_balance_target,
    set_streaming_steps_per_window_schedule!
import ...MetDrivers.BinaryReaders: delta_semantics
import ...Parameters: DEFAULT_POISSON_BALANCE_MAX_ITER,
    DEFAULT_POISSON_BALANCE_TOL,
    DEFAULT_POSITIVITY_CFL_LIMIT,
    HORIZONTAL_BALANCE_TAG,
    STANDARD_GRAVITY
import ..Common: validate_tm5_pbl_target, AbstractColumnWeights,
    ERA5SpectralSettings,
    MAX_SPECIFIC_HUMIDITY,
    MassWeightedColumn,
    ReducedGaussianTargetGeometry,
    check_spectral_column_weights,
    nlon,
    open_preprocessing_staging,
    spectral_column_weights,
    with_column_weights_record,
    with_replay_record
import ..MassClosure: ColumnResidualWeight,
    CompressedLaplacian,
    balance_compressed_column_fluxes!,
    build_compressed_laplacian
import ..Spectral: compute_legendre_column!,
    era5_spectral_paths,
    fill_ring_spectrum!,
    maybe_preload_qv_day,
    next_day_hour0,
    pin_global_mean_ps!,
    pin_global_mean_ps_using_qv!,
    read_day_spectral,
    read_qv_from_thermo,
    spectral_to_native_fields!,
    vod2uv!
import ...State: TotalSurfacePressure
import Dates: Date,
    Dates,
    Day
import FFTW: FFTW
import GRIB: GRIB
import ..LatLonTarget: process_window!, TM5PBLInputCache, load_tm5_pbl_target_window, tm5_pbl_sample, log_mass_fix_configuration,
    should_log_window
import NCDatasets: NCDataset, dimnames
import ..TransportBinary: AbstractWindowContract,
    FluxFidelityEvidenceBundle,
    NoColumnClosureRecord,
    NoSourceOmegaEvidence,
    PreverifiedWindow,
    ReadyWindow,
    ReducedGaussianBinaryWriter,
    SourceHasNoOmega,
    SubstepSchedulePolicy,
    UnifiedPreprocessorDay,
    allocate_window_workspace,
    capture_faceindexed_flux_samples,
    drain_ready_windows!,
    driver_before_close_writer!,
    driver_drain_ready_windows!,
    driver_flush_final_windows!,
    driver_ingest_window!,
    driver_windows_per_day,
    exact_steps_per_window,
    faceindexed_wind_evidence,
    flush_final_windows!,
    ingest_window!,
    init_positivity_accumulator,
    initial_substeps,
    next_substeps,
    output_binary_path,
    process_day,
    rescale_substep_amounts!,
    run_unified_preprocessor_day!,
    scale_captured_flux_samples,
    set_contract_steps_schedule!,
    splitmix64,
    summarize_positivity_status,
    summarize_status!,
    update_accumulator!,
    update_positivity_accumulator,
    validate_cfl_limit,
    validate_replay_tol,
    verify_window!
import ..Vertical: sum_merged_layers!
using Printf: @sprintf
using FFTW

import ..TransportBinary: cached_or_build!, ranked_fidelity_ids

export allocate_reduced_spectral_thread_cache, RGSpectralUnifiedDriverContext,
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

include("reduced_gaussian_contracts.jl")
include("reduced_transport_helpers.jl")
include("reduced_window_buffer.jl")
include("reduced_spectral_day.jl")
include("tm5_spectral_day.jl")

end # module ReducedGaussianTarget
