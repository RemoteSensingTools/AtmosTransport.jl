"""Observation sources, retrieval columns, placement, sampling and NetCDF sinks."""
module Observations

include("Retrievals/Retrievals.jl")
using .Retrievals
include("Sources/Sources.jl")
using .Sources
include("Sampling/Sampling.jl")
using .Sampling

# Shared interfaces used by inversion, terrain readers and runner glue.

export AbstractObservationSource, OCO2LiteSource, ObsPackSource, TableSource,
       AbstractObservationMode, SoundingMode, SiteMode, AbstractSiteGrouping, SiteCodeGrouping,
       LocationGrouping, AbstractTableFormat, AutoTableFormat, CSVTableFormat, TOMLTableFormat,
       NetCDFTableFormat, AbstractQualityFilter, QualityFlagFilter, NoQualityFilter,
       AbstractSiteSchedule, EveryWindow, TimeRange, TimeList, AbstractObservationTimeInterpolation,
       LinearWindowInterpolation, NearestWindowSampling, AbstractObservationOutput,
       NoObservationOutput, ObservationOutputSpec, observation_output_spec, SoundingRequest,
       SiteRequest, ObservationSet, AbstractModelSurfaceHeight, NoModelSurfaceHeight,
       ModelSurfaceHeight, model_surface_height, AbstractPointElevation, NoPointElevation,
       point_elevation, ObservationTerrain, AbstractUnplacedPolicy, ErrorOnUnplaced, SkipUnplaced,
       AbstractAltitudePlacement, AboveGroundPlacement, AbsoluteAltitudeAbove,
       VerticalPlacementSpec, AbstractPlacementCoordinate, HeightAboveGroundCoordinate,
       PressureCoordinate, AbstractPressureBasis, TotalFromHumidity, DryInterfaces,
       VerticalPlacement, VERTICAL_PLACEMENT_CODES, resolve_placement, placement_layer,
       PlacementColumn, column_layer!, sample_column_layer!, needs_column_humidity,
       AbstractMissingHeightRule, LowestLayerWithoutHeight, UnplacedWithoutHeight,
       missing_height_rule, AbstractRetrievalMapping, LayerRebin, HatLevelProjection,
       tm5_layer_column, RetrievalColumn, retrieval_layer_count, stretch_model_interfaces!,
       retrieval_column_weights!, retrieval_prior_offset, retrieval_profile!,
       retrieval_column_value, retrieval_model_error_variance, read_oco2_retrieval_columns,
       AbstractAveragingKernelOutput, NoAveragingKernels, AveragingKernelColumns,
       read_observation_requests, build_observation_set, AbstractObservationSampler,
       NoObservationSampler, build_observation_sampler, samples_observations,
       observation_needs_humidity, observe_window_boundary!, begin_observation_day!,
       finish_observations!, AbstractCellLocator, AbstractLayerTemperature,
       AbstractObservationStream, AnyRetrievalColumn, BELOW_GROUND_TOLERANCE_M, CellLocation,
       ConstantLayerTemperature, CubedSphereCellLocator, DEFAULT_VERTICAL_COORDINATES, GatherPlan,
       HEIGHT_METHOD_CODES, HEIGHT_METHOD_NOT_SAMPLED, INTERP_FLAG_CODES, LOWEST_GROUND_ELEVATION_M,
       LatLonCellLocator, OCO2_LEVEL_FORMS, ObservationGatherBuffers, ObservationSampler,
       PendingSoundings, ProfileLayerTemperature, QualityFlagValues, RETRIEVAL_WEIGHT_SUM_TOLERANCE,
       ReducedGaussianCellLocator, SiteNetCDFStream, SiteRecord, SoundingBatch,
       SoundingNetCDFStream, SurfaceLapseTemperature, VERTICAL_PLACEMENT_KEYS,
       VirtualLayerTemperature, append_site_record!, append_soundings!, averaging_kernel_tracers,
       below_ground, cell_locator, check_observation_tracer_names, check_pressure_basis,
       check_surface_height_layout, check_terrain, column_mean_vmr, count_unplaced!,
       expand_observation_paths, gather_columns!, gather_field!, handle_unplaced,
       handle_unplaced_nonzero, height_method_code, height_method_codes_description,
       intake_layer_index, interface_pressures!, interp_flag, isvalid_lonlat, layer_heights_agl!,
       layer_temperature, locate, mixing_ratio_profile!, mode_label, mole_fraction_scale,
       mole_fraction_unit_scale, nc_variable, nslots, observation_output_path,
       observation_source_from_cfg, oco2_level_form, oco2_level_form_from_label, parse_iso_utc,
       parse_retrieval_mapping, parse_vertical_placement, placement_heights!, placement_pressures!,
       pressure_basis_label, pressure_layer_index, quality_filter_keys, raw_observation_block,
       record_height_above_ground, retrieval_mapping_label, retrieval_pressure_count,
       schedule_label, site_active, source_keys, source_kind, source_mode, source_path_template,
       terrain_source, time_interpolation_label, total_interface_pressures!, unplaced_example,
       unplaced_policy_label, valid_elevation, valid_intake_height, valid_pressure,
       vertical_coordinate_label, vertical_placement_codes_description, warn_below_ground,
       write_summary_attributes!, pressure_unit_scale, ObservationCounters,
       write_observation_variable!

end # module Observations
