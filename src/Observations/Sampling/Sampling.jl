"""Sampling for observation columns and sampling."""
module Sampling

using ...Grids: cell_centers

using ...Grids: panel_convention_tag

using ...State: basis_symbol

using Dates
using JSON3
using NCDatasets
using KernelAbstractions: @kernel, @index, get_backend, synchronize, CPU as KA_CPU

using ...ConfigChecks: config_bool as _config_bool, check_known_keys as _check_known_keys

using ...Parameters: DEFAULT_OBSERVATION_LAYER_TEMPERATURE_KELVIN, R_DRY_AIR, STANDARD_LAPSE_RATE,
                     STANDARD_TROPOPAUSE_TEMPERATURE, VIRTUAL_TEMPERATURE_FACTOR
import ...expand_data_path
using ...Grids: AtmosGrid, LatLonMesh, ReducedGaussianMesh, CubedSphereMesh,
               GnomonicPanelConvention, GEOSNativePanelConvention,
               nx, ny, nrings, ring_longitudes, cell_index, ncells,
               cell_area, panel_cell_center_lonlat, panel_cell_corner_lonlat,
               cs_definition, coordinate_law, center_law, longitude_offset_deg,
               cs_definition_tag, coordinate_law_tag, center_law_tag,
               lonlat_to_panel_xy, gravity
using ...State: AbstractMassBasis, DryBasis, MoistBasis, mass_basis, tracer_names, get_tracer,
               CellState, CubedSphereState, tracer_index


using ..Retrievals
using ..Sources
using ..Sources: _parse_choice
using ...Output: define_product_attributes!, _parse_tracer_names, _output_partition, _substitute_day_template,
                 _insert_suffix_before_extension, _has_day_token, _ensure_parent_dir,
                 _create_netcdf_dataset, _tracer_units, _payload_fill_value, _grid_type_string,
                 _NETCDF_IO_LOCK, with_netcdf_lock, define_provenance_attributes!,
                 mixing_ratio_field, AbstractOutputPartition, SingleOutputFile, DailyOutputFiles

export AboveGroundPlacement, AbsoluteAltitudeAbove, AbstractAltitudePlacement, AbstractCellLocator,
       AbstractLayerTemperature, AbstractMissingHeightRule, AbstractModelSurfaceHeight,
       AbstractObservationOutput, AbstractObservationSampler, AbstractObservationStream,
       AbstractObservationTimeInterpolation, AbstractPlacementCoordinate, AbstractPointElevation,
       AbstractPressureBasis, AbstractUnplacedPolicy, BELOW_GROUND_TOLERANCE_M, CellLocation,
       ConstantLayerTemperature, CubedSphereCellLocator, DEFAULT_VERTICAL_COORDINATES,
       DryInterfaces, ErrorOnUnplaced, GatherPlan, HEIGHT_METHOD_CODES, HEIGHT_METHOD_NOT_SAMPLED,
       HeightAboveGroundCoordinate, INTERP_FLAG_CODES, LatLonCellLocator, LinearWindowInterpolation,
       LowestLayerWithoutHeight, ModelSurfaceHeight, NearestWindowSampling, NoModelSurfaceHeight,
       NoObservationOutput, NoObservationSampler, NoPointElevation, ObservationGatherBuffers,
       ObservationOutputSpec, ObservationSampler, ObservationTerrain, PendingSoundings,
       PlacementColumn, PressureCoordinate, ProfileLayerTemperature, ReducedGaussianCellLocator,
       SiteNetCDFStream, SiteRecord, SkipUnplaced, SoundingBatch, SoundingNetCDFStream,
       SurfaceLapseTemperature, TotalFromHumidity, UnplacedWithoutHeight, VERTICAL_PLACEMENT_CODES,
       VERTICAL_PLACEMENT_KEYS, VerticalPlacement, VerticalPlacementSpec, VirtualLayerTemperature,
       append_site_record!, append_soundings!, begin_observation_day!, below_ground,
       build_observation_sampler, cell_locator, check_observation_tracer_names,
       check_pressure_basis, check_surface_height_layout, check_terrain, column_layer!,
       column_mean_vmr, count_unplaced!, finish_observations!, gather_columns!, gather_field!,
       handle_unplaced, handle_unplaced_nonzero, height_method_code,
       height_method_codes_description, intake_layer_index, interface_pressures!, interp_flag,
       layer_heights_agl!, layer_temperature, locate, missing_height_rule, mixing_ratio_profile!,
       model_surface_height, needs_column_humidity, nslots, observation_needs_humidity,
       observation_output_path, observation_output_spec, observe_window_boundary!,
       parse_vertical_placement, placement_heights!, placement_layer, placement_pressures!,
       point_elevation, pressure_basis_label, pressure_layer_index, resolve_placement,
       sample_column_layer!, samples_observations, terrain_source, time_interpolation_label,
       total_interface_pressures!, unplaced_example, unplaced_policy_label,
       vertical_coordinate_label, vertical_placement_codes_description, warn_below_ground,
       write_summary_attributes!

include("cell_locator.jl")
include("observation_placement.jl")
include("observation_output_spec.jl")
include("observation_gather.jl")
include("observation_netcdf.jl")
include("observation_sampler.jl")

export ObservationCounters

export write_observation_variable!

end # module Sampling
