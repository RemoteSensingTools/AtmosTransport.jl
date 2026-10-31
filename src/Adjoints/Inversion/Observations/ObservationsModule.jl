module Observations
using ....Operators.Advection: NoLimiter
using ....Operators.Advection: PPMScheme
using ...Footprint: _validate_step_sequences
using ....Grids: cs_definition
using ....Grids: cs_definition_tag
using ....Grids: panel_convention

using ....Grids: CubedSphereMesh
using ....Parameters: DEFAULT_OBSERVATION_LAYER_TEMPERATURE_KELVIN, STANDARD_GRAVITY
using ....State: AbstractMassBasis, DryBasis
using ....Observations: AbstractRetrievalMapping, raw_observation_block, nc_variable, oco2_level_form, AbstractLayerTemperature, CellLocation, ConstantLayerTemperature, HatLevelProjection, LayerRebin, ObservationCounters, ObservationTerrain, PlacementColumn, ProfileLayerTemperature, RetrievalColumn, SiteRequest, SkipUnplaced, SoundingRequest, UnplacedWithoutHeight, VERTICAL_PLACEMENT_CODES, VerticalPlacement, VerticalPlacementSpec, cell_locator, check_pressure_basis, check_surface_height_layout, check_terrain, count_unplaced!, handle_unplaced, locate, mole_fraction_scale, mole_fraction_unit_scale, needs_column_humidity, parse_retrieval_mapping, placement_pressures!, pressure_unit_scale, read_oco2_retrieval_columns, resolve_placement, retrieval_column_weights!, retrieval_layer_count, retrieval_mapping_label, retrieval_model_error_variance, retrieval_pressure_count, retrieval_prior_offset, retrieval_profile!, sample_column_layer!, unplaced_example, vertical_placement_codes_description, warn_below_ground, write_observation_variable!
using ...Adjoints: AbstractCSFootprintObjective, CSCellAverageObjective, CSColumnMeanObjective, CSLayerMeanObjective, CSLinearFunctional, CSPlacedPointObjective, CSRetrievalColumnObjective, _linear_functional, _objective_mass_floor, _validate_objective, evaluate_objective
using ...Footprint: _cs_replay_for_call, _replay_m, _replay_nsteps, _replay_rm, _run_forward!
import Dates, NCDatasets

export CELL_AVERAGE_PLACEMENT_CODE, AbstractFlaggedRecordPolicy, DropFlaggedRecords,
    CosampleFlaggedRecords, AbstractCellAveraging, NoCellAveraging, CellAverage,
    inflate_cell_model_error, read_oco2_mip_observations, convert_oco2_mip_observations,
    CSTimeSampling, CSObservation, CSObservationStencil, COLUMN_MEAN_PLACEMENT_CODE,
    FIXED_LEVEL_PLACEMENT_CODE, RETRIEVAL_COLUMN_PLACEMENT_CODE, CSObservationColumnMet,
    AbstractObservationKind, PointObservationKind, ColumnObservationKind, CSObservationRecord,
    CSObservationSet, read_observations, write_observations, AbstractOutOfRangePolicy,
    RejectOutOfRange, SkipOutOfRange, ClampOutOfRange, bind_to_mesh, CSDepartureRecord,
    CSDepartureSet, build_departure_set, read_departures, write_departures

include("Observations.jl")
include("ObservationColumns.jl")
include("ObservationsIO.jl")
include("ObservationBinding.jl")
include("DeparturesIO.jl")
include("OCO2MIPObservations.jl")
include("ForwardSampling.jl")

end # module Observations
