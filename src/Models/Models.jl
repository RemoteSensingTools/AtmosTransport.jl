"""
    Models

Minimal standalone runtime layer for `src`.
"""
module Models

using ..Operators.Convection: convection_workspace_for

using ..State: panel_interior

using ..State: basis_symbol

using Adapt
using ..Architectures
using ..SectionTimer
using ..State
using ..Grids
using ..Operators
# The step plan (Operators/StepPlan): `TransportModel.step!` and
# `DrivenSimulation` execute it; the cubed-sphere adjoint records the same plan.
using ..Operators.StepPlans: TransportStage, ConvectionStage, ChemistryBeforeStage,
                             ChemistryAfterStage, StepDt, WindowDt,
                             WindowStartAirMass, WindowEndAirMass,
                             execute_stage!, execute_window_advance!,
                             execute_window_start!, execute_step_stages!,
                             execute_window_end!,
                             _resolve_physics_cadence, _PHYSICS_CADENCES,
                             _normalize_air_mass_reset_mode, _AIR_MASS_RESET_MODES,
                             _reset_air_mass!, _refresh_state_halos!,
                             _cs_advection_workspace_for, _reject_binary_cfl_check,
                             _symmetric_decay_split, _half_step
import ..Operators.StepPlans: air_mass_reset_target
using ..Operators.StepPlans: AirMassResetStage
# The cubed-sphere adjoint replays the plan a DrivenSimulation executes
# (`CSStepSchedule(sim)`, `require_adjoint_support(sim)`, adjoint_schedule.jl).
import ..Adjoints
using ..Adjoints: CSStepSchedule, require_adjoint_support
using ..MetDrivers
using ..MetDrivers: same_window_layout, AbstractRawWindowStaging, raw_window_staging,
                    read_window_payload!, unpack_window_payload!
# Regridding is available to the nested InitialConditionIO module.
using ..Regridding
# Snapshot and observation APIs used by DrivenRunner.
using ..Output
using ..Observations

using ..ConfigChecks: config_bool as _config_bool
using ..ConfigChecks: config_int
using ..Parameters: DEFAULT_LINROOD_PPM_ORDER

include("TransportModel.jl")
include("RuntimeRecipeStyles.jl")    # runtime-style traits (dispatched on by specs)
include("RuntimePhysicsSpecs.jl")    # typed config specs + materialize (Oceananigans-style)
include("RuntimePhysicsRecipe.jl")
include("RuntimePhysicsMetadata.jl")
include("InitialConditionIO.jl")
using .InitialConditionIO: FileInitialConditionSource,
                           AbstractInitialConditionSpec,
                           UniformInitialConditionSpec,
                           LatitudeStepInitialConditionSpec,
                           BoundaryLayerEnhancedInitialConditionSpec,
                           GaussianBlobInitialConditionSpec,
                           FileInitialConditionSpec,
                           CatrineCO2InitialConditionSpec,
                           PressureLayerInitialConditionSpec,
                           CSNativeInitialConditionSpec,
                           initial_condition_spec,
                           build_initial_mixing_ratio,
                           pack_initial_tracer_mass,
                           FileSurfaceFluxField,
                           build_surface_flux_source,
                           build_surface_flux_sources
export FileInitialConditionSource, build_initial_mixing_ratio, pack_initial_tracer_mass
export AbstractInitialConditionSpec, UniformInitialConditionSpec, initial_condition_spec
export LatitudeStepInitialConditionSpec, BoundaryLayerEnhancedInitialConditionSpec
export GaussianBlobInitialConditionSpec, FileInitialConditionSpec
export CatrineCO2InitialConditionSpec, PressureLayerInitialConditionSpec
export CSNativeInitialConditionSpec
export FileSurfaceFluxField, build_surface_flux_source, build_surface_flux_sources
include("BinaryPathExpander.jl")  # `[input]` folder+date-range
using .BinaryPathExpander: expand_binary_paths
export expand_binary_paths
include("InputStaging.jl")        # rolling NVMe staging of the daily binaries
using .InputStaging: InputStager, staged_path_for!, cleanup_staging!
export InputStager, staged_path_for!, cleanup_staging!
include("Simulation.jl")
include("Callbacks.jl")          # schedules, Callback, NaNChecker (Oceananigans-style)
export AbstractSchedule, IterationInterval, WindowInterval, WallTimeInterval, Callback, NaNChecker
include("DrivenSimulation.jl")
include("driven_window_state.jl")
include("driven_physics_refresh.jl")
include("driven_stepping.jl")
include("adjoint_schedule.jl")    # the adjoint of a DrivenSimulation configuration
export cs_adjoint_inputs
include("DrivenRunner.jl")        # library-level driven runner
using .DrivenRunner: run_driven_simulation, validate_config, TransportTracerSpec
export run_driven_simulation, validate_config, TransportTracerSpec

end # module Models
