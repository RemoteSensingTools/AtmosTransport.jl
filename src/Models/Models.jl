"""
    Models

Runtime composition with ordered Physics, Core, InitialConditionIO and Runner
submodules. Their public bindings remain available through this facade.
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
using ..Operators.StepPlans: window_clock, window_step_dt, TransportStage, ConvectionStage, ChemistryBeforeStage,
                             ChemistryAfterStage, StepDt, WindowDt,
                             WindowStartAirMass, WindowEndAirMass,
                             execute_stage!, execute_window_advance!,
                             execute_window_start!, execute_step_stages!,
                             execute_window_end!
import ..Operators.StepPlans: air_mass_reset_target
using ..Operators.StepPlans: AirMassResetStage
# The cubed-sphere adjoint replays the plan a DrivenSimulation executes
# (`CSStepSchedule(sim)`, `require_adjoint_support(sim)`, adjoint_schedule.jl).
import ..Adjoints
using ..Adjoints: CSStepSchedule, require_adjoint_support
using ..MetDrivers
using ..MetDrivers: same_window_layout, AbstractRawWindowStaging, raw_window_staging,
                    read_window_payload!, unpack_window_payload!
using ..Regridding
# Snapshot and observation APIs used by DrivenRunner.
using ..Output
using ..Observations

using ..ConfigChecks: config_int
using ..Parameters: DEFAULT_LINROOD_PPM_ORDER

# Dependency order: Physics -> Core -> InitialConditionIO -> Runner.
# The facade re-exports the public names of each submodule. Private helpers stay
# with their owners; tests reach them through the owning submodule.
include("Physics/Physics.jl")
using .Physics:
    AbstractAdvectionContext, AbstractAdvectionSpec, AbstractChemistrySpec, AbstractCollabLUConvectionSpec, AbstractConvectionSpec,
    AbstractDiffusionSpec, AbstractRuntimeRecipeStyle, AbstractStructuredRuntimeRecipeStyle, CMFMCConvectionSpec, CMFMCMatrixConvectionSpec,
    ConstantDiffusionSpec, CubedSphereRuntimeRecipeStyle, DEFAULT_AIR_MASS_RESET_MODE, DecayChemistrySpec, ForwardAdvectionContext,
    GCHPNonlocalVdiffDiffusionSpec, HoltslagBovilleVdiffDiffusionSpec, InversionAdvectionContext, LatLonRuntimeRecipeStyle, LinRoodAdvectionSpec,
    NoAdvectionSpec, NoChemistrySpec, NoConvectionSpec, NoDiffusionSpec, PPMAdvectionSpec,
    ReducedGaussianRuntimeRecipeStyle, RuntimePhysicsRecipe, SlopesAdvectionSpec, TM5ConvectionSpec, TM5DkgDiffusionSpec,
    UpwindAdvectionSpec, WindowPBLKzDiffusionSpec,
    advection_spec, binary_cfl_check, build_runtime_advection,
    build_runtime_chemistry, build_runtime_convection, build_runtime_diffusion, build_runtime_physics_recipe, chemistry_spec,
    configured_halo_width, convection_spec, diffusion_spec, materialize, runtime_physics_attributes,
    validate_runtime_advection, validate_runtime_convection, validate_runtime_diffusion, validate_runtime_halo_width, validate_runtime_physics_recipe
include("Core/Core.jl")
using ..Models.Core:
    AbstractSchedule, Callback, IterationInterval, NaNChecker, NonfiniteScan,
    Simulation, TransportModel, TransportModelWorkspace, WallTimeInterval, WindowInterval,
    run!, step!, step_operators, with_chemistry, with_convection,
    with_convection_forcing, with_diffusion, with_emissions
include("InitialConditions/InitialConditionIO.jl")
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
include("Runner/Runner.jl")
using .Runner: DrivenSimulation, cs_adjoint_inputs, current_qv, run_window!, substep_index, window_index
using .Runner: DrivenRunner, BinaryPathExpander, InputStaging,
               expand_binary_paths, InputStager, staged_path_for!, cleanup_staging!,
               run_driven_simulation, validate_config, TransportTracerSpec
export expand_binary_paths
export InputStager, staged_path_for!, cleanup_staging!
export AbstractSchedule, IterationInterval, WindowInterval, WallTimeInterval, Callback, NaNChecker
export cs_adjoint_inputs
export run_driven_simulation, validate_config, TransportTracerSpec
export uniform_background
export binary_date_label, configured_float_type, check_multifile_window_count, layout_run_hours, parse_tracer_specs, run_reference_time, run_time_origin, run_time_span, run_window_index, runtime_window_layout, structured_tracer_specs, runtime_output_origin, window_end_hours, observation_terrain, run_tracer_specs
export ADVECTION_SPEC_KEYS
# Public runtime-layout aliases and the IC background helper.
using .InitialConditionIO: uniform_background
using .DrivenRunner: binary_date_label, configured_float_type, check_multifile_window_count, layout_run_hours, parse_tracer_specs, run_reference_time, run_time_origin, run_time_span, run_window_index, runtime_window_layout, structured_tracer_specs, runtime_output_origin, window_end_hours, observation_terrain, run_tracer_specs
const ADVECTION_SPEC_KEYS = Physics._ADVECTION_SPEC_KEYS

export RuntimePhysicsRecipe
export build_runtime_advection, build_runtime_diffusion, build_runtime_convection
export build_runtime_chemistry
export build_runtime_physics_recipe, validate_runtime_physics_recipe
export configured_halo_width
export AbstractConvectionSpec, AbstractCollabLUConvectionSpec, NoConvectionSpec,
       TM5ConvectionSpec, CMFMCConvectionSpec, CMFMCMatrixConvectionSpec
export AbstractAdvectionSpec, UpwindAdvectionSpec, SlopesAdvectionSpec,
       PPMAdvectionSpec, NoAdvectionSpec, LinRoodAdvectionSpec
export AbstractAdvectionContext, ForwardAdvectionContext, InversionAdvectionContext
export AbstractChemistrySpec, NoChemistrySpec, DecayChemistrySpec
export AbstractDiffusionSpec, NoDiffusionSpec, ConstantDiffusionSpec,
       WindowPBLKzDiffusionSpec, HoltslagBovilleVdiffDiffusionSpec,
       GCHPNonlocalVdiffDiffusionSpec,
       TM5DkgDiffusionSpec
export convection_spec, advection_spec, chemistry_spec, diffusion_spec, materialize

export Simulation, run!
export TransportModel, step!
export with_chemistry, with_diffusion, with_emissions
export with_convection, with_convection_forcing
export DrivenSimulation, run_window!, window_index, substep_index, current_qv
end # module Models
