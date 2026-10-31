"""
    Runner

Driven simulation, window storage and forcing refresh, adjoint capture, input
staging and the configured transport runner. Depends only on lower siblings.
"""
module Runner

using ...Operators.Convection: convection_workspace_for

using ...State: basis_symbol

using Adapt
using ...Architectures
using ...SectionTimer
using ...State
using ...Grids
using ...Operators
# The step plan (Operators/StepPlan): `TransportModel.step!` and
# `DrivenSimulation` execute it; the cubed-sphere adjoint records the same plan.
using ...Operators.StepPlans: window_clock, window_step_dt,
                             WindowStartAirMass, WindowEndAirMass,
                             execute_window_advance!, execute_window_start!,
                             execute_step_stages!, execute_window_end!
import ...Operators.StepPlans: air_mass_reset_target
using ...Operators.StepPlans: AirMassResetStage
# The cubed-sphere adjoint replays the plan a DrivenSimulation executes
# (`CSStepSchedule(sim)`, `require_adjoint_support(sim)`, adjoint_schedule.jl).
import ...Adjoints
using ...Adjoints: CSStepSchedule, require_adjoint_support
using ...MetDrivers
using ...MetDrivers: same_window_layout, AbstractRawWindowStaging, raw_window_staging,
                    read_window_payload!, unpack_window_payload!
using ...Models.Core: TransportModel, with_chemistry, with_emissions, with_convection_forcing, step_operators
import ...Models.Core: step!, run!

include("BinaryPathExpander.jl")
using .BinaryPathExpander: expand_binary_paths
export expand_binary_paths
include("InputStaging.jl")
using .InputStaging: InputStager, staged_path_for!, cleanup_staging!
export InputStager, staged_path_for!, cleanup_staging!
include("DrivenSimulation.jl")
include("driven_window_state.jl")
include("driven_physics_refresh.jl")
include("driven_stepping.jl")
include("adjoint_schedule.jl")
export cs_adjoint_inputs
include("DrivenRunner/DrivenRunner.jl")
using .DrivenRunner: run_driven_simulation, validate_config, TransportTracerSpec
export run_driven_simulation, validate_config, TransportTracerSpec

end # module Runner
