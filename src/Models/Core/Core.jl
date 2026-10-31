"""
    Core

Model construction, fixed-step simulation and callbacks. Driven methods live
with DrivenSimulation in Runner, which extends Core's step! and run! generics.
"""
module Core

using Adapt
using ...Architectures
using ...SectionTimer
using ...State
using ...Grids
using ...Operators
using ...MetDrivers
using ...Operators.Convection: convection_workspace_for
using ...Operators.StepPlans: _cs_advection_workspace_for, execute_step_stages!
import ...Adjoints
using ...Adjoints: require_adjoint_support

# Stepping generics shared with Runner, which implements the DrivenSimulation
# methods beside their type in ../Runner/driven_stepping.jl.
function step! end
function run! end

include("TransportModel.jl")
include("Simulation.jl")
include("Callbacks.jl")
include("adjoint_support.jl")
export AbstractSchedule, IterationInterval, WindowInterval, WallTimeInterval, Callback, NaNChecker

end # module Core
