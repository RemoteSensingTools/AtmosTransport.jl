module Solver
using ....Architectures: zeros_like

using KernelAbstractions: @Const, @index, @kernel, get_backend, synchronize
using ....Grids: CubedSphereMesh
using ....Operators.Diffusion: NoDiffusion
using ....Operators.Convection: NoConvection
using ....Parameters: DEFAULT_CFL_LIMIT
using ....Tape: AbstractCheckpointSchedule, FullCheckpoint
using ...Adjoints: AbstractCSFootprintObjective, CSAdjointSupportedScheme, _assemble_step_forcing, _require_cs_convection_workspace, _validate_cs_diffusion_inputs, _zero_surface_rates
using ..Observations: CSObservation, CSObservationSet, CSObservationStencil, _observation_replay,
    _observe_cs_forward!
import ..Observations: build_departure_set
using ...Footprint: CSFootprintResult, _UntapedRun, _begin_taped_run, _cs_objective_footprints,
    _end_taped_run!, _forced_reverse, _require_tape_storage_supported, _validate_step_sequences

export CSSurfaceFluxWindow, CSSurfaceFluxJacobianResult, CSSurfaceFluxControl, CS4DVarResult,
    CS4DVarSolveResult, AbstractCSSurfaceFluxCovariance, DiagonalCSCovariance,
    IsotropicGaussianCSCovariance, apply_B_half!, apply_B_half_adjoint!, AbstractCSOptimType,
    LinearOptimType, LogNormalOptimType, CSSurfaceFluxPreconditioner, apply_preconditioner!,
    apply_preconditioner_adjoint!, cs_surface_flux_jacobian, cs_surface_flux_4dvar,
    cs_surface_flux_4dvar_optimize, AbstractCS4DVarEvaluation, CSCostAndGradient, CSCostOnly,
    AbstractCSOptimizer, CSGradientDescent, CSLBFGS, cs_surface_flux_4dvar_solve, CSIterationLog,
    CSIterationLogEntry

include("Types.jl")
include("Covariance.jl")
include("Preconditioning.jl")
include("Jacobian.jl")
include("CostGradient.jl")
include("Optimizer.jl")
include("Departures.jl")

end # module Solver
