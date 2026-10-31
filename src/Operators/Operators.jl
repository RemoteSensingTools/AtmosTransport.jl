"""
    Operators

Physics operators for the basis-explicit transport architecture.

Provides:
- Abstract operator types (`AbstractOperator`, `AbstractDiffusion`, etc.)
- Advection hierarchy: `AbstractAdvectionScheme` → `AbstractConstantScheme`,
  `AbstractLinearScheme`, `AbstractQuadraticScheme` with concrete
  `UpwindScheme`, `SlopesScheme`, and structured-grid `PPMScheme`
- Strang splitting orchestrator: `strang_split!`, `apply!`
"""
module Operators

# Re-export State and Grids into Operators scope for sub-submodules
using ..SectionTimer
using ..State
using ..Grids
using ..MetDrivers

include("AbstractOperators.jl")

# Step recorders: the observation hooks the production step calls (no-ops for
# `NoStepRecorder`) and the cubed-sphere adjoint records its tape through.
include("StepRecording.jl")
include("OperatorState.jl")

# Physics families load before StepPlans, which supplies the transport midpoint.
include("Diffusion/Diffusion.jl")
using .Diffusion

include("SurfaceFlux/SurfaceFlux.jl")
using .SurfaceFlux

include("Convection/Convection.jl")
using .Convection

include("Advection/Advection.jl")
using .Advection

include("Chemistry/Chemistry.jl")
using .Chemistry

# The model step as data (`StepPlan`): executed by `TransportModel.step!` and
# `DrivenSimulation`, recorded and transposed by the cubed-sphere adjoint.
include("StepPlan/StepPlan.jl")
using .StepPlans

export AdvectionWorkspace, strang_split!, strang_split_mt!
export TracerView

# Advection scheme hierarchy
export AbstractAdvectionScheme
export AbstractConstantScheme, AbstractLinearScheme, AbstractQuadraticScheme
export AbstractLimiter, NoLimiter, MonotoneLimiter, PositivityLimiter, CW84Limiter
export UpwindScheme, SlopesScheme, PPMScheme, LinRoodPPMScheme, NoAdvection
export AbstractVerticalReconstruction, SameAsHorizontal, FV3ScalarProfile
export required_halo_width

# Chemistry
export AbstractChemistryOperator, NoChemistry, ExponentialDecay, CompositeChemistry

# Diffusion solver infrastructure + operator types
export AbstractDiffusion, NoDiffusion, ImplicitVerticalDiffusion, DiffusionWorkspace
export AbstractSurfaceFluxCoupling, SplitSurfaceFluxCoupling,
       DiffusiveSurfaceFluxBoundary, uses_diffusive_surface_flux_boundary
export apply_vertical_diffusion_vmr!
export refresh_gchp_nonlocal_pbl!
export WindowPBLKzField, LocalHoltslagBovilleKzField, PBLPhysicsParameters
export refresh_pbl_kz_cache!, refresh_local_holtslag_boville_kz_cache!

# SurfaceFlux data types + operator hierarchy
export SurfaceFluxSource, AbstractSurfaceFluxSource, TimeVaryingSurfaceFluxSource,
       PerTracerFluxMap, flux_for
export AbstractFluxTemporalScheme, StepwiseFlux, LinearInterpFlux, ConservativeMeanFlux,
       flux_temporal_scheme
export AbstractSurfaceFluxOperator, NoSurfaceFlux, SurfaceFluxOperator
export apply_surface_flux!
export total_emission_rate, emission_segments, integrated_emission

# Convection operator hierarchy.
export AbstractConvection, NoConvection
export CMFMCConvection, AbstractCloudBase, CMFMCEdgeCloudBase, ArchivedCloudBase
export CMFMCWorkspace, invalidate_cmfmc_cache!
export TM5Convection
export TM5Workspace
export CMFMCMatrixConvection                    # GEOS rates → TM5 LU (conservative CMFMC)
export CMFMCMatrixWorkspace, invalidate_cmfmc_matrix_cache!
export apply_convection!

# Step plans and recorders
export StepPlan, StepOperators, StepTiming
export AbstractStepRecorder, NoStepRecorder

# Cubed-sphere advection
export fill_panel_halos!, copy_corners!, strang_split_cs_mt!,
       CSAdvectionWorkspace
export AbstractBinaryCFLCheck, NoBinaryCFLCheck, BinaryCFLCheck

# Lin-Rood cross-term advection (FV3 fv_tp_2d)
export LinRoodWorkspace, CSLinRoodAdvectionWorkspace
export fv_tp_2d_cs!, fv_tp_2d_cs_q!, strang_split_linrood_ppm!
export apply_divergence_damping_cs!

end # module Operators
