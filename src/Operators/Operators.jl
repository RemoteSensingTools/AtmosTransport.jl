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

# Diffusion is included BEFORE Advection so `strang_split_mt!`
# (palindrome integration) can import
# `NoDiffusion`, `AbstractDiffusion`, and
# `apply_vertical_diffusion_vmr!`. Diffusion has no dependency on
# Advection; reordering preserves correctness.
include("Diffusion/Diffusion.jl")
using .Diffusion

# SurfaceFlux is included BEFORE Advection so `strang_split_mt!`
# (palindrome integration) can import
# `NoSurfaceFlux`, `AbstractSurfaceFluxOperator`, and
# `apply_surface_flux!`.
include("SurfaceFlux/SurfaceFlux.jl")
using .SurfaceFlux

# Convection is included before Advection. Convection
# runs as a SEPARATE block in `TransportModel.step!`
# (between the transport palindrome and the chemistry block), so
# `strang_split_mt!` doesn't need the convection types. The include
# order still puts Convection alongside Diffusion/SurfaceFlux (both
# column/point-local physics) for consistency.
include("Convection/Convection.jl")
using .Convection

include("Advection/Advection.jl")
using .Advection

include("Chemistry/Chemistry.jl")
using .Chemistry

export AdvectionWorkspace, strang_split!, strang_split_mt!
export TracerView

# Advection scheme hierarchy
export AbstractAdvectionScheme
export AbstractConstantScheme, AbstractLinearScheme, AbstractQuadraticScheme
export AbstractLimiter, NoLimiter, MonotoneLimiter, PositivityLimiter, CW84Limiter
export UpwindScheme, SlopesScheme, PPMScheme, LinRoodPPMScheme, NoAdvection
export AbstractVerticalReconstruction, SameAsHorizontal, FV3ScalarProfile
export reconstruction_order, required_halo_width

# Chemistry
export AbstractChemistryOperator, NoChemistry, ExponentialDecay, CompositeChemistry

# Diffusion solver infrastructure + operator types
export solve_tridiagonal!
export AbstractDiffusion, NoDiffusion, ImplicitVerticalDiffusion, DiffusionWorkspace
export AbstractSurfaceFluxCoupling, SplitSurfaceFluxCoupling,
       DiffusiveSurfaceFluxBoundary, uses_diffusive_surface_flux_boundary
export apply_vertical_diffusion_vmr!

# SurfaceFlux data types + operator hierarchy
export SurfaceFluxSource, AbstractSurfaceFluxSource, TimeVaryingSurfaceFluxSource,
       PerTracerFluxMap, flux_for
export AbstractFluxTemporalScheme, StepwiseFlux, LinearInterpFlux, ConservativeMeanFlux,
       flux_temporal_scheme
export AbstractSurfaceFluxOperator, NoSurfaceFlux, SurfaceFluxOperator
export apply_surface_flux!

# Convection operator hierarchy.
export AbstractConvection, NoConvection
export CMFMCConvection, AbstractCloudBase, CMFMCEdgeCloudBase, ArchivedCloudBase
export CMFMCWorkspace, invalidate_cmfmc_cache!
export TM5Convection
export TM5Workspace
export CMFMCMatrixConvection                    # GEOS rates → TM5 LU (conservative CMFMC)
export CMFMCMatrixWorkspace, invalidate_cmfmc_matrix_cache!
export apply_convection!

# Cubed-sphere advection
export fill_panel_halos!, copy_corners!, strang_split_cs!, strang_split_cs_mt!,
       CSAdvectionWorkspace
export AbstractBinaryCFLCheck, NoBinaryCFLCheck, BinaryCFLCheck

# Lin-Rood cross-term advection (FV3 fv_tp_2d)
export LinRoodWorkspace, CSLinRoodAdvectionWorkspace
export fv_tp_2d_cs!, fv_tp_2d_cs_q!, strang_split_linrood_ppm!
export apply_divergence_damping_cs!

end # module Operators
