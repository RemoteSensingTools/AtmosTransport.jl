"""
    Advection

Advection operators for the basis-explicit transport architecture.

Provides:

**Scheme hierarchy**:
- `UpwindScheme <: AbstractConstantScheme` — first-order upwind via generic kernels
- `SlopesScheme <: AbstractLinearScheme`   — van Leer slopes (limiter-dispatched)
- `PPMScheme <: AbstractQuadraticScheme`   — structured-grid PPM (not yet an official real-data reference path)
- `LinRoodPPMScheme <: AbstractAdvectionScheme` — cubed-sphere FV3/Lin-Rood PPM path
- `AbstractLimiter` subtypes: `NoLimiter`, `MonotoneLimiter`, `PositivityLimiter`

**Multi-tracer optimization**:
- `TracerView` — zero-cost 3D slice adapter for 4D tracer arrays
- Multi-tracer kernel shells fuse the N-tracer loop into GPU kernels,
  reducing launches from 6N to 6 per Strang split

**Infrastructure**:
- `AdvectionWorkspace` + `strang_split!` — Strang splitting orchestrator
- CFL utilities for subcycling decisions
"""
module Advection

using Adapt
using DocStringExtensions

import ..AbstractOperator, ..apply!
using ...SectionTimer
# Diffusion is loaded before Advection in Operators.jl so the palindrome
# center of `strang_split_mt!` can dispatch on `AbstractDiffusion`
# concretions. `NoDiffusion`'s `apply_vertical_diffusion_vmr!` method is
# `= nothing`, keeping the default path bit-exact with the no-op behavior.
using ..Diffusion: AbstractDiffusion, DiffusionWorkspace, NoDiffusion,
                   apply_vertical_diffusion_vmr!,
                   uses_diffusive_surface_flux_boundary
# SurfaceFlux is loaded before Advection in Operators.jl so the palindrome
# center can dispatch on `AbstractSurfaceFluxOperator`.
# `NoSurfaceFlux`'s `apply_surface_flux!` method returns `nothing`, keeping
# the default path bit-exact with the no-op behavior.
using ..SurfaceFlux: AbstractSurfaceFluxOperator, NoSurfaceFlux,
                     apply_surface_flux!, emission_deposit
using ...State: CellState, CubedSphereState,
    AbstractStructuredFaceFluxState, AbstractFaceFluxState,
    StructuredFaceFluxState, AbstractUnstructuredFaceFluxState,
    DryBasis, AbstractMassBasis,
    FaceIndexedFluxState, CubedSphereFaceFluxState,
    ntracers, tracer_index, tracer_name, get_tracer, eachtracer
using ...Grids: AtmosGrid, AbstractHorizontalMesh, AbstractStructuredMesh,
    LatLonMesh, CubedSphereMesh, face_cells, nfaces,
    PanelConnectivity, reciprocal_edge,
    EDGE_NORTH, EDGE_SOUTH, EDGE_EAST, EDGE_WEST
using ...MetDrivers: uses_binary_substep_contract
using ...Architectures: _kahan_add

# New scheme hierarchy (include before anything that references these types)
include("schemes.jl")
include("limiters.jl")
include("reconstruction.jl")
include("structured_kernels.jl")
include("multitracer_kernels.jl")

# Cubed-sphere halo exchange and Strang splitting
include("HaloExchange.jl")
include("CubedSphereStrang.jl")
include("CubedSphereSeams.jl")
include("vertical_fv3_profile.jl")

# PPM subgrid distributions (shared by CS PPM kernels and LinRood)
include("ppm_subgrid_distributions.jl")

# Lin-Rood cross-term advection for cubed-sphere grids (FV3 fv_tp_2d)
include("LinRoodSeams.jl")
include("LinRood.jl")
include("linrood_adjoint_kernels.jl")

include("StrangSplitting.jl")

end # module Advection
