"""
    Advection

Advection operators for the basis-explicit transport architecture.

Provides:

**Scheme hierarchy**:
- `UpwindScheme <: AbstractConstantScheme` — first-order upwind via generic kernels
- `SlopesScheme <: AbstractLinearScheme`   — van Leer slopes (limiter-dispatched)
- `PPMScheme <: AbstractQuadraticScheme`   — structured-grid PPM (not yet an official real-data reference path)
- `LinRoodPPMScheme <: AbstractAdvectionScheme` — cubed-sphere FV3/Lin-Rood PPM path
- `AbstractLimiter` subtypes: `NoLimiter`, `MonotoneLimiter`, `PositivityLimiter`, `CW84Limiter` (PPM only)

**Multi-tracer optimization**:
- `TracerView` — zero-cost 3D slice adapter for 4D tracer arrays
- Multi-tracer kernel shells fuse the N-tracer loop into GPU kernels,
  reducing launches from 6N to 6 per Strang split

**Infrastructure**:
- `AdvectionWorkspace` + `strang_split!` — Strang splitting orchestrator
- CFL utilities for subcycling decisions
"""
module Advection

using ...Grids: cs_panel_edge_seam, canonical_seams

using ...Grids: face_connectivity

using ...Grids: edge_interior_ij, edge_halo_ij, corner_source_ij, cs_seam_axis, cs_seam_sign, cs_seam_face_index, cs_seam

using Adapt
using DocStringExtensions

import ..AbstractOperator, ..apply!
import ..AbstractStepRecorder, ..NoStepRecorder, ..record_halo!, ..record_sweep!,
       ..record_linrood_horizontal!
using ...SectionTimer
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
using ...Parameters: DEFAULT_CFL_LIMIT, DEFAULT_MAX_SUBCYCLES, DEFAULT_LINROOD_PPM_ORDER

include("midpoint.jl")

# New scheme hierarchy (include before anything that references these types)
include("Reconstruction/schemes.jl")
include("Reconstruction/limiters.jl")
include("Reconstruction/reconstruction.jl")
include("Structured/multitracer_kernels.jl")

# Cubed-sphere halo exchange and Strang splitting
include("CubedSphere/HaloExchange.jl")
include("CubedSphere/cs_sweep_common.jl")
include("CubedSphere/cs_sweep_x.jl")
include("CubedSphere/cs_sweep_y.jl")
include("CubedSphere/cs_sweep_z.jl")
include("cs_workspace.jl")
include("Structured/subcycling.jl")
include("CubedSphere/cs_subcycling.jl")
include("CubedSphere/CubedSphereStrang.jl")
include("CubedSphere/CubedSphereSeams.jl")
include("CubedSphere/vertical_fv3_profile.jl")
include("CubedSphere/vertical_fv3_profile_adjoint.jl")

# PPM subgrid distributions (shared by CS PPM kernels and LinRood)
include("Reconstruction/ppm_subgrid_distributions.jl")

# Lin-Rood cross-term advection for cubed-sphere grids (FV3 fv_tp_2d)
include("LinRoodSeams.jl")
include("LinRood.jl")
include("linrood_horizontal.jl")
include("linrood_adjoint_kernels.jl")
include("linrood_adjoint_rm_faces.jl")
include("linrood_adjoint_panel.jl")

include("Structured/workspace.jl")
include("Structured/sweeps.jl")
include("Structured/StrangSplitting.jl")
include("Structured/strang_apply.jl")
include("cs_dispatch.jl")
include("Structured/multitracer_strang.jl")

include("CubedSphere/adjoint_workspace.jl")
include("CubedSphere/AdvectionAdjoint.jl")
include("CubedSphere/CubedSphereSeamsAdjoint.jl")
include("CubedSphere/HaloAdjoint.jl")

end # module Advection
