"""
    State

Prognostic and diagnostic state containers for the basis-explicit transport architecture.

Provides:
- `CellState` — cell-centered air mass + conservative tracer storage
- `AbstractFaceFluxState` hierarchy — face mass fluxes
  - `AbstractStructuredFaceFluxState` → `StructuredFaceFluxState` (am, bm, cm)
  - `AbstractUnstructuredFaceFluxState` → `FaceIndexedFluxState`
- Tracer allocation and mixing-ratio utilities
"""
module State

using ..Grids: CS_PANEL_COUNT

using Adapt
using ..Architectures: compensated_total
using ..Grids: AbstractHorizontalMesh, AbstractStructuredMesh, CubedSphereMesh,
    StructuredFluxTopology, FaceIndexedFluxTopology,
    ncells, nfaces, nx, ny

include("Basis.jl")
"""Common basis-tagged prognostic state interface for structured and panel storage."""
abstract type AbstractCellState{B <: AbstractMassBasis} end
export AbstractCellState

include("CellState.jl")
include("CubedSphereState.jl")
include("FaceFluxState.jl")
include("Tracers.jl")
include("Fields/Fields.jl")

using .Fields: AbstractTimeVaryingField, AbstractCubedSphereField,
               ConstantField, PreComputedKzField,
               CubedSphereField, AbstractCSDkgField, PrecomputedCSDkgField,
               GCHPNonlocalPBLField, GCHPVdiffParameters,
               StepwiseField,
               field_value, update_field!, refresh_precomputed_cs_dkg_cache!,
               panel_field
export AbstractTimeVaryingField, AbstractCubedSphereField,
       ConstantField, PreComputedKzField,
       CubedSphereField, AbstractCSDkgField, PrecomputedCSDkgField,
       GCHPNonlocalPBLField, GCHPVdiffParameters,
       StepwiseField,
       field_value, update_field!, refresh_precomputed_cs_dkg_cache!,
       panel_field

end # module State
