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
               ConstantField, ProfileKzField, PreComputedKzField,
               CubedSphereField, DerivedKzField, WindowPBLKzField,
               LocalHoltslagBovilleKzField,
               AbstractCSDkgField, PrecomputedCSDkgField,
               GCHPNonlocalPBLField, GCHPVdiffParameters, refresh_gchp_nonlocal_pbl!,
               PBLPhysicsParameters, StepwiseField,
               field_value, update_field!, refresh_pbl_kz_cache!,
               refresh_local_holtslag_boville_kz_cache!,
               refresh_precomputed_cs_dkg_cache!,
               integral_between, panel_field
export AbstractTimeVaryingField, AbstractCubedSphereField,
       ConstantField, ProfileKzField, PreComputedKzField,
       CubedSphereField, DerivedKzField, WindowPBLKzField,
       LocalHoltslagBovilleKzField,
       AbstractCSDkgField, PrecomputedCSDkgField,
       GCHPNonlocalPBLField, GCHPVdiffParameters, refresh_gchp_nonlocal_pbl!,
       PBLPhysicsParameters, StepwiseField,
       field_value, update_field!, refresh_pbl_kz_cache!,
       refresh_local_holtslag_boville_kz_cache!,
       refresh_precomputed_cs_dkg_cache!,
       integral_between, panel_field

end # module State
