# ---------------------------------------------------------------------------
# FaceFluxState — face-centered mass fluxes for transport
#
# Abstraction lives at the operator contract level, not the storage level:
#
#   abstract type AbstractFaceFluxState end
#       ├── AbstractStructuredFaceFluxState   (am, bm, cm on logically rectangular meshes)
#       └── AbstractUnstructuredFaceFluxState  (face-indexed connectivity)
#
# Each concrete flux state carries a `Basis <: AbstractMassBasis` type
# parameter that records whether the stored fluxes are on a moist or dry
# basis. This prevents accidentally mixing state/flux basis at dispatch time.
#
# Structured meshes keep the proven directional storage (am, bm, cm) and
# cell-loop kernels.  Unstructured meshes will get face-indexed storage and
# face-loop kernels.  The operator dispatch selects the right realization:
#
#   apply!(state, fluxes::StructuredFaceFluxState{DryBasis}, grid, scheme, dt)
#
# Same math, same high-level API, different low-level memory layout,
# different kernels where justified.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Abstract hierarchy
# ---------------------------------------------------------------------------

"""
    AbstractFaceFluxState

Root type for all face-centered mass flux representations.

The transport operator contract is written in terms of this abstract type.
Concrete subtypes differ in storage layout to match the mesh's natural
`flux_topology`, and carry a `Basis <: AbstractMassBasis` type parameter
to enforce moist/dry safety.
"""
abstract type AbstractFaceFluxState{Basis <: AbstractMassBasis} end

"""
    AbstractStructuredFaceFluxState <: AbstractFaceFluxState

Face fluxes stored as separate directional arrays on a logically rectangular
mesh.  Concrete subtypes expose `am` (x-face), `bm` (y-face), `cm` (z-face).

Structured cell-loop kernels access these arrays directly for performance.
"""
abstract type AbstractStructuredFaceFluxState{Basis <: AbstractMassBasis} <: AbstractFaceFluxState{Basis} end

"""
    AbstractUnstructuredFaceFluxState <: AbstractFaceFluxState

Face fluxes stored as a single face-indexed array with explicit connectivity.
Natural for reduced Gaussian and other unstructured meshes.
"""
abstract type AbstractUnstructuredFaceFluxState{Basis <: AbstractMassBasis} <: AbstractFaceFluxState{Basis} end

# ---------------------------------------------------------------------------
# Concrete structured type — keeps am, bm, cm for the fast path
# ---------------------------------------------------------------------------

"""
    StructuredFaceFluxState{Basis, AX, AY, AZ} <: AbstractStructuredFaceFluxState

Face-centered mass fluxes for structured grids, tagged with `Basis` to
indicate whether the stored values are moist or dry.

# Type parameters
- `Basis <: AbstractMassBasis` — `MoistBasis` or `DryBasis`

# Fields
- `am :: AX` — x-face (longitude) prepared substep mass transport [kg for the active transport substep].
  Layout: `(Nx+1, Ny, Nz)` for LatLon, `(Nc+1, Nc, Nz)` per panel for CS.
- `bm :: AY` — y-face (latitude) prepared substep mass transport [kg for the active transport substep].
  Layout: `(Nx, Ny+1, Nz)` for LatLon, `(Nc, Nc+1, Nz)` per panel for CS.
- `cm :: AZ` — z-face (vertical) prepared substep mass transport [kg for the active transport substep].
  Layout: `(Nx, Ny, Nz+1)` for LatLon.

# Convention
- Positive `am` = eastward mass transport
- Positive `bm` = northward mass transport
- Positive `cm` = downward (increasing k / pressure) mass transport

# Examples
```jldoctest
julia> using AtmosTransport.State: StructuredFaceFluxState, DryBasis,
       MoistBasis, mass_basis

julia> am = zeros(11, 8, 4); bm = zeros(10, 9, 4); cm = zeros(10, 8, 5);

julia> dry = StructuredFaceFluxState{DryBasis}(am, bm, cm);

julia> mass_basis(dry)
DryBasis()

julia> moist = StructuredFaceFluxState{MoistBasis}(am, bm, cm);

julia> mass_basis(moist)
MoistBasis()
```
"""
function _validate_structured_flux_storage(am, bm, cm)
    all(array -> ndims(array) == 3, (am, bm, cm)) || throw(DimensionMismatch(
        "StructuredFaceFluxState fields must all be rank 3"))
    expected_am = (size(cm, 1) + 1, size(cm, 2), size(cm, 3) - 1)
    expected_bm = (size(cm, 1), size(cm, 2) + 1, size(cm, 3) - 1)
    size(am) == expected_am || throw(DimensionMismatch(
        "StructuredFaceFluxState am has shape $(size(am)); expected $(expected_am) from cm"))
    size(bm) == expected_bm || throw(DimensionMismatch(
        "StructuredFaceFluxState bm has shape $(size(bm)); expected $(expected_bm) from cm"))
    eltype(am) === eltype(cm) && eltype(bm) === eltype(cm) || throw(ArgumentError(
        "StructuredFaceFluxState fields must share one element type; got " *
        "$(eltype(am)), $(eltype(bm)), and $(eltype(cm))"))
    return nothing
end

"""
    StructuredFaceFluxState{Basis}(am, bm, cm)

Prepared substep mass transport on a logically rectangular grid. Arrays use
shapes (Nx+1, Ny, Nz), (Nx, Ny+1, Nz), and (Nx, Ny, Nz+1). Positive horizontal
fluxes point east/north; positive vertical flux points toward increasing level
index. Basis must match the prognostic state.
"""
struct StructuredFaceFluxState{Basis <: AbstractMassBasis,
                                AX <: AbstractArray,
                                AY <: AbstractArray,
                                AZ <: AbstractArray} <: AbstractStructuredFaceFluxState{Basis}
    am :: AX
    bm :: AY
    cm :: AZ

    function StructuredFaceFluxState{Basis, AX, AY, AZ}(
            am::AX, bm::AY, cm::AZ) where
            {Basis <: AbstractMassBasis, AX <: AbstractArray,
             AY <: AbstractArray, AZ <: AbstractArray}
        _validate_structured_flux_storage(am, bm, cm)
        return new{Basis, AX, AY, AZ}(am, bm, cm)
    end
end

function StructuredFaceFluxState{B}(am::AX, bm::AY, cm::AZ) where {B <: AbstractMassBasis,
                                                                       AX <: AbstractArray,
                                                                       AY <: AbstractArray,
                                                                       AZ <: AbstractArray}
    StructuredFaceFluxState{B, AX, AY, AZ}(am, bm, cm)
end

StructuredFaceFluxState(am, bm, cm) = StructuredFaceFluxState{DryBasis}(am, bm, cm)

function Adapt.adapt_structure(to, fluxes::StructuredFaceFluxState{B}) where {B <: AbstractMassBasis}
    am = Adapt.adapt(to, fluxes.am)
    bm = Adapt.adapt(to, fluxes.bm)
    cm = Adapt.adapt(to, fluxes.cm)
    return StructuredFaceFluxState{B, typeof(am), typeof(bm), typeof(cm)}(am, bm, cm)
end

# ---------------------------------------------------------------------------
# Concrete unstructured type
# ---------------------------------------------------------------------------

"""
    FaceIndexedFluxState{Basis, A, AZ} <: AbstractUnstructuredFaceFluxState

Face-centered mass fluxes for unstructured meshes, tagged with
`Basis` for moist/dry safety.

# Type parameters
- `Basis <: AbstractMassBasis` — `MoistBasis` or `DryBasis`

# Fields
- `horizontal_flux :: A` — prepared substep mass transport per horizontal face [kg for the active transport substep].
  Layout: `(nfaces, Nz)`. Positive = flow in face-normal direction.
- `cm :: AZ` — vertical flux, same convention as structured.

# Vertical storage convention
The `cm` field assumes vertical fluxes are columnar (one column per horizontal
cell, same for all mesh types). This is a convenience that holds for every
atmospheric grid we currently target (ERA5, GEOS-FP, GEOS-IT, reduced Gaussian).
If a future mesh requires non-columnar vertical connectivity, define a new
concrete subtype of `AbstractUnstructuredFaceFluxState` with different storage —
the abstract hierarchy supports this without breaking existing code.
"""
function _validate_face_indexed_flux_storage(horizontal_flux, cm)
    ndims(horizontal_flux) == 2 || throw(DimensionMismatch(
        "FaceIndexedFluxState horizontal_flux must be rank 2, got shape $(size(horizontal_flux))"))
    ndims(cm) == 2 || throw(DimensionMismatch(
        "FaceIndexedFluxState cm must be rank 2, got shape $(size(cm))"))
    size(cm, 2) == size(horizontal_flux, 2) + 1 || throw(DimensionMismatch(
        "FaceIndexedFluxState cm must have one more vertical interface than " *
        "horizontal_flux levels; got $(size(cm, 2)) and $(size(horizontal_flux, 2))"))
    eltype(horizontal_flux) === eltype(cm) || throw(ArgumentError(
        "FaceIndexedFluxState fields must share one element type; got " *
        "$(eltype(horizontal_flux)) and $(eltype(cm))"))
    return nothing
end

"""
    FaceIndexedFluxState{Basis}(horizontal_flux, cm)

Prepared substep mass transport for a connected-face mesh. Horizontal storage
has shape (nfaces, Nz), while vertical storage has shape (ncells, Nz+1).
Positive horizontal flux follows the mesh face orientation.
"""
struct FaceIndexedFluxState{Basis <: AbstractMassBasis,
                             A <: AbstractArray,
                             AZ <: AbstractArray} <: AbstractUnstructuredFaceFluxState{Basis}
    horizontal_flux :: A
    cm              :: AZ

    function FaceIndexedFluxState{Basis, A, AZ}(
            horizontal_flux::A, cm::AZ) where
            {Basis <: AbstractMassBasis, A <: AbstractArray, AZ <: AbstractArray}
        _validate_face_indexed_flux_storage(horizontal_flux, cm)
        return new{Basis, A, AZ}(horizontal_flux, cm)
    end
end

"""
    CubedSphereFaceFluxState{Basis, AX, AY, AZ} <: AbstractStructuredFaceFluxState

Panel-native structured-directional flux storage for cubed-sphere transport.
Each field is an `NTuple{CS_PANEL_COUNT}` of halo-padded panel arrays matching the
`strang_split_cs_mt!` contract.
"""
function _validate_cs_flux_storage(am, bm, cm)
    all(panel -> ndims(panel) == 3, (am..., bm..., cm...)) ||
        throw(DimensionMismatch(
            "CubedSphereFaceFluxState panels must all be rank 3"))
    reference_cm = size(cm[1])
    all(panel -> size(panel) == reference_cm, cm) || throw(DimensionMismatch(
        "CubedSphereFaceFluxState cm panels must have identical shapes"))
    expected_am = (reference_cm[1] + 1, reference_cm[2], reference_cm[3] - 1)
    expected_bm = (reference_cm[1], reference_cm[2] + 1, reference_cm[3] - 1)
    all(panel -> size(panel) == expected_am, am) || throw(DimensionMismatch(
        "CubedSphereFaceFluxState am panels must all have shape $(expected_am)"))
    all(panel -> size(panel) == expected_bm, bm) || throw(DimensionMismatch(
        "CubedSphereFaceFluxState bm panels must all have shape $(expected_bm)"))
    reference_eltype = eltype(cm[1])
    all(panel -> eltype(panel) === reference_eltype, (am..., bm..., cm...)) ||
        throw(ArgumentError(
            "CubedSphereFaceFluxState panels must share one element type"))
    return nothing
end

"""
    CubedSphereFaceFluxState{Basis}(am, bm, cm)

Six-panel, halo-padded directional mass-flux storage for cubed-sphere
transport. Each tuple contains one structured directional array per panel and
uses the same mass basis as CubedSphereState.
"""
struct CubedSphereFaceFluxState{Basis <: AbstractMassBasis,
                                AX <: AbstractArray,
                                AY <: AbstractArray,
                                AZ <: AbstractArray} <: AbstractStructuredFaceFluxState{Basis}
    am :: NTuple{CS_PANEL_COUNT, AX}
    bm :: NTuple{CS_PANEL_COUNT, AY}
    cm :: NTuple{CS_PANEL_COUNT, AZ}

    function CubedSphereFaceFluxState{Basis, AX, AY, AZ}(
            am::NTuple{CS_PANEL_COUNT, AX}, bm::NTuple{CS_PANEL_COUNT, AY}, cm::NTuple{CS_PANEL_COUNT, AZ}) where
            {Basis <: AbstractMassBasis, AX <: AbstractArray,
             AY <: AbstractArray, AZ <: AbstractArray}
        _validate_cs_flux_storage(am, bm, cm)
        return new{Basis, AX, AY, AZ}(am, bm, cm)
    end
end

function CubedSphereFaceFluxState{B}(am::NTuple{CS_PANEL_COUNT}, bm::NTuple{CS_PANEL_COUNT}, cm::NTuple{CS_PANEL_COUNT}) where {B <: AbstractMassBasis}
    return CubedSphereFaceFluxState{B, typeof(am[1]), typeof(bm[1]), typeof(cm[1])}(am, bm, cm)
end

function FaceIndexedFluxState{B}(hflux::A, cm::AZ) where {B <: AbstractMassBasis,
                                                            A <: AbstractArray,
                                                            AZ <: AbstractArray}
    FaceIndexedFluxState{B, A, AZ}(hflux, cm)
end

FaceIndexedFluxState(hflux, cm) = FaceIndexedFluxState{DryBasis}(hflux, cm)

function Adapt.adapt_structure(to, fluxes::FaceIndexedFluxState{B}) where {B <: AbstractMassBasis}
    hflux = Adapt.adapt(to, fluxes.horizontal_flux)
    cm = Adapt.adapt(to, fluxes.cm)
    return FaceIndexedFluxState{B, typeof(hflux), typeof(cm)}(hflux, cm)
end

function Adapt.adapt_structure(to, fluxes::CubedSphereFaceFluxState{B}) where {B <: AbstractMassBasis}
    am = Adapt.adapt(to, fluxes.am)
    bm = Adapt.adapt(to, fluxes.bm)
    cm = Adapt.adapt(to, fluxes.cm)
    return CubedSphereFaceFluxState{B, typeof(am[1]), typeof(bm[1]), typeof(cm[1])}(am, bm, cm)
end

# ---------------------------------------------------------------------------
# Basis accessor
# ---------------------------------------------------------------------------

"""
    mass_basis(state) → AbstractMassBasis

Return the mass flux basis tag for the given flux state.
"""
@inline mass_basis(::AbstractFaceFluxState{B}) where {B <: AbstractMassBasis} = B()

# ---------------------------------------------------------------------------
# Scoped accessor functions
#
# These are available for generic code and validation but the structured
# fast-path kernels bypass them and use am/bm/cm arrays directly.
# Accessors are basis-agnostic — they don't care whether fluxes are moist
# or dry.
# ---------------------------------------------------------------------------

@inline face_flux_x(s::AbstractStructuredFaceFluxState, i, j, k) = s.am[i, j, k]
@inline face_flux_y(s::AbstractStructuredFaceFluxState, i, j, k) = s.bm[i, j, k]
@inline face_flux_z(s::AbstractFaceFluxState, i, j, k)           = s.cm[i, j, k]
@inline face_flux(s::AbstractUnstructuredFaceFluxState, f, k)     = s.horizontal_flux[f, k]
@inline face_flux_x(s::CubedSphereFaceFluxState, panel, i, j, k) = s.am[panel][i, j, k]
@inline face_flux_y(s::CubedSphereFaceFluxState, panel, i, j, k) = s.bm[panel][i, j, k]
@inline face_flux_z(s::CubedSphereFaceFluxState, panel, i, j, k) = s.cm[panel][i, j, k]

# ---------------------------------------------------------------------------
# Allocation helpers
# ---------------------------------------------------------------------------

"""
    allocate_face_fluxes(::StructuredFluxTopology, Nx, Ny, Nz;
                         FT=Float64, ArrayType=Array,
                         basis::Type{<:AbstractMassBasis}=DryBasis)

Allocate zeroed face flux arrays for a structured mesh.
The returned `StructuredFaceFluxState` is tagged with the specified `basis`.
"""
function allocate_face_fluxes(::StructuredFluxTopology,
                              Nx::Int, Ny::Int, Nz::Int;
                              FT::Type{<:AbstractFloat} = Float64,
                              ArrayType = Array,
                              basis::Type{B} = DryBasis) where {B <: AbstractMassBasis}
    am = ArrayType(zeros(FT, Nx + 1, Ny,     Nz))
    bm = ArrayType(zeros(FT, Nx,     Ny + 1, Nz))
    cm = ArrayType(zeros(FT, Nx,     Ny,     Nz + 1))
    return StructuredFaceFluxState{B}(am, bm, cm)
end

"""
    allocate_face_fluxes(::FaceIndexedFluxTopology, nfaces, ncells, Nz;
                         FT=Float64, ArrayType=Array,
                         basis::Type{<:AbstractMassBasis}=DryBasis)

Allocate zeroed face-indexed flux arrays for a connected-face mesh.
"""
function allocate_face_fluxes(::FaceIndexedFluxTopology,
                              nfaces::Int, ncells::Int, Nz::Int;
                              FT::Type{<:AbstractFloat} = Float64,
                              ArrayType = Array,
                              basis::Type{B} = DryBasis) where {B <: AbstractMassBasis}
    hflux = ArrayType(zeros(FT, nfaces, Nz))
    cm = ArrayType(zeros(FT, ncells, Nz + 1))
    return FaceIndexedFluxState{B}(hflux, cm)
end

"""
    allocate_face_fluxes(mesh::AbstractStructuredMesh, Nz; kwargs...)

Allocate a flux container using the mesh's natural structured topology.
"""
function allocate_face_fluxes(mesh::AbstractStructuredMesh, Nz::Int;
                              FT::Type{<:AbstractFloat} = Float64,
                              ArrayType = Array,
                              basis::Type{B} = DryBasis) where {B <: AbstractMassBasis}
    return allocate_face_fluxes(StructuredFluxTopology(), nx(mesh), ny(mesh), Nz;
                                FT=FT, ArrayType=ArrayType, basis=B)
end

function allocate_face_fluxes(mesh::CubedSphereMesh, Nz::Int;
                              FT::Type{<:AbstractFloat} = Float64,
                              ArrayType = Array,
                              basis::Type{B} = DryBasis) where {B <: AbstractMassBasis}
    N = mesh.Nc + 2 * mesh.Hp
    am = ntuple(_ -> ArrayType(zeros(FT, N + 1, N, Nz)), CS_PANEL_COUNT)
    bm = ntuple(_ -> ArrayType(zeros(FT, N, N + 1, Nz)), CS_PANEL_COUNT)
    cm = ntuple(_ -> ArrayType(zeros(FT, N, N, Nz + 1)), CS_PANEL_COUNT)
    return CubedSphereFaceFluxState{B}(am, bm, cm)
end

"""
    allocate_face_fluxes(mesh::AbstractHorizontalMesh, Nz; kwargs...)

Allocate a flux container using the mesh's natural face-connected topology.
"""
function allocate_face_fluxes(mesh::AbstractHorizontalMesh, Nz::Int;
                              FT::Type{<:AbstractFloat} = Float64,
                              ArrayType = Array,
                              basis::Type{B} = DryBasis) where {B <: AbstractMassBasis}
    return allocate_face_fluxes(FaceIndexedFluxTopology(), nfaces(mesh), ncells(mesh), Nz;
                                FT=FT, ArrayType=ArrayType, basis=B)
end

export AbstractFaceFluxState
export AbstractStructuredFaceFluxState, AbstractUnstructuredFaceFluxState
export StructuredFaceFluxState, FaceIndexedFluxState, CubedSphereFaceFluxState
export face_flux_x, face_flux_y, face_flux_z, face_flux
export allocate_face_fluxes
