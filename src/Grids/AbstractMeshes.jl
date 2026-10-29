# ---------------------------------------------------------------------------
# Abstract mesh hierarchy for the dry-mass transport architecture
#
# Three orthogonal concepts:
#   1. Horizontal mesh   — cell/face connectivity and metrics on the sphere
#   2. Vertical coordinate — hybrid pressure levels
#   3. AtmosGrid          — composition of (1) + (2) + architecture
#
# Structured grids (LatLon, CubedSphere) provide fast indexed access under
# the hood, but the abstract interface is face/cell oriented so that
# unstructured meshes (ReducedGaussian) fit the same contracts.
# ---------------------------------------------------------------------------

"""
    AbstractHorizontalMesh

Supertype for all horizontal mesh topologies.

## Required methods

Any subtype must implement the geometry API defined in `GeometryOps.jl`:
- `ncells(mesh)`
- `nfaces(mesh)`
- `cell_area(mesh, c)`
- `face_length(mesh, f)`
- `face_normal(mesh, f)`
- `face_cells(mesh, f) -> (left, right)`
- `cell_faces(mesh, c) -> indices`

Structured meshes additionally provide:
- `nx(mesh)`, `ny(mesh)` — logical grid dimensions
"""
abstract type AbstractHorizontalMesh end

"""
    AbstractStructuredMesh <: AbstractHorizontalMesh

Meshes with a logically rectangular (i, j) structure per region.
Both `LatLonMesh` and `CubedSphereMesh` are structured.
Structured meshes expose `nx(mesh)`, `ny(mesh)` and support
direct `(i, j)` indexing of areas and face fluxes.
"""
abstract type AbstractStructuredMesh <: AbstractHorizontalMesh end

# ---------------------------------------------------------------------------
# Flux topology trait — describes the natural face-flux storage for a mesh
# ---------------------------------------------------------------------------

"""
    AbstractFluxTopology

Supertype for flux topology traits. Each mesh advertises its natural
flux representation via `flux_topology(mesh)`.

- `StructuredFluxTopology` — directional (x, y, z) face fluxes on a
  logically rectangular mesh.
- `FaceIndexedFluxTopology` — face-indexed connectivity; the natural
  representation for reduced Gaussian and other unstructured meshes.
"""
abstract type AbstractFluxTopology end

"""
    StructuredFluxTopology <: AbstractFluxTopology

Fluxes stored as separate directional arrays (am, bm, cm)
on a logically rectangular grid.
"""
struct StructuredFluxTopology <: AbstractFluxTopology end

"""
    FaceIndexedFluxTopology <: AbstractFluxTopology

Fluxes stored as a single face-indexed array with explicit
connectivity. Natural for reduced Gaussian and other unstructured meshes.
"""
struct FaceIndexedFluxTopology <: AbstractFluxTopology end

flux_topology(::AbstractStructuredMesh) = StructuredFluxTopology()
flux_topology(::AbstractHorizontalMesh) = FaceIndexedFluxTopology()


"""
    AbstractVerticalCoordinate{FT}

Supertype for vertical coordinate systems.

## Required methods
- `n_levels(vc)` — number of vertical levels
- `pressure_at_level(vc, k, p_surface)` — center pressure
- `pressure_at_interface(vc, k, p_surface)` — interface pressure
- `level_thickness(vc, k, p_surface)` — layer thickness [Pa]
"""
abstract type AbstractVerticalCoordinate{FT} end

"""
    AtmosGrid{H, V, Arch, FT}

Composite grid: horizontal mesh + vertical coordinate + architecture.

This is the single grid object passed to all transport operators. Dispatch
on `H` selects mesh-specific geometry; dispatch on `Arch` selects backend.

# Fields
- `horizontal :: H` — horizontal mesh (LatLonMesh, CubedSphereMesh, ReducedGaussianMesh)
- `vertical   :: V` — vertical coordinate (HybridSigmaPressure)
- `arch       :: Arch` — compute architecture (CPU, GPU)
- `planet     :: PlanetParameters{FT}` — physical constants [m, m/s², Pa]
"""
struct AtmosGrid{H <: AbstractHorizontalMesh,
                 V <: AbstractVerticalCoordinate,
                 Arch,
                 P,
                 FT <: AbstractFloat} 
    horizontal         :: H
    vertical           :: V
    arch               :: Arch
    planet             :: P
end

@inline _vertical_floattype(::AbstractVerticalCoordinate{FT}) where FT = FT
@inline _planet_floattype(::PlanetParameters{FT}) where FT = FT
@inline _float_typeof_number(x::Real) = typeof(float(x))

function AtmosGrid(horizontal::H, vertical::V, arch::Arch;
                   FT::Union{Nothing, Type{<:AbstractFloat}} = nothing,
                   planet::PlanetParameters{<:AbstractFloat} = earth_parameters(),
                   radius = planet.radius,
                   gravity = planet.gravity,
                   reference_pressure = planet.reference_pressure) where {H, V, Arch}
    FT_out = isnothing(FT) ?
        promote_type(eltype(horizontal),
                     _vertical_floattype(vertical),
                     _planet_floattype(planet),
                     _float_typeof_number(radius),
                     _float_typeof_number(gravity),
                     _float_typeof_number(reference_pressure)) :
        FT
    planet_ft = PlanetParameters(; FT=FT_out,
                                 radius=radius,
                                 gravity=gravity,
                                 reference_pressure=reference_pressure)
    if hasproperty(horizontal, :radius)
        mesh_radius = FT_out(getproperty(horizontal, :radius))
        mesh_radius == planet_ft.radius ||
            throw(ArgumentError(
                "horizontal mesh radius $(mesh_radius) does not match AtmosGrid planet radius $(planet_ft.radius); " *
                "construct the mesh and PlanetParameters with the same radius"))
    end
    return AtmosGrid{H, V, Arch, typeof(planet_ft), FT_out}(horizontal, vertical, arch, planet_ft)
end

floattype(::AtmosGrid{H, V, Arch, P, FT}) where {H, V, Arch, P, FT} = FT
nlevels(g::AtmosGrid) = n_levels(g.vertical)

radius(g::AtmosGrid) = g.planet.radius
gravity(g::AtmosGrid) = g.planet.gravity
reference_pressure(g::AtmosGrid) = g.planet.reference_pressure

function Base.getproperty(g::AtmosGrid, s::Symbol)
    if s === :radius
        return getfield(getfield(g, :planet), :radius)
    elseif s === :gravity
        return getfield(getfield(g, :planet), :gravity)
    elseif s === :reference_pressure
        return getfield(getfield(g, :planet), :reference_pressure)
    else
        return getfield(g, s)
    end
end

function Base.propertynames(::AtmosGrid, private::Bool = false)
    names = (:horizontal, :vertical, :arch, :planet,
             :radius, :gravity, :reference_pressure)
    return names
end

export AbstractHorizontalMesh, AbstractStructuredMesh
export AbstractFluxTopology, StructuredFluxTopology, FaceIndexedFluxTopology
export flux_topology
export AbstractVerticalCoordinate
export AtmosGrid, floattype, nlevels  # `architecture` re-exported via Architectures
export radius, gravity, reference_pressure
