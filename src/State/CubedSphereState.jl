# ---------------------------------------------------------------------------
# CubedSphereState — panel-native prognostic state for cubed-sphere runtime
# ---------------------------------------------------------------------------

function _validate_cs_state_storage(air_mass, tracers_raw, tracer_names, halo_width)
    halo_width >= 0 || throw(ArgumentError(
        "CubedSphereState halo_width must be non-negative, got $(halo_width)"))
    reference_shape = size(air_mass[1])
    length(reference_shape) == 3 || throw(DimensionMismatch(
        "CubedSphereState air-mass panels must be rank 3, got shape $(reference_shape)"))
    reference_shape[1] > 2 * halo_width && reference_shape[2] > 2 * halo_width ||
        throw(DimensionMismatch(
            "CubedSphereState halo_width=$(halo_width) leaves no physical panel " *
            "interior for shape $(reference_shape)"))
    all(panel -> size(panel) == reference_shape, air_mass) ||
        throw(DimensionMismatch("CubedSphereState air-mass panels must have identical shapes"))
    all(panel -> ndims(panel) == 4, tracers_raw) || throw(DimensionMismatch(
        "CubedSphereState tracer panels must be rank 4"))
    all(panel -> size(panel)[1:3] == reference_shape, tracers_raw) ||
        throw(DimensionMismatch(
            "CubedSphereState tracer panel spatial shapes must match air-mass panels"))
    all(panel -> size(panel, 4) == length(tracer_names), tracers_raw) ||
        throw(DimensionMismatch(
            "CubedSphereState tracer axes must match $(length(tracer_names)) tracer names"))
    all(name -> name isa Symbol, tracer_names) || throw(ArgumentError(
        "CubedSphereState tracer_names must contain only Symbol values"))
    length(unique(tracer_names)) == length(tracer_names) || throw(ArgumentError(
        "CubedSphereState tracer_names must be unique; got $(tracer_names)"))
    all(panel -> eltype(panel) === eltype(air_mass[1]), air_mass) ||
        throw(ArgumentError("CubedSphereState air-mass panels must share one element type"))
    all(panel -> eltype(panel) === eltype(air_mass[1]), tracers_raw) ||
        throw(ArgumentError(
            "CubedSphereState air-mass and tracer panels must share one element type"))
    return nothing
end

"""
    CubedSphereState{Basis, A3, Raw4, Names}
    CubedSphereState(Basis, mesh, air_mass; tracers...)
    CubedSphereState(Basis, air_mass; halo_width, tracers...)

Panel-native prognostic state. Air mass is stored as six halo-padded
three-dimensional panels. Tracers are packed into one trailing tracer axis on
each panel, with names retained in storage order. Diagnostics and global sums
exclude halos.

# Fields
- `air_mass :: NTuple{6, A3}` — halo-padded panel air mass arrays with shape
  `(Nc + 2Hp, Nc + 2Hp, Nz)`.
- `tracers_raw :: NTuple{6, Raw4}` — packed per-panel model tracer storage
  (conventionally `dry VMR × dry-air mass`, not physical kg species) with shape
  `(Nc + 2Hp, Nc + 2Hp, Nz, Nt)`.
- `tracer_names :: Names` — names of the tracer axis in storage order.
- `halo_width :: Int` — halo width `Hp` needed to recover the physical interior.
"""
struct CubedSphereState{Basis <: AbstractMassBasis,
                        A3 <: AbstractArray,
                        Raw4 <: AbstractArray,
                        Names <: Tuple}
    air_mass     :: NTuple{6, A3}
    tracers_raw  :: NTuple{6, Raw4}
    tracer_names :: Names
    halo_width   :: Int

    function CubedSphereState{Basis, A3, Raw4, Names}(
            air_mass::NTuple{6, A3}, tracers_raw::NTuple{6, Raw4},
            tracer_names::Names, halo_width::Int) where
            {Basis <: AbstractMassBasis, A3 <: AbstractArray,
             Raw4 <: AbstractArray, Names <: Tuple}
        _validate_cs_state_storage(air_mass, tracers_raw, tracer_names, halo_width)
        return new{Basis, A3, Raw4, Names}(
            air_mass, tracers_raw, tracer_names, halo_width)
    end
end

@inline halo_width(state::CubedSphereState) = state.halo_width

@inline function _pack_cs_tracers_raw(panels_m::NTuple{6, A3}, tr::NamedTuple) where {A3 <: AbstractArray}
    Nt = length(tr)
    return ntuple(6) do p
        panel_m = panels_m[p]
        raw = similar(panel_m, eltype(panel_m), size(panel_m)..., Nt)
        if Nt == 0
            return raw
        end
        @inbounds for (i, tracer_panels) in enumerate(values(tr))
            selectdim(raw, ndims(raw), i) .= tracer_panels[p]
        end
        raw
    end
end

function CubedSphereState(::Type{B}, mesh::CubedSphereMesh,
                          air_mass::NTuple{6}; tracers...) where {B <: AbstractMassBasis}
    return CubedSphereState(B, air_mass; halo_width = mesh.Hp, tracers...)
end

function CubedSphereState(::Type{B},
                          air_mass::NTuple{6};
                          halo_width::Integer,
                          tracers...) where {B <: AbstractMassBasis}
    tr = NamedTuple(tracers)
    raw = _pack_cs_tracers_raw(air_mass, tr)
    names = keys(tr)
    return CubedSphereState{B, typeof(air_mass[1]), typeof(raw[1]), typeof(names)}(
        air_mass, raw, names, Int(halo_width))
end

function CubedSphereState(::Type{B},
                          air_mass::NTuple{6},
                          tracers_raw::NTuple{6},
                          tracer_names::Tuple;
                          halo_width::Integer) where {B <: AbstractMassBasis}
    return CubedSphereState{B, typeof(air_mass[1]), typeof(tracers_raw[1]), typeof(tracer_names)}(
        air_mass, tracers_raw, tracer_names, Int(halo_width))
end

mass_basis(::CubedSphereState{B}) where {B <: AbstractMassBasis} = B()

function Adapt.adapt_structure(to, state::CubedSphereState{B}) where {B <: AbstractMassBasis}
    air_mass = Adapt.adapt(to, state.air_mass)
    tracers_raw = Adapt.adapt(to, state.tracers_raw)
    names = state.tracer_names
    return CubedSphereState{B, typeof(air_mass[1]), typeof(tracers_raw[1]), typeof(names)}(
        air_mass, tracers_raw, names, state.halo_width)
end

function Base.getproperty(state::CubedSphereState, name::Symbol)
    if name === :tracers
        return TracerAccessor(state)
    else
        return getfield(state, name)
    end
end

function Base.propertynames(::CubedSphereState, private::Bool = false)
    names = (:air_mass, :tracers_raw, :tracer_names,
             :tracers, :halo_width)
    return private ? names : names
end

@inline function _panel_interior(panel::AbstractArray, Hp::Int)
    nx_panel = size(panel, 1)
    ny_panel = size(panel, 2)
    return @view panel[Hp + 1:nx_panel - Hp, Hp + 1:ny_panel - Hp, :]
end

@inline function _panel_interior(panel::AbstractArray, Hp::Int, tracer_idx::Int)
    nx_panel = size(panel, 1)
    ny_panel = size(panel, 2)
    return @view panel[Hp + 1:nx_panel - Hp, Hp + 1:ny_panel - Hp, :, tracer_idx]
end

mixing_ratio(state::CubedSphereState, name::Symbol) =
    ntuple(6) do p
        get_tracer(state, name)[p] ./ state.air_mass[p]
    end

function total_mass(state::CubedSphereState, name::Symbol)
    idx = tracer_index(state, name)
    idx === nothing && throw(KeyError(name))
    Hp = halo_width(state)
    return _compensated_total(ntuple(p -> _panel_interior(state.tracers_raw[p], Hp, idx), 6))
end

total_air_mass(state::CubedSphereState) =
    _compensated_total(ntuple(p -> _panel_interior(state.air_mass[p], halo_width(state)), 6))

tracer_names(state::CubedSphereState) = getfield(state, :tracer_names)

export CubedSphereState
export halo_width
