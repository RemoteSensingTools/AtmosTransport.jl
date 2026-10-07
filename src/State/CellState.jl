# ---------------------------------------------------------------------------
# CellState — prognostic cell-centered fields for transport
#
# Storage contract:
#   Tracers are stored in a SINGLE packed array `tracers_raw`:
#     structured grids  → (Nx, Ny, Nz, Nt)
#     face-indexed grids → (ncells, Nz, Nt)
#   A companion `tracer_names::NTuple{Nt, Symbol}` keeps the names.
#   Property access `state.tracers.CO2` is preserved for readability via
#   a `TracerAccessor` lazy wrapper returned by `getproperty` — it
#   forwards to `get_tracer(state, :CO2)` without allocating.
# ---------------------------------------------------------------------------

function _validate_cell_state_storage(air_mass, tracers_raw, tracer_names)
    ndims(tracers_raw) == ndims(air_mass) + 1 || throw(DimensionMismatch(
        "CellState tracers_raw must have exactly one trailing tracer dimension; " *
        "got ndims(air_mass)=$(ndims(air_mass)), ndims(tracers_raw)=$(ndims(tracers_raw))"))
    size(tracers_raw)[1:ndims(air_mass)] == size(air_mass) || throw(DimensionMismatch(
        "CellState tracer spatial shape $(size(tracers_raw)[1:ndims(air_mass)]) " *
        "must match air_mass shape $(size(air_mass))"))
    size(tracers_raw, ndims(tracers_raw)) == length(tracer_names) ||
        throw(DimensionMismatch(
            "CellState tracer axis has $(size(tracers_raw, ndims(tracers_raw))) entries " *
            "but tracer_names has $(length(tracer_names))"))
    all(name -> name isa Symbol, tracer_names) ||
        throw(ArgumentError("CellState tracer_names must contain only Symbol values"))
    length(unique(tracer_names)) == length(tracer_names) ||
        throw(ArgumentError("CellState tracer_names must be unique; got $(tracer_names)"))
    eltype(tracers_raw) === eltype(air_mass) || throw(ArgumentError(
        "CellState air_mass and tracers_raw must have the same element type; got " *
        "$(eltype(air_mass)) and $(eltype(tracers_raw))"))
    return nothing
end

"""
    CellState{Basis, A, Raw, Names}

Cell-centered prognostic state for transport.

# Fields
- `air_mass :: A` — air mass per cell [kg] on the basis carried by `Basis`.
  Layout matches grid:
  `(Nx, Ny, Nz)` for structured, `(ncells, Nz)` for unstructured.
- `tracers_raw :: Raw` — packed model tracer storage, conventionally
  `dry VMR × dry-air mass` (not physical kg species). Shape is
  `(size(air_mass)..., Nt)`: `(Nx, Ny, Nz, Nt)` on structured grids,
  `(ncells, Nz, Nt)` on face-indexed grids. Kernels dispatch directly
  on this field; non-kernel code uses the accessor API
  (`ntracers`, `get_tracer`, `eachtracer`).
- `tracer_names :: Names` — `NTuple{Nt, Symbol}` of tracer names in
  storage order.

# Property access
`state.tracers` returns a lazy `TracerAccessor` that forwards
`state.tracers.CO2` to `get_tracer(state, :CO2)` (a `selectdim` view
into `tracers_raw`).

# Invariant
After each closed transport step, `sum(air_mass)` and the per-tracer storage
sum `sum(view(tracers_raw, ..., t))` must each be conserved (within
floating-point tolerance).
"""
struct CellState{Basis <: AbstractMassBasis, A <: AbstractArray, Raw <: AbstractArray, Names <: Tuple}
    air_mass     :: A
    tracers_raw  :: Raw
    tracer_names :: Names

    function CellState{Basis, A, Raw, Names}(air_mass::A, tracers_raw::Raw,
                                             tracer_names::Names) where
            {Basis <: AbstractMassBasis, A <: AbstractArray,
             Raw <: AbstractArray, Names <: Tuple}
        _validate_cell_state_storage(air_mass, tracers_raw, tracer_names)
        return new{Basis, A, Raw, Names}(air_mass, tracers_raw, tracer_names)
    end
end

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

function _pack_tracers_raw(m::AbstractArray{FT, N}, tr::NamedTuple) where {FT, N}
    Nt = length(tr)
    raw = similar(m, FT, size(m)..., Nt)
    if Nt == 0
        return raw
    end
    @inbounds for (i, rm) in enumerate(values(tr))
        selectdim(raw, N + 1, i) .= rm
    end
    return raw
end

"""
    CellState(m::AbstractArray; tracers...)

Convenience constructor: `CellState(m; CO2=rm_co2, SF6=rm_sf6)`.
Defaults to `DryBasis`. The input 3D/2D arrays are COPIED into the
packed `tracers_raw` buffer at construction — they are not aliased.
Use `get_tracer(state, :CO2)` (or `state.tracers.CO2`) to read or
mutate the stored tracer after construction.
"""
function CellState(m::AbstractArray; tracers...)
    return CellState(DryBasis, m; tracers...)
end

function CellState(::Type{B}, m::AbstractArray; tracers...) where {B <: AbstractMassBasis}
    tr = NamedTuple(tracers)
    raw = _pack_tracers_raw(m, tr)
    names = keys(tr)  # NTuple{Nt, Symbol}
    return CellState{B, typeof(m), typeof(raw), typeof(names)}(m, raw, names)
end

"""
    CellState(::Type{B}, air_mass, tracers_raw, tracer_names)

Direct construction from already-packed storage. Used by adaptation
and low-level code that has a 4D buffer in hand.
"""
function CellState(::Type{B}, air_mass::AbstractArray,
                   tracers_raw::AbstractArray,
                   tracer_names::Tuple) where {B <: AbstractMassBasis}
    return CellState{B, typeof(air_mass), typeof(tracers_raw), typeof(tracer_names)}(
        air_mass, tracers_raw, tracer_names)
end

mass_basis(::CellState{B}) where {B <: AbstractMassBasis} = B()

function Adapt.adapt_structure(to, state::CellState{B}) where {B <: AbstractMassBasis}
    air_mass    = Adapt.adapt(to, state.air_mass)
    tracers_raw = Adapt.adapt(to, state.tracers_raw)
    names       = state.tracer_names
    return CellState{B, typeof(air_mass), typeof(tracers_raw), typeof(names)}(
        air_mass, tracers_raw, names)
end

# ---------------------------------------------------------------------------
# TracerAccessor — lazy wrapper preserving state.tracers.CO2 syntax
# ---------------------------------------------------------------------------

"""
    TracerAccessor{S}

Lazy, allocation-free wrapper that forwards property-style tracer
access (`state.tracers.CO2`) to `get_tracer(state, :CO2)`. Returned by
`getproperty(::CellState, :tracers)`. Reading a tracer that does not
exist throws `KeyError`.

`state.tracers[:CO2]` is also supported and forwards to
`get_tracer(state, :CO2)` for code that prefers index syntax.
"""
struct TracerAccessor{S}
    state::S
end

Base.@propagate_inbounds function Base.getproperty(acc::TracerAccessor, name::Symbol)
    state = getfield(acc, :state)
    return get_tracer(state, name)
end

Base.@propagate_inbounds Base.getindex(acc::TracerAccessor, name::Symbol) =
    get_tracer(getfield(acc, :state), name)

Base.propertynames(acc::TracerAccessor) = getfield(acc, :state).tracer_names
Base.length(acc::TracerAccessor) = ntracers(getfield(acc, :state))
Base.keys(acc::TracerAccessor)   = getfield(acc, :state).tracer_names

# ---------------------------------------------------------------------------
# Lazy property-style tracer access
# ---------------------------------------------------------------------------

function Base.getproperty(state::CellState, name::Symbol)
    if name === :tracers
        return TracerAccessor(state)
    else
        return getfield(state, name)
    end
end

function Base.propertynames(::CellState, private::Bool = false)
    names = (:air_mass, :tracers_raw, :tracer_names, :tracers)
    return private ? names : names
end

# ---------------------------------------------------------------------------
# Convenience diagnostics
# ---------------------------------------------------------------------------

"""
    mixing_ratio(state::CellState, name::Symbol)

Compute mixing ratio `q = tracer_storage / air_mass` for the named tracer.
"""
mixing_ratio(state::CellState, name::Symbol) =
    get_tracer(state, name) ./ state.air_mass

"""
    total_mass(state::CellState, name::Symbol) -> Float64

Sum the named tracer's conservative model storage across all cells and levels
(in Float64, whatever the storage precision).
For dry VMR tracers this is `sum(χ_dry × dry_air_mass)`, not physical kg
species unless an explicit molecular-weight conversion is applied.
"""
total_mass(state::CellState, name::Symbol) = _compensated_total(get_tracer(state, name))

"""
    total_air_mass(state::CellState) -> Float64

Sum of air mass across all cells and levels (in Float64).
"""
total_air_mass(state::CellState) = _compensated_total(state.air_mass)

"""
    tracer_names(state::CellState) -> NTuple{Nt, Symbol}

Names of all tracers in `state`, in stored order.
"""
tracer_names(state::CellState) = getfield(state, :tracer_names)

export CellState, TracerAccessor
export mixing_ratio, total_mass, total_air_mass, tracer_names
