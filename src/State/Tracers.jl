# ---------------------------------------------------------------------------
# Tracer utilities and storage-agnostic accessor API
#
# The accessor functions below are the PRIMARY interface for non-kernel
# code that needs to touch `state.tracers`. They read directly from
# `state.tracers_raw` + `state.tracer_names` (see CellState.jl).
#
# Kernels (multi-tracer sweeps, vertical remap) bypass this API and
# dispatch on raw storage directly — that is intentional and fast.
# ---------------------------------------------------------------------------

"""
    allocate_tracers(names::NTuple{N, Symbol}, Nx, Ny, Nz;
                     FT=Float64, ArrayType=Array, fill_value=zero(FT))

Allocate a NamedTuple of 3D model tracer-storage arrays. Still useful for test
fixtures that want to pre-build per-tracer 3D arrays for a
`CellState(m; tracers...)` keyword-form constructor call.
"""
function allocate_tracers(names::NTuple{N, Symbol}, Nx::Int, Ny::Int, Nz::Int;
                          FT::Type{<:AbstractFloat} = Float64,
                          ArrayType = Array,
                          fill_value = zero(FT)) where N
    arrays = ntuple(N) do _
        arr = ArrayType(zeros(FT, Nx, Ny, Nz))
        fill_value != zero(FT) && fill!(arr, fill_value)
        arr
    end
    return NamedTuple{names}(arrays)
end

# =========================================================================
# Accessor API
# =========================================================================

"""
    ntracers(state::CellState) -> Int

Number of tracers carried by `state`.
"""
ntracers(state::AbstractCellState) = length(state.tracer_names)

"""
    tracer_index(state::AbstractCellState, name::Symbol) -> Union{Int, Nothing}

Index of the tracer named `name` in `state.tracer_names`, or `nothing`
if absent.
"""
function tracer_index(state::AbstractCellState, name::Symbol)
    return findfirst(==(name), state.tracer_names)
end


"""
    tracer_name(state::CellState, idx::Integer) -> Symbol

Name of the tracer at index `idx`. Throws `BoundsError` if out of
range.
"""
tracer_name(state::AbstractCellState, idx::Integer) = state.tracer_names[Int(idx)]

"""
    get_tracer(state::AbstractCellState, name::Symbol)
    get_tracer(state::CellState, idx::Integer)

Return a view of the model tracer-storage slice. For a structured grid with
`state.tracers_raw :: Array{FT, 4}`, this is
`selectdim(state.tracers_raw, 4, idx)`, a contiguous
`SubArray{FT, 3}` (because Julia is column-major and the tracer axis
is the slowest-varying). Mutations through the returned view are
reflected in `state.tracers_raw`.

Throws `KeyError(name)` if `name` is not a tracer in `state`.
"""
function get_tracer(state::AbstractCellState, name::Symbol)
    idx = tracer_index(state, name)
    idx === nothing && throw(KeyError(name))
    return get_tracer(state, idx)
end


function get_tracer(state::CellState, idx::Integer)
    raw = state.tracers_raw
    return selectdim(raw, ndims(raw), Int(idx))
end

function get_tracer(state::CubedSphereState, idx::Integer)
    tracer_idx = Int(idx)
    return ntuple(CS_PANEL_COUNT) do p
        raw = state.tracers_raw[p]
        selectdim(raw, ndims(raw), tracer_idx)
    end
end

"""
    eachtracer(state::AbstractCellState)

Iterate `name => tracer_slice` pairs for every tracer in `state`, in
storage order. The yielded shape matches the previous
`pairs(::NamedTuple)` contract so callers that destructure
`for (name, rm) in eachtracer(state)` continue to work.
"""
function eachtracer(state::AbstractCellState)
    return (n => get_tracer(state, i) for (i, n) in enumerate(state.tracer_names))
end


# =========================================================================
# Mutating utilities
# =========================================================================

"""
    set_uniform_mixing_ratio!(state::CellState, name::Symbol, χ)

Set tracer `name` to uniform mixing ratio χ: tracer_mass = χ × air_mass.
"""
function set_uniform_mixing_ratio!(state::CellState, name::Symbol, χ)
    rm = get_tracer(state, name)
    rm .= χ .* state.air_mass
    return nothing
end

function set_uniform_mixing_ratio!(state::CubedSphereState, name::Symbol, χ)
    rm_panels = get_tracer(state, name)
    @inbounds for p in 1:CS_PANEL_COUNT
        rm_panels[p] .= χ .* state.air_mass[p]
    end
    return nothing
end

export allocate_tracers, set_uniform_mixing_ratio!
export ntracers, tracer_index, tracer_name, get_tracer, eachtracer

mass_basis(::AbstractCellState{B}) where {B <: AbstractMassBasis} = B()
"""
    tracer_names(state::AbstractCellState) -> NTuple{Nt, Symbol}

Names of all tracers in `state`, in stored order.
"""
tracer_names(state::AbstractCellState) = getfield(state, :tracer_names)
