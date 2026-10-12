"""
    PerTracerFluxMap{S <: Tuple}

An ordered tuple of `SurfaceFluxSource`s, keyed by the `tracer_name`
field on each entry. Supplies the surface flux operator
with per-tracer surface-rate data and ensures efficient tuple-splatting
at kernel-launch time.

The map is **NTuple-backed** rather than Dict-backed. Rationale:

- Runtime source construction naturally produces a small tuple; the map keeps
  that storage and adds a lookup helper.
- Tuples are bits-stable on GPU (captured by kernels without
  boxing / hashing); Dicts would require special Adapt handling and
  per-launch lookup cost.
- Small-N (typically 1-5 emitting tracers) makes linear scan cheaper
  than hashing anyway.

Tracers absent from the map have **zero surface flux**. The consumer
operator iterates the map and applies each source; it does NOT walk
the state's full `tracer_names` and look up each by name.

# Construction

```julia
co2_rate  = fill(2.0e-7, Nx, Ny)    # model-storage units/s per cell
sf6_rate  = fill(1.5e-9, Nx, Ny)    # already converted from physical flux
rn222_rate = fill(3.0e-11, Nx, Ny)

map = PerTracerFluxMap(
    SurfaceFluxSource(:CO2,   co2_rate),
    SurfaceFluxSource(:SF6,   sf6_rate),
    SurfaceFluxSource(:Rn222, rn222_rate),
)

length(map)                 # 3
flux_for(map, :CO2) === ... # the :CO2 SurfaceFluxSource
flux_for(map, :CH4)         # nothing (not emitting)
```

# Adapt / GPU

`Adapt.adapt(CuArray, map)` adapts each source through its
`adapt_structure`; tuples adapt element-wise. A `SurfaceFluxSource` moves its
`cell_mass_rate` to the device. A `TimeVaryingSurfaceFluxSource` moves only its
compensation and its small window buffer; its rate series stays in host memory.

# Fields
- `sources :: S` — `NTuple{N, <:AbstractSurfaceFluxSource}` for some
  `N`. Each entry carries a tracer name and its rate data (a static
  array for `SurfaceFluxSource`, a time series for
  `TimeVaryingSurfaceFluxSource`).
"""
struct PerTracerFluxMap{S <: Tuple}
    sources :: S

    function PerTracerFluxMap(sources::S) where {S <: Tuple}
        # Validate that every entry is an AbstractSurfaceFluxSource.
        for (k, src) in enumerate(sources)
            src isa AbstractSurfaceFluxSource ||
                throw(ArgumentError("PerTracerFluxMap entry $k is a $(typeof(src)), expected AbstractSurfaceFluxSource"))
        end
        # Validate tracer names are unique — duplicate entries would
        # double-apply the emission for the same tracer.
        names = map(s -> s.tracer_name, sources)
        length(unique(names)) == length(names) ||
            throw(ArgumentError("PerTracerFluxMap: duplicate tracer names $(names)"))
        new{S}(sources)
    end
end

"""
    PerTracerFluxMap(sources::AbstractSurfaceFluxSource...)

Variadic constructor: `PerTracerFluxMap(src1, src2, src3)` wraps the
three sources into an NTuple-backed map.
"""
PerTracerFluxMap(sources::AbstractSurfaceFluxSource...) = PerTracerFluxMap(sources)

"""
    PerTracerFluxMap(sources::AbstractVector{<:AbstractSurfaceFluxSource})
    PerTracerFluxMap(sources::Tuple)

Generic collection constructor. The input is frozen into an NTuple.
"""
PerTracerFluxMap(sources::AbstractVector{<:AbstractSurfaceFluxSource}) =
    PerTracerFluxMap(Tuple(sources))

"""
    flux_for(map, tracer_name::Symbol) -> SurfaceFluxSource | nothing

Return the `SurfaceFluxSource` for the named tracer, or `nothing` if the
tracer has no surface source in this map. O(N) linear scan, which is
fine for typical N ≤ 10.
"""
@inline function flux_for(map::PerTracerFluxMap, tracer_name::Symbol)
    for src in map.sources
        src.tracer_name === tracer_name && return src
    end
    return nothing
end

# =========================================================================
# Iteration + collection interface
# =========================================================================

Base.length(m::PerTracerFluxMap) = length(m.sources)
Base.eltype(::Type{PerTracerFluxMap{S}}) where {S} = eltype(S)
Base.iterate(m::PerTracerFluxMap, args...) = iterate(m.sources, args...)
Base.firstindex(m::PerTracerFluxMap) = firstindex(m.sources)
Base.lastindex(m::PerTracerFluxMap) = lastindex(m.sources)
Base.getindex(m::PerTracerFluxMap, i) = m.sources[i]

"""
    tracer_names(map::PerTracerFluxMap) -> NTuple{N, Symbol}

Ordered tuple of tracer names in the map, matching the storage order of
`map.sources`.
"""
tracer_names(m::PerTracerFluxMap) = map(s -> s.tracer_name, m.sources)

# =========================================================================
# Adapt — GPU dispatch
# =========================================================================

# Tuples adapt element-wise; each source's adapt_structure decides what moves
# to the device (see the docstring above). The map stays a thin wrapper.
Adapt.adapt_structure(to, m::PerTracerFluxMap) =
    PerTracerFluxMap(map(s -> Adapt.adapt(to, s), m.sources))
