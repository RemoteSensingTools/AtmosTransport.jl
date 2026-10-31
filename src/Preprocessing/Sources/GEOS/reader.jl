# ---------------------------------------------------------------------------
# GEOSNativeReader — concrete reader for GEOS-IT / GEOS-FP native NetCDF.
#
# Wraps the existing `(GEOSSettings, GEOSDayHandles)` pair plus a typed
# chained-mass slot. `read_window!` delegates to the existing
# `read_window!(::RawWindow, ::GEOSSettings, ::GEOSDayHandles, date, w)`
# so bit-exact behavior with today's `cubed_sphere_geos.jl::process_day`
# is preserved.
# ---------------------------------------------------------------------------

"""
    GEOSNativeReader{FT, S, CP, V, H} <: AbstractMetReader{FT, S, CP}

Typed reader wrapping `(GEOSSettings, handles)` for one day, plus
optional chained-mass state. Type parameters:

  - `FT` — preprocessing float type.
  - `S <: AbstractGEOSSettings` — concrete settings (GEOSITSettings /
    GEOSFPSettings / …).
  - `CP <: AbstractChainPolicy` — `NoChain` or `ChainedMass{T}`.
  - `V` — seed-field slot type. `Nothing` for `NoChain`;
    `Union{Nothing, NTuple{6, Array{FT, 3}}}` for `ChainedMass`. The V
    parameter is fixed at construction (Julia-style review round-1:
    value-dependent V was the type-instability foot-gun; we pin V
    to the unconditional Union on the chained path so the inner
    constructor specializes once).
  - `H` — concrete handles type. Pinned to `typeof(open_day(...))` at
    construction so `reader.handles` access is type-stable (replaces
    the previous `:: Any` field). Different GEOS flavors produce
    different handle structs (`GEOSDayHandles` for GEOS-IT,
    `GEOSFPNativeDayHandles` for GEOS-FP); each becomes its own
    concrete reader type via this parameter.

Constructor: `open_reader(settings::AbstractGEOSSettings, date, FT;
seed, next_day_handle)`. See the function docstring.
"""
mutable struct GEOSNativeReader{FT, S <: AbstractGEOSSettings,
                                  CP <: AbstractChainPolicy, V, H} <:
                AbstractMetReader{FT, S, CP}
    settings :: S
    handles  :: H           # concrete handle struct, typed via the H parameter
    date     :: Date
    seed     :: V           # Nothing (NoChain) or Union{Nothing, NTuple{6,Array{FT,3}}}
end

function open_reader(settings::AbstractGEOSSettings, date::Date, ::Type{FT};
                     seed = nothing,
                     next_day_handle::Bool = true,
                     chain_mass::Bool = true,
                     adjacent_omega::Bool = false) where {FT <: AbstractFloat}
    # `adjacent_omega` opens the prev/next-day A3dyn+I3 handles the
    # `:omega_consistent` cm closure needs to interpolate across midnight; every
    # other closure leaves them `nothing` (no extra file opens on the default path).
    handles = open_day(settings, date; next_day_handle = next_day_handle,
                       adjacent_omega = adjacent_omega)
    H = typeof(handles)
    # `chain_mass = false` opts out of cross-day carry entirely (NoChain).
    # Otherwise the policy is `ChainedMass{NTuple{6, Array{FT,3}}}`.
    if !chain_mass
        return GEOSNativeReader{FT, typeof(settings), NoChain, Nothing, H}(
            settings, handles, date, nothing)
    end
    # ChainedMass with NTuple{6, Array{FT,3}} seed shape. Julia-style
    # review round-1: V was previously value-dependent
    # (`V = typeof(seed_value)` vs `Union{Nothing, NTuple{...}}` when
    # `seed === nothing`), which made the compiler unable to specialize
    # the constructor. Pin V unconditionally to the Union so the
    # specialization happens once per (FT, S) pair.
    V = Union{Nothing, NTuple{6, Array{FT, 3}}}
    seed isa V ||
        throw(ArgumentError("GEOSNativeReader: seed must be `nothing` or " *
                             "`NTuple{6, Array{$FT, 3}}`; got $(typeof(seed))."))
    CP = ChainedMass{NTuple{6, Array{FT, 3}}}
    return GEOSNativeReader{FT, typeof(settings), CP, V, H}(
        settings, handles, date, seed)
end

@inline windows_per_day(reader::GEOSNativeReader) =
    windows_per_day(reader.settings, reader.date)

@inline function read_window!(raw::RawWindow{FT}, reader::GEOSNativeReader{FT},
                                w::Int) where FT
    return read_window!(raw, reader.settings, reader.handles, reader.date, w)
end

@inline function close_reader!(reader::GEOSNativeReader)
    close_day!(reader.handles)
    return nothing
end
