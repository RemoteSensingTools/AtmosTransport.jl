"""Type tag for cross-day seed policy."""
abstract type AbstractChainPolicy end

"""
    NoChain <: AbstractChainPolicy

The reader does not carry mass state across days. Day N starts from the
raw source endpoint. The seed slot is `nothing`.
"""
struct NoChain <: AbstractChainPolicy end

"""
    ChainedMass{T} <: AbstractChainPolicy

The reader carries an end-of-day mass field of shape `T` into the next
day's open. The shape `T` is the seed array type (e.g.
`NTuple{6, Array{Float32, 3}}`); the actual seed VALUE lives in the
reader struct, but its STATIC type is encoded here so seed access has a known type.
"""
struct ChainedMass{T} <: AbstractChainPolicy end

# ---------------------------------------------------------------------------
# Abstract reader surface.
# ---------------------------------------------------------------------------

"""
    AbstractMetReader{FT, S, CP}

Typed met-reader nominal. Bundles a met-source's settings, per-day handle
context, and chained-mass-policy carry into one struct that the unified
preprocessor driver dispatches on. Type parameters:

- `FT <: AbstractFloat` — preprocessing float type (`Float32` for GPU,
  `Float64` for diagnostic runs).
- `S <: AbstractMetSettings` — concrete settings type (`GEOSITSettings`,
  `GEOSFPSettings`, `ERA5SpectralSettings`, …).
- `CP <: AbstractChainPolicy` — `NoChain` or `ChainedMass{T}` for some
  seed array type `T`.

Concrete readers implement open, close, and per-window access.
"""
abstract type AbstractMetReader{FT <: AbstractFloat,
                                  S <: AbstractMetSettings,
                                  CP <: AbstractChainPolicy} end

"""
    open_reader(settings::AbstractMetSettings, date::Date, ::Type{FT};
                seed = nothing, next_day_handle::Bool = true)
        → reader::AbstractMetReader{FT, typeof(settings), CP}

Construct a typed met-reader for one calendar day. The reader owns the
underlying day-handle context (file handles, vertical coefficients,
chained-mass seed) and exposes the per-window trait surface.

`seed` carries cross-day mass state for chained-mass readers; pass the
previous day workflow’s `final_m` result. For `NoChain`
readers, `seed` must be `nothing`.

`next_day_handle` controls whether the reader opens a handle into the
next day's first-hour instantaneous file. Required when the day produces
endpoint mass for the last window (the standard case for hourly sources).

Dispatch by `settings` type to select the concrete reader; concrete
readers register their own method.
"""
function open_reader end

"""
    close_reader!(reader::AbstractMetReader) → nothing

Close all per-day file handles and release scratch held by the reader.
Idempotent — safe to call from a `finally` block. Concrete readers
implement.
"""
function close_reader! end

# `windows_per_day` and `read_window!` are already declared as generic
# functions in `met_sources.jl` (the `AbstractMetSettings` surface). This
# file adds reader-typed methods on them; redeclaring the generics here
# would warn-replace the existing docstrings.

# Source/target support matrix for the TOML entrypoint. Methods live here
# because all participating types (`ERA5SpectralSettings`,
# `AbstractGEOSSettings`, and the target geometries) are visible by the time
# this file is included.
preprocessor_pair_supported(::LatLonTargetGeometry, ::ERA5SpectralSettings) = true
preprocessor_pair_supported(::ReducedGaussianTargetGeometry, ::ERA5SpectralSettings) = true
preprocessor_pair_supported(::CubedSphereTargetGeometry, ::ERA5SpectralSettings) = true
preprocessor_pair_supported(::CubedSphereTargetGeometry, ::AbstractGEOSSettings) = true
preprocessor_pair_supported(::CubedSphereTargetGeometry, ::AbstractERA5GRIBSettings) = true

"""
    supports_day_threading(settings) -> Bool

True when `process_day(date, grid, settings, ...)` is genuinely day-local and
the unified driver may launch `Threads.@threads` over a date range. Sources
that maintain cross-day state (e.g. GEOS pressure-flux endpoint chaining)
must keep the serial default. The driver still respects `chain_mass=true` —
threading is enabled only when this trait AND `!chain_mass` both hold.
"""
supports_day_threading(::AbstractMetSettings) = false
supports_day_threading(::AbstractERA5GRIBSettings) = true

function day_threading_enabled(settings, ndays::Integer, chain_mass::Bool,
                                global_mass_pin::Bool,
                                configured_global_mass_target_kg::Real)
    return Threads.nthreads() > 1 && ndays > 1 &&
           supports_day_threading(settings) && !chain_mass &&
           !(global_mass_pin && !isfinite(configured_global_mass_target_kg))
end

function require_configured_terminal_day(date::Date,
                                          configured_last_day::Union{Nothing, Date},
                                          source::AbstractString,
                                          missing_endpoint::AbstractString)
    configured_last_day === nothing && error(
        "$source cannot use a zero-tendency endpoint on $date because no configured " *
        "last day was supplied; missing $missing_endpoint")
    date == configured_last_day || error(
        "$source is missing $missing_endpoint for $date, but the configured last day " *
        "is $configured_last_day; refusing a zero-tendency endpoint before the " *
        "configured archive boundary")
    return nothing
end

"""
    supports_initial_endpoint_mass_pin(settings) -> Bool

True when the source's `process_day` implements `[mass_fix] mode =
"initial_endpoint"`: the dry mass at the start of the first window becomes the
pin target of every later window, carried across the days of one run. Only
the GEOS native path does; the other native sources pin to an explicit target.
"""
supports_initial_endpoint_mass_pin(::AbstractMetSettings) = false
supports_initial_endpoint_mass_pin(::AbstractGEOSSettings) = true
