# ===========================================================================
# Met source abstraction for the preprocessor pipeline.
#
# A met source describes how raw NetCDF / GRIB data on its native grid is
# read into the canonical per-window intermediate `RawWindow`. The
# orchestrator dispatches on `<: AbstractMetSettings` (source axis) to choose
# the reader, and on `<: AbstractTargetGeometry` (target axis) to choose the
# regrid / flux-reconstruction / write path. The two axes never branch on
# each other — they meet in `RawWindow`.
# ===========================================================================

"""
    AbstractMetSettings

Top-level supertype for typed met-data source descriptors used by the
preprocessor. Concrete subtypes (e.g. `GEOSITSettings`, `MERRA2Settings`,
`SpectralERA5Settings`) carry source-specific paths and parameters and
implement the `read_window!` / `source_grid` / `windows_per_day` interface.

`process_day(date, grid::AbstractTargetGeometry, settings::AbstractMetSettings,
vertical; ...)` dispatches on `settings` to pick the reader.
"""
abstract type AbstractMetSettings end

"""
    ERA5SpectralSettings <: AbstractMetSettings

Typed wrapper for the historical ERA5 spectral settings `NamedTuple`.
The spectral preprocessor still performs spectral synthesis inside the
topology workspaces, but the source axis is now explicit: TOML parsing
constructs this settings type and topology methods dispatch on it.
"""
struct ERA5SpectralSettings <: AbstractMetSettings
    nt :: NamedTuple
end

# Forward the existing settings surface to the wrapped NamedTuple so the
# spectral implementation can be ported to typed dispatch without changing
# every field access in the same patch.
Base.getproperty(s::ERA5SpectralSettings, k::Symbol) =
    k === :nt ? getfield(s, :nt) : getfield(s, :nt)[k]
Base.propertynames(s::ERA5SpectralSettings) =
    propertynames(getfield(s, :nt))
Base.haskey(s::ERA5SpectralSettings, k::Symbol) =
    haskey(getfield(s, :nt), k)
Base.get(s::ERA5SpectralSettings, k::Symbol, default) =
    haskey(s, k) ? getfield(s, :nt)[k] : default

"""
    RawWindow{FT, A2, A3}

Per-window source-grid intermediate carrying **both window endpoints**
(t_n and t_{n+1}) and the **window-integrated horizontal mass fluxes**
between them.

The right endpoint of window n is the left endpoint of window n+1, so the
reader caches it across calls. For the last window of the day, the right
endpoint comes from the next day's first instantaneous file (the existing
`next_day_hour0` plumbing in the orchestrator).

Cell-center winds `u`, `v` (geographic frame) are filled only when the
target grid differs from the source grid and fluxes must be reconstructed
downstream. For native passthrough (source mesh == target mesh) `u`/`v`
stay `nothing` and `am`/`bm` are written through directly after vertical
merging.

`cmfmc`/`dtrain` are filled only when the source supports convection and
the user has enabled it via `settings.include_convection`. `vdiff` carries
optional Holtslag-Boville VDIFF source fields `(u, v, t, qv)` when the source
can archive them into the transport binary for runtime diffusion.
"""
struct RawWindow{FT <: AbstractFloat, P, V, S, VD}
    # left endpoint (t_n)
    m       :: V
    ps      :: P
    qv      :: Union{Nothing, V}
    # right endpoint (t_{n+1})
    m_next  :: V
    ps_next :: P
    qv_next :: Union{Nothing, V}
    # window-mean horizontal mass flux on the source grid (units source-defined:
    # for GEOS, Pa·m²/s = MFXC / mass_flux_dt; the orchestrator handles any
    # unit conversion at write time).
    am      :: V
    bm      :: V
    # cell-center winds (geographic frame); set only when regrid happens downstream
    u       :: Union{Nothing, V}
    v       :: Union{Nothing, V}
    # optional raw PBL surface fields `(pblh, ustar, hflux, t2m)`
    surface :: S
    # optional GCHP Holtslag-Boville VDIFF fields `(u, v, t, qv)`
    vdiff   :: VD
    # optional convection sources
    cmfmc   :: Union{Nothing, V}   # at interfaces
    dtrain  :: Union{Nothing, V}   # at centers
end

function RawWindow{FT, P, V}(m, ps, qv,
                             m_next, ps_next, qv_next,
                             am, bm,
                             u, v,
                             cmfmc, dtrain) where {FT <: AbstractFloat, P, V}
    return RawWindow{FT, P, V, Nothing, Nothing}(
        m, ps, qv,
        m_next, ps_next, qv_next,
        am, bm,
        u, v,
        nothing, nothing,
        cmfmc, dtrain)
end

function RawWindow{FT, P, V}(m, ps, qv,
                             m_next, ps_next, qv_next,
                             am, bm,
                             u, v,
                             surface,
                             cmfmc, dtrain) where {FT <: AbstractFloat, P, V}
    return RawWindow{FT, P, V, typeof(surface), Nothing}(
        m, ps, qv,
        m_next, ps_next, qv_next,
        am, bm,
        u, v,
        surface, nothing,
        cmfmc, dtrain)
end

function RawWindow{FT, P, V}(m, ps, qv,
                             m_next, ps_next, qv_next,
                             am, bm,
                             u, v,
                             surface, vdiff,
                             cmfmc, dtrain) where {FT <: AbstractFloat, P, V}
    return RawWindow{FT, P, V, typeof(surface), typeof(vdiff)}(
        m, ps, qv,
        m_next, ps_next, qv_next,
        am, bm,
        u, v,
        surface, vdiff,
        cmfmc, dtrain)
end

# `P` is the per-window 2D field type (PS-shaped); `V` is the per-window 3D
# field type (DELP / mass-flux shaped). For structured (LL/RG) sources these
# are `Matrix{FT}` and `Array{FT, 3}`. For CS native sources they are
# `NTuple{6, Matrix{FT}}` and `NTuple{6, Array{FT, 3}}` so that downstream
# code can dispatch panel-aware operations naturally.

# ---------------------------------------------------------------------------
# Source contract (concrete subtypes implement these). All preprocessing
# orchestrators consume this surface — no source-specific side doors.
# ---------------------------------------------------------------------------

"""
    open_day(settings::AbstractMetSettings, date::Date; ...) -> ctx

Open per-day source-specific context (file handles, caches, vertical
coefficients, …). The orchestrator calls this once per day at the start
of `process_day`, threads `ctx` through every `read_window!` call, and
calls `close_day!(ctx)` in a `finally` block.

`ctx` is opaque to the orchestrator — only the source knows its layout.
"""
function open_day end

"""
    close_day!(ctx)

Close all resources held by a day context. Must be idempotent; safe to
call from a `finally` block.
"""
function close_day! end

"""
    allocate_raw_window(settings::AbstractMetSettings;
                        FT::Type, Nc=nothing, Nz=nothing) -> RawWindow

Allocate a pre-zeroed `RawWindow` sized for one window of `settings`'s
source data. The orchestrator calls this once before the per-window
loop and reuses the same buffer across all windows in a day. The shape
parameters (`Nc`, `Nz`, …) are source-specific — concrete subtypes pick
the keys they need.
"""
function allocate_raw_window end

"""
    read_window!(raw::RawWindow, settings::AbstractMetSettings, ctx,
                 date::Date, win_idx::Int) -> raw

Fill `raw` in place with one window of source data on the source's
native grid. `ctx` is the day context returned by `open_day`.
Implementations must NOT allocate per call beyond bounded scratch —
`raw` is a pre-allocated workspace owned by the orchestrator — and
should be idempotent in `(date, win_idx)`.
"""
function read_window! end

"""
    source_grid(settings::AbstractMetSettings)

Return the source grid mesh that `read_window!` produces data on. Used
by the orchestrator to build the appropriate regridder for the target
grid (or to detect the source==target passthrough case).
"""
function source_grid end

"""
    windows_per_day(settings::AbstractMetSettings, date::Date) -> Int

Number of preprocessing windows per UTC day for this source on `date`.
For most sources this is constant (24 for hourly), but date-dependent
sources (e.g. a leap-second day) may override.
"""
function windows_per_day end

"""
    has_convection(settings::AbstractMetSettings) -> Bool

Whether this source can populate `RawWindow.cmfmc` and `RawWindow.dtrain`.
Defaults to `false`; sources that support convection override and gate
the actual output behind a user flag (e.g. `settings.include_convection`).
"""
has_convection(::AbstractMetSettings) = false

"""
    has_surface(settings::AbstractMetSettings) -> Bool

Whether this source can populate raw PBL surface fields in `RawWindow.surface`.
Sources gate actual reads behind their settings flags.
"""
has_surface(::AbstractMetSettings) = false

"""
    has_vdiff_fields(settings::AbstractMetSettings) -> Bool

Whether this source can populate `RawWindow.vdiff` with layer-center fields
needed by the GCHP Holtslag-Boville vertical-diffusion data contract.
"""
has_vdiff_fields(::AbstractMetSettings) = false

"""
    has_pbl_eflux(settings) -> Bool
    has_cmfmc_cloud_base(settings) -> Bool

Whether this source writes the surface latent heat flux (GCHP non-local PBL
scheme) or the convective cloud-base layer (GCHP cloud convection).
"""
has_pbl_eflux(::AbstractMetSettings) = false
has_cmfmc_cloud_base(::AbstractMetSettings) = false
