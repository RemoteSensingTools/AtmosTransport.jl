# ===========================================================================
# Native-GRIB ERA5 reader for the preprocessor.
#
# Consumes the canonical ECMWF download layout produced by
# `config/downloads/era5_native_daily.toml`:
#
#   <root_dir>/ml_an_native_core/era5_core_YYYYMMDD.grib
#   <root_dir>/ml_fc_convection/era5_convection_YYYYMMDD.grib   # gated by include_convection
#   <root_dir>/sfc_an_native/era5_surface_YYYYMMDD.grib          # gated by include_surface
#
# The `core` stream is the model-level analysis bundle (T, Q, VO, LNSP, D).
# T/VO/D/LNSP are stored as spherical-harmonic coefficients (`gridType=sh`),
# while Q is already on the reduced linear-Gaussian mesh (`gridType=reduced_gg`).
# The reader emits source-grid fields on the N-th reduced linear-Gaussian mesh
# selected by the `flavor` type parameter.
#
# Files: this one holds the settings and the day-handle surface;
# era5_n320_window.jl the per-window spectral synthesis (T / VO / D / LNSP →
# grid + reduced_gg Q); era5_n320_mass_convection.jl the dry-mass derivation
# and the convection forecast reader; era5_n320_to_cs.jl the conservative
# regrid to a cubed-sphere target and the per-window pipeline.
# ===========================================================================

"""
    AbstractERA5GRIBSettings <: AbstractMetSettings

Abstract supertype for ERA5 native-GRIB sources. Concrete subtypes pick the
source mesh via the `flavor` parameter on [`ERA5GRIBSettings`](@ref).
"""
abstract type AbstractERA5GRIBSettings <: AbstractMetSettings end

"""
    ERA5GRIBSettings{flavor} <: AbstractERA5GRIBSettings

Typed settings for one ERA5 native-GRIB flavor:

- `flavor = :n320` — reduced linear-Gaussian N320, the default MARS native grid
  for ERA5 analyses (640 longitudes at the equator, 137 hybrid levels).

ERA5 archives and the runtime both use top-down hybrid levels (`k = 1` is the
TOA-side layer). Only the GRIB horizontal ring order is reversed on ingestion.
"""
Base.@kwdef struct ERA5GRIBSettings{flavor} <: AbstractERA5GRIBSettings
    root_dir              :: String
    include_surface       :: Bool   = false
    include_convection    :: Bool   = false
    # Precompute exact TM5 boundary-layer diffusion interface exchange (`dkg`)
    # into the binary. Requires the surface stream (sshf + slhf + ustar),
    # target dry mass, and the synthesised 3D fields.
    include_tm5_diffusion :: Bool   = false
    # Source surface pressure from the ARCO single_level netCDF (0.25° `sp`,
    # bilinear-interpolated to N320) instead of synthesising spectral `lnsp`.
    # Google's ARCO-ERA5 raw model-level GRIB omits `lnsp`; enable this when
    # the `core` GRIB was assembled from ARCO (see config/downloads/era5_arco.toml).
    arco_surface_pressure :: Bool   = false
    coefficients_file     :: String = "config/era5_L137_coefficients.toml"
    level_orientation     :: Symbol = :top_down
    # Flux construction (shared with MERRA-2; see `flux_construction.jl` and
    # docs/src/config/data_sources.md). The defaults reproduce the historical path.
    column_balance_weights :: Symbol = :mass
    face_lengths          :: Symbol = :cell_centerline
    face_fluxes           :: Symbol = :panel_average
    face_interpolation    :: Symbol = :linear
    wind_regrid           :: Symbol = :scalar
    # Time sampling of the hourly window fluxes: the winds at the window start
    # (`:window_start`, historical) or the mean of the face fluxes at the start
    # and end of the window (`:window_mean`, trapezoidal rule for ∫ u Δp dt).
    flux_time_sampling    :: Symbol = :window_start
end

const ERA5N320Settings = ERA5GRIBSettings{:n320}
const ERA5_FLUX_TIME_SAMPLINGS = (:window_start, :window_mean)
_has_source_line_integrals(::ERA5GRIBSettings) = true
_validate_flux_time_sampling(s::ERA5GRIBSettings, source) =
    s.flux_time_sampling in ERA5_FLUX_TIME_SAMPLINGS || throw(ArgumentError(
        "$source flux_time_sampling must be one of $(join(ERA5_FLUX_TIME_SAMPLINGS, ", ")); got :$(s.flux_time_sampling)"))

# ---------------------------------------------------------------------------
# Stream layout.
#
# `ERA5_GRIB_STREAMS` keeps the on-disk subdirectory and filename stem for each
# GRIB stream in one place so adding a new flavor (e.g. `:o320`) does not need
# to touch any of the call sites.
# ---------------------------------------------------------------------------

const ERA5_GRIB_STREAMS = (
    core       = (subdir = "ml_an_native_core", stem = "era5_core"),
    convection = (subdir = "ml_fc_convection",  stem = "era5_convection"),
    surface    = (subdir = "sfc_an_native",     stem = "era5_surface"),
)

"""
    era5_grib_path(settings, date, stream) -> String

Resolve the on-disk GRIB path for `stream` on `date`. `stream` must be one of
`:core`, `:convection`, or `:surface`. Existence is *not* checked here — the
caller (typically [`open_era5_day`](@ref)) decides whether a missing file is
fatal or merely "no next-day endpoint available".
"""
function era5_grib_path(settings::AbstractERA5GRIBSettings, date::Date,
                        stream::Symbol)
    hasproperty(ERA5_GRIB_STREAMS, stream) ||
        throw(ArgumentError("unknown ERA5 GRIB stream $(stream); expected one of " *
                            string(propertynames(ERA5_GRIB_STREAMS))))
    layout   = getproperty(ERA5_GRIB_STREAMS, stream)
    datestr  = Dates.format(date, "yyyymmdd")
    filename = "$(layout.stem)_$(datestr).grib"
    return joinpath(settings.root_dir, layout.subdir, filename)
end

"""
    era5_arco_sp_path(settings, date) -> String

Resolve the ARCO single_level surface-pressure netCDF for `date`, written by the
ERA5-ARCO downloader under `sfc_an_native/arco/YYYYMMDD/surface_pressure.nc`.
Only consulted when `settings.arco_surface_pressure` is set.
"""
function era5_arco_sp_path(settings::AbstractERA5GRIBSettings, date::Date)
    datestr = Dates.format(date, "yyyymmdd")
    return joinpath(settings.root_dir, "sfc_an_native", "arco", datestr,
                    "surface_pressure.nc")
end

# ---------------------------------------------------------------------------
# Day-handle container.
#
# The handle holds resolved paths plus a lazily built ecCodes index of the
# core GRIB file (see `_core_messages`); `close_era5_day!` releases the index
# and is idempotent.
# ---------------------------------------------------------------------------

"""
    ERA5GRIBDayHandles{S<:AbstractERA5GRIBSettings}

Per-day source-file context. Carries the resolved on-disk paths to the day's
GRIB streams plus the optional next-day `core` path that supplies the right
endpoint of the last window.

`convection_path` is `nothing` unless `settings.include_convection` is set;
likewise `surface_path` is `nothing` unless `settings.include_surface` is set.
`next_core_path` is `nothing` either when the caller passed
`next_day_handle=false` or when the next-day file is not on disk (last day of
the available archive). `core_index` holds the ecCodes index of `core_path`
once the first window has been read.
"""
struct ERA5GRIBDayHandles{S <: AbstractERA5GRIBSettings}
    settings             :: S
    date                 :: Date
    core_path            :: String
    convection_path      :: Union{Nothing, String}
    surface_path         :: Union{Nothing, String}
    next_core_path       :: Union{Nothing, String}
    # Previous-day convection file. ERA5 convection forecasts run from 06 UTC
    # and 18 UTC bases with 1-12 h steps; hours 0..5 of `date` are covered
    # by the previous day's 18 UTC base (steps 6..11). For dates at the
    # start of an archive (where date-1 isn't downloaded) this stays
    # `nothing` and the convection reader rejects requests for hours 0..5.
    prev_convection_path :: Union{Nothing, String}
    # ARCO surface-pressure netCDF for `date` (only set when
    # `settings.arco_surface_pressure`; supplies PS in place of spectral LNSP).
    arco_sp_path         :: Union{Nothing, String}
    core_index           :: Base.RefValue{Union{Nothing, GRIB.Index}}
end

ERA5GRIBDayHandles(settings::S, date, core_path, convection_path, surface_path, next_core_path,
                   prev_convection_path, arco_sp_path) where S <: AbstractERA5GRIBSettings =
    ERA5GRIBDayHandles{S}(settings, date, core_path, convection_path, surface_path, next_core_path,
                          prev_convection_path, arco_sp_path, Ref{Union{Nothing, GRIB.Index}}(nothing))

const ERA5_VALID_LEVEL_ORIENTATIONS = (:top_down, :bottom_up)

"""
    open_era5_day(settings, date; next_day_handle=true) -> ERA5GRIBDayHandles

Resolve the GRIB stream paths for `date` and assert that today's required
files are on disk. When `next_day_handle=true` and `<date+1>` has a `core`
GRIB available, records its path so the last window's right endpoint can be
read from the next day's hour-0 fields.

Errors are explicit about which file is missing so a misconfigured `root_dir`
or an incomplete download is immediately visible.
"""
function open_era5_day(settings::AbstractERA5GRIBSettings, date::Date;
                       next_day_handle::Bool = true)
    settings.level_orientation in ERA5_VALID_LEVEL_ORIENTATIONS ||
        throw(ArgumentError("ERA5 level_orientation must be one of " *
                            join(ERA5_VALID_LEVEL_ORIENTATIONS, ", ") *
                            "; got :$(settings.level_orientation)"))

    core_path = era5_grib_path(settings, date, :core)
    isfile(core_path) ||
        error("ERA5 core GRIB not found: $core_path")

    convection_path = nothing
    if settings.include_convection
        candidate = era5_grib_path(settings, date, :convection)
        isfile(candidate) ||
            error("ERA5 convection GRIB not found: $candidate " *
                  "(include_convection=true)")
        convection_path = candidate
    end

    surface_path = nothing
    if settings.include_surface
        candidate = era5_grib_path(settings, date, :surface)
        if isfile(candidate)
            surface_path = candidate
        else
            # The ARCO downloader stores surface fields as per-variable
            # netCDFs under `<subdir>/arco/YYYYMMDD/` instead of one GRIB;
            # `open_era5_surface_reader` auto-detects that layout, so it
            # satisfies `include_surface` here too.
            arco_dir = joinpath(dirname(candidate), "arco",
                                Dates.format(date, "yyyymmdd"))
            has_arco_nc = isdir(arco_dir) &&
                any(f -> endswith(lowercase(f), ".nc"), readdir(arco_dir))
            has_arco_nc ||
                error("ERA5 surface data not found: neither GRIB $candidate " *
                      "nor ARCO netCDF directory $arco_dir " *
                      "(include_surface=true)")
            surface_path = arco_dir
        end
    end

    next_core_path = nothing
    if next_day_handle
        candidate = era5_grib_path(settings, date + Day(1), :core)
        next_core_path = isfile(candidate) ? candidate : nothing
    end

    prev_convection_path = nothing
    if settings.include_convection
        candidate = era5_grib_path(settings, date - Day(1), :convection)
        prev_convection_path = isfile(candidate) ? candidate : nothing
    end

    arco_sp_path = nothing
    if settings.arco_surface_pressure
        candidate = era5_arco_sp_path(settings, date)
        isfile(candidate) ||
            error("ERA5 ARCO surface-pressure netCDF not found: $candidate " *
                  "(arco_surface_pressure=true)")
        arco_sp_path = candidate
    end

    return ERA5GRIBDayHandles(settings, date,
                              core_path, convection_path, surface_path, next_core_path,
                              prev_convection_path, arco_sp_path)
end

"""
    close_era5_day!(handles::ERA5GRIBDayHandles)

Release the ecCodes index of the day's core GRIB file, if one was built.
Idempotent — safe to call from a `finally` block.
"""
function close_era5_day!(handles::ERA5GRIBDayHandles)
    idx = handles.core_index[]
    idx === nothing && return nothing
    GRIB.destroy(idx)
    handles.core_index[] = nothing
    return nothing
end

"""
    _core_messages(handles, date, hour) -> GRIB.Index

The core GRIB messages of `date` at `hour`. The ecCodes index is built in one
pass over the file on the first call; later calls seek straight to the messages
instead of scanning the whole file (12–24 GB per day) once per hourly window.
"""
function _core_messages(handles::ERA5GRIBDayHandles, date::Date, hour::Integer)
    if handles.core_index[] === nothing
        handles.core_index[] = GRIB.Index(handles.core_path, "dataDate", "dataTime")
    end
    idx = handles.core_index[]::GRIB.Index
    GRIB.select!(idx, "dataDate", parse(Int, Dates.format(date, "yyyymmdd")))
    GRIB.select!(idx, "dataTime", Int(hour) * 100)
    return idx
end

# ---------------------------------------------------------------------------
# AbstractMetSettings trait hooks.
# ---------------------------------------------------------------------------

open_day(settings::AbstractERA5GRIBSettings, date::Date;
         next_day_handle::Bool = true) =
    open_era5_day(settings, date; next_day_handle = next_day_handle)

close_day!(handles::ERA5GRIBDayHandles) = close_era5_day!(handles)

windows_per_day(::AbstractERA5GRIBSettings, ::Date) = 24

# Trait predicates report whether the source populates the optional
# `RawWindow` fields (`surface`, `cmfmc`/`dtrain`, `vdiff`). The ERA5 GRIB
# source has no `read_window!` method and never fills a `RawWindow`, so all
# three report `false` regardless of the `include_*` flags. The ERA5 N320
# writer (`process_era5_n320_to_cs_day`) reads the `include_*` settings
# directly and emits the surface, TM5 `dkg`, and TM5 convection sections
# itself, using its own per-window containers (`ERA5N320ConvectionFields`,
# etc.); it never writes CMFMC/DTRAIN or GCHP VDIFF fields.
has_surface(::AbstractERA5GRIBSettings)      = false
has_convection(::AbstractERA5GRIBSettings)   = false
has_vdiff_fields(::AbstractERA5GRIBSettings) = false
