# ---------------------------------------------------------------------------
# Sampling requests: what the runtime sampler is asked to extract.
#
# A *point event* (`SoundingRequest`) is sampled once, at its own time: a
# satellite sounding, an ObsPack flask or aircraft record, or one entry of a
# station's time list. A *site* (`SiteRequest`) is a fixed station written
# into the hourly site series according to its `AbstractSiteSchedule`.
# ---------------------------------------------------------------------------

"""
    AbstractSiteSchedule

When a station is sampled. Chosen per site from the site table:

| Table keys | Type | Output |
|---|---|---|
| none | [`EveryWindow`](@ref) | every met-window end, `_sites` file |
| `start_time`, `end_time` | [`TimeRange`](@ref) | every met-window end inside the range, `_sites` file (NaN outside) |
| `times` | [`TimeList`](@ref) | one point event per listed UTC time, `_soundings` file |
"""
abstract type AbstractSiteSchedule end

"Sample the site at every met-window end of the run."
struct EveryWindow <: AbstractSiteSchedule end

"""
    TimeRange(start_seconds, stop_seconds)

Sample the site at every met-window end with `start <= t <= stop` (seconds
after the run origin); records outside the range are written as NaN.
"""
struct TimeRange <: AbstractSiteSchedule
    start_seconds::Float64
    stop_seconds::Float64
    function TimeRange(start_seconds::Real, stop_seconds::Real)
        isfinite(start_seconds) && isfinite(stop_seconds) && stop_seconds >= start_seconds ||
            throw(ArgumentError("site time range must be finite with end >= start; got ($(start_seconds), $(stop_seconds))"))
        return new(Float64(start_seconds), Float64(stop_seconds))
    end
end

"""
    TimeList(times_seconds)

Sample the site once at each listed time (seconds after the run origin),
like a sounding: blended between the bracketing met-window ends and written
to the `_soundings` file with the intake-layer value.
"""
struct TimeList <: AbstractSiteSchedule
    times_seconds::Vector{Float64}
    function TimeList(times_seconds::AbstractVector{<:Real})
        times = sort!(Float64.(collect(times_seconds)))
        isempty(times) && throw(ArgumentError("site time list must not be empty"))
        all(isfinite, times) || throw(ArgumentError("site time list must contain finite times"))
        return new(times)
    end
end

schedule_label(::EveryWindow) = :every_window
schedule_label(::TimeRange) = :time_range
schedule_label(::TimeList) = :time_list

"Whether a site is sampled at the window end `t` (seconds after the origin)."
site_active(::EveryWindow, t::Real) = true
site_active(range::TimeRange, t::Real) = range.start_seconds <= t <= range.stop_seconds
# Time-list sites become point events (`_expand_site!`) and never reach the
# per-window series, which `ObservationSet` enforces.
_check_series_schedule(::AbstractSiteSchedule, id) = nothing
_check_series_schedule(::TimeList, id) = throw(ArgumentError(
    "site $(repr(id)) has a time list; expand it into point events (build_observation_set does)"))

"""
    SoundingRequest(id, time_seconds, lon, lat, source[, elevation_m, intake_height_m])

One point event, sampled at `time_seconds` after the run origin (Float64 UTC
seconds). `source` indexes the spec's `sources`. `intake_height_m` (above the
model surface) selects the layer for the `<tracer>_intake` value; `NaN`
means the lowest layer. `elevation_m` (above sea level) is informational.
"""
struct SoundingRequest
    id::String
    time_seconds::Float64
    lon::Float64
    lat::Float64
    source::Int
    elevation_m::Float64
    intake_height_m::Float64
end
SoundingRequest(id, time_seconds, lon, lat, source) =
    SoundingRequest(id, time_seconds, lon, lat, source, NaN, NaN)

"""
    SiteRequest(id, lon, lat, elevation_m, intake_height_m, source[, schedule])

A fixed station. `intake_height_m` (above the model surface) selects the
model layer, with `NaN` meaning the lowest layer; `elevation_m` (above sea
level) is informational. `schedule` defaults to [`EveryWindow`](@ref).
"""
struct SiteRequest
    id::String
    lon::Float64
    lat::Float64
    elevation_m::Float64
    intake_height_m::Float64
    source::Int
    schedule::AbstractSiteSchedule
end
SiteRequest(id, lon, lat, elevation_m, intake_height_m, source) =
    SiteRequest(id, lon, lat, elevation_m, intake_height_m, source, EveryWindow())

"""
    ObservationSet(origin, soundings, sites[, dropped_outside_window, skipped_invalid])

Every sampling request of a run. `soundings` are sorted by `time_seconds`
(stable) and `sites` are unique by `id`. `dropped_outside_window` counts the
point events outside the transported span; `skipped_invalid` counts source
rows skipped for fill values, invalid coordinates, or the quality filter.
"""
struct ObservationSet
    origin::DateTime
    soundings::Vector{SoundingRequest}
    sites::Vector{SiteRequest}
    dropped_outside_window::Int
    skipped_invalid::Int
    function ObservationSet(origin::DateTime, soundings::Vector{SoundingRequest}, sites::Vector{SiteRequest},
                            dropped_outside_window::Integer, skipped_invalid::Integer)
        foreach(site -> _check_series_schedule(site.schedule, site.id), sites)
        return new(origin, soundings, sites, dropped_outside_window, skipped_invalid)
    end
end
ObservationSet(origin::DateTime, soundings::Vector{SoundingRequest}, sites::Vector{SiteRequest},
               dropped_outside_window::Integer = 0) =
    ObservationSet(origin, soundings, sites, Int(dropped_outside_window), 0)

Base.isempty(set::ObservationSet) = isempty(set.soundings) && isempty(set.sites)

# Sites with a time list become point events; the others stay in the series.
_expand_site!(soundings, sites, site::SiteRequest) = _expand_site!(soundings, sites, site, site.schedule)
_expand_site!(soundings, sites, site::SiteRequest, ::AbstractSiteSchedule) = push!(sites, site)
function _expand_site!(soundings, sites, site::SiteRequest, schedule::TimeList)
    for t in schedule.times_seconds
        push!(soundings, SoundingRequest(site.id, t, site.lon, site.lat, site.source,
                                         site.elevation_m, site.intake_height_m))
    end
    return soundings
end
