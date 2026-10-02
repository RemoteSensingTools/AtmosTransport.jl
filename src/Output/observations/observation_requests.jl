# ---------------------------------------------------------------------------
# Sampling requests: what the runtime sampler is asked to extract.
# ---------------------------------------------------------------------------

"""
    SoundingRequest

One time-stamped request for a full profile at (`lon`, `lat`), `time_seconds`
after the run origin (Float64 UTC seconds). `source` indexes the spec's
`sources` so output can attribute each row.
"""
struct SoundingRequest
    id::String
    time_seconds::Float64
    lon::Float64
    lat::Float64
    source::Int
end

"""
    SiteRequest

A fixed station sampled at every met-window end. `elevation_m` (above sea
level) is informational; `intake_height_m` (above ground) selects the model
layer, with `NaN` meaning the lowest layer.
"""
struct SiteRequest
    id::String
    lon::Float64
    lat::Float64
    elevation_m::Float64
    intake_height_m::Float64
    source::Int
end

"""
    ObservationSet(origin, soundings, sites[, dropped_outside_window])

Every sampling request of a run. `soundings` are sorted by `time_seconds`
(stable) and `sites` are unique by `id`. `dropped_outside_window` counts the
soundings the sources provided outside the run days.
"""
struct ObservationSet
    origin::DateTime
    soundings::Vector{SoundingRequest}
    sites::Vector{SiteRequest}
    dropped_outside_window::Int
end
ObservationSet(origin::DateTime, soundings::Vector{SoundingRequest}, sites::Vector{SiteRequest}) =
    ObservationSet(origin, soundings, sites, 0)

Base.isempty(set::ObservationSet) = isempty(set.soundings) && isempty(set.sites)
