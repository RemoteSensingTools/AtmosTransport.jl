# ---------------------------------------------------------------------------
# Runtime sampler interface. `NoObservationSampler` is the default that keeps
# every runner path identical to a run without `[output.observations]`.
# Runners call the interface once per met window, never per cell, so the
# dynamic dispatch on the abstract resource field is negligible.
# ---------------------------------------------------------------------------

abstract type AbstractObservationSampler end

"No-op sampler installed when observation output is absent or disabled."
struct NoObservationSampler <: AbstractObservationSampler end

"""
    build_observation_sampler(spec, args...; kwargs...)

Construct the runtime sampler for a parsed `[output.observations]` spec.
`NoObservationOutput` yields `NoObservationSampler()`.
"""
build_observation_sampler(::NoObservationOutput, args...; kwargs...) = NoObservationSampler()
# TODO(observation sampling, later phase): replace with the real sampler.
build_observation_sampler(::ObservationOutputSpec, args...; kwargs...) =
    throw(ArgumentError(OBSERVATION_RUNTIME_UNAVAILABLE_MESSAGE))

"""
    observe_window_boundary!(sampler, state, time_seconds; temperature = nothing, t2m = nothing)

Sample `state` at a met-window end, `time_seconds` after the run origin
(Float64). Called once at `t = 0` on the initial state and after every window.
"""
observe_window_boundary!(::NoObservationSampler, state, time_seconds::Real; kwargs...) = nothing

"Switch daily observation files before the first boundary of a new input binary."
begin_observation_day!(::NoObservationSampler, date_label::AbstractString, day_index::Integer) = nothing

"Flush held samples and write summary attributes at the end of a successful run."
finish_observations!(::NoObservationSampler) = nothing

Base.close(::NoObservationSampler) = nothing
