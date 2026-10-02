# Observation sampling glue. The runner parses `[output.observations]`, builds
# the sampler once the model state exists, and hands ownership to
# `RunSnapshotOutput` so the sampler is closed on every exit path together
# with the snapshot stream. Sampling happens at met-window ends only.
#
# Clock contract: the run origin (t = 0) is the start of window 1 of the first
# input binary, i.e. `[input].start_date` at 00:00 UTC, or `start_time` when
# the inputs are an explicit `binary_paths` list. A single-file run starting at
# `start_window > 1` begins `(start_window - 1)` windows after the origin.

const _OBSERVATION_ORIGIN_MESSAGE =
    "[output.observations] needs an absolute run origin: set [input].start_date " *
    "or [output.observations].start_time"

# Run origin (t = 0): `[input].start_date` at 00:00 UTC, else `start_time`.
function _observation_time_origin(spec::ObservationOutputSpec, cfg)
    reference = _run_reference_time(cfg)
    if reference !== nothing
        spec.start_time === nothing || spec.start_time == reference || throw(ArgumentError(
            "[output.observations].start_time = $(spec.start_time) disagrees with " *
            "[input].start_date = $(Date(reference)); the run origin is start_date at 00:00 UTC"))
        return reference
    end
    spec.start_time === nothing && throw(ArgumentError(_OBSERVATION_ORIGIN_MESSAGE))
    return spec.start_time
end

# `validate_config` preflight.
_check_observation_time_origin(::NoObservationOutput, cfg) = nothing
_check_observation_time_origin(spec::ObservationOutputSpec, cfg) =
    (_observation_time_origin(spec, cfg); nothing)

function _binary_label_date(path::AbstractString)
    label = _binary_date_label(path)
    isempty(label) && return nothing
    date = tryparse(Date, label, dateformat"yyyymmdd")
    date === nothing && @warn "observation sampling: 8-digit token $(label) in $(basename(path)) is not a date; ignoring it"
    return date
end

# The labelled day of the first binary must be the origin's day, otherwise
# every sounding time would be shifted silently.
function _check_origin_against_binaries(origin::DateTime, binary_paths)
    isempty(binary_paths) && return nothing
    first_day = _binary_label_date(first(binary_paths))
    first_day === nothing || first_day == Date(origin) || throw(ArgumentError(
        "observation run origin $(origin) is not on the first binary's date $(first_day) " *
        "($(basename(first(binary_paths))))"))
    Time(origin) == Time(0) ||
        @warn "observation run origin $(origin) is not 00:00 UTC; it must be the start of window 1 of the first binary"
    return nothing
end

function _install_observation_sampler!(output::RunSnapshotOutput, output_cfg::AbstractDict,
                                       partition::AbstractOutputPartition, state, grid;
                                       cfg, binary_paths, halo_width::Integer,
                                       offset_seconds::Real = 0.0, span_seconds::Real = Inf)
    spec = observation_output_spec(output_cfg; partition)
    sampler = _build_sampler(spec, state, grid; cfg, binary_paths, halo_width,
                             offset_seconds, span_seconds)
    output.observations = sampler
    return sampler
end

_build_sampler(::NoObservationOutput, state, grid; kwargs...) = NoObservationSampler()
function _build_sampler(spec::ObservationOutputSpec, state, grid; cfg, binary_paths, halo_width,
                        offset_seconds, span_seconds)
    origin = _observation_time_origin(spec, cfg)
    _check_origin_against_binaries(origin, binary_paths)
    t0 = Float64(offset_seconds)
    window_seconds = (t0, t0 + Float64(span_seconds))
    @info "observation sampling window: $(origin + Millisecond(round(Int, 1000 * window_seconds[1]))) " *
          "to $(origin + Millisecond(round(Int, 1000 * window_seconds[2]))) UTC"
    return build_observation_sampler(spec, state, grid; origin, window_seconds, halo_width)
end

# Per-binary entry: switch daily files, and on the first binary sample the
# initial state at the start of the run.
_begin_observation_binary!(timer, ::NoObservationSampler, sim, path, idx::Integer, seconds::Real) = nothing
function _begin_observation_binary!(timer, sampler::AbstractObservationSampler, sim, path,
                                    idx::Integer, seconds::Real)
    begin_observation_day!(sampler, _binary_date_label(path), idx)
    idx == 1 && _observe_window_end!(timer, sampler, sim, seconds)
    return nothing
end

# GCHP VDIFF layer temperature (cubed-sphere binaries only, interior panels,
# k = 1 at the top) of the window that just ended; otherwise the sampler's
# constant applies. Virtual-temperature effects are neglected.
_window_temperature(sim) = sim.window.vdiff === nothing ? nothing : sim.window.vdiff.t

_observe_window_end!(timer, ::NoObservationSampler, sim, seconds::Real) = nothing
function _observe_window_end!(timer, sampler::AbstractObservationSampler, sim, seconds::Real)
    temperature = _window_temperature(sim)
    next_window_seconds = Float64(window_dt(sim.driver))
    timed_io_write!(timer, () -> observe_window_boundary!(sampler, sim.model.state, Float64(seconds);
                                                          next_window_seconds, temperature))
    return nothing
end
