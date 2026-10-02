# Observation sampling glue. The runner parses `[output.observations]`, builds
# the sampler, and hands ownership to `RunSnapshotOutput` so the sampler is
# closed on every exit path together with the snapshot stream.

function _install_observation_sampler!(output::RunSnapshotOutput, output_cfg::AbstractDict,
                                       partition::AbstractOutputPartition)
    spec = observation_output_spec(output_cfg; partition)
    sampler = build_observation_sampler(spec)
    output.observations = sampler
    return sampler
end

# `validate_config` preflight: an enabled observation table needs an absolute
# run origin, because sounding times are absolute UTC instants.
_check_observation_time_origin(::NoObservationOutput, cfg) = nothing
function _check_observation_time_origin(spec::ObservationOutputSpec, cfg)
    spec.start_time === nothing || return nothing
    input_cfg = get(cfg, "input", nothing)
    input_cfg isa AbstractDict && haskey(input_cfg, "start_date") && return nothing
    throw(ArgumentError(
        "[output.observations] needs an absolute run origin: set [input].start_date " *
        "or [output.observations].start_time"))
end

# TODO(observation sampling, later phase): drop this gate once the runtime
# sampler exists; until then an enabled table fails at config validation.
_check_observation_runtime_support(::NoObservationOutput) = nothing
_check_observation_runtime_support(::ObservationOutputSpec) =
    throw(ArgumentError(OBSERVATION_RUNTIME_UNAVAILABLE_MESSAGE))
