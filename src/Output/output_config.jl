# `[output]` (`runtime_output_spec`), in precedence order within each alias group.
const OUTPUT_PATH_KEYS = ("path", "snapshot_file", "filename")
const OUTPUT_KEYS = (OUTPUT_PATH_KEYS..., SNAPSHOT_SCHEDULE_KEYS..., "start_hour",
                      "stop_hour", "split", "format", "deflate_level", "shuffle", "enabled",
                      "fields", "observations")
const OUTPUT_FIELDS_KEYS = ("tracers", "layers", "levels", "column_mean",
                             "column_mass_per_area", "column_mass", "air_mass_layers",
                             "air_mass", "air_mass_per_area", "column_air_mass_per_area",
                             "per_tracer", "tracer_fields")
const TRACER_OUTPUT_FIELDS_KEYS = ("layers", "column_mean", "column_mass_per_area",
                                    "column_mass")

"""
    check_output_targets(spec, output_cfg)

Reject snapshot settings that would silently write nothing while
`[output].enabled` is true: a path without any snapshot-schedule key, or
snapshot times without a path. An explicitly empty list (`hours = []`) is a
deliberate "no snapshots" (the path then only names the timing CSV of
`ATMOSTR_TIMERS` runs).
"""
function check_output_targets(spec::RuntimeOutputSpec, output_cfg::AbstractDict)
    spec.enabled || return nothing
    has_path = !isempty(output_path(spec))
    has_times = schedule_has_times(spec.schedule)
    has_path && !any(k -> haskey(output_cfg, k), SNAPSHOT_SCHEDULE_KEYS) && throw(ArgumentError(
        "[output] sets a path but no snapshot times, so no snapshot would be written; " *
        "add `hours = [...]` or `cadence_hours`, set `hours = []` for none, or set " *
        "`enabled = false`"))
    has_times && !has_path && throw(ArgumentError(
        "[output] sets snapshot times but no `path`, so no snapshot would be written; " *
        "add `path` or set `enabled = false`"))
    return nothing
end
