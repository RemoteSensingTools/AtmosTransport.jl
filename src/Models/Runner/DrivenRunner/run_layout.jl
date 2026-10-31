# Public runtime layout interface shared with diagnostics and scripts.
# Aliases keep the legacy qualified bindings and the exact same generic methods.
const binary_date_label = _binary_date_label
const configured_float_type = _cfg_float_type
const check_multifile_window_count = _check_multifile_window_count
const layout_run_hours = _layout_run_hours
const parse_tracer_specs = _parse_tracer_specs
const run_reference_time = _run_reference_time
const run_time_origin = _run_time_origin
const run_time_span = _run_time_span
const run_window_index = _run_window_index
const runtime_window_layout = _runtime_window_layout
const structured_tracer_specs = _structured_tracer_specs
const runtime_output_origin = _runtime_output_origin
const window_end_hours = _window_end_hours
const observation_terrain = _observation_terrain

export binary_date_label, configured_float_type, check_multifile_window_count, layout_run_hours, parse_tracer_specs, run_reference_time, run_time_origin, run_time_span, run_window_index, runtime_window_layout, structured_tracer_specs, runtime_output_origin, window_end_hours, observation_terrain

"""
    run_tracer_specs(cfg, mesh)

Read tracer specifications with the runtime's topology rules. Cubed-sphere
runs require `[tracers]`; structured runs retain the legacy single-tracer form.
"""
function run_tracer_specs(cfg, ::CubedSphereMesh)
    specs = _parse_tracer_specs(cfg)
    specs === nothing && throw(ArgumentError("cubed-sphere runs require a [tracers] table, as the runner does"))
    return specs
end
run_tracer_specs(cfg, mesh) = _structured_tracer_specs(cfg)
export run_tracer_specs
