function _output_display_path(spec::RuntimeOutputSpec)
    return output_enabled(spec) ? output_path(spec) : "(disabled)"
end

function _output_basename(spec::RuntimeOutputSpec)
    return output_enabled(spec) ? basename(output_path(spec)) : "(disabled)"
end

function _binary_date_label(path::AbstractString)
    m = match(r"(\d{8})", basename(path))
    return m === nothing ? "" : String(something(m.captures[1], ""))
end

function _output_default_cap_hours(driver, binary_count::Integer;
                                   start_window::Integer = 1,
                                   stop_window_override = nothing)
    stop_window = stop_window_override === nothing ?
                  total_windows(driver) :
                  min(Int(stop_window_override), total_windows(driver))
    nw = max(stop_window - start_window + 1, 0)
    return Float64(nw * Int(binary_count)) * Float64(window_dt(driver)) / 3600.0
end

"""
    _check_output_targets(spec, output_cfg)

Reject snapshot settings that would silently write nothing while
`[output].enabled` is true: a path without any snapshot-schedule key, or
snapshot times without a path. An explicitly empty list (`hours = []`) is a
deliberate "no snapshots" (the path then only names the timing CSV of
`ATMOSTR_TIMERS` runs).
"""
function _check_output_targets(spec::RuntimeOutputSpec, output_cfg::AbstractDict)
    spec.enabled || return nothing
    has_path = !isempty(output_path(spec))
    has_times = _schedule_has_times(spec.schedule)
    has_path && !any(k -> haskey(output_cfg, k), _SNAPSHOT_SCHEDULE_KEYS) && throw(ArgumentError(
        "[output] sets a path but no snapshot times, so no snapshot would be written; " *
        "add `hours = [...]` or `cadence_hours`, set `hours = []` for none, or set " *
        "`enabled = false`"))
    has_times && !has_path && throw(ArgumentError(
        "[output] sets snapshot times but no `path`, so no snapshot would be written; " *
        "add `path` or set `enabled = false`"))
    return nothing
end

"""
    _snapshot_due(elapsed_hours, hour, window_hours) -> Bool

Whether the snapshot requested at `hour` is taken at the window end reached
after `elapsed_hours`: within half a window, and at most half an hour, of it.
"""
@inline _snapshot_due(elapsed_hours, hour, window_hours) =
    abs(elapsed_hours - hour) < min(0.5, window_hours / 2)

# Hours since the run start of every met-window end, accumulated as the run
# loops accumulate them. `layout` holds `(window_seconds, windows)` per binary.
function _window_end_hours(layout; start_window::Integer = 1, stop_window_override = nothing)
    ends = Float64[]
    elapsed = 0.0
    for (window_seconds, nwindow) in layout
        stop = stop_window_override === nothing ? nwindow : min(Int(stop_window_override), nwindow)
        for _ in start_window:stop
            elapsed += Float64(window_seconds) / 3600
            push!(ends, elapsed)
        end
    end
    return ends
end

# Hours covered by the run: the selected windows of every binary (`layout`).
function _layout_run_hours(layout; start_window::Integer = 1, stop_window_override = nothing)
    seconds = 0.0
    for (window_seconds, nwindow) in layout
        stop = stop_window_override === nothing ? nwindow : min(Int(stop_window_override), nwindow)
        seconds += max(stop - start_window + 1, 0) * Float64(window_seconds)
    end
    return seconds / 3600
end

# Layout of `binary_paths` when only the first driver is known: every binary
# like the first.
_uniform_window_layout(driver, binary_paths) =
    fill((Float64(window_dt(driver)), total_windows(driver)), length(binary_paths))

"""
    _check_snapshot_schedule(spec, window_end_hours)

Snapshots are taken at met-window ends (`window_end_hours`, counted from the
run start) and at hour 0, once each. Reject other and repeated hours: such an
hour is never matched, and every later snapshot is lost with it. Warn about hours after the last
window end, which are never written.
"""
function _check_snapshot_schedule(spec::RuntimeOutputSpec, window_end_hours::AbstractVector)
    output_enabled(spec) || return nothing
    hours = snapshot_hours(spec)
    # Each hour is taken once; a repeated hour (within the matching tolerance
    # below) blocks every later snapshot. `hours` is sorted.
    repeated = unique(hours[[i for i in 2:length(hours) if hours[i] - hours[i - 1] <= 1e-6]])
    isempty(repeated) || throw(ArgumentError(
        "[output] snapshot hour(s) $(join(repeated, ", ")) are listed more than once"))
    last_end = isempty(window_end_hours) ? 0.0 : last(window_end_hours)
    on_end(h) = (i = searchsortedfirst(window_end_hours, h - 1e-6);
                 i <= length(window_end_hours) && window_end_hours[i] <= h + 1e-6)
    off_grid = [h for h in hours if h < 0 || (h <= last_end + 1e-6 && h != 0 && !on_end(h))]
    isempty(off_grid) || throw(ArgumentError(
        "[output] snapshot hour(s) $(join(first(off_grid, 5), ", "))" *
        "$(length(off_grid) > 5 ? ", ..." : "") do not fall on a met-window end; " *
        "snapshots are taken at hour 0 and at window ends only (counted from the run " *
        "start), and a missed hour also loses every later snapshot"))
    late = count(>(last_end + 1e-6), hours)
    late > 0 && @warn "[output]: $(late) snapshot hour(s) fall after the end of the run " *
                      "($(last_end) h) and will not be written."
    return nothing
end

"""
    _check_unique_day_paths(partition, path_for, binary_paths, key)

Fail before any transport when two input binaries resolve to the same daily
output file (`path_for(date_label, index)`), which would otherwise overwrite
the earlier day silently. `key` names the TOML path setting in the error.
"""
_check_unique_day_paths(::SingleOutputFile, path_for, binary_paths, key::AbstractString) = nothing
function _check_unique_day_paths(::DailyOutputFiles, path_for, binary_paths, key::AbstractString)
    seen = Dict{String, String}()
    for (idx, binary) in enumerate(binary_paths)
        path = path_for(_binary_date_label(binary), idx)
        previous = get(seen, path, nothing)
        previous === nothing || throw(ArgumentError(
            "daily output $(path) would be written for both $(basename(previous)) and " *
            "$(basename(binary)); add a {day} token to $(key) or give the binaries distinct dates"))
        seen[path] = binary
    end
    return nothing
end

_check_snapshot_day_paths(spec::RuntimeOutputSpec, binary_paths) =
    output_enabled(spec) ? _check_unique_day_paths(spec.partition,
        (label, idx) -> output_path_for_day(spec, label, idx), binary_paths, "[output].path") : nothing

_output_path_for_partition(spec::RuntimeOutputSpec, ::SingleOutputFile,
                           ::AbstractString, ::Integer) = output_path(spec)
_output_path_for_partition(spec::RuntimeOutputSpec, ::DailyOutputFiles,
                           date_label::AbstractString, day_index::Integer) =
    output_path_for_day(spec, date_label, day_index)

function _push_snapshot_frame!(::SingleOutputFile,
                               snapshots::AbstractVector{<:AbstractSnapshotFrame},
                               ::AbstractVector{<:AbstractSnapshotFrame},
                               frame::AbstractSnapshotFrame)
    push!(snapshots, frame)
    return nothing
end

# The outer run owns this resource, including exceptional exits from either topology.
# `observations` is the `[output.observations]` sampler (a no-op by default); it
# is closed after the snapshot stream so both get the same lifetime guarantees.
mutable struct RunSnapshotOutput
    stream::Union{Nothing,NetCDFSnapshotStream}
    pending_write::Union{Nothing,Task}
    observations::AbstractObservationSampler
end
RunSnapshotOutput() = RunSnapshotOutput(nothing, nothing, NoObservationSampler())
RunSnapshotOutput(stream, pending_write) =
    RunSnapshotOutput(stream, pending_write, NoObservationSampler())

function _wait_pending_output!(output::RunSnapshotOutput)
    task = output.pending_write
    task === nothing && return nothing
    try
        wait(task)
    finally
        # Release the task and its captured frames even if the write failed.
        output.pending_write = nothing
    end
    return nothing
end

function Base.close(output::RunSnapshotOutput)
    # Take the sampler so a second close cannot close it twice.
    sampler = output.observations
    output.observations = NoObservationSampler()
    _with_run_resource(sampler) do
        stream = output.stream
        if stream === nothing
            _wait_pending_output!(output)
        else
            _with_run_resource(stream) do
                _wait_pending_output!(output)
            end
        end
    end
    return nothing
end

function _single_netcdf_stream(output::RunSnapshotOutput, spec::RuntimeOutputSpec, grid; mass_basis)
    output_enabled(spec) && spec.format === :netcdf && spec.partition isa SingleOutputFile ||
        return nothing
    output.stream = NetCDFSnapshotStream(output_path(spec), grid;
                                         mass_basis, options=spec.options, fields=spec.fields)
    return output.stream
end

_record_snapshot!(::Nothing, partition, snapshots, day_snapshots, frame) =
    _push_snapshot_frame!(partition, snapshots, day_snapshots, frame)
_record_snapshot!(stream::NetCDFSnapshotStream, partition, snapshots, day_snapshots, frame) =
    append_snapshot!(stream, frame)

function _push_snapshot_frame!(::DailyOutputFiles,
                               ::AbstractVector{<:AbstractSnapshotFrame},
                               day_snapshots::AbstractVector{<:AbstractSnapshotFrame},
                               frame::AbstractSnapshotFrame)
    push!(day_snapshots, frame)
    return nothing
end

function _write_output_frames!(timer::RunProgressTimer,
                               spec::RuntimeOutputSpec,
                               partition::AbstractOutputPartition,
                               frames::AbstractVector{<:AbstractSnapshotFrame},
                               grid;
                               mass_basis::Symbol,
                               date_label::AbstractString = "",
                               day_index::Integer = 1)
    output_enabled(spec) || return nothing
    isempty(frames) && return nothing
    path = _output_path_for_partition(spec, partition, date_label, day_index)
    timed_io_write!(timer, () -> if spec.format === :binary_mmap
        write_snapshot_binary(path, frames, grid;
                              mass_basis = mass_basis,
                              options = spec.options)
    else
        write_snapshot_netcdf(path, frames, grid;
                              mass_basis = mass_basis,
                              options = spec.options,
                              fields = spec.fields)
    end)
    return path
end

# Write accumulated HOST-side snapshot frames to disk. Used by the async
# daily-flush path (Threads.@spawn): runs off the main loop so the next day's
# GPU transport overlaps the disk write. Deliberately does NOT touch the run
# timer (the overlapped write is not charged to wall io_write) and never touches
# GPU memory (frames are `Array(...)` copies captured at snapshot time).
function _write_frames_to_disk(spec::RuntimeOutputSpec, path::AbstractString,
                               frames::AbstractVector{<:AbstractSnapshotFrame}, grid, mass_basis::Symbol)
    isempty(frames) && return path
    if spec.format === :binary_mmap
        write_snapshot_binary(path, frames, grid; mass_basis = mass_basis,
                              options = spec.options)
    else
        # The writer takes the shared NetCDF lock itself.
        write_snapshot_netcdf(path, frames, grid; mass_basis = mass_basis,
                              options = spec.options, fields = spec.fields)
    end
    return path
end

function _start_daily_output!(output::RunSnapshotOutput, spec::RuntimeOutputSpec,
                              path::AbstractString, frames, grid, mass_basis::Symbol)
    # Only one background write may own frames at a time. The enclosing run
    # drains it on both successful and exceptional exits through close(output).
    _wait_pending_output!(output)
    owned_frames = copy(frames)
    empty!(frames)
    output.pending_write = Threads.@spawn _write_frames_to_disk(
        spec, path, owned_frames, grid, mass_basis)
    return nothing
end

_flush_daily_output!(::SingleOutputFile, timer, spec, frames, grid;
                     mass_basis, date_label, day_index) = nothing

function _flush_daily_output!(partition::DailyOutputFiles, timer, spec, frames, grid;
                              mass_basis, date_label, day_index)
    isempty(frames) && return nothing
    written = _write_output_frames!(timer, spec, partition, frames, grid;
                                    mass_basis = mass_basis,
                                    date_label = date_label,
                                    day_index = day_index)
    empty!(frames)
    return written
end

_flush_single_output!(::DailyOutputFiles, timer, spec, frames, grid;
                      mass_basis) = nothing

function _flush_single_output!(partition::SingleOutputFile, timer, spec, frames, grid;
                               mass_basis)
    return _write_output_frames!(timer, spec, partition, frames, grid;
                                 mass_basis = mass_basis)
end
