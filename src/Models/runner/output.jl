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
