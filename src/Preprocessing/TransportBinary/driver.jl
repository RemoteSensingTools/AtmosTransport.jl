# Unified-preprocessor lifecycle shared by production topology drivers:
# ingest source windows, drain ready windows, verify and accumulate the
# contract, write, close, summarize, promote, and quarantine on failure.

"""
    UnifiedPreprocessorDay(reader, workspace, contract, writer; context=nothing)

Bundle the four typed axes a unified preprocessing day needs. `context` is an
opaque topology/source adapter payload used by hook methods during migration.
"""
struct UnifiedPreprocessorDay{R, W, C, B, X}
    reader    :: R
    workspace :: W
    contract  :: C
    writer    :: B
    context   :: X
end

UnifiedPreprocessorDay(reader, workspace, contract, writer; context=nothing) =
    UnifiedPreprocessorDay{typeof(reader), typeof(workspace), typeof(contract),
                           typeof(writer), typeof(context)}(
        reader, workspace, contract, writer, context)

"""
    driver_windows_per_day(reader, context) -> Int

Migration hook for source readers whose window count needs adapter context.
"""
driver_windows_per_day(reader, _context) = windows_per_day(reader)

"""
    driver_ingest_window!(workspace, reader, win, context)

Migration hook for ingesting one source window into the target workspace.
Topology/source adapters may override while legacy workspace signatures are
being collapsed.
"""
driver_ingest_window!(workspace, reader, win::Int, _context) =
    ingest_window!(workspace, reader, win)

"""
    driver_drain_ready_windows!(workspace, contract, win, context)

Return ready windows produced by the most recent ingest. A hook may return a
single `ReadyWindow`, a single `PreverifiedWindow`, or any iterator of either
shape.
"""
driver_drain_ready_windows!(workspace, _contract, _win::Int, _context) =
    drain_ready_windows!(workspace)

"""
    driver_flush_final_windows!(workspace, reader, contract, context)

Return final ready windows after all source windows have been ingested.
"""
driver_flush_final_windows!(workspace, reader, contract, _context) =
    flush_final_windows!(workspace, reader, contract)

"""
    driver_after_write_window!(workspace, reader, ready, context)

Post-write migration hook. GEOS-native CS uses this to advance its chained
endpoint state; most topologies do nothing.
"""
driver_after_write_window!(_workspace, _reader, _ready, _context) = nothing

"""
    driver_before_close_writer!(workspace, reader, contract, writer, context)

Final metadata hook before the streaming writer is closed. Variable-step
writers use this to patch schedules into the fixed-size JSON header.
"""
driver_before_close_writer!(_workspace, _reader, _contract, _writer, _context) = nothing

"""Validate a closed staging artifact before it can replace the final path."""
validate_staged_binary!(_writer) = nothing

function _ready_events(result)
    result === nothing && return ()
    result isa ReadyWindow && return (result,)
    result isa PreverifiedWindow && return (result,)
    return result
end

function _verify_ready_event!(ready::ReadyWindow, contract)
    diag = verify_window!(ready, contract, ready.index)
    update_accumulator!(contract, diag.positivity, ready.index)
    return ready
end

function _verify_ready_event!(event::PreverifiedWindow, contract)
    ready = event.ready
    event.accumulated ||
        update_accumulator!(contract, event.contract.positivity, ready.index)
    return ready
end

function _handle_ready_event!(event, day::UnifiedPreprocessorDay)
    ready = _verify_ready_event!(event, day.contract)
    write_window!(day.writer, ready)
    driver_after_write_window!(day.workspace, day.reader, ready, day.context)
    return ready
end

"""
    run_unified_preprocessor_day!(day::UnifiedPreprocessorDay; close_reader=true)

Execute the additive unified-driver lifecycle for one day. The function is
generic over concrete reader/workspace/contract/writer types and depends on
hook methods for topology-specific ingest/drain/flush behavior.

The writer is closed before `summarize_status!`, so fatal positivity summaries
can quarantine a closed staging file. Any exception before promotion closes and
quarantines the writer, then closes the reader.
"""
function run_unified_preprocessor_day!(day::UnifiedPreprocessorDay;
                                       close_reader::Bool=true)
    writer_closed = false
    promoted = false
    windows_written = 0
    last_ready_index = 0

    try
        for win in 1:driver_windows_per_day(day.reader, day.context)
            driver_ingest_window!(day.workspace, day.reader, win, day.context)
            for event in _ready_events(
                    driver_drain_ready_windows!(day.workspace, day.contract,
                                                win, day.context))
                ready = _handle_ready_event!(event, day)
                windows_written += 1
                last_ready_index = ready.index
            end
        end

        for event in _ready_events(
                driver_flush_final_windows!(day.workspace, day.reader,
                                            day.contract, day.context))
            ready = _handle_ready_event!(event, day)
            windows_written += 1
            last_ready_index = ready.index
        end

        driver_before_close_writer!(day.workspace, day.reader, day.contract,
                                    day.writer, day.context)
        close_streaming_binary!(day.writer)
        writer_closed = true
        validate_staged_binary!(day.writer)
        summarize_status!(day.contract; quarantine_path = writer_staging_path(day.writer))
        promote_streaming_binary!(day.writer)
        promoted = true

        return (windows_written = windows_written,
                last_ready_index = last_ready_index,
                out_path = writer_final_path(day.writer),
                promoted = promoted)
    finally
        if !writer_closed
            try
                close_streaming_binary!(day.writer)
            catch err
                @warn("Unified preprocessor: failed to close writer during cleanup",
                      exception = (err, catch_backtrace()))
            end
        end
        if !promoted
            try
                quarantine_streaming_binary!(day.writer)
            catch err
                @warn("Unified preprocessor: failed to quarantine writer during cleanup",
                      exception = (err, catch_backtrace()))
            end
        end
        if close_reader
            try
                close_reader!(day.reader)
            catch err
                @warn("Unified preprocessor: failed to close reader during cleanup",
                      exception = (err, catch_backtrace()))
            end
        end
    end
end
