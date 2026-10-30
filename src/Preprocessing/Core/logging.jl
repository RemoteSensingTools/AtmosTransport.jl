# ---------------------------------------------------------------------------
# Flushing logger wrapper — calls flush(stderr) after every log message.
# Without this, @info output is invisible for 5+ minutes when stderr is
# redirected to a file (Julia's libuv stream buffering).
# ---------------------------------------------------------------------------
"""Forward log messages and flush standard output/error for redirected CLI logs."""
struct FlushingLogger{L<:AbstractLogger} <: AbstractLogger
    inner :: L
end

Logging.min_enabled_level(l::FlushingLogger) = Logging.min_enabled_level(l.inner)
Logging.shouldlog(l::FlushingLogger, level, _module, group, id) =
    Logging.shouldlog(l.inner, level, _module, group, id)
Logging.catch_exceptions(l::FlushingLogger) = Logging.catch_exceptions(l.inner)

function Logging.handle_message(l::FlushingLogger, level, message, _module, group, id, file, line; kwargs...)
    Logging.handle_message(l.inner, level, message, _module, group, id, file, line; kwargs...)
    try
        flush(stderr)
    catch
    end
    try
        flush(stdout)
    catch
    end
    return nothing
end
