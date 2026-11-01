"""
    LogLevels

The `[logging] level` of run and preprocessing configs: parse it and run code
at that level. `Preprocessing` and `Models` both use it, so this module is
loaded with the foundation modules before its consumers.
"""
module LogLevels

using ..ConfigChecks: config_int, parse_choice

using Logging
import Logging: min_enabled_level, shouldlog, handle_message, catch_exceptions

export log_level, with_log_level, LevelFilterLogger, progress_settings

"""
    progress_settings(section) -> (kind, interval)

Parse `[logging].progress` and the positive integer interval for line reports.
Reporter types belong to the runner; this module owns the configuration vocabulary.
"""
function progress_settings(section::AbstractDict)
    kind = parse_choice(get(section, "progress", "bar"),
                        (bar = :bar, lines = :lines, none = :none), "[logging].progress")
    interval = kind === :lines ?
        config_int(get(section, "progress_interval_windows", 24),
                   "[logging].progress_interval_windows"; min = 1) : nothing
    return (; kind, interval)
end

const LOG_LEVELS = Dict("debug" => Logging.Debug, "info" => Logging.Info,
                        "warn" => Logging.Warn, "error" => Logging.Error)

"""
    log_level(cfg) -> Union{Nothing, Logging.LogLevel}

The `[logging] level` of a config: `"debug"`, `"info"`, `"warn"` or `"error"`,
or `nothing` when it is not set (the caller's logger is then left as it is).
Anything else is an `ArgumentError`.
"""
function log_level(cfg::AbstractDict)
    section = get(cfg, "logging", Dict{String, Any}())
    section isa AbstractDict || throw(ArgumentError("[logging] must be a table"))
    haskey(section, "level") || return nothing
    raw = section["level"]
    level = raw isa AbstractString ? get(LOG_LEVELS, lowercase(raw), nothing) : nothing
    level === nothing && throw(ArgumentError(
        "[logging].level must be \"debug\", \"info\", \"warn\" or \"error\"; got $(repr(raw))"))
    return level
end

"""
    LevelFilterLogger(parent, min_level)

A logger that sets the minimum level of `parent`: it passes the messages at or
above `min_level` to `parent` (also below the parent's own minimum, so debug
messages reach a console logger) and drops the others.
"""
struct LevelFilterLogger{L <: AbstractLogger} <: AbstractLogger
    parent    :: L
    min_level :: LogLevel
end

min_enabled_level(logger::LevelFilterLogger) = logger.min_level
shouldlog(logger::LevelFilterLogger, level, args...) =
    level >= logger.min_level && shouldlog(logger.parent, level, args...)
handle_message(logger::LevelFilterLogger, args...; kwargs...) =
    handle_message(logger.parent, args...; kwargs...)
catch_exceptions(logger::LevelFilterLogger) = catch_exceptions(logger.parent)

"""
    with_log_level(f, level)

Run `f()` with the current logger at minimum level `level` (a
`Logging.LogLevel`); `level = nothing` leaves the logger as it is.
"""
with_log_level(f, level::LogLevel) = with_logger(f, LevelFilterLogger(current_logger(), level))
with_log_level(f, ::Nothing) = f()

end # module LogLevels
