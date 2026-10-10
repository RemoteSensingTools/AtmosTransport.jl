# ===========================================================================
# `[logging]` and `[checks]` of a run config: how much a run reports and which
# runtime checks it carries, after Oceananigans' log levels and `NaNChecker`.
# ===========================================================================

"""
    RunLogging(level, progress)

Parsed `[logging]` table: the minimum log level of the run (`nothing` when
unset: the caller's logger is left as it is) and its progress reporter
(`ProgressBar()` by default).
"""
struct RunLogging{R <: AbstractProgressReporter}
    level    :: Union{Nothing, LogLevel}
    progress :: R
end

"""
    run_logging(cfg) -> RunLogging

Parse `[logging]`: `level` (`"debug"`, `"info"`, `"warn"` or `"error"`),
`progress` (`"bar"`, `"lines"` or `"none"`) and, for `"lines"`,
`progress_interval_windows` (default 24).
"""
function run_logging(cfg::AbstractDict)
    level = log_level(cfg)             # also rejects a [logging] that is not a table
    return RunLogging(level, _progress_reporter(get(cfg, "logging", Dict{String, Any}())))
end

function _progress_reporter(section)
    raw = get(section, "progress", "bar")
    kind = raw isa AbstractString ? lowercase(raw) : nothing
    kind == "bar" && return ProgressBar()
    kind == "none" && return NoProgress()
    kind == "lines" &&
        return ProgressLines(_positive_int(section, "progress_interval_windows", 24, "[logging]"))
    throw(ArgumentError("[logging].progress must be \"bar\", \"lines\" or \"none\"; got $(repr(raw))"))
end

function _positive_int(section, key, default, label)
    value = _spec_int(section, key, default, label)    # rejects Booleans
    value >= 1 || throw(ArgumentError("$(label).$(key) must be at least 1; got $(value)"))
    return value
end

"""
    run_checks(cfg) -> NamedTuple

Parse `[checks]` into the run's callbacks: `nan_check` (default `true`)
installs `Callback(NaNChecker(), WindowInterval(nan_check_interval_windows))`
(default every window), which stops the run on a NaN or Inf in the air mass or
a tracer.
"""
function run_checks(cfg::AbstractDict)
    section = get(cfg, "checks", Dict{String, Any}())
    section isa AbstractDict || throw(ArgumentError("[checks] must be a table"))
    _config_bool(section, "nan_check", true, "[checks].nan_check") || return NamedTuple()
    interval = _positive_int(section, "nan_check_interval_windows", 1, "[checks]")
    return (nan_checker = Callback(NaNChecker(), WindowInterval(interval)),)
end
