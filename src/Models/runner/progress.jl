# ===========================================================================
# Forward-run progress timer — Transport vs IO wall-clock breakdown.
#
# Three accumulators (driver-open + window loads / transport / snapshot
# capture+write) and a progress reporter over windows (`[logging] progress`):
# a `ProgressMeter` bar, one log line every few windows, or nothing. The
# end-of-run summary lands via `@info` with the other run-completion logs.
# ===========================================================================

"""
    AbstractProgressReporter

How a run shows its progress over met windows (`[logging] progress`):
`ProgressBar()`, `ProgressLines(interval)` or `NoProgress()`.
"""
abstract type AbstractProgressReporter end

"A redrawn progress bar with status and timing lines (interactive terminals)."
struct ProgressBar <: AbstractProgressReporter end

"One `@info` line every `interval` windows and at the end (batch logs)."
struct ProgressLines <: AbstractProgressReporter
    interval :: Int
    function ProgressLines(interval::Integer)
        interval >= 1 || throw(ArgumentError("ProgressLines needs interval ≥ 1; got $(interval)"))
        return new(Int(interval))
    end
end

"No progress output; the end-of-run summary is still logged."
struct NoProgress <: AbstractProgressReporter end

mutable struct RunProgressTimer{R <: AbstractProgressReporter}
    reporter        :: R
    prog            :: Union{Nothing, Progress}   # the bar of `ProgressBar`
    windows_done    :: Int
    t_start         :: Float64
    t_io_read       :: Float64   # TransportBinaryDriver open + window loads
    t_transport     :: Float64   # advection + diffusion + convection + emissions
    t_io_write      :: Float64   # snapshot capture + final NetCDF write
    windows_total   :: Int
    status_line     :: String
    detail_line     :: String
    gc_ns_start     :: UInt64    # Base.gc_time_ns() at start
    compile_ns_start :: UInt64   # Base.cumulative_compile_time_ns()[1] at start
    bytes_start     :: Int64     # total allocated bytes (Base.gc_bytes) at start
    compile_timing  :: Bool      # true while this timer holds compile timing enabled
end

# Wall clock, cumulative GC time, cumulative compile time, and total allocated
# bytes, sampled back to back so start and end values cover the same interval.
# `Base.gc_bytes(::Ref{Int64})` overwrites the Ref with the process total.
function _sample_run_counters()
    bytes = Ref{Int64}(0)
    Base.gc_bytes(bytes)
    return (wall = time(), gc_ns = Base.gc_time_ns(),
            compile_ns = first(Base.cumulative_compile_time_ns()), bytes = bytes[])
end

function RunProgressTimer(total_windows::Integer;
                          reporter::AbstractProgressReporter = ProgressBar(),
                          label::AbstractString = "Forward run ")
    prog = reporter isa ProgressBar ?
           Progress(max(Int(total_windows), 1); desc = label, showspeed = true, barlen = 40) :
           nothing
    # Compile time is only accumulated while timing is enabled (as in `@time`).
    # The enable is a process-wide reference count; `stop_compile_timing!`
    # releases this timer's hold exactly once.
    Base.cumulative_compile_timing(true)
    start = _sample_run_counters()
    return RunProgressTimer(
        reporter, prog, 0, start.wall, 0.0, 0.0, 0.0, Int(total_windows),
        "initializing", "transport 0.0s | io_read 0.0s | io_write 0.0s",
        start.gc_ns, start.compile_ns, start.bytes, true)
end

# Release the compile-timing hold taken by the constructor. Idempotent:
# `summarize_progress!` calls it on success, and the runners call it again in
# `finally` so a run that throws does not leave compile timing enabled.
function stop_compile_timing!(timer::RunProgressTimer)
    timer.compile_timing || return timer
    timer.compile_timing = false
    Base.cumulative_compile_timing(false)
    return timer
end

@inline function _timed!(field::Symbol, timer::RunProgressTimer, f)
    t0 = time()
    val = f()
    delta = time() - t0
    setproperty!(timer, field, getproperty(timer, field) + delta)
    return val
end

# Mark IO read (e.g. opening a daily binary driver).
@inline timed_io_read!(timer, f) = _timed!(:t_io_read, timer, f)

# Mark transport (a single `run_window!` / `step!` block).
@inline timed_transport!(timer, f) = _timed!(:t_transport, timer, f)

# Mark IO write (snapshot capture + final NetCDF write).
@inline timed_io_write!(timer, f) = _timed!(:t_io_write, timer, f)

function _progress_detail_line(timer::RunProgressTimer)
    wall = max(time() - timer.t_start, eps())
    return @sprintf("transport %.1fs (%4.1f%%) | io_read %.1fs | io_write %.1fs | wall %.1fs",
                    timer.t_transport, 100 * timer.t_transport / wall,
                    timer.t_io_read, timer.t_io_write, wall)
end

@inline function _progress_showvalues(timer::RunProgressTimer)
    detail = isempty(timer.detail_line) ?
             _progress_detail_line(timer) :
             string(_progress_detail_line(timer), " | ", timer.detail_line)
    return [(:status, timer.status_line), (:timing, detail)]
end

function set_progress_status!(timer::RunProgressTimer;
                              status::Union{Nothing, AbstractString} = nothing,
                              detail::Union{Nothing, AbstractString} = nothing,
                              redraw::Bool = false)
    status === nothing || (timer.status_line = String(status))
    detail === nothing || (timer.detail_line = String(detail))
    redraw && timer.prog !== nothing &&
        update!(timer.prog, timer.prog.counter; showvalues = _progress_showvalues(timer))
    return timer
end

# Report one advanced window. With the bar, routine runtime status lives in
# the two redrawable lines below it so `@info` output does not interrupt the
# ETA rendering during long runs.
@inline function tick_window!(timer::RunProgressTimer;
                              status::Union{Nothing, AbstractString} = nothing,
                              detail::Union{Nothing, AbstractString} = nothing)
    status === nothing || (timer.status_line = String(status))
    detail === nothing || (timer.detail_line = String(detail))
    timer.windows_done += 1
    _report_window!(timer.reporter, timer)
    return timer
end

_report_window!(::ProgressBar, timer) = next!(timer.prog; showvalues = [
    (:status, timer.status_line),
    (:timing, string(_progress_detail_line(timer), " | ", timer.detail_line)),
])
_report_window!(::NoProgress, timer) = nothing

function _report_window!(reporter::ProgressLines, timer)
    n, total = timer.windows_done, max(timer.windows_total, timer.windows_done)
    n % reporter.interval == 0 || n == total || return nothing
    elapsed = time() - timer.t_start
    @info @sprintf("Window %d/%d (%.0f%%), elapsed %s, ETA %s | %s | %s", n, total, 100n / total,
                   _clock_string(elapsed), _clock_string(elapsed / n * (total - n)),
                   timer.status_line, _progress_detail_line(timer))
    return nothing
end

# Seconds as h:mm:ss.
function _clock_string(seconds::Real)
    s = round(Int, max(seconds, 0))
    return @sprintf("%d:%02d:%02d", s ÷ 3600, (s % 3600) ÷ 60, s % 60)
end

function summarize_progress!(timer::RunProgressTimer)
    timer.prog === nothing || finish!(timer.prog)
    stop = _sample_run_counters()
    stop_compile_timing!(timer)
    wall = stop.wall - timer.t_start
    accounted = timer.t_io_read + timer.t_transport + timer.t_io_write
    other = max(wall - accounted, 0.0)
    w = max(wall, eps())
    msg = @sprintf("Forward run wall %.1fs   transport %.1fs (%.1f%%)   io_read %.1fs (%.1f%%)   io_write %.1fs (%.1f%%)   other %.1fs (%.1f%%)", wall, timer.t_transport, 100*timer.t_transport/w, timer.t_io_read, 100*timer.t_io_read/w, timer.t_io_write, 100*timer.t_io_write/w, other, 100*other/w)
    # Overlapping wall-clock shares: GC pauses and JIT compilation fall inside
    # the sections above. Compile time dominates short runs in a fresh process.
    gc_s = (stop.gc_ns - timer.gc_ns_start) / 1e9
    compile_s = (stop.compile_ns - timer.compile_ns_start) / 1e9
    alloc_gib = (stop.bytes - timer.bytes_start) / 2^30
    msg *= @sprintf("\n  of which GC %.1fs (%.1f%%), compilation %.1fs (%.1f%%); allocated %.1f GiB",
                    gc_s, 100*gc_s/w, compile_s, 100*compile_s/w, alloc_gib)
    @info msg
    return timer
end
