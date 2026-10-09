# ===========================================================================
# Forward-run progress timer — Transport vs IO wall-clock breakdown.
#
# Three accumulators (driver-open + window loads / transport / snapshot
# capture+write) plus a `ProgressMeter.Progress` bar over windows. Always
# on — no env var gating, no SectionTimer dep. End-of-run summary lands
# via `@info` so it surfaces alongside the existing run-completion logs.
# ===========================================================================

mutable struct RunProgressTimer
    prog            :: Progress
    t_start         :: Float64
    t_io_read       :: Float64   # TransportBinaryDriver open + window loads
    t_transport     :: Float64   # advection + diffusion + convection + emissions
    t_io_write      :: Float64   # snapshot capture + final NetCDF write
    windows_total   :: Int
    status_line     :: String
    detail_line     :: String
end

RunProgressTimer(total_windows::Integer; label::AbstractString = "Forward run ") =
    RunProgressTimer(
        Progress(max(Int(total_windows), 1);
                  desc = label, showspeed = true, barlen = 40),
        time(), 0.0, 0.0, 0.0, Int(total_windows),
        "initializing", "transport 0.0s | io_read 0.0s | io_write 0.0s")

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
    redraw && update!(timer.prog, timer.prog.counter;
                      showvalues = _progress_showvalues(timer))
    return timer
end

# Tick the progress bar after one window has advanced. Keep routine runtime
# status in the two redrawable lines below the bar so `@info` output does not
# interrupt ETA/progress rendering during long runs.
@inline function tick_window!(timer::RunProgressTimer;
                              status::Union{Nothing, AbstractString} = nothing,
                              detail::Union{Nothing, AbstractString} = nothing)
    status === nothing || (timer.status_line = String(status))
    detail === nothing || (timer.detail_line = String(detail))
    next!(timer.prog; showvalues = [
        (:status, timer.status_line),
        (:timing, string(_progress_detail_line(timer), " | ", timer.detail_line)),
    ])
end

function summarize_progress!(timer::RunProgressTimer)
    finish!(timer.prog)
    wall = time() - timer.t_start
    accounted = timer.t_io_read + timer.t_transport + timer.t_io_write
    other = max(wall - accounted, 0.0)
    w = max(wall, eps())
    msg = @sprintf("Forward run wall %.1fs   transport %.1fs (%.1f%%)   io_read %.1fs (%.1f%%)   io_write %.1fs (%.1f%%)   other %.1fs (%.1f%%)", wall, timer.t_transport, 100*timer.t_transport/w, timer.t_io_read, 100*timer.t_io_read/w, timer.t_io_write, 100*timer.t_io_write/w, other, 100*other/w)
    @info msg
    return timer
end
