# RunProgressTimer end-of-run summary: GC, compilation, and allocation fields,
# and the idempotent release of the process-wide compile-timing hold that the
# constructor takes (`summarize_progress!` and the runners' `finally` both
# call `stop_compile_timing!`).

using Test, AtmosTransport, Logging

const R = AtmosTransport.Models.DrivenRunner

# Run `f` with stderr silenced (the progress bar draws there) and return its
# value together with the log records it emitted.
function captured_logs(f)
    logger = Test.TestLogger(; min_level = Logging.Info)
    value = with_logger(logger) do
        redirect_stderr(f, devnull)
    end
    return value, logger.logs
end

const SUMMARY_COUNTERS =
    r"of which GC (\S+)s \((\S+)%\), compilation (\S+)s \((\S+)%\); allocated (\S+) GiB"

@testset "RunProgressTimer summary reports GC, compilation, allocation" begin
    timer, logs = captured_logs() do
        t = R.RunProgressTimer(2; label = "progress test ")
        @test t.compile_timing
        @test sum(sum, [rand(64) for _ in 1:64]) > 0
        R.tick_window!(t)
        R.tick_window!(t)
        R.summarize_progress!(t)
    end
    @test !timer.compile_timing

    summaries = [string(log.message) for log in logs
                 if occursin("Forward run wall", string(log.message))]
    @test length(summaries) == 1
    counters = match(SUMMARY_COUNTERS, only(summaries))
    @test counters !== nothing
    values = parse.(Float64, counters.captures)
    @test length(values) == 5
    @test all(>=(0), values)
end

@testset "RunProgressTimer compile-timing release is idempotent" begin
    timer = redirect_stderr(() -> R.RunProgressTimer(1), devnull)
    @test timer.compile_timing
    # Failure path: the runner's `finally` releases the hold without a summary.
    R.stop_compile_timing!(timer)
    @test !timer.compile_timing
    R.stop_compile_timing!(timer)
    @test !timer.compile_timing
    # A summary after the release does not release a second time.
    captured_logs(() -> R.summarize_progress!(timer))
    @test !timer.compile_timing
end
