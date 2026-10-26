#!/usr/bin/env julia
# Schedules, callbacks and the NaN checker of a driven simulation, and the
# run config's `[logging]` and `[checks]` tables (after Oceananigans).

using Test
using Logging
using AtmosTransport
using .AtmosTransport.State: DryBasis, get_tracer

const M = AtmosTransport.Models
const R = AtmosTransport.Models.DrivenRunner

include(joinpath(@__DIR__, "..", "fixtures", "window_prefetch.jl"))

# The fields a schedule or the NaN checker reads.
mutable struct FakeSim{S}
    iteration                   :: Int
    current_window_end_iteration :: Int
    current_window_index        :: Int
    start_window                :: Int
    time                        :: Float64
    model                       :: S
end
FakeSim(; iteration = 0, window_end = 2, window = 1, start = 1, state = nothing) =
    FakeSim(iteration, window_end, window, start, 3600.0 * iteration, (; state))

@testset "schedules" begin
    every3 = IterationInterval(3)
    @test [every3(FakeSim(; iteration = i)) for i in 1:6] == [false, false, true, false, false, true]
    @test_throws ArgumentError IterationInterval(0)

    # Window ends only, every second window this schedule has seen (the count
    # does not depend on the simulation's own window index).
    w2 = WindowInterval(2)
    window_end(w) = FakeSim(; iteration = 2w, window_end = 2w, window = 1)
    @test !w2(FakeSim(; iteration = 1, window_end = 2))          # mid-window step
    @test [w2(window_end(w)) for w in 1:4] == [false, true, false, true]
    @test_throws ArgumentError WindowInterval(0)
    @test_throws ArgumentError WindowInterval(true)
    @test_throws ArgumentError IterationInterval(true)
    @test_throws ArgumentError WallTimeInterval(true)

    wall = WallTimeInterval(3600.0)
    @test !wall(FakeSim())
    wall.previous -= 7200.0                                      # an hour and more ago
    @test wall(FakeSim())
    @test !wall(FakeSim())                                       # restarted
    @test_throws ArgumentError WallTimeInterval(0)

    calls = Ref(0)
    cb = Callback(sim -> (calls[] += 1), IterationInterval(2))
    foreach(i -> cb(FakeSim(; iteration = i)), 1:5)
    @test calls[] == 2
    @test Callback(identity).schedule isa IterationInterval
end

@testset "NaNChecker" begin
    FT = Float64
    check(state) = NaNChecker()(FakeSim(; iteration = 2, window = 3, state))

    # Lat-lon cell state.
    m = ones(FT, 4, 3, 2)
    state = CellState(DryBasis, copy(m); co2 = fill(4e-4, 4, 3, 2), ch4 = fill(2e-6, 4, 3, 2))
    @test check(state) === nothing
    get_tracer(state, :ch4)[2, 2, 1] = NaN
    err = try check(state); nothing catch e; e end
    @test err isa ErrorException && contains(err.msg, "tracer ch4") &&
          contains(err.msg, "met window 3")
    get_tracer(state, :ch4)[2, 2, 1] = 2e-6
    state.air_mass[1] = Inf
    err = try check(state); nothing catch e; e end
    @test err isa ErrorException && contains(err.msg, "the air mass")

    # Cubed sphere: halo cells are not checked.
    mesh = CubedSphereMesh(; FT, Nc = 4)
    N = mesh.Nc + 2mesh.Hp
    cs = CubedSphereState(DryBasis, mesh, ntuple(_ -> ones(FT, N, N, 2), 6);
                          co2 = ntuple(_ -> fill(4e-4, N, N, 2), 6))
    cs.air_mass[2][1, 1, 1] = NaN                 # a halo corner
    @test check(cs) === nothing
    cs.tracers_raw[5][mesh.Hp + 1, mesh.Hp + 2, 2, 1] = NaN
    err = try check(cs); nothing catch e; e end
    @test err isa ErrorException && contains(err.msg, "tracer co2")

    # A halo NaN in one tracer does not take the blame for another's interior NaN.
    cs2 = CubedSphereState(DryBasis, mesh, ntuple(_ -> ones(FT, N, N, 2), 6);
                           co2 = ntuple(_ -> fill(4e-4, N, N, 2), 6),
                           ch4 = ntuple(_ -> fill(2e-6, N, N, 2), 6))
    cs2.tracers_raw[1][1, 1, 1, 1] = NaN                        # co2 halo
    cs2.tracers_raw[2][mesh.Hp + 2, mesh.Hp + 2, 1, 2] = NaN    # ch4 interior
    err = try check(cs2); nothing catch e; e end
    @test err isa ErrorException && contains(err.msg, "tracer ch4")

    # One checker over several checks keeps its flag and clears it after a hit.
    checker = NaNChecker()
    run_check(state) = checker(FakeSim(; iteration = 2, window = 3, state))
    cs.tracers_raw[5][mesh.Hp + 1, mesh.Hp + 2, 2, 1] = 4e-4    # undo the NaN above
    @test run_check(cs) === nothing && checker.flag isa Vector{Int32}
    flag = checker.flag
    cs.tracers_raw[5][mesh.Hp + 1, mesh.Hp + 2, 2, 1] = NaN
    @test_throws ErrorException run_check(cs)
    cs.tracers_raw[5][mesh.Hp + 1, mesh.Hp + 2, 2, 1] = 4e-4
    @test run_check(cs) === nothing && checker.flag === flag && iszero(only(flag))
    # Another state (possibly on another device) gets its own flag.
    @test run_check(deepcopy(cs)) === nothing && checker.flag !== flag
end

@testset "a driven simulation runs its callbacks on their schedules" begin
    model, driver = WindowPrefetchFixtures.prefetch_fixture()
    for w in driver.windows, a in (w.fluxes.am, w.fluxes.bm, w.fluxes.cm)
        fill!(a, 0)
    end
    window_ends = Ref(0)
    steps = Ref(0)
    sim = DrivenSimulation(model, driver;
                           callbacks = (ends = Callback(sim -> (window_ends[] += 1), WindowInterval(1)),
                                        every = sim -> (steps[] += 1)))
    run!(sim)
    @test window_ends[] == 2                     # two windows
    @test steps[] == 4                           # two steps each

    # A NaN that appears mid-window stops the run at the window end.
    model, driver = WindowPrefetchFixtures.prefetch_fixture()
    poison = Callback(sim -> (get_tracer(sim.model.state, :co2)[1] = NaN), IterationInterval(3))
    sim = DrivenSimulation(model, driver;
                           callbacks = (poison = poison,
                                        nan_checker = Callback(NaNChecker(), WindowInterval(1))))
    err = try run!(sim); nothing catch e; e end
    @test err isa ErrorException && contains(err.msg, "tracer co2") &&
          contains(err.msg, "met window 2")

    # One callback shared by the simulations of two input files (as the runner
    # does) counts windows over both: interval 3 actuates once, at window 3.
    actuated = Ref(0)
    shared = (every3 = Callback(sim -> (actuated[] += 1), WindowInterval(3)),)
    for _ in 1:2
        model, driver = WindowPrefetchFixtures.prefetch_fixture()
        for w in driver.windows, a in (w.fluxes.am, w.fluxes.bm, w.fluxes.cm)
            fill!(a, 0)
        end
        run!(DrivenSimulation(model, driver; callbacks = shared))
    end
    @test actuated[] == 1
end

@testset "[logging] and [checks]" begin
    logging(d...) = R.run_logging(Dict{String, Any}("logging" => Dict{String, Any}(d...)))
    @test R.run_logging(Dict{String, Any}()) == R.RunLogging(nothing, R.ProgressBar())
    @test logging("level" => "info").level == Logging.Info
    @test logging("level" => "WARN").level == Logging.Warn
    @test logging("level" => "debug").level == Logging.Debug
    @test logging("progress" => "none").progress === R.NoProgress()
    @test logging("progress" => "lines").progress == R.ProgressLines(24)
    @test logging("progress" => "lines", "progress_interval_windows" => 6).progress ==
          R.ProgressLines(6)
    for bad in (("level" => "verbose",), ("level" => 1,), ("progress" => "dots",),
                ("progress" => "lines", "progress_interval_windows" => 0),
                ("progress" => "lines", "progress_interval_windows" => 1.5),
                ("progress" => "lines", "progress_interval_windows" => true))
        @test_throws ArgumentError logging(bad...)
    end

    checks(d...) = R.run_checks(Dict{String, Any}("checks" => Dict{String, Any}(d...)))
    default = R.run_checks(Dict{String, Any}())
    @test keys(default) == (:nan_checker,)
    @test default.nan_checker.func isa NaNChecker
    @test default.nan_checker.schedule.interval == 1
    @test checks("nan_check_interval_windows" => 24).nan_checker.schedule.interval == 24
    @test checks("nan_check" => false) == NamedTuple()
    @test_throws ArgumentError checks("nan_check" => "yes")
    @test_throws ArgumentError checks("nan_check_interval_windows" => 0)
    @test_throws ArgumentError checks("nan_check_interval_windows" => true)
    @test_throws ArgumentError R.run_checks(Dict{String, Any}("checks" => "on"))

    # Unknown keys are errors; a key the settings leave unread is a warning.
    f = R._config_key_findings(Dict{String, Any}(
        "logging" => Dict("levle" => "warn", "progress_interval_windows" => 6),
        "checks" => Dict("nan_check" => false, "nan_check_interval_windows" => 2)))
    @test any(contains("levle (did you mean `level`?)"), f.errors)
    @test any(contains("progress_interval_windows"), f.warnings)
    @test any(contains("nan_check_interval_windows"), f.warnings)
end

@testset "log levels" begin
    LL = AtmosTransport.LogLevels
    emit() = (@debug "debug message"; @info "info message"; @warn "warn message"; 42)
    levels(level) = begin
        logs, value = Test.collect_test_logs(min_level = Logging.Debug) do
            LL.with_log_level(emit, level)
        end
        @test value == 42
        [l.level for l in logs]
    end
    @test levels(Logging.Warn) == [Logging.Warn]
    @test levels(Logging.Error) == []
    @test levels(Logging.Debug) == [Logging.Debug, Logging.Info, Logging.Warn]
    @test levels(Logging.Info) == [Logging.Info, Logging.Warn]       # also under a debug logger
    @test levels(nothing) == [Logging.Debug, Logging.Info, Logging.Warn]  # unset: logger untouched
    # Debug reaches a console logger whose own minimum is Info: the run keeps
    # the caller's logger and only sets its level.
    io = IOBuffer()
    with_logger(ConsoleLogger(io, Logging.Info)) do
        LL.with_log_level(emit, Logging.Debug)
    end
    @test contains(String(take!(io)), "debug message")

    # Preprocessing configs read `[logging] level` before doing anything else.
    @test LL.log_level(Dict{String, Any}()) === nothing
    @test_throws ArgumentError LL.log_level(Dict{String, Any}("logging" => "quiet"))
    err = try
        AtmosTransport.Preprocessing.process_day(Dict{String, Any}("logging" => Dict("level" => "loud")))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError && contains(err.msg, "[logging].level")
end

@testset "progress reporters" begin
    lines(reporter, total, ticks) = begin
        timer = R.RunProgressTimer(total; reporter)
        logs, _ = Test.collect_test_logs() do
            foreach(_ -> R.tick_window!(timer; status = "day 1"), 1:ticks)
        end
        R.stop_compile_timing!(timer)
        [string(l.message) for l in logs]
    end
    out = lines(R.ProgressLines(2), 5, 5)
    @test length(out) == 3                       # windows 2, 4 and the last
    @test startswith(out[1], "Window 2/5 (40%)") && contains(out[1], "day 1")
    @test startswith(out[3], "Window 5/5 (100%)")
    @test isempty(lines(R.NoProgress(), 5, 5))
    # An underestimated total does not report more than 100 %.
    @test startswith(last(lines(R.ProgressLines(1), 2, 3)), "Window 3/3 (100%)")
end
