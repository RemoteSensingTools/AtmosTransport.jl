#!/usr/bin/env julia
# The golden harness (test/golden) runs its cases in worker sessions: one Julia
# process per group of cases that share what is fixed at process start. These
# tests cover the grouping and the worker's per-case isolation with stand-in
# entry scripts; the real cases need the group's data and a GPU
# (test/golden/README.md).

using Test
using TOML
using Logging

const GOLDEN_DIR = normpath(joinpath(@__DIR__, "..", "golden"))

function _golden_module(name::Symbol, file::AbstractString)
    mod = Module(name)
    Core.eval(mod, :(include(path::AbstractString) = Base.include($mod, path)))
    Core.eval(mod, :(eval(expr) = Core.eval($mod, expr)))
    Base.include(mod, joinpath(GOLDEN_DIR, file))
    return mod
end

const Harness = _golden_module(:GoldenHarness, "run_goldens.jl")
const Worker = _golden_module(:GoldenWorker, "golden_worker.jl")

@testset "golden cases share a session exactly when their process start agrees" begin
    cases = TOML.parsefile(Harness.CASES_FILE)["case"]
    mktempdir() do dir
        cfgs = [Harness.case_config(c, joinpath(dir, c["name"]), dir) for c in cases]
        keys = [Harness.session_key(c, cfg, 8) for (c, cfg) in zip(cases, cfgs)]
        groups = Harness.session_groups(keys)
        @test sort!(reduce(vcat, groups)) == collect(eachindex(cases))   # every case once
        @test all(issorted, groups)                                      # case order kept
        @test length(unique(keys)) == length(groups)
        # For the current cases a session is one entry point on CPU or on the GPU.
        kind_and_gpu(c) = (c["kind"], "gpu" in c["tags"])
        @test all(g -> allequal(kind_and_gpu.(cases[g])), groups)
        @test length(groups) == length(unique(kind_and_gpu.(cases)))
        @test all(c -> c["kind"] == "preprocess", cases[groups[1]])
        @test Harness.check_session_order(cases, groups) === nothing     # a record's order holds
    end
end

@testset "session keys, groups and the dependency order" begin
    key(kind, arch; threads = 8) = (kind = kind, architecture = arch, threads = threads)
    @test Harness.session_groups([key("preprocess", "default"), key("run", "cpu"), key("run", "gpu"),
                                  key("run", "cpu"), key("preprocess", "default")]) == [[1, 5], [2, 4], [3]]
    @test Harness.session_groups([key("run", "cpu"), key("run", "cpu"; threads = 4)]) == [[1], [2]]
    @test Harness.architecture_label(Dict{String, Any}()) == "default"
    @test Harness.architecture_label(Dict("architecture" => Dict("use_gpu" => true, "backend" => "cuda"))) ==
          "backend=cuda use_gpu=true"

    cases = TOML.parsefile(Harness.CASES_FILE)["case"]
    case(name) = only(filter(c -> c["name"] == name, cases))
    @test Harness.case_dependencies(case("run_ll72_ppm_cpu_f64")) == ["pre_ll72"]           # from `set`
    @test Harness.case_dependencies(case("run_c90_merra2_gpu_f32")) == ["pre_merra2_c90"]   # from the template
    @test isempty(Harness.case_dependencies(case("pre_ll72")))
    reader, producer = case("run_c90_merra2_gpu_f32"), case("pre_merra2_c90")
    @test Harness.check_session_order([reader, producer], [[2], [1]]) === nothing
    @test Harness.check_session_order([reader, producer], [[2, 1]]) === nothing
    @test_throws ErrorException Harness.check_session_order([reader, producer], [[1], [2]])
    @test_throws ErrorException Harness.check_session_order([reader, producer], [[1, 2]])
    @test Harness.case_entry(producer, "c.toml") ==
          (joinpath(Harness.REPO, "scripts", "preprocessing", "preprocess_transport_binary.jl"),
           ["c.toml", "--day", "2022-01-15"])
    @test Harness.case_entry(reader, "c.toml") == (joinpath(Harness.REPO, "scripts", "run_transport.jl"), ["c.toml"])
end

# A stand-in entry script: writes to stdout, stderr, C stdio and the logging
# system (also from a spawned task), counts its calls and, depending on its
# argument, fails, changes process state, exits, is killed or hangs ("hang"
# first creates `<args[2]>.hanging`). "poison" and "die_between" install a
# stand-in CUDA module (only ever in a worker process) whose `synchronize`,
# called when the worker frees memory after the case, throws a
# sticky-error-like exception or ends the process with status 0.
const STAND_IN = """
using Logging
const calls = Ref(0)
function main(args::Vector{String})
    calls[] += 1
    println("stdout of ", args[1], ", call ", calls[], ", in ", pwd())
    println(stderr, "stderr of ", args[1])
    @info "log of \$(args[1])" cache = get(ENV, "ATMOSTR_REGRID_CACHE_DIR", "unset")
    wait(Threads.@spawn @warn "warning from a task of \$(args[1])")
    ccall(:puts, Cint, (Cstring,), "C output of " * args[1])
    args[1] == "fail" && error("stand-in failure")
    if args[1] == "leak"                   # changes all the process state the worker restores
        ENV["GOLDEN_TEST_LEAK"] = "1"
        global_logger(SimpleLogger(stderr))
        cd(Sys.BINDIR)
    end
    args[1] == "crash" && exit(3)
    args[1] == "killed" && ccall(:raise, Cint, (Cint,), 9)   # SIGKILL, as from the OOM killer
    args[1] == "hang" && (touch(args[2] * ".hanging"); sleep(600))
    if args[1] in ("poison", "die_between")
        cuda = Module(:CUDA)
        Core.eval(cuda, :(functional() = true))
        Core.eval(cuda, :(reclaim() = nothing))
        Core.eval(cuda, args[1] == "poison" ?
            :(synchronize() = error("CUDA error: an illegal memory access was encountered (code 700, ERROR_ILLEGAL_ADDRESS)")) :
            :(synchronize() = exit(0)))
        Base.loaded_modules[Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")] = cuda
    end
    return zeros(4)
end
"""

"""
    run_stand_in_session(dir, names; script, kwargs...) -> (ok, summary, output)

Run the harness's `run_session` on one stand-in case per name (the name is the
script's first argument), with the workers' output in `output`.
"""
function run_stand_in_session(dir, names; script = joinpath(dir, "entry.jl"), kwargs...)
    isfile(script) || write(script, STAND_IN)
    cases = [Dict{String, Any}("name" => n, "kind" => "run") for n in names]
    cfgs = [Dict{String, Any}("output" => Dict("path" => joinpath(dir, n, "out.nc"))) for n in names]
    entry(case, config_path) = (script, [case["name"], config_path])
    output = joinpath(dir, "harness_output.txt")             # the workers' progress and errors
    ok, summary = open(output, "w") do io
        redirect_stdout(io) do
            redirect_stderr(() -> Harness.run_session(cases, cfgs, dir, 1, "stand-in session"; entry, kwargs...), io)
        end
    end
    return ok, summary, read(output, String)
end

stand_in_status(dir, name) = TOML.parsefile(joinpath(dir, name, "status.toml"))
stand_in_position(dir, name) = (s = stand_in_status(dir, name)["session"]; (s["position"], s["cases"]))

# Fake workers for the timeout paths: shell scripts that start in milliseconds,
# so the timeouts can be short. The k-th worker the harness starts runs
# `scripts[k]` with the plan and results paths as `$1` and `$2`.
function fake_worker(scripts)
    started = Ref(0)
    command(plan_path, results_path, threads) =
        (started[] += 1; `/bin/sh -c $(scripts[started[]]) sh $plan_path $results_path`)
    return command, started
end

"A shell command that publishes `report` as the results file, atomically like the worker."
function publish(dir, report)
    path = tempname(dir) * ".toml"
    open(io -> TOML.print(io, report), path, "w")
    return "cp '$path' \"\$2.tmp\" && mv \"\$2.tmp\" \"\$2\""
end

fake_result(name, position) = Dict{String, Any}("name" => name, "ok" => true, "wall_seconds" => 0.1, "error" => "",
                                                "finished" => "2026-01-01T00:00:00", "position" => position,
                                                "restored" => String[])
fake_report(results...; kw...) = Dict{String, Any}("case" => collect(results), "load_seconds" => 0.1,
                                                   (String(k) => v for (k, v) in kw)...)
const FAST = (timeout = 1.0, poll = 0.1, grace = 1.0)

@testset "worker isolates each case and continues after a failure" begin
    mktempdir() do dir
        script, broken = joinpath(dir, "entry.jl"), joinpath(dir, "broken.jl")
        write(script, STAND_IN)
        write(broken, "main(args) = (\n")                 # does not parse
        names = ["first", "fail", "broken", "leak", "last"]
        plan = Dict("label" => "stand-in", "case" => [
            Dict("name" => n, "script" => n == "broken" ? broken : script, "args" => [n],
                 "log" => joinpath(dir, "$n.log"),
                 "env" => Dict("ATMOSTR_REGRID_CACHE_DIR" => joinpath(dir, "cache_$n"))) for n in names])
        plan_path, results_path = joinpath(dir, "plan.toml"), joinpath(dir, "results.toml")
        open(io -> TOML.print(io, plan), plan_path, "w")
        logger, env, cwd = global_logger(), copy(ENV), pwd()

        results = with_logger(() -> Worker.run_session(plan_path, results_path), NullLogger())

        @test [r["name"] for r in results] == names
        @test [r["ok"] for r in results] == [true, false, false, true, true]
        @test [r["position"] for r in results] == 1:5
        @test all(r -> r["wall_seconds"] > 0, results)
        @test occursin("stand-in failure", results[2]["error"])
        @test occursin("ParseError", results[3]["error"])
        @test all(r -> isempty(r["error"]), results[[1, 4, 5]])
        @test results[4]["restored"] == ["working directory", "global logger", "ENV[\"GOLDEN_TEST_LEAK\"]"]
        @test all(r -> isempty(r["restored"]), results[[1, 2, 3, 5]])
        written = TOML.parsefile(results_path)
        @test !haskey(written, "running") && written["load_seconds"] > 0
        @test [r["ok"] for r in written["case"]] == [true, false, false, true, true]

        log(n) = read(joinpath(dir, "$n.log"), String)
        first_log = log("first")
        for text in ("stdout of first, call 1, in $cwd", "stderr of first", "log of first",
                     joinpath(dir, "cache_first"), "warning from a task of first", "C output of first")
            @test occursin(text, first_log)
        end
        @test !occursin("stdout of fail", first_log)
        @test occursin("ERROR: stand-in failure", log("fail")) && occursin("Stacktrace", log("fail"))
        @test occursin("ParseError", log("broken")) && occursin("broken.jl", log("broken"))
        @test occursin("call 4", log("last"))            # the script was included once per session
        @test occursin(joinpath(dir, "cache_last"), log("last"))

        @test global_logger() === logger                  # the process is as before
        @test copy(ENV) == env
        @test pwd() == cwd
    end
end

@testset "harness writes each status and survives a worker that dies" begin
    mktempdir() do dir
        names = ["first", "crash", "fail", "last"]
        ok, summary, output = run_stand_in_session(dir, names)

        @test ok == [true, false, false, true] || print(output)
        @test length(summary) == 2                        # the crash started a second worker
        @test occursin("worker exited with 3 during crash", summary[1])
        status(n) = stand_in_status(dir, n)
        for n in names
            @test issubset(["ok", "wall_seconds", "finished", "commit", "dirty", "host", "scrubbed_environment",
                            "environment", "inputs", "nonfinite", "session"], keys(status(n)))
            @test status(n)["session"]["label"] == "stand-in session"
            @test TOML.parsefile(joinpath(dir, n, "config.toml")) ==
                  Dict("output" => Dict("path" => joinpath(dir, n, "out.nc")))
        end
        session(n) = stand_in_position(dir, n)
        @test session("first") == (1, 4) && session("crash") == (2, 4)
        @test session("fail") == (1, 2) && session("last") == (2, 2)
        @test occursin("exited with 3 during the case", status("crash")["error"])
        @test occursin("stand-in failure", status("fail")["error"])
        @test !haskey(status("first"), "error")
        @test occursin("stdout of first", read(joinpath(dir, "first", "log.txt"), String))
        @test occursin(joinpath(dir, "first", "_regrid_cache"), read(joinpath(dir, "first", "log.txt"), String))
        @test Harness.wall_label(joinpath(dir, "last")) == "$(round(Int, status("last")["wall_seconds"])) s (session case 2 of 2)"
        @test Harness.wall_label(joinpath(dir, "crash")) == "failed"     # its wall time is NaN
        @test Harness.wall_label(joinpath(dir, "fail")) == "failed"
        @test Harness.wall_label(joinpath(dir, "never")) == "not run"
        @test !occursin("NaN", output)
    end
end

@testset "harness names the signal of a killed worker" begin
    mktempdir() do dir
        ok, summary, output = run_stand_in_session(dir, ["killed", "after"])

        @test ok == [false, true] || print(output)
        @test length(summary) == 2                        # the kill started a second worker
        @test occursin("was killed by signal 9 during the case", stand_in_status(dir, "killed")["error"])
    end
end

@testset "harness fails the cases of a worker that dies before its first case" begin
    mktempdir() do dir
        script = joinpath(dir, "exits.jl")
        write(script, "exit(5)\n")                       # ends the worker while it loads the entry scripts
        ok, summary, output = run_stand_in_session(dir, ["a", "b"]; script)

        @test ok == [false, false] || print(output)
        @test length(summary) == 1                        # a second worker would die the same way
        @test occursin("0 of  2 cases", summary[1]) && occursin("loading failed", summary[1])
        @test occursin("worker exited with 5 before its first case", summary[1])
        for n in ("a", "b")
            @test occursin("exited with 5 before the case", stand_in_status(dir, n)["error"])
            @test stand_in_position(dir, n) == (0, 2)
            @test Harness.wall_label(joinpath(dir, n)) == "failed"
        end
        @test !occursin("NaN", output)
    end
end

@testset "harness continues in a new worker after one that dies between cases" begin
    mktempdir() do dir
        ok, summary, output = run_stand_in_session(dir, ["die_between", "after"])

        @test ok == [true, true] || print(output)         # the case had finished; the rest ran anew
        @test length(summary) == 2
        @test occursin("worker exited with 0 after die_between", summary[1])   # status 0 but cases left
        @test stand_in_position(dir, "die_between") == (1, 2) && stand_in_position(dir, "after") == (1, 1)
        @test occursin("continuing the remaining 1 in a new worker", output)
    end
end

@testset "a failing CUDA cleanup poisons the session; the rest run in a new worker" begin
    mktempdir() do dir
        ok, summary, output = run_stand_in_session(dir, ["poison", "after"]; poll = 0.5)

        @test ok == [true, true] || print(output)         # the case's outcome stands
        @test length(summary) == 2
        @test occursin("session poisoned after poison", summary[1])
        poisoned = get(stand_in_status(dir, "poison")["session"], "poisoned", "")
        @test occursin("freeing memory after case poison failed", poisoned)
        @test occursin("ERROR_ILLEGAL_ADDRESS", poisoned)  # the status is rewritten with the poisoning
        @test !haskey(stand_in_status(dir, "after")["session"], "poisoned")
        @test stand_in_position(dir, "after") == (1, 1)
        @test occursin("the remaining cases continue in a new worker", output)
        @test occursin("Stacktrace", output)              # the worker logs the failure with its backtrace
    end
end

@testset "watch_worker stops a hung worker: SIGTERM, then SIGKILL" begin
    mktempdir() do dir
        results = joinpath(dir, "results.toml")
        watch(script; kw...) = Harness.watch_worker(_ -> nothing, `/bin/sh -c $script sh plan $results`, results;
                                                    FAST..., kw...)
        w = watch("exec sleep 30")
        @test w.stopped === :timeout && w.process.termsignal == 15 && w.wall < 10
        w = watch("trap '' TERM; exec sleep 30")         # ignores SIGTERM: killed after the grace
        @test w.stopped === :timeout && w.process.termsignal == 9 && 2 <= w.wall < 10
        w = watch("exit 0")
        @test w.stopped === nothing && success(w.process)
        w = watch(publish(dir, fake_report(; poisoned = "stand-in")) * "; exec sleep 30"; timeout = 60.0)
        @test w.stopped === :poisoned && w.report["poisoned"] == "stand-in" && w.wall < 10
        rm(results)
    end
end

@testset "harness timeouts: loading, a case, the cleanup after a case" begin
    mktempdir() do dir                                   # loading: the remaining cases fail
        command, started = fake_worker(["exec sleep 30"])
        ok, summary, output = run_stand_in_session(dir, ["a", "b"]; command, FAST...)
        @test ok == [false, false] || print(output)
        @test started[] == 1
        @test occursin("worker was stopped after the timeout (1 s) before its first case", summary[1])
        for n in ("a", "b")
            @test occursin("still loading the entry scripts after the timeout (1 s)", stand_in_status(dir, n)["error"])
            @test stand_in_position(dir, n) == (0, 2)
        end
    end
    mktempdir() do dir                                   # a case, ignoring SIGTERM: it fails, the rest go on
        command, started = fake_worker([publish(dir, fake_report(; running = "a")) * "; trap '' TERM; exec sleep 30",
                                        publish(dir, fake_report(fake_result("b", 1)))])
        ok, summary, output = run_stand_in_session(dir, ["a", "b"]; command, FAST...)
        @test ok == [false, true] || print(output)
        @test started[] == 2
        @test occursin("worker was stopped after the timeout (1 s) during a", summary[1])
        @test occursin("the case ran longer than the timeout (1 s), so its worker process was killed " *
                       "(it did not exit on SIGTERM)", stand_in_status(dir, "a")["error"])
        @test stand_in_position(dir, "a") == (1, 2) && stand_in_position(dir, "b") == (1, 1)
    end
    mktempdir() do dir                                   # the cleanup: the case keeps its result
        command, started = fake_worker([publish(dir, fake_report(fake_result("a", 1))) * "; exec sleep 30",
                                        publish(dir, fake_report(fake_result("b", 1)))])
        ok, summary, output = run_stand_in_session(dir, ["a", "b"]; command, FAST...)
        @test ok == [true, true] || print(output)
        @test started[] == 2
        @test occursin("worker was stopped after the timeout (1 s) after a", summary[1])
        @test summary[2] == rstrip(summary[2]) && !occursin(";", summary[2])     # the second ended normally
        @test stand_in_position(dir, "a") == (1, 2) && stand_in_position(dir, "b") == (1, 1)
        @test occursin("continuing the remaining 1 in a new worker", output)
    end
end

@testset "a worker stopped during a case leaves a backtrace in the case's log" begin
    mktempdir() do dir
        script, log, marker = joinpath(dir, "entry.jl"), joinpath(dir, "hang.log"), joinpath(dir, "hang.hanging")
        write(script, STAND_IN)
        plan = Dict("label" => "stand-in", "case" => [Dict("name" => "hang", "script" => script,
                                                           "args" => ["hang", joinpath(dir, "hang")], "log" => log,
                                                           "env" => Dict{String, String}())])
        plan_path, results_path = joinpath(dir, "plan.toml"), joinpath(dir, "results.toml")
        open(io -> TOML.print(io, plan), plan_path, "w")
        process = run(pipeline(Harness.worker_command(plan_path, results_path, 1); stdout = devnull, stderr = devnull);
                      wait = false)
        @test timedwait(() -> isfile(marker) || process_exited(process), 300; pollint = 0.1) === :ok   # in the case
        @test get(TOML.parsefile(results_path), "running", "") == "hang"
        Harness.stop_worker!(process, 30)
        @test process.termsignal == 15                   # SIGTERM sufficed
        text = read(log, String)
        @test occursin("stdout of hang", text)           # the case's output up to the stop
        @test occursin(r"signal \(?15\)?", text)         # and Julia's report of the signal with its backtrace
    end
end

@testset "the entry scripts answer main([\"--help\"]) when a session includes them" begin
    # Included as the worker includes them: without a thread re-execution or a
    # GPU preload (the test process would be replaced or load CUDA).
    for (script, expected) in (("run_transport.jl", "scripts/run_transport.jl <config.toml>"),
                               (joinpath("preprocessing", "preprocess_transport_binary.jl"),
                                "Usage: preprocess_transport_binary.jl <config.toml>"))
        main = Worker.load_entry(joinpath(Harness.REPO, "scripts", script))
        mktemp() do path, io
            value = redirect_stdout(() -> Base.invokelatest(main, ["--help"]), io)
            flush(io)
            @test value === nothing
            @test occursin(expected, read(path, String))
        end
    end
end
