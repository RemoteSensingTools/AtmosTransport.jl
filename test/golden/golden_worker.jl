#!/usr/bin/env julia
# =============================================================================
# golden_worker.jl — one Julia session that runs a group of golden cases
# =============================================================================
#
# run_goldens.jl starts one worker per session group (cases that share the
# process-start settings: entry point, [architecture], threads) so that package
# loading and compilation happen once per group instead of once per case:
#
#   julia --project=<repo> --threads=N test/golden/golden_worker.jl <plan.toml> <results.toml>
#
# The plan lists the cases in order: name, entry script, arguments, log file
# and the environment variables set around the case. The worker first includes
# every entry script of the plan, each into its own module, which loads
# AtmosTransport and its dependencies once; package initialisation (which sets
# environment variables, for instance ECCODES_* when GRIB loads) belongs to
# the session, as it belonged to each process before. Each case then calls its
# script's `main(args)`, with its stdout, stderr and log messages in its log
# file. A case that throws is recorded as failed and the session continues.
# After every case the worker rewrites <results.toml> (before a case it names
# the running case), so run_goldens.jl writes each case's status as soon as it
# finishes and knows which case a crashed or stopped worker was running.
# Between cases the worker frees memory; if that fails (a CUDA synchronize or
# reclaim throws, the mark of a sticky device error), the session is poisoned:
# the worker records it in <results.toml> and exits with code 75, and the
# remaining cases continue in a new worker. See README.md, "Sessions".
# =============================================================================

using TOML, Dates, Logging, Printf

const CUDA_PKGID = Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")

"Exit code of a worker that stopped because its session is poisoned (EX_TEMPFAIL: retry in a new process)."
const POISONED_EXIT_CODE = 75

"""
    SessionPoisoned(message)

Thrown by `run_session` when freeing memory after a case fails: with CUDA
loaded, a failing `synchronize` or `reclaim` means the context holds a sticky
error (an illegal address, a launch failure), which every later CUDA call of
the process rethrows, so the session cannot run another case.
"""
struct SessionPoisoned <: Exception
    message::String
end

Base.showerror(io::IO, e::SessionPoisoned) = print(io, e.message)

"""
    run_isolated(f, log_path; env = Dict(), dir = pwd()) -> NamedTuple

Run `f()` as one case of a session: stdout, stderr (also of C libraries) and
the logging system go to `log_path`, the variables of `env` are set, and the
working directory is `dir`. An exception is printed to the log with its
backtrace and returned as a failure (an `InterruptException` is rethrown).
Afterwards the global logger, the working directory and `ENV` are as before;
anything the case changed in them is undone and named in `restored`.

Returns `(ok, wall_seconds, error, restored)`, with `error` the one-line
message of a failure (empty on success).
"""
function run_isolated(f, log_path::AbstractString; env = Dict{String, String}(), dir::AbstractString = pwd())
    env_before, dir_before = copy(ENV), pwd()
    ok, message, wall, restored = false, "", NaN, String[]
    open(log_path, "w") do io
        redirect_stdout(io) do
            redirect_stderr(io) do
                # The case's tasks log through the scoped logger; tasks that do
                # not descend from it (finalizers, library tasks) through the
                # global one: both write to the log.
                logger = ConsoleLogger(stderr, Logging.Info; show_limited = false)
                previous = global_logger(logger)
                start = time()
                try
                    with_logger(logger) do
                        withenv(env...) do
                            cd(dir) do                  # `cd` goes back to `dir_before` afterwards
                                inside = pwd()
                                try
                                    f()
                                finally
                                    pwd() == inside || push!(restored, "working directory")
                                end
                            end
                        end
                    end
                    ok = true
                catch err
                    err isa InterruptException && rethrow()
                    # In the latest world: the error and its backtrace may name
                    # functions of the entry script, which is newer than this code.
                    backtrace = catch_backtrace()
                    message = first(split(Base.invokelatest(sprint, showerror, err), '\n'))
                    print(stderr, "ERROR: ")
                    Base.invokelatest(showerror, stderr, err, backtrace)
                    println(stderr)
                finally
                    wall = time() - start
                    global_logger() === logger || push!(restored, "global logger")
                    global_logger(previous)
                    Libc.flush_cstdio()
                end
            end
        end
    end
    pwd() == dir_before || (cd(dir_before); push!(restored, "working directory"))
    for k in sort!(collect(union(keys(env_before), keys(ENV))))
        before = get(env_before, k, nothing)
        isequal(get(ENV, k, nothing), before) && continue
        before === nothing ? delete!(ENV, k) : (ENV[k] = before)
        push!(restored, "ENV[\"$k\"]")
    end
    return (ok = ok, wall_seconds = wall, error = message, restored = restored)
end

"""
    load_entry(script) -> Function

Include an entry script into a module of its own and return its `main`. A
script that fails to load returns a function that rethrows the load error with
its backtrace, so every case of the script fails with it in its log.
"""
function load_entry(script::AbstractString)
    m = Module(Symbol(splitext(basename(script))[1]))
    try
        Core.eval(m, :(include(path::AbstractString) = Base.include($m, path)))
        Core.eval(m, :(eval(expr) = Core.eval($m, expr)))
        Base.include(m, script)
        return Base.invokelatest(getglobal, m, :main)
    catch err
        err isa InterruptException && rethrow()
        failure = CapturedException(err, catch_backtrace())
        return (_...) -> throw(failure)
    end
end

"""
    release_memory!()

Free what the previous case left behind before the next one starts: a full
garbage collection and, when CUDA is loaded, the device memory cached by its
pool (after a `synchronize`). A CUDA error propagates: `run_session` treats it
as a poisoned session (`SessionPoisoned`).
"""
function release_memory!()
    GC.gc(true)
    CUDA = get(Base.loaded_modules, CUDA_PKGID, nothing)
    CUDA === nothing || Base.invokelatest(reclaim_device_memory!, CUDA)   # CUDA is newer than this code
    return nothing
end

reclaim_device_memory!(CUDA::Module) = (CUDA.functional() && (CUDA.synchronize(); CUDA.reclaim()); nothing)

function write_results(path, results; load_seconds, running = nothing, poisoned = nothing)
    report = Dict{String, Any}("case" => results, "load_seconds" => load_seconds)
    running === nothing || (report["running"] = running)
    poisoned === nothing || (report["poisoned"] = poisoned)
    open(io -> TOML.print(io, report), path * ".tmp", "w")
    mv(path * ".tmp", path; force = true)
    return path
end

"""
    run_session(plan_path, results_path) -> Vector{Dict}

Run the cases of the plan in order in this process (see the file header) and
return their results: name, `ok`, `wall_seconds`, `error`, `finished`,
`position` (1 for the first case of the session) and `restored`. The results
file also records `load_seconds`, the time to load the entry scripts.

If freeing memory after a case fails, the session is poisoned: that case's
result gets a `poisoned` message (its outcome stands), the results file
records it as `poisoned`, and `SessionPoisoned` is thrown instead of running
the next case.
"""
function run_session(plan_path::AbstractString, results_path::AbstractString)
    plan = TOML.parsefile(plan_path)
    cases = plan["case"]
    results = Dict{String, Any}[]
    entries = Dict{String, Function}()
    load_seconds = @elapsed for script in unique(c["script"] for c in cases)
        entries[script] = load_entry(script)
    end
    write_results(results_path, results; load_seconds)
    dir = pwd()
    for (position, case) in enumerate(cases)
        name = case["name"]
        write_results(results_path, results; load_seconds, running = name)
        main = entries[case["script"]]
        outcome = run_isolated(case["log"]; env = Dict{String, String}(get(case, "env", Dict())), dir) do
            Base.invokelatest(main, String.(case["args"]))
            nothing                                       # the model is dropped here
        end
        push!(results, Dict{String, Any}("name" => name, "ok" => outcome.ok, "wall_seconds" => outcome.wall_seconds,
                                         "error" => outcome.error, "finished" => string(now()),
                                         "position" => position, "restored" => outcome.restored))
        write_results(results_path, results; load_seconds)
        @info @sprintf("golden session %s: %d/%d %s %s in %.1f s", get(plan, "label", ""), position,
                       length(cases), name, outcome.ok ? "ok" : "FAILED ($(outcome.error))", outcome.wall_seconds)
        isempty(outcome.restored) ||
            @warn "case $name changed process state; restored for the next case" outcome.restored
        flush(stderr)                                     # progress shows in a redirected harness log
        released = try
            release_memory!()
            nothing
        catch err
            err isa InterruptException && rethrow()
            (err, catch_backtrace())
        end
        released === nothing && continue
        err, backtrace = released
        detail = first(split(Base.invokelatest(sprint, showerror, err), '\n'))
        message = "freeing memory after case $name failed ($detail): the CUDA context probably holds a " *
                  "sticky error that would fail every later case of this process, so the worker stops " *
                  "and the remaining cases continue in a new worker"
        results[end]["poisoned"] = message
        write_results(results_path, results; load_seconds, poisoned = message)
        @error "golden session $(get(plan, "label", "")): poisoned; $message"
        Base.invokelatest(showerror, stderr, err, backtrace)
        println(stderr)
        flush(stderr)
        throw(SessionPoisoned(message))
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("usage: golden_worker.jl <plan.toml> <results.toml>")
    try
        run_session(ARGS[1], ARGS[2])
    catch err
        err isa SessionPoisoned || rethrow()
        exit(POISONED_EXIT_CODE)                          # the message is logged and in the results file
    end
end
