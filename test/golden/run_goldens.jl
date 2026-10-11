#!/usr/bin/env julia
# =============================================================================
# run_goldens.jl — golden-output regression harness
# =============================================================================
#
# Runs the production entry points on the fixed cases of cases.toml and
# compares their outputs with a reference set, bit for bit (compare.jl). A
# refactor must leave every golden unchanged; a change that alters results
# states its deltas.
#
#   julia --project=. test/golden/run_goldens.jl record  <ref>        [options]
#   julia --project=. test/golden/run_goldens.jl check   <ref> <new>  [options]
#   julia --project=. test/golden/run_goldens.jl compare <ref> <new>  [options]
#
#   record   run the cases into <ref>.
#   check    run the cases into <new>, then compare with <ref>. Runtime cases
#            read the reference's preprocessed binaries, so a runtime
#            difference is never a preprocessing difference.
#   compare  compare two existing trees.
#
#   --cases=a,b        only these cases (whatever their tags)
#   --tags=fast,cpu    only cases carrying all of these tags
#   --skip-tags=slow   without cases carrying any of these tags ("known_failure"
#                      cases are skipped unless --tags or --cases names them)
#   --threads=N        Julia threads of each worker session (default 8)
#   --timeout=MIN      stop a worker that spends longer than MIN minutes
#                      (default 45) in one case, in loading its entry scripts
#                      or in the cleanup after a case. A case it was running
#                      fails; after a loading timeout the remaining cases
#                      fail; after a cleanup timeout the finished case keeps
#                      its result. Other remaining cases continue in a new
#                      worker.
#
# The cases run in worker sessions (golden_worker.jl): one Julia process per
# group of cases that share what is fixed at process start (entry point,
# [architecture] table, threads), so loading and compilation happen once per
# group, not once per case. Sessions run one after the other, in the order of
# their first case in cases.toml; each case's status.toml is written as soon
# as the case finishes. A case never overwrites: its directory must not exist
# yet. Exit status 1 if a case fails or differs. See README.md.
# =============================================================================

using TOML, Dates, Printf, SHA
include(joinpath(@__DIR__, "compare.jl"))

const REPO = dirname(dirname(@__DIR__))
const CASES_FILE = joinpath(@__DIR__, "cases.toml")
const WORKER = joinpath(@__DIR__, "golden_worker.jl")
const OPTIONS = ("cases", "tags", "skip-tags", "threads", "timeout")
const DEFAULT_TIMEOUT_MINUTES = 45      # ≈ 6× the slowest case with one process per case (7 min)

# --- configuration --------------------------------------------------------------

"""
    case_config(case, out, inputs) -> Dict

The case's config template with its placeholders filled — `@OUTPUT@` (this
case's directory), `@CASE:<name>@` (the directory of case `<name>` under
`inputs`) and `@REPO@` — and its `set` overrides applied (`"section.key" => value`,
each key must exist in the template, so a typo cannot silently do nothing).
"""
function case_config(case, out, inputs)
    subst(s::AbstractString) = replace(s, "@OUTPUT@" => out, "@REPO@" => REPO,
                                       r"@CASE:[^@]+@" => m -> joinpath(inputs, chop(m; head = 6, tail = 1)))
    subst(v::AbstractVector) = map(subst, v)
    subst(x) = x
    cfg = TOML.parse(subst(read(joinpath(@__DIR__, case["config"]), String)))
    for (field, present) in (("set", true), ("add", false)), (path, value) in get(case, field, Dict())
        sections..., key = split(path, ".")
        table = foldl((t, s) -> t[s], sections; init = cfg)
        haskey(table, key) == present || error("case $(case["name"]): `$field` key $path is " *
                                               (present ? "not" : "already") * " in $(case["config"])")
        table[key] = subst(value)
    end
    text = sprint(TOML.print, cfg)
    occursin(r"@[A-Z]+(:[^@]*)?@", text) && error("case $(case["name"]): unfilled placeholder in its config")
    return cfg
end

"""
    case_dependencies(case) -> Vector{String}

The cases whose outputs `case` reads: the `@CASE:<name>@` placeholders of its
config template and of its `set` and `add` overrides.
"""
function case_dependencies(case)
    text = read(joinpath(@__DIR__, case["config"]), String) *
           sprint(TOML.print, Dict(field => get(case, field, Dict()) for field in ("set", "add")))
    return unique!([String(m[1]) for m in eachmatch(r"@CASE:([^@]+)@", text)])
end

"""
    case_entry(case, config_path) -> (script, args)

The production entry script of a case and its command-line arguments; the
worker includes the script and calls its `main(args)`.
"""
function case_entry(case, config_path)
    case["kind"] == "preprocess" &&
        return joinpath(REPO, "scripts", "preprocessing", "preprocess_transport_binary.jl"),
               [config_path, "--day", case["day"]]
    case["kind"] == "run" && return joinpath(REPO, "scripts", "run_transport.jl"), [config_path]
    error("case $(case["name"]): unknown kind $(repr(case["kind"]))")
end

# --- sessions -----------------------------------------------------------------------

"""
    session_key(case, cfg, threads) -> NamedTuple

What a case fixes when its Julia process starts; cases with equal keys share a
worker session. `kind` selects the entry point, `architecture` is the filled
config's `[architecture]` table as written (it decides whether CUDA or Metal is
loaded; two spellings of one backend get two sessions) and `threads` the Julia
threads. The environment is the same for every case (`session_environment`)
except the regridding-weight cache directory, which the worker sets around each
case (it is read when weights are built, not when the package loads).
"""
session_key(case, cfg, threads) =
    (kind = case["kind"], architecture = architecture_label(cfg), threads = threads)

function architecture_label(cfg)
    arch = get(cfg, "architecture", Dict{String, Any}())
    arch isa AbstractDict || return repr(arch)
    return isempty(arch) ? "default" : join(("$k=$(arch[k])" for k in sort!(collect(keys(arch)))), " ")
end

session_label(key) = "$(key.kind), architecture $(key.architecture), $(key.threads) threads"

"""
    session_groups(case_keys) -> Vector{Vector{Int}}

The indices of the cases with equal session keys, one group per key in the
order of its first case; each group keeps the case order.
"""
session_groups(case_keys) = [findall(isequal(k), case_keys) for k in unique(case_keys)]

"""
    check_session_order(cases, groups)

Error unless every case that reads the outputs of another selected case runs
after it (sessions run one after the other, in the order of `groups`). Holds
for cases.toml, which lists the preprocessing cases first.
"""
function check_session_order(cases, groups)
    slot = Dict(cases[i]["name"] => (g, p) for (g, group) in enumerate(groups) for (p, i) in enumerate(group))
    for case in cases, dep in case_dependencies(case)
        haskey(slot, dep) && slot[dep] > slot[case["name"]] &&
            error("case $(case["name"]) reads the outputs of case $dep, which would run after it: " *
                  "list $dep in cases.toml before the first case of $(case["name"])'s session")
    end
    return nothing
end

"""
    session_environment() -> (env, scrubbed)

The environment of a worker process: the parent's without the switches that
change what a run does or writes (`ATMOSTR_*`, `ERA5_N320_PROFILE`; the
scrubbed names are returned), with the data root defaulted.
"""
function session_environment()
    switch(k) = startswith(k, "ATMOSTR_") || k == "ERA5_N320_PROFILE"
    scrubbed = sort!(filter(switch, collect(keys(ENV))))
    env = Dict(k => v for (k, v) in ENV if !switch(k))
    get!(env, "ATMOSTRANSPORT_DATA_ROOT", expanduser("~/data/AtmosTransport"))
    return env, scrubbed
end

"""
    case_environment(out) -> Dict

What the worker sets around one case: the regridding-weight cache of runtime
sources in the case directory, so weights are recomputed, never reused. (The
preprocessing configs put their `regridder_cache_dir` there too.)
"""
case_environment(out) = Dict("ATMOSTR_REGRID_CACHE_DIR" => joinpath(out, "_regrid_cache"))

"""
    input_files(cfg, env) -> Dict

Size and modification time of every file the filled config names (inputs that
change between a record and a check make cases differ for a non-code reason).
"""
function input_files(cfg, env)
    paths = String[]
    resolve(x) = (p = expanduser(replace(x, "\$ATMOSTRANSPORT_DATA_ROOT" => env["ATMOSTRANSPORT_DATA_ROOT"]));
                  isabspath(p) ? p : joinpath(REPO, p))          # cases run in REPO
    collect_paths(x::AbstractString) = push!(paths, resolve(x))
    collect_paths(x::AbstractDict) = foreach(collect_paths, values(x))
    collect_paths(x::AbstractVector) = foreach(collect_paths, x)
    collect_paths(_) = nothing
    collect_paths(cfg)
    return Dict(p => Dict("bytes" => filesize(p), "modified" => string(unix2datetime(mtime(p))))
                for p in paths if isfile(p))
end

"""
    run_environment(threads) -> Dict

What, besides the commit, determines a run's floating-point results.
"""
function run_environment(threads)
    manifest = joinpath(REPO, "Manifest.toml")
    devices = get(ENV, "CUDA_VISIBLE_DEVICES", "")
    gpu = try
        names = readlines(`nvidia-smi --query-gpu=name --format=csv,noheader`)
        index = tryparse(Int, first(split(devices, ",")))         # an index, or a UUID
        isempty(devices) ? first(names) : index === nothing ? devices : names[index + 1]
    catch
        "none"
    end
    return Dict("threads" => threads, "julia" => string(VERSION), "gpu" => gpu, "cuda_visible_devices" => devices,
                "manifest_sha256" => isfile(manifest) ? bytes2hex(open(sha256, manifest)) : "none")
end

"""
    write_status(case, out, cfg, result, session, threads, env, scrubbed) -> Bool

Write `status.toml` next to a case's outputs and return whether it succeeded.
From the worker's `result`: `ok`, `wall_seconds` (the case alone: package
loading and the compilation it shares with later cases are paid by the first
cases of its session), `finished` and, for a failure, `error`. `session` holds
the session's `label`, the case's `position` in it and the number of `cases`
it was started with. Besides: commit, run environment, input files and
non-finite output values.
"""
function write_status(case, out, cfg, result, session, threads, env, scrubbed)
    git(args...) = readchomp(`git -C $REPO $args`)
    status = Dict("ok" => result["ok"], "wall_seconds" => result["wall_seconds"], "finished" => result["finished"],
                  "commit" => git("rev-parse", "HEAD"),
                  "dirty" => !isempty(git("status", "--porcelain", "--", "src", "ext", "scripts", "config",
                                          "Project.toml", "test/golden")),
                  "host" => gethostname(), "scrubbed_environment" => scrubbed, "session" => session,
                  "environment" => run_environment(threads), "inputs" => input_files(cfg, env),
                  "nonfinite" => nonfinite_values(out))
    isempty(get(result, "error", "")) || (status["error"] = result["error"])
    open(io -> TOML.print(io, status), joinpath(out, "status.toml"), "w")
    wall = status["wall_seconds"]                       # NaN when the worker died or was stopped
    status["ok"] || @error "case $(case["name"]) failed" * (isfinite(wall) ? " after $(round(wall; digits = 1)) s" : "") *
                           ": $(get(status, "error", "")); see $(joinpath(out, "log.txt"))"
    isempty(status["nonfinite"]) || @warn "case $(case["name"]) wrote non-finite values" status["nonfinite"]
    return status["ok"]
end

"""
    duration(seconds) -> String

`seconds` as minutes from one minute on ("45 min"), else as seconds ("20 s").
"""
duration(seconds) = seconds >= 60 ? @sprintf("%g min", seconds / 60) : @sprintf("%g s", seconds)

"""
    read_report(path, previous) -> Dict

The worker's results file, or `previous` while there is none. (The worker
replaces the file by a rename, so a read sees the old or the new report.)
"""
read_report(path, previous) = isfile(path) ? TOML.parsefile(path) : previous

"""
    stop_worker!(process, grace)

Stop a worker: SIGTERM, on which Julia prints the backtrace of the main task
to its stderr (during a case, the case's `log.txt`), then SIGKILL if it has not
exited after `grace` seconds (a process blocked in a driver call may not).
"""
function stop_worker!(process, grace)
    kill(process)
    timedwait(() -> process_exited(process), grace; pollint = 0.5) === :ok || kill(process, Base.SIGKILL)
    wait(process)
    return nothing
end

"""
    watch_worker(on_report, cmd, results_path; timeout, poll, grace) -> NamedTuple

Start the worker command without waiting for it and read its results file
every `poll` seconds, passing each report to `on_report` (also once after the
worker exits). A stage of the worker (loading its entry scripts, one case, the
cleanup after a case) that lasts longer than `timeout` seconds gets the worker
stopped (`stop_worker!`), as does a worker that reported a poisoned session
and has not exited `grace` seconds later. Returns the `process`, its last
`report`, its `wall` seconds and why the harness `stopped` it (`:timeout`,
`:poisoned`, or `nothing`).
"""
function watch_worker(on_report, cmd, results_path; timeout, poll, grace)
    start = time()
    process = run(ignorestatus(cmd), devnull, stdout, stderr; wait = false)
    report, stopped = Dict{String, Any}(), nothing
    stage, since = (0, nothing, false), start
    try
        while timedwait(() -> process_exited(process), poll; pollint = min(poll, 0.5)) === :timed_out
            report = read_report(results_path, report)
            on_report(report)
            poisoned = haskey(report, "poisoned")
            current = (length(get(report, "case", [])), get(report, "running", nothing), poisoned)
            current == stage || ((stage, since) = (current, time()))
            if time() - since > (poisoned ? grace : timeout)
                stopped = poisoned ? :poisoned : :timeout
                stop_worker!(process, grace)
                break
            end
        end
    catch
        process_exited(process) || kill(process, Base.SIGKILL)  # the harness itself failed or was interrupted
        rethrow()
    end
    wait(process)
    report = read_report(results_path, report)
    on_report(report)
    return (process = process, report = report, wall = time() - start, stopped = stopped)
end

"""
    worker_command(plan_path, results_path, threads) -> Cmd

The command of a worker process; `run_session` runs it in the repository with
the session environment.
"""
worker_command(plan_path, results_path, threads) =
    `$(Base.julia_cmd()) --project=$REPO --threads=$threads $WORKER $plan_path $results_path`

"""
    run_session(cases, cfgs, dir, threads, label; entry = case_entry, command = worker_command,
                timeout = 60 * DEFAULT_TIMEOUT_MINUTES, poll = 5, grace = 30) -> (ok, summary)

Run `cases` (one session key; `cfgs` are their filled configs) into `dir` in a
worker process: write each case's `config.toml`, start the worker with the
plan (each case's script and arguments from `entry`), and write each case's
`status.toml` as soon as the worker reports the case finished (again if the
worker revises the result, as when it marks the session poisoned), so an
interrupted run keeps the status of its finished cases. The worker process
runs `command(plan_path, results_path, threads)` and is watched by
`watch_worker` (`timeout`, `poll` and `grace` in seconds).

If the worker dies or is stopped during a case (a crash, an `exit`, the
timeout), that case fails and the remaining cases continue in a new worker,
as they do if it ends between cases (a poisoned session, a cleanup timeout:
the finished case keeps its result); if it dies or times out before its first
case, the remaining cases fail (with `session.position = 0`). A worker ended
normally only if it exited successfully after reporting every case of its
plan. Returns whether each case succeeded and one summary line per worker
process.
"""
function run_session(cases, cfgs, dir, threads, label; entry = case_entry, command = worker_command,
                     timeout = 60.0 * DEFAULT_TIMEOUT_MINUTES, poll = 5.0, grace = 30.0)
    env, scrubbed = session_environment()
    outs = [joinpath(dir, c["name"]) for c in cases]
    for (case, cfg, out) in zip(cases, cfgs, outs)
        mkpath(out)
        open(io -> TOML.print(io, cfg), joinpath(out, "config.toml"), "w")
    end
    plan = map(cases, outs) do case, out
        script, args = entry(case, joinpath(out, "config.toml"))
        Dict("name" => case["name"], "script" => script, "args" => args, "log" => joinpath(out, "log.txt"),
             "env" => case_environment(out))
    end
    index = Dict(c["name"] => i for (i, c) in enumerate(cases))
    ok, summary = fill(false, length(cases)), String[]
    pending = collect(eachindex(cases))
    function finish(i, result, position, n)
        session = Dict{String, Any}("label" => label, "position" => position, "cases" => n,
                                    "restored" => get(result, "restored", String[]))
        haskey(result, "poisoned") && (session["poisoned"] = result["poisoned"])
        ok[i] = write_status(cases[i], outs[i], cfgs[i], result, session, threads, env, scrubbed)
    end
    failure(message) = Dict("ok" => false, "wall_seconds" => NaN, "finished" => string(now()), "error" => message)
    mktempdir() do tmp
        plan_path, results_path = joinpath(tmp, "plan.toml"), joinpath(tmp, "results.toml")
        while !isempty(pending)
            n = length(pending)
            open(io -> TOML.print(io, Dict("label" => label, "case" => plan[pending])), plan_path, "w")
            rm(results_path; force = true)
            @info "golden session $label: starting a worker for $n case$(n == 1 ? "" : "s")"
            flush(stderr)                               # before the worker's lines in a redirected log
            cmd = setenv(command(plan_path, results_path, threads), env; dir = REPO)
            seen = Dict{String, Any}()                  # the result each case's status was written from
            worker = watch_worker(cmd, results_path; timeout, poll, grace) do report
                for r in get(report, "case", [])
                    get(seen, r["name"], nothing) == r && continue
                    seen[r["name"]] = r
                    finish(index[r["name"]], r, r["position"], n)
                end
            end
            process, report = worker.process, worker.report
            results = get(report, "case", [])
            running = get(report, "running", nothing)
            filter!(i -> !haskey(seen, cases[i]["name"]), pending)
            # A killed worker (for instance by the OOM killer) has exit code 0 and its signal in `termsignal`.
            ended = worker.stopped === :timeout ? "was stopped after the timeout ($(duration(timeout)))" :
                    worker.stopped === :poisoned ? "was stopped ($(duration(grace)) after it reported a poisoned session)" :
                    process.termsignal > 0 ? "was killed by signal $(process.termsignal)" :
                    "exited with $(process.exitcode)"
            at = running !== nothing ? "during $running" :
                 isempty(results) ? "before its first case" : "after $(results[end]["name"])"
            normal = success(process) && isempty(pending)    # exited 0 after reporting every case of its plan
            note = haskey(report, "poisoned") ? "; session poisoned $at" : normal ? "" : "; worker $ended $at"
            load = get(report, "load_seconds", NaN)
            push!(summary, @sprintf("%-58s %2d of %2d cases %6.0f s (loading %s, cases %5.0f s)%s", label,
                                    length(results), n, worker.wall, isfinite(load) ? @sprintf("%3.0f s", load) : "failed",
                                    sum(r["wall_seconds"] for r in results; init = 0.0), note))
            isempty(pending) && break
            if running == cases[pending[1]]["name"]
                finish(popfirst!(pending),
                       failure(worker.stopped === :timeout ?
                               "the case ran longer than the timeout ($(duration(timeout))), so its worker process " *
                               (process.termsignal == Base.SIGKILL ?
                                "was killed (it did not exit on SIGTERM)" :
                                "was stopped (its log.txt has the backtrace at the stop)") :
                               "its worker process $ended during the case"), length(results) + 1, n)
            elseif isempty(results)                     # died before its first case: so would the next worker
                message = worker.stopped === :timeout ?
                          "its worker process was still loading the entry scripts after the timeout " *
                          "($(duration(timeout))) and was stopped before the case" :
                          "its worker process $ended before the case"
                foreach(i -> finish(i, failure(message), 0, n), pending)
                empty!(pending)
            else                                        # ended between cases: the rest go to a new worker
                @warn "golden session $label: " *
                      (haskey(report, "poisoned") ? "the session was poisoned $at" : "the worker $ended $at") *
                      "; continuing the remaining $(length(pending)) in a new worker"
            end
        end
    end
    return ok, summary
end

"""
    run_cases(cases, dir, inputs, threads, timeout) -> Vector{Bool}

Run the cases into `dir` (with `@CASE:` placeholders resolved under `inputs`),
one worker session per session key (`timeout` in seconds, see
`watch_worker`), and print each session's wall time.
"""
function run_cases(cases, dir, inputs, threads, timeout)
    cfgs = [case_config(c, joinpath(dir, c["name"]), inputs) for c in cases]
    case_keys = [session_key(c, cfg, threads) for (c, cfg) in zip(cases, cfgs)]
    groups = session_groups(case_keys)
    inputs == dir && check_session_order(cases, groups)    # a check reads the reference's outputs
    ok, summary = fill(false, length(cases)), String[]
    wall = @elapsed for group in groups
        done, lines = run_session(cases[group], cfgs[group], dir, threads, session_label(case_keys[group[1]]);
                                  timeout)
        ok[group] = done
        append!(summary, lines)
    end
    println("\nSessions:")
    foreach(line -> println("  ", line), summary)
    @printf("  total %.0f s (%.1f min) for %d cases\n\n", wall, wall / 60, length(cases))
    return ok
end

"""
    run_notes(a, ref, b, new) -> Vector{String}

How two runs of a case differed besides the code: the filled config (with the
tree roots normalised), the run environment and the input files.
"""
function run_notes(a, ref, b, new)
    roots = sort([ref, new]; by = length, rev = true)          # longest first: a root may prefix the other
    unroot(text) = replace(text, (r => "<tree>" for r in roots)...)
    config(dir) = (p = joinpath(dir, "config.toml"); isfile(p) ? unroot(read(p, String)) : nothing)
    status(dir) = (p = joinpath(dir, "status.toml"); isfile(p) ? TOML.parsefile(p) : Dict{String, Any}())
    notes = String[]
    config(a) == config(b) || push!(notes, "the filled configs differ (diff $a/config.toml $b/config.toml)")
    sa, sb = status(a), status(b)
    ea, eb = get(sa, "environment", Dict()), get(sb, "environment", Dict())
    for k in sort!(collect(union(keys(ea), keys(eb))))
        isequal(get(ea, k, nothing), get(eb, k, nothing)) || push!(notes, "$k: $(get(ea, k, nothing)) vs $(get(eb, k, nothing))")
    end
    inputs(st) = Dict(unroot(p) => v for (p, v) in get(st, "inputs", Dict()))
    ia, ib = inputs(sa), inputs(sb)
    for p in sort!(collect(union(keys(ia), keys(ib))))
        isequal(get(ia, p, nothing), get(ib, p, nothing)) || push!(notes, "input $p changed: $(get(ia, p, nothing)) vs $(get(ib, p, nothing))")
    end
    return notes
end

status_of(dir) = (p = joinpath(dir, "status.toml"); isfile(p) ? TOML.parsefile(p) : Dict{String, Any}())
succeeded(dir) = get(status_of(dir), "ok", false)

"""
    wall_label(dir) -> String

A case's wall time and how it ran: in its own process (references recorded
before sessions) or as case k of n of a worker session, whose first cases pay
the loading and compilation the later ones reuse; `failed` for a failed case
and `not run` without a status. Wall times are reported, never compared: a
session case against a per-process reference is not a speed-up of the code.
"""
function wall_label(dir)
    status = status_of(dir)
    isempty(status) && return "not run"
    get(status, "ok", false) || return "failed"
    wall = status["wall_seconds"]
    session = get(status, "session", nothing)
    return session === nothing ? @sprintf("%.0f s (own process)", wall) :
           @sprintf("%.0f s (session case %d of %d)", wall, session["position"], session["cases"])
end

"""
    compare_cases(cases, ref, new) -> Bool

Compare every case of `new` with `ref` and print one line per case (and its
differences); true if all are identical.
"""
function compare_cases(cases, ref, new)
    clean = true
    width = maximum(c -> length(c["name"]), cases; init = 0) + 2   # name column
    for case in cases
        a, b = joinpath(ref, case["name"]), joinpath(new, case["name"])
        timing = "wall $(wall_label(a)) → $(wall_label(b))"
        if !(succeeded(a) && succeeded(b))
            println(rpad(case["name"], width), "NOT RUN OR FAILED (reference ok: $(succeeded(a)), new ok: $(succeeded(b)))",
                    "   ", timing)
            clean = false
            continue
        end
        diffs = compare_case(a, b)
        println(rpad(case["name"], width), isempty(diffs) ? "identical" : "DIFFERS ($(length(diffs)))", "   ", timing)
        foreach(d -> println("    ", d), first(diffs, 20))
        length(diffs) > 20 && println("    … $(length(diffs) - 20) more")
        foreach(n -> println("    note: ", n), run_notes(a, ref, b, new))
        clean &= isempty(diffs)
    end
    return clean
end

# --- command line -------------------------------------------------------------------

function selected_cases(opts)
    cases = TOML.parsefile(CASES_FILE)["case"]
    if haskey(opts, "cases")                      # named cases run whatever their tags
        wanted = split(opts["cases"], ",")
        unknown = setdiff(wanted, [c["name"] for c in cases])
        isempty(unknown) || error("unknown cases $(unknown)")
        return filter(c -> c["name"] in wanted, cases)
    end
    tags = String.(split(get(opts, "tags", ""), ",", keepempty = false))
    skip = String.(split(get(opts, "skip-tags", ""), ",", keepempty = false))
    "known_failure" in tags || push!(skip, "known_failure")
    return filter(c -> all(in(c["tags"]), tags) && !any(in(c["tags"]), skip), cases)
end

function parse_options(args)
    opts = Dict{String, String}()
    for a in filter(startswith("--"), args)
        m = match(r"^--([\w-]+)=(.+)$", a)
        m !== nothing && m[1] in OPTIONS || error("unknown option $a (options: $(join("--" .* OPTIONS .* "=…", ", ")))")
        opts[m[1]] = m[2]
    end
    return opts
end

function main(args)
    opts = parse_options(args)
    positional = filter(!startswith("--"), args)
    length(positional) in 2:3 || error("usage: run_goldens.jl record <ref> | check <ref> <new> | compare <ref> <new> [options]")
    command, dirs = positional[1], abspath.(positional[2:end])
    cases = selected_cases(opts)
    isempty(cases) && error("no case selected")
    threads = parse(Int, get(opts, "threads", "8"))
    timeout = 60 * parse(Float64, get(opts, "timeout", string(DEFAULT_TIMEOUT_MINUTES)))
    timeout > 0 || error("--timeout must be positive (minutes)")
    run_into(dir, inputs) = begin                 # refuse before running anything if a case exists
        existing = filter(c -> ispath(joinpath(dir, c["name"])), cases)
        isempty(existing) || error("$(dir) already has $(join([c["name"] for c in existing], ", ")); golden runs never overwrite")
        run_cases(cases, dir, inputs, threads, timeout)
    end
    if command == "record" && length(dirs) == 1
        ok = all(run_into(dirs[1], dirs[1]))
    elseif command == "check" && length(dirs) == 2
        run_into(dirs[2], dirs[1])
        ok = compare_cases(cases, dirs[1], dirs[2])
    elseif command == "compare" && length(dirs) == 2
        ok = compare_cases(cases, dirs[1], dirs[2])
    else
        error("unknown command or wrong number of directories: $(join(positional, ' '))")
    end
    exit(ok ? 0 : 1)
end

(abspath(PROGRAM_FILE) == @__FILE__) && main(ARGS)
