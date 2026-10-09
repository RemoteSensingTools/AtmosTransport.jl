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
#   --threads=N        Julia threads of each case process (default 8)
#
# A case never overwrites: its directory must not exist yet. Exit status 1 if
# a case fails or differs. See README.md.
# =============================================================================

using TOML, Dates, Printf, SHA
include(joinpath(@__DIR__, "compare.jl"))

const REPO = dirname(dirname(@__DIR__))
const CASES_FILE = joinpath(@__DIR__, "cases.toml")
const OPTIONS = ("cases", "tags", "skip-tags", "threads")

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

function case_command(case, config_path, threads)
    julia = `$(Base.julia_cmd()) --project=$REPO --threads=$threads`
    case["kind"] == "preprocess" &&
        return `$julia $(joinpath(REPO, "scripts", "preprocessing", "preprocess_transport_binary.jl")) $config_path --day $(case["day"])`
    case["kind"] == "run" && return `$julia $(joinpath(REPO, "scripts", "run_transport.jl")) $config_path`
    error("case $(case["name"]): unknown kind $(repr(case["kind"]))")
end

# --- running ----------------------------------------------------------------------

"""
    case_environment(out) -> (env, scrubbed)

The environment of a case process: the parent's without the switches that
change what a run does or writes (`ATMOSTR_*`, `ERA5_N320_PROFILE`), with the
data root defaulted, threads set by the harness and the regridding-weight
cache in the case directory (recomputed, never reused).
"""
function case_environment(out)
    switch(k) = startswith(k, "ATMOSTR_") || k == "ERA5_N320_PROFILE"
    scrubbed = sort!(filter(switch, collect(keys(ENV))))
    env = Dict(k => v for (k, v) in ENV if !switch(k))
    get!(env, "ATMOSTRANSPORT_DATA_ROOT", expanduser("~/data/AtmosTransport"))
    env["ATMOSTR_NO_AUTO_THREADS"] = "1"                       # threads come from --threads
    env["ATMOSTR_REGRID_CACHE_DIR"] = joinpath(out, "_regrid_cache")
    return env, scrubbed
end

"""
    input_files(cfg, env) -> Dict

Size and modification time of every file the filled config names (inputs that
change between a record and a check make cases differ for a non-code reason).
"""
function input_files(cfg, env)
    paths = String[]
    resolve(x) = (p = expanduser(replace(x, "\$ATMOSTRANSPORT_DATA_ROOT" => env["ATMOSTRANSPORT_DATA_ROOT"]));
                  isabspath(p) ? p : joinpath(REPO, p))          # the case process runs in REPO
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
    run_case(case, dir, inputs, threads) -> Bool

Run one case into `dir/<name>`: the filled config (`config.toml`), the process
log (`log.txt`) and `status.toml` (exit, wall time, commit, run environment,
input files, non-finite output values) next to its outputs.
"""
function run_case(case, dir, inputs, threads)
    out = joinpath(dir, case["name"])
    mkpath(out)
    cfg = case_config(case, out, inputs)
    config_path = joinpath(out, "config.toml")
    open(io -> TOML.print(io, cfg), config_path, "w")
    @info "golden case $(case["name"])"
    env, scrubbed = case_environment(out)
    cmd = setenv(case_command(case, config_path, threads), env; dir = REPO)
    log = joinpath(out, "log.txt")
    wall = @elapsed ok = open(io -> success(pipeline(cmd; stdout = io, stderr = io)), log, "w")
    git(args...) = readchomp(`git -C $REPO $args`)
    status = Dict("ok" => ok, "wall_seconds" => wall, "commit" => git("rev-parse", "HEAD"),
                  "dirty" => !isempty(git("status", "--porcelain", "--", "src", "ext", "scripts", "config",
                                          "Project.toml", "test/golden")),
                  "host" => gethostname(), "finished" => string(now()), "scrubbed_environment" => scrubbed,
                  "environment" => run_environment(threads), "inputs" => input_files(cfg, env),
                  "nonfinite" => nonfinite_values(out))
    open(io -> TOML.print(io, status), joinpath(out, "status.toml"), "w")
    ok || @error "case $(case["name"]) failed after $(round(wall; digits = 1)) s; see $log"
    isempty(status["nonfinite"]) || @warn "case $(case["name"]) wrote non-finite values" status["nonfinite"]
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

wall_seconds(dir) = (p = joinpath(dir, "status.toml"); isfile(p) ? TOML.parsefile(p)["wall_seconds"] : NaN)
succeeded(dir) = (p = joinpath(dir, "status.toml"); isfile(p) && TOML.parsefile(p)["ok"])

"""
    compare_cases(cases, ref, new) -> Bool

Compare every case of `new` with `ref` and print one line per case (and its
differences); true if all are identical.
"""
function compare_cases(cases, ref, new)
    clean = true
    for case in cases
        a, b = joinpath(ref, case["name"]), joinpath(new, case["name"])
        if !(succeeded(a) && succeeded(b))
            println(rpad(case["name"], 34), "NOT RUN OR FAILED (reference ok: $(succeeded(a)), new ok: $(succeeded(b)))")
            clean = false
            continue
        end
        diffs = compare_case(a, b)
        timing = @sprintf("wall %.0f s → %.0f s", wall_seconds(a), wall_seconds(b))
        println(rpad(case["name"], 34), isempty(diffs) ? "identical" : "DIFFERS ($(length(diffs)))", "   ", timing)
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
    run_into(dir, inputs) = begin                 # refuse before running anything if a case exists
        existing = filter(c -> ispath(joinpath(dir, c["name"])), cases)
        isempty(existing) || error("$(dir) already has $(join([c["name"] for c in existing], ", ")); golden runs never overwrite")
        [run_case(c, dir, inputs, threads) for c in cases]
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
