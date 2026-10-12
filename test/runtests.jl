#!/usr/bin/env julia
#
# Main test suite entrypoint for AtmosTransport.
#
# Tiered layout (see test/README.md):
#   test/core/        — canonical CI baseline (no external data); must stay green
#   test/real_data/   — needs preprocessed binaries / ERA5 GRIB; opt-in via --all
#   test/diagnostic/  — large numerical sweeps; opt-in via --diagnostic or --all
#   test/orphan/      — promotion candidates; CI-excluded; opt-in via --orphan
#   test/regridding/  — conservative-remapping correctness; part of CI baseline
#
# Usage:
#   julia --project=test test/runtests.jl                # core + regridding
#   julia --project=test test/runtests.jl --all          # default + real_data + diagnostic
#   julia --project=test test/runtests.jl --diagnostic   # default + diagnostic
#   julia --project=test test/runtests.jl --orphan       # default + orphan watchlist
#   julia --project=test test/runtests.jl --tiers=core,real_data,orphan
#
# Selectors pick files of the default tiers (core, regridding) from the topics
# and the quick set of test/groups.toml instead of running them all:
#   julia --project=test test/runtests.jl --quick              # crucial fast set
#   julia --project=test test/runtests.jl --topics=advection,diffusion
#   julia --project=test test/runtests.jl --changed            # topics of the changes vs origin/main
#   julia --project=test test/runtests.jl --changed=<git-ref> --quick
#   julia --project=test test/runtests.jl --quick --list       # print the selection, run nothing
# Selectors combine with each other and with the tier flags (the union runs):
# `--quick --diagnostic` runs the quick set and the diagnostic tier.
#
# Each test file runs in its own module to isolate helpers and constants.
# Core tests import the cached package rather than compiling a new copy per file.
# A failing file does not stop the suite (an interrupt does): every file runs,
# then the failed files and the slowest files are listed and the suite fails if
# any file failed.

const TIER_FOLDERS = (
    core       = "core",
    regridding = "regridding",
    real_data  = "real_data",
    diagnostic = "diagnostic",
    orphan     = "orphan",
)

const DEFAULT_TIERS = (:core, :regridding)

# Command-line options: whole tiers, selectors, and `--list`.
function _parse_args(args::Vector{String})
    explicit_tiers = any(startswith(arg, "--tiers=") for arg in args)
    quick = false
    topics = String[]
    changed = nothing
    list = false
    extra_tiers = Set{Symbol}()
    for arg in args
        if arg == "--all"
            push!(extra_tiers, :real_data); push!(extra_tiers, :diagnostic)
        elseif arg == "--diagnostic"
            push!(extra_tiers, :diagnostic)
        elseif arg == "--real-data" || arg == "--real_data"
            push!(extra_tiers, :real_data)
        elseif arg == "--orphan"
            push!(extra_tiers, :orphan)
        elseif startswith(arg, "--tiers=")
            for s in split(arg[length("--tiers=")+1:end], ",")
                key = Symbol(replace(strip(s), "-" => "_"))
                haskey(TIER_FOLDERS, key) ||
                    error("Unknown tier '$key' (known: $(keys(TIER_FOLDERS)))")
                push!(extra_tiers, key)
            end
        elseif arg == "--quick"
            quick = true
        elseif startswith(arg, "--topics=")
            append!(topics, filter!(!isempty, strip.(split(arg[length("--topics=")+1:end], ","))))
            isempty(topics) && error("--topics= needs at least one topic name")
        elseif arg == "--changed"
            changed = "origin/main"
        elseif startswith(arg, "--changed=")
            changed = arg[length("--changed=")+1:end]
            isempty(changed) && error("--changed= needs a git ref")
        elseif arg == "--list"
            list = true
        else
            error("Unknown argument '$arg'. Tier flags: --all --diagnostic --real-data " *
                  "--orphan --tiers=a,b; selectors: --quick --topics=a,b --changed[=<ref>]; --list")
        end
    end
    selector = quick || !isempty(topics) || changed !== nothing
    tiers = explicit_tiers || selector ? Set{Symbol}() : Set{Symbol}(DEFAULT_TIERS)
    union!(tiers, extra_tiers)
    return (; tiers, quick, topics, changed, list, selector)
end

# Files of the default tiers chosen by the selectors (paths relative to test/),
# and the tiers to run whole when a changed `src/` file maps to no topic.
# Needs `TestGroups` (helpers/test_groups.jl), included only for selectors.
function _selected_subset(opts)
    groups = TestGroups.load_groups(joinpath(@__DIR__, "groups.toml"))
    subset = Set{String}()
    whole = Set{Symbol}()
    parts = String[]
    if opts.quick
        union!(subset, TestGroups.quick_files(groups))
        push!(parts, "quick set")
    end
    if !isempty(opts.topics)
        union!(subset, TestGroups.topic_files(groups, opts.topics))
        push!(parts, "topics " * join(opts.topics, ","))
    end
    if opts.changed !== nothing
        paths = TestGroups.changed_paths(dirname(@__DIR__), opts.changed)
        sel = TestGroups.select_changed(groups, @__DIR__, paths)
        topics = sort!(collect(sel.topics))
        union!(subset, sel.files)
        union!(subset, TestGroups.topic_files(groups, topics))
        foreach(note -> @info(note), sel.notes)
        sel.full && union!(whole, DEFAULT_TIERS)
        push!(parts, "changes vs $(opts.changed) ($(length(paths)) files; topics: " *
                     (isempty(topics) ? "none" : join(topics, ",")) * ")")
    end
    @info "Selected " * join(parts, " + ")
    return subset, whole
end

function _tier_files(tier::Symbol)
    folder = joinpath(@__DIR__, TIER_FOLDERS[tier])
    isdir(folder) || return String[]
    tier === :regridding && return [TIER_FOLDERS[tier] * "/runtests.jl"]
    files = sort([TIER_FOLDERS[tier] * "/" * f
                  for f in readdir(folder) if endswith(f, ".jl")])
    if tier === :core
        # Keep the inference baseline independent of later optional extensions
        # and methods added by individual tests.
        health_gates = [TIER_FOLDERS[tier] * "/" * name
                        for name in ("test_aqua.jl", "test_jet.jl")]
        return vcat(filter(in(files), health_gates), setdiff(files, health_gates))
    end
    return files
end

function run_test_file_isolated(test_file::AbstractString)
    mod_name = Symbol("Test_", replace(replace(basename(test_file), "." => "_"), "/" => "_"))
    mod = Module(mod_name)
    Core.eval(mod, :(include(path::AbstractString) = Base.include($mod, path)))
    Core.eval(mod, :(include(mapexpr::Function, path::AbstractString) = Base.include(mapexpr, $mod, path)))
    Core.eval(mod, :(eval(expr) = Core.eval($mod, expr)))
    return Base.include(mod, joinpath(@__DIR__, test_file))
end

# `include` wraps an error thrown by the file, Ctrl-C included, in a `LoadError`.
_is_interrupt(err) = err isa InterruptException || (err isa LoadError && _is_interrupt(err.error))

# Run one file; return its wall time in seconds and the error it threw, if any.
function run_test_file_timed(test_file::AbstractString)
    t0 = time()
    failure = try
        run_test_file_isolated(test_file)
        nothing
    catch err
        _is_interrupt(err) && rethrow()
        @error "Test file failed: $(test_file)" exception = (err, catch_backtrace())
        err
    end
    return time() - t0, failure
end

const N_SLOWEST = 15

opts = _parse_args(ARGS)
opts.selector && include(joinpath(@__DIR__, "helpers", "test_groups.jl"))
subset, whole = opts.selector ? _selected_subset(opts) : (Set{String}(), Set{Symbol}())
selected = union(opts.tiers, whole)
plan = Pair{Symbol, Vector{String}}[]
for tier in (:core, :regridding, :real_data, :diagnostic, :orphan)
    files = _tier_files(tier)
    tier in selected || (files = filter(in(subset), files))
    push!(plan, tier => files)
end
if opts.list
    for (tier, files) in plan, f in files
        println(f)
    end
    exit(0)
end

timings = Pair{String, Float64}[]
failed = String[]
for (tier, files) in plan
    (tier in selected || !isempty(files)) || continue
    if isempty(files)
        @info "Tier $(tier): no files"
        continue
    end
    @info "── Tier $(tier) — $(length(files)) files ──"
    for f in files
        @info "Running $f"
        seconds, failure = run_test_file_timed(f)
        push!(timings, f => seconds)
        failure === nothing || push!(failed, f)
    end
end

skipped = [tier for (tier, files) in plan if !(tier in selected) && isempty(files)]
isempty(skipped) || @info "Skipped tiers: $(sort(skipped)) (opt-in flags: --real-data --diagnostic --orphan --all)"
opts.selector && @info "Selectors ran $(length(timings)) files; the merge check runs the full suite (no flags)."
slowest = first(sort(timings; by = last, rev = true), N_SLOWEST)
@info "Slowest $(length(slowest)) of $(length(timings)) test files:\n" *
      join((string(lpad(round(t; digits = 1), 8), " s  ", f) for (f, t) in slowest), "\n")
isempty(failed) || error("$(length(failed)) of $(length(timings)) test files failed:\n  " *
                         join(failed, "\n  "))
@info "Test suite complete: $(length(timings)) files passed."
