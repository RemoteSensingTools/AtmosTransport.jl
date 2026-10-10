#!/usr/bin/env julia
#
# Main test suite entrypoint for AtmosTransport.
#
# Tiered layout (see test/README.md):
#   test/core/        — canonical CI baseline (no external data); must stay green
#   test/real_data/   — needs preprocessed binaries / ERA5 GRIB; opt-in via --all
#   test/diagnostic/  — large numerical sweeps; opt-in via --diagnostic or --all
#   test/archived/    — never run; kept as reference (see archived/legacy_README.md)
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

function _selected_tiers(args::Vector{String})
    explicit_tiers = any(startswith(arg, "--tiers=") for arg in args)
    selected = explicit_tiers ? Set{Symbol}() : Set{Symbol}((:core, :regridding))
    for arg in args
        if arg == "--all"
            push!(selected, :real_data); push!(selected, :diagnostic)
        elseif arg == "--diagnostic"
            push!(selected, :diagnostic)
        elseif arg == "--real-data" || arg == "--real_data"
            push!(selected, :real_data)
        elseif arg == "--orphan"
            push!(selected, :orphan)
        elseif startswith(arg, "--tiers=")
            for s in split(arg[length("--tiers=")+1:end], ",")
                key = Symbol(replace(strip(s), "-" => "_"))
                haskey(TIER_FOLDERS, key) ||
                    error("Unknown tier '$key' (known: $(keys(TIER_FOLDERS)))")
                push!(selected, key)
            end
        end
    end
    return selected
end

function _tier_files(tier::Symbol)
    folder = joinpath(@__DIR__, TIER_FOLDERS[tier])
    isdir(folder) || return String[]
    tier === :regridding && return [joinpath(TIER_FOLDERS[tier], "runtests.jl")]
    files = sort([joinpath(TIER_FOLDERS[tier], f)
                  for f in readdir(folder) if endswith(f, ".jl")])
    if tier === :core
        # Keep the inference baseline independent of later optional extensions
        # and methods added by individual tests.
        health_gates = [joinpath(TIER_FOLDERS[tier], name)
                        for name in ("test_aqua.jl", "test_jet.jl")]
        return vcat(health_gates, setdiff(files, health_gates))
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

selected = _selected_tiers(ARGS)
timings = Pair{String, Float64}[]
failed = String[]
for tier in (:core, :regridding, :real_data, :diagnostic, :orphan)
    tier in selected || continue
    files = _tier_files(tier)
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

skipped = setdiff(Set(keys(TIER_FOLDERS)), selected)
isempty(skipped) || @info "Skipped tiers: $(sort(collect(skipped))) (opt-in flags: --real-data --diagnostic --orphan --all)"
slowest = first(sort(timings; by = last, rev = true), N_SLOWEST)
@info "Slowest $(length(slowest)) of $(length(timings)) test files:\n" *
      join((string(lpad(round(t; digits = 1), 8), " s  ", f) for (f, t) in slowest), "\n")
isempty(failed) || error("$(length(failed)) of $(length(timings)) test files failed:\n  " *
                         join(failed, "\n  "))
@info "Test suite complete: $(length(timings)) files passed."
