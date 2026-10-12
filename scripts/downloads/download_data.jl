#!/usr/bin/env julia
# ===========================================================================
# Unified download entry point.
#
# Downloads meteorological data from any supported source using TOML configs.
#
# Usage:
#   julia --project=. scripts/downloads/download_data.jl config.toml \
#       [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--dry-run] [--verify]
# ===========================================================================

using Logging
using TOML
using Dates

# Standalone modules: downloading does not require the full AtmosTransport
# package, only its constants, logging support, and the Downloads module.
include(joinpath(@__DIR__, "..", "..", "src", "Parameters", "Parameters.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "LogLevels.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "Downloads", "Downloads.jl"))
using .DataDownloads

const USAGE = "Usage: download_data.jl <config.toml> " *
              "[--start YYYY-MM-DD] [--end YYYY-MM-DD] [--dry-run] [--verify]"

function _parse_cli(args::Vector{String})
    isempty(args) && throw(ArgumentError(USAGE))
    cfg_path = expanduser(args[1])
    startswith(args[1], "-") && throw(ArgumentError("missing <config.toml>; $USAGE"))
    isfile(cfg_path) || throw(ArgumentError("Config not found: $cfg_path"))

    dates = Dict{String, Union{Nothing, Date}}("--start" => nothing, "--end" => nothing)
    switches = Dict("--dry-run" => false, "--verify" => false)
    i = 2
    while i <= length(args)
        flag = args[i]
        if haskey(dates, flag)
            dates[flag] === nothing || throw(ArgumentError("duplicate option: $flag"))
            i < length(args) || throw(ArgumentError("missing value after $flag"))
            value = args[i + 1]
            startswith(value, "--") && throw(ArgumentError("missing value after $flag"))
            parsed = tryparse(Date, value)
            parsed === nothing &&
                throw(ArgumentError("$flag must be YYYY-MM-DD; got $(repr(value))"))
            dates[flag] = parsed
            i += 2
        elseif haskey(switches, flag)
            switches[flag] && throw(ArgumentError("duplicate option: $flag"))
            switches[flag] = true
            i += 1
        else
            throw(ArgumentError("unknown option: $flag"))
        end
    end

    start_date = dates["--start"]
    end_date = dates["--end"]
    start_date === nothing || end_date === nothing || start_date <= end_date ||
        throw(ArgumentError("--start must be on or before --end"))
    return (; cfg_path, start_date, end_date,
            dry_run = switches["--dry-run"], verify = switches["--verify"])
end

function main()
    global_logger(ConsoleLogger(stderr, Logging.Info; show_limited = false))
    opts = _parse_cli(ARGS)
    cfg = TOML.parsefile(opts.cfg_path)
    return download_data!(cfg; start_date = opts.start_date,
                          end_date = opts.end_date,
                          dry_run = opts.dry_run,
                          verify_only = opts.verify)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
