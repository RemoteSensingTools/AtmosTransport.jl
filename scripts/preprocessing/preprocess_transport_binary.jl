#!/usr/bin/env julia
# ===========================================================================
# Unified transport-binary preprocessor — THE entry point.
#
# Generates v4 transport binaries from any supported met source onto any
# supported target grid. Config-driven via TOML; the canonical entrypoint
# detects native sources via `[source].toml` and routes through the
# `AbstractMetSettings` factory, otherwise falls back to the ERA5 spectral
# path. New sources plug in through `AbstractMetSettings` + a
# `config/met_sources/<source>.toml` descriptor — never via a parallel CLI.
#
# Usage:
#
#   # Single day
#   julia -t8 --project=. scripts/preprocessing/preprocess_transport_binary.jl \
#       <config.toml> --day 2021-12-01
#
#   # Date range
#   julia -t8 --project=. scripts/preprocessing/preprocess_transport_binary.jl \
#       <config.toml> --start 2021-12-01 --end 2021-12-03
#
# Configs:
# - ERA5 spectral:   `[input].spectral_dir = "..."` — `ERA5SpectralSettings`.
# - Native sources:  `[source].toml = "config/met_sources/<src>.toml"` plus
#                    `[source].root_dir = "..."` — typed dispatch.
# ===========================================================================

using Logging
using TOML
using Dates

using AtmosTransport
using AtmosTransport.Preprocessing

function _parse_cli(args::Vector{String})
    usage = "Usage: preprocess_transport_binary.jl <config.toml> " *
            "[--day YYYY-MM-DD | --start YYYY-MM-DD --end YYYY-MM-DD]"
    isempty(args) && throw(ArgumentError(usage))
    cfg_path = expanduser(args[1])
    startswith(args[1], "-") && throw(ArgumentError("missing <config.toml>; $usage"))
    isfile(cfg_path) || throw(ArgumentError("Config not found: $cfg_path"))

    values = Dict{String, Union{Nothing, String}}(
        "--day" => nothing, "--start" => nothing, "--end" => nothing)
    i = 2
    while i <= length(args)
        flag = args[i]
        haskey(values, flag) || throw(ArgumentError("unknown option: $flag"))
        values[flag] === nothing || throw(ArgumentError("duplicate option: $flag"))
        i < length(args) || throw(ArgumentError("missing value after $flag"))
        value = args[i + 1]
        startswith(value, "--") && throw(ArgumentError("missing value after $flag"))
        tryparse(Date, value) === nothing &&
            throw(ArgumentError("$flag must be YYYY-MM-DD; got $(repr(value))"))
        values[flag] = value
        i += 2
    end

    day_override = values["--day"]
    start_date = values["--start"]
    end_date = values["--end"]
    day_override === nothing || (start_date === nothing && end_date === nothing) ||
        throw(ArgumentError("--day cannot be combined with --start or --end"))
    (start_date === nothing) == (end_date === nothing) ||
        throw(ArgumentError("--start and --end must be provided together"))
    start_date === nothing || Date(start_date) <= Date(end_date) ||
        throw(ArgumentError("--start must be on or before --end"))
    return cfg_path, day_override, start_date, end_date
end

function main()
    base_logger = ConsoleLogger(stderr, Logging.Info; show_limited = false)
    global_logger(AtmosTransport.Preprocessing._FlushingLogger(base_logger))

    cfg_path, day_override, start_date, end_date = _parse_cli(ARGS)
    cfg = TOML.parsefile(cfg_path)

    process_day(cfg; day_override = day_override,
                     start_date   = start_date,
                     end_date     = end_date)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
