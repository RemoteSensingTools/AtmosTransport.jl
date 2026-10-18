#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# OCO-2 / OCO-3 v11 MIP 10-second files → observation files of schema v2.
#
# Usage:
#     julia --project=. scripts/preprocessing/oco2_mip_to_observations.jl INPUT OUTPUT [options]
#
#     INPUT   a MIP 10-second file (OCO2_b11.2_10sec_GOOD_r2.nc4, ..._r3.nc4,
#             OCO3_b11_10sec_GOOD_r2.nc4); read only
#     OUTPUT  output path template with {YYYYMMDD} (one file per day, the
#             default) or {YYYYMM} (--partition month)
#
# Options:
#     --start YYYY-MM-DD[THH:MM:SS]   first record time (inclusive; default: the file's first)
#     --stop  YYYY-MM-DD[THH:MM:SS]   last record time (exclusive; default: after the file's last)
#     --data-types LN,LG,OG           the experiment's data types (flag 0 when assimilable; SAM for OCO-3 maps)
#     --assimilate-flags 1            MIP assimilate_flag values to assimilate (e.g. 1,2)
#     --model-sigma file              the MIP model_error, or a constant in ppm (e.g. 0.5)
#     --mapping hat                   hat: the 20 levels as given (default); tm5: TM5's 19 layers
#     --partition day                 day or month
#     --tracer CO2                    the records' tracer name
#     --instrument OCO2               instrument label (default: from the file's ShortName)
#
# Every record of the time range is written; records outside the experiment's
# data types and assimilation flags get flag = 1 + assimilate_flag, so that
# `bind_to_mesh(...; flagged_records = "cosample")` co-samples them (the MIP
# asks for co-samples of all records). See `convert_oco2_mip_observations`.
# ---------------------------------------------------------------------------

using Dates

import AtmosTransport
const AT = AtmosTransport

const _OPTIONS = ("--start", "--stop", "--data-types", "--assimilate-flags", "--model-sigma", "--mapping",
                  "--partition", "--tracer", "--instrument")

function _parse_arguments(args)
    positional = String[]
    options = Dict{String, String}()
    k = 1
    while k <= length(args)
        arg = args[k]
        if startswith(arg, "--")
            arg in _OPTIONS || error("unknown option $(arg); options: $(join(_OPTIONS, ", "))")
            k < length(args) || error("option $(arg) needs a value")
            options[arg] = args[k + 1]
            k += 2
        else
            push!(positional, arg)
            k += 1
        end
    end
    length(positional) == 2 || error("usage: oco2_mip_to_observations.jl INPUT OUTPUT [options]")
    return positional[1], positional[2], options
end

_time(s) = occursin('T', s) ? DateTime(s) : DateTime(Date(s))
_list(s) = [String(strip(x)) for x in split(s, ',') if !isempty(strip(x))]

function main(args = ARGS)
    input, output, options = _parse_arguments(args)
    has_range = haskey(options, "--start") || haskey(options, "--stop")
    time_range = has_range ? (_time(get(options, "--start", "0001-01-01")), _time(get(options, "--stop", "9999-01-01"))) :
                 nothing
    model_sigma = get(options, "--model-sigma", "file")
    paths = AT.convert_oco2_mip_observations(input, output;
        partition = Symbol(get(options, "--partition", "day")),
        time_range,
        data_types = _list(get(options, "--data-types", "LN,LG,OG")),
        assimilate_flags = parse.(Int, _list(get(options, "--assimilate-flags", "1"))),
        model_sigma = model_sigma == "file" ? :file : parse(Float64, model_sigma),
        mapping = AT.Output.oco2_level_form_from_label(get(options, "--mapping", "hat"), "--mapping"),
        tracer = get(options, "--tracer", "CO2"),
        instrument = get(options, "--instrument", nothing))
    println("wrote $(length(paths)) observation files", isempty(paths) ? "" : ": $(first(paths)) … $(last(paths))")
    return paths
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
