#!/usr/bin/env julia
#
# Lab lane: benchmark one published commit on the lab machines
# (benchmarking/lab/machines.toml), copy the results back into
# benchmarking/lab/results/<commit>/, validate them, and with --publish hand
# them to the Benchmarks workflow for the dashboard. Run from wurst:
#
#   julia --project=benchmarking benchmarking/lab/run_lab_benchmarks.jl --dry-run
#   julia --project=benchmarking benchmarking/lab/run_lab_benchmarks.jl --machines=wurst,curry,mac --publish
#
# Only Base, TOML, JSON3, and Dates are needed; see benchmarking/README.md.

include(joinpath(@__DIR__, "LabLane.jl"))
using .AtmosTransportLabLane: parse_lab_options, run_lab, LAB_USAGE

function main(args)
    opts = parse_lab_options(args)
    if opts.help
        print(LAB_USAGE)
        return nothing
    end
    run_lab(opts)
    return nothing
end

main(ARGS)
