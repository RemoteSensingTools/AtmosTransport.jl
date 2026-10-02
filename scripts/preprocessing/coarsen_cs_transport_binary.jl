#!/usr/bin/env julia

# EXPERIMENTAL: nested cubed-sphere operator restriction (for example C90->C30).
# Outputs pass replay and positivity gates but still require comparison against
# a directly preprocessed target-grid binary before scientific production use.

using ArgParse
using AtmosTransport

# The active C90 campaign is pinned to a format-v4 runtime worktree. Allow
# this development-tree CLI to supply the experimental coarsener to that
# module without copying or modifying the production worktree.
if !isdefined(AtmosTransport.Preprocessing, :coarsen_nested_cs_transport_binary)
    Base.include(
        AtmosTransport.Preprocessing,
        joinpath(@__DIR__, "..", "..", "src", "Preprocessing",
                 "transport_binary", "cubed_sphere_coarsen.jl"))
end
const coarsen_nested_cs_transport_binary =
    AtmosTransport.Preprocessing.coarsen_nested_cs_transport_binary

function settings()
    s = ArgParseSettings(
        description = "EXPERIMENTAL nested CS transport-binary coarsener; testing use only")
    @add_arg_table! s begin
        "input"
            help = "source cubed-sphere transport binary"
            required = true
        "output"
            help = "target coarsened transport binary"
            required = true
        "--target-nc"
            help = "target cubed-sphere edge resolution"
            dest_name = "target_nc"
            arg_type = Int
            default = 30
        "--substep-cfl-target"
            help = "adaptive target below the hard positivity limit"
            dest_name = "substep_cfl_target"
            arg_type = Float64
            default = 0.90
        "--positivity-cfl-limit"
            help = "hard write-time positivity limit"
            dest_name = "positivity_cfl_limit"
            arg_type = Float64
            default = 0.95
        "--max-steps-per-window"
            help = "abort if a window requires more substeps"
            dest_name = "max_steps_per_window"
            arg_type = Int
            default = 4096
        "--force"
            help = "replace an existing output"
            action = :store_true
        "--mark-gated"
            help = "write .coarsen-gated after structural/replay gates pass (not scientific validation)"
            dest_name = "mark_gated"
            action = :store_true
    end
    return s
end

function main(args)
    parsed = parse_args(args, settings())
    @warn "EXPERIMENTAL C90-to-coarser-CS operator restriction: testing only; direct-target and tracer validation are still required"
    result = coarsen_nested_cs_transport_binary(
        parsed["input"], parsed["output"];
        target_Nc = parsed["target_nc"],
        substep_cfl_target = parsed["substep_cfl_target"],
        positivity_cfl_limit = parsed["positivity_cfl_limit"],
        max_steps_per_window = parsed["max_steps_per_window"],
        force = parsed["force"],
        mark_gated = parsed["mark_gated"])
    println("EXPERIMENTAL COARSEN COMPLETE")
    println("  input:  ", result.input)
    println("  output: ", result.output)
    println("  grid:   C", result.source_Nc, " -> C", result.target_Nc)
    println("  steps:  ", result.schedule)
    println("  replay: ", result.worst_replay_rel)
    println("  CFL:    ", result.worst_positivity_ratio)
    println("  size:   ", round(result.input_bytes / result.output_bytes; digits=2), "x smaller")
    println("  time:   ", round(result.elapsed_seconds; digits=2), " s")
    return 0
end

exit(main(ARGS))
