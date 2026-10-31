#!/usr/bin/env julia
#
# Bitwise gate of the cubed-sphere adjoint tape, full matrix (opt-in, about
# 10 minutes on one CPU thread): every split-sweep scheme (upwind, slopes, PPM,
# monotone and CW84 PPM) × Float64 and Float32 × five physics families
# (convection operator, cadence, diffusion coupling, reset mode, CFL
# subcycles), each with both host storages. Variants must equal the reference
# variant bit for bit and the limited schemes must meet every limiter branch;
# the core set (`core/test_cs_adjoint_tape_bitwise.jl`) also compares
# recorded digests. A same-host A/B of the full matrix against main uses the
# harness command line (`--record` on main, `--check` on the branch).
using Test

include(joinpath(@__DIR__, "..", "helpers", "cs_adjoint_tape_harness.jl"))
using .CSAdjointTapeHarness: full_cases, missing_limiter_codes, run_subprocess, variant_mismatches

@testset "adjoint tape bitwise gate (full matrix)" begin
    out = run_subprocess("full")
    @test out["threads"] == 1
    @test issetequal(keys(out["cases"]), [c.name for c in full_cases()])
    for (problems, what) in ((variant_mismatches(out), "tape variants differ from the reference"),
                             (missing_limiter_codes(out), "limiter branches never occur"))
        isempty(problems) || @error "$(length(problems)) $what:\n  " * join(problems, "\n  ")
        @test isempty(problems)
    end
end
