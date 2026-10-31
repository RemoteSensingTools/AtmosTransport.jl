#!/usr/bin/env julia
#
# Bitwise gate of the cubed-sphere adjoint tape (core set).
#
# Every `AdjointTape` storage (device, pinned host, mmap) and checkpoint
# schedule (Full, Stride with K not dividing the met windows) must give the
# gradients of the default tape (device, FullCheckpoint) bit for bit:
# footprints and initial adjoints of the objective, seeded and multi-objective
# entry points (the Jacobian also under Stride, whose windows serve both
# objectives), the 4D-Var cost, simulated values and gradient (on two cases
# with every observation operator, background terms and the preconditioned
# space), and a two-iteration 4D-Var optimization whose evaluations share one
# tape pool. Two
# further cases run every entry point on inputs that `cs_adjoint_inputs`
# captures from a production simulation, with each met window's fluxes shared
# by its steps. The fields are rough, so the limited PPM reverse runs through
# every limiter branch on every axis and at the seams, which the test
# requires. On the host that recorded
# `fixtures/cs_adjoint_tape_digests.toml` the reference digests must also equal
# the recorded ones, so a change of the tape internals has to reproduce the
# recorded gradients bit for bit. Every array the tape frees explicitly is
# filled with NaN when it is freed, so a free placed before the last read of
# an array (a use-after-free on a GPU, a no-op on the CPU) changes the digests.
#
# The cases run in a single-threaded subprocess (`helpers/cs_adjoint_tape_harness.jl`,
# whose header documents its command line): CPU atomics are reproducible only on
# one thread. The full matrix is `diagnostic/test_cs_adjoint_tape_bitwise_full.jl`.
using Test
using TOML

include(joinpath(@__DIR__, "..", "helpers", "cs_adjoint_tape_harness.jl"))
using .CSAdjointTapeHarness: CORE_CASES, entry_failures, missing_limiter_codes,
    reference_mismatches, run_subprocess, same_host, variant_mismatches

const FIXTURE = joinpath(@__DIR__, "..", "fixtures", "cs_adjoint_tape_digests.toml")

# Report every problem, not only the count.
function _report(problems, what)
    isempty(problems) || @error "$(length(problems)) $what:\n  " * join(problems, "\n  ")
    return problems
end

@testset "adjoint tape bitwise gate (core)" begin
    out = run_subprocess("core")
    @test out["threads"] == 1
    @test issetequal(keys(out["cases"]), [c.name for c in CORE_CASES])
    @test isempty(_report(entry_failures(out), "entry points failed"))
    @test isempty(_report(variant_mismatches(out), "tape variants differ from the reference"))
    # Freed tape arrays were poisoned (NaN), so a premature free would show.
    @test out["poisoned_arrays"] > 0
    @test isempty(_report(missing_limiter_codes(out), "limiter branches never occur"))

    reference = TOML.parsefile(FIXTURE)
    @test issetequal(keys(reference["cases"]), [c.name for c in CORE_CASES])
    if same_host(out, reference)
        @test isempty(_report(reference_mismatches(out, reference),
                              "reference digests differ from the recording"))
    else
        @info "Reference digests not compared: recorded on $(reference["host"]), " *
              "this host is $(out["host"]) (variants are compared within the run)"
    end
end
