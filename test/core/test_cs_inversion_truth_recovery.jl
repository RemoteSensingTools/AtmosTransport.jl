#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan 26 stretch — synthetic 4D-Var truth-recovery regression test.
#
# Loads `scripts/inversions/synthetic_experiment.jl`, runs the
# truth-emission → forward → noisy-obs → invert pipeline once with a
# deterministic seed, and asserts the inversion recovers the truth
# field within bounds.
#
# Acceptance bands (regression-grade, not science-grade):
#   - Cost reduction:        final / initial < 0.05
#                            (>20× reduction over the 15-iteration
#                            budget).
#   - Peak ratio:            recovered_peak / truth_peak in
#                            (0.5, 1.5) — the truth's central cell
#                            should be reconstructed within a factor
#                            of 2.
#   - Relative L2 error:     <= 0.5 — the recovered field is
#                            within 50% of the truth in L2 norm
#                            (the prior pulls cells far from the
#                            obs patch toward zero, contributing to
#                            this number).
# ---------------------------------------------------------------------------

using Test

include(joinpath(@__DIR__, "..", "..", "scripts", "inversions",
                  "synthetic_experiment.jl"))

@testset "synthetic 4D-Var inversion — truth recovery" begin
    result = run_synthetic_experiment()
    @test result isa SyntheticExperimentResult

    # Cost reduction: L-BFGS should drive the χ-space cost well
    # below the initial-prior cost.
    @test result.cost_reduction_ratio < 0.05

    # Recovered peak is near the truth peak.
    @test 0.5 < result.peak_ratio < 1.5

    # L2 closeness — the prior pulls unobserved cells toward zero,
    # so the bound is loose. Cells observed by the patch should be
    # nearly exact; the residual error is concentrated in
    # unobserved cells where the prior dominates.
    @test result.relative_l2_error < 0.5

    # Iteration log was requested.
    @test result.solve.log isa AT.CSIterationLog
    @test !isempty(result.solve.log)

    # Recovered emissions live on the configured panel (panel 1 by
    # default); other panels should be near zero.
    cfg = result.config
    rec = result.recovered_emissions
    truth = result.truth_emissions
    truth_peak = maximum(p -> maximum(truth[p]), 1:6)
    @test maximum(rec[cfg.obs_panel]) > 0.5 * truth_peak
    @inbounds for p in 1:6
        p == cfg.obs_panel && continue
        @test maximum(abs, rec[p]) < 0.1 * truth_peak
    end

    # Cost-decomposition invariant on the final result.
    @test result.solve.last.cost ≈ result.solve.last.observation_cost +
                                    result.solve.last.background_cost atol = 1e-9
end



# Issue #16: the experiment used to hard-code `PPMScheme(NoLimiter())`; the
# scheme is now a config field, checked for adjoint support.
