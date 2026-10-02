#!/usr/bin/env julia
"""
JET.jl static analysis (plan 21 Phase 6B/6C).

This test runs JET.report_package on AtmosTransport and records the
number of inference errors filtered to hot-path modules. It's
structured as a **snapshot gate**: the test passes if the count is
at or below a documented baseline, and fails loudly if new reports
appear. That catches regressions without making CI red from day
one (plan 21 budget for JET iteration was 1 day).

## Known-tolerated noise sources

Two patterns dominate the current report count and are NOT bugs:

1. **KernelAbstractions `kwcall` dispatch** — JET can't prove that
   `kernel(args...; ndrange=(...))` dispatches to the CPU or GPU
   backend because `KA.Kernel` is generic over its Backend type
   parameter. Every KA-using package hits this. Documented at
   https://github.com/aviatesk/JET.jl/issues/?q=KernelAbstractions.
   JET/Julia 1.12 reports more of these known-tolerated paths than
   the Julia 1.10 JET stack, so the baseline is version-aware. The
   exact-TM5 `dkg` diffusion path adds two kernels and four reports
   (the failed GPU union-split branch and its propagated return type
   for each kernel), without introducing runtime type instability.
   Conservative CS seam caching and paired application add two more kernels
   with the same four-report pattern on Julia 1.12 (measured separately with
   the V100 paths passing through 65 tracers). The two conservative Dkg mass
   kernels add the same four-report pattern; their CPU and V100 reference and
   transpose checks pass through 65 tracers.
   Lin-Rood halo-gradient kernels add two instances of the same unresolved
   GPU `kwcall` pattern. Their Float32/Float64 transporting-footprint and
   checkpoint checks pass with scalar GPU indexing disabled.

2. **Parametric `@kwdef` zero-arg constructors** — `Base.@kwdef`
   auto-generates a zero-arg constructor for
   `struct Foo{FT}; x::FT = FT(1.0); end`, but `FT` is unbound at
   that call site. Users must call `Foo{Float64}()` explicitly.
   Documented in `CLAUDE.md` under "Julia / language gotchas".

## Escape hatch

Set `ATMOSTRANSPORT_JET_ADVISORY=1` to demote any count increase to
a warning instead of a test failure. Useful during local dev when
intermediate refactors temporarily surface new reports.

## Expected behavior

- Baseline count stable: test passes silently.
- Count drops: test passes and prints a "baseline can tighten" hint.
- Count rises: test fails (unless advisory env var is set), printing
  the new reports so the author can triage.
"""

using Test
using JET
using AtmosTransport

# Invocation:
#   julia --project=test test/test_jet.jl        (targeted)
#   julia --project=. -e 'using Pkg; Pkg.test()' (full suite)

# Hot-path modules that plan 21 cares about. IO-heavy modules
# (MetDrivers, Preprocessing) are excluded because they carry
# ~120 JSON3 header-parsing union-split reports that are known-
# tolerated (header field types are guaranteed by the preprocessor
# but JET can't prove it).
const HOT_PATH_MODULES = (
    AtmosTransport.Operators,
    AtmosTransport.State,
    AtmosTransport.Models,
    AtmosTransport.Grids,
)

# Snapshot baselines captured during CI runs. Dominant sources are the
# known-tolerated patterns documented above.
# The 1.12 baseline was re-measured with JET 0.11.5 after conservative Dkg:
# 148 prior reports plus four from the two new mass-kernel call sites above.
# Lin-Rood halo gradients add two kernel-dispatch reports (152 → 154),
# confirmed by comparing all reports with the pre-fix source on the same stack.
# The 1.10 baseline retains its prior CI-measured allowance. An isolated
# compatible JET run produced zero reports, so it is not comparable evidence
# for tightening that allowance.
# Keep these at the expected counts so the snapshot remains a real gate.
const JET_HOT_PATH_BASELINE_1_10 = 130
const JET_HOT_PATH_BASELINE_1_12 = 153
const JET_HOT_PATH_BASELINE =
    VERSION >= v"1.12" ? JET_HOT_PATH_BASELINE_1_12 :
                         JET_HOT_PATH_BASELINE_1_10

const ADVISORY_ONLY = get(ENV, "ATMOSTRANSPORT_JET_ADVISORY", "0") == "1"

@testset "JET: hot-path inference snapshot" begin
    result = JET.report_package(AtmosTransport;
                                target_modules = HOT_PATH_MODULES,
                                toplevel_logger = nothing)
    reports = JET.get_reports(result)
    n = length(reports)

    @info "JET hot-path modules: $n reports (baseline $JET_HOT_PATH_BASELINE)"

    if n > JET_HOT_PATH_BASELINE
        # Print new reports so the author can triage
        println()
        println("⚠  JET report count rose above the baseline.")
        println("   Baseline: $JET_HOT_PATH_BASELINE")
        println("   Current:  $n")
        println("   Delta:    +$(n - JET_HOT_PATH_BASELINE)")
        println()
        println("   First 20 reports:")
        for (i, rep) in enumerate(first(reports, 20))
            println("   [$i] ", rep)
        end
        println()
        println("   If the new reports are genuine bugs: fix them.")
        println("   If they're known-tolerated patterns (see docstring):")
        println("     1. Update JET_HOT_PATH_BASELINE in this file.")
        println("     2. Document the pattern in the 'Known-tolerated")
        println("        noise sources' section of the docstring.")
        println()

        if ADVISORY_ONLY
            @warn "ATMOSTRANSPORT_JET_ADVISORY=1 set — demoting to warning."
            @test true
        else
            @test n <= JET_HOT_PATH_BASELINE
        end
    elseif n < JET_HOT_PATH_BASELINE
        @info "JET baseline can be tightened: $n < $JET_HOT_PATH_BASELINE. " *
              "Lower JET_HOT_PATH_BASELINE in test_jet.jl when this is stable."
        @test true
    else
        @test true
    end
end
