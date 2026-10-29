"""
The global-total and twin-run numerics of `AtmosTransport.Diagnostics`:
`drift_statistics` and `twin_relative_difference` (`src/Diagnostics/mass_totals.jl`).

Asserts:
- `drift_statistics`: drift and increment statistics against a direct
  computation and against values pinned from `increment_stats` of
  `float32_conservation_attribution.py` (≤ 1e-15 relative); the degenerate
  cases (constant series: t = 0; equal non-zero increments: t = ±Inf; two
  values: NaN spread); errors for short, non-finite or non-positive-start
  series;
- `twin_relative_difference`: raw and net series by hand; the net value is
  bitwise the Python `f32_minus_f64` expression for positive, negative and
  zero starts (pinned values too); NaN where the reference is zero; errors,
  including non-finite totals;
"""

using Test
using Statistics: mean, std
using AtmosTransport.Diagnostics: drift_statistics, twin_relative_difference

relerr(a, b) = abs(a - b) / abs(b)

@testset "drift_statistics" begin
    s = [10.0, 11.0, 13.0, 12.0]
    r = drift_statistics(s)
    d = [0.1, 0.2, -0.1]
    @test r.drift ≈ 0.2 rtol = 1e-15
    @test r.n == 3
    @test r.increment_mean ≈ mean(d) rtol = 1e-15
    @test r.increment_std ≈ sqrt(sum(abs2, d .- mean(d)) / 2) rtol = 1e-15
    @test r.t ≈ mean(d) / (r.increment_std / sqrt(3)) rtol = 1e-15

    # Pinned from increment_stats and the drift column of the Python script
    # (NumPy 1.26), on a global total with ~1e-9 relative noise.
    total = [2.0541261065359e15, 2.05412610539308e15, 2.05412610546246e15, 2.05412610736816e15,
             2.05412610870099e15, 2.05412611142221e15, 2.05412611427735e15]
    p = drift_statistics(total)
    @test p.drift == 3.768731615536138e-9
    @test relerr(p.increment_mean, 6.281219359226898e-10) <= 1e-15
    @test relerr(p.increment_std, 7.631671620718519e-10) <= 1e-15
    @test relerr(p.t, 2.016043556018314) <= 1e-15
    @test drift_statistics(Float32.(s)).drift ≈ 0.2 rtol = 1e-7     # converted to Float64

    # Degenerate spreads.
    c = drift_statistics(fill(2.0e15, 9))
    @test (c.drift, c.increment_mean, c.increment_std, c.t) == (0.0, 0.0, 0.0, 0.0)
    lin = drift_statistics([4.0, 5.0, 6.0, 7.0])
    @test lin.increment_std == 0.0 && lin.t == Inf
    @test drift_statistics([4.0, 3.0, 2.0]).t == -Inf
    two = drift_statistics([4.0, 5.0])
    @test two.n == 1 && two.drift == 0.25 && isnan(two.increment_std) && isnan(two.t)

    @test_throws ArgumentError drift_statistics([1.0])
    @test_throws ArgumentError drift_statistics([0.0, 1.0])
    @test_throws ArgumentError drift_statistics([-1.0, 1.0])
    @test_throws ArgumentError drift_statistics([1.0, NaN, 2.0])
end

@testset "twin_relative_difference" begin
    r = twin_relative_difference([99.0, 202.0, 404.0], [100.0, 200.0, 400.0])
    @test r.raw == [(99.0 - 100.0) / 100.0, (202.0 - 200.0) / 200.0, (404.0 - 400.0) / 400.0]
    @test r.net == r.raw .- r.raw[1]
    @test r.net[1] == 0

    # Positive start: net is the Python (a_k - b_k)/b_k - (a_0 - b_0)/b_0, bitwise.
    a = [1.0e15, 1.0000000123e15, 1.0000000377e15]
    b = [1.0000000253e15, 1.0000000301e15, 1.0000000599e15]
    net = twin_relative_difference(a, b).net
    @test net[end] == (a[3] - b[3]) / b[3] - (a[1] - b[1]) / b[1]
    @test net[end] == 3.1000006896899347e-9                  # f32_minus_f64 (Python)

    # A signed diagnostic may start below zero. Its net difference still
    # removes the initial offset; only an exactly zero reference is special.
    neg = twin_relative_difference([-99.0, -202.0, -396.0], [-100.0, -200.0, -400.0])
    @test neg.net == neg.raw .- neg.raw[1]
    @test neg.net == [0.0, 0.02, 0.0]

    # Zero start (a tracer emitted from zero): relative to the emitted mass,
    # the Python (a_k - b_k)/(b_k - b_0) with b_0 = 0; NaN where b is zero.
    a0 = [0.0, 3.2e9, 6.4000001e9]
    b0 = [0.0, 3.2e9 + 17.0, 6.4e9]
    z = twin_relative_difference(a0, b0)
    @test isnan(z.raw[1]) && isnan(z.net[1])
    @test z.net[2:3] == z.raw[2:3]
    @test z.net[3] == (a0[3] - b0[3]) / (b0[3] - b0[1]) == 1.5625e-8
    @test twin_relative_difference(Float32[1, 2], [1.0, 2.0]).raw == [0.0, 0.0]

    @test_throws DimensionMismatch twin_relative_difference([1.0, 2.0], [1.0])
    @test_throws ArgumentError twin_relative_difference(Float64[], Float64[])
    @test_throws ArgumentError twin_relative_difference([NaN, 2.0], [1.0, 2.0])   # e.g. a fill value
    @test_throws ArgumentError twin_relative_difference([1.0, 2.0], [1.0, Inf])
end
