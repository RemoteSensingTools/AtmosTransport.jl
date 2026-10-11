"""
Seasonal, trend and interannual decompositions of `AtmosTransport.Analysis`
(`src/Analysis/decomposition.jl`), plus the skill metrics used to score them.

The first four test sets port `test/diagnostics/test_trendy_hovmoller.py`,
which tested `decompose` and `metrics` of
`scripts/diagnostics/rank_trendy_hovmoller.py`, case by case:
- a noise-free trend plus seasonal cycle is removed to 1e-10 with a polar
  observation gap (a band observed only January-July, so five calendar-month
  columns of the design are zero and the minimum-norm solution is needed);
  unobserved seasonal months and unobserved anomalies are NaN;
- a constant carrier (+1300) does not change the anomalies or the cycle;
- the decomposition is linear: decompose(x) + decompose(fire) =
  decompose(x + fire) for anomaly, seasonal cycle and detrended series;
- correlation and RMSE separate pattern from amplitude.
The random inputs are drawn with Julia's RNG; the cases need only the
properties, not NumPy's random stream.

Further asserts:
- the anomaly equals an independent least-squares solve with the all-zero
  columns dropped (QR), so the pseudoinverse gives the minimum-norm fit;
- `first_month` labels the seasonal cycle by calendar month;
- fewer than `min_months` observed months leave a location all NaN, and a
  non-finite value at an observed month is an error;
- values pinned from the Python `decompose`, `metrics` and `interannual`
  (`compare_sif_gpp_transport_control.py`) on small deterministic inputs agree
  to 1e-12;
- `doy_linear_detrend` removes a per-calendar-day linear evolution exactly,
  folds 29 February onto 28 February, gives 0 for a day seen once, and a NaN
  sample makes only its own day and series NaN.
"""

using Test
using Dates
using LinearAlgebra
using Random
using Statistics
using AtmosTransport.Analysis: seasonal_trend_decomposition, skill_metrics, doy_linear_detrend

# Element-wise |a - b| ≤ atol with NaN equal to NaN (numpy assert_allclose).
function allclose_nan(a, b; atol)
    size(a) == size(b) || return false
    return all((isnan(x) && isnan(y)) || abs(x - y) <= atol for (x, y) in zip(a, b))
end

const NB, NT = 40, 120

function hovmoller_case()
    rng = Xoshiro(47)
    t = [(k - 1 - 59.5) / 12 for k in 1:NT]
    cycle = [3 * sin((k - 1) * 2π / 12) for k in 1:NT]
    x = repeat(transpose(400 .+ 0.7 .* t .+ cycle), NB, 1)       # (band, month)
    seen = rand(rng, NB, NT) .> 0.15
    seen[1, :] .= [mod(k - 1, 12) < 7 for k in 1:NT]                # band 1 observed Jan-Jul only
    return rng, x, seen
end

@testset "trend and season removed with polar gaps" begin
    _, x, seen = hovmoller_case()
    d = seasonal_trend_decomposition(x, seen)
    @test maximum(abs, filter(!isnan, d.anomaly)) < 1e-10
    @test all(isnan, d.seasonal[1, 8:12])
    @test all(isnan, d.anomaly[.!seen])
    @test all(isnan, d.detrended[.!seen])
    @test !any(isnan, d.anomaly[seen])
    # The cycle is recovered: 3 sin(2π(m-1)/12) minus its mean (zero), all months seen in band 2.
    @test maximum(abs, d.seasonal[2, :] .- [3 * sin((m - 1) * 2π / 12) for m in 1:12]) < 1e-10
end

@testset "carriers do not change anomalies" begin
    rng, x, seen = hovmoller_case()
    noisy = x .+ randn(rng, size(x))
    d = seasonal_trend_decomposition(cat(noisy, noisy .+ 1300; dims = 3), seen)
    @test allclose_nan(d.anomaly[:, :, 1], d.anomaly[:, :, 2]; atol = 1e-10)
    @test allclose_nan(d.seasonal[:, :, 1], d.seasonal[:, :, 2]; atol = 1e-10)
    @test allclose_nan(d.detrended[:, :, 1], d.detrended[:, :, 2]; atol = 1e-10)
end

@testset "fire addition commutes with the shared projection" begin
    rng, x, seen = hovmoller_case()
    fire = randn(rng, size(x))
    d = seasonal_trend_decomposition(cat(x, fire, x .+ fire; dims = 3), seen)
    for f in (d.anomaly, d.seasonal, d.detrended)
        @test allclose_nan(f[:, :, 1] .+ f[:, :, 2], f[:, :, 3]; atol = 1e-10)
    end
end

@testset "correlation and RMSE distinguish pattern from amplitude" begin
    x = collect(0.0:19.0) .- 9.5
    doubled = skill_metrics(2 .* x, x)
    @test doubled.r ≈ 1
    @test doubled.amplitude_ratio ≈ 2
    @test doubled.rmse > 0
    @test skill_metrics(-x, x).r ≈ -1
    @test skill_metrics(x, x).rmse == 0
end

@testset "pseudoinverse gives the minimum-norm fit" begin
    rng, x, seen = hovmoller_case()
    noisy = x .+ randn(rng, size(x))
    d = seasonal_trend_decomposition(noisy, seen)
    # Band 1: drop the zero (never observed) month columns and solve by QR.
    ok = findall(seen[1, :])
    t = [(k - (NT + 1) / 2) / 12 for k in 1:NT]
    D = hcat([Float64(mod(k - 1, 12) + 1 == m) for k in 1:NT, m in 1:12], t)[ok, :]
    used = [j for j in 1:13 if any(!iszero, D[:, j])]
    @test length(used) == 8                                   # 7 months + trend
    v = noisy[1, ok] .- mean(noisy[1, ok])
    coef = qr(D[:, used]) \ v
    @test maximum(abs, d.anomaly[1, ok] .- (v .- D[:, used] * coef)) < 1e-11
    @test maximum(abs, d.detrended[1, ok] .- ((v .- coef[end] .* t[ok]) .- mean(v .- coef[end] .* t[ok]))) < 1e-11
end

@testset "first_month labels calendar months" begin
    nt = 96
    k = 1:nt
    # Series starting in April: calendar month of entry k is mod(k + 2, 12) + 1.
    cal = [mod(i + 2, 12) + 1 for i in k]
    cycle_of = m -> 2 * cos(2π * (m - 1) / 12)
    x = reshape([5.0 + 0.1 * i + cycle_of(cal[i]) for i in k], 1, nt)
    d = seasonal_trend_decomposition(x, trues(1, nt); first_month = 4)
    @test maximum(abs, d.seasonal[1, :] .- [cycle_of(m) for m in 1:12]) < 1e-10
    @test maximum(abs, d.anomaly) < 1e-10
    @test_throws ArgumentError seasonal_trend_decomposition(x, trues(1, nt); first_month = 13)
end

@testset "thresholds and input checks" begin
    x = randn(Xoshiro(1), 2, 60)
    seen = trues(2, 60)
    seen[2, 24:end] .= false                                  # 23 observed months
    d = seasonal_trend_decomposition(x, seen)
    @test all(isnan, d.anomaly[2, :]) && all(isnan, d.seasonal[2, :]) && all(isnan, d.detrended[2, :])
    @test !any(isnan, d.anomaly[1, :])
    @test all(!isnan, seasonal_trend_decomposition(x, seen; min_months = 23).anomaly[2, 1:23])
    # 60 months = 5 years per calendar month: valid at 5, NaN at 6.
    @test !any(isnan, d.seasonal[1, :])
    @test all(isnan, seasonal_trend_decomposition(x, seen; min_years_per_month = 6).seasonal[1, :])
    bad = copy(x); bad[1, 10] = NaN
    @test_throws ArgumentError seasonal_trend_decomposition(bad, seen)
    bad[1, 10] = x[1, 10]; bad[2, 40] = NaN                   # not observed: never read
    @test isequal(seasonal_trend_decomposition(bad, seen).anomaly, d.anomaly)
    @test_throws DimensionMismatch seasonal_trend_decomposition(x, trues(2, 59))
    @test_throws ArgumentError seasonal_trend_decomposition(randn(2, 3, 4, 5), trues(2, 3))
end

# Pinned from rank_trendy_hovmoller.decompose (NumPy lstsq) on the input below:
# 72 months, 3 locations, 2 series; location 2 misses calendar months 7-9 in the
# first two years and month 12 always; location 3 has 20 observed months.
@testset "pinned against the Python decompose" begin
    nt, nb, ns = 72, 3, 2
    t = [(k - 1 - (nt - 1) / 2) / 12 for k in 1:nt]
    x = [400 + 0.5 * t[k] + 2 * sin(2π * (k - 1) / 12 + (b - 1)) + 0.3 * sin(1.7 * (k - 1) + (s - 1) + (b - 1)) +
         0.05 * (s - 1) * t[k]^2 for b in 1:nb, k in 1:nt, s in 1:ns]
    seen = trues(nb, nt)
    for k in 1:nt
        m0 = mod(k - 1, 12)
        ((k - 1 < 24 && m0 in (6, 7, 8)) || m0 == 11) && (seen[2, k] = false)
    end
    seen[3, :] .= false
    seen[3, 1:20] .= true
    d = seasonal_trend_decomposition(x, seen)
    # Julia indices (location, month, series) are Python's 0-based (b, k, s) plus one.
    @test d.anomaly[1, 1, 1] ≈ -0.04656730693277367 atol = 1e-12
    @test d.anomaly[1, 14, 2] ≈ -0.22513168566771818 atol = 1e-12
    @test d.anomaly[2, 31, 1] ≈ 0.3042634083623743 atol = 1e-12
    @test isnan(d.anomaly[2, 72, 2])
    @test d.anomaly[2, 6, 1] ≈ 0.037105198457231925 atol = 1e-12
    seasonal_b1s1 = [0.04462462228823629, 1.0431136574296693, 1.6690711408122798, 1.9657781487993595,
                     1.796419477441448, 1.01011293194815, -0.0745887136024676, -0.9985985250125083,
                     -1.6656217525798, -2.0264102556304184, -1.7996572058542812, -0.9642435260396676]
    @test maximum(abs, d.seasonal[1, :, 1] .- seasonal_b1s1) < 1e-12
    seasonal_b2s2 = [1.1581828549018327, 1.3722773213596324, 1.2065810416063172, 0.5891987882077996,
                     -0.4779974825560003, -1.552151274407978, NaN, NaN, NaN, -1.6958673029158844,
                     -0.6002239461957197, NaN]
    @test allclose_nan(d.seasonal[2, :, 2], seasonal_b2s2; atol = 1e-12)
    @test d.detrended[1, 1, 1] ≈ -0.0019426846445373834 atol = 1e-12
    @test d.detrended[1, 41, 2] ≈ 1.54918234370844 atol = 1e-12
    @test d.detrended[2, 51, 2] ≈ 1.3674363373818026 atol = 1e-12
    @test all(isnan, d.anomaly[3, :, :]) && all(isnan, d.seasonal[3, :, :])
end

@testset "pinned against the Python metrics" begin
    model = [1.0, 2.5, NaN, 4.0, 3.2, 0.7, 2.2]
    reference = [1.2, 2.0, 3.0, Inf, 3.0, 1.1, 2.0]
    s = skill_metrics(model, reference)
    @test s.n == 5
    @test s.r ≈ 0.9693188091298008 atol = 1e-14
    @test s.rmse ≈ 0.3255764119219942 atol = 1e-14
    @test s.bias ≈ 0.06000000000000005 atol = 1e-14
    @test s.amplitude_ratio ≈ 1.365886619929854 atol = 1e-14
    # Population moments: amplitude ratio is std(…; corrected = false) ratio.
    x, y = [1.0, 2.5, 3.2, 0.7, 2.2], [1.2, 2.0, 3.0, 1.1, 2.0]
    @test s.amplitude_ratio ≈ std(x; corrected = false) / std(y; corrected = false)
    @test isnan(skill_metrics([1.0, 2.0], [2.0, 1.0]).r)                 # n < 3
    @test isnan(skill_metrics(fill(2.0, 5), collect(1.0:5.0)).r)         # degenerate std
    @test skill_metrics([1.0, 2.0], [2.0, 1.0]; min_n = 2).r ≈ -1
    empty = skill_metrics([NaN], [1.0])
    @test empty.n == 0 && isnan(empty.r) && isnan(empty.rmse) && isnan(empty.bias)
    @test_throws DimensionMismatch skill_metrics([1.0, 2.0], [1.0])
end

@testset "doy_linear_detrend" begin
    # Pinned from compare_sif_gpp_transport_control.interannual (NumPy lstsq).
    dates = Date.(["2019-02-27", "2019-02-28", "2019-03-01", "2020-02-27", "2020-02-28", "2020-02-29",
                   "2020-03-01", "2021-02-27", "2021-02-28", "2021-03-01", "2022-02-27", "2022-02-28",
                   "2022-03-01", "2023-03-02"])
    n = length(dates)
    X = vcat(transpose([sin(0.3 * i) + 0.01 * i * i for i in 0:n-1]),
             transpose([cos(0.7 * i) * (1 + 0.1 * i) for i in 0:n-1]))
    out0 = [-0.2564276989039076, -0.3983649097194673, -0.2763883870151209, 0.22657496260347665,
            0.17481880048120668, 0.33027470111803503, 0.34461242053944174, 0.31613317150476794,
            0.18490772595991878, 0.13994031996647927, -0.28628043520433777, -0.29163631783969324,
            -0.2081643534908001, 0.0]
    out1 = [0.8323737733274097, 1.2146712695953652, -0.043505051622054924, -1.0736046116582065,
            -1.1889604287323687, -1.274534182732442, -0.8963791264137272, -0.34991209666581674,
            1.2829754141435268, 1.9232734076936193, 0.5911429349966133, -0.03415207227408079,
            -0.9833892296578373, 0.0]
    r = doy_linear_detrend(X, dates)
    @test size(r) == size(X)
    @test maximum(abs, r[1, :] .- out0) < 1e-12
    @test maximum(abs, r[2, :] .- out1) < 1e-12
    @test r[:, end] == [0.0, 0.0]                             # a calendar day seen once

    # A per-calendar-day line in years (29 Feb sharing 28 Feb's line) is removed exactly.
    days = collect(Date(2018, 1, 1):Day(1):Date(2024, 12, 31))
    key(d) = (month(d), month(d) == 2 && day(d) == 29 ? 28 : day(d))
    f(d) = 400 + 3 * sin(2π * dayofyear(Date(2001, key(d)...)) / 365)
    g(d) = 0.5 + 0.2 * cos(2π * key(d)[1] / 12)
    series = [f(d) + g(d) * (year(d) - 2021) for d in days]
    @test maximum(abs, doy_linear_detrend(series, days)) < 1e-10
    # Grid-shaped input: time is the last dimension, other dimensions independent.
    field = cat([reshape(series .* s, 1, 1, :) for s in (1.0, -2.0)]...; dims = 2)
    @test size(doy_linear_detrend(field, days)) == size(field)
    @test maximum(abs, doy_linear_detrend(field, days)) < 1e-9

    # Leap day folded onto 28 February: the 2020 key has two samples; independent fit.
    feb28 = findall(d -> key(d) == (2, 28), days)
    @test length(feb28) == 9                                  # 7 × 28 Feb + 2020 and 2024 29 Feb
    y = [sin(Float64(i)) for i in eachindex(days)]
    yy = Float64.(year.(days[feb28])); yy .-= mean(yy)
    xs = y[feb28]
    β = sum(yy .* xs) / sum(yy .^ 2)
    @test maximum(abs, doy_linear_detrend(y, days)[feb28] .- (xs .- mean(xs) .- β .* yy)) < 1e-12

    # NaN poisons only its own calendar day and series.
    Y = vcat(transpose(y), transpose(2 .* y))
    i = feb28[3]
    Y[1, i] = NaN
    R = doy_linear_detrend(Y, days)
    @test all(isnan, R[1, feb28])
    @test count(isnan, R) == length(feb28)
    @test_throws DimensionMismatch doy_linear_detrend(y, days[1:end-1])
end
