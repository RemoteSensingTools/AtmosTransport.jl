"""
Latitude-band means, skill statistics and column profile tools of
`AtmosTransport.Analysis` (`src/Analysis/latitude_bands.jl`, `skill.jl`,
`column_profiles.jl`).

Asserts:
- `EqualAreaBands(n)` edges are `(k / n) * 2 - 1`; `band_index` is the number
  of edges ≤ sin φ (IEEE comparison, so -0.0 is in the band of 0.0, as in
  NumPy), also at and next to every edge, without allocating; band indices
  equal NumPy
  `clip(digitize(sin(deg2rad(lat)), edges) - 1, 0, n - 1) + 1` for pinned
  latitudes (poles, edges, the equator, ±30°); the bands have equal area to
  the discretisation of a fine latitude grid;
- a band set with a gap (`band_index` 0) leaves those cells out of every
  mean; a band set returning an index outside `0:nbands` is rejected;
- `band_means`: NaN cells are left out of the numerator and the denominator,
  a band of NaN cells is NaN, Inf propagates; the result equals a loop over
  cells in linear order bit for bit (the NumPy `bincount` order) on a C8 cubed
  sphere; a uniform field gives the constant; trailing dimensions are looped;
  equal-area rows average to the band midpoint in sin φ; a reused
  `BandAssignment` equals the four-argument call; shape, area and latitude
  errors are raised;
- `weighted_fit_stats` (model, reference, weights) equals the Python `wstats`
  of `catrine_compare_vs_geoschem.py` on a pinned case with NaN pairs, and
  gives r² = 1, slope and bias of an exact linear relation; weights are
  scale-invariant and must be non-negative;
- `column_band_weights` gives the hand-computed layer masses of four bands,
  equals the Python `band_sums` weights bit for bit, sums to (b − a) M, and
  rejects bad bands and masses;
- `logp_interpolate` equals the Python zonal-mean interpolation inside the
  column, reproduces a profile linear in ln p, returns `outside` beyond the
  top and lowest mid levels, includes both end levels, and rejects
  non-monotone or non-positive pressures.
"""

using Test
using Random
using Statistics
using AtmosTransport
using AtmosTransport: CubedSphereMesh
using AtmosTransport.Grids: panel_cell_center_lonlat
using AtmosTransport.Analysis: AbstractLatitudeBands, EqualAreaBands, BandAssignment, nbands, band_edges,
                               band_index, band_means, band_means!, weighted_fit_stats,
                               column_band_weights, column_band_weights!, logp_interpolate,
                               logp_interpolate!

# Cell-centre latitudes and areas of a cubed sphere in NCDatasets order (Xdim, Ydim, nf).
function cs_lat_area(Nc)
    mesh = CubedSphereMesh(; Nc)
    lat = Array{Float64}(undef, Nc, Nc, 6)
    area = Array{Float64}(undef, Nc, Nc, 6)
    for p in 1:6
        _, φ = panel_cell_center_lonlat(mesh, p)
        lat[:, :, p] .= φ
        area[:, :, p] .= mesh.cell_areas
    end
    return lat, area
end

# The Python reduction: band = clip(digitize(sin lat, edges) - 1), then
# bincount(band, weights = x * area) / bincount(band, weights = area), in cell order.
function loop_band_means(x, lat, area, n)
    edges = [k / n * 2 - 1 for k in 0:n]
    num, den = zeros(n), zeros(n)
    for c in eachindex(x, lat, area)
        b = clamp(count(<=(sin(deg2rad(lat[c]))), edges), 1, n)
        num[b] += x[c] * area[c]
        den[b] += area[c]
    end
    return num ./ den
end

@testset "equal-area bands" begin
    b = EqualAreaBands(40)
    @test b isa AbstractLatitudeBands
    @test nbands(b) == 40
    @test band_edges(b) == [k / 40 * 2 - 1 for k in 0:40]
    @test band_edges(b)[[1, 21, 41]] == [-1.0, 0.0, 1.0]
    # Pinned from NumPy digitize (1-based).
    lats = [-90.0, -89.9, -48.59037789072914, -30.0, -1e-9, 0.0, 1e-9, 30.0, 45.0, 89.99, 90.0]
    @test band_index.(Ref(b), lats) == [1, 1, 6, 11, 20, 21, 21, 30, 35, 40, 40]
    # The band is the number of edges ≤ sin φ (IEEE comparison, as NumPy
    # digitize: -0.0 equals 0.0), also at and next to every edge.
    for n in (1, 2, 7, 40, 60, 360)
        bn = EqualAreaBands(n)
        e = band_edges(bn)
        lat_test = vcat(collect(range(-90, 90; length = 2001)), asind.(e), nextfloat.(asind.(e)), prevfloat.(asind.(e)))
        lat_test = clamp.(lat_test, -90.0, 90.0)
        @test all(band_index(bn, φ) == clamp(count(<=(sin(deg2rad(φ))), e), 1, n) for φ in lat_test)
    end
    @test band_index(b, -0.0) == band_index(b, 0.0) == band_index(b, prevfloat(0.0)) == 21
    band_index(b, 12.3)
    @test (@allocated band_index(b, 12.3)) == 0
    @test_throws ArgumentError band_index(b, 90.5)
    @test_throws ArgumentError band_index(b, NaN)
    @test_throws ArgumentError EqualAreaBands(0)
    # Equal areas: 0.01° rows, area 2πR²(sin φ_n − sin φ_s), assigned by their centres.
    faces = collect(-90.0:0.01:90.0)
    centres = (faces[1:end-1] .+ faces[2:end]) ./ 2
    row_area = 2π .* (sind.(faces[2:end]) .- sind.(faces[1:end-1]))
    a = BandAssignment(EqualAreaBands(12), centres, row_area)
    band_area = [sum(row_area[a.band .== k]) for k in 1:12]
    @test maximum(abs, band_area ./ (4π / 12) .- 1) < 2e-3
end

# A band set with a gap: two bands poleward of ±10°, the tropics in no band.
struct ExtratropicalBands <: AbstractLatitudeBands end
AtmosTransport.Analysis.nbands(::ExtratropicalBands) = 2
AtmosTransport.Analysis.band_edges(::ExtratropicalBands) = [-90.0, -10.0, 10.0, 90.0]
AtmosTransport.Analysis.band_index(::ExtratropicalBands, lat::Real) = lat < -10 ? 1 : lat >= 10 ? 2 : 0
struct BrokenBands <: AbstractLatitudeBands end
AtmosTransport.Analysis.nbands(::BrokenBands) = 2
AtmosTransport.Analysis.band_index(::BrokenBands, lat::Real) = 3

@testset "band sets with gaps" begin
    lat = [-45.0, -5.0, 5.0, 45.0]
    a = BandAssignment(ExtratropicalBands(), lat, ones(4))
    @test a.band == [1, 0, 0, 2]
    @test band_means([1.0, 100.0, 100.0, 2.0], a) == [1.0, 2.0]   # cells in no band enter no mean
    @test_throws ArgumentError BandAssignment(BrokenBands(), lat, ones(4))
    @test_throws DimensionMismatch BandAssignment(EqualAreaBands(2), (4,), [1, 1, 2], ones(3))
end

@testset "band means" begin
    bands = EqualAreaBands(2)                        # south and north hemispheres
    lat = [-45.0, -10.0, 10.0, 45.0, 80.0]
    area = [1.0, 2.0, 3.0, 4.0, 5.0]
    @test band_means([1.0, 4.0, 2.0, 3.0, 6.0], lat, area, bands) ≈ [(1 + 8) / 3, (6 + 12 + 30) / 12]
    # NaN out of numerator and denominator; a band of NaNs is NaN; Inf propagates.
    @test band_means([NaN, 4.0, 2.0, NaN, 6.0], lat, area, bands) ≈ [4.0, (6 + 30) / 8]
    m = band_means([NaN, NaN, 2.0, 3.0, 6.0], lat, area, bands)
    @test isnan(m[1]) && m[2] ≈ (6 + 12 + 30) / 12
    @test band_means([1.0, Inf, 2.0, 3.0, 6.0], lat, area, bands)[1] == Inf
    @test band_means(Float32[1, 4, 2, 3, 6], lat, area, bands) ≈ [3.0, 4.0]

    # Linear-order sums on a C8 cubed sphere: bitwise equal to the loop reference.
    lat8, area8 = cs_lat_area(8)
    x = 400 .+ randn(Xoshiro(3), size(lat8))
    a = BandAssignment(EqualAreaBands(10), lat8, area8)
    @test band_means(x, a) == loop_band_means(x, lat8, area8, 10)
    @test band_means(x, lat8, area8, EqualAreaBands(10)) == band_means(x, a)
    @test maximum(abs, band_means(fill(412.5, size(lat8)), a) .- 412.5) < 1e-12
    @test sum(a.area) ≈ 4π * 6.371e6^2 rtol = 1e-12

    # Trailing dimensions are looped; the result has size (nbands, trailing...).
    xs = cat(x, 2 .* x, x .- 1; dims = 4)
    xs = cat(xs, -xs; dims = 5)
    m = band_means(xs, a)
    @test size(m) == (10, 3, 2)
    @test m[:, 2, 1] == band_means(2 .* x, a)
    @test m[:, 3, 2] == band_means(-(x .- 1), a)
    out = zeros(10, 3, 2)
    @test band_means!(out, xs, a) === out && out == m

    # Equal-area rows (equal Δ sin φ, 5 per band): the mean of sin φ is the band midpoint.
    n, k = 8, 5
    s = [-1 + (2i - 1) / (n * k) for i in 1:n*k]
    rows = BandAssignment(EqualAreaBands(n), asind.(s), fill(2π * 2 / (n * k), n * k))
    edges = band_edges(EqualAreaBands(n))
    @test maximum(abs, band_means(s, rows) .- (edges[1:end-1] .+ edges[2:end]) ./ 2) < 1e-14

    @test_throws DimensionMismatch band_means(x[:, :, 1:5], a)
    @test_throws DimensionMismatch band_means!(zeros(9), x, a)
    @test_throws DimensionMismatch BandAssignment(EqualAreaBands(2), lat, area[1:4])
    @test_throws ArgumentError BandAssignment(EqualAreaBands(2), lat, [1.0, -2.0, 3.0, 4.0, 5.0])
    @test_throws ArgumentError BandAssignment(EqualAreaBands(2), [-45.0, 91.0, 0.0, 0.0, 0.0], area)
end

@testset "weighted fit statistics" begin
    # Pinned from catrine_compare_vs_geoschem.wstats(reference, model, w).
    reference = [1.0, 2.0, 3.5, NaN, 0.5, 2.2]
    model = [1.1, 2.3, 3.0, 1.0, 0.4, NaN]
    w = [2.0, 1.0, 0.5, 3.0, 1.5, 1.0]
    s = weighted_fit_stats(model, reference, w)
    @test s.r2 ≈ 0.9388330724160935 atol = 1e-14
    @test s.slope ≈ 0.930864197530864 atol = 1e-14
    @test s.bias ≈ 0.02000000000000024 atol = 1e-14
    @test s.rmse ≈ 0.22360679774997894 atol = 1e-14
    @test s.ref_mean ≈ 1.3 atol = 1e-14
    @test s.ref_std ≈ 0.9000000000000001 atol = 1e-14
    @test weighted_fit_stats(model, reference, 7.5 .* w).r2 ≈ s.r2 rtol = 1e-14

    x = randn(Xoshiro(5), 200)
    area = rand(Xoshiro(6), 200) .+ 0.5
    same = weighted_fit_stats(x, x, area)
    @test same.r2 ≈ 1 && same.slope ≈ 1 && same.bias == 0 && same.rmse == 0
    lin = weighted_fit_stats(2 .* x .+ 1, x, area)
    wn = area ./ sum(area)
    @test lin.r2 ≈ 1 && lin.slope ≈ 2
    @test lin.bias ≈ sum(wn .* x) + 1
    @test lin.ref_std ≈ sqrt(sum(wn .* (x .- sum(wn .* x)) .^ 2))
    y = x .+ 0.3 .* randn(Xoshiro(7), 200)
    @test weighted_fit_stats(y, x, ones(200)).r2 ≈ cor(x, y)^2
    flat = weighted_fit_stats(x, zeros(200), area)            # s_xx = 0 exactly
    @test isnan(flat.slope) && isnan(flat.r2) && flat.ref_std == 0
    none = weighted_fit_stats([NaN, 1.0], [1.0, NaN], [1.0, 1.0])
    @test all(isnan, values(none))
    @test_throws ArgumentError weighted_fit_stats([1.0, 2.0], [1.0, 2.5], [1.0, -1.0])
    @test_throws DimensionMismatch weighted_fit_stats([1.0, 2.0], [1.0, 2.5], [1.0])
end

@testset "column-mass band weights" begin
    # Layers top first; M = 10, so the fractions from the surface are
    # layer 4: 0-0.4, layer 3: 0.4-0.7, layer 2: 0.7-0.9, layer 1: 0.9-1.
    m = [1.0, 2.0, 3.0, 4.0]
    bands = (sfc = (0.0, 0.1), low = (0.1, 0.6), mid = (0.6, 0.9), top = (0.9, 1.0))
    w = column_band_weights(m, bands)
    @test size(w) == (4, 4)
    @test w[:, 1] ≈ [0, 0, 0, 1]
    @test w[:, 2] ≈ [0, 0, 2, 3]
    @test w[:, 3] ≈ [0, 2, 1, 0]
    @test w[:, 4] ≈ [1, 0, 0, 0]
    @test column_band_weights(m, [(0, 1)])[:, 1] ≈ m

    # Pinned from catrine_compare_vs_geoschem.band_sums: the same bits.
    m7 = [0.31, 1.7, 2.9, 4.4, 6.1, 7.3, 8.05]
    q7 = [400.1, 401.0, 402.5, 405.0, 410.2, 412.0, 415.5]
    python_bands = (column = (0.0, 1.0), sfc_910hPa = (0.0, 0.1), var"910_400hPa" = (0.1, 0.6),
                    var"400_100hPa" = (0.6, 0.9), above_100hPa = (0.9, 1.0))
    python_w = [[0.30999999999999733, 1.7000000000000002, 2.9000000000000004, 4.399999999999997,
                 6.099999999999999, 7.300000000000001, 8.05],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0760000000000005],
                [0.0, 0.0, 0.0, 0.0, 3.1059999999999963, 7.300000000000001, 4.974000000000001],
                [0.0, 0.0, 1.8340000000000019, 4.399999999999997, 2.994000000000003, 0.0, 0.0],
                [0.30999999999999733, 1.7000000000000002, 1.0659999999999985, 0.0, 0.0, 0.0, 0.0]]
    w7 = column_band_weights(m7, python_bands)
    @test [w7[:, j] for j in 1:5] == python_w
    @test sum(w7[:, 2] .* q7) ≈ 1278.0780000000002 rtol = 1e-15
    for (j, (a, b)) in enumerate(python_bands)
        @test sum(w7[:, j]) ≈ (b - a) * sum(m7) rtol = 1e-14
    end
    buf = zeros(7, 5)
    @test column_band_weights!(buf, m7, python_bands) === buf && buf == w7

    @test_throws ArgumentError column_band_weights(m, [(0.5, 0.5)])
    @test_throws ArgumentError column_band_weights(m, [(-0.1, 0.5)])
    @test_throws ArgumentError column_band_weights(m, [(0.0, 0.5, 1.0)])
    @test_throws ArgumentError column_band_weights(m, ())
    @test_throws ArgumentError column_band_weights([1.0, -1.0], [(0, 1)])
    @test_throws ArgumentError column_band_weights([0.0, 0.0], [(0, 1)])
    @test_throws DimensionMismatch column_band_weights!(zeros(3, 4), m, bands)
end

@testset "log-pressure interpolation" begin
    p_mid = [150.0, 2000.0, 8000.0, 20000.0, 45000.0, 70000.0, 85000.0, 95000.0]
    q = [1.0, 1.5, 2.0, 2.4, 2.9, 3.3, 3.6, 3.9]
    # Pinned from the zonal-mean interpolation of catrine_compare_vs_geoschem.py.
    targets = [100.0, 1000.0, 10000.0, 50000.0, 90000.0, 95000.0, 99000.0]
    python = [NaN, 1.366201604644005, 2.097411681053588, 2.9953849755269264, 3.7541688131067223, 3.9, NaN]
    v = logp_interpolate(targets, p_mid, q)
    @test all(isnan(a) ? isnan(b) : abs(a - b) <= 1e-14 for (a, b) in zip(v, python))
    # The top mid level is inside the column (the Python script returned NaN there).
    @test logp_interpolate(150.0, p_mid, q) == 1.0
    @test logp_interpolate(95000.0, p_mid, q) == 3.9
    @test logp_interpolate(100.0, p_mid, q; outside = -1.0) == -1.0
    @test logp_interpolate(1000.0, p_mid, q) == v[2]

    # A profile linear in ln p is reproduced exactly at any pressure inside the column.
    lin(p) = 2.0 - 0.3 * log(p)
    tp = exp.(range(log(151.0), log(94999.0); length = 50))
    @test maximum(abs, logp_interpolate(tp, p_mid, lin.(p_mid)) .- lin.(tp)) < 1e-13
    out = zeros(length(tp))
    @test logp_interpolate!(out, tp, p_mid, lin.(p_mid)) === out
    @test logp_interpolate(5.0, [5.0], [7.0]) == 7.0
    @test isnan(logp_interpolate(6.0, [5.0], [7.0]))

    @test_throws ArgumentError logp_interpolate(1000.0, [150.0, 150.0, 2000.0], [1.0, 2.0, 3.0])
    @test_throws ArgumentError logp_interpolate(1000.0, [2000.0, 150.0], [1.0, 2.0])
    @test_throws ArgumentError logp_interpolate(1000.0, [0.0, 150.0], [1.0, 2.0])
    @test_throws ArgumentError logp_interpolate(-5.0, p_mid, q)
    @test_throws DimensionMismatch logp_interpolate(1000.0, p_mid, q[1:end-1])
end
