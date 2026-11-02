#!/usr/bin/env julia
# Per-ring spectral synthesis (`spectral_to_ring!`, `spectral_to_grid!`) and the
# zonal truncation every synthesizer shares (`ring_max_wavenumber`, issue #13):
#   1. ring_max_wavenumber(T, N) is the largest m ≤ T with 2m < N, and equals T
#      on lat-lon targets (T = N ÷ 2 − 1);
#   2. a single harmonic f̂_n^m = c with m ≤ ring_max_wavenumber(T, N) is
#      reproduced as w_m P̃_n^m(sin φ) Re(c e^{i m λ_k}) (w_0 = 1, w_m = 2) at
#      every ring point λ_k = λ_0 + 2πk/N, on even and odd rings, for real and
#      complex c and for the face (λ_0 = 0) and centre (λ_0 = π/N) origins;
#      P̃_n^m comes from the explicit polynomial formula, not the recurrence;
#   3. a harmonic above the limit, such as the Nyquist term m = N/2 of an even
#      ring, contributes nothing;
#   4. both synthesizers give the same ring for a random spectrum.

using Test
import AtmosTransport
const P = AtmosTransport.Preprocessing
const G = AtmosTransport.Grids
using FastGaussQuadrature: gausslegendre

# Fully normalised P̃_n^m(μ) = √((2n+1)(n−m)!/(n+m)!) (1−μ²)^{m/2} dᵐP_n/dμᵐ
# (no Condon–Shortley phase, ECMWF's convention), with
# P_n(μ) = 2⁻ⁿ Σ_k (−1)^k C(n,k) C(2n−2k,n) μ^{n−2k}, evaluated in BigFloat.
function legendre_reference(n::Int, m::Int, μ::Real)
    x = BigFloat(μ)
    dmP = zero(BigFloat)
    for k in 0:(n ÷ 2)
        p = n - 2k
        p < m && continue
        c = big(-1)^k * binomial(big(n), k) * binomial(big(2n - 2k), n) / big(2)^n
        dmP += c * factorial(big(p)) / factorial(big(p - m)) * x^(p - m)
    end
    norm = sqrt((2n + 1) * factorial(big(n - m)) / factorial(big(n + m)))
    return Float64(norm * (1 - x^2)^(BigFloat(m) / 2) * dmP)
end

# The field of the single coefficient f̂_n^m = c at λ_k = λ0 + 2πk/N.
single_harmonic(n, m, c, φ, λ0, N) =
    [(m == 0 ? 1 : 2) * real(c * cis(m * (λ0 + 2π * k / N))) * legendre_reference(n, m, sind(φ))
     for k in 0:(N - 1)]

ring_cache(T, N) = P.ReducedSpectralThreadCache(
    zeros(T + 1, T + 1), Dict(N => zeros(ComplexF64, N)), Dict(N => zeros(N)),
    zeros(ComplexF64, T + 1, T + 1), zeros(ComplexF64, T + 1, T + 1))

function synthesize_ring(spec, T, φ, N, λ0)
    dest = zeros(N)
    P.spectral_to_ring!(dest, spec, T, φ, ring_cache(T, N); lon_shift_rad = λ0)
    return dest
end

function synthesize_grid(spec, T, φ, N, λ0)
    field = zeros(N, 1)
    P.spectral_to_grid!(field, spec, T, [φ], N, zeros(T + 1, T + 1), zeros(ComplexF64, N);
                        lon_shift_rad = λ0)
    return vec(field)
end

@testset "Spectral ring synthesis and zonal truncation" begin
    @testset "ring_max_wavenumber" begin
        for T in 0:24, N in 1:60
            @test P.ring_max_wavenumber(T, N) == maximum(m for m in 0:T if 2m < N)
        end
        @test P.ring_max_wavenumber(639, 1280) == 639   # N320 widest ring at T639
        @test P.ring_max_wavenumber(639, 18) == 8       # N320 polar ring
        @test P.ring_max_wavenumber(55, 20) == 9        # O24 polar ring (T = 112 ÷ 2 − 1)
        for N in (8, 72, 360, 720)                      # lat-lon targets keep every m ≤ T
            @test P.ring_max_wavenumber(N ÷ 2 - 1, N) == N ÷ 2 - 1
        end
    end

    T = 12
    lats = (-70.0, -35.0, 10.0, 50.0, 80.0)
    @testset "single harmonics, $name, N = $N" for (name, synthesize) in
            (("spectral_to_ring!", synthesize_ring), ("spectral_to_grid!", synthesize_grid)),
            N in (16, 15, 20)
        M = maximum(m for m in 0:T if 2m < N)           # top resolved wavenumber
        # m = N ÷ 2 is the Nyquist term of the even rings and the top resolved
        # wavenumber (N − 1)/2 of the odd ring
        for m in unique((0, 1, M - 1, M, N ÷ 2, N ÷ 2 + 1)), n in (m, m + 1),
                c in (1.0 + 0.0im, 0.7 + 0.4im), λ0 in (0.0, π / N), φ in lats
            n <= T || continue
            spec = zeros(ComplexF64, T + 1, T + 1)
            spec[n + 1, m + 1] = c
            got = synthesize(spec, T, φ, N, λ0)
            expected = 2m < N ? single_harmonic(n, m, c, φ, λ0, N) : zeros(N)
            @test got ≈ expected atol = 1e-12 * max(1, maximum(abs, expected))
        end
    end

    @testset "spectral_to_ring! and spectral_to_grid! agree" begin
        for N in (16, 15, 20)
            spec = zeros(ComplexF64, T + 1, T + 1)
            for m in 0:T, n in m:T
                spec[n + 1, m + 1] = m == 0 ? complex(randn()) : complex(randn(), randn())
            end
            for φ in lats, λ0 in (0.0, π / N)
                @test synthesize_ring(spec, T, φ, N, λ0) ≈ synthesize_grid(spec, T, φ, N, λ0) rtol = 1e-13
            end
        end
    end
end

# Small reduced Gaussian grid with even and odd ring lengths, symmetric about
# the equator, so the top wavenumber nlon ÷ 2 of the odd rings lies below T.
function small_reduced_grid()
    lats = asind.(gausslegendre(8)[1])
    nlon = [9, 15, 20, 25, 25, 20, 15, 9]
    mesh = G.ReducedGaussianMesh(lats, nlon; FT = Float64, radius = 6.371e6)
    lons = [G.ring_longitudes(mesh, j) for j in eachindex(nlon)]
    return P.ReducedGaussianTargetGeometry{Float64, typeof(mesh)}(mesh, "", 4, nlon, lats, lons)
end

# f(λ, φ) on every cell centre for the single coefficient f̂_n^m = c.
function single_mode(grid, T, n, m, c)
    out = Float64[]
    for (j, φ) in enumerate(grid.lats)
        append!(out, [(m == 0 ? 1 : 2) * real(c * cis(m * deg2rad(λ))) *
                      legendre_reference(n, m, sind(φ)) for λ in grid.lons_by_ring[j]])
    end
    return out
end

@testset "Reduced spectral synthesis" begin
    grid = small_reduced_grid()
    T, Nf = 10, 3
    s = P.ReducedSpectralSynthesis(grid, T, Nf)
    nc = G.ncells(grid.mesh)
    out = zeros(nc, Nf)

    # (7, 4) and (10, 7) are the top wavenumbers of the 9- and 15-point rings;
    # (10, 10) is the Nyquist term of the 20-point rings
    @testset "single harmonics (n, m) = $nm" for nm in ((0, 0), (3, 0), (5, 2), (7, 4), (10, 7), (9, 9), (10, 6), (10, 10))
        n, m = nm
        spec = zeros(ComplexF64, T + 1, T + 1, Nf)
        spec[n + 1, m + 1, 2] = 0.7 - 0.3im
        P.synthesize_reduced!(out, spec, s)
        expected = single_mode(grid, T, n, m, 0.7 - 0.3im)
        cells(f) = vcat([fill(f(nl), nl) for nl in grid.nlon_per_ring]...)
        resolved = cells(nl -> 2m < nl)                 # full weight (all m ≤ T here)
        @test out[resolved, 2] ≈ expected[resolved] atol = 1e-12
        @test all(iszero, out[.!resolved, 2])           # dropped, incl. the Nyquist term 2m = nl
        @test all(iszero, out[:, 1]) && all(iszero, out[:, 3])
    end

    @testset "agrees with the per-ring path" begin
        spec = zeros(ComplexF64, T + 1, T + 1, Nf)
        for k in 1:Nf, m in 0:T, n in m:T
            spec[n + 1, m + 1, k] = m == 0 ? complex(randn()) : complex(randn(), randn())
        end
        P.synthesize_reduced!(out, spec, s)
        cache = P.ReducedSpectralThreadCache(
            zeros(T + 1, T + 1),
            Dict(n => zeros(ComplexF64, n) for n in unique(grid.nlon_per_ring)),
            Dict(n => zeros(n) for n in unique(grid.nlon_per_ring)),
            zeros(ComplexF64, T + 1, T + 1), zeros(ComplexF64, T + 1, T + 1))
        for k in 1:Nf
            ref = zeros(nc)
            P.spectral_to_reduced_scalar!(ref, spec[:, :, k], T, grid, cache)
            @test out[:, k] ≈ ref rtol = 1e-12
        end
    end

    @testset "level packing and buffer reuse are bitwise stable" begin
        # An uneven level count exercises independent pack/scatter iterations
        # with multiple threads; reversing a view also checks strided input.
        nf = 19
        synthesis = P.ReducedSpectralSynthesis(grid, T, nf)
        spec = zeros(ComplexF64, T + 1, T + 1, nf)
        for k in 1:nf, m in 0:T, n in m:T
            spec[n + 1, m + 1, k] = complex(sin(n + k), cos(m + k))
        end
        original = copy(spec)
        expected = zeros(nc, nf)
        P.synthesize_reduced!(expected, spec, synthesis)
        actual = fill(NaN, nc, nf)
        for _ in 1:3
            P.synthesize_reduced!(actual, view(spec, :, :, nf:-1:1), synthesis)
            @test reinterpret(UInt64, vec(actual)) ==
                  reinterpret(UInt64, vec(expected[:, nf:-1:1]))
            P.synthesize_reduced!(actual, spec, synthesis)
            @test reinterpret(UInt64, vec(actual)) == reinterpret(UInt64, vec(expected))
        end
        @test reinterpret(UInt64, vec(spec)) == reinterpret(UInt64, vec(original))
    end

    @test_throws DimensionMismatch P.synthesize_reduced!(zeros(nc, Nf + 1), zeros(ComplexF64, T + 1, T + 1, Nf + 1), s)
end
