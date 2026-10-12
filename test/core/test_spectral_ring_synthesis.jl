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
