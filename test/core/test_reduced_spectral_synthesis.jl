#!/usr/bin/env julia
# Batched spectral → reduced-Gaussian synthesis (`ReducedSpectralSynthesis`):
#   1. single spherical harmonics reproduce P̃_n^m(sin φ)·2Re(f̂ e^{imλ}) at the
#      cell centres for m < nlon/2, including the top wavenumber of odd-length
#      rings; the Nyquist term m = nlon/2 of even rings enters with weight 1;
#   2. random fields agree with the per-ring `spectral_to_reduced_scalar!` to
#      round-off;
#   3. levels are synthesised independently.

using Test
using FastGaussQuadrature: gausslegendre
import AtmosTransport
const P = AtmosTransport.Preprocessing
const G = AtmosTransport.Grids

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
    Pcol = zeros(T + 1, T + 1)
    out = Float64[]
    for (j, φ) in enumerate(grid.lats)
        P.compute_legendre_column!(Pcol, T, sind(φ))
        w = m == 0 ? 1 : 2
        append!(out, [w * real(c * cis(m * deg2rad(λ))) * Pcol[n + 1, m + 1] for λ in grid.lons_by_ring[j]])
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
        resolved = cells(nl -> 2m < nl)                 # full weight
        nyquist = cells(nl -> 2m == nl)                 # half weight (single FFT bin)
        @test out[resolved, 2] ≈ expected[resolved] atol = 1e-12
        @test out[nyquist, 2] ≈ expected[nyquist] ./ 2 atol = 1e-12
        @test all(iszero, out[cells(nl -> 2m > nl), 2])
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
            P.spectral_to_reduced_scalar!(ref, spec[:, :, k], T, grid, cache; centered = true)
            @test out[:, k] ≈ ref rtol = 1e-12
        end
    end

    @test_throws DimensionMismatch P.synthesize_reduced!(zeros(nc, Nf + 1), zeros(ComplexF64, T + 1, T + 1, Nf + 1), s)
end
