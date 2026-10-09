#!/usr/bin/env julia
# Float32 conservation contracts (docs/src/theory/float32_conservation.md):
# error-free and compensated sums, the column-ledger guard, geometry evaluated
# in Float64 and rounded once, the decay decrement, exact global extents for
# regridding sources, and the emission profile deposit. The ledgers on
# production matrices are checked in test_tm5_hessenberg.jl and
# helpers/conservative_dkg.jl.

using Test, Random, LinearAlgebra
using AtmosTransport
using AtmosTransport.Architectures: _two_sum, _neumaier_add, _compensated_total
using AtmosTransport.Grids: CubedSphereMesh, GEOSNativePanelConvention, LatLonMesh
using AtmosTransport.Operators.SurfaceFlux: SurfaceFluxSource, SurfaceFluxOperator,
                                            ProfileDeposit, apply_surface_flux!
const Chem = AtmosTransport.Operators.Chemistry
const TM5 = AtmosTransport.Operators.Convection
const RG = AtmosTransport.Regridding
const ICIO = AtmosTransport.Models.InitialConditionIO

@testset "Error-free sums" begin
    rng = MersenneTwister(2026)
    draw() = Float32(randn(rng) * 10.0^rand(rng, -3:3))
    @test all(1:10_000) do _
        a, b = draw(), draw()
        s, e = _two_sum(a, b)
        big(s) + big(e) == big(a) + big(b)
    end

    # 0.01 is below half an ulp of 1e6 in Float32: a running sum drops all of
    # it, while the compensated sum stays within a fraction of that ulp.
    x = [1f6; fill(0.01f0, 10_000)]
    s, c = foldl((sc, v) -> _neumaier_add(sc..., v), x; init = (0f0, 0f0))
    exact = sum(big, x)
    @test abs(foldl(+, x) - exact) > 50
    @test abs(big(s) + big(c) - exact) < eps(1f6) / 8
end

@testset "Compensated Float64 totals" begin
    rng = MersenneTwister(11)
    panels = ntuple(_ -> 4f-4 .* (1 .+ rand(rng, Float32, 9, 9, 5)), 6)
    exact = sum(p -> sum(big, p), panels)
    @test _compensated_total(panels[1]) isa Float64
    @test abs(_compensated_total(panels) - exact) <= eps(Float64) * exact
    interior = view(panels[1], 2:8, 2:8, :)
    @test abs(_compensated_total(interior) - sum(big, interior)) <= eps(Float64) * sum(big, interior)
end

@testset "Column ledger returns rounding, not leaks" begin
    n, rng = 40, MersenneTwister(7)
    for FT in (Float32, Float64), leak in (zero(FT), FT(1e-3))
        # I + B, where B exchanges mass between neighbours (columns sum to 1);
        # `leak` makes column n sum to 1 + leak.
        A = Matrix{FT}(I, n, n)
        for (k, a) in enumerate(FT(0.3) .* rand(rng, FT, n - 1))
            A[k, k] += a; A[k + 1, k] -= a; A[k + 1, k + 1] += a; A[k, k + 1] -= a
        end
        A[n, n] += leak
        pivots = zeros(Int, n)
        TM5._tm5_lu!(A, pivots, n)
        q = FT(4) .* (1 .+ rand(rng, FT, n, 1))
        q0 = copy(q)
        TM5._tm5_conserving_solve_tracer!(q, A, pivots, n, 1, 1, false)
        gap = sum(big, q) - sum(big, q0)
        if iszero(leak)
            @test abs(gap) <= eps(FT) * maximum(q)
        else
            @test abs(gap) > 1e-5 * sum(big, q0)     # the leak stays visible
        end
    end
end

@testset "Cubed-sphere geometry is Float64, rounded once" begin
    m32, m64 = (CubedSphereMesh(; FT, Nc = 48, convention = GEOSNativePanelConvention())
                for FT in (Float32, Float64))
    @test m32.cell_areas == Float32.(m64.cell_areas)
    @test m32.Δx == Float32.(m64.Δx) && m32.Δy == Float32.(m64.Δy)
    @test RG.cubed_sphere_face_corners(m32) == RG.cubed_sphere_face_corners(m64)
    @test RG.GOCore.best_manifold(m32) == RG.GOCore.best_manifold(m64)
end

@testset "Decay decrement" begin
    λ = Float32(log(2) / (3.8235 * 86400))                    # Rn-222
    for dt in (400.0, 450.0, 900.0)
        d = Chem.decay_decrement(Float32, λ, dt)
        @test d isa Float32
        @test d == Float32(expm1(-Float64(λ) * dt))
    end
end

@testset "Global source grids snap to exact extents" begin
    # GridFED stores its 0.1° centres in Float32.
    lon = Float64.(Float32.(range(-179.95, 179.95; length = 3600)))
    lat = Float64.(Float32.(range(-89.95, 89.95; length = 1800)))
    mesh = ICIO._build_source_latlon_mesh(lon, lat)
    @test mesh isa LatLonMesh{Float64}
    @test (first(mesh.φᶠ), last(mesh.φᶠ)) == (-90.0, 90.0)
    @test last(mesh.λᶠ) - first(mesh.λᶠ) ≈ 360 atol = 1e-9
    @test RG._latlon_full_sphere(mesh)

    regional = ICIO._build_source_latlon_mesh(collect(0.5:1.0:29.5), collect(-9.5:1.0:9.5))
    @test (first(regional.φᶠ), last(regional.φᶠ)) == (-10.0, 10.0)
    @test !RG._latlon_full_sphere(regional)
end

@testset "Profile deposit conserves column mass in Float32" begin
    FT, Nc, Nz, Hp = Float32, 2, 40, 1
    # 0.45 kg profile shares into 2e6 kg cells (400 ppm of 5e9 kg): ~4 ulps each.
    q = ntuple(_ -> fill(FT(400e-6 * 5e9), Nc + 2Hp, Nc + 2Hp, Nz), 6)
    q0 = map(copy, q)
    src = SurfaceFluxSource(:x, ntuple(_ -> fill(FT(0.01), Nc, Nc), 6))   # kg s⁻¹
    op = SurfaceFluxOperator(src)
    profile = zeros(FT, Nc, Nc, Nz)
    profile[:, :, Nz-4:Nz] .= reshape(FT[0.1, 0.15, 0.2, 0.25, 0.3], 1, 1, 5)
    deposit = ntuple(_ -> ProfileDeposit(profile), 6)
    dt, steps = 450.0, 240
    for _ in 1:steps
        apply_surface_flux!(q, op, nothing, dt, nothing, nothing; tracer_names = (:x,),
                            halo_width = Hp, deposit)
    end
    # Kahan: mass added = steps·x + final compensation (mass deposited ahead).
    emitted = steps * 6Nc^2 * big(FT(0.01) * FT(dt))
    added = sum(p -> sum(big, p), q) - sum(p -> sum(big, p), q0)
    ahead = sum(p -> sum(big, p), src.compensation)
    @test abs(added - (emitted + ahead)) < 1e-7 * emitted
end
