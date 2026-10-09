#!/usr/bin/env julia

using Test

import AtmosTransport
const AT = AtmosTransport

@testset "physical and state constructors reject ambiguous inputs" begin
    @test_throws ArgumentError AT.Parameters.PlanetParameters(radius=0.0)
    @test_throws ArgumentError AT.Parameters.PlanetParameters(gravity=Inf)
    @test AT.Parameters.PlanetParameters(6.0f6, 9.8f0, 1.0f5) isa
          AT.Parameters.PlanetParameters{Float32}
    @test_throws ArgumentError AT.HybridSigmaPressure([0.0], [0.0])
    @test_throws ArgumentError AT.HybridSigmaPressure([0.0, NaN], [0.0, 1.0])

    ps = zeros(Float32, 3, 2)
    q = zeros(Float64, 3, 2, 4)
    met = @test_deprecated AT.State.MetState(ps, q)
    @test met.ps === ps
    @test met.q === q
    @test_deprecated @test_throws DimensionMismatch AT.State.MetState(ps, zeros(3, 3, 4))

    # Deprecated, no caller in the package (removed in the next minor release).
    am, bm, cm = zeros(3, 2, 2), zeros(2, 3, 2), ones(2, 2, 3)
    vc2 = AT.HybridSigmaPressure([0.0, 0.0, 0.0], [0.0, 0.5, 1.0])
    @test_deprecated AT.MetDrivers.diagnose_cm_from_continuity_vc!(cm, am, bm, vc2, 2, 2, 2)
    @test all(iszero, cm)
    cm .= 1
    @test_deprecated AT.MetDrivers.diagnose_cm_from_continuity_ka!(cm, am, bm, [0.5, 0.5], 2, 2, 2)
    @test all(iszero, cm)

    mesh = AT.LatLonMesh(Nx=2, Ny=2, radius=6.0e6)
    vc = AT.HybridSigmaPressure([0.0, 100.0], [0.0, 1.0])
    @test_throws ArgumentError AT.AtmosGrid(mesh, vc, AT.CPU())
    mesh32 = AT.LatLonMesh(FT=Float32, Nx=2, Ny=2, radius=6.371f6)
    @test_throws ArgumentError AT.AtmosGrid(
        mesh32, AT.HybridSigmaPressure(Float32[0, 100], Float32[0, 1]), AT.CPU();
        radius=6.372f6)
    cs_grid_1 = AT.AtmosGrid(AT.CubedSphereMesh(Nc=2, Hp=1), vc, AT.CPU())
    cs_grid_2 = AT.AtmosGrid(AT.CubedSphereMesh(Nc=2, Hp=2), vc, AT.CPU())
    @test_throws ArgumentError AT.Models._check_grid_compatibility(cs_grid_1, cs_grid_2)
end

@testset "chemistry and inversion inputs are finite and unambiguous" begin
    @test_throws ArgumentError AT.ExponentialDecay()
    @test_throws ArgumentError AT.ExponentialDecay(; CO2=0.0)
    @test_throws ArgumentError AT.ExponentialDecay(; CO2=Inf)

    @test_throws ArgumentError AT.CSSurfaceFluxWindow(:duplicate, [1, 1])
    @test_throws ArgumentError AT.CSSurfaceFluxWindow(:nan, [1, 2]; weights=[1.0, NaN])
    @test_throws ArgumentError AT.CSSurfaceFluxWindow(
        :overflow, [1, 2]; weights=fill(floatmax(Float64), 2), normalize=true)
    panels = ntuple(_ -> zeros(2, 2), 6)
    window = AT.CSSurfaceFluxWindow(:valid, 1)
    @test_throws ArgumentError AT.CSSurfaceFluxControl(window, panels; sigma=Inf)
    bad_sigma = ntuple(_ -> fill(1.0, 2, 2), 6)
    bad_sigma[3][1, 1] = 0.0
    @test_throws ArgumentError AT.CSSurfaceFluxControl(window, panels; sigma=bad_sigma)
end

@testset "configuration booleans are not numeric coercions" begin
    @test_throws ArgumentError AT.Architectures.architecture_from_config(
        Dict{String,Any}("use_gpu" => 1))
    @test_throws ArgumentError AT.Output.runtime_output_spec(
        Dict{String,Any}("enabled" => 1), Float64)
end
