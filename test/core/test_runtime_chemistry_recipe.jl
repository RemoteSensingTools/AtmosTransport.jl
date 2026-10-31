#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# build_runtime_chemistry + recipe.chemistry wiring through DrivenSimulation.
#
# Covers:
#   1. Builder dispatch on `[chemistry].kind`:
#      - missing / "none" → NoChemistry
#      - "decay" with empty half-lives → NoChemistry
#      - "decay" with `half_lives_seconds.rn222 = 330350.4` →
#        ExponentialDecay with first-order rate `ln(2)/T`
#      - unknown kind → ArgumentError
#   2. End-to-end decay correctness: a one-tracer state with VMR ≡ 1.0
#      stepped through one day under `ExponentialDecay(Rn222=330350.4)`
#      yields VMR ≈ exp(-ln(2) · 86400 / 330350.4) ≈ 0.834.
# ---------------------------------------------------------------------------

using Test

import AtmosTransport
const AT = AtmosTransport
using .AT.Models: build_runtime_chemistry,
    chemistry_spec, NoChemistrySpec, DecayChemistrySpec
using .AT.State: field_value

const RN222_HALF_LIFE_SECONDS = 330_350.4


@testset "only constant decay rates use the symmetric split" begin
    constant = AT.ExponentialDecay(Float64; rn222 = RN222_HALF_LIFE_SECONDS)
    @test AT.Operators.StepPlans._symmetric_decay_split(constant)

    rate = AT.State.StepwiseField([1e-6, 2e-6], [0.0, 10.0, 20.0])
    rates = (rate,)
    varying = AT.Operators.Chemistry.ExponentialDecay{
        Float64, 1, typeof(rates)}(rates, (:rn222,))
    @test !AT.Operators.StepPlans._symmetric_decay_split(varying)
end

@testset "Rn-222 symmetric window split has the analytic steady burden" begin
    FT = Float64
    Nx, Ny, Nz = 2, 2, 1
    T = FT(3600)
    E = FT(3.0)
    λ = log(FT(2)) / FT(RN222_HALF_LIFE_SECONDS)
    x = λ * T

    # For half-decay → window emissions → half-decay,
    # B[n+1] = exp(-x)B[n] + E*T*exp(-x/2). Its exact discrete fixed point is
    # E*T/(2sinh(x/2)); E/λ is the continuous-source steady burden.
    Bstar = E * T / (2sinh(x / 2))
    Bcontinuous = E / λ
    Bold = E * T / expm1(x)

    mesh = AT.LatLonMesh(; FT, Nx, Ny)
    vertical = AT.HybridSigmaPressure(FT[0, 0], FT[0, 1])
    grid = AT.AtmosGrid(mesh, vertical, AT.CPU(); FT)
    air = ones(FT, Nx, Ny, Nz)
    state = AT.CellState(AT.DryBasis, air; rn222 = fill(Bstar, Nx, Ny, Nz))
    fluxes = AT.allocate_face_fluxes(mesh, Nz; FT, basis = AT.DryBasis)
    emissions = AT.SurfaceFluxOperator(AT.SurfaceFluxSource(:rn222, fill(E, Nx, Ny)))
    decay = AT.ExponentialDecay(FT; rn222 = RN222_HALF_LIFE_SECONDS)
    model = AT.TransportModel(state, fluxes, grid, AT.UpwindScheme();
                              emissions, chemistry = decay)

    AT.step!(model, T)

    @test all(isapprox.(model.state.tracers.rn222, Bstar; rtol = 2e-15))
    @test Bstar / Bcontinuous - 1 ≈ x / (2sinh(x / 2)) - 1 rtol = 1e-10
    @test Bstar / Bcontinuous - 1 ≈ -2.3773563538886933e-6 atol = 1e-14
    @test Bold / Bcontinuous - 1 ≈ x / expm1(x) - 1 rtol = 1e-12
    @test Bold / Bcontinuous - 1 ≈ -0.0037720378199672666 atol = 1e-14
end

@testset "ChemistrySpec parse" begin
    # Multi-tracer decay (Rn222 + Kr85)
    op = build_runtime_chemistry(
        Dict("chemistry" => Dict("kind" => "decay",
                                  "half_lives_seconds" =>
                                      Dict("rn222" => 330350.4,
                                            "kr85"  => 3.394e8))),
        Float64)
    @test op isa AT.Operators.Chemistry.ExponentialDecay
    @test length(op.tracer_names) == 2
    @test :rn222 in op.tracer_names
    @test :kr85 in op.tracer_names

    @test chemistry_spec(Dict{String,Any}())          isa NoChemistrySpec  # default
    @test chemistry_spec(Dict("kind" => "none"))      isa NoChemistrySpec
    @test chemistry_spec(Dict("kind" => "decay"))     isa NoChemistrySpec  # empty table
    s = chemistry_spec(Dict("kind" => "decay",
                            "half_lives_seconds" => Dict("rn222" => 330350.4)))
    @test s isa DecayChemistrySpec
    @test s.half_lives.rn222 == 330350.4
    @test_throws ArgumentError chemistry_spec(Dict("kind" => "photolysis"))
    # parse-time boundary validation on half-lives (positive number).
    @test_throws ArgumentError chemistry_spec(
        Dict("kind" => "decay", "half_lives_seconds" => Dict("rn222" => 0.0)))
    @test_throws ArgumentError chemistry_spec(
        Dict("kind" => "decay", "half_lives_seconds" => Dict("rn222" => -1.0)))
    @test_throws ArgumentError chemistry_spec(
        Dict("kind" => "decay", "half_lives_seconds" => Dict("rn222" => "3d")))
    # Bool <: Real, but a boolean half-life is a typo, not 1 second.
    @test_throws ArgumentError chemistry_spec(
        Dict("kind" => "decay", "half_lives_seconds" => Dict("rn222" => true)))

    # Float32 decay rate is bit-identical to the old pre-conversion builder:
    # ExponentialDecay forms FT(log(2)/FT(T)), so the half-life must be cast to
    # FT *before* the division (not Float32(log(2)/Float64(T))).
    T = 330350.4
    op32 = build_runtime_chemistry(
        Dict("chemistry" => Dict("kind" => "decay",
                                  "half_lives_seconds" => Dict("rn222" => T))), Float32)
    @test eltype(op32.decay_rates[1].value) === Float32
    @test field_value(op32.decay_rates[1], ()) === Float32(log(2) / Float32(T))
end


using AtmosTransport
include(joinpath(@__DIR__, "..", "helpers", "convection_runtime.jl"))

@testset "binary window symmetrically splits Rn-222 decay around emissions" begin
    FT = Float64
    Nx, Ny, Nz = 4, 3, 5
    driver, _, _ = _make_convection_window_driver(
        FT = FT, steps = 4, binary_contract = true)
    T = driver.dt
    half_life = FT(330_350.4)
    λ = log(FT(2)) / half_life
    E = FT(2.5)
    Bstar = E * T / (2sinh(λ * T / 2))

    air = fill(FT(_REALISTIC_AIR_MASS_KG), Nx, Ny, Nz)
    rn222 = zeros(FT, Nx, Ny, Nz)
    rn222[:, :, Nz] .= Bstar
    state = CellState(DryBasis, air; rn222)
    fluxes = allocate_face_fluxes(driver.grid.horizontal, Nz;
                                  FT, basis = DryBasis)
    model = TransportModel(state, fluxes, driver.grid, UpwindScheme())
    source = SurfaceFluxSource(:rn222, fill(E, Nx, Ny))
    decay = ExponentialDecay(FT; rn222 = half_life)
    sim = DrivenSimulation(model, driver;
                           start_window = 1, stop_window = 1,
                           surface_sources = (source,), chemistry = decay)

    run!(sim)

    burden = dropdims(sum(sim.model.state.tracers.rn222; dims = 3); dims = 3)
    @test all(isapprox.(burden, Bstar; rtol = 3e-15))
end
