#!/usr/bin/env julia
"""
CubedSphere chemistry dispatch tests (plan 21 follow-up: close the CS
chemistry gap in TOPOLOGY_SUPPORT.md).

Mirrors `test_chemistry.jl` but for `CubedSphereState` storage:

1. `NoChemistry` — identity across all six panels.
2. `ExponentialDecay{FT, N}` — decays selected tracers by
   `exp(-rate * dt)` exactly (F64) or to ULP (F32) on every panel,
   including halo cells. Unselected tracers are unchanged.
3. `CompositeChemistry` — sequential sub-operator application.
4. Argument error when a named tracer isn't present.

`get_tracer(state, name)` returns an `NTuple{6, Array}`; we check each
panel independently, respecting the plan-14 accessor-API discipline.
"""

using Test

using AtmosTransport

# -------------------------------------------------------------------------
# Tiny CubedSphere test problem
# -------------------------------------------------------------------------

"""
    _make_cs_state(FT; Nc=4, Hp=1, Nz=2, co2_init, rn_init)

Build a small `CubedSphereState` with two tracers initialized to known
values on every cell of every panel (interior + halo). Returns the
state and the panel shape `(Nc + 2Hp, Nc + 2Hp, Nz)` for reference.
"""
function _make_cs_state(FT::Type{<:AbstractFloat}; Nc::Int = 4, Hp::Int = 1,
                        Nz::Int = 2, co2_init = FT(1.0), rn_init = FT(2.0))
    panel_shape = (Nc + 2Hp, Nc + 2Hp, Nz)
    air_mass = ntuple(_ -> fill(FT(1.0), panel_shape...), 6)
    co2     = ntuple(_ -> fill(FT(co2_init), panel_shape...), 6)
    rn      = ntuple(_ -> fill(FT(rn_init),  panel_shape...), 6)
    mesh    = CubedSphereMesh(; FT = FT, Nc = Nc, Hp = Hp,
                              convention = AtmosTransport.Grids.GnomonicPanelConvention())
    state   = CubedSphereState(DryBasis, mesh, air_mass;
                               CO2 = co2, Rn222 = rn)
    return state, panel_shape
end

# Rn-222 half-life (seconds). Matches CATRINE config.
const RN222_HALF_LIFE = 330_350.4
const RN222_LAMBDA    = log(2) / RN222_HALF_LIFE

# =========================================================================
# Test suite
# =========================================================================

@testset "Chemistry on CubedSphereState (plan 21 follow-up)" begin

for FT in (Float64, Float32)
    tag = FT === Float64 ? "F64" : "F32"

    # -------------------------------------------------------------------
    # NoChemistry: identity
    # -------------------------------------------------------------------
    @testset "CS $tag: NoChemistry is identity" begin
        state, _ = _make_cs_state(FT)
        co2_before = deepcopy(get_tracer(state, :CO2))
        rn_before  = deepcopy(get_tracer(state, :Rn222))
        apply!(state, nothing, nothing, NoChemistry(), FT(3600))
        for p in 1:6
            @test get_tracer(state, :CO2)[p]   == co2_before[p]
            @test get_tracer(state, :Rn222)[p] == rn_before[p]
        end
    end

    # -------------------------------------------------------------------
    # ExponentialDecay: single decaying tracer
    # -------------------------------------------------------------------
    @testset "CS $tag: Rn-222 decays by exp(-λ·dt) on every panel" begin
        state, _ = _make_cs_state(FT)
        op  = ExponentialDecay(FT; Rn222 = RN222_HALF_LIFE)
        dt  = FT(3600)
        co2_before = deepcopy(get_tracer(state, :CO2))
        apply!(state, nothing, nothing, op, dt)
        expected = FT(2.0) * exp(-FT(RN222_LAMBDA) * dt)
        for p in 1:6
            @test all(get_tracer(state, :Rn222)[p] .≈ expected)
            @test get_tracer(state, :CO2)[p] == co2_before[p]   # untouched
        end
    end

    # -------------------------------------------------------------------
    # Multi-tracer independence: each tracer decays at its own rate
    # -------------------------------------------------------------------
    @testset "CS $tag: multi-tracer independent decay" begin
        state, _ = _make_cs_state(FT)
        op  = ExponentialDecay(FT; Rn222 = RN222_HALF_LIFE, CO2 = 1e15)  # CO2: essentially no decay at small dt
        dt  = FT(3600)
        apply!(state, nothing, nothing, op, dt)
        rn_expected   = FT(2.0) * exp(-FT(RN222_LAMBDA) * dt)
        co2_expected  = FT(1.0) * exp(-FT(log(2) / 1e15) * dt)
        for p in 1:6
            @test all(get_tracer(state, :Rn222)[p] .≈ rn_expected)
            @test all(get_tracer(state, :CO2)[p]   .≈ co2_expected)
        end
    end

    # -------------------------------------------------------------------
    # Halo cells decay alongside interior
    # -------------------------------------------------------------------
    @testset "CS $tag: halo cells decay uniformly" begin
        state, panel_shape = _make_cs_state(FT)
        op  = ExponentialDecay(FT; Rn222 = RN222_HALF_LIFE)
        dt  = FT(3600)
        apply!(state, nothing, nothing, op, dt)
        expected = FT(2.0) * exp(-FT(RN222_LAMBDA) * dt)
        # Panel halo corners (specifically the Hp=1 border row/col) should
        # match the interior value since we initialized uniformly and the
        # kernel launches over the full panel including halos.
        for p in 1:6
            panel = get_tracer(state, :Rn222)[p]
            @test panel[1, 1, 1]                       ≈ expected
            @test panel[end, end, end]                 ≈ expected
            @test panel[panel_shape[1]÷2, panel_shape[2]÷2, 1] ≈ expected
        end
    end

    # -------------------------------------------------------------------
    # CompositeChemistry: sequential application
    # -------------------------------------------------------------------
    @testset "CS $tag: CompositeChemistry chains sub-operators" begin
        state, _ = _make_cs_state(FT)
        op1 = ExponentialDecay(FT; Rn222 = RN222_HALF_LIFE)
        op2 = ExponentialDecay(FT; Rn222 = RN222_HALF_LIFE * 2)
        composite = CompositeChemistry(op1, op2)
        dt = FT(3600)
        apply!(state, nothing, nothing, composite, dt)
        # After two sequential decays at λ₁ and λ₂, value is
        #   rn * exp(-λ₁·dt) * exp(-λ₂·dt) = rn * exp(-(λ₁+λ₂)·dt)
        λ₁ = FT(log(2) / RN222_HALF_LIFE)
        λ₂ = FT(log(2) / (RN222_HALF_LIFE * 2))
        expected = FT(2.0) * exp(-(λ₁ + λ₂) * dt)
        for p in 1:6
            @test all(get_tracer(state, :Rn222)[p] .≈ expected)
        end
    end

    # -------------------------------------------------------------------
    # ArgumentError when a named tracer is missing
    # -------------------------------------------------------------------
    @testset "CS $tag: missing tracer throws ArgumentError" begin
        state, _ = _make_cs_state(FT)
        op = ExponentialDecay(FT; MissingTracer = 1000.0)
        @test_throws ArgumentError apply!(state, nothing, nothing, op, FT(3600))
    end

    # -------------------------------------------------------------------
    # More decaying tracers than one launch carries: each decays exactly once;
    # t11, t22, t33, t44 between them do not decay
    # -------------------------------------------------------------------
    @testset "CS $tag: 40 decaying tracers" begin
        panel_shape = (6, 6, 2)
        mesh  = CubedSphereMesh(; FT = FT, Nc = 4, Hp = 1,
                                convention = AtmosTransport.Grids.GnomonicPanelConvention())
        state = CubedSphereState(DryBasis, mesh, ntuple(_ -> ones(FT, panel_shape), 6);
                                 (Symbol(:t, n) => ntuple(_ -> fill(FT(n), panel_shape), 6) for n in 1:44)...)
        decaying = filter(n -> n % 11 != 0, 1:44)
        op = ExponentialDecay(FT; (Symbol(:t, n) => 3_600.0 * n for n in decaying)...)
        d  = AtmosTransport.Operators.Chemistry._decay_decrements(op, nothing, FT(3600))
        expected = FT.(1:44)
        for (k, n) in enumerate(decaying)
            expected[n] = muladd(FT(n), d[k], FT(n))
        end
        apply!(state, nothing, nothing, op, FT(3600))
        @test all(n -> all(p -> all(x -> isapprox(x, expected[n]; rtol = 2eps(FT)), get_tracer(state, Symbol(:t, n))[p]), 1:6), 1:44)
    end

    # -------------------------------------------------------------------
    # Zero-operator ExponentialDecay is identity (N == 0 branch)
    # -------------------------------------------------------------------
    @testset "CS $tag: zero-tracer ExponentialDecay is identity" begin
        state, _ = _make_cs_state(FT)
        # Construct an N=0 operator directly (keyword ctor won't allow it)
        op = ExponentialDecay{FT, 0, Tuple{}}((), ())
        co2_before = deepcopy(get_tracer(state, :CO2))
        rn_before  = deepcopy(get_tracer(state, :Rn222))
        apply!(state, nothing, nothing, op, FT(3600))
        for p in 1:6
            @test get_tracer(state, :CO2)[p]   == co2_before[p]
            @test get_tracer(state, :Rn222)[p] == rn_before[p]
        end
    end

end # FT loop

# =========================================================================
# CompositeChemistry dispatches cleanly to CubedSphereState via apply!.
# =========================================================================
@testset "CompositeChemistry on CubedSphereState" begin
    state, _ = _make_cs_state(Float64)
    ops = (ExponentialDecay(Float64; Rn222 = RN222_HALF_LIFE),
           NoChemistry())
    dt = 3600.0
    apply!(state, nothing, nothing, CompositeChemistry(ops), dt)
    expected = 2.0 * exp(-(log(2) / RN222_HALF_LIFE) * dt)
    for p in 1:6
        @test all(get_tracer(state, :Rn222)[p] .≈ expected)
    end
end

end # outer testset
