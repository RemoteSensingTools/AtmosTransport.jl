#!/usr/bin/env julia
"""
Chemistry operator unit tests (plan 15).

Validates `apply!(state, meteo, grid, op, dt; workspace)` for the active
chemistry operators. Optional defaults are checked in test_operator_defaults.jl:

1. `ExponentialDecay{FT, N}` — multi-tracer first-order decay:
   * zero-rate tracer unchanged (by not being registered).
   * nonzero-rate tracer decays by `exp(-rate * dt)` to ULP.
   * multi-tracer independence (per-tracer rates).
   * mass sum conservation under uniform-rate decay.
   * CPU-GPU agreement (ULP tolerance).
   * throws for names not in state.
2. `CompositeChemistry` — sequential composition.
3. End-of-operator observation through the accessor API
   (`state.tracers.X`, `get_tracer(state, :X)`), NOT through the caller's
   input array (plan-14 test discipline).
"""

using Test

using AtmosTransport
using AtmosTransport.State: field_value

# Try to load CUDA; skip GPU tests gracefully if unavailable
const HAS_GPU_CHEM = try
    using CUDA
    CUDA.functional()
catch
    false
end

# =========================================================================
# Tiny test problem (3D air_mass; no grid needed — apply! ignores grid/meteo
# for chemistry).
# =========================================================================

"Build a small CellState with two tracers at known initial mass densities."
function _make_state(FT; Nx = 4, Ny = 3, Nz = 2,
                     co2_init = FT(1.0), rn_init = FT(2.0))
    m = ones(FT, Nx, Ny, Nz)
    rm_co2 = fill(co2_init, Nx, Ny, Nz)
    rm_rn  = fill(rn_init,  Nx, Ny, Nz)
    return CellState(m; CO2 = rm_co2, Rn222 = rm_rn)
end

"""
Tracers `t1 … t44` holding `n`, of which the 40 with `n % 11 != 0` decay with
distinct half-lives (more than one decay launch carries); `t11, t22, t33, t44`
sit between them and do not decay. Returns the state, the operator and each
tracer's value after one `dt = 3600` step from the mathematical exponential.
"""
function _make_many_decay_case(FT, AT)
    state = CellState(AT(ones(FT, 4, 3, 2)); (Symbol(:t, n) => AT(fill(FT(n), 4, 3, 2)) for n in 1:44)...)
    decaying = filter(n -> n % 11 != 0, 1:44)
    op = ExponentialDecay(FT; (Symbol(:t, n) => 3_600.0 * n for n in decaying)...)
    expected = FT.(1:44)
    for n in decaying
        rate = FT(log(2) / (3_600.0 * n))
        expected[n] = FT(n) * exp(-rate * FT(3600))
    end
    return state, op, expected
end

# Every cell of every tracer within a rounding of `expected`: the host and the
# device may or may not fuse `muladd`. One decay more or less is far outside.
_matches_many_decay(get, expected, FT) =
    all(n -> all(x -> isapprox(x, expected[n]; rtol = 2eps(FT)), Array(get(Symbol(:t, n)))), 1:44)

# Half-life of Rn-222 (3.8235 days) in seconds, matches CATRINE config value.
const RN222_HALF_LIFE = 330_350.4
const RN222_LAMBDA    = log(2) / RN222_HALF_LIFE

# =========================================================================
# Test suite
# =========================================================================

@testset "Chemistry operators (plan 15)" begin

for FT in (Float64, Float32)
    precision_tag = FT === Float64 ? "F64" : "F32"

    # -------------------------------------------------------------------
    # ExponentialDecay: single decaying tracer
    # -------------------------------------------------------------------
    @testset "CPU $precision_tag: Rn-222 decays by exp(-λ·dt) exactly" for dt in FT.((3600, 86400))
        state = _make_state(FT)
        op    = ExponentialDecay(FT; Rn222 = RN222_HALF_LIFE)
        # decay_rates are AbstractTimeVaryingField{FT, 0} (ConstantField)
        # after plan 16a retrofit — observe via field_value, not struct equality.
        @test field_value(op.decay_rates[1], ()) ≈ FT(RN222_LAMBDA)
        @test op.tracer_names == (:Rn222,)

        sum_before = sum(get_tracer(state, :Rn222))
        apply!(state, nothing, nothing, op, dt)

        expected = FT(2.0) * exp(-FT(RN222_LAMBDA) * dt)
        # Access through accessor API per plan-14 discipline
        rn_after = get_tracer(state, :Rn222)
        @test all(rn_after .≈ expected)
        # Property-syntax access gives the same thing
        @test all(state.tracers.Rn222 .≈ expected)
        tolerance = FT === Float64 ? 1e-12 : FT(1e-5)
        @test sum(rn_after) ≈ sum_before * exp(-FT(RN222_LAMBDA) * dt) rtol=tolerance
    end

    # -------------------------------------------------------------------
    # Name-not-in-state throws
    # -------------------------------------------------------------------
    @testset "CPU $precision_tag: unknown tracer throws" begin
        state = _make_state(FT)
        op = ExponentialDecay(FT; NotInState = 1.0)
        @test_throws ArgumentError apply!(state, nothing, nothing, op, FT(3600))
    end

    # -------------------------------------------------------------------
    # CompositeChemistry: sequential application
    # -------------------------------------------------------------------
    @testset "CPU $precision_tag: CompositeChemistry composes" begin
        state = _make_state(FT)
        op1 = ExponentialDecay(FT; Rn222 = RN222_HALF_LIFE)
        op2 = ExponentialDecay(FT; Rn222 = RN222_HALF_LIFE)  # apply twice
        comp = CompositeChemistry(op1, op2)
        dt = FT(3600)
        apply!(state, nothing, nothing, comp, dt)
        expected = FT(2.0) * exp(-FT(RN222_LAMBDA) * 2 * dt)  # two decay steps
        @test all(state.tracers.Rn222 .≈ expected)
    end

    # -------------------------------------------------------------------
    # More decaying tracers than one launch carries: each decays exactly once
    # -------------------------------------------------------------------
    @testset "CPU $precision_tag: 40 decaying tracers" begin
        state, op, expected = _make_many_decay_case(FT, Array)
        apply!(state, nothing, nothing, op, FT(3600))
        @test _matches_many_decay(name -> get_tracer(state, name), expected, FT)
    end

    # -------------------------------------------------------------------
    # GPU agreement (F32 only on L40S / F64 only on A100)
    # -------------------------------------------------------------------
    if HAS_GPU_CHEM
        @testset "GPU $precision_tag: 40 decaying tracers" begin
            state, op, expected = _make_many_decay_case(FT, CUDA.CuArray)
            apply!(state, nothing, nothing, op, FT(3600))
            @test _matches_many_decay(name -> get_tracer(state, name), expected, FT)
        end

        @testset "GPU $precision_tag: matches CPU to ULP" begin
            state_cpu = _make_state(FT)
            state_gpu = CellState(CUDA.CuArray(Array(state_cpu.air_mass));
                                  CO2   = CUDA.CuArray(Array(state_cpu.tracers.CO2)),
                                  Rn222 = CUDA.CuArray(Array(state_cpu.tracers.Rn222)))

            op = ExponentialDecay(FT; Rn222 = RN222_HALF_LIFE)
            dt = FT(3600)
            apply!(state_cpu, nothing, nothing, op, dt)
            apply!(state_gpu, nothing, nothing, op, dt)

            rn_cpu = state_cpu.tracers.Rn222
            rn_gpu = Array(state_gpu.tracers.Rn222)
            ulp_tol = FT === Float64 ? FT(4) : FT(4)
            max_ulp = maximum(abs.(rn_gpu .- rn_cpu)) / eps(maximum(abs.(rn_cpu)))
            @test max_ulp < ulp_tol
        end
    end
end  # for FT

# =========================================================================
# End-to-end: advection + decay composition via TransportModel.step!
# =========================================================================

@testset "End-to-end advection + decay composition (TransportModel.step!)" begin
    FT = Float64

    # Small synthetic LatLon problem. Uniform Rn-222 + uniform CO2 + zero
    # mass fluxes ⇒ advection is exactly mass-preserving, decay is exact.
    # This verifies that `step!` composes the two operators correctly.
    Nx, Ny, Nz = 6, 4, 2
    mesh = LatLonMesh(; Nx = Nx, Ny = Ny, FT = FT)
    A_ifc = zeros(FT, Nz + 1)
    B_ifc = FT.(collect(range(0, 1, length = Nz + 1)))
    vc = AtmosTransport.HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosTransport.AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT = FT)

    # Uniform air mass, uniform tracers.
    m = ones(FT, Nx, Ny, Nz)
    rm_co2 = fill(FT(1.5), Nx, Ny, Nz)    # non-decaying
    rm_rn  = fill(FT(2.5), Nx, Ny, Nz)    # decaying
    state = CellState(m; CO2 = rm_co2, Rn222 = rm_rn)

    # Zero mass fluxes — advection is a no-op on mass (and on tracers
    # since the field is uniform).
    am = zeros(FT, Nx + 1, Ny, Nz)
    bm = zeros(FT, Nx, Ny + 1, Nz)
    cm = zeros(FT, Nx, Ny, Nz + 1)
    fluxes = AtmosTransport.StructuredFaceFluxState(am, bm, cm)

    chem = ExponentialDecay(FT; Rn222 = RN222_HALF_LIFE)
    model = TransportModel(state, fluxes, grid, UpwindScheme();
                           chemistry = chem)

    n_steps = 8
    dt = FT(3600)    # 1-hour steps, 8 hours total
    m_rn_initial = sum(state.tracers.Rn222)
    m_co2_initial = sum(state.tracers.CO2)

    for _ in 1:n_steps
        step!(model, dt)
    end

    # Rn-222: total mass = M₀ · exp(-λ · n_steps · dt)
    expected_rn = m_rn_initial * exp(-FT(RN222_LAMBDA) * n_steps * dt)
    rn_final = sum(state.tracers.Rn222)
    @test abs(rn_final - expected_rn) / expected_rn < 1e-13

    # CO2: exactly preserved (no decay, no advection flux, no source).
    @test sum(state.tracers.CO2) == m_co2_initial

    # Air mass: preserved (zero fluxes).
    @test sum(state.air_mass) == Nx * Ny * Nz
end

end  # @testset
