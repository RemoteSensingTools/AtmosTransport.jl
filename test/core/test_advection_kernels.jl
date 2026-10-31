#!/usr/bin/env julia
"""
Cross-backend and cross-precision validation for the new generic advection
kernel shells (UpwindScheme, SlopesScheme through multitracer_kernels.jl).

Tests the 2×2 matrix:  {CPU, GPU} × {Float32, Float64}

For each combination, verifies:
1. Uniform field invariance (zero mixing ratio change under advection)
2. Mass conservation (total tracer mass preserved to machine precision)
3. CPU-GPU agreement within the scheme's ULP budget
4. F32 vs F64 systematic bias (no drift beyond expected truncation error)

Uses semi-realistic fluxes: sinusoidal zonal wind with latitude taper and
vertically-consistent cm diagnosed from horizontal continuity. This exercises
all three sweep directions with non-trivial Courant numbers.
"""

using Test
using AtmosTransport.Operators.StepPlans: transport_step!

using AtmosTransport
using .AtmosTransport: Operators, Grids
using .AtmosTransport.Grids: cell_areas_by_latitude
using .AtmosTransport.Operators: MonotoneLimiter, PositivityLimiter, CW84Limiter,
                                  strang_split!, strang_split_mt!
using .AtmosTransport.Operators.Advection: _limited_moment, _xface_tracer_flux
using .AtmosTransport.State: StructuredFaceFluxState

# Try to load CUDA; tests gracefully skip if unavailable
const HAS_GPU = try
    using CUDA
    CUDA.functional()
catch
    false
end

# =========================================================================
# Helper: build semi-realistic test problem
# =========================================================================

"""
    build_test_problem(FT; Nx=36, Ny=18, Nz=4, cfl=0.15)

Build a semi-realistic structured-grid advection test case.

Returns `(grid, m, rm_uniform, rm_gradient, am, bm, cm)` all in `FT` precision.

Fluxes:
- am: sinusoidal zonal mass flux, tapered to zero at poles
- bm: weak meridional convergence in the tropics
- cm: diagnosed from continuity (exact mass closure)
- CFL ≈ `cfl` at the equator

Initial fields:
- rm_uniform: uniform mixing ratio (400 ppm)
- rm_gradient: latitude-dependent gradient (300-500 ppm)
"""
function build_test_problem(FT; Nx=36, Ny=18, Nz=4, cfl=FT(0.15))
    mesh = LatLonMesh(; Nx=Nx, Ny=Ny, FT=FT)
    A_ifc = FT[0, 500, 5000, 30000, 0]
    B_ifc = FT[0, 0, FT(0.1), FT(0.5), 1]
    vc = HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT=FT)

    g  = gravity(grid)
    ps = reference_pressure(grid)
    areas = cell_areas_by_latitude(mesh)

    m = zeros(FT, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        m[i, j, k] = FT(level_thickness(vc, k, ps)) * areas[j] / g
    end

    χ_uniform = FT(400e-6)
    rm_uniform = m .* χ_uniform

    rm_gradient = similar(m)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        lat_frac = FT(j - 1) / FT(Ny - 1)
        χ_j = FT(300e-6) + FT(200e-6) * lat_frac
        rm_gradient[i, j, k] = m[i, j, k] * χ_j
    end

    # Semi-realistic mass fluxes
    m_min = minimum(m)
    am = zeros(FT, Nx + 1, Ny, Nz)
    bm = zeros(FT, Nx, Ny + 1, Nz)

    for k in 1:Nz, j in 1:Ny, i in 1:Nx+1
        lat = FT(-90) + FT(j - FT(0.5)) * (FT(180) / FT(Ny))
        lon = FT(i - 1) * (FT(360) / FT(Nx))
        cos_lat = cos(lat * FT(π) / FT(180))
        am[i, j, k] = cfl * m_min * cos_lat * (FT(1) + FT(0.3) * sin(lon * FT(π) / FT(180)))
    end
    am[:, 1, :]  .= zero(FT)
    am[:, Ny, :] .= zero(FT)

    for k in 1:Nz, j in 2:Ny, i in 1:Nx
        lat = FT(-90) + FT(j - 1) * (FT(180) / FT(Ny))
        bm[i, j, k] = cfl * m_min * FT(0.1) * sin(FT(2) * lat * FT(π) / FT(180))
    end

    cm = zeros(FT, Nx, Ny, Nz + 1)
    bt = FT[Grids.b_diff(vc, k) for k in 1:Nz]
    # Fixture pressure-fixer closure; the production explicit-dm closure is
    # tested separately by the replay tests.
    for j in 1:Ny, i in 1:Nx
        convergence = [am[i, j, k] - am[i+1, j, k] +
                       bm[i, j, k] - bm[i, j+1, k] for k in 1:Nz]
        pit = sum(convergence)
        for k in 1:Nz
            cm[i, j, k+1] = cm[i, j, k] + convergence[k] - bt[k] * pit
        end
    end

    return grid, m, rm_uniform, rm_gradient, am, bm, cm
end

"""
    run_strang!(m, rm, am, bm, cm, scheme; n_steps=1)

Run `n_steps` Strang splits on CPU arrays, return (m_out, rm_out).
"""
function run_strang!(m, rm, am, bm, cm, grid, scheme; n_steps=1)
    m_work  = copy(m)
    rm_work = copy(rm)
    state   = CellState(m_work; tracer=rm_work)
    fluxes  = StructuredFaceFluxState(copy(am), copy(bm), copy(cm))
    ws      = AdvectionWorkspace(state)
    for _ in 1:n_steps
        strang_split!(state, fluxes, grid, scheme; workspace=ws)
    end
    # Post-plan-14 storage: advection mutates state.tracers_raw, not the
    # input rm_work. Materialize the view into a fresh array to match
    # the pre-refactor return contract (a 3D Array, not a SubArray).
    return m_work, Array(state.tracers.tracer)
end

"""
    to_gpu(arrays...)

Move arrays to GPU (CuArray). Returns a tuple.
"""
function to_gpu(arrays...)
    return map(a -> CuArray(a), arrays)
end

# =========================================================================
# Test suite
# =========================================================================

@testset "default monotone moment is signed and offset-neutral" begin
    limiter = MonotoneLimiter()
    @test _limited_moment(0.0, -1.0, limiter) == 0.0
    @test _limited_moment(-0.25, -1.0, limiter) == -0.25
    @test _limited_moment(2.0, 1.0, limiter) == 2.0
    @test _limited_moment(-0.25, -1.0, PositivityLimiter()) == 0.0
end

# Issue #62: the positivity slope keeps its sign and is clipped to |s| ≤ 2c, the
# bound that keeps both face values c ± s/2 non-negative. A sign test that
# zeroed every downhill slope made the transport depend on the flow direction.
@testset "PositivityLimiter slope keeps its sign, |s| ≤ 2c" begin
    lim = PositivityLimiter()
    slope = Operators.Advection._limited_slope
    @test slope(10.5, 10.0, 9.5, lim) == -0.5      # downhill slope kept
    @test slope(9.5, 10.0, 10.5, lim) == 0.5
    @test slope(0.0, 0.1, 0.6, lim) == 0.2         # clipped at 2c
    @test slope(0.6, 0.1, 0.0, lim) == -0.2
    @test slope(3.0, 1.0, 0.5, lim) == -1.25       # within the bound: centred slope
    @test slope(0.2, 0.0, 0.4, lim) == 0.0         # empty cell: no slope

    # A mirrored profile advected the other way gives the mirrored result, and a
    # positive tracer stays non-negative.
    N, C = 100, 0.4
    nrev = round(Int, N / C)
    xc = (1:N) .- 0.5
    c0 = 0.02 .+ exp.(-((xc .- 30) ./ 5) .^ 2) .* (1 .+ 0.5 .* tanh.((xc .- 30) ./ 3)) .+
         0.8 .* (abs.(xc .- 72) .< 6)
    function advect_1d(c, courant, nsweeps)
        m = ones(N, 1, 1); rm = reshape(copy(c), N, 1, 1)
        am = fill(courant, N + 1, 1, 1); ws = AdvectionWorkspace(m)
        for _ in 1:nsweeps
            Operators.Advection.sweep_x_mt!(reshape(rm, size(rm)..., 1),
                reshape(ws.rm_B, size(rm)..., 1), m, ws.m_B, am, SlopesScheme(lim), ws)
            copyto!(rm, ws.rm_B); copyto!(m, ws.m_B)
        end
        return vec(rm) ./ vec(m)
    end
    right = advect_1d(c0, C, nrev)
    left = reverse(advect_1d(reverse(c0), -C, nrev))
    @test maximum(abs.(right .- left)) < 1e-12
    @test minimum(advect_1d(c0, C, 10nrev)) >= 0
end

@testset "structured monotone advection supports signed VMR" begin
    FT = Float64
    grid, m, _, rm_absolute, am, bm, cm = build_test_problem(FT)
    q0 = FT(400e-6)

    for scheme in (SlopesScheme(MonotoneLimiter()),
                   PPMScheme(MonotoneLimiter()),
                   PPMScheme(CW84Limiter()))
        @testset "$(nameof(typeof(scheme)))" begin
            # A uniform negative VMR is a valid signed contribution and must
            # remain uniform while the carrier mass evolves.
            rm_negative = -q0 .* m
            m_negative, rm_negative_out =
                run_strang!(m, rm_negative, am, bm, cm, grid, scheme)
            q_negative_out = rm_negative_out ./ m_negative
            @test q_negative_out ≈ fill(-q0, size(q_negative_out)) rtol=2e-13 atol=0

            # Carrier-aware affine contract:
            # A(r + q0*m_old) = A(r) + q0*m_new.
            rm_anomaly = rm_absolute .- q0 .* m
            m_abs, rm_abs_out =
                run_strang!(m, rm_absolute, am, bm, cm, grid, scheme; n_steps=2)
            m_anom, rm_anom_out =
                run_strang!(m, rm_anomaly, am, bm, cm, grid, scheme; n_steps=2)
            @test m_anom == m_abs
            reconstructed = rm_anom_out .+ q0 .* m_anom
            scale = maximum(abs, rm_abs_out)
            @test maximum(abs, reconstructed .- rm_abs_out) <= 256eps(scale)
        end
    end
end

@testset "SlopesScheme face flux uses half-slope moment" begin
    m0 = 10.0
    q = collect(1.0:6.0)
    m = fill(m0, 6, 1, 1)
    rm = reshape(m0 .* q, 6, 1, 1)
    scheme = SlopesScheme(MonotoneLimiter())

    # Face 4 lies between q=3 and q=4. For this linear stencil the limited
    # slope is 1, so the donor edge offset is 0.5.
    α = 0.2
    flux_pos = _xface_tracer_flux(Int32(4), 1, 1, rm, m, 2.0, scheme, Int32(6))
    expected_pos = α * (m0 * 3.0 + (1 - α) * (m0 * 0.5))
    @test flux_pos ≈ expected_pos

    flux_neg = _xface_tracer_flux(Int32(4), 1, 1, rm, m, -2.0, scheme, Int32(6))
    expected_neg = -α * (m0 * 4.0 - (1 - α) * (m0 * 0.5))
    @test flux_neg ≈ expected_neg
end

@testset "active diffusion is validated before advection mutates state" begin
    FT = Float64
    grid, m, rm, _, am, bm, cm = build_test_problem(FT; Nx=8, Ny=4, Nz=4)
    state = CellState(copy(m); tracer=copy(rm))
    fluxes = StructuredFaceFluxState(copy(am), copy(bm), copy(cm))
    advection_ws = AdvectionWorkspace(state)
    diffusion = ImplicitVerticalDiffusion(; kz_field=ConstantField{FT, 3}(one(FT)))
    m_before = copy(state.air_mass)
    rm_before = copy(state.tracers_raw)

    @test_throws ArgumentError transport_step!(
        state, fluxes, grid, UpwindScheme(), FT(60);
        workspace=advection_ws,
        diffusion_workspace=advection_ws,
        diffusion_op=diffusion,
    )
    @test state.air_mass == m_before
    @test state.tracers_raw == rm_before

    wrong_shape = DiffusionWorkspace(zeros(FT, size(m, 1), size(m, 2), size(m, 3) - 1))
    @test_throws DimensionMismatch transport_step!(
        state, fluxes, grid, UpwindScheme(), FT(60);
        workspace=advection_ws,
        diffusion_workspace=wrong_shape,
        diffusion_op=diffusion,
    )
    @test state.air_mass == m_before
    @test state.tracers_raw == rm_before

    diffusion_ws = DiffusionWorkspace(state)
    @test_throws ArgumentError transport_step!(
        state, fluxes, grid, UpwindScheme(), nothing;
        workspace=advection_ws,
        diffusion_workspace=diffusion_ws,
        diffusion_op=diffusion,
    )
    @test state.air_mass == m_before
    @test state.tracers_raw == rm_before
end

@testset "advection workspace shape is validated before mutation" begin
    FT = Float64
    grid, m, rm, _, am, bm, cm = build_test_problem(FT; Nx=8, Ny=4, Nz=4)
    state = CellState(copy(m); tracer=copy(rm))
    fluxes = StructuredFaceFluxState(copy(am), copy(bm), copy(cm))
    wrong_workspace = AdvectionWorkspace(state.air_mass)
    m_before = copy(state.air_mass)
    rm_before = copy(state.tracers_raw)

    @test_throws DimensionMismatch strang_split!(
        state, fluxes, grid, UpwindScheme(); workspace=wrong_workspace)
    @test state.air_mass == m_before
    @test state.tracers_raw == rm_before
end

@testset "Advection kernels: $(typeof(scheme))" for scheme in
        (UpwindScheme(), SlopesScheme(MonotoneLimiter()), PPMScheme(MonotoneLimiter()), PPMScheme(CW84Limiter()))

    for FT in (Float64, Float32)
        precision_tag = FT == Float64 ? "F64" : "F32"

        grid, m_cpu, rm_uni_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu =
            build_test_problem(FT)

        Nx, Ny, Nz = size(m_cpu)

        # --- CPU tests ---

        @testset "CPU $precision_tag: uniform invariance" begin
            m_out, rm_out = run_strang!(m_cpu, rm_uni_cpu, am_cpu, bm_cpu, cm_cpu, grid, scheme)
            χ_out = rm_out ./ max.(m_out, eps(FT))
            @test maximum(abs.(χ_out .- FT(400e-6))) / FT(400e-6) < FT(1e-6)
        end

        @testset "CPU $precision_tag: mass conservation (uniform)" begin
            m_out, rm_out = run_strang!(m_cpu, rm_uni_cpu, am_cpu, bm_cpu, cm_cpu, grid, scheme)
            tol = FT == Float64 ? 1e-13 : FT(1e-5)
            @test abs(sum(m_out) - sum(m_cpu)) / sum(m_cpu) < tol
            @test abs(sum(rm_out) - sum(rm_uni_cpu)) / sum(rm_uni_cpu) < tol
        end

        @testset "CPU $precision_tag: mass conservation (gradient, 4 steps)" begin
            m_out, rm_out = run_strang!(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu, grid, scheme; n_steps=4)
            tol = FT == Float64 ? 1e-12 : FT(5e-5)
            @test abs(sum(m_out) - sum(m_cpu)) / sum(m_cpu) < tol
            @test abs(sum(rm_out) - sum(rm_grad_cpu)) / sum(rm_grad_cpu) < tol
        end

        # --- GPU tests (skip if no CUDA) ---

        if HAS_GPU

            @testset "GPU $precision_tag: mass conservation (gradient, 4 steps)" begin
                m_g, rm_g, am_g, bm_g, cm_g = to_gpu(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu)

                state_g  = CellState(copy(m_g); tracer=copy(rm_g))
                fluxes_g = StructuredFaceFluxState(copy(am_g), copy(bm_g), copy(cm_g))
                ws_g     = AdvectionWorkspace(state_g)

                for _ in 1:4
                    strang_split!(state_g, fluxes_g, grid, scheme; workspace=ws_g)
                end

                m_gpu  = Array(state_g.air_mass)
                rm_gpu = Array(state_g.tracers[:tracer])

                tol = FT == Float64 ? 1e-12 : FT(5e-5)
                @test abs(sum(m_gpu) - sum(m_cpu)) / sum(m_cpu) < tol
                @test abs(sum(rm_gpu) - sum(rm_grad_cpu)) / sum(rm_grad_cpu) < tol
            end

            @testset "GPU $precision_tag: uniform invariance" begin
                m_g, rm_g, am_g, bm_g, cm_g = to_gpu(m_cpu, rm_uni_cpu, am_cpu, bm_cpu, cm_cpu)

                state_g  = CellState(copy(m_g); tracer=copy(rm_g))
                fluxes_g = StructuredFaceFluxState(copy(am_g), copy(bm_g), copy(cm_g))
                ws_g     = AdvectionWorkspace(state_g)

                strang_split!(state_g, fluxes_g, grid, scheme; workspace=ws_g)

                rm_gpu = Array(state_g.tracers[:tracer])
                m_gpu  = Array(state_g.air_mass)
                χ_gpu  = rm_gpu ./ max.(m_gpu, eps(FT))
                @test maximum(abs.(χ_gpu .- FT(400e-6))) / FT(400e-6) < FT(1e-6)
            end

            @testset "CPU-GPU agreement $precision_tag (1 step, gradient)" begin
                m_out_cpu, rm_out_cpu = run_strang!(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu, grid, scheme)

                m_g, rm_g, am_g, bm_g, cm_g = to_gpu(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu)
                state_g  = CellState(copy(m_g); tracer=copy(rm_g))
                fluxes_g = StructuredFaceFluxState(copy(am_g), copy(bm_g), copy(cm_g))
                ws_g     = AdvectionWorkspace(state_g)
                strang_split!(state_g, fluxes_g, grid, scheme; workspace=ws_g)

                rm_gpu = Array(state_g.tracers[:tracer])
                m_gpu  = Array(state_g.air_mass)

                # GPU FMA instructions may produce ≤ few ULP difference vs CPU
                ulp_tol = scheme isa UpwindScheme ? FT(4) : FT(16)
                max_rm_ulp = maximum(abs.(rm_gpu .- rm_out_cpu)) / eps(maximum(abs.(rm_out_cpu)))
                max_m_ulp  = maximum(abs.(m_gpu .- m_out_cpu))   / eps(maximum(abs.(m_out_cpu)))
                println("    CPU-GPU $precision_tag 1-step: rm_ulp=", max_rm_ulp, " m_ulp=", max_m_ulp)
                @test max_rm_ulp < ulp_tol
                @test max_m_ulp  < ulp_tol
            end

            @testset "CPU-GPU agreement $precision_tag (4 steps, gradient)" begin
                m_out_cpu, rm_out_cpu = run_strang!(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu, grid, scheme; n_steps=4)

                m_g, rm_g, am_g, bm_g, cm_g = to_gpu(m_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu)
                state_g  = CellState(copy(m_g); tracer=copy(rm_g))
                fluxes_g = StructuredFaceFluxState(copy(am_g), copy(bm_g), copy(cm_g))
                ws_g     = AdvectionWorkspace(state_g)
                for _ in 1:4
                    strang_split!(state_g, fluxes_g, grid, scheme; workspace=ws_g)
                end

                rm_gpu = Array(state_g.tracers[:tracer])
                m_gpu  = Array(state_g.air_mass)

                # After 4 steps, FMA-induced ULP drift accumulates modestly
                ulp_tol = FT == Float64 ? FT(16) : FT(16)
                max_rm_ulp = maximum(abs.(rm_gpu .- rm_out_cpu)) / eps(maximum(abs.(rm_out_cpu)))
                max_m_ulp  = maximum(abs.(m_gpu .- m_out_cpu))   / eps(maximum(abs.(m_out_cpu)))
                println("    CPU-GPU $precision_tag 4-step: rm_ulp=", max_rm_ulp, " m_ulp=", max_m_ulp)
                @test max_rm_ulp < ulp_tol
                @test max_m_ulp  < ulp_tol
            end

        else
            @testset "GPU $precision_tag: SKIPPED (no CUDA)" begin
                @test_skip false
            end
        end
    end

    # --- F32 vs F64 systematic bias check ---

    @testset "F32 vs F64 systematic bias (4 steps)" begin
        grid64, m64, _, rm_grad64, am64, bm64, cm64 = build_test_problem(Float64)
        grid32, m32, _, rm_grad32, am32, bm32, cm32 = build_test_problem(Float32)

        _, rm_out64 = run_strang!(m64, rm_grad64, am64, bm64, cm64, grid64, scheme; n_steps=4)
        _, rm_out32 = run_strang!(m32, rm_grad32, am32, bm32, cm32, grid32, scheme; n_steps=4)

        rm64_as32 = Float32.(rm_out64)
        rel_diff  = maximum(abs.(rm_out32 .- rm64_as32)) / maximum(abs.(rm64_as32))

        println("  F32 vs F64 max relative difference (4 steps): ", rel_diff)
        @test rel_diff < Float32(1e-4)

        bias = (sum(Float64.(rm_out32)) - sum(rm_out64)) / sum(rm_out64)
        @test abs(bias) < 1e-6
        @test abs(bias) < 1e-6
    end

end

# =========================================================================
# Multi-tracer kernel fusion: equivalence with per-tracer path
# =========================================================================

@testset "Multi-tracer kernel fusion" begin

    for FT in (Float64, Float32)
        precision_tag = FT == Float64 ? "F64" : "F32"

        grid, m_cpu, rm_uni_cpu, rm_grad_cpu, am_cpu, bm_cpu, cm_cpu =
            build_test_problem(FT)

        Nx, Ny, Nz = size(m_cpu)
        Nt = 5

        rm_tracers = [rm_grad_cpu .* FT(0.5 + 0.1 * t) for t in 1:Nt]

        for (scheme, scheme_tag) in (
            (UpwindScheme(), "Upwind"),
            (SlopesScheme(MonotoneLimiter()), "Slopes"),
            (PPMScheme(MonotoneLimiter()), "PPM"),
            (PPMScheme(CW84Limiter()), "PPM-CW84"),
        )
            @testset "CPU $precision_tag $scheme_tag: MT ≡ per-tracer (4 steps)" begin
                # Per-tracer path (reference)
                m_ref = copy(m_cpu)
                rm_ref = [copy(rm) for rm in rm_tracers]
                ws_ref = AdvectionWorkspace(m_ref; n_tracers=1)
                for _ in 1:4
                    m_save = copy(m_ref)
                    for t in 1:Nt
                        if t > 1
                            copyto!(m_ref, m_save)
                        end
                        strang_split_mt!(reshape(rm_ref[t], size(m_ref)..., 1), m_ref,
                                         am_cpu, bm_cpu, cm_cpu, scheme, ws_ref)
                    end
                end

                # Multi-tracer path
                m_mt = copy(m_cpu)
                rm_4d = zeros(FT, Nx, Ny, Nz, Nt)
                for t in 1:Nt
                    rm_4d[:, :, :, t] .= rm_tracers[t]
                end
                ws_mt = AdvectionWorkspace(m_mt; n_tracers=Nt)
                for _ in 1:4
                    strang_split_mt!(rm_4d, m_mt, am_cpu, bm_cpu, cm_cpu, scheme, ws_mt)
                end

                # Compare: must be BIT-IDENTICAL (same arithmetic, same order)
                for t in 1:Nt
                    max_diff = maximum(abs.(rm_4d[:, :, :, t] .- rm_ref[t]))
                    @test max_diff == zero(FT)
                end
                @test maximum(abs.(m_mt .- m_ref)) == zero(FT)
            end


        end

        if HAS_GPU
            @testset "GPU $precision_tag: MT CPU-GPU agreement (Slopes, 4 steps)" begin
                scheme = SlopesScheme(MonotoneLimiter())

                # CPU reference
                m_mt_cpu = copy(m_cpu)
                rm_4d_cpu = zeros(FT, Nx, Ny, Nz, Nt)
                for t in 1:Nt
                    rm_4d_cpu[:, :, :, t] .= rm_tracers[t]
                end
                ws_cpu = AdvectionWorkspace(m_mt_cpu; n_tracers=Nt)
                for _ in 1:4
                    strang_split_mt!(rm_4d_cpu, m_mt_cpu, am_cpu, bm_cpu, cm_cpu, scheme, ws_cpu)
                end

                # GPU
                m_g = CuArray(copy(m_cpu))
                rm_4d_g = CuArray(zeros(FT, Nx, Ny, Nz, Nt))
                for t in 1:Nt
                    rm_4d_g[:, :, :, t] .= CuArray(rm_tracers[t])
                end
                am_g, bm_g, cm_g = CuArray(am_cpu), CuArray(bm_cpu), CuArray(cm_cpu)
                ws_g = AdvectionWorkspace(m_g; n_tracers=Nt)
                for _ in 1:4
                    strang_split_mt!(rm_4d_g, m_g, am_g, bm_g, cm_g, scheme, ws_g)
                end

                rm_4d_back = Array(rm_4d_g)
                m_back = Array(m_g)
                ulp_tol = FT(16)
                for t in 1:Nt
                    max_ulp = maximum(abs.(rm_4d_back[:,:,:,t] .- rm_4d_cpu[:,:,:,t])) /
                              eps(maximum(abs.(rm_4d_cpu[:,:,:,t])))
                    @test max_ulp < ulp_tol
                end
                max_m_ulp = maximum(abs.(m_back .- m_mt_cpu)) / eps(maximum(abs.(m_mt_cpu)))
                println("    MT CPU-GPU $precision_tag: m_ulp=$max_m_ulp")
                @test max_m_ulp < ulp_tol
            end
        else
            @testset "GPU $precision_tag MT: SKIPPED (no CUDA)" begin
                @test_skip false
            end
        end
    end

end

# =========================================================================
# Plan 13 regression test: the static palindrome CFL pilot must return
# n_sub=1 for the reference synthetic flow (CFL ≈ 0.15 at the equator) with
# the default cfl_limit of 0.95.
# =========================================================================

println("\n✓ All advection kernel cross-backend/precision tests completed!")
