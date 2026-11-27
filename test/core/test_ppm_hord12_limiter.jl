#!/usr/bin/env julia
#
# `PPMScheme(Hord12Limiter())`: FV3's horizontal tracer PPM with hord = 12, as
# in `tp_core.F90` (`xppm`) of the GCHP fork of FV3
# (geoschem/GFDL_atmos_cubed_sphere, gchp/main, 643f3c44): edges lines 534-541,
# the profile constraint lines 556-575, the face value lines 621-627. The tests
# compare it with hand-computed profiles and with an independent loop
# transcription of `xppm` on a periodic row, and check positivity, mass
# conservation, convergence and uniform fields.

using Test
using Random
using AtmosTransport
using .AtmosTransport.Operators: PPMScheme, SlopesScheme, MonotoneLimiter, CW84Limiter,
    Hord12Limiter, ppm_limit_profile
using .AtmosTransport.Operators.Advection: _xface_tracer_flux, _yface_tracer_flux,
    _zface_tracer_flux, _ppm_edge_value
using .AtmosTransport.Operators: FV3ScalarProfile
using .AtmosTransport.Models: advection_spec, materialize, LatLonRuntimeRecipeStyle,
    CubedSphereRuntimeRecipeStyle

const HORD12 = PPMScheme(Hord12Limiter())

# The limited edges (q_L, q_R) of the middle cell of five cell means.
function hord12_profile(c)
    q_L = _ppm_edge_value(c[1], c[2], c[3], c[4], Hord12Limiter())
    q_R = _ppm_edge_value(c[2], c[3], c[4], c[5], Hord12Limiter())
    return ppm_limit_profile(q_L, c[3], q_R, (c[1], c[2], c[4], c[5]), Hord12Limiter())
end

# Independent transcription of FV3 `xppm`, hord = 12, on a periodic row: loop
# form, with FV3's names. `courant[i]` is the Courant number of the face
# between cells i-1 and i; returns the face values (flux / mass flux).
function fv3_xppm_hord12(q, courant)
    n = length(q)
    w(i) = q[mod1(i, n)]
    dm = map(1:n) do i
        xt = 0.25 * (w(i + 1) - w(i - 1))
        sign(xt) * min(abs(xt), max(w(i - 1), w(i), w(i + 1)) - w(i),
                       w(i) - min(w(i - 1), w(i), w(i + 1)))
    end
    d(i) = dm[mod1(i, n)]
    al(i) = 0.5 * (w(i - 1) + w(i)) + (1 / 3) * (d(i - 1) - d(i))
    dq(i) = 2 * (w(i + 1) - w(i))
    bl, br = zeros(n), zeros(n)
    for i in 1:n
        bl[i] = al(i) - w(i)
        br[i] = al(i + 1) - w(i)
        if abs(d(i - 1)) + abs(d(i)) + abs(d(i + 1)) < 1e-25
            bl[i] = 0.0
            br[i] = 0.0
        elseif abs(3 * (bl[i] + br[i])) > abs(bl[i] - br[i])
            pmp_2 = dq(i - 1)
            lac_2 = pmp_2 - 0.75 * dq(i - 2)
            br[i] = min(max(0.0, pmp_2, lac_2), max(br[i], min(0.0, pmp_2, lac_2)))
            pmp_1 = -dq(i)
            lac_1 = pmp_1 + 0.75 * dq(i + 1)
            bl[i] = min(max(0.0, pmp_1, lac_1), max(bl[i], min(0.0, pmp_1, lac_1)))
        end
    end
    return map(1:n) do i
        c, im = courant[i], mod1(i - 1, n)
        c > 0 ? w(i - 1) + (1 - c) * (br[im] - c * (bl[im] + br[im])) :
                w(i) + (1 + c) * (bl[i] + c * (bl[i] + br[i]))
    end
end

# `steps` flux-form updates around a periodic ring with uniform Courant number.
function advect_ring!(rm, m, courant, steps, scheme)
    Nx = Int32(size(rm, 1))
    F = courant * m[1, 1, 1]
    flux = similar(rm, Nx + 1)
    for _ in 1:steps
        for f in 1:Nx + 1
            flux[f] = _xface_tracer_flux(Int32(f), 1, 1, rm, m, F, scheme, Nx)
        end
        for i in 1:Nx
            rm[i, 1, 1] += flux[i] - flux[i + 1]
        end
    end
    return rm
end

sine_means(Nx) = [(cos(2π * (i - 1) / Nx) - cos(2π * i / Nx)) * Nx / 2π for i in 1:Nx]

function sine_error(scheme, Nx; courant = 0.4)
    m = ones(Float64, Nx, 1, 1)
    exact = sine_means(Nx)
    rm = reshape(copy(exact), Nx, 1, 1)
    advect_ring!(rm, m, courant, round(Int, Nx / courant), scheme)
    return sqrt(sum(abs2, vec(rm) .- exact) / Nx)
end

@testset "PPMScheme(Hord12Limiter())" begin
    @testset "hand-computed FV3 hord = 12 profiles" begin
        # q = (0, 2, 3, 0, 0), cell 3. dm = (3/4, 0, 0): cell 3 is a maximum,
        # cell 4 a minimum. al_i = 5/2 + (3/4 - 0)/3 = 11/4, al_{i+1} = 3/2, so
        # bl = -1/4, br = -3/2. |3(bl + br)| = 21/4 > |bl - br| = 5/4 applies the
        # constraint with dq = (4, 2, -6, 0): br = min(max(0, 2, -1),
        # max(-3/2, min(0, 2, -1))) = -1, bl = min(max(0, 6, 6), max(-1/4, 0)) = 0.
        @test hord12_profile((0.0, 2.0, 3.0, 0.0, 0.0)) === (3.0, 2.0)
        # FV3 face value for Courant 1/2 out of that cell: 3 + (1/2)(-1 - (1/2)(-1)).
        rm = reshape([0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0], 8, 1, 1)
        @test _xface_tracer_flux(Int32(4), 1, 1, rm, ones(8, 1, 1), 0.5, HORD12, Int32(8)) === 0.5 * 2.75
        # q = (0, 0, 1, 4, 4): dm = (0, 1, 0), bl = 1/6 - 1, br = 17/6 - 1; the
        # constraint applies but both bounds hold, so the edges stay 1/6, 17/6.
        q_L, q_R = hord12_profile((0.0, 0.0, 1.0, 4.0, 4.0))
        @test q_L ≈ 1 / 6 rtol = 1e-15
        @test q_R ≈ 17 / 6 rtol = 1e-15
        # A step: every dm is 0 (sum below 1e-25), the cell is flattened.
        @test hord12_profile((0.0, 0.0, 1.0, 1.0, 1.0)) === (1.0, 1.0)
        # Smooth data: no constraint, the edges are the fourth-order ones.
        c = (1.0, 4.0, 9.0, 16.0, 25.0)
        @test all(hord12_profile(c) .≈ (_ppm_edge_value(c[1:4]...), _ppm_edge_value(c[2:5]...)))
    end

    @testset "matches a loop transcription of FV3 xppm" begin
        rng = MersenneTwister(12)
        worst = 0.0
        for trial in 1:500
            n = 24
            q = trial % 3 == 0 ? Float64.(rand(rng, n) .< 0.3) : randn(rng, n)
            courant = (2rand(rng) - 1) * 0.95
            m = ones(n, 1, 1)
            rm = reshape(copy(q), n, 1, 1)
            ours = [_xface_tracer_flux(Int32(f), 1, 1, rm, m, courant, HORD12, Int32(n)) / courant
                    for f in 1:n]
            worst = max(worst, maximum(abs.(ours .- fv3_xppm_hord12(q, fill(courant, n)))))
        end
        # FV3 multiplies by r3 = 1/3 and works with offsets b = q_edge - q;
        # AT divides and passes edge values: rounding only.
        @test worst < 1e-14
    end

    @testset "step and cosine bell on a ring stay in bounds, mass conserved ($FT)" for FT in (Float64, Float32)
        N = 100
        x = ((1:N) .- 0.5) ./ N
        step = FT.(0.2 .< x .< 0.4)
        bell = FT[abs(xi - 0.5) < 0.15 ? (1 + cospi((xi - 0.5) / 0.15)) / 2 : 0 for xi in x]
        m = ones(FT, N, 1, 1)
        for (shape, q0) in ((:step, step), (:bell, bell)), courant in (0.3, 0.7, 1.0, -0.45)
            rm = reshape(copy(q0), N, 1, 1)
            advect_ring!(rm, m, FT(courant), 3 * round(Int, N / abs(courant)), HORD12)
            @test minimum(rm) >= -4eps(FT)
            # Not strictly monotone: a step overshoots by at most ~1e-5.
            @test maximum(rm) <= maximum(q0) + 1e-4
            @test sum(Float64, rm) ≈ sum(Float64, q0) rtol = 16eps(FT)
        end
        # The default PPM undershoots on the same step.
        rm = reshape(copy(step), N, 1, 1)
        advect_ring!(rm, m, FT(0.7), round(Int, N / 0.7), PPMScheme())
        @test minimum(rm) < -0.05
    end

    @testset "smooth profile converges at second order or better" begin
        e80, e160 = sine_error(HORD12, 80), sine_error(HORD12, 160)
        @test e80 / e160 > 3.8
        @test e160 < sine_error(PPMScheme(), 160)
    end

    @testset "uniform mixing ratio stays uniform on non-uniform air mass" begin
        rng = MersenneTwister(7)
        N, q = 9, 4.1e-4
        m = 1 .+ rand(rng, N)
        for F in (0.3, -0.3)
            mx, my, mz = reshape(m, N, 1, 1), reshape(m, 1, N, 1), reshape(m, 1, 1, N)
            for f in 3:N - 1
                @test _xface_tracer_flux(Int32(f), 1, 1, q .* mx, mx, F, HORD12, Int32(N)) ≈ F * q rtol = 1e-14
                @test _yface_tracer_flux(1, Int32(f), 1, q .* my, my, F, HORD12, Int32(N)) ≈ F * q rtol = 1e-14
                @test _zface_tracer_flux(1, 1, Int32(f), q .* mz, mz, F, HORD12, Int32(N)) ≈ F * q rtol = 1e-14
            end
        end
    end

    @testset "x, y and z face fluxes agree on the same column" begin
        rng = MersenneTwister(3)
        N = 16
        m = 1 .+ rand(rng, N)
        rm = m .* randn(rng, N)
        for F in (0.4, -0.4), f in 5:N - 3       # faces whose donors are interior in y and z
            x = _xface_tracer_flux(Int32(f), 1, 1, reshape(rm, N, 1, 1), reshape(m, N, 1, 1), F, HORD12, Int32(N))
            @test _yface_tracer_flux(1, Int32(f), 1, reshape(rm, 1, N, 1), reshape(m, 1, N, 1), F, HORD12, Int32(N)) === x
            @test _zface_tracer_flux(1, 1, Int32(f), reshape(rm, 1, 1, N), reshape(m, 1, 1, N), F, HORD12, Int32(N)) === x
        end
    end

    @testset "TOML limiter key" begin
        spec = advection_spec(Dict("scheme" => "ppm", "limiter" => "hord12"))
        @test materialize(spec, LatLonRuntimeRecipeStyle()) == HORD12
        @test materialize(spec, CubedSphereRuntimeRecipeStyle()) == HORD12
        gchp = advection_spec(Dict("scheme" => "ppm", "limiter" => "hord12", "vertical" => "fv3_kord8"))
        @test materialize(gchp, CubedSphereRuntimeRecipeStyle()) ==
              PPMScheme(Hord12Limiter(); vertical = FV3ScalarProfile())
        @test_throws ArgumentError materialize(gchp, LatLonRuntimeRecipeStyle())
        @test_throws ArgumentError advection_spec(Dict("scheme" => "slopes", "limiter" => "hord12"))
        @test_throws ArgumentError advection_spec(Dict("scheme" => "linrood", "limiter" => "hord12"))
        @test_throws ArgumentError advection_spec(Dict("scheme" => "ppm", "limiter" => "hord_12"))
        @test AtmosTransport.Models.DrivenRunner._advection_label(HORD12) == "PPM, FV3 hord 12"
        @test AtmosTransport.Models.Physics._physics_limiter(Hord12Limiter()) == "hord12"
    end

    @testset "constructors" begin
        @test_throws ArgumentError SlopesScheme(Hord12Limiter())
        @test !(HORD12 isa AtmosTransport.Adjoints.CSAdjointSupportedScheme)
        @test_throws ArgumentError AtmosTransport.Adjoints.require_adjoint_support(HORD12)
    end
end
