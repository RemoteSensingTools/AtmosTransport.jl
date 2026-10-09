#!/usr/bin/env julia
#
# `PPMScheme(CW84Limiter())`: the complete Colella–Woodward (1984) PPM, with
# van Leer-limited edges and the parabolic swept-mean flux. Its 1-D update is
# monotone, so it keeps non-negative tracers non-negative, which the default
# `PPMScheme()` does not. The face-flux adjoint (`_ppm_face_coeffs`) is checked
# against finite differences of the forward.

using Test
using Random
using AtmosTransport
using .AtmosTransport.Operators: PPMScheme, SlopesScheme, MonotoneLimiter, CW84Limiter
using .AtmosTransport.Operators.Advection: _xface_tracer_flux, _yface_tracer_flux,
    _zface_tracer_flux, _ppm_edge_value
using .AtmosTransport.Adjoints: _add_x_face_adjoint!, _add_y_face_adjoint!,
    _add_z_face_adjoint!
using .AtmosTransport.Models: advection_spec, materialize, LatLonRuntimeRecipeStyle,
    CubedSphereRuntimeRecipeStyle, PPMAdvectionSpec
using .AtmosTransport.Operators: FV3ScalarProfile

# `steps` flux-form updates around a periodic ring with a uniform mass flux
# F = courant · m (negative courant: flow to the left).
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

# Cell means of sin(2πx) on `Nx` cells of [0, 1].
sine_means(Nx) = [(cos(2π * (i - 1) / Nx) - cos(2π * i / Nx)) * Nx / 2π for i in 1:Nx]

function sine_error(scheme, Nx; courant = 0.4)
    m = ones(Float64, Nx, 1, 1)
    exact = sine_means(Nx)
    rm = reshape(copy(exact), Nx, 1, 1)
    advect_ring!(rm, m, courant, round(Int, Nx / courant), scheme)   # one revolution
    return sqrt(sum(abs2, vec(rm) .- exact) / Nx)
end

@testset "PPMScheme(CW84Limiter())" begin
    cw84 = PPMScheme(CW84Limiter())

    @testset "1-D ring: no undershoot or overshoot, mass conserved ($FT)" for FT in (Float64, Float32)
        Nx = 40
        m = ones(FT, Nx, 1, 1)
        for courant in (0.3, 0.7, 1.0, -0.45), shape in (:spike, :box)
            rm = zeros(FT, Nx, 1, 1)
            shape === :spike ? (rm[10, 1, 1] = 1) : (rm[10:14, 1, 1] .= 1)
            total = sum(Float64, rm)
            advect_ring!(rm, m, FT(courant), 100, cw84)
            @test minimum(rm) >= 0
            @test maximum(rm) <= 1 + 4eps(FT)
            @test sum(Float64, rm) ≈ total rtol = 8eps(FT)
        end
        # The default PPM does undershoot on the same problem.
        rm = zeros(FT, Nx, 1, 1)
        rm[10:14, 1, 1] .= 1
        advect_ring!(rm, m, FT(0.7), 100, PPMScheme())
        @test minimum(rm) < -0.1
    end

    # Divergent and convergent flow on a ring: random face fluxes with each
    # cell's outflow at most its air mass keep the tracer mass non-negative.
    @testset "one sweep stays non-negative while no cell exports more than its mass" begin
        rng = MersenneTwister(1)
        Nx = 40
        for trial in 1:200
            m = reshape(1 .+ rand(rng, Nx), Nx, 1, 1)
            rm = m .* reshape(rand(rng, Nx) .* (rand(rng, Nx) .< 0.3), Nx, 1, 1)
            # |F| ≤ m/2 of both neighbours, so two outflow faces export at most m
            F = [(2rand(rng) - 1) / 2 * min(m[mod1(f - 1, Nx)], m[mod1(f, Nx)]) for f in 1:Nx + 1]
            F[Nx + 1] = F[1]
            flux = [_xface_tracer_flux(Int32(f), 1, 1, rm, m, F[f], cw84, Int32(Nx)) for f in 1:Nx + 1]
            rm_new = [rm[i] + flux[i] - flux[i + 1] for i in 1:Nx]
            @test minimum(rm_new) >= -1e-15
        end
    end

    @testset "smooth profile converges at second order or better" begin
        e80, e160 = sine_error(cw84, 80), sine_error(cw84, 160)
        @test e80 / e160 > 3.8
        @test e160 < sine_error(PPMScheme(), 160)
    end

    @testset "limited edge lies between its neighbours" begin
        rng = MersenneTwister(84)
        for _ in 1:1000
            c = randn(rng, 4)
            e = _ppm_edge_value(c..., CW84Limiter())
            @test min(c[2], c[3]) - 1e-15 <= e <= max(c[2], c[3]) + 1e-15
        end
        # Unlimited slopes give the fourth-order edge of the default PPM.
        c = [1.0, 4.0, 9.0, 16.0]
        @test _ppm_edge_value(c..., CW84Limiter()) ≈ _ppm_edge_value(c...) rtol = 1e-14
    end

    @testset "uniform mixing ratio stays uniform on non-uniform air mass" begin
        rng = MersenneTwister(7)
        N, q = 9, 4.1e-4
        m = 1 .+ rand(rng, N)
        for F in (0.3, -0.3)
            mx, my, mz = reshape(m, N, 1, 1), reshape(m, 1, N, 1), reshape(m, 1, 1, N)
            for f in 3:N - 1
                @test _xface_tracer_flux(Int32(f), 1, 1, q .* mx, mx, F, cw84, Int32(N)) ≈ F * q rtol = 1e-14
                @test _yface_tracer_flux(1, Int32(f), 1, q .* my, my, F, cw84, Int32(N)) ≈ F * q rtol = 1e-14
                @test _zface_tracer_flux(1, 1, Int32(f), q .* mz, mz, F, cw84, Int32(N)) ≈ F * q rtol = 1e-14
            end
        end
    end

    @testset "TOML limiter key" begin
        spec = advection_spec(Dict("scheme" => "ppm", "limiter" => "cw84"))
        @test materialize(spec, LatLonRuntimeRecipeStyle()) isa PPMScheme{CW84Limiter}
        @test materialize(spec, CubedSphereRuntimeRecipeStyle()) isa PPMScheme{CW84Limiter}
        @test materialize(advection_spec(Dict("scheme" => "ppm")), LatLonRuntimeRecipeStyle()) ==
              PPMScheme()
        @test_throws ArgumentError advection_spec(Dict("scheme" => "ppm", "limiter" => "minmod"))
        @test_throws ArgumentError advection_spec(Dict("scheme" => "slopes", "limiter" => "cw84"))
        @test_throws ArgumentError SlopesScheme(CW84Limiter())
        @test PPMAdvectionSpec(FV3ScalarProfile()) == PPMAdvectionSpec(MonotoneLimiter(), FV3ScalarProfile())
        @test AtmosTransport.Models.DrivenRunner._advection_label(cw84) == "PPM, CW84"
        @test cw84 isa AtmosTransport.Adjoints.CSAdjointSupportedScheme
    end
end

# Central differences of one face flux with respect to every cell's tracer mass.
function fd_face_gradient(flux_of, rm; h = 1e-7)
    g = zero(rm)
    for n in eachindex(rm)
        rp = copy(rm); rp[n] += h
        rn = copy(rm); rn[n] -= h
        g[n] = (flux_of(rp) - flux_of(rn)) / 2h
    end
    return g
end

@testset "PPM face adjoint matches finite differences" begin
    rng = MersenneTwister(1984)
    N = 10
    for limiter in (MonotoneLimiter(), CW84Limiter()), trial in 1:40
        scheme = PPMScheme(limiter)
        m = 1 .+ rand(rng, N)
        rm = m .* (1 .+ 0.3 .* randn(rng, N))
        F = (trial % 2 == 0 ? 1 : -1) * 0.9 * rand(rng)
        for (dir, shape) in ((:x, (N, 1, 1)), (:y, (1, N, 1)), (:z, (1, 1, N)))
            M, RM = reshape(m, shape), reshape(rm, shape)
            for f in 2:N
                face = Int32(f)
                flux_of = dir === :x ? (r -> _xface_tracer_flux(face, 1, 1, r, M, F, scheme, Int32(N))) :
                          dir === :y ? (r -> _yface_tracer_flux(1, face, 1, r, M, F, scheme, Int32(N))) :
                                       (r -> _zface_tracer_flux(1, 1, face, r, M, F, scheme, Int32(N)))
                adj = zero(RM)
                dir === :x && _add_x_face_adjoint!(adj, M, RM, face, 1, 1, F, 1.0, scheme, Int32(N))
                dir === :y && _add_y_face_adjoint!(adj, M, RM, 1, face, 1, F, 1.0, scheme, Int32(N))
                dir === :z && _add_z_face_adjoint!(adj, M, RM, 1, 1, face, F, 1.0, scheme, Int32(N))
                @test adj ≈ fd_face_gradient(flux_of, RM) atol = 1e-6
            end
        end
    end
end
