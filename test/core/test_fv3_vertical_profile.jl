#!/usr/bin/env julia
# FV3 vertical tracer profile (scalar_profile, kord = 8) in the cubed-sphere
# vertical sweep: the profile matches an independent transcription of
# fv_mapz.F90 (positive-definite iv = 0 and signed iv = 1), the flux integrates
# that profile exactly, column tracer mass changes only by rounding in Float32
# and Float64, and the sweep is less diffusive than the default PPM face flux.

using Test, Random

using AtmosTransport
using .AtmosTransport.Operators: MonotoneLimiter, NoLimiter, SameAsHorizontal, FV3ScalarProfile
using .AtmosTransport.Operators.Advection: CSAdvectionWorkspace, strang_split_cs!,
    strang_split_cs_mt!, needs_column_scratch, _sweep_z_panel!,
    _sweep_z_panel_mt_pingpong!, _sweep_z_panels_mt_pingpong!, _sweep_z_panel_fv3!,
    _fv3_unlimited_edges!, _fv3_layer_profile, _fv3_interface_flux, FV3Column,
    _require_structured_vertical
using .AtmosTransport.State: CubedSphereState, DryBasis

include(joinpath(@__DIR__, "..", "helpers", "fv3_scalar_profile_reference.jl"))

const POSITIVE = FV3ScalarProfile()
const SIGNED   = FV3ScalarProfile(; positive_definite = false)
const FV3_PPM  = PPMScheme(; vertical = POSITIVE)

# Top-down layer air masses with thickness growing ~400× from top to surface,
# like a hybrid L72 column.
column_masses(Nz, FT = Float64) =
    FT.(diff(1e5 .* exp.(range(log(1e-4), 0; length = Nz + 1))))

column(x) = reshape(x, 1, 1, :)

function kernel_profile(q, m, profile = POSITIVE)
    Nz = length(q)
    col = FV3Column(zeros(eltype(q), 1, 1, Nz + 1, 3), 1, 1, 1)
    _fv3_unlimited_edges!(col, column(q .* m), column(m), 1, 1, Int32(Nz))
    return [_fv3_layer_profile(col, Int32(k), Int32(Nz), profile) for k in 1:Nz]
end

# One vertical sweep of a single column with the production kernel.
function kernel_sweep(q, m, cm, profile = POSITIVE)
    Nz, FT = length(q), eltype(q)
    rm = reshape(q .* m, 1, 1, Nz, 1)
    rm_out, m_out = similar(rm), similar(column(m))
    scratch = zeros(FT, 1, 1, Nz + 1, 3)
    _sweep_z_panel_fv3!(rm_out, m_out, rm, column(m), column(cm), scratch, profile, 1, 0, Nz, 1)
    return vec(rm_out), vec(m_out)
end
signed_sweep(q, m, cm) = kernel_sweep(q, m, cm, SIGNED)

function default_ppm_sweep(q, m, cm)
    Nz = length(q)
    rm = reshape(q .* m, 1, 1, Nz, 1)
    rm_out, m_out = similar(rm), similar(column(m))
    _sweep_z_panel_mt_pingpong!(rm_out, m_out, rm, column(m), column(cm), PPMScheme(), 1, 0, Nz, 1)
    return vec(rm_out), vec(m_out)
end

random_cm(rng, m, c) = [zero(eltype(m)); c .* (2 .* rand(rng, eltype(m), length(m) - 1) .- 1) .*
                        min.(m[1:end-1], m[2:end]); zero(eltype(m))]

# Minimum of q(s) = q_L + s[(q_R − q_L) + q_6 (1 − s)] over 0 ≤ s ≤ 1.
function parabola_min((q_L, q_R, q_6))
    s_star = q_6 == 0 ? 0.0 : clamp(((q_R - q_L) + q_6) / (2q_6), 0.0, 1.0)
    return minimum(s -> q_L + s * ((q_R - q_L) + q_6 * (1 - s)), (0.0, 1.0, s_star))
end

test_profiles(Nz) = (
    smooth  = [400 + 20exp(-((k - Nz / 2) / 6)^2) for k in 1:Nz],
    step    = [k < Nz ÷ 3 ? 1.0 : 5.0 for k in 1:Nz],
    spike   = [k == Nz ÷ 2 ? 10.0 : 0.1 for k in 1:Nz],
    zeros   = [max(0.0, sin(k / 3)) for k in 1:Nz],
    noisy   = 1 .+ rand(MersenneTwister(7), Nz),
    surface = [exp(-(Nz - k) / 3) for k in 1:Nz],
)

@testset "FV3 vertical profile (kord = 8)" begin
    Nz = 40
    m = column_masses(Nz)

    @testset "profile matches fv_mapz.F90 transcription: $name" for (name, q) in pairs(test_profiles(Nz))
        qbar = (q .* m) ./ m                      # the layer means the kernel sees
        a4 = fv3_scalar_profile_kord8(qbar, m; iv = 0)
        profile = kernel_profile(q, m)
        @test all(k -> profile[k] == (a4[2, k], a4[3, k], a4[4, k]), 1:Nz)
        @test all(p -> parabola_min(p) >= -8eps(maximum(q)), profile)   # non-negative inside each layer

        signed = (q .- 0.6maximum(q)) .* m ./ m   # crosses zero
        a4s = fv3_scalar_profile_kord8(signed, m; iv = 1)
        @test all(k -> kernel_profile(signed, m, SIGNED)[k] == (a4s[2, k], a4s[3, k], a4s[4, k]), 1:Nz)
    end

    @testset "flux integrates the donor parabola over the swept fraction" begin
        rng = MersenneTwister(11)
        for q in test_profiles(Nz)
            cm = random_cm(rng, m, 0.45)
            a4 = fv3_scalar_profile_kord8((q .* m) ./ m, m)
            f = zeros(Nz + 1)
            for e in 2:Nz
                F = cm[e]
                f[e] = F >= 0 ? F * fv3_parabola_mean(a4, e - 1, 1 - F / m[e - 1], 1.0) :
                                F * fv3_parabola_mean(a4, e, 0.0, -F / m[e])
            end
            rm_new, m_new = kernel_sweep(q, m, cm)
            @test rm_new ≈ q .* m .+ f[1:Nz] .- f[2:Nz+1] rtol = 1e-12
            @test m_new == m .+ cm[1:Nz] .- cm[2:Nz+1]
        end
    end

    @testset "flux form equals FV3's conservative remap onto shifted interfaces" begin
        # In cumulative air mass, a sweep with interface fluxes F moves interface
        # e from M_e to M_e − F_e; with each flux below its donor's mass and no
        # flux through the top or surface, the flux-form update is the remap
        # onto that grid (as in `mapn_tracer`), computed here by overlap
        # integration. The signed profile is tested on fields crossing zero.
        rng = MersenneTwister(12)
        for q in test_profiles(Nz), (profile, iv, offset) in ((POSITIVE, 0, 0.0), (SIGNED, 1, 0.6))
            q0 = q .- offset * maximum(q)
            cm = random_cm(rng, m, 0.49)
            pe1 = [0.0; cumsum(m)]
            qbar = (q0 .* m) ./ m                 # the layer means the kernel sees
            rm_new, _ = kernel_sweep(q0, m, cm, profile)
            reference = fv3_remap_content(pe1, qbar, pe1 .- cm; iv)
            @test all(isapprox.(rm_new, reference; rtol = 1e-12, atol = 1e-12 * maximum(abs, reference)))
        end
    end

    @testset "positivity limiter multiplies by a rounded 1/12, as FV3 does" begin
        # Float32 case where q₆/12 and q₆·(1/12) fall on opposite sides of the
        # threshold; FV3 flattens this parabola.
        q, q_L, q_6 = 1.7074982f0, 5.1224947f0, -20.489979f0
        @test q + q_6 * (1f0 / 12) < 0 <= q + q_6 / 12
        @test AtmosTransport.Operators.Advection._fv3_positive_limit(q, q_L, q_L, q_6) == (q, q, 0f0)
    end

    @testset "column mass, uniform field and positivity ($FT)" for FT in (Float32, Float64)
        rng = MersenneTwister(3)
        mF = column_masses(Nz, FT)
        for q in test_profiles(Nz)
            qF = FT.(q)
            rm_new, m_new = kernel_sweep(qF, mF, random_cm(rng, mF, FT(0.45)))
            @test eltype(rm_new) === FT
            rm0 = qF .* mF
            # Each interface flux enters both layers with opposite sign; the only
            # change in the column total is the rounding of the cell updates.
            @test abs(sum(Float64, rm_new) - sum(Float64, rm0)) <= 4Nz * eps(FT) * maximum(rm0)
            @test all(>=(0), rm_new)
        end
        c = FT(411.7)
        rm_new, m_new = kernel_sweep(fill(c, Nz), mF, random_cm(rng, mF, FT(0.4)))
        @test maximum(abs.(rm_new ./ m_new .- c)) <= 8eps(c)
    end

    @testset "signed profile is symmetric under q → −q" begin
        # FV3 resolves exactly equal neighbouring means (δ = 0) in the
        # large-scale constraint with its local-minimum branch whatever the
        # sign, so the symmetry is tested on profiles without such ties.
        rng = MersenneTwister(4)
        tie_free = (test_profiles(Nz).smooth, test_profiles(Nz).noisy,
                    test_profiles(Nz).surface, randn(rng, Nz), randn(rng, Nz))
        for q in tie_free
            q0 = q .- 0.6maximum(q)
            cm = random_cm(rng, m, 0.45)
            @test signed_sweep(-q0, m, cm)[1] == -signed_sweep(q0, m, cm)[1]
        end
    end

    @testset "flux is capped at the donor content beyond Courant 1" begin
        # A CFL violation: twice the top layer's air mass leaves through its base.
        m3 = [1.0, 4.0, 4.0, 4.0, 4.0]
        cm = [0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
        q = [1.0, 0.5, 0.5, 0.5, 0.5]
        for profile in (POSITIVE, SIGNED)
            rm_new, _ = kernel_sweep(q, m3, cm, profile)
            @test abs(rm_new[1]) <= 4eps()            # the whole content leaves, no more
            @test sum(rm_new) ≈ sum(q .* m3)
        end
    end

    @testset "type-stable in Float32" begin
        m32 = column_masses(Nz, Float32)
        col = FV3Column(zeros(Float32, 1, 1, Nz + 1, 3), 1, 1, 1)
        _fv3_unlimited_edges!(col, column(fill(400f0, Nz) .* m32), column(m32), 1, 1, Int32(Nz))
        for k in (1, 2, 5, Nz - 1, Nz), profile in (POSITIVE, SIGNED)
            @test (@inferred _fv3_layer_profile(col, Int32(k), Int32(Nz), profile)) isa
                  NTuple{3, Float32}
        end
        p = (400f0, 401f0, 0.5f0)
        @test (@inferred _fv3_interface_flux(0.1f0, p, 1f0, p, 2f0)) isa Float32
        @test (@inferred _fv3_interface_flux(-0.1f0, p, 1f0, p, 2f0)) isa Float32
    end

    @testset "less numerical diffusion than the default PPM face flux" begin
        # Translate a Gaussian bump (σ = 6 layers) 30 layers downward at Courant
        # 0.05. Massive end layers act as reservoirs, so the interior flow is a
        # pure translation and the exact answer is the shifted bump. The signed
        # profile must do as well on a negative bump.
        Nz = 120
        m = ones(Nz); m[1] = m[end] = 1e6
        bump(center) = [exp(-((k - center) / 6)^2) for k in 1:Nz]
        cm = [0.0; fill(0.05, Nz - 1); 0.0]
        interior = 20:Nz-20
        function translation_error(sweep, sign)
            q, mm = sign .* (1 .+ bump(40)), copy(m)
            for _ in 1:600
                rm_new, mm = sweep(q, mm, cm)
                q = rm_new ./ mm
            end
            sqrt(sum(abs2, q[interior] .- sign .* (1 .+ bump(70))[interior]) / length(interior))
        end
        error_default = translation_error(default_ppm_sweep, 1)
        @test translation_error(kernel_sweep, 1) < 0.1 * error_default
        @test translation_error(signed_sweep, 1) < 0.1 * error_default
        @test translation_error(signed_sweep, -1) < 0.1 * error_default
    end

    @testset "cubed-sphere split sweeps" begin
        Nc, Hp, Nz, Nt = 4, 3, 12, 2
        for FT in (Float32, Float64)
            mesh = CubedSphereMesh(; Nc, Hp, FT)
            N = Nc + 2Hp
            rng = MersenneTwister(5)
            mcol = column_masses(Nz, FT)
            m = ntuple(_ -> repeat(column(mcol), N, N, 1) .* (1 .+ FT(0.1) .* rand(rng, FT, N, N, 1)), 6)
            rm = ntuple(p -> m[p] .* (1 .+ rand(rng, FT, N, N, Nz)), 6)
            rm2 = ntuple(p -> m[p] .* FT(411), 6)
            am = ntuple(_ -> zeros(FT, N + 1, N, Nz), 6)
            bm = ntuple(_ -> zeros(FT, N, N + 1, Nz), 6)
            cm = ntuple(6) do p
                c = zeros(FT, N, N, Nz + 1)
                for k in 2:Nz
                    c[:, :, k] .= FT(0.2) .* (2 .* rand(rng, FT, N, N) .- 1) .*
                                  min.(m[p][:, :, k - 1], m[p][:, :, k])
                end
                c
            end
            rm_4d = ntuple(p -> cat(rm[p], rm2[p]; dims = 4), 6)
            I = Hp+1:Hp+Nc
            interior(a) = sum(Float64, view(a, I, I, ntuple(_ -> Colon(), ndims(a) - 2)...))
            total(panels) = sum(interior, panels)
            ws = CSAdvectionWorkspace(mesh, m[1]; n_tracers = Nt, column_scratch = true)

            # One six-panel sweep equals the 1-D kernel column by column, so the
            # halo offset and the `cm` column are the right ones.
            rm_out, m_out = map(similar, rm_4d), map(similar, m)
            _sweep_z_panels_mt_pingpong!(rm_out, m_out, rm_4d, m, cm, mesh, FV3_PPM, ws)
            @test all(Iterators.product(1:6, I, I)) do (p, i, j)
                q = rm_4d[p][i, j, :, 1] ./ m[p][i, j, :]
                rm_col, m_col = kernel_sweep(q, m[p][i, j, :], cm[p][i, j, :])
                rm_out[p][i, j, :, 1] ≈ rm_col && m_out[p][i, j, :] == m_col
            end

            m_mt, rm_mt = map(copy, m), map(copy, rm_4d)
            mass0 = total(rm_mt)
            strang_split_cs_mt!(rm_mt, m_mt, am, bm, cm, mesh, FV3_PPM, ws)
            @test abs(total(rm_mt) - mass0) <= 100eps(FT) * mass0
            q2 = [rm_mt[p][i, j, k, 2] / m_mt[p][i, j, k] for p in 1:6, i in I, j in I, k in 1:Nz]
            @test maximum(abs.(q2 .- 411)) <= 16eps(FT(411))

            # The single-tracer path runs the same kernel.
            ws1 = CSAdvectionWorkspace(mesh, m[1]; column_scratch = true)
            m_1, rm_1 = map(copy, m), map(copy, rm)
            strang_split_cs!(rm_1, m_1, am, bm, cm, mesh, FV3_PPM, ws1)
            @test all(p -> view(rm_1[p], I, I, :) == view(rm_mt[p], I, I, :, 1), 1:6)

            # Without column scratch the sweep refuses to run.
            ws0 = CSAdvectionWorkspace(mesh, m[1]; n_tracers = Nt)
            @test_throws ArgumentError strang_split_cs_mt!(map(copy, rm_4d), map(copy, m),
                                                           am, bm, cm, mesh, FV3_PPM, ws0)
        end
    end

    @testset "TransportModel builds the column scratch and runs the FV3 sweep" begin
        FT, Nc, Hp, Nz = Float64, 4, 3, 8
        N = Nc + 2Hp
        mesh = CubedSphereMesh(; Nc, Hp, FT)
        vertical = HybridSigmaPressure(collect(FT, range(0, 1e4; length = Nz + 1)), zeros(FT, Nz + 1))
        grid = AtmosGrid(mesh, vertical, AtmosTransport.CPU(); FT)
        mcol = column_masses(Nz)
        panels_m = ntuple(_ -> repeat(column(mcol), N, N, 1), 6)
        panels_rm = ntuple(p -> panels_m[p] .* (400e-6 .+ 1e-5 .* sin.(reshape(1:Nz, 1, 1, :))), 6)
        state = CubedSphereState(DryBasis, mesh, panels_m; CO2 = panels_rm)
        fluxes = allocate_face_fluxes(mesh, Nz; FT, basis = DryBasis)
        for p in 1:6, k in 2:Nz
            fluxes.cm[p][:, :, k] .= 0.2 * min(mcol[k - 1], mcol[k]) * (-1)^k
        end
        model = TransportModel(state, fluxes, grid, FV3_PPM)
        ws = model.workspace.advection_ws
        @test size(ws.column_scratch) == (Nc, Nc, Nz + 1, 3)
        mass0 = sum(p -> sum(view(state.tracers.CO2[p], Hp+1:Hp+Nc, Hp+1:Hp+Nc, :)), 1:6)
        q_before = state.tracers.CO2[1][Hp+1, Hp+1, :] ./ state.air_mass[1][Hp+1, Hp+1, :]
        apply!(state, fluxes, grid, FV3_PPM, FT(1); workspace = ws)
        mass1 = sum(p -> sum(view(state.tracers.CO2[p], Hp+1:Hp+Nc, Hp+1:Hp+Nc, :)), 1:6)
        @test mass1 ≈ mass0 rtol = 1e-13
        @test state.tracers.CO2[1][Hp+1, Hp+1, :] ./ state.air_mass[1][Hp+1, Hp+1, :] != q_before
    end

    @testset "Lin-Rood horizontal with the FV3 vertical profile" begin
        # With no horizontal fluxes, Lin-Rood (H Z Z H) and the split PPM path
        # (X Y Z Z Y X) both reduce to two FV3 vertical sweeps; they differ only
        # by Lin-Rood's mixing-ratio round trip in the horizontal step.
        FT, Nc, Hp, Nz = Float64, 4, 3, 8
        N = Nc + 2Hp
        mesh = CubedSphereMesh(; Nc, Hp, FT)
        vertical = HybridSigmaPressure(collect(FT, range(0, 1e4; length = Nz + 1)), zeros(FT, Nz + 1))
        grid = AtmosGrid(mesh, vertical, AtmosTransport.CPU(); FT)
        mcol = column_masses(Nz)
        function run_once(scheme)
            panels_m = ntuple(_ -> repeat(column(mcol), N, N, 1), 6)
            panels_rm = ntuple(p -> panels_m[p] .* (400e-6 .+ 1e-5 .* sin.(reshape(1:Nz, 1, 1, :) .+ p)), 6)
            state = CubedSphereState(DryBasis, mesh, panels_m; CO2 = panels_rm)
            fluxes = allocate_face_fluxes(mesh, Nz; FT, basis = DryBasis)
            for p in 1:6, k in 2:Nz
                fluxes.cm[p][:, :, k] .= 0.2 * min(mcol[k - 1], mcol[k]) * (-1)^k
            end
            model = TransportModel(state, fluxes, grid, scheme)
            apply!(state, fluxes, grid, scheme, FT(1); workspace = model.workspace.advection_ws)
            return state, model.workspace.advection_ws
        end
        linrood = LinRoodPPMScheme(7; vertical = FV3_PPM)
        state_lr, ws_lr = run_once(linrood)
        state_pp, _ = run_once(FV3_PPM)
        @test size(ws_lr.cs.column_scratch, 4) >= 3
        @test all(p -> isapprox(state_lr.tracers.CO2[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :],
                                state_pp.tracers.CO2[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]; rtol = 1e-12), 1:6)
        @test LinRoodPPMScheme(7).vertical isa UpwindScheme
        @test !needs_column_scratch(LinRoodPPMScheme(7)) && needs_column_scratch(linrood)
        @test_throws TypeError LinRoodPPMScheme(7; vertical = PPMScheme())   # not a LinRoodVertical
        @test !(linrood isa AtmosTransport.Adjoints.CSAdjointSupportedScheme)
        @test LinRoodPPMScheme(5) isa AtmosTransport.Adjoints.CSAdjointSupportedScheme
    end

    @testset "dispatch and defaults" begin
        @test PPMScheme() isa PPMScheme{MonotoneLimiter, SameAsHorizontal}
        @test PPMScheme(NoLimiter()) isa PPMScheme{NoLimiter, SameAsHorizontal}
        @test FV3_PPM isa PPMScheme{MonotoneLimiter, FV3ScalarProfile{true}}
        @test SIGNED isa FV3ScalarProfile{false}
        @test needs_column_scratch(FV3_PPM)
        @test needs_column_scratch(PPMScheme(; vertical = SIGNED))
        @test !needs_column_scratch(PPMScheme())
        @test !needs_column_scratch(UpwindScheme())
        a = zeros(10, 10, 8)
        @test_throws ArgumentError _sweep_z_panel!(a, a, zeros(10, 10, 9), FV3_PPM, a, a, 4, 3, 8)
        # structured (lat-lon) sweeps refuse the FV3 profile
        @test _require_structured_vertical(PPMScheme()) === nothing
        @test_throws ArgumentError _require_structured_vertical(FV3_PPM)
        # no adjoint for the FV3 vertical profile yet
        @test PPMScheme() isa AtmosTransport.Adjoints.CSAdjointSupportedScheme
        @test !(FV3_PPM isa AtmosTransport.Adjoints.CSAdjointSupportedScheme)
    end
end
