#!/usr/bin/env julia

using Test

import AtmosTransport
const AT = AtmosTransport
const Adv = AtmosTransport.Operators.Advection
const TapeMod = AtmosTransport.Tape

# Full/Stride/Revolve parity for objective and seeded paths. Physics combinations
# are covered by the single-sweep and production-step parity matrices.

function _stride_problem(; Nc = 4, Nz = 3, nsteps = 6, FT = Float64)
    mesh = AT.CubedSphereMesh(Nc = Nc, Hp = 3, FT = FT)
    N = mesh.Nc + 2mesh.Hp
    Hp = mesh.Hp

    panels_m = ntuple(6) do p
        m = zeros(FT, N, N, Nz)
        @inbounds for k in 1:Nz, j in 1:N, i in 1:N
            m[i, j, k] = FT(2.0 + 0.25k + 0.01p + 0.0001(i + j))
        end
        m
    end
    panels_rm = ntuple(_ -> zeros(FT, N, N, Nz), 6)
    Adv.fill_panel_halos!(panels_m, mesh; dir = 0)
    Adv.fill_panel_halos!(panels_rm, mesh; dir = 0)

    panels_am_steps = Vector{Any}(undef, nsteps)
    panels_bm_steps = Vector{Any}(undef, nsteps)
    panels_cm_steps = Vector{Any}(undef, nsteps)
    for step in 1:nsteps
        panels_am_steps[step] = ntuple(6) do p
            am = zeros(FT, N + 1, N, Nz)
            @inbounds for k in 1:Nz, j in Hp + 1:Hp + mesh.Nc, i in Hp + 1:Hp + mesh.Nc + 1
                am[i, j, k] = FT(0.013) * sin(FT(0.21step + 0.31p + 0.71i + 0.41j + 0.23k))
            end
            am
        end
        panels_bm_steps[step] = ntuple(6) do p
            bm = zeros(FT, N, N + 1, Nz)
            @inbounds for k in 1:Nz, j in Hp + 1:Hp + mesh.Nc + 1, i in Hp + 1:Hp + mesh.Nc
                bm[i, j, k] = FT(0.011) * cos(FT(0.33step + 0.51p + 0.41i + 0.61j + 0.13k))
            end
            bm
        end
        panels_cm_steps[step] = ntuple(6) do p
            cm = zeros(FT, N, N, Nz + 1)
            @inbounds for k in 2:Nz, j in Hp + 1:Hp + mesh.Nc, i in Hp + 1:Hp + mesh.Nc
                cm[i, j, k] = -FT(0.009) * (one(FT) + FT(0.1) * sin(FT(i + j + k + p + step)))
            end
            cm
        end
    end
    return mesh, panels_m, panels_rm, panels_am_steps, panels_bm_steps, panels_cm_steps
end

# Pick a single column on panel 1, midway into the active mesh so it's a
# stable interior cell. Objective indices are 1:Nc (not haloed).
_column_mean_objective(mesh::AT.CubedSphereMesh) =
    AT.CSColumnMeanObjective(1, max(2, mesh.Nc ÷ 2), max(2, mesh.Nc ÷ 2))

# Stride, Revolve and Full call the same forward / reverse kernels in
# the same order on the same inputs. Every recorded step starts with
# production's step-start halo refresh (`fill_cs_step_start_halos!`),
# so a window that restarts at a step boundary records the same states
# as the full tape. What remains is rounding: the reverse walk sums the
# halo adjoints with atomic adds, whose order can differ between runs
# with several threads (~Float64 epsilon). `atol = 1e-12` (~10⁴ ×
# Float64 epsilon) is well below the FD-identity tolerances in this
# suite (1e-9 relative or looser).
function _footprints_equal(a, b)
    length(a.footprints) == length(b.footprints) || return false
    for step in eachindex(a.footprints)
        for p in 1:6
            isapprox(a.footprints[step][p], b.footprints[step][p];
                     atol = 1e-12, rtol = 1e-10) || return false
        end
    end
    return true
end

function _dot_footprint(result, rates)
    total = zero(eltype(rates[1][1]))
    for step in eachindex(result.footprints), p in 1:6
        total += sum(result.footprints[step][p] .* rates[step][p])
    end
    return total
end

_scaled_rates(rates, scale) =
    [ntuple(p -> scale .* rates[step][p], 6) for step in eachindex(rates)]

@testset "checkpoint_window_count / checkpoint_window_range" begin
    @test TapeMod.checkpoint_window_count(TapeMod.FullCheckpoint(), 10) == 1
    @test TapeMod.checkpoint_window_range(TapeMod.FullCheckpoint(), 1, 10) == 1:10

    s2 = TapeMod.StrideCheckpoint(2)
    @test TapeMod.checkpoint_window_count(s2, 6) == 3
    @test TapeMod.checkpoint_window_range(s2, 1, 6) == 1:2
    @test TapeMod.checkpoint_window_range(s2, 2, 6) == 3:4
    @test TapeMod.checkpoint_window_range(s2, 3, 6) == 5:6

    s3 = TapeMod.StrideCheckpoint(3)
    @test TapeMod.checkpoint_window_count(s3, 7) == 3
    @test TapeMod.checkpoint_window_range(s3, 3, 7) == 7:7  # tail is short

    s_big = TapeMod.StrideCheckpoint(100)
    @test TapeMod.checkpoint_window_count(s_big, 7) == 1
    @test TapeMod.checkpoint_window_range(s_big, 1, 7) == 1:7

    @test_throws ArgumentError TapeMod.StrideCheckpoint(0)
    @test_throws ArgumentError TapeMod.StrideCheckpoint(-3)
    @test_throws BoundsError TapeMod.checkpoint_window_range(s2, 4, 6)
    @test_throws BoundsError TapeMod.checkpoint_window_range(s2, 0, 6)
end

@testset "checkpoint parity across scheme, storage and seed paths" begin
    mesh, m, rm, am, bm, cm = _stride_problem(; Nz = 4)
    N, Nz, Hp = size(rm[1], 1), size(rm[1], 3), mesh.Hp
    for p in 1:6, k in 1:Nz, j in 1:N, i in 1:N
        rm[p][i, j, k] = m[p][i, j, k] *
            (0.06 + 0.011sin(0.27i + 0.13j + 0.19k + 0.07p))
    end
    Adv.fill_panel_halos!(rm, mesh; dir = 0)
    seed = ntuple(p -> [7e-4sin(0.31i + 0.17j + 0.13k + 0.25p)
                       for i in 1:N, j in 1:N, k in 1:Nz], 6)
    objective = _column_mean_objective(mesh)
    schemes = (AT.PPMScheme(AT.NoLimiter()), AT.PPMScheme(AT.MonotoneLimiter()),
               AT.LinRoodPPMScheme(5), AT.LinRoodPPMScheme(7))
    @testset "$(typeof(scheme)), seeded=$seeded" for scheme in schemes, seeded in (false, true)
        run(; kwargs...) = seeded ?
            AT.cs_surface_emission_footprint_from_seed(seed, m, am, bm, cm, mesh;
                scheme, dt = 1.0, base_panels_rm0 = rm, kwargs...) :
            AT.cs_surface_emission_footprint(rm, m, am, bm, cm, mesh, objective;
                scheme, dt = 1.0, kwargs...)
        full = run()
        @test _footprints_equal(full, run(; checkpoint = AT.FullCheckpoint()))
        storages = scheme isa AT.LinRoodPPMScheme ? (:device,) : (:device, :mmap)
        for checkpoint in (AT.StrideCheckpoint.((1, 2, 3, 5, 6, 12))..., AT.RevolveCheckpoint()),
            tape_storage in storages
            result = run(; checkpoint, tape_storage)
            @test _footprints_equal(full, result)
            @test all(p -> isapprox(full.initial_adjoint[p], result.initial_adjoint[p];
                                    atol = 1e-12, rtol = 1e-10), 1:6)
        end
    end
end

@testset "LinRood non-unit flux_scale — FD and checkpoint parity" begin
    mesh, panels_m, panels_rm, am_steps, bm_steps, cm_steps = _stride_problem(
        ; Nc = 4, Nz = 4, nsteps = 2, FT = Float64)
    FT = eltype(panels_m[1])
    N = mesh.Nc + 2mesh.Hp
    Nz = size(panels_m[1], 3)
    for p in 1:6
        @inbounds for k in 1:Nz, j in 1:N, i in 1:N
            panels_rm[p][i, j, k] = panels_m[p][i, j, k] *
                (FT(0.05) + FT(0.009) * sin(FT(0.29i + 0.17j + 0.21k + 0.08p)))
        end
    end
    Adv.fill_panel_halos!(panels_rm, mesh; dir = 0)

    objective = _column_mean_objective(mesh)
    rates = [ntuple(6) do p
        [sin(0.23step + 0.19p + 0.13i - 0.17j)
         for i in 1:mesh.Nc, j in 1:mesh.Nc]
    end for step in eachindex(am_steps)]
    flux_scale = 1.7
    epsilon = 2e-6

    for scheme in (AT.LinRoodPPMScheme(5), AT.LinRoodPPMScheme(7))
        full = AT.cs_surface_emission_footprint(
            panels_rm, panels_m, am_steps, bm_steps, cm_steps, mesh, objective;
            scheme, dt = 1.0, flux_scale)
        j_plus = AT.run_cs_footprint_forward(
            panels_rm, panels_m, am_steps, bm_steps, cm_steps, mesh, objective;
            scheme, dt = 1.0, flux_scale,
            emission_rates = _scaled_rates(rates, epsilon))
        j_minus = AT.run_cs_footprint_forward(
            panels_rm, panels_m, am_steps, bm_steps, cm_steps, mesh, objective;
            scheme, dt = 1.0, flux_scale,
            emission_rates = _scaled_rates(rates, -epsilon))
        finite_difference = (j_plus - j_minus) / (2epsilon)
        # Smooth base field, so the Lin-Rood limiter keeps its branch and the
        # identity holds to ~3e-11. Before issue #10 was fixed the Z sweeps
        # staged post-sweep air mass and this case was off by 7.1e-8.
        @test _dot_footprint(full, rates) ≈ finite_difference rtol = 1e-9

        stride = AT.cs_surface_emission_footprint(
            panels_rm, panels_m, am_steps, bm_steps, cm_steps, mesh, objective;
            scheme, dt = 1.0, flux_scale,
            checkpoint = AT.StrideCheckpoint(1))
        revolve = AT.cs_surface_emission_footprint(
            panels_rm, panels_m, am_steps, bm_steps, cm_steps, mesh, objective;
            scheme, dt = 1.0, flux_scale,
            checkpoint = AT.RevolveCheckpoint())
        @test _footprints_equal(full, stride)
        @test _footprints_equal(full, revolve)
    end
end

@testset "from-seed stride nonlinear PPM with base_panels_rm0 = nothing (zero fallback)" begin
    # Reviewer M1: cover the `base_panels_rm0 === nothing ?
    # _zero_panel_tuple_like(panels_m0) : base_panels_rm0` branch in
    # FootprintAPI.jl. Both FullCheckpoint and stride paths take that
    # fallback; a future refactor that swapped the branches would
    # silently preserve stride==full parity but give the wrong
    # adjoint vs the documented contract.
    mesh, panels_m, panels_rm, am_steps, bm_steps, cm_steps = _stride_problem(
        ; Nc = 4, Nz = 3, nsteps = 6, FT = Float64)
    FT = eltype(panels_m[1])
    final_adj = ntuple(p -> begin
        a = similar(panels_rm[p])
        @inbounds for k in axes(a, 3), j in axes(a, 2), i in axes(a, 1)
            a[i, j, k] = FT(5e-4) * sin(FT(0.19i + 0.23j + 0.13k + 0.27p))
        end
        a
    end, 6)
    scheme = AT.PPMScheme(AT.MonotoneLimiter())

    # base_panels_rm0 omitted → entry defaults to zeros.
    ref = AT.cs_surface_emission_footprint_from_seed(
        final_adj, panels_m, am_steps, bm_steps, cm_steps, mesh;
        scheme = scheme, dt = 1.0)

    explicit_zero = AT.cs_surface_emission_footprint_from_seed(
        final_adj, panels_m, am_steps, bm_steps, cm_steps, mesh;
        scheme, dt = 1.0, base_panels_rm0 = ntuple(p -> zeros(size(panels_rm[p])), 6))
    @test _footprints_equal(ref, explicit_zero)
    @test ref.initial_adjoint == explicit_zero.initial_adjoint

    for K in (2, 3, 6)
        stride = AT.cs_surface_emission_footprint_from_seed(
            final_adj, panels_m, am_steps, bm_steps, cm_steps, mesh;
            scheme = scheme, dt = 1.0,
            checkpoint = AT.StrideCheckpoint(K))
        @test _footprints_equal(ref, stride)
    end
end

@testset "RevolveCheckpoint handles nsteps == 1 (single-step base case)" begin
    # Pathological case: bisection driver immediately hits the leaf.
    # Worth a regression assertion since the recursion stop condition
    # is non-trivial (`hi - lo == 1`).
    mesh, panels_m, panels_rm, am_steps, bm_steps, cm_steps = _stride_problem(
        ; Nc = 4, Nz = 3, nsteps = 1, FT = Float64)
    objective = _column_mean_objective(mesh)

    @testset "linear PPM" begin
        scheme = AT.PPMScheme(AT.NoLimiter())
        ref = AT.cs_surface_emission_footprint(
            panels_rm, panels_m, am_steps, bm_steps, cm_steps, mesh, objective;
            scheme = scheme, dt = 1.0)
        rev = AT.cs_surface_emission_footprint(
            panels_rm, panels_m, am_steps, bm_steps, cm_steps, mesh, objective;
            scheme = scheme, dt = 1.0,
            checkpoint = AT.RevolveCheckpoint())
        @test _footprints_equal(ref, rev)
    end

    @testset "nonlinear PPM" begin
        scheme = AT.PPMScheme(AT.MonotoneLimiter())
        ref = AT.cs_surface_emission_footprint(
            panels_rm, panels_m, am_steps, bm_steps, cm_steps, mesh, objective;
            scheme = scheme, dt = 1.0)
        rev = AT.cs_surface_emission_footprint(
            panels_rm, panels_m, am_steps, bm_steps, cm_steps, mesh, objective;
            scheme = scheme, dt = 1.0,
            checkpoint = AT.RevolveCheckpoint())
        @test _footprints_equal(ref, rev)
    end
end
