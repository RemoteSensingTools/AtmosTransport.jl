#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan 26 P0.C1 — `AbstractCSOptimizer` polymorphic dispatch.
#
# Verifies that:
#   * `CSGradientDescent` is constructible via positional + keyword
#     forms and rejects bad arguments.
#   * `cs_surface_flux_4dvar_solve(opt, cost_fn, controls)` is callable
#     directly with a hand-built closure — the multi-dispatch surface
#     future backends will plug into.
# ---------------------------------------------------------------------------

using Test

import AtmosTransport
const AT = AtmosTransport
const Adv = AtmosTransport.Operators.Advection

function _constant_cs_problem(; Nc=3, Nz=3, nsteps=2, FT=Float64)
    mesh = AT.CubedSphereMesh(Nc=Nc, Hp=3, FT=FT)
    N = mesh.Nc + 2mesh.Hp
    panels_m = ntuple(6) do p
        m = zeros(FT, N, N, Nz)
        for k in 1:Nz, j in 1:N, i in 1:N
            m[i, j, k] = FT(2.0 + 0.25k + 0.01p)
        end
        m
    end
    panels_rm = ntuple(_ -> zeros(FT, N, N, Nz), 6)
    Adv.fill_panel_halos!(panels_m, mesh; dir=0)
    Adv.fill_panel_halos!(panels_rm, mesh; dir=0)
    panels_am = [ntuple(_ -> zeros(FT, N + 1, N, Nz), 6) for _ in 1:nsteps]
    panels_bm = [ntuple(_ -> zeros(FT, N, N + 1, Nz), 6) for _ in 1:nsteps]
    panels_cm = [ntuple(_ -> zeros(FT, N, N, Nz + 1), 6) for _ in 1:nsteps]
    return mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm
end

function _baseline_scenario()
    mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
        _constant_cs_problem(Nc=3, Nz=3, nsteps=2)
    dt = 2.0
    zero_panel = ntuple(_ -> zeros(Float64, mesh.Nc, mesh.Nc), 6)
    control = AT.CSSurfaceFluxControl(
        AT.CSSurfaceFluxWindow(:both_steps, 1:2; normalize=true),
        zero_panel)
    observations = [
        AT.CSObservation(2, AT.CSLayerMeanObjective(1, 2, 2, 3), 0.05, 0.2),
    ]
    return (mesh = mesh,
            panels_rm = panels_rm, panels_m = panels_m,
            panels_am = panels_am, panels_bm = panels_bm, panels_cm = panels_cm,
            dt = dt, control = control, observations = observations)
end

# ---------------------------------------------------------------------------
# CSGradientDescent construction
# ---------------------------------------------------------------------------

@testset "CSGradientDescent — construction" begin
    @test_throws ArgumentError AT.CSGradientDescent(iterations = -1)
    @test_throws ArgumentError AT.CSGradientDescent(initial_step = 0.0)
    @test_throws ArgumentError AT.CSGradientDescent(initial_step = -1.0)
    @test_throws ArgumentError AT.CSGradientDescent(min_step = 0.0)
    @test_throws ArgumentError AT.CSGradientDescent(step_shrink = 0.0)
    @test_throws ArgumentError AT.CSGradientDescent(step_shrink = 1.0)
    @test_throws ArgumentError AT.CSGradientDescent(step_shrink = 1.5)
    @test_throws ArgumentError AT.CSGradientDescent(gradient_tolerance = -1e-6)
end

# ---------------------------------------------------------------------------
# Constant-step descent accepts every candidate.
# ---------------------------------------------------------------------------

@testset "cs_surface_flux_4dvar_solve — non-line-search variant accepts every step" begin
    s = _baseline_scenario()
    cost_fn = function (controls)
        return AT.cs_surface_flux_4dvar(
            s.panels_rm, s.panels_m,
            s.panels_am, s.panels_bm, s.panels_cm,
            s.mesh, s.observations, controls;
            scheme = AT.PPMScheme(AT.NoLimiter()), dt = s.dt)
    end

    opt = AT.CSGradientDescent(iterations = 5, initial_step = 0.05,
                                line_search = false)
    solve = AT.cs_surface_flux_4dvar_solve(opt, cost_fn, s.control)

    # `line_search = false` accepts every candidate, so we should
    # advance exactly `iterations` times (assuming the gradient does
    # not vanish first).
    @test solve.iterations == 5
    @test all(step == 0.05 for step in solve.step_history)
end

# ---------------------------------------------------------------------------
# Regression: an explicit Float64-typed optimizer must not blow up when
# the model is Float32. `CS4DVarSolveResult{FT, A2 <: AbstractArray{FT, 2}}`
# requires the result FT to match the gradient eltype; `solve` derives
# FT from the cost result (not from the optimizer's parametric FT) and
# coerces the optimizer's policy scalars to the cost FT.
# ---------------------------------------------------------------------------

@testset "cs_surface_flux_4dvar_solve — Float32 model with Float64 optimizer" begin
    mesh32, panels_m32, panels_rm32, panels_am32, panels_bm32, panels_cm32 =
        _constant_cs_problem(Nc = 3, Nz = 3, nsteps = 2, FT = Float32)
    dt = 2.0f0
    zero_panel = ntuple(_ -> zeros(Float32, mesh32.Nc, mesh32.Nc), 6)
    control32 = AT.CSSurfaceFluxControl(
        AT.CSSurfaceFluxWindow(:both_steps, 1:2; normalize=true),
        zero_panel)
    observations32 = [
        AT.CSObservation(2, AT.CSLayerMeanObjective(1, 2, 2, 3), 0.05f0, 0.2f0),
    ]

    cost_fn = function (controls)
        return AT.cs_surface_flux_4dvar(
            panels_rm32, panels_m32, panels_am32, panels_bm32, panels_cm32,
            mesh32, observations32, controls;
            scheme = AT.PPMScheme(AT.NoLimiter()), dt = dt)
    end

    # Default kwargs give a Float64-typed CSGradientDescent.
    opt = AT.CSGradientDescent(iterations = 2, initial_step = 0.25)
    @test opt isa AT.CSGradientDescent{Float64}

    solve = AT.cs_surface_flux_4dvar_solve(opt, cost_fn, control32)
    @test solve isa AT.CS4DVarSolveResult
    # History FT is derived from the cost result, not the optimizer.
    @test eltype(solve.cost_history) === Float32
    @test eltype(solve.gradient_norm_history) === Float32
    @test eltype(solve.step_history) === Float32
end

# ---------------------------------------------------------------------------
# Regression: a single scalar preconditioner is broadcast across every
# control. Previously `_preconditioner_vector(scalar)` returned a
# length-1 vector and the strict-equality length check rejected
# multi-control calls — contradicting the docstring's broadcast claim.
# ---------------------------------------------------------------------------

@testset "cs_surface_flux_4dvar — scalar preconditioner broadcasts to N controls" begin
    mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
        _constant_cs_problem(Nc = 3, Nz = 3, nsteps = 3)
    dt = 2.0
    zero_panel = ntuple(_ -> zeros(Float64, mesh.Nc, mesh.Nc), 6)
    x_b = ntuple(_ -> fill(1.0, mesh.Nc, mesh.Nc), 6)
    sigma = ntuple(_ -> fill(0.3, mesh.Nc, mesh.Nc), 6)
    cov = AT.DiagonalCSCovariance(sigma)
    single_prec = AT.CSSurfaceFluxPreconditioner(cov, x_b, AT.LinearOptimType())

    controls = [
        AT.CSSurfaceFluxControl(
            AT.CSSurfaceFluxWindow(:first_step, 1), zero_panel),
        AT.CSSurfaceFluxControl(
            AT.CSSurfaceFluxWindow(:late_window, 2:3; normalize=true),
            zero_panel),
    ]
    observations = [
        AT.CSObservation(1, AT.CSLayerMeanObjective(1, 2, 2, 3), 0.03, 0.4),
        AT.CSObservation(3, AT.CSColumnMeanObjective(1, 2, 2), 0.02, 0.3),
    ]

    # Scalar prec → multi-control. Used to throw; now broadcasts.
    result = AT.cs_surface_flux_4dvar(
        panels_rm, panels_m, panels_am, panels_bm, panels_cm,
        mesh, observations, controls;
        scheme = AT.PPMScheme(AT.NoLimiter()), dt = dt,
        preconditioner = single_prec)
    @test length(result.controls) == 2
    @test length(result.gradients) == 2

    # Equivalent to passing a 2-element vector with the same preconditioner.
    result_vec = AT.cs_surface_flux_4dvar(
        panels_rm, panels_m, panels_am, panels_bm, panels_cm,
        mesh, observations, controls;
        scheme = AT.PPMScheme(AT.NoLimiter()), dt = dt,
        preconditioner = [single_prec, single_prec])
    @test result.cost ≈ result_vec.cost atol = 1e-12 rtol = 1e-12

    # Mismatched-length vector is still rejected.
    @test_throws ArgumentError AT.cs_surface_flux_4dvar(
        panels_rm, panels_m, panels_am, panels_bm, panels_cm,
        mesh, observations, controls;
        scheme = AT.PPMScheme(AT.NoLimiter()), dt = dt,
        preconditioner = [single_prec, single_prec, single_prec])
end
