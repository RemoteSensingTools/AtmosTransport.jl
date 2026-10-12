#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan 26 P0.C2 — `CSLBFGS` (limited-memory BFGS via `Optim.jl`)
# dispatching through the polymorphic
# `cs_surface_flux_4dvar_solve(opt, cost_fn, controls)` surface.
#
# Coverage:
#   * `CSLBFGS` keyword + positional construction + bad-arg rejection.
#   * L-BFGS reduces the cost monotonically and ends at a lower
#     value than the GD baseline for the same iteration count on a
#     small unconditioned 4D-Var problem.
#   * L-BFGS works through the preconditioned cost path (B3) — drives
#     χ-space cost to a lower value than GD with the same budget.
#   * Each control point costs one `cost_fn` call (forward + adjoint),
#     including the initial point and the returned minimizer (#17).
# ---------------------------------------------------------------------------

using Test

import AtmosTransport
const AT = AtmosTransport
const Adv = AtmosTransport.Operators.Advection

# Self-contained constant-flow CS problem (matches the helper in
# `test_cs_optimizer_dispatch.jl`).
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

function _unconditioned_scenario(; Nc = 3, nsteps = 2)
    mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
        _constant_cs_problem(; Nc = Nc, Nz = 3, nsteps = nsteps)
    dt = 2.0
    zero_panel = ntuple(_ -> zeros(Float64, mesh.Nc, mesh.Nc), 6)
    control = AT.CSSurfaceFluxControl(
        AT.CSSurfaceFluxWindow(:both_steps, 1:nsteps; normalize = true),
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
# Construction
# ---------------------------------------------------------------------------

@testset "CSLBFGS — construction" begin
    opt = AT.CSLBFGS()
    @test opt isa AT.AbstractCSOptimizer
    @test opt.iterations == 100
    @test opt.m == 10
    @test opt.show_trace == false

    opt2 = AT.CSLBFGS(iterations = 20, m = 5,
                       gradient_tolerance = 1e-6, show_trace = false)
    @test opt2.iterations == 20
    @test opt2.m == 5
    @test opt2.gradient_tolerance == 1e-6

    @test_throws ArgumentError AT.CSLBFGS(iterations = -1)
    @test_throws ArgumentError AT.CSLBFGS(gradient_tolerance = -1.0)
    @test_throws ArgumentError AT.CSLBFGS(m = 0)
end

@testset "CSLBFGS — gradient_tolerance keeps Optim's L∞ meaning" begin
    values = ntuple(_ -> fill(0.75, 1, 1), 6)
    control = AT.CSSurfaceFluxControl(
        AT.CSSurfaceFluxWindow(:one, 1), values)
    cost_fn = function (controls)
        gradients = ntuple(p -> copy(controls[1].value[p]), 6)
        cost = 0.5 * sum(p -> sum(abs2, controls[1].value[p]), 1:6)
        return AT.CS4DVarResult(
            cost, cost, 0.0, Float64[], Float64[], [gradients],
            Dict{Symbol, typeof(gradients)}(:one => gradients),
            collect(controls), AT.CSObservation[])
    end

    # At the initial point ‖g‖∞ = 0.75 <= 0.8 while
    # ‖g‖₂ = sqrt(6)*0.75 > 0.8. Optim's established L∞ convention
    # must stop before taking a step, and the history/log must report 0.75.
    solve = AT.cs_surface_flux_4dvar_solve(
        AT.CSLBFGS(iterations = 5, gradient_tolerance = 0.8, log = true),
        cost_fn, control)
    @test solve.iterations == 0
    @test solve.gradient_norm_history == [0.75]
    @test [entry.gradient_norm for entry in solve.log] == [0.75]
end

# ---------------------------------------------------------------------------
# L-BFGS vs GD: same budget, L-BFGS ends lower.
#
# Comparison rule (NOTES.md "Phase B/C — preconditioner masks
# optimizer comparison"): both optimizers see the same physical-space
# cost function. We compare final `last.cost` (physical cost), not
# trace counts or χ-space metrics.
# ---------------------------------------------------------------------------

@testset "CSLBFGS — outperforms GD on unconditioned 4D-Var (same iter budget)" begin
    s = _unconditioned_scenario()

    cost_fn = function (controls)
        return AT.cs_surface_flux_4dvar(
            s.panels_rm, s.panels_m, s.panels_am, s.panels_bm, s.panels_cm,
            s.mesh, s.observations, controls;
            scheme = AT.PPMScheme(AT.NoLimiter()), dt = s.dt)
    end

    initial_cost = cost_fn(s.control).cost

    gd_opt = AT.CSGradientDescent(iterations = 8, initial_step = 0.25)
    gd_solve = AT.cs_surface_flux_4dvar_solve(gd_opt, cost_fn, s.control)

    lbfgs_opt = AT.CSLBFGS(iterations = 8, gradient_tolerance = 1e-12)
    lbfgs_solve = AT.cs_surface_flux_4dvar_solve(lbfgs_opt, cost_fn, s.control)

    @test gd_solve isa AT.CS4DVarSolveResult
    @test lbfgs_solve isa AT.CS4DVarSolveResult

    # Both make progress.
    @test gd_solve.last.cost < initial_cost
    @test lbfgs_solve.last.cost < initial_cost

    # L-BFGS's superlinear convergence pays off — it reaches a
    # strictly lower cost than GD on the same iteration budget.
    @test lbfgs_solve.last.cost < gd_solve.last.cost

    # Result structure: cost / gradient-norm histories are populated.
    @test length(lbfgs_solve.cost_history) >= 1
    @test length(lbfgs_solve.gradient_norm_history) == length(lbfgs_solve.cost_history)
    # Optim's L-BFGS produces a monotonically non-increasing cost
    # (line search guarantees descent).
    @test all(diff(lbfgs_solve.cost_history) .<= 1e-12)
end

# ---------------------------------------------------------------------------
# L-BFGS through the preconditioned cost path (B3).
# ---------------------------------------------------------------------------

@testset "CSLBFGS — works through preconditioned 4D-Var" begin
    s = _unconditioned_scenario()

    # Build a Diagonal-cov / Linear preconditioner. With x_b = 0
    # and σ = 1 this is just an identity rescaling — the χ-space
    # cost is essentially the same as the unconditioned one but
    # routed through B3's chain rule.
    sigma = ntuple(_ -> fill(1.0, s.mesh.Nc, s.mesh.Nc), 6)
    cov = AT.DiagonalCSCovariance(sigma)
    bg = ntuple(_ -> zeros(Float64, s.mesh.Nc, s.mesh.Nc), 6)
    prec = AT.CSSurfaceFluxPreconditioner(cov, bg, AT.LinearOptimType())

    cost_fn = function (controls)
        return AT.cs_surface_flux_4dvar(
            s.panels_rm, s.panels_m, s.panels_am, s.panels_bm, s.panels_cm,
            s.mesh, s.observations, controls;
            scheme = AT.PPMScheme(AT.NoLimiter()), dt = s.dt,
            preconditioner = prec)
    end

    initial = cost_fn(s.control)

    lbfgs_opt = AT.CSLBFGS(iterations = 8, gradient_tolerance = 1e-12)
    solve = AT.cs_surface_flux_4dvar_solve(lbfgs_opt, cost_fn, s.control)

    @test solve isa AT.CS4DVarSolveResult
    @test solve.last.cost < initial.cost
end

# ---------------------------------------------------------------------------
# Public entrypoint integration: optimizer kwarg forwards through
# cs_surface_flux_4dvar_optimize.
# ---------------------------------------------------------------------------

@testset "cs_surface_flux_4dvar_optimize — accepts CSLBFGS via kwarg" begin
    s = _unconditioned_scenario()
    solve = AT.cs_surface_flux_4dvar_optimize(
        s.panels_rm, s.panels_m, s.panels_am, s.panels_bm, s.panels_cm,
        s.mesh, s.observations, s.control;
        scheme = AT.PPMScheme(AT.NoLimiter()), dt = s.dt,
        optimizer = AT.CSLBFGS(iterations = 6))
    @test solve isa AT.CS4DVarSolveResult
    @test solve.iterations >= 1
end

# ---------------------------------------------------------------------------
# Issue #17: Optim used to receive separate `f` and `g!` closures that each
# ran `cost_fn` (forward + adjoint), so every point was evaluated twice, and
# the initial probe and the final result were recomputed on top.
# ---------------------------------------------------------------------------

@testset "CSLBFGS — one cost_fn call per distinct control point (#17)" begin
    s = _unconditioned_scenario()
    # An observation far from the prior so L-BFGS takes several steps.
    observations = [AT.CSObservation(2, AT.CSLayerMeanObjective(1, 2, 2, 3), 5.0, 0.2)]
    points = Vector{Vector{Float64}}()
    cost_fn = function (controls)
        push!(points, reduce(vcat, [vec(c.value[p]) for c in controls for p in 1:6]))
        return AT.cs_surface_flux_4dvar(
            s.panels_rm, s.panels_m, s.panels_am, s.panels_bm, s.panels_cm,
            s.mesh, observations, controls;
            scheme = AT.PPMScheme(AT.NoLimiter()), dt = s.dt)
    end

    solve = AT.cs_surface_flux_4dvar_solve(
        AT.CSLBFGS(iterations = 10, gradient_tolerance = 0.0, m = 5), cost_fn, s.control)
    @test solve.iterations >= 2
    # Every evaluated point is distinct: no point is evaluated twice.
    @test length(points) == length(unique(points))
    # The minimizer is the last point evaluated (the final result reuses it).
    xmin = reduce(vcat, [vec(solve.controls[1].value[p]) for p in 1:6])
    @test xmin == points[end]
    # The returned result is the evaluation at the returned controls.
    @test solve.last.cost == solve.cost_history[end]
    @test solve.last.controls[1].value == solve.controls[1].value

    # No iterations: only the initial point is evaluated.
    empty!(points)
    solve0 = AT.cs_surface_flux_4dvar_solve(AT.CSLBFGS(iterations = 0), cost_fn, s.control)
    @test length(points) == 1
    @test solve0.iterations == 0
    @test solve0.last.cost == solve0.cost_history[1]
end
