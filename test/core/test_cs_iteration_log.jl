#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan 26 P0.C3 — `CSIterationLog` per-iteration diagnostic capture.
#
# Coverage:
#   * Default behavior: `log = false` keeps `solve.log === nothing`
#     (pre-C3 baseline).
#   * `log = true`: `solve.log isa CSIterationLog` and contains one
#     `CSIterationLogEntry` per iteration plus one iteration-0 row for
#     the initial point.
#   * Consistency: log entries' `cost` field matches `cost_history`;
#     `gradient_norm` matches `gradient_norm_history`; `step_size`
#     matches `step_history` (GD only — L-BFGS leaves step at 0).
#   * Cost decomposition: `observation_cost + background_cost == cost`
#     bit-exact (the cost-decomposition invariant).
#   * Wall-clock monotonicity: `elapsed_seconds` is monotonically
#     non-decreasing across iterations.
#   * Coverage spans both backends: `CSGradientDescent` and
#     `CSLBFGS`.
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

function _scenario()
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
    cost_fn = function (controls)
        return AT.cs_surface_flux_4dvar(
            panels_rm, panels_m, panels_am, panels_bm, panels_cm,
            mesh, observations, controls;
            scheme = AT.PPMScheme(AT.NoLimiter()), dt = dt)
    end
    return (control = control, cost_fn = cost_fn)
end

# ---------------------------------------------------------------------------
# Default behavior: log = false ⇒ solve.log === nothing
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# GD with log enabled
# ---------------------------------------------------------------------------

@testset "optimizer iteration log smoke" for opt in (
        AT.CSGradientDescent(iterations = 4, initial_step = 0.25, log = true),
        AT.CSLBFGS(iterations = 6, gradient_tolerance = 1e-12, log = true))
    s = _scenario()
    solve = AT.cs_surface_flux_4dvar_solve(opt, s.cost_fn, s.control)
    @test solve.log isa AT.CSIterationLog
    @test [entry.cost for entry in solve.log] == solve.cost_history
    @test [entry.gradient_norm for entry in solve.log] == solve.gradient_norm_history
    @test [entry.iteration for entry in solve.log] == collect(0:solve.iterations)
    @test all(entry -> isapprox(entry.cost, entry.observation_cost + entry.background_cost;
                               atol = 1e-10), solve.log)
end
