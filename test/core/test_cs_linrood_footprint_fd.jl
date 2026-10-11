#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Lin-Rood surface-emission footprint vs central finite differences, with
# vertical mass flux.
#
# Regression test for issue #10. The Lin-Rood tape staged the air mass for its
# two Upwind Z half-sweeps AFTER the sweep had updated it, while the reverse Z
# kernel rebuilds each upwind coefficient F / m_donor from the mass ENTERING the
# sweep. The resulting gradient error is ~C_z^2 for a surface-layer objective and
# ~C_h * C_z^2 for a column mean. It vanishes for purely downward cm: surface
# emissions then never leave the surface layer through the vertical, so no Z
# coefficient enters the footprint. The problem below therefore
# uses upward cm at C_z ~ 0.08 (C_h ~ 0.02), where the old tape was off by up to
# 8.6e-5 (surface layer) and 5.7e-8 (column mean) relative; with the pre-sweep
# mass staged the identity holds to ~1e-11, the same as the linear PPM control.
#
# The base tracer field is smooth and nonzero, so the Lin-Rood monotonicity
# limiter keeps one branch everywhere inside the FD stencil and J is affine in
# the emissions there (checked by the linearity witness below). Central
# differences are then exact up to rounding, so any mismatch is an adjoint error.
# ---------------------------------------------------------------------------

using Test

import AtmosTransport
const AT = AtmosTransport
const Adv = AtmosTransport.Operators.Advection

# Courant numbers are set directly: am, bm ~ Ch * m and cm ~ Cz * m. cm < 0 is
# upward (level 1 is the model top, level Nz the surface layer) and is zero at the
# model top and bottom faces.
function _linrood_vertical_flux_problem(; Nc = 4, Nz = 8, nsteps = 2,
                                        Ch = 0.02, Cz = 0.08)
    FT = Float64
    mesh = AT.CubedSphereMesh(Nc = Nc, Hp = 3, FT = FT)
    Hp = mesh.Hp
    N = Nc + 2Hp
    m = ntuple(p -> [FT(2.0 + 0.25k + 0.01p + 0.02sin(0.7i + 0.4j))
                     for i in 1:N, j in 1:N, k in 1:Nz], 6)
    Adv.fill_panel_halos!(m, mesh; dir = 0)
    rm = ntuple(p -> [m[p][i, j, k] * (0.3 + 0.1sin(0.2i + 0.3j) + 0.05k + 0.7sin(0.5p))
                      for i in 1:N, j in 1:N, k in 1:Nz], 6)
    Adv.fill_panel_halos!(rm, mesh; dir = 0)
    mref = 2.0
    am = [ntuple(p -> [(Hp < j <= Hp + Nc && Hp < i <= Hp + Nc + 1) ?
                       Ch * mref * sin(0.2s + 0.3p + 0.7i + 0.4j + 0.2k) : 0.0
                       for i in 1:N+1, j in 1:N, k in 1:Nz], 6) for s in 1:nsteps]
    bm = [ntuple(p -> [(Hp < j <= Hp + Nc + 1 && Hp < i <= Hp + Nc) ?
                       Ch * mref * cos(0.3s + 0.5p + 0.4i + 0.6j + 0.1k) : 0.0
                       for i in 1:N, j in 1:N+1, k in 1:Nz], 6) for s in 1:nsteps]
    cm = [ntuple(p -> [(2 <= k <= Nz && Hp < i <= Hp + Nc && Hp < j <= Hp + Nc) ?
                       -Cz * mref * sin(pi * (k - 1) / Nz) *
                       (0.6 + 0.4sin(i + 0.7j + 0.3k + p + s)) : 0.0
                       for i in 1:N, j in 1:N, k in 1:Nz+1], 6) for s in 1:nsteps]
    return (; mesh, m, rm, am, bm, cm, Nc, Nz, nsteps)
end

_emission_direction(P, seed) =
    [ntuple(p -> [sin(seed + 0.31s + 0.17p + 0.23i - 0.19j) for i in 1:P.Nc, j in 1:P.Nc], 6)
     for s in 1:P.nsteps]
_scaled(P, a, x) = [ntuple(p -> a .* x[s][p], 6) for s in 1:P.nsteps]
_summed(P, x, y) = [ntuple(p -> x[s][p] .+ y[s][p], 6) for s in 1:P.nsteps]
_dot(P, x, y) = sum(sum(x[s][p] .* y[s][p]) for s in 1:P.nsteps, p in 1:6)

_objective_value(P, scheme, objective, rates) = AT.run_cs_footprint_forward(
    P.rm, P.m, P.am, P.bm, P.cm, P.mesh, objective;
    scheme, dt = 1.0, emission_rates = rates)

@testset "Lin-Rood footprint vs FD with upward vertical flux (issue #10)" begin
    P = _linrood_vertical_flux_problem()
    objectives = (surface_layer = AT.CSLayerMeanObjective(1, 1, 1, P.Nz),
                  column_mean   = AT.CSColumnMeanObjective(1, 1, 1))
    h = 1e-4
    for ord in (5, 7), (name, objective) in pairs(objectives)
        scheme = AT.LinRoodPPMScheme(ord)
        @testset "LinRoodPPMScheme($ord), $name" begin
            zero_rates = _scaled(P, 0.0, _emission_direction(P, 0.0))
            a = _emission_direction(P, 0.0)
            b = _emission_direction(P, 1.3)

            # J is affine in the emissions over the FD stencil: the limiter does
            # not switch branch, so central differences are exact up to rounding.
            j0 = _objective_value(P, scheme, objective, zero_rates)
            ja = _objective_value(P, scheme, objective, _scaled(P, h, a))
            jb = _objective_value(P, scheme, objective, _scaled(P, h, b))
            jab = _objective_value(P, scheme, objective,
                                   _summed(P, _scaled(P, h, a), _scaled(P, h, b)))
            @test abs(jab - ja - jb + j0) <= 1e-9 * abs(ja - j0)

            footprints = AT.cs_surface_emission_footprint(
                P.rm, P.m, P.am, P.bm, P.cm, P.mesh, objective;
                scheme, dt = 1.0).footprints
            for direction in (a, b)
                j_plus = _objective_value(P, scheme, objective, _scaled(P, h, direction))
                j_minus = _objective_value(P, scheme, objective, _scaled(P, -h, direction))
                finite_difference = (j_plus - j_minus) / (2h)
                adjoint = _dot(P, footprints, direction)
                @test abs(adjoint - finite_difference) <= 1e-9 * abs(finite_difference)
            end
        end
    end
end
