#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan 25 Commit 6 — LinRood end-to-end adjoint integration test.
#
# Verifies that `cs_surface_emission_footprint(scheme=LinRoodPPMScheme())`
# returns gradients consistent with centered finite-difference probes on
# a small 6-panel CS problem (Nc=4, Hp=3, Nz=3). Exercises the cross-panel
# halo adjoint (`_adjoint_fill_panel_halos!`) via the LinRood tape
# integration shipped in this commit.
# ---------------------------------------------------------------------------

using Test
using Random

using AtmosTransport

# Reuse the existing test helpers (the footprint test lives under test/core/).
include(joinpath(@__DIR__, "..", "core", "test_cs_ppm_adjoint_footprint.jl"))

# LinRoodPPMScheme applies a monotonicity limiter (`_apply_monotonicity` in
# LinRood.jl: clip the PPM reconstruction to first order at local extrema), so
# the forward map is mildly NONLINEAR. Its reverse-mode adjoint is the transpose
# of the tangent-linear model — the limiter switch is decided by the forward
# base value, and the active branch is propagated linearly. Validate that adjoint
# against central finite differences on a SMOOTH nonzero base field, where the
# limiter stays away from its switch points and central FD is a valid gradient
# probe (matches to ~1e-11).
#
# Do NOT validate this scheme at a zero IC + localized emission (the
# `_transport_cs_problem` default): that lands the forward trajectory ON limiter
# switch points (kinks), where central FD averages the two one-sided derivatives
# and disagrees with the correct one-sided adjoint by an eps-INDEPENDENT ~1.4e-3.
# That is a finite-difference artifact, NOT an adjoint error — proven 2026-06-02:
# the recorded forward (`_record_linrood_horizontal_substep!`, record_ops=false)
# is bit-identical to production `fv_tp_2d_cs!`; the single-substep VJP matches FD
# to 1e-7; and this footprint identity matches to ~1e-11 on smooth fields for both
# ORD=5 and ORD=7. It was ~1e-8 until issue #10 was fixed: the Z half-sweeps
# staged the post-sweep air mass. (The split-sweep footprint test below validates
# at a zero IC precisely because it uses NoLimiter / linear schemes.)
function _seed_smooth_cs_ic!(panels_rm, mesh)
    Hp = mesh.Hp
    N = mesh.Nc + 2Hp
    Nz = size(panels_rm[1], 3)
    for p in 1:6
        for k in 1:Nz, j in 1:N, i in 1:N
            panels_rm[p][i, j, k] = 0.3 + 0.1sin(0.2i + 0.3j) + 0.05k + 0.7sin(0.5p)
        end
    end
    Adv.fill_panel_halos!(panels_rm, mesh; dir = 0)
    return panels_rm
end

@testset "Plan 25 Commit 6 — LinRoodPPMScheme footprint vs FD" begin
    mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
        _transport_cs_problem(Nc=3, Nz=4, nsteps=2)
    # Smooth nonzero IC so the monotonicity limiter stays inactive and central FD
    # is a valid adjoint check (see the note above _seed_smooth_cs_ic!).
    _seed_smooth_cs_ic!(panels_rm, mesh)
    dt = 1.5
    scheme = AT.LinRoodPPMScheme()
    obj = AT.CSColumnMeanObjective(1, 2, 2)

    result = AT.cs_surface_emission_footprint(
        panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
        scheme=scheme, dt=dt, epsilon=1e-6,
    )

    rates = [ntuple(6) do p
        [sin(0.31step + 0.17p + 0.23i - 0.19j) for i in 1:mesh.Nc, j in 1:mesh.Nc]
    end for step in 1:2]

    eps_dir = 2e-6
    j_plus = AT.run_cs_footprint_forward(
        panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
        scheme=scheme, dt=dt,
        emission_rates=_scaled_rates(rates, eps_dir))
    j_minus = AT.run_cs_footprint_forward(
        panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
        scheme=scheme, dt=dt,
        emission_rates=_scaled_rates(rates, -eps_dir))

    fd = (j_plus - j_minus) / (2eps_dir)
    predicted = _dot_footprint(result, rates)

    # On a smooth base field the limiter is inactive, so the (correct) adjoint
    # matches central FD to ~1e-11 (6e-12 measured; 1.2e-8 before the issue #10
    # fix). (Resolved 2026-06-02: the former "~7.4e-4/substep WIP residual" was a
    # finite-difference artifact at limiter kinks from the zero-IC default, not
    # an adjoint bug — see _seed_smooth_cs_ic!.)
    @test predicted ≈ fd rtol=1e-9
end

@testset "Plan 25 Commit 3b — LinRoodPPMScheme(7) footprint vs FD" begin
    # End-to-end adjoint identity for the ORD=7 scheme. The reverse
    # path differs from ORD=5 only at panel-edge faces, where the
    # `_apply_ord7_boundary_d6` correction in the per-kernel adjoints
    # propagates through the tape via the `_CSLinRoodHorizRecord{…, ORD}`
    # binding. Identical structure to the ORD=5 test above.
    mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
        _transport_cs_problem(Nc=3, Nz=4, nsteps=2)
    # Smooth nonzero IC so the monotonicity limiter stays inactive (see ORD=5).
    _seed_smooth_cs_ic!(panels_rm, mesh)
    dt = 1.5
    scheme = AT.LinRoodPPMScheme(7)
    obj = AT.CSColumnMeanObjective(1, 2, 2)

    result = AT.cs_surface_emission_footprint(
        panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
        scheme=scheme, dt=dt, epsilon=1e-6,
    )

    rates = [ntuple(6) do p
        [sin(0.31step + 0.17p + 0.23i - 0.19j) for i in 1:mesh.Nc, j in 1:mesh.Nc]
    end for step in 1:2]

    eps_dir = 2e-6
    j_plus = AT.run_cs_footprint_forward(
        panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
        scheme=scheme, dt=dt,
        emission_rates=_scaled_rates(rates, eps_dir))
    j_minus = AT.run_cs_footprint_forward(
        panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
        scheme=scheme, dt=dt,
        emission_rates=_scaled_rates(rates, -eps_dir))

    fd = (j_plus - j_minus) / (2eps_dir)
    predicted = _dot_footprint(result, rates)

    # See the ORD=5 case above: on a smooth base field the limiter is inactive and
    # the correct adjoint matches central FD to ~1e-11 (7e-11 measured).
    @test predicted ≈ fd rtol=1e-9
end
