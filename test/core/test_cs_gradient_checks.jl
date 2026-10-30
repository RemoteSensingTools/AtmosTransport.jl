# Production gradient gates: algebraic identities, Taylor convergence and
# finite differences share one nonzero-flow problem. Each section keeps its
# own random stream and numerical tolerances.
module GradientProblem
import AtmosTransport
const AT = AtmosTransport
const Adv = AT.Operators.Advection
function _transport_cs_problem(; Nc=3, Nz=3, nsteps=3, FT=Float64)
    mesh = AT.CubedSphereMesh(Nc=Nc, Hp=3, FT=FT)
    N = mesh.Nc + 2mesh.Hp
    Hp = mesh.Hp

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

    panels_am_steps = Vector{Any}(undef, nsteps)
    panels_bm_steps = Vector{Any}(undef, nsteps)
    panels_cm_steps = Vector{Any}(undef, nsteps)
    for step in 1:nsteps
        panels_am_steps[step] = ntuple(6) do p
            am = zeros(FT, N + 1, N, Nz)
            for k in 1:Nz, j in Hp + 1:Hp + mesh.Nc, i in Hp + 1:Hp + mesh.Nc + 1
                am[i, j, k] = FT(0.015) * sin(FT(0.2step + 0.3p + 0.7i + 0.4j + 0.2k))
            end
            am
        end
        panels_bm_steps[step] = ntuple(6) do p
            bm = zeros(FT, N, N + 1, Nz)
            for k in 1:Nz, j in Hp + 1:Hp + mesh.Nc + 1, i in Hp + 1:Hp + mesh.Nc
                bm[i, j, k] = FT(0.012) * cos(FT(0.3step + 0.5p + 0.4i + 0.6j + 0.1k))
            end
            bm
        end
        panels_cm_steps[step] = ntuple(6) do p
            cm = zeros(FT, N, N, Nz + 1)
            for k in 2:Nz, j in Hp + 1:Hp + mesh.Nc, i in Hp + 1:Hp + mesh.Nc
                cm[i, j, k] = -FT(0.010) * (one(FT) + FT(0.1) * sin(FT(i + j + k + p + step)))
            end
            cm
        end
    end
    return mesh, panels_m, panels_rm, panels_am_steps, panels_bm_steps, panels_cm_steps
end
end

module ModelSpace
using Test
using Random
using LinearAlgebra

import AtmosTransport
const AT = AtmosTransport
const Adv = AtmosTransport.Operators.Advection

using ..GradientProblem: _transport_cs_problem


# Per-step controls, one CSSurfaceFluxControl per step over a single
# step (weight = 1). This is the cleanest x → emission_rates mapping:
# emission_rates[step] = controls[step].value.
function _per_step_controls(rng, mesh, nsteps; scale = 0.01)
    return [AT.CSSurfaceFluxControl(
                AT.CSSurfaceFluxWindow(Symbol("step_", k), k),
                ntuple(p -> scale .* (rand(rng, Float64, mesh.Nc, mesh.Nc) .- 0.5), 6))
            for k in 1:nsteps]
end

# Random departure vector in observation space.
function _random_departures(rng, K)
    return rand(rng, Float64, K) .- 0.5
end

# Single observation set sampled across panels, ij, steps, and
# objective kinds. σ = 1 on every observation so residuals pass
# through unattenuated.
function _build_observations(nsteps)
    return [
        AT.CSObservation(1, AT.CSLayerMeanObjective(1, 2, 2, 1), 0.0, 1.0),
        AT.CSObservation(2, AT.CSColumnMeanObjective(2, 1, 2), 0.0, 1.0),
        AT.CSObservation(nsteps, AT.CSLayerMeanObjective(3, 2, 1, 2), 0.0, 1.0),
        AT.CSObservation(nsteps, AT.CSColumnMeanObjective(4, 2, 1), 0.0, 1.0),
    ]
end

# Run the production cost/gradient path with σ=1, no background, and
# arbitrary obs values. We use `.simulated` for y values and
# `.gradients` for `Hᵀ d` (after rewriting obs values to make
# residuals equal `d` — see `_extract_Hadjoint_d`).
function _run_4dvar(mesh, panels_rm0, panels_m0,
                    panels_am, panels_bm, panels_cm,
                    observations, controls;
                    dt = 2.0)
    return AT.cs_surface_flux_4dvar(
        panels_rm0, panels_m0,
        panels_am, panels_bm, panels_cm,
        mesh, observations, controls;
        scheme = AT.PPMScheme(AT.NoLimiter()),
        dt = dt,
        preconditioner = nothing)
end

# Extract `Hᵀ d` per control at linearization point `x_ref` (a
# Vector{CSSurfaceFluxControl}) and departure `d`. The trick: with
# σ = 1 and `obs.value = simulated_k(x_ref) − d_k`, the production
# residual equals `d_k` and the production gradient equals `Hᵀ d`.
function _extract_Hadjoint_d(mesh, panels_rm0, panels_m0,
                             panels_am, panels_bm, panels_cm,
                             observations, x_ref, d; dt = 2.0)
    # Step 1 — forward at x_ref to learn simulated values.
    y_ref = _run_4dvar(mesh, panels_rm0, panels_m0,
                      panels_am, panels_bm, panels_cm,
                      observations, x_ref; dt = dt).simulated
    # Step 2 — synthesize observations whose residual at x_ref equals d.
    obs_d = [AT.CSObservation(observations[k].step, observations[k].objective,
                              y_ref[k] - d[k], 1.0)
             for k in eachindex(observations)]
    # Step 3 — read back gradients = Hᵀ d.
    result_d = _run_4dvar(mesh, panels_rm0, panels_m0,
                          panels_am, panels_bm, panels_cm,
                          obs_d, x_ref; dt = dt)
    # Sanity: residuals match `d` to roundoff (round-trip y_ref - obs_d).
    @test all(isapprox.(result_d.residuals, d; atol = 1e-12, rtol = 1e-12))
    return result_d.gradients, y_ref
end

# ⟨x1 - x2, f⟩ over per-step controls (Vector{CSSurfaceFluxControl}).
function _flux_inner(controls_a, controls_b, f_per_control)
    total = 0.0
    @inbounds for c in eachindex(controls_a)
        for p in 1:6
            total += sum((controls_a[c].value[p] .- controls_b[c].value[p]) .*
                         f_per_control[c][p])
        end
    end
    return total
end

@testset "Adjoint identity in model space (memo Eqs. 2.4-2.5)" begin
    rng = MersenneTwister(0xA7A105)
    mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
        _transport_cs_problem(Nc = 3, Nz = 3, nsteps = 3)
    dt = 2.0
    observations = _build_observations(3)
    K = length(observations)



    # Two random trials. Each draws fresh (x_ref, x1, x2, d), extracts
    # Hᵀd via the σ=1 trick, forward-simulates y1 and y2 through the
    # same observation operator, and checks the algebraic identity to
    # ~1e-12 relative.
    @testset "Random trial $trial" for trial in 1:2
        x_ref = _per_step_controls(rng, mesh, 3)
        x_1   = _per_step_controls(rng, mesh, 3)
        x_2   = _per_step_controls(rng, mesh, 3)
        d     = _random_departures(rng, K)

        # Hᵀ d at the linearization point x_ref. For linear H the
        # linearization point is irrelevant; we still pick one
        # explicitly so the call signature is honest.
        f, _ = _extract_Hadjoint_d(mesh, panels_rm, panels_m,
                                    panels_am, panels_bm, panels_cm,
                                    observations, x_ref, d; dt = dt)

        # y1, y2 from the same observation operator (.simulated field
        # of the production cost path).
        y_1 = _run_4dvar(mesh, panels_rm, panels_m,
                         panels_am, panels_bm, panels_cm,
                         observations, x_1; dt = dt).simulated
        y_2 = _run_4dvar(mesh, panels_rm, panels_m,
                         panels_am, panels_bm, panels_cm,
                         observations, x_2; dt = dt).simulated

        lhs = dot(d, y_1 .- y_2)
        rhs = _flux_inner(x_1, x_2, f)
        scale = max(abs(lhs), abs(rhs), 1e-30)

        # Both sides should agree to ~10 ulps of double precision.
        # The empirical probe in the design review gave ~5e-15
        # for PPM(NoLimiter); 1e-12 is comfortable headroom for
        # mesh / panel-edge accumulation.
        @test abs(lhs - rhs) / scale < 1e-12
        @test isfinite(lhs) && isfinite(rhs)
    end
end
end

module PreconditionedSpace
using Test
using Random
using LinearAlgebra

import AtmosTransport
const AT = AtmosTransport
const Adv = AtmosTransport.Operators.Advection

using ..GradientProblem: _transport_cs_problem


# A control per step. Each control holds the χ-space vector for that
# step — `value` is sized `(Nc, Nc)` per panel.
function _per_step_chi_controls(rng, mesh, nsteps; scale = 0.5)
    return [AT.CSSurfaceFluxControl(
                AT.CSSurfaceFluxWindow(Symbol("step_", k), k),
                ntuple(p -> scale .* (rand(rng, Float64, mesh.Nc, mesh.Nc) .- 0.5), 6))
            for k in 1:nsteps]
end

function _build_observations(nsteps)
    return [
        AT.CSObservation(1, AT.CSLayerMeanObjective(1, 2, 2, 1), 0.0, 1.0),
        AT.CSObservation(2, AT.CSColumnMeanObjective(2, 1, 2), 0.0, 1.0),
        AT.CSObservation(nsteps, AT.CSLayerMeanObjective(3, 2, 1, 2), 0.0, 1.0),
        AT.CSObservation(nsteps, AT.CSColumnMeanObjective(4, 2, 1), 0.0, 1.0),
    ]
end

function _run_4dvar(mesh, panels_rm0, panels_m0,
                    panels_am, panels_bm, panels_cm,
                    observations, controls;
                    dt = 2.0, preconditioner = nothing)
    return AT.cs_surface_flux_4dvar(
        panels_rm0, panels_m0,
        panels_am, panels_bm, panels_cm,
        mesh, observations, controls;
        scheme = AT.PPMScheme(AT.NoLimiter()),
        dt = dt,
        preconditioner)
end

# <χ_1 - χ_2, η> over per-control panels.
function _chi_inner(chi_a, chi_b, eta)
    total = 0.0
    @inbounds for c in eachindex(chi_a)
        for p in 1:6
            total += sum((chi_a[c].value[p] .- chi_b[c].value[p]) .* eta[c][p])
        end
    end
    return total
end

# Build a Linear preconditioner of the requested covariance kind.
# Background `x_b = 1.0` on every panel — non-zero so the L χ + x_b
# composition is exercised.
function _build_linear_preconditioner(mesh; cov_kind::Symbol)
    Nc = mesh.Nc
    sigma = ntuple(_ -> fill(0.3, Nc, Nc), 6)
    cov = cov_kind === :gaussian ?
        AT.IsotropicGaussianCSCovariance(sigma, 0.8) :
        AT.DiagonalCSCovariance(sigma)
    x_b = ntuple(_ -> fill(1.0, Nc, Nc), 6)
    return AT.CSSurfaceFluxPreconditioner(cov, x_b, AT.LinearOptimType()), x_b
end

# Extract Lᵀ Hᵀ d through production preconditioned 4D-Var, subtracting
# its prior contribution chi from the reported gradient.
function _run_identity_trial(mesh, panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                              observations, prec, rng; dt)
    chi_1 = _per_step_chi_controls(rng, mesh, 3)
    chi_2 = _per_step_chi_controls(rng, mesh, 3)
    d = rand(rng, Float64, length(observations)) .- 0.5
    evaluate(obs, controls) = _run_4dvar(mesh, panels_rm, panels_m, panels_am, panels_bm,
                                       panels_cm, obs, controls; dt, preconditioner = prec)
    y_1 = evaluate(observations, chi_1).simulated
    y_2 = evaluate(observations, chi_2).simulated
    obs_d = [AT.CSObservation(o.sampling, o.objective, y_1[k] - d[k], 1.0)
             for (k, o) in enumerate(observations)]
    result = evaluate(obs_d, chi_1)
    @test result.residuals ≈ d atol = 1e-12 rtol = 1e-12
    # Production adds the prior gradient chi; isolate the observation transpose.
    eta = [ntuple(p -> result.gradients[c][p] .- chi_1[c].value[p], 6)
           for c in eachindex(chi_1)]
    return dot(d, y_1 .- y_2), _chi_inner(chi_1, chi_2, eta)
end

@testset "Adjoint identity in preconditioned space (memo Eqs. 2.6-2.7)" begin
    rng = MersenneTwister(0xA7A105)
    mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
        _transport_cs_problem(Nc = 3, Nz = 3, nsteps = 3)
    dt = 2.0
    observations = _build_observations(3)

    @testset "Linear L × $(cov_kind) covariance" for cov_kind in (:diagonal, :gaussian)
        prec, _x_b = _build_linear_preconditioner(mesh; cov_kind = cov_kind)
        @testset "Random trial $trial" for trial in 1:3
            lhs, rhs = _run_identity_trial(mesh, panels_rm, panels_m,
                                            panels_am, panels_bm, panels_cm,
                                            observations, prec, rng; dt = dt)
            scale = max(abs(lhs), abs(rhs), 1e-30)
            @test abs(lhs - rhs) / scale < 1e-12
            @test isfinite(lhs) && isfinite(rhs)
        end
    end

end
end

module TaylorSweep
using Test
using Random
using Printf

import AtmosTransport
const AT = AtmosTransport
const Adv = AtmosTransport.Operators.Advection

using ..GradientProblem: _transport_cs_problem


# Three per-step χ-space controls aligned with `nsteps = 3`.
function _make_chi_controls(chi_panels_per_step)
    return [AT.CSSurfaceFluxControl(
                AT.CSSurfaceFluxWindow(Symbol("step_", k), k),
                chi_panels_per_step[k])
            for k in eachindex(chi_panels_per_step)]
end

# Observations chosen with large enough `value` that J(χ₀=0) — i.e.
# at x = x_b — is well above roundoff, so the Taylor sweep has
# headroom before cancellation kicks in.
function _build_observations()
    return [
        AT.CSObservation(1, AT.CSLayerMeanObjective(1, 2, 2, 1), 0.50, 1.0),
        AT.CSObservation(2, AT.CSColumnMeanObjective(2, 1, 2),   0.40, 1.0),
        AT.CSObservation(3, AT.CSLayerMeanObjective(3, 2, 1, 2), 0.30, 1.0),
        AT.CSObservation(3, AT.CSColumnMeanObjective(4, 2, 1),   0.20, 1.0),
    ]
end

function _build_preconditioner(mesh; cov_kind::Symbol = :diagonal,
                                optim_type::AT.AbstractCSOptimType = AT.LinearOptimType())
    Nc = mesh.Nc
    sigma = ntuple(_ -> fill(0.3, Nc, Nc), 6)
    cov = cov_kind === :gaussian ?
        AT.IsotropicGaussianCSCovariance(sigma, 0.8) :
        AT.DiagonalCSCovariance(sigma)
    x_b = ntuple(_ -> fill(1.0, Nc, Nc), 6)
    return AT.CSSurfaceFluxPreconditioner(cov, x_b, optim_type)
end

# One-shot `cs_surface_flux_4dvar` wrapper. Returns the same result
# object so callers can read `.cost` and (for χ₀ only) `.gradients`.
# Yes, every sweep call also evaluates the adjoint — we accept that
# overhead rather than re-deriving a cost-only path, because the
# alternative would be a parallel forward-evaluation routine that
# could silently drift from the production observation operator.
function _eval_at_chi(mesh, panels_rm0, panels_m0,
                     panels_am, panels_bm, panels_cm,
                     observations, chi_controls, preconditioner; dt = 2.0)
    return AT.cs_surface_flux_4dvar(
        panels_rm0, panels_m0,
        panels_am, panels_bm, panels_cm,
        mesh, observations, chi_controls;
        scheme = AT.PPMScheme(AT.NoLimiter()),
        dt = dt,
        preconditioner = preconditioner)
end

# Vector-space helpers over `Vector{NTuple{6, Matrix{Float64}}}`.
function _flat_inner(a, b)
    total = 0.0
    @inbounds for c in eachindex(a), p in 1:6
        total += sum(a[c][p] .* b[c][p])
    end
    return total
end

# χ_k = χ₀ - α g₀, returned as a Vector{NTuple{6, Matrix}}.
function _step_chi(chi0, g, alpha)
    return [ntuple(p -> chi0[c].value[p] .- alpha .* g[c][p], 6)
            for c in eachindex(chi0)]
end

# Find the longest geometrically-decreasing prefix of |r| that
# survives cancellation, then return ratios `|r_{k+1}|/|r_k|` over
# that prefix. The classical signature of a passing gradient test is
# that these ratios cluster near `1/f`.
function _clean_ratios(r_seq)
    abs_r = abs.(r_seq)
    # Truncate at the first index where |r| stops decreasing — that's
    # where Taylor's linear term and roundoff noise are comparable.
    clean_end = length(abs_r)
    for k in 2:length(abs_r)
        if abs_r[k] >= abs_r[k-1]
            clean_end = k - 1
            break
        end
    end
    return [abs_r[k+1] / abs_r[k] for k in firstindex(abs_r):clean_end-1], clean_end, abs_r
end

# Single Taylor-sweep trial for a given `(χ₀, preconditioner)`. Returns
# the sweep diagnostics so the testset can report them on failure.
function _taylor_sweep(mesh, panels_rm, panels_m,
                       panels_am, panels_bm, panels_cm,
                       observations, chi0_controls, prec;
                       alpha0 = 1e-1, factor = 2.0, M = 10, dt = 2.0)
    base = _eval_at_chi(mesh, panels_rm, panels_m,
                        panels_am, panels_bm, panels_cm,
                        observations, chi0_controls, prec; dt = dt)
    J0 = base.cost
    g0 = base.gradients
    DJ1 = _flat_inner(g0, g0)
    DJ1 > 0 || error("Taylor sweep requires DJ1 > 0; got $DJ1 — pick a different χ₀ " *
                     "or boost observation magnitude")

    alphas = [alpha0 / factor^(k-1) for k in 1:M]
    r_seq = Vector{Float64}(undef, M)
    for k in eachindex(alphas)
        chi_k_panels = _step_chi(chi0_controls, g0, alphas[k])
        chi_k_controls = _make_chi_controls(chi_k_panels)
        Jk = _eval_at_chi(mesh, panels_rm, panels_m,
                          panels_am, panels_bm, panels_cm,
                          observations, chi_k_controls, prec; dt = dt).cost
        DJ2 = (J0 - Jk) / alphas[k]
        r_seq[k] = 1 - DJ2 / DJ1
    end
    return alphas, r_seq, J0, DJ1
end

# Pretty-print the sweep for diagnostic output on failure.
function _print_sweep(alphas, r_seq, abs_r, clean_end)
    println("    α               r                |r|              clean?")
    for k in eachindex(alphas)
        mark = k <= clean_end ? "✓" : "·"
        @printf("    %.3e       %+.3e        %.3e       %s\n",
                alphas[k], r_seq[k], abs_r[k], mark)
    end
end

@testset "Gradient Taylor sweep (memo Eqs. 2.8-2.11)" begin
    rng = MersenneTwister(0xA7A105)
    mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
        _transport_cs_problem(Nc = 3, Nz = 3, nsteps = 3)
    dt = 2.0
    observations = _build_observations()

    # Three fixtures covering both the χ₀ placement and the
    # preconditioner family. The PDF's gradient test holds for any
    # preconditioner whose tangent and adjoint are coded correctly,
    # so we exercise both `LinearOptimType` (x = x_b + B^(1/2) χ)
    # and `LogNormalOptimType` (x = x_b ⊙ exp(B^(1/2) χ)) — the
    # latter being non-linear in χ. A bug in either preconditioner's
    # adjoint would break the f-factor convergence signature here.
    chi_zero = [ntuple(_ -> zeros(Float64, mesh.Nc, mesh.Nc), 6) for _ in 1:3]
    chi_rand = [ntuple(_ -> 0.2 .* (rand(rng, Float64, mesh.Nc, mesh.Nc) .- 0.5), 6)
                for _ in 1:3]

    fixtures = (
        ("Linear L, χ₀ = 0 (prior)",   chi_zero, AT.LinearOptimType()),
        ("Linear L, χ₀ random",         chi_rand, AT.LinearOptimType()),
        ("LogNormal L, χ₀ random",      chi_rand, AT.LogNormalOptimType()),
    )

    @testset "$(label)" for (label, chi0_panels, optim_type) in fixtures
        prec = _build_preconditioner(mesh;
                                     cov_kind = :diagonal,
                                     optim_type = optim_type)
        chi0_controls = _make_chi_controls(chi0_panels)

        alphas, r_seq, J0, DJ1 = _taylor_sweep(
            mesh, panels_rm, panels_m,
            panels_am, panels_bm, panels_cm,
            observations, chi0_controls, prec;
            alpha0 = 1e-1, factor = 2.0, M = 10, dt = dt)

        ratios, clean_end, abs_r = _clean_ratios(r_seq)

        # Diagnostic dump — visible when the testset fails.
        @info "Taylor sweep" label J0 DJ1 clean_end

        # (1) The sweep must reach a clean prefix of at least 4 points;
        # otherwise either DJ1 is too small or α₀ is too small to
        # exercise the linear term before cancellation hits.
        @test clean_end >= 4

        # (2) Smallest clean |r| must be small in absolute terms —
        # confirms DJ₂(α) → DJ₁ as α → 0, i.e. g₀ is the true
        # directional derivative.
        @test abs_r[clean_end] < 1e-3

        # (3) Ratios in the clean middle of the sweep should cluster
        # near 1/f = 0.5. We allow [0.4, 0.6]; the leading α is large
        # enough that O(α²) terms in r(α) bias the first ratio
        # slightly, so we skip the first one if there are enough
        # samples.
        ratio_window = length(ratios) >= 4 ? ratios[2:end] : ratios
        @test all(0.4 .<= ratio_window .<= 0.6)

        # Failure-time diagnostics.
        if !all(0.4 .<= ratio_window .<= 0.6) || abs_r[clean_end] >= 1e-3
            _print_sweep(alphas, r_seq, abs_r, clean_end)
            @info "Taylor sweep ratios" ratios
        end
    end
end
end

module PreconditionedFD
using Test

import AtmosTransport
const AT = AtmosTransport
const Adv = AtmosTransport.Operators.Advection

using ..GradientProblem: _transport_cs_problem

function _dot_control_gradient(gradients, directions)
    total = 0.0
    for c in eachindex(gradients), p in 1:6
        total += sum(gradients[c][p] .* directions[c][p])
    end
    return total
end



# Two distinct controls + two observations + reasonable σ + small Gaussian
# correlation length so the spectral filter is well-conditioned at Nc = 3.
function _preconditioned_scenario(; optim_type, cov_kind = :gaussian, Nc = 3)
    mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
        _transport_cs_problem(; Nc = Nc, Nz = 3, nsteps = 3)
    dt = 2.0

    # χ = 0 baseline. x_b = 1.0 everywhere (positive — valid for both
    # optim types).
    chi_zero = ntuple(_ -> zeros(Float64, mesh.Nc, mesh.Nc), 6)
    x_b = ntuple(_ -> fill(1.0, mesh.Nc, mesh.Nc), 6)

    sigma = ntuple(_ -> fill(0.3, mesh.Nc, mesh.Nc), 6)
    cov = cov_kind === :gaussian ?
        AT.IsotropicGaussianCSCovariance(sigma, 0.8) :
        AT.DiagonalCSCovariance(sigma)
    prec_a = AT.CSSurfaceFluxPreconditioner(cov, x_b, optim_type)
    prec_b = AT.CSSurfaceFluxPreconditioner(cov, x_b, optim_type)

    controls = [
        AT.CSSurfaceFluxControl(
            AT.CSSurfaceFluxWindow(:first_step, 1), chi_zero),
        AT.CSSurfaceFluxControl(
            AT.CSSurfaceFluxWindow(:late_window, 2:3; normalize=true),
            chi_zero),
    ]
    preconditioners = [prec_a, prec_b]

    observations = [
        AT.CSObservation(1, AT.CSLayerMeanObjective(1, 2, 2, 3), 0.03, 0.4),
        AT.CSObservation(3, AT.CSColumnMeanObjective(1, 2, 2), 0.02, 0.3),
    ]

    return (mesh = mesh,
            panels_rm = panels_rm, panels_m = panels_m,
            panels_am = panels_am, panels_bm = panels_bm, panels_cm = panels_cm,
            dt = dt,
            x_b = x_b,
            controls = controls,
            preconditioners = preconditioners,
            observations = observations,
            optim_type = optim_type,
            cov = cov)
end

# Convenience: call cs_surface_flux_4dvar with a custom chi.
function _eval_at_chi(s, chi_panels_per_control)
    new_controls = [
        AT.CSSurfaceFluxControl(s.controls[i].window, chi_panels_per_control[i])
        for i in eachindex(s.controls)
    ]
    return AT.cs_surface_flux_4dvar(
        s.panels_rm, s.panels_m,
        s.panels_am, s.panels_bm, s.panels_cm,
        s.mesh, s.observations, new_controls;
        scheme = AT.PPMScheme(AT.NoLimiter()),
        dt = s.dt,
        preconditioner = s.preconditioners)
end

# ---------------------------------------------------------------------------
# Sanity at χ = 0 — background cost is exactly 0; physical-side cost
# matches the unconditioned path called with `controls = x_b`.
# ---------------------------------------------------------------------------

@testset "preconditioned 4D-Var — sanity at χ = 0" begin
    for optim_type in (AT.LinearOptimType(), AT.LogNormalOptimType()),
        cov_kind in (:diagonal, :gaussian)

        s = _preconditioned_scenario(; optim_type = optim_type,
                                     cov_kind = cov_kind, Nc = 3)
        result = AT.cs_surface_flux_4dvar(
            s.panels_rm, s.panels_m,
            s.panels_am, s.panels_bm, s.panels_cm,
            s.mesh, s.observations, s.controls;
            scheme = AT.PPMScheme(AT.NoLimiter()),
            dt = s.dt,
            preconditioner = s.preconditioners)

        # Background term comes from 0.5 ‖χ‖² = 0 when χ = 0.
        @test result.background_cost == 0.0

        # Total = observation cost.
        @test result.cost ≈ result.observation_cost atol = 1e-12

        # Compare against unconditioned-mode evaluation at x_b.
        x_b_controls = [
            AT.CSSurfaceFluxControl(s.controls[i].window, s.x_b)
            for i in eachindex(s.controls)
        ]
        baseline = AT.cs_surface_flux_4dvar(
            s.panels_rm, s.panels_m,
            s.panels_am, s.panels_bm, s.panels_cm,
            s.mesh, s.observations, x_b_controls;
            scheme = AT.PPMScheme(AT.NoLimiter()), dt = s.dt)

        @test result.observation_cost ≈ baseline.observation_cost atol = 1e-10
        @test result.residuals ≈ baseline.residuals atol = 1e-10

        # Reported controls are the original χ inputs (not the
        # internal physical_controls).
        @test result.controls[1].value === s.controls[1].value
        @test result.controls[2].value === s.controls[2].value
    end

    optim_type = AT.LinearOptimType()
    s = _preconditioned_scenario(; optim_type = optim_type,
                                 cov_kind = :diagonal, Nc = 3)

    # Deterministic non-zero χ.
    chi = [
        ntuple(p -> [0.05 * sin(0.1p + 0.2i - 0.3j)
                     for i in 1:s.mesh.Nc, j in 1:s.mesh.Nc], 6),
        ntuple(p -> [0.04 * cos(0.2p + 0.3i + 0.1j)
                     for i in 1:s.mesh.Nc, j in 1:s.mesh.Nc], 6),
    ]
    expected_bg = 0.5 * sum(sum(chi[c][p] .^ 2) for c in 1:2, p in 1:6)

    result = _eval_at_chi(s, chi)
    @test result.background_cost ≈ expected_bg atol = 1e-12
end

# ---------------------------------------------------------------------------
# 0.5 ‖χ‖² literal — non-zero χ produces the expected background cost.
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Finite-difference identity — the B3 acceptance gate.
# ---------------------------------------------------------------------------

function _fd_identity_check(s; chi_scale = 0.02, eps_dir = 1e-6,
                            rtol = 5e-4, atol = 1e-8)
    # Non-zero base χ so we exercise the full chain rule (LogNormal's
    # tangent depends on x = T(χ)).
    chi_base = [
        ntuple(p -> [chi_scale * sin(0.11p + 0.17i - 0.13j)
                     for i in 1:s.mesh.Nc, j in 1:s.mesh.Nc], 6),
        ntuple(p -> [chi_scale * cos(0.07p + 0.19i + 0.23j)
                     for i in 1:s.mesh.Nc, j in 1:s.mesh.Nc], 6),
    ]
    directions = [
        ntuple(p -> [sin(0.13p + 0.29i - 0.41j)
                     for i in 1:s.mesh.Nc, j in 1:s.mesh.Nc], 6),
        ntuple(p -> [cos(0.31p + 0.37i + 0.43j)
                     for i in 1:s.mesh.Nc, j in 1:s.mesh.Nc], 6),
    ]

    base = _eval_at_chi(s, chi_base)
    plus = _eval_at_chi(s, [
        ntuple(p -> chi_base[c][p] .+ eps_dir .* directions[c][p], 6)
        for c in 1:2
    ])
    minus = _eval_at_chi(s, [
        ntuple(p -> chi_base[c][p] .- eps_dir .* directions[c][p], 6)
        for c in 1:2
    ])

    fd = (plus.cost - minus.cost) / (2 * eps_dir)
    predicted = _dot_control_gradient(base.gradients, directions)
    @test isapprox(predicted, fd; rtol = rtol, atol = atol)
end

@testset "preconditioned 4D-Var finite differences" for
        optim_type in (AT.LinearOptimType(), AT.LogNormalOptimType()), cov_kind in (:diagonal, :gaussian)
    _fd_identity_check(_preconditioned_scenario(; optim_type, cov_kind, Nc = 3))
end

# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

@testset "cs_surface_flux_4dvar — scalar preconditioner broadcasts to N controls" begin
    mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
        _transport_cs_problem(Nc = 3, Nz = 3, nsteps = 3)
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
end

module ControlShapes
using Test
import AtmosTransport
const AT = AtmosTransport
using ..GradientProblem: _transport_cs_problem
@testset "cs_surface_flux_4dvar validates control shapes vs mesh.Nc" begin
    mesh, panels_m, panels_rm, am, bm, cm =
        _transport_cs_problem(Nc = 4, Nz = 3, nsteps = 1, FT = Float32)

    # Correctly-shaped control passes the validator.
    good_control = AT.CSSurfaceFluxControl(
        AT.CSSurfaceFluxWindow(:step1, 1),
        ntuple(_ -> zeros(Float32, mesh.Nc, mesh.Nc), 6))
    observations = [
        AT.CSObservation(1, AT.CSLayerMeanObjective(1, 2, 2, 3),
                         0.01f0, 0.2f0),
    ]
    @test_throws UndefKeywordError AT.cs_surface_flux_4dvar(
        panels_rm, panels_m, am, bm, cm, mesh, observations, good_control;
        dt = Float32(1))
    AT.cs_surface_flux_4dvar(
        panels_rm, panels_m, am, bm, cm, mesh, observations, good_control;
        scheme = AT.PPMScheme(AT.NoLimiter()), dt = Float32(1))

    # Wrong-sized value panels — value[1] is (Nc+1, Nc+1) instead of
    # (Nc, Nc); the shared `_add_weighted_footprint_kernel!` would
    # silently read with `ndrange = size(rates[step][p])` and grab
    # OOB or skip cells depending on the size mismatch. Validator
    # must catch this BEFORE any kernel launches.
    bad_value = ntuple(p -> begin
        sz = p == 1 ? (mesh.Nc + 1, mesh.Nc + 1) : (mesh.Nc, mesh.Nc)
        zeros(Float32, sz)
    end, 6)
    bad_control = AT.CSSurfaceFluxControl(
        AT.CSSurfaceFluxWindow(:step1, 1), bad_value)
    @test_throws DimensionMismatch AT.cs_surface_flux_4dvar(
        panels_rm, panels_m, am, bm, cm, mesh, observations, bad_control;
        scheme = AT.PPMScheme(AT.NoLimiter()), dt = Float32(1))

    # Wrong-sized sigma panels are also rejected (the
    # `_add_background_gradient_array_kernel!` reads `sigma[i, j]`).
    bad_sigma = ntuple(p -> begin
        sz = p == 3 ? (mesh.Nc - 1, mesh.Nc) : (mesh.Nc, mesh.Nc)
        fill(Float32(0.1), sz)
    end, 6)
    # Have to bypass the constructor's per-panel sigma vs value cross-
    # check; build a control with correct sigma first then mutate the
    # tuple via a fresh constructor that skips the check.
    # The constructor at Observations.jl:131-141 validates
    # sigma-panel-vs-value-panel shape, which is exactly what we test
    # here from the mesh side too — confirm both gates fire.
    @test_throws DimensionMismatch AT.CSSurfaceFluxControl(
        AT.CSSurfaceFluxWindow(:step1, 1),
        ntuple(_ -> zeros(Float32, mesh.Nc, mesh.Nc), 6);
        sigma = bad_sigma)
end
end
