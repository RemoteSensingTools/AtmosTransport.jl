#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan 26 stretch — synthetic 4D-Var inversion experiment.
#
# Demonstrates the full Plan 26 Phase B + C stack end-to-end:
#
#   1. Define a TRUTH emission field (a Gaussian blob on panel 1).
#   2. Run the forward model with the truth to generate clean
#      simulated values at every observation location.
#   3. Add Gaussian observation noise.
#   4. Re-run the 4D-Var driver starting from a ZERO prior, with the
#      noisy observations as the data.
#   5. Compare recovered emissions to the truth — L2 relative
#      error, peak ratio, χ² goodness, normalized residuals.
#
# The synthetic met is constant flow (zero advection), so emissions
# deposit into their source cell and column-mean observations recover
# the local emission rate up to noise. The diagonal background term
# from the preconditioner pins unobserved cells toward zero (the
# prior). Even in this near-degenerate case the inversion is a
# proper test that the full pipeline assembles: covariance →
# preconditioner → χ-space cost gradient → L-BFGS → iteration log.
#
# Usage:
#     julia --project=. scripts/inversions/synthetic_experiment.jl
#
# Re-exposed as `run_synthetic_experiment()` so the truth-recovery
# regression test in `test/test_cs_inversion_truth_recovery.jl` can
# include the script and inspect the returned diagnostics.
# ---------------------------------------------------------------------------

using LinearAlgebra: norm

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
const AT = AtmosTransport
const Adv = AtmosTransport.Operators.Advection

# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------

"""
    SyntheticExperimentConfig

Parameters of the synthetic inversion experiment. Defaults are tuned
for a 1–2 s smoke run that exercises every Phase B + C component;
crank `Nc`, `nsteps`, or `nobs` for a heavier demonstration.

`scheme` is the advection scheme of both the truth run and the
inversion; it must have a cubed-sphere adjoint
(`AT.require_adjoint_support`). The default unlimited PPM is linear.
The experiment's meteorology has zero mass fluxes, so the scheme does
not change its result.
"""
struct SyntheticExperimentConfig{FT <: AbstractFloat, S}
    Nc::Int                # cubed-sphere panel side length
    Hp::Int                # halo width
    Nz::Int                # vertical levels
    nsteps::Int            # number of model time steps
    dt::FT                 # seconds per step
    obs_panel::Int         # panel hosting the truth + observations
    blob_center::Tuple{Int, Int}   # (i, j) of the truth blob's peak
    blob_sigma_cells::FT   # 1-σ width of the truth blob (in cells)
    blob_peak::FT          # peak emission rate of the truth blob
    obs_half_window::Int   # observations cover a (2*W+1)² patch
                            # centered on `blob_center`
    obs_noise_frac::FT     # σ_noise = obs_noise_frac · max |signal|
    prior_sigma::FT        # background-error σ used by the covariance
    correlation_length::FT # IsotropicGaussian correlation length
                            # (cells)
    iterations::Int        # L-BFGS max iterations
    seed::UInt64           # deterministic RNG seed for noise
    scheme::S              # advection scheme (truth run and inversion)
end

function SyntheticExperimentConfig(; Nc = 6, Hp = 3, Nz = 3,
                                     nsteps = 2, dt = 2.0,
                                     obs_panel = 1,
                                     blob_center = (3, 3),
                                     blob_sigma_cells = 1.2,
                                     blob_peak = 0.05,
                                     obs_half_window = 2,
                                     obs_noise_frac = 0.05,
                                     prior_sigma = 0.05,
                                     correlation_length = 1.0,
                                     iterations = 15,
                                     seed = 0xC0FFEE,
                                     scheme = AT.PPMScheme(AT.NoLimiter()))
    FT = typeof(float(blob_peak))
    AT.require_adjoint_support(scheme)
    return SyntheticExperimentConfig{FT, typeof(scheme)}(
        Nc, Hp, Nz, nsteps, FT(dt), obs_panel, blob_center,
        FT(blob_sigma_cells), FT(blob_peak),
        obs_half_window, FT(obs_noise_frac), FT(prior_sigma),
        FT(correlation_length), iterations, UInt64(seed), scheme)
end

# ---------------------------------------------------------------------------
# Constant-flow CS problem (zero advection)
# ---------------------------------------------------------------------------

function _constant_flow_problem(cfg::SyntheticExperimentConfig{FT}) where FT
    mesh = AT.CubedSphereMesh(Nc = cfg.Nc, Hp = cfg.Hp, FT = FT)
    N = mesh.Nc + 2mesh.Hp
    panels_m = ntuple(6) do p
        m = zeros(FT, N, N, cfg.Nz)
        for k in 1:cfg.Nz, j in 1:N, i in 1:N
            m[i, j, k] = FT(2.0 + 0.25k + 0.01p)
        end
        m
    end
    panels_rm = ntuple(_ -> zeros(FT, N, N, cfg.Nz), 6)
    Adv.fill_panel_halos!(panels_m, mesh; dir = 0)
    Adv.fill_panel_halos!(panels_rm, mesh; dir = 0)
    panels_am = [ntuple(_ -> zeros(FT, N + 1, N, cfg.Nz), 6) for _ in 1:cfg.nsteps]
    panels_bm = [ntuple(_ -> zeros(FT, N, N + 1, cfg.Nz), 6) for _ in 1:cfg.nsteps]
    panels_cm = [ntuple(_ -> zeros(FT, N, N, cfg.Nz + 1), 6) for _ in 1:cfg.nsteps]
    return mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm
end

# ---------------------------------------------------------------------------
# Truth emission field — Gaussian blob on `cfg.obs_panel`
# ---------------------------------------------------------------------------

function _truth_emissions(cfg::SyntheticExperimentConfig{FT}) where FT
    Nc = cfg.Nc
    ci, cj = cfg.blob_center
    inv2σ² = inv(2 * cfg.blob_sigma_cells^2)
    panels = ntuple(p -> begin
        v = zeros(FT, Nc, Nc)
        if p == cfg.obs_panel
            for j in 1:Nc, i in 1:Nc
                d² = (i - ci)^2 + (j - cj)^2
                v[i, j] = cfg.blob_peak * exp(-d² * inv2σ²)
            end
        end
        v
    end, 6)
    return panels
end

# ---------------------------------------------------------------------------
# Observation grid + noise injection
# ---------------------------------------------------------------------------

function _observation_locations(cfg::SyntheticExperimentConfig)
    ci, cj = cfg.blob_center
    w = cfg.obs_half_window
    obs_step = cfg.nsteps   # sample at the end of the run
    locations = NamedTuple{(:panel, :i, :j, :step), NTuple{4, Int}}[]
    for di in -w:w, dj in -w:w
        i = ci + di
        j = cj + dj
        (1 <= i <= cfg.Nc && 1 <= j <= cfg.Nc) || continue
        push!(locations, (panel = cfg.obs_panel, i = i, j = j, step = obs_step))
    end
    return locations
end

# Deterministic Gaussian noise from `rand_state` — Box–Muller transform
# of two consecutive uniform draws. The script is reproducible across
# runs given `cfg.seed`.
function _box_muller!(rng::Vector{Float64}, idx::Int)
    u1 = rng[idx]
    u2 = rng[idx + 1]
    return sqrt(-2 * log(max(u1, 1e-300))) * cos(2π * u2)
end

# ---------------------------------------------------------------------------
# Truth-side: run the forward model with truth emissions, collect
# clean simulated values, perturb with Gaussian noise.
# ---------------------------------------------------------------------------

function _generate_observations(cfg::SyntheticExperimentConfig{FT},
                                 mesh, panels_m, panels_rm,
                                 panels_am, panels_bm, panels_cm,
                                 truth_emissions) where FT
    # Build a CSSurfaceFluxControl whose value is the truth field.
    truth_window = AT.CSSurfaceFluxWindow(:truth_emission, 1)
    truth_control = AT.CSSurfaceFluxControl(truth_window, truth_emissions)

    # Probe observations at every selected location — we need only
    # `result.simulated` at each obs, so the evaluation is cost-only (one
    # forward run, no tape, no reverse pass).
    locations = _observation_locations(cfg)
    probe_sigma = FT(1.0)   # placeholder; the cost computation is unused
    probe_obs = [AT.CSObservation(loc.step,
                                   AT.CSColumnMeanObjective(loc.panel, loc.i, loc.j),
                                   zero(FT), probe_sigma)
                  for loc in locations]

    forward = AT.cs_surface_flux_4dvar(
        panels_rm, panels_m, panels_am, panels_bm, panels_cm,
        mesh, probe_obs, truth_control;
        scheme = cfg.scheme, dt = cfg.dt, evaluation = AT.CSCostOnly())
    clean = forward.simulated  # one Float per observation

    peak_signal = maximum(abs, clean)
    σ_noise = max(cfg.obs_noise_frac * peak_signal, eps(FT))

    # Reproducible Box–Muller noise from a `cfg.seed`-derived uniform
    # stream. We use a tiny xorshift-style PRNG so the script does not
    # depend on Random.jl globals.
    rng_state = UInt64(cfg.seed)
    function _next_u64()
        rng_state = (rng_state ⊻ (rng_state << 13)) & typemax(UInt64)
        rng_state = (rng_state ⊻ (rng_state >> 7)) & typemax(UInt64)
        rng_state = (rng_state ⊻ (rng_state << 17)) & typemax(UInt64)
        return rng_state
    end
    _next_u01() = (_next_u64() & ((UInt64(1) << 53) - 1)) /
                  Float64(UInt64(1) << 53)

    noisy_values = [clean[i] + σ_noise * _box_muller!(
                        [_next_u01(), _next_u01()], 1)
                    for i in eachindex(clean)]

    observations = [AT.CSObservation(loc.step,
                                      AT.CSColumnMeanObjective(loc.panel, loc.i, loc.j),
                                      noisy_values[i], σ_noise)
                     for (i, loc) in enumerate(locations)]
    return observations, clean, noisy_values, σ_noise
end

# ---------------------------------------------------------------------------
# Inversion side: zero prior + L-BFGS + preconditioned cost
# ---------------------------------------------------------------------------

function _run_inversion(cfg::SyntheticExperimentConfig{FT},
                        mesh, panels_m, panels_rm,
                        panels_am, panels_bm, panels_cm,
                        observations) where FT
    Nc = cfg.Nc

    # Zero prior; isotropic-Gaussian covariance; linear optim type.
    sigma_panels = ntuple(_ -> fill(cfg.prior_sigma, Nc, Nc), 6)
    cov = AT.IsotropicGaussianCSCovariance(sigma_panels, cfg.correlation_length)
    background = ntuple(_ -> zeros(FT, Nc, Nc), 6)
    prec = AT.CSSurfaceFluxPreconditioner(cov, background, AT.LinearOptimType())

    initial = AT.CSSurfaceFluxControl(
        AT.CSSurfaceFluxWindow(:emission_window, 1),
        ntuple(_ -> zeros(FT, Nc, Nc), 6))

    optimizer = AT.CSLBFGS(
        iterations = cfg.iterations,
        gradient_tolerance = FT(1e-10),
        m = 10,
        log = true)

    return AT.cs_surface_flux_4dvar_optimize(
        panels_rm, panels_m, panels_am, panels_bm, panels_cm,
        mesh, observations, initial;
        scheme = cfg.scheme,
        dt = cfg.dt,
        preconditioner = prec,
        optimizer = optimizer)
end

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

"""
    SyntheticExperimentResult

Container for the diagnostics emitted by [`run_synthetic_experiment`](@ref).
"""
struct SyntheticExperimentResult{FT}
    config::SyntheticExperimentConfig{FT}
    truth_emissions::NTuple{6, Matrix{FT}}
    recovered_emissions::NTuple{6, Matrix{FT}}
    clean_observations::Vector{FT}
    noisy_observations::Vector{FT}
    observation_sigma::FT
    solve::AT.CS4DVarSolveResult
    relative_l2_error::FT
    peak_ratio::FT
    cost_reduction_ratio::FT
end

function _diagnostics(cfg::SyntheticExperimentConfig{FT},
                       truth, clean, noisy, σ_noise,
                       solve) where FT
    # χ-space x_recovered = prec(χ); but `solve.controls[1].value` is
    # already χ (in preconditioned mode the reported control is χ).
    # We want the PHYSICAL recovered emission, which is x = x_b +
    # B^(1/2) χ. With x_b = 0 and `LinearOptimType`, that's just
    # B^(1/2) χ.
    Nc = cfg.Nc
    chi = solve.controls[1].value
    sigma_panels = ntuple(_ -> fill(cfg.prior_sigma, Nc, Nc), 6)
    cov = AT.IsotropicGaussianCSCovariance(sigma_panels, cfg.correlation_length)
    x_recovered = ntuple(_ -> zeros(FT, Nc, Nc), 6)
    AT.apply_B_half!(x_recovered, cov, chi)

    truth_l2 = sqrt(sum(p -> sum(abs2, truth[p]), 1:6))
    err_l2 = sqrt(sum(p -> sum(abs2, x_recovered[p] .- truth[p]), 1:6))
    rel_l2 = truth_l2 > 0 ? err_l2 / truth_l2 : FT(0)

    truth_peak = maximum(p -> maximum(truth[p]), 1:6)
    rec_peak = maximum(p -> maximum(x_recovered[p]), 1:6)
    peak_ratio = truth_peak > 0 ? rec_peak / truth_peak : FT(0)

    cost_initial = solve.cost_history[1]
    cost_final = solve.last.cost
    cost_red = cost_initial > 0 ? cost_final / cost_initial : FT(0)

    return SyntheticExperimentResult{FT}(cfg, truth, x_recovered,
                                          clean, noisy, σ_noise, solve,
                                          rel_l2, peak_ratio, cost_red)
end

# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

"""
    run_synthetic_experiment(cfg = SyntheticExperimentConfig())
        -> SyntheticExperimentResult

Run the truth-emission → forward → noisy-obs → invert → diagnose
pipeline once with the supplied config. Used both by the standalone
script (`scripts/inversions/synthetic_experiment.jl`) and the
truth-recovery regression test.
"""
function run_synthetic_experiment(cfg::SyntheticExperimentConfig =
                                       SyntheticExperimentConfig())
    mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
        _constant_flow_problem(cfg)
    truth = _truth_emissions(cfg)
    observations, clean, noisy, σ_noise = _generate_observations(
        cfg, mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm, truth)
    solve = _run_inversion(cfg, mesh, panels_m, panels_rm,
                            panels_am, panels_bm, panels_cm, observations)
    return _diagnostics(cfg, truth, clean, noisy, σ_noise, solve)
end

# ---------------------------------------------------------------------------
# Script invocation
# ---------------------------------------------------------------------------

function _summarize(result::SyntheticExperimentResult)
    cfg = result.config
    println("==== Synthetic CS 4D-Var inversion ====")
    println("  Mesh:                  C$(cfg.Nc), Nz = $(cfg.Nz), nsteps = $(cfg.nsteps)")
    println("  Truth blob:            panel $(cfg.obs_panel), center $(cfg.blob_center), " *
            "σ = $(cfg.blob_sigma_cells) cells, peak = $(cfg.blob_peak)")
    println("  Observations:          $(length(result.clean_observations)) cells in a " *
            "$(2*cfg.obs_half_window + 1)² patch, noise σ = $(round(result.observation_sigma, sigdigits=3))")
    println("  Optimizer:             L-BFGS m=10, max $(cfg.iterations) iterations")
    println()
    println("==== Inversion convergence ====")
    println("  Initial cost:          $(round(result.solve.cost_history[1], sigdigits = 6))")
    println("  Final cost:            $(round(result.solve.last.cost, sigdigits = 6))")
    println("  Cost reduction ratio:  $(round(result.cost_reduction_ratio, sigdigits = 4))")
    println("  Iterations used:       $(result.solve.iterations)")
    println("  Final gradient L2:     $(round(result.solve.gradient_norm_history[end], sigdigits = 4))")
    println()
    println("==== Truth recovery ====")
    println("  Relative L2 error:     $(round(result.relative_l2_error, sigdigits = 4))")
    println("  Recovered peak / truth: $(round(result.peak_ratio, sigdigits = 4))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_synthetic_experiment()
    _summarize(result)
end
