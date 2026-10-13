#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan 26 P0.C4 — CS 4D-Var inversion driver.
#
# Usage:
#     julia --project=. scripts/inversions/cs_4dvar.jl <config.toml>
#
# Reads a TOML config, assembles a CS 4D-Var problem (mesh, met,
# observations, controls, covariance / preconditioner, optimizer),
# and runs `cs_surface_flux_4dvar_optimize`. The pipeline is built on
# the public Phase B + C surface — every component selected via type
# dispatch on TOML-named symbols, no flag branches in the driver.
#
# v1 supports a synthetic-met problem for end-to-end smoke testing
# (the Plan 26 "C24 synthetic inversion recovers prior emissions
# within 2σ" stretch goal). Real-met problems plug in by adding new
# `[meteo.source]` branches; the rest of the driver is unchanged.
#
# TOML schema:
#
#     [mesh]
#     Nc = 3                      # cubed-sphere panel side length
#     Hp = 3                      # halo width
#     float_type = "Float64"      # Float32 also supported
#
#     [time]
#     nsteps = 2
#     dt_seconds = 2.0
#
#     [meteo]
#     source = "synthetic_constant"   # only kind shipped in v1
#
#     [advection]                 # required; same parser as run_transport.jl
#     scheme = "ppm"              # upwind | slopes | ppm | linrood | none
#     limiter = "none"            # ppm only: monotone (default) | cw84 | none
#     # vertical = "fv3_kord8"    # linrood: fv3_kord8 (default) | fv3_kord8_signed | upwind
#     # ppm_order = 5             # linrood only: 5 or 7
#     # The gradient is the adjoint of this scheme, so `scheme` has no
#     # default here, and schemes without a cubed-sphere adjoint
#     # (limited slopes, the split PPM with the FV3 vertical profile, none)
#     # are rejected. Lin–Rood's vertical default is that of forward runs;
#     # the FV3 profile is transposed at its recorded limiter branches. It
#     # needs at least four layers; `synthetic_constant` has three, so a
#     # Lin–Rood inversion on it sets vertical = "upwind".
#     # `limiter = "none"` (unlimited, linear PPM) is accepted only here,
#     # not in forward runs.
#
#     [observations]
#     # Inline [[observations.entries]] blocks. Loading a v1 NetCDF
#     # observations file is not wired into this driver yet.
#     [[observations.entries]]
#     step = 2
#     objective = "layer_mean"    # or "column_mean"
#     panel = 1
#     i = 2
#     j = 2
#     level = 3                   # required for layer_mean
#     value = 0.05
#     sigma = 0.2
#
#     [control]
#     name = "both_steps"
#     steps = [1, 2]
#     normalize = true
#     initial = "zeros"           # or "background" to start at x_b
#
#     [covariance]
#     kind = "diagonal"           # "diagonal" or "isotropic_gaussian"
#     sigma_value = 1.0           # constant σ for all panels in v1
#     correlation_length_cells = 1.0   # required for "isotropic_gaussian"
#
#     [preconditioner]
#     enabled = true
#     optim_type = "linear"       # "linear" or "log_normal"
#     background_value = 0.0      # constant x_b for all panels (must
#                                 # be > 0 for log_normal)
#
#     [optimizer]
#     kind = "lbfgs"              # "gradient_descent" or "lbfgs"
#     iterations = 8
#     log = true
#     # gradient_descent-only:
#     initial_step = 0.25
#     # lbfgs-only:
#     m = 5
#     # both:
#     gradient_tolerance = 1e-10
#
# Any other table or key is an error, so nothing in the file is silently
# ignored; diffusion and convection are not wired into this driver yet.
# Keys that the selected kind does not read (e.g. `m` with gradient descent)
# are reported as warnings.
# ---------------------------------------------------------------------------

using TOML

import AtmosTransport
const AT = AtmosTransport
const Adv = AtmosTransport.Operators.Advection

# ---------------------------------------------------------------------------
# Config validation. Every table and key the driver reads is listed here;
# unknown tables and keys are errors, and known keys that the selected kind
# leaves unread are warnings, as in the forward-run checks
# (`src/Models/runner/config_keys.jl`).
# ---------------------------------------------------------------------------

const _GRADIENT_DESCENT_KEYS = ("initial_step", "min_step", "step_shrink", "line_search")
const _LBFGS_KEYS = ("m", "show_trace")
const _INVERSION_TABLE_KEYS = Dict(
    "mesh"           => ("Nc", "Hp", "float_type"),
    "time"           => ("nsteps", "dt_seconds"),
    "meteo"          => ("source",),
    "advection"      => AT.Models._ADVECTION_SPEC_KEYS,   # read by `advection_spec`
    "observations"   => ("entries",),
    "control"        => ("name", "steps", "normalize", "initial"),
    "covariance"     => ("kind", "sigma_value", "correlation_length_cells"),
    "preconditioner" => ("enabled", "optim_type", "background_value"),
    "optimizer"      => ("kind", "iterations", "log", "gradient_tolerance",
                         _GRADIENT_DESCENT_KEYS..., _LBFGS_KEYS...),
)
const _REQUIRED_TABLES = ("mesh", "time", "meteo", "advection", "observations",
                          "control", "optimizer")
const _OBSERVATION_ENTRY_KEYS = ("step", "objective", "panel", "i", "j", "level",
                                 "value", "sigma")

function _warn_unread(table, keys, label, reason)
    present = [k for k in keys if haskey(table, k)]
    isempty(present) || @warn "$(label): $(join(present, ", ")) ignored $(reason)."
    return nothing
end

function _validate_inversion_config(cfg)
    for name in ("diffusion", "convection")
        haskey(cfg, name) && throw(ArgumentError(
            "[$(name)] is not supported by the inversion driver yet; the 4D-Var cost " *
            "runs without it, so the table would be ignored."))
    end
    AT.ConfigChecks.check_known_keys(cfg, sort!(collect(keys(_INVERSION_TABLE_KEYS))),
                                     "top-level inversion config")
    for name in _REQUIRED_TABLES
        haskey(cfg, name) || throw(ArgumentError("the [$(name)] table is required"))
    end
    for (name, table) in cfg
        table isa AbstractDict || throw(ArgumentError("`$(name)` must be a TOML table"))
        AT.ConfigChecks.check_known_keys(table, _INVERSION_TABLE_KEYS[name], "[$(name)]")
    end
    for entry in get(cfg["observations"], "entries", ())
        entry isa AbstractDict || throw(ArgumentError(
            "observations.entries must be [[observations.entries]] tables"))
        AT.ConfigChecks.check_known_keys(entry, _OBSERVATION_ENTRY_KEYS,
                                         "[[observations.entries]]")
        get(entry, "objective", nothing) == "column_mean" &&
            _warn_unread(entry, ("level",), "[[observations.entries]]",
                         "with objective = \"column_mean\"")
    end

    opt = cfg["optimizer"]
    kind = get(opt, "kind", nothing)
    kind == "lbfgs" &&
        _warn_unread(opt, _GRADIENT_DESCENT_KEYS, "[optimizer]", "with kind = \"lbfgs\"")
    kind == "gradient_descent" &&
        _warn_unread(opt, _LBFGS_KEYS, "[optimizer]", "with kind = \"gradient_descent\"")
    cov = get(cfg, "covariance", nothing)
    cov !== nothing && get(cov, "kind", nothing) == "diagonal" &&
        _warn_unread(cov, ("correlation_length_cells",), "[covariance]",
                     "with kind = \"diagonal\"")
    prec = get(cfg, "preconditioner", Dict{String, Any}())
    if get(prec, "enabled", false) !== true
        _warn_unread(prec, ("optim_type", "background_value"), "[preconditioner]",
                     "while the preconditioner is disabled")
        cov === nothing ||
            @warn "[covariance] is ignored while the preconditioner is disabled."
    end
    return nothing
end

# ---------------------------------------------------------------------------
# TOML → Julia dispatch helpers
# ---------------------------------------------------------------------------

# `[advection]` goes through the forward run's parser and constructor
# (`advection_spec` + `materialize`, as in `run_transport.jl`) in the
# inversion context, then through the adjoint-support gate: the 4D-Var
# gradient is the adjoint of this scheme, so a scheme without one is an error.
function _build_advection_scheme(advection_cfg)
    spec = AT.Models.advection_spec(advection_cfg, AT.Models.InversionAdvectionContext())
    scheme = AT.Models.materialize(spec, AT.Models.CubedSphereRuntimeRecipeStyle())
    return AT.require_adjoint_support(scheme)
end

_float_type(name) = name == "Float32" ? Float32 :
                    name == "Float64" ? Float64 :
                    throw(ArgumentError("unknown float_type $(repr(name)); " *
                                        "expected 'Float32' or 'Float64'"))

function _build_objective(entry)
    kind = entry["objective"]
    if kind == "layer_mean"
        return AT.CSLayerMeanObjective(entry["panel"], entry["i"],
                                        entry["j"], entry["level"])
    elseif kind == "column_mean"
        return AT.CSColumnMeanObjective(entry["panel"], entry["i"], entry["j"])
    else
        throw(ArgumentError("unknown objective kind $(repr(kind)); " *
                            "expected 'layer_mean' or 'column_mean'"))
    end
end

function _build_observations(cfg, FT)
    obs_cfg = get(cfg, "observations", Dict())
    entries = get(obs_cfg, "entries", nothing)
    entries === nothing && throw(ArgumentError(
        "[observations.entries] required in v1 (file-loading path is " *
        "wired in P0.D1/D2 but not yet hooked into this driver)"))
    return [AT.CSObservation(entry["step"],
                              _build_objective(entry),
                              FT(entry["value"]),
                              FT(entry["sigma"]))
            for entry in entries]
end

function _build_covariance(cov_cfg, mesh, FT)
    kind = cov_cfg["kind"]
    sigma_value = FT(cov_cfg["sigma_value"])
    sigma = ntuple(_ -> fill(sigma_value, mesh.Nc, mesh.Nc), 6)
    if kind == "diagonal"
        return AT.DiagonalCSCovariance(sigma)
    elseif kind == "isotropic_gaussian"
        L = FT(cov_cfg["correlation_length_cells"])
        return AT.IsotropicGaussianCSCovariance(sigma, L)
    else
        throw(ArgumentError("unknown covariance kind $(repr(kind)); " *
                            "expected 'diagonal' or 'isotropic_gaussian'"))
    end
end

_optim_type(name) = name == "linear"     ? AT.LinearOptimType() :
                    name == "log_normal" ? AT.LogNormalOptimType() :
                    throw(ArgumentError("unknown optim_type $(repr(name)); " *
                                         "expected 'linear' or 'log_normal'"))

function _build_preconditioner(prec_cfg, covariance, mesh, FT)
    get(prec_cfg, "enabled", false) || return nothing
    optim_type = _optim_type(prec_cfg["optim_type"])
    bg_value = FT(prec_cfg["background_value"])
    background = ntuple(_ -> fill(bg_value, mesh.Nc, mesh.Nc), 6)
    return AT.CSSurfaceFluxPreconditioner(covariance, background, optim_type)
end

function _build_optimizer(opt_cfg, FT)
    kind = opt_cfg["kind"]
    log_flag = get(opt_cfg, "log", false)
    iterations = opt_cfg["iterations"]
    if kind == "gradient_descent"
        return AT.CSGradientDescent(
            iterations = iterations,
            initial_step = FT(get(opt_cfg, "initial_step", 1.0)),
            min_step = FT(get(opt_cfg, "min_step", sqrt(eps(FT)))),
            step_shrink = FT(get(opt_cfg, "step_shrink", 0.5)),
            gradient_tolerance = FT(get(opt_cfg, "gradient_tolerance", 0.0)),
            line_search = get(opt_cfg, "line_search", true),
            log = log_flag,
        )
    elseif kind == "lbfgs"
        return AT.CSLBFGS(
            iterations = iterations,
            gradient_tolerance = FT(get(opt_cfg, "gradient_tolerance", 1e-8)),
            m = get(opt_cfg, "m", 10),
            show_trace = get(opt_cfg, "show_trace", false),
            log = log_flag,
        )
    else
        throw(ArgumentError("unknown optimizer kind $(repr(kind)); " *
                            "expected 'gradient_descent' or 'lbfgs'"))
    end
end

# ---------------------------------------------------------------------------
# Met assembly — only `synthetic_constant` is shipped in v1.
# ---------------------------------------------------------------------------

function _build_synthetic_constant_problem(mesh, nsteps, FT)
    N = mesh.Nc + 2mesh.Hp
    Nz = 3
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
    return (panels_m = panels_m, panels_rm = panels_rm,
            panels_am = panels_am, panels_bm = panels_bm,
            panels_cm = panels_cm)
end

function _build_meteo(cfg, mesh, nsteps, FT)
    source = cfg["meteo"]["source"]
    if source == "synthetic_constant"
        return _build_synthetic_constant_problem(mesh, nsteps, FT)
    else
        throw(ArgumentError("unknown meteo source $(repr(source)); " *
                            "v1 only supports 'synthetic_constant'"))
    end
end

# ---------------------------------------------------------------------------
# Control assembly
# ---------------------------------------------------------------------------

function _build_initial_control(cfg, mesh, prec, FT)
    ctrl_cfg = cfg["control"]
    name = Symbol(ctrl_cfg["name"])
    steps = ctrl_cfg["steps"]
    normalize_flag = get(ctrl_cfg, "normalize", false)
    window = AT.CSSurfaceFluxWindow(name, steps; normalize = normalize_flag)

    initial = get(ctrl_cfg, "initial", "zeros")
    value = if initial == "zeros"
        ntuple(_ -> zeros(FT, mesh.Nc, mesh.Nc), 6)
    elseif initial == "background"
        # "Start at background" means the PHYSICAL starting point
        # should be `x_b`. In unconditioned mode that's literally
        # `control.value = x_b`. In preconditioned mode the control
        # value is `χ`, and `χ = 0` maps to `x = x_b` via
        # `T(0) = x_b` for Linear and `x_b ⊙ exp(0) = x_b` for
        # LogNormal — so we set χ = 0, NOT χ = x_b (which would
        # start at `T(x_b)` and leak a nonzero 0.5‖χ‖² prior cost).
        prec === nothing && throw(ArgumentError(
            "control.initial = \"background\" requires preconditioner.enabled = true " *
            "(unconditioned mode uses `initial = \"zeros\"` to start at 0)"))
        ntuple(_ -> zeros(FT, mesh.Nc, mesh.Nc), 6)
    else
        throw(ArgumentError("unknown control.initial $(repr(initial)); " *
                            "expected 'zeros' or 'background'"))
    end
    return AT.CSSurfaceFluxControl(window, value)
end

# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

"""
    run_inversion(config_path::AbstractString) -> CS4DVarSolveResult

Load a TOML config, build the 4D-Var problem, run the configured
optimizer, and return the solve result. Used by the smoke test and
the public driver entrypoint.
"""
function run_inversion(config_path::AbstractString)
    isfile(config_path) || throw(ArgumentError(
        "config not found at $(repr(config_path))"))
    cfg = TOML.parsefile(config_path)
    _validate_inversion_config(cfg)
    scheme = _build_advection_scheme(cfg["advection"])

    FT = _float_type(get(cfg["mesh"], "float_type", "Float64"))
    Nc = cfg["mesh"]["Nc"]
    Hp = get(cfg["mesh"], "Hp", 3)
    mesh = AT.CubedSphereMesh(Nc = Nc, Hp = Hp, FT = FT)

    nsteps = cfg["time"]["nsteps"]
    dt = FT(cfg["time"]["dt_seconds"])

    meteo = _build_meteo(cfg, mesh, nsteps, FT)
    observations = _build_observations(cfg, FT)
    # Build the covariance once and share it with the preconditioner so
    # the FFTW plan in `IsotropicGaussianCSCovariance` is constructed
    # exactly once per `run_inversion` call. The preconditioner builder
    # accepts an already-built covariance rather than re-parsing the
    # TOML.
    prec_cfg = get(cfg, "preconditioner", Dict("enabled" => false))
    prec_enabled = get(prec_cfg, "enabled", false)
    if prec_enabled && !haskey(cfg, "covariance")
        throw(ArgumentError(
            "[preconditioner].enabled = true requires a [covariance] section " *
            "in the TOML; otherwise the run silently degrades to unconditioned " *
            "mode. Either add [covariance] or set [preconditioner].enabled = false."))
    end
    covariance = haskey(cfg, "covariance") ?
        _build_covariance(cfg["covariance"], mesh, FT) : nothing
    prec = prec_enabled ?
        _build_preconditioner(prec_cfg, covariance, mesh, FT) : nothing
    control = _build_initial_control(cfg, mesh, prec, FT)
    optimizer = _build_optimizer(cfg["optimizer"], FT)

    solve = AT.cs_surface_flux_4dvar_optimize(
        meteo.panels_rm, meteo.panels_m,
        meteo.panels_am, meteo.panels_bm, meteo.panels_cm,
        mesh, observations, control;
        scheme = scheme,
        dt = dt,
        optimizer = optimizer,
        preconditioner = prec)

    return solve
end

# Allow the script to be `include`d for testing without running.
if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("Usage: julia --project=. " *
                                "scripts/inversions/cs_4dvar.jl <config.toml>")
    result = run_inversion(ARGS[1])
    println("Initial cost:       ", result.cost_history[1])
    println("Final cost:         ", result.last.cost)
    println("Observation cost:   ", result.last.observation_cost)
    println("Background cost:    ", result.last.background_cost)
    println("Iterations:         ", result.iterations)
    println("Final gradient L2:  ", result.gradient_norm_history[end])
end
