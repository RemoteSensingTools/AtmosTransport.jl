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
# It runs on synthetic meteorology (`synthetic_constant`, or the hydrostatic
# column `synthetic_hydrostatic` that observation files with placed point
# records need); a transport-binary meteorology source is not wired in yet.
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
#     start = "2024-06-15T00:00:00"   # required with [observations].file
#
#     [meteo]
#     source = "synthetic_constant"   # 3 layers, constant air mass, no fluxes
#     # or a hydrostatic column, the same in every cell, without fluxes:
#     # source = "synthetic_hydrostatic"
#     # interface_pressures_pa = [1000.0, 30000.0, 70000.0, 100000.0]   # total, top first
#     # specific_humidity = [1e-5, 2e-3, 1e-2]   # per layer (kg/kg); omit for a dry column
#     # temperature_kelvin = [230.0, 260.0, 285.0]   # per layer, for layer heights;
#     #                                              # omit for layer_height_temperature_kelvin
#     # The dry air mass of layer k in a cell of area A is
#     # m = (1 − q_k)(p_{k+1} − p_k) A / g.
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
#     # Either an observation file (schema v2, `read_observations`) bound
#     # with `bind_to_mesh` (needs [time].start and synthetic_hydrostatic):
#     file = "observations.nc"
#     tracer = "CO2"                  # optional: keep only this tracer
#     out_of_range_policy = "reject"  # reject | skip | clamp
#     [observations.placement]        # optional: how point records are placed
#     vertical_coordinates = ["height_above_ground", "pressure"]
#     pressure_basis = "total"        # or "dry" (needs no specific_humidity); also the
#                                     # interface pressures of averaging-kernel columns,
#                                     # so allowed without "pressure" in vertical_coordinates
#     unplaced_records = "error"      # or "skip"
#     # surface_height_file, dem_file, dem_variable,
#     # absolute_altitude_above_agl_m: as in [output.observations]
#     layer_height_temperature_kelvin = 280.0   # without meteo temperature_kelvin
#     # Averaging-kernel columns (column records with a retrieval block): the
#     # largest |model − retrieval surface pressure| (Pa) the stretch onto the
#     # retrieval column may absorb; above it a record is an error, or is left
#     # out of the cost with unplaced_records = "skip". Default: no limit.
#     max_surface_pressure_mismatch_pa = 5000.0
#
#     # or inline [[observations.entries]] blocks:
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
const _HYDROSTATIC_METEO_KEYS = ("interface_pressures_pa", "specific_humidity", "temperature_kelvin")
const _OBSERVATION_FILE_KEYS = ("file", "tracer", "out_of_range_policy", "placement")
# `[observations.placement]`: the forward sampler's placement keys, plus the
# constant layer temperature of the hypsometric heights and the surface-pressure
# limit of averaging-kernel columns.
const _OBSERVATION_PLACEMENT_KEYS = (AT.Output._VERTICAL_PLACEMENT_KEYS..., "layer_height_temperature_kelvin",
                                     "max_surface_pressure_mismatch_pa")
const _INVERSION_TABLE_KEYS = Dict(
    "mesh"           => ("Nc", "Hp", "float_type"),
    "time"           => ("nsteps", "dt_seconds", "start"),
    "meteo"          => ("source", _HYDROSTATIC_METEO_KEYS...),
    "advection"      => AT.Models._ADVECTION_SPEC_KEYS,   # read by `advection_spec`
    "observations"   => ("entries", _OBSERVATION_FILE_KEYS...),
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
    _validate_observation_source(cfg)
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

# `[observations]` takes inline entries or a file; the file's keys and the
# hydrostatic meteorology's keys are checked against their source.
function _validate_observation_source(cfg)
    obs = cfg["observations"]
    has_file, has_entries = haskey(obs, "file"), haskey(obs, "entries")
    has_file == has_entries && throw(ArgumentError(
        "[observations] needs exactly one of `file` (an observation file, schema v2) and " *
        "[[observations.entries]]"))
    if has_entries
        extra = [k for k in _OBSERVATION_FILE_KEYS if haskey(obs, k)]
        isempty(extra) || throw(ArgumentError(
            "[observations] $(join(extra, ", ")) apply to an observation `file`, not to " *
            "[[observations.entries]]"))
    else
        haskey(cfg["time"], "start") || throw(ArgumentError(
            "[observations].file needs [time].start, the model start time its records are bound to"))
        get(cfg["meteo"], "source", nothing) == "synthetic_hydrostatic" || throw(ArgumentError(
            "[observations].file needs [meteo].source = \"synthetic_hydrostatic\": its point records " *
            "are placed in a column of physical air mass (a transport-binary meteorology is not wired " *
            "into this driver yet)"))
        placement = get(obs, "placement", Dict{String, Any}())
        placement isa AbstractDict || throw(ArgumentError("[observations.placement] must be a table"))
        AT.ConfigChecks.check_known_keys(placement, _OBSERVATION_PLACEMENT_KEYS, "[observations.placement]")
        haskey(placement, "layer_height_temperature_kelvin") && haskey(cfg["meteo"], "temperature_kelvin") &&
            throw(ArgumentError(
                "[observations.placement].layer_height_temperature_kelvin has no effect with " *
                "[meteo].temperature_kelvin, the layer temperatures of the hydrostatic column"))
    end
    if get(cfg["meteo"], "source", nothing) != "synthetic_hydrostatic"
        extra = [k for k in _HYDROSTATIC_METEO_KEYS if haskey(cfg["meteo"], k)]
        isempty(extra) || throw(ArgumentError(
            "[meteo] $(join(extra, ", ")) apply to source = \"synthetic_hydrostatic\" only"))
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

function _build_observations(cfg, mesh, FT)
    obs_cfg = cfg["observations"]
    haskey(obs_cfg, "file") && return _bind_observation_file(cfg, mesh)
    return [AT.CSObservation(entry["step"],
                              _build_objective(entry),
                              FT(entry["value"]),
                              FT(entry["sigma"]))
            for entry in obs_cfg["entries"]]
end

const _PLACEMENT_LABEL = "[observations.placement]"

# `[observations].file` → `read_observations` → `bind_to_mesh`, with the
# placement of `[observations.placement]` (the forward sampler's parser) and
# its terrain files read through GriddedIO.
function _bind_observation_file(cfg, mesh)
    obs_cfg = cfg["observations"]
    raw = obs_cfg["file"]
    raw isa AbstractString && !isempty(strip(raw)) || throw(ArgumentError(
        "[observations].file must be a non-empty path; got $(repr(raw))"))
    path = AT.expand_data_path(String(raw))
    isfile(path) || throw(ArgumentError("[observations].file = $(repr(path)) does not exist"))
    placement = _observation_placement(cfg)
    for (key, file) in (("surface_height_file", placement.surface_height_file), ("dem_file", placement.dem_file))
        file === nothing || isfile(file) || throw(ArgumentError(
            "$(_PLACEMENT_LABEL).$(key) = $(repr(file)) does not exist"))
    end
    tracer = get(obs_cfg, "tracer", nothing)
    tracer === nothing || tracer isa AbstractString || throw(ArgumentError(
        "[observations].tracer must be a tracer name; got $(repr(tracer))"))
    return AT.bind_to_mesh(AT.read_observations(path), mesh, _start_time(cfg), cfg["time"]["dt_seconds"];
                           nsteps = cfg["time"]["nsteps"], tracer_filter = tracer,
                           out_of_range_policy = get(obs_cfg, "out_of_range_policy", "reject"),
                           placement, terrain = AT.Models.DrivenRunner._observation_terrain(placement, mesh),
                           max_surface_pressure_mismatch_pa = _max_surface_pressure_mismatch(cfg))
end

# The surface-pressure limit of averaging-kernel columns (Pa), Inf without one.
function _max_surface_pressure_mismatch(cfg)
    placement = get(cfg["observations"], "placement", Dict{String, Any}())
    value = get(placement, "max_surface_pressure_mismatch_pa", Inf)
    value isa Real && !(value isa Bool) && value > 0 || throw(ArgumentError(
        "$(_PLACEMENT_LABEL).max_surface_pressure_mismatch_pa must be a positive pressure (Pa); got $(repr(value))"))
    return Float64(value)
end

_observation_placement(cfg) =
    AT.Output._parse_vertical_placement(get(cfg["observations"], "placement", Dict{String, Any}()),
                                        _PLACEMENT_LABEL; pressure_basis_for_columns = true)

_start_time(cfg) = AT.Output._parse_iso_utc(cfg["time"]["start"], "[time].start")

# Near-surface layer temperature of the hypsometric heights, as the forward
# sampler's `layer_height_temperature_kelvin` default.
const _DEFAULT_LAYER_HEIGHT_TEMPERATURE = 280.0

function _layer_height_temperature(cfg)
    placement = get(get(cfg, "observations", Dict()), "placement", Dict())
    value = get(placement, "layer_height_temperature_kelvin", _DEFAULT_LAYER_HEIGHT_TEMPERATURE)
    value isa Real && !(value isa Bool) && isfinite(value) && value > 0 || throw(ArgumentError(
        "$(_PLACEMENT_LABEL).layer_height_temperature_kelvin must be a positive number; got $(repr(value))"))
    return Float64(value)
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

# A hydrostatic column, the same in every cell, at rest: the dry air mass of
# layer k is m = (1 − q_k)(p_{k+1} − p_k) A / g for the total interface
# pressures p (top first), the specific humidity q (zero without
# `specific_humidity`) and the cell area A. The column meteorology places
# point observations: model top p[1], g, the layer temperature (the
# `temperature_kelvin` profile, else the constant
# `layer_height_temperature_kelvin`) and the humidity (absent for a dry
# column, so pressure placement then needs pressure_basis = "dry").
function _build_synthetic_hydrostatic_problem(cfg, mesh, nsteps, FT)
    meteo = cfg["meteo"]
    p_half = _float_vector(meteo, "interface_pressures_pa")
    Nz = length(p_half) - 1
    Nz >= 1 && first(p_half) >= 0 && all(>(0), diff(p_half)) || throw(ArgumentError(
        "[meteo].interface_pressures_pa must list at least two total interface pressures (Pa), " *
        "top first, increasing, the top >= 0; got $(p_half)"))
    q = haskey(meteo, "specific_humidity") ? _float_vector(meteo, "specific_humidity") : nothing
    q === nothing || length(q) == Nz && all(x -> 0 <= x < 1, q) || throw(ArgumentError(
        "[meteo].specific_humidity must give $(Nz) layer values in [0, 1) kg/kg; got $(q)"))
    temperature = if haskey(meteo, "temperature_kelvin")
        t = _float_vector(meteo, "temperature_kelvin")
        length(t) == Nz && all(x -> isfinite(x) && x > 0, t) || throw(ArgumentError(
            "[meteo].temperature_kelvin must give $(Nz) positive layer temperatures; got $(t)"))
        AT.Output.ProfileLayerTemperature(t)
    else
        _layer_height_temperature(cfg)
    end
    g = AT.Parameters.STANDARD_GRAVITY
    Hp = mesh.Hp
    N = mesh.Nc + 2Hp
    dry = q === nothing ? ones(Nz) : 1 .- q
    area(i, j) = AT.cell_area(mesh, clamp(i - Hp, 1, mesh.Nc), clamp(j - Hp, 1, mesh.Nc))
    panels_m = ntuple(_ -> [FT(dry[k] * (p_half[k + 1] - p_half[k]) * area(i, j) / g)
                            for i in 1:N, j in 1:N, k in 1:Nz], 6)
    panels_rm = ntuple(_ -> zeros(FT, N, N, Nz), 6)
    Adv.fill_panel_halos!(panels_m, mesh; dir=0)
    panels_am = [ntuple(_ -> zeros(FT, N + 1, N, Nz), 6) for _ in 1:nsteps]
    panels_bm = [ntuple(_ -> zeros(FT, N, N + 1, Nz), 6) for _ in 1:nsteps]
    panels_cm = [ntuple(_ -> zeros(FT, N, N, Nz + 1), 6) for _ in 1:nsteps]
    humidity = q === nothing ? nothing : ntuple(_ -> repeat(reshape(FT.(q), 1, 1, Nz), N, N, 1), 6)
    column_met = AT.CSObservationColumnMet(; p_top = first(p_half), gravity = g, temperature, humidity)
    return (panels_m = panels_m, panels_rm = panels_rm,
            panels_am = panels_am, panels_bm = panels_bm,
            panels_cm = panels_cm, column_met = column_met)
end

function _float_vector(table, key)
    haskey(table, key) || throw(ArgumentError(
        "[meteo].$(key) is required with source = \"synthetic_hydrostatic\""))
    raw = table[key]
    raw isa AbstractVector && all(x -> x isa Real && !(x isa Bool), raw) || throw(ArgumentError(
        "[meteo].$(key) must be an array of numbers; got $(repr(raw))"))
    return Float64.(raw)
end

function _build_meteo(cfg, mesh, nsteps, FT)
    source = cfg["meteo"]["source"]
    if source == "synthetic_constant"
        return (; _build_synthetic_constant_problem(mesh, nsteps, FT)..., column_met = nothing)
    elseif source == "synthetic_hydrostatic"
        return _build_synthetic_hydrostatic_problem(cfg, mesh, nsteps, FT)
    else
        throw(ArgumentError("unknown meteo source $(repr(source)); " *
                            "expected 'synthetic_constant' or 'synthetic_hydrostatic'"))
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
    observations = _build_observations(cfg, mesh, FT)
    isempty(observations) && throw(ArgumentError(
        "no observation is left to assimilate (check [observations].tracer, out_of_range_policy, " *
        "the records' flags and [observations.placement].unplaced_records)"))
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
        preconditioner = prec,
        column_met = meteo.column_met)

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
