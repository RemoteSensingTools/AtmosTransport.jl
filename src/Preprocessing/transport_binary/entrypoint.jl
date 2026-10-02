# =========================================================================
# TOML-driven preprocessing entry point — THE canonical orchestrator.
#
# Every supported source (ERA5 spectral, GEOS-IT, GEOS-FP, MERRA-2 …) and
# every supported target topology (LatLon, ReducedGaussian, CubedSphere)
# routes through this single function. New sources plug in via
# `AbstractMetSettings` + `load_met_settings`, never via a parallel CLI or
# orchestrator (architectural invariant — see plan geos-followups).
# =========================================================================

# ---------------------------------------------------------------------------
# Source detection. A config that declares `[source].toml = "..."` is
# routed through the typed `AbstractMetSettings` factory; otherwise the
# ERA5-spectral config path wraps historical settings in
# `ERA5SpectralSettings`. Both paths converge on
# `process_day(date, grid, settings, vertical; ...)` for actual work.
# ---------------------------------------------------------------------------

@inline _is_native_source_cfg(cfg::AbstractDict) =
    haskey(cfg, "source") && haskey(cfg["source"], "toml")

# ---------------------------------------------------------------------------
# Date selection — supports `--day YYYY-MM-DD` (single) and
# `--start ... --end ...` (range, inclusive).
# ---------------------------------------------------------------------------

function _resolve_dates_native(cfg::AbstractDict;
                               day_override = nothing,
                               start_date   = nothing,
                               end_date     = nothing)
    if day_override !== nothing
        return [Date(String(day_override))]
    elseif start_date !== nothing && end_date !== nothing
        return collect(Date(String(start_date)):Day(1):Date(String(end_date)))
    elseif haskey(cfg, "input") && haskey(cfg["input"], "start_date") &&
           haskey(cfg["input"], "end_date")
        return collect(Date(String(cfg["input"]["start_date"])):Day(1):Date(String(cfg["input"]["end_date"])))
    end
    error("Native-source preprocessing needs `--day`, `--start/--end`, " *
          "or `[input].start_date`/`[input].end_date` in the TOML.")
end

# ---------------------------------------------------------------------------
# Output path resolution for the native-source path. The TOML's
# `[output].directory` is the destination; the date and float type pick the
# filename. Sources can override via `_native_output_filename(settings, date, FT)`.
# ---------------------------------------------------------------------------

"""
    _native_output_filename(settings, date, FT) -> String

Per-source output filename for native-source preprocessing. Concrete
sources override this in their own files (e.g. `sources/geos.jl`); the
default is a source-agnostic prefix.
"""
_native_output_filename(::AbstractMetSettings, date::Date, FT::Type) =
    "transport_$(Dates.format(date, "yyyymmdd"))_$(FT === Float32 ? "float32" : "float64").bin"

function _native_output_path(cfg::AbstractDict, settings::AbstractMetSettings,
                             date::Date, FT::Type)
    out_dir = expand_data_path(String(cfg["output"]["directory"]))
    mkpath(out_dir)
    return joinpath(out_dir, _native_output_filename(settings, date, FT))
end

# ---------------------------------------------------------------------------
# Float-type / numerics resolution (shared between native and spectral paths).
# ---------------------------------------------------------------------------

function _resolve_float_type(cfg::AbstractDict)
    s = String(get(get(cfg, "numerics", Dict()), "float_type", "Float64"))
    return s == "Float32" ? Float32 : Float64
end

_resolve_dt_met(cfg::AbstractDict) =
    Float64(get(get(cfg, "numerics", Dict()), "dt_met_seconds", 3600.0))

_resolve_mass_basis(cfg::AbstractDict) =
    Symbol(get(get(cfg, "output", Dict()), "mass_basis", "dry"))

_resolve_chain_mass(cfg::AbstractDict) =
    _config_bool(get(cfg, "numerics", Dict()), "chain_mass", true, "[numerics].chain_mass")

# Canonical CS substep-positivity contract knobs. Defaults match the gate the
# regrid path enforces — every CS-producing preprocessor
# (spectral, regrid, GEOS-native) reads them from the same `[numerics]` block
# so a config can't silently bypass the contract on one path while honoring it
# on another.
#
# Validation: `positivity_cfl_limit` must be a finite positive value bounded
# by 1.0. A `0.0` or negative value would make
# `summarize_cs_positivity_status` divide by zero (or produce a negative
# recommendation) AFTER a contract violation has been recorded — by then the
# preprocessor has already paid the cost of the loop, so we'd rather refuse
# to start than throw an `InexactError` at the end. Values > 1.0 are also
# nonsensical (the runtime's `_cs_static_subcycle_count` only protects
# against outgoing < cell mass).
function _resolve_positivity_cfl_limit(cfg::AbstractDict)
    raw = get(get(cfg, "numerics", Dict()), "positivity_cfl_limit", 0.95)
    limit = Float64(raw)
    isfinite(limit) && 0 < limit <= 1 ||
        error("Invalid `[numerics].positivity_cfl_limit = $(raw)`: must be a " *
              "finite value in (0, 1]. The default `0.95` is what the regrid " *
              "path has enforced since plan 39.")
    return limit
end

_resolve_require_substep_positivity(cfg::AbstractDict) =
    _config_bool(get(cfg, "numerics", Dict()), "require_substep_positivity", true,
                 "[numerics].require_substep_positivity")

function _resolve_substep_schedule_policy(cfg::AbstractDict,
                                          positivity_cfl_limit::Real)
    numerics = get(cfg, "numerics", Dict())
    mode = Symbol(lowercase(String(get(numerics, "substep_schedule", "adaptive_cfl"))))
    adaptive = mode in (:adaptive, :adaptive_cfl, :cfl)
    mode in (:fixed, :constant, :adaptive, :adaptive_cfl, :cfl) ||
        error("Invalid `[numerics].substep_schedule = $(mode)`: expected " *
              "`fixed` or `adaptive_cfl`.")
    target = Float64(get(numerics, "substep_cfl_target",
                         Float64(positivity_cfl_limit)))
    isfinite(target) && target > 0 ||
        error("Invalid `[numerics].substep_cfl_target = $(target)`: must be finite and > 0.")
    min_steps = Int(get(numerics, "min_steps_per_window", 1))
    max_steps = Int(get(numerics, "max_steps_per_window", typemax(Int)))
    1 <= min_steps <= max_steps ||
        error("Invalid adaptive step bounds: min_steps_per_window=$(min_steps), " *
              "max_steps_per_window=$(max_steps).")
    target <= Float64(positivity_cfl_limit) || @warn(
        "`substep_cfl_target` is looser than `positivity_cfl_limit`; " *
        "adaptive scheduling may still be rejected by the final positivity contract",
        substep_cfl_target = target,
        positivity_cfl_limit = positivity_cfl_limit)
    return SubstepSchedulePolicy(
        adaptive_substeps = adaptive,
        substep_cfl_target = target,
        min_steps_per_window = min_steps,
        max_steps_per_window = max_steps,
    )
end

function _vertical_float(raw, key::AbstractString)
    value = if raw isa AbstractString
        s = strip(raw)
        lowercase(s) in ("inf", "+inf", "infinity", "+infinity") && return Inf
        parse(Float64, s)
    else
        Float64(raw)
    end
    isfinite(value) || value == Inf ||
        error("Invalid `[vertical].$(key) = $(raw)`: must be finite or `Inf`.")
    return value
end

function _build_native_vertical_setup(cfg_vertical::AbstractDict,
                                      vc_raw::HybridSigmaPressure,
                                      ::Type{FT}) where FT
    vc = HybridSigmaPressure(FT.(vc_raw.A), FT.(vc_raw.B))
    transform_name = lowercase(String(get(cfg_vertical, "transform", "identity")))
    transform = if transform_name in ("identity", "none")
        IdentityVertical()
    elseif transform_name in ("level_selection", "echlevs")
        preset = get(cfg_vertical, "preset",
                     get(cfg_vertical, "echlevs", get(cfg_vertical, "name", "")))
        isempty(strip(String(preset))) &&
            error("`[vertical].transform = \"level_selection\"` requires " *
                  "`preset = \"ml137_…\"`.")
        LevelSelection(echlevs_preset(String(preset)))
    elseif transform_name in ("merge_above_pressure", "merge_above_pressure_pa")
        pressure_pa = if haskey(cfg_vertical, "pressure_Pa")
            _vertical_float(cfg_vertical["pressure_Pa"], "pressure_Pa")
        elseif haskey(cfg_vertical, "pressure_hPa")
            100.0 * _vertical_float(cfg_vertical["pressure_hPa"], "pressure_hPa")
        else
            error("`[vertical].transform = \"merge_above_pressure\"` requires " *
                  "`pressure_Pa` or `pressure_hPa`.")
        end
        MergeAbovePressure(
            pressure_Pa = pressure_pa,
            target_min_thickness_Pa =
                _vertical_float(get(cfg_vertical, "target_min_thickness_Pa", Inf),
                                "target_min_thickness_Pa"),
            reference_surface_pressure_Pa =
                _vertical_float(get(cfg_vertical, "reference_surface_pressure_Pa", 101325.0),
                                "reference_surface_pressure_Pa"))
    elseif transform_name in ("merge_layers_thinner_than", "thin_level_merge")
        MergeLayersThinnerThan(
            min_thickness_Pa =
                _vertical_float(get(cfg_vertical, "min_thickness_Pa",
                                    get(cfg_vertical, "target_min_thickness_Pa", 50.0)),
                                "min_thickness_Pa"),
            reference_surface_pressure_Pa =
                _vertical_float(get(cfg_vertical, "reference_surface_pressure_Pa", 101325.0),
                                "reference_surface_pressure_Pa"))
    else
        error("Unsupported native `[vertical].transform = $(repr(transform_name))`. " *
              "Supported: identity, level_selection, merge_above_pressure, " *
              "merge_layers_thinner_than.")
    end

    plan = plan_vertical(transform, vc)
    return (
        plan = plan,
        transform = transform,
        merged_vc = plan.merged_vc,
        native_vc = plan.native_vc,
        merge_map = plan.merge_map,
        groups = plan.groups,
        vertical_mapping_method = Symbol(transform_name),
        Nz = plan.Nz_output,
        Nz_native = plan.Nz_native,
    )
end

# ---------------------------------------------------------------------------
# Native-source preprocessor: typed `AbstractMetSettings` + cross-day state
# carry (e.g. GEOS pressure-fixer chained mass).
# ---------------------------------------------------------------------------

function _native_mass_fix_target_kg(cfg::AbstractDict, grid)
    mass_fix_cfg = get(cfg, "mass_fix", Dict{String, Any}())
    _config_bool(mass_fix_cfg, "enable", false, "[mass_fix].enable") || return NaN
    haskey(mass_fix_cfg, "target_total_kg") &&
        return Float64(mass_fix_cfg["target_total_kg"])

    mode = Symbol(replace(lowercase(String(get(mass_fix_cfg, "mode", "target_ps_dry"))),
                          '-' => '_'))
    if mode in (:target_ps_dry, :target_dry_ps, :fixed_dry_ps)
        hasproperty(grid, :mesh) && hasproperty(grid.mesh, :cell_areas) ||
            error("native-source [mass_fix].mode=\"target_ps_dry\" requires a grid with cell_areas")
        target_ps_dry_pa = Float64(get(mass_fix_cfg, "target_ps_dry_pa", 98726.0))
        total_area = 6.0 * sum(Float64, grid.mesh.cell_areas)
        return target_ps_dry_pa * total_area / GRAV
    elseif mode === :initial_endpoint
        return NaN
    else
        error("native-source [mass_fix].mode must be \"target_ps_dry\" or " *
              "\"initial_endpoint\"; got $(repr(String(get(mass_fix_cfg, "mode", ""))))")
    end
end

function _process_day_native(cfg::AbstractDict;
                             day_override = nothing,
                             start_date   = nothing,
                             end_date     = nothing)
    src_cfg = cfg["source"]
    toml_relpath = String(src_cfg["toml"])
    toml_path = isabspath(toml_relpath) ? toml_relpath :
                joinpath(@__DIR__, "..", "..", "..", toml_relpath)

    # Single source of truth for the float type: build the grid AND the
    # source settings against the same `FT` so cell areas and the reader's
    # mass buffers share an element type. Without this, Float32 configs
    # trip a `MethodError` in `_delp_pa_to_air_mass_kg!` because
    # cell_areas was Matrix{Float64}.
    FT   = _resolve_float_type(cfg)
    grid = build_target_geometry(cfg["grid"], FT)
    ensure_supported_target(grid)

    # Single source of truth for hybrid coefficients: the source descriptor
    # owns it. Overrides flow back into `settings.coefficients_file` so the
    # reader (`open_day` → `endpoint_dry_mass!`) and the writer's vertical
    # setup never desync.
    settings_kwargs = (root_dir = expand_data_path(String(src_cfg["root_dir"])),)
    if haskey(src_cfg, "include_surface")
        settings_kwargs = (settings_kwargs..., include_surface = _config_bool(
            src_cfg["include_surface"], "[source].include_surface"))
    end
    if haskey(src_cfg, "include_convection")
        settings_kwargs = (settings_kwargs..., include_convection = _config_bool(
            src_cfg["include_convection"], "[source].include_convection"))
    end
    if haskey(src_cfg, "include_vdiff_fields")
        settings_kwargs = (settings_kwargs..., include_vdiff_fields = _config_bool(
            src_cfg["include_vdiff_fields"], "[source].include_vdiff_fields"))
    end
    for key in ("physics_dir", "surface_dir")
        if haskey(src_cfg, key)
            settings_kwargs = (settings_kwargs..., physics_dir = expand_data_path(String(src_cfg[key])))
            break
        end
    end
    if haskey(src_cfg, "physics_layout")
        settings_kwargs = (settings_kwargs..., physics_layout = Symbol(src_cfg["physics_layout"]))
    end
    cfg_vertical = get(cfg, "vertical", Dict())
    if haskey(cfg_vertical, "coefficients")
        settings_kwargs = (settings_kwargs..., coefficients_file =
                           expand_data_path(String(cfg_vertical["coefficients"])))
    end
    settings = load_met_settings(toml_path; settings_kwargs...)

    vc = load_hybrid_coefficients(expand_data_path(settings.coefficients_file))
    vertical = _build_native_vertical_setup(cfg_vertical, vc, FT)

    mass_basis     = _resolve_mass_basis(cfg)
    dt_met_seconds = _resolve_dt_met(cfg)
    chain_mass     = _resolve_chain_mass(cfg)
    positivity_cfl_limit       = _resolve_positivity_cfl_limit(cfg)
    require_substep_positivity = _resolve_require_substep_positivity(cfg)
    substep_policy = _resolve_substep_schedule_policy(cfg, positivity_cfl_limit)
    numerics_cfg = get(cfg, "numerics", Dict{String, Any}())
    balance_mode_raw = lowercase(String(get(numerics_cfg, "geos_balance_mode", "column")))
    balance_mode = if balance_mode_raw in ("column", "column_poisson")
        :column
    elseif balance_mode_raw in ("per_layer", "layer", "layer_local", "global")
        :per_layer
    else
        error("[numerics].geos_balance_mode must be \"column\" or \"per_layer\"; got $(repr(balance_mode_raw))")
    end
    cm_closure_raw = lowercase(String(get(numerics_cfg, "geos_cm_closure", "endpoint_balanced")))
    cm_closure = if cm_closure_raw in ("endpoint_balanced", "endpoint", "diagnose", "balanced")
        :endpoint_balanced
    elseif cm_closure_raw in ("pressure_fixer", "pressurefixer", "fv3", "native")
        :pressure_fixer
    elseif cm_closure_raw in ("moisture_filtered", "moisturefiltered", "filtered", "filtered_endpoint")
        :moisture_filtered
    elseif cm_closure_raw in ("pfix_corrected", "pfixcorrected", "pfix", "pressure_fixer_corrected")
        :pfix_corrected
    elseif cm_closure_raw in ("omega_regularized", "omegaregularized", "omega_reg",
                              "regularized_omega", "omega_consistent",
                              "omegaconsistent", "omega", "omega_cm")
        :omega_regularized
    elseif cm_closure_raw in ("omega_full_replacement", "omega_full", "full_omega")
        :omega_full_replacement
    else
        error("[numerics].geos_cm_closure must be \"endpoint_balanced\", \"pressure_fixer\", " *
              "\"moisture_filtered\", \"pfix_corrected\", \"omega_regularized\", or " *
              "\"omega_full_replacement\"; got $(repr(cm_closure_raw))")
    end
    # cm-closure status (2026-06-03): `:endpoint_balanced` is the ONLY validated
    # production default. ALL of `:pressure_fixer`, `:moisture_filtered`, and
    # `:pfix_corrected` are DIAGNOSTIC-ONLY — they explore the SH-UTLS fingering
    # but each fails at the tracer level (see docs/src/preprocessing/geos_native_cs.md):
    #   :pressure_fixer    → smooth cm but unbounded ps drift / negative mass.
    #   :moisture_filtered → VERIFIED NO-OP (balancing re-injects the noise).
    #   :pfix_corrected    → reduces upper-UTLS noise but makes ~164-280 hPa WORSE
    #                        (the drift correction emits a spurious surface cm flux),
    #                        and chain_mass=true accumulates negative UTLS mass.
    #   :omega_full_replacement → replaces all resolved convergence with OMEGA
    #                        (vdiv=+vdiv_om, smooth); continuity exact by
    #                        construction, but changes bottom-layer horizontal
    #                        fluxes by 2-7× and spuriously vents surface plumes.
    #                        It is retained only for reproducing the failed test.
    #   :omega_regularized → preserves endpoint-balanced resolved scales and uses
    #                        OMEGA only as a pressure-local, high-pass, capped prior.
    # The fingering is the intrinsic MFXC↔DELP residual; the validated cure is
    # input-side (wind-derived / ERA5). `:omega_regularized` is the conservative
    # native-cube candidate that avoids the full OMEGA pathway's oversmoothing.
    if cm_closure === :omega_full_replacement
        @warn "[numerics].geos_cm_closure=:omega_full_replacement is a known-failed " *
              "diagnostic: it rewrites PBL horizontal fluxes by 2-7× and corrupts " *
              "surface-plume transport. Use :omega_regularized or :endpoint_balanced."
    elseif cm_closure !== :endpoint_balanced
        @warn "[numerics].geos_cm_closure=$(cm_closure) is DIAGNOSTIC/CANDIDATE, NOT " *
              "the science-validated production default (see " *
              "docs/src/preprocessing/geos_native_cs.md). Use " *
              ":endpoint_balanced for production and the ERA5/wind-derived path " *
              "for UTLS-sensitive science. :omega_regularized limits OMEGA to a " *
              "UTLS high-pass prior and still requires tracer validation."
    end
    # Spatial low-pass sweeps: the `:moisture_filtered` residual smoother and the
    # `:pfix_corrected` column-drift smoother (ignored by the other closures).
    smooth_iters = Int(get(numerics_cfg, "geos_moisture_filter_smooth_iters", 8))
    smooth_iters >= 0 ||
        error("[numerics].geos_moisture_filter_smooth_iters must be ≥ 0; got $(smooth_iters)")
    omega_cfg = get(numerics_cfg, "omega_regularization", Dict{String, Any}())
    omega_cfg isa AbstractDict ||
        error("[numerics.omega_regularization] must be a TOML table")
    taper_raw = get(omega_cfg, "pressure_taper_hpa", [50.0, 80.0, 300.0, 350.0])
    taper_raw isa AbstractVector && length(taper_raw) == 4 ||
        error("[numerics.omega_regularization].pressure_taper_hpa must contain four values")
    omega_regularization = OmegaRegularization(
        pressure_taper_hpa = ntuple(i -> Float64(taper_raw[i]), 4),
        smoothing_steps = Int(get(omega_cfg, "smoothing_steps", 3)),
        smoothing_fraction = Float64(get(omega_cfg, "smoothing_fraction", 0.10)),
        max_relative_flux_correction = Float64(
            get(omega_cfg, "max_relative_flux_correction", 0.10)),
        max_bottom_flux_correction = Float64(
            get(omega_cfg, "max_bottom_flux_correction", 0.01)),
    )
    cm_closure === :omega_regularized &&
        _validate_omega_regularization(omega_regularization)
    mass_fix_cfg = get(cfg, "mass_fix", Dict{String, Any}())
    global_mass_pin = _config_bool(mass_fix_cfg, "enable", false, "[mass_fix].enable")
    _uses_omega(cm_closure) && !global_mass_pin &&
        error("OMEGA-based GEOS cm closures require [mass_fix].enable=true so " *
              "the per-level Poisson targets have zero global column tendency")
    configured_global_mass_target_kg = _native_mass_fix_target_kg(cfg, grid)
    ensure_preprocessor_pair_supported(grid, settings; context = "native-source")

    dates = _resolve_dates_native(cfg; day_override, start_date, end_date)

    @info @sprintf("Preprocessor: %s  → %s  Nz=%d  FT=%s  %d day(s)",
                   typeof(settings), typeof(grid),
                   vertical.Nz, FT, length(dates))
    vertical.Nz == vertical.Nz_native ||
        @info @sprintf("Vertical transform: %s  native Nz=%d → output Nz=%d",
                       typeof(vertical.transform), vertical.Nz_native, vertical.Nz)
    @info "Unified preprocessor driver enabled for $(nameof(typeof(settings))) → $(grid_kind(grid))"

    t_total = time()

    # Per-day worker. Builds a fresh `target_grid` per call so threads can't
    # race on the mutable `poisson_scratch` inside the geometry. Returns the
    # `process_day` result so the serial path can chain `seed_m`.
    run_one_day = function (idx::Int, d::Date, seed_m_in)
        out_path = _native_output_path(cfg, settings, d, FT)
        @info "[$idx/$(length(dates))] $(d) → $(out_path)"
        day_grid = build_target_geometry(cfg["grid"], FT)
        ensure_supported_target(day_grid)
        day_kwargs = (
            out_path        = out_path,
            dt_met_seconds  = dt_met_seconds,
            FT              = FT,
            mass_basis      = mass_basis,
            chain_mass      = chain_mass,
            positivity_cfl_limit       = positivity_cfl_limit,
            require_substep_positivity = require_substep_positivity,
            adaptive_substeps          = substep_policy.adaptive_substeps,
            substep_cfl_target         = substep_policy.substep_cfl_target,
            min_steps_per_window       = substep_policy.min_steps_per_window,
            max_steps_per_window       = substep_policy.max_steps_per_window,
            seed_m          = seed_m_in,
            global_mass_pin = global_mass_pin,
            global_mass_target_kg = configured_global_mass_target_kg,
            balance_mode = balance_mode,
            cm_closure = cm_closure,
            smooth_iters = smooth_iters,
            omega_regularization = omega_regularization,
        )
        return process_day(d, day_grid, settings, vertical; day_kwargs...)
    end

    threaded = Threads.nthreads() > 1 && length(dates) > 1 &&
               supports_day_threading(settings) && !chain_mass &&
               !(global_mass_pin && !isfinite(configured_global_mass_target_kg))
    # When the day loop itself is threaded, the inner OMEGA reconstruction
    # per-level Poisson loop must run SERIAL so day-workers don't each re-grab
    # the whole pool (oversubscription). In the single-day-per-process (`--day`)
    # path the day loop is serial, so the level solve keeps the full pool.
    _OMEGA_LEVEL_PARALLEL[] = !threaded
    if threaded
        # Cold-cache pre-warm: regridder weight caches + JIT specializations
        # land during day 1 serial, then days 2..N run concurrently. Without
        # the warm-up, N threads race to build the same regridder cache file.
        @info @sprintf("  Day-threading enabled (%d threads). Pre-warming on day 1 serial.",
                       Threads.nthreads())
        run_one_day(1, dates[1], nothing)
        if length(dates) >= 2
            remaining = collect(2:length(dates))
            @info @sprintf("  Threaded loop over days 2..%d (n=%d).",
                           length(dates), length(remaining))
            Threads.@threads for idx in remaining
                run_one_day(idx, dates[idx], nothing)
            end
        end
    else
        seed_m = nothing                  # source-defined cross-day state (e.g. GEOS PF endpoint)
        global_mass_target_kg = configured_global_mass_target_kg
        for (idx, d) in enumerate(dates)
            day_grid = build_target_geometry(cfg["grid"], FT)
            ensure_supported_target(day_grid)
            out_path = _native_output_path(cfg, settings, d, FT)
            @info "[$idx/$(length(dates))] $(d) → $(out_path)"
            day_kwargs = (
                out_path        = out_path,
                dt_met_seconds  = dt_met_seconds,
                FT              = FT,
                mass_basis      = mass_basis,
                chain_mass      = chain_mass,
                positivity_cfl_limit       = positivity_cfl_limit,
                require_substep_positivity = require_substep_positivity,
                adaptive_substeps          = substep_policy.adaptive_substeps,
                substep_cfl_target         = substep_policy.substep_cfl_target,
                min_steps_per_window       = substep_policy.min_steps_per_window,
                max_steps_per_window       = substep_policy.max_steps_per_window,
                seed_m          = seed_m,
                global_mass_pin = global_mass_pin,
                global_mass_target_kg = global_mass_target_kg,
                balance_mode = balance_mode,
                cm_closure = cm_closure,
                smooth_iters = smooth_iters,
                omega_regularization = omega_regularization,
            )
            result = process_day(d, day_grid, settings, vertical; day_kwargs...)
            seed_m = get(result, :final_m, nothing)
            global_mass_target_kg = get(result, :global_mass_target_kg, global_mass_target_kg)
        end
    end

    elapsed = time() - t_total
    @info @sprintf("All done! %d days in %.1fs (%.1fs/day)",
                   length(dates), elapsed, elapsed / max(length(dates), 1))
    return nothing
end

# ---------------------------------------------------------------------------
# ERA5-spectral preprocessor: typed `ERA5SpectralSettings` wrapper over the
# historical settings bundle. Existing ERA5 configs keep working unchanged;
# new met sources must use the typed native-source path.
# ---------------------------------------------------------------------------

function _process_day_spectral(cfg::AbstractDict, grid::AbstractTargetGeometry;
                               day_override = nothing,
                               start_date   = nothing,
                               end_date     = nothing)
    settings_nt = resolve_runtime_settings(cfg)
    settings = ERA5SpectralSettings(
        merge(settings_nt, (T_target = target_spectral_truncation(grid),)))
    vertical = build_vertical_setup(settings.coeff_path, settings.level_range,
                                    settings.min_dp, cfg["grid"])

    log_preprocessor_configuration(settings, grid, vertical)

    dates = if day_override !== nothing
        [Date(String(day_override))]
    elseif start_date !== nothing || end_date !== nothing
        all_dates = sort(available_spectral_dates(settings.spectral_dir))
        start_d = start_date === nothing ? first(all_dates) : Date(String(start_date))
        end_d   = end_date   === nothing ? last(all_dates)  : Date(String(end_date))
        start_d <= end_d ||
            error("--start $(start_d) must be <= --end $(end_d)")
        [d for d in all_dates if start_d <= d <= end_d]
    else
        day_filter = parse_day_filter(day_override === nothing ? String[] : ["--day", day_override])
        select_processing_dates(available_spectral_dates(settings.spectral_dir), day_filter)
    end

    positivity_cfl_limit       = _resolve_positivity_cfl_limit(cfg)
    require_substep_positivity = _resolve_require_substep_positivity(cfg)
    substep_policy             = _resolve_substep_schedule_policy(cfg, positivity_cfl_limit)
    ensure_preprocessor_pair_supported(grid, settings; context = "ERA5 spectral")

    @info @sprintf("Processing %d days: %s to %s", length(dates), first(dates), last(dates))
    @info "Unified preprocessor driver enabled for ERA5 spectral → $(grid_kind(grid))"
    t_total = time()
    run_cache = PreprocessorRunCache(typeof(grid), settings.output_float_type)
    for (idx, date) in enumerate(dates)
        @info @sprintf("[%d/%d] %s", idx, length(dates), date)
        next_day_h0 = next_day_hour0(date, dates, settings.spectral_dir, settings.T_target;
                                     cache_dir = settings.spectral_cache_dir)
        next_day_h0 !== nothing && @info("  Next day hour 0 available for last-window delta")
        day_kwargs = (
            positivity_cfl_limit       = positivity_cfl_limit,
            require_substep_positivity = require_substep_positivity,
            substep_policy             = substep_policy,
            next_day_hour0             = next_day_h0,
            run_cache                  = run_cache,
        )
        process_day(date, grid, settings, vertical; day_kwargs...)
    end
    elapsed = time() - t_total
    @info @sprintf("All done! %d days in %.1fs (%.1fs/day)",
                   length(dates), elapsed, elapsed / max(length(dates), 1))
    return nothing
end

# ---------------------------------------------------------------------------
# Public entry point — called by the unified CLI script.
# ---------------------------------------------------------------------------

"""
    process_day(cfg::Dict; day_override=nothing, start_date=nothing, end_date=nothing)

Top-level TOML-driven preprocessor entry. Detects source type from `cfg`:

* `[source].toml = "config/met_sources/<source>.toml"` → typed
  `AbstractMetSettings` path, supports cross-day state carry (e.g. GEOS
  pressure-fixer chained mass) and `--start/--end` date ranges.
* otherwise → typed ERA5 spectral config path (`[input].spectral_dir`).

Both paths converge on `process_day(date, grid::AbstractTargetGeometry,
settings, vertical; ...)` for the per-day work. There is no parallel
GEOS-only or per-source CLI — new sources plug in via
`AbstractMetSettings` + `load_met_settings`.
"""
function process_day(cfg::Dict{String, Any};
                     day_override::Union{String, Nothing} = nothing,
                     start_date::Union{String, Date, Nothing} = nothing,
                     end_date::Union{String, Date, Nothing} = nothing)
    if _is_native_source_cfg(cfg)
        # Native path resolves FT first, then builds grid internally so
        # mesh element type matches reader/state buffers.
        return _process_day_native(cfg; day_override, start_date, end_date)
    else
        # Spectral path: keep the historical Float64 mesh build at the entry
        # (the spectral preprocessor casts to settings.output_float_type later).
        grid = build_target_geometry(cfg["grid"], Float64)
        ensure_supported_target(grid)
        return _process_day_spectral(cfg, grid; day_override, start_date, end_date)
    end
end
