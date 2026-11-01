"""
    resolve_mass_basis(cfg) -> (thermo_dir, include_qv, mass_basis)

Parse humidity-related runtime choices from the TOML configuration.

- `thermo_dir` is the directory containing daily ERA5 thermo NetCDF files.
- `include_qv` controls whether paired `qv_start`/`qv_end` endpoint fields are
  written into the output payload.
- `mass_basis` is `:moist` or `:dry` and determines whether native fields are
  converted to dry basis before vertical merging.

Dry basis is the default runtime contract. All transport binaries should use dry
basis so that Poisson balance and cm diagnosis operate on dry mass. Moist basis
is available as an explicit opt-out for diagnostic purposes only.

Dry-basis preprocessing requires a humidity source (`input.thermo_dir`).
"""
function resolve_mass_basis(cfg)
    thermo_dir = expand_data_path(get(get(cfg, "input", Dict()), "thermo_dir", ""))
    include_qv = _config_bool(get(cfg, "output", Dict{String, Any}()),
                              "include_qv", !isempty(thermo_dir), "[output].include_qv")
    mass_basis = parse_choice(get(get(cfg, "output", Dict{String, Any}()), "mass_basis", "dry"),
                              (dry = :dry, moist = :moist), "[output].mass_basis"; allow_symbol = true)
    if mass_basis == :dry && isempty(thermo_dir)
        error("output.mass_basis='dry' requires input.thermo_dir for native qv")
    end

    return thermo_dir, include_qv, mass_basis
end

"""
    resolve_output_float_type(cfg) -> Type

Return the on-disk floating-point type requested by `[numerics] float_type`
(default `Float32` on this spectral path). Unknown values throw.
"""
resolve_output_float_type(cfg) =
    parse_float_type(get(cfg["numerics"], "float_type", "Float32"), "[numerics] float_type")

"""
    resolve_mass_fix_settings(cfg) -> NamedTuple

Parse the optional global dry-surface-pressure mass-fix settings.
"""
function checked_mass_fix_config(cfg)
    mass_fix_cfg = get(cfg, "mass_fix", Dict{String, Any}())
    stale = filter(key -> haskey(mass_fix_cfg, key),
                   ("target_ps_dry_pa", "target_total_kg"))
    stale_label = join(stale, " and [mass_fix].")
    isempty(stale) || throw(ArgumentError(
        "[mass_fix].$(stale_label) was removed; " *
        "the dry-mass target is the single " *
        "Parameters.GLOBAL_MEAN_DRY_SURFACE_PRESSURE constant " *
        "($(GLOBAL_MEAN_DRY_SURFACE_PRESSURE) Pa) for every met source"))
    check_known_keys(mass_fix_cfg, ("enable", "mode", "qv_global_climatology"),
                     "[mass_fix]")
    return mass_fix_cfg
end

function resolve_mass_fix_settings(cfg)
    mass_fix_cfg = checked_mass_fix_config(cfg)
    return (
        mass_fix_enable = _config_bool(mass_fix_cfg, "enable", true, "[mass_fix].enable"),
        target_ps_dry_pa = GLOBAL_MEAN_DRY_SURFACE_PRESSURE,
        qv_global_climatology = config_float(get(mass_fix_cfg, "qv_global_climatology", GLOBAL_MEAN_SPECIFIC_HUMIDITY), "[mass_fix].qv_global_climatology"),
    )
end

"""
    reject_removed_tm5_convection_route(cfg)

Reject a preprocessing config that still has a `[tm5_convection]` table. That
table selected the lat-lon ERA5 physics-file route (ERA5 convection NetCDF
converted to flat physics files, then to TM5 fields on the lat-lon grid). The
route placed the fields 180° off in longitude and one hour early (issue #55)
and was removed. TM5 convection comes from the ERA5 N320 → cubed-sphere
preprocessor (`include_convection = true` in the met source's
`[preprocessing]` table) or from the TM5-converted CATRINE archive attached
with `scripts/preprocessing/attach_catrine_tm5_convection_cs.jl`.
"""
function reject_removed_tm5_convection_route(cfg::AbstractDict)
    haskey(cfg, "tm5_convection") && throw(ArgumentError(
        "[tm5_convection] is removed: the lat-lon ERA5 physics-file route " *
        "(physics_bin_dir, convert_era5_physics_nc_to_bin) placed the TM5 fields 180° off " *
        "in longitude and one hour early (issue #55). Build TM5 convection on the cubed " *
        "sphere from native ERA5 instead: an ERA5 N320 met source with " *
        "include_convection = true in its [preprocessing] table (for example " *
        "config/preprocessing/era5_n320_arco_full_to_c90.toml), or attach the TM5-converted " *
        "CATRINE convection with scripts/preprocessing/attach_catrine_tm5_convection_cs.jl. " *
        "Delete the [tm5_convection] table."))
    return nothing
end

"""
    resolve_surface_settings(cfg) -> NamedTuple

Parse optional ERA5 single-level surface input for raw PBL diffusion fields.
The daily LL writer stores the raw `pblh/t2m/ustar/pbl_hflux` payload; CS
regridding preserves it and runtime `[diffusion] kind="tm5_beljaars_viterbo_local_kz"` derives Kz.
"""
function resolve_surface_settings(cfg)
    surface_cfg = get(cfg, "surface", Dict{String, Any}())
    input_cfg = get(cfg, "input", Dict{String, Any}())
    surface_dir = expand_data_path(get(input_cfg, "surface_dir", ""))
    include_surface = _config_bool(surface_cfg, "enable", false, "[surface].enable")
    include_tm5_pbl = _config_bool(get(cfg, "preprocessing", Dict{String,Any}()),
        "include_tm5_pbl", false, "[preprocessing].include_tm5_pbl")
    if (include_surface || include_tm5_pbl) && isempty(surface_dir)
        trigger = include_tm5_pbl ? "[preprocessing] include_tm5_pbl=true" : "[surface] enable=true"
        throw(ArgumentError("$trigger requires input.surface_dir with ERA5 single-level PBL fields"))
    end
    return (
        include_surface = include_surface,
        include_tm5_pbl = include_tm5_pbl,
        surface_dir = surface_dir,
    )
end

"""
    resolve_preprocessing_cache_settings(cfg) -> NamedTuple

Parse optional cache and preload controls.

Persistent spectral caching is disabled unless `[cache]` provides
`spectral_coefficients_dir` or `ATMOSTR_SPECTRAL_CACHE_DIR` is set. Daily QV
preload is an in-memory optimization and is enabled by default, bounded by
`qv_preload_max_gb`.
"""
function resolve_preprocessing_cache_settings(cfg)
    cache_cfg = get(cfg, "cache", Dict{String, Any}())
    spectral_dir_default = get(ENV, "ATMOSTR_SPECTRAL_CACHE_DIR", "")
    spectral_cache_dir = expanduser(String(get(cache_cfg, "spectral_coefficients_dir",
                                               spectral_dir_default)))
    qv_preload = _config_bool(cache_cfg, "qv_preload", true, "[cache].qv_preload")
    qv_preload_max_gb = config_float(get(cache_cfg, "qv_preload_max_gb", 8.0), "[cache].qv_preload_max_gb")
    qv_preload_max_bytes = Int64(floor(qv_preload_max_gb * 1024.0^3))
    return (
        spectral_cache_dir = spectral_cache_dir,
        qv_preload = qv_preload,
        qv_preload_max_bytes = qv_preload_max_bytes,
    )
end

"""
    resolve_balance_settings(cfg) -> NamedTuple

Parse optional solver controls for topology-specific mass-flux balancing.

`cs_balance_tol` keeps the historical strict default, `DEFAULT_POISSON_BALANCE_TOL`.
`cs_balance_project_every` (default `DEFAULT_POISSON_PROJECT_EVERY`)
controls how often the CS PCG solver reprojects residuals to the mean-zero
subspace after the initial exact projection; the graph Laplacian preserves that
subspace, so periodic projection removes redundant global reductions while
keeping roundoff bounded. Set it to `1` for the legacy every-iteration path.
"""
function resolve_balance_settings(cfg)
    numerics = get(cfg, "numerics", Dict{String, Any}())
    reject_removed_balance_mode(numerics)
    return (
        cs_balance_tol = config_float(get(numerics, "cs_balance_tol", DEFAULT_POISSON_BALANCE_TOL), "[numerics].cs_balance_tol"),
        cs_balance_project_every = config_int(get(numerics, "cs_balance_project_every", DEFAULT_POISSON_PROJECT_EVERY), "[numerics].cs_balance_project_every"),
        column_balance_weights = resolve_column_balance_weights(numerics),
    )
end

"""
    resolve_spectral_flux_settings(cfg) -> NamedTuple

Resolve the ERA5 spectral-to-cubed-sphere flux construction selected in
`[numerics]`. The defaults are the GCHP-matched path: regrid winds as Cartesian
vectors, interpolate them to true cube edges with FV3's fourth-order filtered
stencil, and average independently constructed start/end face fluxes over the
meteorological window. `face_fluxes = "line_integral"` instead evaluates

```math
F_f = (Δt/g) ∫_f (v · n_f) Δp dl
```

directly on the spectral staging grid.
"""
function resolve_spectral_flux_settings(cfg)
    numerics = get(cfg, "numerics", Dict{String, Any}())
    return merge(parse_flux_construction(numerics, "[numerics]"),
                 (flux_time_sampling = Symbol(lowercase(String(get(numerics, "flux_time_sampling", "window_mean")))),))
end

"""
    reject_removed_balance_mode(numerics)

Throw if a preprocessing `[numerics]` table sets `balance_mode` (or its older
name `geos_balance_mode`): the per-layer balance it selected was removed, and
every path that applies a Poisson balance uses the column balance
(`HORIZONTAL_BALANCE_TAG`). GEOS `pressure_fixer` and `pfix_corrected` keep
their native horizontal fluxes and record `horizontal_balance = "none"`.
"""
function reject_removed_balance_mode(numerics::AbstractDict)
    for key in ("balance_mode", "geos_balance_mode")
        haskey(numerics, key) && throw(ArgumentError(
            "[numerics].$key is removed: every preprocessing path that applies a " *
            "Poisson balance uses the column balance (the per-layer balance left no " *
            "vertical mass flux; GEOS pressure_fixer and pfix_corrected keep native fluxes); " *
            "delete the key"))
    end
    return nothing
end

"""
    reject_removed_horizontal_balance_keyword(kwargs)

Reject the removed writer keyword when a source-specific `process_day` adapter
otherwise absorbs unified-driver keywords through `kwargs...`.
"""
function reject_removed_horizontal_balance_keyword(kwargs)
    haskey(kwargs, :horizontal_balance) && throw(ArgumentError(
        "horizontal_balance is removed: every preprocessing path that applies a " *
        "Poisson balance uses the column balance; delete the keyword"))
    return nothing
end

"""
    resolve_column_balance_weights(numerics) -> Symbol

`[numerics] column_balance_weights` of an ERA5 spectral preprocessing config:
how the column balance spreads its correction over the levels, `:hybrid_mass`
(default, on every target: lat-lon, reduced Gaussian and cubed sphere),
`:hybrid_b` or `:mass` (see `AbstractColumnWeights`). Native sources set it in
their met-source `[preprocessing]` table instead.
"""
function resolve_column_balance_weights(numerics::AbstractDict)
    return validate_column_balance_weights(
        parse_column_balance_weights(numerics, "[numerics]"), "[numerics]")
end

# `kind` must name an entry of `COLUMN_WEIGHT_KINDS`; `source` names the met
# source or table in the error.
function validate_column_balance_weights(kind::Symbol, source::AbstractString)
    kind in COLUMN_WEIGHT_NAMES || throw(ArgumentError(
        "$source column_balance_weights must be one of " *
        join(COLUMN_WEIGHT_NAMES, ", ") * "; got :$(kind)"))
    return kind
end

# The configured column weights of an ERA5 spectral run, validated; checked by
# the entry point before any input is read, and by each writer for direct
# callers. Every target's column balance (lat-lon FFT, reduced-Gaussian CG,
# cubed-sphere CG) spreads its correction with `spread_column_correction!`, so
# every weighting applies to every target.
check_spectral_column_weights(::AbstractTargetGeometry, settings) =
    validate_column_balance_weights(get(settings, :column_balance_weights, :hybrid_mass), "[numerics]")

# The column weights of an ERA5 spectral run on its output levels (`vertical.merged_vc`).
spectral_column_weights(grid::AbstractTargetGeometry, settings, vertical) =
    column_weights(check_spectral_column_weights(grid, settings), vertical.merged_vc.B)

# Record every non-mass weighting in a binary header. This deliberately records
# the new `hybrid_mass` default: legacy keyless binaries remain identifiable as
# mass-weighted, rather than changing the meaning of an absent provenance key.
function with_column_weights_record(header::AbstractDict, kind::Symbol)
    kind === :mass || (header["column_balance_weights"] = String(kind))
    return header
end

# Record a skipped write-time replay gate (`[numerics] write_replay_check =
# false`) in a binary header. A binary that passed the gate records nothing, so
# default headers are unchanged.
function with_replay_record(header::AbstractDict, write_replay_on::Bool)
    write_replay_on || (header["write_replay_check"] = false)
    return header
end

"""
    resolve_runtime_settings(cfg) -> NamedTuple

Resolve the script configuration into a compact runtime settings bundle used by
the day-processing pipeline.
"""
function resolve_runtime_settings(cfg)
    reject_removed_tm5_convection_route(cfg)
    spectral_dir = expand_data_path(cfg["input"]["spectral_dir"])
    coeff_path = expand_data_path(cfg["input"]["coefficients"])
    out_dir = expand_data_path(cfg["output"]["directory"])
    thermo_dir, include_qv, mass_basis = resolve_mass_basis(cfg)

    level_top = config_int(get(cfg["grid"], "level_top", 1), "[grid].level_top")
    level_bot = config_int(get(cfg["grid"], "level_bot", ERA5_NATIVE_LEVEL_COUNT), "[grid].level_bot")
    min_dp = Float64(cfg["grid"]["merge_min_thickness_Pa"])

    dt = Float64(cfg["numerics"]["dt"])
    met_interval = Float64(cfg["numerics"]["met_interval"])

    return merge((
        spectral_dir = spectral_dir,
        coeff_path = coeff_path,
        out_dir = out_dir,
        thermo_dir = thermo_dir,
        include_qv = include_qv,
        mass_basis = mass_basis,
        level_range = level_top:level_bot,
        min_dp = min_dp,
        dt = dt,
        met_interval = met_interval,
        half_dt = dt / 2,
        output_float_type = resolve_output_float_type(cfg),
    ), resolve_mass_fix_settings(cfg),
       resolve_surface_settings(cfg),
       resolve_preprocessing_cache_settings(cfg),
       resolve_balance_settings(cfg),
       resolve_spectral_flux_settings(cfg))
end

"""
    log_preprocessor_configuration(settings, grid, vertical)

Emit the human-readable startup summary for the current preprocessing run.
"""
function log_preprocessor_configuration(settings, grid::AbstractTargetGeometry, vertical)
    mass_fix_summary =
        if !settings.mass_fix_enable
            "OFF"
        elseif settings.include_qv || settings.mass_basis == :dry
            "ON  (target_ps_dry=$(settings.target_ps_dry_pa) Pa, qv_source=native_hourly)"
        else
            "ON  (target_ps_dry=$(settings.target_ps_dry_pa) Pa, qv_global=$(settings.qv_global_climatology))"
        end

    spectral_cache_summary = isempty(settings.spectral_cache_dir) ?
        "OFF" : settings.spectral_cache_dir
    qv_preload_summary = settings.qv_preload ?
        @sprintf("ON  (max %.1f GiB)", settings.qv_preload_max_bytes / 1024.0^3) : "OFF"

    target_vertical = hasproperty(vertical, :target_vertical_name) &&
                      !isempty(vertical.target_vertical_name) ?
        " ($(vertical.target_vertical_name))" : ""
    mapping_method = hasproperty(vertical, :vertical_mapping_method) ?
        String(vertical.vertical_mapping_method) : "merge_map"

    @info """
    Fused Spectral -> v4 Binary Preprocessor
    ==========================================
    Spectral dir:  $(settings.spectral_dir)
    Output dir:    $(settings.out_dir)
    Target grid:   $(target_summary(grid))
    Native levels: $(vertical.Nz_native) ($(first(vertical.level_range))-$(last(vertical.level_range)))
    Output levels: $(vertical.Nz)$(target_vertical)
    Vertical map:  $(mapping_method)
    DT:            $(settings.dt) s (half_dt=$(settings.half_dt) s)
    Met interval:  $(settings.met_interval) s
    T_target:      $(settings.T_target)
    Threads:       $(Threads.nthreads())
    Float type:    $(settings.output_float_type)
    Mass basis:    $(settings.mass_basis)
    Include qv:    $(settings.include_qv)
    Mass fix:      $(mass_fix_summary)
    Spectral cache: $(spectral_cache_summary)
    QV preload:    $(qv_preload_summary)
    """
end

# Native callers deliberately defer choice validation until keyword overrides
# have been applied to settings, as before this extraction.
function parse_column_balance_weights(table, label::AbstractString)
    raw = get(table, "column_balance_weights", "hybrid_mass")
    raw isa AbstractString || throw(ArgumentError(
        "$label.column_balance_weights must be a string; got $(repr(raw))"))
    return Symbol(lowercase(raw))
end

function parse_flux_construction(table, label::AbstractString)
    option(key, default) = Symbol(lowercase(String(get(table, key, default))))
    face_fluxes = option("face_fluxes", "vector")
    face_lengths = option("face_lengths", "edge")
    face_fluxes === :vector && face_lengths === :cell_centerline &&
        haskey(table, "face_lengths") && throw(ArgumentError(
            "$label face_fluxes = \"vector\" always uses face edge lengths; " *
            "remove face_lengths = \"cell_centerline\""))
    return (face_lengths, face_fluxes,
            face_interpolation = option("face_interpolation", face_fluxes === :vector ? "fv3" : "linear"),
            wind_regrid = option("wind_regrid", "cartesian"))
end
