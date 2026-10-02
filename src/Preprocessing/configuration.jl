"""
    parse_day_filter(args) -> Union{Nothing, Date}

Extract the optional `--day YYYY-MM-DD` filter from the script argument list.
Returns `nothing` when no day filter is requested.
"""
function parse_day_filter(args)
    idx = findfirst(==("--day"), args)
    idx === nothing && return nothing
    idx == length(args) && error("Missing date after --day")
    return Date(args[idx + 1])
end

"""
    resolve_mass_basis(cfg) -> (thermo_dir, include_qv, mass_basis)

Parse humidity-related runtime choices from the TOML configuration.

- `thermo_dir` is the directory containing daily ERA5 thermo NetCDF files.
- `include_qv` controls whether paired `qv_start`/`qv_end` endpoint fields are
  written into the output payload.
- `mass_basis` is `:moist` or `:dry` and determines whether native fields are
  converted to dry basis before vertical merging.

Dry-basis is the default (Invariant 14). All transport binaries should use dry
basis so that Poisson balance and cm diagnosis operate on dry mass. Moist basis
is available as an explicit opt-out for diagnostic purposes only.

Dry-basis preprocessing requires a humidity source (`input.thermo_dir`).
"""
function resolve_mass_basis(cfg)
    thermo_dir = expand_data_path(get(get(cfg, "input", Dict()), "thermo_dir", ""))
    include_qv = _config_bool(get(cfg, "output", Dict{String, Any}()),
                              "include_qv", !isempty(thermo_dir), "[output].include_qv")
    basis_str = lowercase(String(get(get(cfg, "output", Dict{String, Any}()), "mass_basis", "dry")))
    mass_basis = basis_str == "dry" ? :dry : :moist

    if basis_str != "moist" && basis_str != "dry"
        error("Invalid output.mass_basis='$basis_str' (expected 'moist' or 'dry')")
    end
    if mass_basis == :dry && isempty(thermo_dir)
        error("output.mass_basis='dry' requires input.thermo_dir for native qv")
    end

    return thermo_dir, include_qv, mass_basis
end

"""
    resolve_output_float_type(cfg) -> Type

Return the on-disk floating-point type requested by the config.
"""
resolve_output_float_type(cfg) =
    String(get(cfg["numerics"], "float_type", "Float32")) == "Float64" ? Float64 : Float32

"""
    resolve_mass_fix_settings(cfg) -> NamedTuple

Parse the optional global dry-surface-pressure mass-fix settings.
"""
function resolve_mass_fix_settings(cfg)
    mass_fix_cfg = get(cfg, "mass_fix", Dict{String, Any}())
    return (
        mass_fix_enable = _config_bool(mass_fix_cfg, "enable", true, "[mass_fix].enable"),
        target_ps_dry_pa = Float64(get(mass_fix_cfg, "target_ps_dry_pa", 98726.0)),
        qv_global_climatology = Float64(get(mass_fix_cfg, "qv_global_climatology", 0.00247)),
    )
end

"""
    resolve_tm5_convection_settings(cfg) -> NamedTuple

Parse the optional `[tm5_convection]` section.  When `enable=true` the
preprocessor reads ERA5 physics binaries (built by `convert_era5_physics_nc_to_bin`),
computes TM5 entu/detu/entd/detd per hour via
`tm5_native_fields_for_hour!`, merges to the transport Nz,
conservatively regrids to the target horizontal grid, and writes the four
TM5 sections into the transport binary.

Fields:
- `tm5_convection_enable :: Bool` — master switch.
- `tm5_physics_bin_dir   :: String` — NVMe directory holding
  `era5_physics_YYYYMMDD.bin` files produced by the physics-binary converter.
  Empty when disabled.
"""
function resolve_tm5_convection_settings(cfg)
    tm5_cfg = get(cfg, "tm5_convection", Dict{String, Any}())
    enable = _config_bool(tm5_cfg, "enable", false, "[tm5_convection].enable")
    bin_dir = expanduser(String(get(tm5_cfg, "physics_bin_dir", "")))
    if enable && isempty(bin_dir)
        error("[tm5_convection] enable=true requires physics_bin_dir " *
              "pointing at the NVMe directory of ERA5 physics binaries " *
              "(run scripts/preprocessing/convert_era5_physics_nc_to_bin.jl first)")
    end
    return (
        tm5_convection_enable = enable,
        tm5_physics_bin_dir = bin_dir,
    )
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
    if include_surface && isempty(surface_dir)
        error("[surface] enable=true requires input.surface_dir with ERA5 single-level PBL fields")
    end
    return (
        include_surface = include_surface,
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
    qv_preload_max_gb = Float64(get(cache_cfg, "qv_preload_max_gb", 8.0))
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

`cs_balance_tol` keeps the historical strict default. `cs_balance_project_every`
controls how often the CS PCG solver reprojects residuals to the mean-zero
subspace after the initial exact projection; the graph Laplacian preserves that
subspace, so periodic projection removes redundant global reductions while
keeping roundoff bounded. Set it to `1` for the legacy every-iteration path.
"""
function resolve_balance_settings(cfg)
    numerics = get(cfg, "numerics", Dict{String, Any}())
    return (
        cs_balance_tol = Float64(get(numerics, "cs_balance_tol", 1e-14)),
        cs_balance_project_every = Int(get(numerics, "cs_balance_project_every", 50)),
    )
end

"""
    resolve_runtime_settings(cfg) -> NamedTuple

Resolve the script configuration into a compact runtime settings bundle used by
the day-processing pipeline.
"""
function resolve_runtime_settings(cfg)
    spectral_dir = expand_data_path(cfg["input"]["spectral_dir"])
    coeff_path = expand_data_path(cfg["input"]["coefficients"])
    out_dir = expand_data_path(cfg["output"]["directory"])
    thermo_dir, include_qv, mass_basis = resolve_mass_basis(cfg)

    level_top = Int(get(cfg["grid"], "level_top", 1))
    level_bot = Int(get(cfg["grid"], "level_bot", 137))
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
       resolve_tm5_convection_settings(cfg),
       resolve_surface_settings(cfg),
       resolve_preprocessing_cache_settings(cfg),
       resolve_balance_settings(cfg))
end

"""
    _resolve_target_vertical_coordinate(cfg_grid) -> (name, coeff_path) or `nothing`

Resolve an optional target hybrid vertical coordinate from the grid config.

Supported forms:

```toml
[grid]
target_vertical = "geos_l72"
```

or, for a custom coefficient file:

```toml
[grid]
target_vertical = "custom"
target_coefficients = "config/some_LN_coefficients.toml"
```
"""
function _resolve_target_vertical_coordinate(cfg_grid)
    target_name = lowercase(String(get(cfg_grid, "target_vertical", "")))
    target_path = String(get(cfg_grid, "target_coefficients", ""))
    isempty(target_name) && isempty(target_path) && return nothing

    if isempty(target_path)
        target_path = if target_name in ("geos_l72", "geos-it_l72", "geosfp_l72",
                                         "geos_fp_l72", "gmao_l72")
            joinpath(@__DIR__, "..", "..", "config", "geos_L72_coefficients.toml")
        else
            error("grid.target_vertical='$target_name' needs grid.target_coefficients")
        end
    end

    return (name = isempty(target_name) ? "custom" : target_name,
            coeff_path = expand_data_path(target_path))
end

"""
    build_vertical_setup(coeff_path, level_range, min_dp, cfg_grid) -> NamedTuple

Construct the native and output vertical-coordinate metadata for the current
run.

Two vertical mapping modes are supported:

* `:merge_map` collapses native ERA5 layers onto selected/native-derived
  output layers using a native-level `merge_map`.
* `:pressure_overlap` remaps native ERA5 layer integrals onto an independent
  target hybrid coordinate, e.g. GEOS-IT L72, by pressure-thickness overlap:

  ```math
  X^{target}_k = \\sum_s X^{source}_s
      \\frac{\\max(0, \\min(p^s_{s+1}, p^t_{k+1})
                    - \\max(p^s_s, p^t_k))}
           {p^s_{s+1} - p^s_s}
  ```

  where `p = A + B p_s`. For face-centered fluxes the same formula is applied
  with the face surface pressure used by the spectral flux construction.
"""
function build_vertical_setup(coeff_path::String, level_range, min_dp::Float64, cfg_grid)
    ab = load_ab_coefficients(coeff_path, level_range)
    vc_native = load_era5_vertical_coordinate(coeff_path, first(level_range), last(level_range))
    echlevs_name = get(cfg_grid, "echlevs", "")
    target_vertical = _resolve_target_vertical_coordinate(cfg_grid)

    if target_vertical !== nothing && !isempty(echlevs_name)
        error("Choose either grid.target_vertical/grid.target_coefficients or grid.echlevs, not both")
    end

    merged_vc, merge_map, mapping_method =
        if !isempty(echlevs_name)
            selected_vc, mm =
                select_levels_echlevs(vc_native, echlevs_preset(echlevs_name))
            selected_vc, mm, :merge_map
        elseif target_vertical !== nothing
            target_vc = load_hybrid_coefficients(target_vertical.coeff_path)
            @info "target vertical coordinate: $(target_vertical.name) " *
                  "($(n_levels(vc_native)) native → $(n_levels(target_vc)) target levels, pressure-overlap remap)"
            target_vc, Int[], :pressure_overlap
        else
            merged, mm = merge_thin_levels(vc_native; min_thickness_Pa=min_dp)
            merged, mm, :merge_map
        end

    return (
        ab = ab,
        level_range = level_range,
        vc_native = vc_native,
        merged_vc = merged_vc,
        merge_map = merge_map,
        vertical_mapping_method = mapping_method,
        target_vertical_name = target_vertical === nothing ? "" : target_vertical.name,
        target_coefficients = target_vertical === nothing ? "" : target_vertical.coeff_path,
        Nz_native = n_levels(vc_native),
        Nz = n_levels(merged_vc),
    )
end

"""
    available_spectral_dates(spectral_dir) -> Vector{Date}

Discover all processing dates available in the spectral-input directory by
scanning for `era5_spectral_YYYYMMDD_lnsp.gb`.
"""
function available_spectral_dates(spectral_dir::String)
    dates = Date[]
    for f in readdir(spectral_dir)
        m = match(r"era5_spectral_(\d{8})_lnsp\.gb", f)
        m !== nothing && push!(dates, Date(m[1], dateformat"yyyymmdd"))
    end
    sort!(dates)
    isempty(dates) && error("No spectral GRIB files found in $spectral_dir")
    return dates
end

"""
    select_processing_dates(dates, day_filter) -> Vector{Date}

Apply the optional single-day filter to the discovered date list.
"""
function select_processing_dates(dates, day_filter)
    day_filter === nothing && return dates
    filtered = filter(==(day_filter), dates)
    isempty(filtered) && error("Date $day_filter not found in spectral data")
    return filtered
end

"""
    next_day_hour0(date, dates, spectral_dir, T_target)

Load the next day's hour-0 spectral fields when available so the final window of
the current day can form forward differences for `dam`, `dbm`, and `dm`.
Returns `nothing` when no next-day file is available.
"""
function next_day_hour0(date::Date, dates, spectral_dir::String, T_target::Int;
                        cache_dir::AbstractString="")
    next_day = date + Day(1)
    next_path = joinpath(spectral_dir, "era5_spectral_$(Dates.format(next_day, "yyyymmdd"))_lnsp.gb")
    if next_day in dates || isfile(next_path)
        return read_hour0_spectral(spectral_dir, next_day; T_target, cache_dir)
    end
    return nothing
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

    tm5_summary = settings.tm5_convection_enable ?
        "ON  ($(settings.tm5_physics_bin_dir))" : "OFF"
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
    TM5 convec:    $(tm5_summary)
    Spectral cache: $(spectral_cache_summary)
    QV preload:    $(qv_preload_summary)
    """
end
