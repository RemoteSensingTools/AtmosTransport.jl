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
            package_path(joinpath("config", "geos_L72_coefficients.toml"))
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
    vc_native = load_era5_vertical_coordinate(coeff_path, first(level_range), last(level_range))
    ab = ab_coefficients(vc_native)
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
