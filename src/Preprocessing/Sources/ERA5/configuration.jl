settings_constructor(::Val{Symbol("ERA5-N320")}) = ERA5N320Settings
supported_flux_keys(::Type{<:ERA5GRIBSettings}) = (FLUX_CONSTRUCTION_KEYS..., "flux_thickness", "flux_time_sampling")

# ---------------------------------------------------------------------------
# ERA5 native-GRIB — per-day hourly files split by stream
# (`ml_an_native_core`, `ml_fc_convection`, `sfc_an_native`). The settings
# type carries no horizontal `Nc` or `mass_flux_dt` — the source-mesh `Nx`
# rings come from GRIB headers at read time, and ERA5 is hourly so there is
# no dynamics-step accumulation analogous to GEOS MFXC/MFYC.
# ---------------------------------------------------------------------------

function build_met_settings(ctor::Type{<:ERA5GRIBSettings}, cfg::AbstractDict,
                             root_dir::AbstractString; kwargs...)
    vertical_cfg = get(cfg, "vertical",      Dict{String,Any}())
    pre_cfg      = get(cfg, "preprocessing", Dict{String,Any}())

    coefs = String(get(vertical_cfg, "coefficients_file",
                       "config/era5_L137_coefficients.toml"))
    haskey(pre_cfg, "level_orientation") && throw(ArgumentError(
        "ERA5-N320 level_orientation was removed; GRIB levels are canonical top-down"))
    include_surface       = _config_bool(pre_cfg, "include_surface", false, "[preprocessing].include_surface")
    include_convection    = _config_bool(pre_cfg, "include_convection", false, "[preprocessing].include_convection")
    haskey(pre_cfg, "include_vdiff_fields") && throw(ArgumentError(
        "ERA5-N320 does not implement include_vdiff_fields; remove the setting"))
    include_tm5_diffusion = _config_bool(pre_cfg, "include_tm5_diffusion", false, "[preprocessing].include_tm5_diffusion")
    arco_surface_pressure = _config_bool(pre_cfg, "arco_surface_pressure", false, "[preprocessing].arco_surface_pressure")

    include_tm5_diffusion && !include_surface &&
        throw(ArgumentError("[preprocessing] include_tm5_diffusion=true requires \
                             include_surface=true (needs sshf/slhf/ustar)."))

    settings = ctor(;
        root_dir,
        include_surface, include_convection,
        include_tm5_diffusion, arco_surface_pressure,
        coefficients_file = coefs, flux_construction_kwargs(pre_cfg)...,
        flux_thickness = Symbol(lowercase(String(get(pre_cfg, "flux_thickness", "dry_mass")))),
        flux_time_sampling = Symbol(lowercase(String(get(pre_cfg, "flux_time_sampling", "window_mean")))),
        kwargs...)
    validate_flux_thickness(settings.flux_thickness, "ERA5")
    return validate_flux_construction(settings, "ERA5")
end
