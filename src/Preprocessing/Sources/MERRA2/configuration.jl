settings_constructor(::Val{Symbol("MERRA-2")}) = MERRA2Settings
supported_flux_keys(::Type{MERRA2Settings}) = (FLUX_CONSTRUCTION_KEYS..., "flux_thickness")

# ---------------------------------------------------------------------------
# MERRA-2 native NetCDF — regular 0.5°×0.625° LL archive split by collection
# (`M2I3NVASM` inst3 PS/QV endpoints, `M2T3NVASM` tavg3 U/V winds). Shares the
# GEOS-5 L72 hybrid coordinate. The settings carry no horizontal `Nc` (the
# source is fixed 576×361) and no `mass_flux_dt` (fluxes are derived from
# winds, not accumulated MFXC).
# ---------------------------------------------------------------------------

function build_met_settings(ctor::Type{MERRA2Settings}, cfg::AbstractDict,
                             root_dir::AbstractString; kwargs...)
    vertical_cfg = get(cfg, "vertical",      Dict{String,Any}())
    pre_cfg      = get(cfg, "preprocessing", Dict{String,Any}())

    coefs = String(get(vertical_cfg, "coefficients_file",
                       "config/geos_L72_coefficients.toml"))
    winds_collection      = Symbol(get(pre_cfg, "winds_collection", "tavg3"))
    archive               = merra2_archive(String(get(pre_cfg, "layout", "nasa")))
    for key in ("include_tm5_diffusion", "arco_surface_pressure")
        haskey(pre_cfg, key) && throw(ArgumentError(
            "MERRA-2 does not implement [preprocessing].$(key); remove the setting"))
    end
    include_surface      = _config_bool(pre_cfg, "include_surface", false, "[preprocessing].include_surface")
    include_convection   = _config_bool(pre_cfg, "include_convection", false, "[preprocessing].include_convection")
    include_vdiff_fields = _config_bool(pre_cfg, "include_vdiff_fields", false, "[preprocessing].include_vdiff_fields")
    include_convective_cloud_base = _config_bool(pre_cfg, "include_convective_cloud_base", false,
                                                 "[preprocessing].include_convective_cloud_base")
    flux_thickness = Symbol(lowercase(String(get(pre_cfg, "flux_thickness", "moist"))))
    return validate_merra2_settings(ctor(; root_dir,
                  coefficients_file = coefs, winds_collection, archive,
                  include_surface, include_convection, include_vdiff_fields,
                  include_convective_cloud_base, flux_thickness,
                  flux_construction_kwargs(pre_cfg)..., kwargs...))
end
