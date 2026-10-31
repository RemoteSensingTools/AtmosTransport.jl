settings_constructor(::Val{Symbol("GEOS-IT")}) = GEOSITSettings
settings_constructor(::Val{Symbol("GEOS-FP")}) = GEOSFPSettings
supported_flux_keys(::Type{<:GEOSSettings}) = ("column_balance_weights",)

# ---------------------------------------------------------------------------
# GEOS-IT / GEOS-FP — flat NetCDF archive with a per-day file per collection.
# ---------------------------------------------------------------------------

function build_met_settings(ctor::Type{<:GEOSSettings}, cfg::AbstractDict,
                             root_dir::AbstractString; kwargs...)
    grid_cfg     = get(cfg, "grid",          Dict{String,Any}())
    vertical_cfg = get(cfg, "vertical",      Dict{String,Any}())
    pre_cfg      = get(cfg, "preprocessing", Dict{String,Any}())

    coefs = String(get(vertical_cfg, "coefficients_file",
                       "config/geos_L72_coefficients.toml"))
    Nc = Int(grid_cfg["Nc"])
    mass_flux_dt = Float64(get(pre_cfg, "mass_flux_dt_seconds", GEOS_MASS_FLUX_DT_SECONDS))
    # The dynamics step must fit inside one hourly met window.
    100.0 <= mass_flux_dt <= GEOS_MET_INTERVAL_SECONDS || throw(ArgumentError(
        "[preprocessing] mass_flux_dt_seconds must be between 100 and " *
        "$(round(Int, GEOS_MET_INTERVAL_SECONDS)); got $(mass_flux_dt)."))
    if abs(mass_flux_dt - GEOS_MASS_FLUX_DT_SECONDS) > 50.0
        @warn "[preprocessing] mass_flux_dt_seconds=$(mass_flux_dt) is more than 50 s from the " *
              "GEOS FV3 dynamics step ($(GEOS_MASS_FLUX_DT_SECONDS) s)."
    end
    # Only `:bottom_up` triggers the flip, so a typo such as "bottom-up" would
    # silently write bottom-up data as top-down.
    level_orientation    = Symbol(get(pre_cfg, "level_orientation", "auto"))
    level_orientation in (:auto, :bottom_up, :top_down) || throw(ArgumentError(
        "[preprocessing] level_orientation must be \"auto\", \"bottom_up\" or " *
        "\"top_down\"; got $(repr(String(level_orientation)))."))
    # The dynamics step must divide the hourly met window exactly: a rounded
    # substep count scales the accumulated fluxes by the wrong factor.
    isinteger(GEOS_MET_INTERVAL_SECONDS / mass_flux_dt) || throw(ArgumentError(
        "[preprocessing] mass_flux_dt_seconds must divide the " *
        "$(round(Int, GEOS_MET_INTERVAL_SECONDS)) s met window into a whole number " *
        "of dynamics steps; got $(mass_flux_dt)."))
    include_surface      = _config_bool(pre_cfg, "include_surface", false, "[preprocessing].include_surface")
    include_convection   = _config_bool(pre_cfg, "include_convection", false, "[preprocessing].include_convection")
    include_vdiff_fields = _config_bool(pre_cfg, "include_vdiff_fields", false, "[preprocessing].include_vdiff_fields")
    haskey(pre_cfg, "arco_surface_pressure") && throw(ArgumentError(
        "GEOS sources do not implement [preprocessing].arco_surface_pressure; remove the setting"))
    physics_dir          = String(get(pre_cfg, "physics_dir", ""))
    physics_layout       = Symbol(get(pre_cfg, "physics_layout", "auto"))

    settings = ctor(; root_dir,
                      Nc, mass_flux_dt, level_orientation,
                      include_surface, include_convection, include_vdiff_fields,
                      physics_dir, physics_layout,
                      column_balance_weights = column_balance_weights_option(pre_cfg),
                      coefficients_file = coefs, kwargs...)
    validate_column_balance_weights(settings.column_balance_weights, "GEOS")
    return settings
end
