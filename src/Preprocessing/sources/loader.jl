# ===========================================================================
# TOML-driven source-descriptor factory.
#
# `load_met_settings(toml_path; root_dir, kwargs...)` reads a met_sources TOML
# and returns the typed `AbstractMetSettings` for that source. The pipeline is
# two-step:
#
#   1. `_settings_constructor(name)` maps `[source].name` to a concrete type.
#   2. `_build_met_settings(ctor, cfg, root_dir; kwargs...)` consumes the
#      TOML tables that constructor cares about and returns a built instance.
#
# Sources plug in by adding two methods on these generics — never by extending
# a central if/else chain.
# ===========================================================================

_settings_constructor(name::AbstractString) =
    _settings_constructor(Val(Symbol(name)))

_settings_constructor(::Val{Symbol("GEOS-IT")})  = GEOSITSettings
_settings_constructor(::Val{Symbol("GEOS-FP")})  = GEOSFPSettings
_settings_constructor(::Val{Symbol("ERA5-N320")}) = ERA5N320Settings
_settings_constructor(::Val{Symbol("MERRA-2")})  = MERRA2Settings

function _settings_constructor(::Val{name}) where name
    error("Unsupported met source `$(String(name))`. " *
          "Supported: GEOS-IT, GEOS-FP, ERA5-N320, MERRA-2.")
end

"""
    load_met_settings(toml_path::String; root_dir, kwargs...) -> AbstractMetSettings

Construct a typed met-source descriptor from `toml_path`. The TOML's
`[source].name` key picks the concrete settings type, and per-type
`_build_met_settings` methods consume the source-specific TOML tables.

`root_dir` is the on-disk directory holding the source's daily files.
Additional `kwargs` are forwarded to the constructor and override any
values supplied by the TOML.
"""
function load_met_settings(toml_path::String;
                           root_dir::AbstractString,
                           kwargs...)
    isfile(toml_path) || error("Met source TOML not found: $toml_path")
    cfg  = TOML.parsefile(toml_path)
    name = cfg["source"]["name"]
    ctor = _settings_constructor(name)
    _reject_merra2_only_keys(ctor, cfg)
    return _build_met_settings(ctor, cfg, String(root_dir); kwargs...)
end

# Flux-construction options wired into the MERRA-2 path only; elsewhere the
# keys would be silently ignored.
const _MERRA2_ONLY_PREPROCESSING_KEYS = ("column_balance_weights", "face_lengths", "flux_thickness",
                                         "face_fluxes", "face_interpolation", "wind_regrid")
_reject_merra2_only_keys(::Type{MERRA2Settings}, cfg) = nothing
function _reject_merra2_only_keys(::Type, cfg)
    pre_cfg = get(cfg, "preprocessing", Dict{String,Any}())
    for key in _MERRA2_ONLY_PREPROCESSING_KEYS
        haskey(pre_cfg, key) && throw(ArgumentError(
            "[preprocessing].$(key) is implemented for MERRA-2 sources only"))
    end
    return nothing
end

# ---------------------------------------------------------------------------
# GEOS-IT / GEOS-FP — flat NetCDF archive with a per-day file per collection.
# ---------------------------------------------------------------------------

function _build_met_settings(ctor::Type{<:GEOSSettings}, cfg::AbstractDict,
                             root_dir::AbstractString; kwargs...)
    grid_cfg     = get(cfg, "grid",          Dict{String,Any}())
    vertical_cfg = get(cfg, "vertical",      Dict{String,Any}())
    pre_cfg      = get(cfg, "preprocessing", Dict{String,Any}())

    coefs = String(get(vertical_cfg, "coefficients_file",
                       "config/geos_L72_coefficients.toml"))
    Nc = Int(grid_cfg["Nc"])
    mass_flux_dt = Float64(get(pre_cfg, "mass_flux_dt_seconds", 450.0))
    100.0 <= mass_flux_dt <= 3600.0 || throw(ArgumentError(
        "[preprocessing] mass_flux_dt_seconds must be between 100 and 3600; got $(mass_flux_dt)."))
    if !(400.0 <= mass_flux_dt <= 500.0)
        @warn "[preprocessing] mass_flux_dt_seconds=$(mass_flux_dt) is outside the usual GEOS 400-500 s range."
    end
    level_orientation    = Symbol(get(pre_cfg, "level_orientation", "auto"))
    include_surface      = _config_bool(pre_cfg, "include_surface", false, "[preprocessing].include_surface")
    include_convection   = _config_bool(pre_cfg, "include_convection", false, "[preprocessing].include_convection")
    include_vdiff_fields = _config_bool(pre_cfg, "include_vdiff_fields", false, "[preprocessing].include_vdiff_fields")
    haskey(pre_cfg, "arco_surface_pressure") && throw(ArgumentError(
        "GEOS sources do not implement [preprocessing].arco_surface_pressure; remove the setting"))
    physics_dir          = String(get(pre_cfg, "physics_dir", ""))
    physics_layout       = Symbol(get(pre_cfg, "physics_layout", "auto"))

    return ctor(; root_dir,
                  Nc, mass_flux_dt, level_orientation,
                  include_surface, include_convection, include_vdiff_fields,
                  physics_dir, physics_layout,
                  coefficients_file = coefs, kwargs...)
end

# ---------------------------------------------------------------------------
# ERA5 native-GRIB — per-day hourly files split by stream
# (`ml_an_native_core`, `ml_fc_convection`, `sfc_an_native`). The settings
# type carries no horizontal `Nc` or `mass_flux_dt` — the source-mesh `Nx`
# rings come from GRIB headers at read time, and ERA5 is hourly so there is
# no dynamics-step accumulation analogous to GEOS MFXC/MFYC.
# ---------------------------------------------------------------------------

function _build_met_settings(ctor::Type{<:ERA5GRIBSettings}, cfg::AbstractDict,
                             root_dir::AbstractString; kwargs...)
    vertical_cfg = get(cfg, "vertical",      Dict{String,Any}())
    pre_cfg      = get(cfg, "preprocessing", Dict{String,Any}())

    coefs = String(get(vertical_cfg, "coefficients_file",
                       "config/era5_L137_coefficients.toml"))
    level_orientation     = Symbol(get(pre_cfg, "level_orientation", "top_down"))
    include_surface       = _config_bool(pre_cfg, "include_surface", false, "[preprocessing].include_surface")
    include_convection    = _config_bool(pre_cfg, "include_convection", false, "[preprocessing].include_convection")
    haskey(pre_cfg, "include_vdiff_fields") && throw(ArgumentError(
        "ERA5-N320 does not implement include_vdiff_fields; remove the setting"))
    include_tm5_diffusion = _config_bool(pre_cfg, "include_tm5_diffusion", false, "[preprocessing].include_tm5_diffusion")
    arco_surface_pressure = _config_bool(pre_cfg, "arco_surface_pressure", false, "[preprocessing].arco_surface_pressure")

    include_tm5_diffusion && !include_surface &&
        throw(ArgumentError("[preprocessing] include_tm5_diffusion=true requires \
                             include_surface=true (needs sshf/slhf/ustar)."))

    return ctor(; root_dir,
                  include_surface, include_convection,
                  include_tm5_diffusion, arco_surface_pressure, level_orientation,
                  coefficients_file = coefs, kwargs...)
end

# ---------------------------------------------------------------------------
# MERRA-2 native NetCDF — regular 0.5°×0.625° LL archive split by collection
# (`M2I3NVASM` inst3 PS/QV endpoints, `M2T3NVASM` tavg3 U/V winds). Shares the
# GEOS-5 L72 hybrid coordinate. The settings carry no horizontal `Nc` (the
# source is fixed 576×361) and no `mass_flux_dt` (fluxes are derived from
# winds, not accumulated MFXC).
# ---------------------------------------------------------------------------

function _build_met_settings(ctor::Type{MERRA2Settings}, cfg::AbstractDict,
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
    column_balance_weights = Symbol(lowercase(String(get(pre_cfg, "column_balance_weights", "mass"))))
    face_lengths   = Symbol(lowercase(String(get(pre_cfg, "face_lengths", "cell_centerline"))))
    flux_thickness = Symbol(lowercase(String(get(pre_cfg, "flux_thickness", "moist"))))
    face_fluxes    = Symbol(lowercase(String(get(pre_cfg, "face_fluxes", "panel_average"))))
    face_interpolation = Symbol(lowercase(String(get(pre_cfg, "face_interpolation", "linear"))))
    wind_regrid    = Symbol(lowercase(String(get(pre_cfg, "wind_regrid", "scalar"))))
    face_fluxes === :vector && face_lengths === :cell_centerline && haskey(pre_cfg, "face_lengths") &&
        throw(ArgumentError("[preprocessing] face_fluxes = \"vector\" always uses face edge lengths; " *
                            "remove face_lengths = \"cell_centerline\""))
    return validate_merra2_settings(ctor(; root_dir,
                  coefficients_file = coefs, winds_collection, archive,
                  include_surface, include_convection, include_vdiff_fields,
                  include_convective_cloud_base, column_balance_weights,
                  face_lengths, flux_thickness, face_fluxes, face_interpolation, wind_regrid, kwargs...))
end
