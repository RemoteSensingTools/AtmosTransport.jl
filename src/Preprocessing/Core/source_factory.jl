function build_met_settings end

# ===========================================================================
# TOML-driven source-descriptor factory.
#
# `load_met_settings(toml_path; root_dir, kwargs...)` reads a met_sources TOML
# and returns the typed `AbstractMetSettings` for that source. The pipeline is
# two-step:
#
#   1. `settings_constructor(name)` maps `[source].name` to a concrete type.
#   2. `build_met_settings(ctor, cfg, root_dir; kwargs...)` consumes the
#      TOML tables that constructor cares about and returns a built instance.
#
# Sources plug in by adding two methods on these generics — never by extending
# a central if/else chain.
# ===========================================================================

settings_constructor(name::AbstractString) =
    settings_constructor(Val(Symbol(name)))


function settings_constructor(::Val{name}) where name
    error("Unsupported met source `$(String(name))`. " *
          "Supported: GEOS-IT, GEOS-FP, ERA5-N320, MERRA-2.")
end

"""
    load_met_settings(toml_path::String; root_dir, kwargs...) -> AbstractMetSettings

Construct a typed met-source descriptor from `toml_path`. The TOML's
`[source].name` key picks the concrete settings type, and per-type
`build_met_settings` methods consume the source-specific TOML tables.

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
    ctor = settings_constructor(name)
    reject_unsupported_flux_keys(ctor, cfg)
    return build_met_settings(ctor, cfg, String(root_dir); kwargs...)
end

# Flux-construction options and the sources whose preprocessors implement them;
# elsewhere the keys would be silently ignored, so they are rejected.
const FLUX_CONSTRUCTION_KEYS = ("column_balance_weights", "face_lengths", "face_fluxes",
                                 "face_interpolation", "wind_regrid")
const ALL_FLUX_KEYS = (FLUX_CONSTRUCTION_KEYS..., "flux_thickness", "flux_time_sampling")
supported_flux_keys(::Type) = ()
function reject_unsupported_flux_keys(ctor::Type, cfg)
    pre_cfg = get(cfg, "preprocessing", Dict{String,Any}())
    for key in ALL_FLUX_KEYS
        haskey(pre_cfg, key) && !(key in supported_flux_keys(ctor)) && throw(ArgumentError(
            "[preprocessing].$(key) is not implemented for $(nameof(ctor)) sources"))
    end
    return nothing
end

# Parse common choices in Common; native validation still follows overrides.
column_balance_weights_option(pre_cfg) =
    parse_column_balance_weights(pre_cfg, "[preprocessing]")
flux_construction_kwargs(pre_cfg) =
    (column_balance_weights = column_balance_weights_option(pre_cfg),
     parse_flux_construction(pre_cfg, "[preprocessing]")...)

"""Prefix used by the common native-source output filename formatter."""
native_output_prefix(::AbstractMetSettings) = ""

"""
    native_output_filename(settings, date, FT) -> String

Output filename for native-source preprocessing:
`<prefix>transport_YYYYMMDD_float32.bin` (or `float64`). Concrete sources
extend `native_output_prefix` (`geos_`, `era5_n320_`, `merra2_`); the default
prefix is empty.
"""
native_output_filename(settings::AbstractMetSettings, date::Date, FT::Type) =
    "$(native_output_prefix(settings))transport_$(Dates.format(date, "yyyymmdd"))_$(FT === Float32 ? "float32" : "float64").bin"


# Sources extend the native closure configuration protocol.
check_native_closure_config(::AbstractMetSettings, _cm_closure::Symbol) = nothing
