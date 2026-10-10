# ===========================================================================
# Known run-configuration keys
#
# Every key the runtime reads, per table, so that `validate_config` can warn
# about keys a run would ignore: unknown (usually misspelled) keys, and known
# keys that the chosen kind or another setting leaves unread. Each list names
# the parser it mirrors; a parser that starts reading a new key adds it here.
# `[output.observations]` is not listed: its parser rejects unknown keys itself.
# ===========================================================================

const _TOP_LEVEL_TABLES = ("input", "architecture", "numerics", "run", "advection",
                           "diffusion", "convection", "chemistry", "tracers", "output",
                           "init")

# `expand_binary_paths`, `_validate_input_binary_expectations`, `InputStager`.
const _INPUT_KEYS = ("binary_paths", "folder", "start_date", "end_date", "file_pattern",
                     "expected_nlevel", "required_preprocessor_contract",
                     "require_adaptive_substeps", "validate_replay", "staging")
const _INPUT_STAGING_KEYS = ("enabled", "dir", "lookahead_days", "keep_behind_days",
                             "cleanup_on_exit")
const _ARCHITECTURE_KEYS = ("use_gpu", "backend")        # `architecture_from_config`
const _NUMERICS_KEYS = ("float_type",)                   # `_cfg_float_type`

# `[run]`. `reset_air_mass_each_window` is read only to reject it. The
# advection keys are the legacy location of `[advection]` (`_advection_section`).
const _RUN_KEYS = ("start_window", "stop_window", "air_mass_reset_mode", "physics_cadence",
                   "Hp", "halo_padding", "tracer_name", "reset_air_mass_each_window")
const _ADVECTION_KEYS = ("scheme", "ppm_order", "vertical", "limiter")   # `advection_spec`
const _DIFFUSION_KEYS = ("kind", "value", "surface_flux_boundary", "type")  # `diffusion_spec`
const _COLLAB_LU_KEYS = ("tile_workspace_gib", "use_collab_lu", "lmax_conv", "n_merge")
const _CONVECTION_KEYS = ("kind", "clamp", "cloud_base", _COLLAB_LU_KEYS...)  # `convection_spec`
const _CHEMISTRY_KEYS = ("kind", "half_lives_seconds")                   # `chemistry_spec`

# Initial conditions (`InitialConditionIO`), by the kinds that read them
# (`kind` applies to every kind; `background` is the south value of a
# latitude step and the base of a blob).
const _INIT_KIND_KEYS = (
    latitude_step  = ("background", "south_value", "south", "north_value", "north",
                      "split_lat_deg"),
    gaussian_blob  = ("background", "lon0_deg", "lat0_deg", "sigma_lon_deg", "sigma_lat_deg",
                      "amplitude"),
    bl_enhanced    = ("background", "n_layers", "enhancement"),
    file           = ("file", "variable", "time_index"),
    cs_native      = ("file", "variable", "time_index", "vertical_order", "clamp_negative"),
    pressure_layer = ("lowest_layer", "psurf_fraction", "total_molecules"),
    uniform        = ("background",),
)
const _INIT_KEYS = Tuple(unique(vcat(["kind", "background"],
                                     collect(Iterators.flatten(values(_INIT_KIND_KEYS))))))

# Every `init.kind` the builders accept (on some topology), with its key group.
const _INIT_KINDS = Dict("uniform" => :uniform, "latitude_step" => :latitude_step,
                         "lat_step" => :latitude_step, "hemisphere_step" => :latitude_step,
                         "gaussian_blob" => :gaussian_blob, "bl_enhanced" => :bl_enhanced,
                         "file" => :file, "netcdf" => :file, "file_field" => :file,
                         "catrine_co2" => :file, "cs_native" => :cs_native,
                         "pressure_layer" => :pressure_layer)

# An unknown kind (`nothing`) is reported by the initial-condition builder.
_init_kind_group(kind::AbstractString) = get(_INIT_KINDS, lowercase(kind), nothing)

# Surface fluxes (`build_surface_flux_source`). `month` is read by two kinds,
# the series keys only by time-varying sources.
const _SURFACE_FLUX_KEYS = ("kind", "file", "variable", "time_index", "month", "year",
                            "files", "file_pattern", "scale", "molar_mass_kg_mol",
                            "time_varying", "temporal_scheme", "regridding")
const _SURFACE_FLUX_MONTH_KINDS = ("gridfed_fossil_co2", "zhang_rn222")
const _SURFACE_FLUX_SERIES_KEYS = ("files", "file_pattern", "temporal_scheme")

# `[output]` (`runtime_output_spec`), in precedence order within each alias group.
const _OUTPUT_PATH_KEYS = ("path", "snapshot_file", "filename")
const _OUTPUT_KEYS = (_OUTPUT_PATH_KEYS..., _SNAPSHOT_SCHEDULE_KEYS..., "start_hour",
                      "stop_hour", "split", "format", "deflate_level", "shuffle", "enabled",
                      "fields", "observations")
const _OUTPUT_FIELDS_KEYS = ("tracers", "layers", "levels", "column_mean",
                             "column_mass_per_area", "column_mass", "air_mass_layers",
                             "air_mass", "air_mass_per_area", "column_air_mass_per_area",
                             "per_tracer", "tracer_fields")
const _TRACER_OUTPUT_FIELDS_KEYS = ("layers", "column_mean", "column_mass_per_area",
                                    "column_mass")

_lower(value) = value isa AbstractString ? lowercase(value) : nothing
_kind_string(section, default) = _lower(get(section, "kind", default))

# Append a warning for each key of `table` not in `allowed`.
function _warn_unknown!(warnings, table, allowed, label)
    table isa AbstractDict || return warnings
    unknown = unknown_key_messages(table, allowed)
    isempty(unknown) ||
        push!(warnings, "$(label): unknown option(s) $(join(unknown, ", ")); the run ignores them.")
    return warnings
end

# Append a warning for each key of `table` in `keys` that the run leaves unread.
function _warn_unread!(warnings, table, keys, label, reason)
    table isa AbstractDict || return warnings
    present = [k for k in keys if haskey(table, k)]
    isempty(present) ||
        push!(warnings, "$(label): $(join(present, ", ")) $(length(present) == 1 ? "is" : "are") " *
                        "ignored $(reason).")
    return warnings
end

# All but the first present key of an alias group are ignored.
function _warn_shadowed!(warnings, table, group, label)
    present = [k for k in group if haskey(table, k)]
    length(present) > 1 &&
        push!(warnings, "$(label): $(join(present[2:end], ", ")) $(length(present) == 2 ? "is" : "are") " *
                        "ignored because `$(present[1])` is set.")
    return warnings
end

"""
    _config_key_warnings(cfg) -> Vector{String}

Keys of a run config that the run would ignore: unknown keys (with a
suggestion when one is close) and known keys that the chosen kind or another
setting leaves unread. Tables of the wrong type are left to
`_check_config_table_shapes!`.
"""
function _config_key_warnings(cfg::AbstractDict)
    w = String[]
    unknown_tables = unknown_key_messages(cfg, _TOP_LEVEL_TABLES)
    isempty(unknown_tables) ||
        push!(w, "unknown top-level table(s) or key(s) $(join(unknown_tables, ", ")); the run ignores them.")

    input = get(cfg, "input", nothing)
    _warn_unknown!(w, input, _INPUT_KEYS, "[input]")
    if input isa AbstractDict
        haskey(input, "binary_paths") &&
            _warn_unread!(w, input, ("end_date", "file_pattern"), "[input]", "with `binary_paths`")
        staging = get(input, "staging", nothing)
        _warn_unknown!(w, staging, _INPUT_STAGING_KEYS, "[input.staging]")
        staging isa AbstractDict && get(staging, "enabled", false) === false &&
            _warn_unread!(w, staging, ("dir", "lookahead_days", "keep_behind_days", "cleanup_on_exit"),
                          "[input.staging]", "while staging is disabled")
    end
    _warn_unknown!(w, get(cfg, "architecture", nothing), _ARCHITECTURE_KEYS, "[architecture]")
    _warn_unknown!(w, get(cfg, "numerics", nothing), _NUMERICS_KEYS, "[numerics]")

    run = get(cfg, "run", nothing)
    has_advection = haskey(cfg, "advection")
    tracers = get(cfg, "tracers", nothing)
    if run isa AbstractDict
        # The advection keys are legacy [run] keys without [advection]; next to it
        # they are an error (`_advection_section`), reported by `validate_config`.
        _warn_unknown!(w, run, (_RUN_KEYS..., _ADVECTION_KEYS...), "[run]")
        tracers isa AbstractDict &&
            _warn_unread!(w, run, ("tracer_name",), "[run]", "because [tracers] defines the tracers")
    end
    _advection_key_warnings!(w, has_advection ? cfg["advection"] : run,
                             has_advection ? "[advection]" : "[run]")
    _diffusion_key_warnings!(w, get(cfg, "diffusion", nothing))
    _convection_key_warnings!(w, get(cfg, "convection", nothing))
    _chemistry_key_warnings!(w, get(cfg, "chemistry", nothing))

    if haskey(cfg, "init")
        tracers isa AbstractDict ?
            push!(w, "top-level [init] is ignored because [tracers] defines the tracers.") :
            _init_key_warnings!(w, cfg["init"], "[init]")
    end
    if tracers isa AbstractDict
        for (name, tracer) in pairs(tracers)
            tracer isa AbstractDict && _tracer_key_warnings!(w, tracer, "tracers.$(name)")
        end
    end
    run_tracer = run isa AbstractDict ? get(run, "tracer_name", "CO2") : "CO2"
    carried = tracers isa AbstractDict ? Set(String.(keys(tracers))) :
              run_tracer isa AbstractString ? Set([run_tracer]) : nothing
    _output_key_warnings!(w, get(cfg, "output", nothing), carried)
    return w
end

function _advection_key_warnings!(w, section, label)
    section isa AbstractDict || return w
    label == "[advection]" && _warn_unknown!(w, section, _ADVECTION_KEYS, label)
    scheme = _lower(get(section, "scheme", "upwind"))
    scheme in ("upwind", "slopes", "none") &&
        _warn_unread!(w, section, ("ppm_order",), label, "with scheme = \"$(scheme)\"")
    return w
end

function _diffusion_key_warnings!(w, section)
    section isa AbstractDict || return w
    _warn_unknown!(w, section, _DIFFUSION_KEYS, "[diffusion]")
    haskey(section, "kind") &&
        _warn_unread!(w, section, ("type",), "[diffusion]", "next to `kind` (legacy key)")
    kind = _kind_string(section, "none")
    kind === nothing && return w
    kind == "constant" ||
        _warn_unread!(w, section, ("value",), "[diffusion]", "unless kind = \"constant\"")
    kind == "none" &&
        _warn_unread!(w, section, ("surface_flux_boundary",), "[diffusion]", "with kind = \"none\"")
    return w
end

function _convection_key_warnings!(w, section)
    section isa AbstractDict || return w
    _warn_unknown!(w, section, _CONVECTION_KEYS, "[convection]")
    kind = _kind_string(section, "none")
    kind === nothing && return w
    kind == "cmfmc" || _warn_unread!(w, section, ("clamp",), "[convection]",
                                     "unless kind = \"cmfmc\"")
    kind in ("tm5", "cmfmc_matrix") ||
        _warn_unread!(w, section, _COLLAB_LU_KEYS, "[convection]",
                      "unless kind = \"tm5\" or \"cmfmc_matrix\"")
    return w
end

function _chemistry_key_warnings!(w, section)
    section isa AbstractDict || return w
    _warn_unknown!(w, section, _CHEMISTRY_KEYS, "[chemistry]")
    _kind_string(section, "none") == "none" &&
        _warn_unread!(w, section, ("half_lives_seconds",), "[chemistry]", "with kind = \"none\"")
    return w
end

function _init_key_warnings!(w, init, label)
    init isa AbstractDict || return w
    _warn_unknown!(w, init, _INIT_KEYS, label)
    kind = _kind_string(init, "uniform")
    group = kind === nothing ? nothing : _init_kind_group(kind)
    group === nothing && return w
    read = ("kind", _INIT_KIND_KEYS[group]...)
    unread = [k for k in _INIT_KEYS if !(k in read)]
    _warn_unread!(w, init, unread, label, "with kind = \"$(kind)\"")
    _warn_shadowed!(w, init, ("south_value", "south"), label)
    return _warn_shadowed!(w, init, ("north_value", "north"), label)
end

function _surface_flux_key_warnings!(w, sf, label)
    sf isa AbstractDict || return w
    _warn_unknown!(w, sf, _SURFACE_FLUX_KEYS, label)
    kind = _kind_string(sf, "none")
    # No source: nothing but `kind` is read (an omitted kind means "none").
    kind == "none" && return _warn_unread!(w, sf, filter(!=("kind"), _SURFACE_FLUX_KEYS), label,
                                           "with kind = \"none\" (the default); no flux is emitted")
    kind === nothing || kind in _SURFACE_FLUX_MONTH_KINDS ||
        _warn_unread!(w, sf, ("month",), label, "unless kind = \"gridfed_fossil_co2\" or \"zhang_rn222\"")
    if get(sf, "time_varying", false) === true
        # A regular-grid series reads `files`, else the twelve months of
        # `file_pattern`, else `file`; a native cubed-sphere series reads
        # `file`. Both read every slice: never `time_index` or `month`.
        if kind in ("lmdz_co2", "gridfed_fossil_co2")
            _warn_shadowed!(w, sf, ("files", "file_pattern", "file"), label)
        elseif kind == "cs_native"
            _warn_unread!(w, sf, ("files", "file_pattern"), label, "by a cs_native series")
        end
        _warn_unread!(w, sf, ("time_index", "month"), label, "with time_varying = true")
    else
        _warn_unread!(w, sf, _SURFACE_FLUX_SERIES_KEYS, label, "unless time_varying = true")
        kind == "lmdz_co2" &&
            _warn_unread!(w, sf, ("time_index",), label, "by a static lmdz_co2 source (time mean)")
    end
    return w
end

# `table` is the tracer's TOML path, e.g. "tracers.co2".
function _tracer_key_warnings!(w, tracer, table)
    flat_init = _TRACER_FLAT_INIT_KEYS
    flat_flux = first.(_TRACER_FLAT_SURFACE_FLUX_KEYS)
    label = "[$(table)]"
    _warn_unknown!(w, tracer, ("init", "surface_flux", flat_init..., flat_flux...), label)
    haskey(tracer, "init") &&
        _warn_unread!(w, tracer, flat_init, label, "because [$(table).init] exists")
    haskey(tracer, "surface_flux") &&
        _warn_unread!(w, tracer, flat_flux, label, "because [$(table).surface_flux] exists")
    if haskey(tracer, "init")
        _init_key_warnings!(w, tracer["init"], "[$(table).init]")
    else    # flat init keys keep their names
        _init_key_warnings!(w, _tracer_init_cfg(tracer), label)
    end
    if haskey(tracer, "surface_flux")
        _surface_flux_key_warnings!(w, tracer["surface_flux"], "[$(table).surface_flux]")
    elseif any(k -> haskey(tracer, k), flat_flux)
        # Flat keys, checked under their `surface_flux` names.
        _surface_flux_key_warnings!(w, _tracer_surface_flux_cfg(tracer),
                                    "$(label) (flat surface_flux_* keys)")
    end
    return w
end

# `carried`: the run's tracer names, or `nothing` when they cannot be known.
function _output_key_warnings!(w, output, carried)
    output isa AbstractDict || return w
    _warn_unknown!(w, output, _OUTPUT_KEYS, "[output]")
    _warn_shadowed!(w, output, _OUTPUT_PATH_KEYS, "[output]")
    _warn_shadowed!(w, output, _SNAPSHOT_SCHEDULE_KEYS, "[output]")
    any(k -> haskey(output, k), _SNAPSHOT_INTERVAL_KEYS) ||
        _warn_unread!(w, output, ("start_hour", "stop_hour"), "[output]",
                      "without an interval key ($(join(_SNAPSHOT_INTERVAL_KEYS, ", ")))")
    format = get(output, "format", "netcdf")
    format isa AbstractString && lowercase(format) in ("binary_mmap", "binary", "mmap", "atmsnap") &&
        _warn_unread!(w, output, ("fields", "deflate_level", "shuffle"), "[output]",
                      "with format = \"binary_mmap\" (snapshots carry every field, uncompressed)")
    fields = get(output, "fields", nothing)
    fields isa AbstractDict || return w
    _warn_unknown!(w, fields, _OUTPUT_FIELDS_KEYS, "[output.fields]")
    _warn_shadowed!(w, fields, ("per_tracer", "tracer_fields"), "[output.fields]")
    _warn_shadowed!(w, fields, ("column_mass_per_area", "column_mass"), "[output.fields]")
    per_tracer = get(fields, "per_tracer", get(fields, "tracer_fields", nothing))
    per_tracer isa AbstractDict || return w
    for (name, entry) in pairs(per_tracer)
        label = "[output.fields.per_tracer.$(name)]"
        carried === nothing || String(name) in carried ||
            push!(w, "$(label): `$(name)` is not a tracer of the run; the setting is ignored.")
        entry isa AbstractDict || continue
        _warn_unknown!(w, entry, _TRACER_OUTPUT_FIELDS_KEYS, label)
        _warn_shadowed!(w, entry, ("column_mass_per_area", "column_mass"), label)
    end
    return w
end
