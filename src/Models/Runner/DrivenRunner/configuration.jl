# ===========================================================================
# TOML parsing — tracer specifications
# ===========================================================================

"""
    TransportTracerSpec

Validated runtime tracer configuration: tracer name, initial-condition
configuration, and optional surface-flux configuration.
"""
struct TransportTracerSpec{I <: AbstractInitialConditionSpec}
    name             :: Symbol
    init_spec        :: I
    init_cfg         :: Dict{String, Any}
    surface_flux_cfg :: Dict{String, Any}
end

function TransportTracerSpec(name::Symbol, init_cfg::Dict{String, Any},
                             surface_flux_cfg::Dict{String, Any})
    return TransportTracerSpec(name, initial_condition_spec(init_cfg), init_cfg,
                               surface_flux_cfg)
end

_copy_cfg_dict(cfg) = Dict{String, Any}(String(k) => v for (k, v) in pairs(cfg))

# Flat `[tracers.<name>]` keys read when the tracer has no `init` table, and
# the flat surface-flux keys (with their `surface_flux` names) read when it has
# no `surface_flux` table.
const _TRACER_FLAT_INIT_KEYS = ("kind", "background", "lon0_deg", "lat0_deg", "sigma_lon_deg",
                                "sigma_lat_deg", "amplitude", "south_value", "north_value",
                                "south", "north", "split_lat_deg", "file", "variable",
                                "humidity_variable", "time_index")
const _TRACER_FLAT_SURFACE_FLUX_KEYS = (("surface_flux_kind", "kind"),
                                        ("surface_flux_file", "file"),
                                        ("surface_flux_variable", "variable"),
                                        ("surface_flux_time_index", "time_index"),
                                        ("surface_flux_month", "month"),
                                        ("surface_flux_scale", "scale"))

function _tracer_init_cfg(tracer_cfg)
    if haskey(tracer_cfg, "init")
        return _copy_cfg_dict(tracer_cfg["init"])
    end
    cfg = Dict{String, Any}()
    for key in _TRACER_FLAT_INIT_KEYS
        haskey(tracer_cfg, key) && (cfg[key] = tracer_cfg[key])
    end
    isempty(cfg) && return Dict{String, Any}("kind" => "uniform", "background" => 0.0)
    return cfg
end

function _tracer_surface_flux_cfg(tracer_cfg)
    if haskey(tracer_cfg, "surface_flux")
        return _copy_cfg_dict(tracer_cfg["surface_flux"])
    end
    cfg = Dict{String, Any}()
    for (src_key, dst_key) in _TRACER_FLAT_SURFACE_FLUX_KEYS
        haskey(tracer_cfg, src_key) && (cfg[dst_key] = tracer_cfg[src_key])
    end
    return cfg
end

# ===========================================================================
# Region-tagged fluxes: `[tagged_fluxes.<group>]` → ordinary `[tracers.<name>]`
# ===========================================================================

const _TAGGED_FLUX_KEYS = ("flux", "mask_file", "mask_variable", "values", "names", "exclude",
                           "init")

# An array of integers (not Booleans) from the group's `key`.
function _tagged_int_list(g, key, group)
    raw = g[key]
    raw isa AbstractVector && all(x -> x isa Integer && !(x isa Bool), raw) || throw(ArgumentError(
        "[tagged_fluxes.$(group)].$(key) must be an array of integers; got $(repr(raw))"))
    values = Int64.(raw)
    allunique(values) || throw(ArgumentError(
        "[tagged_fluxes.$(group)].$(key) lists a value twice: $(repr(raw))"))
    return values
end

# Default tracer name of mask value `v` of `group`: `<group>_<v>`, with `m` for
# the sign of a negative value (`nee_m1`), so names stay valid identifiers.
_tagged_default_name(group, v::Integer) = v < 0 ? "$(group)_m$(-v)" : "$(group)_$(v)"

"""
    _expand_tagged_fluxes(cfg) -> cfg

Expand the `[tagged_fluxes.<group>]` tables of a run config into ordinary
`[tracers.<name>]` entries, one per selected value of the group's integer
region mask, and return a copy of `cfg` without `tagged_fluxes` (`cfg`
itself when it has none, so the expansion is idempotent and every reader of
`[tracers]` sees the tagged tracers). Each tracer gets the group's `init`
table (default uniform 0) and its `flux` table plus a
`InitialConditionIO.TaggedFluxRegion` under `TAGGED_REGION_KEY`, from which
`build_surface_flux_sources` builds the group with one read of the flux.

Keys of a group: `flux` (a surface-flux table, required), `mask_file` and
`mask_variable` (required; read by `GriddedIO.read_region_mask`), `values`
(the mask values to run; default every value present in the mask, fill
values never count), `names` (one per value; default `<group>_<value>`),
`exclude` (values never run) and `init`. An `ArgumentError` reports unknown
keys (with a suggestion), names without values or of another length,
duplicate values or names, a name that is already a tracer, a value that the
mask does not hold (listing the values it holds), a value both selected and
excluded, a group with no values left, a flux of kind `none` or with
`regridding = "bilinear"`, and tracers of one group without one species
(`InitialConditionIO.tagged_flux_species`).
"""
function _expand_tagged_fluxes(cfg::AbstractDict)
    haskey(cfg, "tagged_fluxes") || return cfg
    groups = cfg["tagged_fluxes"]
    groups isa AbstractDict || throw(ArgumentError(
        "[tagged_fluxes] must be a TOML table of groups ([tagged_fluxes.<group>]); got $(typeof(groups))"))
    # An empty table adds no tracers (and keeps the legacy single-tracer
    # fallback of a config without `[tracers]`).
    isempty(groups) && return Dict{String, Any}(String(k) => v for (k, v) in pairs(cfg)
                                                if String(k) != "tagged_fluxes")
    existing = get(cfg, "tracers", Dict{String, Any}())
    existing isa AbstractDict || throw(ArgumentError(
        "[tracers] must be a TOML table; got $(typeof(existing))"))
    tracers = Dict{String, Any}(String(k) => v for (k, v) in pairs(existing))
    origin = Dict{String, String}(name => "[tracers.$(name)]" for name in keys(tracers))
    for group in sort!([String(k) for k in keys(groups)])
        g = groups[group]
        label = "[tagged_fluxes.$(group)]"
        g isa AbstractDict || throw(ArgumentError("$(label) must be a TOML table; got $(typeof(g))"))
        unknown = unknown_key_messages(g, _TAGGED_FLUX_KEYS)
        isempty(unknown) || throw(ArgumentError("$(label): unknown option(s) $(join(unknown, ", "))."))
        for key in ("flux", "mask_file", "mask_variable")
            haskey(g, key) || throw(ArgumentError("$(label) requires `$(key)`"))
        end
        flux = g["flux"]
        flux isa AbstractDict || throw(ArgumentError(
            "$(label).flux must be a surface-flux table, e.g. { kind = \"lmdz_co2\", species = \"co2\" }"))
        init = get(g, "init", Dict{String, Any}("kind" => "uniform", "background" => 0.0))
        init isa AbstractDict || throw(ArgumentError("$(label).init must be a table like [tracers.X.init]"))
        for key in ("mask_file", "mask_variable")
            g[key] isa AbstractString || throw(ArgumentError("$(label).$(key) must be a string"))
        end
        haskey(flux, TAGGED_REGION_KEY) && throw(ArgumentError(
            "$(label).flux: unknown option(s) $(TAGGED_REGION_KEY) (an internal key)."))
        _surface_flux_kind(flux) === :none && throw(ArgumentError(
            "$(label).flux needs a flux kind other than \"none\""))
        # Bilinear sampling renormalises each field on its own, so the tagged
        # tracers would not sum to the flux (and small regions would be refused).
        _regridding_method(flux) isa BilinearFluxRegridding && throw(ArgumentError(
            "$(label).flux: tagged fluxes need conservative regridding; remove " *
            "regridding = \"bilinear\""))
        mask_file = expand_data_path(String(g["mask_file"]))
        mask_variable = String(g["mask_variable"])
        present = region_mask_values(read_region_mask(mask_file, mask_variable))
        exclude = haskey(g, "exclude") ? _tagged_int_list(g, "exclude", group) : Int64[]
        values = if haskey(g, "values")
            selected = _tagged_int_list(g, "values", group)
            absent = filter(v -> !insorted(v, present), selected)
            isempty(absent) || throw(ArgumentError(
                "$(label).values: $(join(absent, ", ")) $(length(absent) == 1 ? "is" : "are") not " *
                "in mask '$(mask_variable)' of $(mask_file); it holds $(join(present, ", "))"))
            both = intersect(selected, exclude)
            isempty(both) || throw(ArgumentError(
                "$(label): $(join(both, ", ")) $(length(both) == 1 ? "is" : "are") both in " *
                "`values` and in `exclude`"))
            selected
        else
            filter(v -> !(v in exclude), present)
        end
        isempty(values) && throw(ArgumentError(
            "$(label): no mask values left to run (mask holds $(isempty(present) ? "no values" :
            join(present, ", ")); exclude = $(exclude))"))
        names = if haskey(g, "names")
            haskey(g, "values") || throw(ArgumentError(
                "$(label).names needs `values`: the names are matched to the values by position"))
            raw = g["names"]
            raw isa AbstractVector && all(x -> x isa AbstractString, raw) || throw(ArgumentError(
                "$(label).names must be an array of strings"))
            length(raw) == length(values) || throw(ArgumentError(
                "$(label): names has $(length(raw)) entries but values has $(length(values))"))
            String.(raw)
        else
            [_tagged_default_name(group, v) for v in values]
        end
        for (name, v) in zip(names, values)
            haskey(origin, name) && throw(ArgumentError(
                "$(label): tracer name `$(name)` (value $(v)) is already defined by $(origin[name])"))
            origin[name] = "$(label) value $(v)"
            sf = _copy_cfg_dict(flux)
            sf[TAGGED_REGION_KEY] = TaggedFluxRegion(group, mask_file, mask_variable, v)
            tracers[name] = Dict{String, Any}("init" => _copy_cfg_dict(init), "surface_flux" => sf)
        end
        tagged_flux_species(group, names, flux)
    end
    out = Dict{String, Any}(String(k) => v for (k, v) in pairs(cfg) if String(k) != "tagged_fluxes")
    out["tracers"] = tracers
    return out
end

function _parse_tracer_specs(cfg)
    cfg = _expand_tagged_fluxes(cfg)
    tracers_cfg = get(cfg, "tracers", nothing)
    tracers_cfg isa AbstractDict || return nothing
    names = sort!(collect(keys(tracers_cfg)))
    isempty(names) && throw(ArgumentError("config has [tracers] but no tracer sections"))
    return Tuple(TransportTracerSpec(Symbol(name),
                                     _tracer_init_cfg(tracers_cfg[name]),
                                     _tracer_surface_flux_cfg(tracers_cfg[name])) for name in names)
end

# Tracer specs of a structured (lat-lon, reduced-Gaussian) run: `[tracers]`,
# or without it the legacy single tracer `[run].tracer_name` (default CO2)
# initialised from `[init]`, without a surface flux. Cubed-sphere runs require
# `[tracers]`. `Analysis.RunContext` reads a run's tracers with this function.
function _structured_tracer_specs(cfg)
    specs = _parse_tracer_specs(cfg)
    specs === nothing || return specs
    run_cfg = get(cfg, "run", Dict{String, Any}())
    return (TransportTracerSpec(Symbol(get(run_cfg, "tracer_name", "CO2")),
                                _copy_cfg_dict(get(cfg, "init", Dict{String, Any}())),
                                Dict{String, Any}()),)
end

# ===========================================================================
# Execution architecture
# ===========================================================================

@inline _cfg_architecture_section(cfg) = get(cfg, "architecture", Dict{String, Any}())
@inline _cfg_architecture(cfg) = architecture_from_config(_cfg_architecture_section(cfg))

_cfg_float_type(cfg) =
    parse_float_type(get(get(cfg, "numerics", Dict{String, Any}()), "float_type", "Float64"),
                     "[numerics] float_type")

function _capture_config_error!(f, errors::Vector{String})
    try
        f()
    catch err
        push!(errors, sprint(showerror, err))
    end
    return errors
end

function _check_config_table!(value, label, errors)
    value isa AbstractDict && return true
    push!(errors, "$(label) must be a TOML table; got $(typeof(value)).")
    return false
end

# Check container shapes before calling get/pairs on their contents. In
# particular, malformed tracer subtables must not survive until initialization.
function _check_config_table_shapes!(cfg, errors)
    for name in ("input", "architecture", "numerics", "run", "advection",
                 "diffusion", "convection", "chemistry", "output", "tracers", "init",
                 "logging", "checks")
        haskey(cfg, name) || continue
        _check_config_table!(cfg[name], "[$name]", errors)
    end

    input = get(cfg, "input", nothing)
    if input isa AbstractDict && haskey(input, "staging")
        _check_config_table!(input["staging"], "[input.staging]", errors)
    end

    output = get(cfg, "output", nothing)
    if output isa AbstractDict
        for table in ("time_mean", "local_time")
            haskey(output, table) || continue
            _check_config_table!(output[table], "[output.$(table)]", errors) || continue
            haskey(output[table], "fields") &&
                _check_config_table!(output[table]["fields"], "[output.$(table).fields]", errors)
        end
    end
    if output isa AbstractDict && haskey(output, "observations") &&
       _check_config_table!(output["observations"], "[output.observations]", errors)
        sources = get(output["observations"], "sources", nothing)
        if sources !== nothing && !(sources isa AbstractVector)
            push!(errors, "[output.observations].sources must be an array of tables " *
                          "([[output.observations.sources]]); got $(typeof(sources)).")
        elseif sources isa AbstractVector
            for (i, source) in enumerate(sources)
                _check_config_table!(source, "[[output.observations.sources]] entry $i", errors)
            end
        end
    end

    tracers = get(cfg, "tracers", nothing)
    tracers isa AbstractDict || return nothing
    for (name, tracer) in pairs(tracers)
        label = "[tracers.$name]"
        _check_config_table!(tracer, label, errors) || continue
        for section in ("init", "surface_flux")
            haskey(tracer, section) || continue
            _check_config_table!(tracer[section], "[tracers.$name.$section]", errors)
        end
    end
    return nothing
end

_run_window_index(run_cfg, key, default) =
    config_int(get(run_cfg, key, default), "[run] $key")

function _check_run_window_bounds!(cfg, errors::Vector{String})
    run_cfg = get(cfg, "run", Dict{String, Any}())
    _capture_config_error!(errors) do
        start_window = _run_window_index(run_cfg, "start_window", 1)
        start_window >= 1 ||
            throw(ArgumentError("[run] start_window must be >= 1; got $(start_window)."))
        # Programmatic configurations may use nothing for an open-ended run.
        if get(run_cfg, "stop_window", nothing) !== nothing
            stop_window = _run_window_index(run_cfg, "stop_window", nothing)
            stop_window >= start_window ||
                throw(ArgumentError("[run] stop_window=$(stop_window) must be >= start_window=$(start_window)."))
        end
    end
    return nothing
end

"""
    validate_config(cfg::AbstractDict) -> (ok::Bool, errors::Vector{String})

Check a driven runtime config and return errors without opening binary readers
or allocating model state. Checks cover runtime table shapes (including
`input.staging`, tracer `init` and `surface_flux` subtables, and
`output.observations` with its `sources` array), resolved binary paths,
numeric type, backend/float compatibility, the `[output.observations]`
contract, and integer run-window bounds. The `[advection]`, `[diffusion]`,
`[convection]`, `[chemistry]`, `[logging]` and `[checks]` sections,
`air_mass_reset_mode`, `physics_cadence`, `air_mass_update` (against the
advection scheme and reset mode), `prefetch_windows` and `[output]` are
parsed as the run parses them; decay
half-lives must name a tracer, and an enabled `[output]` needs both a path and
snapshot times (an explicit `hours = []` means none). Shape errors are
reported before value checks. `[tagged_fluxes]` groups are expanded into
tracers first (`_expand_tagged_fluxes`, which reads each group's mask), and
the checks see those tracers. Window indices accept integers, not Booleans or
floating-point values. Explicit halo widths must be integers ≥ 1; input dates
accept TOML dates or date strings in either input mode.

Unknown keys and tables are returned as errors (with a suggestion when a known
key is close, e.g. `order` → `ppm_order`); known keys that the chosen kind or
another setting leaves unread are logged as warnings (e.g. `value` unless
`[diffusion] kind = "constant"`).

This is not a complete physics or binary validation. Topology and payload
capability checks run when `run_driven_simulation` inspects the first binary,
which also checks that snapshot hours fall on met-window ends.
GPU backend auto-detection can load and probe optional runtime packages.
"""
function validate_config(cfg::AbstractDict)
    errors = String[]

    # `[tagged_fluxes]` groups become ordinary tracers before any other check.
    expanded = Ref{Any}(nothing)
    _capture_config_error!(() -> (expanded[] = _expand_tagged_fluxes(cfg)), errors)
    isempty(errors) || return false, errors
    cfg = expanded[]

    _check_config_table_shapes!(cfg, errors)
    isempty(errors) || return false, errors

    input_cfg = get(cfg, "input", nothing)
    if !(input_cfg isa AbstractDict)
        push!(errors, "[input] must be a TOML table with `binary_paths` or `folder + start_date + end_date`.")
    else
        binary_paths_ref = Ref(String[])
        _capture_config_error!(errors) do
            paths = expand_binary_paths(input_cfg)
            isempty(paths) && throw(ArgumentError("[input] resolved to an empty binary list."))
            binary_paths_ref[] = paths
        end
        for path in binary_paths_ref[]
            isfile(path) || push!(errors, "[input] resolved path does not exist: $(path)")
        end
    end

    ft_ref = Ref{Union{Nothing, DataType}}(nothing)
    _capture_config_error!(errors) do
        ft_ref[] = _cfg_float_type(cfg)
    end
    _capture_config_error!(errors) do
        arch = _cfg_architecture(cfg)
        ft_ref[] === nothing || assert_float_type!(arch, ft_ref[])
    end

    tracers_cfg = get(cfg, "tracers", nothing)
    if tracers_cfg !== nothing && isempty(tracers_cfg)
        push!(errors, "[tracers] was provided but contains no tracer subtables.")
    end

    # `[output.observations]` is parsed here so unknown keys and invalid
    # choices fail before any binary is opened. Observation files themselves
    # are per-day templates and are resolved by the runner.
    output_cfg = get(cfg, "output", nothing)
    if output_cfg isa AbstractDict
        spec_ref = Ref{Any}(nothing)
        _capture_config_error!(errors) do
            spec_ref[] = observation_output_spec(output_cfg)
        end
        spec = spec_ref[]
        if spec !== nothing
            _capture_config_error!(errors) do
                _check_observation_time_origin(spec, cfg)
            end
            _capture_config_error!(errors) do
                _check_observation_tracers(spec, cfg)
            end
            _capture_config_error!(errors) do
                _check_observation_placement_files(spec)
            end
        end
    end

    _check_run_window_bounds!(cfg, errors)
    _check_physics_and_output_settings!(cfg, ft_ref[], errors)

    findings = _config_key_findings(cfg)
    append!(errors, findings.errors)
    # The tracers of a `[tagged_fluxes]` group share one flux table: report a
    # repeated message once.
    unique!(errors)
    for warning in findings.warnings
        @warn "Run config: $(warning)"
    end
    return isempty(errors), errors
end

# `[run] air_mass_update` against the advection scheme and the reset mode, as
# `DrivenSimulation` resolves it. An `[advection]` table or reset mode that does
# not parse is reported by its own check; the value is not checked then.
function _check_air_mass_update!(cfg, run_cfg, errors)
    advects, mode = try
        (advects_air_mass(advection_spec(_advection_section(cfg))),
         _normalize_air_mass_reset_mode(get(run_cfg, "air_mass_reset_mode",
                                            DEFAULT_AIR_MASS_RESET_MODE)))
    catch err
        err isa ArgumentError || rethrow()
        return errors
    end
    return _capture_config_error!(errors) do
        resolve_air_mass_update(get(run_cfg, "air_mass_update", nothing);
                                advects_air_mass = advects, air_mass_reset = mode)
    end
end

# The physics, run and output settings, parsed as the run would parse them so
# that their errors appear with the others, before any binary is opened.
function _check_physics_and_output_settings!(cfg, FT, errors)
    _check_input_expectations!(get(cfg, "input", nothing), errors)
    _check_surface_flux_sources!(get(cfg, "tracers", nothing), errors)
    _check_time_varying_flux_origin!(cfg, errors)
    _capture_config_error!(() -> advection_spec(_advection_section(cfg)), errors)
    _capture_config_error!(() -> binary_cfl_check(cfg), errors)
    _capture_config_error!(() -> run_logging(cfg), errors)
    _capture_config_error!(() -> run_checks(cfg), errors)
    _capture_config_error!(() -> diffusion_spec(_diffusion_section(cfg)), errors)
    _capture_config_error!(() -> convection_spec(_convection_section(cfg)), errors)
    _capture_config_error!(() -> chemistry_spec(_chemistry_section(cfg)), errors)
    _check_decay_tracers!(cfg, errors)

    run_cfg = get(cfg, "run", Dict{String, Any}())
    if run_cfg isa AbstractDict
        for key in ("Hp", "halo_padding")
            haskey(run_cfg, key) && _capture_config_error!(errors) do
                config_int(run_cfg[key], "[run].$(key)"; min = 1)
            end
        end
        haskey(run_cfg, "reset_air_mass_each_window") && push!(errors,
            "run.reset_air_mass_each_window was replaced by run.air_mass_reset_mode = " *
            "\"none\", \"preserve_vmr\", or \"preserve_tracer_mass\"")
        _capture_config_error!(errors) do
            _normalize_air_mass_reset_mode(get(run_cfg, "air_mass_reset_mode",
                                               DEFAULT_AIR_MASS_RESET_MODE))
        end
        _capture_config_error!(() -> _resolve_physics_cadence(get(run_cfg, "physics_cadence", nothing)),
                               errors)
        _check_air_mass_update!(cfg, run_cfg, errors)
        _capture_config_error!(() -> _config_bool(run_cfg, "prefetch_windows", true,
                                                  "[run].prefetch_windows"), errors)
    end

    output_cfg = get(cfg, "output", nothing)
    if output_cfg isa AbstractDict && FT !== nothing
        # The run end comes from the binaries; without it an omitted stop_hour
        # must not be capped (the runner applies the binaries' length).
        _capture_config_error!(errors) do
            spec = runtime_output_spec(output_cfg, FT; default_cap_hours = floatmax(Float64))
            check_output_targets(spec, output_cfg)
        end
        # `[output.time_mean]` / `[output.local_time]`; their period against the
        # met window and the 00 UTC start are checked when the binaries are open.
        _capture_config_error!(() -> aggregate_output_specs(output_cfg, FT), errors)
        _capture_config_error!(() -> _check_output_path_collisions(output_cfg, FT), errors)
    end
    return errors
end

# A sample day for `_check_output_path_collisions`: every daily template is
# expanded with the same label and index, as one day of the run expands them.
const _SAMPLE_DAY_LABEL = "20000101"

"""
    _check_output_path_collisions(output_cfg, FT)

Reject two outputs that would write one file on the same day: snapshots,
`[output.time_mean]`, `[output.local_time]` and both observation files,
each path expanded for one sample day as its writer expands it (a daily
template and a plain path can name the same file). Parse errors of the
individual tables are reported by their own checks.
"""
function _check_output_path_collisions(output_cfg::AbstractDict, ::Type{FT}) where {FT}
    paths = Tuple{String, String}[]
    snapshots = try
        runtime_output_spec(output_cfg, FT; default_cap_hours = floatmax(Float64))
    catch err
        err isa ArgumentError || rethrow()
        nothing
    end
    snapshots !== nothing && output_enabled(snapshots) && push!(paths,
        ("[output].path", _output_path_for_partition(snapshots, snapshots.partition,
                                                     _SAMPLE_DAY_LABEL, 1)))
    specs = try
        aggregate_output_specs(output_cfg, FT)
    catch err
        err isa ArgumentError || rethrow()
        []
    end
    for spec in specs
        push!(paths, (aggregate_label(spec) * ".path", aggregate_output_path(spec, _SAMPLE_DAY_LABEL, 1)))
    end
    observations = try
        observation_output_spec(output_cfg)
    catch err
        err isa ArgumentError || rethrow()
        NoObservationOutput()
    end
    if observations isa ObservationOutputSpec
        for mode in (SoundingMode(), SiteMode())
            push!(paths, ("[output.observations].path",
                          observation_output_path(observations, mode, _SAMPLE_DAY_LABEL, 1)))
        end
    end
    for i in eachindex(paths), j in 1:(i - 1)
        paths[i][2] == paths[j][2] && throw(ArgumentError(
            "$(paths[j][1]) and $(paths[i][1]) would both write $(paths[i][2]) " *
            "(day $(_SAMPLE_DAY_LABEL) shown); give them distinct paths"))
    end
    return nothing
end

# The `[input]` expectations compared with the first binary's header: their
# types are checked here, the comparison when the binary is open.
function _check_input_expectations!(input, errors)
    input isa AbstractDict || return errors
    if haskey(input, "expected_nlevel")
        n = input["expected_nlevel"]
        n isa Integer && !(n isa Bool) && n >= 1 || push!(errors,
            "[input].expected_nlevel must be a positive integer; got $(repr(n))")
    end
    haskey(input, "required_preprocessor_contract") &&
        !(input["required_preprocessor_contract"] isa AbstractString) && push!(errors,
            "[input].required_preprocessor_contract must be a string; got " *
            "$(repr(input["required_preprocessor_contract"]))")
    for key in ("require_adaptive_substeps", "validate_replay")
        _capture_config_error!(() -> _config_bool(input, key, false, "[input].$(key)"), errors)
    end
    return errors
end

# Surface-flux kind, regridding method, coverage opt-in and species of every
# tracer (nested or flat). A source (kind other than "none") needs a species:
# `species` or a tracer name that implies one.
function _check_surface_flux_sources!(tracers, errors)
    tracers isa AbstractDict || return errors
    for (name, tracer) in pairs(tracers)
        tracer isa AbstractDict || continue
        sf = get(tracer, "surface_flux", nothing)
        sf = sf === nothing ? _tracer_surface_flux_cfg(tracer) : sf
        sf isa AbstractDict || continue
        kind = nothing
        _capture_config_error!(() -> (kind = _surface_flux_kind(sf)), errors)
        haskey(sf, "regridding") && _capture_config_error!(() -> _regridding_method(sf), errors)
        haskey(sf, "hold_outside_coverage") &&
            _capture_config_error!(() -> _hold_outside_coverage(sf), errors)
        kind === nothing || kind === :none ||
            _capture_config_error!(() -> surface_flux_species(Symbol(name), sf), errors)
    end
    return errors
end

# Time-varying surface fluxes place their slices on the run clock, which needs
# the run's first day: without `[input].start_date` there is no calendar anchor.
function _check_time_varying_flux_origin!(cfg, errors)
    tracers = get(cfg, "tracers", nothing)
    tracers isa AbstractDict || return errors
    input = get(cfg, "input", nothing)
    input isa AbstractDict && haskey(input, "start_date") && return errors
    for (name, tracer) in pairs(tracers)
        tracer isa AbstractDict || continue
        sf = get(tracer, "surface_flux", nothing)
        sf = sf === nothing ? _tracer_surface_flux_cfg(tracer) : sf
        sf isa AbstractDict && get(sf, "time_varying", false) === true &&
            push!(errors, "tracers.$(name): a time-varying surface flux needs [input].start_date " *
                          "(the run's first day) to place its time slices")
    end
    return errors
end

# A decay half-life for a tracer the run does not carry fails at the first
# chemistry step; report it here instead. Without [tracers] the run carries
# the single legacy tracer `[run].tracer_name` (lat-lon and reduced Gaussian).
function _check_decay_tracers!(cfg, errors)
    chemistry = _chemistry_section(cfg)
    chemistry isa AbstractDict || return errors
    kind = get(chemistry, "kind", "none")
    kind isa AbstractString && lowercase(kind) == "decay" || return errors
    half_lives = get(chemistry, "half_lives_seconds", nothing)
    half_lives isa AbstractDict || return errors
    tracers = get(cfg, "tracers", nothing)
    run_cfg = get(cfg, "run", Dict{String, Any}())
    carried = tracers isa AbstractDict ? sort!([String(k) for k in keys(tracers)]) :
              run_cfg isa AbstractDict ? [String(get(run_cfg, "tracer_name", "CO2"))] :
              return errors
    for name in sort!([String(k) for k in keys(half_lives)])
        name in carried || push!(errors,
            "[chemistry.half_lives_seconds].$(name) names no tracer of the run " *
            "($(join(carried, ", ")))")
    end
    return errors
end

# A partial window range across files skips forcing between handoffs while
# carrying tracer state forward. Restrict these ranges to single-file debugging.
_check_multifile_window_range(driver, start_window, stop_window, file_count) =
    _check_multifile_window_count(total_windows(driver), start_window, stop_window, file_count)

# The same check for a binary of `last_window` windows (`Analysis.RunContext`
# applies it to every binary's header).
function _check_multifile_window_count(last_window::Integer, start_window, stop_window, file_count)
    file_count <= 1 && return nothing
    if start_window != 1 || (stop_window !== nothing && Int(stop_window) < last_window)
        throw(ArgumentError(
            "Partial window ranges with multiple input files would skip forcing " *
            "between files. Use one input file for a partial-window run, or " *
            "start_window=1 and the full window range for every file."))
    end
    return nothing
end
