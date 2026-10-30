# ---------------------------------------------------------------------------
# `[output.observations]` — typed runtime contract for observation sampling.
#
# Sampling happens at met-window ends only (where state is physically
# complete). Soundings are blended linearly between the two bracketing ends
# by default; sites are written at every end.
# ---------------------------------------------------------------------------

"How point events are placed in time between met-window ends."
abstract type AbstractObservationTimeInterpolation end

"Blend the two met-window-end samples bracketing each sounding time (default)."
struct LinearWindowInterpolation <: AbstractObservationTimeInterpolation end

"Use the single met-window-end sample nearest to each sounding time."
struct NearestWindowSampling <: AbstractObservationTimeInterpolation end

time_interpolation_label(::LinearWindowInterpolation) = :linear
time_interpolation_label(::NearestWindowSampling) = :nearest_window

"Parsed `[output.observations]`: either `NoObservationOutput` or a spec."
abstract type AbstractObservationOutput end

"Default when `[output.observations]` is absent or `enabled = false`."
struct NoObservationOutput <: AbstractObservationOutput end

"""
    ObservationOutputSpec

Parsed `[output.observations]` table. `path` is a template: `_soundings` or
`_sites` is inserted before the extension and, with daily partitions,
`{date}` / `{YYYYMMDD}` / `{day}` are substituted per input binary exactly as
for snapshot output. `tracers === nothing` means every model tracer.
`start_time` is only consulted when `[input].start_date` is absent.
`placement` is the [`VerticalPlacementSpec`](@ref).
"""
struct ObservationOutputSpec{P <: AbstractOutputPartition,
                             TI <: AbstractObservationTimeInterpolation} <: AbstractObservationOutput
    path::String
    partition::P
    time_interpolation::TI
    tracers::Union{Nothing, Vector{Symbol}}
    write_profile_for_sites::Bool
    layer_height_temperature_kelvin::Float64
    start_time::Union{Nothing, DateTime}
    deflate_level::Int
    sources::Vector{AbstractObservationSource}
    placement::VerticalPlacementSpec
end

# Keyword constructor: the struct has many positional fields of similar types.
function ObservationOutputSpec(; path::AbstractString,
                                 partition::AbstractOutputPartition,
                                 time_interpolation::AbstractObservationTimeInterpolation,
                                 tracers::Union{Nothing, Vector{Symbol}},
                                 write_profile_for_sites::Bool,
                                 layer_height_temperature_kelvin::Real,
                                 start_time::Union{Nothing, DateTime},
                                 deflate_level::Integer,
                                 sources::Vector{AbstractObservationSource},
                                 placement::VerticalPlacementSpec)
    return ObservationOutputSpec(String(path), partition, time_interpolation, tracers,
                                 write_profile_for_sites, Float64(layer_height_temperature_kelvin),
                                 start_time, Int(deflate_level), sources, placement)
end

# The vertical-placement keys (`parse_vertical_placement`), shared with any
# config table that places observation records in the model column.
const VERTICAL_PLACEMENT_KEYS = ("surface_height_file", "dem_file", "dem_variable", "unplaced_records",
                                  "absolute_altitude_above_agl_m", "vertical_coordinates", "pressure_basis")
const _OBSERVATION_OUTPUT_KEYS = ("enabled", "path", "time_interpolation", "tracers",
                                  "write_profile_for_sites", "layer_height_temperature_kelvin",
                                  "start_time", "deflate_level", "sources", VERTICAL_PLACEMENT_KEYS...)
# Near-surface fallback for hypsometric layer heights when the binary carries
# no temperature. The diffusion helpers' 260 K is a mid-troposphere value and
# would bias intake-height placement in the lowest layers.
const _TIME_INTERPOLATIONS = (linear = LinearWindowInterpolation(),
                              nearest_window = NearestWindowSampling())
_parse_start_time(value) = parse_iso_utc(value, "[output.observations].start_time")

function _parse_observation_tracers(value)
    tracers = _parse_tracer_names(value; label = "[output.observations].tracers")
    tracers === nothing && return nothing
    isempty(tracers) && throw(ArgumentError(
        "[output.observations].tracers must name at least one tracer " *
        "(omit the key to sample every tracer)"))
    any(t -> isempty(String(t)), tracers) &&
        throw(ArgumentError("[output.observations].tracers must not contain empty names"))
    allunique(tracers) ||
        throw(ArgumentError("[output.observations].tracers must not repeat a tracer name"))
    return tracers
end

function _parse_layer_height_temperature(value)
    value isa Real && !(value isa Bool) && isfinite(value) && value > 0 ||
        throw(ArgumentError(
            "[output.observations].layer_height_temperature_kelvin must be a positive " *
            "number; got $(repr(value))"))
    return Float64(value)
end

function _parse_observation_deflate_level(value)
    _is_plain_int(value) && 0 <= value <= 9 || throw(ArgumentError(
        "[output.observations].deflate_level must be an integer in 0..9; got $(repr(value))"))
    return Int(value)
end

function _parse_optional_path(cfg::AbstractDict, key::AbstractString, label::AbstractString)
    haskey(cfg, key) || return nothing
    raw = cfg[key]
    raw isa AbstractString && !isempty(strip(raw)) || throw(ArgumentError(
        "$(label).$(key) must be a non-empty path; got $(repr(raw))"))
    return expand_data_path(String(raw))
end

function _parse_altitude_placement(cfg::AbstractDict, has_surface::Bool, label::AbstractString)
    haskey(cfg, "absolute_altitude_above_agl_m") || return AboveGroundPlacement()
    value = cfg["absolute_altitude_above_agl_m"]
    value isa Real && !(value isa Bool) && isfinite(value) && value >= 0 || throw(ArgumentError(
        "$(label).absolute_altitude_above_agl_m must be a number >= 0; got $(repr(value))"))
    has_surface || throw(ArgumentError(
        "$(label).absolute_altitude_above_agl_m places records by altitude above sea " *
        "level and needs the model surface height: set surface_height_file"))
    return AbsoluteAltitudeAbove(value)
end

# `vertical_coordinates`: a non-empty array of distinct coordinate names, in
# preference order → a tuple of `AbstractPlacementCoordinate`s.
function _parse_vertical_coordinates(cfg::AbstractDict, label::AbstractString)
    haskey(cfg, "vertical_coordinates") || return DEFAULT_VERTICAL_COORDINATES
    key = "$(label).vertical_coordinates"
    choices = join(("\"$(k)\"" for k in keys(_VERTICAL_COORDINATES)), ", ")
    raw = cfg["vertical_coordinates"]
    raw isa AbstractVector && all(v -> v isa AbstractString, raw) || throw(ArgumentError(
        "$(key) must be an array of coordinate names in preference order, from $(choices); got $(repr(raw))"))
    isempty(raw) && throw(ArgumentError(
        "$(key) must name at least one vertical coordinate ($(choices)); omit the key for the default " *
        "[\"height_above_ground\", \"pressure\"]"))
    coordinates = Tuple(_parse_choice(name, _VERTICAL_COORDINATES, key) for name in raw)
    allunique(coordinates) || throw(ArgumentError("$(key) must not repeat a coordinate; got $(repr(raw))"))
    return coordinates
end

"""
    parse_vertical_placement(cfg, label) -> VerticalPlacementSpec

Parse the vertical-placement keys (`VERTICAL_PLACEMENT_KEYS`) of the config
table `cfg`, named `label` in error messages (`"[output.observations]"`):
`surface_height_file`, `dem_file`/`dem_variable`, `unplaced_records`,
`absolute_altitude_above_agl_m`, `vertical_coordinates` and
`pressure_basis`. The caller checks for unknown keys. Keys that would have
no effect are errors: `dem_variable` without `dem_file`, and `dem_file` or
`absolute_altitude_above_agl_m` (both act on placements by height) when
`vertical_coordinates` leaves out `"height_above_ground"`. `pressure_basis`
applies to every coordinate (the layer heights are built on it too).
"""
function parse_vertical_placement(cfg::AbstractDict, label::AbstractString)
    surface_height_file = _parse_optional_path(cfg, "surface_height_file", label)
    dem_file = _parse_optional_path(cfg, "dem_file", label)
    dem_variable = get(cfg, "dem_variable", "z")
    dem_variable isa AbstractString && !isempty(strip(dem_variable)) || throw(ArgumentError(
        "$(label).dem_variable must be a non-empty variable name; got $(repr(dem_variable))"))
    dem_file === nothing && haskey(cfg, "dem_variable") && throw(ArgumentError(
        "$(label).dem_variable has no effect without dem_file"))
    unplaced = _parse_choice(get(cfg, "unplaced_records", "error"), _UNPLACED_POLICIES,
                             "$(label).unplaced_records")
    coordinates = _parse_vertical_coordinates(cfg, label)
    if !_uses_coordinate(coordinates, HeightAboveGroundCoordinate)
        for key in ("dem_file", "absolute_altitude_above_agl_m")
            haskey(cfg, key) && throw(ArgumentError(
                "$(label).$(key) acts on records placed by their height above the ground and has no " *
                "effect without \"height_above_ground\" in $(label).vertical_coordinates"))
        end
    end
    pressure_basis = _parse_choice(get(cfg, "pressure_basis", "total"), _PRESSURE_BASES, "$(label).pressure_basis")
    return VerticalPlacementSpec(; surface_height_file, dem_file, dem_variable = String(dem_variable),
                                 unplaced,
                                 altitude_placement = _parse_altitude_placement(cfg, surface_height_file !== nothing,
                                                                                label),
                                 coordinates, pressure_basis)
end

function _parse_observation_sources(obs_cfg::AbstractDict)
    haskey(obs_cfg, "sources") || throw(ArgumentError(
        "[output.observations] requires at least one [[output.observations.sources]] table"))
    raw = obs_cfg["sources"]
    raw isa AbstractVector && !isempty(raw) || throw(ArgumentError(
        "[output.observations].sources must be a non-empty array of tables " *
        "([[output.observations.sources]])"))
    sources = AbstractObservationSource[]
    for (i, entry) in enumerate(raw)
        label = "[[output.observations.sources]] entry $(i)"
        entry isa AbstractDict || throw(ArgumentError("$(label) must be a table"))
        push!(sources, observation_source_from_cfg(entry, label))
    end
    return sources
end

"""
    observation_output_spec(output_cfg; partition = nothing)

Parse `[output.observations]` from the `[output]` table. Returns
`NoObservationOutput()` when the subtable is absent or `enabled = false`,
otherwise an [`ObservationOutputSpec`](@ref). Unknown keys, a missing `path`
or `sources`, and invalid choices raise `ArgumentError`. The file partition
(one file vs. one per daily binary) follows `[output].split` unless
`partition` is given; `split` is only read when observations are enabled.
"""
function observation_output_spec(output_cfg::AbstractDict;
                                 partition::Union{Nothing, AbstractOutputPartition} = nothing)
    haskey(output_cfg, "observations") || return NoObservationOutput()
    obs_cfg = output_cfg["observations"]
    obs_cfg isa AbstractDict || throw(ArgumentError(
        "[output.observations] must be a TOML table; got $(typeof(obs_cfg))"))
    _check_known_keys(obs_cfg, _OBSERVATION_OUTPUT_KEYS, "[output.observations]")
    _config_bool(get(obs_cfg, "enabled", true), "[output.observations].enabled") ||
        return NoObservationOutput()

    raw_path = get(obs_cfg, "path", "")
    raw_path isa AbstractString && !isempty(raw_path) ||
        throw(ArgumentError("[output.observations] requires a non-empty `path`"))
    return ObservationOutputSpec(;
        path = expand_data_path(String(raw_path)),
        partition = partition === nothing ? _output_partition(output_cfg) : partition,
        time_interpolation = _parse_choice(get(obs_cfg, "time_interpolation", "linear"),
                                           _TIME_INTERPOLATIONS,
                                           "[output.observations].time_interpolation"),
        tracers = _parse_observation_tracers(get(obs_cfg, "tracers", nothing)),
        write_profile_for_sites = _config_bool(get(obs_cfg, "write_profile_for_sites", false),
                                               "[output.observations].write_profile_for_sites"),
        layer_height_temperature_kelvin = _parse_layer_height_temperature(
            get(obs_cfg, "layer_height_temperature_kelvin",
                DEFAULT_OBSERVATION_LAYER_TEMPERATURE_KELVIN)),
        start_time = _parse_start_time(get(obs_cfg, "start_time", nothing)),
        deflate_level = _parse_observation_deflate_level(get(obs_cfg, "deflate_level", 0)),
        sources = _parse_observation_sources(obs_cfg),
        placement = parse_vertical_placement(obs_cfg, "[output.observations]"))
end

"""
    observation_output_path(spec, mode, date_label, day_index) -> String

Resolve the file for `mode::AbstractObservationMode`. `_soundings` / `_sites`
is inserted before the extension. Daily runs substitute the day template
exactly like `output_path_for_day`; single-file runs substitute it only when
the path carries a `{date}` / `{YYYYMMDD}` / `{day}` token, using the first
day's label, and otherwise use the path as given.
"""
function observation_output_path(spec::ObservationOutputSpec{SingleOutputFile}, mode::AbstractObservationMode,
                                 date_label::AbstractString, day_index::Integer)
    path = _has_day_token(spec.path) ? _substitute_day_template(spec.path, date_label, day_index) : spec.path
    return _insert_suffix_before_extension(path, "_" * String(mode_label(mode)))
end

function observation_output_path(spec::ObservationOutputSpec{DailyOutputFiles},
                                 mode::AbstractObservationMode,
                                 date_label::AbstractString, day_index::Integer)
    day_path = _substitute_day_template(spec.path, date_label, day_index)
    return _insert_suffix_before_extension(day_path, "_" * String(mode_label(mode)))
end
