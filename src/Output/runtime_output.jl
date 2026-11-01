abstract type AbstractOutputSchedule end
abstract type AbstractOutputPartition end
abstract type AbstractLayerSelection end
abstract type AbstractOutputFormat end

struct ExplicitSnapshotSchedule <: AbstractOutputSchedule
    hours::Vector{Float64}

    function ExplicitSnapshotSchedule(hours::AbstractVector)
        out = sort!([config_float(h, "output snapshot hours") for h in hours])
        any(!isfinite, out) &&
            throw(ArgumentError("output snapshot hours must be finite"))
        return new(out)
    end
end

struct IntervalSnapshotSchedule <: AbstractOutputSchedule
    start_hour::Float64
    interval_hours::Float64
    stop_hour::Float64
end

struct SingleOutputFile <: AbstractOutputPartition end
struct DailyOutputFiles <: AbstractOutputPartition end

struct NetCDFOutput <: AbstractOutputFormat end
struct BinaryMmapOutput <: AbstractOutputFormat end

output_format_label(::NetCDFOutput) = :netcdf
output_format_label(::BinaryMmapOutput) = :binary_mmap

struct FullLayerSelection <: AbstractLayerSelection end
struct SelectedLayerSelection <: AbstractLayerSelection end
struct NoLayerSelection <: AbstractLayerSelection end

struct TracerOutputFields{L <: AbstractLayerSelection}
    layers::L
    column_mean::Bool
    column_mass_per_area::Bool
end

struct OutputFieldSpec{L <: AbstractLayerSelection}
    tracers::Union{Nothing, Vector{Symbol}}
    selected_levels::Vector{Int}
    default_tracer::TracerOutputFields
    tracer_overrides::Dict{Symbol, TracerOutputFields}
    air_mass_layers::L
    air_mass::Bool
    air_mass_per_area::Bool
    column_air_mass_per_area::Bool
end

"Metadata resolved from the typed physics objects of one runtime invocation."
mutable struct RuntimeOutputMetadata
    physics_attributes::Dict{String, String}
    time_origin::Union{Nothing, DateTime}
end

RuntimeOutputMetadata() = RuntimeOutputMetadata(Dict{String, String}(), nothing)

struct RuntimeOutputSpec{S <: AbstractOutputSchedule,
                         P <: AbstractOutputPartition,
                         O <: SnapshotWriteOptions,
                         F <: OutputFieldSpec,
                         M <: AbstractOutputFormat}
    path::String
    schedule::S
    partition::P
    options::O
    fields::F
    enabled::Bool
    format::M
    metadata::RuntimeOutputMetadata
end

# `[output]` keys (`runtime_output_spec`), in precedence order within each alias
# group: the first key present wins.
const OUTPUT_PATH_KEYS = ("path", "snapshot_file", "filename")
# An explicit hour list wins over an interval.
const _SNAPSHOT_HOURS_KEYS = ("snapshot_hours", "hours")
const _SNAPSHOT_SECONDS_KEYS = ("cadence_seconds", "interval_seconds")
const SNAPSHOT_INTERVAL_KEYS = ("snapshot_interval_hours", "cadence_hours", "interval_hours",
                                _SNAPSHOT_SECONDS_KEYS...)
const SNAPSHOT_SCHEDULE_KEYS = (_SNAPSHOT_HOURS_KEYS..., SNAPSHOT_INTERVAL_KEYS...)
const OUTPUT_KEYS = (OUTPUT_PATH_KEYS..., SNAPSHOT_SCHEDULE_KEYS..., "start_hour",
                      "stop_hour", "split", "format", "deflate_level", "shuffle", "enabled",
                      "fields", "observations")
const OUTPUT_COLUMN_MASS_KEYS = ("column_mass_per_area", "column_mass")
const OUTPUT_PER_TRACER_KEYS = ("per_tracer", "tracer_fields")
const OUTPUT_FIELDS_KEYS = ("tracers", "layers", "levels", "column_mean", OUTPUT_COLUMN_MASS_KEYS...,
                             "air_mass_layers", "air_mass", "air_mass_per_area",
                             "column_air_mass_per_area", OUTPUT_PER_TRACER_KEYS...)
const TRACER_OUTPUT_FIELDS_KEYS = ("layers", "column_mean", OUTPUT_COLUMN_MASS_KEYS...)

# Choice tables for `parse_choice`: the canonical spelling first, then aliases.
const _LAYER_SELECTIONS = (full = FullLayerSelection(), all = FullLayerSelection(),
                           var"true" = FullLayerSelection(),
                           selected = SelectedLayerSelection(), select = SelectedLayerSelection(),
                           levels = SelectedLayerSelection(),
                           none = NoLayerSelection(), off = NoLayerSelection(),
                           var"false" = NoLayerSelection(), column = NoLayerSelection(),
                           columns = NoLayerSelection())
const _OUTPUT_PARTITIONS = (single = SingleOutputFile(), one_file = SingleOutputFile(),
                            var"one-file" = SingleOutputFile(),
                            daily = DailyOutputFiles(), per_day = DailyOutputFiles(),
                            var"per-day" = DailyOutputFiles(), day = DailyOutputFiles())
const OUTPUT_FORMATS = (netcdf = NetCDFOutput(), nc = NetCDFOutput(),
                         binary_mmap = BinaryMmapOutput(), binary = BinaryMmapOutput(),
                         mmap = BinaryMmapOutput(), atmsnap = BinaryMmapOutput())
const _DEFAULT_TRACER_FIELDS = TracerOutputFields(FullLayerSelection(), true, true)

# The first of the alias keys `names` present in `cfg` (`nothing` if none), and
# its value (`default` if none).
_first_present_key(cfg::AbstractDict, names) =
    (i = findfirst(k -> haskey(cfg, k), names); i === nothing ? nothing : names[i])
function _first_present(cfg::AbstractDict, names, default)
    key = _first_present_key(cfg, names)
    return key === nothing ? default : cfg[key]
end

function IntervalSnapshotSchedule(; start_hour::Real=0.0,
                                  interval_hours::Real,
                                  stop_hour::Real)
    start = Float64(start_hour)
    interval = Float64(interval_hours)
    stop = Float64(stop_hour)
    isfinite(start) || throw(ArgumentError("output start_hour must be finite, got $(start_hour)"))
    isfinite(interval) && interval > 0 ||
        throw(ArgumentError("output cadence/interval hours must be finite and positive, got $(interval_hours)"))
    isfinite(stop) && stop >= start ||
        throw(ArgumentError("output stop_hour must be finite and >= start_hour, got $(stop_hour)"))
    return IntervalSnapshotSchedule(start, interval, stop)
end

function snapshot_hours(schedule::ExplicitSnapshotSchedule)
    return copy(schedule.hours)
end

function snapshot_hours(schedule::IntervalSnapshotSchedule)
    return collect(schedule.start_hour:schedule.interval_hours:schedule.stop_hour)
end

snapshot_hours(spec::RuntimeOutputSpec) = snapshot_hours(spec.schedule)
schedule_has_times(schedule::ExplicitSnapshotSchedule) = !isempty(schedule.hours)
schedule_has_times(schedule::IntervalSnapshotSchedule) = schedule.stop_hour >= schedule.start_hour
output_enabled(spec::RuntimeOutputSpec) =
    spec.enabled && !isempty(spec.path) && schedule_has_times(spec.schedule)
output_path(spec::RuntimeOutputSpec) = spec.path
output_fields(spec::RuntimeOutputSpec) = spec.fields

layer_selection_label(::FullLayerSelection) = :full
layer_selection_label(::SelectedLayerSelection) = :selected
layer_selection_label(::NoLayerSelection) = :none

function _parse_tracer_names(value; label::AbstractString = "[output.fields].tracers")
    if value === nothing
        return nothing
    elseif value isa AbstractString
        s = lowercase(String(value))
        s in ("*", "all") && return nothing
        s in ("none", "false", "off") && return Symbol[]
        return [Symbol(String(value))]
    elseif value isa AbstractVector && all(v -> v isa AbstractString, value)
        return Symbol[Symbol(String(v)) for v in value]
    else
        throw(ArgumentError("$(label) must be a string or array of tracer names; got $(repr(value))"))
    end
end

function _parse_levels(value)
    value === nothing && return Int[]
    value isa AbstractVector ||
        throw(ArgumentError("[output.fields].levels must be an array of 1-based model levels"))
    levels = Int[config_int(level, "[output.fields].levels"; min = 1) for level in value]
    return sort!(unique(levels))
end

function _tracer_fields_from_cfg(cfg::AbstractDict, fallback::TracerOutputFields;
                                 key_prefix::AbstractString = "[output.fields]")
    layers = haskey(cfg, "layers") ?
             parse_choice(cfg["layers"], _LAYER_SELECTIONS, "$(key_prefix).layers") :
             fallback.layers
    column_mean = config_bool(get(cfg, "column_mean", fallback.column_mean),
                              "$(key_prefix).column_mean")
    column_mass = config_bool(_first_present(cfg, OUTPUT_COLUMN_MASS_KEYS, fallback.column_mass_per_area),
                              "$(key_prefix).column_mass_per_area")
    return TracerOutputFields(layers, column_mean, column_mass)
end

function _parse_tracer_overrides(fields_cfg::AbstractDict,
                                 fallback::TracerOutputFields)
    raw = _first_present(fields_cfg, OUTPUT_PER_TRACER_KEYS, Dict{String, Any}())
    raw isa AbstractDict ||
        throw(ArgumentError("[output.fields].per_tracer must be a table keyed by tracer name"))
    out = Dict{Symbol, TracerOutputFields}()
    for (name, cfg) in pairs(raw)
        cfg isa AbstractDict ||
            throw(ArgumentError("[output.fields.per_tracer.$(name)] must be a table"))
        out[Symbol(String(name))] =
            _tracer_fields_from_cfg(cfg, fallback;
                                    key_prefix = "[output.fields.per_tracer.$(name)]")
    end
    return out
end

function output_field_spec(fields_cfg::AbstractDict)
    default_tracer = _tracer_fields_from_cfg(fields_cfg, _DEFAULT_TRACER_FIELDS)
    air_layers = parse_choice(get(fields_cfg, "air_mass_layers", get(fields_cfg, "layers", "full")),
                              _LAYER_SELECTIONS, "[output.fields].air_mass_layers")
    return OutputFieldSpec(
        _parse_tracer_names(get(fields_cfg, "tracers", nothing)),
        _parse_levels(get(fields_cfg, "levels", nothing)),
        default_tracer,
        _parse_tracer_overrides(fields_cfg, default_tracer),
        air_layers,
        config_bool(get(fields_cfg, "air_mass", true), "[output.fields].air_mass"),
        config_bool(get(fields_cfg, "air_mass_per_area", true),
                    "[output.fields].air_mass_per_area"),
        config_bool(get(fields_cfg, "column_air_mass_per_area", true),
                    "[output.fields].column_air_mass_per_area"),
    )
end

output_field_spec() = output_field_spec(Dict{String, Any}())
tracer_fields(fields::OutputFieldSpec, name::Symbol) =
    get(fields.tracer_overrides, name, fields.default_tracer)

function _output_options(output_cfg::AbstractDict, ::Type{FT},
                         format::AbstractOutputFormat = NetCDFOutput()) where FT <: AbstractFloat
    # On-disk snapshot precision is independent of the model compute type FT.
    # NetCDF defaults to FT (back-compatible), but the binary/ATMSNAP payload is
    # Float32-only on disk (`binary_writer.jl`), so for that format the on-disk
    # dtype is coerced to Float32. Float32 snapshots are ample for
    # visualization/diagnostics even when the model integrates in Float64; the
    # Float64 precision benefit lives in the in-run transport accumulation.
    # SnapshotFrame also records a compensated Float64 global total per tracer;
    # ATMSNAP stores those totals in its JSON header even though its spatial
    # payload remains Float32.
    on_disk = _output_float_type(format, FT)
    _log_output_float_coercion(format, FT)
    return SnapshotWriteOptions(float_type = on_disk,
                                deflate_level = config_int(get(output_cfg, "deflate_level", 0),
                                                           "[output].deflate_level"),
                                shuffle = config_bool(get(output_cfg, "shuffle", true),
                                                      "[output].shuffle"))
end

_output_float_type(::NetCDFOutput, ::Type{FT}) where FT <: AbstractFloat = FT
_output_float_type(::BinaryMmapOutput, ::Type{<:AbstractFloat}) = Float32
_log_output_float_coercion(::NetCDFOutput, ::Type{<:AbstractFloat}) = nothing
_log_output_float_coercion(::BinaryMmapOutput, ::Type{Float32}) = nothing
function _log_output_float_coercion(::BinaryMmapOutput, ::Type{FT}) where FT <: AbstractFloat
    @info "Binary (ATMSNAP) output is Float32-only on disk; storing Float32 \
           snapshots while the model integrates in $(FT)."
    return nothing
end

_output_path(output_cfg::AbstractDict, default_path::AbstractString) =
    expand_data_path(String(_first_present(output_cfg, OUTPUT_PATH_KEYS, default_path)))

_output_partition(output_cfg::AbstractDict) =
    parse_choice(get(output_cfg, "split", "single"), _OUTPUT_PARTITIONS, "[output].split")

function _output_schedule(output_cfg::AbstractDict;
                          default_cap_hours::Real,
                          fallback_hours::AbstractVector{<:Real})
    hours_key = _first_present_key(output_cfg, _SNAPSHOT_HOURS_KEYS)
    hours_key === nothing || return ExplicitSnapshotSchedule(output_cfg[hours_key])
    interval_key = _first_present_key(output_cfg, SNAPSHOT_INTERVAL_KEYS)
    interval_key === nothing && return ExplicitSnapshotSchedule(fallback_hours)
    interval = config_float(output_cfg[interval_key], "[output].$(interval_key)")
    cadence_hours = interval_key in _SNAPSHOT_SECONDS_KEYS ? interval / 3600.0 : interval
    start_hour = config_float(get(output_cfg, "start_hour", 0.0), "[output].start_hour")
    stop_hour = config_float(get(output_cfg, "stop_hour", default_cap_hours), "[output].stop_hour")
    return IntervalSnapshotSchedule(start_hour = start_hour,
                                    interval_hours = cadence_hours,
                                    stop_hour = stop_hour)
end

"""
    runtime_output_spec(output_cfg, FT; default_path="", default_cap_hours=8760, fallback_hours=[])

Parse the run-time output contract from TOML-compatible `[output]` settings.

Preferred keys are `path`, `cadence_hours` or `hours`, and
`split = "single" | "daily"`. Legacy `snapshot_file`,
`snapshot_hours`, and `snapshot_interval_hours` remain accepted.
"""
function runtime_output_spec(output_cfg::AbstractDict, ::Type{FT};
                             default_path::AbstractString = "",
                             default_cap_hours::Real = 8760.0,
                             fallback_hours::AbstractVector{<:Real} = Float64[]) where
        {FT <: AbstractFloat}
    schedule = _output_schedule(output_cfg;
                                default_cap_hours = default_cap_hours,
                                fallback_hours = fallback_hours)
    partition = _output_partition(output_cfg)
    format = parse_choice(get(output_cfg, "format", "netcdf"), OUTPUT_FORMATS, "[output].format")
    options = _output_options(output_cfg, FT, format)
    fields = output_field_spec(get(output_cfg, "fields", Dict{String, Any}()))
    enabled = config_bool(get(output_cfg, "enabled", true), "[output].enabled")
    path = _output_path(output_cfg, default_path)
    return RuntimeOutputSpec(path, schedule, partition, options, fields, enabled, format,
                             RuntimeOutputMetadata())
end

"""
    check_output_targets(spec, output_cfg)

Reject snapshot settings that would silently write nothing while
`[output].enabled` is true: a path without any snapshot-schedule key, or
snapshot times without a path. An explicitly empty list (`hours = []`) is a
deliberate "no snapshots" (the path then only names the timing CSV of
`ATMOSTR_TIMERS` runs).
"""
function check_output_targets(spec::RuntimeOutputSpec, output_cfg::AbstractDict)
    spec.enabled || return nothing
    has_path = !isempty(output_path(spec))
    has_times = schedule_has_times(spec.schedule)
    has_path && !any(k -> haskey(output_cfg, k), SNAPSHOT_SCHEDULE_KEYS) && throw(ArgumentError(
        "[output] sets a path but no snapshot times, so no snapshot would be written; " *
        "add `hours = [...]` or `cadence_hours`, set `hours = []` for none, or set " *
        "`enabled = false`"))
    has_times && !has_path && throw(ArgumentError(
        "[output] sets snapshot times but no `path`, so no snapshot would be written; " *
        "add `path` or set `enabled = false`"))
    return nothing
end

# Runtime physics attribute names carry the `physics_` prefix in every sink
# (`RuntimeOutputSpec` metadata, NetCDF global attributes, ATMSNAP header).
function _physics_attribute_key(key)
    startswith(key, "physics_") || throw(ArgumentError(
        "runtime physics attribute $(repr(key)) must start with `physics_`"))
    return key
end

function set_physics_attributes!(spec::RuntimeOutputSpec,
                                 attributes::AbstractDict{<:AbstractString, <:AbstractString})
    empty!(spec.metadata.physics_attributes)
    for (key, value) in pairs(attributes)
        spec.metadata.physics_attributes[String(_physics_attribute_key(key))] = String(value)
    end
    return spec
end

function set_time_origin!(spec::RuntimeOutputSpec, origin::DateTime)
    spec.metadata.time_origin = origin
    return spec
end

function _insert_suffix_before_extension(path::AbstractString, suffix::AbstractString)
    root, ext = splitext(String(path))
    return isempty(ext) ? string(root, suffix, ".nc") : string(root, suffix, ext)
end

"Whether an output path carries a per-day token (`{date}`, `{YYYYMMDD}`, `{day}`)."
_has_day_token(path::AbstractString) =
    occursin("{date}", path) || occursin("{YYYYMMDD}", path) || occursin("{day}", path)

# Shared by snapshot and observation output: substitute `{date}`, `{YYYYMMDD}`
# and `{day}` when present, otherwise insert the day label before the
# extension. Binaries without a date in their name fall back to the
# zero-padded day index, so daily files never collide on an empty label.
function _substitute_day_template(path::AbstractString, date_label::AbstractString,
                                  day_index::Integer)
    day = lpad(string(day_index), 3, '0')
    label = isempty(date_label) ? day : date_label
    if _has_day_token(path)
        out = replace(path, "{date}" => label)
        out = replace(out, "{YYYYMMDD}" => label)
        return replace(out, "{day}" => day)
    end
    return _insert_suffix_before_extension(path, "_" * label)
end

output_path_for_day(spec::RuntimeOutputSpec, date_label::AbstractString, day_index::Integer) =
    _substitute_day_template(spec.path, date_label, day_index)
