abstract type AbstractOutputSchedule end
abstract type AbstractOutputPartition end
abstract type AbstractLayerSelection end

struct ExplicitSnapshotSchedule <: AbstractOutputSchedule
    hours::Vector{Float64}
end

struct IntervalSnapshotSchedule <: AbstractOutputSchedule
    start_hour::Float64
    interval_hours::Float64
    stop_hour::Float64
end

struct SingleOutputFile <: AbstractOutputPartition end
struct DailyOutputFiles <: AbstractOutputPartition end

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

struct RuntimeOutputSpec{S <: AbstractOutputSchedule,
                         P <: AbstractOutputPartition,
                         O <: SnapshotWriteOptions,
                         F <: OutputFieldSpec}
    path::String
    schedule::S
    partition::P
    options::O
    fields::F
    enabled::Bool
    format::Symbol   # :netcdf (default) or :binary_mmap (offline NetCDF conversion)
end

# Backward-compatible 6-arg constructor: defaults format to :netcdf.
function RuntimeOutputSpec(path::AbstractString, schedule::AbstractOutputSchedule,
                            partition::AbstractOutputPartition,
                            options::SnapshotWriteOptions,
                            fields::OutputFieldSpec, enabled::Bool)
    return RuntimeOutputSpec(String(path), schedule, partition, options, fields,
                              enabled, :netcdf)
end

function ExplicitSnapshotSchedule(hours::AbstractVector{<:Real})
    out = sort!(Float64.(hours))
    any(!isfinite, out) &&
        throw(ArgumentError("output snapshot hours must be finite"))
    return ExplicitSnapshotSchedule(out)
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
_schedule_has_times(schedule::ExplicitSnapshotSchedule) = !isempty(schedule.hours)
_schedule_has_times(schedule::IntervalSnapshotSchedule) = schedule.stop_hour >= schedule.start_hour
output_enabled(spec::RuntimeOutputSpec) =
    spec.enabled && !isempty(spec.path) && _schedule_has_times(spec.schedule)
output_path(spec::RuntimeOutputSpec) = spec.path
output_fields(spec::RuntimeOutputSpec) = spec.fields

layer_selection_label(::FullLayerSelection) = :full
layer_selection_label(::SelectedLayerSelection) = :selected
layer_selection_label(::NoLayerSelection) = :none
layer_selection(fields::TracerOutputFields) = layer_selection_label(fields.layers)
air_mass_layer_selection(fields::OutputFieldSpec) = layer_selection_label(fields.air_mass_layers)
layer_selection(s::FullLayerSelection) = layer_selection_label(s)
layer_selection(s::SelectedLayerSelection) = layer_selection_label(s)
layer_selection(s::NoLayerSelection) = layer_selection_label(s)

function _parse_layer_selection(value, key::AbstractString)
    s = lowercase(String(value))
    if s in ("full", "all", "true")
        return FullLayerSelection()
    elseif s in ("selected", "select", "levels")
        return SelectedLayerSelection()
    elseif s in ("none", "off", "false", "column", "columns")
        return NoLayerSelection()
    else
        throw(ArgumentError("$(key) must be \"full\", \"selected\", or \"none\", got $(repr(s))"))
    end
end

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
    levels = Int.(value)
    any(<=(0), levels) &&
        throw(ArgumentError("[output.fields].levels must contain positive 1-based model levels"))
    return sort!(unique(levels))
end

function _tracer_fields_from_cfg(cfg::AbstractDict, fallback::TracerOutputFields;
                                 key_prefix::AbstractString = "[output.fields]")
    layers = haskey(cfg, "layers") ?
             _parse_layer_selection(cfg["layers"], "$(key_prefix).layers") :
             fallback.layers
    column_mean = _config_bool(get(cfg, "column_mean", fallback.column_mean),
                               "$(key_prefix).column_mean")
    column_mass = _config_bool(get(cfg, "column_mass_per_area",
                                   get(cfg, "column_mass", fallback.column_mass_per_area)),
                               "$(key_prefix).column_mass_per_area")
    return TracerOutputFields(layers, column_mean, column_mass)
end

function _parse_tracer_overrides(fields_cfg::AbstractDict,
                                 fallback::TracerOutputFields)
    raw = get(fields_cfg, "per_tracer",
              get(fields_cfg, "tracer_fields", Dict{String, Any}()))
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
    default_tracer = TracerOutputFields(
        _parse_layer_selection(get(fields_cfg, "layers", "full"), "[output.fields].layers"),
        _config_bool(get(fields_cfg, "column_mean", true),
                     "[output.fields].column_mean"),
        _config_bool(get(fields_cfg, "column_mass_per_area",
                         get(fields_cfg, "column_mass", true)),
                     "[output.fields].column_mass_per_area"),
    )
    air_layers = _parse_layer_selection(get(fields_cfg, "air_mass_layers",
                                            get(fields_cfg, "layers", "full")),
                                        "[output.fields].air_mass_layers")
    return OutputFieldSpec(
        _parse_tracer_names(get(fields_cfg, "tracers", nothing)),
        _parse_levels(get(fields_cfg, "levels", nothing)),
        default_tracer,
        _parse_tracer_overrides(fields_cfg, default_tracer),
        air_layers,
        _config_bool(get(fields_cfg, "air_mass", true), "[output.fields].air_mass"),
        _config_bool(get(fields_cfg, "air_mass_per_area", true),
                     "[output.fields].air_mass_per_area"),
        _config_bool(get(fields_cfg, "column_air_mass_per_area", true),
                     "[output.fields].column_air_mass_per_area"),
    )
end

output_field_spec() = output_field_spec(Dict{String, Any}())
tracer_fields(fields::OutputFieldSpec, name::Symbol) =
    get(fields.tracer_overrides, name, fields.default_tracer)

function _output_options(output_cfg::AbstractDict, ::Type{FT},
                         format::Symbol = :netcdf) where FT <: AbstractFloat
    # On-disk snapshot precision is independent of the model compute type FT.
    # NetCDF defaults to FT (back-compatible), but the binary/ATMSNAP payload is
    # Float32-only on disk (`binary_writer.jl`), so for that format the on-disk
    # dtype is coerced to Float32. Float32 snapshots are ample for
    # visualization/diagnostics even when the model integrates in Float64; the
    # Float64 precision benefit lives in the in-run transport accumulation.
    # SnapshotFrame also records a compensated Float64 global total per tracer;
    # ATMSNAP stores those totals in its JSON header even though its spatial
    # payload remains Float32.
    on_disk = FT
    if format === :binary_mmap && on_disk !== Float32
        @info "Binary (ATMSNAP) output is Float32-only on disk; storing Float32 \
               snapshots while the model integrates in $(FT)."
        on_disk = Float32
    end
    return SnapshotWriteOptions(float_type = on_disk,
                                deflate_level = Int(get(output_cfg, "deflate_level", 0)),
                                shuffle = _config_bool(get(output_cfg, "shuffle", true),
                                                       "[output].shuffle"))
end

function _output_path(output_cfg::AbstractDict, default_path::AbstractString)
    raw = get(output_cfg, "path",
              get(output_cfg, "snapshot_file",
                  get(output_cfg, "filename", default_path)))
    return expand_data_path(String(raw))
end

function _output_partition(output_cfg::AbstractDict)
    split = lowercase(String(get(output_cfg, "split", "single")))
    if split in ("single", "one_file", "one-file")
        return SingleOutputFile()
    elseif split in ("daily", "per_day", "per-day", "day")
        return DailyOutputFiles()
    else
        throw(ArgumentError("[output].split must be \"single\" or \"daily\", got $(repr(split))"))
    end
end

function _output_schedule(output_cfg::AbstractDict;
                          default_cap_hours::Real,
                          fallback_hours::AbstractVector{<:Real})
    if haskey(output_cfg, "snapshot_hours")
        return ExplicitSnapshotSchedule(output_cfg["snapshot_hours"])
    elseif haskey(output_cfg, "hours")
        return ExplicitSnapshotSchedule(output_cfg["hours"])
    end

    cadence_hours = if haskey(output_cfg, "snapshot_interval_hours")
        Float64(output_cfg["snapshot_interval_hours"])
    elseif haskey(output_cfg, "cadence_hours")
        Float64(output_cfg["cadence_hours"])
    elseif haskey(output_cfg, "interval_hours")
        Float64(output_cfg["interval_hours"])
    elseif haskey(output_cfg, "cadence_seconds")
        Float64(output_cfg["cadence_seconds"]) / 3600.0
    elseif haskey(output_cfg, "interval_seconds")
        Float64(output_cfg["interval_seconds"]) / 3600.0
    else
        nothing
    end
    if cadence_hours !== nothing
        start_hour = Float64(get(output_cfg, "start_hour", 0.0))
        stop_hour = Float64(get(output_cfg, "stop_hour", default_cap_hours))
        return IntervalSnapshotSchedule(start_hour = start_hour,
                                        interval_hours = cadence_hours,
                                        stop_hour = stop_hour)
    end
    return ExplicitSnapshotSchedule(fallback_hours)
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
    format = _parse_output_format(get(output_cfg, "format", "netcdf"))
    options = _output_options(output_cfg, FT, format)
    fields = output_field_spec(get(output_cfg, "fields", Dict{String, Any}()))
    enabled = _config_bool(get(output_cfg, "enabled", true), "[output].enabled")
    path = _output_path(output_cfg, default_path)
    return RuntimeOutputSpec(path, schedule, partition, options, fields, enabled, format)
end

function _parse_output_format(value)
    s = lowercase(String(value))
    s in ("netcdf", "nc") && return :netcdf
    s in ("binary_mmap", "binary", "mmap", "atmsnap") && return :binary_mmap
    throw(ArgumentError(
        "[output].format must be \"netcdf\" or \"binary_mmap\", got \"$(value)\""))
end

function _insert_suffix_before_extension(path::AbstractString, suffix::AbstractString)
    root, ext = splitext(String(path))
    return isempty(ext) ? string(root, suffix, ".nc") : string(root, suffix, ext)
end

# Shared by snapshot and observation output: substitute `{date}`, `{YYYYMMDD}`
# and `{day}` when present, otherwise insert the day label before the extension.
function _substitute_day_template(path::AbstractString, date_label::AbstractString,
                                  day_index::Integer)
    day = lpad(string(day_index), 3, '0')
    if occursin("{date}", path) || occursin("{YYYYMMDD}", path) || occursin("{day}", path)
        out = replace(path, "{date}" => date_label)
        out = replace(out, "{YYYYMMDD}" => date_label)
        return replace(out, "{day}" => day)
    end
    label = isempty(date_label) ? day : date_label
    return _insert_suffix_before_extension(path, "_" * label)
end

output_path_for_day(spec::RuntimeOutputSpec, date_label::AbstractString, day_index::Integer) =
    _substitute_day_template(spec.path, date_label, day_index)
