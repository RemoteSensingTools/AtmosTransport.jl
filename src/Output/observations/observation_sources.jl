# ---------------------------------------------------------------------------
# Observation source descriptors for `[output.observations]`.
#
# A source says where sampling requests come from and whether they are
# soundings (time-stamped point events, e.g. OCO-2 Lite files) or sites
# (fixed stations sampled at every met-window end, e.g. NOAA ObsPack).
# Modes, groupings, and table formats are singleton types so readers and
# writers dispatch on them instead of branching on symbols. This file owns
# the typed descriptors and their TOML validation; the readers that turn a
# source into requests live alongside them.
# ---------------------------------------------------------------------------

"""
    AbstractObservationSource

Where sampling requests come from. Concrete sources carry a path template
(`{YYYYMMDD}` or `{date}`, `{YYMMDD}`, `{YYYY}`, `{MM}`, `{DD}` substituted per
run day; `*` and `?` wildcards in the file name) and declare whether they
yield point events (`SoundingMode`) or station series (`SiteMode`).
"""
abstract type AbstractObservationSource end

"How a source's records are sampled."
abstract type AbstractObservationMode end
"Time-stamped point events; each record is sampled at its own time."
struct SoundingMode <: AbstractObservationMode end
"Fixed stations; written at every met-window end."
struct SiteMode <: AbstractObservationMode end
mode_label(::SoundingMode) = :soundings
mode_label(::SiteMode) = :sites

"How ObsPack records are grouped into sites."
abstract type AbstractSiteGrouping end
"""
One site per dataset file and distinct intake height. The id is the dataset
name (ObsPack files already encode site, platform, and lab), suffixed with
`_<h>magl` only when a file holds several intake heights.
"""
struct SiteCodeGrouping <: AbstractSiteGrouping end
"""
One site per distinct (lat, lon, intake height) rounded to 0.01° and 0.1 m,
with ids `<site_code>_<lat>N_<lon>E_<h>magl` in signed decimal degrees.
"""
struct LocationGrouping <: AbstractSiteGrouping end

"Encoding of a generic point table."
abstract type AbstractTableFormat end
"Derive the format from the file extension (`.csv`, `.toml`, `.nc`/`.nc4`)."
struct AutoTableFormat <: AbstractTableFormat end
"Comma-separated with a header line; `#` comments; no quoting. List-valued `times` use `;`."
struct CSVTableFormat <: AbstractTableFormat end
"`[[soundings]]` or `[[sites]]` arrays of tables; `times` may be a TOML array."
struct TOMLTableFormat <: AbstractTableFormat end
"One-dimensional NetCDF variables along the record dimension; CF `time` units."
struct NetCDFTableFormat <: AbstractTableFormat end

"Which satellite records become requests."
abstract type AbstractQualityFilter end
"""
    QualityFlagFilter(variable, max)

`quality_filter = "flag_max"`: keep records whose integer flag `variable` is
`<= max` (Lite files: `xco2_quality_flag`, 0 = good).
"""
struct QualityFlagFilter <: AbstractQualityFilter
    variable::String
    max::Int
    function QualityFlagFilter(variable::AbstractString, max::Integer)
        max isa Bool && throw(ArgumentError("quality flag max must be an integer, not a Bool"))
        return new(String(variable), Int(max))
    end
end
"""
    QualityFlagValues(variable, values)

`quality_filter = "flag_values"`: keep records whose integer flag `variable`
is one of `values`. Use it for categorical flags such as the OCO-2 v11 MIP
`assimilate_flag` (0 = not assimilated, 1 = assimilated, 2 = withheld), where
`<= max` cannot select the wanted set.
"""
struct QualityFlagValues <: AbstractQualityFilter
    variable::String
    values::Vector{Int}
    function QualityFlagValues(variable::AbstractString, values::AbstractVector{<:Integer})
        isempty(values) && throw(ArgumentError("quality flag values must not be empty"))
        any(v -> v isa Bool, values) && throw(ArgumentError("quality flag values must be integers"))
        return new(String(variable), sort!(unique!(Int.(values))))
    end
end
"`quality_filter = \"none\"`: keep every record (the OCO-2 v11 MIP co-samples all 10-s averages)."
struct NoQualityFilter <: AbstractQualityFilter end

"""
    OCO2LiteSource(path_template, quality_filter)

NASA OCO-2/OCO-3 Lite XCO2 files, or the OCO-2 v11 MIP 10-second average
files (same `sounding_id`, `time`, `latitude`, `longitude` layout). Records
passing `quality_filter` become point events.
"""
struct OCO2LiteSource{Q <: AbstractQualityFilter} <: AbstractObservationSource
    path_template::String
    quality_filter::Q
    OCO2LiteSource(path_template::AbstractString, quality_filter::Q) where {Q <: AbstractQualityFilter} =
        new{Q}(String(path_template), quality_filter)
end
OCO2LiteSource(path_template::AbstractString, quality_flag_max::Integer) =
    OCO2LiteSource(path_template, QualityFlagFilter("xco2_quality_flag", quality_flag_max))

"""
    ObsPackSource(path_template, mode, site_grouping)

NOAA ObsPack NetCDF dataset files. `SiteMode()` groups records into fixed
stations by `site_grouping`; `SoundingMode()` samples every record at its
own time.
"""
struct ObsPackSource{M <: AbstractObservationMode,
                     G <: AbstractSiteGrouping} <: AbstractObservationSource
    path_template::String
    mode::M
    site_grouping::G
end

"""
    TableSource(path_template, mode, format)

Generic point list as CSV, TOML, or NetCDF. Point events need `id`, `time`
(ISO-8601 UTC), `lat`, `lon`; sites need `id`, `lat`, `lon`. Both accept
`elevation` (m asl) and `intake_height` (m above ground) or `altitude`
(m asl, converted with `elevation`). Sites may add `start_time`/`end_time`
([`TimeRange`](@ref)) or `times` ([`TimeList`](@ref)).
"""
struct TableSource{M <: AbstractObservationMode,
                   F <: AbstractTableFormat} <: AbstractObservationSource
    path_template::String
    mode::M
    format::F
end

source_kind(::Type{<:OCO2LiteSource}) = :oco2_lite
source_kind(::Type{<:ObsPackSource}) = :obspack
source_kind(::Type{<:TableSource}) = :table
source_kind(source::AbstractObservationSource) = source_kind(typeof(source))
source_mode(::OCO2LiteSource) = SoundingMode()
source_mode(source::ObsPackSource) = source.mode
source_mode(source::TableSource) = source.mode
source_path_template(source::AbstractObservationSource) = source.path_template

# One table maps the TOML `kind` to its type; the per-type methods below hold
# the allowed keys and the parser, so adding a kind touches one type only.
const _SOURCE_TYPES = (oco2_lite = OCO2LiteSource, obspack = ObsPackSource, table = TableSource)
source_keys(::Type{<:OCO2LiteSource}) = ("kind", "path", "mode", "quality_filter", "quality_variable",
                                          "quality_flag_max", "quality_flag_values")
source_keys(::Type{<:ObsPackSource}) = ("kind", "path", "mode", "site_grouping")
source_keys(::Type{<:TableSource}) = ("kind", "path", "mode", "format")

const _OBSERVATION_MODES = (soundings = SoundingMode(), sites = SiteMode())
const _SITE_GROUPINGS = (site_code = SiteCodeGrouping(), location = LocationGrouping())
const _TABLE_FORMATS = (auto = AutoTableFormat(), csv = CSVTableFormat(),
                        toml = TOMLTableFormat(), netcdf = NetCDFTableFormat())
const _QUALITY_FILTERS = (flag_max = QualityFlagFilter, flag_values = QualityFlagValues,
                          none = NoQualityFilter)
# Keys each filter consumes; the other filters' keys are rejected so a
# setting can never be silently ignored.
quality_filter_keys(::Type{QualityFlagFilter}) = ("quality_variable", "quality_flag_max")
quality_filter_keys(::Type{QualityFlagValues}) = ("quality_variable", "quality_flag_values")
quality_filter_keys(::Type{NoQualityFilter}) = ()
const _QUALITY_FILTER_KEYS = ("quality_variable", "quality_flag_max", "quality_flag_values")

_is_plain_int(x) = x isa Integer && !(x isa Bool)

# Map a config string onto one of the singleton choices in `choices`.
function _parse_choice(value, choices::NamedTuple, label::AbstractString)
    value isa AbstractString ||
        throw(ArgumentError("$(label) must be a string; got $(repr(value))"))
    key = Symbol(lowercase(String(value)))
    haskey(choices, key) || throw(ArgumentError(
        "$(label) must be one of $(join(string.(keys(choices)), ", ")); got $(repr(value))"))
    return choices[key]
end

function _source_path(cfg::AbstractDict, label::AbstractString)
    haskey(cfg, "path") || throw(ArgumentError("$(label) requires a `path`"))
    raw = cfg["path"]
    raw isa AbstractString && !isempty(raw) ||
        throw(ArgumentError("$(label).path must be a non-empty string"))
    return expand_data_path(String(raw))
end

function _source_mode(cfg::AbstractDict, label::AbstractString; default = nothing)
    if !haskey(cfg, "mode")
        default === nothing && throw(ArgumentError(
            "$(label) requires `mode = \"soundings\"` or `mode = \"sites\"`"))
        return default
    end
    return _parse_choice(cfg["mode"], _OBSERVATION_MODES, "$(label).mode")
end

"""
    observation_source_from_cfg(cfg, label) -> AbstractObservationSource

Build one source from a `[[output.observations.sources]]` table. Unknown keys
and invalid choices fail with an `ArgumentError` naming the table.
"""
function observation_source_from_cfg(cfg::AbstractDict, label::AbstractString)
    haskey(cfg, "kind") || throw(ArgumentError("$(label) requires a `kind`"))
    kind_raw = cfg["kind"]
    kind_raw isa AbstractString ||
        throw(ArgumentError("$(label).kind must be a string; got $(repr(kind_raw))"))
    kind = Symbol(lowercase(String(kind_raw)))
    haskey(_SOURCE_TYPES, kind) || throw(ArgumentError(
        "$(label).kind must be one of $(join(string.(keys(_SOURCE_TYPES)), ", ")); " *
        "got $(repr(kind_raw))"))
    T = _SOURCE_TYPES[kind]
    _check_known_keys(cfg, source_keys(T), label)
    return _parse_source(T, cfg, _source_path(cfg, label), label)
end

function _parse_source(::Type{<:OCO2LiteSource}, cfg, path, label)
    _check_oco2_mode(_source_mode(cfg, label; default = SoundingMode()), label)
    return OCO2LiteSource(path, _parse_quality_filter(cfg, label))
end

_check_oco2_mode(::SoundingMode, label) = nothing
_check_oco2_mode(::AbstractObservationMode, label) = throw(ArgumentError(
    "$(label): oco2_lite sources only support mode = \"soundings\""))

function _parse_quality_filter(cfg, label)
    T = _parse_choice(get(cfg, "quality_filter", "flag_max"), _QUALITY_FILTERS, "$(label).quality_filter")
    for key in _QUALITY_FILTER_KEYS
        haskey(cfg, key) && !(key in quality_filter_keys(T)) && throw(ArgumentError(
            "$(label).$(key) has no effect with quality_filter = " *
            "\"$(_choice_name(_QUALITY_FILTERS, T))\""))
    end
    return _quality_filter(T, cfg, label)
end

_choice_name(choices::NamedTuple, value) = String(findfirst(==(value), choices))

function _quality_variable(cfg, label)
    variable = get(cfg, "quality_variable", "xco2_quality_flag")
    variable isa AbstractString && !isempty(strip(variable)) || throw(ArgumentError(
        "$(label).quality_variable must be a non-empty variable name; got $(repr(variable))"))
    return String(variable)
end

_quality_filter(::Type{NoQualityFilter}, cfg, label) = NoQualityFilter()

function _quality_filter(::Type{QualityFlagFilter}, cfg, label)
    qmax = get(cfg, "quality_flag_max", 0)
    _is_plain_int(qmax) && qmax >= 0 || throw(ArgumentError(
        "$(label).quality_flag_max must be a non-negative integer; got $(repr(qmax))"))
    return QualityFlagFilter(_quality_variable(cfg, label), qmax)
end

function _quality_filter(::Type{QualityFlagValues}, cfg, label)
    haskey(cfg, "quality_flag_values") || throw(ArgumentError(
        "$(label): quality_filter = \"flag_values\" requires `quality_flag_values`, e.g. [1]"))
    values = cfg["quality_flag_values"]
    values isa AbstractVector && !isempty(values) && all(_is_plain_int, values) || throw(ArgumentError(
        "$(label).quality_flag_values must be a non-empty array of integers; got $(repr(values))"))
    return QualityFlagValues(_quality_variable(cfg, label), Int.(values))
end

function _parse_source(::Type{<:ObsPackSource}, cfg, path, label)
    mode = _source_mode(cfg, label)
    _check_grouping_mode(mode, haskey(cfg, "site_grouping"), label)
    grouping = haskey(cfg, "site_grouping") ?
        _parse_choice(cfg["site_grouping"], _SITE_GROUPINGS, "$(label).site_grouping") :
        SiteCodeGrouping()
    return ObsPackSource(path, mode, grouping)
end

# Point events are never grouped, so a grouping there would be ignored silently.
_check_grouping_mode(::AbstractObservationMode, has_grouping::Bool, label) = nothing
_check_grouping_mode(::SoundingMode, has_grouping::Bool, label) = has_grouping && throw(ArgumentError(
    "$(label).site_grouping only applies to mode = \"sites\""))

function _parse_source(::Type{<:TableSource}, cfg, path, label)
    mode = _source_mode(cfg, label)
    format = haskey(cfg, "format") ?
        _parse_choice(cfg["format"], _TABLE_FORMATS, "$(label).format") :
        AutoTableFormat()
    return TableSource(path, mode, format)
end
