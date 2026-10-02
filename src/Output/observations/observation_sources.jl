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
(`{YYYY}`, `{YYYYMMDD}`, `{YYMMDD}` and `*` wildcards expand per run day) and
declare whether they yield soundings or sites.
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
"Group by the `site_code` attribute plus intake height."
struct SiteCodeGrouping <: AbstractSiteGrouping end
"Group by rounded (lat, lon, intake height)."
struct LocationGrouping <: AbstractSiteGrouping end

"Encoding of a generic point table."
abstract type AbstractTableFormat end
"Derive the format from the file extension."
struct AutoTableFormat <: AbstractTableFormat end
struct CSVTableFormat <: AbstractTableFormat end
struct TOMLTableFormat <: AbstractTableFormat end
struct NetCDFTableFormat <: AbstractTableFormat end

"""
    OCO2LiteSource(path_template, quality_flag_max)

NASA OCO-2/OCO-3 Lite XCO2 files. Every sounding with
`xco2_quality_flag <= quality_flag_max` becomes a sounding request.
"""
struct OCO2LiteSource <: AbstractObservationSource
    path_template::String
    quality_flag_max::Int
end

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

Generic point list with columns `id, time, lat, lon[, altitude_agl]` as CSV,
TOML, or NetCDF.
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
source_keys(::Type{<:OCO2LiteSource}) = ("kind", "path", "mode", "quality_flag_max")
source_keys(::Type{<:ObsPackSource}) = ("kind", "path", "mode", "site_grouping")
source_keys(::Type{<:TableSource}) = ("kind", "path", "mode", "format")

const _OBSERVATION_MODES = (soundings = SoundingMode(), sites = SiteMode())
const _SITE_GROUPINGS = (site_code = SiteCodeGrouping(), location = LocationGrouping())
const _TABLE_FORMATS = (auto = AutoTableFormat(), csv = CSVTableFormat(),
                        toml = TOMLTableFormat(), netcdf = NetCDFTableFormat())

_is_plain_int(x) = x isa Integer && !(x isa Bool)

function _check_known_keys(cfg::AbstractDict, allowed, label::AbstractString)
    unknown = sort!([String(k) for k in keys(cfg) if !(String(k) in allowed)])
    isempty(unknown) || throw(ArgumentError(
        "Unknown $(label) option(s): $(join(unknown, ", ")). " *
        "Supported: $(join(allowed, ", "))."))
    return nothing
end

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
    mode = _source_mode(cfg, label; default = SoundingMode())
    mode isa SoundingMode || throw(ArgumentError(
        "$(label): oco2_lite sources only support mode = \"soundings\""))
    qmax = get(cfg, "quality_flag_max", 0)
    _is_plain_int(qmax) && qmax >= 0 || throw(ArgumentError(
        "$(label).quality_flag_max must be a non-negative integer; got $(repr(qmax))"))
    return OCO2LiteSource(path, Int(qmax))
end

function _parse_source(::Type{<:ObsPackSource}, cfg, path, label)
    mode = _source_mode(cfg, label)
    grouping = haskey(cfg, "site_grouping") ?
        _parse_choice(cfg["site_grouping"], _SITE_GROUPINGS, "$(label).site_grouping") :
        SiteCodeGrouping()
    return ObsPackSource(path, mode, grouping)
end

function _parse_source(::Type{<:TableSource}, cfg, path, label)
    mode = _source_mode(cfg, label)
    format = haskey(cfg, "format") ?
        _parse_choice(cfg["format"], _TABLE_FORMATS, "$(label).format") :
        AutoTableFormat()
    return TableSource(path, mode, format)
end
