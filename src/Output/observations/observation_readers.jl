# ---------------------------------------------------------------------------
# Readers that turn `[[output.observations.sources]]` into sampling requests.
#
# `read_observation_requests(source, index, origin, dates)` expands the
# source's path template for the run days and calls `_read_requests_file!`
# once per file, dispatched on the source type and its mode. Times become
# Float64 seconds since `origin` (UTC). Satellite and ObsPack rows with fill
# values, impossible coordinates, or a failed quality filter are skipped and
# counted in `ReadStats`; generic tables are user-written, so an invalid row
# there is an error. `build_observation_set` merges every source, expands
# site time lists into point events, keeps only events inside the
# transported span, and logs what it kept, skipped, and dropped.
# ---------------------------------------------------------------------------

# -- path templates ---------------------------------------------------------

const _DATE_TOKENS = (("{YYYYMMDD}", dateformat"yyyymmdd"), ("{date}", dateformat"yyyymmdd"),
                      ("{YYMMDD}", dateformat"yymmdd"), ("{YYYY}", dateformat"yyyy"),
                      ("{MM}", dateformat"mm"), ("{DD}", dateformat"dd"))

_has_date_token(template::AbstractString) = any(occursin(tok, template) for (tok, _) in _DATE_TOKENS)
_has_wildcard(s::AbstractString) = occursin('*', s) || occursin('?', s)

function _substitute_date_tokens(template::AbstractString, date::Date)
    out = String(template)
    for (tok, fmt) in _DATE_TOKENS
        out = replace(out, tok => Dates.format(date, fmt))
    end
    return out
end

const _REGEX_SPECIALS = Set(".^\$|()[]{}+\\")

function _wildcard_regex(name::AbstractString)
    io = IOBuffer()
    print(io, '^')
    for ch in name
        if ch == '*'
            print(io, ".*")
        elseif ch == '?'
            print(io, '.')
        elseif ch in _REGEX_SPECIALS
            print(io, '\\', ch)
        else
            print(io, ch)
        end
    end
    print(io, '$')
    return Regex(String(take!(io)))
end

# Wildcards are supported in the file name only; the directory must be literal.
function _glob_basename(pattern::AbstractString, listings::Dict{String, Vector{String}})
    dir, name = dirname(pattern), basename(pattern)
    _has_wildcard(dir) && throw(ArgumentError(
        "observation path wildcards are only supported in the file name; got $(repr(pattern))"))
    entries = get!(listings, dir) do
        isdir(dir) ? sort!(readdir(dir)) : String[]
    end
    rx = _wildcard_regex(name)
    return [joinpath(dir, f) for f in entries if occursin(rx, f) && isfile(joinpath(dir, f))]
end

"""
    expand_observation_paths(template, dates) -> Vector{String}

Resolve a source path template for the run days: `{YYYYMMDD}` (alias
`{date}`), `{YYMMDD}`, `{YYYY}`, `{MM}`, `{DD}` are substituted per day and
`*` / `?` wildcards are expanded in the file name. Returns unique existing
paths in order. A path without date tokens must match at least one file;
templated paths may resolve to nothing on days without data.
"""
function expand_observation_paths(template::AbstractString, dates::AbstractVector{Date})
    templated = _has_date_token(template)
    candidates = templated ? [_substitute_date_tokens(template, d) for d in dates] : [String(template)]
    unique!(candidates)
    listings = Dict{String, Vector{String}}()
    paths = String[]
    for candidate in candidates
        if _has_wildcard(candidate)
            append!(paths, _glob_basename(candidate, listings))
        elseif isfile(candidate)
            push!(paths, candidate)
        elseif !templated
            throw(ArgumentError("observation source file not found: $(candidate)"))
        end
    end
    templated || !isempty(paths) || throw(ArgumentError(
        "observation source pattern matched no files: $(template)"))
    return unique!(paths)
end

# -- NetCDF helpers ---------------------------------------------------------

const _CF_UNIT_SECONDS = Dict(
    "second" => 1.0, "seconds" => 1.0, "sec" => 1.0, "s" => 1.0,
    "minute" => 60.0, "minutes" => 60.0, "min" => 60.0,
    "hour" => 3600.0, "hours" => 3600.0, "hr" => 3600.0, "h" => 3600.0,
    "day" => 86400.0, "days" => 86400.0, "d" => 86400.0)

# "1970-01-01 00:00:00", "...T00:00:00Z", "... UTC", "... +00:00". Non-zero
# offsets are rejected: every supported product is UTC. Unlike config times
# this accepts non-padded CF origins such as "1900-1-1", so it does not use
# the strict `_ISO_UTC_RE` gate.
function _cf_time_origin(s::AbstractString, label::AbstractString)
    t = String(strip(s))
    t = String(strip(chopsuffix(t, "UTC")))
    offset = match(r"([+-])(\d{1,2})(?::?(\d{2}))?$", t)
    if offset !== nothing && occursin(r"\d[T ]\d", t)
        hours = parse(Int, something(offset.captures[2], "0"))
        minutes = parse(Int, something(offset.captures[3], "0"))
        hours == 0 && minutes == 0 || throw(ArgumentError(
            "$(label): time origin $(repr(s)) has a non-UTC offset"))
        t = String(strip(t[1:prevind(t, offset.offset)]))
    end
    t = chopsuffix(t, "Z")
    parsed = tryparse(DateTime, replace(t, ' ' => 'T'))
    parsed === nothing && throw(ArgumentError("$(label): cannot parse CF time origin $(repr(s))"))
    return parsed
end

# Raw CF time values -> Float64 seconds since 1970-01-01T00:00:00 UTC (NaN stays NaN).
function _cf_unix_seconds(values::AbstractVector, units::AbstractString, label::AbstractString)
    m = match(r"^\s*(\w+)\s+since\s+(.+?)\s*$", String(units))
    m === nothing && throw(ArgumentError("$(label): cannot parse time units $(repr(units))"))
    scale = get(_CF_UNIT_SECONDS, lowercase(m.captures[1]), nothing)
    scale === nothing && throw(ArgumentError("$(label): unsupported time unit $(repr(m.captures[1]))"))
    origin_unix = datetime2unix(_cf_time_origin(m.captures[2], label))
    return Float64[origin_unix + Float64(v) * scale for v in values]
end

function _nc_variable(ds, candidates, label::AbstractString)
    for name in candidates
        haskey(ds, name) && return ds[name]
    end
    throw(ArgumentError("$(label) is missing variable $(join(repr.(candidates), " / "))"))
end

_scalar_attr(var, name) = (v = get(var.attrib, name, nothing); v isa AbstractArray && length(v) == 1 ? first(v) : v)

# Raw numeric values as Float64 with fill entries replaced by NaN. The fill
# comparison happens at the stored element type (a Float64 `missing_value`
# on Float32 data would otherwise never match), then CF packing is undone.
function _raw_numeric(var, label::AbstractString)
    raw = Array(var.var)
    T = eltype(raw)
    T <: Real || throw(ArgumentError("$(label): expected a numeric variable, got $(T)"))
    fill_attr = _scalar_attr(var, "_FillValue")
    fill_attr === nothing && (fill_attr = _scalar_attr(var, "missing_value"))
    fill_raw = fill_attr isa Real ? _fill_as(T, fill_attr) : nothing
    out = Vector{Float64}(undef, length(raw))
    @inbounds for (k, value) in enumerate(vec(raw))
        out[k] = (fill_raw !== nothing && value == fill_raw) ? NaN : Float64(value)
    end
    scale = _scalar_attr(var, "scale_factor")
    offset = _scalar_attr(var, "add_offset")
    if scale isa Real || offset isa Real
        a = scale isa Real ? Float64(scale) : 1.0
        b = offset isa Real ? Float64(offset) : 0.0
        @inbounds for k in eachindex(out)
            isnan(out[k]) || (out[k] = out[k] * a + b)
        end
    end
    return out
end

_fill_as(::Type{T}, fill::Real) where {T <: AbstractFloat} = T(fill)
function _fill_as(::Type{T}, fill::Real) where {T <: Integer}
    fill isa Integer && typemin(T) <= fill <= typemax(T) && return T(fill)
    fill isa AbstractFloat && isinteger(fill) && typemin(T) <= fill <= typemax(T) && return T(fill)
    return nothing
end

function _time_unix_seconds(var, label::AbstractString)
    units = get(var.attrib, "units", nothing)
    units isa AbstractString || throw(ArgumentError("$(label): time variable has no units attribute"))
    return _cf_unix_seconds(_raw_numeric(var, label), units, label)
end

_is_char_matrix(var) = eltype(var.var) <: AbstractChar && ndims(var) == 2

# String-valued variable stored either as NC_STRING or as an NC_CHAR matrix.
function _string_vector(var, label::AbstractString)
    raw = Array(var.var)   # NC_CHAR matrices must keep their (length, obs) shape
    if eltype(raw) <: AbstractString
        return String.(vec(raw))
    elseif eltype(raw) <: AbstractChar && ndims(raw) == 2
        return [String(rstrip(String(view(raw, :, k)), ('\0', ' '))) for k in axes(raw, 2)]
    end
    throw(ArgumentError("$(label): expected a string variable, got $(typeof(raw))"))
end

function _check_same_length(label::AbstractString, n::Integer, pairs...)
    for (name, v) in pairs
        length(v) == n || throw(ArgumentError(
            "$(label): variable $(name) has length $(length(v)); expected $(n)"))
    end
    return nothing
end

# -- generic entry point ----------------------------------------------------

"Rows a reader skipped instead of turning into requests."
Base.@kwdef mutable struct ReadStats
    skipped_invalid::Int = 0     # fill values or impossible coordinates/times
    skipped_quality::Int = 0     # rejected by the source's quality filter
    outside_window::Int = 0      # point events dropped by the reader's window prefilter
end

# Reader-side window prefilter, in seconds after the origin. Large satellite
# files (the MIP OCO-2 file spans the whole mission) are cut before their
# request records are built.
_in_window(::Nothing, t::Real) = true
_in_window((t0, t1)::Tuple{Float64, Float64}, t::Real) = t0 <= t <= t1

"""
    read_observation_requests(source, source_index, origin, dates) -> (soundings, sites)

Expand the source's path template for the run days and read every file into
`SoundingRequest`s and `SiteRequest`s (one of the two is empty, depending on
the source mode). Times are Float64 seconds since `origin`. Rows with fill
values or impossible coordinates are skipped; a source that resolves to no
file at all logs a warning.
"""
function read_observation_requests(source::AbstractObservationSource, source_index::Int,
                                   origin::DateTime, dates::AbstractVector{Date};
                                   stats::ReadStats = ReadStats(),
                                   window::Union{Nothing, Tuple{Float64, Float64}} = nothing)
    soundings = SoundingRequest[]
    sites = SiteRequest[]
    paths = expand_observation_paths(source_path_template(source), dates)
    isempty(paths) && @warn "observation source $(source_index) ($(source_kind(source))) matched no files for the run days" template = source_path_template(source)
    for path in paths
        _read_requests_file!(soundings, sites, source, path, source_index, origin, stats, window)
    end
    return soundings, sites
end

# -- OCO-2 Lite -------------------------------------------------------------

function _read_requests_file!(soundings::Vector{SoundingRequest}, ::Vector{SiteRequest},
                              source::OCO2LiteSource, path::AbstractString, source_index::Int,
                              origin::DateTime, stats::ReadStats, window)
    label = "oco2_lite $(path)"
    origin_unix = datetime2unix(origin)
    NCDataset(path, "r") do ds
        ids = vec(Array(_nc_variable(ds, ("sounding_id",), label).var))
        n = length(ids)
        times = _time_unix_seconds(_nc_variable(ds, ("time",), label), label)
        lats = _raw_numeric(_nc_variable(ds, ("latitude",), label), label)
        lons = _raw_numeric(_nc_variable(ds, ("longitude",), label), label)
        keep = _quality_mask(source.quality_filter, ds, n, label)
        _check_same_length(label, n, ("time", times), ("latitude", lats), ("longitude", lons))
        for k in 1:n
            if !keep[k]
                stats.skipped_quality += 1
            elseif isfinite(times[k]) && !_in_window(window, times[k] - origin_unix)
                stats.outside_window += 1
            elseif !(isvalid_lonlat(lons[k], lats[k]) && isfinite(times[k]) && !iszero(ids[k]))
                stats.skipped_invalid += 1
            else
                push!(soundings, SoundingRequest(string(ids[k]), times[k] - origin_unix,
                                                 lons[k], lats[k], source_index))
            end
        end
    end
    return nothing
end

_quality_mask(::NoQualityFilter, ds, n::Int, label) = trues(n)
function _quality_mask(filter::QualityFlagFilter, ds, n::Int, label)
    flags = _raw_numeric(_nc_variable(ds, (filter.variable,), label), label)
    _check_same_length(label, n, (filter.variable, flags))
    return [isfinite(f) && f <= filter.max for f in flags]
end
function _quality_mask(filter::QualityFlagValues, ds, n::Int, label)
    flags = _raw_numeric(_nc_variable(ds, (filter.variable,), label), label)
    _check_same_length(label, n, (filter.variable, flags))
    return [isfinite(f) && insorted(f, filter.values) for f in flags]
end

# -- NOAA ObsPack -----------------------------------------------------------

Base.@kwdef struct _ObsPackRecords
    dataset::String
    site_code::String
    times::Vector{Float64}       # unix seconds
    lats::Vector{Float64}
    lons::Vector{Float64}
    elevation::Vector{Float64}   # m asl, NaN when absent
    intake::Vector{Float64}      # m agl, NaN when absent
    ids::Vector{String}
    site_lat::Float64            # file attributes, NaN when absent
    site_lon::Float64
    site_elevation::Float64
end

_attr_float(ds, name) = (v = get(ds.attrib, name, nothing); v isa Real ? Float64(v) : NaN)

function _obspack_records(path::AbstractString)
    label = "obspack $(path)"
    dataset = first(splitext(basename(path)))
    return NCDataset(path, "r") do ds
        times = _time_unix_seconds(_nc_variable(ds, ("time",), label), label)
        n = length(times)
        lats = _raw_numeric(_nc_variable(ds, ("latitude",), label), label)
        lons = _raw_numeric(_nc_variable(ds, ("longitude",), label), label)
        elevation = haskey(ds, "elevation") ? _raw_numeric(ds["elevation"], label) : fill(NaN, n)
        intake = if haskey(ds, "intake_height")
            _raw_numeric(ds["intake_height"], label)
        elseif haskey(ds, "altitude")
            _raw_numeric(ds["altitude"], label) .- elevation
        else
            fill(NaN, n)
        end
        ids = haskey(ds, "obspack_id") ? _string_vector(ds["obspack_id"], label) :
              [string(dataset, "~", k) for k in 1:n]
        _check_same_length(label, n, ("latitude", lats), ("longitude", lons),
                           ("elevation", elevation), ("intake_height", intake), ("obspack_id", ids))
        site_code = String(get(ds.attrib, "site_code", dataset))
        _ObsPackRecords(; dataset, site_code, times, lats, lons, elevation, intake, ids,
                        site_lat = _attr_float(ds, "site_latitude"),
                        site_lon = _attr_float(ds, "site_longitude"),
                        site_elevation = _attr_float(ds, "site_elevation"))
    end
end

# Every record is a point event at its own time and intake height (flasks,
# continuous records, and aircraft alike), as the OCO-2 v11 MIP requires.
function _read_requests_file!(soundings::Vector{SoundingRequest}, ::Vector{SiteRequest},
                              ::ObsPackSource{SoundingMode}, path::AbstractString,
                              source_index::Int, origin::DateTime, stats::ReadStats, window)
    rec = _obspack_records(path)
    origin_unix = datetime2unix(origin)
    for k in eachindex(rec.times)
        if isfinite(rec.times[k]) && !_in_window(window, rec.times[k] - origin_unix)
            stats.outside_window += 1
        elseif isvalid_lonlat(rec.lons[k], rec.lats[k]) && isfinite(rec.times[k])
            push!(soundings, SoundingRequest(rec.ids[k], rec.times[k] - origin_unix,
                                             rec.lons[k], rec.lats[k], source_index,
                                             rec.elevation[k], rec.intake[k]))
        else
            stats.skipped_invalid += 1
        end
    end
    return nothing
end

function _read_requests_file!(::Vector{SoundingRequest}, sites::Vector{SiteRequest},
                              source::ObsPackSource{SiteMode}, path::AbstractString,
                              source_index::Int, ::DateTime, stats::ReadStats, window)
    rec = _obspack_records(path)
    stats.skipped_invalid += count(k -> !isvalid_lonlat(rec.lons[k], rec.lats[k]), eachindex(rec.times))
    append!(sites, _obspack_sites(source.site_grouping, rec, source_index))
    return nothing
end

_nanmedian(v::AbstractVector{Float64}) = (x = filter(isfinite, v); isempty(x) ? NaN : median!(x))
# `+ 0.0` folds -0.0 into 0.0 so rounding cannot split one site in two.
_round_intake(h) = isfinite(h) ? round(h; digits = 1) + 0.0 : NaN
_height_label(h) = isfinite(h) ? @sprintf("%gmagl", h) : "surface"

# One site per distinct intake height in the dataset, named after the dataset
# file (which already encodes site, platform, and lab), suffixed with the
# intake height when the file holds more than one.
function _obspack_sites(::SiteCodeGrouping, rec::_ObsPackRecords, source_index::Int)
    valid = [k for k in eachindex(rec.times) if isvalid_lonlat(rec.lons[k], rec.lats[k])]
    isempty(valid) && return SiteRequest[]
    heights = unique!(sort!([_round_intake(rec.intake[k]) for k in valid]))   # NaN sorts last
    lat = isfinite(rec.site_lat) ? rec.site_lat : _nanmedian(rec.lats[valid])
    lon = isfinite(rec.site_lon) ? rec.site_lon : _nanmedian(rec.lons[valid])
    elev = isfinite(rec.site_elevation) ? rec.site_elevation : _nanmedian(rec.elevation[valid])
    return [SiteRequest(length(heights) == 1 ? rec.dataset : string(rec.dataset, "_", _height_label(h)),
                        lon, lat, elev, h, source_index) for h in heights]
end

# One site per distinct rounded (lat, lon, intake) triple across the records;
# ids read `<site_code>_<lat>N_<lon>E_<height>` with signed decimal degrees.
function _obspack_sites(::LocationGrouping, rec::_ObsPackRecords, source_index::Int)
    seen = Dict{Tuple{Float64, Float64, Float64}, SiteRequest}()
    for k in eachindex(rec.times)
        isvalid_lonlat(rec.lons[k], rec.lats[k]) || continue
        key = (round(rec.lats[k]; digits = 2) + 0.0, round(rec.lons[k]; digits = 2) + 0.0,
               _round_intake(rec.intake[k]))
        haskey(seen, key) && continue
        id = @sprintf("%s_%.2fN_%.2fE_%s", rec.site_code, key[1], key[2], _height_label(key[3]))
        elevation = isfinite(rec.site_elevation) ? rec.site_elevation : rec.elevation[k]
        seen[key] = SiteRequest(id, key[2], key[1], elevation, key[3], source_index)
    end
    return sort!(collect(values(seen)); by = s -> s.id)
end

# -- generic tables ---------------------------------------------------------

const _TABLE_ID_KEYS = ("id", "sounding_id", "site_id", "site")
const _TABLE_TIME_KEYS = ("time", "datetime")
const _TABLE_LAT_KEYS = ("lat", "latitude")
const _TABLE_LON_KEYS = ("lon", "longitude")
const _TABLE_INTAKE_KEYS = ("intake_height", "intake_height_m", "altitude_agl", "height_agl")
const _TABLE_ALTITUDE_KEYS = ("altitude", "altitude_asl")
const _TABLE_ELEVATION_KEYS = ("elevation", "elevation_m")
const _TABLE_START_KEYS = ("start_time",)
const _TABLE_END_KEYS = ("end_time", "stop_time")
const _TABLE_TIMES_KEYS = ("times",)
const _TABLE_DECODED_TIME_KEYS = (_TABLE_TIME_KEYS..., _TABLE_START_KEYS..., _TABLE_END_KEYS...)
# Every column a table may carry; anything else is reported, because a
# misspelt optional column (e.g. `intake_hieght`) would otherwise fall back
# silently to its default.
const _TABLE_KNOWN_KEYS = (_TABLE_ID_KEYS..., _TABLE_TIME_KEYS..., _TABLE_LAT_KEYS..., _TABLE_LON_KEYS...,
                           _TABLE_INTAKE_KEYS..., _TABLE_ALTITUDE_KEYS..., _TABLE_ELEVATION_KEYS...,
                           _TABLE_START_KEYS..., _TABLE_END_KEYS..., _TABLE_TIMES_KEYS...)

# A table is a Dict from lower-case column name to a column vector. CSV
# columns are Vector{String}, NetCDF columns are concrete numeric or string
# vectors (times already as unix seconds), TOML columns are Vector{Any}.
const _TableColumns = Dict{String, AbstractVector}

function _column(columns::_TableColumns, names, label::AbstractString; required::Bool = true)
    for name in names
        haskey(columns, name) && return columns[name]
    end
    required && throw(ArgumentError("$(label) is missing column $(join(repr.(names), " / "))"))
    return nothing
end

function _table_float(v, label::AbstractString, column::AbstractString)
    v isa Real && !(v isa Bool) && return Float64(v)
    if v isa AbstractString
        parsed = tryparse(Float64, strip(v))
        parsed === nothing || return parsed
    end
    throw(ArgumentError("$(label): column $(column) has non-numeric entry $(repr(v))"))
end

_table_optional_float(::Nothing, ::AbstractString, ::AbstractString, ::Int) = NaN
function _table_optional_float(column_values, label::AbstractString, column::AbstractString, k::Int)
    x = column_values[k]
    (x === nothing || x === missing || (x isa AbstractString && isempty(strip(x)))) && return NaN
    x isa Real && isnan(x) && return NaN
    return _table_float(x, label, column)
end

# A NetCDF time cell already decoded from its CF units (unix seconds; NaN for
# fill values). Wrapping it keeps a decoded time distinct from a bare number,
# which is rejected because its units would be a guess.
struct _UnixTime
    seconds::Float64
end

# Time cells: decoded NetCDF times, TOML date-times, or ISO-8601 UTC strings.
function _table_utc_seconds(v::_UnixTime, origin_unix::Float64, label::AbstractString, k::Int)
    isfinite(v.seconds) || throw(ArgumentError("$(label): row $(k) has a fill or non-finite time"))
    return v.seconds - origin_unix
end
_table_utc_seconds(v::Real, ::Float64, label::AbstractString, k::Int) = throw(ArgumentError(
    "$(label): row $(k) has a bare number $(repr(v)) as a time; use an ISO-8601 UTC string, " *
    "a TOML date-time, or a NetCDF variable with CF time units"))
_table_utc_seconds(v, origin_unix::Float64, label::AbstractString, k::Int) =
    datetime2unix(_parse_iso_utc(v, "$(label) row $(k) time")) - origin_unix

# Elevation (m asl) and intake height (m above ground); `altitude` (m asl)
# is converted with the elevation when no intake height is given.
function _table_heights(columns::_TableColumns, label::AbstractString, k::Int)
    elevation = _table_optional_float(_column(columns, _TABLE_ELEVATION_KEYS, label; required = false),
                                      label, "elevation", k)
    intake = _table_optional_float(_column(columns, _TABLE_INTAKE_KEYS, label; required = false),
                                   label, "intake_height", k)
    isnan(intake) || return elevation, intake
    altitude = _table_optional_float(_column(columns, _TABLE_ALTITUDE_KEYS, label; required = false),
                                     label, "altitude", k)
    isnan(altitude) && return elevation, intake
    isnan(elevation) && throw(ArgumentError(
        "$(label): row $(k) gives an altitude above sea level but no elevation to convert it"))
    return elevation, altitude - elevation
end

function _soundings_from_columns(columns::_TableColumns, source_index::Int, origin::DateTime,
                                 label::AbstractString)
    ids = _column(columns, _TABLE_ID_KEYS, label)
    times = _column(columns, _TABLE_TIME_KEYS, label)
    lats = _column(columns, _TABLE_LAT_KEYS, label)
    lons = _column(columns, _TABLE_LON_KEYS, label)
    n = length(ids)
    _check_same_length(label, n, ("time", times), ("lat", lats), ("lon", lons))
    origin_unix = datetime2unix(origin)
    out = SoundingRequest[]
    for k in 1:n
        lon = _table_float(lons[k], label, "lon")
        lat = _table_float(lats[k], label, "lat")
        isvalid_lonlat(lon, lat) || throw(ArgumentError(
            "$(label): row $(k) has an invalid location ($(lon), $(lat))"))
        elevation, intake = _table_heights(columns, label, k)
        push!(out, SoundingRequest(string(ids[k]), _table_utc_seconds(times[k], origin_unix, label, k),
                                   lon, lat, source_index, elevation, intake))
    end
    return out
end

# A table cell, with empty strings, `missing`, and NetCDF fill times read as
# `nothing` so the schedule methods below can dispatch on presence.
_table_cell(::Nothing, k::Int) = nothing
_table_cell(column, k::Int) = _present(column[k])
_present(x) = x
_present(::Missing) = nothing
_present(x::AbstractString) = isempty(strip(x)) ? nothing : x
_present(x::_UnixTime) = isnan(x.seconds) ? nothing : x

# `times` holds a TOML array or a `;`-separated CSV string of UTC times.
_table_time_list(x::AbstractString) = [String(strip(t)) for t in split(x, ';') if !isempty(strip(t))]
_table_time_list(x::AbstractVector) = collect(x)
_table_time_list(x) = [x]

function _site_schedule(columns::_TableColumns, origin_unix::Float64, label::AbstractString, k::Int)
    start = _table_cell(_column(columns, _TABLE_START_KEYS, label; required = false), k)
    stop = _table_cell(_column(columns, _TABLE_END_KEYS, label; required = false), k)
    times = _table_cell(_column(columns, _TABLE_TIMES_KEYS, label; required = false), k)
    return _make_schedule(start, stop, times, origin_unix, label, k)
end

# (start_time, end_time, times) cells -> schedule; `nothing` marks an empty cell.
_make_schedule(::Nothing, ::Nothing, ::Nothing, origin_unix, label, k) = EveryWindow()
_make_schedule(start, stop, ::Nothing, origin_unix, label, k) =
    TimeRange(_table_utc_seconds(start, origin_unix, label, k),
              _table_utc_seconds(stop, origin_unix, label, k))
_make_schedule(::Nothing, ::Nothing, times, origin_unix, label, k) =
    TimeList([_table_utc_seconds(t, origin_unix, label, k) for t in _table_time_list(times)])
_make_schedule(::Nothing, stop, ::Nothing, origin_unix, label, k) = _half_range(label, k)
_make_schedule(start, ::Nothing, ::Nothing, origin_unix, label, k) = _half_range(label, k)
_make_schedule(start, stop, times, origin_unix, label, k) = throw(ArgumentError(
    "$(label): row $(k) sets both a time range and a time list; choose one"))
_half_range(label, k) = throw(ArgumentError("$(label): row $(k) needs both start_time and end_time"))

function _sites_from_columns(columns::_TableColumns, source_index::Int, origin::DateTime,
                             label::AbstractString)
    ids = _column(columns, _TABLE_ID_KEYS, label)
    lats = _column(columns, _TABLE_LAT_KEYS, label)
    lons = _column(columns, _TABLE_LON_KEYS, label)
    n = length(ids)
    _check_same_length(label, n, ("lat", lats), ("lon", lons))
    origin_unix = datetime2unix(origin)
    out = SiteRequest[]
    for k in 1:n
        lon = _table_float(lons[k], label, "lon")
        lat = _table_float(lats[k], label, "lat")
        isvalid_lonlat(lon, lat) || throw(ArgumentError(
            "$(label): row $(k) has an invalid location ($(lon), $(lat))"))
        elevation, intake = _table_heights(columns, label, k)
        push!(out, SiteRequest(string(ids[k]), lon, lat, elevation, intake, source_index,
                               _site_schedule(columns, origin_unix, label, k)))
    end
    return out
end

# CSV: a header line, comma separated, `#` comments and blank lines ignored,
# no quoting support (a quoted comma is reported as a field-count mismatch).
function _read_csv_columns(path::AbstractString, label::AbstractString)
    lines = [String(strip(l)) for l in readlines(path)]
    isempty(lines) || (lines[1] = lstrip(lines[1], '﻿'))
    filter!(l -> !isempty(l) && !startswith(l, '#'), lines)
    isempty(lines) && throw(ArgumentError("$(label) has no header line"))
    header = [lowercase(strip(h)) for h in split(lines[1], ',')]
    allunique(header) || throw(ArgumentError("$(label) has repeated header names"))
    data = [String[] for _ in header]
    for (row, line) in enumerate(lines[2:end])
        fields = [String(strip(f)) for f in split(line, ',')]
        length(fields) == length(header) || throw(ArgumentError(
            "$(label): row $(row) has $(length(fields)) fields; header has $(length(header)) " *
            "(quoted fields are not supported)"))
        for (col, f) in zip(data, fields)
            push!(col, f)
        end
    end
    return _TableColumns(h => col for (h, col) in zip(header, data))
end

function _lower_keys(row::AbstractDict, label::AbstractString, k::Int)
    out = Dict{String, Any}()
    for (key, value) in pairs(row)
        name = lowercase(String(key))
        haskey(out, name) && throw(ArgumentError(
            "$(label): row $(k) has keys that differ only by case ($(name))"))
        out[name] = value
    end
    return out
end

# TOML: `[[soundings]]` / `[[sites]]` tables. Required keys must appear in
# every row; optional ones (intake height, elevation) may be absent.
function _read_toml_columns(path::AbstractString, mode::AbstractObservationMode, label::AbstractString)
    doc = TOML.parsefile(path)
    key = String(mode_label(mode))
    haskey(doc, key) || throw(ArgumentError("$(label) has no [[$(key)]] tables"))
    rows = doc[key]
    rows isa AbstractVector && all(r -> r isa AbstractDict, rows) ||
        throw(ArgumentError("$(label): $(key) must be an array of tables"))
    lowered = [_lower_keys(row, label, k) for (k, row) in enumerate(rows)]
    names = unique!(sort!(vcat([collect(keys(r)) for r in lowered]...)))
    columns = _TableColumns()
    for name in names
        columns[name] = Any[get(r, name, nothing) for r in lowered]
    end
    return columns
end

# NetCDF: every 1-D variable (plus 2-D NC_CHAR string arrays) along the row
# dimension. Time columns (`time`, `start_time`, `end_time`, ...) are decoded
# from their CF units into `_UnixTime`; integer ids stay integers so their
# decimal strings are exact. A per-site `times` list cannot be a 1-D NetCDF
# column; use CSV or TOML for time lists.
function _read_netcdf_columns(path::AbstractString, label::AbstractString)
    return NCDataset(path, "r") do ds
        columns = _TableColumns()
        for name in keys(ds)
            var = ds[name]
            lname = lowercase(String(name))
            if _is_char_matrix(var) || (ndims(var) == 1 && eltype(var.var) <: AbstractString)
                columns[lname] = _string_vector(var, label)
            elseif ndims(var) != 1
                continue
            elseif lname in _TABLE_DECODED_TIME_KEYS
                columns[lname] = _UnixTime.(_time_unix_seconds(var, label))
            elseif eltype(var.var) <: Integer
                columns[lname] = vec(Array(var.var))
            elseif eltype(var.var) <: Real
                columns[lname] = _raw_numeric(var, label)
            end
        end
        columns
    end
end

function _table_format(::AutoTableFormat, path::AbstractString, label::AbstractString)
    ext = lowercase(last(splitext(path)))
    ext == ".csv" && return CSVTableFormat()
    ext == ".toml" && return TOMLTableFormat()
    ext in (".nc", ".nc4", ".netcdf") && return NetCDFTableFormat()
    throw(ArgumentError("$(label): cannot infer the table format from extension $(repr(ext)); set format"))
end
_table_format(format::AbstractTableFormat, ::AbstractString, ::AbstractString) = format

_read_table_columns(::CSVTableFormat, path::AbstractString, ::AbstractObservationMode, label::AbstractString) =
    _read_csv_columns(path, label)
_read_table_columns(::TOMLTableFormat, path::AbstractString, mode::AbstractObservationMode, label::AbstractString) =
    _read_toml_columns(path, mode, label)
_read_table_columns(::NetCDFTableFormat, path::AbstractString, ::AbstractObservationMode, label::AbstractString) =
    _read_netcdf_columns(path, label)

# Unknown columns are an error for CSV and TOML (user-written tables); NetCDF
# files often carry extra variables, so those are only listed at debug level.
_check_table_columns(::Union{CSVTableFormat, TOMLTableFormat}, columns, label) =
    _check_known_keys(columns, _TABLE_KNOWN_KEYS, "$(label) column")
function _check_table_columns(::NetCDFTableFormat, columns, label)
    extra = sort!([k for k in keys(columns) if !(k in _TABLE_KNOWN_KEYS)])
    isempty(extra) || @debug "$(label): ignoring variables $(join(extra, ", "))"
    return nothing
end

function _table_columns(source::TableSource, path::AbstractString, label::AbstractString)
    format = _table_format(source.format, path, label)
    columns = _read_table_columns(format, path, source.mode, label)
    _check_table_columns(format, columns, label)
    return columns
end

function _read_requests_file!(soundings::Vector{SoundingRequest}, ::Vector{SiteRequest},
                              source::TableSource{SoundingMode}, path::AbstractString,
                              source_index::Int, origin::DateTime, ::ReadStats, window)
    label = "table $(path)"
    columns = _table_columns(source, path, label)
    for key in (_TABLE_START_KEYS..., _TABLE_END_KEYS..., _TABLE_TIMES_KEYS...)
        haskey(columns, key) && throw(ArgumentError(
            "$(label): column $(key) is a site schedule; it has no meaning with mode = \"soundings\""))
    end
    append!(soundings, _soundings_from_columns(columns, source_index, origin, label))
    return nothing
end

function _read_requests_file!(::Vector{SoundingRequest}, sites::Vector{SiteRequest},
                              source::TableSource{SiteMode}, path::AbstractString,
                              source_index::Int, origin::DateTime, ::ReadStats, window)
    label = "table $(path)"
    columns = _table_columns(source, path, label)
    append!(sites, _sites_from_columns(columns, source_index, origin, label))
    return nothing
end

# -- assembly ---------------------------------------------------------------

# Run window in seconds since `origin`: the run days as a half-open interval.
function _run_window_seconds(origin::DateTime, dates::AbstractVector{Date})
    isempty(dates) && throw(ArgumentError("observation sampling needs at least one run day"))
    origin_unix = datetime2unix(origin)
    t0 = datetime2unix(DateTime(minimum(dates))) - origin_unix
    t1 = datetime2unix(DateTime(maximum(dates)) + Day(1)) - origin_unix
    return t0, t1
end

# Calendar days whose files may hold requests inside the seconds window.
function _window_days(origin::DateTime, (t0, t1)::Tuple{Real, Real})
    isfinite(t0) && isfinite(t1) && t1 > t0 || throw(ArgumentError(
        "observation window must be a finite, non-empty interval; got ($(t0), $(t1))"))
    first_day = Date(origin + Millisecond(floor(Int, 1000 * t0)))
    last_day = Date(origin + Millisecond(ceil(Int, 1000 * t1)) - Millisecond(1))
    return collect(first_day:Day(1):last_day)
end

"""
    build_observation_set(sources, origin, window_seconds) -> ObservationSet
    build_observation_set(sources, origin, dates)

Read every source for the days covering the window, merge sites by `id`,
expand site time lists into point events, keep the point events with
`window_seconds[1] <= t <= window_seconds[2]` (seconds after `origin`; the
`dates` form uses whole days), and sort them by time (stable). A site `id`
that repeats (across rows, daily files, or sources) must keep its location;
time lists of a repeated id are merged, and other schedules must be equal.
Per-source counts are logged.
"""
build_observation_set(sources::AbstractVector{<:AbstractObservationSource}, origin::DateTime,
                      dates::AbstractVector{Date}) =
    build_observation_set(sources, origin, _run_window_seconds(origin, dates))

function build_observation_set(sources::AbstractVector{<:AbstractObservationSource},
                               origin::DateTime, window_seconds::Tuple{Real, Real})
    t0, t1 = Float64.(window_seconds)
    dates = _window_days(origin, (t0, t1))
    events = SoundingRequest[]
    merged = Dict{String, SiteRequest}()
    order = String[]
    stats = [ReadStats() for _ in sources]
    for (index, source) in enumerate(sources)
        source_events, stations = read_observation_requests(source, index, origin, dates;
                                                           stats = stats[index], window = (t0, t1))
        append!(events, source_events)
        for site in stations
            previous = get(merged, site.id, nothing)
            previous === nothing && push!(order, site.id)
            merged[site.id] = previous === nothing ? site : _merge_site(previous, site)
        end
    end
    series = SiteRequest[]
    for id in order
        _expand_site!(events, series, merged[id])
    end
    nsource = length(sources)
    kept, outside = zeros(Int, nsource), [s.outside_window for s in stats]
    soundings = SoundingRequest[]
    for request in events
        if t0 <= request.time_seconds <= t1
            push!(soundings, request)
            kept[request.source] += 1
        else
            outside[request.source] += 1
        end
    end
    for (index, source) in enumerate(sources)
        nseries = count(site -> site.source == index, series)
        @info "observation source $(index) ($(source_kind(source)), $(mode_label(source_mode(source)))): " *
              "$(kept[index]) point events in the run window, $(outside[index]) outside, " *
              "$(nseries) station series, $(stats[index].skipped_quality) rows failed the quality filter, " *
              "$(stats[index].skipped_invalid) rows invalid"
    end
    sort!(soundings; by = r -> r.time_seconds, alg = MergeSort)
    skipped = sum(s -> s.skipped_invalid + s.skipped_quality, stats; init = 0)
    return ObservationSet(origin, soundings, series, sum(outside; init = 0), skipped)
end

# A repeated site id: same place, merged schedule. The first occurrence keeps
# its source index and metadata.
function _merge_site(a::SiteRequest, b::SiteRequest)
    _same_location(a, b) || throw(ArgumentError(
        "observation site id $(repr(a.id)) appears with different locations " *
        "($(a.lon), $(a.lat), $(a.intake_height_m)) vs ($(b.lon), $(b.lat), $(b.intake_height_m))"))
    schedule = _merge_schedule(a.schedule, b.schedule, a.id)
    return SiteRequest(a.id, a.lon, a.lat, a.elevation_m, a.intake_height_m, a.source, schedule)
end

_same_location(a::SiteRequest, b::SiteRequest) =
    isapprox(a.lon, b.lon; atol = 1e-6) && isapprox(a.lat, b.lat; atol = 1e-6) &&
    (isequal(a.intake_height_m, b.intake_height_m) ||
     isapprox(a.intake_height_m, b.intake_height_m; atol = 0.05))

# Time lists of one site (e.g. a daily flask table) are combined; any other
# pair of schedules must agree, since a site series has one schedule.
_merge_schedule(a::TimeList, b::TimeList, id) = TimeList(unique!(vcat(a.times_seconds, b.times_seconds)))
_merge_schedule(a::AbstractSiteSchedule, b::AbstractSiteSchedule, id) =
    a == b ? a : throw(ArgumentError(
        "observation site id $(repr(id)) appears with different schedules " *
        "($(schedule_label(a)) vs $(schedule_label(b))); give each schedule its own id"))
