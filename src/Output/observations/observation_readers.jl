# ---------------------------------------------------------------------------
# Readers that turn `[[output.observations.sources]]` into sampling requests.
#
# `read_observation_requests(source, index, origin, dates)` expands the
# source's path template for the run days and calls `_read_requests_file!`
# once per file, dispatched on the source type and its mode. Times become
# Float64 seconds since `origin` (UTC); rows with fill values or impossible
# coordinates are skipped. `build_observation_set` merges every source, keeps
# only requests inside the run window, and logs what it kept and dropped.
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
# offsets are rejected: every supported product is UTC.
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
    units = var.attrib isa AbstractDict ? get(var.attrib, "units", nothing) : get(var.attrib, "units", nothing)
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

"""
    read_observation_requests(source, source_index, origin, dates) -> (soundings, sites)

Expand the source's path template for the run days and read every file into
`SoundingRequest`s and `SiteRequest`s (one of the two is empty, depending on
the source mode). Times are Float64 seconds since `origin`. Rows with fill
values or impossible coordinates are skipped; a source that resolves to no
file at all logs a warning.
"""
function read_observation_requests(source::AbstractObservationSource, source_index::Int,
                                   origin::DateTime, dates::AbstractVector{Date})
    soundings = SoundingRequest[]
    sites = SiteRequest[]
    paths = expand_observation_paths(source_path_template(source), dates)
    isempty(paths) && @warn "observation source $(source_index) ($(source_kind(source))) matched no files for the run days" template = source_path_template(source)
    for path in paths
        _read_requests_file!(soundings, sites, source, path, source_index, origin)
    end
    return soundings, sites
end

# -- OCO-2 Lite -------------------------------------------------------------

function _read_requests_file!(soundings::Vector{SoundingRequest}, ::Vector{SiteRequest},
                              source::OCO2LiteSource, path::AbstractString, source_index::Int,
                              origin::DateTime)
    label = "oco2_lite $(basename(path))"
    origin_unix = datetime2unix(origin)
    NCDataset(path, "r") do ds
        ids = vec(Array(_nc_variable(ds, ("sounding_id",), label).var))
        n = length(ids)
        times = _time_unix_seconds(_nc_variable(ds, ("time",), label), label)
        lats = _raw_numeric(_nc_variable(ds, ("latitude",), label), label)
        lons = _raw_numeric(_nc_variable(ds, ("longitude",), label), label)
        flags = _raw_numeric(_nc_variable(ds, ("xco2_quality_flag",), label), label)
        _check_same_length(label, n, ("time", times), ("latitude", lats),
                           ("longitude", lons), ("xco2_quality_flag", flags))
        for k in 1:n
            isfinite(flags[k]) && flags[k] <= source.quality_flag_max || continue
            isvalid_lonlat(lons[k], lats[k]) && isfinite(times[k]) && !iszero(ids[k]) || continue
            push!(soundings, SoundingRequest(string(ids[k]), times[k] - origin_unix,
                                             lons[k], lats[k], source_index))
        end
    end
    return nothing
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
    label = "obspack $(basename(path))"
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

function _read_requests_file!(soundings::Vector{SoundingRequest}, ::Vector{SiteRequest},
                              ::ObsPackSource{SoundingMode}, path::AbstractString,
                              source_index::Int, origin::DateTime)
    rec = _obspack_records(path)
    origin_unix = datetime2unix(origin)
    for k in eachindex(rec.times)
        isvalid_lonlat(rec.lons[k], rec.lats[k]) && isfinite(rec.times[k]) || continue
        push!(soundings, SoundingRequest(rec.ids[k], rec.times[k] - origin_unix,
                                         rec.lons[k], rec.lats[k], source_index))
    end
    return nothing
end

function _read_requests_file!(::Vector{SoundingRequest}, sites::Vector{SiteRequest},
                              source::ObsPackSource{SiteMode}, path::AbstractString,
                              source_index::Int, ::DateTime)
    append!(sites, _obspack_sites(source.site_grouping, _obspack_records(path), source_index))
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
const _TABLE_ELEVATION_KEYS = ("elevation", "elevation_m")

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

# NetCDF time columns arrive as unix seconds; CSV/TOML values are parsed strictly.
function _table_utc_seconds(v, origin_unix::Float64, label::AbstractString, k::Int)
    if v isa Real && !(v isa Bool)
        isfinite(v) || throw(ArgumentError("$(label): row $(k) has a fill or non-finite time"))
        return Float64(v) - origin_unix
    end
    return datetime2unix(_parse_iso_utc(v, "$(label) row $(k) time")) - origin_unix
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
        push!(out, SoundingRequest(string(ids[k]), _table_utc_seconds(times[k], origin_unix, label, k),
                                   lon, lat, source_index))
    end
    return out
end

function _sites_from_columns(columns::_TableColumns, source_index::Int, label::AbstractString)
    ids = _column(columns, _TABLE_ID_KEYS, label)
    lats = _column(columns, _TABLE_LAT_KEYS, label)
    lons = _column(columns, _TABLE_LON_KEYS, label)
    intake = _column(columns, _TABLE_INTAKE_KEYS, label; required = false)
    elevation = _column(columns, _TABLE_ELEVATION_KEYS, label; required = false)
    n = length(ids)
    _check_same_length(label, n, ("lat", lats), ("lon", lons))
    out = SiteRequest[]
    for k in 1:n
        lon = _table_float(lons[k], label, "lon")
        lat = _table_float(lats[k], label, "lat")
        isvalid_lonlat(lon, lat) || throw(ArgumentError(
            "$(label): row $(k) has an invalid location ($(lon), $(lat))"))
        push!(out, SiteRequest(string(ids[k]), lon, lat,
                               _table_optional_float(elevation, label, "elevation", k),
                               _table_optional_float(intake, label, "intake_height", k), source_index))
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
# dimension. `time`/`datetime` are converted to unix seconds; integer ids
# stay integers so their decimal strings are exact.
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
            elseif lname in _TABLE_TIME_KEYS
                columns[lname] = _time_unix_seconds(var, label)
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

function _read_requests_file!(soundings::Vector{SoundingRequest}, ::Vector{SiteRequest},
                              source::TableSource{SoundingMode}, path::AbstractString,
                              source_index::Int, origin::DateTime)
    label = "table $(basename(path))"
    columns = _read_table_columns(_table_format(source.format, path, label), path, SoundingMode(), label)
    append!(soundings, _soundings_from_columns(columns, source_index, origin, label))
    return nothing
end

function _read_requests_file!(::Vector{SoundingRequest}, sites::Vector{SiteRequest},
                              source::TableSource{SiteMode}, path::AbstractString,
                              source_index::Int, ::DateTime)
    label = "table $(basename(path))"
    columns = _read_table_columns(_table_format(source.format, path, label), path, SiteMode(), label)
    append!(sites, _sites_from_columns(columns, source_index, label))
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

"""
    build_observation_set(sources, origin, dates) -> ObservationSet

Read every source, keep the soundings inside the run window (the run days
as a half-open interval), sort them by time (stable), and merge sites by
`id`. Sites that repeat with the same location are deduplicated; the same
`id` with a different location is an error. Per-source counts are logged.
"""
function build_observation_set(sources::AbstractVector{<:AbstractObservationSource},
                               origin::DateTime, dates::AbstractVector{Date})
    t0, t1 = _run_window_seconds(origin, dates)
    soundings = SoundingRequest[]
    sites = SiteRequest[]
    seen = Dict{String, SiteRequest}()
    dropped_outside = 0
    for (index, source) in enumerate(sources)
        s, p = read_observation_requests(source, index, origin, dates)
        kept = 0
        for request in s
            if t0 <= request.time_seconds < t1
                push!(soundings, request)
                kept += 1
            else
                dropped_outside += 1
            end
        end
        for site in p
            previous = get(seen, site.id, nothing)
            if previous === nothing
                seen[site.id] = site
                push!(sites, site)
            elseif !_same_site(previous, site)
                throw(ArgumentError(
                    "observation site id $(repr(site.id)) appears with different locations " *
                    "($(previous.lon), $(previous.lat), $(previous.intake_height_m)) vs " *
                    "($(site.lon), $(site.lat), $(site.intake_height_m))"))
            end
        end
        @info "observation source $(index) ($(source_kind(source)), $(mode_label(source_mode(source)))): " *
              "$(kept) soundings in the run window, $(length(s) - kept) outside, $(length(p)) sites"
    end
    sort!(soundings; by = r -> r.time_seconds, alg = MergeSort)
    return ObservationSet(origin, soundings, sites, dropped_outside)
end

_same_site(a::SiteRequest, b::SiteRequest) =
    isapprox(a.lon, b.lon; atol = 1e-6) && isapprox(a.lat, b.lat; atol = 1e-6) &&
    (isequal(a.intake_height_m, b.intake_height_m) ||
     isapprox(a.intake_height_m, b.intake_height_m; atol = 0.05))
