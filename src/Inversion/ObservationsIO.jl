# ---------------------------------------------------------------------------
# On-disk observation IO for the CS 4D-Var path.
#
# Loose-compat NetCDF schema documented in `schemas/cs_observations_v1.toml`.
# Single file. Dim `obs` (unlimited) plus `date_component` (length 6).
# Vars: id (i64), date_components (i16[6, obs]), lat / lon / alt (f32),
# value / value_sigma (f64), instrument_type / tracer (string). Root
# attrs: `cs_observations_schema = "v1"`, `time_origin` (ISO-8601 string).
#
# In-memory layer:
#   * `CSObservationRecord`     — one observation row.
#   * `CSObservationSet`        — vector of records plus the file-level
#                                 `time_origin` string.
#   * `read_observations(path)` — open + parse + validate -> CSObservationSet.
#   * `write_observations(path, set)` — emit a v1-compliant NetCDF file.
#
# Bridge to the 4D-Var path (`bind_to_mesh`, `CSObservation` mapping) lives
# in `ObservationBinding.jl`, also included into the `Adjoints` module.
# This file is the pure IO layer.
# ---------------------------------------------------------------------------

const _CS_OBSERVATIONS_SCHEMA_VERSION = "v1"

# Fixed-size date tuple matches the on-disk `(date_component, obs)`
# layout. Each component is Int16 (year fits comfortably).
const _CSObservationDate = NTuple{6, Int16}

"""
    CSObservationRecord(id, date_components, lat, lon, alt,
                        value, value_sigma, instrument_type, tracer)

One point observation row in the [`CSObservationSet`](@ref) on-disk
schema. `date_components` is `(year, month, day, hour, minute, second)`
as Int16 to match the NetCDF layout; `(lat, lon, alt)` are geophysical
coordinates in degrees / metres; `value` + `value_sigma` are stored
in the tracer's native unit (mole fraction, column density, etc.);
`instrument_type` is the instrument-family tag (e.g. `"TCCON"`,
`"ICOS"`, `"OCO-2"`); `tracer` names the species (e.g. `"CO2"`,
`"CH4"`).
"""
struct CSObservationRecord
    id::Int64
    date_components::_CSObservationDate
    lat::Float32
    lon::Float32
    alt::Float32
    value::Float64
    value_sigma::Float64
    instrument_type::String
    tracer::String

    # Inner constructor so positional invocations are validated. The
    # keyword constructor below is a type-coercion wrapper that
    # forwards here.
    function CSObservationRecord(id::Int64,
                                  date_components::_CSObservationDate,
                                  lat::Float32, lon::Float32, alt::Float32,
                                  value::Float64, value_sigma::Float64,
                                  instrument_type::String,
                                  tracer::String)
        isfinite(lat) || throw(ArgumentError(
            "CSObservationRecord lat must be finite, got $lat"))
        -90 <= lat <= 90 || throw(ArgumentError(
            "CSObservationRecord lat must be in [-90, 90], got $lat"))
        isfinite(lon) || throw(ArgumentError(
            "CSObservationRecord lon must be finite, got $lon"))
        isfinite(value) || throw(ArgumentError(
            "CSObservationRecord value must be finite, got $value"))
        isfinite(value_sigma) || throw(ArgumentError(
            "CSObservationRecord value_sigma must be finite, got $value_sigma"))
        value_sigma > 0 || throw(ArgumentError(
            "CSObservationRecord value_sigma must be positive, got $value_sigma"))
        return new(id, date_components, lat, lon, alt,
                   value, value_sigma, instrument_type, tracer)
    end
end

function CSObservationRecord(; id::Integer,
                              date_components,
                              lat::Real, lon::Real, alt::Real,
                              value::Real, value_sigma::Real,
                              instrument_type::AbstractString,
                              tracer::AbstractString)
    dc = _coerce_date_components(date_components)
    return CSObservationRecord(Int64(id), dc,
                               Float32(lat), Float32(lon), Float32(alt),
                               Float64(value), Float64(value_sigma),
                               String(instrument_type), String(tracer))
end

function _coerce_date_components(dc)
    length(dc) == 6 || throw(ArgumentError(
        "date_components must have 6 entries (year, month, day, hour, " *
        "minute, second); got $(length(dc))"))
    return ntuple(i -> Int16(dc[i]), 6)
end

"""
    CSObservationSet(records, time_origin)

Collection of [`CSObservationRecord`](@ref) rows that share an absolute
`time_origin` string (an ISO-8601 timestamp, e.g.
`"1900-01-01 00:00:00"`). The on-disk format always carries the origin
even though the per-record dates are stored as absolute components —
the origin tags the convention so a downstream consumer that wants to
convert to seconds-since-origin can do so without parsing the per-
record dates.
"""
struct CSObservationSet
    records::Vector{CSObservationRecord}
    time_origin::String
end

CSObservationSet(records::AbstractVector{CSObservationRecord};
                 time_origin::AbstractString = "1900-01-01 00:00:00") =
    CSObservationSet(collect(records), String(time_origin))

Base.length(set::CSObservationSet) = length(set.records)
Base.isempty(set::CSObservationSet) = isempty(set.records)
Base.iterate(set::CSObservationSet, args...) = iterate(set.records, args...)
Base.getindex(set::CSObservationSet, i::Integer) = set.records[i]

# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

"""
    write_observations(path::AbstractString, set::CSObservationSet) -> String

Write `set` to `path` as a v1-compliant NetCDF observations file.
Returns the expanded path. Overwrites `path` if it already exists.
The file follows `schemas/cs_observations_v1.toml`:

- Dimensions: `obs` (unlimited), `date_component` (= 6).
- Variables (typed exactly per spec): `id`, `date_components`,
  `lat`, `lon`, `alt`, `value`, `value_sigma`, `instrument_type`,
  `tracer`.
- Root attributes: `cs_observations_schema = "v1"`,
  `time_origin = set.time_origin`.
"""
function write_observations(path::AbstractString, set::CSObservationSet)
    isempty(set.time_origin) && throw(ArgumentError(
        "CSObservationSet.time_origin must be a non-empty ISO-8601 string"))
    mkpath(dirname(abspath(path)))
    n = length(set)
    isfile(path) && rm(path)
    NCDatasets.NCDataset(path, "c") do ds
        NCDatasets.defDim(ds, "obs", n == 0 ? Inf : n)
        NCDatasets.defDim(ds, "date_component", 6)

        ds.attrib["cs_observations_schema"] = _CS_OBSERVATIONS_SCHEMA_VERSION
        ds.attrib["time_origin"] = set.time_origin

        ids = Vector{Int64}(undef, n)
        dates = Matrix{Int16}(undef, 6, n)
        lats = Vector{Float32}(undef, n)
        lons = Vector{Float32}(undef, n)
        alts = Vector{Float32}(undef, n)
        vals = Vector{Float64}(undef, n)
        sigs = Vector{Float64}(undef, n)
        instr = Vector{String}(undef, n)
        tracs = Vector{String}(undef, n)
        @inbounds for (i, r) in enumerate(set.records)
            ids[i] = r.id
            for c in 1:6
                dates[c, i] = r.date_components[c]
            end
            lats[i] = r.lat
            lons[i] = r.lon
            alts[i] = r.alt
            vals[i] = r.value
            sigs[i] = r.value_sigma
            instr[i] = r.instrument_type
            tracs[i] = r.tracer
        end

        _write_obs_var(ds, "id", Int64, ("obs",), ids,
                       Dict("description" => "unique observation identifier"))
        _write_obs_var(ds, "date_components", Int16,
                       ("date_component", "obs"), dates,
                       Dict("description" =>
                            "(year, month, day, hour, minute, second) per observation"))
        _write_obs_var(ds, "lat", Float32, ("obs",), lats,
                       Dict("units" => "degrees_north",
                            "description" => "observation latitude"))
        _write_obs_var(ds, "lon", Float32, ("obs",), lons,
                       Dict("units" => "degrees_east",
                            "description" => "observation longitude"))
        _write_obs_var(ds, "alt", Float32, ("obs",), alts,
                       Dict("units" => "m",
                            "description" => "observation altitude above sea level"))
        _write_obs_var(ds, "value", Float64, ("obs",), vals,
                       Dict("description" =>
                            "observation value in the tracer's native unit"))
        _write_obs_var(ds, "value_sigma", Float64, ("obs",), sigs,
                       Dict("description" =>
                            "1-sigma observation error in the tracer's native unit"))
        _write_obs_var(ds, "instrument_type", String, ("obs",), instr,
                       Dict("description" => "instrument family label"))
        _write_obs_var(ds, "tracer", String, ("obs",), tracs,
                       Dict("description" => "tracer name"))
    end
    return path
end

function _write_obs_var(ds, name, T, dims, data, attrs)
    v = NCDatasets.defVar(ds, name, T, dims)
    for (k, val) in attrs
        v.attrib[k] = val
    end
    v[:] = data
    return v
end

# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

"""
    read_observations(path::AbstractString) -> CSObservationSet

Parse a v1-compliant observation NetCDF written by
[`write_observations`](@ref) and return an in-memory
[`CSObservationSet`](@ref).

Validation (fails fast with `ArgumentError`):

- Root attribute `cs_observations_schema` must equal `"v1"`.
- Root attribute `time_origin` must be present.
- Required variables (`id`, `date_components`, `lat`, `lon`, `alt`,
  `value`, `value_sigma`, `instrument_type`, `tracer`) must all be
  present and dimensioned exactly per the schema.
- `date_components` must have dimensions `(date_component=6, obs)`.
- All `obs`-indexed variables must share the same length.
"""
function read_observations(path::AbstractString)
    isfile(path) || throw(ArgumentError(
        "observation NetCDF not found at $(repr(path))"))
    return NCDatasets.NCDataset(path, "r") do ds
        _validate_schema_version(ds, path)
        time_origin = _require_attr(ds, "time_origin", path)

        for v in ("id", "date_components", "lat", "lon", "alt", "value",
                  "value_sigma", "instrument_type", "tracer")
            haskey(ds, v) || throw(ArgumentError(
                "observation NetCDF $(repr(path)) is missing required " *
                "variable $(repr(v))"))
        end

        n = ds.dim["obs"]
        ds.dim["date_component"] == 6 || throw(ArgumentError(
            "observation NetCDF $(repr(path)) has date_component = " *
            "$(ds.dim["date_component"]); expected 6"))

        ids = Array(ds["id"][:])
        dates = Array(ds["date_components"][:, :])
        lats = Array(ds["lat"][:])
        lons = Array(ds["lon"][:])
        alts = Array(ds["alt"][:])
        vals = Array(ds["value"][:])
        sigs = Array(ds["value_sigma"][:])
        instr = Array(ds["instrument_type"][:])
        tracs = Array(ds["tracer"][:])

        size(dates) == (6, n) || throw(ArgumentError(
            "observation NetCDF $(repr(path)) date_components has shape " *
            "$(size(dates)); expected (6, $n)"))
        for (var, len, expected) in (("id", length(ids), n),
                                     ("lat", length(lats), n),
                                     ("lon", length(lons), n),
                                     ("alt", length(alts), n),
                                     ("value", length(vals), n),
                                     ("value_sigma", length(sigs), n),
                                     ("instrument_type", length(instr), n),
                                     ("tracer", length(tracs), n))
            len == expected || throw(ArgumentError(
                "observation NetCDF $(repr(path)) variable $(repr(var)) " *
                "has length $(len); expected $(expected) to match obs dim"))
        end

        records = Vector{CSObservationRecord}(undef, n)
        @inbounds for i in 1:n
            dc = ntuple(c -> Int16(dates[c, i]), 6)
            records[i] = CSObservationRecord(Int64(ids[i]), dc,
                                             Float32(lats[i]),
                                             Float32(lons[i]),
                                             Float32(alts[i]),
                                             Float64(vals[i]),
                                             Float64(sigs[i]),
                                             String(instr[i]),
                                             String(tracs[i]))
        end
        return CSObservationSet(records, String(time_origin))
    end
end

function _validate_schema_version(ds, path)
    haskey(ds.attrib, "cs_observations_schema") || throw(ArgumentError(
        "observation NetCDF $(repr(path)) is missing root attribute " *
        "`cs_observations_schema`; not a CS observations file"))
    got = ds.attrib["cs_observations_schema"]
    got == _CS_OBSERVATIONS_SCHEMA_VERSION || throw(ArgumentError(
        "observation NetCDF $(repr(path)) has cs_observations_schema = " *
        "$(repr(got)); this build supports " *
        "$(repr(_CS_OBSERVATIONS_SCHEMA_VERSION)) only"))
    return nothing
end

function _require_attr(ds, name, path)
    haskey(ds.attrib, name) || throw(ArgumentError(
        "observation NetCDF $(repr(path)) is missing root attribute " *
        "$(repr(name))"))
    return ds.attrib[name]
end
