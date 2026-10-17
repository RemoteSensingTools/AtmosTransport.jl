# ---------------------------------------------------------------------------
# On-disk observation IO for the CS 4D-Var path.
#
# NetCDF schema v2, documented in `schemas/cs_observations_v2.toml`. Single
# file; dimension `obs` (unlimited) plus `date_component` (length 6). Root
# attributes: `cs_observations_schema = "v2"`, `time_origin` (ISO-8601).
# Coordinates are Float64; the vertical fields (`elevation_m`,
# `intake_height_m`, `altitude_m`, `pressure_pa`) use NaN for "absent".
# Files of schema v1 (Float32 lat/lon/alt) are rejected.
#
# In-memory layer:
#   * `CSObservationRecord`     — one observation row.
#   * `CSObservationSet`        — vector of records plus the file-level
#                                 `time_origin` string.
#   * `read_observations(path)` — open + parse + validate -> CSObservationSet.
#   * `write_observations(path, set)` — emit a v2 NetCDF file.
#
# Bridge to the 4D-Var path (`bind_to_mesh`) lives in `ObservationBinding.jl`,
# also included into the `Adjoints` module. This file is the pure IO layer.
# ---------------------------------------------------------------------------

const _CS_OBSERVATIONS_SCHEMA_VERSION = "v2"

# Fixed-size date tuple matches the on-disk `(date_component, obs)`
# layout. Each component is Int16 (year fits comfortably).
const _CSObservationDate = NTuple{6, Int16}

"""
    AbstractObservationKind

What an observation record measures, the `kind` column of the observation
file: [`PointObservationKind`](@ref) (`"point"`) or
[`ColumnObservationKind`](@ref) (`"column"`). `bind_to_mesh` turns a point
record into a [`CSPlacedPointObjective`](@ref) and a column record into a
[`CSColumnMeanObjective`](@ref).
"""
abstract type AbstractObservationKind end

"""
`kind = "point"`: an in-situ sample at one place in the column (surface
station, tower, flask, aircraft, AirCore). Placed in the model column by its
height above the ground or its pressure (`intake_height_m`, `altitude_m` with
`elevation_m` or a DEM, `pressure_pa`); a point record that gives none of the
vertical coordinates in use cannot be placed (`unplaced_records`).
"""
struct PointObservationKind <: AbstractObservationKind end

"""
`kind = "column"`: a column-averaged dry mole fraction (TCCON, a satellite
sounding). Bound to the air-mass-weighted column mean of its cell
([`CSColumnMeanObjective`](@ref)); averaging kernels are not applied yet.
A column record states no intake height, altitude or pressure.
"""
struct ColumnObservationKind <: AbstractObservationKind end

const _ObservationKind = Union{PointObservationKind, ColumnObservationKind}
const _OBSERVATION_KINDS = (point = PointObservationKind(), column = ColumnObservationKind())
observation_kind_label(::PointObservationKind) = "point"
observation_kind_label(::ColumnObservationKind) = "column"

_parse_observation_kind(kind::_ObservationKind) = kind
function _parse_observation_kind(kind::AbstractString)
    key = Symbol(kind)
    haskey(_OBSERVATION_KINDS, key) || throw(ArgumentError(
        "observation kind must be \"point\" or \"column\"; got $(repr(kind))"))
    return _OBSERVATION_KINDS[key]
end
_parse_observation_kind(kind) = throw(ArgumentError(
    "observation kind must be PointObservationKind(), ColumnObservationKind(), \"point\" or " *
    "\"column\"; got $(repr(kind))"))

# NaN means "absent"; any other value must be finite.
_absent_or_finite(x::Float64) = isnan(x) || isfinite(x)

"""
    CSObservationRecord(id, date_components, lat, lon, kind,
                        elevation_m, intake_height_m, altitude_m, pressure_pa,
                        time_window_s, flag, value, value_sigma, model_sigma,
                        instrument_type, tracer)
    CSObservationRecord(; id, date_components, lat, lon, value, value_sigma,
                          instrument_type, tracer, kind = PointObservationKind(),
                          elevation_m = NaN, intake_height_m = NaN, altitude_m = NaN,
                          pressure_pa = NaN, time_window_s = 0, flag = 0, model_sigma = 0)

One observation row of the schema-v2 [`CSObservationSet`](@ref).

- `date_components`: `(year, month, day, hour, minute, second)` UTC, Int16, the
  centre of the sampling window.
- `lat`, `lon`: degrees, Float64 (a point within Float32 rounding of a cell
  edge must bind to the cell that contains it).
- `kind`: [`PointObservationKind`](@ref) or [`ColumnObservationKind`](@ref)
  (the keyword form also takes `"point"` / `"column"`).
- Vertical fields, NaN when the record does not give them: `elevation_m`
  (ground, m above sea level), `intake_height_m` (m above the ground),
  `altitude_m` (m above sea level), `pressure_pa` (total pressure, Pa,
  positive). Column records give none of `intake_height_m`, `altitude_m`,
  `pressure_pa`.
- `time_window_s`: length (s, ≥ 0) of the sampling window centred on the
  record time; 0 samples the model state of one step.
- `flag`: quality flag, `0` = assimilate; `bind_to_mesh` drops the others.
- `value`, `value_sigma`: observation and its 1-σ error, in the tracer's
  native unit (dry mole fraction for trace gases); `model_sigma` (≥ 0) is a
  model/representation error added in quadrature (σ² = value_sigma² +
  model_sigma²).
- `instrument_type` (e.g. `"ICOS"`, `"TCCON"`), `tracer` (e.g. `"CO2"`).
"""
struct CSObservationRecord
    id::Int64
    date_components::_CSObservationDate
    lat::Float64
    lon::Float64
    kind::_ObservationKind
    elevation_m::Float64
    intake_height_m::Float64
    altitude_m::Float64
    pressure_pa::Float64
    time_window_s::Float64
    flag::Int8
    value::Float64
    value_sigma::Float64
    model_sigma::Float64
    instrument_type::String
    tracer::String

    # Inner constructor so positional invocations are validated. The
    # keyword constructor below is a type-coercion wrapper that forwards here.
    function CSObservationRecord(id::Int64, date_components::_CSObservationDate,
                                 lat::Float64, lon::Float64, kind::_ObservationKind,
                                 elevation_m::Float64, intake_height_m::Float64,
                                 altitude_m::Float64, pressure_pa::Float64,
                                 time_window_s::Float64, flag::Int8,
                                 value::Float64, value_sigma::Float64, model_sigma::Float64,
                                 instrument_type::String, tracer::String)
        label = "CSObservationRecord id $(id)"
        isfinite(lat) && -90 <= lat <= 90 || throw(ArgumentError(
            "$(label): lat must be finite and in [-90, 90], got $lat"))
        isfinite(lon) && abs(lon) <= 720 || throw(ArgumentError(
            "$(label): lon must be finite with |lon| <= 720, got $lon"))
        isfinite(value) || throw(ArgumentError("$(label): value must be finite, got $value"))
        isfinite(value_sigma) && value_sigma > 0 || throw(ArgumentError(
            "$(label): value_sigma must be finite and positive, got $value_sigma"))
        isfinite(model_sigma) && model_sigma >= 0 || throw(ArgumentError(
            "$(label): model_sigma must be finite and >= 0, got $model_sigma"))
        isfinite(time_window_s) && time_window_s >= 0 || throw(ArgumentError(
            "$(label): time_window_s must be finite and >= 0, got $time_window_s"))
        for (name, x) in (("elevation_m", elevation_m), ("intake_height_m", intake_height_m),
                          ("altitude_m", altitude_m))
            _absent_or_finite(x) || throw(ArgumentError(
                "$(label): $(name) must be finite or NaN (absent), got $x"))
        end
        isnan(pressure_pa) || (isfinite(pressure_pa) && pressure_pa > 0) || throw(ArgumentError(
            "$(label): pressure_pa must be a finite positive pressure or NaN (absent), got $pressure_pa"))
        if kind isa ColumnObservationKind
            for (name, x) in (("intake_height_m", intake_height_m), ("altitude_m", altitude_m),
                              ("pressure_pa", pressure_pa))
                isnan(x) || throw(ArgumentError(
                    "$(label) is a column record but gives $(name) = $x; a column record is " *
                    "not placed at one height (set it to NaN, or make the record kind \"point\")"))
            end
        end
        return new(id, date_components, lat, lon, kind, elevation_m, intake_height_m, altitude_m,
                   pressure_pa, time_window_s, flag, value, value_sigma, model_sigma,
                   instrument_type, tracer)
    end
end

function CSObservationRecord(; id::Integer, date_components, lat::Real, lon::Real,
                             value::Real, value_sigma::Real,
                             instrument_type::AbstractString, tracer::AbstractString,
                             kind = PointObservationKind(),
                             elevation_m::Real = NaN, intake_height_m::Real = NaN,
                             altitude_m::Real = NaN, pressure_pa::Real = NaN,
                             time_window_s::Real = 0, flag::Integer = 0,
                             model_sigma::Real = 0)
    return CSObservationRecord(Int64(id), _coerce_date_components(date_components),
                               Float64(lat), Float64(lon), _parse_observation_kind(kind),
                               Float64(elevation_m), Float64(intake_height_m),
                               Float64(altitude_m), Float64(pressure_pa),
                               Float64(time_window_s), Int8(flag),
                               Float64(value), Float64(value_sigma), Float64(model_sigma),
                               String(instrument_type), String(tracer))
end

function _coerce_date_components(dc)
    length(dc) == 6 || throw(ArgumentError(
        "date_components must have 6 entries (year, month, day, hour, " *
        "minute, second); got $(length(dc))"))
    return ntuple(i -> Int16(dc[i]), 6)
end

# The 1-σ error of the record's departure, σ = √(value_sigma² + model_sigma²);
# exactly `value_sigma` without a model error.
_total_sigma(r::CSObservationRecord) =
    r.model_sigma == 0 ? r.value_sigma : hypot(r.value_sigma, r.model_sigma)

"""
    CSObservationSet(records, time_origin)

Collection of [`CSObservationRecord`](@ref) rows, with unique ids, that share
an absolute `time_origin` string (an ISO-8601 timestamp, e.g.
`"1900-01-01 00:00:00"`). The on-disk format always carries the origin
even though the per-record dates are stored as absolute components —
the origin tags the convention so a downstream consumer that wants to
convert to seconds-since-origin can do so without parsing the per-
record dates.
"""
struct CSObservationSet
    records::Vector{CSObservationRecord}
    time_origin::String
    function CSObservationSet(records::Vector{CSObservationRecord}, time_origin::String)
        _check_unique_ids(records, "CSObservationSet")
        return new(records, time_origin)
    end
end

# Record ids are unique: departures are joined to records by id.
function _check_unique_ids(records, label::AbstractString)
    seen = Dict{Int64, Int}()
    for (row, r) in enumerate(records)
        first_row = get(seen, r.id, 0)
        first_row == 0 || throw(ArgumentError(
            "$(label): observation id $(r.id) appears in rows $(first_row) and $(row); record ids must be " *
            "unique (departures are joined to records by id)"))
        seen[r.id] = row
    end
    return nothing
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

# (name, on-disk type, record field, attributes) of the per-observation
# variables other than `date_components` and `kind`.
const _OBSERVATION_VARIABLES = (
    ("id", Int64, :id, ("description" => "unique observation identifier",)),
    ("lat", Float64, :lat, ("units" => "degrees_north", "description" => "observation latitude")),
    ("lon", Float64, :lon, ("units" => "degrees_east", "description" => "observation longitude")),
    ("elevation_m", Float64, :elevation_m,
     ("units" => "m", "description" => "ground elevation above sea level; NaN = absent")),
    ("intake_height_m", Float64, :intake_height_m,
     ("units" => "m", "description" => "sampling height above the ground; NaN = absent")),
    ("altitude_m", Float64, :altitude_m,
     ("units" => "m", "description" => "sampling altitude above sea level; NaN = absent")),
    ("pressure_pa", Float64, :pressure_pa,
     ("units" => "Pa", "description" => "total (ambient) pressure at the sample; NaN = absent")),
    ("time_window_s", Float64, :time_window_s,
     ("units" => "s", "description" => "length of the sampling window centred on the record time; 0 = one model step")),
    ("flag", Int8, :flag, ("description" => "quality flag: 0 = assimilate, any other value = do not assimilate",)),
    ("value", Float64, :value, ("description" => "observation value in the tracer's native unit",)),
    ("value_sigma", Float64, :value_sigma,
     ("description" => "1-sigma observation error in the tracer's native unit",)),
    ("model_sigma", Float64, :model_sigma,
     ("description" => "1-sigma model/representation error added in quadrature to value_sigma",)),
    ("instrument_type", String, :instrument_type, ("description" => "instrument family label",)),
    ("tracer", String, :tracer, ("description" => "tracer name",)),
)

const _OBSERVATION_REQUIRED_VARIABLES =
    ("date_components", "kind", (first(v) for v in _OBSERVATION_VARIABLES)...)

"""
    write_observations(path::AbstractString, set::CSObservationSet) -> String

Write `set` to `path` as a schema-v2 NetCDF observations file
(`schemas/cs_observations_v2.toml`) and return `path`; an existing file is
overwritten.

- Dimensions: `obs` (unlimited), `date_component` (= 6).
- Variables: `id` (i64), `date_components` (i16), `lat`, `lon` (f64),
  `kind` (string `"point"` / `"column"`), `elevation_m`, `intake_height_m`,
  `altitude_m`, `pressure_pa` (f64, NaN = absent), `time_window_s` (f64),
  `flag` (i8), `value`, `value_sigma`, `model_sigma` (f64),
  `instrument_type`, `tracer` (string).
- Root attributes: `cs_observations_schema = "v2"`,
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

        dates = Matrix{Int16}(undef, 6, n)
        @inbounds for (i, r) in enumerate(set.records), c in 1:6
            dates[c, i] = r.date_components[c]
        end
        _write_obs_var(ds, "date_components", Int16, ("date_component", "obs"), dates,
                       ("description" => "(year, month, day, hour, minute, second) UTC per observation",))
        _write_obs_var(ds, "kind", String, ("obs",),
                       String[observation_kind_label(r.kind) for r in set.records],
                       ("description" => "\"point\" (in-situ sample in the column) or \"column\" (column average)",))
        for (name, T, field, attrs) in _OBSERVATION_VARIABLES
            _write_obs_var(ds, name, T, ("obs",), T[getfield(r, field) for r in set.records], attrs)
        end
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

Parse a schema-v2 observation NetCDF written by
[`write_observations`](@ref) and return an in-memory
[`CSObservationSet`](@ref).

Validation (fails fast with `ArgumentError`):

- Root attribute `cs_observations_schema` must equal `"v2"`. Schema-v1 files
  are rejected: their Float32 coordinates can bind a point near a cell edge
  to the neighbouring cell, and their single `alt` cannot place a record by
  its height above the ground. Rewrite them as v2 with `write_observations`.
- Root attribute `time_origin` must be present.
- Every v2 variable must be present, `lat` and `lon` stored as Float64, and
  `date_components` dimensioned `(date_component = 6, obs)`.
- `elevation_m`, `intake_height_m`, `altitude_m` and `pressure_pa` read NaN or
  a declared fill value (`_FillValue`, e.g. xarray's NaN) as absent; any other
  variable with a fill value is an error.
- Every row must pass the [`CSObservationRecord`](@ref) checks, and record
  ids must be unique.
"""
function read_observations(path::AbstractString)
    isfile(path) || throw(ArgumentError(
        "observation NetCDF not found at $(repr(path))"))
    return NCDatasets.NCDataset(path, "r") do ds
        _validate_schema_version(ds, path)
        time_origin = _require_attr(ds, "time_origin", path)
        for v in _OBSERVATION_REQUIRED_VARIABLES
            haskey(ds, v) || throw(ArgumentError(
                "observation NetCDF $(repr(path)) is missing required variable $(repr(v))"))
        end
        for v in ("lat", "lon")
            T = eltype(ds[v].var)
            T === Float64 || throw(ArgumentError(
                "observation NetCDF $(repr(path)) stores $(v) as $(T); schema v2 requires Float64 " *
                "coordinates (a Float32 point near a cell edge can bind to the neighbouring cell)"))
        end
        n = ds.dim["obs"]
        ds.dim["date_component"] == 6 || throw(ArgumentError(
            "observation NetCDF $(repr(path)) has date_component = " *
            "$(ds.dim["date_component"]); expected 6"))
        dates = Array(ds["date_components"][:, :])
        size(dates) == (6, n) || throw(ArgumentError(
            "observation NetCDF $(repr(path)) date_components has shape " *
            "$(size(dates)); expected (6, $n)"))
        columns = Dict{String, Vector}(v => Array(ds[v][:])
                                       for v in _OBSERVATION_REQUIRED_VARIABLES if v != "date_components")
        for (v, data) in columns
            length(data) == n || throw(ArgumentError(
                "observation NetCDF $(repr(path)) variable $(repr(v)) has length " *
                "$(length(data)); expected $(n) to match obs dim"))
        end
        any(ismissing, dates) && throw(ArgumentError(
            "observation NetCDF $(repr(path)) has fill values in date_components"))
        # The optional vertical fields read fill values (e.g. xarray's
        # `_FillValue = NaN`) as absent; any other variable must be complete.
        value(v, i) = _observation_value(columns[v][i], v, i, path)
        absent(v, i) = (x = columns[v][i]; ismissing(x) ? NaN : Float64(x))
        records = Vector{CSObservationRecord}(undef, n)
        @inbounds for i in 1:n
            records[i] = CSObservationRecord(
                Int64(value("id", i)), ntuple(c -> Int16(dates[c, i]), 6),
                Float64(value("lat", i)), Float64(value("lon", i)),
                _parse_observation_kind(String(value("kind", i))),
                absent("elevation_m", i), absent("intake_height_m", i),
                absent("altitude_m", i), absent("pressure_pa", i),
                Float64(value("time_window_s", i)), Int8(value("flag", i)),
                Float64(value("value", i)), Float64(value("value_sigma", i)),
                Float64(value("model_sigma", i)),
                String(value("instrument_type", i)), String(value("tracer", i)))
        end
        _check_unique_ids(records, "observation NetCDF $(repr(path))")
        return CSObservationSet(records, String(time_origin))
    end
end

_observation_value(x, v, i, path) = ismissing(x) ? throw(ArgumentError(
    "observation NetCDF $(repr(path)) variable $(repr(v)) has a fill value in row $(i); only " *
    "elevation_m, intake_height_m, altitude_m and pressure_pa may be absent (NaN or fill)")) : x

function _validate_schema_version(ds, path)
    haskey(ds.attrib, "cs_observations_schema") || throw(ArgumentError(
        "observation NetCDF $(repr(path)) is missing root attribute " *
        "`cs_observations_schema`; not a CS observations file"))
    got = ds.attrib["cs_observations_schema"]
    got == "v1" && throw(ArgumentError(
        "observation NetCDF $(repr(path)) uses schema v1, which this build no longer reads. " *
        "Schema v2 stores Float64 lat/lon (Float32 coordinates can bind a point near a cell edge " *
        "to the neighbouring cell) and replaces `alt` by elevation_m, intake_height_m, altitude_m " *
        "and pressure_pa (NaN = absent), with kind, time_window_s, flag and model_sigma. Rebuild " *
        "the file from its source data with write_observations; see schemas/cs_observations_v2.toml"))
    got == _CS_OBSERVATIONS_SCHEMA_VERSION || throw(ArgumentError(
        "observation NetCDF $(repr(path)) has cs_observations_schema = " *
        "$(repr(got)); this build supports $(repr(_CS_OBSERVATIONS_SCHEMA_VERSION)) only"))
    return nothing
end

function _require_attr(ds, name, path)
    haskey(ds.attrib, name) || throw(ArgumentError(
        "observation NetCDF $(repr(path)) is missing root attribute " *
        "$(repr(name))"))
    return ds.attrib[name]
end
