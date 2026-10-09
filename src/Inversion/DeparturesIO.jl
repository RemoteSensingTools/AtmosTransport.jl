# ---------------------------------------------------------------------------
# On-disk departures (one observation set's forward-pass
# simulated values, paired with the originating observations) for the
# CS 4D-Var inversion path.
#
# Strict v1 NetCDF schema documented in `schemas/cs_departures_v1.toml`.
# Single file. Dim `obs` (unlimited) plus `date_component = 6`.
# Required vars: id (i64), tracer (string), instrument_type (string),
# date_components (i16[6, obs]), lat / lon / alt (f32), step (i64),
# panel (i8), i / j (i32), observed_value / simulated_value /
# departure / value_sigma / normalized_departure (f64). Required root
# attrs: `cs_departures_schema = "v1"`, `mesh_Nc`,
# `mesh_panel_convention`, `mesh_cs_definition_tag`, `t_start`,
# `dt_seconds`, `nsteps`, `departure_sign_convention =
# "simulated_minus_observed"`. Optional root attrs: `run_id`,
# `iteration`.
#
# Sign convention: `departure = simulated_value - observed_value`
# (i.e. the 4D-Var innovation `H(x) - y`). The root attribute
# `departure_sign_convention` pins this on disk. A v2 schema would
# have to bump the version to change it.
#
# In-memory layer:
#   * `CSDepartureRecord`   — one departure row.
#   * `CSDepartureSet`      — vector + run-level metadata.
#   * `build_departure_set` — alignment + finite gatekeeper that
#                              constructs a `CSDepartureSet` from a
#                              `CSObservationSet`, the bound
#                              `Vector{CSObservation}`, and the
#                              forward-pass simulated values.
#   * `write_departures(path, set)` / `read_departures(path)` —
#                              pure IO. The writer fails-fast on
#                              non-finite simulated or departure
#                              values; the reader rejects any file
#                              that deviates from the v1 schema.
# ---------------------------------------------------------------------------

const _CS_DEPARTURES_SCHEMA_VERSION = "v1"
const _CS_DEPARTURE_SIGN_CONVENTION = "simulated_minus_observed"

"""
    CSDepartureRecord(; id, tracer, instrument_type, date_components,
                        lat, lon, alt, step, panel, i, j,
                        observed_value, simulated_value, departure,
                        value_sigma, normalized_departure)

One audit row of the CS departure file. Stores the original
observation provenance (`id`, `tracer`, `instrument_type`,
`date_components`, `lat`, `lon`, `alt`), the bound CS mesh location
(`step`, `panel`, `i`, `j`), the observed value + sigma copied from
the parent observation, and the forward-pass triplet
(`simulated_value`, `departure`, `normalized_departure`). The
keyword constructor coerces dtypes to the on-disk widths and
rejects non-finite `simulated_value`, `departure`, or
`normalized_departure`.

Sign convention is `departure = simulated_value - observed_value`,
pinned by the v1 schema root attribute `departure_sign_convention`.
"""
struct CSDepartureRecord
    id::Int64
    tracer::String
    instrument_type::String
    date_components::_CSObservationDate
    lat::Float32
    lon::Float32
    alt::Float32
    step::Int64
    panel::Int8
    i::Int32
    j::Int32
    observed_value::Float64
    simulated_value::Float64
    departure::Float64
    value_sigma::Float64
    normalized_departure::Float64

    # Inner constructor — all validation goes through here so that
    # both keyword and positional invocations are caught. Without an
    # inner ctor, Julia auto-generates a positional one that bypasses
    # every check (review finding 2026-05-13 "Medium").
    function CSDepartureRecord(id::Int64,
                                tracer::String,
                                instrument_type::String,
                                date_components::_CSObservationDate,
                                lat::Float32, lon::Float32, alt::Float32,
                                step::Int64,
                                panel::Int8, i::Int32, j::Int32,
                                observed_value::Float64,
                                simulated_value::Float64,
                                departure::Float64,
                                value_sigma::Float64,
                                normalized_departure::Float64)
        isfinite(lat) || throw(ArgumentError(
            "CSDepartureRecord lat must be finite, got $lat"))
        -90 <= lat <= 90 || throw(ArgumentError(
            "CSDepartureRecord lat must be in [-90, 90], got $lat"))
        isfinite(lon) || throw(ArgumentError(
            "CSDepartureRecord lon must be finite, got $lon"))
        step > 0 || throw(ArgumentError(
            "CSDepartureRecord step must be positive, got $step"))
        1 <= panel <= 6 || throw(ArgumentError(
            "CSDepartureRecord panel must be in 1:6, got $panel"))
        i > 0 || throw(ArgumentError(
            "CSDepartureRecord i must be positive, got $i"))
        j > 0 || throw(ArgumentError(
            "CSDepartureRecord j must be positive, got $j"))
        isfinite(observed_value) || throw(ArgumentError(
            "CSDepartureRecord observed_value must be finite, got " *
            "$observed_value"))
        isfinite(simulated_value) || throw(ArgumentError(
            "CSDepartureRecord simulated_value must be finite, got " *
            "$simulated_value"))
        isfinite(departure) || throw(ArgumentError(
            "CSDepartureRecord departure must be finite, got $departure"))
        isfinite(normalized_departure) || throw(ArgumentError(
            "CSDepartureRecord normalized_departure must be finite, got " *
            "$normalized_departure"))
        isfinite(value_sigma) || throw(ArgumentError(
            "CSDepartureRecord value_sigma must be finite, got $value_sigma"))
        value_sigma > 0 || throw(ArgumentError(
            "CSDepartureRecord value_sigma must be positive, got $value_sigma"))
        # Arithmetic invariants pinned by the v1 schema's sign convention
        # `departure = simulated_value - observed_value` and the derived
        # `normalized_departure = departure / value_sigma`. Enforced
        # inside the inner ctor so neither keyword nor positional
        # construction can bypass it; `read_departures` constructs one
        # record per row and therefore inherits the check.
        expected_departure = simulated_value - observed_value
        isapprox(departure, expected_departure; atol = 1e-12, rtol = 1e-9) ||
            throw(ArgumentError(
                "CSDepartureRecord arithmetic invariant violated: " *
                "stored departure = $departure but simulated_value - " *
                "observed_value = $expected_departure (the v1 sign " *
                "convention is `simulated_minus_observed`)"))
        expected_normalized = departure / value_sigma
        isapprox(normalized_departure, expected_normalized;
                 atol = 1e-12, rtol = 1e-9) ||
            throw(ArgumentError(
                "CSDepartureRecord arithmetic invariant violated: " *
                "stored normalized_departure = $normalized_departure " *
                "but departure / value_sigma = $expected_normalized"))
        return new(id, tracer, instrument_type, date_components,
                   lat, lon, alt, step, panel, i, j,
                   observed_value, simulated_value, departure,
                   value_sigma, normalized_departure)
    end
end

# Keyword constructor — thin type-coercion wrapper that forwards to the
# inner positional constructor above. The actual validation lives in
# the inner ctor so a positional call cannot bypass it.
function CSDepartureRecord(; id::Integer,
                            tracer::AbstractString,
                            instrument_type::AbstractString,
                            date_components,
                            lat::Real, lon::Real, alt::Real,
                            step::Integer,
                            panel::Integer, i::Integer, j::Integer,
                            observed_value::Real,
                            simulated_value::Real,
                            departure::Real,
                            value_sigma::Real,
                            normalized_departure::Real)
    dc = _coerce_date_components(date_components)
    return CSDepartureRecord(Int64(id),
                             String(tracer),
                             String(instrument_type),
                             dc,
                             Float32(lat), Float32(lon), Float32(alt),
                             Int64(step),
                             Int8(panel), Int32(i), Int32(j),
                             Float64(observed_value),
                             Float64(simulated_value),
                             Float64(departure),
                             Float64(value_sigma),
                             Float64(normalized_departure))
end

"""
    CSDepartureSet(records, mesh_Nc, mesh_panel_convention,
                   mesh_cs_definition_tag, t_start, dt_seconds, nsteps;
                   run_id = nothing, iteration = nothing)

Collection of [`CSDepartureRecord`](@ref) rows together with the
run-level metadata that lets a downstream consumer reconstruct the
forward run that produced the file. `mesh_panel_convention` and
`mesh_cs_definition_tag` pin the cubed-sphere geometry; `t_start`,
`dt_seconds`, and `nsteps` pin the time grid. Optional `run_id` and
`iteration` are caller-supplied attribution tags.
"""
struct CSDepartureSet
    records::Vector{CSDepartureRecord}
    mesh_Nc::Int
    mesh_panel_convention::String
    mesh_cs_definition_tag::String
    t_start::String
    dt_seconds::Float64
    nsteps::Int
    run_id::Union{Nothing, String}
    iteration::Union{Nothing, Int}
end

function CSDepartureSet(records::AbstractVector{CSDepartureRecord};
                        mesh_Nc::Integer,
                        mesh_panel_convention::AbstractString,
                        mesh_cs_definition_tag::AbstractString,
                        t_start::AbstractString,
                        dt_seconds::Real,
                        nsteps::Integer,
                        run_id::Union{Nothing, AbstractString} = nothing,
                        iteration::Union{Nothing, Integer} = nothing)
    mesh_Nc > 0 || throw(ArgumentError(
        "CSDepartureSet mesh_Nc must be positive, got $mesh_Nc"))
    isempty(t_start) && throw(ArgumentError(
        "CSDepartureSet t_start must be a non-empty ISO-8601 string"))
    isfinite(dt_seconds) || throw(ArgumentError(
        "CSDepartureSet dt_seconds must be finite, got $dt_seconds"))
    dt_seconds > 0 || throw(ArgumentError(
        "CSDepartureSet dt_seconds must be positive, got $dt_seconds"))
    nsteps > 0 || throw(ArgumentError(
        "CSDepartureSet nsteps must be positive, got $nsteps"))
    # Bounds check every record against the run-level metadata so a
    # file (or hand-built set) with `step > nsteps` or `i > mesh_Nc`
    # cannot slip past either direct construction or `read_departures`.
    # `panel ∈ 1:6` is already enforced by the `CSDepartureRecord` ctor.
    @inbounds for (k, r) in enumerate(records)
        1 <= r.step <= nsteps || throw(ArgumentError(
            "CSDepartureSet record $k has step $(r.step) outside " *
            "[1, $nsteps] (run nsteps)"))
        1 <= r.i <= mesh_Nc || throw(ArgumentError(
            "CSDepartureSet record $k has i $(r.i) outside [1, $mesh_Nc] " *
            "(mesh.Nc)"))
        1 <= r.j <= mesh_Nc || throw(ArgumentError(
            "CSDepartureSet record $k has j $(r.j) outside [1, $mesh_Nc] " *
            "(mesh.Nc)"))
    end
    return CSDepartureSet(collect(records),
                          Int(mesh_Nc),
                          String(mesh_panel_convention),
                          String(mesh_cs_definition_tag),
                          String(t_start),
                          Float64(dt_seconds),
                          Int(nsteps),
                          run_id === nothing ? nothing : String(run_id),
                          iteration === nothing ? nothing : Int(iteration))
end

Base.length(set::CSDepartureSet) = length(set.records)
Base.isempty(set::CSDepartureSet) = isempty(set.records)
Base.iterate(set::CSDepartureSet, args...) = iterate(set.records, args...)
Base.getindex(set::CSDepartureSet, i::Integer) = set.records[i]

# ---------------------------------------------------------------------------
# Builder — alignment + finite-value gatekeeper
# ---------------------------------------------------------------------------

# Parse an ISO-8601 datetime in either `T`-separated (strict) or
# space-separated (common variant used by the test fixtures) form.
# Used by `build_departure_set` to re-derive the expected step index
# from each observation record's `date_components`.
function _parse_iso_datetime(s::AbstractString)
    for fmt in (Dates.DateFormat("yyyy-mm-ddTHH:MM:SS"),
                Dates.DateFormat("yyyy-mm-dd HH:MM:SS"))
        try
            return Dates.DateTime(s, fmt)
        catch
        end
    end
    throw(ArgumentError(
        "could not parse t_start = $(repr(s)) as an ISO-8601 datetime " *
        "(accepted formats: 'yyyy-mm-ddTHH:MM:SS' or 'yyyy-mm-dd HH:MM:SS')"))
end

"""
    build_departure_set(set::CSObservationSet,
                        observations::AbstractVector{<:CSObservation{CSColumnMeanObjective}},
                        simulated::AbstractVector{<:Real},
                        mesh::CubedSphereMesh,
                        t_start::AbstractString,
                        dt::Real,
                        nsteps::Integer;
                        run_id = nothing,
                        iteration = nothing) -> CSDepartureSet

Build a [`CSDepartureSet`](@ref) from aligned `(set, observations,
simulated)` triples. The three vectors must be the same length and
must correspond row-for-row: `set.records[k]` is the on-disk record
whose `CSObservation` is `observations[k]` and whose forward-pass
simulated value is `simulated[k]`.

Validation (`ArgumentError` on any violation):

- `length(set) == length(observations) == length(simulated)`.
- Every `simulated[k]` is finite. Sign convention:
  `departure = simulated - observed`. The resulting `departure` and
  `normalized_departure = departure / value_sigma` are also finite
  by construction (the parent observations are already finite via
  the observation-IO hardening).

Mesh metadata is captured at build time so the writer can pin it
into the file's root attributes.

Pass `t_start` as an ISO-8601 string. Use `Dates.format(dt, "yyyy-mm-dd
HH:MM:SS")` if you have a `DateTime` in hand.
"""
function build_departure_set(set::CSObservationSet,
                              observations::AbstractVector{<:CSObservation{CSColumnMeanObjective}},
                              simulated::AbstractVector{<:Real},
                              mesh::CubedSphereMesh,
                              t_start::AbstractString,
                              dt::Real,
                              nsteps::Integer;
                              run_id::Union{Nothing, AbstractString} = nothing,
                              iteration::Union{Nothing, Integer} = nothing)
    n = length(set)
    length(observations) == n || throw(ArgumentError(
        "build_departure_set length mismatch: length(set) = $n but " *
        "length(observations) = $(length(observations))"))
    length(simulated) == n || throw(ArgumentError(
        "build_departure_set length mismatch: length(set) = $n but " *
        "length(simulated) = $(length(simulated))"))
    nsteps > 0 || throw(ArgumentError(
        "build_departure_set nsteps must be positive, got $nsteps"))

    # Spatio-temporal alignment check: re-derive the expected
    # `(step, panel, i, j)` that `bind_to_mesh` would have produced
    # for each `set.records[k]` and compare to `observations[k]`. This
    # uses the record's `date_components` + `lat`/`lon` as the stable
    # row identity, so a swap that happens to preserve `value` /
    # `value_sigma` (e.g. two distinct records that share a sigma) is
    # still caught. The check is stricter than the bind_to_mesh
    # default — if you used `out_of_range_policy = :clamp` /
    # `:skip` / `tracer_filter`, you must pre-filter `set` to match
    # the kept observations before calling `build_departure_set`.
    t_start_dt = _parse_iso_datetime(t_start)
    dt_f = float(dt)
    locator = cell_locator(mesh)

    records = Vector{CSDepartureRecord}(undef, n)
    @inbounds for k in 1:n
        sim = simulated[k]
        isfinite(sim) || throw(ArgumentError(
            "build_departure_set simulated[$k] is not finite (got $sim); " *
            "reject before write, or add a deliberate missing-data policy"))

        rec = set.records[k]
        obs = observations[k]

        # Re-derive expected step from the record's date and the run grid.
        expected_step = _step_index_from_date(rec.date_components,
                                              t_start_dt, dt_f)
        obs.step == expected_step || throw(ArgumentError(
            "build_departure_set alignment check failed at row $k: " *
            "observations[$k].step = $(obs.step) but the record's " *
            "date_components map to step $expected_step under " *
            "t_start = $(repr(t_start)), dt = $dt. The records and " *
            "observations vectors must align by row — did you filter / " *
            "reorder one without the other, or use `:clamp` / `:skip` / " *
            "`tracer_filter` in `bind_to_mesh` without re-filtering set?"))

        # Re-derive expected (panel, i, j) from the record's lat/lon.
        expected_p, expected_i, expected_j = _locate_cs_cell(
            Float64(rec.lat), Float64(rec.lon), locator)
        obs.objective.panel == expected_p || throw(ArgumentError(
            "build_departure_set alignment check failed at row $k: " *
            "observations[$k].objective.panel = $(obs.objective.panel) " *
            "but the record's (lat, lon) = ($(rec.lat), $(rec.lon)) " *
            "maps to panel $expected_p"))
        obs.objective.i == expected_i || throw(ArgumentError(
            "build_departure_set alignment check failed at row $k: " *
            "observations[$k].objective.i = $(obs.objective.i) " *
            "but the record's (lat, lon) maps to i $expected_i"))
        obs.objective.j == expected_j || throw(ArgumentError(
            "build_departure_set alignment check failed at row $k: " *
            "observations[$k].objective.j = $(obs.objective.j) " *
            "but the record's (lat, lon) maps to j $expected_j"))

        # Defense in depth: value/sigma should also match the record.
        # `bind_to_mesh` copies these fields through verbatim, so any
        # difference here points at a hand-built observation.
        Float64(obs.value) == Float64(rec.value) || throw(ArgumentError(
            "build_departure_set provenance check failed at row $k: " *
            "observations[$k].value = $(obs.value) but " *
            "set.records[$k].value = $(rec.value)"))
        Float64(obs.sigma) == Float64(rec.value_sigma) || throw(ArgumentError(
            "build_departure_set provenance check failed at row $k: " *
            "observations[$k].sigma = $(obs.sigma) but " *
            "set.records[$k].value_sigma = $(rec.value_sigma)"))

        observed = Float64(rec.value)
        sigma = Float64(rec.value_sigma)
        departure = Float64(sim) - observed
        normalized = departure / sigma
        # `value_sigma > 0` is guaranteed by `CSObservationRecord`; we
        # therefore only need to assert finiteness of the derived
        # quantities to catch overflow from a pathological `sim`.
        isfinite(departure) || throw(ArgumentError(
            "build_departure_set departure at row $k overflowed to $departure"))
        isfinite(normalized) || throw(ArgumentError(
            "build_departure_set normalized_departure at row $k " *
            "overflowed to $normalized"))

        records[k] = CSDepartureRecord(
            id = rec.id,
            tracer = rec.tracer,
            instrument_type = rec.instrument_type,
            date_components = rec.date_components,
            lat = rec.lat, lon = rec.lon, alt = rec.alt,
            step = obs.step,
            panel = obs.objective.panel,
            i = obs.objective.i,
            j = obs.objective.j,
            observed_value = observed,
            simulated_value = Float64(sim),
            departure = departure,
            value_sigma = sigma,
            normalized_departure = normalized,
        )
    end

    return CSDepartureSet(records;
                          mesh_Nc = mesh.Nc,
                          mesh_panel_convention =
                              string(nameof(typeof(panel_convention(mesh)))),
                          mesh_cs_definition_tag =
                              string(cs_definition_tag(cs_definition(mesh))),
                          t_start = t_start,
                          dt_seconds = float(dt),
                          nsteps = Int(nsteps),
                          run_id = run_id,
                          iteration = iteration)
end

# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

"""
    write_departures(path::AbstractString, set::CSDepartureSet) -> String

Write `set` to `path` as a v1-compliant NetCDF departures file (see
`schemas/cs_departures_v1.toml`). Returns the expanded path.
Overwrites `path` if it already exists.
"""
function write_departures(path::AbstractString, set::CSDepartureSet)
    mkpath(dirname(abspath(path)))
    n = length(set)
    isfile(path) && rm(path)
    NCDatasets.NCDataset(path, "c") do ds
        NCDatasets.defDim(ds, "obs", n == 0 ? Inf : n)
        NCDatasets.defDim(ds, "date_component", 6)

        ds.attrib["cs_departures_schema"] = _CS_DEPARTURES_SCHEMA_VERSION
        ds.attrib["mesh_Nc"] = Int64(set.mesh_Nc)
        ds.attrib["mesh_panel_convention"] = set.mesh_panel_convention
        ds.attrib["mesh_cs_definition_tag"] = set.mesh_cs_definition_tag
        ds.attrib["t_start"] = set.t_start
        ds.attrib["dt_seconds"] = set.dt_seconds
        ds.attrib["nsteps"] = Int64(set.nsteps)
        ds.attrib["departure_sign_convention"] = _CS_DEPARTURE_SIGN_CONVENTION
        if set.run_id !== nothing
            ds.attrib["run_id"] = set.run_id
        end
        if set.iteration !== nothing
            ds.attrib["iteration"] = Int64(set.iteration)
        end

        ids = Vector{Int64}(undef, n)
        tracers = Vector{String}(undef, n)
        instruments = Vector{String}(undef, n)
        dates = Matrix{Int16}(undef, 6, n)
        lats = Vector{Float32}(undef, n)
        lons = Vector{Float32}(undef, n)
        alts = Vector{Float32}(undef, n)
        steps = Vector{Int64}(undef, n)
        panels = Vector{Int8}(undef, n)
        is = Vector{Int32}(undef, n)
        js = Vector{Int32}(undef, n)
        observed = Vector{Float64}(undef, n)
        simulated = Vector{Float64}(undef, n)
        departures = Vector{Float64}(undef, n)
        sigmas = Vector{Float64}(undef, n)
        normalized = Vector{Float64}(undef, n)
        @inbounds for (k, r) in enumerate(set.records)
            isfinite(r.simulated_value) || throw(ArgumentError(
                "write_departures row $k simulated_value = $(r.simulated_value) is not finite"))
            isfinite(r.departure) || throw(ArgumentError(
                "write_departures row $k departure = $(r.departure) is not finite"))
            ids[k] = r.id
            tracers[k] = r.tracer
            instruments[k] = r.instrument_type
            for c in 1:6
                dates[c, k] = r.date_components[c]
            end
            lats[k] = r.lat
            lons[k] = r.lon
            alts[k] = r.alt
            steps[k] = r.step
            panels[k] = r.panel
            is[k] = r.i
            js[k] = r.j
            observed[k] = r.observed_value
            simulated[k] = r.simulated_value
            departures[k] = r.departure
            sigmas[k] = r.value_sigma
            normalized[k] = r.normalized_departure
        end

        _write_dep_var(ds, "id", Int64, ("obs",), ids,
                       Dict("description" =>
                            "unique observation identifier, copied from the parent observations file"))
        _write_dep_var(ds, "tracer", String, ("obs",), tracers,
                       Dict("description" => "tracer name"))
        _write_dep_var(ds, "instrument_type", String, ("obs",), instruments,
                       Dict("description" => "instrument family label"))
        _write_dep_var(ds, "date_components", Int16,
                       ("date_component", "obs"), dates,
                       Dict("description" =>
                            "(year, month, day, hour, minute, second) per observation"))
        _write_dep_var(ds, "lat", Float32, ("obs",), lats,
                       Dict("units" => "degrees_north",
                            "description" => "observation latitude"))
        _write_dep_var(ds, "lon", Float32, ("obs",), lons,
                       Dict("units" => "degrees_east",
                            "description" => "observation longitude"))
        _write_dep_var(ds, "alt", Float32, ("obs",), alts,
                       Dict("units" => "m",
                            "description" => "observation altitude above sea level"))
        _write_dep_var(ds, "step", Int64, ("obs",), steps,
                       Dict("description" => "bound model-step index"))
        _write_dep_var(ds, "panel", Int8, ("obs",), panels,
                       Dict("description" => "bound cubed-sphere panel (1:6)"))
        _write_dep_var(ds, "i", Int32, ("obs",), is,
                       Dict("description" => "bound CS interior cell index along panel X"))
        _write_dep_var(ds, "j", Int32, ("obs",), js,
                       Dict("description" => "bound CS interior cell index along panel Y"))
        _write_dep_var(ds, "observed_value", Float64, ("obs",), observed,
                       Dict("description" =>
                            "observation value in the tracer's native unit"))
        _write_dep_var(ds, "simulated_value", Float64, ("obs",), simulated,
                       Dict("description" =>
                            "forward-pass simulated value in the tracer's native unit"))
        _write_dep_var(ds, "departure", Float64, ("obs",), departures,
                       Dict("description" =>
                            "simulated_value - observed_value (4D-Var innovation H(x) - y)"))
        _write_dep_var(ds, "value_sigma", Float64, ("obs",), sigmas,
                       Dict("description" => "1-sigma observation error"))
        _write_dep_var(ds, "normalized_departure", Float64, ("obs",), normalized,
                       Dict("description" =>
                            "departure / value_sigma — the sigma-normalized residual"))
    end
    return path
end

function _write_dep_var(ds, name, T, dims, data, attrs)
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

const _CS_DEPARTURES_REQUIRED_VARS = (
    "id", "tracer", "instrument_type", "date_components",
    "lat", "lon", "alt",
    "step", "panel", "i", "j",
    "observed_value", "simulated_value", "departure",
    "value_sigma", "normalized_departure",
)

const _CS_DEPARTURES_REQUIRED_ATTRS = (
    "mesh_Nc", "mesh_panel_convention", "mesh_cs_definition_tag",
    "t_start", "dt_seconds", "nsteps", "departure_sign_convention",
)

"""
    read_departures(path::AbstractString) -> CSDepartureSet

Parse a v1-compliant departures NetCDF written by
[`write_departures`](@ref) and return an in-memory
[`CSDepartureSet`](@ref).

Validation (fails fast with `ArgumentError`):

- Root attribute `cs_departures_schema` must equal `"v1"`.
- Every required root attribute must be present:
  `mesh_Nc`, `mesh_panel_convention`, `mesh_cs_definition_tag`,
  `t_start`, `dt_seconds`, `nsteps`, `departure_sign_convention`.
- `departure_sign_convention` must equal `"simulated_minus_observed"`
  (the sign is pinned by the v1 schema).
- Every required variable must be present.
- `date_components` must have dimensions `(date_component=6, obs)`.
- All `obs`-indexed variables must share the same length.
"""
function read_departures(path::AbstractString)
    isfile(path) || throw(ArgumentError(
        "departures NetCDF not found at $(repr(path))"))
    return NCDatasets.NCDataset(path, "r") do ds
        _validate_departures_schema_version(ds, path)
        for attr in _CS_DEPARTURES_REQUIRED_ATTRS
            haskey(ds.attrib, attr) || throw(ArgumentError(
                "departures NetCDF $(repr(path)) is missing root attribute " *
                "$(repr(attr))"))
        end
        sign_attr = ds.attrib["departure_sign_convention"]
        sign_attr == _CS_DEPARTURE_SIGN_CONVENTION || throw(ArgumentError(
            "departures NetCDF $(repr(path)) has departure_sign_convention = " *
            "$(repr(sign_attr)); this build supports " *
            "$(repr(_CS_DEPARTURE_SIGN_CONVENTION)) only"))

        for v in _CS_DEPARTURES_REQUIRED_VARS
            haskey(ds, v) || throw(ArgumentError(
                "departures NetCDF $(repr(path)) is missing required " *
                "variable $(repr(v))"))
        end

        n = ds.dim["obs"]
        ds.dim["date_component"] == 6 || throw(ArgumentError(
            "departures NetCDF $(repr(path)) has date_component = " *
            "$(ds.dim["date_component"]); expected 6"))

        ids = Array(ds["id"][:])
        tracers = Array(ds["tracer"][:])
        instruments = Array(ds["instrument_type"][:])
        dates = Array(ds["date_components"][:, :])
        lats = Array(ds["lat"][:])
        lons = Array(ds["lon"][:])
        alts = Array(ds["alt"][:])
        steps = Array(ds["step"][:])
        panels = Array(ds["panel"][:])
        is = Array(ds["i"][:])
        js = Array(ds["j"][:])
        observed = Array(ds["observed_value"][:])
        simulated = Array(ds["simulated_value"][:])
        departures = Array(ds["departure"][:])
        sigmas = Array(ds["value_sigma"][:])
        normalized = Array(ds["normalized_departure"][:])

        size(dates) == (6, n) || throw(ArgumentError(
            "departures NetCDF $(repr(path)) date_components has shape " *
            "$(size(dates)); expected (6, $n)"))
        for (var, len, expected) in (("id", length(ids), n),
                                     ("tracer", length(tracers), n),
                                     ("instrument_type", length(instruments), n),
                                     ("lat", length(lats), n),
                                     ("lon", length(lons), n),
                                     ("alt", length(alts), n),
                                     ("step", length(steps), n),
                                     ("panel", length(panels), n),
                                     ("i", length(is), n),
                                     ("j", length(js), n),
                                     ("observed_value", length(observed), n),
                                     ("simulated_value", length(simulated), n),
                                     ("departure", length(departures), n),
                                     ("value_sigma", length(sigmas), n),
                                     ("normalized_departure", length(normalized), n))
            len == expected || throw(ArgumentError(
                "departures NetCDF $(repr(path)) variable $(repr(var)) " *
                "has length $(len); expected $(expected) to match obs dim"))
        end

        records = Vector{CSDepartureRecord}(undef, n)
        @inbounds for k in 1:n
            dc = ntuple(c -> Int16(dates[c, k]), 6)
            records[k] = CSDepartureRecord(
                id = Int64(ids[k]),
                tracer = String(tracers[k]),
                instrument_type = String(instruments[k]),
                date_components = dc,
                lat = Float32(lats[k]),
                lon = Float32(lons[k]),
                alt = Float32(alts[k]),
                step = Int64(steps[k]),
                panel = Int8(panels[k]),
                i = Int32(is[k]),
                j = Int32(js[k]),
                observed_value = Float64(observed[k]),
                simulated_value = Float64(simulated[k]),
                departure = Float64(departures[k]),
                value_sigma = Float64(sigmas[k]),
                normalized_departure = Float64(normalized[k]),
            )
        end

        run_id = haskey(ds.attrib, "run_id") ? String(ds.attrib["run_id"]) : nothing
        iteration = haskey(ds.attrib, "iteration") ? Int(ds.attrib["iteration"]) : nothing

        return CSDepartureSet(records;
                              mesh_Nc = Int(ds.attrib["mesh_Nc"]),
                              mesh_panel_convention =
                                  String(ds.attrib["mesh_panel_convention"]),
                              mesh_cs_definition_tag =
                                  String(ds.attrib["mesh_cs_definition_tag"]),
                              t_start = String(ds.attrib["t_start"]),
                              dt_seconds = Float64(ds.attrib["dt_seconds"]),
                              nsteps = Int(ds.attrib["nsteps"]),
                              run_id = run_id,
                              iteration = iteration)
    end
end

function _validate_departures_schema_version(ds, path)
    haskey(ds.attrib, "cs_departures_schema") || throw(ArgumentError(
        "departures NetCDF $(repr(path)) is missing root attribute " *
        "`cs_departures_schema`; not a CS departures file"))
    got = ds.attrib["cs_departures_schema"]
    got == _CS_DEPARTURES_SCHEMA_VERSION || throw(ArgumentError(
        "departures NetCDF $(repr(path)) has cs_departures_schema = " *
        "$(repr(got)); this build supports " *
        "$(repr(_CS_DEPARTURES_SCHEMA_VERSION)) only"))
    return nothing
end
