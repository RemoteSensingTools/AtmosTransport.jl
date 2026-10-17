# ---------------------------------------------------------------------------
# On-disk departures (one observation set's forward-pass simulated values and
# sampling stencils, paired with the originating observation records) for the
# CS 4D-Var inversion path.
#
# Strict NetCDF schema v2, documented in `schemas/cs_departures_v2.toml`.
# Dimensions `obs` (unlimited), `date_component = 6` and `sample` (the sampled
# steps of every observation, a contiguous ragged array: observation `n` owns
# `sample_count[n]` consecutive samples). Per observation: the record's id,
# tracer, instrument, date, Float64 lat/lon, kind, time window and whether
# ClampOutOfRange kept it from outside the run; the bound cell; the vertical
# placement (code, height above the model surface, pressure); observed and
# simulated values, departure, value and model sigma and the normalized
# departure. Per sample: step, weight and the level read. Files with
# averaging-kernel column rows add a retrieval block: the model minus
# retrieval surface pressure, the model profile on the retrieval layers (a
# `retrieval_level` dimension, NaN-padded), the profile model sigma and whether
# the row was assimilated. Root attributes:
# `cs_departures_schema = "v2"`, mesh identity, time grid,
# `departure_sign_convention = "simulated_minus_observed"`, the placement
# codes, `n_clamped`, and optional `run_id` / `iteration`.
#
# Sign convention: `departure = simulated_value - observed_value` (the 4D-Var
# innovation `H(x) - y`); `normalized_departure = departure / σ` with
# `σ = √(value_sigma² + model_sigma² + profile_model_sigma²)`, the σ of the
# cost (`profile_model_sigma` is zero except for averaging-kernel columns with
# a profile uncertainty).
#
# In-memory layer:
#   * `CSDepartureRecord`   — one departure row.
#   * `CSDepartureSet`      — vector + run-level metadata.
#   * `build_departure_set` — joins a `CS4DVarResult` (observations,
#                             simulated values, stencils) to the records of
#                             a `CSObservationSet` by record id, re-deriving
#                             each observation's cell and time sampling.
#   * `write_departures(path, set)` / `read_departures(path)` — pure IO.
# ---------------------------------------------------------------------------

const _CS_DEPARTURES_SCHEMA_VERSION = "v2"
const _CS_DEPARTURE_SIGN_CONVENTION = "simulated_minus_observed"

# Stencil placement codes a departure row may carry for each record kind.
_departure_placement_codes(::PointObservationKind) =
    (VERTICAL_PLACEMENT_CODES.record_height, VERTICAL_PLACEMENT_CODES.altitude_minus_dem,
     VERTICAL_PLACEMENT_CODES.pressure, VERTICAL_PLACEMENT_CODES.absolute_altitude)
_departure_placement_codes(::ColumnObservationKind) = (COLUMN_MEAN_PLACEMENT_CODE, RETRIEVAL_COLUMN_PLACEMENT_CODE)

_departure_placement_codes_description() =
    vertical_placement_codes_description() * "; $(COLUMN_MEAN_PLACEMENT_CODE) = column_mean: " *
    "air-mass-weighted column mean (kind = column, sample_level = 0); $(RETRIEVAL_COLUMN_PLACEMENT_CODE) = " *
    "retrieval_column: averaging-kernel column on the model column stretched onto the retrieval column " *
    "(kind = column, sample_level = 0)"

"""
    CSDepartureRecord(; id, tracer, instrument_type, date_components, lat, lon,
                        kind, time_window_s, clamped, panel, i, j,
                        vertical_placement, placement_height_agl_m, placement_pressure_pa,
                        steps, weights, levels,
                        observed_value, simulated_value, departure,
                        value_sigma, model_sigma, normalized_departure,
                        surface_pressure_mismatch_pa = NaN, retrieval_profile = Float64[],
                        profile_model_sigma = 0, assimilated = true)

One audit row of the CS departure file (schema v2): the observation record's
provenance (`id`, `tracer`, `instrument_type`, `date_components`, Float64
`lat`/`lon`, `kind`, `time_window_s`; `clamped` when its time lies outside
the run and `ClampOutOfRange` kept it), where it was sampled (the cell
`panel, i, j`; the `vertical_placement` code — `Output.VERTICAL_PLACEMENT_CODES`
for point records, [`COLUMN_MEAN_PLACEMENT_CODE`](@ref) for column records —
with the height above the model surface or the pressure that placed it, NaN
when unused; the sampled `steps`, their time-sampling `weights` and the
`levels` read at each, k = 1 at the top and 0 for a column mean), and the
forward-pass values. An averaging-kernel column row
([`RETRIEVAL_COLUMN_PLACEMENT_CODE`](@ref)) also carries the model minus
retrieval surface pressure `surface_pressure_mismatch_pa` (Pa), the model
profile on the retrieval layers `retrieval_profile` (surface first, mol/mol),
its profile model error `profile_model_sigma` (``\\sqrt{\\sum_i (h_i a_i
σ_i)^2}``, 0 without a profile uncertainty) and `assimilated` (`false` when
the surface-pressure screen left it out of the cost); other rows have NaN,
an empty profile, 0 and `true`. The constructor rejects non-finite values and
checks the sign convention `departure = simulated_value − observed_value` and
`normalized_departure = departure / σ`, ``σ = \\sqrt{\\text{value\\_sigma}^2 +
\\text{model\\_sigma}^2 + \\text{profile\\_model\\_sigma}^2}``.
"""
struct CSDepartureRecord
    id::Int64
    tracer::String
    instrument_type::String
    date_components::_CSObservationDate
    lat::Float64
    lon::Float64
    kind::_ObservationKind
    time_window_s::Float64
    clamped::Bool
    panel::Int8
    i::Int32
    j::Int32
    vertical_placement::Int8
    placement_height_agl_m::Float64
    placement_pressure_pa::Float64
    steps::Vector{Int64}
    weights::Vector{Float64}
    levels::Vector{Int32}
    observed_value::Float64
    simulated_value::Float64
    departure::Float64
    value_sigma::Float64
    model_sigma::Float64
    normalized_departure::Float64
    surface_pressure_mismatch_pa::Float64
    retrieval_profile::Vector{Float64}
    profile_model_sigma::Float64
    assimilated::Bool

    # Inner constructor: every route (keyword, positional, reader) is checked.
    function CSDepartureRecord(id::Int64, tracer::String, instrument_type::String,
                               date_components::_CSObservationDate, lat::Float64, lon::Float64,
                               kind::_ObservationKind, time_window_s::Float64, clamped::Bool,
                               panel::Int8, i::Int32, j::Int32, vertical_placement::Int8,
                               placement_height_agl_m::Float64, placement_pressure_pa::Float64,
                               steps::Vector{Int64}, weights::Vector{Float64}, levels::Vector{Int32},
                               observed_value::Float64, simulated_value::Float64, departure::Float64,
                               value_sigma::Float64, model_sigma::Float64,
                               normalized_departure::Float64, surface_pressure_mismatch_pa::Float64,
                               retrieval_profile::Vector{Float64}, profile_model_sigma::Float64,
                               assimilated::Bool)
        label = "CSDepartureRecord id $(id)"
        isfinite(lat) && -90 <= lat <= 90 || throw(ArgumentError(
            "$(label): lat must be finite and in [-90, 90], got $lat"))
        isfinite(lon) || throw(ArgumentError("$(label): lon must be finite, got $lon"))
        isfinite(time_window_s) && time_window_s >= 0 || throw(ArgumentError(
            "$(label): time_window_s must be finite and >= 0, got $time_window_s"))
        1 <= panel <= 6 || throw(ArgumentError("$(label): panel must be in 1:6, got $panel"))
        i > 0 || throw(ArgumentError("$(label): i must be positive, got $i"))
        j > 0 || throw(ArgumentError("$(label): j must be positive, got $j"))
        vertical_placement in _departure_placement_codes(kind) || throw(ArgumentError(
            "$(label): vertical_placement $(vertical_placement) is not a placement code of a " *
            "$(observation_kind_label(kind)) record; allowed: $(_departure_placement_codes(kind))"))
        _absent_or_finite(placement_height_agl_m) && _absent_or_finite(placement_pressure_pa) ||
            throw(ArgumentError("$(label): placement height and pressure must be finite or NaN"))
        vertical_placement != VERTICAL_PLACEMENT_CODES.pressure || isfinite(placement_pressure_pa) ||
            throw(ArgumentError("$(label): a pressure placement needs a finite placement_pressure_pa"))
        vertical_placement in (VERTICAL_PLACEMENT_CODES.pressure, COLUMN_MEAN_PLACEMENT_CODE,
                               RETRIEVAL_COLUMN_PLACEMENT_CODE) ||
            isfinite(placement_height_agl_m) || throw(ArgumentError(
            "$(label): a placement by height needs a finite placement_height_agl_m"))
        n = length(steps)
        n >= 1 && length(weights) == n && length(levels) == n || throw(ArgumentError(
            "$(label): steps, weights and levels must be non-empty and of equal length; got " *
            "$(n), $(length(weights)), $(length(levels))"))
        CSTimeSampling(steps, weights)          # positive, increasing steps; weights sum to 1
        if kind isa ColumnObservationKind
            all(iszero, levels) || throw(ArgumentError(
                "$(label): a column mean reads every level; levels must be 0, got $(levels)"))
        else
            all(>(0), levels) || throw(ArgumentError(
                "$(label): levels of a point record must be positive, got $(levels)"))
        end
        for (name, x) in (("observed_value", observed_value), ("simulated_value", simulated_value),
                          ("departure", departure), ("normalized_departure", normalized_departure))
            isfinite(x) || throw(ArgumentError("$(label): $(name) must be finite, got $x"))
        end
        isfinite(value_sigma) && value_sigma > 0 || throw(ArgumentError(
            "$(label): value_sigma must be finite and positive, got $value_sigma"))
        isfinite(model_sigma) && model_sigma >= 0 || throw(ArgumentError(
            "$(label): model_sigma must be finite and >= 0, got $model_sigma"))
        isfinite(profile_model_sigma) && profile_model_sigma >= 0 || throw(ArgumentError(
            "$(label): profile_model_sigma must be finite and >= 0, got $profile_model_sigma"))
        if vertical_placement == RETRIEVAL_COLUMN_PLACEMENT_CODE
            isfinite(surface_pressure_mismatch_pa) && !isempty(retrieval_profile) &&
                all(isfinite, retrieval_profile) || throw(ArgumentError(
                "$(label): an averaging-kernel column row needs a finite surface_pressure_mismatch_pa and a " *
                "non-empty, finite retrieval_profile"))
        else
            isnan(surface_pressure_mismatch_pa) && isempty(retrieval_profile) && profile_model_sigma == 0 &&
                assimilated || throw(ArgumentError(
                "$(label): only averaging-kernel column rows (vertical_placement " *
                "$(RETRIEVAL_COLUMN_PLACEMENT_CODE)) carry a surface-pressure mismatch, a retrieval profile, a " *
                "profile model sigma or assimilated = false"))
        end
        # Arithmetic invariants of the sign convention.
        expected_departure = simulated_value - observed_value
        isapprox(departure, expected_departure; atol = 1e-12, rtol = 1e-9) || throw(ArgumentError(
            "$(label): departure = $departure but simulated_value - observed_value = " *
            "$expected_departure (the sign convention is `simulated_minus_observed`)"))
        sigma = _combined_sigma(value_sigma, model_sigma, profile_model_sigma)
        expected_normalized = departure / sigma
        isapprox(normalized_departure, expected_normalized; atol = 1e-12, rtol = 1e-9) ||
            throw(ArgumentError(
                "$(label): normalized_departure = $normalized_departure but departure / " *
                "sqrt(value_sigma^2 + model_sigma^2 + profile_model_sigma^2) = $expected_normalized"))
        return new(id, tracer, instrument_type, date_components, lat, lon, kind, time_window_s, clamped,
                   panel, i, j, vertical_placement, placement_height_agl_m, placement_pressure_pa,
                   steps, weights, levels, observed_value, simulated_value, departure,
                   value_sigma, model_sigma, normalized_departure, surface_pressure_mismatch_pa,
                   retrieval_profile, profile_model_sigma, assimilated)
    end
end

# A row without averaging-kernel column diagnostics, positionally.
CSDepartureRecord(id::Int64, tracer::String, instrument_type::String, date_components::_CSObservationDate,
                  lat::Float64, lon::Float64, kind::_ObservationKind, time_window_s::Float64, clamped::Bool,
                  panel::Int8, i::Int32, j::Int32, vertical_placement::Int8, placement_height_agl_m::Float64,
                  placement_pressure_pa::Float64, steps::Vector{Int64}, weights::Vector{Float64},
                  levels::Vector{Int32}, observed_value::Float64, simulated_value::Float64,
                  departure::Float64, value_sigma::Float64, model_sigma::Float64,
                  normalized_departure::Float64) =
    CSDepartureRecord(id, tracer, instrument_type, date_components, lat, lon, kind, time_window_s, clamped,
                      panel, i, j, vertical_placement, placement_height_agl_m, placement_pressure_pa, steps,
                      weights, levels, observed_value, simulated_value, departure, value_sigma, model_sigma,
                      normalized_departure, NaN, Float64[], 0.0, true)

function CSDepartureRecord(; id::Integer, tracer::AbstractString, instrument_type::AbstractString,
                           date_components, lat::Real, lon::Real, kind = PointObservationKind(),
                           time_window_s::Real = 0, clamped::Bool = false,
                           panel::Integer, i::Integer, j::Integer,
                           vertical_placement::Integer, placement_height_agl_m::Real = NaN,
                           placement_pressure_pa::Real = NaN, steps, weights, levels,
                           observed_value::Real, simulated_value::Real, departure::Real,
                           value_sigma::Real, model_sigma::Real = 0, normalized_departure::Real,
                           surface_pressure_mismatch_pa::Real = NaN, retrieval_profile = Float64[],
                           profile_model_sigma::Real = 0, assimilated::Bool = true)
    return CSDepartureRecord(Int64(id), String(tracer), String(instrument_type),
                             _coerce_date_components(date_components), Float64(lat), Float64(lon),
                             _parse_observation_kind(kind), Float64(time_window_s), clamped,
                             Int8(panel), Int32(i), Int32(j), Int8(vertical_placement),
                             Float64(placement_height_agl_m), Float64(placement_pressure_pa),
                             collect(Int64, steps), collect(Float64, weights), collect(Int32, levels),
                             Float64(observed_value), Float64(simulated_value), Float64(departure),
                             Float64(value_sigma), Float64(model_sigma), Float64(normalized_departure),
                             Float64(surface_pressure_mismatch_pa), collect(Float64, retrieval_profile),
                             Float64(profile_model_sigma), assimilated)
end

"""
    CSDepartureSet(records; mesh_Nc, mesh_panel_convention, mesh_cs_definition_tag,
                   t_start, dt_seconds, nsteps, run_id = nothing, iteration = nothing)

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
    # Bounds check every record against the run-level metadata, so neither
    # direct construction nor `read_departures` admits a step past the run or a
    # cell outside the mesh.
    @inbounds for (k, r) in enumerate(records)
        last(r.steps) <= nsteps || throw(ArgumentError(
            "CSDepartureSet record $k samples step $(last(r.steps)), outside [1, $nsteps] (run nsteps)"))
        1 <= r.i <= mesh_Nc || throw(ArgumentError(
            "CSDepartureSet record $k has i $(r.i) outside [1, $mesh_Nc] (mesh.Nc)"))
        1 <= r.j <= mesh_Nc || throw(ArgumentError(
            "CSDepartureSet record $k has j $(r.j) outside [1, $mesh_Nc] (mesh.Nc)"))
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
# Builder — join by record id + consistency gatekeeper
# ---------------------------------------------------------------------------

# Parse an ISO-8601 datetime in either `T`-separated (strict) or
# space-separated form.
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

# Whether an observation's objective is the one `bind_to_mesh` makes of its
# record: a placed point for a point record; an averaging-kernel column for a
# column record with a retrieval block (the same block), a column mean for one
# without.
_kind_matches(r::CSObservationRecord, o::AbstractCSFootprintObjective) = _kind_matches(r.kind, r, o)
_kind_matches(::PointObservationKind, r, ::CSPlacedPointObjective) = true
_kind_matches(::ColumnObservationKind, r, ::CSColumnMeanObjective) = r.retrieval === nothing
_kind_matches(::ColumnObservationKind, r, o::CSRetrievalColumnObjective) = r.retrieval == o.retrieval
_kind_matches(::AbstractObservationKind, r, ::AbstractCSFootprintObjective) = false

"""
    build_departure_set(set::CSObservationSet, result::CS4DVarResult, mesh, t_start, dt, nsteps;
                        run_id = nothing, iteration = nothing) -> CSDepartureSet
    build_departure_set(set, observations, simulated, stencils, mesh, t_start, dt, nsteps; ...)

Build a [`CSDepartureSet`](@ref) from a 4D-Var evaluation: its observations
(bound from `set` by `bind_to_mesh`), simulated values and stencils
([`CSObservationStencil`](@ref)), joined to the records of `set` by record
id, one row per observation in observation order.

Validation (`ArgumentError` on any violation):

- equal lengths; unique record ids in `set`; every observation's `id` names a
  record of `set` (hand-built observations have id 0 and no record);
- every simulated value is finite;
- each observation matches its record as `bind_to_mesh` binds it: the kind
  (point → [`CSPlacedPointObjective`](@ref), column with a retrieval block →
  [`CSRetrievalColumnObjective`](@ref) with that block, column without one →
  [`CSColumnMeanObjective`](@ref)), the containing cell of the record's
  Float64 `(lat, lon)`, the time sampling re-derived from the record date and
  window on the run grid `t_start`, `dt`, `nsteps` (for a record outside the
  run, as `ClampOutOfRange` binds it; such rows are marked `clamped`), the
  value, and
  ``σ = \\sqrt{\\text{value\\_sigma}^2 + \\text{model\\_sigma}^2 +
  \\text{profile\\_model\\_sigma}^2}``;
- each stencil matches its observation's cell and time sampling.

Averaging-kernel column rows take the surface-pressure mismatch, the
retrieval-level model profile and `assimilated` from their stencil.

`t_start` is an ISO-8601 string (`Dates.format(t, "yyyy-mm-dd HH:MM:SS")`).
"""
build_departure_set(set::CSObservationSet, result::CS4DVarResult, mesh::CubedSphereMesh,
                    t_start::AbstractString, dt::Real, nsteps::Integer; kwargs...) =
    build_departure_set(set, result.observations, result.simulated, result.stencils, mesh,
                        t_start, dt, nsteps; kwargs...)

function build_departure_set(set::CSObservationSet,
                             observations::AbstractVector{<:CSObservation},
                             simulated::AbstractVector{<:Real},
                             stencils::AbstractVector{CSObservationStencil},
                             mesh::CubedSphereMesh,
                             t_start::AbstractString,
                             dt::Real,
                             nsteps::Integer;
                             run_id::Union{Nothing, AbstractString} = nothing,
                             iteration::Union{Nothing, Integer} = nothing)
    n = length(observations)
    length(simulated) == n == length(stencils) || throw(ArgumentError(
        "build_departure_set length mismatch: $(n) observations, $(length(simulated)) simulated " *
        "values, $(length(stencils)) stencils"))
    nsteps > 0 || throw(ArgumentError(
        "build_departure_set nsteps must be positive, got $nsteps"))
    row_of = Dict{Int64, Int}()
    for (k, r) in enumerate(set.records)
        haskey(row_of, r.id) && throw(ArgumentError(
            "build_departure_set: record id $(r.id) appears more than once in the observation set; " *
            "departures are joined to records by id"))
        row_of[r.id] = k
    end
    t_start_dt = _parse_iso_datetime(t_start)
    dt_f = Float64(dt)
    locator = cell_locator(mesh)

    records = Vector{CSDepartureRecord}(undef, n)
    @inbounds for k in 1:n
        obs = observations[k]
        stencil = stencils[k]
        sim = simulated[k]
        isfinite(sim) || throw(ArgumentError(
            "build_departure_set simulated[$k] is not finite (got $sim); " *
            "reject before write, or add a deliberate missing-data policy"))
        haskey(row_of, obs.id) || throw(ArgumentError(
            "build_departure_set: observation $k has id $(obs.id), which is not a record of the " *
            "observation set (departures are built for observations bound by bind_to_mesh)"))
        rec = set.records[row_of[obs.id]]
        label = "build_departure_set observation $k (record id $(rec.id))"
        _kind_matches(rec, obs.objective) || throw(ArgumentError(
            "$(label): a $(observation_kind_label(rec.kind)) record " *
            "$(rec.retrieval === nothing ? "" : "with a retrieval block ")bound to a " *
            "$(nameof(typeof(obs.objective)))$(obs.objective isa CSRetrievalColumnObjective ?
                                               " of another retrieval block" : "")"))
        cell = _locate_cs_cell(rec.lat, rec.lon, locator)
        cell == (obs.objective.panel, obs.objective.i, obs.objective.j) || throw(ArgumentError(
            "$(label): the record's (lat, lon) = ($(rec.lat), $(rec.lon)) is in cell $(cell), " *
            "the observation in $((obs.objective.panel, obs.objective.i, obs.objective.j))"))
        # A record outside the run reaches the 4D-Var only through ClampOutOfRange.
        t_rec = _seconds_after(rec.date_components, t_start_dt)
        k_rec = floor(Int, t_rec / dt_f) + 1
        clamped = !(1 <= k_rec <= nsteps)
        expected = clamped ? _clamped_sampling(t_rec, rec.time_window_s, k_rec, dt_f, Int(nsteps)) :
                             _window_sampling(t_rec, rec.time_window_s, k_rec, dt_f, Int(nsteps))
        expected == obs.sampling || throw(ArgumentError(
            "$(label): the observation samples steps $(obs.sampling.steps) with weights " *
            "$(obs.sampling.weights), but the record maps to $(expected.steps) with " *
            "$(expected.weights) on this run grid (t_start = $(repr(t_start)), dt = $dt, nsteps = $nsteps)"))
        Float64(obs.value) == rec.value || throw(ArgumentError(
            "$(label): observation value $(obs.value) differs from the record's $(rec.value)"))
        Float64(obs.sigma) == _total_sigma(rec) || throw(ArgumentError(
            "$(label): observation sigma $(obs.sigma) differs from the record's " *
            "sqrt(value_sigma^2 + model_sigma^2 + profile_model_sigma^2) = $(_total_sigma(rec))"))
        (stencil.panel, stencil.i, stencil.j) == cell && stencil.steps == obs.sampling.steps &&
            stencil.weights == obs.sampling.weights || throw(ArgumentError(
            "$(label): its stencil (cell $((stencil.panel, stencil.i, stencil.j)), steps " *
            "$(stencil.steps)) does not belong to the observation"))

        observed = rec.value
        departure = Float64(sim) - observed
        normalized = departure / _total_sigma(rec)
        isfinite(departure) && isfinite(normalized) || throw(ArgumentError(
            "$(label): the departure overflowed ($departure, normalized $normalized)"))
        records[k] = CSDepartureRecord(
            id = rec.id, tracer = rec.tracer, instrument_type = rec.instrument_type,
            date_components = rec.date_components, lat = rec.lat, lon = rec.lon,
            kind = rec.kind, time_window_s = rec.time_window_s, clamped = clamped,
            panel = stencil.panel, i = stencil.i, j = stencil.j,
            vertical_placement = stencil.placement_method,
            placement_height_agl_m = stencil.height_agl_m, placement_pressure_pa = stencil.pressure_pa,
            steps = stencil.steps, weights = stencil.weights, levels = stencil.levels,
            observed_value = observed, simulated_value = Float64(sim), departure = departure,
            value_sigma = rec.value_sigma, model_sigma = rec.model_sigma,
            normalized_departure = normalized,
            surface_pressure_mismatch_pa = stencil.surface_pressure_mismatch_pa,
            retrieval_profile = stencil.retrieval_profile, profile_model_sigma = _profile_model_sigma(rec),
            assimilated = stencil.assimilated)
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

# (name, on-disk type, record field, attributes) of the per-observation
# variables other than `date_components`, `kind` and the sample arrays.
const _DEPARTURE_VARIABLES = (
    ("id", Int64, :id, ("description" => "unique observation identifier, the record id of the parent observations file",)),
    ("tracer", String, :tracer, ("description" => "tracer name",)),
    ("instrument_type", String, :instrument_type, ("description" => "instrument family label",)),
    ("lat", Float64, :lat, ("units" => "degrees_north", "description" => "observation latitude")),
    ("lon", Float64, :lon, ("units" => "degrees_east", "description" => "observation longitude")),
    ("time_window_s", Float64, :time_window_s,
     ("units" => "s", "description" => "sampling window centred on the record time; 0 = one model step")),
    ("clamped", Int8, :clamped,
     ("description" => "1 when the record time lies outside the run and ClampOutOfRange kept it, else 0",)),
    ("panel", Int8, :panel, ("description" => "bound cubed-sphere panel (1:6)",)),
    ("i", Int32, :i, ("description" => "bound CS interior cell index along panel X",)),
    ("j", Int32, :j, ("description" => "bound CS interior cell index along panel Y",)),
    ("vertical_placement", Int8, :vertical_placement,
     ("description" => "how the observation is placed in the column; see the vertical_placement_codes attribute",)),
    ("placement_height_agl_m", Float64, :placement_height_agl_m,
     ("units" => "m", "description" => "height above the model surface that selects the layer; NaN when not placed by height")),
    ("placement_pressure_pa", Float64, :placement_pressure_pa,
     ("units" => "Pa", "description" => "pressure that selects the layer; NaN when not placed by pressure")),
    ("observed_value", Float64, :observed_value,
     ("description" => "observation value in the tracer's native unit",)),
    ("simulated_value", Float64, :simulated_value,
     ("description" => "forward-pass simulated value (time-sampling-weighted) in the tracer's native unit",)),
    ("departure", Float64, :departure,
     ("description" => "simulated_value - observed_value (4D-Var innovation H(x) - y)",)),
    ("value_sigma", Float64, :value_sigma, ("description" => "1-sigma observation error",)),
    ("model_sigma", Float64, :model_sigma,
     ("description" => "1-sigma model/representation error, added in quadrature",)),
    ("normalized_departure", Float64, :normalized_departure,
     ("description" => "departure / sqrt(value_sigma^2 + model_sigma^2 + profile_model_sigma^2), the residual " *
                       "of the 4D-Var cost (profile_model_sigma: averaging-kernel columns, else 0)",)),
)

const _CS_DEPARTURES_REQUIRED_VARS = ("date_components", "kind", "sample_count", "sample_step",
                                      "sample_weight", "sample_level",
                                      (first(v) for v in _DEPARTURE_VARIABLES)...)

const _CS_DEPARTURES_REQUIRED_ATTRS = (
    "mesh_Nc", "mesh_panel_convention", "mesh_cs_definition_tag",
    "t_start", "dt_seconds", "nsteps", "departure_sign_convention", "n_clamped",
)

"""
    write_departures(path::AbstractString, set::CSDepartureSet) -> String

Write `set` to `path` as a schema-v2 NetCDF departures file (see
`schemas/cs_departures_v2.toml`) and return `path`; an existing file is
overwritten. The sampled steps form a contiguous ragged array on the `sample`
dimension: observation `n` owns `sample_count[n]` consecutive entries of
`sample_step`, `sample_weight` and `sample_level`. When any row is an
averaging-kernel column, the file adds `surface_pressure_mismatch_pa`,
`retrieval_model_profile` (on a `retrieval_level` dimension, surface first,
NaN-padded), `profile_model_sigma` and `assimilated` for every row.
"""
function write_departures(path::AbstractString, set::CSDepartureSet)
    mkpath(dirname(abspath(path)))
    n = length(set)
    nsample = sum(r -> length(r.steps), set.records; init = 0)
    isfile(path) && rm(path)
    NCDatasets.NCDataset(path, "c") do ds
        NCDatasets.defDim(ds, "obs", n == 0 ? Inf : n)
        NCDatasets.defDim(ds, "date_component", 6)
        NCDatasets.defDim(ds, "sample", nsample == 0 ? Inf : nsample)

        ds.attrib["cs_departures_schema"] = _CS_DEPARTURES_SCHEMA_VERSION
        ds.attrib["mesh_Nc"] = Int64(set.mesh_Nc)
        ds.attrib["mesh_panel_convention"] = set.mesh_panel_convention
        ds.attrib["mesh_cs_definition_tag"] = set.mesh_cs_definition_tag
        ds.attrib["t_start"] = set.t_start
        ds.attrib["dt_seconds"] = set.dt_seconds
        ds.attrib["nsteps"] = Int64(set.nsteps)
        ds.attrib["departure_sign_convention"] = _CS_DEPARTURE_SIGN_CONVENTION
        ds.attrib["vertical_placement_codes"] = _departure_placement_codes_description()
        ds.attrib["n_clamped"] = Int64(count(r -> r.clamped, set.records))
        set.run_id === nothing || (ds.attrib["run_id"] = set.run_id)
        set.iteration === nothing || (ds.attrib["iteration"] = Int64(set.iteration))

        dates = Matrix{Int16}(undef, 6, n)
        @inbounds for (k, r) in enumerate(set.records), c in 1:6
            dates[c, k] = r.date_components[c]
        end
        _write_dep_var(ds, "date_components", Int16, ("date_component", "obs"), dates,
                       ("description" => "(year, month, day, hour, minute, second) UTC per observation",))
        _write_dep_var(ds, "kind", String, ("obs",),
                       String[observation_kind_label(r.kind) for r in set.records],
                       ("description" => "\"point\" or \"column\"",))
        for (name, T, field, attrs) in _DEPARTURE_VARIABLES
            _write_dep_var(ds, name, T, ("obs",), T[getfield(r, field) for r in set.records], attrs)
        end
        _write_dep_var(ds, "sample_count", Int32, ("obs",), Int32[length(r.steps) for r in set.records],
                       ("description" => "number of sampled model steps; observation n owns the next sample_count[n] entries of the sample dimension",))
        _write_dep_var(ds, "sample_step", Int64, ("sample",),
                       reduce(vcat, (r.steps for r in set.records); init = Int64[]),
                       ("description" => "sampled model step (the state after it)",))
        _write_dep_var(ds, "sample_weight", Float64, ("sample",),
                       reduce(vcat, (r.weights for r in set.records); init = Float64[]),
                       ("description" => "time-sampling weight of the step (an observation's weights sum to 1)",))
        _write_dep_var(ds, "sample_level", Int32, ("sample",),
                       reduce(vcat, (r.levels for r in set.records); init = Int32[]),
                       ("description" => "model level read at the step (k = 1 at the top); 0 = every level (column mean)",))
        _write_departure_retrieval_block(ds, set.records)
    end
    return path
end

# The averaging-kernel column diagnostics, written when any row is such a
# column: `retrieval_level` (the most retrieval layers of any row), the
# surface-pressure mismatch, the retrieval-level model profile (NaN-padded,
# surface first), the profile model sigma and the assimilated flag.
const _DEPARTURE_RETRIEVAL_VARIABLES = ("surface_pressure_mismatch_pa", "retrieval_model_profile",
                                        "profile_model_sigma", "assimilated")

function _write_departure_retrieval_block(ds, records)
    any(r -> r.vertical_placement == RETRIEVAL_COLUMN_PLACEMENT_CODE, records) || return nothing
    n = length(records)
    nlevel = maximum(r -> length(r.retrieval_profile), records)
    NCDatasets.defDim(ds, "retrieval_level", nlevel)
    profiles = fill(NaN, nlevel, n)
    for (k, r) in enumerate(records)
        profiles[1:length(r.retrieval_profile), k] .= r.retrieval_profile
    end
    _write_dep_var(ds, "surface_pressure_mismatch_pa", Float64, ("obs",),
                   Float64[r.surface_pressure_mismatch_pa for r in records],
                   ("units" => "Pa", "description" =>
                    "averaging-kernel columns: model minus retrieval surface pressure (time-weighted), absorbed " *
                    "by the stretch of the model column onto the retrieval column; NaN for other rows"))
    _write_dep_var(ds, "retrieval_model_profile", Float64, ("retrieval_level", "obs"), profiles,
                   ("units" => "mol mol-1", "ordering" => "surface_to_top", "description" =>
                    "averaging-kernel columns: the model's dry mole fraction on the retrieval layers (the " *
                    "stretched model column rebinned by pressure overlap, time-weighted), from the surface up; " *
                    "NaN-padded, NaN for other rows"))
    _write_dep_var(ds, "profile_model_sigma", Float64, ("obs",), Float64[r.profile_model_sigma for r in records],
                   ("description" => "1-sigma model error of an averaging-kernel column, sqrt(sum_i (h_i a_i " *
                    "sigma_i)^2) from its profile uncertainty, added in quadrature; 0 for other rows",))
    _write_dep_var(ds, "assimilated", Int8, ("obs",), Int8[r.assimilated for r in records],
                   ("description" => "1 when the row is in the 4D-Var cost; 0 for an averaging-kernel column " *
                    "left out by max_surface_pressure_mismatch_pa (unplaced_records = skip)",))
    return nothing
end

# The block as per-row vectors (defaults when the file has none).
function _read_departure_retrieval_block(ds, n::Int, path)
    present = [v for v in _DEPARTURE_RETRIEVAL_VARIABLES if haskey(ds, v)]
    isempty(present) && return (fill(NaN, n), [Float64[] for _ in 1:n], zeros(n), trues(n))
    length(present) == length(_DEPARTURE_RETRIEVAL_VARIABLES) || throw(ArgumentError(
        "departures NetCDF $(repr(path)) has a partial retrieval block ($(join(present, ", "))); it needs " *
        "all of $(join(_DEPARTURE_RETRIEVAL_VARIABLES, ", "))"))
    nomissing(name, x) = ismissing(x) ? throw(ArgumentError(
        "departures NetCDF $(repr(path)) variable $(repr(name)) has fill values")) : x
    profile_var = ds["retrieval_model_profile"]
    Tuple(NCDatasets.dimnames(profile_var)) == ("retrieval_level", "obs") || throw(ArgumentError(
        "departures NetCDF $(repr(path)) variable \"retrieval_model_profile\" has dimensions " *
        "$(NCDatasets.dimnames(profile_var)); expected (retrieval_level, obs)"))
    # The profile states its ordering and a mole-fraction unit, the mismatch a
    # pressure unit; supported alternatives are converted to mol/mol and Pa.
    ordering = get(profile_var.attrib, "ordering", nothing)
    ordering == "surface_to_top" || throw(ArgumentError(
        "departures NetCDF $(repr(path)) variable \"retrieval_model_profile\" has ordering = $(repr(ordering)); " *
        "it must state ordering = \"surface_to_top\" (index 1 is the retrieval surface)"))
    profile_scale = mole_fraction_unit_scale(get(profile_var.attrib, "units", nothing),
                                             "departures NetCDF $(repr(path)) variable \"retrieval_model_profile\"")
    mismatch_var = ds["surface_pressure_mismatch_pa"]
    mismatch_units = get(mismatch_var.attrib, "units", nothing)
    mismatch_key = _pressure_unit_key(mismatch_units)
    mismatch_scale = mismatch_key === nothing ? nothing : get(_PRESSURE_UNITS_PA, mismatch_key, nothing)
    mismatch_scale === nothing && throw(ArgumentError(
        "departures NetCDF $(repr(path)) variable \"surface_pressure_mismatch_pa\" has units " *
        "$(repr(mismatch_units)); expected a pressure unit, one of $(join(sort!(collect(keys(_PRESSURE_UNITS_PA))), ", "))"))
    mismatch = Float64[ismissing(x) ? NaN : Float64(x) * mismatch_scale for x in Array(mismatch_var)]
    sigma = Float64[Float64(nomissing("profile_model_sigma", x)) for x in Array(ds["profile_model_sigma"])]
    assimilated = Bool[nomissing("assimilated", x) != 0 for x in Array(ds["assimilated"])]
    raw = map(x -> ismissing(x) ? NaN : Float64(x) * profile_scale, Array(profile_var))
    length(mismatch) == length(sigma) == length(assimilated) == size(raw, 2) == n || throw(ArgumentError(
        "departures NetCDF $(repr(path)) retrieval block variables do not have $(n) rows"))
    profiles = [begin
                    col = view(raw, :, k)
                    count = something(findfirst(isnan, col), length(col) + 1) - 1
                    all(isnan, view(col, (count + 1):length(col))) || throw(ArgumentError(
                        "departures NetCDF $(repr(path)) retrieval_model_profile row $(k) is not NaN-padded"))
                    col[1:count]
                end for k in 1:n]
    return (mismatch, profiles, sigma, assimilated)
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

"""
    read_departures(path::AbstractString) -> CSDepartureSet

Parse a schema-v2 departures NetCDF written by [`write_departures`](@ref)
and return an in-memory [`CSDepartureSet`](@ref).

Validation (fails fast with `ArgumentError`):

- Root attribute `cs_departures_schema` must equal `"v2"` (schema-v1 files,
  with Float32 coordinates and a single step, are rejected).
- Every required root attribute must be present:
  `mesh_Nc`, `mesh_panel_convention`, `mesh_cs_definition_tag`,
  `t_start`, `dt_seconds`, `nsteps`, `departure_sign_convention` (equal to
  `"simulated_minus_observed"`), and `n_clamped` (the number of rows marked
  `clamped`).
- Every required variable must be present, `date_components` dimensioned
  `(date_component = 6, obs)`, every `obs` variable of the `obs` length, and
  the `sample_count`s must add up to the `sample` length.
- The averaging-kernel column block (`surface_pressure_mismatch_pa`,
  `retrieval_model_profile (retrieval_level, obs)`, `profile_model_sigma`,
  `assimilated`) is all or nothing; without it every row has the defaults
  (NaN, no profile, 0, assimilated). The profile must state
  `ordering = "surface_to_top"` and a mole-fraction unit, the mismatch a
  pressure unit (converted to mol/mol and Pa).
- Every row must pass the [`CSDepartureRecord`](@ref) checks.
"""
function read_departures(path::AbstractString)
    isfile(path) || throw(ArgumentError(
        "departures NetCDF not found at $(repr(path))"))
    return NCDatasets.NCDataset(path, "r") do ds
        _validate_departures_schema_version(ds, path)
        for attr in _CS_DEPARTURES_REQUIRED_ATTRS
            haskey(ds.attrib, attr) || throw(ArgumentError(
                "departures NetCDF $(repr(path)) is missing root attribute $(repr(attr))"))
        end
        sign_attr = ds.attrib["departure_sign_convention"]
        sign_attr == _CS_DEPARTURE_SIGN_CONVENTION || throw(ArgumentError(
            "departures NetCDF $(repr(path)) has departure_sign_convention = " *
            "$(repr(sign_attr)); this build supports " *
            "$(repr(_CS_DEPARTURE_SIGN_CONVENTION)) only"))
        for v in _CS_DEPARTURES_REQUIRED_VARS
            haskey(ds, v) || throw(ArgumentError(
                "departures NetCDF $(repr(path)) is missing required variable $(repr(v))"))
        end

        n = ds.dim["obs"]
        ds.dim["date_component"] == 6 || throw(ArgumentError(
            "departures NetCDF $(repr(path)) has date_component = " *
            "$(ds.dim["date_component"]); expected 6"))
        dates = Array(ds["date_components"][:, :])
        size(dates) == (6, n) || throw(ArgumentError(
            "departures NetCDF $(repr(path)) date_components has shape " *
            "$(size(dates)); expected (6, $n)"))
        obs_vars = ("kind", "sample_count", (first(v) for v in _DEPARTURE_VARIABLES)...)
        columns = Dict{String, Vector}(v => Array(ds[v][:]) for v in obs_vars)
        for (v, data) in columns
            length(data) == n || throw(ArgumentError(
                "departures NetCDF $(repr(path)) variable $(repr(v)) has length $(length(data)); " *
                "expected $(n) to match obs dim"))
        end
        sample_step = Array(ds["sample_step"][:])
        sample_weight = Array(ds["sample_weight"][:])
        sample_level = Array(ds["sample_level"][:])
        counts = columns["sample_count"]
        nsample = length(sample_step)
        length(sample_weight) == nsample == length(sample_level) && sum(Int, counts; init = 0) == nsample &&
            all(>(0), counts) || throw(ArgumentError(
            "departures NetCDF $(repr(path)) has inconsistent sample arrays: sample_count adds up to " *
            "$(sum(Int, counts; init = 0)) (each must be positive); sample_step/weight/level have " *
            "$(nsample)/$(length(sample_weight))/$(length(sample_level)) entries"))

        for (v, data) in columns
            v in ("placement_height_agl_m", "placement_pressure_pa") && continue
            any(ismissing, data) && throw(ArgumentError(
                "departures NetCDF $(repr(path)) variable $(repr(v)) has fill values"))
        end
        (any(ismissing, sample_step) || any(ismissing, sample_weight) || any(ismissing, sample_level)) &&
            throw(ArgumentError("departures NetCDF $(repr(path)) has fill values in its sample arrays"))
        mismatch, profiles, profile_sigma, assimilated = _read_departure_retrieval_block(ds, n, path)
        records = Vector{CSDepartureRecord}(undef, n)
        stop = 0
        @inbounds for k in 1:n
            start = stop + 1
            stop += Int(counts[k])
            # The placement height and pressure read a fill value as unused (NaN).
            col(name) = (x = columns[name][k]; ismissing(x) ? NaN : x)
            records[k] = CSDepartureRecord(
                id = col("id"), tracer = String(col("tracer")),
                instrument_type = String(col("instrument_type")),
                date_components = ntuple(c -> Int16(dates[c, k]), 6),
                lat = col("lat"), lon = col("lon"), kind = String(col("kind")),
                time_window_s = col("time_window_s"), clamped = col("clamped") != 0,
                panel = col("panel"), i = col("i"), j = col("j"),
                vertical_placement = col("vertical_placement"),
                placement_height_agl_m = col("placement_height_agl_m"),
                placement_pressure_pa = col("placement_pressure_pa"),
                steps = sample_step[start:stop], weights = sample_weight[start:stop],
                levels = sample_level[start:stop],
                observed_value = col("observed_value"), simulated_value = col("simulated_value"),
                departure = col("departure"), value_sigma = col("value_sigma"),
                model_sigma = col("model_sigma"), normalized_departure = col("normalized_departure"),
                surface_pressure_mismatch_pa = mismatch[k], retrieval_profile = profiles[k],
                profile_model_sigma = profile_sigma[k], assimilated = assimilated[k])
        end

        n_clamped = count(r -> r.clamped, records)
        Int(ds.attrib["n_clamped"]) == n_clamped || throw(ArgumentError(
            "departures NetCDF $(repr(path)) has n_clamped = $(ds.attrib["n_clamped"]) but " *
            "$(n_clamped) rows marked clamped"))
        run_id = haskey(ds.attrib, "run_id") ? String(ds.attrib["run_id"]) : nothing
        iteration = haskey(ds.attrib, "iteration") ? Int(ds.attrib["iteration"]) : nothing
        return CSDepartureSet(records;
                              mesh_Nc = Int(ds.attrib["mesh_Nc"]),
                              mesh_panel_convention = String(ds.attrib["mesh_panel_convention"]),
                              mesh_cs_definition_tag = String(ds.attrib["mesh_cs_definition_tag"]),
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
    got == "v1" && throw(ArgumentError(
        "departures NetCDF $(repr(path)) uses schema v1, which this build no longer reads " *
        "(Float32 coordinates and one step per observation); schema v2 stores Float64 coordinates, " *
        "the vertical placement and every sampled step. Rewrite it from the 4D-Var result with " *
        "build_departure_set / write_departures"))
    got == _CS_DEPARTURES_SCHEMA_VERSION || throw(ArgumentError(
        "departures NetCDF $(repr(path)) has cs_departures_schema = " *
        "$(repr(got)); this build supports " *
        "$(repr(_CS_DEPARTURES_SCHEMA_VERSION)) only"))
    return nothing
end
