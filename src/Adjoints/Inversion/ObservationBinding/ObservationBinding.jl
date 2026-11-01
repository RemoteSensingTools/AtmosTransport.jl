# ---------------------------------------------------------------------------
# Bridge from on-disk `CSObservationSet` records to the `CSObservation`s of
# the CS 4D-Var surface-flux path (`cs_surface_flux_4dvar`).
#
# Three mappings:
#
#   * Time. Step `k` covers `[t_start + (k-1)*dt, t_start + k*dt)` and is
#     represented by the state at its end (after convection). A record without
#     a time window samples the step whose interval contains its time; a record
#     with `time_window_s = W > 0` samples every step whose interval overlaps
#     `[t - W/2, t + W/2]`, weighted by the overlap and renormalised over the
#     part of the window inside the run (`CSTimeSampling`). A record whose time
#     is outside the run goes to the out-of-range policy.
#
#   * Geography. `(lat, lon)` (Float64) -> the cell that contains the point,
#     from the inverse gnomonic map of the mesh: the locator the forward
#     observation sampler uses (`Observations.cell_locator`/`locate`), so an
#     observation is sampled and inverted in the same cell.
#
#   * Vertical. A point record becomes an `Observations.SoundingRequest` and is
#     placed by the shared rule of the forward sampler
#     (`Observations.resolve_placement` with the run's `VerticalPlacementSpec` and
#     `ObservationTerrain`) into a `CSPlacedPointObjective`; its layer is chosen
#     at every sampled state by `Observations.sample_column_layer!`. A column record
#     with a retrieval block becomes a `CSRetrievalColumnObjective` (its
#     averaging kernel on the model column stretched onto the retrieval
#     column, built at every sampled state), one without a block a
#     `CSColumnMeanObjective`.
# ---------------------------------------------------------------------------

"""
    AbstractOutOfRangePolicy

Policy applied when an observation maps outside the valid model-step range.
Use [`RejectOutOfRange`](@ref), [`SkipOutOfRange`](@ref), or
[`ClampOutOfRange`](@ref).
"""
abstract type AbstractOutOfRangePolicy end

"""Reject the first observation outside the valid model-step range."""
struct RejectOutOfRange <: AbstractOutOfRangePolicy end

"""Drop observations outside the valid model-step range."""
struct SkipOutOfRange <: AbstractOutOfRangePolicy end

"""
Keep observations outside the valid model-step range: a record whose time
window reaches into the run samples the part inside it (truncated and
renormalised), any other record the nearest step (first or last).
"""
struct ClampOutOfRange <: AbstractOutOfRangePolicy end

_parse_out_of_range_policy(policy::AbstractOutOfRangePolicy) = policy

_parse_out_of_range_policy(raw) =
    parse_choice(raw, (reject = RejectOutOfRange(), skip = SkipOutOfRange(), clamp = ClampOutOfRange()),
                 "out_of_range_policy"; case_sensitive = true)

"""
    AbstractFlaggedRecordPolicy

What `bind_to_mesh` does with records whose `flag` is not 0 (not to be
assimilated): [`DropFlaggedRecords`](@ref) (`"drop"`, the default) or
[`CosampleFlaggedRecords`](@ref) (`"cosample"`).
"""
abstract type AbstractFlaggedRecordPolicy end

"""`flagged_records = "drop"` (default): records with a nonzero flag are not bound."""
struct DropFlaggedRecords <: AbstractFlaggedRecordPolicy end

"""
`flagged_records = "cosample"`: records with a nonzero flag are bound as
co-samples (`CSObservation(...; assimilate = false)`): simulated on the
forward replay and reported with their flag, never in the cost or the
gradient. The OCO-2 MIP needs co-samples of every record, the withheld and
evaluation-only ones included.
"""
struct CosampleFlaggedRecords <: AbstractFlaggedRecordPolicy end

_parse_flagged_records(policy::AbstractFlaggedRecordPolicy) = policy
_parse_flagged_records(raw) =
    parse_choice(raw, (drop = DropFlaggedRecords(), cosample = CosampleFlaggedRecords()),
                 "flagged_records"; case_sensitive = true)

_binds_flagged(::DropFlaggedRecords) = false
_binds_flagged(::CosampleFlaggedRecords) = true

"""
    AbstractCellAveraging

Whether `bind_to_mesh` averages column records that fall in one model cell:
[`NoCellAveraging`](@ref) (the default) or [`CellAverage`](@ref). The
alternative that keeps every record and inflates its model error instead is
[`inflate_cell_model_error`](@ref).
"""
abstract type AbstractCellAveraging end

"""`cell_averaging = NoCellAveraging()` (default): every record is its own observation."""
struct NoCellAveraging <: AbstractCellAveraging end

"""
    CellAverage(; time_tolerance_s = 60.0, correlation = 0.0)

Average the assimilated column records (flag 0) of one tracer that fall in
the same model cell, sample the same model steps with the same weights, and
lie within `time_tolerance_s` of the earliest record of their group (records
of one overpass: an OCO-2 orbit crosses a C90 cell in about 15 s, and the
next orbit passes 99 minutes later). A group of ``n ≥ 2`` records
``y_i`` with measurement errors ``σ_i`` (`value_sigma`) becomes one
observation with weights ``w_i = 1/σ_i^2``:

```math
\\bar y = \\frac{\\sum_i w_i y_i}{\\sum_i w_i}, \\qquad
\\bar H = \\frac{\\sum_i w_i H_i}{\\sum_i w_i}, \\qquad
σ^2_\\mathrm{meas} = \\frac{\\sum_i \\sum_j w_i w_j ρ_{ij} σ_i σ_j}{(\\sum_i w_i)^2}, \\qquad
\\bar σ^2 = σ^2_\\mathrm{meas} + \\frac{\\sum_i w_i σ^2_{\\mathrm{mod},i}}{\\sum_i w_i},
```

with ``ρ_{ii} = 1`` and ``ρ_{ij} =`` `correlation` between distinct records
(0 by default: separate 10-s spans; their within-span correlation is already
in ``σ_i``), and ``σ^2_{\\mathrm{mod},i}`` the record's model error
(`model_sigma²` plus the profile term of an averaging-kernel column), added
once, not reduced by the averaging. ``\\bar H`` is a
[`CSCellAverageObjective`](@ref); the operators are linear, so its gradient is
the ``w``-weighted combination of the members'. The members stay bound as
co-samples (`assimilate = false`), so every record keeps its own simulated
value. Needs `max_surface_pressure_mismatch_pa = Inf` (the screen applies to
records, not averages).
"""
struct CellAverage <: AbstractCellAveraging
    time_tolerance_s::Float64
    correlation::Float64
    function CellAverage(time_tolerance_s::Real, correlation::Real)
        isfinite(time_tolerance_s) && time_tolerance_s >= 0 || throw(ArgumentError(
            "CellAverage time_tolerance_s must be finite and >= 0; got $(time_tolerance_s)"))
        isfinite(correlation) && 0 <= correlation <= 1 || throw(ArgumentError(
            "CellAverage correlation must be in [0, 1]; got $(correlation)"))
        return new(Float64(time_tolerance_s), Float64(correlation))
    end
end
CellAverage(; time_tolerance_s::Real = 60.0, correlation::Real = 0.0) = CellAverage(time_tolerance_s, correlation)

"""
    bind_to_mesh(set::CSObservationSet, mesh::CubedSphereMesh, t_start::Dates.DateTime, dt::Real;
                 nsteps = nothing, tracer_filter = nothing, out_of_range_policy = RejectOutOfRange(),
                 placement = VerticalPlacementSpec(), terrain = ObservationTerrain(),
                 max_surface_pressure_mismatch_pa = Inf, flagged_records = DropFlaggedRecords(),
                 cell_averaging = NoCellAveraging())
        -> Vector{CSObservation}

Bind each record of `set` to a [`CSObservation`](@ref) of the CS 4D-Var path:
its time sampling, the cell that contains it, and its objective.

- **Time.** `t_start` is the model start (UTC) and `dt` the step length (s):
  step `k` covers `[t_start + (k−1)dt, t_start + k dt)` and is represented by
  the state at its end, after convection. A record with `time_window_s = 0`
  samples the step whose interval contains its time. A record with
  `time_window_s = W > 0` samples the window `[t − W/2, t + W/2]` around its
  time `t`: every step whose interval overlaps the window, with weight
  ``\\Delta t_k/\\sum\\Delta t`` where ``\\Delta t_k`` is the overlap of step
  `k`'s interval with the window (a step only partly inside counts its
  overlap; steps fully inside count `dt`, TM5-4DVAR's Δt/ΣΔt mean). The window
  is truncated at the run's ends (`0` and `nsteps·dt`) and the weights are
  renormalised over the part inside the run. See [`CSTimeSampling`](@ref).
- **Out of range.** A record whose time falls outside `1:nsteps` (only
  `step >= 1` is enforced when `nsteps === nothing`) is handled by
  `out_of_range_policy`: [`RejectOutOfRange`](@ref) (default, `"reject"`)
  throws, [`SkipOutOfRange`](@ref) (`"skip"`) drops it,
  [`ClampOutOfRange`](@ref) (`"clamp"`) samples the part of its window inside
  the run (truncated and renormalised) when the window reaches into the run,
  else step `1` or `nsteps`. Skipped and clamped records are counted in an
  `@info`. Records with a time window need `nsteps`.
- **Ids.** Record ids must be unique (departures are joined to records by
  id).
- **Cell.** `(lat, lon)` (Float64) → the cell containing the point
  (`Observations.locate`, the forward observation sampler's locator).
- **Point records** ([`PointObservationKind`](@ref)) become
  [`CSPlacedPointObjective`](@ref)s, placed by the forward sampler's rule
  (`Observations.resolve_placement`) with `placement` (an
  `Observations.VerticalPlacementSpec`: the ordered `vertical_coordinates`, height
  above ground and/or pressure; the pressure basis; the altitude option; the
  unplaced policy) and the static ground fields of `terrain` (an
  `Observations.ObservationTerrain`: model surface height, point DEM). A point
  record that gives none of the coordinates in use is unplaced: an
  `ArgumentError` with `unplaced_records = "error"` (default), dropped and
  counted with `SkipUnplaced()`.
- **Column records** ([`ColumnObservationKind`](@ref)) with a retrieval
  block become [`CSRetrievalColumnObjective`](@ref)s: the averaging-kernel
  column on the interface pressures of `placement.pressure_basis` (total
  pressures from the humidity by default), with the surface-pressure limit
  `max_surface_pressure_mismatch_pa` (Pa, positive; `Inf`, the default, lets
  the stretch absorb any mismatch, as TM5) and the unplaced policy of
  `placement` as its mismatch policy: a record whose model surface pressure
  differs from the retrieval's by more than the limit is an error under
  `unplaced_records = "error"`, and is left out of the cost (and counted)
  under `"skip"`. The model surface pressure is known only at the sampled
  states, so the check runs in the 4D-Var's forward replay; it is the surface
  pressure of the pressure basis, so with `DryInterfaces` the mismatch also
  contains the weight of the water vapour (about 0.5 % of the surface
  pressure, more in the tropics). Column records without a block become
  [`CSColumnMeanObjective`](@ref)s.
- Records with a nonzero `flag` are not assimilated: dropped with
  `flagged_records = DropFlaggedRecords()` (`"drop"`, the default), bound as
  co-samples (`CSObservation(...; assimilate = false)`: simulated and
  reported, never in the cost or the gradient) with
  [`CosampleFlaggedRecords`](@ref) (`"cosample"`). Records of another tracer
  are dropped when `tracer_filter` names one.
- **Cell averaging.** With `cell_averaging = CellAverage(...)`, assimilated
  column records of one cell, step sampling and overpass become one
  inverse-variance-weighted observation ([`CellAverage`](@ref),
  [`CSCellAverageObjective`](@ref)), appended after the records, whose own
  observations become co-samples.
- The observation `sigma` is ``\\sqrt{\\sigma_\\mathrm{value}^2 +
  \\sigma_\\mathrm{model}^2 + \\sigma_\\mathrm{prof}^2}`` (exactly
  `value_sigma` without a model error), where ``\\sigma_\\mathrm{prof}^2 =
  \\sum_i (h_i a_i \\sigma_i)^2`` is the model-error term of an
  averaging-kernel column with a profile uncertainty ``\\sigma_i``
  (`Observations.retrieval_model_error_variance`; fixed by the record, so the cost
  stays quadratic); the observation carries the record `id`.

`set.time_origin` is documentation only.
"""
function bind_to_mesh(set::CSObservationSet,
                      mesh::CubedSphereMesh,
                      t_start::Dates.DateTime,
                      dt::Real;
                      nsteps::Union{Nothing, Integer} = nothing,
                      tracer_filter::Union{Nothing, AbstractString} = nothing,
                      out_of_range_policy = RejectOutOfRange(),
                      placement::VerticalPlacementSpec = VerticalPlacementSpec(),
                      terrain::ObservationTerrain = ObservationTerrain(),
                      max_surface_pressure_mismatch_pa::Real = Inf,
                      flagged_records = DropFlaggedRecords(),
                      cell_averaging::AbstractCellAveraging = NoCellAveraging())
    policy = _parse_out_of_range_policy(out_of_range_policy)
    max_surface_pressure_mismatch_pa > 0 || throw(ArgumentError(
        "max_surface_pressure_mismatch_pa must be a positive pressure (Pa), or Inf for no limit; got " *
        "$(max_surface_pressure_mismatch_pa)"))
    cell_averaging isa CellAverage && isfinite(max_surface_pressure_mismatch_pa) && throw(ArgumentError(
        "cell averaging (CellAverage) needs max_surface_pressure_mismatch_pa = Inf: the surface-pressure screen " *
        "applies to records, and an average would assimilate a screened record"))
    observations, records = _bind_to_mesh(set, mesh, t_start, dt, policy; nsteps, tracer_filter, placement,
                                          terrain, max_surface_pressure_mismatch_pa =
                                              Float64(max_surface_pressure_mismatch_pa),
                                          flagged = _parse_flagged_records(flagged_records))
    return _average_in_cells(cell_averaging, observations, records)
end

# The observations and, for each, the record it was bound from.
function _bind_to_mesh(set::CSObservationSet,
                       mesh::CubedSphereMesh,
                       t_start::Dates.DateTime,
                       dt::Real,
                       policy::AbstractOutOfRangePolicy;
                       nsteps::Union{Nothing, Integer},
                       tracer_filter::Union{Nothing, AbstractString},
                       placement::VerticalPlacementSpec,
                       terrain::ObservationTerrain,
                       max_surface_pressure_mismatch_pa::Float64,
                       flagged::AbstractFlaggedRecordPolicy)
    dt_f = Float64(dt)
    isfinite(dt_f) && dt_f > 0 || throw(ArgumentError("dt must be positive, got $dt"))
    if nsteps !== nothing
        nsteps >= 1 || throw(ArgumentError(
            "nsteps must be >= 1 when provided, got $nsteps"))
    end
    check_terrain(placement, terrain, _INVERSION_PLACEMENT_LABEL)
    check_surface_height_layout(terrain.model_surface, mesh)
    nsteps_max = nsteps === nothing ? typemax(Int) : Int(nsteps)
    tracer_match = tracer_filter === nothing ? nothing : String(tracer_filter)

    _check_unique_ids(set.records, "bind_to_mesh")
    locator = cell_locator(mesh)
    out = CSObservation[]
    bound = CSObservationRecord[]
    sizehint!(out, length(set))
    counters = ObservationCounters()
    examples = String[]
    placed_requests = SoundingRequest[]
    placements = VerticalPlacement[]
    n_flagged = n_cosampled = n_skipped = n_clamped = 0

    for record in set.records
        tracer_match === nothing || record.tracer == tracer_match || continue
        assimilate = record.flag == 0
        assimilate || _binds_flagged(flagged) || (n_flagged += 1; continue)
        sampling, outside, t = _record_sampling(record, t_start, dt_f, nsteps, nsteps_max, policy)
        sampling === nothing && (n_skipped += 1; continue)
        outside && (n_clamped += 1)
        assimilate || (n_cosampled += 1)
        cell = locate(locator, record.lon, record.lat)
        request = _sounding_request(record, t)
        objective = _bind_objective(record.kind, record, request, cell, placement, terrain,
                                    max_surface_pressure_mismatch_pa)
        if objective === nothing
            count_unplaced!(counters, request)
            length(examples) < 5 && push!(examples, unplaced_example(request))
            continue
        end
        if objective isa CSPlacedPointObjective
            push!(placed_requests, request)
            push!(placements, objective.placement)
        end
        push!(out, CSObservation(sampling, objective, record.value, _total_sigma(record); id = record.id,
                                 assimilate))
        push!(bound, record)
    end
    handle_unplaced(placement, counters, examples; label = _INVERSION_PLACEMENT_LABEL,
                     context = "bind_to_mesh")
    warn_below_ground(placed_requests, placements, SiteRequest[], VerticalPlacement[]; context = "bind_to_mesh")
    n_flagged > 0 && @info "bind_to_mesh: $(n_flagged) records with a nonzero flag are not assimilated"
    n_cosampled > 0 && @info "bind_to_mesh: $(n_cosampled) records with a nonzero flag are co-sampled " *
                             "(flagged_records = \"cosample\"): simulated and reported, not assimilated"
    n_skipped > 0 && @info "bind_to_mesh: $(n_skipped) records outside the run are skipped (SkipOutOfRange)"
    n_clamped > 0 && @info "bind_to_mesh: $(n_clamped) records outside the run are clamped (ClampOutOfRange): " *
                           "a record whose window overlaps the run samples that part of the window, the " *
                           "others the first or last step"
    return out, bound
end

# The time sampling of a record (`nothing` for one the out-of-range policy
# skips), whether its time lies outside the run, and its time (seconds after
# `t_start`).
function _record_sampling(record::CSObservationRecord, t_start::Dates.DateTime, dt::Float64,
                          nsteps::Union{Nothing, Integer}, nsteps_max::Int, policy::AbstractOutOfRangePolicy)
    nsteps === nothing && record.time_window_s > 0 && throw(ArgumentError(
        "observation id $(record.id) has a time window (time_window_s = $(record.time_window_s)); " *
        "bind_to_mesh needs `nsteps` to clip windows to the run"))
    t = _seconds_after(record.date_components, t_start)
    k = floor(Int, t / dt) + 1
    1 <= k <= nsteps_max && return (_window_sampling(t, record.time_window_s, k, dt, nsteps_max), false, t)
    return (_out_of_range_sampling(policy, record, t, k, dt, nsteps, nsteps_max), true, t)
end

# ---------------------------------------------------------------------------
# Cell averaging
# ---------------------------------------------------------------------------

# Groups of two or more indices whose `keys` are equal and whose `times` lie
# within `tolerance` of the group's earliest time; groups in order of their
# first index, members in time order.
function _time_clusters(keys::AbstractVector, times::AbstractVector{Float64}, tolerance::Float64)
    by_key = Dict{Any, Vector{Int}}()
    order = Any[]
    for (idx, key) in enumerate(keys)
        key === nothing && continue
        haskey(by_key, key) || push!(order, key)
        push!(get!(by_key, key, Int[]), idx)
    end
    groups = Vector{Int}[]
    for key in order
        members = sort(by_key[key]; by = idx -> (times[idx], idx))
        start = 1
        for n in 2:(length(members) + 1)
            n <= length(members) && times[members[n]] - times[members[start]] <= tolerance && continue
            n - start >= 2 && push!(groups, members[start:(n - 1)])
            start = n
        end
    end
    sort!(groups; by = first)
    return groups
end

# The grouping key of a bound observation: tracer, cell and time sampling of an
# assimilated column observation, `nothing` for any other.
_cell_key(obs::CSObservation, record::CSObservationRecord) =
    obs.assimilate && record.kind isa ColumnObservationKind &&
        obs.objective isa Union{CSRetrievalColumnObjective, CSColumnMeanObjective} ?
    (record.tracer, obs.objective.panel, obs.objective.i, obs.objective.j, obs.sampling.steps,
     obs.sampling.weights) : nothing

_average_in_cells(::NoCellAveraging, observations, records) = observations

function _average_in_cells(averaging::CellAverage, observations::Vector{CSObservation},
                           records::Vector{CSObservationRecord})
    times = Float64[Dates.value(_record_datetime(r.date_components)) / 1000 for r in records]
    groups = _time_clusters([_cell_key(o, r) for (o, r) in zip(observations, records)], times,
                            averaging.time_tolerance_s)
    out = copy(observations)
    for group in groups
        for idx in group
            o = observations[idx]
            out[idx] = CSObservation(o.sampling, o.objective, o.value, o.sigma; id = o.id, assimilate = false)
        end
        push!(out, _cell_average_observation(averaging, observations[group], records[group]))
    end
    isempty(groups) || @info "bind_to_mesh: $(sum(length, groups)) column records are averaged into " *
                             "$(length(groups)) cell averages (CellAverage, time_tolerance_s = " *
                             "$(averaging.time_tolerance_s)); the records are co-sampled"
    return out
end

"""
    _cell_average_observation(averaging, members, records) -> CSObservation

The averaged observation of a group of bound column observations of one cell
and time sampling ([`CellAverage`](@ref)): weights ``w_i = 1/σ_i^2`` from the
records' `value_sigma`, value and objective the normalized weighted means,
``σ^2 = σ^2_\\mathrm{meas} + \\sum_i \\tilde w_i σ^2_{\\mathrm{mod},i}`` with
``σ^2_\\mathrm{meas} = \\sum_{ij} \\tilde w_i \\tilde w_j ρ_{ij} σ_i σ_j``. Its
id is the id of the group's earliest record.
"""
function _cell_average_observation(averaging::CellAverage, members::AbstractVector{<:CSObservation},
                                   records::AbstractVector{CSObservationRecord})
    w = [1 / r.value_sigma^2 for r in records]
    weights = w ./ sum(w)
    σ = [r.value_sigma for r in records]
    measurement = 0.0
    for i in eachindex(σ), j in eachindex(σ)
        measurement += weights[i] * weights[j] * (i == j ? 1.0 : averaging.correlation) * σ[i] * σ[j]
    end
    model = sum(weights[i] * (records[i].model_sigma^2 + _profile_model_sigma(records[i])^2) for i in eachindex(σ))
    first_member = first(members).objective
    objective = CSCellAverageObjective(first_member.panel, first_member.i, first_member.j,
                                       [m.objective for m in members], weights, [r.id for r in records];
                                       measurement_sigma = sqrt(measurement), model_sigma = sqrt(model),
                                       correlation = averaging.correlation)
    value = sum(weights[i] * records[i].value for i in eachindex(σ))
    return CSObservation(first(members).sampling, objective, value, sqrt(measurement + model);
                         id = first(records).id)
end

"""
    inflate_cell_model_error(set::CSObservationSet, mesh, t_start, dt; nsteps = nothing,
                             time_tolerance_s = 60.0, tracer_filter = nothing,
                             out_of_range_policy = RejectOutOfRange()) -> CSObservationSet

The alternative to [`CellAverage`](@ref) that keeps every record: the
assimilated column records grouped as `CellAverage` groups them (same tracer,
model cell and step sampling as `bind_to_mesh` would bind them, within
`time_tolerance_s` of the group's earliest record) get their model-error
variance multiplied by the number ``k`` of records in their group,
`model_sigma → √k · model_sigma`, i.e. `model_sigma²` times ``k``; the
profile-error term ``\\sum_i (h_i a_i σ_i)^2`` of an averaging-kernel column
(from its retrieval block) is not inflated. The model error of
records that sample the same model column is correlated, so ``k`` records
should not reduce it ``k``-fold; inflating each by ``k`` gives the group the
weight of one record's model error. Bind the returned set with the same run
grid; departures built from it report the inflated `model_sigma`. Every
other record is returned unchanged.
"""
function inflate_cell_model_error(set::CSObservationSet, mesh::CubedSphereMesh, t_start::Dates.DateTime, dt::Real;
                                  nsteps::Union{Nothing, Integer} = nothing, time_tolerance_s::Real = 60.0,
                                  tracer_filter::Union{Nothing, AbstractString} = nothing,
                                  out_of_range_policy = RejectOutOfRange())
    isfinite(time_tolerance_s) && time_tolerance_s >= 0 || throw(ArgumentError(
        "time_tolerance_s must be finite and >= 0; got $(time_tolerance_s)"))
    policy = _parse_out_of_range_policy(out_of_range_policy)
    dt_f = Float64(dt)
    isfinite(dt_f) && dt_f > 0 || throw(ArgumentError("dt must be positive, got $dt"))
    nsteps_max = nsteps === nothing ? typemax(Int) : Int(nsteps)
    locator = cell_locator(mesh)
    keys = map(set.records) do r
        tracer_filter === nothing || r.tracer == tracer_filter || return nothing
        r.flag == 0 && r.kind isa ColumnObservationKind || return nothing
        sampling, _, _ = _record_sampling(r, t_start, dt_f, nsteps, nsteps_max, policy)
        sampling === nothing && return nothing
        cell = locate(locator, r.lon, r.lat)
        (r.tracer, cell.panel, cell.i, cell.j, sampling.steps, sampling.weights)
    end
    times = Float64[Dates.value(_record_datetime(r.date_components)) / 1000 for r in set.records]
    records = copy(set.records)
    for group in _time_clusters(keys, times, Float64(time_tolerance_s)), idx in group
        records[idx] = _with_model_sigma(records[idx], sqrt(length(group)) * records[idx].model_sigma)
    end
    return CSObservationSet(records, set.time_origin)
end

_with_model_sigma(r::CSObservationRecord, model_sigma::Float64) =
    CSObservationRecord(r.id, r.date_components, r.lat, r.lon, r.kind, r.elevation_m, r.intake_height_m,
                        r.altitude_m, r.pressure_pa, r.time_window_s, r.flag, r.value, r.value_sigma, model_sigma,
                        r.instrument_type, r.tracer, r.retrieval)

# The record as a forward-sampler point request (source 1), so it is placed
# by exactly the forward rule.
_sounding_request(r::CSObservationRecord, time_seconds::Float64) =
    SoundingRequest(string(r.id), time_seconds, r.lon, r.lat, 1, r.elevation_m, r.intake_height_m,
                    r.altitude_m, r.pressure_pa)

# Point records: placed in the column (nothing when unplaced). Column records:
# the averaging-kernel column of their cell when they carry a retrieval block,
# else its column mean.
function _bind_objective(::PointObservationKind, record::CSObservationRecord, request::SoundingRequest,
                         cell::CellLocation, spec::VerticalPlacementSpec, terrain::ObservationTerrain,
                         max_surface_pressure_mismatch_pa::Float64)
    vp = resolve_placement(spec, UnplacedWithoutHeight(), request, cell, terrain)
    vp === nothing && return nothing
    return CSPlacedPointObjective(cell.panel, cell.i, cell.j, cell.area, vp, spec.pressure_basis)
end

function _bind_objective(::ColumnObservationKind, record::CSObservationRecord, request::SoundingRequest,
                         cell::CellLocation, spec::VerticalPlacementSpec, terrain::ObservationTerrain,
                         max_surface_pressure_mismatch_pa::Float64)
    record.retrieval === nothing && return CSColumnMeanObjective(cell.panel, cell.i, cell.j)
    return CSRetrievalColumnObjective(cell.panel, cell.i, cell.j, cell.area, record.retrieval,
                                      spec.pressure_basis, max_surface_pressure_mismatch_pa, spec.unplaced)
end

# ---------------------------------------------------------------------------
# Time sampling
# ---------------------------------------------------------------------------

# Seconds of the record time after `t_start` (millisecond date arithmetic).
_seconds_after(dc::NTuple{6, Int16}, t_start::Dates.DateTime) =
    Dates.value(_record_datetime(dc) - t_start) / 1000

_record_datetime(dc::NTuple{6, Int16}) =
    Dates.DateTime(Int(dc[1]), Int(dc[2]), Int(dc[3]), Int(dc[4]), Int(dc[5]), Int(dc[6]))

"""
    _window_sampling(t, window, k, dt, nsteps_max) -> CSTimeSampling

Time sampling of a record at `t` seconds after the run start, in step `k` of
the run: step `k` alone without a window; with a window `W > 0`, the steps
overlapping it (`_window_overlap`), or step `k` when rounding leaves none.
"""
function _window_sampling(t::Float64, window::Float64, k::Int, dt::Float64, nsteps_max::Int)
    window == 0 && return CSTimeSampling(k)
    return something(_window_overlap(t, window, dt, nsteps_max), CSTimeSampling(k))
end

"""
    _window_overlap(t, window, dt, nsteps) -> Union{CSTimeSampling, Nothing}

Every step `s` whose interval `[(s−1)dt, s·dt)` overlaps the window
`[t − W/2, t + W/2]` clipped to the run `[0, nsteps·dt]`, weighted by its
overlap and renormalised; `nothing` when the window does not reach into the
run. At most `nsteps` steps, whatever the window.
"""
function _window_overlap(t::Float64, window::Float64, dt::Float64, nsteps::Int)
    window > 0 || return nothing
    a = max(t - window / 2, 0.0)
    b = min(t + window / 2, nsteps * dt)
    b > a || return nothing
    steps = Int[]
    overlaps = Float64[]
    for s in max(floor(Int, a / dt) + 1, 1):min(ceil(Int, b / dt), nsteps)
        overlap = min(b, s * dt) - max(a, (s - 1) * dt)
        overlap > 0 || continue
        push!(steps, s)
        push!(overlaps, overlap)
    end
    isempty(steps) && return nothing
    return CSTimeSampling(steps, overlaps ./ sum(overlaps))
end

# Records whose time lies outside the run: the policy's sampling, or `nothing`
# to drop the record. A clamped record samples the part of its window inside
# the run (truncated and renormalised, as a window at the run's end), and a
# record without a window, or whose window misses the run, the nearest step.
function _out_of_range_sampling(::RejectOutOfRange, record, t::Float64, k::Int, dt::Float64,
                                nsteps::Union{Nothing, Integer}, nsteps_max::Int)
    throw(ArgumentError(
        "observation id $(record.id) at " *
        "$(_date_components_string(record.date_components)) " *
        "maps to step $k, outside [1, " *
        "$(nsteps === nothing ? "Inf" : string(nsteps_max))]; pass " *
        "`out_of_range_policy = SkipOutOfRange()` or `ClampOutOfRange()` to handle"))
end

_out_of_range_sampling(::SkipOutOfRange, record, t::Float64, k::Int, dt::Float64,
                       nsteps::Union{Nothing, Integer}, nsteps_max::Int) = nothing

_out_of_range_sampling(::ClampOutOfRange, record, t::Float64, k::Int, dt::Float64,
                       nsteps::Union{Nothing, Integer}, nsteps_max::Int) =
    _clamped_sampling(t, record.time_window_s, k, dt, nsteps_max)

_clamped_sampling(t::Float64, window::Float64, k::Int, dt::Float64, nsteps_max::Int) =
    something(_window_overlap(t, window, dt, nsteps_max), CSTimeSampling(clamp(k, 1, nsteps_max)))

_date_components_string(dc::NTuple{6, Int16}) =
    string(lpad(Int(dc[1]), 4, '0'), "-",
           lpad(Int(dc[2]), 2, '0'), "-",
           lpad(Int(dc[3]), 2, '0'), "T",
           lpad(Int(dc[4]), 2, '0'), ":",
           lpad(Int(dc[5]), 2, '0'), ":",
           lpad(Int(dc[6]), 2, '0'))

# ---------------------------------------------------------------------------
# Geographic mapping
# ---------------------------------------------------------------------------

# Observations are bound to the cell that contains them, with the locator the
# forward observation sampler uses (`Observations.cell_locator`): the inverse
# gnomonic map of the mesh; a point on a face belongs to the cell to its east or
# north (`locate`). The longitude is passed as given, as the sampler passes it:
# `locate` takes any convention, while reducing it first (`mod(lon, 360)`)
# rounds a longitude within ~1e-14 below 0 to 360 and can change the cell.
@inline function _locate_cs_cell(lat_deg::Float64, lon_deg::Float64, locator)
    cell = locate(locator, lon_deg, lat_deg)
    return (cell.panel, cell.i, cell.j)
end
