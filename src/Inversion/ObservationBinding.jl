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
#     observation sampler uses (`Output.cell_locator`/`locate`), so an
#     observation is sampled and inverted in the same cell.
#
#   * Vertical. A point record becomes an `Output.SoundingRequest` and is
#     placed by the shared rule of the forward sampler
#     (`Output.resolve_placement` with the run's `VerticalPlacementSpec` and
#     `ObservationTerrain`) into a `CSPlacedPointObjective`; its layer is chosen
#     at every sampled state by `Output.sample_column_layer!`. A column record
#     with a retrieval block becomes a `CSRetrievalColumnObjective` (its
#     averaging kernel on the model column stretched onto the retrieval
#     column, built at every sampled state), one without a block a
#     `CSColumnMeanObjective`.
# ---------------------------------------------------------------------------

import Dates

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

function _parse_out_of_range_policy(raw::AbstractString)
    raw == "reject" && return RejectOutOfRange()
    raw == "skip" && return SkipOutOfRange()
    raw == "clamp" && return ClampOutOfRange()
    throw(ArgumentError(
        "out_of_range_policy must be \"reject\", \"skip\", or \"clamp\"; " *
        "got $(repr(raw))."))
end

_parse_out_of_range_policy(raw) = throw(ArgumentError(
    "out_of_range_policy must be a concrete AbstractOutOfRangePolicy or one " *
    "of the strings \"reject\", \"skip\", or \"clamp\"; got $(repr(raw)). " *
    "Symbol policies are not supported; use RejectOutOfRange(), " *
    "SkipOutOfRange(), or ClampOutOfRange()."))

"""
    bind_to_mesh(set::CSObservationSet, mesh::CubedSphereMesh, t_start::Dates.DateTime, dt::Real;
                 nsteps = nothing, tracer_filter = nothing, out_of_range_policy = RejectOutOfRange(),
                 placement = VerticalPlacementSpec(), terrain = ObservationTerrain(),
                 max_surface_pressure_mismatch_pa = Inf)
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
  (`Output.locate`, the forward observation sampler's locator).
- **Point records** ([`PointObservationKind`](@ref)) become
  [`CSPlacedPointObjective`](@ref)s, placed by the forward sampler's rule
  (`Output.resolve_placement`) with `placement` (an
  `Output.VerticalPlacementSpec`: the ordered `vertical_coordinates`, height
  above ground and/or pressure; the pressure basis; the altitude option; the
  unplaced policy) and the static ground fields of `terrain` (an
  `Output.ObservationTerrain`: model surface height, point DEM). A point
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
- Records with a nonzero `flag` are not assimilated (dropped); records of
  another tracer are dropped when `tracer_filter` names one.
- The observation `sigma` is ``\\sqrt{\\sigma_\\mathrm{value}^2 +
  \\sigma_\\mathrm{model}^2 + \\sigma_\\mathrm{prof}^2}`` (exactly
  `value_sigma` without a model error), where ``\\sigma_\\mathrm{prof}^2 =
  \\sum_i (h_i a_i \\sigma_i)^2`` is the model-error term of an
  averaging-kernel column with a profile uncertainty ``\\sigma_i``
  (`Output.retrieval_model_error_variance`; fixed by the record, so the cost
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
                      max_surface_pressure_mismatch_pa::Real = Inf)
    policy = _parse_out_of_range_policy(out_of_range_policy)
    max_surface_pressure_mismatch_pa > 0 || throw(ArgumentError(
        "max_surface_pressure_mismatch_pa must be a positive pressure (Pa), or Inf for no limit; got " *
        "$(max_surface_pressure_mismatch_pa)"))
    return _bind_to_mesh(set, mesh, t_start, dt, policy; nsteps, tracer_filter, placement, terrain,
                         max_surface_pressure_mismatch_pa = Float64(max_surface_pressure_mismatch_pa))
end

function _bind_to_mesh(set::CSObservationSet,
                       mesh::CubedSphereMesh,
                       t_start::Dates.DateTime,
                       dt::Real,
                       policy::AbstractOutOfRangePolicy;
                       nsteps::Union{Nothing, Integer},
                       tracer_filter::Union{Nothing, AbstractString},
                       placement::VerticalPlacementSpec,
                       terrain::ObservationTerrain,
                       max_surface_pressure_mismatch_pa::Float64)
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
    sizehint!(out, length(set))
    counters = ObservationCounters()
    examples = String[]
    placed_requests = SoundingRequest[]
    placements = VerticalPlacement[]
    n_flagged = n_skipped = n_clamped = 0

    for record in set.records
        tracer_match === nothing || record.tracer == tracer_match || continue
        record.flag == 0 || (n_flagged += 1; continue)
        nsteps === nothing && record.time_window_s > 0 && throw(ArgumentError(
            "observation id $(record.id) has a time window (time_window_s = $(record.time_window_s)); " *
            "bind_to_mesh needs `nsteps` to clip windows to the run"))
        t = _seconds_after(record.date_components, t_start)
        k = floor(Int, t / dt_f) + 1
        sampling = if 1 <= k <= nsteps_max
            _window_sampling(t, record.time_window_s, k, dt_f, nsteps_max)
        else
            outside = _out_of_range_sampling(policy, record, t, k, dt_f, nsteps, nsteps_max)
            outside === nothing ? (n_skipped += 1; continue) : (n_clamped += 1; outside)
        end
        cell = locate(locator, record.lon, record.lat)
        request = _sounding_request(record, t)
        objective = _bind_objective(record.kind, record, request, cell, placement, terrain,
                                    max_surface_pressure_mismatch_pa)
        if objective === nothing
            _count_unplaced!(counters, request)
            length(examples) < 5 && push!(examples, _unplaced_example(request))
            continue
        end
        if objective isa CSPlacedPointObjective
            push!(placed_requests, request)
            push!(placements, objective.placement)
        end
        push!(out, CSObservation(sampling, objective, record.value, _total_sigma(record); id = record.id))
    end
    _handle_unplaced(placement, counters, examples; label = _INVERSION_PLACEMENT_LABEL,
                     context = "bind_to_mesh")
    _warn_below_ground(placed_requests, placements, SiteRequest[], VerticalPlacement[]; context = "bind_to_mesh")
    n_flagged > 0 && @info "bind_to_mesh: $(n_flagged) records with a nonzero flag are not assimilated"
    n_skipped > 0 && @info "bind_to_mesh: $(n_skipped) records outside the run are skipped (SkipOutOfRange)"
    n_clamped > 0 && @info "bind_to_mesh: $(n_clamped) records outside the run are clamped (ClampOutOfRange): " *
                           "a record whose window overlaps the run samples that part of the window, the " *
                           "others the first or last step"
    return out
end

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

@inline function _step_index_from_date(dc::NTuple{6, Int16},
                                       t_start::Dates.DateTime,
                                       dt_f::Real)
    return floor(Int, _seconds_after(dc, t_start) / dt_f) + 1
end

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
# forward observation sampler uses (`Output.cell_locator`): the inverse
# gnomonic map of the mesh; a point on a face belongs to the cell to its east or
# north (`locate`). The longitude is passed as given, as the sampler passes it:
# `locate` takes any convention, while reducing it first (`mod(lon, 360)`)
# rounds a longitude within ~1e-14 below 0 to 360 and can change the cell.
@inline function _locate_cs_cell(lat_deg::Float64, lon_deg::Float64, locator)
    cell = locate(locator, lon_deg, lat_deg)
    return (cell.panel, cell.i, cell.j)
end
