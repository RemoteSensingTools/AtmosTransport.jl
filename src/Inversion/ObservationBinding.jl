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
#     becomes a `CSColumnMeanObjective`.
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
Move observations outside the valid model-step range to its nearest end: the
record is sampled once, at the first or last step, whatever its time window.
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
                 placement = VerticalPlacementSpec(), terrain = ObservationTerrain())
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
  [`ClampOutOfRange`](@ref) (`"clamp"`) samples it once at step `1` or
  `nsteps`.
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
- **Column records** ([`ColumnObservationKind`](@ref)) become
  [`CSColumnMeanObjective`](@ref)s (averaging kernels are not applied yet).
- Records with a nonzero `flag` are not assimilated (dropped); records of
  another tracer are dropped when `tracer_filter` names one.
- The observation `sigma` is ``\\sqrt{\\sigma_\\mathrm{value}^2 +
  \\sigma_\\mathrm{model}^2}`` (exactly `value_sigma` without a model
  error); the observation carries the record `id`.

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
                      terrain::ObservationTerrain = ObservationTerrain())
    policy = _parse_out_of_range_policy(out_of_range_policy)
    return _bind_to_mesh(set, mesh, t_start, dt, policy; nsteps, tracer_filter, placement, terrain)
end

function _bind_to_mesh(set::CSObservationSet,
                       mesh::CubedSphereMesh,
                       t_start::Dates.DateTime,
                       dt::Real,
                       policy::AbstractOutOfRangePolicy;
                       nsteps::Union{Nothing, Integer},
                       tracer_filter::Union{Nothing, AbstractString},
                       placement::VerticalPlacementSpec,
                       terrain::ObservationTerrain)
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

    locator = cell_locator(mesh)
    out = CSObservation[]
    sizehint!(out, length(set))
    counters = ObservationCounters()
    examples = String[]
    placed_requests = SoundingRequest[]
    placements = VerticalPlacement[]
    n_flagged = 0

    for record in set.records
        tracer_match === nothing || record.tracer == tracer_match || continue
        record.flag == 0 || (n_flagged += 1; continue)
        sampling = _record_time_sampling(policy, record, t_start, dt_f, nsteps, nsteps_max)
        sampling === nothing && continue
        cell = locate(locator, mod(record.lon, 360.0), record.lat)
        request = _sounding_request(record, _seconds_after(record.date_components, t_start))
        objective = _bind_objective(record.kind, request, cell, placement, terrain)
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
    return out
end

# The record as a forward-sampler point request (source 1), so it is placed
# by exactly the forward rule.
_sounding_request(r::CSObservationRecord, time_seconds::Float64) =
    SoundingRequest(string(r.id), time_seconds, r.lon, r.lat, 1, r.elevation_m, r.intake_height_m,
                    r.altitude_m, r.pressure_pa)

# Point records: placed in the column (nothing when unplaced). Column records:
# the column mean of their cell.
function _bind_objective(::PointObservationKind, request::SoundingRequest, cell::CellLocation,
                         spec::VerticalPlacementSpec, terrain::ObservationTerrain)
    vp = resolve_placement(spec, UnplacedWithoutHeight(), request, cell, terrain)
    vp === nothing && return nothing
    return CSPlacedPointObjective(cell.panel, cell.i, cell.j, cell.area, vp, spec.pressure_basis)
end

_bind_objective(::ColumnObservationKind, request::SoundingRequest, cell::CellLocation,
                spec::VerticalPlacementSpec, terrain::ObservationTerrain) =
    CSColumnMeanObjective(cell.panel, cell.i, cell.j)

# ---------------------------------------------------------------------------
# Time sampling
# ---------------------------------------------------------------------------

# Seconds of the record time after `t_start` (millisecond date arithmetic).
_seconds_after(dc::NTuple{6, Int16}, t_start::Dates.DateTime) =
    Dates.value(_record_datetime(dc) - t_start) / 1000

_record_datetime(dc::NTuple{6, Int16}) =
    Dates.DateTime(Int(dc[1]), Int(dc[2]), Int(dc[3]), Int(dc[4]), Int(dc[5]), Int(dc[6]))

# The record's `CSTimeSampling` on the run grid, or the out-of-range policy's
# answer (`nothing` to drop the record).
function _record_time_sampling(policy::AbstractOutOfRangePolicy, record::CSObservationRecord,
                               t_start::Dates.DateTime, dt_f::Float64,
                               nsteps::Union{Nothing, Integer}, nsteps_max::Int)
    k = _step_index_from_date(record.date_components, t_start, dt_f)
    if !(1 <= k <= nsteps_max)
        adjusted = _out_of_range_step(policy, record, k, nsteps, nsteps_max)
        return adjusted === nothing ? nothing : CSTimeSampling(adjusted)
    end
    return _window_sampling(_seconds_after(record.date_components, t_start), record.time_window_s,
                            k, dt_f, nsteps_max)
end

"""
    _window_sampling(t, window, k, dt, nsteps_max) -> CSTimeSampling

Time sampling of a record at `t` seconds after the run start, in step `k`:
step `k` alone without a window; with a window `W > 0`, every step `s` whose
interval `[(s−1)dt, s·dt)` overlaps `[t − W/2, t + W/2]` clipped to the run
`[0, nsteps_max·dt]`, weighted by its overlap and renormalised.
"""
function _window_sampling(t::Float64, window::Float64, k::Int, dt::Float64, nsteps_max::Int)
    window == 0 && return CSTimeSampling(k)
    run_end = nsteps_max == typemax(Int) ? Inf : nsteps_max * dt
    a = max(t - window / 2, 0.0)
    b = min(t + window / 2, run_end)
    first_step = max(floor(Int, a / dt) + 1, 1)
    last_step = min(ceil(Int, b / dt), nsteps_max)
    steps = Int[]
    overlaps = Float64[]
    for s in first_step:last_step
        overlap = min(b, s * dt) - max(a, (s - 1) * dt)
        overlap > 0 || continue
        push!(steps, s)
        push!(overlaps, overlap)
    end
    # A window narrower than rounding around a step boundary keeps its step.
    isempty(steps) && return CSTimeSampling(k)
    return CSTimeSampling(steps, overlaps ./ sum(overlaps))
end

function _out_of_range_step(::RejectOutOfRange, record, k::Int,
                            nsteps::Union{Nothing, Integer}, nsteps_max::Int)
    throw(ArgumentError(
        "observation id $(record.id) at " *
        "$(_date_components_string(record.date_components)) " *
        "maps to step $k, outside [1, " *
        "$(nsteps === nothing ? "Inf" : string(nsteps_max))]; pass " *
        "`out_of_range_policy = SkipOutOfRange()` or `ClampOutOfRange()` to handle"))
end

@inline _out_of_range_step(::SkipOutOfRange, record, k::Int,
                           nsteps::Union{Nothing, Integer}, nsteps_max::Int) = nothing

@inline _out_of_range_step(::ClampOutOfRange, record, k::Int,
                           nsteps::Union{Nothing, Integer}, nsteps_max::Int) =
    clamp(k, 1, nsteps_max)

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
# north (`locate`). Longitudes of any wrap are reduced to [0, 360) first.
@inline function _locate_cs_cell(lat_deg::Float64, lon_deg::Float64, locator)
    cell = locate(locator, mod(lon_deg, 360.0), lat_deg)
    return (cell.panel, cell.i, cell.j)
end
