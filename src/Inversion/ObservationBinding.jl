# ---------------------------------------------------------------------------
# Bridge on-disk CSObservationSet to in-memory
# Vector{CSObservation{CSColumnMeanObjective, Float64}} consumed by the
# CS 4D-Var surface-flux path (`cs_surface_flux_4dvar`,
# `cs_surface_flux_jacobian`).
#
# Two mappings:
#
#   * Time. `date_components -> DateTime -> step index` via the left-
#     closed half-open interval `[t_start + (k-1)*dt, t_start + k*dt)`.
#     Observation exactly at `t_start` lands on step 1; an observation
#     at `t_start + nsteps*dt` (one past the end) lands on step
#     `nsteps + 1` and is rejected by the `:reject` policy.
#
#   * Geography. `(lat, lon) -> (panel, i, j)` is the cell that contains
#     the point, from the inverse gnomonic map of the mesh: the locator the
#     forward observation sampler uses (`Output.cell_locator`/`locate`), so
#     an observation is sampled and inverted in the same cell. Records store
#     Float32 coordinates; a point within Float32 rounding of a cell edge can
#     fall on either side of it.
#
# Altitude is dropped: v1 always projects to a column-mean objective.
# ---------------------------------------------------------------------------

import Dates

const _OUT_OF_RANGE_POLICIES = (:reject, :skip, :clamp)

"""
    bind_to_mesh(set::CSObservationSet,
                 mesh::CubedSphereMesh,
                 t_start::Dates.DateTime,
                 dt::Real;
                 nsteps::Union{Nothing, Integer} = nothing,
                 tracer_filter::Union{Nothing, AbstractString} = nothing,
                 out_of_range_policy::Symbol = :reject)
        -> Vector{CSObservation{CSColumnMeanObjective, Float64}}

Map each `CSObservationRecord` in `set` to a `CSObservation` tied to a
model step and a [`CSColumnMeanObjective`](@ref) on `mesh`, ready to
feed `cs_surface_flux_jacobian` / `cs_surface_flux_4dvar`.

- `t_start` is the absolute model start time (UTC).
- `dt` is the model step length in seconds; the step grid is
  `t_start, t_start + dt, ..., t_start + nsteps*dt`.
- `nsteps`, if given, bounds the valid step range to `1:nsteps`.
  When `nothing`, only the lower bound (`step >= 1`) is enforced.
- `tracer_filter` keeps only records whose `tracer` field matches
  exactly (e.g. `"CO2"`). `nothing` keeps every record.
- `out_of_range_policy` chooses what happens when an observation's
  date falls outside the valid step range:
    - `:reject` (default) — throw an `ArgumentError`.
    - `:skip` — drop the record silently.
    - `:clamp` — snap to the nearest valid step (`1` or `nsteps`).

Altitude (`record.alt`) is ignored: every observation becomes a
column-mean objective. The on-disk `set.time_origin` is not consulted
during the computation; it is documentation only.
"""
function bind_to_mesh(set::CSObservationSet,
                      mesh::CubedSphereMesh,
                      t_start::Dates.DateTime,
                      dt::Real;
                      nsteps::Union{Nothing, Integer} = nothing,
                      tracer_filter::Union{Nothing, AbstractString} = nothing,
                      out_of_range_policy::Symbol = :reject)
    out_of_range_policy in _OUT_OF_RANGE_POLICIES || throw(ArgumentError(
        "out_of_range_policy must be one of $(_OUT_OF_RANGE_POLICIES); " *
        "got $(repr(out_of_range_policy))"))
    dt_f = float(dt)
    dt_f > 0 || throw(ArgumentError("dt must be positive, got $dt"))
    if nsteps !== nothing
        nsteps >= 1 || throw(ArgumentError(
            "nsteps must be >= 1 when provided, got $nsteps"))
    end
    nsteps_max = nsteps === nothing ? typemax(Int) : Int(nsteps)
    tracer_match = tracer_filter === nothing ? nothing : String(tracer_filter)

    locator = cell_locator(mesh)
    out = Vector{CSObservation{CSColumnMeanObjective, Float64}}()
    sizehint!(out, length(set))

    @inbounds for record in set.records
        tracer_match === nothing || record.tracer == tracer_match || continue
        _validate_bind_record(record)

        k = _step_index_from_date(record.date_components, t_start, dt_f)
        in_range = 1 <= k <= nsteps_max
        if !in_range
            if out_of_range_policy === :reject
                throw(ArgumentError(
                    "observation id $(record.id) at " *
                    "$(_date_components_string(record.date_components)) " *
                    "maps to step $k, outside [1, " *
                    "$(nsteps === nothing ? "Inf" : string(nsteps_max))]; " *
                    "pass `out_of_range_policy = :skip` or `:clamp` to handle"))
            elseif out_of_range_policy === :skip
                continue
            else  # :clamp
                k = clamp(k, 1, nsteps_max)
            end
        end

        p, i, j = _locate_cs_cell(Float64(record.lat),
                                  Float64(record.lon), locator)
        push!(out, CSObservation(k,
                                 CSColumnMeanObjective(p, i, j),
                                 record.value, record.value_sigma))
    end
    return out
end

# ---------------------------------------------------------------------------
# Per-record fail-fast validation
#
# Every `CSObservationRecord` is already checked by its inner constructor,
# which the positional form, the keyword wrapper and `read_observations`
# all go through. The repeat check here is defensive; its error messages
# name the offending `record.id`, which is more useful at debug time than
# the constructor's generic message.
# ---------------------------------------------------------------------------

@inline function _validate_bind_record(record::CSObservationRecord)
    isfinite(record.lat) || throw(ArgumentError(
        "observation id $(record.id) has non-finite lat = $(record.lat)"))
    -90 <= record.lat <= 90 || throw(ArgumentError(
        "observation id $(record.id) has lat = $(record.lat) outside [-90, 90]"))
    isfinite(record.lon) || throw(ArgumentError(
        "observation id $(record.id) has non-finite lon = $(record.lon)"))
    isfinite(record.value) || throw(ArgumentError(
        "observation id $(record.id) has non-finite value = $(record.value)"))
    isfinite(record.value_sigma) || throw(ArgumentError(
        "observation id $(record.id) has non-finite value_sigma = " *
        "$(record.value_sigma)"))
    return nothing
end

# ---------------------------------------------------------------------------
# Time mapping
# ---------------------------------------------------------------------------

@inline function _step_index_from_date(dc::NTuple{6, Int16},
                                       t_start::Dates.DateTime,
                                       dt_f::Real)
    t_obs = Dates.DateTime(Int(dc[1]), Int(dc[2]), Int(dc[3]),
                            Int(dc[4]), Int(dc[5]), Int(dc[6]))
    ms = Dates.value(t_obs - t_start)   # Int64 milliseconds (signed)
    seconds = ms / 1000
    return floor(Int, seconds / dt_f) + 1
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
