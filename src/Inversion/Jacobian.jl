# ---------------------------------------------------------------------------
# Surface-flux Jacobian + per-window aggregation.
#
# `_aggregate_surface_window` sums per-step `CSFootprintResult` footprints
# according to a `CSSurfaceFluxWindow`'s `(steps, weights)`; this is what
# converts a per-step emission-rate footprint into a control-window-level
# Jacobian column.
#
# `cs_surface_flux_jacobian` is the user-facing entry: it computes one
# footprint per objective (one recorded tape walked once per objective under
# `FullCheckpoint`, `_cs_objective_footprints`) and aggregates each into the
# requested windows.
# ---------------------------------------------------------------------------

@kernel function _add_weighted_footprint_kernel!(dst, @Const(src), weight)
    i, j = @index(Global, NTuple)
    @inbounds dst[i, j] += weight * src[i, j]
end

function _aggregate_surface_window(result::CSFootprintResult,
                                   window::CSSurfaceFluxWindow;
                                   ignore_future::Bool = false)
    nsteps = length(result.footprints)
    FT = eltype(result.footprints[1][1])
    aggregate = ntuple(p -> begin
        a = similar(result.footprints[1][p])
        fill!(a, zero(FT))
        a
    end, 6)
    @inbounds for (idx, step) in enumerate(window.steps)
        if !(1 <= step <= nsteps)
            if ignore_future && step > nsteps
                continue
            end
            throw(ArgumentError(
                "surface-flux window $(window.name) references step $step, " *
                "but the footprint has $nsteps steps"))
        end
        weight = FT(window.weights[idx])
        for p in 1:6
            backend = get_backend(aggregate[p])
            kernel! = _add_weighted_footprint_kernel!(backend, (16, 16))
            kernel!(aggregate[p], result.footprints[step][p], weight;
                    ndrange = size(aggregate[p]))
        end
    end
    # The launches above are queued in order; one wait covers them all.
    synchronize(get_backend(aggregate[1]))
    return aggregate
end

function _zero_surface_like(values::NTuple{6, A2}) where {FT, A2 <: AbstractArray{FT, 2}}
    return ntuple(p -> begin
        a = similar(values[p])
        fill!(a, zero(FT))
        a
    end, 6)
end

# ---------------------------------------------------------------------------
# Public Jacobian entry
# ---------------------------------------------------------------------------

_objective_vector(obj::AbstractCSFootprintObjective) =
    AbstractCSFootprintObjective[obj]
_objective_vector(objectives) =
    AbstractCSFootprintObjective[objectives...]
_window_vector(window::CSSurfaceFluxWindow) =
    CSSurfaceFluxWindow[window]
_window_vector(windows) =
    CSSurfaceFluxWindow[windows...]

"""
    cs_surface_flux_jacobian(..., objectives, windows; kwargs...)

Compute surface-flux Jacobian maps for several layer/column objectives and
named time windows. Each returned `footprints[obj, window]` entry is an
`NTuple{6}` of `(Nc, Nc)` arrays. Window aggregation is a weighted sum of
per-step emission-rate footprints; use `CSSurfaceFluxWindow(...;
normalize=true)` for average-rate controls or explicit `weights` for a
custom temporal basis. `kwargs` are those of `cs_surface_emission_footprint`.
Every checkpoint schedule records (and recomputes) the forward run once and
walks it once per objective.
"""
function cs_surface_flux_jacobian(panels_rm0, panels_m0,
                                  panels_am_steps,
                                  panels_bm_steps,
                                  panels_cm_steps,
                                  mesh::CubedSphereMesh,
                                  objectives,
                                  windows;
                                  kwargs...)
    objective_vec = _objective_vector(objectives)
    window_vec = _window_vector(windows)
    isempty(objective_vec) && throw(ArgumentError("at least one objective is required"))
    isempty(window_vec) && throw(ArgumentError("at least one surface-flux window is required"))

    # Under the default `FullCheckpoint` the forward tape is recorded once and
    # walked once per objective; the objectives share the trajectory, so each
    # walk is the one a separate `cs_surface_emission_footprint` call makes.
    per_step = CSFootprintResult[result for result in _cs_objective_footprints(
        panels_rm0, panels_m0, panels_am_steps, panels_bm_steps, panels_cm_steps,
        mesh, objective_vec; kwargs...)]
    first_result = per_step[1]
    first_agg = _aggregate_surface_window(first_result, window_vec[1])
    footprints = Matrix{typeof(first_agg)}(undef, length(objective_vec), length(window_vec))
    for o in eachindex(objective_vec), w in eachindex(window_vec)
        footprints[o, w] = (o, w) == (1, 1) ? first_agg :
            _aggregate_surface_window(per_step[o], window_vec[w])
    end
    A2 = typeof(first_agg[1])
    FT = eltype(first_agg[1])
    return CSSurfaceFluxJacobianResult{FT, A2}(
        objective_vec, window_vec, footprints, per_step, first_result.dt)
end
