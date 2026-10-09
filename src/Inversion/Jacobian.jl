# ---------------------------------------------------------------------------
# Surface-flux Jacobian + per-window aggregation.
#
# `_aggregate_surface_window` sums per-step `CSFootprintResult` footprints
# according to a `CSSurfaceFluxWindow`'s `(steps, weights)`; this is what
# converts a per-step emission-rate footprint into a control-window-level
# Jacobian column.
#
# `cs_surface_flux_jacobian` is the user-facing entry that loops over
# `objectives` and `windows`, calling `cs_surface_emission_footprint` per
# objective and then aggregating into the requested windows.
#
# Relocated unchanged from `src/Adjoints/Adjoints.jl` lines 199-241 and
# 476-532; no semantic change.
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
            synchronize(backend)
        end
    end
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
custom temporal basis.
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

    per_step = Vector{CSFootprintResult}(undef, length(objective_vec))
    first_result = cs_surface_emission_footprint(
        panels_rm0, panels_m0, panels_am_steps, panels_bm_steps, panels_cm_steps,
        mesh, objective_vec[1]; kwargs...)
    per_step[1] = first_result
    first_agg = _aggregate_surface_window(first_result, window_vec[1])
    footprints = Matrix{typeof(first_agg)}(undef, length(objective_vec), length(window_vec))
    footprints[1, 1] = first_agg
    for w in 2:length(window_vec)
        footprints[1, w] = _aggregate_surface_window(first_result, window_vec[w])
    end
    for o in 2:length(objective_vec)
        result = cs_surface_emission_footprint(
            panels_rm0, panels_m0, panels_am_steps, panels_bm_steps, panels_cm_steps,
            mesh, objective_vec[o]; kwargs...)
        per_step[o] = result
        for w in eachindex(window_vec)
            footprints[o, w] = _aggregate_surface_window(result, window_vec[w])
        end
    end
    A2 = typeof(first_agg[1])
    FT = eltype(first_agg[1])
    return CSSurfaceFluxJacobianResult{FT, A2}(
        objective_vec, window_vec, footprints, per_step, first_result.dt)
end
