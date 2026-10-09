# ---------------------------------------------------------------------------
# Observation, control, and 4D-Var result types.
#
# This file defines the user-facing types consumed by the prototype CS
# 4D-Var path:
#
#   * `CSSurfaceFluxWindow`           — named temporal aggregation window.
#   * `CSSurfaceFluxJacobianResult`   — return type of
#                                         `cs_surface_flux_jacobian`.
#   * `CSObservation`                 — scalar observation with sigma.
#   * `CSSurfaceFluxControl`          — surface-flux control block
#                                         tied to a window; optional
#                                         diagonal background + sigma.
#   * `CS4DVarResult`                 — per-iteration 4D-Var output.
#   * `CS4DVarSolveResult`            — full optimizer history wrapper.
#
# Helper: `_step_vector` is the small utility that normalises step
# specifiers (Integer / iterable) for the `CSSurfaceFluxWindow`
# constructor.
#
# Loaded into the `Adjoints` module via an `include` from
# `src/Adjoints/Adjoints.jl`.
# ---------------------------------------------------------------------------

"""
    CSSurfaceFluxWindow(name, steps; weights=nothing, normalize=false)

Named surface-flux control window for Jacobian aggregation. `steps` can be
a single step or any iterable of step indices. By default, a window sums
per-step surface-emission footprints, which corresponds to perturbing the
same rate over the whole window. Pass `normalize=true` for an average-rate
control, or explicit `weights` for custom temporal basis functions.
"""
struct CSSurfaceFluxWindow
    name::Symbol
    steps::Vector{Int}
    weights::Vector{Float64}
end

_step_vector(step::Integer) = [Int(step)]
_step_vector(steps) = Int.(collect(steps))

function CSSurfaceFluxWindow(name::Symbol, steps; weights = nothing,
                             normalize::Bool = false)
    step_vec = _step_vector(steps)
    isempty(step_vec) && throw(ArgumentError("surface-flux window $name has no steps"))
    any(<=(0), step_vec) &&
        throw(ArgumentError("surface-flux window $name contains non-positive step indices"))
    length(unique(step_vec)) == length(step_vec) ||
        throw(ArgumentError("surface-flux window $name contains duplicate step indices"))
    weight_vec = if weights === nothing
        ones(Float64, length(step_vec))
    else
        Float64.(collect(weights))
    end
    length(weight_vec) == length(step_vec) || throw(ArgumentError(
        "surface-flux window $name has $(length(step_vec)) steps but " *
        "$(length(weight_vec)) weights"))
    all(isfinite, weight_vec) || throw(ArgumentError(
        "surface-flux window $name weights must be finite"))
    if normalize
        total = sum(weight_vec)
        isfinite(total) && total != 0 || throw(ArgumentError(
            "surface-flux window $name cannot normalize non-finite or zero-sum weights"))
        weight_vec ./= total
    end
    return CSSurfaceFluxWindow(name, step_vec, weight_vec)
end

CSSurfaceFluxWindow(name::AbstractString, steps; kwargs...) =
    CSSurfaceFluxWindow(Symbol(name), steps; kwargs...)

struct CSSurfaceFluxJacobianResult{FT, A2 <: AbstractArray{FT, 2}}
    objectives::Vector{AbstractCSFootprintObjective}
    windows::Vector{CSSurfaceFluxWindow}
    footprints::Matrix{NTuple{6, A2}}
    per_step_results::Vector{<:CSFootprintResult}
    dt::FT
end

"""
    CSObservation(step, objective, value, sigma)

Scalar observation used by the prototype CS 4D-Var surface-flux path.
`step` is the model-step index after which the objective is sampled.
`sigma` is the observation-error standard deviation.
"""
struct CSObservation{O <: AbstractCSFootprintObjective, FT <: AbstractFloat}
    step::Int
    objective::O
    value::FT
    sigma::FT
end

function CSObservation(step::Integer, objective::AbstractCSFootprintObjective,
                       value::Real, sigma::Real)
    step > 0 || throw(ArgumentError("CSObservation step must be positive, got $step"))
    isfinite(value) || throw(ArgumentError("CSObservation value must be finite, got $value"))
    isfinite(sigma) || throw(ArgumentError("CSObservation sigma must be finite, got $sigma"))
    sigma > 0 || throw(ArgumentError("CSObservation sigma must be positive, got $sigma"))
    FT = promote_type(typeof(float(value)), typeof(float(sigma)))
    return CSObservation{typeof(objective), FT}(Int(step), objective, FT(value), FT(sigma))
end

"""
    CSSurfaceFluxControl(window, value; background=nothing, sigma=nothing)

Surface-flux control block for the prototype 4D-Var path. `value` is an
`NTuple{6}` of `(Nc, Nc)` rate arrays tied to `window`. Optional
`background` and `sigma` add the standard diagonal background term
`0.5 * ((value - background) / sigma)^2`; `sigma` can be a scalar or
matching panel arrays.
"""
struct CSSurfaceFluxControl{FT, A2 <: AbstractArray{FT, 2}, B, S}
    window::CSSurfaceFluxWindow
    value::NTuple{6, A2}
    background::B
    sigma::S
end

function CSSurfaceFluxControl(window::CSSurfaceFluxWindow,
                              value::NTuple{6, A2};
                              background = nothing,
                              sigma = nothing) where {FT, A2 <: AbstractArray{FT, 2}}
    background === nothing || sigma !== nothing || throw(ArgumentError(
        "surface-flux control background requires `sigma`"))
    if background !== nothing
        background isa NTuple{6} || throw(ArgumentError(
            "surface-flux control background must be nothing or an NTuple{6}"))
        @inbounds for p in 1:6
            size(background[p]) == size(value[p]) || throw(DimensionMismatch(
                "surface-flux control background panel $p has shape $(size(background[p])); " *
                "expected $(size(value[p]))"))
        end
    end
    if sigma !== nothing && !(sigma isa Real) && !(sigma isa NTuple{6})
        throw(ArgumentError("surface-flux control sigma must be nothing, a scalar, or an NTuple{6}"))
    end
    if sigma isa Real
        isfinite(sigma) && sigma > 0 || throw(ArgumentError(
            "surface-flux control scalar sigma must be finite and positive"))
    elseif sigma isa NTuple{6}
        @inbounds for p in 1:6
            size(sigma[p]) == size(value[p]) || throw(DimensionMismatch(
                "surface-flux control sigma panel $p has shape $(size(sigma[p])); " *
                "expected $(size(value[p]))"))
            all(x -> isfinite(x) && x > 0, sigma[p]) || throw(ArgumentError(
                "surface-flux control sigma panel $p must contain finite positive values"))
        end
    end
    return CSSurfaceFluxControl{FT, A2, typeof(background), typeof(sigma)}(
        window, value, background, sigma)
end

struct CS4DVarResult{FT, A2 <: AbstractArray{FT, 2}}
    cost::FT
    observation_cost::FT
    background_cost::FT
    simulated::Vector{FT}
    residuals::Vector{FT}
    gradients::Vector{NTuple{6, A2}}
    gradient_by_name::Dict{Symbol, NTuple{6, A2}}
    controls::Vector{<:CSSurfaceFluxControl}
    observations::Vector{<:CSObservation}
end

"""
    CSIterationLogEntry(iteration, cost, observation_cost, background_cost,
                        gradient_norm, step_size, elapsed_seconds)

One row of per-iteration solver diagnostics. Captures the
cost-decomposition (observation vs background terms), the L2
gradient norm, the line-search step accepted on this iteration
(`0` for optimizers that don't expose a step — e.g. L-BFGS), and the
wall-clock time since the solve started.

The convention is that an iteration is logged AFTER the descent
direction has been accepted; the `iteration = 0` row records the
initial probe before any descent step.
"""
struct CSIterationLogEntry{FT}
    iteration::Int
    cost::FT
    observation_cost::FT
    background_cost::FT
    gradient_norm::FT
    step_size::FT
    elapsed_seconds::Float64
end

"""
    CSIterationLog(entries)

Per-iteration diagnostic log for a CS 4D-Var solve. Populated by
optimizers when their `log::Bool` field is set to `true`; otherwise
the `CS4DVarSolveResult.log` field is `nothing`.

The log carries enough information for after-the-fact convergence
diagnostics: a monotone cost trajectory check, gradient-norm decay
curve, observation-vs-background cost decomposition, and a
wall-clock time series.
"""
struct CSIterationLog{FT}
    entries::Vector{CSIterationLogEntry{FT}}
end

CSIterationLog{FT}() where FT = CSIterationLog{FT}(CSIterationLogEntry{FT}[])

Base.length(log::CSIterationLog) = length(log.entries)
Base.isempty(log::CSIterationLog) = isempty(log.entries)
Base.iterate(log::CSIterationLog, args...) = iterate(log.entries, args...)
Base.getindex(log::CSIterationLog, i::Integer) = log.entries[i]
Base.firstindex(log::CSIterationLog) = firstindex(log.entries)
Base.lastindex(log::CSIterationLog) = lastindex(log.entries)
Base.eachindex(log::CSIterationLog) = eachindex(log.entries)

struct CS4DVarSolveResult{FT, A2 <: AbstractArray{FT, 2}}
    controls::Vector{<:CSSurfaceFluxControl}
    last::CS4DVarResult{FT, A2}
    cost_history::Vector{FT}
    gradient_norm_history::Vector{FT}
    step_history::Vector{FT}
    iterations::Int
    log::Union{Nothing, CSIterationLog{FT}}
end

# Backward-compat constructor — pre-C3 call sites that don't pass a
# log argument default to `nothing` (matching the pre-C3 behavior).
function CS4DVarSolveResult{FT, A2}(controls, last, cost_history,
                                     gradient_norm_history, step_history,
                                     iterations) where {FT, A2}
    return CS4DVarSolveResult{FT, A2}(controls, last, cost_history,
                                       gradient_norm_history, step_history,
                                       iterations, nothing)
end
