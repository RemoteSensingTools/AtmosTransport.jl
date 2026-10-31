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

"""
    AbstractCS4DVarEvaluation

What one `cs_surface_flux_4dvar` evaluation computes: [`CSCostAndGradient`](@ref)
(the default) or [`CSCostOnly`](@ref).
"""
abstract type AbstractCS4DVarEvaluation end

"""
    CSCostAndGradient()

A 4D-Var evaluation of the cost and its gradient (the default of
`cs_surface_flux_4dvar` and the evaluation of every optimizer step): one
forward run that samples the observations and records the tape of its
`checkpoint` schedule, then one reverse pass.
"""
struct CSCostAndGradient <: AbstractCS4DVarEvaluation end

"""
    CSCostOnly()

A 4D-Var evaluation of the cost and the simulated values only: one forward
run without a tape, no reverse pass and no tape storage. The result's
`gradients` and `gradient_by_name` are empty. For callers that need the
simulated observations or the cost at a point but not the gradient there
(for example, synthetic observations).
"""
struct CSCostOnly <: AbstractCS4DVarEvaluation end

"""
    CS4DVarResult

One 4D-Var cost and gradient evaluation (`cs_surface_flux_4dvar`): the
total, observation and background costs, the simulated values and
residuals `H(x) − y` per observation, the gradient per control (and by
control name; both empty for a [`CSCostOnly`](@ref) evaluation), the
controls and observations evaluated, and `stencils`, one
[`CSObservationStencil`](@ref) per observation (the steps, weights and
levels it was sampled at). An observation whose stencil is not `assimilated`
(a co-sample, or an averaging-kernel column screened out by its
surface-pressure mismatch) keeps its simulated value and residual but adds
nothing to the cost and the gradient.
"""
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
    stencils::Vector{CSObservationStencil}
end

"""
    CSIterationLogEntry(iteration, cost, observation_cost, background_cost,
                        gradient_norm, step_size, elapsed_seconds)

One row of per-iteration solver diagnostics. Captures the
cost-decomposition (observation vs background terms), the optimizer's
documented gradient norm (L2 for `CSGradientDescent`, L-infinity for
`CSLBFGS`), the line-search step accepted on this iteration
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
