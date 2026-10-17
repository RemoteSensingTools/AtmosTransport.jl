# ---------------------------------------------------------------------------
# Observation, control, and 4D-Var result types.
#
# This file defines the user-facing types consumed by the prototype CS
# 4D-Var path:
#
#   * `CSSurfaceFluxWindow`           — named temporal aggregation window.
#   * `CSSurfaceFluxJacobianResult`   — return type of
#                                         `cs_surface_flux_jacobian`.
#   * `CSTimeSampling`                — the steps an observation samples
#                                         and their weights (one step, or
#                                         a Δt/ΣΔt time window).
#   * `CSObservation`                 — scalar observation with sigma,
#                                         time sampling and record id.
#   * `CSSurfaceFluxControl`          — surface-flux control block
#                                         tied to a window; optional
#                                         diagonal background + sigma.
#   * `CSObservationStencil`          — where an observation was sampled
#                                         (steps, weights, levels).
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
    CSTimeSampling(steps, weights)
    CSTimeSampling(step)

The model states an observation samples: the states after `steps` (strictly
increasing, after convection: the state runtime output sees), averaged with
`weights` (positive, summing to 1). The simulated value is
``H = \\sum_s w_s\\, h(\\mathbf{r}_s)``, with ``h`` the objective evaluated on
the state after step ``s``. `CSTimeSampling(step)` samples one state (weight 1).

`bind_to_mesh` builds an instantaneous sampling for a record without a time
window (the step whose interval contains the record time) and, for a record
with `time_window_s > 0`, the steps whose intervals overlap the window with
weights proportional to the overlap (TM5-4DVAR's Δt/ΣΔt window mean).
"""
struct CSTimeSampling
    steps::Vector{Int}
    weights::Vector{Float64}
    function CSTimeSampling(steps::AbstractVector{<:Integer}, weights::AbstractVector{<:Real})
        n = length(steps)
        n >= 1 || throw(ArgumentError("CSTimeSampling needs at least one step"))
        length(weights) == n || throw(DimensionMismatch(
            "CSTimeSampling has $n steps but $(length(weights)) weights"))
        first(steps) >= 1 || throw(ArgumentError(
            "CSTimeSampling steps must be positive, got $(first(steps))"))
        all(>(0), diff(steps)) || throw(ArgumentError(
            "CSTimeSampling steps must be strictly increasing, got $(steps)"))
        all(w -> isfinite(w) && w > 0, weights) || throw(ArgumentError(
            "CSTimeSampling weights must be finite and positive, got $(weights)"))
        total = sum(Float64, weights)
        abs(total - 1) <= 8n * eps(Float64) || throw(ArgumentError(
            "CSTimeSampling weights must sum to 1, got $(total)"))
        return new(collect(Int, steps), collect(Float64, weights))
    end
end

CSTimeSampling(step::Integer) = CSTimeSampling([Int(step)], [1.0])

Base.:(==)(a::CSTimeSampling, b::CSTimeSampling) = a.steps == b.steps && a.weights == b.weights
Base.hash(s::CSTimeSampling, h::UInt) = hash(s.weights, hash(s.steps, hash(:CSTimeSampling, h)))

"""
    CSObservation(step, objective, value, sigma; id = 0)
    CSObservation(sampling::CSTimeSampling, objective, value, sigma; id = 0)

Scalar observation of the CS 4D-Var surface-flux path: `objective` sampled
as `sampling` says (one step, or a weighted window of steps; see
[`CSTimeSampling`](@ref)), with observed `value` and error standard
deviation `sigma`. `step` is the last sampled step, after which the
observation is complete (the only one for an instantaneous observation).
`id` is the identifier of the observation record it was bound from
(`bind_to_mesh`), `0` for an observation built by hand.
"""
struct CSObservation{O <: AbstractCSFootprintObjective, FT <: AbstractFloat}
    step::Int
    sampling::CSTimeSampling
    objective::O
    value::FT
    sigma::FT
    id::Int64
    function CSObservation{O, FT}(sampling::CSTimeSampling, objective::O, value::FT, sigma::FT,
                                  id::Int64) where {O <: AbstractCSFootprintObjective,
                                                    FT <: AbstractFloat}
        isfinite(value) || throw(ArgumentError("CSObservation value must be finite, got $value"))
        isfinite(sigma) || throw(ArgumentError("CSObservation sigma must be finite, got $sigma"))
        sigma > 0 || throw(ArgumentError("CSObservation sigma must be positive, got $sigma"))
        return new{O, FT}(last(sampling.steps), sampling, objective, value, sigma, id)
    end
end

function CSObservation(sampling::CSTimeSampling, objective::AbstractCSFootprintObjective,
                       value::Real, sigma::Real; id::Integer = 0)
    FT = promote_type(typeof(float(value)), typeof(float(sigma)))
    return CSObservation{typeof(objective), FT}(sampling, objective, FT(value), FT(sigma), Int64(id))
end

function CSObservation(step::Integer, objective::AbstractCSFootprintObjective,
                       value::Real, sigma::Real; id::Integer = 0)
    step > 0 || throw(ArgumentError("CSObservation step must be positive, got $step"))
    return CSObservation(CSTimeSampling(step), objective, value, sigma; id)
end

"""
Stencil placement code of an observation whose objective is a column mean
([`CSColumnMeanObjective`](@ref)): every level of the column, no vertical
placement. The point objectives use the `Output.VERTICAL_PLACEMENT_CODES`.
"""
const COLUMN_MEAN_PLACEMENT_CODE = Int8(-1)
"""
Stencil placement code of an observation whose objective has a fixed stencil
instead of being placed: a [`CSLayerMeanObjective`](@ref) naming its level,
or another fixed operator.
"""
const FIXED_LEVEL_PLACEMENT_CODE = Int8(-2)
"""
Stencil placement code of an observation whose objective is an
averaging-kernel column ([`CSRetrievalColumnObjective`](@ref)): every level
of the column, weighted by the retrieval's pressure weights and averaging
kernel on the model column stretched onto the retrieval column.
"""
const RETRIEVAL_COLUMN_PLACEMENT_CODE = Int8(-3)

"""
    CSObservationStencil(panel, i, j, steps, weights, levels, placement_method,
                         height_agl_m, pressure_pa,
                         surface_pressure_mismatch_pa = NaN, retrieval_profile = Float64[],
                         assimilated = true)

Where an observation was sampled in a 4D-Var evaluation, returned per
observation in [`CS4DVarResult`](@ref)`.stencils`: the cell `(panel, i, j)`,
the sampled `steps` and their `weights` (its [`CSTimeSampling`](@ref)), the
model level sampled at each step (`levels`, k = 1 at the top, resolved from
the replayed column at the end of that step; `0` for an operator that reads
more than one level, as a column mean), the `placement_method`
(`Output.VERTICAL_PLACEMENT_CODES` for a [`CSPlacedPointObjective`](@ref),
[`COLUMN_MEAN_PLACEMENT_CODE`](@ref),
[`RETRIEVAL_COLUMN_PLACEMENT_CODE`](@ref) or
[`FIXED_LEVEL_PLACEMENT_CODE`](@ref) otherwise), and the height above the
model surface (m) or the pressure (Pa) that placed it (NaN when not used).
An operator without a single cell has the cell `(0, 0, 0)`.

For an averaging-kernel column ([`CSRetrievalColumnObjective`](@ref)):
`surface_pressure_mismatch_pa`, the model minus the retrieval surface
pressure (Pa; time-weighted over the sampled steps; NaN for other
operators); `retrieval_profile`, the model's dry mixing ratios on the
retrieval layers (surface first, the rebinned profile ``R x``,
time-weighted; empty for other operators); and `assimilated`, `false` when
the surface-pressure mismatch exceeded `max_surface_pressure_mismatch_pa`
under `SkipUnplaced()`: the observation's simulated value is reported, but
it is not in the cost.
"""
struct CSObservationStencil
    panel::Int
    i::Int
    j::Int
    steps::Vector{Int}
    weights::Vector{Float64}
    levels::Vector{Int}
    placement_method::Int8
    height_agl_m::Float64
    pressure_pa::Float64
    surface_pressure_mismatch_pa::Float64
    retrieval_profile::Vector{Float64}
    assimilated::Bool
end

CSObservationStencil(panel, i, j, steps, weights, levels, placement_method, height_agl_m, pressure_pa) =
    CSObservationStencil(panel, i, j, steps, weights, levels, placement_method, height_agl_m, pressure_pa,
                         NaN, Float64[], true)

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
    CS4DVarResult

One 4D-Var cost and gradient evaluation (`cs_surface_flux_4dvar`): the
total, observation and background costs, the simulated values and
residuals `H(x) − y` per observation, the gradient per control (and by
control name), the controls and observations evaluated, and `stencils`, one
[`CSObservationStencil`](@ref) per observation (the steps, weights and
levels it was sampled at). An observation whose stencil is not `assimilated`
(an averaging-kernel column screened out by its surface-pressure mismatch)
keeps its simulated value and residual but adds nothing to the cost and the
gradient.
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

# Backward-compat constructor — pre-C3 call sites that don't pass a
# log argument default to `nothing` (matching the pre-C3 behavior).
function CS4DVarSolveResult{FT, A2}(controls, last, cost_history,
                                     gradient_norm_history, step_history,
                                     iterations) where {FT, A2}
    return CS4DVarSolveResult{FT, A2}(controls, last, cost_history,
                                       gradient_norm_history, step_history,
                                       iterations, nothing)
end
