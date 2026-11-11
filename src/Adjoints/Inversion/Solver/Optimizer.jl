# ---------------------------------------------------------------------------
# Polymorphic optimizer dispatch for CS 4D-Var.
#
# Layout (new backends — hand-rolled L-BFGS-B, etc. — plug in as new
# concrete subtypes without touching the public entrypoint):
#
#   AbstractCSOptimizer
#       │
#       ├── CSGradientDescent          (the descent shim, wrapped as a
#       │                               concrete subtype)
#       └── CSLBFGS                    (limited-memory BFGS via
#                                       `Optim.jl`)
#
# Backends implement `cs_surface_flux_4dvar_solve(opt, cost_fn,
# controls)` where `cost_fn(controls) -> CS4DVarResult` is the
# closure built by `cs_surface_flux_4dvar_optimize`. The public
# entrypoint accepts a concrete optimizer and defaults to gradient descent.
# Every backend carries the same `_CSStopRules` and passes each evaluation
# through one `_CSStopTracker`, which keeps the best evaluation and decides
# the opt-in stops (cost decrease, stall, evaluation cap, user callback).
# ---------------------------------------------------------------------------

import Optim

"""
    AbstractCSOptimizer

Supertype for CS 4D-Var optimization backends. Concrete subtypes
implement `cs_surface_flux_4dvar_solve(opt, cost_fn, controls)`
where `cost_fn(controls) -> CS4DVarResult` evaluates one
cost-and-gradient pass through the 4D-Var driver.

Shipped backends:

- [`CSGradientDescent`](@ref) — the dependency-free
  backtracking-line-search descent loop.

Additional backends:

- `CSLBFGS` — `Optim.jl` L-BFGS wrapper.

Both accept the same stop rules (`cost_rtol`, `cost_rtol_improvements`,
`stall_evaluations`, `max_evaluations`, `callback`); see [`CSLBFGS`](@ref).
"""
abstract type AbstractCSOptimizer end

# ---------------------------------------------------------------------------
# Stop rules shared by every backend
# ---------------------------------------------------------------------------

# The opt-in stop rules of one optimizer, checked after every cost evaluation
# (line-search trials included). Zero turns a rule off; the defaults keep
# every rule off. The meaning of each field is documented on `CSLBFGS`.
struct _CSStopRules{C}
    cost_rtol::Float64
    cost_rtol_improvements::Int
    stall_evaluations::Int
    max_evaluations::Int
    callback::C

    function _CSStopRules(cost_rtol::Real, cost_rtol_improvements::Integer,
                          stall_evaluations::Integer, max_evaluations::Integer,
                          callback::C) where C
        cost_rtol >= 0 || throw(ArgumentError(
            "cost_rtol must be non-negative (0 = off), got $cost_rtol"))
        cost_rtol_improvements > 0 || throw(ArgumentError(
            "cost_rtol_improvements must be positive, got $cost_rtol_improvements"))
        stall_evaluations >= 0 || throw(ArgumentError(
            "stall_evaluations must be non-negative (0 = off), got $stall_evaluations"))
        max_evaluations >= 0 || throw(ArgumentError(
            "max_evaluations must be non-negative (0 = no cap), got $max_evaluations"))
        return new{C}(Float64(cost_rtol), Int(cost_rtol_improvements),
                      Int(stall_evaluations), Int(max_evaluations), callback)
    end
end

# The state of the stop rules during one solve. Every cost evaluation passes
# through `_record_evaluation!`; a rule that fires sets `reason`, and the
# backend stops before its next evaluation (`_may_evaluate!` returns false).
mutable struct _CSStopTracker{R, C}
    const rules::_CSStopRules{C}
    evaluations::Int
    best::R                        # lowest-cost evaluation (earliest on ties)
    small_improvements::Int        # consecutive best-cost decreases below cost_rtol
    stalled_evaluations::Int       # evaluations since the best cost last decreased
    reason::Symbol                 # :none until a rule fires
end

# The tracker of a solve whose first evaluation is `first`.
function _CSStopTracker(rules::_CSStopRules, first)
    tracker = _CSStopTracker(rules, 1, first, 0, 0, :none)
    _apply_callback!(tracker, first)
    return tracker
end

_stopped(tracker::_CSStopTracker) = tracker.reason !== :none

# The first rule that fires names the stop.
function _stop!(tracker::_CSStopTracker, reason::Symbol)
    _stopped(tracker) || (tracker.reason = reason)
    return nothing
end

_callback_requests_stop(::Nothing, _evaluation, _result) = false
_callback_requests_stop(callback, evaluation, result) = callback(evaluation, result)::Bool

function _apply_callback!(tracker::_CSStopTracker, result)
    _callback_requests_stop(tracker.rules.callback, tracker.evaluations, result) &&
        _stop!(tracker, :callback)
    return nothing
end

# Record one cost evaluation; returns whether it lowered the best cost.
function _record_evaluation!(tracker::_CSStopTracker, result)
    rules = tracker.rules
    tracker.evaluations += 1
    best_cost = tracker.best.cost
    improved = result.cost < best_cost
    if improved
        decrease = (best_cost - result.cost) / abs(best_cost)
        tracker.small_improvements = decrease < rules.cost_rtol ?
            tracker.small_improvements + 1 : 0
        tracker.best = result
        tracker.stalled_evaluations = 0
        tracker.small_improvements >= rules.cost_rtol_improvements && _stop!(tracker, :cost_rtol)
    else
        tracker.stalled_evaluations += 1
        rules.stall_evaluations > 0 &&
            tracker.stalled_evaluations >= rules.stall_evaluations && _stop!(tracker, :stall)
    end
    _apply_callback!(tracker, result)
    return improved
end

# Whether the solve may run another cost evaluation: no rule has fired and the
# evaluation cap leaves room. Reaching the cap is itself a stop.
function _may_evaluate!(tracker::_CSStopTracker)
    cap = tracker.rules.max_evaluations
    cap > 0 && tracker.evaluations >= cap && _stop!(tracker, :evaluations)
    return !_stopped(tracker)
end

# ---------------------------------------------------------------------------
# CSGradientDescent
# ---------------------------------------------------------------------------

"""
    CSGradientDescent(; iterations, initial_step, min_step,
                        step_shrink, gradient_tolerance, line_search,
                        log, cost_rtol, cost_rtol_improvements,
                        stall_evaluations, max_evaluations, callback)

Plain backtracking-line-search gradient descent. `iterations` is the
maximum count of accepted descent steps; the loop terminates early
when the gradient L2 norm falls below `gradient_tolerance` or the
line search shrinks `step` below `min_step`. With
`line_search = false` every candidate step is accepted (constant
step size unless the loop exits on tolerance).

`log = true` captures per-iteration diagnostics into
`CS4DVarSolveResult.log` (see [`CSIterationLog`](@ref)): cost
decomposition (observation vs background), gradient L2 norm,
accepted step size, and wall-clock elapsed seconds since the solve
started. Default `false` matches the pre-C3 behavior.

The stop rules `cost_rtol`, `cost_rtol_improvements`, `stall_evaluations`,
`max_evaluations` and `callback` are the same as [`CSLBFGS`](@ref)'s and are
off by default. Every line-search candidate is an evaluation.
"""
struct CSGradientDescent{FT <: AbstractFloat, R <: _CSStopRules} <: AbstractCSOptimizer
    iterations::Int
    initial_step::FT
    min_step::FT
    step_shrink::FT
    gradient_tolerance::FT
    line_search::Bool
    log::Bool
    stop_rules::R

    function CSGradientDescent(iterations::Integer,
                                initial_step::FT,
                                min_step::FT,
                                step_shrink::FT,
                                gradient_tolerance::FT,
                                line_search::Bool,
                                log::Bool,
                                stop_rules::R) where {FT <: AbstractFloat, R <: _CSStopRules}
        iterations >= 0 || throw(ArgumentError(
            "CSGradientDescent iterations must be non-negative, got $iterations"))
        initial_step > 0 || throw(ArgumentError(
            "CSGradientDescent initial_step must be positive, got $initial_step"))
        min_step > 0 || throw(ArgumentError(
            "CSGradientDescent min_step must be positive, got $min_step"))
        0 < step_shrink < 1 || throw(ArgumentError(
            "CSGradientDescent step_shrink must be in (0, 1), got $step_shrink"))
        gradient_tolerance >= 0 || throw(ArgumentError(
            "CSGradientDescent gradient_tolerance must be non-negative, got " *
            "$gradient_tolerance"))
        return new{FT, R}(Int(iterations), initial_step, min_step, step_shrink,
                          gradient_tolerance, line_search, log, stop_rules)
    end
end

function CSGradientDescent(; iterations::Integer = 10,
                            initial_step::Real = 1.0,
                            min_step::Real = sqrt(eps(Float64)),
                            step_shrink::Real = 0.5,
                            gradient_tolerance::Real = 0.0,
                            line_search::Bool = true,
                            log::Bool = false,
                            cost_rtol::Real = 0.0,
                            cost_rtol_improvements::Integer = 2,
                            stall_evaluations::Integer = 0,
                            max_evaluations::Integer = 0,
                            callback = nothing)
    FT = promote_type(typeof(float(initial_step)),
                      typeof(float(min_step)),
                      typeof(float(step_shrink)),
                      typeof(float(gradient_tolerance)))
    return CSGradientDescent(Int(iterations),
                             FT(initial_step), FT(min_step),
                             FT(step_shrink), FT(gradient_tolerance),
                             line_search, log,
                             _CSStopRules(cost_rtol, cost_rtol_improvements,
                                          stall_evaluations, max_evaluations, callback))
end

# ---------------------------------------------------------------------------
# Polymorphic backend dispatch
# ---------------------------------------------------------------------------

"""
    cs_surface_flux_4dvar_solve(optimizer::AbstractCSOptimizer,
                                 cost_fn,
                                 controls) -> CS4DVarSolveResult

Run `optimizer` against the cost closure `cost_fn(controls) ->
CS4DVarResult` starting from `controls`. Dispatches on the
optimizer's concrete type — additional backends are added by
defining a new method here, not by branching inside an existing one.
The solver keeps earlier results (the best and the last accepted
evaluation), so `cost_fn` must return arrays that later calls do not
overwrite. See [`CS4DVarSolveResult`](@ref) for what it returns.
"""
function cs_surface_flux_4dvar_solve end

function cs_surface_flux_4dvar_solve(optimizer::CSGradientDescent,
                                      cost_fn, controls)
    current = cost_fn(controls)
    isempty(current.gradients) && throw(ArgumentError(
        "cost_fn returned a CS4DVarResult with no gradients; " *
        "cannot derive optimizer element type"))
    tracker = _CSStopTracker(optimizer.stop_rules, current)
    FT = eltype(current.gradients[1][1])
    initial_step = FT(optimizer.initial_step)
    min_step = FT(optimizer.min_step)
    step_shrink = FT(optimizer.step_shrink)
    gradient_tolerance = FT(optimizer.gradient_tolerance)

    cost_history = FT[FT(current.cost)]
    grad_norm = FT(_control_gradient_norm(current.gradients))
    gradient_norm_history = FT[grad_norm]
    step_history = FT[]

    # Optional per-iteration diagnostic log. `iteration = 0` records
    # the initial probe before any descent step.
    log = optimizer.log ? CSIterationLog{FT}() : nothing
    t0 = time()
    if log !== nothing
        push!(log.entries, CSIterationLogEntry{FT}(
            0, 1, FT(current.cost), FT(current.observation_cost),
            FT(current.background_cost), grad_norm, zero(FT),
            time() - t0))
    end

    reason = :iterations
    for iter in 1:optimizer.iterations
        if grad_norm <= gradient_tolerance
            reason = :gradient
            break
        end
        step = initial_step
        accepted = false
        candidate = nothing
        # A stop rule ends the line search before its next candidate.
        while step >= min_step && _may_evaluate!(tracker)
            candidate_controls = _gradient_step_controls(
                current.controls, current.gradients, step)
            candidate = cost_fn(candidate_controls)
            _record_evaluation!(tracker, candidate)
            if !optimizer.line_search || candidate.cost <= current.cost
                accepted = true
                break
            end
            step *= step_shrink
        end
        if accepted
            current = candidate
            grad_norm = FT(_control_gradient_norm(current.gradients))
            push!(cost_history, FT(current.cost))
            push!(gradient_norm_history, grad_norm)
            push!(step_history, step)
            if log !== nothing
                push!(log.entries, CSIterationLogEntry{FT}(
                    iter, tracker.evaluations, FT(current.cost),
                    FT(current.observation_cost), FT(current.background_cost),
                    grad_norm, step, time() - t0))
            end
        end
        _stopped(tracker) && break
        if !accepted
            reason = :line_search
            break
        end
    end

    A2 = typeof(current.gradients[1][1])
    return CS4DVarSolveResult{FT, A2}(
        current.controls,
        current,
        cost_history,
        gradient_norm_history,
        step_history,
        length(step_history),
        log,
        tracker.best,
        tracker.evaluations,
        _stopped(tracker) ? tracker.reason : reason)
end

# ---------------------------------------------------------------------------
# CSLBFGS — limited-memory BFGS via `Optim.jl`.
# ---------------------------------------------------------------------------

"""
    CSLBFGS(; iterations, gradient_tolerance, m, show_trace, log,
              cost_rtol, cost_rtol_improvements, stall_evaluations,
              max_evaluations, callback)

Limited-memory BFGS (L-BFGS) optimizer backed by
[`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl). `m` is the
history length; the line search is Optim's `LBFGS` default
(Hager–Zhang). `iterations` is the max outer iteration count;
`gradient_tolerance` is Optim's L-infinity stopping threshold on the
χ-space / x-space gradient. The same L-infinity norm is recorded in
`gradient_norm_history` and `CSIterationLogEntry.gradient_norm` for this
backend. `show_trace = true` forwards verbose
progress to Optim's logger. `log = true` records one
[`CSIterationLogEntry`](@ref) for the initial state (iteration 0) and
one per accepted iterate.

Stop rules (shared with [`CSGradientDescent`](@ref); all off by default).
They are checked after every cost evaluation, line-search trials included,
against the best (lowest) cost evaluated so far, ``J_\\mathrm{best}``:

- `cost_rtol` (default `0`, off): an evaluation that lowers the best cost
  from ``J_\\mathrm{best}`` to ``J`` is a small improvement when
  ``(J_\\mathrm{best} - J)/|J_\\mathrm{best}| <`` `cost_rtol`. The solve stops
  after `cost_rtol_improvements` (default `2`) consecutive small improvements
  (stop reason `:cost_rtol`); a larger improvement resets the count, and
  evaluations that do not lower the best cost leave it unchanged.
- `stall_evaluations` (default `0`, off): stop after this many consecutive
  evaluations that do not lower the best cost (`:stall`). Near the optimum a
  Float32 cost has a noise floor, and the line search can keep evaluating
  without finding a decrease; this rule ends it. Every trial counts, also
  ordinary backtracking (each step halving of `CSGradientDescent`, an
  overshooting first L-BFGS line search), so choose it above the number of
  trials a healthy line search needs (the flux OSSE used 6 with L-BFGS).
- `max_evaluations` (default `0`, no cap): run at most this many cost
  evaluations, the initial one included (`:evaluations`). `iterations`
  caps accepted iterates; this caps the work. The cap is checked when
  another evaluation is requested, so a solve that ends on its own at the
  last allowed evaluation reports its own reason.
- `callback` (default `nothing`): `callback(evaluation::Int,
  result::CS4DVarResult) -> Bool`, called after every evaluation with its
  1-based index; returning `true` stops the solve (`:callback`). The
  result's controls are working storage that the next evaluation
  overwrites; copy what you keep.

A rule that fires stops the solve before its next evaluation, so no
evaluation is wasted: at an accepted iterate Optim stops through its
callback; inside a line search the wrapper interrupts Optim and returns the
last accepted iterate. Either way the result's `best` holds the
lowest-cost evaluation and `stop_reason` names the rule (see
[`CS4DVarSolveResult`](@ref)).

Used by the polymorphic dispatch surface
[`cs_surface_flux_4dvar_solve`](@ref). The cost closure
`cost_fn(controls) -> CS4DVarResult` is handed to Optim as one
objective with value, gradient and combined value-and-gradient methods,
flattening each control's `NTuple{6, Matrix}` to a `Vector{FT}` and
back. Each `cost_fn` call is a forward run plus the adjoint, so all
three methods share a cache of the last evaluation: `cost_fn` runs
once per point Optim visits, and the initial point and the returned
result reuse those evaluations.
"""
struct CSLBFGS{FT <: AbstractFloat, R <: _CSStopRules} <: AbstractCSOptimizer
    iterations::Int
    gradient_tolerance::FT
    m::Int
    show_trace::Bool
    log::Bool
    stop_rules::R

    function CSLBFGS(iterations::Integer, gradient_tolerance::FT,
                      m::Integer, show_trace::Bool,
                      log::Bool, stop_rules::R) where {FT <: AbstractFloat, R <: _CSStopRules}
        iterations >= 0 || throw(ArgumentError(
            "CSLBFGS iterations must be non-negative, got $iterations"))
        gradient_tolerance >= 0 || throw(ArgumentError(
            "CSLBFGS gradient_tolerance must be non-negative, got " *
            "$gradient_tolerance"))
        m > 0 || throw(ArgumentError(
            "CSLBFGS m (L-BFGS history length) must be positive, got $m"))
        return new{FT, R}(Int(iterations), gradient_tolerance, Int(m),
                          show_trace, log, stop_rules)
    end
end

function CSLBFGS(; iterations::Integer = 100,
                  gradient_tolerance::Real = 1e-8,
                  m::Integer = 10,
                  show_trace::Bool = false,
                  log::Bool = false,
                  cost_rtol::Real = 0.0,
                  cost_rtol_improvements::Integer = 2,
                  stall_evaluations::Integer = 0,
                  max_evaluations::Integer = 0,
                  callback = nothing)
    FT = typeof(float(gradient_tolerance))
    return CSLBFGS(Int(iterations), FT(gradient_tolerance), Int(m),
                   show_trace, log,
                   _CSStopRules(cost_rtol, cost_rtol_improvements,
                                stall_evaluations, max_evaluations, callback))
end

# ---------- flatten / unflatten helpers --------------------------------------

# Flatten a `Vector{<:CSSurfaceFluxControl}` to a `Vector{FT}` for
# Optim's vector-based optimization API. Pre-allocated buffer is
# overwritten in place.
function _flatten_controls_into!(flat::AbstractVector{FT},
                                  controls) where FT
    idx = 1
    @inbounds for c in controls
        for p in 1:6
            v = c.value[p]
            n = length(v)
            @views flat[idx:idx + n - 1] .= vec(v)
            idx += n
        end
    end
    return flat
end

function _control_flat_length(controls)
    total = 0
    @inbounds for c in controls
        for p in 1:6
            total += length(c.value[p])
        end
    end
    return total
end

# Inverse of `_flatten_controls_into!`. Mutates the underlying panel
# matrices, so `controls` here is treated as a scratch vector whose
# `.value` matrices are reused across Optim cost evaluations.
function _unflatten_into_controls!(controls, flat::AbstractVector)
    idx = 1
    @inbounds for c in controls
        for p in 1:6
            v = c.value[p]
            n = length(v)
            @views copyto!(vec(v), flat[idx:idx + n - 1])
            idx += n
        end
    end
    return controls
end

# Mirror of `_flatten_controls_into!` for `Vector{NTuple{6, Matrix}}`
# (the shape of `CS4DVarResult.gradients`).
function _flatten_gradients_into!(flat::AbstractVector{FT},
                                   gradients) where FT
    idx = 1
    @inbounds for grad in gradients
        for p in 1:6
            g = grad[p]
            n = length(g)
            @views flat[idx:idx + n - 1] .= vec(g)
            idx += n
        end
    end
    return flat
end

# Independent copies of `controls` (scratch the optimizer may overwrite).
_copy_controls(controls) = [
    CSSurfaceFluxControl(c.window, ntuple(p -> copy(c.value[p]), 6);
                          background = c.background,
                          sigma = c.sigma)
    for c in controls
]

# `result` reported at `controls` (the same point, held in other storage).
_result_with_controls(result::CS4DVarResult{FT, A2}, controls) where {FT, A2} =
    CS4DVarResult{FT, A2}(result.cost, result.observation_cost, result.background_cost,
                          result.simulated, result.residuals, result.gradients,
                          result.gradient_by_name, controls, result.observations,
                          result.stencils)

# ---------- single-point evaluation cache -------------------------------------

# One `cost_fn` call returns the cost AND its gradient (one forward run, which
# records the tape, plus the reverse pass), so a value-only or gradient-only
# request costs as much as a combined one. L-BFGS with Optim's Hager–Zhang
# line search asks for the value and the slope at every point it visits
# (`value_gradient!`), so no evaluation of it could be a cheaper `CSCostOnly`
# one. `_CSCostCache` remembers the most recent evaluation keyed
# on the flat control vector: Optim's `f`, `g!` and `fg!` callbacks and the
# iteration callback read from it, and `cost_fn` runs only when the requested
# point differs from the cached one. One entry suffices: `NLSolversBase`
# requests value and gradient at the same point back to back, and Optim
# evaluates each accepted iterate last, just before its iteration callback.
#
# `working_controls` is the scratch control vector `cost_fn` reads. The
# cached result's `controls` alias its matrices, and they always hold `x`,
# the cached point, because the cache only moves when `cost_fn` runs. Every
# evaluation passes through the stop `tracker`; `best_x` is the point of its
# best evaluation and `evaluation` the index of the cached one.
mutable struct _CSCostCache{FT, R, C, F, T}
    const cost_fn :: F
    const working_controls :: C
    const tracker :: T
    const x :: Vector{FT}
    const best_x :: Vector{FT}
    result :: R
    evaluation :: Int
end

# Thrown by `_evaluate!` when a stop rule forbids the evaluation Optim asks
# for (inside a line search); the backend catches it and returns.
struct _CSStopRequested <: Exception end

# The `CS4DVarResult` at the flat control `x`, evaluating `cost_fn` only on a
# cache miss. `==` matches `NLSolversBase`'s own cache test (a NaN never hits).
function _evaluate!(cache::_CSCostCache, x::AbstractVector)
    x == cache.x && return cache.result
    _may_evaluate!(cache.tracker) || throw(_CSStopRequested())
    _unflatten_into_controls!(cache.working_controls, x)
    cache.result = cache.cost_fn(cache.working_controls)
    copyto!(cache.x, x)
    _record_evaluation!(cache.tracker, cache.result) && copyto!(cache.best_x, x)
    cache.evaluation = cache.tracker.evaluations
    return cache.result
end

# Why Optim ended a solve that no stop rule ended (those are reported by the
# tracker). A failed line search is read from `stopped_by` first: it leaves the
# iterate in place, so Optim's termination code would call it `NoXChange`.
# `CSLBFGS` sets no time, call or trust-region limits and allows cost
# increases; any other code is reported as `:optim_<code>` rather than thrown
# away at the end of a long solve.
function _optim_stop_reason(opt_result)
    opt_result.stopped_by.ls_failed && return :line_search
    code = Optim.termination_code(opt_result)
    TC = Optim.TerminationCode
    code == TC.GradientNorm && return :gradient
    code == TC.Iterations && return :iterations
    code == TC.FailedLinesearch && return :line_search
    code in (TC.NoXChange, TC.NoObjectiveChange,
             TC.SmallXChange, TC.SmallObjectiveChange) && return :no_progress
    code in (TC.ObjectiveNotFinite, TC.GradientNotFinite) && return :non_finite
    return Symbol(:optim_, code)
end

# ---------- backend method ----------------------------------------------------

function cs_surface_flux_4dvar_solve(optimizer::CSLBFGS, cost_fn, controls)
    initial_controls = _control_vector(controls)
    isempty(initial_controls) && throw(ArgumentError(
        "cs_surface_flux_4dvar_solve(CSLBFGS, ...) requires at least one control"))

    # Working scratch — copy the user's input so Optim's inner
    # iterations don't mutate the original control matrices.
    working_controls = _copy_controls(initial_controls)

    # The first evaluation fixes the numerical type and seeds the cache, so
    # Optim's own evaluation at `x0` reuses it.
    probe = cost_fn(working_controls)
    isempty(probe.gradients) && throw(ArgumentError(
        "cost_fn returned a CS4DVarResult with no gradients; " *
        "cannot derive optimizer element type"))
    FT = eltype(probe.gradients[1][1])
    tracker = _CSStopTracker(optimizer.stop_rules, probe)

    n_total = _control_flat_length(working_controls)
    x0 = Vector{FT}(undef, n_total)
    _flatten_controls_into!(x0, initial_controls)
    cache = _CSCostCache(cost_fn, working_controls, tracker, copy(x0), copy(x0), probe, 1)

    # Value, gradient and value-and-gradient all come from one cached
    # evaluation per distinct point.
    f = x -> FT(_evaluate!(cache, x).cost)
    g! = function (G, x)
        _flatten_gradients_into!(G, _evaluate!(cache, x).gradients)
        return G
    end
    fg! = function (G, x)
        result = _evaluate!(cache, x)
        _flatten_gradients_into!(G, result.gradients)
        return FT(result.cost)
    end
    objective = Optim.OnceDifferentiable(f, g!, fg!, x0, zero(FT))

    # Optim's native L-BFGS convergence and trace use L∞. Stopping, history,
    # and optional logs all consume that same value at the initial state and
    # every accepted iterate, preserving Optim's established tolerance meaning.
    log = optimizer.log ? CSIterationLog{FT}() : nothing
    gradient_tolerance = FT(optimizer.gradient_tolerance)
    cost_history = FT[]
    gradient_norm_history = FT[]
    accepted = Ref(probe)          # evaluation at the last accepted iterate
    accepted_x = copy(x0)
    t0 = time()
    # Optim calls this at the initial state and after every accepted iterate
    # (the same points its trace records). The accepted iterate is the cached
    # point, so the lookup does not run `cost_fn`; it recovers the
    # observation/background split that Optim's state does not carry.
    # Returning `true` stops Optim at this iterate once a stop rule has fired.
    callback = function (state)
        r = _evaluate!(cache, state.x)
        # Optim's L-BFGS convergence check and trace use the L∞ norm of `g_x`.
        g_norm = FT(maximum(abs, state.g_x))
        push!(cost_history, FT(state.f_x))
        push!(gradient_norm_history, g_norm)
        accepted[] = r
        copyto!(accepted_x, state.x)
        if log !== nothing
            push!(log.entries, CSIterationLogEntry{FT}(
                state.pseudo_iteration, cache.evaluation, FT(r.cost),
                FT(r.observation_cost), FT(r.background_cost),
                g_norm, zero(FT), time() - t0))
        end
        return _stopped(tracker)
    end

    # `Optim.LBFGS(m = ...)` selects the L-BFGS algorithm with `m`
    # history terms. The default line search is Hager–Zhang.
    options = Optim.Options(iterations = optimizer.iterations,
                            g_abstol = gradient_tolerance,
                            show_trace = optimizer.show_trace,
                            callback = callback)
    iterations, reason = try
        opt_result = Optim.optimize(objective, x0, Optim.LBFGS(m = optimizer.m), options)
        # Optim ≥ 2.1 (allowing cost increases, its default) returns its last
        # accepted iterate, also after a failed line search.
        isequal(Optim.minimizer(opt_result), accepted_x) || error(
            "Optim returned a minimizer other than its last accepted iterate")
        (Optim.iterations(opt_result),
         _stopped(tracker) ? tracker.reason : _optim_stop_reason(opt_result))
    catch err
        # A stop rule refused an evaluation inside a line search.
        err isa _CSStopRequested || rethrow()
        (length(cost_history) - 1, tracker.reason)
    end

    # The line search may have moved the cache past the last accepted
    # iterate; `working_controls` (which `accepted[].controls` alias) get
    # that point back without another evaluation.
    _unflatten_into_controls!(working_controls, accepted_x)
    best = _result_with_controls(
        tracker.best, _unflatten_into_controls!(_copy_controls(working_controls), cache.best_x))

    final = accepted[]
    A2 = typeof(final.gradients[1][1])
    return CS4DVarSolveResult{FT, A2}(
        working_controls,
        final,
        cost_history,
        gradient_norm_history,
        FT[],      # Optim does not expose the L-BFGS step size
        iterations,
        log,
        best,
        tracker.evaluations,
        reason)
end

# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

"""
    cs_surface_flux_4dvar_optimize(..., observations, controls;
                                    optimizer = CSGradientDescent(...),
                                    tape = AutoTape(), kwargs...)
        -> CS4DVarSolveResult

Run an optimization loop around [`cs_surface_flux_4dvar`](@ref).

`optimizer::AbstractCSOptimizer` selects the backend; configure descent and
logging on that object. The default uses the model precision for its step
settings, including `min_step = sqrt(eps(FT))`.

`tape` (default `AutoTape()`) resolves to one
[`AdjointTape`](@ref AtmosTransport.Tape.AdjointTape), shared by every evaluation: they all record into one tape pool, opened by the first
and rewound by each later one (an mmap file is reserved and its bytes counted
once), and released when the optimization returns.

Remaining keyword arguments are forwarded to `cs_surface_flux_4dvar`
on every cost evaluation — including `preconditioner = ...` for the
preconditioned-cost path. Every evaluation computes the cost and the gradient,
so `evaluation` is not accepted.
"""
function cs_surface_flux_4dvar_optimize(panels_rm0, panels_m0,
                                        panels_am_steps,
                                        panels_bm_steps,
                                        panels_cm_steps,
                                        mesh::CubedSphereMesh,
                                        observations,
                                        controls;
                                        optimizer::AbstractCSOptimizer = CSGradientDescent(;
                                            initial_step = one(eltype(panels_rm0[1])),
                                            min_step = sqrt(eps(eltype(panels_rm0[1]))),
                                            step_shrink = eltype(panels_rm0[1])(0.5),
                                            gradient_tolerance = zero(eltype(panels_rm0[1]))),
                                        tape::Union{AdjointTape, AutoTape} = AutoTape(),
                                        kwargs...)
    haskey(kwargs, :evaluation) && throw(ArgumentError(
        "cs_surface_flux_4dvar_optimize evaluates the cost and its gradient at every point; " *
        "do not pass `evaluation`"))
    return _with_tape_runtime(tape) do runtime
        cost_fn = function (active_controls)
            return _cs_surface_flux_4dvar(
                runtime, panels_rm0, panels_m0, panels_am_steps, panels_bm_steps, panels_cm_steps,
                mesh, observations, active_controls; kwargs...)
        end
        cs_surface_flux_4dvar_solve(optimizer, cost_fn, controls)
    end
end
