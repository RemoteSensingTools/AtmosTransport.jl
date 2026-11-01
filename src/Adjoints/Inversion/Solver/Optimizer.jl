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
"""
abstract type AbstractCSOptimizer end

"""
    CSGradientDescent(; iterations, initial_step, min_step,
                        step_shrink, gradient_tolerance, line_search,
                        log)

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
"""
struct CSGradientDescent{FT <: AbstractFloat} <: AbstractCSOptimizer
    iterations::Int
    initial_step::FT
    min_step::FT
    step_shrink::FT
    gradient_tolerance::FT
    line_search::Bool
    log::Bool

    function CSGradientDescent(iterations::Integer,
                                initial_step::FT,
                                min_step::FT,
                                step_shrink::FT,
                                gradient_tolerance::FT,
                                line_search::Bool,
                                log::Bool) where FT <: AbstractFloat
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
        return new{FT}(Int(iterations), initial_step, min_step, step_shrink,
                       gradient_tolerance, line_search, log)
    end
end

function CSGradientDescent(; iterations::Integer = 10,
                            initial_step::Real = 1.0,
                            min_step::Real = sqrt(eps(Float64)),
                            step_shrink::Real = 0.5,
                            gradient_tolerance::Real = 0.0,
                            line_search::Bool = true,
                            log::Bool = false)
    FT = promote_type(typeof(float(initial_step)),
                      typeof(float(min_step)),
                      typeof(float(step_shrink)),
                      typeof(float(gradient_tolerance)))
    return CSGradientDescent(Int(iterations),
                             FT(initial_step), FT(min_step),
                             FT(step_shrink), FT(gradient_tolerance),
                             line_search, log)
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
"""
function cs_surface_flux_4dvar_solve end

function cs_surface_flux_4dvar_solve(optimizer::CSGradientDescent,
                                      cost_fn, controls)
    current = cost_fn(controls)
    isempty(current.gradients) && throw(ArgumentError(
        "cost_fn returned a CS4DVarResult with no gradients; " *
        "cannot derive optimizer element type"))
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
            0, FT(current.cost), FT(current.observation_cost),
            FT(current.background_cost), grad_norm, zero(FT),
            time() - t0))
    end

    for iter in 1:optimizer.iterations
        grad_norm <= gradient_tolerance && break
        step = initial_step
        accepted = false
        candidate = nothing
        candidate_controls = nothing
        while step >= min_step
            candidate_controls = _gradient_step_controls(
                current.controls, current.gradients, step)
            candidate = cost_fn(candidate_controls)
            if !optimizer.line_search || candidate.cost <= current.cost
                accepted = true
                break
            end
            step *= step_shrink
        end
        accepted || break
        current = candidate
        grad_norm = FT(_control_gradient_norm(current.gradients))
        push!(cost_history, FT(current.cost))
        push!(gradient_norm_history, grad_norm)
        push!(step_history, step)
        if log !== nothing
            push!(log.entries, CSIterationLogEntry{FT}(
                iter, FT(current.cost), FT(current.observation_cost),
                FT(current.background_cost), grad_norm, step,
                time() - t0))
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
        log)
end

# ---------------------------------------------------------------------------
# CSLBFGS — limited-memory BFGS via `Optim.jl`.
# ---------------------------------------------------------------------------

"""
    CSLBFGS(; iterations, gradient_tolerance, m, show_trace, log)

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

Used by the polymorphic dispatch surface
[`cs_surface_flux_4dvar_solve`](@ref). The cost closure
`cost_fn(controls) -> CS4DVarResult` is handed to Optim as one
objective with value, gradient and combined value-and-gradient methods,
flattening each control's `NTuple{6, Matrix}` to a `Vector{FT}` and
back. Each `cost_fn` call is a forward run plus the adjoint, so all
three methods share a cache of the last evaluation: `cost_fn` runs
once per point Optim visits, and the initial point and the returned
result reuse those evaluations. (The result costs one more evaluation
only when Optim returns an earlier iterate, after a failed line search
or a cost increase.)
"""
struct CSLBFGS{FT <: AbstractFloat} <: AbstractCSOptimizer
    iterations::Int
    gradient_tolerance::FT
    m::Int
    show_trace::Bool
    log::Bool

    function CSLBFGS(iterations::Integer, gradient_tolerance::FT,
                      m::Integer, show_trace::Bool,
                      log::Bool) where FT <: AbstractFloat
        iterations >= 0 || throw(ArgumentError(
            "CSLBFGS iterations must be non-negative, got $iterations"))
        gradient_tolerance >= 0 || throw(ArgumentError(
            "CSLBFGS gradient_tolerance must be non-negative, got " *
            "$gradient_tolerance"))
        m > 0 || throw(ArgumentError(
            "CSLBFGS m (L-BFGS history length) must be positive, got $m"))
        return new{FT}(Int(iterations), gradient_tolerance, Int(m),
                       show_trace, log)
    end
end

function CSLBFGS(; iterations::Integer = 100,
                  gradient_tolerance::Real = 1e-8,
                  m::Integer = 10,
                  show_trace::Bool = false,
                  log::Bool = false)
    FT = typeof(float(gradient_tolerance))
    return CSLBFGS(Int(iterations), FT(gradient_tolerance), Int(m),
                   show_trace, log)
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

# ---------- single-point evaluation cache -------------------------------------

# One `cost_fn` call returns the cost AND its gradient (one forward run, which
# records the tape, plus the reverse pass), so a value-only or gradient-only
# request costs as much as a combined one. L-BFGS with Optim's Hager–Zhang
# line search asks for the value and the slope at every point it visits
# (`value_gradient!`), so no evaluation of it could be a cheaper `CSCostOnly`
# one. `_CSCostCache` remembers the most recent evaluation keyed
# on the flat control vector: Optim's `f`, `g!` and `fg!` callbacks, the
# iteration-log callback and the final result all read from it, and
# `cost_fn` runs only when the requested point differs from the cached one.
# One entry suffices: `NLSolversBase` requests value and gradient at the same
# point back to back, and Optim evaluates each accepted iterate last, just
# before its iteration callback.
#
# `working_controls` is the scratch control vector `cost_fn` reads. The
# cached result's `controls` alias its matrices, and they always hold `x`,
# the cached point, because the cache only moves when `cost_fn` runs.
mutable struct _CSCostCache{FT, R, C, F}
    const cost_fn :: F
    const working_controls :: C
    const x :: Vector{FT}
    result :: R
end

# The `CS4DVarResult` at the flat control `x`, evaluating `cost_fn` only on a
# cache miss. `==` matches `NLSolversBase`'s own cache test (a NaN never hits).
function _evaluate!(cache::_CSCostCache, x::AbstractVector)
    x == cache.x && return cache.result
    _unflatten_into_controls!(cache.working_controls, x)
    cache.result = cache.cost_fn(cache.working_controls)
    copyto!(cache.x, x)
    return cache.result
end

# ---------- backend method ----------------------------------------------------

function cs_surface_flux_4dvar_solve(optimizer::CSLBFGS, cost_fn, controls)
    initial_controls = _control_vector(controls)
    isempty(initial_controls) && throw(ArgumentError(
        "cs_surface_flux_4dvar_solve(CSLBFGS, ...) requires at least one control"))

    # Working scratch — copy the user's input so Optim's inner
    # iterations don't mutate the original control matrices.
    working_controls = [
        CSSurfaceFluxControl(c.window, ntuple(p -> copy(c.value[p]), 6);
                              background = c.background,
                              sigma = c.sigma)
        for c in initial_controls
    ]

    # The first evaluation fixes the numerical type and seeds the cache, so
    # Optim's own evaluation at `x0` reuses it.
    probe = cost_fn(working_controls)
    isempty(probe.gradients) && throw(ArgumentError(
        "cost_fn returned a CS4DVarResult with no gradients; " *
        "cannot derive optimizer element type"))
    FT = eltype(probe.gradients[1][1])

    n_total = _control_flat_length(working_controls)
    x0 = Vector{FT}(undef, n_total)
    _flatten_controls_into!(x0, initial_controls)
    cache = _CSCostCache(cost_fn, working_controls, copy(x0), probe)

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
    t0 = time()
    callback = log === nothing ? nothing : function (state)
        # Optim's `OptimizationState` for L-BFGS exposes
        # `pseudo_iteration`, `f_x`, `g_x` etc. — different field
        # names from the `OptimizationTrace` entry returned by
        # `opt_result.trace[k]`. We project to the same shape as
        # CSGradientDescent's log entries.
        #
        # The accepted iterate is the cached point, so this lookup does
        # not run `cost_fn`; it recovers the observation/background split
        # that Optim's state does not carry.
        #
        # Optim's L-BFGS convergence check uses the L∞ norm of `g_x`.
        # The callback state does not expose the derived `g_norm`, so compute
        # the identical value directly from its flat gradient.
        r = _evaluate!(cache, state.x)
        g_norm = FT(maximum(abs, state.g_x))
        push!(log.entries, CSIterationLogEntry{FT}(
            state.pseudo_iteration, FT(r.cost),
            FT(r.observation_cost), FT(r.background_cost),
            g_norm, zero(FT), time() - t0))
        return false
    end

    # `Optim.LBFGS(m = ...)` selects the L-BFGS algorithm with `m`
    # history terms. The default line search is Hager–Zhang.
    # `store_trace` powers `cost_history` / `gradient_norm_history`.
    options = callback === nothing ?
        Optim.Options(iterations = optimizer.iterations,
                      g_abstol = gradient_tolerance,
                      show_trace = optimizer.show_trace,
                      store_trace = true) :
        Optim.Options(iterations = optimizer.iterations,
                      g_abstol = gradient_tolerance,
                      show_trace = optimizer.show_trace,
                      store_trace = true,
                      callback = callback)
    opt_result = Optim.optimize(objective, x0, Optim.LBFGS(m = optimizer.m), options)

    # The result at the minimizer, normally the cached last evaluation (the
    # accepted iterate). If Optim stops on a failed line search, or returns
    # the previous iterate because the last step increased the cost, this
    # costs one more evaluation. Either way `working_controls` then holds
    # the minimizer.
    final = _evaluate!(cache, Optim.minimizer(opt_result))

    # Pull per-iteration cost + gradient-norm history out of the Optim
    # trace. Optim doesn't separately track step size for L-BFGS, so
    # `step_history` stays empty.
    trace_costs = FT[FT(state.value) for state in opt_result.trace]
    trace_gnorms = FT[FT(state.g_norm) for state in opt_result.trace]

    A2 = typeof(final.gradients[1][1])
    return CS4DVarSolveResult{FT, A2}(
        working_controls,
        final,
        trace_costs,
        trace_gnorms,
        FT[],
        Optim.iterations(opt_result),
        log)
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
                                        optimizer::AbstractCSOptimizer = CSGradientDescent(
                                            10, one(eltype(panels_rm0[1])),
                                            sqrt(eps(eltype(panels_rm0[1]))),
                                            eltype(panels_rm0[1])(0.5),
                                            zero(eltype(panels_rm0[1])), true, false),
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
