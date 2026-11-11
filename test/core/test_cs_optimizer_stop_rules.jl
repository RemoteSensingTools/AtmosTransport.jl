#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Stop rules of the CS 4D-Var optimizers (`CSLBFGS`, `CSGradientDescent`):
# `cost_rtol` / `cost_rtol_improvements`, `stall_evaluations`,
# `max_evaluations` and `callback`, the best-evaluation result and the stop
# reason.
#
# Coverage:
#   * Construction rejects invalid rules; the defaults turn every rule off and
#     equal passing the off values explicitly. (That the default path is
#     bitwise unchanged against the code before the stop rules was checked
#     outside the suite, by solving the same problems on both.)
#   * Each rule stops at the evaluation an independent replay of the rule on
#     the recorded cost sequence names, runs no further evaluation, and leaves
#     the path before the stop unchanged (the costs are a prefix of the
#     unstopped run's).
#   * `best` is the lowest-cost evaluation (point, cost and its
#     observation/background split); `last`/`controls` are the last accepted
#     iterate, also when L-BFGS is interrupted inside a line search.
#   * The iteration log's `evaluation` indices point at the evaluations that
#     computed each iterate. L-BFGS stops both at an accepted iterate (through
#     Optim's callback) and inside a line search (interrupted).
#   * A failed L-BFGS line search is reported as `:line_search`.
#   * A quadratic with pseudo-noise in the cost (a Float32-like noise floor)
#     keeps L-BFGS's line search evaluating for thousands of evaluations
#     without the stall rule and stops after a few with it.
# ---------------------------------------------------------------------------

using Test

import AtmosTransport
const AT = AtmosTransport

# A separable quadratic over 6 panels of Nc×Nc controls with minimum `floor`
# (a 4D-Var cost does not reach zero), with optional deterministic
# pseudo-noise of relative size `eta` in the cost only (the gradient stays
# exact), as near the optimum of a Float32 transport cost. The floor and the
# first half of the controls are the "observation" term, the second half plus
# the noise the "background" term. Every call records its point and cost.
function quadratic_problem(; FT = Float64, eta = 0.0, Nc = 2, floor = 100)
    n = 6 * Nc^2
    a = FT[1 + 9 * (i - 1) / (n - 1) for i in 1:n]
    target = FT[sin(FT(0.7) * i) for i in 1:n]
    control = AT.CSSurfaceFluxControl(AT.CSSurfaceFluxWindow(:w, 1),
                                      ntuple(_ -> zeros(FT, Nc, Nc), 6))
    points = Vector{Vector{FT}}()
    costs = FT[]
    cost_fn = function (controls)
        controls = controls isa AbstractVector ? controls : [controls]
        x = reduce(vcat, [vec(controls[1].value[p]) for p in 1:6])
        terms = FT(0.5) .* a .* (x .- target) .^ 2
        noise = FT(eta) * (floor + sum(terms)) * FT(sin(1.0e4 * sum(Float64.(x) .* (1:n))))
        observation_cost = floor + sum(terms[1:n ÷ 2])
        background_cost = sum(terms[n ÷ 2 + 1:end]) + noise
        cost = observation_cost + background_cost
        g = a .* (x .- target)
        gradient = ntuple(p -> reshape(g[(p - 1) * Nc^2 + 1:p * Nc^2], Nc, Nc), 6)
        push!(points, copy(x))
        push!(costs, cost)
        return AT.CS4DVarResult(cost, observation_cost, background_cost, FT[], FT[],
                                [gradient], Dict(:w => gradient), collect(controls),
                                AT.CSObservation[], AT.CSObservationStencil[])
    end
    return (; cost_fn, control, points, costs)
end

flat(controls) = reduce(vcat, [vec(c.value[p]) for c in controls for p in 1:6])

# Independent replay of the rules on a cost sequence: the 1-based evaluation
# at which the first rule fires (and the rule), or `nothing`.
function replay_rules(costs; cost_rtol = 0.0, cost_rtol_improvements = 2,
                      stall_evaluations = 0, max_evaluations = 0)
    best = costs[1]
    small = 0
    stalled = 0
    for k in 2:length(costs)
        if costs[k] < best
            small = (best - costs[k]) / abs(best) < cost_rtol ? small + 1 : 0
            best = costs[k]
            stalled = 0
            small >= cost_rtol_improvements && return (k, :cost_rtol)
        else
            stalled += 1
            stall_evaluations > 0 && stalled >= stall_evaluations && return (k, :stall)
        end
        max_evaluations > 0 && k >= max_evaluations && return (k, :evaluations)
    end
    return nothing
end

# What every solve must satisfy, whatever ended it.
function check_solve_invariants(solve, problem)
    @test solve.evaluations == length(problem.costs)
    kbest = argmin(problem.costs)                  # first minimum on ties
    @test solve.best.cost == problem.costs[kbest]
    @test flat(solve.best.controls) == problem.points[kbest]
    @test solve.best.observation_cost + solve.best.background_cost == solve.best.cost
    # `last`/`controls`: the last accepted iterate.
    @test solve.last.cost == solve.cost_history[end]
    @test flat(solve.controls) == flat(solve.last.controls)
    @test solve.best.cost <= solve.last.cost
    if solve.log !== nothing
        evaluations = [entry.evaluation for entry in solve.log]
        @test evaluations[1] == 1
        @test issorted(evaluations; lt = <=)       # strictly increasing
        @test evaluations[end] <= solve.evaluations
        @test [entry.cost for entry in solve.log] == problem.costs[evaluations]
        @test flat(solve.controls) == problem.points[evaluations[end]]
    end
end

const OPTIMIZERS = (
    lbfgs = (; kw...) -> AT.CSLBFGS(; iterations = 40, gradient_tolerance = 0.0, m = 5,
                                    log = true, kw...),
    gradient_descent = (; kw...) -> AT.CSGradientDescent(; iterations = 60, initial_step = 0.15,
                                                         log = true, kw...),
)

solve_with(make, problem; kw...) =
    AT.cs_surface_flux_4dvar_solve(make(; kw...), problem.cost_fn, problem.control)

@testset "stop rules — construction" begin
    for make in OPTIMIZERS
        @test_throws ArgumentError make(cost_rtol = -1e-4)
        @test_throws ArgumentError make(cost_rtol_improvements = 0)
        @test_throws ArgumentError make(stall_evaluations = -1)
        @test_throws ArgumentError make(max_evaluations = -1)
        rules = make().stop_rules
        @test rules.cost_rtol == 0
        @test rules.stall_evaluations == 0
        @test rules.max_evaluations == 0
        @test rules.callback === nothing
    end
end

@testset "stop rules — defaults equal the explicit off values ($name)" for (name, make) in pairs(OPTIMIZERS)
    a = quadratic_problem(eta = 1e-5)
    b = quadratic_problem(eta = 1e-5)
    default = solve_with(make, a)
    off = solve_with(make, b; cost_rtol = 0.0, stall_evaluations = 0, max_evaluations = 0,
                     callback = nothing)
    @test a.costs == b.costs && a.points == b.points
    @test flat(default.controls) == flat(off.controls)
    @test default.cost_history == off.cost_history
    @test default.gradient_norm_history == off.gradient_norm_history
    @test default.iterations == off.iterations
    @test default.stop_reason == off.stop_reason
    check_solve_invariants(default, a)
end

@testset "stop rules — each rule stops where its replay says ($name)" for (name, make) in pairs(OPTIMIZERS)
    reference = quadratic_problem(eta = 1e-5)
    unstopped = solve_with(make, reference)
    rules = (
        (cost_rtol = 1e-2,),
        (cost_rtol = 1e-2, cost_rtol_improvements = 3),
        (stall_evaluations = 3,),
        (max_evaluations = 7,),
        (cost_rtol = 1e-3, stall_evaluations = 4, max_evaluations = 30),
    )
    for kw in rules
        expected = replay_rules(reference.costs; kw...)
        @test expected !== nothing           # every rule fires on this problem
        problem = quadratic_problem(eta = 1e-5)
        solve = solve_with(make, problem; kw...)
        k, reason = expected
        @test solve.stop_reason == reason
        @test length(problem.costs) == k     # no evaluation after the one that fired
        @test problem.costs == reference.costs[1:k]
        @test solve.iterations == length(solve.cost_history) - 1
        check_solve_invariants(solve, problem)
    end
    check_solve_invariants(unstopped, reference)
end

@testset "stop rules — callback ($name)" for (name, make) in pairs(OPTIMIZERS)
    seen = Int[]
    problem = quadratic_problem()
    callback = function (evaluation, result)
        push!(seen, evaluation)
        @test result.cost == problem.costs[evaluation]
        return evaluation == 4
    end
    solve = solve_with(make, problem; callback = callback)
    @test seen == 1:4
    @test solve.stop_reason == :callback
    @test solve.evaluations == length(problem.costs) == 4
    check_solve_invariants(solve, problem)

    # A stop requested at the initial point: no step, one evaluation.
    problem = quadratic_problem()
    solve = solve_with(make, problem; callback = (evaluation, result) -> true)
    @test solve.stop_reason == :callback
    @test solve.iterations == 0
    @test length(problem.costs) == 1
    check_solve_invariants(solve, problem)
end

@testset "stop rules — the cap counts the initial evaluation" begin
    for make in OPTIMIZERS
        problem = quadratic_problem()
        solve = solve_with(make, problem; max_evaluations = 1)
        @test solve.stop_reason == :evaluations
        @test solve.iterations == 0
        @test length(problem.costs) == 1
        @test solve.best.cost == solve.last.cost == problem.costs[1]
    end
end

# L-BFGS stopped at an accepted iterate goes through Optim's callback: the last
# evaluation is that iterate, and `last` is it.
@testset "stop rules — L-BFGS stop at an accepted iterate" begin
    reference = quadratic_problem(eta = 1e-5)
    unstopped = solve_with(OPTIMIZERS.lbfgs, reference)
    k = unstopped.log[4].evaluation              # the third accepted iterate
    problem = quadratic_problem(eta = 1e-5)
    solve = solve_with(OPTIMIZERS.lbfgs, problem;
                       callback = (evaluation, result) -> evaluation == k)
    @test solve.stop_reason == :callback
    @test solve.evaluations == k
    @test solve.iterations == 3
    @test solve.log[end].evaluation == k
    @test solve.last.cost == reference.costs[k]
    check_solve_invariants(solve, problem)
end

@testset "stop rules — Float32 ($name)" for (name, make) in pairs(OPTIMIZERS)
    reference = quadratic_problem(FT = Float32, eta = 3e-6)
    solve_with(make, reference)
    expected = replay_rules(reference.costs; stall_evaluations = 3)
    problem = quadratic_problem(FT = Float32, eta = 3e-6)
    solve = solve_with(make, problem; stall_evaluations = 3)
    @test solve isa AT.CS4DVarSolveResult{Float32}
    @test (solve.evaluations, solve.stop_reason) == expected
    check_solve_invariants(solve, problem)
end

# A gradient tolerance met before any rule fires keeps Optim's own reason.
@testset "stop rules — optimizer reasons" begin
    problem = quadratic_problem()
    solve = AT.cs_surface_flux_4dvar_solve(
        AT.CSLBFGS(iterations = 100, gradient_tolerance = 1e-6, stall_evaluations = 50),
        problem.cost_fn, problem.control)
    @test solve.stop_reason == :gradient
    check_solve_invariants(solve, problem)

    problem = quadratic_problem()
    solve = AT.cs_surface_flux_4dvar_solve(
        AT.CSGradientDescent(iterations = 200, initial_step = 0.15, gradient_tolerance = 1e-3),
        problem.cost_fn, problem.control)
    @test solve.stop_reason == :gradient

    problem = quadratic_problem()
    solve = AT.cs_surface_flux_4dvar_solve(
        AT.CSGradientDescent(iterations = 5, initial_step = 1e3, min_step = 1.0),
        problem.cost_fn, problem.control)
    @test solve.stop_reason == :line_search
    @test solve.iterations == 0

    # Every point but the initial one has a non-finite cost: L-BFGS's line
    # search fails, and the solve keeps the initial point as last and best.
    problem = quadratic_problem()
    nonfinite = function (controls)
        result = problem.cost_fn(controls)
        length(problem.costs) == 1 && return result
        return AT.CS4DVarResult(NaN, NaN, 0.0, result.simulated, result.residuals,
                                result.gradients, result.gradient_by_name, result.controls,
                                result.observations, result.stencils)
    end
    solve = AT.cs_surface_flux_4dvar_solve(
        AT.CSLBFGS(iterations = 10, gradient_tolerance = 0.0), nonfinite, problem.control)
    @test solve.stop_reason == :line_search
    @test solve.last.cost == solve.best.cost == problem.costs[1]
    @test flat(solve.controls) == problem.points[1]
end

@testset "stop rules — the stall rule ends a line search at the cost noise floor" begin
    # Without the rule, the noise floor keeps the Hager–Zhang line search
    # evaluating: more than a thousand evaluations for at most 80 iterations.
    noisy = quadratic_problem(eta = 1e-5)
    unstopped = AT.cs_surface_flux_4dvar_solve(
        AT.CSLBFGS(iterations = 80, gradient_tolerance = 0.0, m = 5, log = true),
        noisy.cost_fn, noisy.control)
    @test unstopped.stop_reason != :stall
    @test unstopped.evaluations > 1000
    # At the noise floor a line-search trial beats the last accepted iterate.
    @test unstopped.best.cost < unstopped.last.cost
    check_solve_invariants(unstopped, noisy)

    problem = quadratic_problem(eta = 1e-5)
    solve = AT.cs_surface_flux_4dvar_solve(
        AT.CSLBFGS(iterations = 80, gradient_tolerance = 0.0, m = 5, log = true,
                   stall_evaluations = 6),
        problem.cost_fn, problem.control)
    k, reason = replay_rules(noisy.costs; stall_evaluations = 6)
    @test solve.stop_reason == reason == :stall
    @test solve.evaluations == k < 100
    @test problem.costs == noisy.costs[1:k]
    check_solve_invariants(solve, problem)
    # The stop lands at the noise floor: the best cost is within the noise
    # (1e-5 relative) of the best of the full run.
    @test (solve.best.cost - unstopped.best.cost) / unstopped.best.cost < 1e-4
    # Interrupted inside a line search: `last` is the last accepted iterate,
    # whose evaluation precedes the stall.
    @test solve.log[end].evaluation < solve.evaluations
end
