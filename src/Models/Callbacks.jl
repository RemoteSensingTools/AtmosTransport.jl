# ===========================================================================
# Schedules and callbacks of a driven simulation, after Oceananigans'
# `Callback(func, schedule)`. A schedule is callable, `schedule(sim) -> Bool`;
# `DrivenSimulation` calls every entry of its `callbacks` after each step, and
# a `Callback` runs its function only when its schedule actuates. Plain
# functions in `callbacks` still run after every step.
# ===========================================================================

"""
    AbstractSchedule

When a [`Callback`](@ref) runs: a callable `schedule(sim) -> Bool`, evaluated
after every step of a `DrivenSimulation`.
"""
abstract type AbstractSchedule end

"""
    IterationInterval(interval)

Actuates after every `interval`-th step of the simulation (iterations
`interval`, `2 interval`, …).
"""
struct IterationInterval <: AbstractSchedule
    interval :: Int
    function IterationInterval(interval::Integer)
        !(interval isa Bool) && interval >= 1 ||
            throw(ArgumentError("IterationInterval needs an integer interval ≥ 1; got $(repr(interval))"))
        return new(Int(interval))
    end
end

(schedule::IterationInterval)(sim) = sim.iteration % schedule.interval == 0

"""
    WindowInterval(interval)

Actuates at the end of every `interval`-th met window this schedule has seen.
The count lives in the schedule, so a callback reused by the simulations of
consecutive input files (as the runner does) counts windows over the whole
run. Give each `Callback` its own schedule.
"""
mutable struct WindowInterval <: AbstractSchedule
    interval     :: Int
    windows_seen :: Int
    function WindowInterval(interval::Integer)
        !(interval isa Bool) && interval >= 1 ||
            throw(ArgumentError("WindowInterval needs an integer interval ≥ 1; got $(repr(interval))"))
        return new(Int(interval), 0)
    end
end

function (schedule::WindowInterval)(sim)
    sim.iteration == sim.current_window_end_iteration || return false
    schedule.windows_seen += 1
    return schedule.windows_seen % schedule.interval == 0
end

"""
    WallTimeInterval(seconds)

Actuates at the first step at least `seconds` of wall time (monotonic clock)
after it last actuated, or after it was constructed.
"""
mutable struct WallTimeInterval <: AbstractSchedule
    interval :: Float64
    previous :: Float64
    function WallTimeInterval(seconds::Real)
        !(seconds isa Bool) && seconds > 0 ||
            throw(ArgumentError("WallTimeInterval needs seconds > 0; got $(repr(seconds))"))
        return new(Float64(seconds), _monotonic_seconds())
    end
end

_monotonic_seconds() = time_ns() * 1e-9

function (schedule::WallTimeInterval)(sim)
    now = _monotonic_seconds()
    now - schedule.previous >= schedule.interval || return false
    schedule.previous = now
    return true
end

"""
    Callback(func, schedule = IterationInterval(1))

Run `func(sim)` after a step of a `DrivenSimulation` whenever `schedule`
actuates. Pass callbacks to `DrivenSimulation(...; callbacks = (; name = cb))`.
"""
struct Callback{F, S <: AbstractSchedule}
    func     :: F
    schedule :: S
end

Callback(func) = Callback(func, IterationInterval(1))

function (callback::Callback)(sim)
    callback.schedule(sim) && callback.func(sim)
    return nothing
end

"""
    NaNChecker()

Callback function that stops the run with an error when the air mass or a
tracer holds a NaN or Inf (interior cells only). It runs `mapreduce` on the
state's backend, without copying fields to the host: one reduction for the air
mass and one for all tracers on structured grids, one per panel for each on the
cubed sphere (about 0.1 ms each on an L40S). The run config installs it as
`Callback(NaNChecker(), WindowInterval(n))` (`[checks]`).
"""
struct NaNChecker end

_has_nonfinite(field::AbstractArray) = mapreduce(!isfinite, |, field; init = false)
_has_nonfinite(fields::Tuple) = any(_has_nonfinite, fields)

# Interior views, without the cubed-sphere halos (which are not prognostic).
_interior_view(panel::AbstractArray{<:Any, 3}, Hp) =
    view(panel, Hp + 1:size(panel, 1) - Hp, Hp + 1:size(panel, 2) - Hp, :)
_interior_view(panel::AbstractArray{<:Any, 4}, Hp) =
    view(panel, Hp + 1:size(panel, 1) - Hp, Hp + 1:size(panel, 2) - Hp, :, :)

_checked_air_mass(state::CellState) = state.air_mass
_checked_air_mass(state::CubedSphereState) =
    map(panel -> _interior_view(panel, halo_width(state)), state.air_mass)
_checked_tracers(state::CellState) = state.tracers_raw
_checked_tracers(state::CubedSphereState) =
    map(panel -> _interior_view(panel, halo_width(state)), state.tracers_raw)
# Tracer `i` alone, over the same cells.
_checked_tracer(state::CellState, i) = selectdim(state.tracers_raw, ndims(state.tracers_raw), i)
_checked_tracer(state::CubedSphereState, i) =
    map(panel -> view(panel, :, :, :, i), _checked_tracers(state))

function (::NaNChecker)(sim)
    state = sim.model.state
    _has_nonfinite(_checked_air_mass(state)) && _nonfinite_error(sim, "the air mass")
    if _has_nonfinite(_checked_tracers(state))       # name the tracer only on failure
        names = tracer_names(state)
        i = findfirst(i -> _has_nonfinite(_checked_tracer(state, i)), eachindex(names))
        _nonfinite_error(sim, i === nothing ? "a tracer" : "tracer $(names[i])")
    end
    return nothing
end

_nonfinite_error(sim, what) = error(
    "NaN or Inf in $(what) after met window $(sim.current_window_index) (iteration " *
    "$(sim.iteration), model time $(round(sim.time / 3600; digits = 2)) h); stopping the run. " *
    "Check the transport binary (scripts/diagnostics/inspect_transport_binary.jl) and the " *
    "physics settings.")
