# DrivenSimulation: window advance, the constructor, step!, run_window! and run!.
# Split from DrivenSimulation.jl (refactor phase 4); included by Models.jl in this order.

function _load_window(driver::D, win::Int) where {D <: AbstractMetDriver}
    return load_transport_window(driver, win)
end

function _maybe_advance_window!(sim::DrivenSimulation)
    if sim.iteration > 0 && sim.iteration == sim.current_window_end_iteration
        next_window = sim.current_window_index + 1
        next_window <= sim.stop_window ||
            throw(ArgumentError("DrivenSimulation attempted to step past stop_window=$(sim.stop_window)"))
        sim.current_window_start_iteration = sim.iteration
        sim.current_window_index = next_window
        sim.steps_per_window = sim.steps_per_window_schedule[next_window]
        sim.current_window_end_iteration = sim.iteration + sim.steps_per_window
        sim.Δt = window_step_dt(sim.window_dt, sim.steps_per_window)
        _take_prefetched_window!(sim, next_window)
        _stage_surface_flux_window!(sim)
        # The plan's window-advance stages: the air-mass reset to the window
        # just entered (`air_mass_reset_mode`).
        execute_window_advance!(sim.plan, step_operators(sim.model), _step_timing(sim), sim)
        if sim.qv_buffer !== nothing && !has_humidity_endpoints(sim.window)
            throw(ArgumentError("driver humidity endpoint support changed between windows"))
        end
        _validate_convection_runtime(sim.model, sim.driver, sim.window)
        _refresh_dz_for_window!(sim)
        _refresh_pbl_kz_for_window!(sim.model.diffusion, sim)
        # Under the canonical `:window_constant` contract, the runtime's own
        # flux divergence should integrate to `(m_next - m)` over each window.
        # `air_mass_reset_mode` controls whether the binary endpoint is still
        # treated as authoritative at window boundaries.
        invalidate_cmfmc_cache!(sim.model.workspace.convection_ws)
        _start_window_prefetch!(sim, next_window + 1)
    end
    return nothing
end

# The air-mass reset targets of the plan's `AirMassResetStage`s: the air mass
# of the window just entered, and the binary's authoritative window-end mass.
air_mass_reset_target(::WindowStartAirMass, sim::DrivenSimulation) = sim.window.air_mass

function air_mass_reset_target(::WindowEndAirMass, sim::DrivenSimulation)
    expected_air_mass!(sim.expected_air_mass, sim.window, one(typeof(sim.Δt)))
    return sim.expected_air_mass
end

# The time steps of the current step: the advection step and its met window.
_step_timing(sim::DrivenSimulation) = StepTiming(sim.Δt, sim.window_dt)

"""
    DrivenSimulation(model, driver; kwargs...)

Construct a window-driven `src` runtime.

Keyword arguments:
- `start_window=1`
- `stop_window=total_windows(driver)`
- `initialize_air_mass=true`
- `use_midpoint_forcing=true`
- `interpolate_fluxes_within_window=nothing` (derive from driver)
- `air_mass_reset_mode=:preserve_tracer_mass` — one of `:none`, `:preserve_vmr`, or
  `:preserve_tracer_mass`. When non-`:none`, each newly loaded window
  replaces prognostic air mass using the selected tracer invariant. For
  binary-scheduled runs, the same endpoint reset is applied before the
  once-per-window convection and closing chemistry half-step so physics sees
  the binary's authoritative window-end mass.
- `physics_cadence=nothing` — `:window` (the default for `nothing`) or
  `:substep`. On a binary with a per-window physics contract, `:window` runs
  convection once per met window and splits exponential decay around all
  stored advection/emission substeps; `:substep` runs the full composition
  every advection substep (the cadence before 2026-05-31), for
  cadence-sensitivity comparisons on the same binary. The window-end air-mass
  reset is the same for both. Other binaries run the full operator suite every
  substep either way. The resulting [`StepPlan`](@ref) is `sim.plan`; the
  cubed-sphere adjoint replays the same plan (`CSStepSchedule`).
- `prefetch_windows=true` — on a device backend with more than one Julia
  thread, load the next met window on a second thread while the current one
  runs. `false` loads each window when it is needed and keeps one device
  window buffer instead of two. Window contents are the same either way.
- `surface_sources=()`
- `chemistry=NoChemistry()` — exponential decay is applied in exact half-window
  steps before and after the window's transport + surface sources; other
  chemistry operators run after transport and convection
- `callbacks=NamedTuple()`
- `reuse_buffers_of=nothing` — a previous `DrivenSimulation` of the same model,
  typically that of the preceding daily binary. If its driver has the same
  window layout (`same_window_layout`) and it kept a host staging window, its
  device windows, host staging window, expected-air-mass and humidity buffers
  and the model's convection forcing are refilled in place instead of
  allocated, and the startup trim of the device memory pool (a garbage
  collection and pool reclaim) is skipped. Its prefetch is finished first.
  Otherwise new buffers are allocated as without it. Window contents are the
  same either way.
- `start_time=0` — simulation clock origin [s]. Multi-binary runners MUST pass
  the accumulated run time here when rebuilding the sim per binary: `sim.time`
  feeds `current_time(meteo)`, which time-varying surface-flux sources use to
  select their emission slice (seconds since the RUN start, not the binary
  start). Restarting the clock at 0 each day silently replays day-1 fluxes —
  the December-2021 co2_natural +1 Pg/month surplus (plan 45 Stage-4 A/B
  experiment attributed the leak to exactly this).
"""
function DrivenSimulation(model::TransportModel,
                          driver::D;
                          start_window::Integer = 1,
                          stop_window::Integer = total_windows(driver),
                          initialize_air_mass::Bool = true,
                          use_midpoint_forcing::Bool = true,
                          interpolate_fluxes_within_window = nothing,
                          air_mass_reset_mode = :preserve_tracer_mass,
                          physics_cadence = nothing,
                          prefetch_windows::Bool = true,
                          surface_sources = (),
                          chemistry::AbstractChemistryOperator = NoChemistry(),
                          callbacks = NamedTuple(),
                          reuse_buffers_of::Union{Nothing, DrivenSimulation} = nothing,
                          start_time::Real = 0) where {D <: AbstractMetDriver}
    1 <= start_window <= stop_window <= total_windows(driver) ||
        throw(ArgumentError("invalid window range: start_window=$(start_window), stop_window=$(stop_window), total_windows=$(total_windows(driver))"))
    supports_native_vertical_flux(driver) ||
        throw(ArgumentError("DrivenSimulation requires native vertical mass fluxes in the met-driver contract"))
    isfinite(start_time) ||
        throw(ArgumentError("DrivenSimulation start_time must be finite; got $(start_time)"))

    _check_grid_compatibility(model.grid, driver_grid(driver))
    _check_basis_compatibility(model, driver)

    prefetch_enabled = _prefetch_enabled(prefetch_windows, model.state.air_mass)
    separate_prefetch = prefetch_enabled && start_window < stop_window
    buffers = _reusable_window_buffers(reuse_buffers_of, model, driver, separate_prefetch)
    if buffers === nothing
        # Read/decode the first forcing window once. Adapt its host payload twice
        # when prefetching so the two device buffers remain independently writable.
        loaded_window = _load_window(driver, start_window)
        window = _adapt_window_to_model_backend(loaded_window, model.state.air_mass)
        prefetch_window = if separate_prefetch
            # A custom driver may already supply device arrays; adapting those to
            # the same backend can alias, so copy that window explicitly instead.
            _window_backend_adapter(loaded_window.air_mass) === Array ?
                _adapt_window_to_model_backend(loaded_window, model.state.air_mass) : deepcopy(window)
        else
            window
        end
        expected_air_mass = _allocate_storage_like(model.state.air_mass)
        qv_buffer = _allocate_qv_buffer(window)
        host_staging = _host_staging_window(loaded_window, driver, model.state.air_mass)
    else
        # Refill the previous simulation's storage, as a prefetch would.
        host_staging = buffers.host_staging
        window = _load_window_into_existing_backend!(buffers.window, driver, start_window,
                                                     model.state.air_mass, host_staging)
        prefetch_window = separate_prefetch ? buffers.prefetch_window : window
        expected_air_mass = buffers.expected_air_mass
        qv_buffer = buffers.qv_buffer
    end
    prefetch_task = _empty_prefetch_task()
    surface_sources_adapted = _adapt_sources_to_model_backend(Tuple(surface_sources), model.state.air_mass)
    foreach(source -> _check_surface_source_compatibility(model.state, source), surface_sources_adapted)

    # Chemistry + emissions are applied by the model-level composition, not as
    # an independent sim-level post-step. `with_emissions` installs the
    # user-supplied surface sources as a `SurfaceFluxOperator` inside the
    # wrapped model so the palindrome's S slot runs at the correct
    # center-of-transport position. `with_chemistry` installs the user's
    # chemistry in the model. `step!(model)` symmetrically splits constant-rate
    # exponential decay around its transport/emissions block; the
    # binary-scheduled path below performs the same split around the complete
    # met window. Time-varying-rate decay runs unsplit after transport.
    model = with_chemistry(model, chemistry)
    if !isempty(surface_sources_adapted)
        emissions_op = SurfaceFluxOperator(PerTracerFluxMap(surface_sources_adapted))
        model = with_emissions(model, emissions_op)
    end
    model = buffers === nothing ? _install_convection_forcing(model, driver, window) :
                                  _reinstall_convection_forcing(model, driver, window)
    FT = _storage_eltype(model.state.air_mass)
    step_schedule = _driver_step_schedule(driver)
    isfinite(window_dt(driver)) && window_dt(driver) > 0 || throw(ArgumentError(
        "DrivenSimulation driver window_dt must be finite and positive"))
    steps_current = step_schedule[Int(start_window)]
    Δt = window_step_dt(FT(window_dt(driver)), steps_current)
    nsteps_total = sum(@view step_schedule[Int(start_window):Int(stop_window)])

    flux_interp = interpolate_fluxes_within_window === nothing ?
                  (flux_interpolation_mode(driver) === :interpolate) : Bool(interpolate_fluxes_within_window)
    reset_mode = _normalize_air_mass_reset_mode(air_mass_reset_mode)
    plan = StepPlan(; physics_cadence, window_contract = uses_binary_substep_contract(driver),
                    air_mass_reset = reset_mode)

    sim = DrivenSimulation{typeof(model), typeof(driver), typeof(window),
                           typeof(expected_air_mass), typeof(qv_buffer), FT,
                           typeof(callbacks), typeof(prefetch_task),
                           typeof(host_staging), typeof(plan)}(
        model,
        driver,
        window,
        prefetch_window,
        prefetch_task,
        0,
        prefetch_enabled,
        expected_air_mass,
        qv_buffer,
        Δt,
        FT(window_dt(driver)),
        steps_current,
        step_schedule,
        Float64(start_time),
        Float64(start_time),
        0,
        Int(start_window),
        Int(start_window),
        0,
        steps_current,
        Int(stop_window),
        Int(nsteps_total),
        callbacks,
        initialize_air_mass,
        use_midpoint_forcing,
        flux_interp,
        reset_mode,
        plan,
        host_staging,
    )

    if initialize_air_mass
        _copy_storage!(sim.model.state.air_mass, sim.window.air_mass)
        _refresh_state_halos!(sim.model.state, sim.model.grid.horizontal)
    elseif sim.air_mass_reset_mode !== :none
        _reset_air_mass!(sim.model.state, sim.window.air_mass,
                         sim.model.grid.horizontal,
                         sim.air_mass_reset_mode)
    else
        _refresh_state_halos!(sim.model.state, sim.model.grid.horizontal)
    end
    copy_fluxes!(sim.model.fluxes, sim.window.fluxes)
    _copy_storage!(sim.expected_air_mass, sim.window.air_mass)
    if sim.qv_buffer !== nothing
        _copy_storage!(sim.qv_buffer, sim.window.qv_start)
    end
    _refresh_dz_for_window!(sim)
    _refresh_pbl_kz_for_window!(sim.model.diffusion, sim)
    _stage_surface_flux_window!(sim)
    # With reused window buffers there are no large startup allocations to trim
    # (the surface sources' device copies are small).
    buffers === nothing && _reclaim_backend_pool_after_startup!(sim.model.state.air_mass)
    _start_window_prefetch!(sim, Int(start_window) + 1)
    return sim
end

window_index(sim::DrivenSimulation) = sim.current_window_index
function substep_index(sim::DrivenSimulation)
    if sim.iteration == sim.current_window_end_iteration &&
       sim.current_window_index < sim.stop_window
        return 1
    end
    return min(sim.steps_per_window,
               sim.iteration - sim.current_window_start_iteration + 1)
end
current_qv(sim::DrivenSimulation) = sim.qv_buffer

"""
    current_time(sim::DrivenSimulation) -> Float64

Simulation time [s] at the start of the next step. Returns
`sim.time` (Float64), which is initialized to `start_time` at sim construction
(seconds since the RUN start for multi-binary runs) and recomputed from the
window and step counters at the end of each `step!(sim)`, so window ends fall
exactly on multiples of `window_dt`.

`sim` is threaded through operators via the `meteo` kwarg:

    step!(sim.model, sim.Δt; meteo = sim)   # not sim.driver

so operators that need time (`StepwiseField` emission rates,
time-varying Kz, future convection DerivedConvMassFluxField) read
`current_time(meteo)` and get `sim.time`. `meteo.driver` remains
accessible for operator code that needs driver-level capabilities
(e.g. `supports_cmfmc(meteo.driver)`).

Meteorological drivers are stateless and deliberately do not implement
`current_time`; operators receive the simulation clock, not `sim.driver`.
"""
MetDrivers.current_time(sim::DrivenSimulation) = sim.time

# The clock follows the counters instead of accumulating steps: a Float32
# clock adding 3600/7 s steps is 3 h off after a year, and even a Float64 sum
# misses window ends by an ulp.
_clock_time(sim::DrivenSimulation) =
    _window_clock(sim, sim.current_window_index - sim.start_window,
                  (sim.iteration - sim.current_window_start_iteration) / sim.steps_per_window)

_window_clock(sim::DrivenSimulation, offset::Integer, fraction::Float64) =
    window_clock(sim.start_time, sim.window_dt, offset, fraction)

# Run-clock interval `[t0, t1]` of the current met window: `_clock_time` at the
# window's first and last step boundary.
function _window_time_span(sim::DrivenSimulation)
    offset = sim.current_window_index - sim.start_window
    return (_window_clock(sim, offset, 0.0), _window_clock(sim, offset, 1.0))
end

# Time-varying surface-flux series stay in host memory; copy the slices the
# current window reads into their device buffers. This runs on the task that
# launches the window's kernels, after the window's met data became current,
# so the copies are ordered before the window's first surface-flux kernel
# (CUDA.jl issues both on the task's stream; the Metal bulk hook synchronizes
# once before eligible dense host copies, whose low-level blits each wait for
# completion). The prefetch task never touches the buffers.
function _stage_surface_flux_window!(sim::DrivenSimulation)
    t0, t1 = _window_time_span(sim)
    SectionTimer.@section :surface_flux_upload stage_surface_flux_window!(sim.model.emissions, t0, t1)
    return nothing
end

# A binary with a per-window contract makes its endpoint mass authoritative at
# window ends. Convection runs once per window there and exponential decay is
# split into exact half-steps around the window's transport/emissions, unless
# `physics_cadence = :substep` (`[run] physics_cadence`) asks for every advection
# substep; the window-end mass reset is the same for both cadences, so an A/B
# comparison on one binary changes only where physics runs. `sim.plan`
# (`StepPlan`) encodes this schedule.

# `physics_cadence` (`:window`, `:substep` or `nothing` for `:window`) is
# resolved by `Operators.StepPlans._resolve_physics_cadence`.

function step!(sim::DrivenSimulation)
    sim.iteration < sim.final_iteration ||
        throw(ArgumentError("DrivenSimulation has already completed all scheduled steps"))

    SectionTimer.@section :window_advance _maybe_advance_window!(sim)
    substep = substep_index(sim)
    SectionTimer.@section :forcing_refresh _refresh_forcing!(sim, substep)

    # The plan (`sim.plan`) holds the schedule. Without a binary window
    # contract, or with `physics_cadence = :substep`, every step runs the full
    # operator suite. Binary-scheduled `:window` runs apply only the transport
    # block at each stored substep: exact exponential decay half-steps bracket
    # the complete met window, placing its integrated emissions at the
    # midpoint, and convection and the closing chemistry run after the
    # window's last step, behind the window-end air-mass reset.
    ops = step_operators(sim.model)
    timing = _step_timing(sim)
    substep == 1 && execute_window_start!(sim.plan, ops, timing, sim)
    execute_step_stages!(sim.plan, ops, timing, sim)
    sim.iteration += 1
    sim.time = _clock_time(sim)
    sim.iteration == sim.current_window_end_iteration &&
        execute_window_end!(sim.plan, ops, timing, sim)
    for callback in values(sim.callbacks)
        callback(sim)
    end
    return nothing
end

"""
    run_window!(sim::DrivenSimulation)

Advance exactly the current meteorological window and return sim. If sim is
positioned at a completed non-final window, the next window is loaded first.
"""
function run_window!(sim::DrivenSimulation)
    if sim.iteration == sim.current_window_end_iteration &&
       sim.current_window_index < sim.stop_window
        SectionTimer.@section :window_advance _maybe_advance_window!(sim)
    end
    target_iteration = min(sim.final_iteration, sim.current_window_end_iteration)
    while sim.iteration < target_iteration
        step!(sim)
    end
    return sim
end

function run!(sim::DrivenSimulation)
    while sim.iteration < sim.final_iteration
        step!(sim)
    end
    return sim
end

export DrivenSimulation, run_window!, window_index, substep_index, current_qv
