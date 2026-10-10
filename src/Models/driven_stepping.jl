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
        sim.Δt = sim.window_dt / typeof(sim.Δt)(sim.steps_per_window)
        _take_prefetched_window!(sim, next_window)
        if sim.air_mass_reset_mode !== :none
            _reset_air_mass!(sim.model.state, sim.window.air_mass,
                             sim.model.grid.horizontal,
                             sim.air_mass_reset_mode)
        end
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

function _maybe_reset_to_window_endpoint!(sim::DrivenSimulation)
    (sim.air_mass_reset_mode !== :none && _binary_window_contract(sim)) ||
        return nothing
    expected_air_mass!(sim.expected_air_mass, sim.window, one(typeof(sim.Δt)))
    _reset_air_mass!(sim.model.state, sim.expected_air_mass,
                     sim.model.grid.horizontal,
                     sim.air_mass_reset_mode)
    return nothing
end

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
  once-per-window convection/chemistry block so physics sees the binary's
  authoritative window-end mass.
- `physics_cadence=nothing` — `:window` (the default for `nothing`) or
  `:substep`. On a binary with a per-window physics contract, `:window` runs
  convection and chemistry once per met window after the stored advection
  substeps; `:substep` runs them every advection substep (the cadence before
  2026-05-31), for cadence-sensitivity comparisons on the same binary. The
  window-end air-mass reset is the same for both. Other binaries run the full
  operator suite every substep either way.
- `surface_sources=()`
- `chemistry=NoChemistry()` — applied after advection + surface sources each step
- `callbacks=NamedTuple()`
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
                          surface_sources = (),
                          chemistry::AbstractChemistryOperator = NoChemistry(),
                          callbacks = NamedTuple(),
                          start_time::Real = 0) where {D <: AbstractMetDriver}
    1 <= start_window <= stop_window <= total_windows(driver) ||
        throw(ArgumentError("invalid window range: start_window=$(start_window), stop_window=$(stop_window), total_windows=$(total_windows(driver))"))
    supports_native_vertical_flux(driver) ||
        throw(ArgumentError("DrivenSimulation requires native vertical mass fluxes in the met-driver contract"))
    isfinite(start_time) ||
        throw(ArgumentError("DrivenSimulation start_time must be finite; got $(start_time)"))

    _check_grid_compatibility(model.grid, driver_grid(driver))
    _check_basis_compatibility(model, driver)

    # Read/decode the first forcing window once. Adapt its host payload twice
    # when prefetching so the two device buffers remain independently writable.
    loaded_window = _load_window(driver, start_window)
    window = _adapt_window_to_model_backend(loaded_window, model.state.air_mass)
    prefetch_window = if _prefetch_enabled(model.state.air_mass) && start_window < stop_window
        # A custom driver may already supply device arrays; adapting those to
        # the same backend can alias, so copy that window explicitly instead.
        _window_backend_adapter(loaded_window.air_mass) === Array ?
            _adapt_window_to_model_backend(loaded_window, model.state.air_mass) : deepcopy(window)
    else
        window
    end
    prefetch_task = _empty_prefetch_task()
    expected_air_mass = _allocate_storage_like(model.state.air_mass)
    qv_buffer = _allocate_qv_buffer(window)
    surface_sources_adapted = _adapt_sources_to_model_backend(Tuple(surface_sources), model.state.air_mass)
    foreach(source -> _check_surface_source_compatibility(model.state, source), surface_sources_adapted)

    # Chemistry + emissions are applied inside the model's transport block,
    # not as a sim-level post-step. `with_emissions` installs the
    # user-supplied surface sources as a `SurfaceFluxOperator` inside the
    # wrapped model so the palindrome's S slot runs at the correct
    # center-of-transport position. `with_chemistry` installs the user's
    # chemistry in the model; `step!(model)` runs
    # `advection → emissions → diffusion → chemistry` as ONE composed
    # call. The step loop delegates entirely to the model-level operator
    # composition.
    #
    # The palindrome integration preserves TM5's
    # `advection → emissions → chemistry` order with emissions inside the
    # palindrome, so the sim delegates entirely to `step!(model)`.
    model = with_chemistry(model, chemistry)
    if !isempty(surface_sources_adapted)
        emissions_op = SurfaceFluxOperator(PerTracerFluxMap(surface_sources_adapted))
        model = with_emissions(model, emissions_op)
    end
    model = _install_convection_forcing(model, driver, window)
    FT = _storage_eltype(model.state.air_mass)
    step_schedule = _driver_step_schedule(driver)
    all(>(0), step_schedule) || throw(ArgumentError(
        "DrivenSimulation driver step schedule must contain only positive integers"))
    isfinite(window_dt(driver)) && window_dt(driver) > 0 || throw(ArgumentError(
        "DrivenSimulation driver window_dt must be finite and positive"))
    steps_current = step_schedule[Int(start_window)]
    Δt = FT(window_dt(driver)) / FT(steps_current)
    nsteps_total = sum(@view step_schedule[Int(start_window):Int(stop_window)])

    flux_interp = interpolate_fluxes_within_window === nothing ?
                  (flux_interpolation_mode(driver) === :interpolate) : Bool(interpolate_fluxes_within_window)
    reset_mode = _normalize_air_mass_reset_mode(air_mass_reset_mode)
    every_substep = _resolve_physics_cadence(physics_cadence) === :substep
    host_staging = _host_staging_window(loaded_window, driver, model.state.air_mass)

    sim = DrivenSimulation{typeof(model), typeof(driver), typeof(window),
                           typeof(expected_air_mass), typeof(qv_buffer), FT,
                           typeof(callbacks), typeof(prefetch_task),
                           typeof(host_staging)}(
        model,
        driver,
        window,
        prefetch_window,
        prefetch_task,
        0,
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
        every_substep,
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
    _reclaim_backend_pool_after_startup!(sim.model.state.air_mass)
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
_clock_time(sim::DrivenSimulation) = sim.start_time + Float64(sim.window_dt) *
    ((sim.current_window_index - sim.start_window) +
     (sim.iteration - sim.current_window_start_iteration) / sim.steps_per_window)

# A binary with a per-window contract makes its endpoint mass authoritative at
# window ends. Convection and chemistry run once per window there, unless
# `physics_cadence = :substep` (`[run] physics_cadence`) asks for every advection
# substep; the window-end mass reset is the same for both cadences, so an A/B
# comparison on one binary changes only where physics runs.
@inline _binary_window_contract(sim::DrivenSimulation) = uses_binary_substep_contract(sim.driver)
@inline _uses_binary_transport_schedule(sim::DrivenSimulation) =
    _binary_window_contract(sim) && !sim.physics_every_substep

# `physics_cadence` (`:window`, `:substep` or `nothing`), resolved once. Without
# a value, the deprecated `ATMOSTR_FORCE_PER_SUBSTEP_PHYSICS=1` still selects
# `:substep`, with a warning.
function _resolve_physics_cadence(physics_cadence)
    from_env = get(ENV, "ATMOSTR_FORCE_PER_SUBSTEP_PHYSICS", "0") == "1"
    if physics_cadence === nothing
        from_env || return :window
        @warn "ATMOSTR_FORCE_PER_SUBSTEP_PHYSICS=1 is deprecated and will be removed; " *
              "set [run] physics_cadence = \"substep\"." maxlog = 1
        return :substep
    end
    cadence = Symbol(physics_cadence)
    cadence in (:window, :substep) || throw(ArgumentError(
        "physics_cadence must be :window or :substep; got $(repr(physics_cadence))"))
    from_env && cadence === :window && throw(ArgumentError(
        "ATMOSTR_FORCE_PER_SUBSTEP_PHYSICS=1 conflicts with physics_cadence = :window; " *
        "remove the environment variable"))
    return cadence
end

function step!(sim::DrivenSimulation)
    sim.iteration < sim.final_iteration ||
        throw(ArgumentError("DrivenSimulation has already completed all scheduled steps"))

    SectionTimer.@section :window_advance _maybe_advance_window!(sim)
    substep = substep_index(sim)
    SectionTimer.@section :forcing_refresh _refresh_forcing!(sim, substep)

    # The default path keeps the live operator suite in one call.
    # Transport binaries carry an advection substep contract, not a
    # physics cadence contract, so driven
    # binary-scheduled runs apply only the transport block at each stored
    # substep and defer convection + chemistry to the end of the met window.
    if _uses_binary_transport_schedule(sim)
        transport_step!(sim.model, sim.Δt; meteo = sim)
    else
        step!(sim.model, sim.Δt; meteo = sim)
    end
    sim.iteration += 1
    sim.time = _clock_time(sim)
    if _binary_window_contract(sim) && sim.iteration == sim.current_window_end_iteration
        _maybe_reset_to_window_endpoint!(sim)
        sim.physics_every_substep ||
            convection_chemistry_step!(sim.model, sim.window_dt; meteo = sim)
    end
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
