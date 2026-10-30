# The cubed-sphere adjoint of a DrivenSimulation configuration: its step plan
# and met windows as a `CSStepSchedule`, and the setup guard for its operators.
# Included by Models.jl after driven_stepping.jl.

"""
    CSStepSchedule(sim::DrivenSimulation)

The production step of `sim` for the cubed-sphere adjoint: `sim.plan`, the
driver's step schedule and window length for windows `sim.start_window`
through `sim.stop_window`, the binary subcycle contract, and the air-mass
reset targets the plan reads (each window's air mass and its end air mass
`expected_air_mass!(…, window, 1)`, from the driver's windows). The adjoint
replays the same plan; its per-step face fluxes and the per-window convection
forcing are the caller's (see `cs_surface_emission_footprint`).
"""
function Adjoints.CSStepSchedule(sim::DrivenSimulation)
    require_adjoint_support(sim)
    windows = sim.start_window:sim.stop_window
    plan = sim.plan
    needs_start = any(s -> s isa AirMassResetStage, plan.window_advance)
    needs_end = any(s -> s isa AirMassResetStage, plan.window_end)
    starts = needs_start ? [_window_air_mass(sim, w, 0) for w in windows] : nothing
    ends = needs_end ? [_window_air_mass(sim, w, 1) for w in windows] : nothing
    return CSStepSchedule(plan; steps_per_window = sim.steps_per_window_schedule[windows],
                          window_dt = sim.window_dt, start_time = sim.start_time,
                          binary_substeps = uses_binary_substep_contract(sim.driver),
                          chemistry = sim.model.chemistry,
                          window_start_air_mass = starts, window_end_air_mass = ends)
end

"""
    cs_adjoint_inputs(sim::DrivenSimulation; tracer = first tracer) -> NamedTuple

Everything a cubed-sphere adjoint of `sim`'s configuration needs, captured from
the production simulation itself: `panels_rm0` (the tracer `tracer`) and
`panels_m0` (the initial state), the face fluxes of every step as production
refreshes them (`panels_am_steps`, `panels_bm_steps`, `panels_cm_steps`:
window loading, intra-window interpolation, full-window flux scaling), `mesh`,
and `kwargs` for the adjoint entry points: `scheme`, `schedule =
CSStepSchedule(sim)`, `surface_flux`, `emission_rates` (the tracer's static
surface source per step, or `nothing`), the diffusion operator and a
workspace with production's layer thickness per met window (`diffusion_op`,
`diffusion_workspace`; the operator is copied per window when its Kz field is
refreshed per window), and the convection operator, its forcing per step (one
object per met window) and a fresh workspace.

`require_adjoint_support(sim)` runs first. The capture runs the simulation's
met refresh through every window (not its tracer transport, which it leaves
in an unspecified state), so `sim` cannot be run afterwards: build a second,
identical simulation for the production run. Memory grows with the run: one
set of face fluxes per step. A time-varying surface source of `tracer` is an
error (pass its per-step rates yourself). Initial persistent operator state must
be fresh: nonzero surface-source Kahan compensation, diffusion/Dkg or convection
ledger carries, or a valid CMFMC CFL cache are rejected before capture. This
also rejects continued models whose new simulation has `iteration == 0`.

```julia
inputs = cs_adjoint_inputs(sim; tracer = :CO2)
footprint = cs_surface_emission_footprint(inputs.panels_rm0, inputs.panels_m0,
    inputs.panels_am_steps, inputs.panels_bm_steps, inputs.panels_cm_steps,
    inputs.mesh, objective; base_emission_rates = inputs.kwargs.emission_rates,
    Base.structdiff(inputs.kwargs, NamedTuple{(:surface_flux, :emission_rates)})...)
```
"""
function cs_adjoint_inputs(sim::DrivenSimulation; tracer::Symbol = first(sim.model.state.tracer_names))
    model = sim.model
    model.state isa CubedSphereState || throw(ArgumentError(
        "cs_adjoint_inputs needs a cubed-sphere simulation; got $(typeof(model.state))"))
    sim.iteration == 0 || throw(ArgumentError(
        "cs_adjoint_inputs needs a simulation that has not stepped (iteration $(sim.iteration))"))
    idx = findfirst(==(tracer), model.state.tracer_names)
    idx === nothing && throw(ArgumentError(
        "tracer $(repr(tracer)) is not in the simulation ($(model.state.tracer_names))"))
    require_adjoint_support(sim)
    _require_fresh_adjoint_state(model)

    panels_rm0 = map(copy, get_tracer(model.state, idx))
    panels_m0 = map(copy, model.state.air_mass)
    schedule = CSStepSchedule(sim)
    emission_rates = _adjoint_emission_rates(model.emissions, tracer, sim.final_iteration)

    has_diffusion = !(model.diffusion isa NoDiffusion)
    has_convection = !(model.convection isa NoConvection)
    am, bm, cm = Any[], Any[], Any[]
    diffusion_op, diffusion_workspace, forcing = Any[], Any[], Any[]
    window_op = window_ws = window_forcing = nothing
    try
        while sim.iteration < sim.final_iteration
            _maybe_advance_window!(sim)
            substep = substep_index(sim)
            _refresh_forcing!(sim, substep)
            push!(am, map(copy, model.fluxes.am))
            push!(bm, map(copy, model.fluxes.bm))
            push!(cm, map(copy, model.fluxes.cm))
            if substep == 1
                has_convection && (window_forcing = deepcopy(model.convection_forcing))
                if has_diffusion
                    window_op = deepcopy(model.diffusion)
                    window_ws = DiffusionWorkspace(model.state.air_mass, model.state.halo_width, 0)
                    foreach(copyto!, window_ws.layer_thickness,
                            model.workspace.diffusion_ws.layer_thickness)
                end
            end
            push!(forcing, window_forcing)
            push!(diffusion_op, window_op)
            push!(diffusion_workspace, window_ws)
            sim.iteration += 1
            sim.time = _clock_time(sim)
        end
    finally
        _finish_window_prefetch!(sim)
    end
    kwargs = (; scheme = model.advection, schedule,
              surface_flux = !(model.emissions isa NoSurfaceFlux), emission_rates,
              diffusion_op = has_diffusion ? [diffusion_op...] : NoDiffusion(),
              diffusion_workspace = has_diffusion ? [diffusion_workspace...] : nothing,
              convection_op = model.convection,
              convection_forcing = has_convection ? [forcing...] : nothing,
              convection_workspace = convection_workspace_for(model.convection, model.state,
                                                               model.grid))
    return (; panels_rm0, panels_m0, panels_am_steps = [am...], panels_bm_steps = [bm...],
            panels_cm_steps = [cm...], mesh = model.grid.horizontal, kwargs)
end

# Capture currently supports fresh operator state only. Iteration zero alone
# does not establish freshness: the runner reuses a model across daily binaries.
# Reject before met refresh consumes the simulation, including a valid CMFMC
# CFL cache (which replay would otherwise invalidate/reset).
_initial_adjoint_state(::Nothing) = true
_initial_adjoint_state(a::AbstractArray) = all(iszero, a)
_initial_adjoint_state(t::Tuple) = all(_initial_adjoint_state, t)
_initial_adjoint_state(s::NamedTuple) =
    _initial_adjoint_state(s.carry) && (!hasproperty(s, :valid) || !s.valid)

function _require_fresh_adjoint_state(model)
    for (name, op) in (("surface emissions", model.emissions),
                       ("diffusion", model.workspace.diffusion_ws),
                       ("convection", model.workspace.convection_ws))
        _initial_adjoint_state(Operators.StepPlans.operator_state(op)) ||
            throw(ArgumentError("cs_adjoint_inputs cannot discard initial persistent operator state " *
                "of $name (Kahan compensation, ledger carry or valid CMFMC CFL cache); " *
                "capture a fresh model, rather than one continued from a previous binary"))
    end
    return nothing
end

# The per-step emission rates of `tracer`: its static surface source, or
# nothing (no source for it).
_adjoint_emission_rates(::NoSurfaceFlux, _tracer, _n) = nothing
function _adjoint_emission_rates(op, tracer, n)
    src = flux_for(op.flux_map, tracer)
    src === nothing && return nothing
    src isa SurfaceFluxSource || throw(ArgumentError(
        "cs_adjoint_inputs: the surface source of $(repr(tracer)) is a $(nameof(typeof(src))); " *
        "pass its per-step rates as the adjoint's emission rates"))
    return fill(src.cell_mass_rate, n)
end

# The air mass of window `w` at fraction `λ` (0: start, 1: the end the binary
# makes authoritative), computed as the simulation computes its reset target.
function _window_air_mass(sim::DrivenSimulation, w::Int, λ)
    window = _adapt_window_to_model_backend(_load_window(sim.driver, w),
                                            sim.model.state.air_mass)
    dest = _allocate_storage_like(window.air_mass)
    λ == 0 ? _copy_storage!(dest, window.air_mass) :
             expected_air_mass!(dest, window, one(typeof(sim.Δt)))
    return dest
end

"""
    require_adjoint_support(sim::DrivenSimulation) -> sim
    require_adjoint_support(model::TransportModel; plan = StepPlan(; physics_cadence = :substep)) -> model

Check that the cubed-sphere adjoint records and transposes every stage of the
simulation's step plan and every operator of its model. Throws an
`ArgumentError` naming each unsupported piece.
"""
function Adjoints.require_adjoint_support(sim::DrivenSimulation)
    require_adjoint_support(sim.plan, step_operators(sim.model))
    return sim
end

function Adjoints.require_adjoint_support(model::TransportModel;
                                          plan::StepPlan = _MODEL_STEP_PLAN)
    require_adjoint_support(plan, step_operators(model))
    return model
end
