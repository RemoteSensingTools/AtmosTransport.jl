# The cubed-sphere adjoint of a DrivenSimulation configuration: its step plan
# and met windows as a `CSStepSchedule`, and the setup guard for its operators.
# Included by Runner.jl after driven_stepping.jl.

"""
    CSStepSchedule(sim::DrivenSimulation)

The production step of `sim` for the cubed-sphere adjoint: `sim.plan`, the
driver's step schedule and window length for windows `sim.start_window`
through `sim.stop_window`, the binary subcycle contract, and the air-mass
reset targets the plan reads (each window's air mass and its end air mass
`expected_air_mass!(…, window, 1)`, from the driver's windows). The adjoint
replays the same plan; its per-step face fluxes and the per-window convection
forcing are the caller's (see `cs_surface_emission_footprint`).

Where the plan resets to both targets, window `w + 1`'s start target is
window `w`'s end target, the same array, when the two are equal bit for bit
(halos included; continuous binaries store `dm` so that `m + dm` is the next
window's `m`), and a separate copy otherwise (see `_window_reset_targets`).
"""
function Adjoints.CSStepSchedule(sim::DrivenSimulation)
    require_adjoint_support(sim)
    return _cs_step_schedule(sim, _window_reset_targets(sim, identity)...)
end

# The schedule of `sim` with reset targets `starts` and `ends` (`nothing` when
# a met source carries them with its windows).
function _cs_step_schedule(sim::DrivenSimulation, starts, ends)
    windows = sim.start_window:sim.stop_window
    return CSStepSchedule(sim.plan; steps_per_window = sim.steps_per_window_schedule[windows],
                          window_dt = sim.window_dt, start_time = sim.start_time,
                          binary_substeps = uses_binary_substep_contract(sim.driver),
                          chemistry = sim.model.chemistry,
                          window_start_air_mass = starts, window_end_air_mass = ends)
end

"""
    cs_adjoint_inputs(sim::DrivenSimulation; tracer = first tracer, residency = :device,
                      device_slots = nothing, host_budget_bytes = nothing) -> NamedTuple

Everything a cubed-sphere adjoint of `sim`'s configuration needs, captured from
the production simulation itself: `panels_rm0` (the tracer `tracer`) and
`panels_m0` (the initial state), the face fluxes of every step as production
refreshes them (`panels_am_steps`, `panels_bm_steps`, `panels_cm_steps`:
window loading, intra-window interpolation, full-window flux scaling), `mesh`,
and `kwargs` for the adjoint entry points: `scheme`, `schedule =
CSStepSchedule(sim)`, `surface_flux`, `emission_rates` (the tracer's static
surface source per step, or `nothing`), the diffusion operator and
production's layer thickness per step (`diffusion_op`,
`diffusion_layer_thickness`: both copied once per met window, after
production refreshed them, and shared by the window's steps; `nothing` for
Dkg fields and TM5 Calc_Kzz, which carry their exchange), and the convection operator,
its forcing per step (one object per met window) and a fresh workspace. The
replay allocates its own diffusion workspace and reverse scratch.

`residency = :host` keeps the met windows in host memory instead: each
window's captured fluxes, convection forcing, diffusion operator, layer
thickness and reset targets are copied to the host when the window is
complete (bit copies of what `:device` keeps) and the device copies freed.
The three flux vectors are then `nothing`, the schedule carries no reset
targets, and `kwargs.met_source` is a [`HostCSWindowSource`](@ref) with
`device_slots` (default 2) windows on the model's backend, replacing `diffusion_op`,
`diffusion_layer_thickness` and `convection_forcing`. The host bytes must fit
`host_budget_bytes` (default: the host or cgroup memory limit less AutoTape's
host margin; set it on a shared host); the capture checks the projected total
after every window and fails before the run. Streamed inputs need a
`MetWindowCheckpoint` tape.

The face fluxes are copied each time production rewrites the model's
fluxes. With window-constant fluxes (`sim.interpolate_fluxes_within_window ==
false`: the driver's default for a binary with `flux_sampling =
:window_constant` or without flux deltas, unless the `DrivenSimulation`
keyword overrides it) that is a window's first step, and the window's steps
reference that one copy (`panels_am_steps[s] === panels_am_steps[s + 1]`
inside a window), as production's steps read the same arrays. Fluxes
interpolated within the window are rewritten and copied every step. The
adjoint only reads these arrays; do not modify them before the reverse pass
ends (a change to one step's arrays changes every step of its window).

`require_adjoint_support(sim)` runs first. The capture runs the simulation's
met refresh through every window (and production advection for live TM5
with no mass reset, to advance its carrier mass); it leaves tracers in an
unspecified state, so `sim` cannot be run afterwards: build a second,
identical simulation for the production run. Memory grows with the run: one
set of face fluxes per met window (per step when interpolated), the
per-window convection forcing, diffusion operator and layer thickness, and
the reset targets (two per window, or one where a window's end target equals
the next window's start target bit for bit, see `CSStepSchedule(sim)`). A
time-varying surface source of `tracer` is an error (pass its per-step rates
yourself). Initial persistent operator state must be fresh: nonzero
surface-source Kahan compensation, diffusion/Dkg or convection ledger
carries, or a valid CMFMC CFL cache are rejected before capture. This also
rejects continued models whose new simulation has `iteration == 0`.

```julia
inputs = cs_adjoint_inputs(sim; tracer = :CO2)
footprint = cs_surface_emission_footprint(inputs.panels_rm0, inputs.panels_m0,
    inputs.panels_am_steps, inputs.panels_bm_steps, inputs.panels_cm_steps,
    inputs.mesh, objective; base_emission_rates = inputs.kwargs.emission_rates,
    Base.structdiff(inputs.kwargs, NamedTuple{(:surface_flux, :emission_rates)})...)
```
"""
function cs_adjoint_inputs(sim::DrivenSimulation; tracer::Symbol = first(sim.model.state.tracer_names),
                           residency::Symbol = :device, device_slots = nothing,
                           host_budget_bytes = nothing)
    model = sim.model
    model.state isa CubedSphereState || throw(ArgumentError(
        "cs_adjoint_inputs needs a cubed-sphere simulation; got $(typeof(model.state))"))
    sim.iteration == 0 || throw(ArgumentError(
        "cs_adjoint_inputs needs a simulation that has not stepped (iteration $(sim.iteration))"))
    idx = findfirst(==(tracer), model.state.tracer_names)
    idx === nothing && throw(ArgumentError(
        "tracer $(repr(tracer)) is not in the simulation ($(model.state.tracer_names))"))
    capture = _adjoint_capture(residency, sim; device_slots, host_budget_bytes)
    require_adjoint_support(sim)
    _require_fresh_adjoint_state(model)

    panels_rm0 = map(copy, get_tracer(model.state, idx))
    panels_m0 = map(copy, model.state.air_mass)
    schedule = _capture_schedule!(capture, sim)
    emission_rates = _adjoint_emission_rates(model.emissions, tracer, sim.final_iteration)

    has_diffusion = !(model.diffusion isa NoDiffusion)
    has_convection = !(model.convection isa NoConvection)
    fluxes = window_op = window_dz = window_forcing = nothing
    try
        while sim.iteration < sim.final_iteration
            _maybe_advance_window!(sim)
            substep = substep_index(sim)
            _refresh_forcing!(sim, substep)
            # One copy each time production rewrites the fluxes; the steps in
            # between read the same arrays and share it.
            if _refreshes_fluxes(sim, substep)
                fluxes = map(f -> map(copy, f), (model.fluxes.am, model.fluxes.bm, model.fluxes.cm))
            end
            if substep == 1
                _window_captured!(capture)          # the previous window is complete
                has_convection && (window_forcing = deepcopy(model.convection_forcing))
                if has_diffusion
                    window_op = _adjoint_window_operator(model.diffusion)
                    window_dz = _adjoint_window_layer_thickness(model.diffusion, model)
                end
            end
            _capture_step!(capture, fluxes, window_forcing, window_op, window_dz)
            _advance_adjoint_carrier!(model.diffusion,sim)
            _advance_clock!(sim)
        end
        _window_captured!(capture)
    finally
        _finish_window_prefetch!(sim)
    end
    common = (; scheme = model.advection, schedule,
              surface_flux = !(model.emissions isa NoSurfaceFlux), emission_rates)
    convection = (; convection_op = model.convection,
                  convection_workspace = convection_workspace_for(model.convection, model.state,
                                                                  model.grid))
    return _captured_inputs(capture, panels_rm0, panels_m0, model.grid.horizontal, common, convection,
                            has_diffusion, has_convection)
end

# Where the capture keeps the met: every window on the model's backend, as
# per-step vectors (`:device`), or each completed window copied to the host
# for a `HostCSWindowSource` (`:host`).
function _adjoint_capture(residency::Symbol, sim; device_slots, host_budget_bytes)
    residency === :host && return _HostCapture(sim; device_slots = something(device_slots, 2),
                                               host_budget_bytes)
    residency === :device || throw(ArgumentError(
        "residency = $(repr(residency)); expected :device (every window on the model's backend) or " *
        ":host (a HostCSWindowSource)"))
    device_slots === nothing && host_budget_bytes === nothing || throw(ArgumentError(
        "device_slots and host_budget_bytes configure residency = :host"))
    return _DeviceCapture(Any[], Any[], Any[], Any[], Any[], Any[])
end

struct _DeviceCapture
    am              :: Vector{Any}
    bm              :: Vector{Any}
    cm              :: Vector{Any}
    forcing         :: Vector{Any}
    diffusion_op    :: Vector{Any}
    layer_thickness :: Vector{Any}
end

_capture_schedule!(::_DeviceCapture, sim) = CSStepSchedule(sim)

function _capture_step!(c::_DeviceCapture, fluxes, forcing, op, dz)
    push!(c.am, fluxes[1]); push!(c.bm, fluxes[2]); push!(c.cm, fluxes[3])
    push!(c.forcing, forcing); push!(c.diffusion_op, op); push!(c.layer_thickness, dz)
    return nothing
end
_window_captured!(::_DeviceCapture) = nothing

function _captured_inputs(c::_DeviceCapture, panels_rm0, panels_m0, mesh, common, convection,
                          has_diffusion, has_convection)
    kwargs = (; common...,
              diffusion_op = has_diffusion ? [c.diffusion_op...] : NoDiffusion(),
              diffusion_layer_thickness = has_diffusion && !all(isnothing, c.layer_thickness) ?
                                          map(identity, c.layer_thickness) : nothing,
              convection_op = convection.convection_op,
              convection_forcing = has_convection ? [c.forcing...] : nothing,
              convection_workspace = convection.convection_workspace)
    return (; panels_rm0, panels_m0, panels_am_steps = [c.am...], panels_bm_steps = [c.bm...],
            panels_cm_steps = [c.cm...], mesh, kwargs)
end

# The host capture: the steps of the window being captured (device copies),
# the completed windows in host memory, and their host bytes against the
# budget (each array once: a start target shared with the previous window's
# end target counts once).
mutable struct _HostCapture
    const am            :: Vector{Any}
    const bm            :: Vector{Any}
    const cm            :: Vector{Any}
    forcing             :: Any
    diffusion_op        :: Any
    layer_thickness     :: Any
    const windows       :: Vector{Any}
    starts              :: Any
    ends                :: Any
    const seen          :: IdDict{Any, Nothing}
    target_bytes        :: Int
    window_bytes        :: Int
    const budget        :: Int
    const nwindows      :: Int
    const device_slots  :: Int
end

_HostCapture(sim; device_slots, host_budget_bytes) =
    _HostCapture(Any[], Any[], Any[], nothing, nothing, nothing, Any[], nothing, nothing,
                 IdDict{Any, Nothing}(), 0, 0, Adjoints.Footprint._host_window_budget(host_budget_bytes),
                 length(sim.start_window:sim.stop_window), Int(device_slots))

# The schedule without targets; the targets, computed as `CSStepSchedule(sim)`
# computes them, in host memory.
function _capture_schedule!(c::_HostCapture, sim)
    c.starts, c.ends = _window_reset_targets(sim, Adjoints.Footprint._host_copy)
    c.target_bytes = Adjoints.Footprint._new_array_bytes!(c.seen, (Tuple(something(c.starts, ())),
                                                                   Tuple(something(c.ends, ()))))
    Adjoints.Footprint._require_host_budget(c.target_bytes, c.budget, "the reset targets")
    return _cs_step_schedule(sim, nothing, nothing)
end

function _capture_step!(c::_HostCapture, fluxes, forcing, op, dz)
    push!(c.am, fluxes[1]); push!(c.bm, fluxes[2]); push!(c.cm, fluxes[3])
    c.forcing, c.diffusion_op, c.layer_thickness = forcing, op, dz
    return nothing
end

# A completed window: its met in host memory (`_host_window_met`), its device
# copies freed, and the projected host bytes of all windows checked.
function _window_captured!(c::_HostCapture)
    isempty(c.am) && return nothing
    FP = Adjoints.Footprint
    w = length(c.windows) + 1
    first_step = w == 1 ? 1 : last(c.windows[end].steps) + 1
    met = FP._host_window_met(w, first_step:(first_step + length(c.am) - 1), c.am, c.bm, c.cm;
                              convection_forcing = c.forcing,
                              diffusion_op = something(c.diffusion_op, NoDiffusion()),
                              layer_thickness = c.layer_thickness,
                              start_air_mass = c.starts === nothing ? nothing : c.starts[w],
                              end_air_mass = c.ends === nothing ? nothing : c.ends[w])
    FP._free_met_arrays!((Tuple(unique(objectid, zip(c.am, c.bm, c.cm))), c.forcing, c.diffusion_op,
                          c.layer_thickness))
    push!(c.windows, met)
    c.window_bytes += FP._new_array_bytes!(c.seen, FP._met_pieces(met))
    FP._require_host_budget(c.target_bytes + cld(c.window_bytes * c.nwindows, w), c.budget,
                            "the met of $(c.nwindows) windows (projected from $w)")
    empty!(c.am); empty!(c.bm); empty!(c.cm)
    return nothing
end

function _captured_inputs(c::_HostCapture, panels_rm0, panels_m0, mesh, common, convection,
                          _has_diffusion, _has_convection)
    source = HostCSWindowSource([c.windows...]; device_like = first(panels_m0),
                                device_slots = c.device_slots, host_budget_bytes = c.budget)
    kwargs = (; common..., met_source = source, convection...)
    return (; panels_rm0, panels_m0, panels_am_steps = nothing, panels_bm_steps = nothing,
            panels_cm_steps = nothing, mesh, kwargs)
end

# With no mass reset, the next window's live D reads prognostic air mass.
# Advance it with production advection (including its sweep order/rounding),
# without physics at the midpoint. Tracers are disposable during capture.
_advance_adjoint_carrier!(_op,_sim) = nothing
function _advance_adjoint_carrier!(::ImplicitVerticalDiffusion{FT,<:TM5CalcKzzCSField},sim) where FT
    sim.air_mass_reset_mode === :none || return nothing
    m = sim.model
    Operators.apply!(m.state,m.fluxes,m.grid,m.advection,sim.Δt;
        workspace=m.workspace.advection_ws,meteo=sim)
    return nothing
end

# D is met-only in every reset mode. Capture just the immutable per-window
# coefficients; refresh scratch and geometry are never needed during replay.
_adjoint_window_operator(op) = deepcopy(op)
function _adjoint_window_operator(op::ImplicitVerticalDiffusion{FT,<:TM5CalcKzzCSField}) where FT
    f = op.kz_field
    frozen = TM5CalcKzzCSField(map(copy,f.dkg),nothing,f.params,nothing,Ref(true))
    return ImplicitVerticalDiffusion(;kz_field=frozen,surface_flux_coupling=op.surface_flux_coupling)
end
# The window's layer thickness [m] as production refreshed it (the replay
# owns its workspace and the reverse scratch), or `nothing` for an operator
# whose solve reads none (Dkg fields, TM5 D: `Adjoints._reads_layer_thickness`).
_adjoint_window_layer_thickness(op, model) =
    Adjoints._reads_layer_thickness(op) ? map(copy, model.workspace.diffusion_ws.layer_thickness) :
                                          nothing

# Capture currently supports fresh operator state only. Iteration zero alone
# does not establish freshness: the runner reuses a model across daily binaries.
# Reject before met refresh consumes the simulation, including a valid CMFMC
# CFL cache (which replay would otherwise invalidate/reset).
_has_initial_operator_state(::Nothing) = true
_has_initial_operator_state(a::AbstractArray) = all(iszero, a)
_has_initial_operator_state(t::Tuple) = all(_has_initial_operator_state, t)
_has_initial_operator_state(s::NamedTuple) =
    _has_initial_operator_state(s.carry) && (!hasproperty(s, :valid) || !s.valid)

function _require_fresh_adjoint_state(model)
    for (name, op) in (("surface emissions", model.emissions),
                       ("diffusion", model.workspace.diffusion_ws),
                       ("convection", model.workspace.convection_ws))
        _has_initial_operator_state(Operators.StepPlans.operator_state(op)) ||
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

# The air-mass reset targets of the simulation's windows (vectors of six
# halo-padded panels on the model backend, or `nothing` for a target the plan does not read),
# computed as the simulation computes them: each window's air mass (start) and
# its end air mass `m + 1 * dm` (end, `expected_air_mass!`). Each window is
# loaded once.
#
# A start target equal bit for bit (interior and halos) to the previous
# window's end target is that array itself; any other is its own copy. The
# sharing cannot change a result: the replay only reads a target, except that
# a reset first refreshes its target's halos in place (`fill_panel_halos!(…;
# dir = 1)`, which writes every halo cell: the edges from the interior, then
# the corners from those edges, within one call). Both resets of a shared target
# therefore install the same values the two separate copies would, whichever
# runs first, and the interior is never written. The one read of a target
# before its own refresh, the `:preserve_vmr` reset record staging it, may
# then see refreshed instead of initial halos; the reverse multiplies those
# halo cells by an adjoint state the next halo record has zeroed (the reason
# reset-target identity checksums cover interiors only), as for any later
# forward run of separate targets.
#
# `store` keeps each target: `identity` (on the model backend), or a host copy
# (`residency = :host`); a shared target is stored once.
function _window_reset_targets(sim::DrivenSimulation, store)
    plan = sim.plan
    needs_start = any(s -> s isa AirMassResetStage, plan.window_advance)
    needs_end = any(s -> s isa AirMassResetStage, plan.window_end)
    needs_start || needs_end || return nothing, nothing
    starts, ends = Any[], Any[]
    previous_end = stored_end = nothing
    for w in sim.start_window:sim.stop_window
        window = _adapt_window_to_model_backend(_load_window(sim.driver, w),
                                                sim.model.state.air_mass)
        if needs_start
            if previous_end !== nothing && _bitwise_equal(previous_end, window.air_mass)
                push!(starts, stored_end)
            else
                start = _allocate_storage_like(window.air_mass)
                push!(starts, store(_copy_storage!(start, window.air_mass)))
            end
        end
        if needs_end
            previous_end = _allocate_storage_like(window.air_mass)
            expected_air_mass!(previous_end, window, one(typeof(sim.Δt)))
            stored_end = store(previous_end)
            push!(ends, stored_end)
        end
    end
    return needs_start ? [starts...] : nothing, needs_end ? [ends...] : nothing
end

# Bit-for-bit equality of two six-panel fields (`===` per element: -0.0 and
# 0.0 differ), as one reduction per panel on the arrays' backend; queued
# writes to either field are finished first.
function _bitwise_equal(a::NTuple{6, AbstractArray}, b::NTuple{6, AbstractArray})
    all(p -> size(a[p]) == size(b[p]) && eltype(a[p]) === eltype(b[p]), 1:6) || return false
    Architectures.synchronize_storage!(a); Architectures.synchronize_storage!(b)
    return all(p -> mapreduce(===, &, a[p], b[p]; init = true), 1:6)
end
# Storage that is not six panels (e.g. lat-lon): never shared.
_bitwise_equal(_a, _b) = false

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
