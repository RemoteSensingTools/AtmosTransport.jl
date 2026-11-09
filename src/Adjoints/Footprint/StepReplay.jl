function _validate_step_sequences(panels_am_steps, panels_bm_steps, panels_cm_steps)
    nsteps = length(panels_am_steps)
    nsteps > 0 || throw(ArgumentError("footprint generation requires at least one step"))
    length(panels_bm_steps) == nsteps ||
        throw(ArgumentError("panels_bm_steps length $(length(panels_bm_steps)) does not match panels_am_steps length $nsteps"))
    length(panels_cm_steps) == nsteps ||
        throw(ArgumentError("panels_cm_steps length $(length(panels_cm_steps)) does not match panels_am_steps length $nsteps"))
    return nsteps
end

function _validate_emission_rates(rates, nsteps::Int, mesh::CubedSphereMesh,
                                  name::AbstractString)
    rates === nothing && return nothing
    length(rates) == nsteps || throw(ArgumentError(
        "$name length $(length(rates)) does not match nsteps $nsteps"))
    @inbounds for step in 1:nsteps, p in 1:6
        size(rates[step][p]) == (mesh.Nc, mesh.Nc) || throw(DimensionMismatch(
            "$name step $step panel $p has shape $(size(rates[step][p])); " *
            "expected $((mesh.Nc, mesh.Nc))"))
    end
    return nothing
end



# ---------------------------------------------------------------------------
# The adjoint's forward model: the production step plan, replayed.
#
# `CSStepSchedule` says which production step the adjoint differentiates: the
# `StepPlan` (physics cadence, window contract, air-mass reset), the met
# window structure and the reset targets. `_CSReplay` executes that plan with
# the production operators on a single-tracer `CubedSphereState`: the packed
# cubed-sphere palindrome with StepPlan’s typed midpoint, surface-flux, diffusion and
# convection operators with their persistent state (Kahan compensation,
# ledger carries, the CMFMC CFL cache). The forward replays (finite-difference
# reference, a cost-only 4D-Var evaluation) run it with `NoStepRecorder()`,
# the production call path itself; the tape recorders run the same steps with
# a recorder that stages each operator's input (`TapeRecorder.jl`), and a 4D-Var
# gradient samples its observations in that recording run. There is no second
# composition of the step that could drift from production.
# ---------------------------------------------------------------------------

"""
    CSStepSchedule(; physics_cadence = nothing, binary_window_contract = false,
                   air_mass_reset = :none, steps_per_window = 1, window_dt,
                   start_time = 0, window_start_air_mass = nothing,
                   window_end_air_mass = nothing)
    CSStepSchedule(plan::StepPlan; steps_per_window = 1, window_dt, start_time = 0,
                   binary_substeps = false, chemistry = NoChemistry(),
                   window_start_air_mass = nothing,
                   window_end_air_mass = nothing)

The production step a cubed-sphere adjoint differentiates: the step plan and
the met windows it runs on. The keyword form builds the plan as
`DrivenSimulation` does (`StepPlan(; physics_cadence, window_contract =
binary_window_contract, air_mass_reset)`); `DrivenSimulation`'s own schedule
is `CSStepSchedule(sim)`. That constructor validates the simulation's operators
and retains its chemistry operator; replay never replaces it with a no-op.
Unsupported chemistry is rejected by the adjoint setup guard.

- `steps_per_window` — advection steps per met window: an integer (every
  window) or one entry per window (the run may not go beyond them). Step `s`
  of window `w` advances `window_dt / steps_per_window[w]`, computed as
  `DrivenSimulation` computes `Δt`.
- `start_time` — the run clock [s] at the first step, as `DrivenSimulation`'s
  `start_time`; the replay's `current_time` follows production's clock.
- `binary_window_contract` / `binary_substeps` — the transport binary's
  per-window contract: one palindrome subcycle per step (the binary's
  schedule), and with `physics_cadence = :window` (the default) convection
  and chemistry once per met window over `window_dt`.
- `air_mass_reset` — `:none`, `:preserve_tracer_mass` or `:preserve_vmr`:
  the air mass is reset to `window_start_air_mass[w]` on entering window
  `w > 1` and, with the window contract, to `window_end_air_mass[w]` before
  the window-end physics (vectors of `NTuple{6}` halo-padded panels, one per
  window; a reset refreshes the halos of its target in place, as production
  does).

The default (`CSStepSchedule(window_dt = dt)`) is one step per window and the
full operator suite at every step, `TransportModel.step!`'s composition.
"""
struct CSStepSchedule{P <: StepPlan, FT, WS, WE, CH}
    plan                  :: P
    steps_per_window      :: Vector{Int}
    periodic              :: Bool      # one entry for every window (integer `steps_per_window`)
    window_dt             :: FT
    start_time            :: Float64
    binary_substeps       :: Bool
    chemistry             :: CH
    window_start_air_mass :: WS
    window_end_air_mass   :: WE
end

function CSStepSchedule(plan::StepPlan; steps_per_window = 1, window_dt,
                        start_time::Real = 0,
                        binary_substeps::Bool = false,
                        chemistry = NoChemistry(),
                        window_start_air_mass = nothing,
                        window_end_air_mass = nothing)
    periodic = steps_per_window isa Integer
    steps = periodic ? [Int(steps_per_window)] : collect(Int, steps_per_window)
    isempty(steps) && throw(ArgumentError("steps_per_window must not be empty"))
    all(>=(1), steps) || throw(ArgumentError("steps_per_window must be positive; got $(steps)"))
    isfinite(window_dt) && window_dt > 0 ||
        throw(ArgumentError("window_dt must be finite and positive; got $(window_dt)"))
    isfinite(start_time) || throw(ArgumentError("start_time must be finite; got $(start_time)"))
    nwindows = periodic ? nothing : length(steps)
    _validate_reset_targets(plan.window_advance, window_start_air_mass, "window_start_air_mass",
                            nwindows)
    _validate_reset_targets(plan.window_end, window_end_air_mass, "window_end_air_mass", nwindows)
    return CSStepSchedule{typeof(plan), typeof(window_dt), typeof(window_start_air_mass),
                          typeof(window_end_air_mass), typeof(chemistry)}(
        plan, steps, periodic, window_dt, Float64(start_time), binary_substeps, chemistry,
        window_start_air_mass, window_end_air_mass)
end

function CSStepSchedule(; physics_cadence = nothing, binary_window_contract::Bool = false,
                        air_mass_reset = :none, kwargs...)
    plan = StepPlan(; physics_cadence, window_contract = binary_window_contract,
                    air_mass_reset)
    return CSStepSchedule(plan; binary_substeps = binary_window_contract, kwargs...)
end

_has_reset(stages::Tuple) = any(s -> s isa AirMassResetStage, stages)

function _validate_reset_targets(stages::Tuple, targets, name, nwindows)
    if _has_reset(stages)
        targets isa AbstractVector || throw(ArgumentError(
            "the plan resets the air mass ($(name)); pass one NTuple{6} of air-mass panels per met window"))
        nwindows === nothing || length(targets) == nwindows || throw(ArgumentError(
            "$(name) has $(length(targets)) entries for $(nwindows) met windows"))
    else
        targets === nothing || throw(ArgumentError(
            "$(name) given, but the plan has no air-mass reset that reads it"))
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", s::CSStepSchedule)
    println(io, "CSStepSchedule: ", length(s.steps_per_window), " window pattern(s), steps per window ",
            s.steps_per_window, ", window_dt ", s.window_dt,
            s.binary_substeps ? ", binary subcycles" : ", CFL subcycles")
    show(io, MIME"text/plain"(), s.plan)
end

# Steps `1:nsteps` of a schedule: window of each step, first/last flags, the
# step's dt (as `DrivenSimulation` computes `Δt`) and its clock interval (as
# `DrivenSimulation` computes `sim.time`). An integer `steps_per_window`
# repeats for as many windows as `nsteps` needs.
struct _ResolvedSchedule{FT}
    window       :: Vector{Int}
    first        :: Vector{Bool}
    last         :: Vector{Bool}
    dt           :: Vector{FT}
    window_dt    :: FT
    start_time   :: Vector{Float64}
    end_time     :: Vector{Float64}
end

function _resolve_schedule(s::CSStepSchedule, nsteps::Int, ::Type{FT}) where FT
    window = Int[]; first = Bool[]; last = Bool[]; dt = FT[]
    start_time = Float64[]; end_time = Float64[]
    w = 0
    wdt = FT(s.window_dt)
    while length(window) < nsteps
        w += 1
        n = s.periodic ? s.steps_per_window[1] :
            (w <= length(s.steps_per_window) ? s.steps_per_window[w] : throw(ArgumentError(
                "the schedule has $(length(s.steps_per_window)) met windows " *
                "($(sum(s.steps_per_window)) steps); the run has $nsteps steps")))
        step_dt = window_step_dt(wdt, n)
        for k in 1:n
            push!(window, w); push!(first, k == 1); push!(last, k == n); push!(dt, step_dt)
            push!(start_time, window_clock(s.start_time, s.window_dt, w - 1, (k - 1) / n))
            push!(end_time, window_clock(s.start_time, s.window_dt, w - 1, k / n))
        end
    end
    resize!(window, nsteps); resize!(first, nsteps); resize!(last, nsteps)
    resize!(dt, nsteps); resize!(start_time, nsteps); resize!(end_time, nsteps)
    # A run that stops inside a met window never reaches that window's end.
    return _ResolvedSchedule{FT}(window, first, last, dt, wdt, start_time, end_time)
end

_default_schedule(dt) = CSStepSchedule(; physics_cadence = :substep, window_dt = dt)

# The schedule of a call and its step dt: `schedule = nothing` is one step of
# `dt` per window (`dt` defaults to one); a given schedule fixes the step dt,
# and an explicit `dt` must equal it. The returned dt (that of the first
# window) labels the result; each emission record carries its own step dt.
function _call_schedule(schedule, dt, ::Type{FT}) where FT
    if schedule === nothing
        step_dt = dt === nothing ? one(FT) : FT(dt)
        return _default_schedule(step_dt), step_dt
    end
    schedule isa CSStepSchedule || throw(ArgumentError(
        "schedule must be a CSStepSchedule or nothing; got $(typeof(schedule))"))
    step_dt = window_step_dt(FT(schedule.window_dt), first(schedule.steps_per_window))
    if dt !== nothing
        steps = schedule.steps_per_window
        all(n -> window_step_dt(FT(schedule.window_dt), n) == FT(dt), steps) || throw(ArgumentError(
            "dt = $(dt) disagrees with the schedule's step dt window_dt / steps_per_window; " *
            "omit dt when passing a schedule"))
    end
    return schedule, step_dt
end

# ---------------------------------------------------------------------------
# Replay meteorology: the clock and the air-mass reset targets the plan reads
# ---------------------------------------------------------------------------

mutable struct _CSReplayMeteo{DM, AS, AE}
    time                  :: Float64
    binary_substeps       :: Bool
    diffusion_meteo       :: DM
    window_start_air_mass :: AS
    window_end_air_mass   :: AE
end

# Operators read `current_time(meteo)`: the caller's diffusion meteorology when
# given, else the replay clock, which follows production's (`start_time` plus
# the elapsed windows). The tape records it with each diffusion solve, so the
# reverse evaluates time-varying Kz fields at the forward's clock.
current_time(m::_CSReplayMeteo) =
    m.diffusion_meteo === nothing ? m.time : current_time(m.diffusion_meteo)
# The palindrome takes one subcycle per step under a binary window contract.
uses_binary_substep_contract(m::_CSReplayMeteo) = m.binary_substeps
air_mass_reset_target(::WindowStartAirMass, m::_CSReplayMeteo) = m.window_start_air_mass
air_mass_reset_target(::WindowEndAirMass, m::_CSReplayMeteo) = m.window_end_air_mass

# ---------------------------------------------------------------------------
# The replay
# ---------------------------------------------------------------------------

# A window the replay holds, and whether a recording run uses it (its records
# reference the window's met until the driver releases them).
mutable struct _HeldWindow
    const lease :: CSWindowLease
    retained    :: Bool
end

mutable struct _CSReplay{P, ST, G, M, SRC, AS, AW, DW, DS, E, RB, BR, C, CW, CH, ME, FT}
    plan                 :: P
    steps                :: _ResolvedSchedule{FT}
    state                :: ST
    grid                 :: G
    mesh                 :: M
    source               :: SRC      # the met windows (`WindowSource.jl`)
    leases               :: Vector{_HeldWindow}   # the windows the replay holds
    advection            :: AS
    advection_workspace  :: AW
    diffusion_workspace  :: DW       # replay workspace (one tracer), or nothing
    diffusion_scratch    :: DS       # reverse-solve scratch (six column panels), or nothing
    emissions            :: E
    rate_buffer          :: RB
    base_rates           :: BR
    convection           :: C
    convection_workspace :: CW
    chemistry            :: CH       # retained from the production schedule
    meteo                :: ME
end

_replay_nsteps(r::_CSReplay) = length(r.steps.window)
_has_surface_flux(r::_CSReplay) = !(r.emissions isa NoSurfaceFlux)

# The tracer name of the replay state and its surface source.
const _REPLAY_TRACER = :tracer

# The replay grid: the mesh, and a vertical coordinate the step's operators
# never read (they take layer geometry from their workspaces).
function _replay_grid(mesh::CubedSphereMesh{FT}, Nz::Int) where FT
    vertical = HybridSigmaPressure(zeros(FT, Nz + 1), collect(FT, range(0, 1; length = Nz + 1)))
    return AtmosGrid(mesh, vertical, CPU(); FT = FT, radius = mesh.radius)
end

_sequence_at(v::AbstractVector, s) = v[s]
_sequence_at(v, _s) = v

"""
    _build_cs_replay(panels_rm0, panels_m0, am_steps, bm_steps, cm_steps, mesh;
                     scheme, schedule, nsteps, flux_scale, surface_flux,
                     emission_rates, diffusion_op,
                     diffusion_layer_thickness, diffusion_meteo, convection_op,
                     convection_forcing, convection_workspace) -> _CSReplay

A single-tracer replay of steps `1:nsteps` of `schedule`, starting from
`(panels_rm0, panels_m0)`. With `surface_flux = true` the plan carries a
surface-flux operator whose per-cell rates at step `s` are
`emission_rates[s]` (zero when `nothing`); without one the
palindrome center is one diffusion solve `V(dt)`, as production runs a
configuration without surface sources. The caller's convection workspace is
used, with the persistent state of a fresh run. The replay owns its diffusion
workspace and the reverse solves' scratch; `diffusion_layer_thickness` (six
panels, or one per step: a met window's steps share its tuple) is copied into
that workspace at each step and is never written. The met inputs are grouped
by met window into a `ResidentCSWindowSource` (`WindowSource.jl`), which
validates each window once.
"""
function _build_cs_replay(panels_rm0, panels_m0, panels_am_steps, panels_bm_steps,
                          panels_cm_steps, mesh::CubedSphereMesh;
                          scheme, schedule::CSStepSchedule, nsteps::Int,
                          flux_scale = one(eltype(panels_m0[1])),
                          surface_flux::Bool = true,
                          emission_rates = nothing,
                          diffusion_op = NoDiffusion(),
                          diffusion_layer_thickness = nothing,
                          diffusion_meteo = nothing,
                          convection_op = NoConvection(),
                          convection_forcing = nothing,
                          convection_workspace = nothing)
    FT = eltype(panels_m0[1])
    Nz = size(panels_m0[1], 3)
    1 <= nsteps <= length(panels_am_steps) || throw(ArgumentError(
        "replay of $nsteps steps with $(length(panels_am_steps)) flux steps"))
    steps = _resolve_schedule(schedule, nsteps, FT)
    surface_flux || emission_rates === nothing || throw(ArgumentError(
        "emission rates given to a replay without a surface-flux operator"))

    m = ntuple(p -> copy(panels_m0[p]), 6)
    rm = ntuple(p -> copy(panels_rm0[p]), 6)
    state = CubedSphereState(DryBasis, mesh, m; (_REPLAY_TRACER => rm,)...)
    fill_panel_halos!(state.air_mass, mesh; dir = 0)
    fill_panel_halos!(state.tracers_raw, mesh; dir = 0)
    grid = _replay_grid(mesh, Nz)
    advection_ws = _cs_advection_workspace_for(scheme, state, grid)

    uses_diffusion = any(s -> !(_sequence_at(diffusion_op, s) isa NoDiffusion), 1:nsteps)
    diffusion_ws = uses_diffusion ? DiffusionWorkspace(state) : nothing
    # One scratch for every reverse solve: each overwrites a column before it
    # reads it, so scratch carries nothing between solves (or windows).
    diffusion_scratch = uses_diffusion ? map(similar, diffusion_ws.factors) : nothing
    source = ResidentCSWindowSource(panels_am_steps, panels_bm_steps, panels_cm_steps, steps.window;
                                    flux_scale = FT(flux_scale), convection_forcing, diffusion_op,
                                    layer_thickness = diffusion_layer_thickness,
                                    start_air_mass = schedule.window_start_air_mass,
                                    end_air_mass = schedule.window_end_air_mass,
                                    layer_thickness_like = uses_diffusion ? diffusion_ws.layer_thickness : nothing,
                                    air_mass_like = panels_m0)

    rate_buffer = surface_flux ? _zero_surface_rates(mesh, panels_m0[1]) : nothing
    emissions = surface_flux ?
        SurfaceFluxOperator(SurfaceFluxSource(_REPLAY_TRACER, rate_buffer)) : NoSurfaceFlux()

    convection_op isa NoConvection || reset_operator_state!(convection_workspace)

    meteo = _CSReplayMeteo(schedule.start_time, schedule.binary_substeps, diffusion_meteo,
                           _first_target(schedule.window_start_air_mass),
                           _first_target(schedule.window_end_air_mass))
    return _CSReplay(schedule.plan, steps, state, grid, mesh, source, _HeldWindow[], scheme,
                     advection_ws, diffusion_ws, diffusion_scratch, emissions, rate_buffer,
                     emission_rates, convection_op, convection_workspace, schedule.chemistry, meteo)
end

_first_target(::Nothing) = nothing
_first_target(targets) = first(targets)

# ---------------------------------------------------------------------------
# The windows the replay holds
# ---------------------------------------------------------------------------

# The met of window `w`, acquired from the source unless the replay holds it.
# A recording run (`retain = true`) retains every window it reads; the
# windows no recording has read are released when the next one is acquired.
function _window_met!(r::_CSReplay, w::Int, retain::Bool)
    for held in Iterators.reverse(r.leases)      # the current window is the last one acquired
        held.lease.met.window == w || continue
        held.retained |= retain
        return held.lease.met
    end
    _release_windows!(r, held -> !held.retained)
    lease = acquire_window!(r.source, w)
    push!(r.leases, _HeldWindow(lease, retain))
    return lease.met
end

"""
    _release_replay_windows!(replay)

Return every window the replay holds to its source. Drivers call it after the
records that reference the windows' met are released (`_release_records!`),
and in their `finally` blocks.
"""
_release_replay_windows!(r::_CSReplay) = _release_windows!(r, _ -> true)

function _release_windows!(r::_CSReplay, which)
    any(which, r.leases) || return r
    kept = _HeldWindow[]
    for held in r.leases
        which(held) ? release_window!(r.source, held.lease) : push!(kept, held)
    end
    copy!(r.leases, kept)
    return r
end

# A recording run's records reference the met of the windows it reads.
_retains_windows(::NoStepRecorder) = false
_retains_windows(::AbstractStepRecorder) = true

# Met inputs of step `s` that are not prognostic state: the step's emission
# rates in the surface source, the layer thickness of its diffusion, the clock
# and the window's reset targets; at a window's first step the convection
# CFL cache is stale (production invalidates it when it installs the window's
# convection forcing). Returns the met of the step's window.
function _load_step!(r::_CSReplay, s::Int, retain::Bool)
    met = _window_met!(r, r.steps.window[s], retain)
    r.steps.first[s] && invalidate_cmfmc_cache!(r.convection_workspace)
    if r.rate_buffer !== nothing
        rates = r.base_rates === nothing ? nothing : r.base_rates[s]
        for p in 1:6
            rates === nothing ? fill!(r.rate_buffer[p], zero(eltype(r.rate_buffer[p]))) :
                                copyto!(r.rate_buffer[p], rates[p])
        end
    end
    layer_thickness = met.layer_thickness
    if !(met.diffusion_op isa NoDiffusion) && layer_thickness !== nothing
        for (dst, src) in zip(r.diffusion_workspace.layer_thickness, layer_thickness)
            size(dst) == size(src) || throw(DimensionMismatch(
                "diffusion_layer_thickness panel has shape $(size(src)); expected $(size(dst))"))
            copyto!(dst, src)
        end
    end
    r.meteo.time = r.steps.start_time[s]
    r.meteo.window_start_air_mass = met.start_air_mass
    r.meteo.window_end_air_mass = met.end_air_mass
    return met
end

function _replay_operators(r::_CSReplay, s::Int, met::CSWindowMet)
    diffusion_op = met.diffusion_op
    return StepOperators(r.state, _fluxes_at(met, s), r.grid, r.advection, r.advection_workspace,
                         diffusion_op, diffusion_op isa NoDiffusion ? nothing : r.diffusion_workspace,
                         r.emissions, r.convection, r.convection_workspace,
                         met.convection_forcing, r.chemistry)
end

"""
    _replay_steps!(replay, steps; recorder = NoStepRecorder(), on_step_end = nothing)

Advance the replay through `steps` (a contiguous range starting where the
replay state is), executing each step's plan phases in production order:
window advance (windows after the first) and window start at a window's first
step, the step stages, and the window end after its last step. `recorder`
observes every operator; `on_step_end(s, state)` sees the state after step
`s`, where runtime output and the observation sampler see it. A run with a
recorder retains the windows it reads until `_release_replay_windows!`.
"""
function _replay_steps!(r::_CSReplay, steps::AbstractUnitRange;
                        recorder::AbstractStepRecorder = NoStepRecorder(),
                        on_step_end = nothing)
    retain = _retains_windows(recorder)
    for s in steps
        met = _load_step!(r, s, retain)
        ops = _replay_operators(r, s, met)
        timing = StepTiming(r.steps.dt[s], r.steps.window_dt)
        _begin_recorded_step!(recorder, s, met.layer_thickness, current_time(r.meteo))
        if r.steps.first[s]
            r.steps.window[s] > 1 && execute_window_advance!(r.plan, ops, timing, r.meteo, recorder)
            execute_window_start!(r.plan, ops, timing, r.meteo, recorder)
        end
        execute_step_stages!(r.plan, ops, timing, r.meteo, recorder)
        r.meteo.time = r.steps.end_time[s]
        r.steps.last[s] && execute_window_end!(r.plan, ops, timing, r.meteo, recorder)
        _end_recorded_step!(recorder, s)
        on_step_end === nothing || on_step_end(s, r.state)
    end
    return r
end

@inline _begin_recorded_step!(::NoStepRecorder, _s, _layer_thickness, _time) = nothing
@inline _end_recorded_step!(::NoStepRecorder, _s) = nothing

# The single-tracer panels of the replay state, as the reverse pass and the
# objectives read them (3D halo-padded panels sharing the state's memory).
_replay_rm(r::_CSReplay) = _single_tracer(r.state.tracers_raw)
_replay_m(r::_CSReplay) = r.state.air_mass

_single_tracer(rm::NTuple{6, <:AbstractArray{<:Any, 3}}) = rm
function _single_tracer(rm::NTuple{6, <:AbstractArray{<:Any, 4}})
    size(rm[1], 4) == 1 || throw(ArgumentError(
        "the cubed-sphere adjoint records one tracer; got $(size(rm[1], 4))"))
    return ntuple(p -> reshape(rm[p], size(rm[p], 1), size(rm[p], 2), size(rm[p], 3)), 6)
end

# Checkpoints: the prognostic state with its halos and the persistent operator
# state, so a restart reproduces the recording run bit for bit.
function _replay_snapshot(r::_CSReplay)
    return (rm = map(copy, r.state.tracers_raw), m = map(copy, r.state.air_mass),
            emissions = operator_state(r.emissions),
            convection = operator_state(r.convection_workspace),
            diffusion = operator_state(r.diffusion_workspace))
end

function _replay_restore!(r::_CSReplay, snap)
    foreach(copyto!, r.state.tracers_raw, snap.rm)
    foreach(copyto!, r.state.air_mass, snap.m)
    restore_operator_state!(r.emissions, snap.emissions)
    restore_operator_state!(r.convection_workspace, snap.convection)
    restore_operator_state!(r.diffusion_workspace, snap.diffusion)
    return r
end
