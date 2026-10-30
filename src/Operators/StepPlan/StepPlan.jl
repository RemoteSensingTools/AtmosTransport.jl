"""
    StepPlans

The model step as data: which operators run, in which order, with which time
step, at every advection step and at the met-window boundaries.

A [`StepPlan`](@ref) is built once from the run configuration (physics
cadence, the transport binary's window contract, the air-mass reset mode).
Production executes it: `TransportModel.step!` runs its `step` stages and
`DrivenSimulation` runs all four phases. The cubed-sphere adjoint executes the
same plan with a tape recorder (`AbstractStepRecorder`) and walks the recorded
tape backwards, so the forward model it differentiates is the production step
by construction.

Phases, in the order a met window runs them:

| phase            | runs                                                   |
|------------------|--------------------------------------------------------|
| `window_advance` | on entering a met window after the first (air-mass reset to the new window) |
| `window_start`   | before the first step of every met window              |
| `step`           | every advection step                                   |
| `window_end`     | after the last step of every met window                |
"""
module StepPlans

using ...SectionTimer
using ...Architectures: copy_arrays!
using ...State: CellState, CubedSphereState, ConstantField, ntracers, FaceIndexedFluxState
using ...Grids: AtmosGrid, CubedSphereMesh
import ..apply!
import ..Advection: apply_midpoint!
using ..Advection: AbstractMidpointHook
import ..operator_state, ..restore_operator_state!, ..reset_operator_state!
using ..Operators: _copy_state, _copy_state!, _zero_state!
import ..AbstractStepRecorder, ..NoStepRecorder,
       ..record_center!, ..record_diffusion!, ..record_emission!,
       ..record_convection!, ..record_air_mass_reset!, ..record_chemistry!
using ..Advection: fill_panel_halos!, AbstractAdvectionScheme, LinRoodPPMScheme, NoAdvection,
                   CSAdvectionWorkspace, CSLinRoodAdvectionWorkspace, needs_column_scratch,
                   AbstractBinaryCFLCheck, NoBinaryCFLCheck, BinaryCFLCheck
using ..Diffusion: AbstractDiffusion, NoDiffusion, DiffusionWorkspace,
                   apply_vertical_diffusion_vmr!, preflight_diffusion,
                   uses_diffusive_surface_flux_boundary, surface_flux_coupling, surface_flux_center!
using ..SurfaceFlux: AbstractSurfaceFluxOperator, NoSurfaceFlux, SurfaceFluxOperator,
                     apply_surface_flux!, center_emission_deposit
using ..Convection: AbstractConvection, NoConvection
using ..Chemistry: AbstractChemistryOperator, NoChemistry, ExponentialDecay, CompositeChemistry

include("midpoint.jl")

export StepPlan, StepOperators, StepTiming
export AbstractStepStage, TransportStage, ConvectionStage, ChemistryBeforeStage,
       ChemistryAfterStage, AirMassResetStage
export AbstractStageDt, StepDt, WindowDt, WindowStartAirMass, WindowEndAirMass
export AbstractPhysicsCadence, SubstepPhysics, WindowPhysics
export execute_stage!, execute_stages!, execute_window_advance!, execute_window_start!,
       execute_step_stages!, execute_window_end!
export plan_stages

# Clock arithmetic shared by driven stepping and adjoint replay. Preserve the
# conversion and operation order: window ends must agree bit for bit.
"""Run clock at `fraction` of window `offset`, relative to `start_time`."""
window_clock(start_time, window_dt, offset::Integer, fraction::Float64) =
    start_time + Float64(window_dt) * (offset + fraction)

"""Step duration in the precision of `window_dt`, with one division."""
window_step_dt(window_dt::FT, steps_per_window::Integer) where FT =
    window_dt / FT(steps_per_window)

export window_clock, window_step_dt

# ---------------------------------------------------------------------------
# Configuration vocabulary (also used by the runner's config checks)
# ---------------------------------------------------------------------------

# `[run] physics_cadence`: `:window`, `:substep` or `nothing` for `:window`.
const _PHYSICS_CADENCES = (:window, :substep)

function _resolve_physics_cadence(physics_cadence)
    physics_cadence === nothing && return :window
    cadence = Symbol(physics_cadence)
    cadence in _PHYSICS_CADENCES || throw(ArgumentError(
        "physics_cadence must be :window or :substep; got $(repr(physics_cadence))"))
    return cadence
end

const _AIR_MASS_RESET_MODES = (:none, :preserve_vmr, :preserve_tracer_mass)

function _normalize_air_mass_reset_mode(air_mass_reset_mode)
    mode = air_mass_reset_mode === nothing ? :none : Symbol(air_mass_reset_mode)
    mode in _AIR_MASS_RESET_MODES ||
        throw(ArgumentError("air_mass_reset_mode must be one of :none, " *
                            ":preserve_vmr, or :preserve_tracer_mass; got $(repr(mode))"))
    return mode
end

# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

"""
    AbstractStageDt

The time step a stage integrates over: [`StepDt`](@ref) (the advection step)
or [`WindowDt`](@ref) (the met window).
"""
abstract type AbstractStageDt end
"""    StepDt()  — the stage integrates over the advection step `dt`."""
struct StepDt <: AbstractStageDt end
"""    WindowDt()  — the stage integrates over the met window `window_dt`."""
struct WindowDt <: AbstractStageDt end

"""
    StepTiming(dt, window_dt = dt)

The time steps of one advection step: its own `dt` and the length
`window_dt` of the met window it belongs to.
"""
struct StepTiming{T}
    dt        :: T
    window_dt :: T
end
StepTiming(dt) = StepTiming(dt, dt)
StepTiming(dt, window_dt) = StepTiming(promote(dt, window_dt)...)

@inline stage_dt(::StepDt, timing::StepTiming) = timing.dt
@inline stage_dt(::WindowDt, timing::StepTiming) = timing.window_dt

"""
    AbstractStepStage

One operator application of a [`StepPlan`](@ref). Concrete stages:
[`TransportStage`](@ref), [`ConvectionStage`](@ref),
[`ChemistryBeforeStage`](@ref), [`ChemistryAfterStage`](@ref) and
[`AirMassResetStage`](@ref). `execute_stage!(stage, ops, timing, meteo,
recorder)` applies it.
"""
abstract type AbstractStepStage end

"""
    TransportStage()

The transport block of one advection step: the step-start halo refresh, the
advection palindrome and its center (vertical diffusion and surface fluxes in
the diffusion operator's coupling, or one diffusion solve `V(dt)` without a
surface-flux operator).
"""
struct TransportStage <: AbstractStepStage end

"""
    ConvectionStage(StepDt() | WindowDt())

The convection operator over the advection step or the met window.
"""
struct ConvectionStage{D <: AbstractStageDt} <: AbstractStepStage
    dt :: D
end

"""
    ChemistryBeforeStage(StepDt() | WindowDt())

The first half-step of a symmetrically split chemistry operator (constant-rate
exponential decay); a no-op for other chemistry.
"""
struct ChemistryBeforeStage{D <: AbstractStageDt} <: AbstractStepStage
    dt :: D
end

"""
    ChemistryAfterStage(StepDt() | WindowDt())

The second half-step of a symmetrically split chemistry operator, or the full
step of any other chemistry.
"""
struct ChemistryAfterStage{D <: AbstractStageDt} <: AbstractStepStage
    dt :: D
end

"""    WindowStartAirMass()  — the reset target is the air mass of the met window just entered."""
struct WindowStartAirMass end
"""    WindowEndAirMass()  — the reset target is the window's authoritative end air mass."""
struct WindowEndAirMass end

"""
    AirMassResetStage(mode, target)

Replace the prognostic air mass by the met target (`WindowStartAirMass()` or
`WindowEndAirMass()`), preserving the tracer mass (`:preserve_tracer_mass`)
or the mixing ratio (`:preserve_vmr`).
"""
struct AirMassResetStage{P <: Union{WindowStartAirMass, WindowEndAirMass}} <: AbstractStepStage
    mode   :: Symbol
    target :: P
end

# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

"""
    AbstractPhysicsCadence

When the physics blocks (convection, chemistry) run: every advection step
([`SubstepPhysics`](@ref)) or once per met window ([`WindowPhysics`](@ref)).
"""
abstract type AbstractPhysicsCadence end
"""    SubstepPhysics()  — convection and chemistry run every advection step."""
struct SubstepPhysics <: AbstractPhysicsCadence end
"""    WindowPhysics()  — convection and chemistry run once per met window."""
struct WindowPhysics <: AbstractPhysicsCadence end

"""
    StepPlan(; physics_cadence = nothing, window_contract = false,
             air_mass_reset = :none)

The model step of a run: the stages of its four phases (`window_advance`,
`window_start`, `step`, `window_end`).

- Without a window contract, or with `physics_cadence = :substep`, every step
  runs `ChemistryBefore(dt) → Transport → Convection(dt) → ChemistryAfter(dt)`
  (constant-rate decay split symmetrically around transport).
- With a transport binary's per-window contract (`window_contract = true`)
  and `physics_cadence = :window` (the default), every step runs the
  transport block only; the window starts with the first decay half over the
  window and ends with the air-mass reset to the window's end mass, convection
  over the window and the closing chemistry.
- `air_mass_reset` other than `:none` resets the air mass to each newly
  entered window (`window_advance`) and, with a window contract, to the
  window's end mass before the window-end physics.

`TransportModel.step!` executes the `step` stages of the default plan;
`DrivenSimulation` executes all phases of the plan of its configuration; the
cubed-sphere adjoint records and transposes the same plan.
"""
struct StepPlan{C <: AbstractPhysicsCadence, A <: Tuple, S <: Tuple, T <: Tuple, E <: Tuple}
    cadence        :: C
    window_advance :: A
    window_start   :: S
    step           :: T
    window_end     :: E
end

function StepPlan(; physics_cadence = nothing, window_contract::Bool = false,
                  air_mass_reset = :none)
    cadence = _resolve_physics_cadence(physics_cadence)
    mode = _normalize_air_mass_reset_mode(air_mass_reset)
    advance = mode === :none ? () : (AirMassResetStage(mode, WindowStartAirMass()),)
    end_reset = mode !== :none && window_contract ?
        (AirMassResetStage(mode, WindowEndAirMass()),) : ()
    if window_contract && cadence === :window
        return StepPlan(WindowPhysics(), advance,
                        (ChemistryBeforeStage(WindowDt()),),
                        (TransportStage(),),
                        (end_reset..., ConvectionStage(WindowDt()),
                         ChemistryAfterStage(WindowDt())))
    end
    return StepPlan(SubstepPhysics(), advance, (),
                    (ChemistryBeforeStage(StepDt()), TransportStage(),
                     ConvectionStage(StepDt()), ChemistryAfterStage(StepDt())),
                    end_reset)
end

"""
    plan_stages(plan) -> Tuple

Every stage of `plan`, in phase order (`window_advance`, `window_start`,
`step`, `window_end`).
"""
plan_stages(plan::StepPlan) =
    (plan.window_advance..., plan.window_start..., plan.step..., plan.window_end...)

_stage_label(::TransportStage) = "Transport"
_stage_label(s::ConvectionStage) = "Convection(" * _dt_label(s.dt) * ")"
_stage_label(s::ChemistryBeforeStage) = "ChemistryBefore(" * _dt_label(s.dt) * ")"
_stage_label(s::ChemistryAfterStage) = "ChemistryAfter(" * _dt_label(s.dt) * ")"
_stage_label(s::AirMassResetStage{WindowStartAirMass}) = "AirMassReset($(s.mode), window start)"
_stage_label(s::AirMassResetStage{WindowEndAirMass}) = "AirMassReset($(s.mode), window end)"
_dt_label(::StepDt) = "dt"
_dt_label(::WindowDt) = "window_dt"
_phase_label(stages) = isempty(stages) ? "—" : join(map(_stage_label, stages), " → ")

function Base.show(io::IO, ::MIME"text/plain", plan::StepPlan)
    println(io, "StepPlan (", nameof(typeof(plan.cadence)), ")")
    println(io, "  window_advance: ", _phase_label(plan.window_advance))
    println(io, "  window_start:   ", _phase_label(plan.window_start))
    println(io, "  step:           ", _phase_label(plan.step))
    print(io,   "  window_end:     ", _phase_label(plan.window_end))
end

# ---------------------------------------------------------------------------
# What a plan acts on
# ---------------------------------------------------------------------------

"""
    StepOperators(; state, fluxes, grid, advection, advection_workspace,
                  diffusion = NoDiffusion(), diffusion_workspace = nothing,
                  emissions = NoSurfaceFlux(), convection = NoConvection(),
                  convection_workspace = nothing, convection_forcing = nothing,
                  chemistry = NoChemistry())

The state, the step's face fluxes and the operators with their workspaces that
the stages of a [`StepPlan`](@ref) act on. `TransportModel` provides them
(`step_operators(model)`); the cubed-sphere adjoint builds them for its
single-tracer replay.
"""
struct StepOperators{S, F, G, A, AW, D, DW, E, C, CW, CF, Ch}
    state                :: S
    fluxes               :: F
    grid                 :: G
    advection            :: A
    advection_workspace  :: AW
    diffusion            :: D
    diffusion_workspace  :: DW
    emissions            :: E
    convection           :: C
    convection_workspace :: CW
    convection_forcing   :: CF
    chemistry            :: Ch
end

StepOperators(; state, fluxes, grid, advection, advection_workspace,
              diffusion = NoDiffusion(), diffusion_workspace = nothing,
              emissions = NoSurfaceFlux(), convection = NoConvection(),
              convection_workspace = nothing, convection_forcing = nothing,
              chemistry = NoChemistry()) =
    StepOperators(state, fluxes, grid, advection, advection_workspace, diffusion,
                  diffusion_workspace, emissions, convection, convection_workspace,
                  convection_forcing, chemistry)

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

"""
    execute_stages!(stages::Tuple, ops, timing, meteo, recorder = NoStepRecorder())

Apply `stages` in order. `meteo` is the run clock and met context
(`current_time(meteo)`, the air-mass reset targets); `recorder` observes every
operator.
"""
@inline execute_stages!(::Tuple{}, _ops, _timing, _meteo,
                        _recorder::AbstractStepRecorder = NoStepRecorder()) = nothing
@inline function execute_stages!(stages::Tuple, ops, timing, meteo,
                                 recorder::AbstractStepRecorder = NoStepRecorder())
    execute_stage!(first(stages), ops, timing, meteo, recorder)
    return execute_stages!(Base.tail(stages), ops, timing, meteo, recorder)
end

"""    execute_window_advance!(plan, ops, timing, meteo, recorder = NoStepRecorder())"""
execute_window_advance!(plan::StepPlan, ops, timing, meteo,
                        recorder::AbstractStepRecorder = NoStepRecorder()) =
    execute_stages!(plan.window_advance, ops, timing, meteo, recorder)
"""    execute_window_start!(plan, ops, timing, meteo, recorder = NoStepRecorder())"""
execute_window_start!(plan::StepPlan, ops, timing, meteo,
                      recorder::AbstractStepRecorder = NoStepRecorder()) =
    execute_stages!(plan.window_start, ops, timing, meteo, recorder)
"""    execute_step_stages!(plan, ops, timing, meteo, recorder = NoStepRecorder())"""
execute_step_stages!(plan::StepPlan, ops, timing, meteo,
                     recorder::AbstractStepRecorder = NoStepRecorder()) =
    execute_stages!(plan.step, ops, timing, meteo, recorder)
"""    execute_window_end!(plan, ops, timing, meteo, recorder = NoStepRecorder())"""
execute_window_end!(plan::StepPlan, ops, timing, meteo,
                    recorder::AbstractStepRecorder = NoStepRecorder()) =
    execute_stages!(plan.window_end, ops, timing, meteo, recorder)

"""
    execute_stage!(stage, ops, timing, meteo, recorder)

Apply one stage to `ops.state`.
"""
function execute_stage! end

# Transport. The production call (`NoStepRecorder`) is the plain topology
# `apply!`; a recorder rides along on the cubed-sphere palindrome.
execute_stage!(::TransportStage, ops::StepOperators, timing::StepTiming, meteo,
               recorder::AbstractStepRecorder) =
    _transport!(ops, timing.dt, meteo, recorder)

function _transport!(ops::StepOperators, dt, meteo, recorder::AbstractStepRecorder)
    SectionTimer.@section :advection transport_step!(
        ops.state, ops.fluxes, ops.grid, ops.advection, dt;
        workspace = ops.advection_workspace,
        diffusion_workspace = ops.diffusion_workspace,
        diffusion_op = ops.diffusion,
        emissions_op = ops.emissions,
        meteo, recorder)
    return nothing
end

# Convection. Convection leaves the air mass unchanged, so the recorder reads
# it after the call, together with what the call decided (the CFL sub-step
# count of `CMFMCConvection`).
execute_stage!(stage::ConvectionStage, ops::StepOperators, timing::StepTiming, meteo,
               recorder::AbstractStepRecorder) =
    _convection!(ops.convection, ops, stage_dt(stage.dt, timing), recorder)

@inline _convection!(::NoConvection, ::StepOperators, _dt, _recorder) = nothing

function _convection!(op::AbstractConvection, ops::StepOperators, dt, recorder)
    SectionTimer.@section :convection apply!(ops.state, ops.convection_forcing,
                                             ops.grid, op, dt;
                                             workspace = ops.convection_workspace)
    record_convection!(recorder, op, ops.convection_forcing, ops.state.air_mass, dt,
                       ops.convection_workspace)
    return nothing
end

# Chemistry. Constant-rate ExponentialDecay is linear and autonomous over a
# met window, so two exact half-steps place the window-integrated emissions at
# the window midpoint. A time-varying rate is not autonomous and retains the
# historical end-step order: sampling its two halves on opposite sides of a
# forcing boundary would apply part of the next interval's rate to this
# interval. NoChemistry is included so the default remains a compile-time
# no-op. A CompositeChemistry is split only when every member has the same
# safe property; future/non-decay chemistry retains the historical end-step
# order.
@inline _symmetric_decay_split(
    ::ExponentialDecay{FT, N, R}) where {FT, N, R <: NTuple{N, ConstantField}} = true
@inline _symmetric_decay_split(::ExponentialDecay) = false
@inline _symmetric_decay_split(::NoChemistry) = true
@inline _symmetric_decay_split(op::CompositeChemistry) =
    all(_symmetric_decay_split, op.schemes)
@inline _symmetric_decay_split(::AbstractChemistryOperator) = false

@inline _half_step(dt) = dt / 2

function execute_stage!(stage::ChemistryBeforeStage, ops::StepOperators, timing::StepTiming,
                        meteo, recorder::AbstractStepRecorder)
    _symmetric_decay_split(ops.chemistry) || return nothing
    dt = _half_step(stage_dt(stage.dt, timing))
    record_chemistry!(recorder, ops.chemistry, dt)
    SectionTimer.@section :chemistry apply!(ops.state, meteo, ops.grid, ops.chemistry, dt)
    return nothing
end

function execute_stage!(stage::ChemistryAfterStage, ops::StepOperators, timing::StepTiming,
                        meteo, recorder::AbstractStepRecorder)
    full = stage_dt(stage.dt, timing)
    dt = _symmetric_decay_split(ops.chemistry) ? _half_step(full) : full
    record_chemistry!(recorder, ops.chemistry, dt)
    SectionTimer.@section :chemistry apply!(ops.state, meteo, ops.grid, ops.chemistry, dt)
    return nothing
end

# Air-mass reset to the met target of the window.
"""
    air_mass_reset_target(target, meteo) -> air mass

The air mass an [`AirMassResetStage`](@ref) installs: the air mass of the met
window just entered (`WindowStartAirMass()`) or the window's end air mass
(`WindowEndAirMass()`), as the run context `meteo` provides it
(`DrivenSimulation`, or the adjoint's replay meteorology).
"""
function air_mass_reset_target end

function execute_stage!(stage::AirMassResetStage, ops::StepOperators, timing::StepTiming,
                        meteo, recorder::AbstractStepRecorder)
    target = air_mass_reset_target(stage.target, meteo)
    record_air_mass_reset!(recorder, stage.mode, ops.state, target)
    _reset_air_mass!(ops.state, target, ops.grid.horizontal, stage.mode)
    return nothing
end

# ---------------------------------------------------------------------------
# Air-mass reset (state halos, preserve-VMR and preserve-tracer-mass)
# ---------------------------------------------------------------------------

_refresh_state_halos!(state, _mesh) = state

function _refresh_state_halos!(state::CubedSphereState, mesh::CubedSphereMesh)
    fill_panel_halos!(state.air_mass, mesh; dir = 1)
    fill_panel_halos!(state.tracers_raw, mesh; dir = 1)
    return state
end

function _reset_air_mass_preserve_vmr!(state::CellState, new_air_mass, _mesh)
    old_air_mass = state.air_mass
    FT = eltype(old_air_mass)
    floor_m = eps(FT)
    # One broadcast over the packed tracer array: the air masses broadcast
    # along its trailing tracer axis.
    raw = state.tracers_raw
    @. raw = ifelse(old_air_mass > floor_m,
                    raw / old_air_mass * new_air_mass,
                    zero(FT))
    copyto!(old_air_mass, new_air_mass)
    return state
end

function _reset_air_mass_preserve_vmr!(state::CubedSphereState,
                                       new_air_mass::NTuple{6},
                                       mesh::CubedSphereMesh)
    fill_panel_halos!(new_air_mass, mesh; dir = 1)
    FT = eltype(state.air_mass[1])
    floor_m = eps(FT)
    for p in 1:6
        old_p = state.air_mass[p]
        new_p = new_air_mass[p]
        raw_p = state.tracers_raw[p]
        # One broadcast per panel over all tracers (trailing axis).
        @. raw_p = ifelse(old_p > floor_m,
                          raw_p / old_p * new_p,
                          zero(FT))
    end
    # One queued launch for the six panels (each panel's rescaling above read
    # only its own old air mass); see `copy_arrays!`.
    copy_arrays!(state.air_mass, new_air_mass)
    return _refresh_state_halos!(state, mesh)
end

function _reset_air_mass_preserve_tracer_mass!(state::CellState, new_air_mass, _mesh)
    copyto!(state.air_mass, new_air_mass)
    return state
end

function _reset_air_mass_preserve_tracer_mass!(state::CubedSphereState,
                                               new_air_mass::NTuple{6},
                                               mesh::CubedSphereMesh)
    fill_panel_halos!(new_air_mass, mesh; dir = 1)
    copy_arrays!(state.air_mass, new_air_mass)
    return _refresh_state_halos!(state, mesh)
end

function _reset_air_mass!(state, new_air_mass, mesh, mode::Symbol)
    mode === :none && return state
    mode === :preserve_vmr &&
        return _reset_air_mass_preserve_vmr!(state, new_air_mass, mesh)
    mode === :preserve_tracer_mass &&
        return _reset_air_mass_preserve_tracer_mass!(state, new_air_mass, mesh)
    throw(ArgumentError("unknown air_mass_reset_mode $(repr(mode))"))
end

# ---------------------------------------------------------------------------
# Cubed-sphere advection workspaces (shared by TransportModel and the adjoint
# replay, so both run the palindrome with the same storage)
# ---------------------------------------------------------------------------

_cs_advection_workspace_for(scheme::AbstractAdvectionScheme,
                            state::CubedSphereState,
                            grid::AtmosGrid{<:CubedSphereMesh};
                            binary_cfl_check::AbstractBinaryCFLCheck = NoBinaryCFLCheck()) =
    CSAdvectionWorkspace(grid.horizontal, state.air_mass[1];
                         n_tracers = ntracers(state),
                         column_scratch = needs_column_scratch(scheme),
                         binary_cfl_check)

# Lin-Rood and no advection take no subcycle count from the binary, so they
# cannot honor `BinaryCFLCheck()`.
function _cs_advection_workspace_for(scheme::LinRoodPPMScheme,
                                     state::CubedSphereState,
                                     grid::AtmosGrid{<:CubedSphereMesh};
                                     binary_cfl_check::AbstractBinaryCFLCheck = NoBinaryCFLCheck())
    _reject_binary_cfl_check(binary_cfl_check, scheme)
    return CSLinRoodAdvectionWorkspace(grid.horizontal, state.air_mass[1];
                                       column_scratch = needs_column_scratch(scheme))
end

# No advection means no advection buffers. Diffusion has an independent
# workspace and therefore does not affect this dispatch.
function _cs_advection_workspace_for(scheme::NoAdvection,
                                     state::CubedSphereState,
                                     grid::AtmosGrid{<:CubedSphereMesh};
                                     binary_cfl_check::AbstractBinaryCFLCheck = NoBinaryCFLCheck())
    _reject_binary_cfl_check(binary_cfl_check, scheme)
    return nothing
end

_reject_binary_cfl_check(::NoBinaryCFLCheck, scheme) = nothing
_reject_binary_cfl_check(::BinaryCFLCheck, scheme) = throw(ArgumentError(
    "BinaryCFLCheck() needs a split-sweep advection scheme; $(nameof(typeof(scheme))) " *
    "takes no subcycle count from the binary."))

# ---------------------------------------------------------------------------
# Persistent operator state
#
# Some operators keep state from one call to the next: the Kahan compensation
# of the surface sources, the column ledger carries of convection and Dkg
# diffusion, and the CFL sub-step count `CMFMCConvection` caches for a met
# window. The cubed-sphere adjoint restarts its replay at checkpoints; to
# reproduce the production trajectory bit for bit it saves this state with
# the tracer and air mass (`operator_state`), restores it
# (`restore_operator_state!`), and starts each replay from the state of a
# fresh run (`reset_operator_state!`). Production never calls these.
# ---------------------------------------------------------------------------

# Surface fluxes: each source's Kahan compensation.
operator_state(op::SurfaceFluxOperator) =
    map(src -> _copy_state(src.compensation), op.flux_map.sources)
restore_operator_state!(op::SurfaceFluxOperator, snap::Tuple) =
    (foreach((src, s) -> _copy_state!(src.compensation, s), op.flux_map.sources, snap); nothing)
reset_operator_state!(op::SurfaceFluxOperator) =
    (foreach(src -> _zero_state!(src.compensation), op.flux_map.sources); nothing)

# Dkg diffusion: the ledger carry.
operator_state(ws::DiffusionWorkspace) = (carry = _copy_state(ws.ledger_carry),)
restore_operator_state!(ws::DiffusionWorkspace, snap::NamedTuple) =
    (_copy_state!(ws.ledger_carry, snap.carry); nothing)
reset_operator_state!(ws::DiffusionWorkspace) = (_zero_state!(ws.ledger_carry); nothing)

end # module StepPlans
