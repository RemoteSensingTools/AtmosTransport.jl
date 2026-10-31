"""
    TransportMidpoint(; dt, workspace, diffusion_op = NoDiffusion(),
                      diffusion_workspace = nothing, emissions_op = NoSurfaceFlux(),
                      tracer_names = nothing, meteo = nothing, grid = nothing,
                      halo_width = nothing, recorder = NoStepRecorder())

StepPlan's column-local emission/diffusion composition. Advection supplies the
current buffers (and, for RG/Lin–Rood, the tracer index). All fields are concrete;
no midpoint captures a binding reassigned by the sweep loop.
"""
struct TransportMidpoint{D, W, E, A, N, M, G, T, H, R, P} <: AbstractMidpointHook
    diffusion::D
    diffusion_workspace::W
    emissions::E
    workspace::A
    tracer_names::N
    meteo::M
    grid::G
    dt::T
    halo_width::H
    recorder::R
    deposit::P
end

function TransportMidpoint(; dt, workspace, diffusion_op = NoDiffusion(),
                            diffusion_workspace = nothing, emissions_op = NoSurfaceFlux(),
                            tracer_names = nothing, meteo = nothing, grid = nothing,
                            halo_width = nothing, recorder = NoStepRecorder())
    emissions_op isa NoSurfaceFlux || tracer_names !== nothing || throw(ArgumentError(
        "TransportMidpoint: active surface flux requires `tracer_names`"))
    emissions_op isa NoSurfaceFlux || dt !== nothing || throw(ArgumentError(
        "TransportMidpoint: active surface flux requires the step `dt`"))
    deposit = center_emission_deposit(surface_flux_coupling(diffusion_op), diffusion_op)
    return TransportMidpoint(diffusion_op, diffusion_workspace, emissions_op, workspace,
                             tracer_names, meteo, grid, dt, halo_width, recorder, deposit)
end

# Callable objects retain the existing Diffusion-owned coupling order, with
# exactly the original dt/2 expression and full-dt emission call.
struct _MidpointDiffusion{H, R, M, T}
    hook::H
    rm::R
    m::M
    tracer::T
end
struct _MidpointEmission{H, R, T}
    hook::H
    rm::R
    tracer::T
end
@inline (d::_MidpointDiffusion)(dt) = _midpoint_diffusion!(d.hook, d.rm, d.m, d.tracer, dt)
@inline (e::_MidpointEmission)(dt) = _midpoint_emission!(e.hook, e.rm, e.tracer, dt)

@inline function apply_midpoint!(hook::TransportMidpoint, rm, m, tracer = nothing)
    _record_midpoint!(hook, m)
    diffuse! = _MidpointDiffusion(hook, rm, m, tracer)
    emit! = _MidpointEmission(hook, rm, tracer)
    _palindrome_center!(diffuse!, emit!, hook.diffusion, hook.emissions, hook.dt)
    return nothing
end

@inline _record_midpoint!(hook, m::AbstractArray) = nothing
@inline _record_midpoint!(hook, m::NTuple{6}) =
    record_center!(hook.recorder, m, hook.diffusion)

@inline _palindrome_center!(diffuse!::D, emit!::E, _op, ::NoSurfaceFlux, dt) where {D, E} =
    (diffuse!(dt); nothing)
@inline _palindrome_center!(diffuse!::D, emit!::E, op, ::AbstractSurfaceFluxOperator, dt) where {D, E} =
    surface_flux_center!(diffuse!, emit!, surface_flux_coupling(op), dt)

# LL packed midpoint: no timer/recorder wrapper in the original driver.
@inline function _midpoint_diffusion!(h, rm::AbstractArray{FT, 4}, m, ::Nothing, dt) where FT
    apply_vertical_diffusion_vmr!(rm, m, h.diffusion, h.diffusion_workspace, dt, h.meteo)
    return nothing
end
# RG single-tracer midpoint: preserve the per-tracer ledger and timer.
@inline function _midpoint_diffusion!(h, rm::AbstractMatrix, m, tracer, dt)
    SectionTimer.@section :diffusion apply_vertical_diffusion_vmr!(
        rm, m, h.diffusion, h.diffusion_workspace, dt, h.meteo; tracer)
    return nothing
end
# CS packed and Lin–Rood midpoint: record before every solve.
@inline function _midpoint_diffusion!(h, rm::NTuple{6}, m, tracer::Nothing, dt)
    record_diffusion!(h.recorder, h.diffusion, h.diffusion_workspace, dt)
    SectionTimer.@section :diffusion apply_vertical_diffusion_vmr!(
        rm, m, h.diffusion, h.diffusion_workspace, dt, h.meteo; halo_width = h.halo_width)
    return nothing
end
@inline function _midpoint_diffusion!(h, rm::NTuple{6}, m, tracer::Integer, dt)
    record_diffusion!(h.recorder, h.diffusion, h.diffusion_workspace, dt)
    SectionTimer.@section :diffusion apply_vertical_diffusion_vmr!(
        rm, m, h.diffusion, h.diffusion_workspace, dt, h.meteo; halo_width = h.halo_width, tracer)
    return nothing
end

@inline _midpoint_names(names, ::Nothing) = names
@inline _midpoint_names(names, tracer::Integer) = (names[tracer],)

@inline function _midpoint_emission!(h, rm::AbstractArray, tracer, dt)
    apply_surface_flux!(rm, h.emissions, h.workspace, dt, h.meteo, h.grid;
                        tracer_names = _midpoint_names(h.tracer_names, tracer))
    return nothing
end
@inline function _midpoint_emission!(h, rm::NTuple{6}, tracer, dt)
    record_emission!(h.recorder, dt)
    apply_surface_flux!(rm, h.emissions, h.workspace, dt, h.meteo, h.grid;
                        tracer_names = _midpoint_names(h.tracer_names, tracer),
                        halo_width = h.halo_width, deposit = h.deposit)
    return nothing
end

# Validation belongs to the composer and to Diffusion's public validator, and
# finishes before a sweep changes state. NoAdvection has the same checks.
_preflight_transport!(state::CellState, fluxes, op, ws, dt) =
    preflight_diffusion(op, ws, dt, state.air_mass, ntracers(state))
_preflight_transport!(state::CubedSphereState, fluxes, op, ws, dt) =
    preflight_diffusion(op, ws, dt, state.air_mass, ntracers(state), state.halo_width)

_check_midpoint_coupling(fluxes, op) = nothing
function _check_midpoint_coupling(::FaceIndexedFluxState, op)
    uses_diffusive_surface_flux_boundary(op) && throw(ArgumentError(
        "DiffusiveSurfaceFluxBoundary is not implemented for reduced-Gaussian runs; " *
        "use SplitSurfaceFluxCoupling."))
    return nothing
end

"""
    transport_step!(state, fluxes, grid, scheme, dt; workspace = nothing, ...)

Compose transport with its emission/diffusion midpoint. This is the shared
StepPlan entry point for production and adjoint replay. Direct advection
`apply!` only executes sweeps and the supplied abstract midpoint hook.
"""
function transport_step!(state, fluxes, grid, scheme::AbstractAdvectionScheme, dt;
                          workspace = nothing, diffusion_workspace = nothing,
                          diffusion_op = NoDiffusion(), emissions_op = NoSurfaceFlux(),
                          meteo = nothing, recorder::AbstractStepRecorder = NoStepRecorder(),
                          kwargs...)
    _preflight_transport!(state, fluxes, diffusion_op, diffusion_workspace, dt)
    _check_midpoint_coupling(fluxes, diffusion_op)
    hook = TransportMidpoint(; dt, workspace, diffusion_op, diffusion_workspace,
                              emissions_op, tracer_names = state.tracer_names, meteo, grid,
                              halo_width = _midpoint_halo_width(state), recorder)
    _apply_transport!(state, fluxes, grid, scheme, dt, workspace, hook, meteo, recorder; kwargs...)
    return nothing
end

@inline _midpoint_halo_width(::CellState) = nothing
@inline _midpoint_halo_width(state::CubedSphereState) = state.halo_width

@inline function _apply_transport!(state, fluxes, grid, scheme, dt, workspace, hook,
                                   meteo, ::NoStepRecorder; kwargs...)
    apply!(state, fluxes, grid, scheme, dt; workspace, midpoint = hook, meteo, kwargs...)
    return nothing
end
@inline function _apply_transport!(state, fluxes, grid, scheme, dt, workspace, hook,
                                   meteo, recorder::AbstractStepRecorder; kwargs...)
    apply!(state, fluxes, grid, scheme, dt; workspace, midpoint = hook, meteo, recorder, kwargs...)
    return nothing
end

# NoAdvection is a diffusion-only composition, with no palindrome or midpoint
# records. Advection has no `apply!` method for it: this is its only path, and
# it holds the emission rejection and the CS recorder guard.
function transport_step!(state, fluxes, grid, scheme::NoAdvection, dt;
                          workspace = nothing, diffusion_workspace = nothing,
                          diffusion_op = NoDiffusion(), emissions_op = NoSurfaceFlux(),
                          meteo = nothing, recorder::AbstractStepRecorder = NoStepRecorder(),
                          kwargs...)
    _check_noadvection_recorder(state, recorder)
    _noadvection_reject_emissions_op(emissions_op)
    _preflight_transport!(state, fluxes, diffusion_op, diffusion_workspace, dt)
    _diffusion_only!(state, diffusion_op, diffusion_workspace, dt, meteo)
    return nothing
end

_check_noadvection_recorder(::CellState, recorder) = nothing
function _check_noadvection_recorder(::CubedSphereState, recorder)
    recorder isa NoStepRecorder || throw(ArgumentError(
        "NoAdvection (scheme = \"none\") has no record in the cubed-sphere adjoint tape"))
    return nothing
end

@inline _noadvection_reject_emissions_op(::NoSurfaceFlux) = nothing
function _noadvection_reject_emissions_op(::AbstractSurfaceFluxOperator)
    throw(ArgumentError(
        "NoAdvection is incompatible with non-NoSurfaceFlux: surface " *
        "emissions are applied as Strang half-steps wrapping the " *
        "advection palindrome, so running them without advection " *
        "drops the symmetric splitting. Remove `[tracers.*.surface_flux]` " *
        "blocks for a convection-only or diffusion-only run."))
end

@inline _diffusion_only!(state::CellState, ::NoDiffusion, ws, dt, meteo) = nothing
@inline _diffusion_only!(state::CubedSphereState, ::NoDiffusion, ws, dt, meteo) = nothing
@inline function _diffusion_only!(state::CellState, op::AbstractDiffusion, ws, dt, meteo)
    apply_vertical_diffusion_vmr!(state.tracers_raw, state.air_mass, op, ws, dt, meteo)
    return nothing
end
@inline function _diffusion_only!(state::CubedSphereState, op::AbstractDiffusion, ws, dt, meteo)
    apply_vertical_diffusion_vmr!(state.tracers_raw, state.air_mass, op, ws, dt, meteo;
                                 halo_width = state.halo_width)
    return nothing
end

export TransportMidpoint, transport_step!
