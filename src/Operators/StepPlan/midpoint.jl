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
# it holds the emission check and the CS recorder guard. Surface emissions run
# only where the palindrome center has no half-steps to drop: on the cubed
# sphere with `DiffusiveSurfaceFluxBoundary`, whose center is S(dt) → V(dt).
# There the step runs that same center (`TransportMidpoint`, with the scheme's
# emission deposit) on the state's own buffers.
function transport_step!(state, fluxes, grid, scheme::NoAdvection, dt;
                          workspace = nothing, diffusion_workspace = nothing,
                          diffusion_op = NoDiffusion(), emissions_op = NoSurfaceFlux(),
                          meteo = nothing, recorder::AbstractStepRecorder = NoStepRecorder(),
                          kwargs...)
    _check_noadvection_recorder(state, recorder)
    _check_noadvection_emissions(state, diffusion_op, emissions_op)
    _preflight_transport!(state, fluxes, diffusion_op, diffusion_workspace, dt)
    _noadvection_step!(state, grid, dt, workspace, diffusion_op, diffusion_workspace,
                       emissions_op, meteo)
    return nothing
end

_check_noadvection_recorder(::CellState, recorder) = nothing
function _check_noadvection_recorder(::CubedSphereState, recorder)
    recorder isa NoStepRecorder || throw(ArgumentError(
        "NoAdvection (scheme = \"none\") has no record in the cubed-sphere adjoint tape"))
    return nothing
end

@inline _check_noadvection_emissions(state, diffusion_op, ::NoSurfaceFlux) = nothing
@inline _check_noadvection_emissions(state, diffusion_op, ::AbstractSurfaceFluxOperator) =
    _noadvection_emissions_supported(state, surface_flux_coupling(diffusion_op))

@inline _noadvection_emissions_supported(::CubedSphereState, ::DiffusiveSurfaceFluxBoundary) = nothing
function _noadvection_emissions_supported(state, coupling)
    throw(ArgumentError(
        "NoAdvection (scheme = \"none\") with surface emissions is supported only on " *
        "the cubed sphere with DiffusiveSurfaceFluxBoundary (`[diffusion] " *
        "surface_flux_boundary = true`, always on for kind = \"geoschem_nonlocal_vdiff\"), " *
        "whose palindrome center S(dt) → V(dt) has no half-steps. This run has " *
        "$(nameof(typeof(coupling))) on $(_noadvection_topology(state)), where the " *
        "emission belongs inside the advection palindrome. Remove the " *
        "`[tracers.*.surface_flux]` blocks, switch advection on, or use that " *
        "coupling on a cubed-sphere grid."))
end

_noadvection_topology(::CubedSphereState) = "the cubed sphere"
_noadvection_topology(::CellState) = "a lat-lon or reduced-Gaussian grid"

@inline _noadvection_step!(state, grid, dt, workspace, diffusion_op, ws,
                           ::NoSurfaceFlux, meteo) =
    _diffusion_only!(state, diffusion_op, ws, dt, meteo)
@inline _noadvection_step!(state, grid, dt, workspace, diffusion_op, ws,
                           emissions_op::AbstractSurfaceFluxOperator, meteo) =
    _noadvection_center!(state, grid, dt, workspace, diffusion_op, ws, emissions_op, meteo)

# Only reached after `_check_noadvection_emissions` (CS + DiffusiveSurfaceFluxBoundary).
@inline function _noadvection_center!(state::CubedSphereState, grid, dt, workspace,
                                      diffusion_op, ws, emissions_op, meteo)
    hook = TransportMidpoint(; dt, workspace, diffusion_op, diffusion_workspace = ws,
                              emissions_op, tracer_names = state.tracer_names, meteo, grid,
                              halo_width = state.halo_width)
    apply_midpoint!(hook, state.tracers_raw, state.air_mass)
    return nothing
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
