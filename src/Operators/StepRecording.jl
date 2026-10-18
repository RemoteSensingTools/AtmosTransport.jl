# ---------------------------------------------------------------------------
# Step recorders: the observation hooks of one model step.
#
# The production step (`StepPlans.execute_stages!`, the cubed-sphere
# palindromes in `Advection`) calls a hook at every point whose input the
# discrete adjoint needs: before each advection sweep, after each halo fill,
# at the palindrome center, around convection and air-mass resets. Production
# passes `NoStepRecorder()`, for which every hook is an inlined no-op, so the
# production kernels and their order are unchanged. The cubed-sphere adjoint
# (`Adjoints`) passes a tape recorder that stages the hook arguments and pushes
# one tape record per hook, so the step it later transposes is the step
# production ran, not a second composition of it.
#
# Hooks only read their arguments; a recorder never changes the model state.
# A recorder type must implement every hook a step it observes can reach: a
# missing method is a `MethodError`, never a silently skipped record.
# ---------------------------------------------------------------------------

"""
    AbstractStepRecorder

Observer of one model step. Production passes [`NoStepRecorder`](@ref); the
cubed-sphere adjoint passes a recorder that writes its tape from the same
calls. Hooks (`record_halo!`, `record_sweep!`, `record_center!`,
`record_diffusion!`, `record_emission!`, `record_linrood_horizontal!`,
`record_convection!`, `record_air_mass_reset!`, `record_chemistry!`) read their
arguments and never modify model state.
"""
abstract type AbstractStepRecorder end

"""
    NoStepRecorder()

The production recorder: every hook is an inlined no-op.
"""
struct NoStepRecorder <: AbstractStepRecorder end

"""
    record_halo!(recorder, dir)

A cross-panel halo fill of the tracer (and air) mass in direction `dir`
(`0`: edges, `1`/`2`: edges and the X/Y-oriented corners) has just run.
"""
function record_halo! end

"""
    record_sweep!(recorder, direction, scheme, panels_m, panels_rm, panels_flux, flux_scale)

An advection sweep (`direction ∈ (:x, :y, :z)`) of `scheme` with face fluxes
`flux_scale * panels_flux` is about to run on air mass `panels_m` and tracer
mass `panels_rm` (single-tracer 3D panels or packed 4D panels).
"""
function record_sweep! end

"""
    record_center!(recorder, panels_m, diffusion_op)

A palindrome center (diffusion and surface fluxes) is about to run on air mass
`panels_m`, which the center does not change.
"""
function record_center! end

"""
    record_diffusion!(recorder, diffusion_op, diffusion_workspace, dt)

A vertical-diffusion solve over `dt` is about to run at the palindrome center.
"""
function record_diffusion! end

"""
    record_emission!(recorder, dt)

The surface-flux operator is about to add `dt` worth of emissions at the
palindrome center.
"""
function record_emission! end

"""
    record_linrood_horizontal!(recorder, phase::Val, args...)

One phase of the Lin-Rood horizontal operator `fv_tp_2d_cs!`: `Val(:start)`
(rm, m after the halo and corner fills), `Val(:phase2)` and `Val(:phase3)`
(the predictor buffer after phases 1 and 2), `Val(:faces)` (the inner face
values before the seam projection) and `Val(:finish)` (the outer X faces after
it, the interior face fluxes and `Val(ORD)`).
"""
function record_linrood_horizontal! end

"""
    record_convection!(recorder, op, forcing, panels_m, dt, workspace)

Convection `op` with `forcing` has just advanced the tracer over `dt` on air
mass `panels_m` (unchanged by convection).
"""
function record_convection! end

"""
    record_air_mass_reset!(recorder, mode, state, target_air_mass)

The prognostic air mass of `state` is about to be replaced by
`target_air_mass` under reset `mode` (`:preserve_tracer_mass` or
`:preserve_vmr`), followed by a halo refresh of air and tracer mass.
"""
function record_air_mass_reset! end

"""
    record_chemistry!(recorder, op, dt)

Chemistry `op` is about to run over `dt`.
"""
function record_chemistry! end

@inline record_halo!(::NoStepRecorder, _dir) = nothing
@inline record_sweep!(::NoStepRecorder, _direction, _scheme, _m, _rm, _flux, _scale) = nothing
@inline record_center!(::NoStepRecorder, _m, _diffusion_op) = nothing
@inline record_diffusion!(::NoStepRecorder, _op, _workspace, _dt) = nothing
@inline record_emission!(::NoStepRecorder, _dt) = nothing
@inline record_linrood_horizontal!(::NoStepRecorder, _phase::Val, _args...) = nothing
@inline record_convection!(::NoStepRecorder, _op, _forcing, _m, _dt, _workspace) = nothing
@inline record_air_mass_reset!(::NoStepRecorder, _mode, _state, _target) = nothing
@inline record_chemistry!(::NoStepRecorder, _op, _dt) = nothing
