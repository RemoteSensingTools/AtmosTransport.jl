"""A column-local operation supplied by the step composer between advection halves."""
abstract type AbstractMidpointHook end

"""The default advection midpoint; performs no work on either active buffer."""
struct NoMidpoint <: AbstractMidpointHook end

"""
    apply_midpoint!(hook, active_rm, active_m[, tracer])

Apply a midpoint to the current ping-pong buffers. Per-tracer drivers pass the
tracer index; packed drivers omit it. Hooks must leave air mass unchanged and
must not read halos. StepPlans owns the emission/diffusion implementation.
"""
function apply_midpoint! end

@inline apply_midpoint!(::NoMidpoint, _rm, _m, _tracer = nothing) = nothing

# The fenced Lin–Rood driver calls a zero-argument midpoint. Bind its current
# tracer arrays in a concrete adapter, without a captured/reassigned binding.
struct BoundMidpoint{H, R, M, T}
    hook::H
    rm::R
    m::M
    tracer::T
end
@inline (hook::BoundMidpoint)() =
    apply_midpoint!(hook.hook, hook.rm, hook.m, hook.tracer)

export AbstractMidpointHook, NoMidpoint, apply_midpoint!
