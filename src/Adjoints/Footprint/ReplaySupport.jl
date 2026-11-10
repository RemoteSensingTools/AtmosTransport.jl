# The replay of an adjoint entry point: its plan with the operators of every
# met window (the diffusion operator may change per window). Only the
# operators' types are checked.
function require_adjoint_support(replay::_CSReplay)
    require_adjoint_support(replay.plan, _window_operators(replay, replay.source)...)
    return replay
end

# One `StepOperators` per distinct diffusion operator of the windows.
function _window_operators(replay::_CSReplay, source::AbstractCSWindowSource)
    seen = IdDict{Any, Nothing}()
    ops = StepOperators[]
    for w in 1:window_count(source)
        lease = acquire_window!(source, w)
        try
            met = lease.met
            haskey(seen, met.diffusion_op) && continue
            seen[met.diffusion_op] = nothing
            push!(ops, _replay_operators(replay, first(met.steps), met))
        finally
            release_window!(source, lease)
        end
    end
    return ops
end

# A streamed source's windows share one layout (operator types included), so
# its layout stands for every window: no window is acquired.
_window_operators(replay::_CSReplay, source::AbstractStreamedCSWindowSource) =
    (_replay_operators(replay, 1, window_layout(source)),)
