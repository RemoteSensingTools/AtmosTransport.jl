# The replay of an adjoint entry point: its plan with the operators of every
# met window (the diffusion operator may change per window). Only the
# operators' types are checked.
function require_adjoint_support(replay::_CSReplay)
    source = replay.source
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
    require_adjoint_support(replay.plan, ops...)
    return replay
end
