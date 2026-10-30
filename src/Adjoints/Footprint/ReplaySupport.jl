# The replay of an adjoint entry point: its plan with the operators of every
# step (the diffusion operator may change per step).
function require_adjoint_support(replay::_CSReplay)
    ops = [_replay_operators(replay, s, _sequence_at(replay.diffusion_op, s))
           for s in _distinct_diffusion_steps(replay)]
    require_adjoint_support(replay.plan, ops...)
    return replay
end

function _distinct_diffusion_steps(replay::_CSReplay)
    replay.diffusion_op isa AbstractVector || return 1:1
    seen = IdDict{Any, Int}()
    for s in 1:_replay_nsteps(replay)
        get!(seen, replay.diffusion_op[s], s)
    end
    return sort!(collect(values(seen)))
end
