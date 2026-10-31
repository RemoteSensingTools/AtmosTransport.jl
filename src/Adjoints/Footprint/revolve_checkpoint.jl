# RevolveCheckpoint driver (recursive bisection) of the footprint reverse pass.
# Included by Adjoints.jl after StrideCheckpoint.jl, whose shared helpers it uses.

# ---------------------------------------------------------------------------
# RevolveCheckpoint driver.
#
# Recursive-bisection variant of Griewank-Walther Revolve. The reverse
# pass walks the step range via depth-first bisection:
#
#   reverse_range!(snapshot at lo, lo, hi):
#       if hi == lo:   return
#       if hi == lo+1: restore, record step hi, walk reverse, drop
#       else:
#           mid = (lo + hi) ÷ 2
#           restore lo, propagate lo+1:mid, snapshot mid
#           reverse_range!(snapshot at mid, mid, hi)
#           reverse_range!(snapshot at lo,  lo, mid)
#
# Snapshot memory is the recursion depth — `ceil(log2(nsteps))` replay
# snapshots (prognostic state with halos and persistent operator state).
# Every propagation and recording starts from a restored snapshot, so each
# step is replayed from the state the full forward run had at its start.
#
# Bisection only; this is not the optimal binomial schedule of
# Griewank-Walther Algorithm 799.
# ---------------------------------------------------------------------------

# The forward run keeps the initial state (and counts the step tapes for a
# storage that reserves its extent); the storage of the call holds one base
# step at a time: it is opened when the reverse pass starts and rewound before
# each base step.
struct _RevolveTapedRun{T, C, S}
    taped_steps  :: Int
    tape_storage :: T
    counter      :: C
    initial      :: S     # the replay state before step 1
end

_begin_taped_run(::RevolveCheckpoint, replay::_CSReplay, tape_storage, taped_steps::Int) =
    _RevolveTapedRun(taped_steps, tape_storage, _tape_size_counter(tape_storage),
                     _replay_snapshot(replay))

_taped_steps(run::_RevolveTapedRun) = run.taped_steps
_run_taped_steps!(run::_RevolveTapedRun, replay, on_step_end) =
    (_replay_steps!(replay, _taped_range(run); recorder = run.counter, on_step_end); nothing)
_end_taped_run!(::_RevolveTapedRun) = nothing

function _reverse_taped_run(run::_RevolveTapedRun, replay::_CSReplay, starts, dt; nsteps::Int, forcing)
    final = _replay_snapshot(replay)
    storage = _open_tape_storage(run.tape_storage,
                                 maximum(s -> _counted_slot_bytes(run.counter, s:s), 1:nsteps))
    try
        return map(starts) do start
            _replay_restore!(replay, final)
            lambda, footprints, ws = _start_reverse(replay, start, nsteps)

            function reverse_range!(snap_lo, lo, hi)
                if hi == lo
                    return nothing
                elseif hi - lo == 1
                    _rewind_tape_storage!(storage)
                    _replay_restore!(replay, snap_lo)
                    ops = _record_cs_tape!(replay, hi:hi, storage)
                    _walk!(footprints, lambda, ops, replay, ws, forcing)
                else
                    mid = (lo + hi) ÷ 2
                    _replay_restore!(replay, snap_lo)
                    _replay_steps!(replay, (lo + 1):mid)
                    snap_mid = _replay_snapshot(replay)
                    reverse_range!(snap_mid, mid, hi)
                    reverse_range!(snap_lo, lo, mid)
                end
                return nothing
            end

            reverse_range!(run.initial, 0, nsteps)
            _footprint_result(start, footprints, lambda, dt, replay)
        end
    finally
        _release_opened_tape_storage!(storage, run.tape_storage)
    end
end
