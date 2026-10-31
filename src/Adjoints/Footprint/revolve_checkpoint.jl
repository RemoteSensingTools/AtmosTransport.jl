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

# The storage of the call holds one base step at a time: it is opened after
# the initial propagation (which counts the step tapes for a storage that
# reserves its extent) and rewound before each base step.
function _collect_footprints(::RevolveCheckpoint, replay::_CSReplay, starts, dt;
                             tape_storage, forcing)
    n = _replay_nsteps(replay)
    initial = _replay_snapshot(replay)
    counter = _tape_size_counter(tape_storage)
    _replay_steps!(replay, 1:n; recorder = counter)
    final = _replay_snapshot(replay)
    storage = _open_tape_storage(tape_storage,
                                 maximum(s -> _counted_slot_bytes(counter, s:s), 1:n))
    try
        return map(starts) do start
            _replay_restore!(replay, final)
            lambda, footprints, ws = _start_reverse(replay, start)

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

            reverse_range!(initial, 0, n)
            _footprint_result(start, footprints, lambda, dt, replay)
        end
    finally
        _release_opened_tape_storage!(storage, tape_storage)
    end
end
