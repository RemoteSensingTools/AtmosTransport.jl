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

function _collect_footprints(::RevolveCheckpoint, replay::_CSReplay, starts, dt;
                             tape_storage, tape_path, forcing)
    n = _replay_nsteps(replay)
    initial = _replay_snapshot(replay)
    _replay_steps!(replay, 1:n)
    final = _replay_snapshot(replay)
    return map(starts) do start
        _replay_restore!(replay, final)
        lambda, footprints, ws = _start_reverse(replay, start)

        function reverse_range!(snap_lo, lo, hi)
            if hi == lo
                return nothing
            elseif hi - lo == 1
                storage_w = _build_window_storage(tape_storage, tape_path,
                                                  _revolve_step_subdir(hi))
                try
                    _replay_restore!(replay, snap_lo)
                    ops = _record_cs_tape!(replay, hi:hi, storage_w)
                    _walk!(footprints, lambda, ops, replay, ws, forcing)
                finally
                    finalize_tape!(storage_w; quiet = true, strict = tape_path !== nothing)
                end
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
end
