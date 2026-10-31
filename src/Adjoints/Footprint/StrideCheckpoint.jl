# ---------------------------------------------------------------------------
# Checkpoint drivers of the reverse pass: FullCheckpoint and StrideCheckpoint
# (RevolveCheckpoint is in revolve_checkpoint.jl).
#
# Every driver works on one replay (`_CSReplay`, StepReplay.jl) of the
# production step plan, whatever the advection scheme: it propagates the
# replay with the production call path, saves checkpoints of its state
# (prognostic state with halos and persistent operator state,
# `_replay_snapshot`), restores them, records windows of steps with a tape
# recorder and walks each window's tape backwards. A checkpoint is a step
# boundary of the absolute schedule, so a window that restarts inside a met
# window runs the remaining steps of that met window and its window-end
# stages exactly as the recording run did: whether a met window's convection
# has run is a function of the step index.
# ---------------------------------------------------------------------------

# The tape storage of a call is opened once, after the replay has counted the
# bytes of the largest tape the call records, when the storage reserves them up
# front (`MmapCSTapeStorage`): the recorder that counts them during a
# propagation run from step 1, else none.
_tape_size_counter(tape_storage) =
    _needs_tape_size(tape_storage) ? _CSCountingRecorder() : NoStepRecorder()

# The mmap bytes (`_tape_slot_bytes`) of the tape of `steps` (counted from step 1).
_counted_slot_bytes(::NoStepRecorder, _steps::AbstractUnitRange) = 0
function _counted_slot_bytes(counter::_CSCountingRecorder, steps::AbstractUnitRange)
    through = counter.slot_bytes_through_step
    return through[last(steps)] - (first(steps) == 1 ? 0 : through[first(steps) - 1])
end

# The reverse state of one adjoint start at the end of the replayed run: the
# starting adjoint state (from the run's final air mass), zero footprints for
# every step with emissions, and the adjoint workspace.
function _start_reverse(replay::_CSReplay, start)
    mesh = replay.mesh
    m = _replay_m(replay)
    lambda = _initial_adjoint_state(start, m, mesh)
    n = _replay_nsteps(replay)
    footprints = [_zero_surface_rates(mesh, m[1]) for _ in 1:(_has_surface_flux(replay) ? n : 0)]
    ws = CSAdjointWorkspace(mesh, lambda[1]; column_scratch = needs_column_scratch(replay.advection))
    return lambda, footprints, ws
end

function _walk!(footprints, lambda, ops, replay::_CSReplay, ws, forcing)
    _walk_window_reverse!(footprints, lambda, ops, replay.mesh, ws;
                          convection_workspace = replay.convection_workspace, forcing)
    return nothing
end

# The `CSFootprintResult` of a reverse pass started from `start`: per-step
# footprints and the adjoint state at the initial time.
function _footprint_result(start, footprints, lambda, dt, replay::_CSReplay)
    nsteps = length(footprints)
    A2 = typeof(_zero_surface_rates(replay.mesh, lambda[1])[1])
    FT = eltype(lambda[1])
    objective = _result_objective(start)
    lag_steps = [nsteps - step for step in 1:nsteps]
    return CSFootprintResult{FT, typeof(objective), A2, typeof(lambda[1])}(
        objective, footprints, lag_steps, FT(dt), lambda)
end

# FullCheckpoint: one tape for all starts. A storage that reserves its extent
# first counts the tape on a run of all steps, then the replay restarts from
# its initial snapshot, which reproduces the recording run bit for bit.
function _collect_footprints(::FullCheckpoint, replay::_CSReplay, starts, dt;
                             tape_storage, forcing)
    steps = 1:_replay_nsteps(replay)
    counter = _tape_size_counter(tape_storage)
    if counter isa _CSCountingRecorder
        initial = _replay_snapshot(replay)
        _replay_steps!(replay, steps; recorder = counter)
        _replay_restore!(replay, initial)
    end
    storage = _open_tape_storage(tape_storage, _counted_slot_bytes(counter, steps))
    try
        ops = _record_cs_tape!(replay, steps, storage)
        return map(starts) do start
            lambda, footprints, ws = _start_reverse(replay, start)
            _walk!(footprints, lambda, ops, replay, ws, forcing)
            _footprint_result(start, footprints, lambda, dt, replay)
        end
    finally
        _release_opened_tape_storage!(storage, tape_storage)
    end
end

# StrideCheckpoint(K): one propagation pass saves the replay state at every
# K-step boundary (and counts the window tapes for a storage that reserves its
# extent); the reverse pass re-records each window from its saved left edge
# into the one storage of the call, walks it backwards and rewinds the storage.
# Peak tape is one window; the propagation pass is shared by all starts.
function _collect_footprints(schedule::StrideCheckpoint, replay::_CSReplay, starts, dt;
                             tape_storage, forcing)
    n = _replay_nsteps(replay)
    windows = [checkpoint_window_range(schedule, w, n) for w in 1:checkpoint_window_count(schedule, n)]
    checkpoints = Vector{Any}(undef, length(windows))
    counter = _tape_size_counter(tape_storage)
    for (w, steps) in enumerate(windows)
        checkpoints[w] = _replay_snapshot(replay)
        _replay_steps!(replay, steps; recorder = counter)
    end
    final = _replay_snapshot(replay)
    storage = _open_tape_storage(tape_storage,
                                 maximum(steps -> _counted_slot_bytes(counter, steps), windows))
    try
        return map(starts) do start
            _replay_restore!(replay, final)
            lambda, footprints, ws = _start_reverse(replay, start)
            for w in reverse(eachindex(windows))
                _rewind_tape_storage!(storage)
                _replay_restore!(replay, checkpoints[w])
                ops = _record_cs_tape!(replay, windows[w], storage)
                _walk!(footprints, lambda, ops, replay, ws, forcing)
            end
            _footprint_result(start, footprints, lambda, dt, replay)
        end
    finally
        _release_opened_tape_storage!(storage, tape_storage)
    end
end

# The storage checks the drivers rely on, made before any forward run: the
# `tape_storage` argument is supported, and the Lin-Rood tape is device-only.
# The 4D-Var gradient calls this before its forward run, so a rejected
# combination does not waste that run.
function _require_tape_storage_supported(scheme, tape_storage)
    _validate_tape_storage(tape_storage)
    return _require_scheme_tape_storage(scheme, tape_storage)
end

_require_scheme_tape_storage(::CSAdjointLinRoodScheme, tape_storage) =
    _linrood_validate_tape_storage(tape_storage)
_require_scheme_tape_storage(_scheme, _tape_storage) = nothing
