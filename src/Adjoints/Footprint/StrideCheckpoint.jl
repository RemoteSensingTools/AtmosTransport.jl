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
#
# A reverse-pass call runs one forward run of the replay and then its reverse
# pass, in four calls on one replay:
#
#   1. `_begin_taped_run(checkpoint, replay, tape_storage, taped_steps)`,
#      before the forward run: what the schedule needs before it (Full: the
#      tape storage, sized first by a counting run when it reserves its
#      extent);
#   2. `_run_forward!(run, replay; on_step_end)`: the forward run of every
#      step of the replay, which keeps what the schedule's reverse pass needs
#      for steps `1:taped_steps` (Full: the tape; Stride: the state at every
#      K-step boundary; Revolve: the initial state) and passes each step to
#      `on_step_end`;
#   3. `_reverse_taped_run(run, replay, starts, dt; nsteps, forcing)`, with
#      the replay where the forward run left it: the reverse pass of steps
#      `1:nsteps` (`nsteps <= taped_steps`) from each adjoint start;
#   4. `_end_taped_run!(run)`: releases the storage the call opened.
#
# The footprint entry points run them back to back over every step
# (`_collect_footprints`). The 4D-Var gradient samples its observations in
# the forward run (`on_step_end`), records through the last step of an
# observation with `assimilate = true`, and runs the reverse pass from a zero
# start through the last assimilated step after it has assembled the
# observation forcing (`_forced_reverse`): one forward run gives the simulated
# values and the tape.
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

# The reverse state of one adjoint start at the end of a run of `n` steps: the
# starting adjoint state (from the replay's air mass, the state after step
# `n`), zero footprints for every step with emissions, and the adjoint
# workspace.
function _start_reverse(replay::_CSReplay, start, n::Int)
    _check_reverse_length(start, replay, n)
    mesh = replay.mesh
    m = _replay_m(replay)
    lambda = _initial_adjoint_state(start, m, mesh)
    footprints = [_zero_surface_rates(mesh, m[1]) for _ in 1:(_has_surface_flux(replay) ? n : 0)]
    ws = CSAdjointWorkspace(mesh, lambda[1]; column_scratch = needs_column_scratch(replay.advection))
    return lambda, footprints, ws
end

# A final-time start (an objective or a seed: `dJ/drm` after the run) reverses
# every step of the forward run; only the zero start of the 4D-Var reverse may
# stop before its end (the forward continues past the last assimilated step).
_check_reverse_length(::_ZeroStart, ::_CSReplay, ::Int) = nothing
_check_reverse_length(_start, replay::_CSReplay, n::Int) =
    n == _replay_nsteps(replay) || throw(ArgumentError(
        "a final-time adjoint start reverses all $(_replay_nsteps(replay)) steps of its forward run, not $n"))

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

# ---------------------------------------------------------------------------
# The forward run of a call
# ---------------------------------------------------------------------------

# A forward run that keeps nothing for a reverse pass: the forward of a 4D-Var
# evaluation without a gradient (`CSCostOnly`) or without an assimilated
# observation.
struct _UntapedRun end

_taped_steps(::_UntapedRun) = 0
_run_taped_steps!(::_UntapedRun, _replay, _on_step_end) = nothing
_end_taped_run!(::_UntapedRun) = nothing

# Every step of the replay once: the taped steps as the schedule needs them,
# then the remaining steps (the 4D-Var co-sample tail) without a recorder.
function _run_forward!(run, replay::_CSReplay; on_step_end = nothing)
    _run_taped_steps!(run, replay, on_step_end)
    _replay_steps!(replay, (_taped_steps(run) + 1):_replay_nsteps(replay); on_step_end)
    return nothing
end

# The steps of the run with a tape: `1:taped_steps(run)`.
_taped_range(run) = 1:_taped_steps(run)

# ---------------------------------------------------------------------------
# FullCheckpoint: one tape for all starts, recorded by the forward run. A
# storage that reserves its extent is sized first by a counting run of the
# taped steps, from which the replay restarts at its initial snapshot; that
# reproduces the recording run bit for bit.
# ---------------------------------------------------------------------------

struct _FullTapedRun{T, R <: _CSTapeRecorder}
    taped_steps  :: Int
    tape_storage :: T       # the call's `tape_storage` argument
    recorder     :: R       # records into the storage the call opened
end

function _begin_taped_run(::FullCheckpoint, replay::_CSReplay, tape_storage, taped_steps::Int)
    steps = 1:taped_steps
    counter = _tape_size_counter(tape_storage)
    if counter isa _CSCountingRecorder
        initial = _replay_snapshot(replay)
        _replay_steps!(replay, steps; recorder = counter)
        _replay_restore!(replay, initial)
    end
    storage = _open_tape_storage(tape_storage, _counted_slot_bytes(counter, steps))
    return _FullTapedRun(taped_steps, tape_storage, _CSTapeRecorder(storage))
end

_taped_steps(run::_FullTapedRun) = run.taped_steps
_run_taped_steps!(run::_FullTapedRun, replay, on_step_end) =
    (_replay_steps!(replay, _taped_range(run); recorder = run.recorder, on_step_end); nothing)
_end_taped_run!(run::_FullTapedRun) =
    _release_opened_tape_storage!(run.recorder.storage, run.tape_storage)

# The records of the tape through the end of step `nsteps` (its end-of-step
# marker); the whole tape when `nsteps` is its last step.
function _tape_through(ops, nsteps::Int)
    last_record = findlast(op -> op isa _CSStepEndRecord && op.step == nsteps, ops)
    last_record === nothing && throw(ArgumentError("the tape has no record of step $nsteps"))
    return view(ops, 1:last_record)
end

function _reverse_taped_run(run::_FullTapedRun, replay::_CSReplay, starts, dt; nsteps::Int, forcing)
    ops = _tape_through(run.recorder.ops, nsteps)
    return map(starts) do start
        lambda, footprints, ws = _start_reverse(replay, start, nsteps)
        _walk!(footprints, lambda, ops, replay, ws, forcing)
        _footprint_result(start, footprints, lambda, dt, replay)
    end
end

# ---------------------------------------------------------------------------
# StrideCheckpoint(K): the forward run saves the replay state at every K-step
# boundary of the taped steps (and counts the window tapes for a storage that
# reserves its extent); the reverse pass re-records each window from its saved
# left edge into the one storage of the call, walks it backwards and rewinds
# the storage. Peak tape is one window; the forward run is shared by all
# starts. The window boundaries do not depend on the run length, so the saved
# states serve the reverse pass of any `nsteps <= taped_steps`.
# ---------------------------------------------------------------------------

struct _StrideTapedRun{T, C}
    schedule     :: StrideCheckpoint
    taped_steps  :: Int
    tape_storage :: T
    counter      :: C              # counts the window tapes, or `NoStepRecorder()`
    checkpoints  :: Vector{Any}    # the replay state at the left edge of each window
end

_begin_taped_run(schedule::StrideCheckpoint, ::_CSReplay, tape_storage, taped_steps::Int) =
    _StrideTapedRun(schedule, taped_steps, tape_storage, _tape_size_counter(tape_storage), Any[])

_taped_steps(run::_StrideTapedRun) = run.taped_steps
_end_taped_run!(::_StrideTapedRun) = nothing

_stride_windows(schedule::StrideCheckpoint, n::Int) =
    [checkpoint_window_range(schedule, w, n) for w in 1:checkpoint_window_count(schedule, n)]

function _run_taped_steps!(run::_StrideTapedRun, replay, on_step_end)
    for steps in _stride_windows(run.schedule, run.taped_steps)
        push!(run.checkpoints, _replay_snapshot(replay))
        _replay_steps!(replay, steps; recorder = run.counter, on_step_end)
    end
    return nothing
end

function _reverse_taped_run(run::_StrideTapedRun, replay::_CSReplay, starts, dt; nsteps::Int, forcing)
    windows = _stride_windows(run.schedule, nsteps)
    final = _replay_snapshot(replay)
    storage = _open_tape_storage(run.tape_storage,
                                 maximum(steps -> _counted_slot_bytes(run.counter, steps), windows))
    try
        return map(starts) do start
            _replay_restore!(replay, final)
            lambda, footprints, ws = _start_reverse(replay, start, nsteps)
            for w in reverse(eachindex(windows))
                _rewind_tape_storage!(storage)
                _replay_restore!(replay, run.checkpoints[w])
                ops = _record_cs_tape!(replay, windows[w], storage)
                _walk!(footprints, lambda, ops, replay, ws, forcing)
            end
            _footprint_result(start, footprints, lambda, dt, replay)
        end
    finally
        _release_opened_tape_storage!(storage, run.tape_storage)
    end
end

# ---------------------------------------------------------------------------
# The calls
# ---------------------------------------------------------------------------

# Every step of the replay forward and its reverse pass from each start (the
# footprint entry points).
function _collect_footprints(checkpoint::AbstractCheckpointSchedule, replay::_CSReplay, starts, dt;
                             tape_storage)
    n = _replay_nsteps(replay)
    run = _begin_taped_run(checkpoint, replay, tape_storage, n)
    try
        _run_forward!(run, replay)
        return _reverse_taped_run(run, replay, starts, dt; nsteps = n, forcing = NoStepForcing())
    finally
        _end_taped_run!(run)
    end
end

"""
    _forced_reverse(run, replay, dt; nsteps, forcing) -> CSFootprintResult

The reverse pass of steps `1:nsteps` of a taped forward run (`run`, after
`_run_forward!`, with the replay where that run left it) from a zero adjoint
state, adding `forcing` at the end of each step: the footprints of the 4D-Var
observation forcing. `nsteps` may end before the run's taped steps (the last
step of an assimilated observation after the surface-pressure screen) and
before the replay's last step (co-samples after it); a zero start does not
read the replay's final state.
"""
function _forced_reverse(run, replay::_CSReplay, dt; nsteps::Int, forcing::AbstractStepForcing)
    1 <= nsteps <= _taped_steps(run) || throw(ArgumentError(
        "the reverse pass of $nsteps steps needs a tape of them; the run taped $(_taped_steps(run))"))
    _validate_step_forcing(forcing, nsteps, size(_replay_m(replay)[1], 3))
    return only(_reverse_taped_run(run, replay, [_ZeroStart()], dt; nsteps, forcing))
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
