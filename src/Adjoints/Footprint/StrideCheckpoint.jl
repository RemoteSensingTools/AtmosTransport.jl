# ---------------------------------------------------------------------------
# Checkpoint drivers of the reverse pass: FullCheckpoint and StrideCheckpoint.
#
# Every driver works on one replay (`_CSReplay`, StepReplay.jl) of the
# production step plan, whatever the advection scheme: it propagates the
# replay with the production call path, saves checkpoints of its state
# (prognostic state with halos and persistent operator state,
# `_replay_snapshot`), restores them, records windows of steps with a tape
# recorder into the call's tape pool (`TapeRuntime.jl`) and walks each
# window's tape backwards. A checkpoint is a step boundary of the absolute
# schedule, so a window that restarts inside a met window runs the remaining
# steps of that met window and its window-end stages exactly as the recording
# run did: whether a met window's convection has run is a function of the
# step index.
#
# A reverse-pass call runs one forward run of the replay and then its reverse
# pass, in four calls on one replay and one tape runtime:
#
#   1. `_begin_taped_run(checkpoint, replay, runtime, taped_steps)`, before
#      the forward run: what the schedule needs before it (Full: the tape
#      pool, sized first by a counting run when the storage reserves its
#      extent; Stride: the checkpoint of the initial state);
#   2. `_run_forward!(run, replay; on_step_end)`: the forward run of every
#      step of the replay, which keeps what the schedule's reverse pass needs
#      for steps `1:taped_steps` (Full: the tape; Stride: the state at every
#      K-step boundary) and passes each step to `on_step_end`;
#   3. `_reverse_taped_run(run, replay, starts, dt; nsteps, forcing)`, with
#      the replay where the forward run left it: the reverse pass of steps
#      `1:nsteps` (`nsteps <= taped_steps`) from each adjoint start;
#   4. `_end_taped_run!(run)`: releases what the run holds outside the pool
#      (checkpoints, Lin–Rood records); the pool stays with the runtime, which
#      its entry point releases (after one call, or after every evaluation of
#      an optimization).
#
# The footprint entry points run them back to back over every step
# (`_collect_footprints`). The 4D-Var gradient samples its observations in
# the forward run (`on_step_end`), records through the last step of an
# observation with `assimilate = true`, and runs the reverse pass from a zero
# start through the last assimilated step after it has assembled the
# observation forcing (`_forced_reverse`): one forward run gives the simulated
# values and the tape.
# ---------------------------------------------------------------------------

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
# FullCheckpoint: one tape for all starts, recorded by the forward run into
# the runtime's pool. A storage that reserves its extent is sized first by a
# counting run of the taped steps (once per runtime), from which the replay
# restarts at its initial snapshot; that reproduces the recording run bit for
# bit. With `AirMassTape` content the runtime holds the first forward run's
# tape and every later forward run of the runtime (a later evaluation of an
# optimization) runs without a recorder and walks the held tape, after its
# identity is checked (`_held_air_mass_tape`).
# ---------------------------------------------------------------------------

struct _FullTapedRun{R <: _CSTapeRecorder, H}
    taped_steps :: Int
    recorder    :: R       # records into the runtime's pool
    hold        :: H       # `AirMassTape`: the runtime and identity to hold the tape with; else nothing
end

# A forward run that reuses the air-mass tape its runtime holds.
struct _ReplayedTapedRun
    taped_steps :: Int
    ops         :: Vector{Any}
end

function _begin_taped_run(::FullCheckpoint, replay::_CSReplay, runtime::_TapeRuntime, taped_steps::Int)
    return _begin_full_run(runtime.tape.content, replay, runtime, taped_steps)
end

function _begin_full_run(content::StateTape, replay::_CSReplay, runtime::_TapeRuntime, taped_steps::Int)
    pool = _tape_pool!(runtime, _counted_tape_bytes!(runtime, replay, 1:taped_steps))
    return _FullTapedRun(taped_steps, _CSTapeRecorder(content, pool, replay.advection), nothing)
end

function _begin_full_run(content::AirMassTape, replay::_CSReplay, runtime::_TapeRuntime, taped_steps::Int)
    identity = _tape_identity(replay, taped_steps)
    held = _held_air_mass_tape(runtime, identity)
    held === nothing || return _ReplayedTapedRun(taped_steps, held)
    pool = _tape_pool!(runtime, _counted_tape_bytes!(runtime, replay, 1:taped_steps))
    return _FullTapedRun(taped_steps, _CSTapeRecorder(content, pool, replay.advection),
                         (; runtime, identity))
end

_taped_steps(run::Union{_FullTapedRun, _ReplayedTapedRun}) = run.taped_steps

function _run_taped_steps!(run::_FullTapedRun, replay, on_step_end)
    try
        _replay_steps!(replay, _taped_range(run); recorder = run.recorder, on_step_end)
        _finish_recording!(run.recorder)
        _hold_tape!(run.hold, run.recorder.ops)
    finally
        _free_recorder_scratch!(run.recorder)
    end
    return nothing
end
_run_taped_steps!(run::_ReplayedTapedRun, replay, on_step_end) =
    (_replay_steps!(replay, _taped_range(run); on_step_end); nothing)

_hold_tape!(::Nothing, _ops) = nothing
_hold_tape!(hold::NamedTuple, ops) = _hold_air_mass_tape!(hold.runtime, ops, hold.identity)

# A held air-mass tape lives until its runtime is released.
_end_taped_run!(run::_FullTapedRun{<:_CSTapeRecorder, Nothing}) = _release_records!(run.recorder.ops)
_end_taped_run!(::_FullTapedRun) = nothing
_end_taped_run!(::_ReplayedTapedRun) = nothing

_tape_ops(run::_FullTapedRun) = run.recorder.ops
_tape_ops(run::_ReplayedTapedRun) = run.ops

# The records of the tape through the end of step `nsteps` (its end-of-step
# marker); the whole tape when `nsteps` is its last step.
function _tape_through(ops, nsteps::Int)
    last_record = findlast(op -> op isa _CSStepEndRecord && op.step == nsteps, ops)
    last_record === nothing && throw(ArgumentError("the tape has no record of step $nsteps"))
    return view(ops, 1:last_record)
end

function _reverse_taped_run(run::Union{_FullTapedRun, _ReplayedTapedRun}, replay::_CSReplay, starts, dt;
                            nsteps::Int, forcing)
    ops = _tape_through(_tape_ops(run), nsteps)
    return map(starts) do start
        lambda, footprints, ws = _start_reverse(replay, start, nsteps)
        _walk!(footprints, lambda, ops, replay, ws, forcing)
        _footprint_result(start, footprints, lambda, dt, replay)
    end
end

# ---------------------------------------------------------------------------
# StrideCheckpoint(K): the forward run saves the replay state at every K-step
# boundary of the taped steps (and counts the window tapes for a storage that
# reserves its extent, once per runtime); the reverse pass re-records each
# window from its saved left edge into the runtime's pool, rewound for each
# window, and walks it backwards for every start. Peak tape is one window;
# the forward run and each window's recording are shared by all starts. The
# window boundaries do not depend on the run length, so the saved states serve
# the reverse pass of any `nsteps <= taped_steps`.
# ---------------------------------------------------------------------------

struct _StrideTapedRun{R <: _TapeRuntime, S <: StrideCheckpoint}
    schedule::S
    taped_steps::Int
    runtime::R
    checkpoints::Vector{Any}
    tail::Any
end

function _begin_taped_run(schedule::StrideCheckpoint, replay::_CSReplay, runtime::_TapeRuntime, n::Int)
    checkpoints = runtime.snapshots
    isempty(checkpoints) || (checkpoints[1] = _save_snapshot!(checkpoints[1], replay, schedule.snapshots))
    runtime.tail_pool === nothing || _rewind_tape_pool!(runtime.tail_pool)
    tail = schedule.resident_tail == 0 ? nothing :
        _CSTapeRecorder(runtime.tape.content, runtime.tail_pool, replay.advection)
    return _StrideTapedRun(schedule, n, runtime, checkpoints, tail)
end

_taped_steps(run::_StrideTapedRun) = run.taped_steps
_end_taped_run!(run::_StrideTapedRun) =
    (run.tail === nothing || _release_records!(run.tail.ops); nothing)  # snapshots belong to the runtime

function _run_taped_steps!(run::_StrideTapedRun, replay, on_step_end)
    for (w, steps) in enumerate(_head_windows(run.schedule, run.taped_steps))
        w == 1 || (run.checkpoints[w] = _save_snapshot!(run.checkpoints[w], replay, run.schedule.snapshots))
        _replay_steps!(replay, steps; on_step_end)
    end
    if run.tail !== nothing
        try
            head = _head_steps(run.schedule, run.taped_steps)
            _replay_steps!(replay, (head + 1):run.taped_steps; recorder = run.tail, on_step_end)
            _finish_recording!(run.tail)
        finally
            _free_recorder_scratch!(run.tail)
        end
    end
    return nothing
end

function _reverse_taped_run(run::_StrideTapedRun, replay::_CSReplay, starts, dt; nsteps::Int, forcing)
    runtime = run.runtime
    pool = _tape_pool!(runtime, 0)
    reverses = [_start_reverse(replay, start, nsteps) for start in starts]
    head = _head_steps(run.schedule, run.taped_steps)
    if nsteps > head
        ops = _tape_through(run.tail.ops, nsteps)
        for (lambda, footprints, ws) in reverses
            _walk!(footprints, lambda, ops, replay, ws, forcing)
        end
    end
    # Preserve the original head boundaries even if the surface-pressure
    # screen shortens the reverse pass into a head segment or into the tail.
    windows = _head_windows(run.schedule, run.taped_steps)
    for w in reverse(eachindex(windows))
        first(windows[w]) > nsteps && continue
        steps = first(windows[w]):min(last(windows[w]), nsteps)
        _rewind_tape_pool!(pool)
        _replay_restore!(replay, run.checkpoints[w])
        ops = _record_cs_tape!(replay, steps, runtime.tape.content, pool)
        try
            for (lambda, footprints, ws) in reverses
                _walk!(footprints, lambda, ops, replay, ws, forcing)
            end
        finally
            _release_records!(ops)
        end
    end
    return map((start, (lambda, footprints, _)) -> _footprint_result(start, footprints, lambda, dt, replay),
               starts, reverses)
end

# ---------------------------------------------------------------------------
# The calls
# ---------------------------------------------------------------------------

# Every step of the replay forward and its reverse pass from each start (the
# footprint entry points), with the tape of `runtime`.
function _collect_footprints(replay::_CSReplay, starts, dt, runtime::_TapeRuntime)
    n = _replay_nsteps(replay)
    run = _begin_taped_run(replay, runtime, n; nstarts = length(starts))
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

# The tape checks the drivers rely on, made before any forward run: the
# Lin–Rood tape is device-only (its records hold device copies, not pool
# slots; the check is the fenced Lin–Rood validator), and an `AirMassTape`
# needs a scheme whose reverse reads the air mass only (`tape_dependence`).
# The 4D-Var gradient calls this before its forward run, so a rejected
# combination does not waste that run.
function _require_tape_supported(scheme, tape::AdjointTape)
    _require_tape_storage(scheme, tape.storage)
    _require_tape_content(tape.content, scheme)
    return nothing
end

_require_tape_storage(::CSAdjointLinRoodScheme, storage) = _linrood_validate_tape_storage(storage)
_require_tape_storage(_scheme, _storage) = nothing

_require_tape_content(::StateTape, _scheme) = nothing
_require_tape_content(::AirMassTape, scheme) = _require_air_mass_dependence(tape_dependence(scheme), scheme)
_require_air_mass_dependence(::AirMassDependence, _scheme) = nothing
_require_air_mass_dependence(::TracerDependence, scheme) = throw(ArgumentError(
    "AirMassTape needs an advection scheme whose reverse reads the air mass only " *
    "(tape_dependence(scheme) isa AirMassDependence: UpwindScheme(), SlopesScheme(NoLimiter()), " *
    "PPMScheme(NoLimiter())); the reverse of $(nameof(typeof(scheme))) replays limiter branches " *
    "of the tracer (TracerDependence), so every forward run records its own tape: use StateTape()."))
