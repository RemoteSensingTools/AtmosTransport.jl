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

# Per-window / per-base-case subdirectory under a user-supplied `tape_path`.
# Five zero-padded digits keep alphabetical sort agreeing with numeric order
# through nsteps = 99999. The leading tag distinguishes Stride (`window_*`)
# from Revolve (`step_*`) subdirs so one `tape_path` tree can host both.
_stride_window_subdir(w::Integer) = "window_" * lpad(w, 5, '0')
_revolve_step_subdir(step::Integer) = "step_" * lpad(step, 5, '0')

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

# FullCheckpoint: one tape for all starts. The storage is constructed here so
# it is finalized deterministically after the last reverse walk. With
# `tape_path !== nothing` this is load-bearing — the user's directory keeps
# `records.bin` + `manifest.toml` after the call returns rather than relying
# on GC to flush the manifest. A pre-constructed `AbstractCSTapeStorage` is
# used as is (`_build_window_storage` returns it unchanged) and not finalized
# here, so the caller's manual lifecycle is preserved. When the caller
# supplied `tape_path`, manifest/close failures propagate (a saved tape
# without a valid manifest defeats the purpose of `tape_path`); temp-dir
# tapes keep warn-and-continue, as `cleanup_on_finalize` deletes them anyway.
function _collect_footprints(::FullCheckpoint, replay::_CSReplay, starts, dt;
                             tape_storage, tape_path, forcing)
    storage = _build_window_storage(tape_storage, tape_path)
    owns_storage = !(tape_storage isa AbstractCSTapeStorage)
    strict_finalize = tape_path !== nothing
    try
        ops = _record_cs_tape!(replay, 1:_replay_nsteps(replay), storage)
        return map(starts) do start
            lambda, footprints, ws = _start_reverse(replay, start)
            _walk!(footprints, lambda, ops, replay, ws, forcing)
            _footprint_result(start, footprints, lambda, dt, replay)
        end
    finally
        owns_storage && finalize_tape!(storage; quiet = true, strict = strict_finalize)
    end
end

# StrideCheckpoint(K): one propagation pass saves the replay state at every
# K-step boundary; the reverse pass re-records each window from its saved
# left edge into a fresh window tape, walks it backwards and drops it. Peak
# tape is one window; the propagation pass is shared by all starts.
function _collect_footprints(schedule::StrideCheckpoint, replay::_CSReplay, starts, dt;
                             tape_storage, tape_path, forcing)
    n = _replay_nsteps(replay)
    nw = checkpoint_window_count(schedule, n)
    checkpoints = Vector{Any}(undef, nw)
    for w in 1:nw
        checkpoints[w] = _replay_snapshot(replay)
        _replay_steps!(replay, checkpoint_window_range(schedule, w, n))
    end
    final = _replay_snapshot(replay)
    return map(starts) do start
        _replay_restore!(replay, final)
        lambda, footprints, ws = _start_reverse(replay, start)
        for w in nw:-1:1
            # Per-window storage, finalized after its walk: for `:mmap` each
            # window opens its own `records.bin` (under
            # `joinpath(tape_path, "window_NNNNN")` with `tape_path`).
            storage_w = _build_window_storage(tape_storage, tape_path, _stride_window_subdir(w))
            try
                _replay_restore!(replay, checkpoints[w])
                ops = _record_cs_tape!(replay, checkpoint_window_range(schedule, w, n), storage_w)
                _walk!(footprints, lambda, ops, replay, ws, forcing)
            finally
                finalize_tape!(storage_w; quiet = true, strict = tape_path !== nothing)
            end
        end
        _footprint_result(start, footprints, lambda, dt, replay)
    end
end

# The storage checks the drivers rely on, made up front: Stride and Revolve
# build and finalize one storage per window, so they reject a pre-constructed
# storage, and the Lin-Rood tape is device-only. The 4D-Var gradient calls
# this before its forward run, so a rejected combination does not waste that
# run.
_require_tape_storage_supported(scheme, ::FullCheckpoint, tape_storage) =
    _require_scheme_tape_storage(scheme, tape_storage)

function _require_tape_storage_supported(scheme, schedule::AbstractCheckpointSchedule,
                                         tape_storage)
    tape_storage isa AbstractCSTapeStorage && throw(ArgumentError(
        "$(nameof(typeof(schedule))) requires `tape_storage` to be a Symbol " *
        "(:device, :pinned_host, or :mmap), not a pre-constructed " *
        "$(typeof(tape_storage)); the driver builds and finalize_tape!s one " *
        "storage instance per window."))
    return _require_scheme_tape_storage(scheme, tape_storage)
end

_require_scheme_tape_storage(::CSAdjointLinRoodScheme, tape_storage) =
    _linrood_validate_tape_storage(tape_storage)
_require_scheme_tape_storage(_scheme, _tape_storage) = nothing

# Per-scheme `tape_path` compatibility. LinRood is pinned to `:device`
# storage, so a non-nothing `tape_path` is meaningless and would only produce
# empty `records.bin` / `manifest.toml` side effects under the user's tree.
# Reject here, BEFORE `_resolve_tape_path` runs `mkpath`, so an invalid
# request leaves the filesystem untouched.
function _require_tape_path_supported(scheme,
                                       tape_path::Union{Nothing, AbstractString})
    tape_path === nothing && return nothing
    scheme isa CSAdjointLinRoodScheme && throw(ArgumentError(
        "tape_path is not supported with LinRoodPPMScheme; the LinRood " *
        "reverse tape is :device-only. Omit tape_path or pick a non-" *
        "LinRood scheme. (Mmap eviction for LinRood is reserved for a " *
        "Plan 26 follow-up that refactors the LinRood tape.)"))
    return nothing
end
