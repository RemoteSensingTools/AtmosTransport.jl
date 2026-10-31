# ---------------------------------------------------------------------------
# The runtime of an `AdjointTape`: what one reverse-pass call (or one
# `cs_surface_flux_4dvar_optimize` call, for all its evaluations) records into.
#
#   * The tape pool, opened from `tape.storage` before the first tape
#     (`_open_tape_pool`), rewound before each further tape (every stride
#     window, every later 4D-Var evaluation) so they all reuse its memory,
#     and released explicitly when the call returns (`_release_tape_runtime!`):
#     device arrays and pinned host chunks are freed now, not at a later GC,
#     and an mmap file is unmapped and closed.
#   * The counted tape bytes per replay step (`_CSCountingRecorder`), which an
#     mmap pool reserves. They depend on the meteorology and the step plan
#     only (the air-mass trajectory and its CFL subcycles do not depend on the
#     tracer or the emissions), so a runtime counts once: the later
#     evaluations of an optimization skip the counting run.
#
# The drivers (`StrideCheckpoint.jl`) take the runtime; the entry points build
# it with `_with_tape_runtime`.
# ---------------------------------------------------------------------------

mutable struct _TapeRuntime{T <: AdjointTape}
    const tape :: T
    pool :: Any                             # `nothing` until the first tape
    slot_bytes_through_step :: Vector{Int}  # counted bytes through each step (empty: not counted)
end

_TapeRuntime(tape::AdjointTape) = _TapeRuntime(tape, nothing, Int[])

"""
    _with_tape_runtime(f, tape::AdjointTape)

`f(runtime)` with a fresh runtime of `tape`, whose pool is released when `f`
returns or throws.
"""
function _with_tape_runtime(f, tape::AdjointTape)
    runtime = _TapeRuntime(tape)
    try
        return f(runtime)
    finally
        _release_tape_runtime!(runtime)
    end
end

function _release_tape_runtime!(runtime::_TapeRuntime)
    pool = runtime.pool
    runtime.pool = nothing
    pool === nothing || _release_tape_pool!(pool)
    return nothing
end

"""
    _tape_pool!(runtime, nbytes) -> pool

The pool the next tape of `runtime` records into, rewound: the open pool when
it holds `nbytes` (counted bytes; 0 unless the storage reserves them), else a
new one. Device and host pools grow on demand, and a runtime's counted bytes
are fixed by its first count, so in practice only an mmap file opened for a
smaller tape than a later one is replaced (released first).
"""
function _tape_pool!(runtime::_TapeRuntime, nbytes::Integer)
    pool = runtime.pool
    pool !== nothing && _holds_tape_bytes(pool, nbytes) && return _rewind_tape_pool!(pool)
    _release_tape_runtime!(runtime)
    return runtime.pool = _open_tape_pool(runtime.tape.storage, nbytes)
end

# The counted bytes of the tape of `steps` (counted from step 1).
function _counted_slot_bytes(through::Vector{Int}, steps::AbstractUnitRange)
    isempty(through) && return 0
    return through[last(steps)] - (first(steps) == 1 ? 0 : through[first(steps) - 1])
end

# The recorder that counts the tape bytes during a forward run from step 1,
# when the storage reserves them and the runtime has not counted them yet.
_tape_size_counter(runtime::_TapeRuntime, taped_steps::Int) =
    _needs_tape_size(runtime.tape.storage) && length(runtime.slot_bytes_through_step) < taped_steps ?
        _CSCountingRecorder(runtime.tape.content) : NoStepRecorder()

_store_tape_count!(::_TapeRuntime, ::NoStepRecorder) = nothing
_store_tape_count!(runtime::_TapeRuntime, counter::_CSCountingRecorder) =
    (runtime.slot_bytes_through_step = counter.slot_bytes_through_step; nothing)

# The bytes of the tape of `steps` for `runtime`'s storage: 0 when it does not
# reserve them; else the count, made now by a counting run of the replay from
# step 1 (restored to its initial state afterwards, which reproduces the
# recording run bit for bit) unless the runtime counted these steps before.
function _counted_tape_bytes!(runtime::_TapeRuntime, replay::_CSReplay, steps::AbstractUnitRange)
    _needs_tape_size(runtime.tape.storage) || return 0
    counter = _tape_size_counter(runtime, last(steps))
    if counter isa _CSCountingRecorder
        initial = _replay_snapshot(replay)
        _replay_steps!(replay, 1:last(steps); recorder = counter)
        _replay_restore!(replay, initial)
        _release_snapshot!(initial)
        _store_tape_count!(runtime, counter)
    end
    return _counted_slot_bytes(runtime.slot_bytes_through_step, steps)
end

# ---------------------------------------------------------------------------
# Explicit release of what a tape holds outside its pool
# ---------------------------------------------------------------------------

# A replay snapshot's tracer and air masses (operator state is small and left
# to the GC).
function _release_snapshot!(snap)
    foreach(_free_tape_array!, snap.rm)
    foreach(_free_tape_array!, snap.m)
    return nothing
end

# Records that own arrays outside the pool: the Lin–Rood horizontal records
# copy their states and fluxes (`TapeRecorder.jl`). Released once every start
# has walked the tape.
_release_records!(ops) = (foreach(_release_record!, ops); nothing)
_release_record!(_op) = nothing
function _release_record!(op::_CSLinRoodHorizRecord)
    foreach(_free_tape_panels!, (op.panels_rm, op.panels_m, op.panels_q_buf_phase2,
                                 op.panels_q_buf_phase3, op.panels_fx_in, op.panels_fx_out,
                                 op.panels_fy_in, op.panels_am, op.panels_bm))
    return nothing
end
