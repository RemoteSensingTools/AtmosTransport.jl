# ---------------------------------------------------------------------------
# Pinned-host tape storage: `_PinnedHostArena`, the runtime pool of
# `PinnedHostCSTapeStorage()`.
#
# The arena allocates host memory in large chunks (`Libc.malloc`) and hands
# out 64-byte aligned slot views into them. Each chunk is registered with the
# GPU driver (page-locked) once, when it is allocated (`_pin_host_chunk`, CUDA
# and AMDGPU extensions), so recording a state is one device→host copy into
# memory that is already pinned, not an allocation and a registration per
# panel. A rewind (the next stride window, the next 4D-Var evaluation) reuses
# the chunks in order; `_release_tape_pool!` unregisters and frees them at
# once.
#
# Reservation allocates the complete arena before recording. Recording and
# rewind never grow it.
# ---------------------------------------------------------------------------

# One chunk: the `malloc`ed block, its first 64-byte boundary, the usable bytes
# from there and the backend registration (`nothing` for CPU panels).
struct _HostTapeChunk
    memory::Ptr{Cvoid}
    base::Ptr{UInt8}
    capacity::Int64
    registration::Any
end

mutable struct _PinnedHostArena <: _AbstractHostTapePool
    chunks::Vector{_HostTapeChunk}
    chunk::Int               # the chunk being filled since the last rewind (0: none yet)
    used::Int64              # bytes handed out in it
    generation::Int
    released::Bool
    read_template::Union{Nothing, AbstractArray}
    read_caches::Dict{CSTapeReadCacheKey, CSPanelCache}
end

"""Bytes of host memory the arena holds."""
_arena_bytes(arena::_PinnedHostArena) = sum(c -> c.capacity, arena.chunks; init = Int64(0))

# A chunk registered for the backend of `template`, the arena's zero-size
# array on the source backend (`read_template`): `_pin_host_chunk` dispatches
# on its array type.
function _new_host_chunk(capacity::Int64, template::AbstractArray)
    budget = first(_host_tape_memory())
    capacity <= budget || throw(TapeReservationError(Int(capacity), budget,
        "pinned host chunk exceeds host/cgroup headroom"))
    memory = Libc.malloc(capacity + _TAPE_ALIGN)
    memory == C_NULL && throw(OutOfMemoryError())
    base = Ptr{UInt8}(cld(UInt(memory), _TAPE_ALIGN) * _TAPE_ALIGN)
    registration = try
        _pin_host_chunk(template, base, capacity)
    catch
        Libc.free(memory)
        rethrow()
    end
    return _HostTapeChunk(memory, base, capacity, registration)
end

function _free_host_chunk!(chunk::_HostTapeChunk)
    _unpin_host_chunk(chunk.registration)
    Libc.free(chunk.memory)
    return nothing
end

# Take a whole slot from the reserved chunks in order.
function _arena_take!(arena::_PinnedHostArena, nbytes::Int64)
    arena.chunk == 0 && (arena.chunk = 1)
    if arena.chunk <= length(arena.chunks) && arena.used + nbytes > arena.chunks[arena.chunk].capacity
        arena.chunk += 1
        arena.used = 0
    end
    arena.chunk <= length(arena.chunks) ||
        throw(TapeReservationExceeded(_arena_bytes(arena), _arena_bytes(arena) + nbytes))
    chunk = arena.chunks[arena.chunk]
    nbytes <= chunk.capacity || throw(TapeReservationExceeded(chunk.capacity, nbytes))
    base = chunk.base + arena.used
    arena.used += nbytes
    return base
end

function _allocate_tape_slot(arena::_PinnedHostArena, panels::NTuple{6})
    _require_open(arena)
    _register_tape_read_backend!(arena, panels)
    base = _arena_take!(arena, _tape_slot_bytes(panels))
    return HostCSTapeSlot(arena, _host_views(base, panels), arena.generation)
end

function _rewind_tape_pool!(arena::_PinnedHostArena)
    _require_open(arena)
    arena.chunk = 0
    arena.used = 0
    arena.generation += 1
    return arena
end

# The backend first finishes every copy that may still read or write the
# chunks. Each chunk then leaves the arena before it is freed, so a failing
# unregistration (or an interrupt) cannot leave a freed chunk behind to be
# freed again; the other chunks are still freed, and the first error is
# rethrown.
function _release_tape_pool!(arena::_PinnedHostArena)
    arena.released && return nothing
    arena.released = true
    arena.generation += 1
    arena.chunk = 0
    arena.used = 0
    _tape_synchronize(arena.read_template)
    failure = nothing
    while !isempty(arena.chunks)
        chunk = pop!(arena.chunks)
        try
            _free_host_chunk!(chunk)
        catch err
            failure === nothing && (failure = err)
        end
    end
    _free_read_caches!(arena)
    failure === nothing || throw(failure)
    return nothing
end

function Base.show(io::IO, arena::_PinnedHostArena)
    print(io, "_PinnedHostArena(", length(arena.chunks), " chunks, ",
          Base.format_bytes(_arena_bytes(arena)), arena.released ? ", released" : "", ")")
end
