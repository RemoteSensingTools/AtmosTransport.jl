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
# Chunks grow geometrically with the arena (from 64 MiB to 1 GiB): a tape of
# T bytes takes about log2 of the first gigabyte plus T / 1 GiB chunks, and
# wastes less than one slot per chunk at its end.
# ---------------------------------------------------------------------------

const _HOST_ARENA_MIN_CHUNK = Int64(64) << 20
const _HOST_ARENA_MAX_CHUNK = Int64(1) << 30
const _HOST_PAGE = Int64(4096)

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

function _open_tape_pool(::PinnedHostCSTapeStorage, _nbytes::Integer)
    arena = _PinnedHostArena(_HostTapeChunk[], 0, 0, 1, false, nothing,
                             Dict{CSTapeReadCacheKey, CSPanelCache}())
    # An arena dropped without `_release_tape_pool!`: the GC unregisters and
    # frees its chunks.
    finalizer(_release_tape_pool!, arena)
    return arena
end

"""Bytes of host memory the arena holds."""
_arena_bytes(arena::_PinnedHostArena) = sum(c -> c.capacity, arena.chunks; init = Int64(0))

# The size of the next chunk: at least one slot; otherwise the arena's size so
# far (doubling it), between 64 MiB and 1 GiB, in whole pages.
_next_chunk_bytes(arena::_PinnedHostArena, nbytes::Int64) =
    cld(max(nbytes, clamp(_arena_bytes(arena), _HOST_ARENA_MIN_CHUNK, _HOST_ARENA_MAX_CHUNK)),
        _HOST_PAGE) * _HOST_PAGE

# A chunk registered for the backend of `template`, the arena's zero-size
# array on the source backend (`read_template`): `_pin_host_chunk` dispatches
# on its array type.
function _new_host_chunk(capacity::Int64, template::AbstractArray)
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

# The address of `nbytes` more bytes: in the current chunk if they fit, else
# at the start of the next one (an existing chunk after a rewind, a new one
# otherwise; a chunk too small for one slot is replaced).
function _arena_take!(arena::_PinnedHostArena, nbytes::Int64)
    template = arena.read_template
    if arena.chunk > 0 && arena.used + nbytes <= arena.chunks[arena.chunk].capacity
        offset = arena.used
    else
        arena.chunk += 1
        if arena.chunk > length(arena.chunks)
            push!(arena.chunks, _new_host_chunk(_next_chunk_bytes(arena, nbytes), template))
        elseif arena.chunks[arena.chunk].capacity < nbytes
            # Allocate before freeing: if the allocation fails, the arena
            # still owns the old chunk (and releases it once).
            old = arena.chunks[arena.chunk]
            arena.chunks[arena.chunk] = _new_host_chunk(_next_chunk_bytes(arena, nbytes), template)
            _free_host_chunk!(old)
        end
        offset = Int64(0)
    end
    arena.used = offset + nbytes
    return arena.chunks[arena.chunk].base + offset
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
