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

# Locations are assigned at reservation, by element type and panel shape.
# Recording can consume the banks in any order without stranding chunk tails.
mutable struct _ReservedHostSlots
    locations::Vector{Tuple{Int, Int64}}
    cursor::Int
end

const _HOST_TAPE_CHUNK_BYTES = 1 << 30

mutable struct _PinnedHostArena <: _AbstractHostTapePool
    chunks::Vector{_HostTapeChunk}
    banks::Dict{Any, _ReservedHostSlots}
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

# Take the next reserved slot of this type and shape. Slots never straddle
# chunks; a slot larger than the chunk target has a dedicated chunk.
function _arena_take!(arena::_PinnedHostArena, key, nbytes::Int64)
    bank = get(arena.banks, key, nothing)
    bank === nothing &&
        throw(TapeReservationExceeded(_arena_bytes(arena), _arena_bytes(arena) + nbytes))
    bank.cursor < length(bank.locations) ||
        throw(TapeReservationExceeded(length(bank.locations) * nbytes, (bank.cursor + 1) * nbytes))
    bank.cursor += 1
    chunk, offset = bank.locations[bank.cursor]
    return arena.chunks[chunk].base + offset
end

function _allocate_tape_slot(arena::_PinnedHostArena, panels::NTuple{6})
    _require_open(arena)
    _register_tape_read_backend!(arena, panels)
    key = (eltype(first(panels)), map(size, panels))
    base = _arena_take!(arena, key, _tape_slot_bytes(panels))
    return HostCSTapeSlot(arena, _host_views(base, panels), arena.generation)
end

function _rewind_tape_pool!(arena::_PinnedHostArena)
    _require_open(arena)
    for bank in values(arena.banks)
        bank.cursor = 0
    end
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
    empty!(arena.banks)
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
