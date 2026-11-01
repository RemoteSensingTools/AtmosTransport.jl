# ---------------------------------------------------------------------------
# Tape storage of the CS adjoint: descriptors, runtime pools and slots.
#
# A "tape" is the sequence of per-step state snapshots the forward pass
# records so the reverse pass can re-read them. Where those snapshots live is
# chosen with an immutable storage descriptor (the `storage` of an
# `AdjointTape`), and recorded into a runtime pool that the reverse-pass driver
# opens from it, reuses and releases explicitly:
#
#   descriptor (user-facing)       runtime pool (internal, one per call or
#                                  per 4D-Var optimization)
#   `DeviceCSTapeStorage()`        `_DeviceSlabPool`: backend arrays, reused
#   `PinnedHostCSTapeStorage()`    `_PinnedHostArena` (HostTapeArena.jl): host
#                                  memory allocated and pinned once per chunk
#   `MmapCSTapeStorage(dir)`       `_MmapTapeFile` (MmapTapeStorage.jl): one
#                                  reserved, mapped scratch file
#
# A pool hands out slots (`_allocate_tape_slot`, `stage_panels!`,
# `_tape_panels(slot, role)`). `_rewind_tape_pool!` starts a new tape in the
# same memory (the next stride window, the next 4D-Var evaluation): the slots
# of the previous tape become stale, and reading one throws a
# `StaleTapeSlotError` instead of returning overwritten data.
# `_release_tape_pool!` frees the memory now, not at a later GC.
#
# Host pools (`_AbstractHostTapePool`) copy every read into a device read
# cache per read role (`AirMassRead`, `TracerMassRead`, `LimiterCodeRead`) and
# panel shape; their slot views never leave the pool.
# ---------------------------------------------------------------------------

"""
    AbstractCSTapeStorage

Where the staged states of a cubed-sphere adjoint tape live: the `storage` of
an [`AdjointTape`](@ref). The descriptors are immutable values
(`DeviceCSTapeStorage()`, `PinnedHostCSTapeStorage()`, `MmapCSTapeStorage(dir)`);
each reverse-pass call (or each `cs_surface_flux_4dvar_optimize` call) opens
its own runtime pool from the descriptor and releases it when it returns, so
one `AdjointTape` can serve any number of calls.

A new storage subtypes it and implements `_open_tape_pool(storage, nbytes)`,
returning a pool that implements `_allocate_tape_slot`, `stage_panels!`,
`_tape_panels(slot, role)`, `_rewind_tape_pool!` and `_release_tape_pool!`.
"""
abstract type AbstractCSTapeStorage end

"""
    AbstractHostCSTapeStorage <: AbstractCSTapeStorage

Storage whose staged states live off the source device: host memory
(`PinnedHostCSTapeStorage`) or a scratch file (`MmapCSTapeStorage`). Every read
copies the state into a device read cache per read role and panel shape, so
the two operands that one reverse kernel reads together (the air mass and
tracer mass of a limited-PPM sweep record) are always distinct buffers.
"""
abstract type AbstractHostCSTapeStorage <: AbstractCSTapeStorage end

"""
    DeviceCSTapeStorage()

Keep the staged states on the backend of the source panels (device memory on
a GPU). The fastest storage; the tape must fit beside the model on the device.
The pool of a call reuses its arrays for every tape it records (each stride
window, each 4D-Var evaluation of an optimization) and frees them when the
call returns.
"""
struct DeviceCSTapeStorage <: AbstractCSTapeStorage end

"""
    PinnedHostCSTapeStorage(; max_bytes = nothing)

Keep the staged states in host memory and read them back through device read
caches, one per read role and panel shape. The pool of a call allocates its
host memory in large chunks, each registered with the GPU driver (page-locked)
once when it is allocated (the CUDA and AMDGPU extensions), reuses them for
every tape it records and frees them when the call returns; staging and
reading are synchronous device↔host copies. With CPU panels it is plain host
memory. `max_bytes` caps the reservation; current host/cgroup headroom
always applies as well.
"""
struct PinnedHostCSTapeStorage{B <: Union{Nothing, Int}} <: AbstractHostCSTapeStorage
    max_bytes::B
    function PinnedHostCSTapeStorage(; max_bytes::Union{Nothing, Integer} = nothing)
        max_bytes === nothing || max_bytes >= 0 || throw(ArgumentError("max_bytes must be >= 0"))
        cap = max_bytes === nothing ? nothing : Int(max_bytes)
        new{typeof(cap)}(cap)
    end
end

"""
    AbstractTapeReadRole

The reverse-kernel operand a tape read feeds, passed as
`_tape_panels(slot, role)`. Host-side storage keeps one read cache per role,
and a read stays valid until the next read with the same role. A reverse
kernel that takes two staged states therefore reads them with two different
roles. Reading both through one shared cache gave the limited-PPM reverse
sweep the same buffer for the air mass and the tracer mass (issue #57).
"""
abstract type AbstractTapeReadRole end

"""
    AirMassRead()

Read role of staged air-mass states (sweep, diffusion and convection records).
"""
struct AirMassRead <: AbstractTapeReadRole end

"""
    TracerMassRead()

Read role of the staged tracer-mass state of an FV3-profile sweep record,
which the reverse sweep reads together with the record's air mass, and of the
target air mass of a `:preserve_vmr` reset record.
"""
struct TracerMassRead <: AbstractTapeReadRole end

"""
    LimiterCodeRead()

Read role of the limiter branch codes of a limited-PPM sweep record (six
`UInt8` panels), which the reverse sweep reads together with the record's air
mass.
"""
struct LimiterCodeRead <: AbstractTapeReadRole end

"""
    StaleTapeSlotError(msg)

A tape slot was read or written after its pool was rewound for a new tape
(the next stride window or 4D-Var evaluation) or released (its call
returned). The slot's memory belongs to another tape, or to nobody, so the
read is refused rather than returning other data.
"""
struct StaleTapeSlotError <: Exception
    msg::String
end
Base.showerror(io::IO, e::StaleTapeSlotError) = print(io, "StaleTapeSlotError: ", e.msg)

# ---------------------------------------------------------------------------
# Runtime pools: generation stamps and the lifecycle of a call
# ---------------------------------------------------------------------------

# The runtime pool a reverse-pass call records into. Every pool has the fields
# `generation::Int` (stamped on each slot; `_rewind_tape_pool!` and
# `_release_tape_pool!` advance it) and `released::Bool`.
abstract type _AbstractTapePool end

function _require_open(pool)
    pool.released && throw(StaleTapeSlotError(
        "the adjoint tape pool was released: its reverse-pass call has returned"))
    return nothing
end

function _require_current(pool, generation::Int)
    _require_open(pool)
    pool.generation == generation || throw(StaleTapeSlotError(
        "a tape slot of generation $(generation) was used after its pool was rewound for " *
        "another tape (generation $(pool.generation))"))
    return nothing
end

"""
    _open_tape_pool(storage, nbytes) -> pool

The runtime pool a reverse-pass call records its tapes into, opened from the
storage descriptor. This low-level constructor reserves an mmap extent of
`nbytes`. The reverse-pass resolver uses
`_reserve_tape_pool` to allocate the typed slabs or host chunks before running.
"""
function _open_tape_pool end

"""
    _holds_tape_bytes(pool, nbytes) -> Bool

Whether an open `pool` can record a tape of `nbytes` after a rewind; reserved
pools hold at most their reservation.
"""
_holds_tape_bytes(pool, _nbytes::Integer) = !pool.released

"""
    _rewind_tape_pool!(pool) -> pool

Start a new tape in the pool's memory: the next slots reuse it, and the slots
of the previous tape become stale (`StaleTapeSlotError` on use).
"""
function _rewind_tape_pool! end

"""
    _release_tape_pool!(pool)

Free the pool's memory now (device arrays, pinned host chunks, the mmap
file), not at a later GC; every slot becomes stale. Releasing twice is a
no-op.
"""
function _release_tape_pool! end

# ---------------------------------------------------------------------------
# Backend hooks (methods for GPU arrays in the CUDA and AMDGPU extensions)
# ---------------------------------------------------------------------------

"""
    _free_tape_array!(a)

Free the memory of a device array the tape owns now (`unsafe_free!` on a GPU).
A no-op for host arrays, which the GC frees.
"""
_free_tape_array!(::AbstractArray) = nothing

"""
    _tape_synchronize(template)

Wait for the backend of `template` (a zero-size array on the source backend)
after a device↔host tape copy. A no-op on the CPU.
"""
_tape_synchronize(::Union{Nothing, AbstractArray}) = nothing

"""
    _pin_host_chunk(template, ptr, nbytes) -> registration

Register (page-lock) `nbytes` of host memory at `ptr` with the backend of
`template`, once, when a pinned-host arena allocates a chunk. Returns a handle
for `_unpin_host_chunk`, or `nothing` (CPU panels: nothing to register).
"""
_pin_host_chunk(::AbstractArray, _ptr::Ptr, _nbytes::Integer) = nothing

"""
    _unpin_host_chunk(registration)

Undo `_pin_host_chunk` before the chunk is freed.
"""
_unpin_host_chunk(::Nothing) = nothing

_free_tape_panels!(panels) = (foreach(_free_tape_array!, panels); nothing)

# ---------------------------------------------------------------------------
# Device storage: `_DeviceSlabPool`
# ---------------------------------------------------------------------------

"""
    CSTapeSlot

One staged state of a device tape: six panel arrays on the source backend,
owned by the pool and stamped with the pool generation that handed them out.
"""
struct CSTapeSlot{P, A}
    pool::P
    panels::A
    generation::Int
end

# The arrays of one device tape. A rewind keeps them: the next tape takes the
# slabs in the same order and copies into them, so a stride window or a later
# 4D-Var evaluation records into the memory of the previous one rather than
# allocating its own (and waiting for a GC to free the old).
mutable struct _ReservedSlabs
    slabs::Vector{Any}
    cursor::Int
end

mutable struct _DeviceSlabPool <: _AbstractTapePool
    slabs::Vector{Any}       # NTuple{6} panel arrays, in staging order
    cursor::Int              # slabs handed out since the last rewind
    generation::Int
    released::Bool
    banks::Dict{Any, _ReservedSlabs}
end


function _rewind_tape_pool!(pool::_DeviceSlabPool)
    _require_open(pool)
    pool.cursor = 0
    foreach(bank -> (bank.cursor = 0), values(pool.banks))
    pool.generation += 1
    return pool
end

# Frees are ordered on the device after the kernels queued before them, so no
# host synchronization is needed. Each slab leaves the pool before it is freed
# (an interrupt cannot leave a freed slab to be freed again).
function _release_tape_pool!(pool::_DeviceSlabPool)
    pool.released && return nothing
    pool.released = true
    pool.generation += 1
    while !isempty(pool.slabs)
        _free_tape_panels!(pop!(pool.slabs))
    end
    empty!(pool.banks)
    return nothing
end

# Every slab is allocated by reservation and indexed by type and shape.
function _take_slab!(pool::_DeviceSlabPool, panels::NTuple{6, A},
                     ::Type{T} = eltype(first(panels))) where {A <: AbstractArray, T}
    bank = get(pool.banks, (T, map(size, panels)), nothing)
    if bank === nothing || bank.cursor == length(bank.slabs)
        capacity = sum(_bytes_per_panel_tuple, pool.slabs; init = 0)
        throw(TapeReservationExceeded(capacity, capacity + _bytes_per_panel_tuple(T, panels)))
    end
    pool.cursor += 1
    bank.cursor += 1
    return bank.slabs[bank.cursor]
end

function _allocate_tape_slot(pool::_DeviceSlabPool, panels::NTuple{6})
    _require_open(pool)
    return CSTapeSlot(pool, _take_slab!(pool, panels), pool.generation)
end

function stage_panels!(slot::CSTapeSlot{_DeviceSlabPool}, src::NTuple{6})
    _require_current(slot.pool, slot.generation)
    @inbounds for p in 1:6
        copyto!(slot.panels[p], src[p])
    end
    return slot
end

# `_tape_panels(slot, role::AbstractTapeReadRole) -> NTuple{6}`: the readable
# panels of a staged slot. Device slots return their own arrays; host slots
# (`HostCSTapeSlot`) copy into the read cache of `role` and return it, valid
# until the next read with the same role.
function _tape_panels(slot::CSTapeSlot{_DeviceSlabPool}, ::AbstractTapeReadRole)
    _require_current(slot.pool, slot.generation)
    return slot.panels
end

"""
    _stage_panels(pool, panels) -> slot

Allocate a slot in `pool` and copy `panels` into it.
"""
function _stage_panels(pool, panels::NTuple{6})
    slot = _allocate_tape_slot(pool, panels)
    return stage_panels!(slot, panels)
end

"""
    _stage_computed(fill_panels!, pool, template[, T]) -> slot

Allocate a slot shaped like `template` (six panels on the source backend) and
fill it with `fill_panels!(dest)`, which writes six panels of that shape (a
kernel queued on the backend). Device pools accept an element type `T`, so
the source panels can template a code slot without a scratch allocation.
A device slot is filled in place; any other pool's slot (host memory) is
filled through `template` as scratch, which is then staged.
"""
function _stage_computed(fill_panels!, pool::_DeviceSlabPool, template::NTuple{6},
                         ::Type{T} = eltype(first(template))) where {T}
    _require_open(pool)
    slot = CSTapeSlot(pool, _take_slab!(pool, template, T), pool.generation)
    fill_panels!(slot.panels)
    return slot
end

# ---------------------------------------------------------------------------
# Host pools: slots are views into host memory, read through device caches
# ---------------------------------------------------------------------------

const CSPanelCache = NTuple{6, <:AbstractArray}
const CSTapeReadCacheKey = Tuple{AbstractTapeReadRole, DataType, NTuple{6, NTuple{3, Int}}}

# A pool whose slots live in host memory it owns: the pinned-host arena and
# the mmap file. Besides `generation` and `released` it has the fields
# `read_template` (a zero-size array on the source backend, set when the first
# slot is allocated) and `read_caches` (`Dict{CSTapeReadCacheKey, CSPanelCache}`).
abstract type _AbstractHostTapePool <: _AbstractTapePool end

"""
    HostCSTapeSlot

One staged state of a host tape (pinned-host arena or mmap file): six views
into the pool's host memory, each on a 64-byte boundary, stamped with the
pool generation that handed them out. The views never leave the pool:
`_tape_panels` copies them into a device read cache.
"""
struct HostCSTapeSlot{P <: _AbstractHostTapePool, A}
    pool::P
    panels::NTuple{6, A}
    generation::Int
end

# Every panel of a slot starts on a 64-byte boundary of the pool's memory, so
# its view is aligned for any element type, also when slots of different
# element types share one pool.
const _TAPE_ALIGN = 64
_align_tape_bytes(n::Integer) = cld(Int64(n), _TAPE_ALIGN) * _TAPE_ALIGN

"""
    _tape_slot_bytes(panels::NTuple{6}) -> Int64

Bytes one staged state of `panels` takes in a host pool (each panel padded to
the 64-byte alignment). The counting recorder sums these to size an mmap
reservation.
"""
_tape_slot_bytes(panels::NTuple{6}) =
    sum(p -> _align_tape_bytes(sizeof(eltype(panels[p])) * length(panels[p])), 1:6)
# ... of six panels of element type `T` shaped like `like`.
_tape_slot_bytes(::Type{T}, like::NTuple{6}) where T =
    sum(p -> _align_tape_bytes(sizeof(T) * length(like[p])), 1:6)

# The six panel views of a slot whose memory starts at `base` (64-byte aligned).
function _host_views(base::Ptr{UInt8}, panels::NTuple{6})
    offsets = _panel_offsets(ntuple(p -> _align_tape_bytes(sizeof(eltype(panels[p])) *
                                                             length(panels[p])), 6))
    return ntuple(p -> unsafe_wrap(Array, Ptr{eltype(panels[p])}(base + offsets[p]),
                                   size(panels[p])), 6)
end

function _panel_offsets(nbytes::NTuple{6, Int64})
    o2 = nbytes[1]
    o3 = o2 + nbytes[2]
    o4 = o3 + nbytes[3]
    o5 = o4 + nbytes[4]
    o6 = o5 + nbytes[5]
    return (Int64(0), o2, o3, o4, o5, o6)
end

# Record the backend of the source panels when a slot is allocated: the read
# caches are `similar` to this zero-size template (a `CuArray` for GPU
# panels), and the device↔host copies synchronize on it.
function _register_tape_read_backend!(pool::_AbstractHostTapePool, panels::NTuple{6})
    pool.read_template === nothing && (pool.read_template = similar(panels[1], 0, 0, 0))
    return pool
end

function _cache_matches(cache, panels::NTuple{6})
    cache isa NTuple{6} || return false
    @inbounds for p in 1:6
        if eltype(cache[p]) !== eltype(panels[p]) || size(cache[p]) != size(panels[p])
            return false
        end
    end
    return true
end

# The read cache of `role` for the shapes of `panels`, allocated on first use
# on the backend of `read_template`. One cache per role keeps the operands of
# one reverse kernel apart; one per shape lets a pool hold slots of different
# shapes. Two reads of the same role and shape return the same buffer, so a
# record whose reverse kernel reads two staged states must give them
# different roles (a new role type if needed), or the kernel sees one operand
# twice (issue #57).
function _tape_read_cache!(pool::_AbstractHostTapePool, role::AbstractTapeReadRole,
                           panels::NTuple{6})
    key = (role, eltype(first(panels)), ntuple(p -> size(panels[p]), 6))
    cached = get(pool.read_caches, key, nothing)
    _cache_matches(cached, panels) && return cached
    cached === nothing || _free_tape_panels!(cached)
    template = pool.read_template
    fresh = ntuple(p -> similar(template, eltype(panels[p]), size(panels[p])), 6)
    pool.read_caches[key] = fresh
    return fresh
end

function _free_read_caches!(pool::_AbstractHostTapePool)
    foreach(_free_tape_panels!, values(pool.read_caches))
    empty!(pool.read_caches)
    return nothing
end

# Staging copies device → host and waits for the copy; reading copies host →
# device cache and waits, so the next read of the role may reuse the cache
# (synchronous copies; no prefetch).
function stage_panels!(slot::HostCSTapeSlot, src::NTuple{6})
    pool = slot.pool
    _require_current(pool, slot.generation)
    @inbounds for p in 1:6
        copyto!(slot.panels[p], src[p])
    end
    _tape_synchronize(pool.read_template)
    return slot
end

function _stage_computed(fill_panels!, pool, scratch::NTuple{6})
    fill_panels!(scratch)
    return _stage_panels(pool, scratch)
end

function _tape_panels(slot::HostCSTapeSlot, role::AbstractTapeReadRole)
    pool = slot.pool
    _require_current(pool, slot.generation)
    cache = _tape_read_cache!(pool, role, slot.panels)
    @inbounds for p in 1:6
        copyto!(cache[p], slot.panels[p])
    end
    _tape_synchronize(pool.read_template)
    return cache
end

# ---------------------------------------------------------------------------
# Byte estimation helper
# ---------------------------------------------------------------------------

_bytes_per_panel_tuple(panels::NTuple{6}) =
    sum(sizeof(eltype(panels[p])) * length(panels[p]) for p in 1:6)
_bytes_per_panel_tuple(::Type{T}, like::NTuple{6}) where T =
    sum(sizeof(T) * length(like[p]) for p in 1:6)
