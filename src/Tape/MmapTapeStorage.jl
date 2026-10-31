# ---------------------------------------------------------------------------
# On-disk cubed-sphere tape storage: the eviction tier of the adjoint tape.
#
# `MmapCSTapeStorage(dir)` is the value a caller passes as `tape_storage`: it
# names the scratch directory and nothing else. Each reverse-pass call opens one
# `_MmapTapeFile` in that directory (`_open_tape_storage`), sized to the
# largest tape the call records, and releases it when the call returns. The file
#
#   * checks the free space of `dir` (statvfs) before anything is written and
#     throws with the required and available byte counts;
#   * is unlinked as soon as it is created, so no name is left behind, also
#     after a crash;
#   * reserves its whole extent at once (`fallocate` on Linux, zero fill
#     elsewhere), so a full disk fails at the reservation, before the forward
#     run is recorded, not as a SIGBUS from a sparse mmap page; a file system
#     that cannot reserve blocks for in-place writes (no `fallocate`, ZFS) is
#     an error naming it;
#   * maps the extent once (POSIX `mmap`); slots are views into that one
#     mapping, and `_release_tape_storage!` unmaps it (`munmap`) and closes the
#     file, so the space goes back to the file system when the call returns,
#     without waiting for a GC.
#
# `stage_panels!` copies the six panels into a slot's views; `_tape_panels(slot,
# role)` returns the views (CPU panels) or copies them into a device read cache
# per read role and panel shape (GPU panels, see the CUDA and AMDGPU
# extensions). Nothing is persisted: a tape lives for one call.
# ---------------------------------------------------------------------------

"""
    MmapCSTapeStorage(dir)

Tape storage policy that keeps staged adjoint mass states in a scratch file in
`dir`, for tapes larger than device and host memory. `dir` is required (there
is no default: a tape can be tens to hundreds of GB) and is created if it does
not exist; a relative path is resolved when the storage is constructed.

Each reverse-pass call (footprint, Jacobian, 4D-Var gradient) first sizes its
largest tape by running the forward replay with a counting recorder (one extra
forward run with `FullCheckpoint`; the checkpoint propagation pass with
`StrideCheckpoint` and `RevolveCheckpoint`). It then checks that `dir` has that
many bytes plus 10 % free, and reserves them with `fallocate` in one anonymous
(unlinked) file before recording. Too little space is an `ArgumentError`
naming the required and available bytes, raised after the sizing pass and
before the tape is recorded. A file system that cannot reserve blocks for
in-place writes (no `fallocate`, e.g. some NFS mounts, or ZFS) is an
`ArgumentError` naming it; use a directory on a local file system such as ext4
or XFS. The file is unmapped, closed and its space returned to the file system
when the call returns; tapes are not persisted.

On a GPU, reads come back through device-side read caches keyed by read role
(`AirMassRead`, `TracerMassRead`) and panel shape, so the air mass and tracer
mass that one reverse kernel reads together are distinct buffers.
"""
struct MmapCSTapeStorage <: AbstractCSTapeStorage
    dir::String
    function MmapCSTapeStorage(dir::AbstractString)
        isempty(dir) && throw(ArgumentError(
            "MmapCSTapeStorage needs a scratch directory; got an empty path"))
        return new(abspath(dir))
    end
end

# Free space required beside the tape itself, as a fraction of the tape: the
# scratch file system keeps headroom for other writers (logs, output).
const _MMAP_TAPE_HEADROOM = 0.1

# Seconds the free-space check waits for space that is being released (an
# earlier tape's blocks: XFS frees unlinked inodes lazily) before it fails.
const _MMAP_TAPE_FREE_WAIT = 5.0

# Every panel of a slot starts on a 64-byte boundary of the mapping, so its
# view is aligned for any element type, also when slots of different element
# types share one file.
const _MMAP_TAPE_ALIGN = 64
_align_tape_bytes(n::Integer) = cld(Int64(n), _MMAP_TAPE_ALIGN) * _MMAP_TAPE_ALIGN

"""
    _tape_slot_bytes(panels::NTuple{6}) -> Int64

Bytes one staged state of `panels` takes in an mmap tape (each panel padded
to the 64-byte alignment). The counting recorder sums these to size the
reservation.
"""
_tape_slot_bytes(panels::NTuple{6}) =
    sum(p -> _align_tape_bytes(sizeof(eltype(panels[p])) * length(panels[p])), 1:6)

# The open scratch file of one reverse-pass call: its reserved extent
# (`capacity`) mapped once at `mapping` (`C_NULL` when empty or unmapped), the
# bytes handed out so far (`cursor`), and the device read caches of
# `AbstractHostCSTapeStorage`.
mutable struct _MmapTapeFile <: AbstractHostCSTapeStorage
    dir::String
    io::IOStream
    mapping::Ptr{UInt8}
    capacity::Int64
    cursor::Int64
    # Zero-size array on the source backend, set by `_mmap_prepare_for_panels!`
    # for non-`Array` panels; `nothing` on the CPU, where reads return views.
    read_template::Union{Nothing, AbstractArray}
    read_caches::Dict{CSTapeReadCacheKey, CSPanelCache}
    synchronize::CSSynchronizeHook
    closed::Bool
end

"""
    _MmapTapeFile(dir, capacity) -> _MmapTapeFile

Create an anonymous scratch file of `capacity` bytes in `dir`: check the file
system and its free space, create and unlink the file, reserve the extent and
map it. Throws an `ArgumentError` with the required and available bytes when
the file system cannot hold it, and one naming the file system when it cannot
reserve blocks. `_release_tape_storage!` unmaps and closes it (without it, the
GC does).
"""
function _MmapTapeFile(dir::AbstractString, capacity::Integer)
    capacity >= 0 || throw(ArgumentError("tape capacity must be ≥ 0; got $(capacity)"))
    isdir(dir) || mkpath(dir)
    _check_tape_filesystem(dir, _filesystem_type(dir))
    _check_tape_free_space(dir, Int64(capacity))
    path, io = mktemp(dir; cleanup = false)
    local mapping
    try
        rm(path)
        _reserve_tape_extent!(io, Int64(capacity), dir)
        mapping = _map_tape_extent(io, Int64(capacity), dir)
    catch
        close(io)
        rethrow()
    end
    storage = _MmapTapeFile(String(dir), io, mapping, Int64(capacity), Int64(0), nothing,
                            Dict{CSTapeReadCacheKey, CSPanelCache}(), nothing, false)
    # A storage dropped without `_release_tape_storage!`: the GC unmaps it (and
    # the stream's own finalizer closes the file).
    finalizer(_unmap_tape_extent!, storage)
    return storage
end

# POSIX mmap of the whole extent, read-write and shared with the file.
const _PROT_READ_WRITE = Cint(0x1 | 0x2)
const _MAP_SHARED = Cint(0x01)
const _MAP_FAILED = reinterpret(Ptr{Cvoid}, typemax(UInt))

function _map_tape_extent(io::IOStream, capacity::Int64, dir)
    capacity == 0 && return Ptr{UInt8}(C_NULL)
    ptr = ccall(:mmap, Ptr{Cvoid}, (Ptr{Cvoid}, Csize_t, Cint, Cint, Cint, Int64),
                C_NULL, capacity, _PROT_READ_WRITE, _MAP_SHARED, fd(io), 0)
    ptr == _MAP_FAILED && throw(ArgumentError(
        "MmapCSTapeStorage: could not map $(_format_tape_bytes(capacity)) of the tape in " *
        "$(repr(dir)): $(Libc.strerror(Libc.errno()))"))
    return Ptr{UInt8}(ptr)
end

function _unmap_tape_extent!(storage::_MmapTapeFile)
    if storage.mapping != C_NULL
        ccall(:munmap, Cint, (Ptr{Cvoid}, Csize_t), storage.mapping, storage.capacity)
        storage.mapping = Ptr{UInt8}(C_NULL)
    end
    return nothing
end

_format_tape_bytes(n) = n < 1024 ? "$(n) bytes" : "$(n) bytes ($(Base.format_bytes(n)))"

# The type of the file system holding `path` (Linux: the longest mount point
# of /proc/self/mounts that contains it), or "unknown".
function _filesystem_type(path::AbstractString)
    Sys.islinux() && isfile("/proc/self/mounts") || return "unknown"
    target = realpath(path)
    best, fstype = "", "unknown"
    for line in eachline("/proc/self/mounts")
        fields = split(line)
        length(fields) >= 3 || continue
        mount = replace(fields[2], "\\040" => " ")
        inside = target == mount || startswith(target, rstrip(mount, '/') * "/")
        if inside && length(mount) >= length(best)
            best, fstype = mount, String(fields[3])
        end
    end
    return fstype
end

# File systems whose `fallocate` does not reserve blocks for in-place writes:
# a later write through the mapping can still fail with a SIGBUS.
const _MMAP_TAPE_UNRESERVING_FS = ("zfs",)

_check_tape_filesystem(dir, fstype::AbstractString) =
    fstype in _MMAP_TAPE_UNRESERVING_FS && throw(ArgumentError(
        "MmapCSTapeStorage: $(repr(dir)) is on $(fstype), whose fallocate does not " *
        "reserve blocks for in-place writes; choose a scratch directory on a local file " *
        "system such as ext4 or XFS"))

function _check_tape_free_space(dir, capacity::Int64)
    required = Base.checked_add(capacity, ceil(Int64, _MMAP_TAPE_HEADROOM * capacity))
    stat = diskstat(dir)
    available = Int64(stat.available)
    if available < required && required <= stat.total
        # The blocks of a tape released just before can come back late: XFS
        # frees unlinked inodes lazily, and stray mappings wait for the GC.
        GC.gc()
        deadline = time() + _MMAP_TAPE_FREE_WAIT
        while available < required && time() < deadline
            sleep(0.25)
            available = Int64(diskstat(dir).available)
        end
    end
    available >= required || throw(ArgumentError(
        "MmapCSTapeStorage: the adjoint tape needs $(_format_tape_bytes(capacity)) " *
        "plus $(round(Int, 100 * _MMAP_TAPE_HEADROOM)) % headroom, " *
        "$(_format_tape_bytes(required)) in total, in $(repr(dir)), but only " *
        "$(_format_tape_bytes(available)) are available. Free space there, choose " *
        "another scratch directory, or record a shorter tape (StrideCheckpoint)."))
    return nothing
end

# Allocate the file's blocks now, so writes through the mapping can never hit
# a full file system. Linux calls `fallocate` itself rather than
# `posix_fallocate`, whose glibc fallback writes one byte per block when the
# file system has no `fallocate`, slowly and without a guarantee.
function _reserve_tape_extent!(io::IOStream, capacity::Int64, dir)
    capacity == 0 && return io
    @static if Sys.islinux()
        rc = ccall(:fallocate, Cint, (Cint, Cint, Int64, Int64), fd(io), 0, 0, capacity)
        if rc != 0
            err = Libc.errno()
            fstype = _filesystem_type(dir)
            err == Libc.EOPNOTSUPP && throw(ArgumentError(
                "MmapCSTapeStorage: $(repr(dir)) is on $(fstype), which cannot reserve " *
                "space (fallocate: $(Libc.strerror(err))); choose a scratch directory on " *
                "a local file system such as ext4 or XFS"))
            throw(ArgumentError(
                "MmapCSTapeStorage: could not reserve $(_format_tape_bytes(capacity)) in " *
                "$(repr(dir)) ($(fstype), $(_format_tape_bytes(Int64(diskstat(dir).available))) " *
                "available): $(Libc.strerror(err))"))
        end
    else
        _zero_fill_tape_extent!(io, capacity, dir)
    end
    return io
end

# Portable reservation: write zeros over the extent (does not reserve on
# copy-on-write or compressing file systems).
function _zero_fill_tape_extent!(io::IOStream, capacity::Int64, dir)
    chunk = zeros(UInt8, min(capacity, Int64(64) << 20))
    try
        seekstart(io)
        remaining = capacity
        while remaining > 0
            n = min(remaining, Int64(length(chunk)))
            written = write(io, view(chunk, 1:n))
            written == n || throw(SystemError("short write ($(written) of $(n) bytes)"))
            remaining -= n
        end
        flush(io)
    catch err
        err isa SystemError || rethrow()
        throw(ArgumentError(
            "MmapCSTapeStorage: could not reserve $(_format_tape_bytes(capacity)) in " *
            "$(repr(dir)): $(sprint(showerror, err))"))
    end
    return io
end

_needs_tape_size(::MmapCSTapeStorage) = true

_open_tape_storage(storage::MmapCSTapeStorage, nbytes::Integer) =
    _MmapTapeFile(storage.dir, nbytes)

# Drivers that record several tapes in a row (one per stride window or
# bisection base step) reuse the extent.
_rewind_tape_storage!(storage::_MmapTapeFile) = (storage.cursor = 0; storage)

# Unmap and close now: with the mapping and the descriptor gone, the unlinked
# file's blocks return to the file system. Views handed out before are invalid
# afterwards (their reverse-pass call has returned).
function _release_tape_storage!(storage::_MmapTapeFile)
    storage.closed && return nothing
    storage.closed = true
    _unmap_tape_extent!(storage)
    close(storage.io)
    # Drops the references; device buffers are freed by the next GC.
    empty!(storage.read_caches)
    return nothing
end

"""
    MmapCSTapeSlot

One staged state in an `_MmapTapeFile`: the element type, the byte offset and
the shape of each of the six panels.
"""
struct MmapCSTapeSlot{S <: _MmapTapeFile}
    storage::S
    eltype::DataType
    offsets::NTuple{6, Int64}
    shapes::NTuple{6, NTuple{3, Int}}
end

_require_open(storage::_MmapTapeFile) = storage.closed && throw(ArgumentError(
    "the mmap tape in $(repr(storage.dir)) is closed: its reverse-pass call has returned"))

# The base offset of `nbytes` more bytes of the reserved extent.
function _bump_cursor!(storage::_MmapTapeFile, nbytes::Int64)
    _require_open(storage)
    base = storage.cursor
    base + nbytes <= storage.capacity || error(
        "the mmap adjoint tape in $(repr(storage.dir)) needs more than its reserved " *
        "$(_format_tape_bytes(storage.capacity)): the tape byte count of the reverse-pass " *
        "driver disagrees with the recorder")
    storage.cursor = base + nbytes
    return base
end

function _panel_offsets(base::Int64, nbytes_per::NTuple{6, Int64})
    o1 = base
    o2 = o1 + nbytes_per[1]
    o3 = o2 + nbytes_per[2]
    o4 = o3 + nbytes_per[3]
    o5 = o4 + nbytes_per[4]
    o6 = o5 + nbytes_per[5]
    return (o1, o2, o3, o4, o5, o6)
end

# A panel of a slot: an array over the storage's one mapping (no new mapping).
_tape_view(storage::_MmapTapeFile, ::Type{T}, shape::NTuple{3, Int}, offset::Int64) where {T} =
    unsafe_wrap(Array, Ptr{T}(storage.mapping + offset), shape)

"""
    _mmap_prepare_for_panels!(storage, panels)

Hook called once per slot allocation, before the span is handed out, that
lets the storage install per-backend state (the backend of the device-side
read caches, a backend `synchronize` callback).

The default method assumes non-`Array` device panels (CuArray, ROCArray, …)
and records their backend via `_register_tape_read_backend!`, so the read
caches are allocated there on first read. The method for
`NTuple{6, <:Array}` is a no-op, so the CPU read path returns views of the
mapping without an intermediate copy. The CUDA and AMDGPU extensions add their
`synchronize` hooks.
"""
function _mmap_prepare_for_panels!(storage::_MmapTapeFile, panels::NTuple{6})
    _register_tape_read_backend!(storage, panels)
    return storage
end

_mmap_prepare_for_panels!(storage::_MmapTapeFile, _panels::NTuple{6, <:Array}) = storage

function _allocate_tape_slot(storage::_MmapTapeFile, panels::NTuple{6})
    _require_open(storage)
    _mmap_prepare_for_panels!(storage, panels)
    T = eltype(panels[1])
    shapes = ntuple(p -> size(panels[p]), 6)
    nbytes_per = ntuple(p -> _align_tape_bytes(sizeof(T) * prod(shapes[p])), 6)
    base = _bump_cursor!(storage, sum(nbytes_per))
    return MmapCSTapeSlot(storage, T, _panel_offsets(base, nbytes_per), shapes)
end

# Works for CPU `Array` and GPU sources alike: `copyto!(::Array, ::CuArray)`
# is a device→host transfer.
function stage_panels!(slot::MmapCSTapeSlot, src::NTuple{6})
    storage = slot.storage
    _require_open(storage)
    @inbounds for p in 1:6
        copyto!(_tape_view(storage, slot.eltype, slot.shapes[p], slot.offsets[p]), src[p])
    end
    _after_tape_stage!(storage)
    return slot
end

# The reverse walk reads slots one at a time; each read cache is overwritten
# by the next read with the same role and shape, so this is not safe for
# concurrent reverse walks on one storage. GPU runs (`read_template` set)
# copy the views into the cache of `(role, slot.shapes)`: the role keeps the
# air mass and tracer mass of one limited-PPM sweep record in two buffers
# (issue #57). CPU runs return the views, a distinct array per read, valid
# until the storage is released.
function _tape_panels(slot::MmapCSTapeSlot, role::AbstractTapeReadRole)
    storage = slot.storage
    _require_open(storage)
    views = ntuple(p -> _tape_view(storage, slot.eltype, slot.shapes[p], slot.offsets[p]), 6)
    storage.read_template === nothing && return views
    cache = _tape_read_cache!(storage, role, views)
    @inbounds for p in 1:6
        copyto!(cache[p], views[p])
    end
    _after_tape_read!(storage)
    return cache
end

function Base.show(io::IO, storage::_MmapTapeFile)
    print(io, "_MmapTapeFile(dir=", repr(storage.dir), ", bytes=", storage.cursor, "/",
          storage.capacity, storage.closed ? ", closed" : "", ")")
end
