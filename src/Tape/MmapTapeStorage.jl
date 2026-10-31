# ---------------------------------------------------------------------------
# On-disk cubed-sphere tape storage: the eviction tier of the adjoint tape.
#
# `MmapCSTapeStorage(dir)` is the storage descriptor of an `AdjointTape`: it
# names the scratch directory and nothing else. Each reverse-pass call (or
# 4D-Var optimization) opens one `_MmapTapeFile` in that directory
# (`_open_tape_pool`), sized to the largest tape it records, reuses it for
# every tape that fits, and releases it when it returns. The file
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
#     mapping, and `_release_tape_pool!` unmaps it (`munmap`) and closes the
#     file, so the space goes back to the file system when the call returns,
#     without waiting for a GC.
#
# `stage_panels!` copies the six panels into a slot's views; `_tape_panels(slot,
# role)` copies them into a read cache per read role and panel shape (on the
# device for GPU panels). The views never leave the file, so nothing can read
# them after the unmap. Nothing is persisted: a tape lives for one call.
# ---------------------------------------------------------------------------

"""
    MmapCSTapeStorage(dir)

Keep the staged states in a scratch file in `dir`, for tapes larger than
device and host memory. `dir` is required (there is no default: a tape can be
tens to hundreds of GB) and is created if it does not exist; a relative path
is resolved when the storage is constructed.

Each reverse-pass call (footprint, Jacobian, 4D-Var gradient) first sizes its
largest tape by running the forward replay with a counting recorder (one extra
forward run with `FullCheckpoint`; the checkpoint propagation pass with
`StrideCheckpoint`; a `cs_surface_flux_4dvar_optimize` call counts once for
all its evaluations). It then checks that `dir` has that many bytes plus 10 %
free, and reserves them with `fallocate` in one anonymous (unlinked) file
before recording; later tapes of the call reuse the file. Too little space is an `ArgumentError`
naming the required and available bytes, raised after the sizing pass and
before the tape is recorded. A file system that cannot reserve blocks for
in-place writes (no `fallocate`, e.g. some NFS mounts, or ZFS) is an
`ArgumentError` naming it; use a directory on a local file system such as ext4
or XFS. The file is unmapped, closed and its space returned to the file system
when the call returns; tapes are not persisted.

Reads copy the staged state into read caches keyed by read role
(`AirMassRead`, `TracerMassRead`) and panel shape (on the device for GPU
panels), so the air mass and tracer mass that one reverse kernel reads
together are distinct buffers.
"""
struct MmapCSTapeStorage <: AbstractHostCSTapeStorage
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

# The open scratch file of one reverse-pass call: its reserved extent
# (`capacity`) mapped once at `mapping` (`C_NULL` when empty or unmapped), the
# bytes handed out since the last rewind (`cursor`), and the generation and
# read caches of a host pool.
mutable struct _MmapTapeFile <: _AbstractHostTapePool
    dir::String
    io::IOStream
    mapping::Ptr{UInt8}
    capacity::Int64
    cursor::Int64
    generation::Int
    released::Bool
    read_template::Union{Nothing, AbstractArray}
    read_caches::Dict{CSTapeReadCacheKey, CSPanelCache}
end

"""
    _MmapTapeFile(dir, capacity) -> _MmapTapeFile

Create an anonymous scratch file of `capacity` bytes in `dir`: check the file
system and its free space, create and unlink the file, reserve the extent and
map it. Throws an `ArgumentError` with the required and available bytes when
the file system cannot hold it, and one naming the file system when it cannot
reserve blocks. `_release_tape_pool!` unmaps and closes it (without it, the
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
    storage = _MmapTapeFile(String(dir), io, mapping, Int64(capacity), Int64(0), 1, false,
                            nothing, Dict{CSTapeReadCacheKey, CSPanelCache}())
    # A file dropped without `_release_tape_pool!`: the GC unmaps it (and the
    # stream's own finalizer closes the file).
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
    # Read errno before anything else can allocate (and a GC overwrite it).
    err = ptr == _MAP_FAILED ? Libc.errno() : 0
    err == 0 || throw(ArgumentError(
        "MmapCSTapeStorage: could not map $(_format_tape_bytes(capacity)) of the tape in " *
        "$(repr(dir)): $(Libc.strerror(err))"))
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
# file system has no `fallocate`, slowly and without a guarantee. A signal
# that interrupts it (EINTR, e.g. a profiler's) is retried.
function _reserve_tape_extent!(io::IOStream, capacity::Int64, dir)
    capacity == 0 && return io
    @static if Sys.islinux()
        err = _fallocate(io, capacity)
        if err != 0
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

# `fallocate(fd, 0, 0, capacity)`, retried while a signal interrupts it; the
# errno of the failure, or 0.
function _fallocate(io::IOStream, capacity::Int64)
    while true
        rc = ccall(:fallocate, Cint, (Cint, Cint, Int64, Int64), fd(io), 0, 0, capacity)
        rc == 0 && return 0
        err = Libc.errno()
        err == Libc.EINTR || return err
    end
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

_open_tape_pool(storage::MmapCSTapeStorage, nbytes::Integer) = _MmapTapeFile(storage.dir, nbytes)

# A rewound file holds any tape up to its reservation; a larger one needs a
# new file (the driver releases this one first).
_holds_tape_bytes(file::_MmapTapeFile, nbytes::Integer) = !file.released && nbytes <= file.capacity

function _rewind_tape_pool!(file::_MmapTapeFile)
    _require_open(file)
    file.cursor = 0
    file.generation += 1
    return file
end

# Unmap and close now: with the mapping and the descriptor gone, the unlinked
# file's blocks return to the file system. Its slots are stale afterwards.
function _release_tape_pool!(file::_MmapTapeFile)
    file.released && return nothing
    file.released = true
    file.generation += 1
    _tape_synchronize(file.read_template)    # no copy still reads the mapping
    _unmap_tape_extent!(file)
    close(file.io)
    _free_read_caches!(file)
    return nothing
end

# The base offset of `nbytes` more bytes of the reserved extent.
function _bump_cursor!(file::_MmapTapeFile, nbytes::Int64)
    base = file.cursor
    base + nbytes <= file.capacity || error(
        "the mmap adjoint tape in $(repr(file.dir)) needs more than its reserved " *
        "$(_format_tape_bytes(file.capacity)): the tape byte count of the reverse-pass " *
        "driver disagrees with the recorder")
    file.cursor = base + nbytes
    return base
end

function _allocate_tape_slot(file::_MmapTapeFile, panels::NTuple{6})
    _require_open(file)
    _register_tape_read_backend!(file, panels)
    base = _bump_cursor!(file, _tape_slot_bytes(panels))
    return HostCSTapeSlot(file, _host_views(file.mapping + base, panels), file.generation)
end

function Base.show(io::IO, file::_MmapTapeFile)
    print(io, "_MmapTapeFile(dir=", repr(file.dir), ", bytes=", file.cursor, "/",
          file.capacity, file.released ? ", released" : "", ")")
end
