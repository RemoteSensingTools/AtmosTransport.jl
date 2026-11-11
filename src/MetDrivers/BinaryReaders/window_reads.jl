# How raw window reads split into concurrent `pread`s: chosen per binary from
# its file system, and on NFS per window from page-cache residency.

# ---------------------------------------------------------------------------
# On Unix a raw window read (`read_window_payload!`, the device staging path)
# splits the window into concurrent `pread`s of contiguous parts; other
# platforms read it with one `read!`. Measured with C90 L72 windows (156 MB):
# four reads copy a page-cache-warm window at 15-23 GB/s against 5.4-6.3 GB/s
# for one, and read cold local NVMe at 3.4 against 3.0 GB/s. On NFS, for
# client-cold windows of files nobody had read recently, four reads defeat
# sequential readahead: 0.40-0.49 GB/s forward and 0.36-0.41 GB/s in reverse
# window order, against 0.72-0.82 and 0.46 GB/s with one read; windows the
# client caches read 3-4 times faster with four. So on NFS each window read
# asks the page cache (`mincore` on the reader's mmap) how much of the window
# is resident: four reads when at least `NFS_RESIDENT_FRACTION_FOR_SPLIT` is,
# one otherwise (always one where the process cannot see the file's page
# cache). Local file systems always use four, with no query. Other
# network and parallel file systems (SMB, Lustre, GPFS, BeeGFS) are unmeasured
# and keep the local rule.
# ---------------------------------------------------------------------------

const LOCAL_WINDOW_READ_CHUNKS = 4
const NFS_WINDOW_READ_CHUNKS = 1

# Resident fraction of a window from which an NFS window read uses
# `LOCAL_WINDOW_READ_CHUNKS`. From the uncached and cached per-window times of
# a C90 window (one read: about 210 and 25 ms; four: 350 and 7 ms), four reads
# win once more than about 88 % of the window is cached. Windows with a cached
# prefix of 50-96 % read 1.6-2.7 times faster with four in one measurement, but
# their uncached tail followed a sequential read that the NFS server had
# likely prefetched; 0.9 stays on the side that both support.
const NFS_RESIDENT_FRACTION_FOR_SPLIT = 0.9

# `mincore` sample of a window's residency: runs of `RESIDENCY_RUN_PAGES`
# pages, `RESIDENCY_SAMPLE_PAGES` pages in all. Querying all 38,094 pages of a
# C90 window took 0.35 ms when none and 1.0 ms (up to 2.8 ms) when all were
# cached; the sample takes 29-43 µs and is within 1 percentage point of the full count for
# cold, half-cached and cached windows.
const RESIDENCY_RUN_PAGES = 16
const RESIDENCY_SAMPLE_PAGES = 1024

# `statfs` `f_type` of NFS on Linux (`NFS_SUPER_MAGIC`, linux/magic.h; v2-v4).
const LINUX_NFS_SUPER_MAGIC = 0x6969

"""
How a raw window read is split into concurrent `pread`s, one per reader
(`window_read_split`): `ConcurrentWindowReads`, `ResidencyWindowReads` or
`SequentialWindowReads`.
"""
abstract type AbstractWindowReadSplit end

"""
    ConcurrentWindowReads()

Every window as `LOCAL_WINDOW_READ_CHUNKS` concurrent `pread`s: binaries on
local (and unidentified) file systems.
"""
struct ConcurrentWindowReads <: AbstractWindowReadSplit end

"""
    SequentialWindowReads()

Every window as one `pread` (`NFS_WINDOW_READ_CHUNKS`): binaries on NFS whose
page-cache residency this process cannot see (`page_cache_visible`).
"""
struct SequentialWindowReads <: AbstractWindowReadSplit end

"""
    ResidencyWindowReads()

Binaries on NFS: a window whose pages are at least
`NFS_RESIDENT_FRACTION_FOR_SPLIT` resident in the page cache is read as
`LOCAL_WINDOW_READ_CHUNKS` concurrent `pread`s, any other window as one
(`NFS_WINDOW_READ_CHUNKS`), which keeps the client's sequential readahead.
`pages` is the `mincore` result buffer, allocated once per reader
(`RESIDENCY_SAMPLE_PAGES`); `lock` serializes its use.
"""
struct ResidencyWindowReads <: AbstractWindowReadSplit
    pages :: Vector{UInt8}
    lock  :: ReentrantLock
    ResidencyWindowReads() = new(Vector{UInt8}(undef, RESIDENCY_SAMPLE_PAGES), ReentrantLock())
end

# The granularity `Mmap` aligns the reader's mapping to.
_page_size() = Int(Mmap.PAGESIZE)

window_read_summary(::ConcurrentWindowReads) =
    string(LOCAL_WINDOW_READ_CHUNKS, " concurrent preads")
window_read_summary(::SequentialWindowReads) = "NFS: one pread"
window_read_summary(::ResidencyWindowReads) =
    string("NFS: ", LOCAL_WINDOW_READ_CHUNKS, " concurrent preads when ≥ ",
           round(Int, 100 * NFS_RESIDENT_FRACTION_FOR_SPLIT), " % of the window is cached, else one")

"""
    window_read_split(path) -> AbstractWindowReadSplit

The read split of the binary at `path`: on NFS `ResidencyWindowReads` when
this process can see the file's page cache (`page_cache_visible`), else
`SequentialWindowReads`; `ConcurrentWindowReads` on other file systems. Linux
identifies NFS by the `statfs` magic number, macOS by the `statfs` file-system
type name. On other platforms, or when the query fails, the result is
`ConcurrentWindowReads`; detection never throws.
"""
function window_read_split(path::AbstractString)
    on_nfs = try
        _on_nfs(path)
    catch err
        @debug "File-system query failed; raw window reads stay concurrent" path exception = err
        false
    end
    on_nfs || return ConcurrentWindowReads()
    return page_cache_visible(path) ? ResidencyWindowReads() : SequentialWindowReads()
end

"""
    page_cache_visible(path) -> Bool

Whether `mincore` on a mapping of the file at `path` reports its page cache to
this process. Linux (5.x; checked on 5.14) reports it only to the file's
owner, to a process with `CAP_FOWNER` or to one that may write the file, and
tells every other process that all pages are resident. The check here is the
conservative part of that rule, by effective credentials: the owner, or write
access (`faccessat` with `AT_EACCESS`); a process it misses reads NFS windows
with one `pread`. macOS is assumed to report it to every process (not
measured); other platforms have no `mincore` here. Never throws (`false` on
failure or for a missing file).
"""
function page_cache_visible(path::AbstractString)
    @static Sys.isapple() && return true
    @static Sys.islinux() || return false
    try
        st = stat(path)
        ispath(st) || return false
        AT_FDCWD, W_OK, AT_EACCESS = Cint(-100), Cint(2), Cint(0x200)
        return st.uid == ccall(:geteuid, Cuint, ()) ||
               ccall(:faccessat, Cint, (Cint, Cstring, Cint, Cint), AT_FDCWD, path, W_OK, AT_EACCESS) == 0
    catch err
        @debug "Page-cache visibility query failed" path exception = err
        return false
    end
end

"""
    window_read_chunks(split, data, first_byte, nbytes) -> Int

Number of concurrent `pread`s for the `nbytes` bytes of the mmapped `data`
starting at zero-based byte `first_byte`. `ConcurrentWindowReads` and
`SequentialWindowReads` answer without a query. `ResidencyWindowReads` queries
the page cache; if the query is unavailable or fails it reads with one `pread`.
"""
window_read_chunks(::ConcurrentWindowReads, data, first_byte, nbytes) = LOCAL_WINDOW_READ_CHUNKS
window_read_chunks(::SequentialWindowReads, data, first_byte, nbytes) = NFS_WINDOW_READ_CHUNKS

function window_read_chunks(split::ResidencyWindowReads, data::Vector, first_byte::Integer,
                            nbytes::Integer)
    fraction = @lock split.lock resident_fraction!(split.pages, data, first_byte, nbytes)
    return fraction !== nothing && fraction >= NFS_RESIDENT_FRACTION_FOR_SPLIT ?
        LOCAL_WINDOW_READ_CHUNKS : NFS_WINDOW_READ_CHUNKS
end

"""
    resident_fraction!(pages, data, first_byte, nbytes) -> Float64 or nothing

Fraction of the memory pages holding bytes `first_byte .+ (0:nbytes-1)`
(zero-based) of the mmapped vector `data` that are resident in memory, from
`mincore` (Linux, macOS) into `pages`. For a file mapping that is the file's
page cache. A range of up to `length(pages)` pages is queried whole; a longer
one is sampled as `length(pages) ÷ RESIDENCY_RUN_PAGES` evenly spaced runs of
`RESIDENCY_RUN_PAGES` pages, first and last page included. Returns `nothing`
on other platforms or when `mincore` fails.

Linux reports the page-cache residency of a file mapping only to some
processes and reports every page resident to the others; see
`page_cache_visible`, which `window_read_split` checks first.
"""
function resident_fraction!(pages::Vector{UInt8}, data::Vector, first_byte::Integer,
                            nbytes::Integer)
    nbytes >= 0 || throw(ArgumentError("nbytes must be non-negative; got $nbytes"))
    0 <= first_byte && first_byte + nbytes <= sizeof(data) ||
        throw(BoundsError(data, (first_byte, nbytes)))
    @static (Sys.islinux() || Sys.isapple()) || return nothing
    nbytes > 0 || return 1.0
    page = _page_size()
    address = UInt(pointer(data)) + UInt(first_byte)
    start = address & ~UInt(page - 1)
    npages = cld(Int(address - start) + Int(nbytes), page)
    run, runs = if npages <= length(pages)
        npages, 1
    else
        RESIDENCY_RUN_PAGES, length(pages) ÷ RESIDENCY_RUN_PAGES
    end
    GC.@preserve data pages for r in 0:(runs - 1)
        first_page = runs == 1 ? 0 : div(r * (npages - run), runs - 1)
        status = ccall(:mincore, Cint, (Ptr{Cvoid}, Csize_t, Ptr{UInt8}),
                       Ptr{Cvoid}(start + UInt(first_page * page)), Csize_t(run * page),
                       pointer(pages, r * run + 1))
        status == 0 || return nothing
    end
    resident = 0
    @inbounds for p in 1:(run * runs)
        resident += pages[p] & 0x01   # Linux and macOS: bit 0 = resident
    end
    return resident / (run * runs)
end

@static if Sys.islinux()
    # libuv `uv_fs_statfs`: `ftype` is `statfs.f_type`.
    _on_nfs(path) = Base.Filesystem.diskstat(path).ftype == LINUX_NFS_SUPER_MAGIC
elseif Sys.isapple()
    # macOS `struct statfs` with 64-bit inodes (the only layout on arm64, the
    # `$INODE64` symbol on x86_64): `char f_fstypename[16]` at byte 72. The
    # buffer exceeds the 2168-byte struct.
    function _on_nfs(path)
        buffer = zeros(UInt8, 4096)
        status = @static if Sys.ARCH === :x86_64
            ccall(:var"statfs$INODE64", Cint, (Cstring, Ptr{UInt8}), path, buffer)
        else
            ccall(:statfs, Cint, (Cstring, Ptr{UInt8}), path, buffer)
        end
        status == 0 || return false
        # NUL-terminated within its 16 bytes; the buffer is zero-filled.
        return GC.@preserve buffer unsafe_string(pointer(buffer, 73)) == "nfs"
    end
else
    _on_nfs(_path) = false
end
