# ---------------------------------------------------------------------------
# Memory-mapped, on-disk cubed-sphere tape storage policy.
#
# `MmapCSTapeStorage` is a sibling of `DeviceCSTapeStorage` and
# `PinnedHostCSTapeStorage` that evicts staged
# panel snapshots from working memory onto a raw appended binary file
# (`records.bin`). A separate human-readable manifest (`manifest.toml`)
# is written at finalisation time and used during resume / inspection.
#
# Layout summary (v1):
#
#   tape/<run-id>/
#   ├── manifest.toml      # written on finalize_tape!
#   └── records.bin        # grown via truncate as slots are allocated
#
# Each slot reserves a contiguous span of `records.bin` and holds six
# `(record_id, panel_offset, panel_shape)` triplets. `stage_panels!`
# copies the source panels into `Mmap.mmap` views over the reserved
# region; `_tape_panels` returns mmap views (CPU runs) or copies them
# into a shared device-side cache (GPU runs, see CUDA extension).
# ---------------------------------------------------------------------------

using Mmap: Mmap
using TOML: TOML
using Dates: now

const _MMAP_TAPE_FORMAT_VERSION = "v1"

# One forward record's locator into records.bin. Kept inside the storage
# so the manifest emitter has a single source of truth at finalise time.
struct MmapTapeRecordEntry
    record_id::Int
    eltype::DataType
    offsets::NTuple{6, Int64}
    nbytes::NTuple{6, Int64}
    shapes::NTuple{6, NTuple{3, Int}}
end

"""
    MmapCSTapeStorage(; dir = mktempdir(prefix="atmostransport-cstape-"),
                       cleanup_on_finalize = (dir == ""))

Tape storage policy that streams staged adjoint mass states onto a
single appended raw binary file (`records.bin`) inside `dir` and emits a
TOML manifest (`manifest.toml`) at finalisation. `dir` is created if it
does not exist; if omitted, a unique temporary directory is created and
deleted when the storage object is garbage-collected.

The policy is backend-agnostic: CPU source panels yield in-place
`Mmap.mmap` views at read time; GPU source panels go through a shared
per-shape device-side read cache. The cache is keyed by the panel
shape signature (`NTuple{6, NTuple{3, Int}}`) so a reverse loop that
alternates between m-shaped (Nx,Nx,Nz), am-shaped (Nx+1,Nx,Nz),
bm-shaped (Nx,Nx+1,Nz), and cm-shaped (Nx,Nx,Nz+1) slots reuses each
shape's cache instead of reallocating six `similar(panel)` device
buffers on every read.

The on-disk format is the `manifest.toml` metadata plus appended raw
panel payloads in `records.bin`.
"""
mutable struct MmapCSTapeStorage <: AbstractCSTapeStorage
    bin_path::String
    manifest_path::String
    dir::String
    bin_io::IOStream
    cursor::Int64
    records::Vector{MmapTapeRecordEntry}
    # Shape-keyed device-side read cache. Populated by
    # `_mmap_prepare_for_panels!` for non-`Array` panel sources;
    # stays empty for CPU runs so `_tape_panels` returns mmap views
    # directly (zero-copy through the page cache).
    device_caches::Dict{NTuple{6, NTuple{3, Int}}, CSPanelCache}
    synchronize::CSSynchronizeHook
    finalised::Bool
    closed::Bool
    cleanup_on_finalize::Bool
    # Read-only tapes block `_allocate_tape_slot` / `_bump_cursor!`.
    # `load_mmap_tape` opens an existing tape with `readonly = true`;
    # the forward-recording constructor leaves this false.
    readonly::Bool

    # Forward-recording constructor: opens a fresh records.bin in
    # read+write mode. `cleanup_on_finalize` defaults to true when
    # `dir` is empty (temp dir) and false when the user supplies one.
    function MmapCSTapeStorage(; dir::AbstractString = "",
                               cleanup_on_finalize::Union{Nothing,Bool} = nothing)
        if isempty(dir)
            dir = mktempdir(; prefix = "atmostransport-cstape-")
            cleanup = cleanup_on_finalize === nothing ? true : cleanup_on_finalize
        else
            isdir(dir) || mkpath(dir)
            cleanup = cleanup_on_finalize === nothing ? false : cleanup_on_finalize
        end
        bin_path = joinpath(dir, "records.bin")
        manifest_path = joinpath(dir, "manifest.toml")
        # A manifest describes one immutable, finalised records.bin. Remove it
        # before truncating that file so a crash during a new recording can
        # never make fresh bytes look like the previous completed tape.
        ispath(manifest_path) && rm(manifest_path; force = true)
        bin_io = open(bin_path, "w+")
        local storage
        try
            storage = new(bin_path, manifest_path, String(dir), bin_io, Int64(0),
                          MmapTapeRecordEntry[],
                          Dict{NTuple{6, NTuple{3, Int}}, CSPanelCache}(),
                          nothing,
                          false, false, cleanup, false)
            finalizer(_mmap_storage_gc_close!, storage)
        catch
            close(bin_io)
            rethrow()
        end
        return storage
    end

    # Read-only constructor used by `load_mmap_tape` (the actual
    # parser lives in a free function below so it can throw clean
    # errors before any IOStream is opened). All slot bookkeeping
    # is restored from a finalised manifest; `cursor` reflects the
    # total bytes written by the original forward pass.
    function MmapCSTapeStorage(dir::AbstractString, records::Vector{MmapTapeRecordEntry},
                               cursor::Int64; mode::AbstractString = "r")
        mode in ("r", "r+") || throw(ArgumentError(
            "MmapCSTapeStorage reopen mode $(repr(mode)) is not supported; " *
            "use \"r\" (default, readonly) or \"r+\" (forward-compat for Phase B)"))
        isdir(dir) || throw(ArgumentError("tape directory $(repr(dir)) does not exist"))
        bin_path = joinpath(dir, "records.bin")
        manifest_path = joinpath(dir, "manifest.toml")
        isfile(bin_path) || throw(ArgumentError(
            "tape directory $(repr(dir)) is missing records.bin"))
        bin_io = open(bin_path, mode)
        local storage
        try
            storage = new(bin_path, manifest_path, String(dir), bin_io, cursor,
                          records,
                          Dict{NTuple{6, NTuple{3, Int}}, CSPanelCache}(),
                          nothing,
                          true, false, false, mode == "r")
            finalizer(_mmap_storage_gc_close!, storage)
        catch
            close(bin_io)
            rethrow()
        end
        return storage
    end
end

_tape_storage(::Val{:mmap}) = MmapCSTapeStorage()

"""
    _build_window_storage(tape_storage, tape_path, subdir) -> AbstractCSTapeStorage

Construct a fresh tape storage instance for one window (Stride) or one
base-case step (Revolve). When `tape_path === nothing` this falls
through to `_tape_storage(tape_storage)` and inherits its temp-dir /
cleanup behaviour. When `tape_path !== nothing`, `tape_storage` must
be `:mmap`; the function builds an [`MmapCSTapeStorage`](@ref) rooted
at `joinpath(tape_path, subdir)` with `cleanup_on_finalize = false` so
the caller-owned directory is preserved past `finalize_tape!`.

An empty `subdir` (the default) is used by the single-tape
`FullCheckpoint` path: the storage is rooted directly at `tape_path`
with no extra nesting. The checkpoint drivers pass per-window /
per-step subdirectory names (e.g. `"window_00007"`, `"step_00042"`) so
multiple windows can coexist under the same user-supplied tree.
"""
function _build_window_storage(tape_storage::Symbol,
                                tape_path::Union{Nothing, AbstractString},
                                subdir::AbstractString = "")
    tape_path === nothing && return _tape_storage(tape_storage)
    tape_storage === :mmap || throw(ArgumentError(
        "tape_path requires tape_storage = :mmap; got " *
        "$(repr(tape_storage))"))
    dir = isempty(subdir) ? String(tape_path) : joinpath(tape_path, subdir)
    return MmapCSTapeStorage(; dir = dir, cleanup_on_finalize = false)
end

# Pre-constructed storage: identity pass-through. The user owns the
# storage's lifecycle, so the caller (`cs_surface_emission_footprint`)
# must skip its own `finalize_tape!` for this branch. Stride / Revolve
# already reject pre-constructed storage up front (they build a fresh
# one per window), so this method only fires through the
# `FullCheckpoint` path.
_build_window_storage(storage::AbstractCSTapeStorage,
                       ::Nothing,
                       ::AbstractString = "") = storage

# ---------------------------------------------------------------------------
# Slot type — minimal descriptor; all data lives on disk.
# ---------------------------------------------------------------------------

struct MmapCSTapeSlot{S <: MmapCSTapeStorage}
    storage::S
    record_id::Int
    eltype::DataType
    offsets::NTuple{6, Int64}
    shapes::NTuple{6, NTuple{3, Int}}
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_mmap_file_size(io::IOStream) = stat(io).size

"""
    _bump_cursor!(storage, nbytes)

Reserve `nbytes` of contiguous space at the end of `records.bin`, extending
the file via `truncate` (sparse on filesystems that support it) so that
subsequent `Mmap.mmap` calls find the region backed.
Returns the base offset of the reserved region.
"""
# NOTE: this assumes POSIX semantics — growing the file via
# `Base.truncate` while earlier mmap views are still alive is
# well-defined on Linux/macOS (existing pages stay valid; the kernel
# refcounts the file). Windows is not supported.
function _bump_cursor!(storage::MmapCSTapeStorage, nbytes::Int64)
    storage.readonly && throw(ArgumentError(
        "MmapCSTapeStorage at $(storage.dir) is readonly; cannot extend records.bin"))
    storage.closed && throw(ArgumentError(
        "MmapCSTapeStorage at $(storage.dir) is closed; cannot extend records.bin"))
    base = storage.cursor
    storage.cursor += nbytes
    if storage.cursor > _mmap_file_size(storage.bin_io)
        Base.truncate(storage.bin_io, storage.cursor)
    end
    return base
end

# Build per-panel offsets from the record base.
function _panel_offsets(base::Int64, nbytes_per::NTuple{6, Int64})
    o1 = base
    o2 = o1 + nbytes_per[1]
    o3 = o2 + nbytes_per[2]
    o4 = o3 + nbytes_per[3]
    o5 = o4 + nbytes_per[4]
    o6 = o5 + nbytes_per[5]
    return (o1, o2, o3, o4, o5, o6)
end

_mmap_view(io::IOStream, ::Type{T}, shape::NTuple{3, Int}, offset::Int64) where {T} =
    Mmap.mmap(io, Array{T, 3}, shape, offset; grow = false, shared = true)

# Register the slot in the manifest queue. Called from the type-specialised
# `_allocate_tape_slot` methods so the queue is consistent across backends.
function _push_record!(storage::MmapCSTapeStorage, ::Type{T},
                       offsets, nbytes_per, shapes) where {T}
    record_id = length(storage.records) + 1
    push!(storage.records,
          MmapTapeRecordEntry(record_id, T, offsets, nbytes_per, shapes))
    return record_id
end

# ---------------------------------------------------------------------------
# Slot allocation + staging — CPU path.
# ---------------------------------------------------------------------------

"""
    _mmap_prepare_for_panels!(storage, panels)

Hook called once per slot allocation, **before** the disk reservation,
that lets the storage install per-policy state (e.g. a shared
device-side read cache, a backend `synchronize` callback).

The **default** method assumes the source panels are non-`Array`
device-resident buffers (CuArray, MtlArray, …) and pre-allocates a
backend-matched read cache via `_ensure_tape_read_cache!`. The
specialised method for `NTuple{6, <:Array}` is a no-op so the CPU read
path can return mmap views directly without an intermediate copy.

The CUDA extension layers a `synchronize = CUDA.synchronize` hook on top
of the default to amortise asynchronous device transfers; correctness
of mmap → device traffic does **not** depend on that extension because
`copyto!(::CuArray, ::Array)` is synchronous on CUDA.jl.
"""
function _mmap_prepare_for_panels!(storage::MmapCSTapeStorage,
                                   panels::NTuple{6})
    _ensure_tape_read_cache!(storage, panels)
    return storage
end

_mmap_prepare_for_panels!(storage::MmapCSTapeStorage,
                          _panels::NTuple{6, <:Array}) = storage

"""
    _allocate_tape_slot(storage::MmapCSTapeStorage, panels)

Reserve a span of `records.bin` for one tape record (six panel arrays)
and return a slot that can be staged into and later read from. The disk
bookkeeping is backend-agnostic; backend-specific setup (read-cache
allocation, synchronisation hooks) happens in
`_mmap_prepare_for_panels!`, which the CUDA extension specialises for
`NTuple{6, <:CuArray}`.
"""
function _allocate_tape_slot(storage::MmapCSTapeStorage,
                             panels::NTuple{6})
    storage.readonly && throw(ArgumentError(
        "MmapCSTapeStorage at $(storage.dir) is readonly; cannot allocate new slot"))
    storage.finalised && throw(ArgumentError(
        "MmapCSTapeStorage at $(storage.dir) is finalised; cannot allocate new slot"))
    storage.closed && throw(ArgumentError(
        "MmapCSTapeStorage at $(storage.dir) is closed; cannot allocate new slot"))
    _mmap_prepare_for_panels!(storage, panels)
    T = eltype(panels[1])
    shapes = ntuple(p -> size(panels[p]), 6)
    nbytes_per = ntuple(p -> Int64(sizeof(T)) * Int64(prod(shapes[p])), 6)
    base = _bump_cursor!(storage, sum(nbytes_per))
    offsets = _panel_offsets(base, nbytes_per)
    record_id = _push_record!(storage, T, offsets, nbytes_per, shapes)
    return MmapCSTapeSlot(storage, record_id, T, offsets, shapes)
end

"""
    stage_panels!(slot::MmapCSTapeSlot, src)

Write the six source panels into the slot's reserved region of
`records.bin` via `Mmap.mmap` views. Works uniformly for CPU `Array` and
GPU `CuArray` sources because `copyto!(::Array, ::CuArray)` performs a
device→host transfer.
"""
function stage_panels!(slot::MmapCSTapeSlot, src::NTuple{6})
    storage = slot.storage
    storage.readonly && throw(ArgumentError(
        "MmapCSTapeStorage at $(storage.dir) is readonly; cannot stage panels"))
    storage.finalised && throw(ArgumentError(
        "MmapCSTapeStorage at $(storage.dir) is finalised; cannot stage panels. " *
        "A reopened tape (`load_mmap_tape`) is for inspection only; slot " *
        "reuse / overwrite semantics are reserved for Phase B."))
    storage.closed && throw(ArgumentError(
        "MmapCSTapeStorage at $(storage.dir) is closed; cannot stage panels"))
    @inbounds for p in 1:6
        view = _mmap_view(storage.bin_io, slot.eltype, slot.shapes[p],
                          slot.offsets[p])
        copyto!(view, src[p])
    end
    _after_tape_stage!(storage)
    return slot
end

# ---------------------------------------------------------------------------
# Slot read — CPU returns mmap views; GPU is overloaded in the CUDA ext.
# ---------------------------------------------------------------------------

# Single-reader during reverse pass: the `_collect_surface_footprints`
# loop in Footprint/ReverseLoop.jl consumes slots sequentially. Each
# shape's cache is mutated in-place across reads, so this method is
# not safe under concurrent multi-threaded reverse walks against the
# same storage. Document via this comment until/unless that scenario
# materialises.
#
# Cache lookup is keyed by `slot.shapes` (the per-panel-shape tuple),
# so future callers that stage heterogeneous shapes (e.g. flux-tape
# work that pushes `panels_am` / `panels_bm` / `panels_cm` snapshots
# alongside the air-mass tape) reuse one six-buffer cache per
# distinct shape signature. As of A.2a the only forward-recorder that
# uses MmapCSTapeStorage is `_record_cs_mass_tape` / _tracer_tape in
# `src/Footprint/TapeRecording.jl`, which only stages m-shaped panels
# (`panels_m`, `panels_rm`) — so in practice `device_caches` holds at
# most one entry today and the multi-shape generality is forward
# insurance.
#
# For GPU runs the per-shape cache is pre-allocated by
# `_mmap_prepare_for_panels!` during `_allocate_tape_slot`, so
# `_ensure_tape_read_cache!` returns the existing `CuArray` cache
# without ever calling `similar(::Array)` on the mmap views.
function _tape_panels(slot::MmapCSTapeSlot)
    storage = slot.storage
    views = ntuple(p -> _mmap_view(storage.bin_io, slot.eltype,
                                   slot.shapes[p], slot.offsets[p]), 6)
    isempty(storage.device_caches) && return views
    cache = _ensure_tape_read_cache!(storage, views)
    @inbounds for p in 1:6
        copyto!(cache[p], views[p])
    end
    _after_tape_read!(storage)
    return cache
end

# ---------------------------------------------------------------------------
# Stage / read hooks — sync the device cache for GPU runs.
# ---------------------------------------------------------------------------

function _sync_mmap_tape_storage!(storage::MmapCSTapeStorage)
    storage.synchronize === nothing || storage.synchronize()
    return nothing
end

_after_tape_stage!(storage::MmapCSTapeStorage) =
    _sync_mmap_tape_storage!(storage)
_after_tape_read!(storage::MmapCSTapeStorage) =
    _sync_mmap_tape_storage!(storage)

# Shape-keyed device cache. The key is `ntuple(p -> size(panels[p]), 6)`,
# which makes consecutive reads of slots with the same shape signature
# (e.g. all `panels_m` slots in a reverse loop) reuse the same six
# device buffers.
#
# Eltype handling: each `MmapCSTapeStorage` derives its on-disk eltype
# from the first staged slot (`slot.eltype` in `_allocate_tape_slot`),
# and `_record_cs_mass_tape` / `_record_cs_tracer_tape` stage all
# their tape slots from arrays of the same `FT`. So in practice every
# `_ensure_tape_read_cache!` call for the same key arrives with the
# same eltype. The defensive `eltype` recheck below reallocates if
# that upstream invariant ever breaks (rather than silently
# corrupting the `copyto!`).
function _ensure_tape_read_cache!(storage::MmapCSTapeStorage,
                                  panels::NTuple{6})
    key = ntuple(p -> size(panels[p]), 6)
    cached = get(storage.device_caches, key, nothing)
    if cached !== nothing && eltype(cached[1]) === eltype(panels[1])
        return cached
    end
    fresh = ntuple(p -> similar(panels[p]), 6)
    storage.device_caches[key] = fresh
    return fresh
end

# ---------------------------------------------------------------------------
# Finalisation + cleanup
# ---------------------------------------------------------------------------

"""
    finalize_tape!(storage::MmapCSTapeStorage; quiet = false, strict = false)

Sync any pending mmap state, write the TOML manifest, and close the
underlying `records.bin` handle. Safe to call more than once; subsequent
calls are no-ops. If the storage owns its temp directory and
`cleanup_on_finalize` is true, the directory is also removed after
manifest emission.

By default (`strict = false`), failures during manifest emission, IO
close, or temp-dir cleanup are caught and surfaced via `@warn` (or
suppressed if `quiet = true`). This matches the temp-dir use case
where leaving a partially-finalised storage behind is benign — the
directory gets cleaned up by GC or by the caller's `mktempdir do`
block anyway.

With `strict = true`, manifest- and close-failures are still caught
into local error slots (so the IO handle and device cache are released
before throwing), but the first such error is rethrown at the end of
the function. This is the right setting whenever the storage is rooted
at a caller-supplied `tape_path` and a missing/corrupt `manifest.toml`
would defeat the purpose of persisting the tape.
"""
function finalize_tape!(storage::MmapCSTapeStorage; quiet::Bool = false,
                        strict::Bool = false)
    storage.closed && return storage
    manifest_err = nothing
    if !storage.finalised && isdir(storage.dir)
        try
            _write_manifest(storage)
            storage.finalised = true
        catch err
            manifest_err = err
            !quiet && !strict &&
                @warn "MmapCSTapeStorage: manifest emission failed" exception = err
        end
    end
    close_err = nothing
    try
        close(storage.bin_io)
    catch err
        close_err = err
        !quiet && !strict &&
            @warn "MmapCSTapeStorage: bin close failed" exception = err
    end
    # Drop the per-shape device cache so any device-resident buffers
    # are released promptly. This matters for long-running 4D-Var
    # loops that reuse one Julia session across many tapes.
    empty!(storage.device_caches)
    storage.closed = true
    if storage.cleanup_on_finalize && isdir(storage.dir)
        try
            rm(storage.dir; recursive = true, force = true)
        catch err
            !quiet && !strict &&
                @warn "MmapCSTapeStorage: temp dir cleanup failed" exception = err
        end
    end
    if strict
        manifest_err !== nothing && throw(manifest_err)
        close_err !== nothing && throw(close_err)
    end
    return storage
end

# GC-driven close. Closes the IOStream and optionally removes the temp
# directory, but does **not** emit the TOML manifest — emitting that
# from the GC thread risks running TOML.print across a task switch
# inside a finalizer. Manifest emission requires an explicit
# `finalize_tape!(storage)` call by the user.
function _mmap_storage_gc_close!(storage::MmapCSTapeStorage)
    storage.closed && return nothing
    try
        close(storage.bin_io)
    catch
        # Finaliser must never throw.
    end
    storage.closed = true
    if storage.cleanup_on_finalize && isdir(storage.dir)
        try
            rm(storage.dir; recursive = true, force = true)
        catch
            # Finaliser must never throw.
        end
    end
    return nothing
end

function _write_manifest(storage::MmapCSTapeStorage)
    meta = Dict{String, Any}(
        "version"             => _MMAP_TAPE_FORMAT_VERSION,
        "created_at"          => string(now()),
        "endianness"          => _machine_endianness(),
        "julia_version"       => string(VERSION),
        "total_bytes"         => storage.cursor,
        "record_count"        => length(storage.records),
        "finalised"           => true,
    )
    records = [
        Dict{String, Any}(
            "id"      => r.record_id,
            "eltype"  => string(r.eltype),
            "offsets" => collect(r.offsets),
            "nbytes"  => collect(r.nbytes),
            "shapes"  => [collect(s) for s in r.shapes],
        ) for r in storage.records
    ]
    tmp_path, io = mktemp(storage.dir; cleanup = false)
    try
        TOML.print(io, Dict{String, Any}("meta" => meta, "record" => records))
        flush(io)
        close(io)
        mv(tmp_path, storage.manifest_path; force = true)
    catch
        isopen(io) && close(io)
        rm(tmp_path; force = true)
        rethrow()
    end
    return storage.manifest_path
end

_machine_endianness() =
    ENDIAN_BOM == 0x04030201 ? "little" :
    ENDIAN_BOM == 0x01020304 ? "big" : "unknown"

# ---------------------------------------------------------------------------
# Resume / inspection API
# ---------------------------------------------------------------------------

# Manifest stores `eltype` as a string for human-readability + portability.
# The loader maps that string back to a Julia type via this whitelist
# rather than `eval(Meta.parse(...))`, since the manifest is data, not
# code, and tape files may have come from another machine.
function _parse_mmap_eltype(s::AbstractString)
    s == "Float32" && return Float32
    s == "Float64" && return Float64
    s == "Float16" && return Float16
    throw(ArgumentError("unsupported tape eltype $(repr(s)); " *
                        "expected Float32, Float64, or Float16"))
end

function _parse_mmap_record(d::AbstractDict, cursor::Int64)
    haskey(d, "id") && haskey(d, "eltype") && haskey(d, "offsets") &&
        haskey(d, "nbytes") && haskey(d, "shapes") ||
        throw(ArgumentError(
            "mmap tape manifest record is missing one of " *
            "id/eltype/offsets/nbytes/shapes; got keys $(collect(keys(d)))"))
    raw_offsets = d["offsets"]
    raw_nbytes = d["nbytes"]
    raw_shapes = d["shapes"]
    length(raw_offsets) == 6 || throw(ArgumentError(
        "mmap tape record id=$(d["id"]) has $(length(raw_offsets)) offsets; expected 6"))
    length(raw_nbytes) == 6 || throw(ArgumentError(
        "mmap tape record id=$(d["id"]) has $(length(raw_nbytes)) nbytes entries; expected 6"))
    length(raw_shapes) == 6 || throw(ArgumentError(
        "mmap tape record id=$(d["id"]) has $(length(raw_shapes)) shape entries; expected 6"))
    offsets = ntuple(p -> Int64(raw_offsets[p]), 6)
    nbytes = ntuple(p -> Int64(raw_nbytes[p]), 6)
    shapes = ntuple(6) do p
        s = raw_shapes[p]
        length(s) == 3 || throw(ArgumentError(
            "mmap tape record id=$(d["id"]) panel $(p) shape has $(length(s)) entries; expected 3"))
        (Int(s[1]), Int(s[2]), Int(s[3]))
    end
    et = _parse_mmap_eltype(d["eltype"])

    # Per-panel sanity: offsets and lengths must fit inside records.bin
    # (cursor is meta.total_bytes), nbytes must agree with shape × sizeof(et),
    # and no field may be negative. Caught here at load time so the user
    # gets a tape-aware diagnostic rather than an opaque `Mmap.mmap` error
    # at first read.
    bytes_per_elt = Int64(sizeof(et))
    @inbounds for p in 1:6
        offsets[p] >= 0 || throw(ArgumentError(
            "mmap tape record id=$(d["id"]) panel $(p) has negative offset $(offsets[p])"))
        nbytes[p] >= 0 || throw(ArgumentError(
            "mmap tape record id=$(d["id"]) panel $(p) has negative nbytes $(nbytes[p])"))
        all(>=(0), shapes[p]) || throw(ArgumentError(
            "mmap tape record id=$(d["id"]) panel $(p) has negative-shape entry $(shapes[p])"))
        offsets[p] + nbytes[p] <= cursor || throw(ArgumentError(
            "mmap tape record id=$(d["id"]) panel $(p) extends past records.bin: " *
            "offset=$(offsets[p]) + nbytes=$(nbytes[p]) > total_bytes=$(cursor)"))
        expected = bytes_per_elt * Int64(prod(shapes[p]))
        nbytes[p] == expected || throw(ArgumentError(
            "mmap tape record id=$(d["id"]) panel $(p) nbytes=$(nbytes[p]) " *
            "disagrees with shape $(shapes[p]) × sizeof($(et))=$(expected)"))
    end
    return MmapTapeRecordEntry(Int(d["id"]), et, offsets, nbytes, shapes)
end

"""
    load_mmap_tape(dir; readonly = true) -> MmapCSTapeStorage

Reopen a finalised `MmapCSTapeStorage` directory written by an earlier
session. The loader parses `manifest.toml`, validates the format
version and machine endianness, and rebuilds the in-memory record
table so callers can fetch slots via [`get_record`](@ref) and read
them through `_tape_panels`.

The returned storage has `finalised = true` and `cleanup_on_finalize
= false`; it never touches the on-disk files except through mmap reads
on the existing slots. With `readonly = true` (the default)
`_bump_cursor!`, `_allocate_tape_slot`, and `stage_panels!` throw on
mutation; `readonly = false` opens the binary in `r+` mode but is
intended for future Phase B work (slot reuse), not for forward
appends — appends are blocked by the `finalised` flag.

Validation:

* `manifest.toml` and `records.bin` must both exist under `dir`.
* Format version must equal `$(_MMAP_TAPE_FORMAT_VERSION)`.
* Endianness must match the loading machine.
* `meta.finalised` must be `true`; a torn tape from an interrupted
  run is rejected here (Phase B is responsible for repairing it).
* `records.bin` must be at least `meta.total_bytes` long; the cursor
  is restored from `meta.total_bytes`.
"""
function load_mmap_tape(dir::AbstractString; readonly::Bool = true)
    isdir(dir) || throw(ArgumentError(
        "tape directory $(repr(dir)) does not exist"))
    manifest_path = joinpath(dir, "manifest.toml")
    bin_path = joinpath(dir, "records.bin")
    isfile(manifest_path) || throw(ArgumentError(
        "tape directory $(repr(dir)) is missing manifest.toml"))
    isfile(bin_path) || throw(ArgumentError(
        "tape directory $(repr(dir)) is missing records.bin"))

    manifest = TOML.parsefile(manifest_path)
    haskey(manifest, "meta") || throw(ArgumentError(
        "mmap tape manifest at $(manifest_path) is missing the [meta] section"))
    meta = manifest["meta"]

    version = get(meta, "version", nothing)
    version === _MMAP_TAPE_FORMAT_VERSION || throw(ArgumentError(
        "mmap tape manifest version $(repr(version)) does not match " *
        "supported version $(repr(_MMAP_TAPE_FORMAT_VERSION))"))

    endian = get(meta, "endianness", nothing)
    local_endian = _machine_endianness()
    endian === local_endian || throw(ArgumentError(
        "mmap tape endianness $(repr(endian)) does not match host endianness " *
        "$(repr(local_endian)); cross-endian replay is not supported"))

    finalised = get(meta, "finalised", false)
    finalised === true || throw(ArgumentError(
        "mmap tape at $(dir) is not marked finalised; refusing to load. " *
        "Run finalize_tape!(storage) on the writer before reloading."))

    total_bytes = get(meta, "total_bytes", nothing)
    total_bytes isa Integer || throw(ArgumentError(
        "mmap tape manifest missing integer meta.total_bytes; got $(repr(total_bytes))"))
    cursor = Int64(total_bytes)
    file_size = filesize(bin_path)
    file_size == cursor || throw(ArgumentError(
        "mmap tape records.bin is $(file_size) bytes but manifest reports " *
        "total_bytes = $(cursor); refusing a truncated or trailing-data tape"))

    raw_records = get(manifest, "record", Any[])
    raw_records isa AbstractVector || throw(ArgumentError(
        "mmap tape manifest [[record]] section is not a list; got $(typeof(raw_records))"))
    record_count = get(meta, "record_count", length(raw_records))
    length(raw_records) == record_count || throw(ArgumentError(
        "mmap tape manifest reports record_count = $(record_count) but " *
        "contains $(length(raw_records)) [[record]] entries"))

    records = MmapTapeRecordEntry[_parse_mmap_record(r, cursor) for r in raw_records]
    for (i, r) in enumerate(records)
        r.record_id == i || throw(ArgumentError(
            "mmap tape manifest record at position $(i) has id $(r.record_id); " *
            "ids must be 1-indexed and contiguous"))
    end

    mode = readonly ? "r" : "r+"
    return MmapCSTapeStorage(dir, records, cursor; mode = mode)
end

"""
    get_record(storage::MmapCSTapeStorage, record_id::Integer) -> MmapCSTapeSlot

Return a slot descriptor for an existing record in `storage`. The slot
exposes the same `offsets`/`shapes`/`eltype` fields as one produced by
`_allocate_tape_slot`, so `_tape_panels(slot)` returns mmap views over
the recorded panels.
"""
function get_record(storage::MmapCSTapeStorage, record_id::Integer)
    storage.closed && throw(ArgumentError(
        "MmapCSTapeStorage at $(storage.dir) is closed; cannot fetch records"))
    1 <= record_id <= length(storage.records) ||
        throw(BoundsError(storage.records, record_id))
    r = storage.records[record_id]
    return MmapCSTapeSlot(storage, r.record_id, r.eltype, r.offsets, r.shapes)
end

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

function Base.show(io::IO, storage::MmapCSTapeStorage)
    print(io, "MmapCSTapeStorage(dir=", repr(storage.dir),
          ", records=", length(storage.records),
          ", bytes=", storage.cursor,
          ", cache_shapes=", length(storage.device_caches),
          storage.closed ? ", closed" : "",
          ")")
end
