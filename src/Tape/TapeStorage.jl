# ---------------------------------------------------------------------------
# Tape storage policies for the CS adjoint pipeline.
#
# A "tape" is the sequence of per-step state snapshots the forward pass
# records so the reverse pass can re-read them. This file defines:
#
#   * `AbstractCSTapeStorage` — policy abstract.
#   * `DeviceCSTapeStorage` — original in-memory/backend-resident policy.
#   * `PinnedHostCSTapeStorage` — stages to host memory (pinned for
#     CuArray panels via the CUDA extension) with a shared device-side
#     read cache during reverse.
#   * `CSTapeSlot` / `PinnedHostCSTapeSlot` — per-state slot containers.
#   * `_stage_panels`, `stage_panels!`, `_tape_panels`,
#     `_allocate_tape_slot` — public-ish staging API.
#
# The on-disk policy `MmapCSTapeStorage` lives in `MmapTapeStorage.jl`.
# ---------------------------------------------------------------------------

abstract type AbstractCSTapeStorage end

const CSPanelCache = NTuple{6, <:AbstractArray}
const CSSynchronizeHook = Union{Nothing, Function}

"""
    DeviceCSTapeStorage()

Tape storage policy that keeps staged adjoint mass states on the same backend
as the source panels. This preserves the original in-memory/device-resident
tape behavior while making the storage policy explicit.
"""
struct DeviceCSTapeStorage <: AbstractCSTapeStorage end

"""
    PinnedHostCSTapeStorage()

Tape storage policy that stages tape states in host memory and uses a shared
device-side read cache during the reverse pass. The generic methods allocate
plain `Array`s, so the policy also works with CPU panel states; the CUDA
extension specialises `CuArray` panel states to use pinned (`CUDA.pin`) host
memory and a `CUDA.synchronize` hook.
"""
mutable struct PinnedHostCSTapeStorage <: AbstractCSTapeStorage
    device_cache::Union{Nothing, CSPanelCache}
    synchronize::CSSynchronizeHook

    PinnedHostCSTapeStorage() = new(nothing, nothing)
end

struct CSTapeSlot{S <: AbstractCSTapeStorage, P}
    storage::S
    panels::P
end

struct PinnedHostCSTapeSlot{S <: PinnedHostCSTapeStorage, H}
    storage::S
    host_panels::H
end

# ---------------------------------------------------------------------------
# Policy resolution
# ---------------------------------------------------------------------------

_tape_storage(storage::AbstractCSTapeStorage) = storage
_tape_storage(::Val{:device}) = DeviceCSTapeStorage()
_tape_storage(::Val{:pinned_host}) = PinnedHostCSTapeStorage()
_tape_storage(storage::Symbol) = _tape_storage(Val(storage))
_tape_storage(storage) = throw(ArgumentError(
    "unsupported CS adjoint tape storage $(storage); " *
    "supported: :device, :pinned_host, :mmap"))

"""
    _resolve_tape_path(tape_storage, tape_path) -> Union{Nothing, String}

Validate that `tape_path` is compatible with `tape_storage`. Returns
`nothing` when `tape_path === nothing` (use a temp dir / no disk
backing). Returns the path string (after `mkpath`) when `tape_storage`
is `:mmap` and a path is supplied. Throws an `ArgumentError` if a path
is supplied with non-`:mmap` storage (since `:device` and
`:pinned_host` keep tape state in memory, a path has no meaning), or
if `tape_storage` is a pre-constructed `AbstractCSTapeStorage` (the
storage's own configuration already determines where the tape lives).
"""
function _resolve_tape_path(tape_storage::Symbol,
                            tape_path::Union{Nothing, AbstractString})
    tape_path === nothing && return nothing
    tape_storage === :mmap || throw(ArgumentError(
        "tape_path requires tape_storage = :mmap; got tape_storage = " *
        "$(repr(tape_storage)). The :device and :pinned_host policies " *
        "keep tape state in memory and have no on-disk path."))
    isempty(tape_path) && throw(ArgumentError(
        "tape_path is empty; pass a non-empty directory path or omit the kwarg"))
    isdir(tape_path) || mkpath(tape_path)
    return String(tape_path)
end

function _resolve_tape_path(::AbstractCSTapeStorage,
                            tape_path::Union{Nothing, AbstractString})
    tape_path === nothing && return nothing
    throw(ArgumentError(
        "tape_path is not compatible with a pre-constructed " *
        "AbstractCSTapeStorage; the storage's own configuration already " *
        "determines where the tape lives. Pass a Symbol (:mmap) plus " *
        "tape_path, or pass the storage instance alone."))
end

# Generic no-op fallback so the strided checkpoint driver can call
# `finalize_tape!(storage)` once per window without dispatching on
# storage type. `MmapCSTapeStorage` overrides this in
# `MmapTapeStorage.jl` to sync + emit manifest + close the bin handle.
# `DeviceCSTapeStorage` and `PinnedHostCSTapeStorage` hold their state
# in plain Julia arrays and do not need an explicit teardown — GC
# handles release once the storage drops out of scope.
finalize_tape!(::AbstractCSTapeStorage; quiet::Bool = false,
                strict::Bool = false) = nothing

# ---------------------------------------------------------------------------
# Slot read / staging hooks
# ---------------------------------------------------------------------------

_tape_panels(slot::CSTapeSlot) = slot.panels

function _sync_pinned_tape_storage!(storage::PinnedHostCSTapeStorage)
    storage.synchronize === nothing || storage.synchronize()
    return nothing
end

_after_tape_stage!(storage::PinnedHostCSTapeStorage) =
    _sync_pinned_tape_storage!(storage)
_after_tape_read!(storage::PinnedHostCSTapeStorage) =
    _sync_pinned_tape_storage!(storage)

_host_tape_panel(a::AbstractArray{T,N}) where {T,N} =
    Array{T,N}(undef, size(a))

function _cache_matches(cache, panels::NTuple{6})
    cache isa NTuple{6} || return false
    @inbounds for p in 1:6
        if eltype(cache[p]) !== eltype(panels[p]) ||
           size(cache[p]) != size(panels[p])
            return false
        end
    end
    return true
end

function _ensure_tape_read_cache!(storage::PinnedHostCSTapeStorage,
                                  panels::NTuple{6})
    if !_cache_matches(storage.device_cache, panels)
        storage.device_cache = ntuple(p -> similar(panels[p]), 6)
    end
    return storage.device_cache
end

# ---------------------------------------------------------------------------
# Per-policy slot allocators + stagers
# ---------------------------------------------------------------------------

function _allocate_tape_slot(storage::DeviceCSTapeStorage, panels::NTuple{6})
    slot_panels = ntuple(p -> similar(panels[p]), 6)
    return CSTapeSlot(storage, slot_panels)
end

function stage_panels!(slot::CSTapeSlot{DeviceCSTapeStorage}, src::NTuple{6})
    @inbounds for p in 1:6
        copyto!(slot.panels[p], src[p])
    end
    return slot
end

function _allocate_tape_slot(storage::PinnedHostCSTapeStorage,
                             panels::NTuple{6})
    _ensure_tape_read_cache!(storage, panels)
    host_panels = ntuple(p -> _host_tape_panel(panels[p]), 6)
    return PinnedHostCSTapeSlot(storage, host_panels)
end

function stage_panels!(slot::PinnedHostCSTapeSlot, src::NTuple{6})
    @inbounds for p in 1:6
        copyto!(slot.host_panels[p], src[p])
    end
    _after_tape_stage!(slot.storage)
    return slot
end

function _tape_panels(slot::PinnedHostCSTapeSlot)
    cache = _ensure_tape_read_cache!(slot.storage, slot.host_panels)
    @inbounds for p in 1:6
        copyto!(cache[p], slot.host_panels[p])
    end
    _after_tape_read!(slot.storage)
    return cache
end

function _stage_panels(storage::AbstractCSTapeStorage, panels::NTuple{6})
    slot = _allocate_tape_slot(storage, panels)
    return stage_panels!(slot, panels)
end

# ---------------------------------------------------------------------------
# Byte estimation helper
# ---------------------------------------------------------------------------

_bytes_per_panel_tuple(panels::NTuple{6}) =
    sum(sizeof(eltype(panels[p])) * length(panels[p]) for p in 1:6)
