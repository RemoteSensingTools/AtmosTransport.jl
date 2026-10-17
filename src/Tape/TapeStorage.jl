# ---------------------------------------------------------------------------
# Tape storage policies for the CS adjoint pipeline.
#
# A "tape" is the sequence of per-step state snapshots the forward pass
# records so the reverse pass can re-read them. This file defines:
#
#   * `AbstractCSTapeStorage` — policy abstract.
#   * `DeviceCSTapeStorage` — original in-memory/backend-resident policy.
#   * `AbstractHostCSTapeStorage` — policies whose slots live off the
#     source device (host RAM or disk) and are read back through
#     device-side read caches, one per read role and panel shape.
#   * `PinnedHostCSTapeStorage` — stages to host memory (pinned for
#     CuArray panels via the CUDA extension).
#   * `AbstractTapeReadRole` (`AirMassRead`, `TracerMassRead`) — which
#     reverse-kernel operand a tape read feeds.
#   * `CSTapeSlot` / `PinnedHostCSTapeSlot` — per-state slot containers.
#   * `_stage_panels`, `stage_panels!`, `_tape_panels(slot, role)`,
#     `_allocate_tape_slot` — public-ish staging API.
#
# The on-disk policy `MmapCSTapeStorage` lives in `MmapTapeStorage.jl`.
# ---------------------------------------------------------------------------

abstract type AbstractCSTapeStorage end

"""
    AbstractHostCSTapeStorage <: AbstractCSTapeStorage

Tape storage whose staged slots live off the source device: host memory for
`PinnedHostCSTapeStorage`, a file for `MmapCSTapeStorage`. Reading a slot
copies it into a device-side read cache and returns that cache. There is one
cache per `(read role, panel shapes)` pair, so the two operands that one
reverse kernel reads together (the air mass and tracer mass of a limited-PPM
sweep record) are always distinct buffers. Subtypes carry the fields
`read_template` (a zero-size array on the source backend, or `nothing`) and
`read_caches`.
"""
abstract type AbstractHostCSTapeStorage <: AbstractCSTapeStorage end

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

Read role of the staged tracer-mass state of a limited-PPM or FV3-profile
sweep record, which the reverse sweep reads together with the record's air
mass.
"""
struct TracerMassRead <: AbstractTapeReadRole end

const CSPanelCache = NTuple{6, <:AbstractArray}
const CSSynchronizeHook = Union{Nothing, Function}
const CSTapeReadCacheKey = Tuple{AbstractTapeReadRole, NTuple{6, NTuple{3, Int}}}

"""
    DeviceCSTapeStorage()

Tape storage policy that keeps staged adjoint mass states on the same backend
as the source panels. This preserves the original in-memory/device-resident
tape behavior while making the storage policy explicit.
"""
struct DeviceCSTapeStorage <: AbstractCSTapeStorage end

"""
    PinnedHostCSTapeStorage()

Tape storage policy that stages tape states in host memory and reads them back
through device-side read caches during the reverse pass, one per read role
(`AirMassRead`, `TracerMassRead`) and panel shape, so at most two staged
states are mirrored on the device at a time. The generic methods allocate
plain `Array`s, so the policy also works with CPU panel states; the CUDA
extension specialises `CuArray` panel states to use pinned (`CUDA.pin`) host
memory and a `CUDA.synchronize` hook.
"""
mutable struct PinnedHostCSTapeStorage <: AbstractHostCSTapeStorage
    # Zero-size array on the source backend. The read caches are `similar` to
    # it, so they live on the device although the slots hold host copies.
    read_template::Union{Nothing, AbstractArray}
    read_caches::Dict{CSTapeReadCacheKey, CSPanelCache}
    synchronize::CSSynchronizeHook

    PinnedHostCSTapeStorage() =
        new(nothing, Dict{CSTapeReadCacheKey, CSPanelCache}(), nothing)
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
function _resolve_tape_path(tape_storage::Union{Symbol, AbstractCSTapeStorage},
                            tape_path::Union{Nothing, AbstractString})
    _validate_tape_path(tape_storage, tape_path)
    tape_path === nothing && return nothing
    isdir(tape_path) || mkpath(tape_path)
    return String(tape_path)
end

# The checks of `_resolve_tape_path` without creating the directory, for
# callers that must reject a bad combination before other validation that can
# still fail, and so before any filesystem side effect.
function _validate_tape_path(tape_storage::Symbol,
                             tape_path::Union{Nothing, AbstractString})
    tape_path === nothing && return nothing
    tape_storage === :mmap || throw(ArgumentError(
        "tape_path requires tape_storage = :mmap; got tape_storage = " *
        "$(repr(tape_storage)). The :device and :pinned_host policies " *
        "keep tape state in memory and have no on-disk path."))
    isempty(tape_path) && throw(ArgumentError(
        "tape_path is empty; pass a non-empty directory path or omit the kwarg"))
    return nothing
end

function _validate_tape_path(::AbstractCSTapeStorage,
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

# `_tape_panels(slot, role::AbstractTapeReadRole) -> NTuple{6}`: the readable
# panels of a staged slot. Device-resident slots return their own panels;
# host-side storage (`AbstractHostCSTapeStorage`) copies the slot into the read
# cache of `role` and returns that cache, valid until the next read with the
# same role.
_tape_panels(slot::CSTapeSlot, ::AbstractTapeReadRole) = slot.panels

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

# Record the backend of the source panels when a slot is allocated. The slots
# hold host copies, so the read caches cannot be `similar` to them; they are
# `similar` to this zero-size template instead (a `CuArray` for GPU panels).
function _register_tape_read_backend!(storage::AbstractHostCSTapeStorage,
                                      panels::NTuple{6})
    storage.read_template === nothing &&
        (storage.read_template = similar(panels[1], 0, 0, 0))
    return storage
end

# The read cache of `role` for the shapes of `panels`, allocated on first use
# on the backend of `read_template` (else of `panels`). One cache per role
# keeps the operands of one reverse kernel apart; one per shape lets a storage
# hold slots of different shapes. Two reads of the same role and shape return
# the same buffer, so a record whose reverse kernel reads two staged states
# must give them different roles (a new role type if needed), or the kernel
# sees one operand twice (issue #57).
function _tape_read_cache!(storage::AbstractHostCSTapeStorage,
                           role::AbstractTapeReadRole, panels::NTuple{6})
    key = (role, ntuple(p -> size(panels[p]), 6))
    cached = get(storage.read_caches, key, nothing)
    _cache_matches(cached, panels) && return cached
    template = storage.read_template === nothing ? panels[1] : storage.read_template
    fresh = ntuple(p -> similar(template, eltype(panels[p]), size(panels[p])), 6)
    storage.read_caches[key] = fresh
    return fresh
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
    _register_tape_read_backend!(storage, panels)
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

function _tape_panels(slot::PinnedHostCSTapeSlot, role::AbstractTapeReadRole)
    cache = _tape_read_cache!(slot.storage, role, slot.host_panels)
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
