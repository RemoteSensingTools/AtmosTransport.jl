"""
CUDA extension for AtmosTransport.

Loaded automatically when `using CUDA` is called alongside AtmosTransport.
Provides CUDA array adaptation, launch policies, and tape storage support.
"""
module AtmosTransportCUDAExt

import AtmosTransport
import CUDA
import AtmosTransport.Adjoints:
    PinnedHostCSTapeStorage,
    PinnedHostCSTapeSlot,
    MmapCSTapeStorage,
    _allocate_tape_slot,
    _register_tape_read_backend!,
    _mmap_prepare_for_panels!,
    stage_panels!
using AtmosTransport.Architectures: GPU
using CUDA: CuArray, CUDABackend

# A scalar workgroup size extends to (size, 1, 1). On C90 this leaves
# 166 of a 256-thread row inactive. A 32×2 tile keeps adjacent i cells in
# one warp and covers panel rows with much less padding. Measured on V100.
AtmosTransport.Operators.Advection._cs_packed_sweep_workgroupsize(
    ::CUDABackend, ::AtmosTransport.Operators.Advection.PPMScheme,
    ::Type{Float32}) = (32, 2)

# Float64 packed sweeps benefit from the same contiguous 32-cell rows,
# with one warp per workgroup on the measured V100 workload.
AtmosTransport.Operators.Advection._cs_packed_sweep_workgroupsize(
    ::CUDABackend, ::AtmosTransport.Operators.Advection.PPMScheme,
    ::Type{Float64}) = 32

# Share one factorization per diffusion column, then distribute independent
# tracer solves across warps. The tracer tile pairs two tracers while keeping
# all 32 i cells of each warp contiguous. No extra workspace is required.
AtmosTransport.Operators.Diffusion._cs_dkg_mass_workgroupsize(
    ::CUDABackend, ::Type) = (32, 2)
AtmosTransport.Operators.Diffusion._cs_dkg_tracer_workgroupsize(
    ::CUDABackend, ::Type) = (32, 1, 2)

# Static shared storage for a six-tracer Float64 batch is
# 8*(L^2 + 9L + 2) + 4*(L + 2) bytes. L=73 uses 48,204 bytes;
# L=74 exceeds the 48 KiB budget. Keep Float32's portable 85-level gate.
AtmosTransport.Operators.Convection.TM5._tm5_collab_max_depth(
    ::Type{Float64}, ::CUDABackend) = AtmosTransport.Operators.Convection.TM5._TM5_COLLAB_MAX_DEPTH_FLOAT64_CUDA

AtmosTransport.Architectures._array_adapter_for(::CUDA.AbstractGPUArray) = CuArray

function AtmosTransport.Architectures._reclaim_backend_pool!(::CUDA.AbstractGPUArray)
    CUDA.synchronize()
    CUDA.reclaim()
    return nothing
end

# Pinned host staging of device windows (`pin_host_memory!`): `CUDA.pin`
# registers each host array in place with the driver.
AtmosTransport.Architectures._pin_host_memory!(storage, ::CUDA.AbstractGPUArray) =
    AtmosTransport.Architectures.foreach_host_array(a -> isempty(a) || CUDA.pin(a), storage)

_pinned_host_panel(a::CuArray{T,N}) where {T,N} =
    CUDA.pin(Array{T,N}(undef, size(a)))

function _allocate_tape_slot(storage::PinnedHostCSTapeStorage,
                             panels::NTuple{6,<:CuArray})
    storage.synchronize = CUDA.synchronize
    _register_tape_read_backend!(storage, panels)
    host_panels = ntuple(p -> _pinned_host_panel(panels[p]), 6)
    return PinnedHostCSTapeSlot(storage, host_panels)
end

function stage_panels!(slot::PinnedHostCSTapeSlot,
                       src::NTuple{6,<:CuArray})
    @inbounds for p in 1:6
        copyto!(slot.host_panels[p], src[p])
    end
    CUDA.synchronize()
    return slot
end

# Plan 26 Phase A.1 — Mmap tape policy on GPU. `_mmap_prepare_for_panels!`
# is called from `_allocate_tape_slot(::MmapCSTapeStorage, panels)` BEFORE
# disk reservation; it records the CuArray backend of the per-role read
# caches (allocated on first read) and installs the CUDA.synchronize hook
# so subsequent `copyto!(::Array, ::CuArray)` and
# `copyto!(::CuArray, ::Array)` operations against the mmap views are
# wrapped by a device sync. No specialised `_allocate_tape_slot` or
# `stage_panels!` is needed: both default to mmap-write / mmap-read,
# and CUDA.jl provides synchronous `copyto!` between `Array` and
# `CuArray` of matching element type and shape.
function _mmap_prepare_for_panels!(storage::MmapCSTapeStorage,
                                   panels::NTuple{6,<:CuArray})
    storage.synchronize = CUDA.synchronize
    _register_tape_read_backend!(storage, panels)
    return storage
end

end # module AtmosTransportCUDAExt
