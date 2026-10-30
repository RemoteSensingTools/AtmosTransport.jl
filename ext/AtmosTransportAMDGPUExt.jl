"""
ROCm extension for AtmosTransport, loaded alongside AMDGPU.jl.

`GPU(:rocm)` selects `ROCArray` and `ROCBackend()`. Loading the extension
only defines methods; it does not initialize a device or require working ROCm.
"""
module AtmosTransportAMDGPUExt

import AtmosTransport
import AMDGPU
import AtmosTransport.Tape:
    PinnedHostCSTapeStorage,
    PinnedHostCSTapeSlot,
    MmapCSTapeStorage,
    _allocate_tape_slot,
    _register_tape_read_backend!,
    _mmap_prepare_for_panels!,
    stage_panels!
using AtmosTransport.Architectures: GPU
using AMDGPU: ROCArray, AnyROCArray, ROCBackend

AtmosTransport.Architectures._array_adapter_for(::AnyROCArray) = ROCArray

function AtmosTransport.Architectures._reclaim_backend_pool!(::AnyROCArray)
    # Public pool-trimming API documented in AMDGPU's api/memory.md.
    AMDGPU.HIP.reclaim()
    return nothing
end

# AMDGPU documents unsafe_wrap(ROCArray, host_pointer, dims; own=true) as
# registering host memory and unregistering it when the wrapper is freed.
# Keep that registration alive until the Julia host array is collected. The
# wrapper owns the registration, NOT the Julia allocation. Never resize a
# registered host array. Empty arrays need no registration.
function _pin_host_array!(a::Array)
    isempty(a) && return a
    registration = GC.@preserve a unsafe_wrap(ROCArray, pointer(a), size(a); own = true)
    finalizer(a) do _
        AMDGPU.unsafe_free!(registration)
    end
    return a
end

AtmosTransport.Architectures._pin_host_memory!(storage, ::AnyROCArray) =
    AtmosTransport.Architectures.foreach_host_array(_pin_host_array!, storage)

_pinned_host_panel(a::ROCArray{T,N}) where {T,N} =
    _pin_host_array!(Array{T,N}(undef, size(a)))

function _allocate_tape_slot(storage::PinnedHostCSTapeStorage,
                             panels::NTuple{6,<:ROCArray})
    storage.synchronize = AMDGPU.synchronize
    _register_tape_read_backend!(storage, panels)
    host_panels = ntuple(p -> _pinned_host_panel(panels[p]), 6)
    return PinnedHostCSTapeSlot(storage, host_panels)
end

function stage_panels!(slot::PinnedHostCSTapeSlot,
                       src::NTuple{6,<:ROCArray})
    @inbounds for p in 1:6
        copyto!(slot.host_panels[p], src[p])
    end
    AMDGPU.synchronize()
    return slot
end

# Mmap storage owns its host views; it need not register every disk mapping.
# The generic tape path copies via its role-specific ROCArray read caches.
# Its synchronization hook waits for H2D transfers before a cache is reused
# and D2H transfers before the mmap payload is consumed.
function _mmap_prepare_for_panels!(storage::MmapCSTapeStorage,
                                   panels::NTuple{6,<:ROCArray})
    storage.synchronize = AMDGPU.synchronize
    _register_tape_read_backend!(storage, panels)
    return storage
end

end # module AtmosTransportAMDGPUExt
