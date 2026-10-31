"""
ROCm extension for AtmosTransport, loaded alongside AMDGPU.jl.

`GPU(:rocm)` selects `ROCArray` and `ROCBackend()`. Loading the extension
only defines methods; it does not initialize a device or require working ROCm.
"""
module AtmosTransportAMDGPUExt

import AtmosTransport
import AMDGPU
import AtmosTransport.Tape:
    _free_tape_array!,
    _pin_host_chunk,
    _tape_synchronize,
    _unpin_host_chunk
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

# Adjoint tape hooks (`src/Tape/`): explicit release of tape memory, and
# synchronous device↔host tape copies (staging and every read wait).
_free_tape_array!(a::ROCArray) = AMDGPU.unsafe_free!(a)
_tape_synchronize(::ROCArray) = AMDGPU.synchronize()

# A pinned-host tape arena registers each chunk once, when it allocates it,
# through the same `unsafe_wrap(...; own = true)` registration as above, and
# unregisters it (frees the wrapper) before freeing the chunk. Mmap tapes are
# not registered: their file mapping is copied through the read caches.
struct _ROCHostRegistration{W}
    wrapper::W
end

_pin_host_chunk(::ROCArray, ptr::Ptr, nbytes::Integer) =
    _ROCHostRegistration(unsafe_wrap(ROCArray, Ptr{UInt8}(ptr), (Int(nbytes),); own = true))

_unpin_host_chunk(registration::_ROCHostRegistration) =
    (AMDGPU.unsafe_free!(registration.wrapper); nothing)

end # module AtmosTransportAMDGPUExt
