"""
CUDA extension for AtmosTransport.

Loaded automatically when `using CUDA` is called alongside AtmosTransport.
Provides CUDA array adaptation, launch policies, and tape storage support.
"""
module AtmosTransportCUDAExt

import AtmosTransport
import CUDA
import AtmosTransport.Tape:
    _free_tape_array!,
    _pin_host_chunk,
    _tape_synchronize,
    _unpin_host_chunk
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

# Query after reclaim so cached backend allocations count as available.
function AtmosTransport.Tape._device_tape_memory(::CUDABackend, a)
    AtmosTransport.Architectures._reclaim_backend_pool!(a)
    free, capacity = CUDA.memory_info()
    return Int(free), Int(capacity)
end

# Adjoint tape hooks (`src/Tape/`). Tape memory is released explicitly when a
# reverse-pass call returns (device slabs, read caches, checkpoints, Lin–Rood
# records), so the CUDA pool gets it back now rather than at a later GC.
_free_tape_array!(a::CuArray) = CUDA.unsafe_free!(a)

# Device↔host tape copies are synchronous: staging and every read wait for
# the device (host pools: `PinnedHostCSTapeStorage`, `MmapCSTapeStorage`).
_tape_synchronize(::CuArray) = CUDA.synchronize()

# A pinned-host tape arena registers each chunk with the driver once, when it
# allocates it (`cuMemHostRegister`), and unregisters it before freeing it:
# a staged state is then one device→host copy into page-locked memory, with
# no per-state allocation or registration. The registration belongs to the
# context that was current when the chunk was allocated.
struct _CUDAHostRegistration
    context::CUDA.CuContext
    ptr::Ptr{Cvoid}
end

function _pin_host_chunk(::CuArray, ptr::Ptr, nbytes::Integer)
    CUDA.cuMemHostRegister_v2(Ptr{Cvoid}(ptr), Csize_t(nbytes), Cuint(0))
    return _CUDAHostRegistration(CUDA.context(), Ptr{Cvoid}(ptr))
end

function _unpin_host_chunk(registration::_CUDAHostRegistration)
    CUDA.context!(registration.context; skip_destroyed = true) do
        CUDA.cuMemHostUnregister(registration.ptr)
    end
    return nothing
end

AtmosTransport.Tape._is_reservation_failure(err::CUDA.OutOfGPUMemoryError) = true

# TM5 spectral synthesis: core owns portable KA/GEMM/FFT and quadrature;
# CUDA owns device selection and the strict precision policy.
function AtmosTransport.Preprocessing.Common.validate_gpu_spectral_backend(::Val{:cuda})
    CUDA.functional() || throw(ArgumentError("GPU spectral synthesis requires a functional CUDA backend"))
    return nothing
end
AtmosTransport.Preprocessing.Spectral.spectral_gpu_adapter(::Val{:cuda}) = CuArray
function AtmosTransport.Preprocessing.Spectral.spectral_gpu_precision(::Val{:cuda})
    cap = CUDA.capability(CUDA.device())
    return cap in (v"6.0",v"7.0",v"8.0",v"9.0",v"10.0") ? Float64 : Float32
end
AtmosTransport.Preprocessing.Spectral.spectral_gpu_math_guard(::CUDABackend) =
    AtmosTransport.Preprocessing.Spectral.spectral_gpu_math_guard(Val(:cuda))
function AtmosTransport.Preprocessing.Spectral.spectral_gpu_math_guard(::Val{:cuda})
    CUDA.functional() || throw(ArgumentError("GPU spectral synthesis requires a functional CUDA backend"))
    CUDA.math_mode() == CUDA.FAST_MATH && throw(ArgumentError("TF32/FAST_MATH is forbidden for spectral synthesis"))
    return nothing
end

function AtmosTransport.Preprocessing.Spectral.with_spectral_math(f, ::CUDABackend)
    AtmosTransport.Preprocessing.Spectral.spectral_gpu_math_guard(Val(:cuda))
    previous = CUDA.math_mode()
    try
        CUDA.math_mode!(CUDA.PEDANTIC_MATH)
        return f()
    finally
        CUDA.math_mode!(previous)
    end
end

end # module AtmosTransportCUDAExt
