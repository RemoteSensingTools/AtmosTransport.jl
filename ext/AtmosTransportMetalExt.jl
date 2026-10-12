"""
Metal extension for AtmosTransport.

Loaded automatically when `using Metal` is called alongside AtmosTransport.
Provides GPU array types and KernelAbstractions device for Apple Silicon GPUs.
"""
module AtmosTransportMetalExt

import AtmosTransport
import Metal
using AtmosTransport.Architectures: GPU
using Metal: MtlArray, MetalBackend

AtmosTransport.Architectures.array_type(::GPU{:metal}) = MtlArray
AtmosTransport.Architectures.device(::GPU{:metal})     = MetalBackend()
AtmosTransport.Architectures.architecture(::MtlArray) = GPU(:metal)
AtmosTransport.Architectures._array_adapter_for(::MtlArray) = MtlArray

function AtmosTransport.Architectures._reclaim_backend_pool!(::MtlArray)
    isdefined(Metal, :synchronize) && Metal.synchronize()
    return nothing
end

AtmosTransport.Architectures._total_accumulator_type(::MetalBackend) = Float32

# Fused point operations run faster on Metal as separate launches with their
# arrays bound directly (see `Architectures.fusion_policy`).
AtmosTransport.Architectures.fusion_policy(::MetalBackend) = AtmosTransport.Architectures.SeparateLaunches()

# Metal's ordinary host-to-MtlArray `copyto!` synchronizes the global queue
# before every transfer. For dense, same-eltype Array-to-MtlArray copies, sync
# once for the whole dispatch group and use the lower-level transfer primitive.
# The primitive is deliberately called without `async=true`: private-storage
# arrays therefore wait for each blit to complete. This removes repeated global
# queue synchronizations, but Metal staging still includes every DMA completion
# wait. Other source/destination pairs retain the ordinary `_copy_slice!` path.
const SurfaceFlux = AtmosTransport.Operators.SurfaceFlux

function _copy_flux_slice_lowlevel!(dest::MtlArray{T}, slot::Integer,
                                    src::Array{T}, k::Integer) where T
    n = length(src) ÷ size(src, ndims(src))
    doffs = (slot - 1) * n + 1
    soffs = (k - 1) * n + 1
    GC.@preserve dest src Base.unsafe_copyto!(Metal.device(dest), pointer(dest, doffs),
                                              pointer(src, soffs), n)
    return dest
end

function _copy_flux_slice_lowlevel!(dest::Tuple, slot::Integer,
                                    src::Tuple, k::Integer)
    foreach((d, s) -> _copy_flux_slice_lowlevel!(d, slot, s, k), dest, src)
    return dest
end

_optimized_flux_slice_copy(::Any, ::Any) = false
_optimized_flux_slice_copy(::MtlArray{T}, ::Array{T}) where T = true
function _optimized_flux_slice_copy(dest::Tuple, src::Tuple)
    length(dest) == length(src) || return false
    return all(((d, s),) -> _optimized_flux_slice_copy(d, s), zip(dest, src))
end

# All concrete MtlArray storage/dimension types dispatch to the method below.
SurfaceFlux._bulk_copy_flux_slices_group(::MtlArray) = MtlArray

function SurfaceFlux._bulk_copy_flux_slices!(::MtlArray, copies::AbstractVector)
    optimized = filter(copy -> _optimized_flux_slice_copy(copy.buffer.slots,
                                                          copy.series), copies)
    fallback = filter(copy -> !_optimized_flux_slice_copy(copy.buffer.slots,
                                                          copy.series), copies)
    isempty(optimized) || Metal.synchronize()
    for copy in optimized
        _copy_flux_slice_lowlevel!(copy.buffer.slots, copy.slot, copy.series, copy.slice)
    end
    for copy in fallback
        SurfaceFlux._copy_slice!(copy.buffer.slots, copy.slot, copy.series, copy.slice)
    end
    return nothing
end

end # module AtmosTransportMetalExt
