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
# before every transfer. Surface-flux staging has six panel transfers per
# missing logical slice, often across many sources, so synchronize once for the
# whole operator/window and use the lower-level ordered transfer primitive.
const SurfaceFlux = AtmosTransport.Operators.SurfaceFlux

function _copy_flux_slice_no_sync!(dest::MtlArray{T}, slot::Integer,
                                   src::Array{T}, k::Integer) where T
    n = length(src) ÷ size(src, ndims(src))
    doffs = (slot - 1) * n + 1
    soffs = (k - 1) * n + 1
    GC.@preserve dest src Base.unsafe_copyto!(Metal.device(dest), pointer(dest, doffs),
                                              pointer(src, soffs), n)
    return dest
end

function _copy_flux_slice_no_sync!(dest::Tuple, slot::Integer,
                                   src::Tuple, k::Integer)
    foreach((d, s) -> _copy_flux_slice_no_sync!(d, slot, s, k), dest, src)
    return dest
end

function SurfaceFlux._bulk_copy_flux_slices!(::MtlArray, copies::AbstractVector)
    Metal.synchronize()
    for copy in copies
        _copy_flux_slice_no_sync!(copy.buffer.slots, copy.slot, copy.series, copy.slice)
    end
    return nothing
end

end # module AtmosTransportMetalExt
