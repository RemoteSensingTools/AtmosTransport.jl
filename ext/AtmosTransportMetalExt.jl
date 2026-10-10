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

end # module AtmosTransportMetalExt
