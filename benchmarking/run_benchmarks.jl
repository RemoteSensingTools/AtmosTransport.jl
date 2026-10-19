#!/usr/bin/env julia

using ArgParse

function _early_backend(args)
    for arg in args
        startswith(arg, "--backend=") && return Symbol(lowercase(split(arg, "=", limit = 2)[2]))
    end
    for i in 1:(length(args) - 1)
        args[i] == "--backend" && return Symbol(lowercase(args[i + 1]))
    end
    return :cpu
end

function _preload_backend!(backend::Symbol)
    if backend === :cuda || backend === :gpu_cuda
        @info "Preloading CUDA for benchmark backend"
        Core.eval(Main, :(using CUDA))
        cuda = Base.invokelatest(getfield, Main, :CUDA)
        Base.invokelatest(getproperty(cuda, :functional)) ||
            error("CUDA backend requested but CUDA.functional() is false")
        isdefined(cuda, :allowscalar) &&
            Base.invokelatest(getproperty(cuda, :allowscalar), false)
    elseif backend === :rocm
        @info "Preloading AMDGPU for benchmark backend"
        Core.eval(Main, :(using AMDGPU))
        amdgpu = Base.invokelatest(getfield, Main, :AMDGPU)
        Base.invokelatest(getproperty(amdgpu, :functional)) ||
            error("ROCm backend requested but AMDGPU.functional() is false")
        Base.invokelatest(getproperty(amdgpu, :allowscalar), false)
    elseif backend === :metal || backend === :gpu_metal
        @info "Preloading Metal for benchmark backend"
        Core.eval(Main, :(using Metal))
        metal = Base.invokelatest(getfield, Main, :Metal)
        if isdefined(metal, :functional)
            Base.invokelatest(getproperty(metal, :functional)) ||
                error("Metal backend requested but Metal.functional() is false")
        end
        isdefined(metal, :allowscalar) &&
            Base.invokelatest(getproperty(metal, :allowscalar), false)
    end
    return nothing
end

_preload_backend!(_early_backend(ARGS))

using AtmosTransportBenchmarks

AtmosTransportBenchmarks.main(ARGS)
