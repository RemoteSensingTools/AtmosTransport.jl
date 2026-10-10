#!/usr/bin/env julia
# Select a device explicitly, then run from the checkout being measured:
# CUDA_VISIBLE_DEVICES=0 ATMOSTR_MATRIX_GPU_NAME=V100 julia --project=. \
#   scripts/benchmarks/heritage/bench_matrix_convection_gpu.jl results.toml native
# `split` measures independent <=6-tracer launches, usable on pre-batching code.
# Inputs/buffers are prepared before timing; each sample restores the same RHS.
# V100 needs a compatible toolkit. In each benchmark environment, run once:
# julia --project=. -e 'using CUDA; CUDA.set_runtime_version!(v"12.6")'
using CUDA
using AtmosTransport
import KernelAbstractions as KA
using Random, Statistics, TOML, Test
const C = AtmosTransport.Operators.Convection

function fixture(nc, nz, nt, downdrafts)
    rng = MersenneTwister(9052026)
    m = 50f0 .+ 150f0 .* rand(rng, Float32, nc, nz)
    q = rand(rng, Float32, nc, nz, nt) .* m
    rates = ntuple(_ -> zeros(Float32, nc, nz), 4)
    for c in 1:nc
        top = 1 + mod(c-1, 8)
        rates[1][c,nz-1] = 0.04f0
        rates[2][c,top] = 0.04f0
        if downdrafts
            rates[3][c,top+1] = 0.01f0
            rates[4][c,nz] = 0.01f0
        end
    end
    (; m, q, rates, nc, nz, nt)
end

function dense_reference(f, c)
    rates = map(r -> view(r,c,:), f.rates)
    top, _, lfs = C._tm5_diagnose_cloud_dims(rates[2], rates[3], f.nz)
    lo = min(lfs, max(top,2)-1)
    A = zeros(Float32, f.nz, f.nz)
    piv = zeros(Int, f.nz)
    C._tm5_build_conv1!(A, rates..., view(f.m,c,:), top, lfs, 1800f0, f.nz;
        f=A, amu=zeros(Float32,f.nz+1), amd=zeros(Float32,f.nz+1))
    C._tm5_lu!(A, piv, f.nz; icltop_eff=lo)
    q = copy(f.q[c,:,:])
    C._tm5_solve!(q, A, piv, f.nz, f.nt; icltop_eff=lo)
    q
end

function kernel_resources(kernel, q, m, rates, areas, nz, nc, wg)
    # Match CUDAKernels' compilation options and metadata for this exact launch.
    nd, _, iterspace, _ = KA.launch_config(kernel, wg*nc, nothing)
    ctx = KA.mkcontext(kernel, nd, iterspace)
    args = (ctx, q, m, rates..., areas, nz, size(q,3), 1800f0, Val(nz))
    compiled = CUDA.cufunction(kernel.f, typeof(CUDA.cudaconvert(args));
        always_inline=KA.backend(kernel).always_inline, maxthreads=wg)
    memory = CUDA.memory(compiled)
    Dict("compiled_shared_bytes"=>memory.shared,
         "compiled_local_bytes_per_thread"=>memory.local,
         "registers_per_thread"=>CUDA.registers(compiled),
         "maximum_active_blocks_per_sm"=>CUDA.active_blocks(compiled.fun,wg),
         "theoretical_occupancy"=>CUDA.occupancy(compiled.fun,wg))
end

function measure(nc, nz, nt, downdrafts, mode)
    launch_tracers = mode == "split" ? min(6,nt) : nt
    C._tm5_collab_supports(nz, launch_tracers) ||
        error("This checkout cannot launch L=$nz Nt=$launch_tracers; use split mode for pre-batching code")
    f = fixture(nc, nz, nt, downdrafts)
    m, rates, areas = CuArray(f.m), map(CuArray, f.rates), CUDA.ones(Float32,nc)
    ranges = mode == "split" ? [i:min(i+5,nt) for i in 1:6:nt] : [1:nt]
    initial = [CuArray(f.q[:,:,r]) for r in ranges]
    qs = copy.(initial)
    backend = KA.get_backend(m)
    wg = C._TM5_COLLAB_WG_SIZE
    kernel = C._tm5_faceindexed_column_collab_kernel!(backend, wg)
    function launch!()
        for q in qs
            kernel(q, m, rates..., areas, nz, size(q,3), 1800f0, Val(nz);
                   ndrange=wg*nc)
        end
        nothing
    end
    function restore!()
        foreach(copyto!, qs, initial)
        CUDA.synchronize()
    end
    launch!() # Includes compilation, excluded from timings.
    CUDA.synchronize()
    result = cat(Array.(qs)...; dims=3)
    max_rel_error = 0.0
    for c in unique([1,2,8,nc])
        ref = dense_reference(f,c)
        actual = result[c,:,:]
        max_rel_error = max(max_rel_error,
            maximum(abs.(Float64.(actual)-ref)) / maximum(abs.(ref)))
        @test isapprox(actual, ref; rtol=100eps(Float32))
    end
    before, after = sum(Float64.(f.q); dims=2), sum(Float64.(result); dims=2)
    mass_error = maximum(abs.(after-before) ./ before)
    @test mass_error < 100eps(Float32)
    @test minimum(result) >= 0
    for _ in 1:3
        restore!()
        launch!()
        CUDA.synchronize()
    end
    device_ms, wall_ms = Float64[], Float64[]
    for _ in 1:9
        restore!()
        start = time_ns()
        elapsed = CUDA.@elapsed launch!()
        CUDA.synchronize()
        push!(wall_ms, (time_ns()-start)*1e-6)
        push!(device_ms, elapsed*1e3)
    end
    # RHS storage is doubled solely to restore identical input between samples.
    shared_bytes = 4*(nz*nz + C._TM5_COLLAB_TRACER_BATCH*nz + 2nz + 2(nz+1) + 2)
    r = Dict("columns"=>nc, "levels"=>nz, "tracers"=>nt,
        "downdrafts"=>downdrafts, "mode"=>mode, "launches"=>length(qs),
        "median_device_ms"=>median(device_ms), "minimum_device_ms"=>minimum(device_ms),
        "median_wall_ms"=>median(wall_ms), "samples_device_ms"=>device_ms,
        "samples_wall_ms"=>wall_ms, "shared_bytes_per_block"=>shared_bytes,
        "maximum_reference_relative_error"=>max_rel_error,
        "maximum_mass_relative_error"=>mass_error)
    merge!(r, kernel_resources(kernel, qs[1], m, rates, areas, nz, nc, wg))
    println("L=$nz Nt=$nt downdrafts=$downdrafts $mode: ",
        round(r["median_device_ms"]; digits=3), " ms")
    flush(stdout)
    r
end

function main()
    length(ARGS) == 2 || error("Usage: bench_matrix_convection_gpu.jl results.toml native|split")
    output, mode = ARGS
    mode in ("native", "split") || error("Mode must be native or split")
    expected = get(ENV, "ATMOSTR_MATRIX_GPU_NAME", "A100")
    isempty(expected) && error("ATMOSTR_MATRIX_GPU_NAME must not be empty")
    CUDA.functional() || error("A functional CUDA GPU is required")
    occursin(expected, CUDA.name(CUDA.device())) || error("Select the requested $expected GPU")
    CUDA.allowscalar(false)
    nc = parse(Int, get(ENV,"ATMOSTR_MATRIX_BENCH_COLUMNS","4096"))
    nc >= 8 || error("At least eight columns are needed by the reference checks")
    results = Dict{String,Any}(
        "device"=>CUDA.name(CUDA.device()), "compute_capability"=>string(CUDA.capability(CUDA.device())),
        "julia_version"=>string(VERSION), "cuda_version"=>string(pkgversion(CUDA)),
        "cuda_runtime_version"=>string(CUDA.runtime_version()),
        "cuda_driver_version"=>string(CUDA.driver_version()),
        "kernelabstractions_version"=>string(pkgversion(KA)),
        "revision"=>get(ENV,"ATMOSTR_MATRIX_BENCH_REVISION","unspecified"),
        "scope"=>"Synthetic Float32 face-indexed columns; all columns convect; no transfer or I/O timing",
        "timing"=>"Median of nine warmed single-step samples; same RHS restored outside timing",
        "benchmark"=>Dict{String,Any}[])
    levels = parse.(Int,split(get(ENV,"ATMOSTR_MATRIX_BENCH_LEVELS","60,85"),','))
    tracers = parse.(Int,split(get(ENV,"ATMOSTR_MATRIX_BENCH_TRACERS","1,6,7,12,32,65"),','))
    all(n -> 8 <= n <= 85,levels) || error("Benchmark levels must be between 8 and 85")
    all(>(0),tracers) || error("Benchmark tracer counts must be positive")
    for nz in levels, downdrafts in (false,true), nt in tracers
        push!(results["benchmark"], measure(nc,nz,nt,downdrafts,mode))
        open(output,"w") do io
            TOML.print(io,results)
        end
        GC.gc()
        CUDA.reclaim()
    end
end

main()
