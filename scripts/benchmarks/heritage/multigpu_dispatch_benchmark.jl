#!/usr/bin/env julia
# ===========================================================================
# Multi-GPU Dispatch Benchmark
#
# Tests different strategies for dispatching 6 cubed-sphere panels across
# 1 or 2 GPUs. Emulates our transport pipeline: heavy stencil (advection),
# boundary copy (halo exchange), column operation (physics).
#
# Usage:
#   julia --threads=4 --project=. scripts/benchmarks/heritage/multigpu_dispatch_benchmark.jl [c180|c720]
# ===========================================================================

using CUDA
using KernelAbstractions
using KernelAbstractions: synchronize
using Printf
using Statistics

# ─────────────────────────────────────────────────────────────────────────────
# Kernels — synthetic but matching our compute profile
# ─────────────────────────────────────────────────────────────────────────────

@kernel function _stencil_kernel!(out, @Const(inp), Hp)
    i, j, k = @index(Global, NTuple)
    @inbounds out[Hp+i, Hp+j, k] = 0.5f0 * inp[Hp+i, Hp+j, k] +
        0.125f0 * (inp[Hp+i-1, Hp+j, k] + inp[Hp+i+1, Hp+j, k] +
                   inp[Hp+i, Hp+j-1, k] + inp[Hp+i, Hp+j+1, k])
end

@kernel function _column_kernel!(data, Hp)
    i, j = @index(Global, NTuple)
    s = 0.0f0
    @inbounds for k in 1:size(data, 3)
        s += data[Hp+i, Hp+j, k]
    end
    @inbounds for k in 1:size(data, 3)
        data[Hp+i, Hp+j, k] += 0.001f0 * s / size(data, 3)
    end
end

@kernel function _copy_edge_kernel!(dst, @Const(src), N, Hp)
    i, k = @index(Global, NTuple)
    @inbounds dst[Hp+i, 1, k] = src[Hp+i, Hp+N, k]
end

# ─────────────────────────────────────────────────────────────────────────────
# Helper: launch kernels for one "window" under a given strategy
# ─────────────────────────────────────────────────────────────────────────────

function launch_stencil!(out, inp, Nc, Nz, Hp, backend)
    k! = _stencil_kernel!(backend, 256)
    k!(out, inp, Hp; ndrange=(Nc, Nc, Nz))
end

function launch_column!(data, Nc, Hp, backend)
    k! = _column_kernel!(backend, 256)
    k!(data, Hp; ndrange=(Nc, Nc))
end

function launch_halo!(dst, src, Nc, Nz, Hp, backend)
    k! = _copy_edge_kernel!(backend, 256)
    k!(dst, src, Nc, Hp; ndrange=(Nc, Nz))
end

# ─────────────────────────────────────────────────────────────────────────────
# Allocation helpers
# ─────────────────────────────────────────────────────────────────────────────

function allocate_panels(Nc, Nz, Hp, gpu_ids)
    panels_a = NTuple{6, CuArray{Float32, 3}}(ntuple(p -> begin
        CUDA.device!(gpu_ids[p])
        CUDA.rand(Float32, Nc + 2Hp, Nc + 2Hp, Nz)
    end, 6))
    panels_b = NTuple{6, CuArray{Float32, 3}}(ntuple(p -> begin
        CUDA.device!(gpu_ids[p])
        CUDA.zeros(Float32, Nc + 2Hp, Nc + 2Hp, Nz)
    end, 6))
    CUDA.device!(0)
    return panels_a, panels_b
end

# ─────────────────────────────────────────────────────────────────────────────
# Strategy A: Single GPU, sync after each kernel
# ─────────────────────────────────────────────────────────────────────────────

function run_strategy_A!(panels_a, panels_b, Nc, Nz, Hp, n_windows)
    for _ in 1:n_windows
        # Phase 1: 3 stencil passes per panel (advection-like)
        for _ in 1:3
            for p in 1:6
                backend = get_backend(panels_a[p])
                launch_stencil!(panels_b[p], panels_a[p], Nc, Nz, Hp, backend)
                synchronize(backend)
            end
            panels_a, panels_b = panels_b, panels_a
        end
        # Phase 2: halo exchange
        for p in 1:6
            q = mod1(p + 1, 6)
            backend = get_backend(panels_a[p])
            launch_halo!(panels_a[p], panels_a[q], Nc, Nz, Hp, backend)
            synchronize(backend)
        end
        # Phase 3: column physics
        for p in 1:6
            backend = get_backend(panels_a[p])
            launch_column!(panels_a[p], Nc, Hp, backend)
            synchronize(backend)
        end
    end
    return panels_a
end

# ─────────────────────────────────────────────────────────────────────────────
# Strategy B: Single GPU, deferred sync (launch all, sync once)
# ─────────────────────────────────────────────────────────────────────────────

function run_strategy_B!(panels_a, panels_b, Nc, Nz, Hp, n_windows)
    backend = get_backend(panels_a[1])
    for _ in 1:n_windows
        for _ in 1:3
            for p in 1:6
                launch_stencil!(panels_b[p], panels_a[p], Nc, Nz, Hp, backend)
            end
            synchronize(backend)
            panels_a, panels_b = panels_b, panels_a
        end
        for p in 1:6
            q = mod1(p + 1, 6)
            launch_halo!(panels_a[p], panels_a[q], Nc, Nz, Hp, backend)
        end
        synchronize(backend)
        for p in 1:6
            launch_column!(panels_a[p], Nc, Hp, backend)
        end
        synchronize(backend)
    end
    return panels_a
end

# ─────────────────────────────────────────────────────────────────────────────
# Strategy C: Two GPUs, Threads.@spawn per GPU
# ─────────────────────────────────────────────────────────────────────────────

function run_strategy_C!(panels_a, panels_b, Nc, Nz, Hp, n_windows)
    gpu_panels = (1:3, 4:6)  # panels per GPU
    for _ in 1:n_windows
        for _ in 1:3
            tasks = map(0:1) do g
                Threads.@spawn begin
                    CUDA.device!(g)
                    for p in gpu_panels[g+1]
                        backend = get_backend(panels_a[p])
                        launch_stencil!(panels_b[p], panels_a[p], Nc, Nz, Hp, backend)
                    end
                    CUDA.synchronize()
                end
            end
            for t in tasks; wait(t); end
            panels_a, panels_b = panels_b, panels_a
        end
        # Halo — sequential (cross-GPU edges need staging)
        for p in 1:6
            q = mod1(p + 1, 6)
            backend = get_backend(panels_a[p])
            launch_halo!(panels_a[p], panels_a[q], Nc, Nz, Hp, backend)
            synchronize(backend)
        end
        # Physics
        tasks = map(0:1) do g
            Threads.@spawn begin
                CUDA.device!(g)
                for p in gpu_panels[g+1]
                    backend = get_backend(panels_a[p])
                    launch_column!(panels_a[p], Nc, Hp, backend)
                end
                CUDA.synchronize()
            end
        end
        for t in tasks; wait(t); end
    end
    return panels_a
end

# ─────────────────────────────────────────────────────────────────────────────
# Strategy D: Two GPUs, launch from main thread (KA auto-routes), deferred sync
# ─────────────────────────────────────────────────────────────────────────────

function run_strategy_D!(panels_a, panels_b, Nc, Nz, Hp, n_windows)
    for _ in 1:n_windows
        for _ in 1:3
            # Launch all 6 — KA routes to correct GPU based on array ownership
            for p in 1:6
                backend = get_backend(panels_a[p])
                launch_stencil!(panels_b[p], panels_a[p], Nc, Nz, Hp, backend)
            end
            # Sync both GPUs
            CUDA.device!(0); CUDA.synchronize()
            CUDA.device!(1); CUDA.synchronize()
            panels_a, panels_b = panels_b, panels_a
        end
        for p in 1:6
            q = mod1(p + 1, 6)
            backend = get_backend(panels_a[p])
            launch_halo!(panels_a[p], panels_a[q], Nc, Nz, Hp, backend)
        end
        CUDA.device!(0); CUDA.synchronize()
        CUDA.device!(1); CUDA.synchronize()
        for p in 1:6
            backend = get_backend(panels_a[p])
            launch_column!(panels_a[p], Nc, Hp, backend)
        end
        CUDA.device!(0); CUDA.synchronize()
        CUDA.device!(1); CUDA.synchronize()
    end
    CUDA.device!(0)
    return panels_a
end

# ─────────────────────────────────────────────────────────────────────────────
# Strategy E: Two GPUs, explicit CUDA streams
# ─────────────────────────────────────────────────────────────────────────────

function run_strategy_E!(panels_a, panels_b, Nc, Nz, Hp, n_windows)
    # Create one stream per GPU
    CUDA.device!(0); s0 = CuStream()
    CUDA.device!(1); s1 = CuStream()
    CUDA.device!(0)
    streams = (s0, s1)
    gpu_panels = (1:3, 4:6)

    for _ in 1:n_windows
        for _ in 1:3
            for g in 0:1
                CUDA.device!(g)
                for p in gpu_panels[g+1]
                    backend = get_backend(panels_a[p])
                    launch_stencil!(panels_b[p], panels_a[p], Nc, Nz, Hp, backend)
                end
            end
            CUDA.device!(0); CUDA.synchronize()
            CUDA.device!(1); CUDA.synchronize()
            panels_a, panels_b = panels_b, panels_a
        end
        for g in 0:1
            CUDA.device!(g)
            for p in gpu_panels[g+1]
                q = mod1(p + 1, 6)
                backend = get_backend(panels_a[p])
                launch_halo!(panels_a[p], panels_a[q], Nc, Nz, Hp, backend)
            end
        end
        CUDA.device!(0); CUDA.synchronize()
        CUDA.device!(1); CUDA.synchronize()
        for g in 0:1
            CUDA.device!(g)
            for p in gpu_panels[g+1]
                backend = get_backend(panels_a[p])
                launch_column!(panels_a[p], Nc, Hp, backend)
            end
        end
        CUDA.device!(0); CUDA.synchronize()
        CUDA.device!(1); CUDA.synchronize()
    end
    CUDA.device!(0)
    return panels_a
end

# ─────────────────────────────────────────────────────────────────────────────
# Strategy F: Two GPUs, Threads.@spawn + batched device switch (1 switch per phase)
# ─────────────────────────────────────────────────────────────────────────────

function run_strategy_F!(panels_a, panels_b, Nc, Nz, Hp, n_windows)
    gpu_panels = (1:3, 4:6)
    for _ in 1:n_windows
        for _ in 1:3
            # Launch both GPUs concurrently via threads
            t0 = Threads.@spawn begin
                CUDA.device!(0)
                for p in gpu_panels[1]
                    launch_stencil!(panels_b[p], panels_a[p], Nc, Nz, Hp, get_backend(panels_a[p]))
                end
                CUDA.synchronize()
            end
            # GPU 1 on main thread (avoids 2nd spawn overhead)
            CUDA.device!(1)
            for p in gpu_panels[2]
                launch_stencil!(panels_b[p], panels_a[p], Nc, Nz, Hp, get_backend(panels_a[p]))
            end
            CUDA.synchronize()
            wait(t0)
            panels_a, panels_b = panels_b, panels_a
        end
        # Halo — sequential
        for p in 1:6
            q = mod1(p + 1, 6)
            backend = get_backend(panels_a[p])
            launch_halo!(panels_a[p], panels_a[q], Nc, Nz, Hp, backend)
            synchronize(backend)
        end
        # Physics — concurrent
        t0 = Threads.@spawn begin
            CUDA.device!(0)
            for p in gpu_panels[1]
                launch_column!(panels_a[p], Nc, Hp, get_backend(panels_a[p]))
            end
            CUDA.synchronize()
        end
        CUDA.device!(1)
        for p in gpu_panels[2]
            launch_column!(panels_a[p], Nc, Hp, get_backend(panels_a[p]))
        end
        CUDA.synchronize()
        wait(t0)
    end
    CUDA.device!(0)
    return panels_a
end

# ─────────────────────────────────────────────────────────────────────────────
# Strategy G: Fused kernel — single launch for all panels on each GPU
# ─────────────────────────────────────────────────────────────────────────────

@kernel function _fused_stencil_kernel!(out1, out2, out3,
                                        @Const(inp1), @Const(inp2), @Const(inp3), Hp)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        out1[Hp+i,Hp+j,k] = 0.5f0*inp1[Hp+i,Hp+j,k] + 0.125f0*(inp1[Hp+i-1,Hp+j,k]+inp1[Hp+i+1,Hp+j,k]+inp1[Hp+i,Hp+j-1,k]+inp1[Hp+i,Hp+j+1,k])
        out2[Hp+i,Hp+j,k] = 0.5f0*inp2[Hp+i,Hp+j,k] + 0.125f0*(inp2[Hp+i-1,Hp+j,k]+inp2[Hp+i+1,Hp+j,k]+inp2[Hp+i,Hp+j-1,k]+inp2[Hp+i,Hp+j+1,k])
        out3[Hp+i,Hp+j,k] = 0.5f0*inp3[Hp+i,Hp+j,k] + 0.125f0*(inp3[Hp+i-1,Hp+j,k]+inp3[Hp+i+1,Hp+j,k]+inp3[Hp+i,Hp+j-1,k]+inp3[Hp+i,Hp+j+1,k])
    end
end

@kernel function _fused_column_kernel!(d1, d2, d3, Hp)
    i, j = @index(Global, NTuple)
    @inbounds for d in (d1, d2, d3)
        s = 0.0f0
        for k in 1:size(d, 3)
            s += d[Hp+i, Hp+j, k]
        end
        for k in 1:size(d, 3)
            d[Hp+i, Hp+j, k] += 0.001f0 * s / size(d, 3)
        end
    end
end

function run_strategy_G!(panels_a, panels_b, Nc, Nz, Hp, n_windows)
    gpu_panels = (1:3, 4:6)
    for _ in 1:n_windows
        for _ in 1:3
            t0 = Threads.@spawn begin
                CUDA.device!(0)
                backend = get_backend(panels_a[1])
                k! = _fused_stencil_kernel!(backend, 256)
                k!(panels_b[1], panels_b[2], panels_b[3],
                   panels_a[1], panels_a[2], panels_a[3], Hp; ndrange=(Nc, Nc, Nz))
                CUDA.synchronize()
            end
            CUDA.device!(1)
            backend = get_backend(panels_a[4])
            k! = _fused_stencil_kernel!(backend, 256)
            k!(panels_b[4], panels_b[5], panels_b[6],
               panels_a[4], panels_a[5], panels_a[6], Hp; ndrange=(Nc, Nc, Nz))
            CUDA.synchronize()
            wait(t0)
            panels_a, panels_b = panels_b, panels_a
        end
        # Halo — sequential
        for p in 1:6
            q = mod1(p + 1, 6)
            backend = get_backend(panels_a[p])
            launch_halo!(panels_a[p], panels_a[q], Nc, Nz, Hp, backend)
            synchronize(backend)
        end
        # Fused column physics
        t0 = Threads.@spawn begin
            CUDA.device!(0)
            backend = get_backend(panels_a[1])
            k! = _fused_column_kernel!(backend, 256)
            k!(panels_a[1], panels_a[2], panels_a[3], Hp; ndrange=(Nc, Nc))
            CUDA.synchronize()
        end
        CUDA.device!(1)
        backend = get_backend(panels_a[4])
        k! = _fused_column_kernel!(backend, 256)
        k!(panels_a[4], panels_a[5], panels_a[6], Hp; ndrange=(Nc, Nc))
        CUDA.synchronize()
        wait(t0)
    end
    CUDA.device!(0)
    return panels_a
end

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark harness
# ─────────────────────────────────────────────────────────────────────────────

function run_benchmark(; Nc=180, Nz=72, Hp=3, n_windows=20, n_warmup=3)
    n_gpus = length(CUDA.devices())
    @info "Multi-GPU Dispatch Benchmark" Nc Nz Hp n_windows n_warmup n_gpus
    @info "GPUs: $(join([CUDA.name(CUDA.device!(i)) for i in 0:n_gpus-1], ", "))"
    CUDA.device!(0)

    strategies = []

    # --- Strategy A: 1GPU sequential sync ---
    gpu1 = ntuple(_ -> 0, 6)
    a, b = allocate_panels(Nc, Nz, Hp, gpu1)
    run_strategy_A!(a, b, Nc, Nz, Hp, n_warmup)  # warmup
    CUDA.device!(0); CUDA.synchronize()
    t = @elapsed run_strategy_A!(a, b, Nc, Nz, Hp, n_windows)
    push!(strategies, ("A: 1GPU seq sync", t))
    @info @sprintf("A: 1GPU seq sync       %8.3fs  %7.1f ms/win", t, t/n_windows*1000)

    # --- Strategy B: 1GPU deferred sync ---
    a, b = allocate_panels(Nc, Nz, Hp, gpu1)
    run_strategy_B!(a, b, Nc, Nz, Hp, n_warmup)
    CUDA.device!(0); CUDA.synchronize()
    t = @elapsed run_strategy_B!(a, b, Nc, Nz, Hp, n_windows)
    push!(strategies, ("B: 1GPU deferred sync", t))
    @info @sprintf("B: 1GPU deferred sync  %8.3fs  %7.1f ms/win", t, t/n_windows*1000)

    if n_gpus >= 2
        gpu2 = (0, 0, 0, 1, 1, 1)

        # --- Strategy C: 2GPU Threads.@spawn ---
        a, b = allocate_panels(Nc, Nz, Hp, gpu2)
        run_strategy_C!(a, b, Nc, Nz, Hp, n_warmup)
        CUDA.device!(0); CUDA.synchronize()
        CUDA.device!(1); CUDA.synchronize()
        t = @elapsed run_strategy_C!(a, b, Nc, Nz, Hp, n_windows)
        push!(strategies, ("C: 2GPU threads", t))
        @info @sprintf("C: 2GPU threads        %8.3fs  %7.1f ms/win", t, t/n_windows*1000)

        # --- Strategy D: 2GPU main-thread KA auto-route ---
        a, b = allocate_panels(Nc, Nz, Hp, gpu2)
        run_strategy_D!(a, b, Nc, Nz, Hp, n_warmup)
        CUDA.device!(0); CUDA.synchronize()
        CUDA.device!(1); CUDA.synchronize()
        t = @elapsed run_strategy_D!(a, b, Nc, Nz, Hp, n_windows)
        push!(strategies, ("D: 2GPU KA auto-route", t))
        @info @sprintf("D: 2GPU KA auto-route  %8.3fs  %7.1f ms/win", t, t/n_windows*1000)

        # --- Strategy E: 2GPU explicit device switch ---
        a, b = allocate_panels(Nc, Nz, Hp, gpu2)
        run_strategy_E!(a, b, Nc, Nz, Hp, n_warmup)
        CUDA.device!(0); CUDA.synchronize()
        CUDA.device!(1); CUDA.synchronize()
        t = @elapsed run_strategy_E!(a, b, Nc, Nz, Hp, n_windows)
        push!(strategies, ("E: 2GPU device switch", t))
        @info @sprintf("E: 2GPU device switch  %8.3fs  %7.1f ms/win", t, t/n_windows*1000)

        # --- Strategy F: 2GPU spawn+main hybrid ---
        a, b = allocate_panels(Nc, Nz, Hp, gpu2)
        run_strategy_F!(a, b, Nc, Nz, Hp, n_warmup)
        CUDA.device!(0); CUDA.synchronize()
        CUDA.device!(1); CUDA.synchronize()
        t = @elapsed run_strategy_F!(a, b, Nc, Nz, Hp, n_windows)
        push!(strategies, ("F: 2GPU spawn+main", t))
        @info @sprintf("F: 2GPU spawn+main     %8.3fs  %7.1f ms/win", t, t/n_windows*1000)

        # --- Strategy G: 2GPU fused kernel ---
        a, b = allocate_panels(Nc, Nz, Hp, gpu2)
        run_strategy_G!(a, b, Nc, Nz, Hp, n_warmup)
        CUDA.device!(0); CUDA.synchronize()
        CUDA.device!(1); CUDA.synchronize()
        t = @elapsed run_strategy_G!(a, b, Nc, Nz, Hp, n_windows)
        push!(strategies, ("G: 2GPU fused kernel", t))
        @info @sprintf("G: 2GPU fused kernel   %8.3fs  %7.1f ms/win", t, t/n_windows*1000)
    end

    # --- Summary ---
    baseline = strategies[1][2]
    println()
    println("=" ^ 72)
    @printf("Multi-GPU Dispatch Benchmark (Nc=%d, %d windows, %d× %s)\n",
            Nc, n_windows, n_gpus, CUDA.name(CUDA.device!(0)))
    println("=" ^ 72)
    @printf("%-25s │ %8s │ %10s │ %7s\n", "Strategy", "Total(s)", "ms/win", "Speedup")
    println("─" ^ 25, "─┼─", "─" ^ 8, "─┼─", "─" ^ 10, "─┼─", "─" ^ 7)
    for (name, t) in strategies
        @printf("%-25s │ %8.3f │ %10.1f │ %6.2f×\n",
                name, t, t/n_windows*1000, baseline/t)
    end
    println("=" ^ 72)

    CUDA.device!(0)
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

size_arg = length(ARGS) >= 1 ? lowercase(ARGS[1]) : "c180"
if size_arg == "c720"
    Nc, n_windows = 720, 10
elseif size_arg == "c180"
    Nc, n_windows = 180, 40
elseif size_arg == "c90"
    Nc, n_windows = 90, 80
else
    error("Unknown size: $size_arg (use c90, c180, or c720)")
end

run_benchmark(; Nc, n_windows)
