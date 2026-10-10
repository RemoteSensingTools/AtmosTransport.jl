#!/usr/bin/env julia
# ===========================================================================
# Plan 16b diffusion overhead bench
#
# Measures `step!(model, dt)` per-step time at four points along the
# Kz-source spectrum:
#
#   1. NoDiffusion                                   — baseline (dead branch)
#   2. ImplicitVerticalDiffusion{ConstantField}      — scalar Kz
#   3. ImplicitVerticalDiffusion{ProfileKzField}     — per-level profile
#   4. ImplicitVerticalDiffusion{PreComputedKzField} — full 3D snapshot
#
# Plan 16 target: ≤30% overhead on GPU at Nt=10, cfl=0.4 vs. NoDiffusion.
# This script also resolves the ProfileKzField GPU-dispatch question (plan
# pitfall 11) — as of Commit 6, `ProfileKzField` is Adapt-compatible, so
# its backing vector is moved to the device at kernel-launch time.
#
# Usage:
#   julia --project=. scripts/benchmarks/heritage/bench_diffusion_overhead.jl [size] [backend]
# where
#   size    ∈ {small, medium, large}   (default: medium)
#   backend ∈ {cpu, gpu}               (default: cpu)
#
# Measurement protocol:
# - 3 warm-up steps (not timed)
# - n_steps timed per configuration
# - GPU: CUDA.synchronize before and after each step (wall-clock with sync);
#   this matches the chemistry bench pattern and avoids CUDA.@elapsed pitfalls.
# - Report median and median-absolute-deviation per configuration
# ===========================================================================

using AtmosTransport
using AtmosTransport: LatLonMesh, AtmosGrid, HybridSigmaPressure, CPU,
    cell_areas_by_latitude, gravity, reference_pressure, level_thickness,
    CellState, StructuredFaceFluxState, DryBasis, AdvectionWorkspace,
    UpwindScheme, SlopesScheme, MonotoneLimiter, TransportModel,
    ConstantField, ProfileKzField, PreComputedKzField,
    NoDiffusion, ImplicitVerticalDiffusion, with_diffusion
using Statistics
using Printf
using Adapt

const POSITIONAL = filter(a -> !startswith(a, "--"), ARGS)
const SIZE    = length(POSITIONAL) >= 1 ? Symbol(POSITIONAL[1]) : :medium
const BACKEND = length(POSITIONAL) >= 2 ? Symbol(POSITIONAL[2]) : :cpu

if BACKEND === :gpu
    using CUDA
    CUDA.functional() || error("CUDA requested but not functional")
end

const DIMS = Dict(
    :small  => (Nx =  72, Ny =  36, Nz =  4, n_steps = 30),
    :medium => (Nx = 288, Ny = 144, Nz = 32, n_steps = 25),
    :large  => (Nx = 576, Ny = 288, Nz = 72, n_steps = 20),
)[SIZE]

median_abs_dev(v::AbstractVector) = median(abs.(v .- median(v)))

function build_problem(::Type{FT}, Nx, Ny, Nz, Nt) where {FT}
    mesh = LatLonMesh(; Nx = Nx, Ny = Ny, FT = FT)
    A_ifc = zeros(FT, Nz + 1)
    B_ifc = FT.(collect(range(0.0, 1.0, length = Nz + 1)))
    vc = HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosGrid(mesh, vc, CPU(); FT = FT)

    g  = gravity(grid)
    ps = reference_pressure(grid)
    areas = cell_areas_by_latitude(mesh)

    m = zeros(FT, Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        m[i, j, k] = FT(level_thickness(vc, k, ps)) * areas[j] / g
    end
    m_min = minimum(m)

    tracer_syms = ntuple(t -> Symbol("tr$t"), Nt)
    rms = map(t -> fill(FT(1e-4) * t, Nx, Ny, Nz), ntuple(identity, Nt))

    cfl = FT(0.4)  # plan 16 target
    am = zeros(FT, Nx + 1, Ny, Nz)
    bm = zeros(FT, Nx, Ny + 1, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:(Nx + 1)
        lat = FT(-90) + (FT(j) - FT(0.5)) * (FT(180) / FT(Ny))
        am[i, j, k] = cfl * m_min * cos(lat * FT(pi) / FT(180))
    end
    am[:, 1, :]  .= zero(FT)
    am[:, Ny, :] .= zero(FT)
    cm = zeros(FT, Nx, Ny, Nz + 1)

    return grid, m, tracer_syms, rms, am, bm, cm
end

function _to_backend(x, backend::Symbol)
    backend === :gpu ? CUDA.CuArray(x) : copy(x)
end

# Build a diffusion operator of the requested Kz variety (or NoDiffusion).
# Sizes refer to the model's Nx, Ny, Nz — needed only for PreComputedKzField.
function _make_diff_op(::Type{FT}, kind::Symbol, Nx, Ny, Nz, backend) where {FT}
    kind === :none && return NoDiffusion()
    if kind === :constant
        return ImplicitVerticalDiffusion(;
            kz_field = ConstantField{FT, 3}(FT(1.0)))
    elseif kind === :profile
        prof = FT.(0.5 .+ 1.5 .* exp.(-(range(FT(0), FT(Nz-1); length=Nz) ./ FT(Nz/3)).^2))
        pf = ProfileKzField(collect(prof))
        pf = backend === :gpu ? Adapt.adapt(CUDA.CuArray, pf) : pf
        return ImplicitVerticalDiffusion(; kz_field = pf)
    elseif kind === :precomputed
        kz_cpu = fill(FT(1.0), Nx, Ny, Nz)
        kz = _to_backend(kz_cpu, backend)
        return ImplicitVerticalDiffusion(; kz_field = PreComputedKzField(kz))
    else
        error("unknown diffusion kind: $kind")
    end
end

function run_bench(::Type{FT}, Nt::Int, scheme, backend::Symbol,
                   diff_kind::Symbol, cfg) where {FT}
    grid, m_cpu, tr_syms, rms_cpu, am_cpu, bm_cpu, cm_cpu =
        build_problem(FT, cfg.Nx, cfg.Ny, cfg.Nz, Nt)

    m      = _to_backend(m_cpu,  backend)
    am     = _to_backend(am_cpu, backend)
    bm     = _to_backend(bm_cpu, backend)
    cm     = _to_backend(cm_cpu, backend)
    tracer_vals = ntuple(t -> _to_backend(rms_cpu[t], backend), Nt)

    tracers = NamedTuple{tr_syms}(tracer_vals)
    state   = CellState(DryBasis, m; tracers...)
    fluxes  = StructuredFaceFluxState(am, bm, cm)

    diff_op = _make_diff_op(FT, diff_kind, cfg.Nx, cfg.Ny, cfg.Nz, backend)

    model = TransportModel(state, fluxes, grid, scheme; diffusion = diff_op)

    # Layer thickness must be populated before the first step — the operator
    # reads it directly from the workspace. Uniform 100 m is a reasonable
    # stand-in; actual hydrostatic dz belongs to met-driver integration.
    if diff_kind !== :none
        fill!(model.workspace.diffusion_ws.layer_thickness, FT(100.0))
    end

    # Warmup
    dt = FT(3600)
    for _ in 1:3
        step!(model, dt)
    end
    backend === :gpu && CUDA.synchronize()

    times_ms = Float64[]
    for _ in 1:cfg.n_steps
        if backend === :gpu
            CUDA.synchronize()
            t0 = time_ns()
            step!(model, dt)
            CUDA.synchronize()
            push!(times_ms, (time_ns() - t0) * 1e-6)
        else
            t = @elapsed step!(model, dt)
            push!(times_ms, t * 1e3)
        end
    end

    med = median(times_ms)
    mad = median_abs_dev(times_ms)
    return med, mad
end

function main()
    @printf("Diffusion overhead bench — %s / %s (grid=%dx%dx%d, steps=%d, cfl=0.4)\n",
            SIZE, BACKEND, DIMS.Nx, DIMS.Ny, DIMS.Nz, DIMS.n_steps)
    @printf("%-8s %-8s %4s %12s %12s %12s %12s %10s\n",
            "dtype", "scheme", "Nt",
            "baseline", "const(ms)", "profile(ms)", "precomp(ms)", "max-Δ%")
    println("-"^90)

    ft_list = BACKEND === :gpu ? (Float32,) : (Float64, Float32)
    scheme_list = [
        ("upwind", UpwindScheme()),
        ("slopes", SlopesScheme(MonotoneLimiter())),
    ]
    nt_list = [5, 10]

    for FT in ft_list, (sname, sch) in scheme_list, Nt in nt_list
        med_none,   _ = run_bench(FT, Nt, sch, BACKEND, :none, DIMS)
        med_const,  _ = run_bench(FT, Nt, sch, BACKEND, :constant, DIMS)
        med_prof,   _ = run_bench(FT, Nt, sch, BACKEND, :profile, DIMS)
        med_pre,    _ = run_bench(FT, Nt, sch, BACKEND, :precomputed, DIMS)
        max_dms = max(med_const, med_prof, med_pre) - med_none
        max_pct = 100 * max_dms / med_none
        @printf("%-8s %-8s %4d %12.3f %12.3f %12.3f %12.3f %9.2f%%\n",
                string(FT), sname, Nt, med_none, med_const, med_prof,
                med_pre, max_pct)
    end
end

main()
