#!/usr/bin/env julia
# ===========================================================================
# Plan 17 emissions overhead benchmark
#
# Measures `step!(model, dt)` per-step time with and without a non-trivial
# surface-emissions operator, at production Nt / grid sizes. Mirrors the
# shape of bench_chemistry_overhead.jl and bench_diffusion_overhead.jl so
# plan 17's perf numbers read as a natural extension of the earlier sets.
#
# Expected: <5% overhead on CPU and GPU at typical CATRINE counts
# (1-5 emitting tracers, one kernel launch per emitter). When `emissions =
# NoSurfaceFlux()` the dead-branch dispatch is zero FP work, so the
# reference run has bit-exact pre-17 behaviour and the delta is entirely
# the N_emitting-tracer V(dt/2) + kernel-launch + V(dt/2) overhead.
#
# Usage:
#   julia --project=. scripts/benchmarks/heritage/bench_emissions_overhead.jl [size] [backend]
# where
#   size    ∈ {small, medium, large}   (default: medium)
#   backend ∈ {cpu, gpu}               (default: cpu)
# ===========================================================================

using AtmosTransport
using AtmosTransport: LatLonMesh, AtmosGrid, HybridSigmaPressure, CPU,
    cell_areas_by_latitude, gravity, reference_pressure, level_thickness,
    CellState, StructuredFaceFluxState, DryBasis, AdvectionWorkspace,
    UpwindScheme, SlopesScheme, MonotoneLimiter, TransportModel,
    NoSurfaceFlux, SurfaceFluxOperator, SurfaceFluxSource
using Statistics
using Printf

const POSITIONAL = filter(a -> !startswith(a, "--"), ARGS)
const SIZE    = length(POSITIONAL) >= 1 ? Symbol(POSITIONAL[1]) : :medium
const BACKEND = length(POSITIONAL) >= 2 ? Symbol(POSITIONAL[2]) : :cpu

if BACKEND === :gpu
    using CUDA
    CUDA.functional() || error("CUDA requested but not functional")
end

const DIMS = Dict(
    :small  => (Nx =  72, Ny =  36, Nz =  4, n_steps = 50),
    :medium => (Nx = 288, Ny = 144, Nz = 32, n_steps = 40),
    :large  => (Nx = 576, Ny = 288, Nz = 72, n_steps = 40),
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

    cfl = FT(0.3)
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

_to_backend(x, backend::Symbol) = backend === :gpu ? CUDA.CuArray(x) : copy(x)

"""
Build a `SurfaceFluxOperator` emitting into the first `n_emitters` tracers
of `tracer_syms`. Rate = 1e-8 × sym_index (kg/s per cell) so the flux is
non-trivial across emitters.
"""
function build_emissions(::Type{FT}, tracer_syms, n_emitters, Nx, Ny,
                         backend) where FT
    n_emitters = min(n_emitters, length(tracer_syms))
    sources = [SurfaceFluxSource(
        tracer_syms[t],
        _to_backend(fill(FT(1e-8 * t), Nx, Ny), backend))
        for t in 1:n_emitters]
    return SurfaceFluxOperator(sources...)
end

function run_bench(::Type{FT}, Nt::Int, n_emitters::Int, scheme,
                   backend::Symbol, emissions_on::Bool, cfg) where {FT}
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

    emissions = emissions_on ?
        build_emissions(FT, tr_syms, n_emitters, cfg.Nx, cfg.Ny, backend) :
        NoSurfaceFlux()

    model = TransportModel(state, fluxes, grid, scheme; emissions = emissions)

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
    @printf("Emissions overhead bench — %s / %s (grid=%dx%dx%d, steps=%d)\n",
            SIZE, BACKEND, DIMS.Nx, DIMS.Ny, DIMS.Nz, DIMS.n_steps)
    @printf("%-10s %-8s %4s %4s %12s %12s %12s %10s\n",
            "dtype", "scheme", "Nt", "NE", "no-em(ms)", "em(ms)", "Δ(ms)", "Δ%")
    println("-"^92)

    ft_list = BACKEND === :gpu ? (Float32,) : (Float64, Float32)
    scheme_list = [
        ("upwind", UpwindScheme()),
        ("slopes", SlopesScheme(MonotoneLimiter())),
    ]
    # (Nt, n_emitters) combos: typical CATRINE configs with 1-3
    # emitting tracers out of 5-30 advected.
    nt_combos = [(5, 1), (5, 3), (10, 3), (30, 5)]

    for FT in ft_list, (sname, sch) in scheme_list, (Nt, NE) in nt_combos
        med_no,  _ = run_bench(FT, Nt, NE, sch, BACKEND, false, DIMS)
        med_yes, _ = run_bench(FT, Nt, NE, sch, BACKEND, true,  DIMS)
        dms = med_yes - med_no
        pct = 100 * dms / med_no
        @printf("%-10s %-8s %4d %4d %12.3f %12.3f %12.3f %9.2f%%\n",
                string(FT), sname, Nt, NE, med_no, med_yes, dms, pct)
    end
end

main()
