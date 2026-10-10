#!/usr/bin/env julia
# ===========================================================================
# Plan 15 chemistry overhead bench
#
# Measures `step!(model, dt)` per-step time with and without a non-trivial
# chemistry operator, at production Nt values. Expected: < 5% overhead on
# CPU and GPU (one KA kernel launch + a single pass over tracers_raw).
#
# Usage:
#   julia --project=. scripts/benchmarks/heritage/bench_chemistry_overhead.jl [size] [backend]
# where
#   size    ∈ {small, medium, large}   (default: medium)
#   backend ∈ {cpu, gpu}               (default: cpu)
# ===========================================================================

using AtmosTransport
using AtmosTransport: LatLonMesh, AtmosGrid, HybridSigmaPressure, CPU,
    cell_areas_by_latitude, gravity, reference_pressure, level_thickness,
    CellState, StructuredFaceFluxState, DryBasis, AdvectionWorkspace,
    UpwindScheme, SlopesScheme, MonotoneLimiter, TransportModel,
    ExponentialDecay, NoChemistry
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

# Rn-222 decay constant (ln(2)/half_life) for plausible production workload.
const RN222_HALF_LIFE = 330_350.4

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

function _to_backend(x, backend::Symbol)
    backend === :gpu ? CUDA.CuArray(x) : copy(x)
end

function run_bench(::Type{FT}, Nt::Int, scheme, backend::Symbol, chem_on::Bool,
                   cfg) where {FT}
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

    chemistry = chem_on ?
        ExponentialDecay(FT; NamedTuple{tr_syms}(ntuple(_ -> RN222_HALF_LIFE, Nt))...) :
        NoChemistry()

    model = TransportModel(state, fluxes, grid, scheme; chemistry = chemistry)

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
    @printf("Chemistry overhead bench — %s / %s (grid=%dx%dx%d, steps=%d)\n",
            SIZE, BACKEND, DIMS.Nx, DIMS.Ny, DIMS.Nz, DIMS.n_steps)
    @printf("%-10s %-8s %4s %12s %12s %12s %10s\n",
            "dtype", "scheme", "Nt", "no-chem(ms)", "chem(ms)", "Δ(ms)", "Δ%")
    println("-"^80)

    ft_list = BACKEND === :gpu ? (Float32,) : (Float64, Float32)
    scheme_list = [
        ("upwind", UpwindScheme()),
        ("slopes", SlopesScheme(MonotoneLimiter())),
    ]
    nt_list = [5, 10, 30]

    for FT in ft_list, (sname, sch) in scheme_list, Nt in nt_list
        med_no,  mad_no  = run_bench(FT, Nt, sch, BACKEND, false, DIMS)
        med_yes, mad_yes = run_bench(FT, Nt, sch, BACKEND, true,  DIMS)
        dms = med_yes - med_no
        pct = 100 * dms / med_no
        @printf("%-10s %-8s %4d %12.3f %12.3f %12.3f %9.2f%%\n",
                string(FT), sname, Nt, med_no, med_yes, dms, pct)
    end
end

main()
