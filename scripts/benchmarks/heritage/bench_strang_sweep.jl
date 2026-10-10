#!/usr/bin/env julia
# ===========================================================================
# Strang split per-step benchmark
#
# Measures per-step wall time for the two advection pipelines on a
# synthetic LatLon problem, so that Plan 14's per-tracer vs. multi-tracer
# decision has a direct measurement rather than a theoretical estimate.
#
# Reports median ± MAD (robust to outliers).
#
# Usage (basic):
#   julia --project=. scripts/benchmarks/heritage/bench_strang_sweep.jl [size] [backend] [--events]
# where
#   size    ∈ {small, medium, large}   (default: medium)
#   backend ∈ {cpu, gpu}               (default: cpu; gpu requires CUDA)
#   --events                           on GPU: also report CUDA-event-based
#                                      device time (vs. host wall time).
#
# Plan 14 matrix flags (comma-separated lists override defaults):
#   --mode=per-tracer,multi-tracer     (default: per-tracer)
#   --ntracers=1,5,10,30               (default: DIMS.Nt for size)
#   --cfl-limits=0.4,Inf               (default: 1.0 — no subcycling cap)
#   --schemes=upwind,slopes,ppm        (default: all three)
#   --dtype=f32,f64                    (default: Float64 on CPU, Float32 on GPU)
#
# Columns emitted (one line per grid point in the matrix):
#   mode  dtype  scheme  Nt  cfl   median(ms)  MAD(ms)  [host/cuda/Δ if --events]
#
# On GPU Float64 is omitted (L40S has no F64 units); pass --dtype=f64 to
# override if you're on A100/H100.
# ===========================================================================

using AtmosTransport
using AtmosTransport: AdvectionWorkspace, strang_split!, strang_split_mt!,
    UpwindScheme, SlopesScheme, PPMScheme, MonotoneLimiter,
    LatLonMesh, AtmosGrid, HybridSigmaPressure, CPU,
    cell_areas_by_latitude, gravity, reference_pressure, level_thickness,
    CellState, StructuredFaceFluxState, DryBasis
using Statistics
using Printf

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

const POSITIONAL = filter(a -> !startswith(a, "--"), ARGS)
const SIZE    = length(POSITIONAL) >= 1 ? Symbol(POSITIONAL[1]) : :medium
const BACKEND = length(POSITIONAL) >= 2 ? Symbol(POSITIONAL[2]) : :cpu
const EVENTS  = "--events" in ARGS

"Return the comma-separated value of `--key=...`, or `nothing` if not present."
function _flag_value(key::AbstractString)
    prefix = key * "="
    for a in ARGS
        if startswith(a, prefix)
            return a[lastindex(prefix)+1:end]
        end
    end
    return nothing
end

function _flag_list(key::AbstractString; default::Vector{String})
    v = _flag_value(key)
    return v === nothing ? default : String.(split(v, ","))
end

const MODE_LIST    = _flag_list("--mode";    default = ["per-tracer"])
const SCHEME_LIST  = _flag_list("--schemes"; default = ["upwind", "slopes", "ppm"])
const DTYPE_LIST_S = _flag_list("--dtype";   default = BACKEND === :gpu ? ["f32"] : ["f64", "f32"])

"Parse `--cfl-limits=0.4,Inf` into Float64 values; `Inf` disables subcycling."
function _parse_cfl_list()
    raw = _flag_value("--cfl-limits")
    return raw === nothing ? [1.0] :
           [lowercase(s) == "inf" ? Inf : parse(Float64, s) for s in split(raw, ",")]
end

const CFL_LIST = _parse_cfl_list()

"Parse `--ntracers=1,5,10,30`; fall back to DIMS default per size."
_parse_nt_list() = (v = _flag_value("--ntracers");
                    v === nothing ? nothing : parse.(Int, split(v, ",")))

const NT_OVERRIDE = _parse_nt_list()

# ---------------------------------------------------------------------------
# Problem presets
# ---------------------------------------------------------------------------

const DIMS = Dict(
    :small  => (Nx =  72, Ny =  36, Nz =  4, Nt = 1,  n_steps = 50),
    :medium => (Nx = 288, Ny = 144, Nz = 32, Nt = 5,  n_steps = 40),
    :large  => (Nx = 576, Ny = 288, Nz = 72, Nt = 10, n_steps = 60),
)[SIZE]

const NT_LIST = NT_OVERRIDE === nothing ? [DIMS.Nt] : NT_OVERRIDE

const SCHEME_TABLE = Dict(
    "upwind" => UpwindScheme(),
    "slopes" => SlopesScheme(MonotoneLimiter()),
    "ppm"    => PPMScheme(MonotoneLimiter()),
)

const DTYPE_TABLE = Dict(
    "f64" => Float64,
    "f32" => Float32,
)

# ---------------------------------------------------------------------------
# GPU setup — CUDA only loaded when backend=gpu
# ---------------------------------------------------------------------------

if BACKEND === :gpu
    using CUDA
    CUDA.functional() || error("CUDA requested but not functional")
    @eval gpu_event_time(f::Function) = CUDA.@elapsed f()
end

# ---------------------------------------------------------------------------
# Problem builder
# ---------------------------------------------------------------------------

"Median absolute deviation (robust scale)."
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

    rms = [similar(m) for _ in 1:Nt]
    @inbounds for t in 1:Nt
        chi0 = FT(100e-6) * t
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            lat_frac = FT(j - 1) / FT(Ny - 1)
            rms[t][i, j, k] = m[i, j, k] * (chi0 + FT(200e-6) * lat_frac)
        end
    end

    m_min = minimum(m)
    cfl = FT(0.3)
    am = zeros(FT, Nx + 1, Ny, Nz)
    bm = zeros(FT, Nx, Ny + 1, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:(Nx + 1)
        lat = FT(-90) + (FT(j) - FT(0.5)) * (FT(180) / FT(Ny))
        am[i, j, k] = cfl * m_min * cos(lat * FT(pi) / FT(180))
    end
    am[:, 1, :]  .= zero(FT)
    am[:, Ny, :] .= zero(FT)
    @inbounds for k in 1:Nz, j in 2:Ny, i in 1:Nx
        lat = FT(-90) + FT(j - 1) * (FT(180) / FT(Ny))
        bm[i, j, k] = cfl * m_min * FT(0.1) * sin(FT(2) * lat * FT(pi) / FT(180))
    end
    cm = zeros(FT, Nx, Ny, Nz + 1)

    return grid, m, rms, am, bm, cm
end

# ---------------------------------------------------------------------------
# Device adaptation — lift CPU arrays to GPU if needed
# ---------------------------------------------------------------------------

function _to_backend(x, backend::Symbol)
    backend === :gpu ? CUDA.CuArray(x) : copy(x)
end

# ---------------------------------------------------------------------------
# Per-tracer (NamedTuple) benchmark — current production path
# ---------------------------------------------------------------------------

function bench_per_tracer(::Type{FT}, scheme, grid, m_cpu, rms_cpu, am_cpu, bm_cpu, cm_cpu,
                          backend::Symbol, cfl_limit::Real, n_steps::Int, use_events::Bool) where {FT}
    Nt = length(rms_cpu)
    m      = _to_backend(m_cpu,  backend)
    am     = _to_backend(am_cpu, backend)
    bm     = _to_backend(bm_cpu, backend)
    cm     = _to_backend(cm_cpu, backend)
    tracer_vals = ntuple(t -> _to_backend(rms_cpu[t], backend), Nt)
    tracer_syms = ntuple(t -> Symbol("tr$t"), Nt)
    tracers = NamedTuple{tracer_syms}(tracer_vals)

    state   = CellState(DryBasis, m; tracers...)
    fluxes  = StructuredFaceFluxState(am, bm, cm)
    ws      = AdvectionWorkspace(state.air_mass; n_tracers = Nt)
    cfl     = FT(cfl_limit)

    for _ in 1:3
        strang_split!(state, fluxes, grid, scheme; workspace = ws, cfl_limit = cfl)
    end
    backend === :gpu && CUDA.synchronize()

    _time_loop(n_steps, backend, use_events) do
        strang_split!(state, fluxes, grid, scheme; workspace = ws, cfl_limit = cfl)
    end
end

# ---------------------------------------------------------------------------
# Multi-tracer (4D) benchmark — target pipeline
# ---------------------------------------------------------------------------

function bench_multi_tracer(::Type{FT}, scheme, grid, m_cpu, rms_cpu, am_cpu, bm_cpu, cm_cpu,
                            backend::Symbol, cfl_limit::Real, n_steps::Int, use_events::Bool) where {FT}
    Nt = length(rms_cpu)
    Nx, Ny, Nz = size(m_cpu)

    rm_4d_cpu = zeros(FT, Nx, Ny, Nz, Nt)
    @inbounds for t in 1:Nt
        rm_4d_cpu[:, :, :, t] .= rms_cpu[t]
    end

    m      = _to_backend(m_cpu,     backend)
    am     = _to_backend(am_cpu,    backend)
    bm     = _to_backend(bm_cpu,    backend)
    cm     = _to_backend(cm_cpu,    backend)
    rm_4d  = _to_backend(rm_4d_cpu, backend)

    ws = AdvectionWorkspace(m; n_tracers = Nt)
    cfl = FT(cfl_limit)

    for _ in 1:3
        strang_split_mt!(rm_4d, m, am, bm, cm, scheme, ws; cfl_limit = cfl)
    end
    backend === :gpu && CUDA.synchronize()

    _time_loop(n_steps, backend, use_events) do
        strang_split_mt!(rm_4d, m, am, bm, cm, scheme, ws; cfl_limit = cfl)
    end
end

# ---------------------------------------------------------------------------
# Timing loop — shared between both paths
# ---------------------------------------------------------------------------

function _time_loop(step_fn::Function, n_steps::Int, backend::Symbol, use_events::Bool)
    host_times_ms = Float64[]
    evt_times_ms  = Float64[]
    for _ in 1:n_steps
        if backend === :gpu
            CUDA.synchronize()
            t0 = time_ns()
            if use_events
                evt_s = gpu_event_time(step_fn)
                push!(evt_times_ms, evt_s * 1e3)
            else
                step_fn()
            end
            CUDA.synchronize()
            push!(host_times_ms, (time_ns() - t0) * 1e-6)
        else
            t = @elapsed step_fn()
            push!(host_times_ms, t * 1e3)
        end
    end

    med_host = median(host_times_ms)
    mad_host = median_abs_dev(host_times_ms)
    med_evt  = isempty(evt_times_ms) ? NaN : median(evt_times_ms)
    mad_evt  = isempty(evt_times_ms) ? NaN : median_abs_dev(evt_times_ms)
    return med_host, mad_host, med_evt, mad_evt
end

# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

function main()
    show_events = EVENTS && BACKEND === :gpu

    @printf("Strang sweep benchmark — %s / %s  (grid=%dx%dx%d, steps=%d%s)\n",
            SIZE, BACKEND, DIMS.Nx, DIMS.Ny, DIMS.Nz, DIMS.n_steps,
            show_events ? ", events" : "")
    @printf("modes=%s  Nt=%s  cfl_limits=%s  schemes=%s  dtypes=%s\n",
            join(MODE_LIST, ","), join(NT_LIST, ","),
            join(CFL_LIST, ","), join(SCHEME_LIST, ","), join(DTYPE_LIST_S, ","))

    if show_events
        @printf("%-12s %-6s %-8s %4s %6s %12s %12s %12s %12s %12s\n",
                "mode", "dtype", "scheme", "Nt", "cfl",
                "host(ms)", "host MAD", "cuda(ms)", "cuda MAD", "Δhost-cuda")
    else
        @printf("%-12s %-6s %-8s %4s %6s %12s %12s\n",
                "mode", "dtype", "scheme", "Nt", "cfl",
                "median(ms)", "MAD(ms)")
    end
    println("-"^90)

    for dtype_s in DTYPE_LIST_S
        FT = DTYPE_TABLE[dtype_s]
        for Nt in NT_LIST
            grid, m_cpu, rms_cpu, am_cpu, bm_cpu, cm_cpu =
                build_problem(FT, DIMS.Nx, DIMS.Ny, DIMS.Nz, Nt)

            for scheme_s in SCHEME_LIST
                scheme = SCHEME_TABLE[scheme_s]
                for cfl in CFL_LIST
                    for mode in MODE_LIST
                        med_host, mad_host, med_evt, mad_evt = if mode == "per-tracer"
                            bench_per_tracer(FT, scheme, grid, m_cpu, rms_cpu,
                                             am_cpu, bm_cpu, cm_cpu,
                                             BACKEND, cfl, DIMS.n_steps, EVENTS)
                        elseif mode == "multi-tracer"
                            bench_multi_tracer(FT, scheme, grid, m_cpu, rms_cpu,
                                               am_cpu, bm_cpu, cm_cpu,
                                               BACKEND, cfl, DIMS.n_steps, EVENTS)
                        else
                            error("Unknown mode: $mode (expected per-tracer or multi-tracer)")
                        end
                        cfl_s = isinf(cfl) ? "Inf" : @sprintf("%.2f", cfl)
                        if show_events
                            @printf("%-12s %-6s %-8s %4d %6s %12.3f %12.3f %12.3f %12.3f %12.3f\n",
                                    mode, string(FT), scheme_s, Nt, cfl_s,
                                    med_host, mad_host, med_evt, mad_evt, med_host - med_evt)
                        else
                            @printf("%-12s %-6s %-8s %4d %6s %12.3f %12.3f\n",
                                    mode, string(FT), scheme_s, Nt, cfl_s,
                                    med_host, mad_host)
                        end
                    end
                end
            end
        end
    end
end

main()
