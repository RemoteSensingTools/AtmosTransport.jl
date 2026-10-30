#!/usr/bin/env julia
# ===========================================================================
# Synthetic cubed-sphere advection benchmark
#
# Measures the advection-only cost of the cubed-sphere runtime path on
# synthetic fields. The benchmark intentionally avoids binary IO, snapshots,
# sources, diffusion, convection, and chemistry so the result is a roofline-style
# estimate for the transport kernels and halo exchange.
#
# Examples:
#
#   julia --project=. scripts/benchmarks/bench_cs_advection_gpu.jl
#
#   julia --project=. scripts/benchmarks/bench_cs_advection_gpu.jl \
#       --grid=C180 --levels=64 --tracers=4 --steps=20 --scheme=ppm --repeat=10
#
#   julia --project=. scripts/benchmarks/bench_cs_advection_gpu.jl \
#       --grid=C90,C180 --steps=8,16,24 --scheme=upwind,ppm --stream
#
# Columns:
#   time/window       median wall time for one met window
#   cell-sweeps/s     6 panels * Nc^2 * Nz * 6 palindrome sweeps * steps / s
#   tracer-sweeps/s   cell-sweeps/s * tracers
#   est GB/s          user-selected byte model divided by time
#
# The byte model is deliberately explicit because the real kernels reuse,
# reread, and write different fields depending on scheme/order. Use
# --bytes-per-tracer-sweep=128,256 to bracket likely memory traffic.
# ===========================================================================

using AtmosTransport
using AtmosTransport: CubedSphereMesh, GEOSNativePanelConvention,
    DryBasis, CubedSphereFaceFluxState,
    CSAdvectionWorkspace, CSLinRoodAdvectionWorkspace,
    CubedSphereState, PPMScheme, SlopesScheme, UpwindScheme,
    LinRoodPPMScheme, MonotoneLimiter, fill_panel_halos!,
    strang_split_cs_mt!
using Adapt
using Statistics
using Printf

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

function _flag_value(name::AbstractString, default::Union{Nothing, String} = nothing)
    prefix = name * "="
    for arg in ARGS
        arg == name && return "true"
        startswith(arg, prefix) && return arg[nextind(arg, lastindex(prefix)):end]
    end
    return default
end

_flag_bool(name::AbstractString) = _flag_value(name, "false") == "true"

function _split_strings(name::AbstractString, default::String)
    raw = _flag_value(name, default)
    return String.(split(raw, ","))
end

function _split_ints(name::AbstractString, default::String)
    return parse.(Int, _split_strings(name, default))
end

function _split_floats(name::AbstractString, default::String)
    return parse.(Float64, _split_strings(name, default))
end

function _parse_grid(s::AbstractString)
    t = uppercase(strip(s))
    startswith(t, "C") && return parse(Int, t[2:end])
    return parse(Int, t)
end

const GRIDS = _parse_grid.(_split_strings("--grid", "C180"))
const LEVELS = _split_ints("--levels", "64")
const TRACERS = _split_ints("--tracers", "4")
const STEPS = _split_ints("--steps", "20")
const SCHEMES = _split_strings("--scheme", "ppm")
const DTYPE_ARGS = _split_strings("--dtype", "f32")
const REPEAT = parse(Int, _flag_value("--repeat", "8"))
const WARMUP = parse(Int, _flag_value("--warmup", "3"))
const BACKEND = Symbol(lowercase(_flag_value("--backend", "gpu")))
const STREAM = _flag_bool("--stream")
const BYTE_MODELS = _split_floats("--bytes-per-tracer-sweep", "128,256")

const DTYPE_TABLE = Dict(
    "f32" => Float32,
    "float32" => Float32,
    "f64" => Float64,
    "float64" => Float64,
)

const SCHEME_TABLE = Dict{String, Any}(
    "upwind" => UpwindScheme(),
    "slopes" => SlopesScheme(MonotoneLimiter()),
    "ppm" => PPMScheme(MonotoneLimiter()),
    "linrood5" => LinRoodPPMScheme(5),
    "linrood7" => LinRoodPPMScheme(7),
)

median_abs_dev(v::AbstractVector) = median(abs.(v .- median(v)))

if BACKEND === :gpu
    using CUDA
    CUDA.functional() || error("CUDA requested but CUDA.functional() is false")
    CUDA.allowscalar(false)
    @eval function stream_bandwidth(::Type{FT}, backend; n::Int = 80_000_000,
                                    repeat::Int = 10, warmup::Int = 3) where {FT}
        backend === Val(:gpu) || return NaN
        x = CUDA.fill(FT(1), n)
        y = CUDA.fill(FT(2), n)
        z = CUDA.fill(FT(3), n)
        a = FT(1.0001)
        for _ in 1:warmup
            y .= a .* x .+ z
        end
        CUDA.synchronize()
        times = Float64[]
        for _ in 1:repeat
            t = CUDA.@elapsed y .= a .* x .+ z
            push!(times, t)
        end
        bytes = 3 * sizeof(FT) * n
        return bytes / median(times) / 1e9
    end
elseif BACKEND !== :cpu
    error("Unsupported --backend=$(BACKEND); expected gpu or cpu")
end

if BACKEND === :cpu
    stream_bandwidth(::Type, _backend; kwargs...) = NaN
end

function _dtype_from_arg(s)
    key = lowercase(string(s))
    haskey(DTYPE_TABLE, key) || error("Unsupported dtype $(s); use f32 or f64")
    return DTYPE_TABLE[key]
end

const DTYPES = _dtype_from_arg.(DTYPE_ARGS)

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

function _to_backend(x, ::Val{:cpu})
    return x
end

function _to_backend(x, ::Val{:gpu})
    return CUDA.CuArray(x)
end

_sync(::Val{:cpu}) = nothing
_sync(::Val{:gpu}) = CUDA.synchronize()

function _array_type(::Val{:cpu})
    return Array
end

function _array_type(::Val{:gpu})
    return CUDA.CuArray
end

# ---------------------------------------------------------------------------
# Synthetic problem construction
# ---------------------------------------------------------------------------

function _haloed_panel(::Type{FT}, Nc::Int, Hp::Int, Nz::Int) where {FT}
    N = Nc + 2Hp
    return zeros(FT, N, N, Nz)
end

function _fill_mass!(panels_m, mesh::CubedSphereMesh, ::Type{FT}) where {FT}
    Nc, Hp = mesh.Nc, mesh.Hp
    @inbounds for p in 1:6
        m = panels_m[p]
        for k in axes(m, 3), j in 1:Nc, i in 1:Nc
            ii = Hp + i
            jj = Hp + j
            x = FT(i - 1) / max(FT(Nc - 1), one(FT))
            y = FT(j - 1) / max(FT(Nc - 1), one(FT))
            z = FT(k - 1) / max(FT(size(m, 3) - 1), one(FT))
            m[ii, jj, k] = FT(1) + FT(0.05) * sinpi(FT(2) * x) +
                           FT(0.04) * cospi(FT(2) * y) +
                           FT(0.02) * z + FT(0.001) * p
        end
    end
    fill_panel_halos!(panels_m, mesh; dir = 0)
    return panels_m
end

function _make_tracer(panels_m, mesh::CubedSphereMesh, ::Type{FT}, tracer_idx::Int) where {FT}
    Nc, Hp = mesh.Nc, mesh.Hp
    panels_rm = ntuple(_ -> similar(panels_m[1]), 6)
    @inbounds for p in 1:6
        m = panels_m[p]
        rm = panels_rm[p]
        fill!(rm, zero(FT))
        for k in axes(m, 3), j in 1:Nc, i in 1:Nc
            ii = Hp + i
            jj = Hp + j
            x = FT(i - 1) / max(FT(Nc - 1), one(FT))
            y = FT(j - 1) / max(FT(Nc - 1), one(FT))
            z = FT(k - 1) / max(FT(size(m, 3) - 1), one(FT))
            q = FT(1e-6) * FT(tracer_idx) *
                (FT(1) + FT(0.15) * sinpi(FT(2) * x + FT(0.1) * p) +
                 FT(0.10) * cospi(FT(2) * y) + FT(0.05) * z)
            rm[ii, jj, k] = m[ii, jj, k] * q
        end
    end
    fill_panel_halos!(panels_rm, mesh; dir = 0)
    return panels_rm
end

function _fill_fluxes!(am, bm, cm, panels_m, mesh::CubedSphereMesh,
                       ::Type{FT}; cfl::Real = 0.45) where {FT}
    Nc, Hp = mesh.Nc, mesh.Hp
    cf = FT(cfl)
    @inbounds for p in 1:6
        m = panels_m[p]
        ax = am[p]
        by = bm[p]
        cz = cm[p]
        fill!(ax, zero(FT))
        fill!(by, zero(FT))
        fill!(cz, zero(FT))
        for k in 1:size(m, 3), j in 1:Nc, i in 1:(Nc + 1)
            ii_l = Hp + max(i - 1, 1)
            ii_r = Hp + min(i, Nc)
            jj = Hp + j
            donor = min(m[ii_l, jj, k], m[ii_r, jj, k])
            phase = FT(2pi) * FT(j + p) / FT(Nc + 6)
            ax[Hp + i, jj, k] = cf * FT(0.08) * donor * sin(phase)
        end
        for k in 1:size(m, 3), j in 1:(Nc + 1), i in 1:Nc
            ii = Hp + i
            jj_l = Hp + max(j - 1, 1)
            jj_r = Hp + min(j, Nc)
            donor = min(m[ii, jj_l, k], m[ii, jj_r, k])
            phase = FT(2pi) * FT(i + 2p) / FT(Nc + 12)
            by[ii, Hp + j, k] = cf * FT(0.06) * donor * cos(phase)
        end
        for k in 2:size(m, 3), j in 1:Nc, i in 1:Nc
            ii = Hp + i
            jj = Hp + j
            donor = min(m[ii, jj, k - 1], m[ii, jj, k])
            phase = FT(2pi) * FT(i + j + p) / FT(2Nc + 6)
            cz[ii, jj, k] = cf * FT(0.03) * donor * sin(phase)
        end
    end
    return nothing
end

function build_problem(::Type{FT}, Nc::Int, Nz::Int, Nt::Int, backend) where {FT}
    Hp = 3
    mesh = CubedSphereMesh(; FT = FT, Nc = Nc, Hp = Hp,
                           convention = GEOSNativePanelConvention())
    panels_m_cpu = ntuple(_ -> _haloed_panel(FT, Nc, Hp, Nz), 6)
    _fill_mass!(panels_m_cpu, mesh, FT)
    tracers_cpu = [_make_tracer(panels_m_cpu, mesh, FT, t) for t in 1:Nt]
    tracers_raw_cpu = ntuple(p -> cat((tracers_cpu[t][p] for t in 1:Nt)...; dims = 4), 6)

    N = Nc + 2Hp
    am_cpu = ntuple(_ -> zeros(FT, N + 1, N, Nz), 6)
    bm_cpu = ntuple(_ -> zeros(FT, N, N + 1, Nz), 6)
    cm_cpu = ntuple(_ -> zeros(FT, N, N, Nz + 1), 6)
    _fill_fluxes!(am_cpu, bm_cpu, cm_cpu, panels_m_cpu, mesh, FT)

    panels_m = Adapt.adapt(_array_type(backend), panels_m_cpu)
    tracers = [Adapt.adapt(_array_type(backend), tr) for tr in tracers_cpu]
    tracers_raw = Adapt.adapt(_array_type(backend), tracers_raw_cpu)
    fluxes = CubedSphereFaceFluxState{DryBasis}(
        Adapt.adapt(_array_type(backend), am_cpu),
        Adapt.adapt(_array_type(backend), bm_cpu),
        Adapt.adapt(_array_type(backend), cm_cpu))
    m_save = ntuple(p -> similar(panels_m[p]), 6)
    return mesh, panels_m, m_save, tracers, tracers_raw, fluxes
end

function _copy_panels!(dst::NTuple{6}, src::NTuple{6})
    for p in 1:6
        copyto!(dst[p], src[p])
    end
    return nothing
end

function _workspace_for_problem(mesh, panels_m, scheme, Nt::Int)
    if scheme isa LinRoodPPMScheme
        return CSLinRoodAdvectionWorkspace(mesh, panels_m[1])
    end
    return CSAdvectionWorkspace(mesh, panels_m[1]; n_tracers = Nt)
end

# ---------------------------------------------------------------------------
# Benchmark kernels
# ---------------------------------------------------------------------------

function run_window!(panels_m, m_save, tracers, tracers_raw, fluxes, mesh,
                     scheme::LinRoodPPMScheme, ws, n_sub::Int)
    _copy_panels!(m_save, panels_m)
    for (idx, rm) in enumerate(tracers)
        idx > 1 && _copy_panels!(panels_m, m_save)
        fill_panel_halos!(rm, mesh; dir = 1)
        Adv = AtmosTransport.Operators.Advection
        Adv._cs_transport_step!(Adv.CSLinRoodStyle(), rm, panels_m, fluxes,
                                 mesh, scheme, ws; subcycle_count=n_sub)
    end
    return nothing
end

function run_window!(panels_m, m_save, tracers, tracers_raw, fluxes, mesh,
                     scheme, ws, n_sub::Int)
    _ = m_save
    _ = tracers
    AtmosTransport.Operators.Advection.fill_cs_step_start_halos!(tracers_raw, panels_m, mesh)
    strang_split_cs_mt!(tracers_raw, panels_m, fluxes.am, fluxes.bm, fluxes.cm,
                        mesh, scheme, ws; subcycle_count = n_sub)
    return nothing
end

function time_case(::Type{FT}, Nc::Int, Nz::Int, Nt::Int, n_sub::Int,
                   scheme, backend; repeat::Int, warmup::Int) where {FT}
    mesh, panels_m, m_save, tracers, tracers_raw, fluxes = build_problem(FT, Nc, Nz, Nt, backend)
    ws = _workspace_for_problem(mesh, panels_m, scheme, Nt)
    _sync(backend)

    for _ in 1:warmup
        run_window!(panels_m, m_save, tracers, tracers_raw, fluxes, mesh, scheme, ws, n_sub)
    end
    _sync(backend)

    times = Float64[]
    for _ in 1:repeat
        t0 = time_ns()
        run_window!(panels_m, m_save, tracers, tracers_raw, fluxes, mesh, scheme, ws, n_sub)
        _sync(backend)
        push!(times, (time_ns() - t0) * 1e-9)
    end
    return median(times), median_abs_dev(times)
end

function _scheme_from_string(s::String)
    key = lowercase(s)
    haskey(SCHEME_TABLE, key) || error("Unsupported scheme $(s); use $(join(sort(collect(keys(SCHEME_TABLE))), ", "))")
    return SCHEME_TABLE[key]
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    backend = Val(BACKEND)
    if BACKEND === :gpu
        @printf("CUDA device: %s\n", CUDA.name(CUDA.device()))
    end
    @printf("CS synthetic advection benchmark: backend=%s repeat=%d warmup=%d\n",
            string(BACKEND), REPEAT, WARMUP)
    @printf("grids=%s levels=%s tracers=%s steps=%s schemes=%s dtypes=%s byte_models=%s\n",
            join("C" .* string.(GRIDS), ","), join(LEVELS, ","), join(TRACERS, ","),
            join(STEPS, ","), join(SCHEMES, ","), join(string.(DTYPES), ","),
            join(BYTE_MODELS, ","))

    if STREAM && BACKEND === :gpu
        for FT in DTYPES
            gbps = stream_bandwidth(FT, backend)
            @printf("STREAM triad %-7s: %.1f GB/s\n", string(FT), gbps)
        end
    end

    @printf("\n%-8s %-7s %5s %4s %7s %9s %12s %10s %14s %14s",
            "scheme", "dtype", "Nc", "Nz", "tracers", "n_sub",
            "time/window", "MAD", "cell-sweeps/s", "tracer-sweeps/s")
    for b in BYTE_MODELS
        @printf(" %10s", @sprintf("GB/s@%.0fB", b))
    end
    println()
    println("-"^132)

    for FT in DTYPES, Nc in GRIDS, Nz in LEVELS, Nt in TRACERS, n_sub in STEPS, scheme_s in SCHEMES
        scheme = _scheme_from_string(scheme_s)
        med_s, mad_s = time_case(FT, Nc, Nz, Nt, n_sub, scheme, backend;
                                 repeat = REPEAT, warmup = WARMUP)
        cells = 6 * Nc^2 * Nz
        cell_sweeps = cells * 6 * n_sub
        tracer_sweeps = cell_sweeps * Nt
        cell_rate = cell_sweeps / med_s
        tracer_rate = tracer_sweeps / med_s
        @printf("%-8s %-7s %5d %4d %7d %9d %11.4fs %9.4fs %13.3e %13.3e",
                scheme_s, string(FT), Nc, Nz, Nt, n_sub, med_s, mad_s,
                cell_rate, tracer_rate)
        for b in BYTE_MODELS
            @printf(" %10.1f", tracer_sweeps * b / med_s / 1e9)
        end
        println()
    end
end

main()
