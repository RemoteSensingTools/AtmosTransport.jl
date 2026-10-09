#!/usr/bin/env julia
# Microbenchmark: TM5 convection P6 cache, with the cached factor re-laid
# out for coalesced reads across adjacent workgroups (P17).
#
# Background: in the existing P6 hit kernel the cache is stored as
# cache_A[k, kk, i, j] with shape (Nz, Nz, Nc, Nc) — Julia column-major,
# so the (k, kk) inner axes are the fastest-varying.  Adjacent workgroups
# correspond to adjacent (i, j); when each workgroup loads cache_A[k, kk, i, j]
# for the SAME (k, kk), they hit completely different memory locations
# (stride Nz·Nz·sizeof(Float32) ≈ 29 kB apart) — no coalescing.
#
# P17 idea: re-lay out the cache as cache_A[i, j, k, kk] with shape
# (Nc, Nc, Nz, Nz) so that adjacent workgroups (adjacent i) read adjacent
# memory locations for any fixed (k, kk).
#
# BIT_EXACT on hit (layout-only change, no math change).  Same memory
# footprint (938 MB on Nc=180, Nz=85).
#
# Usage:
#   julia --project=. scripts/benchmarks/bench_tm5_p17_layout.jl \
#       [--binary <path>] [--window 1] [--panel 1] [--nt 2] [--repeat 5]

using Printf
using Statistics
using CUDA
using Adapt
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const, @localmem, @synchronize,
                          @groupsize, get_backend, synchronize
using Random

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.MetDrivers: TransportBinaryReader
using .AtmosTransport.Operators: TM5Convection, TM5Workspace

const DEFAULT_BIN = "/temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps48_v3_20260520/era5_transport_20211202_merged1000Pa_float32.bin"

const NZ_MAX = 85
const WG_SIZE = 32

# Elements of one CS payload section (the library's table, which knows every section).
_section_elements(h, section::Symbol) = AtmosTransport.MetDrivers._cs_section_elements(h, section)

function _section_offset(h, win::Int, section::Symbol)
    o = (win - 1) * h.elems_per_window
    for s in h.payload_sections
        s === section && return o
        o += _section_elements(h, s)
    end
    error("missing section: $section")
end

function _panel_view(reader, win::Int, section::Symbol, panel::Int)
    h = reader.header
    Nc, Nz = h.geometry.Nc, h.nlevel
    panel_elems = Nc * Nc * Nz
    sec_off = _section_offset(h, win, section)
    lo = sec_off + (panel - 1) * panel_elems + 1
    hi = lo + panel_elems - 1
    return reshape(@view(reader.data[lo:hi]), Nc, Nc, Nz)
end

function _panel_cell_areas(::Type{FT}, Nc; radius = FT(6.371e6)) where {FT}
    panel_area = FT(4π) * radius^2 / 6
    return fill(panel_area / Nc^2, Nc, Nc)
end

# ============================================================
# MISS kernel — same as P6 but writes cache to TRANSPOSED layout
# (i, j, k, kk) and (i, j, k) for pivots.
# ============================================================
@kernel function _tm5_p17_miss_kernel!(
    q_raw,
    cache_A_T,       # (Nc, Nc, Nz, Nz) — TRANSPOSED layout: (i, j, k, kk)
    cache_piv_T,     # (Nc, Nc, Nz) — TRANSPOSED layout: (i, j, k)
    cache_klo,
    @Const(air_mass),
    @Const(entu), @Const(detu), @Const(entd), @Const(detd),
    @Const(cell_areas_y),
    Nx::Int, Nz_in::Int, Nt::Int, dt::Float32,
)
    g = @index(Group)
    t = @index(Local)
    i = ((g - 1) % Nx) + 1
    j = ((g - 1) ÷ Nx) + 1

    A_loc = @localmem Float32 (NZ_MAX, NZ_MAX)
    q_loc = @localmem Float32 (NZ_MAX, 4)
    piv_loc = @localmem Int32 (NZ_MAX,)
    bmass_loc = @localmem Float32 (NZ_MAX,)
    amu_loc = @localmem Float32 (NZ_MAX + 1,)
    amd_loc = @localmem Float32 (NZ_MAX + 1,)
    icl_top = @localmem Int32 (1,)
    icl_lfs = @localmem Int32 (1,)

    area = cell_areas_y[j]

    for k in t:WG_SIZE:(NZ_MAX + 1)
        amu_loc[k] = 0f0
        amd_loc[k] = 0f0
    end

    if t == 1
        icl_top[1] = Int32(Nz_in + 1)
        icl_lfs[1] = Int32(Nz_in + 1)
        for k in 1:Nz_in
            d = detu[i, j, k]
            if d > 0f0 && icl_top[1] == Nz_in + 1
                icl_top[1] = Int32(k)
            end
            e = entd[i, j, k]
            if e > 0f0 && icl_lfs[1] == Nz_in + 1
                icl_lfs[1] = Int32(k)
            end
        end
    end
    @synchronize

    icltop = Int(icl_top[1])
    icllfs = Int(icl_lfs[1])
    icltop_eff = min(icllfs, max(icltop, 2) - 1)
    k_lo = max(icltop_eff, 1)
    no_conv = icltop > Nz_in

    if t == 1
        cache_klo[i, j] = Int32(no_conv ? Nz_in + 1 : k_lo)
    end

    for k in t:WG_SIZE:Nz_in
        bmass_loc[k] = air_mass[i, j, k] / area
    end
    @synchronize

    if !no_conv
        for idx in t:WG_SIZE:Nz_in*Nz_in
            k = (idx - 1) % Nz_in + 1
            kk = (idx - 1) ÷ Nz_in + 1
            if k >= k_lo && kk >= k_lo
                A_loc[k, kk] = 0f0
            elseif k == kk
                A_loc[k, kk] = 1f0
            else
                A_loc[k, kk] = 0f0
            end
        end
        @synchronize

        if t == 1
            for k in Nz_in:-1:icltop
                e = entu[i, j, k]
                d = detu[i, j, k]
                amu_loc[k] = amu_loc[k + 1] + e - d
                zxi = 0f0
                if amu_loc[k] > 0f0
                    denom = amu_loc[k + 1] + e
                    zxi = max(0f0, 1f0 - d / denom)
                else
                    amu_loc[k] = 0f0
                end
                for kk in (k + 1):Nz_in
                    f_below = k == Nz_in ? 0f0 : A_loc[k + 1, kk]
                    A_loc[k, kk] = f_below * zxi
                end
                bmass_k = bmass_loc[k]
                A_loc[k, k] = e / bmass_k * zxi
            end

            if icllfs <= Nz_in
                for k in icllfs:(Nz_in - 1)
                    e = entd[i, j, k]
                    d = detd[i, j, k]
                    amd_loc[k + 1] = amd_loc[k] - e + d
                    zxi = 0f0
                    if amd_loc[k + 1] < 0f0
                        denom = amd_loc[k] - e
                        zxi = max(0f0, 1f0 + d / denom)
                    else
                        amd_loc[k + 1] = 0f0
                    end
                    for kk in icllfs:(k - 1)
                        A_loc[k + 1, kk] = A_loc[k, kk] * zxi
                    end
                    bmass_k = bmass_loc[k]
                    A_loc[k + 1, k] = -e / bmass_k * zxi
                end
            end

            for k in Nz_in:-1:2
                bmass_above = bmass_loc[k - 1]
                bmass_k = bmass_loc[k]
                A_loc[k, k - 1] -= amu_loc[k] / bmass_above
                A_loc[k, k]     -= amd_loc[k] / bmass_k
            end

            for k in 1:Nz_in
                for kk in 1:Nz_in
                    if k < k_lo || kk < k_lo
                        A_loc[k, kk] = (k == kk) ? 1f0 : 0f0
                        continue
                    end
                    f_below = k == Nz_in ? 0f0 : A_loc[k + 1, kk]
                    fdiff = f_below - A_loc[k, kk]
                    A_loc[k, kk] = -dt * fdiff
                end
                A_loc[k, k] += 1f0
            end
        end
    end
    @synchronize

    for idx in t:WG_SIZE:(Nz_in * Nt)
        k = (idx - 1) % Nz_in + 1
        tt = (idx - 1) ÷ Nz_in + 1
        q_loc[k, tt] = q_raw[i, j, k, tt]
    end
    @synchronize

    if !no_conv
        for k in k_lo:Nz_in
            if t == 1
                piv = k
                pivmag = abs(A_loc[k, k])
                for r in (k + 1):Nz_in
                    m_ = abs(A_loc[r, k])
                    if m_ > pivmag
                        piv = r
                        pivmag = m_
                    end
                end
                piv_loc[k] = Int32(piv)
            end
            @synchronize
            piv = Int(piv_loc[k])

            if piv != k
                for cc in (k_lo + t - 1):WG_SIZE:Nz_in
                    tmp = A_loc[k, cc]
                    A_loc[k, cc] = A_loc[piv, cc]
                    A_loc[piv, cc] = tmp
                end
            end
            @synchronize

            diag_val = A_loc[k, k]
            for r in (k + t):WG_SIZE:Nz_in
                A_loc[r, k] /= diag_val
            end
            @synchronize

            for cc in (k + t):WG_SIZE:Nz_in
                akc = A_loc[k, cc]
                for r in (k + 1):Nz_in
                    A_loc[r, cc] -= A_loc[r, k] * akc
                end
            end
            @synchronize
        end

        if t == 1
            for k in k_lo:Nz_in
                piv = Int(piv_loc[k])
                if piv != k
                    for tt in 1:Nt
                        tmp = q_loc[k, tt]
                        q_loc[k, tt] = q_loc[piv, tt]
                        q_loc[piv, tt] = tmp
                    end
                end
            end
        end
        @synchronize

        for tt in t:WG_SIZE:Nt
            for k in k_lo:Nz_in
                s = q_loc[k, tt]
                for j2 in k_lo:(k - 1)
                    s -= A_loc[k, j2] * q_loc[j2, tt]
                end
                q_loc[k, tt] = s
            end
        end
        @synchronize

        for tt in t:WG_SIZE:Nt
            for k in Nz_in:-1:k_lo
                s = q_loc[k, tt]
                for j2 in (k + 1):Nz_in
                    s -= A_loc[k, j2] * q_loc[j2, tt]
                end
                q_loc[k, tt] = s / A_loc[k, k]
            end
        end
        @synchronize
    end

    # ---- Persist cache in TRANSPOSED layout ----
    # cache_A_T[i, j, k, kk] — adjacent (i, j) workgroups now read adjacent
    # memory for any fixed (k, kk).
    for idx in t:WG_SIZE:(Nz_in * Nz_in)
        k = (idx - 1) % Nz_in + 1
        kk = (idx - 1) ÷ Nz_in + 1
        cache_A_T[i, j, k, kk] = A_loc[k, kk]
    end
    for k in t:WG_SIZE:Nz_in
        cache_piv_T[i, j, k] = piv_loc[k]
    end

    for idx in t:WG_SIZE:(Nz_in * Nt)
        k = (idx - 1) % Nz_in + 1
        tt = (idx - 1) ÷ Nz_in + 1
        q_raw[i, j, k, tt] = q_loc[k, tt]
    end
end

# ============================================================
# HIT kernel — reads cache from TRANSPOSED layout.
# ============================================================
@kernel function _tm5_p17_hit_kernel!(
    q_raw,
    @Const(cache_A_T),
    @Const(cache_piv_T),
    @Const(cache_klo),
    Nx::Int, Nz_in::Int, Nt::Int,
)
    g = @index(Group)
    t = @index(Local)
    i = ((g - 1) % Nx) + 1
    j = ((g - 1) ÷ Nx) + 1

    A_loc = @localmem Float32 (NZ_MAX, NZ_MAX)
    q_loc = @localmem Float32 (NZ_MAX, 4)
    piv_loc = @localmem Int32 (NZ_MAX,)

    k_lo = Int(cache_klo[i, j])
    no_conv = k_lo > Nz_in

    # Read transposed cache: each thread reads a different (k, kk) but
    # adjacent workgroups (adjacent i) read adjacent locations.
    for idx in t:WG_SIZE:(Nz_in * Nz_in)
        k = (idx - 1) % Nz_in + 1
        kk = (idx - 1) ÷ Nz_in + 1
        A_loc[k, kk] = cache_A_T[i, j, k, kk]
    end
    for k in t:WG_SIZE:Nz_in
        piv_loc[k] = cache_piv_T[i, j, k]
    end
    for idx in t:WG_SIZE:(Nz_in * Nt)
        k = (idx - 1) % Nz_in + 1
        tt = (idx - 1) ÷ Nz_in + 1
        q_loc[k, tt] = q_raw[i, j, k, tt]
    end
    @synchronize

    if !no_conv
        if t == 1
            for k in k_lo:Nz_in
                piv = Int(piv_loc[k])
                if piv != k
                    for tt in 1:Nt
                        tmp = q_loc[k, tt]
                        q_loc[k, tt] = q_loc[piv, tt]
                        q_loc[piv, tt] = tmp
                    end
                end
            end
        end
        @synchronize

        for tt in t:WG_SIZE:Nt
            for k in k_lo:Nz_in
                s = q_loc[k, tt]
                for j2 in k_lo:(k - 1)
                    s -= A_loc[k, j2] * q_loc[j2, tt]
                end
                q_loc[k, tt] = s
            end
        end
        @synchronize

        for tt in t:WG_SIZE:Nt
            for k in Nz_in:-1:k_lo
                s = q_loc[k, tt]
                for j2 in (k + 1):Nz_in
                    s -= A_loc[k, j2] * q_loc[j2, tt]
                end
                q_loc[k, tt] = s / A_loc[k, k]
            end
        end
        @synchronize
    end

    for idx in t:WG_SIZE:(Nz_in * Nt)
        k = (idx - 1) % Nz_in + 1
        tt = (idx - 1) ÷ Nz_in + 1
        q_raw[i, j, k, tt] = q_loc[k, tt]
    end
end

function bench_baseline!(q_after, q_before, m, entu, detu, entd, detd,
                          cell_areas, dt, nrep)
    Nc, _, Nz = size(m)
    backend = CUDA.CUDABackend()
    ws_cpu = TM5Workspace(Array(m); tile_columns = Nc*Nc,
                          cell_metrics = cell_areas)
    ws = Adapt.adapt(CuArray, ws_cpu)
    kernel = AtmosTransport.Operators.Convection._tm5_column_kernel!(backend)
    times_ms = Float64[]
    for r in 0:nrep
        q_work = CUDA.copy(q_before)
        CUDA.synchronize()
        t = CUDA.@elapsed begin
            kernel(q_work, m, entu, detu, entd, detd,
                   cell_areas[1, :],
                   ws.conv1, ws.pivots, ws.cloud_dims,
                   ws.f_scratch, ws.amu_scratch, ws.amd_scratch,
                   Int(0), Int(Nc), Float32(dt);
                   ndrange = Nc*Nc)
            CUDA.synchronize()
        end
        r > 0 && push!(times_ms, t * 1000)
        r == 0 && copyto!(q_after, q_work)
    end
    return minimum(times_ms), median(times_ms)
end

function bench_p17_miss!(q_after, q_before, cache_A_T, cache_piv_T, cache_klo,
                          m, entu, detu, entd, detd, cell_areas, dt, nrep)
    Nc, _, Nz = size(m)
    Nt = size(q_before, 4)
    backend = get_backend(q_before)
    kernel = _tm5_p17_miss_kernel!(backend, WG_SIZE)
    times_ms = Float64[]
    for r in 0:nrep
        q_work = CUDA.copy(q_before)
        CUDA.synchronize()
        t = CUDA.@elapsed begin
            kernel(q_work, cache_A_T, cache_piv_T, cache_klo,
                   m, entu, detu, entd, detd,
                   cell_areas[1, :],
                   Int(Nc), Int(Nz), Int(Nt), Float32(dt);
                   ndrange = WG_SIZE * (Nc * Nc),
                   workgroupsize = WG_SIZE)
            CUDA.synchronize()
        end
        r > 0 && push!(times_ms, t * 1000)
        r == 0 && copyto!(q_after, q_work)
    end
    return minimum(times_ms), median(times_ms)
end

function bench_p17_hit!(q_after, q_before, cache_A_T, cache_piv_T, cache_klo, m, nrep)
    Nc, _, Nz = size(m)
    Nt = size(q_before, 4)
    backend = get_backend(q_before)
    kernel = _tm5_p17_hit_kernel!(backend, WG_SIZE)
    times_ms = Float64[]
    for r in 0:nrep
        q_work = CUDA.copy(q_before)
        CUDA.synchronize()
        t = CUDA.@elapsed begin
            kernel(q_work, cache_A_T, cache_piv_T, cache_klo,
                   Int(Nc), Int(Nz), Int(Nt);
                   ndrange = WG_SIZE * (Nc * Nc),
                   workgroupsize = WG_SIZE)
            CUDA.synchronize()
        end
        r > 0 && push!(times_ms, t * 1000)
        r == 0 && copyto!(q_after, q_work)
    end
    return minimum(times_ms), median(times_ms)
end

function main()
    bin = DEFAULT_BIN; win = 1; panel = 1; nt = 2; dt = 1800.0; nrep = 5
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if     a == "--binary" bin = ARGS[i+1]; i += 2
        elseif a == "--window" win = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--panel"  panel = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--nt"     nt = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--dt"     dt = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--repeat" nrep = parse(Int, ARGS[i+1]); i += 2
        else error("unknown arg `$a`")
        end
    end
    @info "Loading binary" bin win panel nt dt nrep
    reader = TransportBinaryReader(bin; FT = Float32)
    h = reader.header
    @assert h.nlevel == NZ_MAX
    @info "Binary header" Nc=h.geometry.Nc Nz=h.nlevel npanel=h.geometry.npanel nwindow=h.nwindow

    entu = collect(_panel_view(reader, win, :entu, panel))
    detu = collect(_panel_view(reader, win, :detu, panel))
    entd = collect(_panel_view(reader, win, :entd, panel))
    detd = collect(_panel_view(reader, win, :detd, panel))
    m    = collect(_panel_view(reader, win, :m,    panel))
    close(reader)
    FT = Float32
    Nc, _, Nz = size(m)

    rng = MersenneTwister(0)
    q_cpu = randn(rng, FT, Nc, Nc, Nz, nt) .* FT(1e-3) .+ FT(1.0)
    area = _panel_cell_areas(FT, Nc; radius = FT(h.planet_radius_m))
    entu_d = CuArray(entu); detu_d = CuArray(detu)
    entd_d = CuArray(entd); detd_d = CuArray(detd)
    m_d    = CuArray(m);    q_d    = CuArray(q_cpu)
    area_d = CuArray(area)

    # Transposed-layout cache: (Nc, Nc, Nz, Nz).
    cache_A_T   = CuArray{Float32}(undef, Nc, Nc, Nz, Nz)
    cache_piv_T = CuArray{Int32}(undef, Nc, Nc, Nz)
    cache_klo   = CuArray{Int32}(undef, Nc, Nc)
    @info "Allocated transposed cache" cache_A_T_bytes=length(cache_A_T)*4

    @info "Running baseline"
    q_base = similar(q_d)
    t_base_min, t_base_med = bench_baseline!(q_base, q_d, m_d, entu_d, detu_d,
                                              entd_d, detd_d, area_d, dt, nrep)
    @printf "  baseline:    min %.2f ms  median %.2f ms  (n=%d)\n" t_base_min t_base_med nrep

    @info "Running P17 miss (transposed write)"
    q_miss = similar(q_d)
    t_miss_min, t_miss_med = bench_p17_miss!(q_miss, q_d, cache_A_T, cache_piv_T, cache_klo,
                                              m_d, entu_d, detu_d, entd_d, detd_d,
                                              area_d, dt, nrep)
    @printf "  P17 miss:    min %.2f ms  median %.2f ms\n" t_miss_min t_miss_med
    err_miss = maximum(abs.(Array(q_miss) .- Array(q_base)))
    @printf "  max|Δq| baseline vs P17 miss: %.3e\n" err_miss

    @info "Running P17 hit (transposed read, solve only)"
    q_hit = similar(q_d)
    t_hit_min, t_hit_med = bench_p17_hit!(q_hit, q_d, cache_A_T, cache_piv_T, cache_klo,
                                            m_d, nrep)
    @printf "  P17 hit:     min %.2f ms  median %.2f ms\n" t_hit_min t_hit_med
    err_hit = maximum(abs.(Array(q_hit) .- Array(q_miss)))
    @printf "  max|Δq| miss vs hit (same input): %.3e\n" err_hit
    err_hit_base = maximum(abs.(Array(q_hit) .- Array(q_base)))
    @printf "  max|Δq| baseline vs P17 hit: %.3e\n" err_hit_base

    @printf "  speedup baseline / P17 miss: %.2fx\n" t_base_min / t_miss_min
    @printf "  speedup baseline / P17 hit : %.2fx\n" t_base_min / t_hit_min
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
