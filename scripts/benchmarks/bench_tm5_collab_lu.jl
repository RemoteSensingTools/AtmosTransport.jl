#!/usr/bin/env julia
# Microbenchmark: workgroup-collaborative TM5 convection LU/solve kernel.
#
# Goal: a portable (CUDA + Metal) KA kernel that beats the per-thread serial
# LU baseline, since cuBLAS is CUDA-only.
#
# Design: one workgroup per column, WG_SIZE threads collaborate on a single
# (N, N) LU in `@localmem`.  Each iteration of the outer k-loop:
#   - thread 0: scan column k for the pivot row.
#   - all threads: parallel row swap.
#   - all threads: parallel column scale below diagonal.
#   - all threads: parallel rank-1 trailing update.
#   - @synchronize between phases.
#
# After LU: collaborative pivot-apply, forward solve, back solve on a stored
# `(N, Nt)` RHS in `@localmem`.
#
# We compare against:
#   1. baseline `_tm5_column_kernel!` (per-thread serial)
#   2. cuBLAS strided-batched at lmax=64 (CUDA-only reference upper bound)
#
# Usage:
#   julia --project=. scripts/benchmarks/bench_tm5_collab_lu.jl \
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
using .AtmosTransport.Operators.Convection: _tm5_build_conv1!,
                                              _tm5_diagnose_cloud_dims

const DEFAULT_BIN = "/temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps48_v3_20260520/era5_transport_20211202_merged1000Pa_float32.bin"

# Compile-time matrix size and workgroup size.  Nz is determined by the
# binary header at run time; we use the same value for N here and tile
# everything to a single Nz × Nz matrix per column.  The `Val(N)` /
# `Val(WG)` plumbing keeps the `@localmem` allocation static.
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
# Collaborative LU + solve kernel.
#
# Each workgroup processes one column.  Indexing convention:
#   - @index(Group)  → column index 1..B  (1D launch)
#   - @index(Local)  → thread index 1..WG within the workgroup
#
# Workgroup memory budget at Nz=85, Nt≤4, Float32:
#   A_loc : 85 * 85 * 4  = 28 900 B
#   q_loc : 85 *  4 * 4  =  1 360 B  (Nt up to 4)
#   piv_loc: 85 * 4      =    340 B
# Total: ~30.6 KB per workgroup.  Fits L40S (100 KB/SM) and M2 (32+ KB/WG).
# ============================================================
@kernel function _tm5_collab_lu_kernel!(
    q_raw,                # (Nc, Nc, Nz, Nt) — tracer mass, updated in place
    @Const(air_mass),     # (Nc, Nc, Nz)
    @Const(entu), @Const(detu), @Const(entd), @Const(detd),
    @Const(cell_areas_y), # (Nc,) cell-area-per-latitude (LL parity)
    Nx::Int, Nz_in::Int, Nt::Int, dt::Float32,
)
    g = @index(Group)
    t = @index(Local)

    # Decode (i, j) from group index, column-major.
    i = ((g - 1) % Nx) + 1
    j = ((g - 1) ÷ Nx) + 1

    # Shared workspace.  Sizes are constants so KA can map them to
    # CUDA `__shared__` / Metal `threadgroup` storage.
    A_loc = @localmem Float32 (NZ_MAX, NZ_MAX)
    q_loc = @localmem Float32 (NZ_MAX, 4)        # supports Nt ≤ 4
    piv_loc = @localmem Int32 (NZ_MAX,)
    bmass_loc = @localmem Float32 (NZ_MAX,)
    amu_loc = @localmem Float32 (NZ_MAX + 1,)
    amd_loc = @localmem Float32 (NZ_MAX + 1,)
    icl_top = @localmem Int32 (1,)
    icl_lfs = @localmem Int32 (1,)

    # ---- Phase 1: build the matrix ----------------------------------
    # Each thread handles a chunk of k values.
    area = cell_areas_y[j]

    # Initialize amu/amd boundaries to zero (parallel).
    for k in t:WG_SIZE:(NZ_MAX + 1)
        amu_loc[k] = 0f0
        amd_loc[k] = 0f0
    end

    # Diagnose icltop, icllfs (serial in thread 1 — Nz ops, cheap).
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
    # k_lo follows _tm5_build_conv1!:icltop_eff = min(icllfs, max(icltop,2)-1)
    icltop_eff = min(icllfs, max(icltop, 2) - 1)
    k_lo = max(icltop_eff, 1)

    # If no convection in this column, identity transform — write the
    # input tracer back unchanged and exit early via a one-thread guard.
    no_conv = icltop > Nz_in

    # Pre-compute bmass[k] = m[i,j,k] / area in shared.
    for k in t:WG_SIZE:Nz_in
        bmass_loc[k] = air_mass[i, j, k] / area
    end
    @synchronize

    # Build A_loc in TWO PASSES, mirroring _tm5_build_conv1! exactly so
    # numerical results stay bit-identical to baseline within rounding.
    # Identity rows above k_lo, zero lower-left below k_lo, dense
    # active block from (k_lo, Nz_in).

    # Initialize the active block to zero (parallel sweep over k_lo..Nz_in × k_lo..Nz_in).
    if !no_conv
        for idx in t:WG_SIZE:Nz_in*Nz_in
            k = (idx - 1) % Nz_in + 1
            kk = (idx - 1) ÷ Nz_in + 1
            if k >= k_lo && kk >= k_lo
                A_loc[k, kk] = 0f0
            elseif k == kk
                A_loc[k, kk] = 1f0       # identity above k_lo
            else
                A_loc[k, kk] = 0f0       # off-diag zeros above k_lo
            end
        end
        @synchronize

        # Updraft + downdraft + subsidence passes happen sequentially in
        # thread 1.  The math has strong inter-iteration dependence
        # (amu[k] depends on amu[k+1]; f[k, ...] depends on f[k+1, ...]),
        # so parallelising it is non-trivial — we keep that serial in
        # this first cut.  The build is O(Nz²) flops per column =
        # 7 200 ops at Nz=85: cheap relative to the O(Nz³) LU.
        if t == 1
            # Updraft pass (k from Nz_in down to icltop).
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

            # Downdraft pass.
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

            # Subsidence subtraction.
            for k in Nz_in:-1:2
                bmass_above = bmass_loc[k - 1]
                bmass_k = bmass_loc[k]
                A_loc[k, k - 1] -= amu_loc[k] / bmass_above
                A_loc[k, k]     -= amd_loc[k] / bmass_k
            end

            # Final f -> conv1 conversion: conv1[k, kk] = -dt * (f[k+1, kk] - f[k, kk]) + I.
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

    # ---- Phase 2: load RHS into q_loc -----------------------------
    for idx in t:WG_SIZE:(Nz_in * Nt)
        k = (idx - 1) % Nz_in + 1
        tt = (idx - 1) ÷ Nz_in + 1
        q_loc[k, tt] = q_raw[i, j, k, tt]
    end
    @synchronize

    # ---- Phase 3: collaborative right-looking LU on A_loc[k_lo:Nz, k_lo:Nz]
    # If no_conv, A_loc is identity, k_lo == Nz_in + 1, so the loop body
    # never runs and we fall through to the solve which is a no-op too.
    # We do still need to write back, which the final phase covers.
    if !no_conv
    for k in k_lo:Nz_in
        # Pivot search (serial in thread 1).
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

        # Row swap, parallel across columns from k_lo to Nz_in.
        if piv != k
            for cc in (k_lo + t - 1):WG_SIZE:Nz_in
                tmp = A_loc[k, cc]
                A_loc[k, cc] = A_loc[piv, cc]
                A_loc[piv, cc] = tmp
            end
        end
        @synchronize

        diag_val = A_loc[k, k]
        # Scale column k below diagonal.
        for r in (k + t):WG_SIZE:Nz_in
            A_loc[r, k] /= diag_val
        end
        @synchronize

        # Rank-1 update of trailing submatrix: each thread handles columns
        # (k+1)+t-1, (k+1)+t-1+WG, ...  For each such column, walk rows
        # k+1..Nz_in serially.
        for cc in (k + t):WG_SIZE:Nz_in
            akc = A_loc[k, cc]
            for r in (k + 1):Nz_in
                A_loc[r, cc] -= A_loc[r, k] * akc
            end
        end
        @synchronize
    end

    # ---- Phase 4: pivot-aware forward + back solve, per tracer ---
    # Pivot apply (serial in thread 1, fast for k_lo..Nz_in entries).
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

    # Forward solve L y = b, parallel across tracers.
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

    # Back solve U x = y, parallel across tracers.
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
    end  # close if !no_conv

    # ---- Phase 5: store back ---------------------------------------
    for idx in t:WG_SIZE:(Nz_in * Nt)
        k = (idx - 1) % Nz_in + 1
        tt = (idx - 1) ÷ Nz_in + 1
        q_raw[i, j, k, tt] = q_loc[k, tt]
    end
end

function bench_collab!(q_after, q_before, m, entu, detu, entd, detd,
                        cell_areas, dt, nrep)
    Nc, _, Nz = size(m)
    Nt = size(q_before, 4)
    backend = get_backend(q_before)
    kernel = _tm5_collab_lu_kernel!(backend, WG_SIZE)
    times_ms = Float64[]
    for r in 0:nrep
        q_work = CUDA.copy(q_before)
        CUDA.synchronize()
        t = CUDA.@elapsed begin
            kernel(q_work, m, entu, detu, entd, detd,
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
    @assert h.nlevel == NZ_MAX "this bench is pinned at Nz=$(NZ_MAX), binary has Nz=$(h.nlevel)"
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

    @info "Running baseline"
    q_base = similar(q_d)
    t_base_min, t_base_med = bench_baseline!(q_base, q_d, m_d, entu_d, detu_d,
                                              entd_d, detd_d, area_d, dt, nrep)
    @printf "  baseline:    min %.2f ms  median %.2f ms  (n=%d)\n" t_base_min t_base_med nrep

    @info "Running collab-LU"
    q_collab = similar(q_d)
    t_collab_min, t_collab_med = bench_collab!(q_collab, q_d, m_d, entu_d, detu_d,
                                                 entd_d, detd_d, area_d, dt, nrep)
    @printf "  collab-LU:   min %.2f ms  median %.2f ms\n" t_collab_min t_collab_med

    err = maximum(abs.(Array(q_collab) .- Array(q_base)))
    @printf "  max|Δq| baseline vs collab: %.3e\n" err
    @printf "  speedup baseline / collab : %.2fx\n" t_base_min / t_collab_min
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
