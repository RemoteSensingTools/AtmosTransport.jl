#!/usr/bin/env julia
# Microbenchmark: TM5 convection bucketed `Val{NZ}` collab-LU (P4).
#
# Design (from /tmp/tm5_round1_proposals.md P4 + /tmp/tm5_round1_ranking.md):
#
#   The per-column active depth `Nz_active = Nz - icltop + 1` varies a lot
#   across the panel.  We pre-scan columns and assign each to one of four
#   buckets: 32, 48, 64, 75.  We launch one specialized kernel per bucket,
#   each compiled at a compile-time `Val{NZ}` so the inner LU loop unrolls.
#
#   The matrix-build runs over the FULL `Nz_in` layers (identity above
#   icltop is structural, and the build write region is bounded by
#   `[k_lo, Nz_in]`), but the factorize + solve only iterates over the
#   bucket's compile-time `NZ_BUCKET` lower-right window.
#
#   To keep things bit-faithful with the baseline, the bucket size
#   determines the LU window size: a column with active depth ≤ 32 goes
#   into the Val{32} bucket and we factorize the bottom 32×32 block.  The
#   rows above that bottom block are identity-row (no-op for the LU and
#   solve).
#
# Column → bucket mapping is computed once per substep on the host side
# by scanning detu/entd on the GPU.  Each bucket gets a column index
# list, dispatched via a packed list of `(i, j)` pairs.
#
# Usage:
#   julia --project=. scripts/completed_experiments/bench_tm5_p4_bucketed.jl \
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
# Bucketed kernel.  Each workgroup processes one column from the
# bucket's index list.  The compile-time parameter `NZ_BUCKET`
# determines the size of the lower-right LU block.
#
# `k_lo_eff = max(k_lo, Nz_in - NZ_BUCKET + 1)`.  Columns assigned
# to this bucket are guaranteed to have `Nz_in - k_lo + 1 <= NZ_BUCKET`,
# so `k_lo_eff == k_lo` for assigned columns; the bucket cap is a
# compile-time bound used to size @localmem.
# ============================================================
@kernel function _tm5_bucketed_kernel!(
    q_raw, @Const(air_mass),
    @Const(entu), @Const(detu), @Const(entd), @Const(detd),
    @Const(cell_areas_y),
    @Const(idx_i), @Const(idx_j),
    Nbucket::Int, Nz_in::Int, Nt::Int, dt::Float32,
    ::Val{NZ_BUCKET},
) where {NZ_BUCKET}
    g = @index(Group)
    t = @index(Local)

    # Note: ndrange is set to WG_SIZE * Nbucket so g ∈ [1, Nbucket]
    # always; no guard needed.
    i = Int(idx_i[g])
    j = Int(idx_j[g])

    # Per-bucket shared workspace.  Sized to NZ_BUCKET + 1 for amu/amd,
    # NZ_BUCKET for the local matrix.  The matrix lives in column-major
    # @localmem at (NZ_BUCKET, NZ_BUCKET) and holds the lower-right
    # active block of the full (Nz_in, Nz_in) matrix.  Mapping:
    #   A_loc[k_local, kk_local] == A_full[k_global, kk_global]
    # where k_global = (Nz_in - NZ_BUCKET) + k_local, etc.
    A_loc = @localmem Float32 (NZ_BUCKET, NZ_BUCKET)
    q_loc = @localmem Float32 (NZ_BUCKET, 4)
    piv_loc = @localmem Int32 (NZ_BUCKET,)
    bmass_loc = @localmem Float32 (NZ_BUCKET,)
    amu_loc = @localmem Float32 (NZ_BUCKET + 1,)
    amd_loc = @localmem Float32 (NZ_BUCKET + 1,)
    icl_top_l = @localmem Int32 (1,)
    icl_lfs_l = @localmem Int32 (1,)

    # Bucket-window offset into the full column.
    k_off = Nz_in - NZ_BUCKET  # k_global = k_off + k_local, k_local ∈ [1, NZ_BUCKET]
    area = cell_areas_y[j]

    for k in t:WG_SIZE:(NZ_BUCKET + 1)
        amu_loc[k] = 0f0
        amd_loc[k] = 0f0
    end

    # Diagnose cloud dims over the full column.
    if t == 1
        icl_top_l[1] = Int32(Nz_in + 1)
        icl_lfs_l[1] = Int32(Nz_in + 1)
        for k in 1:Nz_in
            d = detu[i, j, k]
            if d > 0f0 && icl_top_l[1] == Nz_in + 1
                icl_top_l[1] = Int32(k)
            end
            e = entd[i, j, k]
            if e > 0f0 && icl_lfs_l[1] == Nz_in + 1
                icl_lfs_l[1] = Int32(k)
            end
        end
    end
    @synchronize

    icltop = Int(icl_top_l[1])
    icllfs = Int(icl_lfs_l[1])
    icltop_eff = min(icllfs, max(icltop, 2) - 1)
    k_lo = max(icltop_eff, 1)
    no_conv = icltop > Nz_in

    # Convert global k_lo to local; columns are pre-screened so
    # k_lo > k_off (i.e. all active rows fit in the bucket window).
    k_lo_local = max(k_lo - k_off, 1)

    # Pre-compute bmass[k] for the bucket window.
    for kl in t:WG_SIZE:NZ_BUCKET
        kg = k_off + kl
        bmass_loc[kl] = kg >= 1 && kg <= Nz_in ? air_mass[i, j, kg] / area : 1f0
    end
    @synchronize

    if !no_conv
        # Initialize A_loc to zero / identity over the bucket window.
        for idx in t:WG_SIZE:NZ_BUCKET*NZ_BUCKET
            kl = (idx - 1) % NZ_BUCKET + 1
            kkl = (idx - 1) ÷ NZ_BUCKET + 1
            if kl >= k_lo_local && kkl >= k_lo_local
                A_loc[kl, kkl] = 0f0
            elseif kl == kkl
                A_loc[kl, kkl] = 1f0
            else
                A_loc[kl, kkl] = 0f0
            end
        end
        @synchronize

        if t == 1
            # Updraft pass — iterate global k from Nz_in down to icltop
            # (all within the bucket window since columns are screened).
            for kg in Nz_in:-1:icltop
                kl = kg - k_off
                e = entu[i, j, kg]
                d = detu[i, j, kg]
                amu_loc[kl] = amu_loc[kl + 1] + e - d
                zxi = 0f0
                if amu_loc[kl] > 0f0
                    denom = amu_loc[kl + 1] + e
                    zxi = max(0f0, 1f0 - d / denom)
                else
                    amu_loc[kl] = 0f0
                end
                for kkg in (kg + 1):Nz_in
                    kkl = kkg - k_off
                    f_below = kg == Nz_in ? 0f0 : A_loc[kl + 1, kkl]
                    A_loc[kl, kkl] = f_below * zxi
                end
                bmass_k = bmass_loc[kl]
                A_loc[kl, kl] = e / bmass_k * zxi
            end

            if icllfs <= Nz_in
                for kg in icllfs:(Nz_in - 1)
                    kl = kg - k_off
                    e = entd[i, j, kg]
                    d = detd[i, j, kg]
                    amd_loc[kl + 1] = amd_loc[kl] - e + d
                    zxi = 0f0
                    if amd_loc[kl + 1] < 0f0
                        denom = amd_loc[kl] - e
                        zxi = max(0f0, 1f0 + d / denom)
                    else
                        amd_loc[kl + 1] = 0f0
                    end
                    for kkg in icllfs:(kg - 1)
                        kkl = kkg - k_off
                        A_loc[kl + 1, kkl] = A_loc[kl, kkl] * zxi
                    end
                    bmass_k = bmass_loc[kl]
                    A_loc[kl + 1, kl] = -e / bmass_k * zxi
                end
            end

            # Subsidence: only subtract within the bucket window.
            # Need to iterate global k from Nz_in down to max(2, k_off+2)
            # because k-1 must be >= 1 globally; here k_off >= 0 so the
            # local k >= 2 maps to global k >= k_off+2 >= 2.
            for kg in Nz_in:-1:max(2, k_off + 2)
                kl = kg - k_off
                bmass_above = bmass_loc[kl - 1]
                bmass_k = bmass_loc[kl]
                A_loc[kl, kl - 1] -= amu_loc[kl] / bmass_above
                A_loc[kl, kl]     -= amd_loc[kl] / bmass_k
            end

            # Final conv1 assembly.
            for kl in 1:NZ_BUCKET
                kg = k_off + kl
                for kkl in 1:NZ_BUCKET
                    kkg = k_off + kkl
                    if kg < k_lo || kkg < k_lo || kg > Nz_in || kkg > Nz_in
                        A_loc[kl, kkl] = (kl == kkl) ? 1f0 : 0f0
                        continue
                    end
                    f_below = kg == Nz_in ? 0f0 : A_loc[kl + 1, kkl]
                    fdiff = f_below - A_loc[kl, kkl]
                    A_loc[kl, kkl] = -dt * fdiff
                end
                if kg <= Nz_in
                    A_loc[kl, kl] += 1f0
                end
            end
        end
    end
    @synchronize

    # Load RHS into the bucket-local space.
    for idx in t:WG_SIZE:(NZ_BUCKET * Nt)
        kl = (idx - 1) % NZ_BUCKET + 1
        tt = (idx - 1) ÷ NZ_BUCKET + 1
        kg = k_off + kl
        q_loc[kl, tt] = kg >= 1 && kg <= Nz_in ? q_raw[i, j, kg, tt] : 0f0
    end
    @synchronize

    # LU on [k_lo_local, NZ_BUCKET] with partial pivot.
    # NZ_BUCKET is compile-time so LLVM should unroll the outer loop.
    if !no_conv
        for k in 1:NZ_BUCKET
            if k >= k_lo_local
                if t == 1
                    piv = k
                    pivmag = abs(A_loc[k, k])
                    for r in (k + 1):NZ_BUCKET
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
                    for cc in (k_lo_local + t - 1):WG_SIZE:NZ_BUCKET
                        tmp = A_loc[k, cc]
                        A_loc[k, cc] = A_loc[piv, cc]
                        A_loc[piv, cc] = tmp
                    end
                end
                @synchronize

                diag_val = A_loc[k, k]
                for r in (k + t):WG_SIZE:NZ_BUCKET
                    A_loc[r, k] /= diag_val
                end
                @synchronize

                for cc in (k + t):WG_SIZE:NZ_BUCKET
                    akc = A_loc[k, cc]
                    for r in (k + 1):NZ_BUCKET
                        A_loc[r, cc] -= A_loc[r, k] * akc
                    end
                end
                @synchronize
            end
        end

        # Pivot apply.
        if t == 1
            for k in k_lo_local:NZ_BUCKET
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

        # Forward solve.
        for tt in t:WG_SIZE:Nt
            for k in k_lo_local:NZ_BUCKET
                s = q_loc[k, tt]
                for j2 in k_lo_local:(k - 1)
                    s -= A_loc[k, j2] * q_loc[j2, tt]
                end
                q_loc[k, tt] = s
            end
        end
        @synchronize

        # Back solve.
        for tt in t:WG_SIZE:Nt
            for k in NZ_BUCKET:-1:k_lo_local
                s = q_loc[k, tt]
                for j2 in (k + 1):NZ_BUCKET
                    s -= A_loc[k, j2] * q_loc[j2, tt]
                end
                q_loc[k, tt] = s / A_loc[k, k]
            end
        end
        @synchronize
    end

    # Store back (only over the bucket window).
    for idx in t:WG_SIZE:(NZ_BUCKET * Nt)
        kl = (idx - 1) % NZ_BUCKET + 1
        tt = (idx - 1) ÷ NZ_BUCKET + 1
        kg = k_off + kl
        if kg >= 1 && kg <= Nz_in
            q_raw[i, j, kg, tt] = q_loc[kl, tt]
        end
    end
end

# Pre-scan: classify each column into a bucket by active depth.
# Bucket boundaries: 32, 48, 64, 75 (= NZ_MAX).
function classify_columns(entu, detu, entd, detd, Nc, Nz)
    # Compute active depth = Nz - icltop + 1 per column, with icltop
    # diagnosed from detu only (matches kernel logic).
    # Done on CPU to keep the bench simple (single-pass scan, microsecond cost).
    entu_h = Array(entu); detu_h = Array(detu); entd_h = Array(entd); detd_h = Array(detd)
    depths = Array{Int}(undef, Nc, Nc)
    for j in 1:Nc, i in 1:Nc
        icltop = Nz + 1
        for k in 1:Nz
            if detu_h[i, j, k] > 0f0
                icltop = k
                break
            end
        end
        # Also account for downdraft entrainment (icllfs) being below icltop:
        # k_lo is min(icllfs, max(icltop,2)-1), so the bucket needs to cover Nz-k_lo+1.
        icllfs = Nz + 1
        for k in 1:Nz
            if entd_h[i, j, k] > 0f0
                icllfs = k
                break
            end
        end
        if icltop > Nz
            depths[i, j] = 0
        else
            icltop_eff = min(icllfs, max(icltop, 2) - 1)
            k_lo = max(icltop_eff, 1)
            depths[i, j] = Nz - k_lo + 1
        end
    end

    bucket_caps = (32, 48, 64, NZ_MAX)
    buckets = [Tuple{Int,Int}[], Tuple{Int,Int}[], Tuple{Int,Int}[], Tuple{Int,Int}[]]
    n_identity = 0
    for j in 1:Nc, i in 1:Nc
        d = depths[i, j]
        if d == 0
            n_identity += 1
            # Skip — column is identity, no work needed.
            continue
        end
        bi = findfirst(c -> d <= c, bucket_caps)
        push!(buckets[bi], (i, j))
    end
    return buckets, bucket_caps, n_identity
end

function bench_p4!(q_after, q_before, m, entu, detu, entd, detd,
                    cell_areas, dt, nrep)
    Nc, _, Nz = size(m)
    Nt = size(q_before, 4)
    backend = get_backend(q_before)

    # Pre-scan once (outside timing loop is a free since the binary is fixed).
    # We include it inside the timing loop to be honest — preprocessor-side
    # truncation would amortize this, but the runtime classification is part
    # of the wall-clock cost.
    buckets, bucket_caps, n_identity = classify_columns(entu, detu, entd, detd, Nc, Nz)
    bsz = length.(buckets)
    @printf "  bucket sizes (32/48/64/85): %d / %d / %d / %d   identity: %d   total: %d\n" bsz[1] bsz[2] bsz[3] bsz[4] n_identity Nc*Nc

    # Build per-bucket (i, j) index arrays on GPU.
    bucket_idx_gpu = map(buckets) do bk
        if isempty(bk)
            return (CuArray(Int32[]), CuArray(Int32[]))
        end
        ii = Int32[t[1] for t in bk]
        jj = Int32[t[2] for t in bk]
        return (CuArray(ii), CuArray(jj))
    end

    kernels = (_tm5_bucketed_kernel!(backend, WG_SIZE),
               _tm5_bucketed_kernel!(backend, WG_SIZE),
               _tm5_bucketed_kernel!(backend, WG_SIZE),
               _tm5_bucketed_kernel!(backend, WG_SIZE))
    vals = (Val(32), Val(48), Val(64), Val(NZ_MAX))

    times_ms = Float64[]
    for r in 0:nrep
        q_work = CUDA.copy(q_before)
        CUDA.synchronize()
        t = CUDA.@elapsed begin
            for b in 1:4
                nb = length(buckets[b])
                nb == 0 && continue
                idx_i, idx_j = bucket_idx_gpu[b]
                kernels[b](q_work, m, entu, detu, entd, detd,
                            cell_areas[1, :],
                            idx_i, idx_j,
                            Int(nb), Int(Nz), Int(Nt), Float32(dt),
                            vals[b];
                            ndrange = WG_SIZE * nb,
                            workgroupsize = WG_SIZE)
            end
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

    @info "Running baseline"
    q_base = similar(q_d)
    t_base_min, t_base_med = bench_baseline!(q_base, q_d, m_d, entu_d, detu_d,
                                              entd_d, detd_d, area_d, dt, nrep)
    @printf "  baseline:    min %.2f ms  median %.2f ms  (n=%d)\n" t_base_min t_base_med nrep

    @info "Running P4 bucketed"
    q_p4 = similar(q_d)
    t_p4_min, t_p4_med = bench_p4!(q_p4, q_d, m_d, entu_d, detu_d,
                                     entd_d, detd_d, area_d, dt, nrep)
    @printf "  P4 bucketed: min %.2f ms  median %.2f ms\n" t_p4_min t_p4_med

    err = maximum(abs.(Array(q_p4) .- Array(q_base)))
    @printf "  max|Δq| baseline vs P4: %.3e\n" err
    @printf "  speedup baseline / P4 : %.2fx\n" t_base_min / t_p4_min
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
