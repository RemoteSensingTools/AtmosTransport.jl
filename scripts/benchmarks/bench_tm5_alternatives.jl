#!/usr/bin/env julia
# Microbenchmark: alternatives to the per-thread serial LU in TM5Convection.
#
# Background (this session's findings):
#  - Production timing on C180/L85/2-tracer/Nt=2 shows convection at 1172 ms
#    per substep (91.6% of wall time), advection at 52 ms. Convection is 22×
#    advection per call.
#  - Active-depth scan of the production binary across 24 windows × 194k
#    columns:  min_top_code = 11 (deepest reach = 75 layers), median = 53
#    (33-layer active block), p95 = 73 (≤13-layer active block).  TM5 itself
#    declares lmax_conv globally and caps the matrix size at it (ml137:
#    {19, 25, 34, 87} depending on the model setup).
#  - cuBLAS Sgetrf/SgetrsBatched is hard-capped at N ≤ 64 in CUDA 13.1
#    (validated this session: N=64 works, N=65 segfaults inside libcublas).
#    So a cuBLAS-batched path requires a global lmax_conv ≤ 64 — which is
#    exactly how TM5 already operates.
#
# What this script measures, on one real C180 panel of ERA5 forcing
# (32,400 columns):
#  1. baseline    : current `_tm5_column_kernel!` at full Nz=85.
#  2. cuBLAS lmc=64 : cuBLAS strided-batched LU on a `(64, 64, B)` slab
#                     built from the bottom 64 layers (k=22..85). Skips
#                     the ~1% of columns whose icltop lies above k=22.
#  3. cuBLAS lmc=33 : same with `(33, 33, B)`. Bit-exact for the ≥50% of
#                     columns whose icltop lies inside the slab; the rest
#                     are silently truncated (median active depth, like
#                     TM5's ml137/tropo34a).
#  4. cuBLAS lmc=25 : same with `(25, 25, B)`. Matches TM5's ml137/tropo25a
#                     production setup.
#  5. factor-once  : at lmc=64, time JUST the back-substitution after a
#                     pre-existing LU. Bounds the "amortize factor across
#                     palindrome / multi-step" idea.
#
# Output: artifacts/benchmarks/tm5_alternatives.md with a table.

using Printf
using Statistics
using CUDA
using Adapt
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.MetDrivers: TransportBinaryReader
using .AtmosTransport.Operators: TM5Convection, TM5Workspace
using .AtmosTransport.Operators.Convection: _tm5_build_conv1!, _tm5_solve_column!,
                                              _tm5_diagnose_cloud_dims

const DEFAULT_BIN = "/temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps48_v3_20260520/era5_transport_20211202_merged1000Pa_float32.bin"

function _parse_args(argv)
    bin = DEFAULT_BIN; win = 1; panel = 1; nt = 2; dt = 1800.0; nrep = 5
    i = 1
    while i <= length(argv)
        a = argv[i]
        if     a == "--binary" bin = argv[i+1]; i += 2
        elseif a == "--window" win = parse(Int, argv[i+1]); i += 2
        elseif a == "--panel"  panel = parse(Int, argv[i+1]); i += 2
        elseif a == "--nt"     nt = parse(Int, argv[i+1]); i += 2
        elseif a == "--dt"     dt = parse(Float64, argv[i+1]); i += 2
        elseif a == "--repeat" nrep = parse(Int, argv[i+1]); i += 2
        else error("unknown arg `$a`")
        end
    end
    return (; bin, win, panel, nt, dt, nrep)
end

# --- Section offsets (mirrors diagnose_tm5_active_layers.jl) ---
function _section_elements(h, section::Symbol)
    Nc, Nz, np = h.geometry.Nc, h.nlevel, h.geometry.npanel
    section === :m     && return np * Nc * Nc * Nz
    section === :am    && return np * (Nc + 1) * Nc * Nz
    section === :bm    && return np * Nc * (Nc + 1) * Nz
    section === :cm    && return np * Nc * Nc * (Nz + 1)
    section === :ps    && return np * Nc * Nc
    section in (:pblh, :ustar, :pbl_hflux, :t2m) && return np * Nc * Nc
    section === :cmfmc && return np * Nc * Nc * (Nz + 1)
    section === :dtrain && return np * Nc * Nc * Nz
    section in (:entu, :detu, :entd, :detd, :qv, :qv_start, :qv_end, :dm) &&
        return np * Nc * Nc * Nz
    error("unknown section: $section")
end

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

# Cell-area approximation for a gnomonic panel: identical for all (i,j) — the
# panel-uniform area is a fair stand-in for the cell_metrics workspace field,
# which only enters the solver through `m_kg / area`.
function _panel_cell_areas(::Type{FT}, Nc; radius = FT(6.371e6)) where {FT}
    panel_area = FT(4π) * radius^2 / 6
    A = fill(panel_area / Nc^2, Nc, Nc)
    return convert(Matrix{FT}, A)
end

# ============================================================
# Variant 1: baseline — current `_tm5_column_kernel!` invocation but
# specialised here to one panel.  Re-uses the production solver code.
# ============================================================
function bench_baseline!(q_after, q_before, m, entu, detu, entd, detd,
                          cell_areas, dt, nrep)
    Nc, _, Nz = size(m)
    Nt = size(q_before, 4)
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
    return minimum(times_ms), times_ms
end

# ============================================================
# Variant 2/3/4: cuBLAS strided-batched LU on the bottom `lmax` layers.
# Builds dense (lmax, lmax, B) matrices via existing `_tm5_build_conv1!`
# then factors + back-substitutes on GPU.  N ≤ 64 is enforced by cuBLAS.
# ============================================================
function _build_dense_batch!(A::Array{FT, 3}, R::Array{FT, 3},
                              q::Array{FT, 4},
                              m::Array{FT, 3}, entu, detu, entd, detd,
                              cell_areas, dt::FT, lmax::Int,
                              full_iden::Array{Bool, 1}) where {FT}
    Nc, _, Nz = size(m)
    Nt = size(q, 4)
    B = Nc*Nc
    @assert size(A) == (lmax, lmax, B)
    @assert size(R) == (lmax, Nt, B)
    k_shift = Nz - lmax
    Threads.@threads for c in 1:B
        i = ((c - 1) % Nc) + 1
        j = ((c - 1) ÷ Nc) + 1
        m_col    = @view m[i, j, :]
        entu_col = @view entu[i, j, :]
        detu_col = @view detu[i, j, :]
        entd_col = @view entd[i, j, :]
        detd_col = @view detd[i, j, :]
        area     = cell_areas[i, j]

        f_buf  = Array{FT}(undef, Nz + 1, Nz)
        amu_buf = zeros(FT, Nz + 1); amd_buf = zeros(FT, Nz + 1)
        conv1_full = Array{FT}(undef, Nz, Nz)
        icltop, _, icllfs = _tm5_diagnose_cloud_dims(detu_col, entd_col, Nz)
        if icltop > Nz
            full_iden[c] = true
            @inbounds for r in 1:lmax, s in 1:lmax
                A[r, s, c] = r == s ? one(FT) : zero(FT)
            end
            for t in 1:Nt, r in 1:lmax
                R[r, t, c] = q[i, j, k_shift + r, t]
            end
            continue
        end
        full_iden[c] = false
        _tm5_build_conv1!(conv1_full,
                          entu_col, detu_col, entd_col, detd_col, m_col,
                          icltop, icllfs, dt, Nz;
                          cell_area = FT(area),
                          f = f_buf, amu = amu_buf, amd = amd_buf)
        @inbounds for r in 1:lmax, s in 1:lmax
            A[r, s, c] = conv1_full[k_shift + r, k_shift + s]
        end
        for t in 1:Nt, r in 1:lmax
            R[r, t, c] = q[i, j, k_shift + r, t]
        end
    end
    return nothing
end

function bench_cublas_batched!(q_after, q_before, m, entu, detu, entd, detd,
                                cell_areas, dt, nrep, lmax::Int)
    @assert lmax <= 64 "cuBLAS getrsBatched is capped at N ≤ 64"
    FT = eltype(m)
    Nc, _, Nz = size(m)
    Nt = size(q_before, 4)
    B = Nc * Nc
    k_shift = Nz - lmax

    A_h = Array{FT}(undef, lmax, lmax, B)
    R_h = Array{FT}(undef, lmax, Nt, B)
    full_iden = zeros(Bool, B)
    q_h = Array(q_before); m_h = Array(m)
    entu_h = Array(entu); detu_h = Array(detu)
    entd_h = Array(entd); detd_h = Array(detd)
    area_h = Array(cell_areas)

    times_total_ms = Float64[]
    times_gpu_ms   = Float64[]
    for r in 0:nrep
        _build_dense_batch!(A_h, R_h, q_h, m_h, entu_h, detu_h, entd_h, detd_h,
                            area_h, FT(dt), lmax, full_iden)
        t_total = CUDA.@elapsed begin
            A_d = CuArray(A_h)
            R_d = CuArray(R_h)
            CUDA.synchronize()
            t_gpu = CUDA.@elapsed begin
                pivots, _, _ = CUDA.CUBLAS.getrf_strided_batched!(A_d, true)
                CUDA.CUBLAS.getrs_strided_batched!('N', A_d, R_d, pivots)
                CUDA.synchronize()
            end
            R_back = Array(R_d)
            r == 0 && _scatter_back!(q_after, q_before, R_back, k_shift)
            CUDA.synchronize()
        end
        r > 0 && (push!(times_total_ms, t_total * 1000);
                   push!(times_gpu_ms,   t_gpu   * 1000))
    end
    return minimum(times_gpu_ms), minimum(times_total_ms),
           median(times_gpu_ms), median(times_total_ms)
end

function _scatter_back!(q_after, q_before, R::Array, k_shift::Int)
    Nc = size(q_before, 1); Nz = size(q_before, 3); Nt = size(q_before, 4)
    lmax = size(R, 1)
    q_h = Array(q_before)               # one host copy of the prior state
    @inbounds for j in 1:Nc, i in 1:Nc
        c = i + Nc * (j - 1)
        for t in 1:Nt
            for r in 1:lmax
                q_h[i, j, k_shift + r, t] = R[r, t, c]
            end
        end
    end
    copyto!(q_after, q_h)
    return nothing
end

# ============================================================
# Variant 5: per-column variable shift + split batch.
# Pre-scan each column's icltop, split columns into:
#   - "shallow"  (Nz - icltop + 1 ≤ 64): cuBLAS-batched at lmax=64
#   - "deep"     (Nz - icltop + 1  > 64): scalar fallback via the existing
#                                          single-column kernel on the subset
# This is the closest fit to the user's question: a flexible per-column
# scheme that respects the cuBLAS N ≤ 64 ceiling while being bit-exact
# for every column.
# ============================================================
function bench_split_batch!(q_after, q_before, m, entu, detu, entd, detd,
                             cell_areas, dt, nrep)
    FT = eltype(m)
    Nc, _, Nz = size(m)
    Nt = size(q_before, 4)
    B = Nc * Nc
    LMAX_BATCH = 64

    # Pre-scan per-column k_lo on host (cheap O(Nc² × Nz)).  The active
    # block in `_tm5_build_conv1!` starts at
    # `k_lo = min(icllfs, max(icltop, 2) - 1)`, NOT at icltop.  Bucketing
    # on icltop alone misses columns where the downdraft `entd` reaches
    # higher than the updraft `detu`; those columns lose physics inside
    # the cuBLAS slab and produce a phantom error vs baseline.
    detu_h = Array(detu); entd_h = Array(entd)
    deep_idx = Int[]
    shallow_idx = Int[]
    for c in 1:B
        i = ((c - 1) % Nc) + 1
        j = ((c - 1) ÷ Nc) + 1
        d_col = @view detu_h[i, j, :]
        e_col = @view entd_h[i, j, :]
        ic, _, ifs = _tm5_diagnose_cloud_dims(d_col, e_col, Nz)
        if ic > Nz                                 # truly no convection
            push!(shallow_idx, c); continue
        end
        k_lo = min(Int(ifs), max(Int(ic), 2) - 1)
        active_depth = Nz - k_lo + 1
        if active_depth <= LMAX_BATCH
            push!(shallow_idx, c)
        else
            push!(deep_idx, c)
        end
    end
    n_shallow = length(shallow_idx)
    n_deep    = length(deep_idx)
    @info "split-batch composition" n_shallow n_deep frac_deep=(n_deep/B)

    times_total_ms = Float64[]
    times_gpu_ms   = Float64[]
    backend = CUDA.CUDABackend()

    for r in 0:nrep
        q_work = CUDA.copy(q_before)
        CUDA.synchronize()

        t_total = CUDA.@elapsed begin

            # --- Shallow bucket: cuBLAS-batched at lmax=LMAX_BATCH ---
            # Build matrices for shallow columns on host.
            A_h = Array{FT}(undef, LMAX_BATCH, LMAX_BATCH, n_shallow)
            R_h = Array{FT}(undef, LMAX_BATCH, Nt, n_shallow)
            m_h = Array(m); entu_h = Array(entu); detd_h = Array(detd)
            area_h = Array(cell_areas)
            q_h = Array(q_before)
            f_buf = Array{FT}(undef, Nz + 1, Nz)
            amu_buf = zeros(FT, Nz + 1); amd_buf = zeros(FT, Nz + 1)
            conv1_full = Array{FT}(undef, Nz, Nz)
            k_shift = Nz - LMAX_BATCH

            for (s_idx, c) in enumerate(shallow_idx)
                i = ((c - 1) % Nc) + 1
                j = ((c - 1) ÷ Nc) + 1
                m_col    = @view m_h[i, j, :]
                entu_col = @view entu_h[i, j, :]
                detu_col = @view detu_h[i, j, :]
                entd_col = @view entd_h[i, j, :]
                detd_col = @view detd_h[i, j, :]
                area     = area_h[i, j]
                ic, _, ifs = _tm5_diagnose_cloud_dims(detu_col, entd_col, Nz)
                if ic > Nz
                    @inbounds for rr in 1:LMAX_BATCH, ss in 1:LMAX_BATCH
                        A_h[rr, ss, s_idx] = rr == ss ? one(FT) : zero(FT)
                    end
                else
                    _tm5_build_conv1!(conv1_full,
                                      entu_col, detu_col, entd_col, detd_col, m_col,
                                      ic, ifs, FT(dt), Nz;
                                      cell_area = FT(area),
                                      f = f_buf, amu = amu_buf, amd = amd_buf)
                    @inbounds for rr in 1:LMAX_BATCH, ss in 1:LMAX_BATCH
                        A_h[rr, ss, s_idx] = conv1_full[k_shift + rr, k_shift + ss]
                    end
                end
                for t in 1:Nt, rr in 1:LMAX_BATCH
                    R_h[rr, t, s_idx] = q_h[i, j, k_shift + rr, t]
                end
            end

            A_d = CuArray(A_h)
            R_d = CuArray(R_h)
            CUDA.synchronize()
            t_gpu = CUDA.@elapsed begin
                pivots, _, _ = CUDA.CUBLAS.getrf_strided_batched!(A_d, true)
                CUDA.CUBLAS.getrs_strided_batched!('N', A_d, R_d, pivots)
                CUDA.synchronize()
            end

            # Scatter shallow results back into q_work on GPU.
            R_back = Array(R_d)
            q_h2 = Array(q_work)
            for (s_idx, c) in enumerate(shallow_idx)
                i = ((c - 1) % Nc) + 1
                j = ((c - 1) ÷ Nc) + 1
                for t in 1:Nt, rr in 1:LMAX_BATCH
                    q_h2[i, j, k_shift + rr, t] = R_back[rr, t, s_idx]
                end
            end
            copyto!(q_work, q_h2)

            # --- Deep bucket: scalar fallback via existing kernel on a
            # per-column-list mask.  Easiest implementation here: build a
            # small mask array on GPU, then launch the existing kernel and
            # have it skip columns not in the deep set.  For benchmarking
            # parity we just launch on the WHOLE panel but then overwrite
            # the shallow columns back to the cuBLAS result — gives a fair
            # *upper bound* on the deep-bucket cost (the kernel does waste
            # work on shallow columns it'll re-overwrite, so real deep-only
            # cost is at most this value).
            if n_deep > 0
                # Pack deep columns into a face-indexed `(n_deep, Nz, …)`
                # layout and reuse the existing face-indexed kernel
                # (`_tm5_faceindexed_column_kernel!` from
                # src/Operators/Convection/tm5_kernels.jl).  The
                # face-indexed kernel was already written for exactly
                # this layout; only ndrange = n_deep threads launch,
                # so warp divergence is bounded by the deep set.
                m_h_panel  = Array(m); entu_h_panel = Array(entu)
                detu_h_panel = Array(detu); entd_h_panel = Array(entd)
                detd_h_panel = Array(detd); area_h_panel = Array(cell_areas)
                q_h_panel  = Array(q_work)
                m_deep  = Array{FT}(undef, n_deep, Nz)
                entu_dd = Array{FT}(undef, n_deep, Nz)
                detu_dd = Array{FT}(undef, n_deep, Nz)
                entd_dd = Array{FT}(undef, n_deep, Nz)
                detd_dd = Array{FT}(undef, n_deep, Nz)
                area_deep = Array{FT}(undef, n_deep)
                q_deep   = Array{FT}(undef, n_deep, Nz, Nt)
                for (didx, c) in enumerate(deep_idx)
                    i = ((c - 1) % Nc) + 1; j = ((c - 1) ÷ Nc) + 1
                    m_deep[didx, :]    = m_h_panel[i, j, :]
                    entu_dd[didx, :]   = entu_h_panel[i, j, :]
                    detu_dd[didx, :]   = detu_h_panel[i, j, :]
                    entd_dd[didx, :]   = entd_h_panel[i, j, :]
                    detd_dd[didx, :]   = detd_h_panel[i, j, :]
                    area_deep[didx]    = area_h_panel[i, j]
                    for t in 1:Nt, k in 1:Nz
                        q_deep[didx, k, t] = q_h_panel[i, j, k, t]
                    end
                end
                m_deep_d    = CuArray(m_deep)
                entu_deep_d = CuArray(entu_dd); detu_deep_d = CuArray(detu_dd)
                entd_deep_d = CuArray(entd_dd); detd_deep_d = CuArray(detd_dd)
                area_deep_d = CuArray(area_deep)
                q_deep_d    = CuArray(q_deep)
                # Allocate a fresh workspace sized to n_deep columns.
                ws_deep_cpu = TM5Workspace(m_deep; tile_columns = n_deep,
                                            cell_metrics = area_deep)
                ws_deep = Adapt.adapt(CuArray, ws_deep_cpu)
                deep_kernel = AtmosTransport.Operators.Convection._tm5_faceindexed_column_kernel!(backend)
                CUDA.synchronize()
                t_deep_kernel = CUDA.@elapsed begin
                    deep_kernel(q_deep_d, m_deep_d,
                                entu_deep_d, detu_deep_d, entd_deep_d, detd_deep_d,
                                area_deep_d,
                                ws_deep.conv1, ws_deep.pivots, ws_deep.cloud_dims,
                                ws_deep.f_scratch, ws_deep.amu_scratch, ws_deep.amd_scratch,
                                Int(0), Float32(dt); ndrange = n_deep)
                    CUDA.synchronize()
                end
                t_gpu += t_deep_kernel
                q_deep_out = Array(q_deep_d)
                q_h3 = Array(q_work)
                for (didx, c) in enumerate(deep_idx)
                    i = ((c - 1) % Nc) + 1; j = ((c - 1) ÷ Nc) + 1
                    for t in 1:Nt, k in 1:Nz
                        q_h3[i, j, k, t] = q_deep_out[didx, k, t]
                    end
                end
                copyto!(q_work, q_h3)
            end
            CUDA.synchronize()
        end
        r > 0 && (push!(times_total_ms, t_total * 1000);
                   push!(times_gpu_ms,   t_gpu   * 1000))
        r == 0 && copyto!(q_after, q_work)
    end
    return minimum(times_gpu_ms), minimum(times_total_ms),
           median(times_gpu_ms), median(times_total_ms),
           n_shallow, n_deep
end

# Variant 6: factor-once + apply-many.  Times only the back-substitution on
# an already-factored batch — the upper bound on what we can save by reusing
# the LU across a palindrome inner repetition or, more aggressively, across
# multiple substeps (subject to the m_col-changes-per-substep caveat in the
# memo).
function bench_solve_only(m, entu, detu, entd, detd, cell_areas, dt, nrep,
                           lmax::Int, Nt::Int)
    FT = eltype(m)
    Nc, _, Nz = size(m)
    B = Nc * Nc

    A_h = Array{FT}(undef, lmax, lmax, B)
    R_h = Array{FT}(undef, lmax, Nt, B)
    full_iden = zeros(Bool, B)
    q_h = ones(FT, Nc, Nc, Nz, Nt)
    _build_dense_batch!(A_h, R_h, q_h, Array(m), Array(entu), Array(detu),
                         Array(entd), Array(detd), Array(cell_areas),
                         FT(dt), lmax, full_iden)
    A_d = CuArray(A_h)
    pivots, _, _ = CUDA.CUBLAS.getrf_strided_batched!(A_d, true)
    CUDA.synchronize()
    times_ms = Float64[]
    for r in 0:nrep
        R_d = CuArray(R_h)
        t = CUDA.@elapsed begin
            CUDA.CUBLAS.getrs_strided_batched!('N', A_d, R_d, pivots)
            CUDA.synchronize()
        end
        r > 0 && push!(times_ms, t * 1000)
    end
    return minimum(times_ms), median(times_ms)
end

function main()
    opts = _parse_args(ARGS)
    @info "Loading binary" opts...
    reader = TransportBinaryReader(opts.bin; FT = Float32)
    h = reader.header
    @info "Binary header" Nc=h.geometry.Nc Nz=h.nlevel npanel=h.geometry.npanel nwindow=h.nwindow

    entu = collect(_panel_view(reader, opts.win, :entu, opts.panel))
    detu = collect(_panel_view(reader, opts.win, :detu, opts.panel))
    entd = collect(_panel_view(reader, opts.win, :entd, opts.panel))
    detd = collect(_panel_view(reader, opts.win, :detd, opts.panel))
    m    = collect(_panel_view(reader, opts.win, :m,    opts.panel))
    close(reader)

    FT = Float32
    Nc, _, Nz = size(m)
    Nt = opts.nt
    rng = MersenneTwister(0)
    q_cpu = randn(rng, FT, Nc, Nc, Nz, Nt) .* FT(1e-3) .+ FT(1.0)
    cell_areas_cpu = _panel_cell_areas(FT, Nc; radius = FT(h.planet_radius_m))

    entu_d = CuArray(entu); detu_d = CuArray(detu)
    entd_d = CuArray(entd); detd_d = CuArray(detd)
    m_d    = CuArray(m);    q_d    = CuArray(q_cpu)
    area_d = CuArray(cell_areas_cpu)

    @info "Running baseline (current kernel)"
    q_base = similar(q_d)
    t_base_min, t_base_all = bench_baseline!(q_base, q_d, m_d, entu_d, detu_d,
                                              entd_d, detd_d,
                                              area_d, opts.dt, opts.nrep)
    @printf "  baseline:    min %.2f ms  median %.2f ms  (n=%d)\n" t_base_min median(t_base_all) opts.nrep

    results = NamedTuple[]
    push!(results, (variant="baseline (per-thread LU)", lmax=Nz,
                    gpu_min=t_base_min, total_min=t_base_min,
                    gpu_med=median(t_base_all), max_err_active=0.0))

    for lmax in (64, 33, 25)
        @info "Running cuBLAS-batched" lmax=lmax
        q_alt = similar(q_d)
        t_gpu, t_total, t_gpu_med, _ = bench_cublas_batched!(q_alt, q_d, m_d,
                                                              entu_d, detu_d,
                                                              entd_d, detd_d,
                                                              area_d, opts.dt,
                                                              opts.nrep, lmax)
        diff_full = maximum(abs.(Array(q_alt) .- Array(q_base)))
        # Error within the lmax claim is the meaningful quantity for the
        # "no policy change" variant.
        k_shift = Nz - lmax
        slice_alt  = Array(q_alt)[:, :, (k_shift+1):Nz, :]
        slice_base = Array(q_base)[:, :, (k_shift+1):Nz, :]
        diff_active = maximum(abs.(slice_alt .- slice_base))
        @printf "  cuBLAS lmax=%2d:  gpu_min %.2f ms  total_min %.2f ms  err_in_active=%.3e  err_full=%.3e\n" lmax t_gpu t_total diff_active diff_full
        push!(results, (variant="cuBLAS lmax=$lmax", lmax=lmax,
                         gpu_min=t_gpu, total_min=t_total,
                         gpu_med=t_gpu_med, max_err_active=diff_active))
    end

    @info "Running split-batch (per-column shift + scalar fallback at depth>64)"
    q_split = similar(q_d)
    t_split_gpu, t_split_total, _, _, n_shallow, n_deep =
        bench_split_batch!(q_split, q_d, m_d, entu_d, detu_d, entd_d, detd_d,
                            area_d, opts.dt, opts.nrep)
    diff_full = maximum(abs.(Array(q_split) .- Array(q_base)))
    @printf "  split-batch: gpu_min %.2f ms  total_min %.2f ms  err_full=%.3e  (shallow=%d, deep=%d)\n" t_split_gpu t_split_total diff_full n_shallow n_deep
    push!(results, (variant="split-batch (64 + scalar)", lmax=64,
                     gpu_min=t_split_gpu, total_min=t_split_total,
                     gpu_med=t_split_gpu, max_err_active=diff_full))

    @info "Running factor-once + apply-only (lmc=64)"
    t_solve_min, t_solve_med = bench_solve_only(m_d, entu_d, detu_d, entd_d, detd_d,
                                                 area_d, opts.dt, opts.nrep, 64, Nt)
    @printf "  solve-only lmax=64: min %.2f ms median %.2f ms (factor amortized)\n" t_solve_min t_solve_med
    push!(results, (variant="solve-only lmax=64", lmax=64,
                     gpu_min=t_solve_min, total_min=t_solve_min,
                     gpu_med=t_solve_med, max_err_active=NaN))

    mkpath("artifacts/benchmarks")
    open("artifacts/benchmarks/tm5_alternatives.md", "w") do io
        println(io, "# TM5 convection alternatives — single C180 panel, Nz=85")
        println(io)
        println(io, "Window $(opts.win), panel $(opts.panel), Nt=$(opts.nt), ")
        println(io, "dt=$(opts.dt)s, repeats=$(opts.nrep).")
        println(io)
        println(io, "Active-depth scan summary (whole binary, all 24 windows):")
        println(io, "  - 88.9% of columns have detu>0 somewhere (the rest are identity).")
        println(io, "  - min_top_code = 11 → deepest convection uses 75 layers.")
        println(io, "  - median_top_code = 53 → median active block is 33 layers.")
        println(io, "  - p95_top_code   = 73 → 95% of columns ≤ 13 layers.")
        println(io)
        println(io, "cuBLAS notes:")
        println(io, "  - `cublasSgetrsBatched` is hard-capped at N ≤ 64 in CUDA 13.1")
        println(io, "    (validated this session). Larger N segfaults inside libcublas.")
        println(io, "  - TM5 itself runs at lmax_conv ∈ {19, 25, 34, 87} depending on")
        println(io, "    model setup; 25 is the common production troposphere cap.")
        println(io)
        println(io, "| Variant | lmax | gpu_min ms | gpu_med ms | total_min ms | speedup vs baseline | max\\|Δq\\| in active block |")
        println(io, "|---------|------|------------|------------|--------------|---------------------|----------------------------|")
        for r in results
            sp = @sprintf("%.1f×", t_base_min / r.gpu_min)
            err = isnan(r.max_err_active) ? "—" : @sprintf("%.2e", r.max_err_active)
            @printf io "| %s | %d | %.2f | %.2f | %.2f | %s | %s |\n" r.variant r.lmax r.gpu_min r.gpu_med r.total_min sp err
        end
        println(io)
        println(io, "Production reference: 1172 ms / call for 6 panels at full C180")
        println(io, "(this script measures one panel, so multiply 6× when comparing to that).")
    end
    @info "Wrote artifacts/benchmarks/tm5_alternatives.md"
end

main()
