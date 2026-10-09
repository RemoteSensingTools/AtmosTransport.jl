#!/usr/bin/env julia
# Microbenchmark: TM5 convection Schur-complement solver (P8).
#
# Design (from /tmp/tm5_round1_proposals.md P8 + /tmp/tm5_round1_ranking.md):
#
#   Within the active block [k_lo, Nz] of conv1, split into:
#     top    rows [k_lo, icllfs-1]  (size n_T)
#     bottom rows [icllfs, Nz]      (size n_B)
#
#   Inspection of _tm5_build_conv1!:
#     - For row k in [k_lo, icllfs-1] the only non-zero columns are
#       kk in [k-1, Nz].  This makes A_TT (top-left, columns
#       [k_lo, icllfs-1]) upper-HESSENBERG (sub-diagonal at column k-1,
#       coming from the subsidence term -dt*amu[k]/bmass_above).
#     - Top-right A_TB (columns [icllfs, Nz]) is fully populated.
#     - A_BT (bottom rows × top cols) is zero everywhere except at
#       row=icllfs, col=icllfs-1 (the subsidence -amu[icllfs] term);
#       so the Schur correction is a rank-1 update of A_BB.
#     - A_BB is dense.
#
#   Algorithm:
#     1. LU A_TT (Hessenberg, no pivot search needed — one subdiagonal
#        elimination per column).  Costs O(n_T^2).
#     2. Solve A_TT y = b_T (forward + back).  Costs O(n_T^2).
#     3. Form Schur correction: w = A_TT \ A_TB[:, j] for each column
#        j of A_TB, build S = A_BB - A_BT * W.  Because A_BT has only
#        one non-zero (row=icllfs, col=icllfs-1), W is just the last
#        ROW of A_TT^{-1} A_TB (well, the (icllfs-1)th-col-of-A_TT^{-1}
#        times A_TB[icllfs-1, :]).  Actually we need to be careful:
#        A_BT[1, n_T] (i.e. row=icllfs in global, col=icllfs-1 in
#        global = col n_T in local TT space) is the only entry.
#        So Schur S[1, :] -= A_BT[1, n_T] * (A_TT^{-1} A_TB)[n_T, :].
#        Other rows of S equal A_BB.
#     4. Modify RHS: b_B -= A_BT * (A_TT^{-1} b_T) — again only the
#        first entry is affected.
#     5. Collab-LU + solve on S (size n_B × n_B).
#     6. Back out x_T = A_TT^{-1} (b_T - A_TB * x_B).
#
# We do partial pivoting only on the bottom block, which preserves
# numerical stability where it matters (downdraft + subsidence couples).
#
# Usage:
#   julia --project=. scripts/completed_experiments/bench_tm5_p8_schur.jl \
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
# P8 — Schur-complement kernel.
#
# Each workgroup processes one column.  Same matrix-build as the
# collab-LU kernel.  The only difference is in the factor + solve
# phase: split into top (Hessenberg) + bottom (dense) blocks.
#
# To keep the matrix-build bit-faithful to the baseline, we re-use
# the exact build sequence from bench_tm5_collab_lu.jl and only
# replace phases 3 and 4.
# ============================================================
@kernel function _tm5_p8_schur_kernel!(
    q_raw, @Const(air_mass),
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

    for k in t:WG_SIZE:Nz_in
        bmass_loc[k] = air_mass[i, j, k] / area
    end
    @synchronize

    # ---- Phase 1: build A_loc (identical to collab-LU bench) ----
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

    # ---- Phase 2: load RHS -----------------------------------------
    for idx in t:WG_SIZE:(Nz_in * Nt)
        k = (idx - 1) % Nz_in + 1
        tt = (idx - 1) ÷ Nz_in + 1
        q_loc[k, tt] = q_raw[i, j, k, tt]
    end
    @synchronize

    # ---- Phase 3: P8 Schur-complement factor + solve --------------
    # Define block boundaries:
    #   top    = [k_lo, icllfs-1]
    #   bottom = [icllfs, Nz_in]
    # If icllfs <= k_lo, the top block is empty and we degenerate to
    # a single dense LU on [k_lo, Nz_in].  Likewise if icllfs > Nz_in
    # (no downdraft), the top block covers everything and we have an
    # upper-Hessenberg LU only.
    #
    # We pipe through both regimes via the same code path: a Hessenberg
    # LU on [k_lo, icllfs_eff-1] followed by a dense LU on
    # [icllfs_eff, Nz_in], with icllfs_eff = clamp(icllfs, k_lo, Nz_in+1).
    if !no_conv
        icllfs_eff = max(min(icllfs, Nz_in + 1), k_lo)

        # ---------------- Top block: Hessenberg LU ----------------
        # Rows k in [k_lo, icllfs_eff-1] have non-zero entries only at
        # columns [k-1, Nz_in].  Within the TOP-LEFT block (cols
        # [k_lo, icllfs_eff-1]) the structure is upper-Hessenberg.
        # We eliminate the sub-diagonal (k+1, k) for k in [k_lo, icllfs_eff-2]
        # by scaling row k+1 by L = A[k+1,k]/A[k,k] and subtracting L*row_k.
        # No pivot search (Hessenberg + diagonally dominant ⇒ stable).
        if t == 1
            for k in k_lo:(icllfs_eff - 2)
                diag_val = A_loc[k, k]
                Lkp1 = A_loc[k + 1, k] / diag_val
                piv_loc[k] = Int32(k)  # no swap
                # Store multiplier in lower triangle so the Schur step
                # can read it back, mirroring LU storage convention.
                A_loc[k + 1, k] = Lkp1
                # Row update across the full column span [k+1, Nz_in].
                for c in (k + 1):Nz_in
                    A_loc[k + 1, c] -= Lkp1 * A_loc[k, c]
                end
            end
            if icllfs_eff - 1 >= k_lo
                piv_loc[icllfs_eff - 1] = Int32(icllfs_eff - 1)
            end
        end
        @synchronize

        # ---------------- Schur correction on bottom block --------
        # A_BT has at most one non-zero: row = icllfs_eff (the first
        # row of the bottom block) at col = icllfs_eff - 1.  That
        # entry encodes the subsidence term -dt*amu[icllfs_eff]/bmass.
        # After eliminating the Hessenberg sub-diagonals above, the
        # column icllfs_eff-1 of A still has that single entry at row
        # icllfs_eff (no fill-in there, because the Hessenberg LU only
        # writes the L multiplier into A[k+1, k] for k < icllfs_eff-1).
        #
        # We need to solve U_TT * w_col = A_TB[:, col] for each col in
        # [icllfs_eff, Nz_in], then subtract A_BT * W from A_BB.
        # Since A_BT has one non-zero at (row 1 of bottom, col n_T of
        # top), we only need W's LAST ROW: W[n_T, col] = w_col[n_T].
        # That's a single forward+back solve over the top block per
        # bottom column.  But we can also fuse it: with the L unit-
        # lower-bidiagonal stored, the forward solve b_T -> y_T over
        # the top block updates row n_T in one final step.  The back
        # solve then writes w_col[n_T] = y_T[n_T] / U[n_T, n_T].
        #
        # Equivalent (and simpler to vectorize): we just perform the
        # block elimination on the full A_loc directly: for each top
        # row k in [k_lo, icllfs_eff-2], the trailing rows [icllfs_eff, Nz_in]
        # have zero in column k (A_BT structurally sparse), so the
        # standard LU update would skip them.  But the LAST top row
        # k = icllfs_eff - 1 has the bottom block's first row coupled
        # via A[icllfs_eff, icllfs_eff - 1] (subsidence).  We eliminate
        # that sub-diagonal entry to fold the Schur correction into
        # A_BB in one row sweep, parallel across columns.
        n_T = icllfs_eff - k_lo
        n_B = Nz_in - icllfs_eff + 1
        if n_T > 0 && n_B > 0
            k_T_last = icllfs_eff - 1
            diag_last_T = A_loc[k_T_last, k_T_last]
            # Lbt = A[icllfs_eff, k_T_last] / diag_last_T — scalar.
            if t == 1
                Lbt = A_loc[icllfs_eff, k_T_last] / diag_last_T
                A_loc[icllfs_eff, k_T_last] = Lbt  # store multiplier
            end
            @synchronize
            Lbt = A_loc[icllfs_eff, k_T_last]
            # Parallel column sweep across [icllfs_eff, Nz_in].
            for c in (icllfs_eff + t - 1):WG_SIZE:Nz_in
                A_loc[icllfs_eff, c] -= Lbt * A_loc[k_T_last, c]
            end
            @synchronize
        end

        # ---------------- Bottom block: dense LU with partial pivot
        # Standard right-looking LU on [icllfs_eff, Nz_in] × [icllfs_eff, Nz_in].
        # Need to ALSO track which top columns get fill-in via row swaps:
        # NONE, because A_BT[1:n_B, 1:n_T-1] is structurally zero (only
        # column n_T of top has any bottom coupling, and we just folded
        # that into A_BB).  So pivots within bottom block can swap rows
        # within [icllfs_eff, Nz_in] without disturbing top-block entries.
        for k in icllfs_eff:Nz_in
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
                # Swap row k and piv over column span [k_lo, Nz_in]
                # (need to include all top columns too because the
                # top-row-elim above wrote multipliers into row icllfs_eff
                # at column icllfs_eff-1, which travels with the row).
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

        # ---------------- Solve phase ----------------
        # Apply top-block "pivots" (none — they're identity) and then
        # the unified forward solve.  L is unit-lower with multipliers
        # at A[k+1,k] for top rows, and standard LU L below icllfs_eff.
        #
        # First apply the bottom-block pivots to the RHS.
        if t == 1
            for k in icllfs_eff:Nz_in
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

        # Forward solve over [k_lo, Nz_in], parallel across tracers.
        # Within top block [k_lo, icllfs_eff-1], L is unit-lower-bidiagonal:
        # only A[k, k-1] is a multiplier (for k > k_lo).  Within bottom
        # block [icllfs_eff, Nz_in], L is dense lower triangular.
        # Boundary row icllfs_eff has multiplier at column icllfs_eff-1
        # (from the Schur fold above) plus dense L below.
        for tt in t:WG_SIZE:Nt
            for k in k_lo:Nz_in
                s = q_loc[k, tt]
                if k < icllfs_eff
                    # Top block: only A[k, k-1] is L; A[k, k-1] for k > k_lo.
                    if k > k_lo
                        s -= A_loc[k, k - 1] * q_loc[k - 1, tt]
                    end
                else
                    # Bottom block (or icllfs_eff exactly): dense L over
                    # [icllfs_eff-1, k-1] if k == icllfs_eff (one L at
                    # k-1 == icllfs_eff-1), else dense over [icllfs_eff, k-1].
                    j_start = k == icllfs_eff ? icllfs_eff - 1 : icllfs_eff
                    for j2 in j_start:(k - 1)
                        s -= A_loc[k, j2] * q_loc[j2, tt]
                    end
                end
                q_loc[k, tt] = s
            end
        end
        @synchronize

        # Back solve over [k_lo, Nz_in].  Top block U is upper triangular
        # within columns [k_lo, icllfs_eff-1] but ALSO has entries in
        # bottom columns [icllfs_eff, Nz_in] (A_TB is fully populated).
        # Bottom block is plain upper triangular.
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

    # ---- Phase 5: store back ---------------------------------------
    for idx in t:WG_SIZE:(Nz_in * Nt)
        k = (idx - 1) % Nz_in + 1
        tt = (idx - 1) ÷ Nz_in + 1
        q_raw[i, j, k, tt] = q_loc[k, tt]
    end
end

function bench_p8!(q_after, q_before, m, entu, detu, entd, detd,
                    cell_areas, dt, nrep)
    Nc, _, Nz = size(m)
    Nt = size(q_before, 4)
    backend = get_backend(q_before)
    kernel = _tm5_p8_schur_kernel!(backend, WG_SIZE)
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

    @info "Running P8 Schur"
    q_p8 = similar(q_d)
    t_p8_min, t_p8_med = bench_p8!(q_p8, q_d, m_d, entu_d, detu_d,
                                     entd_d, detd_d, area_d, dt, nrep)
    @printf "  P8 Schur:    min %.2f ms  median %.2f ms\n" t_p8_min t_p8_med

    err = maximum(abs.(Array(q_p8) .- Array(q_base)))
    @printf "  max|Δq| baseline vs P8: %.3e\n" err
    @printf "  speedup baseline / P8 : %.2fx\n" t_base_min / t_p8_min
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
