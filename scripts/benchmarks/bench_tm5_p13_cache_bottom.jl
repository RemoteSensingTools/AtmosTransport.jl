#!/usr/bin/env julia
# Microbenchmark: TM5 convection — cache ONLY the bottom Schur block (P13 = P6 ∘ P8).
#
# Round-2 pick #1 from /tmp/tm5_round2_proposals.md and /tmp/tm5_round2_ranking.md.
#
# Strategy:
#   * Build A_loc fully on every substep (cheap, O(Nz²)).
#   * On the MISS path: Hessenberg LU on the top block + Schur fold +
#     dense partial-pivot LU on the bottom block (size n_B = Nz - icllfs_eff + 1).
#     Persist ONLY the factored bottom block (n_B_max × n_B_max per column)
#     and its pivots.  Cap n_B_max at a known global maximum
#     (probed via _probe_panel_depths.jl: max n_B on the production binary is 53).
#   * On the HIT path: re-build A_loc from raw inputs, redo Hessenberg LU
#     on the top + Schur fold, then load the cached bottom factor + pivots
#     into the shared-memory slots and skip straight to the solve.
#
# Memory: cache is (n_B_max, n_B_max, Nc, Nc) Float32 + (n_B_max, Nc, Nc) Int32
#   For Nc=180, n_B_max=60: 60²·180²·4 = 467 MB (vs P6's 938 MB).
#
# APPROX_PHYSICS (inherits P6 staleness assumption — forcings fixed within window).
#
# Usage:
#   julia --project=. scripts/benchmarks/bench_tm5_p13_cache_bottom.jl \
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
const NB_MAX = 60   # global cap on bottom-block size (probed max = 53 on production binary).
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
# Shared helper: build A_loc, do Hessenberg LU on the top block + Schur
# fold (in-place on A_loc), record icllfs_eff / k_lo.  Returns nothing —
# operates on shared memory.
#
# This is the body that BOTH miss and hit kernels share, copy-pasted
# inline below.  We keep them inline (rather than calling a device
# function) because @kernel + @localmem doesn't play nicely with
# function calls across the boundary.
# ============================================================

@kernel function _tm5_p13_miss_kernel!(
    q_raw,
    cache_B,         # (NB_MAX, NB_MAX, Nc, Nc) factored bottom block.
    cache_piv_B,     # (NB_MAX, Nc, Nc) pivots, indexed 1..n_B (local).
    cache_klo,       # (Nc, Nc) global k_lo.
    cache_icllfs,    # (Nc, Nc) global icllfs_eff.
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
    icllfs_eff = max(min(icllfs, Nz_in + 1), k_lo)

    if t == 1
        cache_klo[i, j] = Int32(no_conv ? Nz_in + 1 : k_lo)
        cache_icllfs[i, j] = Int32(icllfs_eff)
    end

    for k in t:WG_SIZE:Nz_in
        bmass_loc[k] = air_mass[i, j, k] / area
    end
    @synchronize

    # ---- Build A_loc (identical to baseline collab-LU) ----
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

    # ---- Load RHS ----
    for idx in t:WG_SIZE:(Nz_in * Nt)
        k = (idx - 1) % Nz_in + 1
        tt = (idx - 1) ÷ Nz_in + 1
        q_loc[k, tt] = q_raw[i, j, k, tt]
    end
    @synchronize

    # ---- Hessenberg LU on top block + Schur fold + bottom dense LU ----
    if !no_conv
        # Hessenberg LU on top block — sub-diagonal elimination per col.
        if t == 1
            for k in k_lo:(icllfs_eff - 2)
                diag_val = A_loc[k, k]
                Lkp1 = A_loc[k + 1, k] / diag_val
                piv_loc[k] = Int32(k)
                A_loc[k + 1, k] = Lkp1
                for c in (k + 1):Nz_in
                    A_loc[k + 1, c] -= Lkp1 * A_loc[k, c]
                end
            end
            if icllfs_eff - 1 >= k_lo
                piv_loc[icllfs_eff - 1] = Int32(icllfs_eff - 1)
            end
        end
        @synchronize

        # Schur fold: collapse the one non-zero of A_BT into A_BB row 1.
        n_T = icllfs_eff - k_lo
        n_B = Nz_in - icllfs_eff + 1
        if n_T > 0 && n_B > 0
            k_T_last = icllfs_eff - 1
            if t == 1
                diag_last_T = A_loc[k_T_last, k_T_last]
                Lbt = A_loc[icllfs_eff, k_T_last] / diag_last_T
                A_loc[icllfs_eff, k_T_last] = Lbt
            end
            @synchronize
            Lbt = A_loc[icllfs_eff, k_T_last]
            for c in (icllfs_eff + t - 1):WG_SIZE:Nz_in
                A_loc[icllfs_eff, c] -= Lbt * A_loc[k_T_last, c]
            end
            @synchronize
        end

        # Dense LU on bottom block with partial pivoting.
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

        # ---- Solve phase (full solve for the miss path's tracer) ----
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

        for tt in t:WG_SIZE:Nt
            for k in k_lo:Nz_in
                s = q_loc[k, tt]
                if k < icllfs_eff
                    if k > k_lo
                        s -= A_loc[k, k - 1] * q_loc[k - 1, tt]
                    end
                else
                    j_start = k == icllfs_eff ? icllfs_eff - 1 : icllfs_eff
                    for j2 in j_start:(k - 1)
                        s -= A_loc[k, j2] * q_loc[j2, tt]
                    end
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

    # ---- Persist ONLY the bottom block + its pivots (local index basis) ----
    if !no_conv
        n_B = Nz_in - icllfs_eff + 1
        # Store A[icllfs_eff:Nz_in, icllfs_eff:Nz_in] into cache_B at local indices 1..n_B.
        for idx in t:WG_SIZE:(n_B * n_B)
            lk = (idx - 1) % n_B + 1
            lkk = (idx - 1) ÷ n_B + 1
            gk = lk + icllfs_eff - 1
            gkk = lkk + icllfs_eff - 1
            cache_B[lk, lkk, i, j] = A_loc[gk, gkk]
        end
        # Pivots stored in LOCAL index space: piv_loc holds global k; we store (piv - icllfs_eff + 1).
        for lk in t:WG_SIZE:n_B
            gk = lk + icllfs_eff - 1
            piv_global = Int(piv_loc[gk])
            cache_piv_B[lk, i, j] = Int32(piv_global - icllfs_eff + 1)
        end
    end

    # ---- Store back tracer.
    for idx in t:WG_SIZE:(Nz_in * Nt)
        k = (idx - 1) % Nz_in + 1
        tt = (idx - 1) ÷ Nz_in + 1
        q_raw[i, j, k, tt] = q_loc[k, tt]
    end
end

@kernel function _tm5_p13_hit_kernel!(
    q_raw,
    @Const(cache_B),
    @Const(cache_piv_B),
    @Const(cache_klo),
    @Const(cache_icllfs),
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

    # Re-diagnose (or read from cache).  Reading icltop / icllfs from cache
    # would still require a re-scan for the build below, so we re-scan.
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
    icllfs_eff = max(min(icllfs, Nz_in + 1), k_lo)

    for k in t:WG_SIZE:Nz_in
        bmass_loc[k] = air_mass[i, j, k] / area
    end
    @synchronize

    # ---- Re-build A_loc (same as miss) ----
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

    # ---- Load RHS ----
    for idx in t:WG_SIZE:(Nz_in * Nt)
        k = (idx - 1) % Nz_in + 1
        tt = (idx - 1) ÷ Nz_in + 1
        q_loc[k, tt] = q_raw[i, j, k, tt]
    end
    @synchronize

    if !no_conv
        # Hessenberg LU on top — REDONE freshly (no cache).
        if t == 1
            for k in k_lo:(icllfs_eff - 2)
                diag_val = A_loc[k, k]
                Lkp1 = A_loc[k + 1, k] / diag_val
                piv_loc[k] = Int32(k)
                A_loc[k + 1, k] = Lkp1
                for c in (k + 1):Nz_in
                    A_loc[k + 1, c] -= Lkp1 * A_loc[k, c]
                end
            end
            if icllfs_eff - 1 >= k_lo
                piv_loc[icllfs_eff - 1] = Int32(icllfs_eff - 1)
            end
        end
        @synchronize

        # Schur fold.
        n_T = icllfs_eff - k_lo
        n_B = Nz_in - icllfs_eff + 1
        if n_T > 0 && n_B > 0
            k_T_last = icllfs_eff - 1
            if t == 1
                diag_last_T = A_loc[k_T_last, k_T_last]
                Lbt = A_loc[icllfs_eff, k_T_last] / diag_last_T
                A_loc[icllfs_eff, k_T_last] = Lbt
            end
            @synchronize
            Lbt = A_loc[icllfs_eff, k_T_last]
            for c in (icllfs_eff + t - 1):WG_SIZE:Nz_in
                A_loc[icllfs_eff, c] -= Lbt * A_loc[k_T_last, c]
            end
            @synchronize
        end

        # ---- Splice in the cached bottom factor + pivots ----
        # Overwrite A_loc[icllfs_eff:Nz_in, icllfs_eff:Nz_in] with cache_B (n_B × n_B),
        # but PRESERVE the A_loc[icllfs_eff, icllfs_eff-1] Schur multiplier we just wrote
        # (which is INDEXED at gk = icllfs_eff and gc = icllfs_eff - 1; cache_B doesn't
        # touch that column).
        for idx in t:WG_SIZE:(n_B * n_B)
            lk = (idx - 1) % n_B + 1
            lkk = (idx - 1) ÷ n_B + 1
            gk = lk + icllfs_eff - 1
            gkk = lkk + icllfs_eff - 1
            A_loc[gk, gkk] = cache_B[lk, lkk, i, j]
        end
        # Pivots stored locally — translate back into global index basis used by the solve.
        for lk in t:WG_SIZE:n_B
            gk = lk + icllfs_eff - 1
            cached = Int(cache_piv_B[lk, i, j])
            piv_loc[gk] = Int32(cached + icllfs_eff - 1)
        end
        @synchronize

        # ---- Solve (identical to miss path) ----
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

        for tt in t:WG_SIZE:Nt
            for k in k_lo:Nz_in
                s = q_loc[k, tt]
                if k < icllfs_eff
                    if k > k_lo
                        s -= A_loc[k, k - 1] * q_loc[k - 1, tt]
                    end
                else
                    j_start = k == icllfs_eff ? icllfs_eff - 1 : icllfs_eff
                    for j2 in j_start:(k - 1)
                        s -= A_loc[k, j2] * q_loc[j2, tt]
                    end
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

# ============================================================
# Collab-LU reference (for cross-check) — re-uses the production
# `_tm5_column_kernel!`.
# ============================================================
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

function bench_p13_miss!(q_after, q_before, cache_B, cache_piv_B, cache_klo, cache_icllfs,
                          m, entu, detu, entd, detd, cell_areas, dt, nrep)
    Nc, _, Nz = size(m)
    Nt = size(q_before, 4)
    backend = get_backend(q_before)
    kernel = _tm5_p13_miss_kernel!(backend, WG_SIZE)
    times_ms = Float64[]
    for r in 0:nrep
        q_work = CUDA.copy(q_before)
        CUDA.synchronize()
        t = CUDA.@elapsed begin
            kernel(q_work, cache_B, cache_piv_B, cache_klo, cache_icllfs,
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

function bench_p13_hit!(q_after, q_before, cache_B, cache_piv_B, cache_klo, cache_icllfs,
                         m, entu, detu, entd, detd, cell_areas, dt, nrep)
    Nc, _, Nz = size(m)
    Nt = size(q_before, 4)
    backend = get_backend(q_before)
    kernel = _tm5_p13_hit_kernel!(backend, WG_SIZE)
    times_ms = Float64[]
    for r in 0:nrep
        q_work = CUDA.copy(q_before)
        CUDA.synchronize()
        t = CUDA.@elapsed begin
            kernel(q_work, cache_B, cache_piv_B, cache_klo, cache_icllfs,
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

    # Bottom-block cache buffers (n_B_max cap = NB_MAX).
    cache_B      = CuArray{Float32}(undef, NB_MAX, NB_MAX, Nc, Nc)
    cache_piv_B  = CuArray{Int32}(undef, NB_MAX, Nc, Nc)
    cache_klo    = CuArray{Int32}(undef, Nc, Nc)
    cache_icllfs = CuArray{Int32}(undef, Nc, Nc)
    @info "Allocated cache" cache_B_bytes=length(cache_B)*4

    @info "Running baseline"
    q_base = similar(q_d)
    t_base_min, t_base_med = bench_baseline!(q_base, q_d, m_d, entu_d, detu_d,
                                              entd_d, detd_d, area_d, dt, nrep)
    @printf "  baseline:    min %.2f ms  median %.2f ms  (n=%d)\n" t_base_min t_base_med nrep

    @info "Running P13 miss (build + factor + solve, populate bottom cache)"
    q_miss = similar(q_d)
    t_miss_min, t_miss_med = bench_p13_miss!(q_miss, q_d, cache_B, cache_piv_B,
                                              cache_klo, cache_icllfs,
                                              m_d, entu_d, detu_d, entd_d, detd_d,
                                              area_d, dt, nrep)
    @printf "  P13 miss:    min %.2f ms  median %.2f ms\n" t_miss_min t_miss_med

    err_miss = maximum(abs.(Array(q_miss) .- Array(q_base)))
    @printf "  max|Δq| baseline vs P13 miss: %.3e\n" err_miss

    @info "Running P13 hit (rebuild top + load bottom cache + solve)"
    q_hit = similar(q_d)
    t_hit_min, t_hit_med = bench_p13_hit!(q_hit, q_d, cache_B, cache_piv_B,
                                           cache_klo, cache_icllfs,
                                           m_d, entu_d, detu_d, entd_d, detd_d,
                                           area_d, dt, nrep)
    @printf "  P13 hit:     min %.2f ms  median %.2f ms\n" t_hit_min t_hit_med

    err_hit_miss = maximum(abs.(Array(q_hit) .- Array(q_miss)))
    @printf "  max|Δq| miss vs hit (same input): %.3e\n" err_hit_miss
    err_hit_base = maximum(abs.(Array(q_hit) .- Array(q_base)))
    @printf "  max|Δq| baseline vs P13 hit: %.3e\n" err_hit_base

    @printf "  speedup baseline / P13 miss: %.2fx\n" t_base_min / t_miss_min
    @printf "  speedup baseline / P13 hit : %.2fx\n" t_base_min / t_hit_min
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
