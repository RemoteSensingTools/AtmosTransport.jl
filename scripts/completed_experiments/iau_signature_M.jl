# ===========================================================================
# Is the MFXC↔DELP continuity residual M the IAU ANALYSIS INCREMENT, or
# sub-hourly advective accumulation noise?  A LOCAL-ONLY temporal-signature
# test (needs only CTM_A1, which we already have for all of Dec 2021).
#
# Logic (from the GFD review + workflow wsz2yeiah):
#   GEOS applies the analysis increment via IAU as a ~constant tendency over a
#   6-h corrector window centered on 00/06/12/18Z (Bloom 1996; Takacs 2018).
#   If M is that increment, the SPATIAL PATTERN of M(i,j) should stay highly
#   correlated hour-to-hour WITHIN a 6-h block and DECORRELATE across block
#   boundaries -> the temporal autocorrelation of M stays high to lag ~5-6 then
#   drops.  If M is sub-hourly dynamics lost to hourly archiving, each hour's
#   accumulation residual is ~independent -> M is temporally WHITE (autocorr
#   collapses at lag 1).
#
#   M[i,j] computed exactly as moist_budget_IT_vs_FP.jl:
#     M = Σ_k (DELP_{w+1}-DELP_w)[k]·area/g/(2·steps) − Σ_k div_h(am,bm)[k]
#     am=MFXC/(2g) face flux; MOIST DELP (dry offset common, normalized out).
#
#   julia --project=. scripts/completed_experiments/iau_signature_M.jl [YYYY-MM-DD]
#
# Optional: if a GES DISC GEOSIT_ASM_I1_C_SLV file with DMDTANA/DQDTANAINT is
# present (see SLV_FILE below), also runs the column M-vs-(DMDTANA−DQDTANAINT)
# spatial-correlation test — the stronger ground-truth probe.
# ===========================================================================
using AtmosTransport, AtmosTransport.Preprocessing, AtmosTransport.Grids
using Dates, NCDatasets, Statistics, Printf, TOML
const P = AtmosTransport.Preprocessing
const GRAV = 9.80665; const DT_MET = 3600.0; const MFDT = 450.0
const CONFIG = joinpath(@__DIR__, "..", "..",
    "config/preprocessing/geosit_c180_dec2021_catrine_f32_fullL72.toml")
const RAW = expanduser("~/data/AtmosTransport/met/geosit/C180/raw_catrine")
const DATE = length(ARGS) >= 1 ? Date(ARGS[1]) : Date(2021, 12, 1)
ctm_path(d::Date) = joinpath(RAW, Dates.format(d, "yyyymmdd"),
    "GEOSIT.$(Dates.format(d,"yyyymmdd")).CTM_A1.C180.nc")
# Optional SLV file (GES DISC) for the DMDTANA correlation; leave as-is if absent.
const SLV_FILE = joinpath(RAW, Dates.format(DATE, "yyyymmdd"),
    "GEOSIT_ASM_I1_C_SLV.$(Dates.format(DATE,"yyyymmdd")).nc")

function setup(FT)
    cfg  = TOML.parsefile(CONFIG)
    grid = P.build_target_geometry(cfg["grid"], FT); Nc = grid.Nc
    vc   = P.load_hybrid_coefficients(AtmosTransport.expand_data_path(String(cfg["vertical"]["coefficients"])))
    Nz   = length(vc.A) - 1
    lats = ntuple(p -> Float64.(P.panel_cell_center_lonlat(grid.mesh, p)[2]), 6)
    sh   = ntuple(p -> lats[p] .< -30.0, 6)
    (grid=grid, Nc=Nc, Nz=Nz, sh=sh,
     areas=grid.mesh.cell_areas, conn=grid.mesh.connectivity)
end

# M[i,j] for one window, given DELP endpoints (dc,dn) and window-mean MFXC/MFYC.
function window_M(s, dc, dn, mfxc, mfyc, FT)
    g = FT(GRAV); steps = round(Int, DT_MET/MFDT); twosteps = FT(2*steps)
    fs = FT(1/(2g)); Nc = s.Nc; Nz = s.Nz
    am = ntuple(_ -> zeros(FT, Nc+1, Nc, Nz), 6)
    bm = ntuple(_ -> zeros(FT, Nc, Nc+1, Nz), 6)
    P.geos_native_to_face_flux!(am, bm, mfxc, mfyc, s.conn, Nc, Nz, fs)
    M = ntuple(_ -> zeros(FT, Nc, Nc), 6); colm = zero(FT)
    for p in 1:6, j in 1:Nc, i in 1:Nc
        a = s.areas[i,j]; pit = zero(FT); sdm = zero(FT); cm = zero(FT)
        @inbounds for k in 1:Nz
            pit += (am[p][i,j,k]-am[p][i+1,j,k]) + (bm[p][i,j,k]-bm[p][i,j+1,k])
            sdm += (dn[p][i,j,k]-dc[p][i,j,k])*a/g/twosteps
            cm  += dc[p][i,j,k]*a/g
        end
        M[p][i,j] = sdm - pit
        colm = max(colm, cm)
    end
    return M, colm
end

shvec(M, sh) = vcat([vec(M[p][sh[p]]) for p in 1:6]...)

function main()
    FT = Float64; s = setup(FT)
    @printf("IAU-signature test for M  —  %s  (Nc=%d Nz=%d)\n", DATE, s.Nc, s.Nz)

    f0 = ctm_path(DATE); f1 = ctm_path(DATE + Day(1))
    isfile(f0) || error("missing $f0")
    ds0 = NCDataset(f0, "r"); or = P.detect_level_orientation(ds0)
    nt  = ds0.dim["time"]
    have_next = isfile(f1)
    ds1 = have_next ? NCDataset(f1, "r") : nothing
    nwin = have_next ? nt : nt - 1
    @printf("  CTM_A1 windows = %d  (next-day wrap: %s)\n\n", nwin, have_next ? "yes" : "no — last window dropped")

    Ms = Vector{NTuple{6,Matrix{FT}}}(undef, nwin)
    rmsM = zeros(nwin); colmass = 0.0
    dc = P._read_panels_3d(ds0, "DELP", 1, or; FT=FT)
    for w in 1:nwin
        mfxc = P._read_panels_3d(ds0, "MFXC", w, or; FT=FT)
        mfyc = P._read_panels_3d(ds0, "MFYC", w, or; FT=FT)
        dn = w < nt ? P._read_panels_3d(ds0, "DELP", w+1, or; FT=FT) :
                      P._read_panels_3d(ds1, "DELP", 1,   or; FT=FT)
        M, colm = window_M(s, dc, dn, mfxc, mfyc, FT)
        Ms[w] = M; colmass = max(colmass, colm)
        rmsM[w] = sqrt(mean(shvec(M, s.sh).^2))
        dc = dn
    end
    close(ds0); have_next && close(ds1)

    # --- (1) magnitude vs clock hour (window w begins ~ (w-1):30 UTC) ---
    println("=== RMS|M|/colmass by window (SH, lat<-30) ===")
    println("  win  ~UTC   RMS|M|/colmass")
    for w in 1:nwin
        @printf("  %3d  %02d:30   %.3e\n", w, w-1, rmsM[w]/colmass)
    end
    cv = std(rmsM)/mean(rmsM)
    @printf("  magnitude CV across windows = %.2f  (flat ⇒ steady forcing; spiky ⇒ episodic)\n\n", cv)

    # --- (2) temporal autocorrelation of the SH pattern of M ---
    shM = [shvec(Ms[w], s.sh) for w in 1:nwin]
    println("=== temporal autocorr of M's SH spatial pattern  ρ(lag) ===")
    println("  IAU 6-h block ⇒ ρ high to lag~5 then drops;  advective noise ⇒ ρ collapses at lag 1")
    println("  lag   mean ρ(M_w, M_{w+lag})   [over SH cells]")
    maxlag = min(12, nwin-1)
    rho = zeros(maxlag+1)
    for lag in 0:maxlag
        acc = Float64[]
        for w in 1:(nwin-lag)
            push!(acc, cor(shM[w], shM[w+lag]))
        end
        rho[lag+1] = mean(acc)
        @printf("  %3d     %+.3f\n", lag, rho[lag+1])
    end

    # --- (3) within- vs across-6h-block lag-1 correlation ---
    # synoptic IAU windows: 21-03, 03-09, 09-15, 15-21 (centered 00/06/12/18Z).
    # window w begins (w-1):30; block boundary crossings at UTC 03,09,15,21.
    block(w) = fld(mod(w-1+3, 24), 6)   # 0..3 label; boundaries at 03/09/15/21:30
    within = Float64[]; across = Float64[]
    for w in 1:(nwin-1)
        ρ = cor(shM[w], shM[w+1])
        push!(block(w) == block(w+1) ? within : across, ρ)
    end
    @printf("\n  lag-1 ρ  WITHIN 6-h block  = %+.3f  (n=%d)\n", mean(within), length(within))
    @printf("  lag-1 ρ  ACROSS block edge = %+.3f  (n=%d)\n", isempty(across) ? NaN : mean(across), length(across))
    drop = mean(within) - (isempty(across) ? NaN : mean(across))
    @printf("  within−across = %+.3f   (large positive ⇒ IAU 6-h block signature)\n", drop)

    # --- verdict heuristic ---
    println("\n=== heuristic verdict ===")
    if rho[2] > 0.6 && rho[min(6,maxlag+1)] > 0.4
        println("  M's pattern is temporally PERSISTENT (high autocorr) ⇒ consistent with an")
        println("  IAU analysis increment (a slowly-varying forcing), NOT sub-hourly noise.")
    elseif rho[2] < 0.25
        println("  M's pattern is temporally WHITE (autocorr collapses at lag 1) ⇒ consistent")
        println("  with sub-hourly advective accumulation residual, NOT a 6-h IAU increment.")
    else
        println("  Intermediate persistence — mixed / inconclusive from the temporal test alone;")
        println("  the DMDTANA correlation (GES DISC GEOSIT_ASM_I1_C_SLV) is the decisive probe.")
    end

    # --- (4) optional: M vs archived dry analysis increment (DMDTANA−DQDTANAINT) ---
    if isfile(SLV_FILE)
        println("\n=== M vs archived dry analysis increment (DMDTANA−DQDTANAINT) ===")
        slv = NCDataset(SLV_FILE, "r")
        # DMDTANA, DQDTANAINT: kg m-2 s-1, instantaneous top-of-hour, 2D on C180.
        # Put on the same per-window-per-cell mass basis as M (kg/cell/window):
        #   σ_dry[i,j] ≈ (DMDTANA − DQDTANAINT)·area·DT_MET   (window-mean ≈ avg of w,w+1)
        function slv_window(w)
            dm  = P._read_panels_2d(slv, "DMDTANA",   w; FT=FT)
            dq  = P._read_panels_2d(slv, "DQDTANAINT", w; FT=FT)
            ntuple(p -> (dm[p] .- dq[p]) .* s.areas .* FT(DT_MET), 6)
        end
        for w in (1, nwin ÷ 2, nwin)
            σ = slv_window(w)
            ρ  = cor(shvec(Ms[w], s.sh), shvec(σ, s.sh))
            rr = sqrt(mean(shvec(σ, s.sh).^2)) / sqrt(mean(shvec(Ms[w], s.sh).^2))
            @printf("  win %3d:  SH cor(M, σ_dry) = %+.3f   RMS ratio |σ|/|M| = %.2f\n", w, ρ, rr)
        end
        close(slv)
        println("  cor>0.7 & ratio~O(1) ⇒ M IS the IAU increment ⇒ treat as non-advective source.")
    else
        println("\n(DMDTANA file absent: $SLV_FILE)")
        println(" To run the decisive ground-truth probe, fetch GES DISC GEOSIT_ASM_I1_C_SLV")
        println(" (DMDTANA + DQDTANAINT, ~18MB/granule, native C180) for $DATE and re-run.")
    end
end
main()
