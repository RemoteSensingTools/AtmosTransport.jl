# ===========================================================================
# FINGERFIX PROTOTYPE — id: two-window-time-averaged-flux
#
# Hypothesis: the MFXC<->DELP continuity residual M (~5.6e-4 colmass, grid-noisy,
# SH-UTLS-concentrated) that the offline closure forces into cm is a TEMPORAL /
# hour-mean-SAMPLING residual — the difference between (hour-mean DELP tendency)
# and (hour-mean MFXC convergence). Sub-hourly dynamics aliased into the hourly
# archive show up as a FIRST-ORDER-in-time error in the single-window budget. If
# so, a TIME-SYMMETRIC (trapezoidal) flux average + a CENTERED DELP tendency,
# both valid at the SAME mid-time, cancels the first-order sampling term and
# produces a smoother div_h (=vdiv) at SH-UTLS.
#
# METHOD (window w, the "center" half-step):
#   am,bm  <- geos_native_to_face_flux! on  MFXC_bar = 0.5*(MFXC_w + MFXC_{w+1})
#                                            MFYC_bar = 0.5*(MFYC_w + MFYC_{w+1})
#   centered mass tendency: m_next_target - m_cur  ~  (DELP_{w+1} - DELP_{w-1})/2,
#     i.e. m_cur = mass(DELP_w),  m_next_target = mass(DELP_w) + 0.5*(mass(DELP_{w+1})-mass(DELP_{w-1}))
#     (the trapezoidal flux is valid at the boundary between half-steps centered on w;
#      the centered difference is the matching second-order tendency at w.)
#   balance_cs_column_mass_fluxes!  (production Poisson balance to the centered target)
#   fill_cs_window_mass_tendency!   (dm = (m_next_target - m_cur)/(2 steps))
#   diagnose_cs_cm!                 (cm closes continuity EXACTLY by construction)
#
# CONTROL (DIRTY, same window): standard endpoint closure
#   am,bm  <- MFXC_w ; target = mass(DELP_{w+1}) ; same balance+diagnose path.
#
# GATING CHECK (cheap, run first): SH-UTLS relrough of raw div_h from
#   single-window MFXC_w vs trapezoidal MFXC_bar. If the trapezoid is materially
#   below single-window, M is first-order-in-time and the method should help;
#   else stop (a cheap, valuable negative result).
#
# METRIC: r_vdiv = SH-UTLS (lat<-30, 80-300 hPa) normalized grid-Laplacian
#   roughness of vdiv = cm[k]-cm[k+1] (== div_h), the score_binary.py S1 formula.
#   Continuity: C1 = RMS_ijk(dm - div_h - vdiv)/colmass over the panel interior.
#   Baselines: DIRTY 0.294, CLEAN 0.227.
#
#   ~/.julia/juliaup/julia-1.12.5+0.x64.linux.gnu/bin/julia --project=. \
#       scripts/completed_experiments/fingerfix_proto_two-window-time-averaged-flux.jl [win]
# ===========================================================================
using AtmosTransport, AtmosTransport.Preprocessing, AtmosTransport.Grids
using Dates, NCDatasets, Statistics, Printf, TOML
const P = AtmosTransport.Preprocessing
const GRAV = 9.80665
const DT_MET = 3600.0
const MFDT = 450.0                       # mass_flux_dt = 450 (high-signal invariant)
const CONFIG = joinpath(@__DIR__, "..", "..",
    "config/preprocessing/geosit_c180_dec2021_catrine_f32_fullL72.toml")
const RAW = expanduser("~/data/AtmosTransport/met/geosit/C180/raw_catrine")
const DATE = Date(2021, 12, 1)
# Center window for the trapezoid: need w-1, w, w+1 DELP and w, w+1 MFXC, all on
# day 1 (24 hourly windows) → 2 <= w <= 23 keeps us in-file (no day wrap).
const WIN = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 12

ctm_path(d::Date) = joinpath(RAW, Dates.format(d, "yyyymmdd"),
    "GEOSIT.$(Dates.format(d,"yyyymmdd")).CTM_A1.C180.nc")

# UTLS band (score_binary.py): nominal L72 layer-mid pressures, 80-300 hPa.
const EDGE72 = Float64[0.0100,0.0200,0.0327,0.0476,0.0660,0.0893,0.1197,0.1595,0.2113,0.2785,
    0.3650,0.4758,0.6168,0.7951,1.0194,1.3005,1.6508,2.0850,2.6202,3.2764,4.0766,
    5.0468,6.2168,7.6198,9.2929,11.2769,13.6434,16.4571,19.7916,23.7304,28.3678,
    33.8100,40.1754,47.6439,56.3879,66.6034,78.5123,92.3657,108.663,127.837,150.393,
    176.930,208.152,244.875,288.083,337.500,375.000,412.500,450.000,487.500,525.000,
    562.500,600.000,637.500,675.000,700.000,725.000,750.000,775.000,800.000,820.000,
    835.000,850.000,865.000,880.000,895.000,910.000,925.000,940.000,955.000,970.000,985.000]
const EDGEP = vcat(EDGE72, 1000.0)
const MIDP  = 0.5 .* (EDGEP[1:end-1] .+ EDGEP[2:end])   # 72 layer mids [hPa]
const UTLS_LO, UTLS_HI = 80.0, 300.0
const SH_LAT = -30.0

# ----------------------------------------------------------------------------
# SH-UTLS normalized grid-Laplacian roughness of a per-layer field, exactly the
# score_binary.py rough_field formula (panel interior; 4-neighbor Laplacian).
# `levs` is the set of UTLS level indices; returns mean over levels.
# ----------------------------------------------------------------------------
function sh_utls_relrough(field3::NTuple{6,Array{Float64,3}}, sh, Nc, Nz, levs)
    rs = Float64[]
    for k in levs
        lap = Float64[]; vals = Float64[]
        for p in 1:6, j in 2:Nc-1, i in 2:Nc-1
            sh[p][i, j] || continue
            f = field3[p]
            (isnan(f[i,j,k]) || isnan(f[i+1,j,k]) || isnan(f[i-1,j,k]) ||
             isnan(f[i,j+1,k]) || isnan(f[i,j-1,k])) && continue
            push!(lap, f[i,j,k] - 0.25*(f[i+1,j,k]+f[i-1,j,k]+f[i,j+1,k]+f[i,j-1,k]))
            push!(vals, f[i,j,k])
        end
        (length(vals) > 2 && std(vals) > 0) || continue
        push!(rs, sqrt(mean(abs2, lap)) / std(vals))
    end
    isempty(rs) ? NaN : mean(rs)
end

# div_h per layer from face fluxes (interior convergence == vdiv).
function divh_from_faces(am::NTuple{6,Array{Float64,3}}, bm, Nc, Nz)
    d = ntuple(_ -> fill(NaN, Nc, Nc, Nz), 6)
    @inbounds for p in 1:6, k in 1:Nz, j in 1:Nc, i in 1:Nc
        d[p][i,j,k] = (am[p][i,j,k]-am[p][i+1,j,k]) + (bm[p][i,j,k]-bm[p][i,j+1,k])
    end
    d
end

# vdiv = cm[k] - cm[k+1] per layer.
function vdiv_from_cm(cm::NTuple{6,Array{Float64,3}}, Nc, Nz)
    v = ntuple(_ -> fill(NaN, Nc, Nc, Nz), 6)
    @inbounds for p in 1:6, k in 1:Nz, j in 1:Nc, i in 1:Nc
        v[p][i,j,k] = cm[p][i,j,k] - cm[p][i,j,k+1]
    end
    v
end

# Continuity RMS over the panel interior / colmass:  dm - div_h - vdiv.
function continuity_rms(dm, am, bm, cm, m, Nc, Nz)
    resid2 = 0.0; n = 0; colmax = 0.0
    @inbounds for p in 1:6, j in 1:Nc, i in 1:Nc
        s = 0.0
        for k in 1:Nz
            s += m[p][i,j,k]
        end
        colmax = max(colmax, s)
    end
    @inbounds for p in 1:6, k in 1:Nz, j in 2:Nc-1, i in 2:Nc-1
        dh = (am[p][i,j,k]-am[p][i+1,j,k]) + (bm[p][i,j,k]-bm[p][i,j+1,k])
        vd = cm[p][i,j,k] - cm[p][i,j,k+1]
        r  = dm[p][i,j,k] - dh - vd
        resid2 += r*r; n += 1
    end
    (sqrt(resid2/n)/colmax, colmax)
end

function main()
    FT = Float64
    cfg  = TOML.parsefile(CONFIG)
    grid = P.build_target_geometry(cfg["grid"], FT)
    Nc   = grid.Nc
    vc   = P.load_hybrid_coefficients(AtmosTransport.expand_data_path(String(cfg["vertical"]["coefficients"])))
    Nz   = length(vc.A) - 1
    g    = FT(GRAV); inv_g = inv(g)
    areas = grid.mesh.cell_areas
    steps = round(Int, DT_MET/MFDT)             # 8 substeps/window (mass_flux_dt=450)
    flux_scale = FT(1/(2g))                      # raw MFXC → am = MFXC/(2g) (dt cancels)
    conn = grid.mesh.connectivity
    lats = ntuple(p -> Float64.(P.panel_cell_center_lonlat(grid.mesh, p)[2]), 6)
    sh   = ntuple(p -> lats[p] .< SH_LAT, 6)
    utls = findall(k -> (k <= Nz) && (UTLS_LO <= MIDP[k] <= UTLS_HI), 1:Nz)

    @printf("two-window-time-averaged-flux prototype  —  %s  win=%d  (Nc=%d Nz=%d steps=%d)\n",
            DATE, WIN, Nc, Nz, steps)
    @printf("  UTLS levels (80-300 hPa): %d  (k=%d..%d, p=%.0f..%.0f hPa)\n\n",
            length(utls), first(utls), last(utls), MIDP[first(utls)], MIDP[last(utls)])
    (2 <= WIN <= 23) || error("WIN must be in 2:23 (need w-1,w,w+1 on day 1, no day wrap)")

    f0 = ctm_path(DATE)
    isfile(f0) || error("missing $f0")
    ds = NCDataset(f0, "r"); or = P.detect_level_orientation(ds)
    @printf("  level orientation = %s\n", or)

    # --- read MFXC/MFYC at w and w+1, DELP at w-1, w, w+1 ---
    mfxc_w  = P._read_panels_3d(ds, "MFXC", WIN,   or; FT=FT)
    mfyc_w  = P._read_panels_3d(ds, "MFYC", WIN,   or; FT=FT)
    mfxc_w1 = P._read_panels_3d(ds, "MFXC", WIN+1, or; FT=FT)
    mfyc_w1 = P._read_panels_3d(ds, "MFYC", WIN+1, or; FT=FT)
    delp_m1 = P._read_panels_3d(ds, "DELP", WIN-1, or; FT=FT)
    delp_w  = P._read_panels_3d(ds, "DELP", WIN,   or; FT=FT)
    delp_w1 = P._read_panels_3d(ds, "DELP", WIN+1, or; FT=FT)
    close(ds)

    # trapezoidal (time-symmetric) flux average
    mfxc_bar = ntuple(p -> 0.5 .* (mfxc_w[p] .+ mfxc_w1[p]), 6)
    mfyc_bar = ntuple(p -> 0.5 .* (mfyc_w[p] .+ mfyc_w1[p]), 6)

    # ----- helpers to build faces / masses -----
    function faces(mfxc, mfyc)
        am = ntuple(_ -> zeros(FT, Nc+1, Nc, Nz), 6)
        bm = ntuple(_ -> zeros(FT, Nc, Nc+1, Nz), 6)
        P.geos_native_to_face_flux!(am, bm, mfxc, mfyc, conn, Nc, Nz, flux_scale)
        am, bm
    end
    mass(delp) = ntuple(p -> begin
        mk = zeros(FT, Nc, Nc, Nz); P._delp_pa_to_air_mass_kg!(mk, delp[p], areas, inv_g); mk
    end, 6)

    # ========================================================================
    # GATING CHECK: raw div_h SH-UTLS relrough, single-window vs trapezoid.
    # (No balance/cm — just the raw native convergence the closure must integrate.)
    # ========================================================================
    am_w,  bm_w  = faces(mfxc_w,  mfyc_w)
    am_bar, bm_bar = faces(mfxc_bar, mfyc_bar)
    divh_w   = divh_from_faces(am_w,   bm_w,   Nc, Nz)
    divh_bar = divh_from_faces(am_bar, bm_bar, Nc, Nz)
    r_divh_single = sh_utls_relrough(divh_w,   sh, Nc, Nz, utls)
    r_divh_trap   = sh_utls_relrough(divh_bar, sh, Nc, Nz, utls)
    println("=== GATING CHECK: raw div_h SH-UTLS relrough ===")
    @printf("  single-window MFXC_w        r(div_h) = %.4f\n", r_divh_single)
    @printf("  trapezoid 0.5(MFXC_w+MFXC_{w+1})  r(div_h) = %.4f   (ratio %.3f)\n",
            r_divh_trap, r_divh_trap/r_divh_single)
    gate_ratio = r_divh_trap / r_divh_single
    if gate_ratio < 0.97
        println("  → trapezoid materially smoother (ratio < 0.97): M has a first-order-in-time")
        println("    component; proceeding to the full cm closure.\n")
    else
        println("  → trapezoid NOT materially smoother (ratio >= 0.97): M is not dominated by")
        println("    a first-order-in-time sampling term. Proceeding anyway to measure cm-level r_vdiv\n")
        println("    (the gate is advisory; the cm-level r_vdiv vs DIRTY is the decisive number).\n")
    end

    # ========================================================================
    # CONTROL (DIRTY): standard endpoint closure on window w.
    #   faces = MFXC_w ; m_cur = mass(DELP_w) ; m_next = mass(DELP_{w+1})
    # ========================================================================
    function run_closure(am0, bm0, m_cur, m_next; label)
        am = ntuple(p -> copy(am0[p]), 6)
        bm = ntuple(p -> copy(bm0[p]), 6)
        bal = P.balance_cs_column_mass_fluxes!(
            am, bm, m_cur, m_next, grid.face_table, grid.cell_degree, steps,
            grid.poisson_scratch)
        dm = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
        P.fill_cs_window_mass_tendency!(dm, m_cur, m_next, steps)
        cm = ntuple(_ -> zeros(FT, Nc, Nc, Nz+1), 6)
        P.diagnose_cs_cm!(cm, am, bm, dm, m_cur, Nc, Nz)
        vdiv = vdiv_from_cm(cm, Nc, Nz)
        rv   = sh_utls_relrough(vdiv, sh, Nc, Nz, utls)
        c1, colmax = continuity_rms(dm, am, bm, cm, m_cur, Nc, Nz)
        @printf("  [%-22s] r_vdiv = %.4f   C1_rms/colmass = %.2e   (balance post-resid %.2e)\n",
                label, rv, c1, bal.final_column_projected_residual)
        (r_vdiv=rv, c1=c1, colmax=colmax)
    end

    m_w   = mass(delp_w)
    m_w1  = mass(delp_w1)
    println("=== CLOSURE RESULTS (production balance + diagnose_cs_cm!) ===")
    dirty = run_closure(am_w, bm_w, m_w, m_w1; label="DIRTY endpoint")

    # ========================================================================
    # METHOD: trapezoidal flux + centered DELP tendency.
    #   m_cur = mass(DELP_w)
    #   m_next_target = mass(DELP_w) + 0.5*(mass(DELP_{w+1}) - mass(DELP_{w-1}))
    # ========================================================================
    m_m1 = mass(delp_m1)
    m_next_centered = ntuple(p -> m_w[p] .+ 0.5 .* (m_w1[p] .- m_m1[p]), 6)
    method = run_closure(am_bar, bm_bar, m_w, m_next_centered; label="trapezoid+centered")

    # ========================================================================
    # SUMMARY vs baselines
    # ========================================================================
    println("\n=== SUMMARY ===")
    @printf("  DIRTY baseline (paper)     r_vdiv = 0.294\n")
    @printf("  CLEAN target  (MERRA-2)    r_vdiv = 0.227\n")
    @printf("  DIRTY in-script control    r_vdiv = %.4f   C1 = %.2e\n", dirty.r_vdiv, dirty.c1)
    @printf("  METHOD trapezoid+centered  r_vdiv = %.4f   C1 = %.2e\n", method.r_vdiv, method.c1)
    Δ = method.r_vdiv - dirty.r_vdiv
    @printf("  Δ(method - DIRTY control)  = %+.4f   (%.1f%% of control)\n",
            Δ, 100*Δ/dirty.r_vdiv)
    if method.r_vdiv < dirty.r_vdiv && method.c1 <= 2e-5
        if method.r_vdiv <= 0.227
            println("  VERDICT: beats CLEAN target while continuity holds.")
        else
            println("  VERDICT: improves over DIRTY (reduces fingering) while continuity holds.")
        end
    elseif method.c1 > 2e-5
        println("  VERDICT: continuity BROKEN (C1 > 2e-5) — invalid.")
    else
        println("  VERDICT: no improvement over DIRTY control.")
    end
end
main()
