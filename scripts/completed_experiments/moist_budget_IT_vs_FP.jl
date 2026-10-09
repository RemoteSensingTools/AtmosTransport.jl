# ===========================================================================
# QV-free MFXC↔DELP continuity residual, GEOS-IT (C180) vs GEOS-FP (C720).
# Decides whether the MFXC-vs-DELP inconsistency (the SH-UTLS fingering source)
# is GEOS-IT-specific (replay product) or intrinsic to the native FV3 mass flux.
#
#   M[i,j] = Σ_k dm[k] − pit ,  dm[k]=(DELP_next−DELP_cur)[k]·area/g/(2·steps)
#   pit = Σ_k column convergence of am/bm (= MFXC/(2g) face fluxes, same as the
#   production GEOS-CS path). MOIST DELP (no QV); the small dry offset is common
#   to both products so the normalized comparison stands. Column M is
#   orientation-independent.  Both: mass_flux_dt=450, MFXC "Pa m2 s-1".
#
#   julia --project=. scripts/completed_experiments/moist_budget_IT_vs_FP.jl
# ===========================================================================
using AtmosTransport, AtmosTransport.Preprocessing, AtmosTransport.Grids
using Dates, NCDatasets, Statistics, Printf, TOML
const P = AtmosTransport.Preprocessing
const GRAV = 9.80665; const DT_MET = 3600.0; const MFDT = 450.0

function load_product(name, config, FT)
    cfg = TOML.parsefile(config)
    grid = P.build_target_geometry(cfg["grid"], FT); Nc = grid.Nc
    vc = P.load_hybrid_coefficients(AtmosTransport.expand_data_path(String(cfg["vertical"]["coefficients"])))
    Nz = length(vc.A) - 1
    g = FT(GRAV); steps = round(Int, DT_MET/MFDT)
    flux_scale = FT(1/(2g))   # raw MFXC → am = MFXC/(2g)  (mass_flux_dt cancels)
    areas = grid.mesh.cell_areas

    # open datasets + (mfxc,mfyc @win) , (delp @win, delp @win+1)
    if name == :IT
        f = expanduser("~/data/AtmosTransport/met/geosit/C180/raw_catrine/20211211/GEOSIT.20211211.CTM_A1.C180.nc")
        ds = NCDataset(f, "r"); or = P.detect_level_orientation(ds); w = 12
        mfxc = P._read_panels_3d(ds, "MFXC", w,   or; FT=FT); mfyc = P._read_panels_3d(ds, "MFYC", w, or; FT=FT)
        dc   = P._read_panels_3d(ds, "DELP", w,   or; FT=FT); dn   = P._read_panels_3d(ds, "DELP", w+1, or; FT=FT)
        close(ds)
    else  # :FP — per-hour files, hour 12 → 13
        dir = expanduser("~/data/AtmosTransport/met/geosfp_c720/raw/20211204")
        path(h) = joinpath(dir, "GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.20211204_$(lpad(h,2,'0'))30.V01.nc4")
        dsc = NCDataset(path(12), "r"); dsn = NCDataset(path(13), "r")
        or = P.detect_level_orientation(dsc)
        mfxc = P._read_panels_3d(dsc, "MFXC", 1, or; FT=FT); mfyc = P._read_panels_3d(dsc, "MFYC", 1, or; FT=FT)
        dc   = P._read_panels_3d(dsc, "DELP", 1, or; FT=FT); dn   = P._read_panels_3d(dsn, "DELP", 1, or; FT=FT)
        close(dsc); close(dsn)
    end

    am = ntuple(_ -> zeros(FT, Nc+1, Nc, Nz), 6); bm = ntuple(_ -> zeros(FT, Nc, Nc+1, Nz), 6)
    P.geos_native_to_face_flux!(am, bm, mfxc, mfyc, grid.mesh.connectivity, Nc, Nz, flux_scale)

    M = ntuple(_ -> zeros(FT, Nc, Nc), 6); colm = ntuple(_ -> zeros(FT, Nc, Nc), 6)
    twosteps = FT(2*steps)
    for p in 1:6, j in 1:Nc, i in 1:Nc
        pit = zero(FT); sdm = zero(FT); cm = zero(FT)
        a = areas[i,j]
        for k in 1:Nz
            pit += (am[p][i,j,k]-am[p][i+1,j,k]) + (bm[p][i,j,k]-bm[p][i,j+1,k])
            sdm += (dn[p][i,j,k]-dc[p][i,j,k])*a/g/twosteps
            cm  += dc[p][i,j,k]*a/g
        end
        M[p][i,j] = sdm - pit; colm[p][i,j] = cm
    end
    lats = ntuple(p -> Float64.(P.panel_cell_center_lonlat(grid.mesh, p)[2]), 6)
    return (Nc=Nc, Nz=Nz, M=M, colmass=maximum(maximum, colm),
            sh = ntuple(p -> lats[p] .< -30.0, 6), or=or)
end

function report(name, r)
    Mv = vcat([vec(r.M[p]) for p in 1:6]...)
    shM = vcat([vec(r.M[p][r.sh[p]]) for p in 1:6]...)
    @printf("%s  (Nc=%d Nz=%d orient=%s)\n", name, r.Nc, r.Nz, r.or)
    @printf("   colmass(max col, kg) = %.4e\n", r.colmass)
    @printf("   RMS|M|/colmass  = %.3e     max|M|/colmass = %.3e\n",
            sqrt(mean(Mv.^2))/r.colmass, maximum(abs.(Mv))/r.colmass)
    @printf("   SH RMS|M|/colmass = %.3e\n", sqrt(mean(shM.^2))/r.colmass)
end

println("Loading GEOS-IT C180 …");  it = load_product(:IT, "config/preprocessing/geosit_c180_dec2021_catrine_f32_fullL72.toml", Float64)
println("Loading GEOS-FP C720 …");  fp = load_product(:FP, "config/preprocessing/geosfp_c720_native_to_cs720.toml", Float32)
println("\n=== QV-free MFXC↔DELP continuity residual M/colmass ===")
report("GEOS-IT C180 (replay)", it)
report("GEOS-FP C720 (online)", fp)
@printf("\n>> RMS ratio FP/IT (normalized) = %.3f\n",
        (sqrt(mean(vcat([vec(fp.M[p]) for p in 1:6]...).^2))/fp.colmass) /
        (sqrt(mean(vcat([vec(it.M[p]) for p in 1:6]...).^2))/it.colmass))
println("   <1 ⇒ FP cleaner (GEOS-IT replay issue);  ≈1 ⇒ intrinsic to native MFXC")
