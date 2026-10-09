# ===========================================================================
# THE DECISIVE IAU TEST: is the MFXC↔DELP continuity residual M the GEOS
# analysis-increment mass (DMDTANA)?
#
# Budget: per column, total mass tendency = DMDTDYN + DMDTPHY + DMDTANA
# (dynamics + physics + analysis; all in GEOS-FP tavg1_2d_mdt_Nx, kg m-2 s-1).
# The archived MFXC convergence IS the dynamics term, so the offline residual
#   M = Σ_k dm(DELP) − Σ_k div_h(MFXC)  ≈  DMDTPHY + DMDTANA.
# M is uncorrelated with the water (physics) tendency (corr 0.01) ⇒ if the IAU
# hypothesis holds, M ≈ DMDTANA and cor(M, DMDTANA) is HIGH.
#
# M computed on the native GEOS-FP C720 cube (same run as the lat-lon DMDTANA),
# exactly as moist_budget_IT_vs_FP.jl. DMDTANA (0.25° lat-lon) is bilinear-
# sampled onto the cube cell centres, then correlated (global / SH / tropics),
# raw and on a 3×3 box-smoothed M (to strip the C720 grid-noise that the 0.25°
# increment cannot carry).
#
#   julia --project=. scripts/completed_experiments/M_vs_DMDTANA_geosfp.jl [h1 h2 ...]
#   (default windows: hours 0 6 12 18 of 2021-12-01)
# ===========================================================================
using AtmosTransport, AtmosTransport.Preprocessing, AtmosTransport.Grids
using Dates, NCDatasets, Statistics, Printf, TOML
const P = AtmosTransport.Preprocessing
const GRAV = 9.80665; const DT_MET = 3600.0; const MFDT = 450.0
const CONFIG = joinpath(@__DIR__, "..", "..",
    "config/preprocessing/geosfp_c720_native_to_cs720.toml")
const FPDIR  = expanduser("~/data/AtmosTransport/met/geosfp_c720/raw")
const MDTDIR = expanduser("~/data/AtmosTransport/met/geosfp_latlon/mdt/20211201")
const DATE   = Date(2021, 12, 1)
const HOURS  = isempty(ARGS) ? [0, 6, 12, 18] : parse.(Int, ARGS)

fp_ctm(d::Date, h::Int) = joinpath(FPDIR, Dates.format(d, "yyyymmdd"),
    "GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.$(Dates.format(d,"yyyymmdd"))_$(lpad(h,2,'0'))30.V01.nc4")
mdt_file(h::Int) = joinpath(MDTDIR,
    "GEOS.fp.asm.tavg1_2d_mdt_Nx.20211201_$(lpad(h,2,'0'))30.V01.nc4")

function setup()
    FT = Float32
    cfg = TOML.parsefile(CONFIG)
    grid = P.build_target_geometry(cfg["grid"], FT); Nc = grid.Nc
    vc = P.load_hybrid_coefficients(AtmosTransport.expand_data_path(String(cfg["vertical"]["coefficients"])))
    Nz = length(vc.A) - 1
    lonlat = ntuple(p -> P.panel_cell_center_lonlat(grid.mesh, p), 6)
    lons = ntuple(p -> Float64.(lonlat[p][1]), 6)
    lats = ntuple(p -> Float64.(lonlat[p][2]), 6)
    (FT=FT, grid=grid, Nc=Nc, Nz=Nz, areas=grid.mesh.cell_areas,
     conn=grid.mesh.connectivity, lons=lons, lats=lats)
end

# M[p][i,j] on the C720 cube for window h (moist, exactly moist_budget_IT_vs_FP)
function compute_M(s, h)
    FT = s.FT; Nc = s.Nc; Nz = s.Nz; g = FT(GRAV)
    steps = round(Int, DT_MET/MFDT); twosteps = FT(2*steps); fs = FT(1/(2g))
    hn = h + 1; dn_date = DATE; (hn == 24) && (hn = 0; dn_date = DATE + Day(1))
    dsc = NCDataset(fp_ctm(DATE, h), "r"); or = P.detect_level_orientation(dsc)
    dsn = NCDataset(fp_ctm(dn_date, hn), "r")
    mfxc = P._read_panels_3d(dsc, "MFXC", 1, or; FT=FT)
    mfyc = P._read_panels_3d(dsc, "MFYC", 1, or; FT=FT)
    dc   = P._read_panels_3d(dsc, "DELP", 1, or; FT=FT)
    dn   = P._read_panels_3d(dsn, "DELP", 1, or; FT=FT)
    close(dsc); close(dsn)
    am = ntuple(_ -> zeros(FT, Nc+1, Nc, Nz), 6); bm = ntuple(_ -> zeros(FT, Nc, Nc+1, Nz), 6)
    P.geos_native_to_face_flux!(am, bm, mfxc, mfyc, s.conn, Nc, Nz, fs)
    M = ntuple(_ -> zeros(Float64, Nc, Nc), 6)
    for p in 1:6, j in 1:Nc, i in 1:Nc
        a = s.areas[i,j]; pit = 0.0; sdm = 0.0
        @inbounds for k in 1:Nz
            pit += (am[p][i,j,k]-am[p][i+1,j,k]) + (bm[p][i,j,k]-bm[p][i,j+1,k])
            sdm += (dn[p][i,j,k]-dc[p][i,j,k])*a/g/twosteps
        end
        M[p][i,j] = sdm - pit
    end
    return M
end

# bilinear sample of a (lon,lat) field at (lo,la); lon ascending, lat ascending
function bilin(lonv, latv, F, lo, la)
    nlon = length(lonv); nlat = length(latv)
    lo = lo < lonv[1] ? lo + 360 : (lo > lonv[end] ? lo - 360 : lo)
    i = searchsortedlast(lonv, lo); i = clamp(i, 1, nlon-1)
    j = searchsortedlast(latv, la); j = clamp(j, 1, nlat-1)
    tx = (lo - lonv[i]) / (lonv[i+1]-lonv[i]); tx = clamp(tx, 0, 1)
    ty = (la - latv[j]) / (latv[j+1]-latv[j]); ty = clamp(ty, 0, 1)
    (1-tx)*(1-ty)*F[i,j] + tx*(1-ty)*F[i+1,j] + (1-tx)*ty*F[i,j+1] + tx*ty*F[i+1,j+1]
end

function read_mdt(h)
    ds = NCDataset(mdt_file(h), "r")
    lonv = Float64.(ds["lon"][:]); latv = Float64.(ds["lat"][:])
    fld(v) = Float64.(replace(ds[v][:,:,1], missing=>NaN))   # (lon,lat)
    out = (lon=lonv, lat=latv, ana=fld("DMDTANA"), dyn=fld("DMDTDYN"), phy=fld("DMDTPHY"))
    close(ds); out
end

# Coarse lat-lon grid: average out the C720 grid-noise so the LARGE-SCALE
# relationship (where DMDTANA lives) is exposed. lon normalized to [0,360) so
# cube vs lat-lon convention cannot scramble geography.
const NLON = 144; const NLAT = 72            # 2.5° × 2.5°
binx(lo) = clamp(floor(Int, mod(lo, 360.0) / (360/NLON)) + 1, 1, NLON)
biny(la) = clamp(floor(Int, (la + 90.0) / (180/NLAT)) + 1, 1, NLAT)
latc(jb) = -90.0 + (jb - 0.5) * (180/NLAT)
cor_masked(x, y, m) = (count(m)>2 ? cor(x[m], y[m]) : NaN)

function coarsen_cube(field, lons, lats, Nc)   # cube cells → coarse means
    S = zeros(NLON, NLAT); C = zeros(NLON, NLAT)
    @inbounds for p in 1:6, j in 1:Nc, i in 1:Nc
        bx = binx(lons[p][i,j]); by = biny(lats[p][i,j])
        S[bx,by] += field[p][i,j]; C[bx,by] += 1
    end
    S ./ max.(C, 1), C
end
function coarsen_ll(F, lonv, latv)             # native 0.25° → coarse means
    S = zeros(NLON, NLAT); C = zeros(NLON, NLAT)
    @inbounds for jl in eachindex(latv), il in eachindex(lonv)
        isfinite(F[il,jl]) || continue
        bx = binx(lonv[il]); by = biny(latv[jl])
        S[bx,by] += F[il,jl]; C[bx,by] += 1
    end
    S ./ max.(C, 1)
end

function main()
    s = setup()
    @printf("C720 M vs GEOS-FP DMDTANA (coarsened to %d×%d ≈2.5°) — %s, hours %s\n\n",
            NLON, NLAT, DATE, HOURS)
    println("  h   region    cor(M,ANA)  cor(M,DYN)  cor(M,PHY)   nbins   RMS|M|/colmass")
    acc = Float64[]
    for h in HOURS
        isfile(mdt_file(h)) || (println("  (missing mdt h=$h, skip)"); continue)
        M = compute_M(s, h)
        Mc, cnt = coarsen_cube(M, s.lons, s.lats, s.Nc)
        md = read_mdt(h)
        Ac = coarsen_ll(md.ana, md.lon, md.lat)
        Dc = coarsen_ll(md.dyn, md.lon, md.lat)
        Pc = coarsen_ll(md.phy, md.lon, md.lat)
        # geography sanity: coarse-cell M-weighted lat must match the bin lat
        rmsM = sqrt(mean(vcat([vec(M[p]) for p in 1:6]...).^2))
        # max column dry mass proxy for normalization (rough; pattern test is the point)
        Mcv = vec(Mc); Acv = vec(Ac); Dcv = vec(Dc); Pcv = vec(Pc)
        latv = vec([latc(jb) for ib in 1:NLON, jb in 1:NLAT])
        valid = vec(cnt) .> 0
        for (nm, reg) in (("GLOBAL", trues(length(Mcv))),
                          ("SH<-30", latv .< -30.0),
                          ("TROPICS", abs.(latv) .< 30.0))
            m = reg .& valid
            @printf("  %2d  %-8s  %+8.3f   %+8.3f   %+8.3f   %5d\n",
                    h, nm, cor_masked(Mcv,Acv,m), cor_masked(Mcv,Dcv,m),
                    cor_masked(Mcv,Pcv,m), count(m))
            nm=="GLOBAL" && push!(acc, cor_masked(Mcv,Acv,m))
        end
        @printf("      RMS|M| = %.3e (internal units); RMS DMDTANA = %.3e kg/m2/s\n\n",
                rmsM, sqrt(mean(filter(isfinite, vec(md.ana)).^2)))
    end
    println("VERDICT (coarse, grid-noise removed): cor(M,DMDTANA) > 0.6 ⇒ M IS the analysis")
    println("  increment (build a non-advective source σ). ~0 ⇒ M is an advective/temporal residual")
    println("  (variational redistribution / OMEGA-cm is the cure, NOT a source term).")
    isempty(acc) || @printf("  mean GLOBAL cor(M,ANA) = %+.3f\n", mean(filter(isfinite,acc)))
end
main()
