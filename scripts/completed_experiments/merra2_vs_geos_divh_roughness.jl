# ===========================================================================
# Route-1 de-risk: does a LAT-LON wind source (MERRA-2, the GEOS-Chem CO2 path)
# give a SMOOTH cubed-sphere div_h, unlike GEOS native A3dyn winds (which share
# MFXC's FV3 CS-grid imprint)?  A lat-lon field bilinearly mapped to C180 cannot
# carry a CS-grid checkerboard, so this is the decisive proxy for whether the
# MERRA-2 build is worth it.
#
# Compares SH-UTLS relative roughness  RMS(∇²div_h)/std(div_h)  of:
#   native : GEOS MFXC/MFYC (CTM_A1)                — same as wind_vs_mfxc probe
#   merra  : MERRA-2 U/V (lat-lon) → bilinear C180 → rotate → reconstruct_cs_fluxes!
# Both on the shared C180/L72 grid, so TOA-level k is the same pressure.
#
#   julia --project=. scripts/completed_experiments/merra2_vs_geos_divh_roughness.jl [ctm_win] [merra_ti]
# defaults: ctm_win=12 (~11:30), merra_ti=5 (12:00 UTC inst3 sample)
# ===========================================================================
using AtmosTransport, AtmosTransport.Preprocessing, AtmosTransport.Grids
using Dates, NCDatasets, Statistics, Printf, TOML
const P = AtmosTransport.Preprocessing
const CONFIG = joinpath(@__DIR__, "..", "..",
    "config/preprocessing/geosit_c180_dec2021_catrine_f32_fullL72.toml")
const MERRA = "/home/cfranken/data/AtmosTransport/met/merra2/M2I3NVASM/2021/12/MERRA2_400.inst3_3d_asm_Nv.20211211.nc4"
const DATE = Date(2021, 12, 11)
const CTM_WIN  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 12
const MERRA_TI = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5

# bilinear bracket on an ascending, uniform-ish axis with optional periodic lon
function bracket(axis, x; periodic=false, period=360.0)
    n = length(axis)
    if periodic
        if x < axis[1]
            i = n; ip = 1; w = (x - (axis[n] - period)) / (axis[1] - (axis[n] - period))
            return i, ip, clamp(w, 0.0, 1.0)
        elseif x >= axis[n]
            i = n; ip = 1; w = (x - axis[n]) / ((axis[1] + period) - axis[n])
            return i, ip, clamp(w, 0.0, 1.0)
        end
    end
    x <= axis[1] && return 1, 2, 0.0
    x >= axis[n] && return n-1, n, 1.0
    i = searchsortedlast(axis, x)
    return i, i+1, (x - axis[i]) / (axis[i+1] - axis[i])
end

function main()
    cfg = TOML.parsefile(CONFIG); FT = Float64
    grid = P.build_target_geometry(cfg["grid"], FT); Nc = grid.Nc
    src = cfg["source"]
    settings = P.load_met_settings(joinpath(@__DIR__, "..", "..", String(src["toml"]));
        root_dir = AtmosTransport.expand_data_path(String(src["root_dir"])),
        include_surface = false, include_convection = false, include_vdiff_fields = false,
        coefficients_file = AtmosTransport.expand_data_path(String(cfg["vertical"]["coefficients"])))
    vc = P.load_hybrid_coefficients(AtmosTransport.expand_data_path(settings.coefficients_file))
    vertical = P._build_native_vertical_setup(cfg["vertical"], vc, FT); Nz = vertical.Nz
    Aifc = Float64.(vertical.merged_vc.A); Bifc = Float64.(vertical.merged_vc.B)
    Δx = grid.mesh.Δx; Δy = grid.mesh.Δy
    sh = ntuple(p -> Float64.(P.panel_cell_center_lonlat(grid.mesh, p)[2]) .< -30.0, 6)
    clon = ntuple(p -> Float64.(P.panel_cell_center_lonlat(grid.mesh, p)[1]), 6)
    clat = ntuple(p -> Float64.(P.panel_cell_center_lonlat(grid.mesh, p)[2]), 6)

    # --- GEOS native MFXC convergence -------------------------------------
    reader = P.open_reader(settings, DATE, FT; chain_mass = false, next_day_handle = true)
    or = reader.handles.orientation
    mfxc = P._read_panels_3d(reader.handles.ctm_a1, "MFXC", CTM_WIN, or; FT = FT)
    mfyc = P._read_panels_3d(reader.handles.ctm_a1, "MFYC", CTM_WIN, or; FT = FT)
    @printf("Nc=%d Nz=%d  ctm_win=%d  merra_ti=%d\n", Nc, Nz, CTM_WIN, MERRA_TI)

    # --- MERRA-2 lat-lon winds → C180 -------------------------------------
    ds = NCDataset(MERRA)
    mlon = Float64.(ds["lon"][:]); mlat = Float64.(ds["lat"][:])
    U = Float64.(ds["U"][:, :, :, MERRA_TI])   # (lon, lat, lev)
    V = Float64.(ds["V"][:, :, :, MERRA_TI])
    PS = Float64.(ds["PS"][:, :, MERRA_TI])     # (lon, lat)
    close(ds)
    nlev_m = size(U, 3)
    nlev_m == Nz || @warn "MERRA lev=$nlev_m ≠ Nz=$Nz; assuming top-to-bottom alignment"

    Ucs = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    Vcs = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    pscs = ntuple(_ -> zeros(FT, Nc, Nc), 6)
    for p in 1:6, j in 1:Nc, i in 1:Nc
        lo = mod(clon[p][i, j] + 180.0, 360.0) - 180.0
        la = clat[p][i, j]
        il, ip, wl = bracket(mlon, lo; periodic=true)
        jl, jp, wb = bracket(mlat, la)
        w11 = (1-wl)*(1-wb); w21 = wl*(1-wb); w12 = (1-wl)*wb; w22 = wl*wb
        @inbounds for k in 1:Nz
            Ucs[p][i,j,k] = w11*U[il,jl,k] + w21*U[ip,jl,k] + w12*U[il,jp,k] + w22*U[ip,jp,k]
            Vcs[p][i,j,k] = w11*V[il,jl,k] + w21*V[ip,jl,k] + w12*V[il,jp,k] + w22*V[ip,jp,k]
        end
        pscs[p][i,j] = w11*PS[il,jl] + w21*PS[ip,jl] + w12*PS[il,jp] + w22*PS[ip,jp]
    end

    u_loc = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    v_loc = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    P.rotate_winds_to_panel_local!(u_loc, v_loc, Ucs, Vcs, grid.mesh, Nz)
    am = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), 6)
    bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), 6)
    dp_scr = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    P.reconstruct_cs_fluxes!(am, bm, u_loc, v_loc, dp_scr, pscs, Aifc, Bifc,
                             Δx, Δy, FT(AtmosTransport.Parameters.STANDARD_GRAVITY), one(FT), Nc, Nz)

    div_merra = ntuple(p -> begin
        d = fill(NaN, Nc, Nc, Nz)
        @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
            d[i,j,k] = (am[p][i,j,k]-am[p][i+1,j,k]) + (bm[p][i,j,k]-bm[p][i,j+1,k])
        end; d
    end, 6)
    div_nat = ntuple(p -> begin
        d = fill(NaN, Nc, Nc, Nz)
        @inbounds for k in 1:Nz, j in 2:Nc, i in 2:Nc
            d[i,j,k] = (mfxc[p][i-1,j,k]-mfxc[p][i,j,k]) + (mfyc[p][i,j-1,k]-mfyc[p][i,j,k])
        end; d
    end, 6)

    function sh_rel_rough(div, k)
        lap = Float64[]; vals = Float64[]
        for p in 1:6, j in 3:Nc-2, i in 3:Nc-2
            sh[p][i, j] || continue
            f = div[p]
            (isnan(f[i,j,k])||isnan(f[i+1,j,k])||isnan(f[i-1,j,k])||isnan(f[i,j+1,k])||isnan(f[i,j-1,k])) && continue
            push!(lap, f[i,j,k]-0.25*(f[i+1,j,k]+f[i-1,j,k]+f[i,j+1,k]+f[i,j-1,k]))
            push!(vals, f[i,j,k])
        end
        (isempty(lap)||length(vals)<2) && return (NaN, NaN)
        sd = std(vals); return (sqrt(mean(abs2,lap)), sd>0 ? sqrt(mean(abs2,lap))/sd : NaN)
    end

    println("\n  TOAlev  p(hPa)   native rel   merra rel   merra/native")
    psref = 1.0e5
    for k in 30:2:50
        _, rn = sh_rel_rough(div_nat, k)
        _, rm = sh_rel_rough(div_merra, k)
        pmid = 0.5*(Aifc[k]+Aifc[k+1] + (Bifc[k]+Bifc[k+1])*psref)/100
        @printf("   %3d   %6.1f    %.4f      %.4f      %.3f\n", k, pmid, rn, rm, rm/rn)
    end
    println("\n(merra/native ≪ 1 ⇒ MERRA-2 lat-lon winds give a much smoother CS div_h than")
    println(" native MFXC ⇒ MERRA-2 Route-1 worth building. ~0.8 like GEOS A3dyn ⇒ not worth it.)")
end
main()
