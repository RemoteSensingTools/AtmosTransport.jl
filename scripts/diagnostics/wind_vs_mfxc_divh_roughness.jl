# ===========================================================================
# Route-1 proxy probe: is the WIND-DERIVED horizontal flux convergence smoother
# at the SH-UTLS than the native GEOS MFXC convergence?
#
# The fingering driver is the grid-scale roughness of the per-level horizontal
# convergence div_h[k] (the diagnosed cm integrates div_h+dm down the column;
# dm from analyzed DELP is smooth, so grid-noise in cm comes from div_h). We
# compare div_h two ways on the SAME window/grid/levels:
#   native : div_h from archived MFXC/MFYC (CTM_A1)
#   winds  : div_h from A3dyn U/V rotated to panel-local + reconstruct_cs_fluxes!
# Metric is SCALE-INVARIANT relative roughness  RMS(∇²div_h)/std(div_h)  over the
# SH interior, so dry-vs-moist and dt_factor/accumulation magnitude differences
# cancel — it measures "what fraction of the convergence is grid-scale
# checkerboard", which is the fingering source. (At the UTLS q~1e-5 so the
# dry/moist factor on the wind flux is negligible.)
#
#   julia --project=. scripts/diagnostics/wind_vs_mfxc_divh_roughness.jl [ctm_win] [a3_win]
# defaults: ctm_win=12 (~11:30), a3_win=4 (10:30, nearest 3-hourly A3dyn sample)
# ===========================================================================
using AtmosTransport, AtmosTransport.Preprocessing, AtmosTransport.Grids
using Dates, NCDatasets, Statistics, Printf, TOML
const P = AtmosTransport.Preprocessing
const CONFIG = joinpath(@__DIR__, "..", "..",
    "config/preprocessing/geosit_c180_dec2021_catrine_f32_fullL72.toml")
const DATE = Date(2021, 12, 11)
const CTM_WIN = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 12
const A3_WIN  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4

function main()
    cfg = TOML.parsefile(CONFIG); FT = Float64
    grid = P.build_target_geometry(cfg["grid"], FT); Nc = grid.Nc
    src = cfg["source"]
    settings = P.load_met_settings(joinpath(@__DIR__, "..", "..", String(src["toml"]));
        root_dir = AtmosTransport.expand_data_path(String(src["root_dir"])),
        include_surface = false, include_convection = false, include_vdiff_fields = true,
        coefficients_file = AtmosTransport.expand_data_path(String(cfg["vertical"]["coefficients"])))
    vc = P.load_hybrid_coefficients(AtmosTransport.expand_data_path(settings.coefficients_file))
    vertical = P._build_native_vertical_setup(cfg["vertical"], vc, FT); Nz = vertical.Nz
    Aifc = Float64.(vertical.merged_vc.A); Bifc = Float64.(vertical.merged_vc.B)
    Δx = grid.mesh.Δx; Δy = grid.mesh.Δy
    sh = ntuple(p -> Float64.(P.panel_cell_center_lonlat(grid.mesh, p)[2]) .< -30.0, 6)

    reader = P.open_reader(settings, DATE, FT; chain_mass = false, next_day_handle = true)
    or = reader.handles.orientation
    reader.handles.a3dyn === nothing && error("A3dyn handle not open (need include_vdiff_fields=true)")
    @printf("Nc=%d Nz=%d  ctm_win=%d  a3_win=%d\n", Nc, Nz, CTM_WIN, A3_WIN)

    # --- native MFXC/MFYC convergence -------------------------------------
    mfxc = P._read_panels_3d(reader.handles.ctm_a1, "MFXC", CTM_WIN, or; FT = FT)
    mfyc = P._read_panels_3d(reader.handles.ctm_a1, "MFYC", CTM_WIN, or; FT = FT)
    delp = P._read_panels_3d(reader.handles.ctm_a1, "DELP", CTM_WIN, or; FT = FT)

    # --- wind-derived convergence -----------------------------------------
    U = P._read_panels_3d(reader.handles.a3dyn, "U", A3_WIN, or; FT = FT)
    V = P._read_panels_3d(reader.handles.a3dyn, "V", A3_WIN, or; FT = FT)
    u_loc = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    v_loc = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    P.rotate_winds_to_panel_local!(u_loc, v_loc, U, V, grid.mesh, Nz)

    ptop = Aifc[1]
    ps = ntuple(p -> begin
        m = zeros(FT, Nc, Nc)
        @inbounds for j in 1:Nc, i in 1:Nc
            s = ptop
            for k in 1:Nz; s += delp[p][i, j, k]; end
            m[i, j] = s
        end; m
    end, 6)
    am = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), 6)
    bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), 6)
    dp_scr = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    P.reconstruct_cs_fluxes!(am, bm, u_loc, v_loc, dp_scr, ps, Aifc, Bifc,
                             Δx, Δy, FT(AtmosTransport.Parameters.STANDARD_GRAVITY), one(FT), Nc, Nz)

    # convergence (in − out) per cell, both conventions identical interior form
    div_wind = ntuple(p -> begin
        d = fill(NaN, Nc, Nc, Nz)
        @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
            d[i, j, k] = (am[p][i, j, k] - am[p][i + 1, j, k]) +
                         (bm[p][i, j, k] - bm[p][i, j + 1, k])
        end; d
    end, 6)
    div_nat = ntuple(p -> begin
        d = fill(NaN, Nc, Nc, Nz)
        @inbounds for k in 1:Nz, j in 2:Nc, i in 2:Nc   # MFXC[i]=east face of cell i
            d[i, j, k] = (mfxc[p][i - 1, j, k] - mfxc[p][i, j, k]) +
                         (mfyc[p][i, j - 1, k] - mfyc[p][i, j, k])
        end; d
    end, 6)

    # SH-interior relative roughness at level k: RMS(5-pt Laplacian)/std
    function sh_rel_rough(div, k)
        lap = Float64[]; vals = Float64[]
        for p in 1:6, j in 3:Nc-2, i in 3:Nc-2
            sh[p][i, j] || continue
            f = div[p]
            (isnan(f[i,j,k]) || isnan(f[i+1,j,k]) || isnan(f[i-1,j,k]) ||
             isnan(f[i,j+1,k]) || isnan(f[i,j-1,k])) && continue
            push!(lap, f[i,j,k] - 0.25*(f[i+1,j,k]+f[i-1,j,k]+f[i,j+1,k]+f[i,j-1,k]))
            push!(vals, f[i,j,k])
        end
        (isempty(lap) || length(vals) < 2) && return (NaN, NaN, NaN)
        rabs = sqrt(mean(abs2, lap)); sd = std(vals)
        return (rabs, sd, sd > 0 ? rabs/sd : NaN)
    end

    println("\n  TOAlev  p(hPa)   native rel   wind rel   wind/native   nat|∇²|RMS   wind|∇²|RMS")
    psref = 1.0e5
    for k in 30:2:50
        rn_abs, sn, rn = sh_rel_rough(div_nat, k)
        rw_abs, sw, rw = sh_rel_rough(div_wind, k)
        pmid = 0.5*(Aifc[k]+Aifc[k+1] + (Bifc[k]+Bifc[k+1])*psref)/100
        @printf("   %3d   %6.1f    %.4f      %.4f      %.3f        %.3e   %.3e\n",
                k, pmid, rn, rw, rw/rn, rn_abs, rw_abs)
    end
    println("\n(relative roughness = RMS(∇²div_h)/std(div_h) over SH interior; scale-invariant.")
    println(" wind/native < 1  ⇒  wind-derived convergence is smoother ⇒ smoother cm ⇒ less fingering.")
    println(" This is the per-level driver; the decisive test is tracer-level after a full build.)")
end
main()
