# ===========================================================================
# exp 1 (input-side reconciliation): quantify the dry-air column residual
#   M = Σ_k dm_dry − pit_native   (should be 0 by dry-air conservation)
# two ways, isolating the PS/DELP time-sampling (QV held identical = CTM_I1):
#   M_I1: endpoints from CTM_I1 instantaneous PS (DELP_total = ΔA+ΔB·PS) × (1−QV)
#   M_A1: endpoints from CTM_A1 hourly-AVERAGED DELP                    × (1−QV)
# pit_native is the CTM_A1 dry MFXC/MFYC convergence for BOTH.
# Report magnitude (RMS/max, frac of column), per-level residual roughness by
# pressure, and SH-vs-global — especially SH UTLS.
#   julia --project=. scripts/completed_experiments/moisture_residual_I1_vs_A1.jl [window]
# Time-centering caveat: CTM_I1 @ :00 (aligned with MFXC [:00,:00]); CTM_A1 @ :30.
# ===========================================================================
using AtmosTransport
using AtmosTransport.Preprocessing
using AtmosTransport.Grids
using Dates, NCDatasets, Statistics, Printf, TOML
const P = AtmosTransport.Preprocessing

const GRAV = 9.80665
const DT_MET = 3600.0
const CONFIG = joinpath(@__DIR__, "..", "..",
    "config/preprocessing/geosit_c180_dec2021_catrine_f32_fullL72.toml")
const DATE = Date(2021, 12, 11)
WIN = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 12

# normalized grid roughness over a mask: RMS(f−0.25·nb)/std
function rough(f_panels, mask, Nc)
    laps = Float64[]; vals = Float64[]
    for p in 1:6
        f = f_panels[p]; m = mask[p]
        for j in 2:Nc-1, i in 2:Nc-1
            m[i,j] || continue
            nb = f[i+1,j]+f[i-1,j]+f[i,j+1]+f[i,j-1]
            (isnan(f[i,j])||isnan(nb)) && continue
            push!(laps, f[i,j]-0.25nb)
        end
        for j in 1:Nc, i in 1:Nc; m[i,j] && !isnan(f[i,j]) && push!(vals, f[i,j]); end
    end
    (isempty(laps)||std(vals)==0) && return NaN
    return sqrt(mean(abs2,laps))/std(vals)
end

function main()
    cfg = TOML.parsefile(CONFIG); FT = Float64
    grid = P.build_target_geometry(cfg["grid"], FT); Nc = grid.Nc
    src = cfg["source"]
    tomlp = joinpath(@__DIR__, "..", "..", String(src["toml"]))
    settings = P.load_met_settings(tomlp;
        root_dir = AtmosTransport.expand_data_path(String(src["root_dir"])),
        include_surface=false, include_convection=false, include_vdiff_fields=false,
        coefficients_file = AtmosTransport.expand_data_path(String(cfg["vertical"]["coefficients"])))
    vc = P.load_hybrid_coefficients(AtmosTransport.expand_data_path(settings.coefficients_file))
    vertical = P._build_native_vertical_setup(cfg["vertical"], vc, FT)
    Nz = vertical.Nz
    Bifc = Float64.(vertical.merged_vc.B); ΔB = [Bifc[k+1]-Bifc[k] for k in 1:Nz]
    Aifc = Float64.(vertical.merged_vc.A)
    lats = ntuple(p -> Float64.(P.panel_cell_center_lonlat(grid.mesh, p)[2]), 6)
    sh = ntuple(p -> lats[p] .< -30.0, 6)
    @printf("Nc=%d Nz=%d  window=%d  DATE=%s\n", Nc, Nz, WIN, DATE)

    reader = P.open_reader(settings, DATE, FT; chain_mass=false, next_day_handle=true)
    nw = P.windows_per_day(reader); or = reader.handles.orientation
    @assert WIN < nw "use an interior window (win<$nw) so win+1 is same-day CTM_A1"
    raw = P.allocate_raw_window(settings; FT=FT, Nz=Nz)
    P.read_window!(raw, reader, WIN)
    g = FT(GRAV); inv_g = inv(g); cell_areas = grid.mesh.cell_areas
    steps = round(Int, FT(DT_MET)/FT(settings.mass_flux_dt))   # 8
    flux_scale = FT(settings.mass_flux_dt/2)/g

    # --- native MFXC/MFYC → am/bm → per-cell column convergence pit (per half-step)
    am = ntuple(_ -> zeros(FT, Nc+1, Nc, Nz), 6); bm = ntuple(_ -> zeros(FT, Nc, Nc+1, Nz), 6)
    P.geos_native_to_face_flux!(am, bm, raw.am, raw.bm, grid.mesh.connectivity, Nc, Nz, flux_scale)
    pit = ntuple(p -> begin
        a = zeros(FT, Nc, Nc)
        @inbounds for j in 1:Nc, i in 1:Nc, k in 1:Nz
            a[i,j] += (am[p][i,j,k]-am[p][i+1,j,k]) + (bm[p][i,j,k]-bm[p][i,j+1,k])
        end; a
    end, 6)

    # --- dry endpoints two ways. raw.m/raw.m_next = CTM_I1 DELP_dry (Pa, TOA-first).
    #     raw.qv/raw.qv_next = CTM_I1 QV at win/win+1. CTM_A1 DELP read directly.
    delp_a1_cur  = P._read_panels_3d(reader.handles.ctm_a1, "DELP", WIN,   or; FT=FT)
    delp_a1_next = P._read_panels_3d(reader.handles.ctm_a1, "DELP", WIN+1, or; FT=FT)

    # per-half-step residual r[k] = dm_dry[k] − ΔB[k]·pit ; column M = Σ_k r
    function residual(m_cur_pa, m_next_pa)   # DELP_dry in Pa per panel-tuple
        r = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6); M = ntuple(_ -> zeros(FT, Nc, Nc), 6)
        for p in 1:6, j in 1:Nc, i in 1:Nc
            acc = 0.0
            for k in 1:Nz
                mcur = m_cur_pa[p][i,j,k]*cell_areas[i,j]*inv_g
                mnxt = m_next_pa[p][i,j,k]*cell_areas[i,j]*inv_g
                dm = (mnxt - mcur)/(2steps)
                r[p][i,j,k] = dm - FT(ΔB[k])*pit[p][i,j]
                acc += dm
            end
            M[p][i,j] = acc - pit[p][i,j]
        end
        return r, M
    end
    # I1 dry DELP (Pa): raw.m, raw.m_next already (1−QV_I1)·(ΔA+ΔB·PS_I1).
    rI1, MI1 = residual(raw.m, raw.m_next)
    # A1 dry DELP (Pa): CTM_A1 DELP_total × (1−QV_I1) at the matching time.
    dpA1_cur  = ntuple(p -> (1 .- raw.qv[p])      .* delp_a1_cur[p],  6)
    dpA1_next = ntuple(p -> (1 .- raw.qv_next[p]) .* delp_a1_next[p], 6)
    rA1, MA1 = residual(dpA1_cur, dpA1_next)

    # --- column mass scale for fractions
    colmass = maximum(begin
        mx=0.0; for p in 1:6, j in 1:Nc, i in 1:Nc
            c=0.0; for k in 1:Nz; c+=raw.m[p][i,j,k]*cell_areas[i,j]*inv_g; end; mx=max(mx,c)
        end; mx
    end for _ in 1:1)
    rms(M) = sqrt(mean(vcat([vec(M[p]).^2 for p in 1:6]...)))
    mx(M)  = maximum(vcat([vec(abs.(M[p])) for p in 1:6]...))
    @printf("\n=== COLUMN residual M = Σ_k dm_dry − pit (per half-step, kg) ===\n")
    @printf("              RMS(kg)      max(kg)     RMS/colmass   max/colmass\n")
    @printf("  M_I1     %.4e   %.4e    %.3e     %.3e\n", rms(MI1), mx(MI1), rms(MI1)/colmass, mx(MI1)/colmass)
    @printf("  M_A1     %.4e   %.4e    %.3e     %.3e\n", rms(MA1), mx(MA1), rms(MA1)/colmass, mx(MA1)/colmass)
    @printf("  ratio A1/I1 (RMS) = %.3f   ⇒ %s\n", rms(MA1)/rms(MI1),
            rms(MA1)/rms(MI1) < 0.5 ? "A1 MUCH smaller ⇒ I1-vs-A1 time-sampling dominates" :
            rms(MA1)/rms(MI1) > 0.9 ? "A1 ≈ I1 ⇒ NOT time-sampling (dry-conv/MFXC convention)" : "partial")
    @printf("  SH-only RMS:  M_I1=%.4e  M_A1=%.4e  ratio=%.3f\n",
            sqrt(mean(vcat([vec(MI1[p][sh[p]]).^2 for p in 1:6]...))),
            sqrt(mean(vcat([vec(MA1[p][sh[p]]).^2 for p in 1:6]...))),
            sqrt(mean(vcat([vec(MA1[p][sh[p]]).^2 for p in 1:6]...)))/sqrt(mean(vcat([vec(MI1[p][sh[p]]).^2 for p in 1:6]...))))

    # --- KEY TEST: is M ≈ −(column water-mass tendency)?  (⇒ MFXC is MOIST) ---
    # water[k] = q·DELP_total = raw.m(=DELP_dry)·q/(1−q) ; tendency per half-step.
    Wten = ntuple(_ -> zeros(FT, Nc, Nc), 6)
    for p in 1:6, j in 1:Nc, i in 1:Nc
        w = 0.0
        for k in 1:Nz
            qc = raw.qv[p][i,j,k]; qn = raw.qv_next[p][i,j,k]
            wc = raw.m[p][i,j,k]      * qc/(1-qc) * cell_areas[i,j]*inv_g
            wn = raw.m_next[p][i,j,k] * qn/(1-qn) * cell_areas[i,j]*inv_g
            w += (wn - wc)
        end
        Wten[p][i,j] = w/(2steps)
    end
    negW = ntuple(p -> -Wten[p], 6)
    # correlation + ratio of M_I1 vs −water tendency
    mv = vcat([vec(MI1[p]) for p in 1:6]...); wv = vcat([vec(negW[p]) for p in 1:6]...)
    cc = cor(mv, wv)
    resid_after = sqrt(mean((mv .- wv).^2))
    @printf("\n=== KEY TEST: M_I1 vs −(column water tendency) ===\n")
    @printf("  corr(M_I1, −Wtend)        = %.4f\n", cc)
    @printf("  RMS(M_I1)                 = %.4e\n", rms(MI1))
    @printf("  RMS(M_I1 − (−Wtend))      = %.4e   (%.1f%% of RMS(M_I1))\n", resid_after, 100*resid_after/rms(MI1))
    @printf("  ⇒ %s\n", cc > 0.9 ? "MFXC is MOIST: M ≈ −water tendency ⇒ dry-convert the flux (×(1−q_face))" :
                          cc < 0.4 ? "NOT the moist-flux water tendency ⇒ deeper (FV3 remap / regrid / scaling)" : "partial — water tendency explains some of M")

    # --- per-level residual roughness (the fingering source) by pressure, SH ---
    println("\n=== per-level residual r[k] SH roughness (≈164hPa band): I1 vs A1 ===")
    println("  TOAlev   p_mid(hPa)   SH rough r_I1   r_A1")
    psref = 1.0e5
    for k in 30:2:48
        pmid = 0.5*(Aifc[k]+Aifc[k+1] + (Bifc[k]+Bifc[k+1])*psref)/100
        fI1 = ntuple(p -> rI1[p][:,:,k], 6); fA1 = ntuple(p -> rA1[p][:,:,k], 6)
        @printf("   %3d     %7.1f      %.4f         %.4f\n", k, pmid, rough(fI1, sh, Nc), rough(fA1, sh, Nc))
    end
    println("\nDone.")
end
main()
