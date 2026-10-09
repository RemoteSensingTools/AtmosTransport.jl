# ===========================================================================
# PROTOTYPE (one window): does flux-adv + vertical REMAP inject LESS SH-UTLS
# structure into a SMOOTH tracer than the current cm-advection?
# Seed q0 = horizontally UNIFORM, vertically-smooth CO2-like profile, so ANY
# output horizontal structure at the SH-UTLS is purely the method's artifact.
#   Path CM (current): advect q0 vertically by Path-A cm over the window.
#   Path REMAP (FV3) : flux-adv (uniform q ⇒ Lagrangian dpA) → conservative
#                      remap from cumsum(dpA) to analyzed cumsum(m_next).
# Metric = RMS over SH (lat<-30) of the horizontal grid-Laplacian of q at UTLS.
#   julia --project=. scripts/completed_experiments/prototype_remap_vs_cm.jl [window]
# ===========================================================================
using AtmosTransport, AtmosTransport.Preprocessing, AtmosTransport.Grids
using Dates, NCDatasets, Statistics, Printf, TOML
const P = AtmosTransport.Preprocessing
const GRAV = 9.80665; const DT_MET = 3600.0
const CONFIG = joinpath(@__DIR__,"..","..","config/preprocessing/geosit_c180_dec2021_catrine_f32_fullL72.toml")
const DATE = Date(2021,12,11); WIN = length(ARGS)>=1 ? parse(Int,ARGS[1]) : 12

# horizontal grid-roughness RMS over SH (the artifact metric)
function sh_rough_rms(f, mask, Nc)
    laps = Float64[]
    for p in 1:6, j in 2:Nc-1, i in 2:Nc-1
        mask[p][i,j] || continue
        push!(laps, Float64(f[p][i,j]) - 0.25*(Float64(f[p][i+1,j])+f[p][i-1,j]+f[p][i,j+1]+f[p][i,j-1]))
    end
    sqrt(mean(abs2, laps))
end
sh_std(f, mask) = std(vcat([vec(f[p][mask[p]]) for p in 1:6]...))

# conservative piecewise-constant column remap of tracer MASS from src edges→dst edges:
# integrate the per-layer src density (rm_src[k]/dp_src[k]) over each dst layer's range.
function remap_col!(rm_dst, rm_src, pe_src, pe_dst, Nz)
    @inbounds for kd in 1:Nz
        a = pe_dst[kd]; b = pe_dst[kd+1]; tot = 0.0
        for ks in 1:Nz
            lo = max(a, pe_src[ks]); hi = min(b, pe_src[ks+1])
            hi > lo || continue
            tot += rm_src[ks]*(hi-lo)/(pe_src[ks+1]-pe_src[ks])
        end
        rm_dst[kd] = tot
    end
end

function main()
    cfg = TOML.parsefile(CONFIG); FT = Float64
    grid = P.build_target_geometry(cfg["grid"], FT); Nc = grid.Nc
    src = cfg["source"]
    settings = P.load_met_settings(joinpath(@__DIR__,"..","..",String(src["toml"]));
        root_dir=AtmosTransport.expand_data_path(String(src["root_dir"])),
        include_surface=false, include_convection=false, include_vdiff_fields=false,
        coefficients_file=AtmosTransport.expand_data_path(String(cfg["vertical"]["coefficients"])))
    vc = P.load_hybrid_coefficients(AtmosTransport.expand_data_path(settings.coefficients_file))
    vertical = P._build_native_vertical_setup(cfg["vertical"], vc, FT); Nz = vertical.Nz
    Bifc = Float64.(vertical.merged_vc.B); Aifc = Float64.(vertical.merged_vc.A)
    sh = ntuple(p -> Float64.(P.panel_cell_center_lonlat(grid.mesh,p)[2]) .< -30.0, 6)
    g=FT(GRAV); inv_g=inv(g); areas=grid.mesh.cell_areas
    steps=round(Int,DT_MET/settings.mass_flux_dt); flux_scale=FT(settings.mass_flux_dt/2)/g
    @printf("Nc=%d Nz=%d win=%d\n", Nc, Nz, WIN)

    reader=P.open_reader(settings,DATE,FT;chain_mass=false,next_day_handle=true)
    raw=P.allocate_raw_window(settings;FT=FT,Nz=Nz); P.read_window!(raw,reader,WIN)
    am=ntuple(_->zeros(FT,Nc+1,Nc,Nz),6); bm=ntuple(_->zeros(FT,Nc,Nc+1,Nz),6)
    P.geos_native_to_face_flux!(am,bm,raw.am,raw.bm,grid.mesh.connectivity,Nc,Nz,flux_scale)
    m_cur=ntuple(_->zeros(FT,Nc,Nc,Nz),6); m_next=ntuple(_->zeros(FT,Nc,Nc,Nz),6); t=ntuple(_->zeros(FT,Nc,Nc,Nz),6)
    for p in 1:6
        P._delp_pa_to_air_mass_kg!(t[p],raw.m[p],areas,inv_g); copyto!(m_cur[p],t[p])
        P._delp_pa_to_air_mass_kg!(t[p],raw.m_next[p],areas,inv_g); copyto!(m_next[p],t[p])
    end
    amB=ntuple(p->copy(am[p]),6); bmB=ntuple(p->copy(bm[p]),6)
    P.balance_cs_column_mass_fluxes!(amB,bmB,m_cur,m_next,grid.face_table,grid.cell_degree,steps,grid.poisson_scratch)
    dm=ntuple(_->zeros(FT,Nc,Nc,Nz),6); P.fill_cs_window_mass_tendency!(dm,m_cur,m_next,steps)
    cm=ntuple(_->zeros(FT,Nc,Nc,Nz+1),6); P.diagnose_cs_cm!(cm,amB,bmB,dm,m_cur,Nc,Nz)

    # smooth CO2-like profile vs pressure (TOA-first): low aloft, high near surface, smooth
    psref=1.0e5
    q0=zeros(Float64,Nz)
    for k in 1:Nz
        pmid=0.5*(Aifc[k]+Aifc[k+1]+(Bifc[k]+Bifc[k+1])*psref)
        q0[k]=400.0 + 40.0/(1.0+exp(-(pmid-1.5e4)/2.0e4))   # smooth sigmoid, UTLS gradient ~164hPa
    end
    twostep=FT(2*steps)

    # ---- Path CM: advect uniform q0 vertically by cm ----
    qcm=ntuple(_->zeros(FT,Nc,Nc,Nz),6)
    @inbounds for p in 1:6, j in 1:Nc, i in 1:Nc
        # horizontal flux-adv keeps uniform q ⇒ Lagrangian dpA; rm_horiz=q0*dpA
        for k in 1:Nz
            conv = (amB[p][i,j,k]-amB[p][i+1,j,k])+(bmB[p][i,j,k]-bmB[p][i,j+1,k])
            dpA = m_cur[p][i,j,k] + twostep*conv
            # vertical cm flux through interface k (top of layer k): 2steps*cm*q_up
            cmk  = cm[p][i,j,k];   cmk1 = cm[p][i,j,k+1]
            qupk  = cmk  > 0 ? (k>1  ? q0[k-1] : q0[k]) : q0[k]
            qupk1 = cmk1 > 0 ? q0[k] : (k<Nz ? q0[k+1] : q0[k])
            rm = q0[k]*dpA + twostep*(cmk*qupk - cmk1*qupk1)
            qcm[p][i,j,k] = rm / m_next[p][i,j,k]
        end
    end

    # ---- Path REMAP: flux-adv (uniform q ⇒ rm=q0*dpA on Lagrangian) → remap to analyzed m_next ----
    qrm=ntuple(_->zeros(FT,Nc,Nc,Nz),6)
    rm_src=zeros(Float64,Nz); rm_dst=zeros(Float64,Nz); peS=zeros(Float64,Nz+1); peD=zeros(Float64,Nz+1)
    @inbounds for p in 1:6, j in 1:Nc, i in 1:Nc
        for k in 1:Nz
            conv=(amB[p][i,j,k]-amB[p][i+1,j,k])+(bmB[p][i,j,k]-bmB[p][i,j+1,k])
            dpA=m_cur[p][i,j,k]+twostep*conv
            rm_src[k]=q0[k]*dpA
            peS[k+1]=peS[k]+dpA
            peD[k+1]=peD[k]+m_next[p][i,j,k]
        end
        peS[1]=0.0; peD[1]=0.0
        # rescale src edges so total column = analyzed (handles the residual M, like calcScalingFactor)
        sc = peD[Nz+1]/peS[Nz+1]
        for k in 1:Nz+1; peS[k]*=sc; end
        remap_col!(rm_dst, rm_src, peS, peD, Nz)
        for k in 1:Nz; qrm[p][i,j,k]=rm_dst[k]/m_next[p][i,j,k]; end
    end

    # ---- metric at UTLS band (TOA-first lev 37-44 ≈ 100-260 hPa) ----
    println("\n=== SH horizontal grid-roughness RMS of q (q0 was UNIFORM ⇒ this IS the artifact) ===")
    println("  lev   p(hPa)   PathCM      PathREMAP    REMAP/CM")
    for k in 37:43
        pmid=0.5*(Aifc[k]+Aifc[k+1]+(Bifc[k]+Bifc[k+1])*psref)/100
        rcm=sh_rough_rms(ntuple(p->qcm[p][:,:,k],6),sh,Nc)
        rrm=sh_rough_rms(ntuple(p->qrm[p][:,:,k],6),sh,Nc)
        @printf("  %3d   %6.1f   %.3e   %.3e   %.3f\n", k, pmid, rcm, rrm, rrm/rcm)
    end
    println("\nDone.")
end
main()
