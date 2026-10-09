# ===========================================================================
# PROTOTYPE (one window): GEOS-Chem Classic fix = pressure-fixer cm + fluxes
# BALANCED to the analyzed PS endpoint. Confirms, before the larger integration:
#   (1) cm SH-UTLS roughness DROPS to the pressure-fixer floor (vs Path-A diagnose)
#   (2) the column mass LANDS on the analyzed dry-PS (no drift — unlike native pf)
#   (3) the replay gate CLOSES (m_evolved == m_next) to roundoff
# Mirrors tpcore_fvdas Calc_Vert_Mass_Flux (wz=cumsum(dpi−dbk·dps)) + delp2=dap+dbk·ps2.
#   julia --project=. scripts/completed_experiments/prototype_pfix_balanced.jl [window]
# ===========================================================================
using AtmosTransport, AtmosTransport.Preprocessing, AtmosTransport.Grids
using Dates, NCDatasets, Statistics, Printf, TOML
const P = AtmosTransport.Preprocessing
const GRAV = 9.80665; const DT_MET = 3600.0
const CONFIG = joinpath(@__DIR__, "..", "..", "config/preprocessing/geosit_c180_dec2021_catrine_f32_fullL72.toml")
const DATE = Date(2021, 12, 11); WIN = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 12

function rough_sh(f, mask, Nc)
    laps = Float64[]; vals = Float64[]
    for p in 1:6, j in 2:Nc-1, i in 2:Nc-1
        mask[p][i,j] || continue
        nb = f[p][i+1,j]+f[p][i-1,j]+f[p][i,j+1]+f[p][i,j-1]
        push!(laps, Float64(f[p][i,j])-0.25Float64(nb))
    end
    for p in 1:6, j in 1:Nc, i in 1:Nc; mask[p][i,j] && push!(vals, Float64(f[p][i,j])); end
    sqrt(mean(abs2,laps))/std(vals)
end

cfg = TOML.parsefile(CONFIG); FT = Float64
grid = P.build_target_geometry(cfg["grid"], FT); Nc = grid.Nc
src = cfg["source"]
settings = P.load_met_settings(joinpath(@__DIR__,"..","..",String(src["toml"]));
    root_dir = AtmosTransport.expand_data_path(String(src["root_dir"])),
    include_surface=false, include_convection=false, include_vdiff_fields=false,
    coefficients_file = AtmosTransport.expand_data_path(String(cfg["vertical"]["coefficients"])))
vc = P.load_hybrid_coefficients(AtmosTransport.expand_data_path(settings.coefficients_file))
vertical = P._build_native_vertical_setup(cfg["vertical"], vc, FT); Nz = vertical.Nz
Bifc = Float64.(vertical.merged_vc.B); ΔB = [Bifc[k+1]-Bifc[k] for k in 1:Nz]
sh = ntuple(p -> Float64.(P.panel_cell_center_lonlat(grid.mesh,p)[2]) .< -30.0, 6)
g = FT(GRAV); inv_g = inv(g); areas = grid.mesh.cell_areas
steps = round(Int, DT_MET/settings.mass_flux_dt); flux_scale = FT(settings.mass_flux_dt/2)/g
@printf("Nc=%d Nz=%d win=%d steps=%d\n", Nc, Nz, WIN, steps)

reader = P.open_reader(settings, DATE, FT; chain_mass=false, next_day_handle=true)
raw = P.allocate_raw_window(settings; FT=FT, Nz=Nz); P.read_window!(raw, reader, WIN)
am0 = ntuple(_->zeros(FT,Nc+1,Nc,Nz),6); bm0 = ntuple(_->zeros(FT,Nc,Nc+1,Nz),6)
P.geos_native_to_face_flux!(am0, bm0, raw.am, raw.bm, grid.mesh.connectivity, Nc, Nz, flux_scale)
m_cur = ntuple(_->zeros(FT,Nc,Nc,Nz),6); m_next = ntuple(_->zeros(FT,Nc,Nc,Nz),6)
tmp = ntuple(_->zeros(FT,Nc,Nc,Nz),6)
for p in 1:6
    P._delp_pa_to_air_mass_kg!(tmp[p], raw.m[p], areas, inv_g);      copyto!(m_cur[p], tmp[p])
    P._delp_pa_to_air_mass_kg!(tmp[p], raw.m_next[p], areas, inv_g); copyto!(m_next[p], tmp[p])
end

# --- balance am/bm to the analyzed endpoint (column closure to analyzed dry-PS) ---
amB = ntuple(p->copy(am0[p]),6); bmB = ntuple(p->copy(bm0[p]),6)
P.balance_cs_column_mass_fluxes!(amB, bmB, m_cur, m_next, grid.face_table, grid.cell_degree, steps, grid.poisson_scratch)

# Path A (current production): diagnose cm from per-level analyzed DELP_dry tendency
dmA = ntuple(_->zeros(FT,Nc,Nc,Nz),6); P.fill_cs_window_mass_tendency!(dmA, m_cur, m_next, steps)
cmA = ntuple(_->zeros(FT,Nc,Nc,Nz+1),6); P.diagnose_cs_cm!(cmA, amB, bmB, dmA, m_cur, Nc, Nz)

# Fix B (GEOS-Chem): pressure-fixer cm on the BALANCED fluxes + PS-hybrid endpoint
cmB = ntuple(_->zeros(FT,Nc,Nc,Nz+1),6); P.compute_cs_cm_pressure_fixer!(cmB, amB, bmB, ΔB, Nc, Nz)
m_next_pfix = ntuple(_->zeros(FT,Nc,Nc,Nz),6)
P._evolve_mass_pressure_fixer!(m_next_pfix, m_cur, amB, bmB, ΔB, FT(2*steps), Nc, Nz)

# ---- (1) cm SH-UTLS roughness, 100-200 hPa band (TOA-first lev 37-42) ----
println("\n=== (1) cm SH roughness @ UTLS interfaces: Path A (diagnose) vs Fix B (pfix) ===")
for iface in 38:2:44
    rA = rough_sh(ntuple(p->cmA[p][:,:,iface],6), sh, Nc)
    rB = rough_sh(ntuple(p->cmB[p][:,:,iface],6), sh, Nc)
    @printf("  iface %2d   PathA=%.4f   FixB=%.4f   (%.0f%% of A)\n", iface, rA, rB, 100rB/rA)
end

# ---- (2) does the PS-hybrid endpoint land on the analyzed dry-PS (per column)? ----
maxps=0.0; den=0.0
for p in 1:6, j in 1:Nc, i in 1:Nc
    ca=0.0; cb=0.0; for k in 1:Nz; ca+=m_next[p][i,j,k]; cb+=m_next_pfix[p][i,j,k]; end
    maxps=max(maxps,abs(cb-ca)); den=max(den,abs(ca))
end
@printf("\n=== (2) column-mass landing: max|Σm_pfix − Σm_analyzed|/maxcol = %.3e (expect ~0 ⇒ lands on analyzed PS) ===\n", maxps/den)

# ---- (3) replay gate: m_evolved (Fix-B cm) == m_next_pfix ? ----
maxg=0.0; deng=0.0
for p in 1:6
    for v in m_next_pfix[p]; deng=max(deng,abs(v)); end
end
for p in 1:6, k in 1:Nz, j in 1:Nc, i in 1:Nc
    divh = (amB[p][i+1,j,k]-amB[p][i,j,k]) + (bmB[p][i,j+1,k]-bmB[p][i,j,k])
    mev  = m_cur[p][i,j,k] - 2steps*(divh + (cmB[p][i,j,k+1]-cmB[p][i,j,k]))
    maxg = max(maxg, abs(mev - m_next_pfix[p][i,j,k]))
end
@printf("=== (3) replay gate: max|m_evolved − m_next_pfix|/maxmass = %.3e (expect roundoff ⇒ gate closes) ===\n", maxg/deng)
println("\nDone.")
