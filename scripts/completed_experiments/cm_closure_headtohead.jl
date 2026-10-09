# ===========================================================================
# cm_closure_headtohead.jl  — STANDALONE DIAGNOSTIC (do not commit)
#
# Decide whether the GEOS-CS transport binary's UTLS "fingering" comes from the
# column-balance + endpoint-DELP cm diagnosis (Path A, current production) by
# comparing it head-to-head against an FV3 pressure-fixer closure that keeps the
# native horizontal fluxes UNBALANCED (Path B, proposed).
#
# Drives ONE window (window 1) of one GEOS-IT C180 day from the NATIVE MFXC/MFYC
# (before any column balance), through the REAL preprocessing internals:
#   geos_native_to_face_flux! → {balance_cs_column_mass_fluxes! + diagnose_cs_cm!}
#                              vs {compute_cs_cm_pressure_fixer! on native am/bm}
#
# Float64 diagnostic math throughout. See task spec for the anchor numbers:
#   native MFXC/MFYC cell-div roughness @164hPa,SH ≈ 0.225
#   written binary am/bm face-div roughness @164hPa,SH ≈ 0.321
# ===========================================================================

using AtmosTransport
using AtmosTransport.Preprocessing
using AtmosTransport.Grids
using Dates
using NCDatasets
using Statistics
using Printf
using TOML

const P  = AtmosTransport.Preprocessing
const G  = AtmosTransport.Grids

const GRAV   = 9.80665
const MASS_FLUX_DT = 450.0
const DT_MET = 3600.0
const NATIVE_RAW = expanduser("~/data/AtmosTransport/met/geosit/C180/daily/raw/20211201/GEOSIT.20211201.CTM_A1.C180.nc")
const CONFIG = joinpath(@__DIR__, "..", "..",
    "config/preprocessing/geosit_c180_dec2021_catrine_f32_fullL72.toml")
const DATE = Date(2021, 12, 1)

# Anchor-faithful UTLS levels (TOA-first, validated by the 164hPa anchor below).
# 164 hPa ↔ native surface-first lev 32 ↔ TOA-first lev 41 (= 72-32+1).
const LEV_164 = 41   # TOA-first
const LEV_250 = 45   # ~native surface-first 28
const LEV_100 = 37   # ~native surface-first 36
const UTLS = 37:45

# ---------------------------------------------------------------------------
# Roughness metric — EXACT match to the session anchor (geos_binary_flux_-
# divergence_noise_2026_06_03.md): normalized grid-scale "averaging-Laplacian"
# roughness, GLOBAL (all 6 panels, interior 2:Nc-1), no lat mask:
#
#   L[i,j] = f[i,j] - 0.25*(f[i+1,j]+f[i-1,j]+f[i,j+1]+f[i,j-1])
#   rough  = sqrt(nanmean(L^2 over interior)) / nanstd(f over full panels)
#
# Reproduces native MFXC/MFYC div = 0.225 and binary am/bm div = 0.321.
# A SH-only (lat<-30) variant is also provided for the cm comparison, but the
# headline anchors are GLOBAL.
# ---------------------------------------------------------------------------
function rough_global(field_panels, Nc)
    laps = Float64[]; vals = Float64[]
    for p in 1:6
        f = field_panels[p]
        for j in 2:Nc-1, i in 2:Nc-1
            nb = f[i+1, j] + f[i-1, j] + f[i, j+1] + f[i, j-1]
            (isnan(f[i, j]) || isnan(nb)) && continue
            push!(laps, f[i, j] - 0.25 * nb)
        end
        for v in f; isnan(v) || push!(vals, v); end
    end
    isempty(laps) && return NaN
    return sqrt(mean(abs2, laps)) / std(vals)
end

function rough_sh(field_panels, mask_panels, Nc)
    laps = Float64[]; vals = Float64[]
    for p in 1:6
        f = field_panels[p]; m = mask_panels[p]
        for j in 2:Nc-1, i in 2:Nc-1
            m[i, j] || continue
            nb = f[i+1, j] + f[i-1, j] + f[i, j+1] + f[i, j-1]
            (isnan(f[i, j]) || isnan(nb)) && continue
            push!(laps, f[i, j] - 0.25 * nb)
        end
        for j in 1:Nc, i in 1:Nc
            m[i, j] && !isnan(f[i, j]) && push!(vals, f[i, j])
        end
    end
    isempty(laps) && return NaN
    return sqrt(mean(abs2, laps)) / std(vals)
end

# Face-divergence of staggered (am,bm): div[i,j] = (am[i]-am[i+1])+(bm[j]-bm[j+1])
# Sign matches diagnose_cs_cm!/pressure_fixer convergence convention. NaN the
# first interior ring (i=1/j=1) to mirror the numpy native stencil's masked edge.
function face_div(am, bm, k, Nc)
    d = fill(NaN, Nc, Nc)
    @inbounds for j in 2:Nc, i in 2:Nc
        d[i, j] = (am[i, j, k] - am[i+1, j, k]) + (bm[i, j, k] - bm[i, j+1, k])
    end
    return d
end

# ---------------------------------------------------------------------------
# Build grid / settings / vertical exactly as the production native path does.
# ---------------------------------------------------------------------------
println("=== Building grid / settings / vertical (mirror of _process_day_native) ===")
cfg = TOML.parsefile(CONFIG)
FT = Float64   # diagnostic in F64

grid = P.build_target_geometry(cfg["grid"], FT)
Nc = grid.Nc
@assert Nc == 180

src_cfg = cfg["source"]
toml_relpath = String(src_cfg["toml"])
toml_path = isabspath(toml_relpath) ? toml_relpath :
            joinpath(@__DIR__, "..", "..", toml_relpath)
settings_kwargs = (root_dir = AtmosTransport.expand_data_path(String(src_cfg["root_dir"])),
                   include_surface = false, include_convection = false,
                   include_vdiff_fields = false,
                   coefficients_file = AtmosTransport.expand_data_path(
                       String(cfg["vertical"]["coefficients"])))
settings = P.load_met_settings(toml_path; settings_kwargs...)
@assert settings.mass_flux_dt == MASS_FLUX_DT "mass_flux_dt=$(settings.mass_flux_dt)"

vc = P.load_hybrid_coefficients(AtmosTransport.expand_data_path(settings.coefficients_file))
vertical = P._build_native_vertical_setup(cfg["vertical"], vc, FT)
Nz = vertical.Nz
Nz_native = vertical.Nz_native
@assert Nz == 72 && Nz_native == 72 "identity transform expected, got Nz=$Nz Nz_native=$Nz_native"
println("  Nc=$Nc  Nz=$Nz (identity, TOA-first)")

# ΔB[k] = B_interface[k+1] - B_interface[k], TOA-first (same orientation as am/bm)
Bifc = Float64.(vertical.merged_vc.B)       # length Nz+1 = 73, TOA-first
@assert length(Bifc) == Nz + 1
ΔB = [Bifc[k+1] - Bifc[k] for k in 1:Nz]
@printf("  ΔB: ΣΔB=%.6f  (expect ≈1)   ΔB[TOA lev1..3]=%.3e,%.3e,%.3e   ΔB[surf lev70..72]=%.4f,%.4f,%.4f\n",
        sum(ΔB), ΔB[1], ΔB[2], ΔB[3], ΔB[70], ΔB[71], ΔB[72])

# cell-center latitudes per panel (binary i,j layout; identical layout for native)
lats = ntuple(p -> Float64.(P.panel_cell_center_lonlat(grid.mesh, p)[2]), 6)
sh_mask = ntuple(p -> lats[p] .< -30.0, 6)
nsh = sum(sum, sh_mask)
@printf("  SH cells (lat<-30): %d of %d (%.1f%%)\n", nsh, 6*Nc*Nc, 100*nsh/(6*Nc*Nc))

# ---------------------------------------------------------------------------
# Read window 1 through the REAL reader (TOA-first, am=MFXC/mass_flux_dt, m=DELP_dry).
# ---------------------------------------------------------------------------
println("\n=== Reading window 1 via real GEOS reader ===")
reader = P.open_reader(settings, DATE, FT; chain_mass = false, next_day_handle = true)
nw = P.windows_per_day(reader)
println("  windows/day=$nw  orientation=$(reader.handles.orientation)")
raw = P.allocate_raw_window(settings; FT = FT, Nz = Nz_native)
P.read_window!(raw, reader, 1)

g = FT(GRAV); inv_g = inv(g)
cell_areas = grid.mesh.cell_areas
steps_per_met = round(Int, FT(DT_MET) / FT(MASS_FLUX_DT))   # = 8
@assert steps_per_met == 8
flux_scale = FT(MASS_FLUX_DT / 2) / g                        # ≈ 22.94
@printf("  steps_per_met=%d  flux_scale=%.4f\n", steps_per_met, flux_scale)

# ---------------------------------------------------------------------------
# Native MFXC/MFYC → staggered face am/bm (BEFORE balance).
# raw.am = MFXC/mass_flux_dt (TOA-first); geos_native_to_face_flux! multiplies by
# flux_scale = (mass_flux_dt/2)/g. Net = MFXC/(2g) per half-sweep, the binary unit.
# ---------------------------------------------------------------------------
am_nat = ntuple(_ -> zeros(FT, Nc+1, Nc, Nz_native), 6)
bm_nat = ntuple(_ -> zeros(FT, Nc, Nc+1, Nz_native), 6)
P.geos_native_to_face_flux!(am_nat, bm_nat, raw.am, raw.bm,
                            grid.mesh.connectivity, Nc, Nz_native, flux_scale)

# Apply the (identity) vertical transform am_native_v4 → am_v4 exactly like the
# production ingest does (no-op reorder here, but keeps the pipeline faithful).
am_v4_nat = ntuple(_ -> zeros(FT, Nc+1, Nc, Nz), 6)
bm_v4_nat = ntuple(_ -> zeros(FT, Nc, Nc+1, Nz), 6)
for p in 1:6
    P.apply_vertical!(am_v4_nat[p], am_nat[p], vertical.plan, P.MassFluxField())
    P.apply_vertical!(bm_v4_nat[p], bm_nat[p], vertical.plan, P.MassFluxField())
end

# ---------------------------------------------------------------------------
# Seed mass m_cur / m_next_target from DELP_dry endpoints (kg).
# ---------------------------------------------------------------------------
m_cur  = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
m_next = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
m_native_kg = ntuple(_ -> zeros(FT, Nc, Nc, Nz_native), 6)
for p in 1:6
    P._delp_pa_to_air_mass_kg!(m_native_kg[p], raw.m[p], cell_areas, inv_g)
    P.apply_vertical!(m_cur[p], m_native_kg[p], vertical.plan, P.MassField())
end
for p in 1:6
    P._delp_pa_to_air_mass_kg!(m_native_kg[p], raw.m_next[p], cell_areas, inv_g)
    P.apply_vertical!(m_next[p], m_native_kg[p], vertical.plan, P.MassField())
end

# Steps for window 1 = source steps_per_met (this diagnostic does NOT run the
# adaptive-substep schedule; we drive one fixed window). The production code at
# the catrine binary used adaptive substeps, but the cm SHAPE comparison and the
# anchor reproduction are at the source steps_per_met scaling.
steps = steps_per_met

# dm = (m_next - m_cur) / (2*steps)
dm = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
P.fill_cs_window_mass_tendency!(dm, m_cur, m_next, steps)

# ===========================================================================
# #1  div(am,bm) roughness BEFORE and AFTER column balance (validate anchors)
#     GLOBAL metric (matches the session anchor: 0.225 native, 0.321 binary).
# ===========================================================================
println("\n=== #1  Flux-divergence roughness @164hPa (GLOBAL, anchor validation) ===")

# (a) NATIVE MFXC/MFYC cell divergence on the RAW native (surface-first) arrays.
#     numpy axis convention (transcript): array (nf,Y,X); MFXC differenced along
#     Y, MFYC along X. Julia ds["MFXC"] is (X,Y,nf) ⟹ numpy-Y=Julia-j, numpy-X=
#     Julia-i ⟹ D = (mfxc[i,j]-mfxc[i,j-1]) + (mfyc[i,j]-mfyc[i-1,j]). The first
#     ring (i=1 or j=1) is left NaN, mirroring numpy's D[:,1:,1:] masked edge.
function native_div_panels(ncfile, native_lev, Nc)
    ds = NCDataset(ncfile)
    mfxc = Float64.(ds["MFXC"][:, :, :, native_lev, 1])  # (X,Y,nf)
    mfyc = Float64.(ds["MFYC"][:, :, :, native_lev, 1])
    close(ds)
    return ntuple(p -> begin
        d = fill(NaN, Nc, Nc)
        @inbounds for j in 2:Nc, i in 2:Nc
            d[i, j] = (mfxc[i, j, p] - mfxc[i, j-1, p]) +
                      (mfyc[i, j, p] - mfyc[i-1, j, p])
        end
        d
    end, 6)
end

native_lev_164 = Nz - LEV_164 + 1   # TOA-first 41 → surface-first 32
@printf("  native lev for 164hPa (surface-first) = %d  (TOA-first %d)\n",
        native_lev_164, LEV_164)
div_native = native_div_panels(NATIVE_RAW, native_lev_164, Nc)
r_native = rough_global(div_native, Nc)
@printf("  [a] NATIVE MFXC/MFYC div roughness        = %.4f   (anchor 0.225)\n", r_native)

# (b) BINARY am/bm face-div BEFORE balance (native staggered, v4) at TOA lev 41.
div_before = ntuple(p -> face_div(am_v4_nat[p], bm_v4_nat[p], LEV_164, Nc), 6)
r_before = rough_global(div_before, Nc)
@printf("  [b] face-div BEFORE balance (native am/bm)= %.4f\n", r_before)

# (c) Column-balance a COPY of native am/bm (Path A horizontal step), then face-div.
am_A = ntuple(p -> copy(am_v4_nat[p]), 6)
bm_A = ntuple(p -> copy(bm_v4_nat[p]), 6)
bal_diag = P.balance_cs_column_mass_fluxes!(am_A, bm_A, m_cur, m_next,
            grid.face_table, grid.cell_degree, steps, grid.poisson_scratch)
div_after = ntuple(p -> face_div(am_A[p], bm_A[p], LEV_164, Nc), 6)
r_after = rough_global(div_after, Nc)
@printf("  [c] face-div AFTER  column balance        = %.4f   (anchor 0.321)\n", r_after)
@printf("      balance: max_pre=%.3e max_post=%.3e col_resid=%.3e\n",
        bal_diag.max_pre_residual, bal_diag.max_post_residual,
        bal_diag.final_column_projected_residual)

# ===========================================================================
# #2  cm roughness, two closures from the SAME native am/bm
# ===========================================================================
println("\n=== #2  cm roughness @164hPa SH — Path A vs Path B ===")

# Path A: balanced am/bm → diagnose_cs_cm!(cm_A, am_A, bm_A, dm, m_cur)
cm_A = ntuple(_ -> zeros(FT, Nc, Nc, Nz+1), 6)
P.diagnose_cs_cm!(cm_A, am_A, bm_A, dm, m_cur, Nc, Nz)

# Path B: NATIVE (unbalanced) am/bm → compute_cs_cm_pressure_fixer!(cm_B, ..., ΔB)
cm_B = ntuple(_ -> zeros(FT, Nc, Nc, Nz+1), 6)
P.compute_cs_cm_pressure_fixer!(cm_B, am_v4_nat, bm_v4_nat, ΔB, Nc, Nz)

# cm lives at Nz+1 interfaces. Interface nearest center lev L is interface L+1
# (bottom of layer L). Report both GLOBAL and SH (lat<-30) roughness.
cm_iface(cm, iface) = ntuple(p -> cm[p][:, :, iface], 6)
println("  level     GLOBAL cm_A  cm_B   B/A  |  SH cm_A   cm_B   B/A")
for (name, lev) in (("164hPa", LEV_164), ("100hPa", LEV_100), ("250hPa", LEV_250))
    iface = lev + 1
    fA = cm_iface(cm_A, iface); fB = cm_iface(cm_B, iface)
    gA = rough_global(fA, Nc); gB = rough_global(fB, Nc)
    sA = rough_sh(fA, sh_mask, Nc); sB = rough_sh(fB, sh_mask, Nc)
    @printf("  %-7s    %.4f  %.4f  %.3f  |  %.4f  %.4f  %.3f\n",
            name, gA, gB, gB/gA, sA, sB, sB/sA)
end

# ===========================================================================
# #2c  Path C — moisture-corrected FILTERED closure (the durable fix candidate).
#   dm_pf[k]      = ΔB[k]·pit           (pit = column horiz convergence; smooth)
#   residual      = dm_dry − dm_pf      (the moisture/temporal mismatch; grid-scale)
#   dm_corrected  = dm_pf + smooth(residual)   (grid-scale fingering filtered out)
#   cm_C          = diagnose(am_nat, bm_nat, dm_corrected, m_cur)   (consistent w/ m)
#   m_next_C      = m_cur + 2·steps·dm_corrected  (≈ DELP, grid-scale removed)
# cm_C and m_next_C come from the SAME dm_corrected → replay closes (no blowup),
# yet the grid-scale moisture artifact is filtered → less fingering. niter=0 ⇒
# Path A (endpoint); large niter ⇒ approaches Path B (pure pf).
# ===========================================================================
println("\n=== #2c  Path C: moisture-corrected filtered closure (native am/bm) ===")

function _col_pit(am, bm, Nc, Nz)
    pit = ntuple(_ -> zeros(FT, Nc, Nc), 6)
    @inbounds for p in 1:6, j in 1:Nc, i in 1:Nc
        s = zero(FT)
        for k in 1:Nz
            s += (am[p][i, j, k] - am[p][i+1, j, k]) +
                 (bm[p][i, j, k] - bm[p][i, j+1, k])
        end
        pit[p][i, j] = s
    end
    pit
end
function _smooth_panels!(f, niter, w, Nc, Nz)
    sc = ntuple(p -> similar(f[p]), 6)
    for _ in 1:niter
        for p in 1:6
            copyto!(sc[p], f[p])
            @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
                im = max(i-1,1); ip = min(i+1,Nc); jm = max(j-1,1); jp = min(j+1,Nc)
                avg = (sc[p][im,j,k] + sc[p][ip,j,k] + sc[p][i,jm,k] + sc[p][i,jp,k]) / 4
                f[p][i,j,k] = (1-w)*sc[p][i,j,k] + w*avg
            end
        end
    end
end

pit = _col_pit(am_v4_nat, bm_v4_nat, Nc, Nz)
sA164 = rough_sh(cm_iface(cm_A, LEV_164+1), sh_mask, Nc)
sB164 = rough_sh(cm_iface(cm_B, LEV_164+1), sh_mask, Nc)
println("  reference: SH cm rough @164 — A(endpoint)=$(round(sA164,digits=4))  B(pure pf)=$(round(sB164,digits=4))")
println("  niter   SH cm rough @164   vs A    min(m_next)/min(DELP)   replay_resid")
minDELP = minimum(minimum(m_next[p]) for p in 1:6)
for niter in (0, 4, 16, 64)
    dmc = ntuple(p -> begin
        d = similar(dm[p])
        @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
            d[i,j,k] = dm[p][i,j,k] - FT(ΔB[k]) * pit[p][i,j]   # residual
        end
        d
    end, 6)
    _smooth_panels!(dmc, niter, FT(0.5), Nc, Nz)
    @inbounds for p in 1:6, k in 1:Nz, j in 1:Nc, i in 1:Nc
        dmc[p][i,j,k] += FT(ΔB[k]) * pit[p][i,j]               # + dm_pf
    end
    cm_C = ntuple(_ -> zeros(FT, Nc, Nc, Nz+1), 6)
    P.diagnose_cs_cm!(cm_C, am_v4_nat, bm_v4_nat, dmc, m_cur, Nc, Nz)
    minmn = Inf; maxres = 0.0
    @inbounds for p in 1:6, j in 1:Nc, i in 1:Nc
        for k in 1:Nz
            mn = m_cur[p][i,j,k] + 2*steps*dmc[p][i,j,k]
            minmn = min(minmn, mn)
            # replay residual: horiz_conv + (cm[k]-cm[k+1]) should == 2*steps*dmc
            hc = (am_v4_nat[p][i,j,k]-am_v4_nat[p][i+1,j,k]) + (bm_v4_nat[p][i,j,k]-bm_v4_nat[p][i,j+1,k])
            res = hc + (cm_C[p][i,j,k]-cm_C[p][i,j,k+1]) - 2*steps*dmc[p][i,j,k]
            maxres = max(maxres, abs(res))
        end
    end
    sC = rough_sh(cm_iface(cm_C, LEV_164+1), sh_mask, Nc)
    @printf("  %5d   %.4f            %.2f×   %+.3e / %.3e        %.2e\n",
            niter, sC, sC/sA164, minmn, minDELP, maxres)
end

# ===========================================================================
# #3  Mass-closure / surface-pressure drift per path over the window.
#     m_new[k] = m_cur[k] + horiz_conv[k] + (cm[k] - cm[k+1])
#     Column-integrated mass error vs m_next → equivalent ps drift (Pa).
# ===========================================================================
println("\n=== #3  Column mass-closure → equivalent ps drift (Pa) ===")

function ps_drift(am, bm, cm, m_cur, m_next, dm, cell_areas, g, Nc, Nz; use_dm::Bool)
    # Replay one window's air-mass update and compare column-integrated mass to
    # the endpoint m_next. The replay uses the SAME convergence the diagnosis
    # closed against: per substep, the net dm step is dm = (m_next-m_cur)/(2*steps),
    # and over the window the cumulative tendency telescopes to (m_next - m_cur)
    # IF the cm/horiz closure is exact. We test the per-cell continuity:
    #   resid[k] = horiz_conv[k] + (cm[k]-cm[k+1]) - dm_target[k]
    # then column-sum |Σ_k resid| · (2*steps) → kg error over the window, → Pa.
    drift = Float64[]
    for p in 1:6
        amp, bmp, cmp = am[p], bm[p], cm[p]
        mc = m_cur[p]; mn = m_next[p]
        for j in 1:Nc, i in 1:Nc
            col_kg = 0.0
            for k in 1:Nz
                conv = (amp[i, j, k] - amp[i+1, j, k]) + (bmp[i, j, k] - bmp[i, j+1, k])
                cmdiv = cmp[i, j, k] - cmp[i, j, k+1]
                tgt = use_dm ? dm[p][i, j, k] : (mn[i, j, k] - mc[i, j, k]) / (2*steps)
                col_kg += (conv + cmdiv - tgt)
            end
            # per-substep column residual in kg; over window = ×(2*steps)
            col_kg_window = col_kg * (2*steps)
            push!(drift, col_kg_window * g / cell_areas[i, j])  # Pa
        end
    end
    return drift
end

# Path A: balanced am/bm, dm target, cm_A
dA = ps_drift(am_A, bm_A, cm_A, m_cur, m_next, dm, cell_areas, g, Nc, Nz; use_dm = true)
# Path B: native am/bm, cm_B; the pressure-fixer enforces cm[Nz+1]=0 and does NOT
# target m_next — so we compare its implied per-cell mass update vs m_next.
dB = ps_drift(am_v4_nat, bm_v4_nat, cm_B, m_cur, m_next, dm, cell_areas, g, Nc, Nz; use_dm = true)

# restrict to SH for reporting (and report global too)
shvec = vcat([vec(sh_mask[p]) for p in 1:6]...)
function drift_stats(d, mask, label)
    dsh = d[mask]
    @printf("  %-8s  global: max|Δps|=%.4e Pa  RMS=%.4e Pa | SH: max=%.4e Pa  RMS=%.4e Pa  mean=%.3e\n",
            label, maximum(abs, d), sqrt(mean(abs2, d)),
            maximum(abs, dsh), sqrt(mean(abs2, dsh)), mean(dsh))
end
drift_stats(dA, shvec, "Path A")
drift_stats(dB, shvec, "Path B")
@printf("  (GEOS-IT ps ~1e5 Pa; Path B max drift fraction = %.3e)\n",
        maximum(abs, dB) / 1e5)

# ===========================================================================
# #4  Vertical profile of cm roughness (UTLS levels) for both paths.
# ===========================================================================
println("\n=== #4  cm roughness vertical profile (UTLS) — GLOBAL and SH ===")
println("  TOAlev  hPa    | GLOBAL cm_A  cm_B   B/A | SH cm_A   cm_B   B/A")
Aifc = Float64.(vertical.merged_vc.A)
psref = 1.0e5
for lev in UTLS
    iface = lev + 1
    phpa = (Aifc[iface] + Bifc[iface]*psref) / 100.0
    fA = cm_iface(cm_A, iface); fB = cm_iface(cm_B, iface)
    gA = rough_global(fA, Nc); gB = rough_global(fB, Nc)
    sA = rough_sh(fA, sh_mask, Nc); sB = rough_sh(fB, sh_mask, Nc)
    @printf("  %4d  %6.1f  |  %.4f  %.4f  %.3f |  %.4f  %.4f  %.3f\n",
            lev, phpa, gA, gB, gB/gA, sA, sB, sB/sA)
end

P.close_reader!(reader)
println("\n=== done ===")
