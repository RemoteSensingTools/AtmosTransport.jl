# ===========================================================================
# fingerfix_proto_anisotropic-divergence-damping.jl  — STANDALONE PROTOTYPE
#   (diagnostic; do not commit to the production pipeline)
#
# METHOD id: anisotropic-divergence-damping
#   "Roughness-gated, UTLS-masked anisotropic divergence damping in the flux
#    null space."
#
# IDEA
#   The SH-UTLS fingering is the grid-noisy MFXC<->DELP residual M forced into
#   cm via the offline closure cm[k+1]=cm[k]+div_h[k]-dm[k].  M carries no
#   physical signal (proven: cor(M,DMDTANA)~=0), so its grid-structure may be
#   smoothed freely PROVIDED discrete continuity is preserved exactly.
#
#   We apply a variable-coefficient biharmonic (divergence-damping/hyper-
#   diffusion) smoothing to the PER-LAYER horizontal flux convergence
#   div_h = div_h(am,bm), through a mass-flux POTENTIAL lambda solved on the
#   global cube: L*lambda = rhs, then am/bm += grad(lambda) (apply_cs_flux_
#   correction!).  Because the correction is a discrete gradient of a scalar
#   potential, it changes div_h but leaves the GLOBAL continuity machinery
#   intact: dm is held FIXED, and diagnose_cs_cm! re-derives cm from the
#   corrected (am,bm,dm) so the replay gate
#       dm = div_h(am,bm) + (cm[k]-cm[k+1])
#   holds EXACTLY by construction (cm is defined as the running cumsum that
#   closes it).  The smoothing increment lives inside continuity's null space.
#
#   rhs = -L( nu .* L(div_h_level) ), projected mean-zero, with
#     nu[i,j] = nu0 * sigmoid((rho[i,j]-rough_target)/scale) * mask_SH_UTLS
#   rho = per-cell |Laplacian|/std  (the exact score_binary.py roughness cell
#   indicator) so damping only fires on rough cells inside the SH-UTLS band,
#   with smooth lat (~-30) and pressure (80/300 hPa) tapers to avoid edge kinks.
#
#   This is anisotropic only in the sense that nu is spatially varying (gated
#   by roughness + masked to SH-UTLS); the operator itself is the isotropic
#   graph Laplacian.  Biharmonic (L .* nu .* L) => scale-selective: damps
#   grid-scale 2-delta noise hard, leaves large scales nearly untouched.
#
# CONTROL
#   In the same script we build the DIRTY (production endpoint) closure on the
#   identical window(s) and score r_vdiv on it, so the comparison is apples-to-
#   apples.  Baselines: DIRTY 0.294, CLEAN 0.227.
#
# METRIC (validated; matches ~/data/AtmosTransport/AI-Training/score_binary.py)
#   r_vdiv = SH-UTLS (lat<-30, layer-mid 80-300 hPa) mean over levels of
#       RMS_ij[ vdiv - 0.25*(4-nbr) ] / std_ij(vdiv),  vdiv=cm[k]-cm[k+1]
#   computed over panel-interior SH cells.  ALSO report continuity RMS/colmass.
#
# RUN (direct binary; the juliaup shim blocks on a config lock):
#   ~/.julia/juliaup/julia-1.12.5+0.x64.linux.gnu/bin/julia \
#       --project=/home/cfranken/code/gitHub/AtmosTransportModel \
#       scripts/completed_experiments/fingerfix_proto_anisotropic-divergence-damping.jl
# ===========================================================================

using AtmosTransport
using AtmosTransport.Preprocessing
using AtmosTransport.Grids
using Dates
using NCDatasets
using Statistics
using Printf
using TOML

const P = AtmosTransport.Preprocessing

const GRAV         = 9.80665
const MASS_FLUX_DT = 450.0
const DT_MET       = 3600.0
const CONFIG = joinpath(@__DIR__, "..", "..",
    "config/preprocessing/geosit_c180_dec2021_catrine_f32_fullL72.toml")
const RAW = expanduser("~/data/AtmosTransport/met/geosit/C180/raw_catrine")

ctm_path(d::Date) = joinpath(RAW, Dates.format(d, "yyyymmdd"),
    "GEOSIT.$(Dates.format(d,"yyyymmdd")).CTM_A1.C180.nc")

# UTLS / SH metric band
const UTLS_LO_HPA = 80.0
const UTLS_HI_HPA = 300.0
const SH_LAT      = -30.0

# ---------------------------------------------------------------------------
# Knobs (tuned to push SH-UTLS r_vdiv toward CLEAN 0.227 while keeping C1~0).
# nu0 has units of (1/L)-ish in the graph metric; biharmonic strength.
# rough_target/scale gate which cells get damped; mask tapers avoid edge kinks.
# ---------------------------------------------------------------------------
# STABILITY: the per-pass update is div_h <- div_h - nu*L^2(div_h). The graph
# Laplacian L=D-A has max eigenvalue ~8 (degree-4 mesh), so L^2 reaches ~64.
# Stable smoothing needs nu*lambda_max^2 < 1  =>  nu < ~1/64 ~ 0.0156.  The
# grid-scale (2-delta) mode (lambda~8) is damped by (1-64*nu); nu=1/64 nulls it
# in one pass.  Use nu just under the bound and several gentle passes.
const NU0          = 0.0145   # biharmonic damping strength (just under 1/64)
const ROUGH_TARGET = 0.04     # gate center (cell roughness rho)
const ROUGH_SCALE  = 0.04     # gate softness
const LAT_TAPER_W  = 8.0      # deg; sigmoid half-width of the SH lat mask edge
const P_TAPER_FRAC = 0.15     # fractional pressure taper at 80/300 hPa band edges
const N_PASSES     = 9        # repeated damp passes (each is gate-exact)

# ---------------------------------------------------------------------------
# Setup: grid, settings, vertical (mirror of the production native path).
# ---------------------------------------------------------------------------
function build_setup(FT)
    cfg  = TOML.parsefile(CONFIG)
    grid = P.build_target_geometry(cfg["grid"], FT)
    Nc   = grid.Nc
    src  = cfg["source"]
    toml_relpath = String(src["toml"])
    toml_path = isabspath(toml_relpath) ? toml_relpath :
                joinpath(@__DIR__, "..", "..", toml_relpath)
    settings = P.load_met_settings(toml_path;
        root_dir = AtmosTransport.expand_data_path(String(src["root_dir"])),
        include_surface = false, include_convection = false,
        include_vdiff_fields = false,
        coefficients_file = AtmosTransport.expand_data_path(
            String(cfg["vertical"]["coefficients"])))
    vc       = P.load_hybrid_coefficients(
                   AtmosTransport.expand_data_path(settings.coefficients_file))
    vertical = P._build_native_vertical_setup(cfg["vertical"], vc, FT)
    Nz       = vertical.Nz

    Aifc = Float64.(vertical.merged_vc.A)   # length Nz+1, TOA-first (Pa)
    Bifc = Float64.(vertical.merged_vc.B)
    # layer-mid pressure at a reference ps (scorer uses nominal ~1000 hPa)
    psref = 1.0e5
    pmid = [0.5*((Aifc[k]+Aifc[k+1]) + (Bifc[k]+Bifc[k+1])*psref) / 100.0
            for k in 1:Nz]                  # hPa, TOA-first
    utls_levels = findall(p -> UTLS_LO_HPA <= p <= UTLS_HI_HPA, pmid)

    lats = ntuple(p -> Float64.(P.panel_cell_center_lonlat(grid.mesh, p)[2]), 6)
    sh   = ntuple(p -> lats[p] .< SH_LAT, 6)

    return (cfg=cfg, grid=grid, Nc=Nc, Nz=Nz, settings=settings,
            vertical=vertical, Aifc=Aifc, Bifc=Bifc, pmid=pmid,
            utls_levels=utls_levels, lats=lats, sh=sh, psref=psref)
end

# ---------------------------------------------------------------------------
# Read one window: native MFXC/MFYC -> v4 face am/bm; DELP-dry endpoints -> m.
# Returns am_nat,bm_nat (native unbalanced), m_cur,m_next, dm, steps.
# (Mirrors cm_closure_headtohead.jl ingest, the production native path.)
# ---------------------------------------------------------------------------
function read_window(s, date::Date, win::Int, FT)
    grid = s.grid; Nc = s.Nc; Nz = s.Nz
    reader = P.open_reader(s.settings, date, FT; chain_mass = false,
                           next_day_handle = true)
    raw = P.allocate_raw_window(s.settings; FT = FT, Nz = Nz)
    P.read_window!(raw, reader, win)
    or = reader.handles.orientation

    g = FT(GRAV); inv_g = inv(g)
    cell_areas = grid.mesh.cell_areas
    steps = round(Int, FT(DT_MET) / FT(MASS_FLUX_DT))     # 8
    flux_scale = FT(MASS_FLUX_DT / 2) / g                 # MFXC/mass_flux_dt -> /(2g)

    am_nat = ntuple(_ -> zeros(FT, Nc+1, Nc, Nz), 6)
    bm_nat = ntuple(_ -> zeros(FT, Nc, Nc+1, Nz), 6)
    P.geos_native_to_face_flux!(am_nat, bm_nat, raw.am, raw.bm,
                                grid.mesh.connectivity, Nc, Nz, flux_scale)

    m_cur  = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    m_next = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    tmp    = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    for p in 1:6
        P._delp_pa_to_air_mass_kg!(tmp[p], raw.m[p], cell_areas, inv_g)
        copyto!(m_cur[p], tmp[p])
        P._delp_pa_to_air_mass_kg!(tmp[p], raw.m_next[p], cell_areas, inv_g)
        copyto!(m_next[p], tmp[p])
    end

    dm = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    P.fill_cs_window_mass_tendency!(dm, m_cur, m_next, steps)

    P.close_reader!(reader)
    return (am_nat=am_nat, bm_nat=bm_nat, m_cur=m_cur, m_next=m_next,
            dm=dm, steps=steps, or=or)
end

# ---------------------------------------------------------------------------
# Metric: r_vdiv over SH-UTLS, exact score_binary.py formula.
#   vdiv[k] = cm[:,:,k] - cm[:,:,k+1]  (== div_h at every layer when continuity
#   holds).  Per UTLS level k: RMS_interior_SH[ vdiv - 0.25*(4-nbr) ] /
#   std_interior_SH(vdiv).  Average over UTLS levels.
# Panel-interior (2:Nc-1) + SH mask, same as the scorer's jj/ii + sh.
# ---------------------------------------------------------------------------
function r_vdiv(cm, s; sh=s.sh, levels=s.utls_levels)
    Nc = s.Nc
    rs = Float64[]
    for k in levels
        laps = Float64[]; vals = Float64[]
        for p in 1:6
            cmp = cm[p]; m = sh[p]
            for j in 2:Nc-1, i in 2:Nc-1
                m[i, j] || continue
                v   = cmp[i, j, k]   - cmp[i, j, k+1]
                vip = cmp[i+1, j, k] - cmp[i+1, j, k+1]
                vim = cmp[i-1, j, k] - cmp[i-1, j, k+1]
                vjp = cmp[i, j+1, k] - cmp[i, j+1, k+1]
                vjm = cmp[i, j-1, k] - cmp[i, j-1, k+1]
                (isnan(v)||isnan(vip)||isnan(vim)||isnan(vjp)||isnan(vjm)) && continue
                push!(laps, v - 0.25*(vip+vim+vjp+vjm))
                push!(vals, v)
            end
        end
        (length(vals) < 3) && continue
        sd = std(vals)
        sd > 0 && push!(rs, sqrt(mean(abs2, laps)) / sd)
    end
    return isempty(rs) ? NaN : mean(rs)
end

# Continuity RMS residual / colmass for a (am,bm,cm,dm,m) set.
#   resid = dm - div_h(am,bm) - (cm[k]-cm[k+1]);  RMS over panel interior / colmass.
function continuity_rms(am, bm, cm, dm, m, Nc, Nz)
    ss = 0.0; n = 0; colmax = 0.0
    for p in 1:6
        amp, bmp, cmp, dmp, mp = am[p], bm[p], cm[p], dm[p], m[p]
        @inbounds for j in 1:Nc, i in 1:Nc
            col = 0.0
            for k in 1:Nz; col += mp[i, j, k]; end
            colmax = max(colmax, col)
        end
        @inbounds for j in 2:Nc-1, i in 2:Nc-1
            for k in 1:Nz
                divh = (amp[i,j,k]-amp[i+1,j,k]) + (bmp[i,j,k]-bmp[i,j+1,k])
                vdiv = cmp[i,j,k] - cmp[i,j,k+1]
                r = dmp[i,j,k] - divh - vdiv
                ss += r*r; n += 1
            end
        end
    end
    return sqrt(ss / n) / colmax
end

# ---------------------------------------------------------------------------
# DIRTY (production endpoint) closure on native am/bm.
#   column-balance native am/bm to (m_next-m_cur), then diagnose_cs_cm!.
# Returns balanced (am,bm,cm).
# ---------------------------------------------------------------------------
function dirty_closure(s, w, FT)
    Nc = s.Nc; Nz = s.Nz
    am = ntuple(p -> copy(w.am_nat[p]), 6)
    bm = ntuple(p -> copy(w.bm_nat[p]), 6)
    P.balance_cs_column_mass_fluxes!(am, bm, w.m_cur, w.m_next,
        s.grid.face_table, s.grid.cell_degree, w.steps, s.grid.poisson_scratch)
    cm = ntuple(_ -> zeros(FT, Nc, Nc, Nz+1), 6)
    P.diagnose_cs_cm!(cm, am, bm, w.dm, w.m_cur, Nc, Nz)
    return (am=am, bm=bm, cm=cm)
end

# ---------------------------------------------------------------------------
# Per-cell roughness indicator rho[i,j] for a single-level field f (panels).
#   rho = |f - 0.25*(4-nbr)| / std_panel(f)   (score_binary.py cell formula).
# std taken over the SH-UTLS-masked cells of THIS field (scale-invariant gate).
# Edge cells (i or j in {1,Nc}) get rho=0 (no damping there; halos handled by
# the potential solve + mirror sync).
# ---------------------------------------------------------------------------
function cell_roughness(f, sd, Nc)
    rho = ntuple(_ -> zeros(Float64, Nc, Nc), 6)
    sd <= 0 && return rho
    for p in 1:6
        fp = f[p]
        @inbounds for j in 2:Nc-1, i in 2:Nc-1
            nb = fp[i+1,j] + fp[i-1,j] + fp[i,j+1] + fp[i,j-1]
            rho[p][i,j] = abs(fp[i,j] - 0.25*nb) / sd
        end
    end
    return rho
end

@inline sigmoid(x) = 1.0 / (1.0 + exp(-x))

# Smooth SH-UTLS mask weight in [0,1]: lat sigmoid around SH_LAT, pressure
# bump that tapers to 0 at the 80/300 hPa band edges.
function shutls_weight(s, p, i, j, phpa)
    lat = s.lats[p][i, j]
    wlat = sigmoid((SH_LAT - lat) / LAT_TAPER_W)      # ~1 deep SH, ~0 north of -30
    lo = UTLS_LO_HPA; hi = UTLS_HI_HPA
    tlo = lo * (1 + P_TAPER_FRAC); thi = hi * (1 - P_TAPER_FRAC)
    wp = if phpa <= lo || phpa >= hi
        0.0
    elseif phpa < tlo
        (phpa - lo) / (tlo - lo)
    elseif phpa > thi
        (hi - phpa) / (hi - thi)
    else
        1.0
    end
    return wlat * wp
end

# ---------------------------------------------------------------------------
# THE METHOD: roughness-gated UTLS anisotropic divergence damping.
#
# Starting from native am/bm, for each UTLS level:
#   1. div_h_level = per-cell horizontal convergence (global cell field).
#   2. rho = cell roughness; nu = NU0*sigmoid((rho-target)/scale)*shutls_weight.
#   3. Lf  = L(div_h)         (graph Laplacian on the global cell vector)
#      g   = nu .* Lf         (variable coefficient)
#      rhs = -L(g)            (biharmonic), projected mean-zero
#   4. solve L*lambda = rhs  (PCG, mean-zero), apply_cs_flux_correction! adds
#      grad(lambda) to am/bm at this level, then mirror-sync.
# Repeat N_PASSES.  Then column-balance to (m_next-m_cur) [restore the column
# budget the damping may have nudged] and diagnose_cs_cm! with FIXED dm.
#
# Why the gate holds: dm is untouched; cm is DEFINED by diagnose_cs_cm! as the
# running cumsum closing dm = div_h + (cm[k]-cm[k+1]) exactly.  The horizontal
# correction (a discrete gradient) only reshapes div_h's grid structure.
# ---------------------------------------------------------------------------
function damped_closure(s, w, FT; verbose=false)
    Nc = s.Nc; Nz = s.Nz; nc = s.grid.face_table.nc
    ft = s.grid.face_table; degree = s.grid.cell_degree
    am = ntuple(p -> copy(w.am_nat[p]), 6)
    bm = ntuple(p -> copy(w.bm_nat[p]), 6)

    # scratch for the potential solve
    sc = P.CSPoissonScratch(nc)
    lam = sc.psi; rhs = sc.rhs
    divh_vec = Vector{Float64}(undef, nc)   # L-input (div_h as cell field)
    Lf       = Vector{Float64}(undef, nc)
    gvec     = Vector{Float64}(undef, nc)

    # map global cell index -> (p,i,j)
    @inline function cell_pij(c)
        p = (c - 1) ÷ (Nc*Nc) + 1
        l = (c - 1) % (Nc*Nc)
        j = l ÷ Nc + 1
        i = l % Nc + 1
        return (p, i, j)
    end

    for pass in 1:N_PASSES
        for k in s.utls_levels
            phpa = s.pmid[k]
            # 1. per-cell div_h at this level into a single-level panel field
            divh_lvl = ntuple(_ -> zeros(Float64, Nc, Nc), 6)
            for p in 1:6
                amp, bmp = am[p], bm[p]
                @inbounds for j in 1:Nc, i in 1:Nc
                    divh_lvl[p][i,j] = (amp[i,j,k]-amp[i+1,j,k]) +
                                       (bmp[i,j,k]-bmp[i,j+1,k])
                end
            end

            # std over SH-UTLS-masked interior cells (gate denominator)
            svals = Float64[]
            for p in 1:6, j in 2:Nc-1, i in 2:Nc-1
                s.sh[p][i,j] && push!(svals, divh_lvl[p][i,j])
            end
            sd = length(svals) > 2 ? std(svals) : 0.0
            sd <= 0 && continue

            rho = cell_roughness(divh_lvl, sd, Nc)

            # cell field for L(): div_h as a global vector
            @inbounds for c in 1:nc
                (p,i,j) = cell_pij(c)
                divh_vec[c] = divh_lvl[p][i,j]
            end
            _cs_proj0!(divh_vec)
            P._cs_graph_laplacian_mul!(Lf, divh_vec, ft, degree)

            # nu .* Lf  with roughness gate * smooth SH-UTLS mask
            @inbounds for c in 1:nc
                (p,i,j) = cell_pij(c)
                nu = NU0 * sigmoid((rho[p][i,j] - ROUGH_TARGET) / ROUGH_SCALE) *
                     shutls_weight(s, p, i, j, phpa)
                gvec[c] = nu * Lf[c]
            end
            # rhs = -L(g)
            P._cs_graph_laplacian_mul!(rhs, gvec, ft, degree)
            @inbounds for c in 1:nc; rhs[c] = -rhs[c]; end
            _cs_proj0!(rhs)

            # solve L*lambda = rhs, then am/bm += grad(lambda) at this level
            P.solve_cs_poisson_pcg!(lam, rhs, ft, degree,
                (r=sc.r, p=sc.p, Ap=sc.Ap, z=sc.z); tol=1e-12, max_iter=5000,
                project_every=50)
            P.apply_cs_flux_correction!(am, bm, lam, ft, k)
        end
        P._sync_cs_mirrors!(am, bm, ft, Nz)
    end

    # Restore the COLUMN mass budget (the per-level damping is column-mean-zero
    # by the mean-zero projection, but re-balancing guarantees the endpoint
    # column tendency is met before diagnosing cm — keeps DIRTY's column basis).
    P.balance_cs_column_mass_fluxes!(am, bm, w.m_cur, w.m_next,
        ft, degree, w.steps, s.grid.poisson_scratch)

    cm = ntuple(_ -> zeros(FT, Nc, Nc, Nz+1), 6)
    P.diagnose_cs_cm!(cm, am, bm, w.dm, w.m_cur, Nc, Nz)
    return (am=am, bm=bm, cm=cm)
end

@inline function _cs_proj0!(v)
    m = sum(v) / length(v)
    @inbounds @simd for i in eachindex(v); v[i] -= m; end
    return v
end

# Collateral-degradation masks: tropics |lat|<30 and NH lat>30.
function tropics_mask(s)
    ntuple(p -> abs.(s.lats[p]) .< 30.0, 6)
end
function nh_mask(s)
    ntuple(p -> s.lats[p] .> -SH_LAT, 6)   # lat > +30
end

# ===========================================================================
# MAIN
# ===========================================================================
function run_window(s, date, win, FT)
    w = read_window(s, date, win, FT)
    dirty = dirty_closure(s, w, FT)
    damp  = damped_closure(s, w, FT)

    rv_dirty = r_vdiv(dirty.cm, s)
    rv_damp  = r_vdiv(damp.cm, s)
    c1_dirty = continuity_rms(dirty.am, dirty.bm, dirty.cm, w.dm, w.m_cur, s.Nc, s.Nz)
    c1_damp  = continuity_rms(damp.am,  damp.bm,  damp.cm,  w.dm, w.m_cur, s.Nc, s.Nz)

    # collateral checks (tropics + NH) on the damped result
    trop = tropics_mask(s); nh = nh_mask(s)
    rv_damp_trop  = r_vdiv(damp.cm, s; sh=trop)
    rv_dirty_trop = r_vdiv(dirty.cm, s; sh=trop)
    rv_damp_nh    = r_vdiv(damp.cm, s; sh=nh)
    rv_dirty_nh   = r_vdiv(dirty.cm, s; sh=nh)

    @printf("\n--- %s  win=%d ---\n", date, win)
    @printf("  SH-UTLS r_vdiv   DIRTY=%.4f   DAMPED=%.4f   (CLEAN target 0.227)\n",
            rv_dirty, rv_damp)
    @printf("  continuity RMS/colmass   DIRTY=%.2e   DAMPED=%.2e   (gate <=~1e-5)\n",
            c1_dirty, c1_damp)
    @printf("  collateral (no-degrade): tropics r_vdiv  DIRTY=%.4f DAMPED=%.4f\n",
            rv_dirty_trop, rv_damp_trop)
    @printf("                            NH      r_vdiv  DIRTY=%.4f DAMPED=%.4f\n",
            rv_dirty_nh, rv_damp_nh)
    return (date=date, win=win, rv_dirty=rv_dirty, rv_damp=rv_damp,
            c1_dirty=c1_dirty, c1_damp=c1_damp)
end

function main()
    FT = Float64
    println("=== fingerfix prototype: anisotropic-divergence-damping ===")
    println("  knobs: NU0=$NU0 ROUGH_TARGET=$ROUGH_TARGET ROUGH_SCALE=$ROUGH_SCALE ",
            "N_PASSES=$N_PASSES LAT_TAPER=$LAT_TAPER_W P_TAPER=$P_TAPER_FRAC")
    s = build_setup(FT)
    @printf("  Nc=%d Nz=%d  UTLS levels (TOA-first)=%s  p=%.0f..%.0f hPa\n",
            s.Nc, s.Nz, string(s.utls_levels),
            s.pmid[first(s.utls_levels)], s.pmid[last(s.utls_levels)])
    nsh = sum(sum, s.sh)
    @printf("  SH cells (lat<%.0f): %d of %d\n", SH_LAT, nsh, 6*s.Nc*s.Nc)

    # Primary window + validation set to guard against per-window overfitting.
    cases = [(Date(2021,12,1), 12),
             (Date(2021,12,1),  6),
             (Date(2021,12,1), 18),
             (Date(2021,12,1), 24),
             (Date(2021,12,2), 12)]

    results = NamedTuple[]
    for (d, wn) in cases
        isfile(ctm_path(d)) || (@printf("  SKIP %s (missing %s)\n", d, ctm_path(d)); continue)
        push!(results, run_window(s, d, wn, FT))
    end

    println("\n=== SUMMARY (DIRTY 0.294 / CLEAN 0.227 published baselines) ===")
    @printf("  %-12s %4s | %-10s %-10s | %-10s %-10s\n",
            "date","win","rvDIRTY","rvDAMP","c1DIRTY","c1DAMP")
    for r in results
        @printf("  %-12s %4d | %-10.4f %-10.4f | %-10.2e %-10.2e\n",
                string(r.date), r.win, r.rv_dirty, r.rv_damp, r.c1_dirty, r.c1_damp)
    end
    if !isempty(results)
        md_dirty = mean(r.rv_dirty for r in results)
        md_damp  = mean(r.rv_damp  for r in results)
        maxc1    = maximum(r.c1_damp for r in results)
        @printf("\n  MEAN r_vdiv  DIRTY=%.4f  DAMPED=%.4f  (improvement %.1f%%)\n",
                md_dirty, md_damp, 100*(md_dirty-md_damp)/md_dirty)
        @printf("  MAX continuity RMS/colmass (damped) = %.2e %s\n",
                maxc1, maxc1 <= 2e-5 ? "[gate OK]" : "[GATE FAIL]")
    end
    return results
end

main()
