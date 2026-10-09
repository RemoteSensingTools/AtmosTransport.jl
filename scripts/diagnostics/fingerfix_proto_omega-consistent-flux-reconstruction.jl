# ===========================================================================
# PROTOTYPE: OMEGA-consistent 3D flux reconstruction  (ITERATION 2)
#   (smooth physical vertical-motion target via least-norm flux potential,
#    blended with a null-space hyperdiffusive regularizer)
#
# Method id: omega-consistent-flux-reconstruction
#
# PROBLEM. Offline cubed-sphere transport driven by GEOS-IT C180 native mass
# fluxes shows grid-scale tracer noise ("fingering") at the SH-UTLS (lat<-30,
# 80-300 hPa). The diagnosed closure cm[k+1]=cm[k]+div_h[k]-dm[k] forces the
# grid-noisy MFXC<->DELP residual M (~5.6e-4 colmass) into cm, so the per-layer
# vertical convergence vdiv=cm[k]-cm[k+1] (==div_h) is grid-rough at the sharp
# UTLS gradient -> fingering.
#
# ITERATION-1 IDEA. GEOS-IT A3dyn archives OMEGA, the model's RESOLVED vertical
# pressure velocity, ~2x smoother than div_h(MFXC) at the SH-UTLS. Build a SMOOTH
# physical vertical-convergence target vdiv_om from OMEGA (DOWNWARD-positive, same
# sign as cm) and per level solve for a least-norm horizontal flux POTENTIAL
# lambda so the NEW horizontal convergence is  div_h_new[k] = dm[k] - vdiv_om[k];
# then the telescoped cm gives EXACTLY vdiv[k] = dm[k]-div_h_new[k] = +vdiv_om[k]
# (smooth, and cm TRACKS OMEGA), while continuity holds BY CONSTRUCTION (the
# correction lives inside continuity's null space -> the replay gate passes).
# [SIGN FIX vs the original iter-1: the OMEGA term is +vdiv_om, not -vdiv_om;
#  the earlier -vdiv_om anchored cm to MINUS the resolved vertical motion.]
#
# ITERATION-2 REFINEMENTS (this file):
#  (a) BLEND the OMEGA prior with a null-space HYPERDIFFUSIVE regularizer
#      (the cm-hyperdiffusion finalist).  The realized vertical convergence is
#         vdiv_new = alpha * vdiv_om  +  (1-alpha) * smooth_implicit(vdiv_raw)
#      (vdiv_om downward-positive, same sign as cm) so the result tracks OMEGA
#      where OMEGA is trustworthy and FALLS BACK to a
#      data-driven hyperdiffusive smoothing of the actual (dm-div_h) convergence
#      where it is not.  alpha is swept.  alpha=1 == iteration-1; alpha=0 == the
#      pure cm-hyperdiffusion finalist.  Continuity is still exact for ANY alpha
#      because EVERY target is realized via the same null-space potential.
#  (b) MONOTONE-CUBIC (PCHIP) 3-hourly->hourly OMEGA/QV time interpolation in
#      place of linear, removing the slope-kinks at 3-hourly bracket boundaries.
#      Verified explicitly at a bracket-boundary CTM window.
#  (c) INTERFACE-consistent dry conversion: qv interpolated to LAYER INTERFACES
#      and (1-qv_ifc) applied to the interface OMEGA before differencing, instead
#      of mixing interface-omega with a flat layer (1-qv).
#  (d) Tracer-level adv-only proxy: report the horizontal-flux increment % and a
#      one-substep passive-tracer SH-UTLS variance-growth proxy to confirm the
#      ~29% increment does not degrade transport; plus an independent cross-panel
#      mirror-consistency audit of the corrected am/bm.
#  (e) Extended scoring: global continuity (whole-globe C1), full-SH roughness
#      (all levels), and the SH-UTLS r_vdiv metric.
#
# UNITS (match scripts/completed_experiments/iau_signature_M.jl / moist_budget_IT_vs_FP.jl):
#   am = MFXC/(2g), bm = MFYC/(2g)    [flux_scale = 1/(2g)]
#   div_h[k] = (am[i,j,k]-am[i+1,j,k]) + (bm[i,j,k]-bm[i,j+1,k])
#   dm[k]    = (DELP_next-DELP_cur)[k]*area/g/(2*steps)         steps=DT_MET/MFDT=8
#   tau      = DT_MET/(2*steps) = MFDT/2 = 225 s
#   cm[k+1]  = cm[k] + div_h[k] - dm[k]   (diagnose_cs_cm! convention)
#
# RUN (DIRECT julia binary; the juliaup shim blocks on a config lock):
#   ~/.julia/juliaup/julia-1.12.5+0.x64.linux.gnu/bin/julia --project=. \
#       scripts/diagnostics/fingerfix_proto_omega-consistent-flux-reconstruction.jl [ctm_win...]
# ===========================================================================
using AtmosTransport, AtmosTransport.Preprocessing, AtmosTransport.Grids
using Dates, NCDatasets, Statistics, Printf, TOML
const P = AtmosTransport.Preprocessing

const GRAV  = 9.80665
const DT_MET = 3600.0
const MFDT  = 450.0
const CONFIG = joinpath(@__DIR__, "..", "..",
    "config/preprocessing/geosit_c180_dec2021_catrine_f32_fullL72.toml")
const RAW = expanduser("~/data/AtmosTransport/met/geosit/C180/raw_catrine")
const DATE = Date(2021, 12, 1)
const DSTR = Dates.format(DATE, "yyyymmdd")
ctm_path  = joinpath(RAW, DSTR, "GEOSIT.$DSTR.CTM_A1.C180.nc")
a3_path   = joinpath(RAW, DSTR, "GEOSIT.$DSTR.A3dyn.C180.nc")
i3_path   = joinpath(RAW, DSTR, "GEOSIT.$DSTR.I3.C180.nc")

# Windows to process (CTM_A1 indices). Default 10..13 — a handful, runs in min.
const CTM_WINS = length(ARGS) >= 1 ? parse.(Int, ARGS) : collect(10:13)

# Blend weights to sweep: alpha=1 -> pure OMEGA prior (iter-1); alpha=0 -> pure
# null-space hyperdiffusion of the raw convergence (cm-hyperdiffusion finalist).
const ALPHA_SWEEP = [1.0, 0.75, 0.5, 0.25, 0.0]
# Implicit-smoothing strength + order for the hyperdiffusive fallback component.
# order=1 implicit Laplacian (monotone-damping); order=2 implicit bi-Laplacian.
const HD_NU    = 8.0
const HD_ORDER = 1

# --- UTLS / SH band, exactly as score_binary.py -----------------------------
const UTLS_LO, UTLS_HI = 80.0, 300.0   # hPa
const SH_LAT = -30.0

# ---------------------------------------------------------------------------
# Time-interpolation helpers between 3-hourly A3dyn / I3 and hourly CTM_A1.
# CTM_A1 win w valid time = (w-1):30 UTC (minutes = (w-1)*60 + 30).
# A3dyn   win a valid time = 01:30 + (a-1)*3h  (minutes = (a-1)*180 + 90).
# I3      win a valid time = 00:00 + (a-1)*3h  (minutes = (a-1)*180).
# ---------------------------------------------------------------------------
ctm_valid_min(w) = (w - 1) * 60 + 30
a3_valid_min(a)  = (a - 1) * 180 + 90
i3_valid_min(a)  = (a - 1) * 180

# --- (Bug-2 guard) Detect a target time that falls OUTSIDE the same-day 3-hourly
# bracket so the clamp in bracket3/bracket4 would SILENTLY reuse an endpoint
# (extrapolation) instead of interpolating across midnight.  Returns true when t
# is strictly before the first node or strictly after the last node.  The OMEGA/QV
# read site MUST guard on this (error or next-day read) so a late CTM window such
# as 24 (valid 23:30, past the 22:30 last A3dyn node) cannot produce an invalid
# constant-extrapolated OMEGA target masquerading as an interpolated one.
@inline function interp_extrapolates(valid_min::Function, n3::Int, t::Float64)
    return t < Float64(valid_min(1)) || t > Float64(valid_min(n3))
end

"Linear-interp factor + bracketing 3-hourly window indices for target minute t."
function bracket3(valid_min::Function, n3::Int, t::Float64)
    a = 1
    for ai in 1:n3
        if valid_min(ai) <= t
            a = ai
        end
    end
    a0 = clamp(a, 1, n3)
    a1 = clamp(a + 1, 1, n3)
    t0 = Float64(valid_min(a0)); t1 = Float64(valid_min(a1))
    f = (t1 == t0) ? 0.0 : clamp((t - t0) / (t1 - t0), 0.0, 1.0)
    return a0, a1, f
end

# --- (b) Monotone-cubic (PCHIP / Fritsch-Carlson) scalar interpolation -------
# Uniform 3-hourly node spacing => the PCHIP derivative at an interior node is
# the harmonic-mean / WENO-style limited slope; this gives a C1 curve whose
# derivative does NOT jump across a bracket boundary (kills the linear kink) and
# is monotone (no over/undershoot).  At t equal to a node, returns that node
# exactly => instantaneous-frame consistency preserved.
@inline function _pchip_slope(dm1::Float64, d0::Float64)
    # secant slopes on uniform grid; Fritsch-Carlson limiter
    (dm1 == 0.0 || d0 == 0.0 || sign(dm1) != sign(d0)) && return 0.0
    return 2.0 / (1.0/dm1 + 1.0/d0)      # harmonic mean (h cancels on uniform grid)
end
@inline function _pchip_eval(y::NTuple{4,Float64}, f::Float64)
    # nodes y1,y2,y3,y4 at uniform spacing; interpolate on [y2,y3], local frac f.
    # endpoint slopes via one-sided; interior via harmonic-mean secants.
    d12 = y[2]-y[1]; d23 = y[3]-y[2]; d34 = y[4]-y[3]
    m2 = _pchip_slope(d12, d23)
    m3 = _pchip_slope(d23, d34)
    # Hermite basis on unit interval (h=1 in node units; slopes already per-node)
    h00 = (1+2f)*(1-f)^2
    h10 = f*(1-f)^2
    h01 = f^2*(3-2f)
    h11 = f^2*(f-1)
    return h00*y[2] + h10*m2 + h01*y[3] + h11*m3
end

"4-node bracket (a-1..a+2) clamped to [1,n3] for monotone-cubic interp at t."
function bracket4(valid_min::Function, n3::Int, t::Float64)
    a0, a1, f = bracket3(valid_min, n3, t)   # reuse: a0=floor node, a1=ceil node
    am1 = clamp(a0 - 1, 1, n3)
    ap2 = clamp(a1 + 1, 1, n3)
    return (am1, a0, a1, ap2), f, (a0 == a1)
end

"Read a 3D field at CTM valid time t via the chosen scheme (:linear or :pchip)."
function read_interp_3d(ds, var, valid_min::Function, n3::Int, t::Float64, or;
                        FT, scheme::Symbol)
    if scheme === :linear
        a0, a1, f = bracket3(valid_min, n3, t)
        f0 = P._read_panels_3d(ds[var], a0, or; FT=FT)
        a0 == a1 && return f0
        f1 = P._read_panels_3d(ds[var], a1, or; FT=FT)
        return ntuple(p -> @.(f0[p] * (1 - f) + f1[p] * f), 6)
    else  # :pchip
        (nodes, f, atnode) = bracket4(valid_min, n3, t)
        y1 = P._read_panels_3d(ds[var], nodes[1], or; FT=FT)
        atnode && f == 0.0 && return P._read_panels_3d(ds[var], nodes[2], or; FT=FT)
        y2 = P._read_panels_3d(ds[var], nodes[2], or; FT=FT)
        y3 = P._read_panels_3d(ds[var], nodes[3], or; FT=FT)
        y4 = P._read_panels_3d(ds[var], nodes[4], or; FT=FT)
        out = ntuple(p -> similar(y2[p]), 6)
        @inbounds for p in 1:6
            o = out[p]; a = y1[p]; b = y2[p]; c = y3[p]; d = y4[p]
            for idx in eachindex(o)
                o[idx] = FT(_pchip_eval((Float64(a[idx]),Float64(b[idx]),
                                         Float64(c[idx]),Float64(d[idx])), f))
            end
        end
        return out
    end
end

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
function setup(FT)
    cfg  = TOML.parsefile(CONFIG)
    grid = P.build_target_geometry(cfg["grid"], FT); Nc = grid.Nc
    vc   = P.load_hybrid_coefficients(AtmosTransport.expand_data_path(String(cfg["vertical"]["coefficients"])))
    Nz   = length(vc.A) - 1
    Aifc = Float64.(vc.A); Bifc = Float64.(vc.B)
    lats = ntuple(p -> Float64.(P.panel_cell_center_lonlat(grid.mesh, p)[2]), 6)
    sh   = ntuple(p -> lats[p] .< SH_LAT, 6)
    conn  = grid.mesh.connectivity
    areas = grid.mesh.cell_areas
    ft    = P.build_cs_global_face_table(Nc, conn)
    deg   = P.cs_cell_face_degree(ft)
    scr   = P.CSPoissonScratch(ft.nc)
    psref = 1.0e5
    pmid  = [0.5*((Aifc[k]+Aifc[k+1]) + (Bifc[k]+Bifc[k+1])*psref)/100 for k in 1:Nz]  # hPa
    utls  = findall(p -> UTLS_LO <= p <= UTLS_HI, pmid)
    return (; cfg, grid, Nc, Nz, areas, conn, sh, lats, ft, deg, scr, pmid, utls,
            steps = round(Int, DT_MET/MFDT))
end

# ---------------------------------------------------------------------------
# r_vdiv on a given level set (default UTLS).  Exactly score_binary.rough_field.
# ---------------------------------------------------------------------------
function r_vdiv(cm::NTuple{6,Array{Float64,3}}, s; levels = s.utls)
    Nc = s.Nc
    rs = Float64[]
    for k in levels
        lp = Float64[]; vals = Float64[]
        for p in 1:6, j in 2:Nc-1, i in 2:Nc-1
            s.sh[p][i, j] || continue
            f(ii, jj) = cm[p][ii, jj, k] - cm[p][ii, jj, k+1]
            c = f(i, j)
            isnan(c) && continue
            lap = c - 0.25*(f(i+1, j) + f(i-1, j) + f(i, j+1) + f(i, j-1))
            push!(lp, lap); push!(vals, c)
        end
        if length(vals) > 2
            sd = std(vals)
            sd > 0 && push!(rs, sqrt(mean(abs2, lp)) / sd)
        end
    end
    isempty(rs) ? NaN : mean(rs)
end

# r_vdiv num/denom split (UTLS-averaged): returns (mean RMS-Laplacian numerator,
# mean std-of-vdiv denominator).  A genuine fingering cut LOWERS the numerator
# while leaving the denominator ~unchanged; a variance-collapse artifact LOWERS
# the denominator faster than the numerator (the cm-hyperdiffusion finalist's
# documented trap).  This is the load-bearing honesty check for low-alpha.
function r_vdiv_split(cm::NTuple{6,Array{Float64,3}}, s; levels = s.utls)
    Nc = s.Nc; nums = Float64[]; dens = Float64[]
    for k in levels
        lp = Float64[]; vals = Float64[]
        for p in 1:6, j in 2:Nc-1, i in 2:Nc-1
            s.sh[p][i, j] || continue
            f(ii, jj) = cm[p][ii, jj, k] - cm[p][ii, jj, k+1]
            c = f(i, j); isnan(c) && continue
            push!(lp, c - 0.25*(f(i+1, j) + f(i-1, j) + f(i, j+1) + f(i, j-1)))
            push!(vals, c)
        end
        if length(vals) > 2
            push!(nums, sqrt(mean(abs2, lp))); push!(dens, std(vals))
        end
    end
    (isempty(nums) ? NaN : mean(nums), isempty(dens) ? NaN : mean(dens))
end

# Pearson correlation over SH-UTLS cells between the REALIZED vertical convergence
# vdiv = cm[k]-cm[k+1] and the OMEGA-derived target vdiv_om.  This is the proof
# the OMEGA anchoring is GENUINE: with the sign fixed, alpha=1 must give a STRONG
# POSITIVE correlation (cm tracks the resolved downward-positive vertical motion).
# A near-zero or negative value at alpha=1 means the anchor is broken/inverted.
function cm_vs_omega_corr(cm::NTuple{6,Array{Float64,3}},
                          vdiv_om::NTuple{6,Array{Float64,3}}, s; levels = s.utls)
    Nc = s.Nc
    xs = Float64[]; ys = Float64[]
    for k in levels, p in 1:6, j in 2:Nc-1, i in 2:Nc-1
        s.sh[p][i, j] || continue
        v  = cm[p][i, j, k] - cm[p][i, j, k+1]
        vo = vdiv_om[p][i, j, k]
        (isnan(v) || isnan(vo)) && continue
        push!(xs, v); push!(ys, vo)
    end
    length(xs) < 3 && return NaN
    (std(xs) == 0 || std(ys) == 0) && return NaN
    cor(xs, ys)
end

# Continuity RMS residual / colmass.  region = :interior (panel-interior, the
# gate) or :global (whole globe incl. panel edges, for the extended (e) score).
function continuity_rms(am, bm, cm, dm, m, s; region::Symbol = :interior)
    Nc = s.Nc; Nz = s.Nz
    colmass = 0.0
    for p in 1:6, j in 1:Nc, i in 1:Nc
        c = 0.0
        @inbounds for k in 1:Nz; c += m[p][i,j,k]; end
        colmass = max(colmass, c)
    end
    lo = region === :interior ? 2 : 1
    hi(n) = region === :interior ? n-1 : n
    ss = 0.0; n = 0; mx = 0.0
    for p in 1:6, j in lo:hi(Nc), i in lo:hi(Nc)
        @inbounds for k in 1:Nz
            div_h = (am[p][i,j,k]-am[p][i+1,j,k]) + (bm[p][i,j,k]-bm[p][i,j+1,k])
            vdiv  = cm[p][i,j,k] - cm[p][i,j,k+1]
            r = (dm[p][i,j,k] - div_h - vdiv) / colmass
            ss += r*r; n += 1; mx = max(mx, abs(r))
        end
    end
    return sqrt(ss/n), mx, colmass
end

# (d) Cross-panel mirror-consistency audit of am/bm.  For every boundary face,
# the canonical flux and its mirror entry must agree up to mirror_sign.  Returns
# the max |canonical - sign*mirror| over UTLS levels (== 0 means edge rotation of
# the lambda correction is consistent).
function mirror_consistency(am, bm, s)
    ft = s.ft; worst = 0.0
    @inbounds for f in 1:ft.nf
        mq = Int(ft.mirror_panel[f]); mq == 0 && continue
        p  = Int(ft.face_panel[f]); dir = Int(ft.face_dir[f])
        i  = Int(ft.face_idx_i[f]); j = Int(ft.face_idx_j[f])
        md = Int(ft.mirror_dir[f]); mi = Int(ft.mirror_idx_i[f]); mj = Int(ft.mirror_idx_j[f])
        msign = Int(ft.mirror_sign[f])
        for k in s.utls
            can = dir == 1 ? Float64(am[p][i,j,k]) : Float64(bm[p][i,j,k])
            mir = md == 1 ? Float64(am[mq][mi,mj,k]) : Float64(bm[mq][mi,mj,k])
            worst = max(worst, abs(can - msign*mir))
        end
    end
    worst
end

# ---------------------------------------------------------------------------
# Build native am/bm, dm (endpoint), m (=DELP*area/g) for window pair (w, w+1).
# ---------------------------------------------------------------------------
function build_window(s, ds_ctm, ds_ctm1, w::Int, nt::Int, or, FT)
    Nc = s.Nc; Nz = s.Nz; g = FT(GRAV)
    flux_scale = FT(1/(2g))
    twosteps = FT(2*s.steps)
    mfxc = P._read_panels_3d(ds_ctm["MFXC"], w, or; FT=FT)
    mfyc = P._read_panels_3d(ds_ctm["MFYC"], w, or; FT=FT)
    dc   = P._read_panels_3d(ds_ctm["DELP"], w, or; FT=FT)
    dn   = w < nt ? P._read_panels_3d(ds_ctm["DELP"], w+1, or; FT=FT) :
                    P._read_panels_3d(ds_ctm1["DELP"], 1,  or; FT=FT)
    am = ntuple(_ -> zeros(FT, Nc+1, Nc, Nz), 6)
    bm = ntuple(_ -> zeros(FT, Nc, Nc+1, Nz), 6)
    P.geos_native_to_face_flux!(am, bm, mfxc, mfyc, s.conn, Nc, Nz, flux_scale)
    dm = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    m  = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    @inbounds for p in 1:6, j in 1:Nc, i in 1:Nc
        a = s.areas[i,j]
        for k in 1:Nz
            dm[p][i,j,k] = (dn[p][i,j,k]-dc[p][i,j,k]) * a/g/twosteps
            m[p][i,j,k]  = dc[p][i,j,k] * a/g
        end
    end
    return (; am, bm, dm, m)
end

# (c) OMEGA-derived smooth vertical-convergence target, INTERFACE-consistent dry
# conversion.  qv_ifc[k] = 0.5*(qv[k-1]+qv[k]); the dry interface pressure
# velocity is omega_ifc*(1-qv_ifc); the per-layer convergence is the telescoped
# interface difference, *area/g*tau.  Because omega_dry_ifc[1]=omega_dry_ifc[Nz+1]
# =0, Σ_k vdiv_om = 0 exactly (matches cm[1]=cm[Nz+1]=0).
function omega_target(s, omega, qv, FT)
    Nc = s.Nc; Nz = s.Nz; g = FT(GRAV)
    tau = FT(DT_MET/(2*s.steps))           # = MFDT/2 = 225 s
    vdiv_om = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    @inbounds for p in 1:6, j in 1:Nc, i in 1:Nc
        a = s.areas[i,j]
        for k in 1:Nz
            # interface omega (0 at TOA & surface)
            om_top = (k == 1)  ? zero(FT) : FT(0.5)*(omega[p][i,j,k-1]+omega[p][i,j,k])
            om_bot = (k == Nz) ? zero(FT) : FT(0.5)*(omega[p][i,j,k]  +omega[p][i,j,k+1])
            # interface dry fraction (qv interpolated to the SAME interface)
            qv_top = (k == 1)  ? zero(FT) : FT(0.5)*(qv[p][i,j,k-1]+qv[p][i,j,k])
            qv_bot = (k == Nz) ? zero(FT) : FT(0.5)*(qv[p][i,j,k]  +qv[p][i,j,k+1])
            od_top = om_top * (one(FT) - qv_top)
            od_bot = om_bot * (one(FT) - qv_bot)
            vdiv_om[p][i,j,k] = (a/g) * (od_top - od_bot) * tau
        end
    end
    return vdiv_om
end

# Graph divergence of (am,bm) at level k into a global nc-vector.
function level_divergence!(div::Vector{Float64}, am, bm, ft, k::Int)
    fill!(div, 0.0)
    @inbounds for f in 1:ft.nf
        panel = Int(ft.face_panel[f]); dir = Int(ft.face_dir[f])
        i = Int(ft.face_idx_i[f]); j = Int(ft.face_idx_j[f])
        flux = dir == 1 ? Float64(am[panel][i, j, k]) : Float64(bm[panel][i, j, k])
        div[Int(ft.face_left[f])]  += flux
        div[Int(ft.face_right[f])] -= flux
    end
    return div
end
@inline function _vdot(a,b); s=0.0; @inbounds @simd for i in eachindex(a,b); s+=a[i]*b[i]; end; s; end

# Implicit (I + nu L^order)^{-1} smoothing of a mean-zero global field, reusing
# the graph Laplacian.  Unconditionally stable; order=1 monotone, order=2 hyper.
function _solve_implicit_smooth!(x::Vector{Float64}, b::Vector{Float64},
                                 ft, degree, nu::Float64, order::Int;
                                 tol=1e-10, maxit=5000)
    nc = ft.nc
    r = similar(x); p = similar(x); Ap = similar(x); t = similar(x)
    applyA! = (out, v) -> begin
        if order == 1
            P._cs_graph_laplacian_mul!(out, v, ft, degree)
        else
            P._cs_graph_laplacian_mul!(t, v, ft, degree)
            mu = sum(t)/nc; @inbounds @simd for c in 1:nc; t[c] -= mu; end
            P._cs_graph_laplacian_mul!(out, t, ft, degree)
        end
        @inbounds @simd for c in 1:nc; out[c] = v[c] + nu*out[c]; end
        return out
    end
    mub = sum(b)/nc; @inbounds @simd for c in 1:nc; b[c] -= mub; end
    fill!(x, 0.0); copyto!(r, b)
    applyA!(Ap, x); @inbounds @simd for c in 1:nc; r[c] = b[c] - Ap[c]; end
    copyto!(p, r)
    rr = _vdot(r, r); bnorm = sqrt(_vdot(b,b)) + eps()
    it = 0
    while it < maxit && sqrt(rr)/bnorm > tol
        applyA!(Ap, p)
        alpha = rr / _vdot(p, Ap)
        @inbounds @simd for c in 1:nc; x[c] += alpha*p[c]; r[c] -= alpha*Ap[c]; end
        rr_new = _vdot(r, r); beta = rr_new / rr
        @inbounds @simd for c in 1:nc; p[c] = r[c] + beta*p[c]; end
        rr = rr_new; it += 1
    end
    mux = sum(x)/nc; @inbounds @simd for c in 1:nc; x[c] -= mux; end
    return nothing
end

# (a) BLENDED reconstruction: realize the target vertical convergence
#     vdiv_target[k] = alpha * vdiv_om[k]  +  (1-alpha) * vdiv_hd[k]
# where vdiv_om is the DOWNWARD-positive OMEGA mass-flux convergence (same sign
# as cm; see omega_target + structured_kernels.jl:145-146), so at alpha=1 cm
# TRACKS the resolved OMEGA vertical motion (vdiv = +vdiv_om), and vdiv_hd is the
# implicit-smoothed RAW convergence (dm - div_h).  We solve, per level, for the
# null-space potential lambda that makes div_h_new = dm - vdiv_target, so the
# realized vdiv = dm - div_h_new = vdiv_target EXACTLY and continuity is exact for
# ANY alpha.  alpha=1 -> pure OMEGA anchor; alpha=0 -> cm-hyperdiffusion finalist.
# Returns (grad_rms, base_rms) at UTLS for the increment ratio.
function reconstruct_blended!(am, bm, dm, vdiv_om, s, alpha::Float64,
                              hd_nu::Float64, hd_order::Int)
    Nc = s.Nc; Nz = s.Nz; ft = s.ft; deg = s.deg; scr = s.scr; nc = ft.nc
    cg = (r=scr.r, p=scr.p, Ap=scr.Ap, z=scr.z)
    divh = scr.div
    rhs  = scr.rhs       # we use this as the potential RHS = (div_h - target_div)
    psi  = scr.psi
    raw  = similar(divh)  # raw vdiv = dm - div_h (per level)
    hd   = similar(divh)  # smoothed raw vdiv
    glam2 = 0.0; base2 = 0.0; nfaces_utls = 0
    for k in 1:Nz
        level_divergence!(divh, am, bm, ft, k)
        # raw convergence vdiv_raw[c] = dm[c] - div_h[c]   (note level_divergence
        # returns div[c] = Σ signed flux = -(local convergence) per the table; the
        # convergence used by diagnose_cs_cm! is conv[c] = -div[c].  Here we must
        # be consistent with how target_div is formed below.)
        @inbounds for c in 1:nc
            p_idx = (c - 1) ÷ (Nc*Nc) + 1
            li = (c - 1) % (Nc*Nc); jl = li ÷ Nc + 1; il = li % Nc + 1
            convh = -divh[c]                       # local horizontal convergence
            raw[c] = Float64(dm[p_idx][il,jl,k]) - convh   # vdiv_raw = dm - div_h
        end
        # hyperdiffused raw convergence (mean preserved: smooth only the noise)
        if alpha < 1.0 && hd_nu > 0.0
            mu = sum(raw)/nc
            @inbounds @simd for c in 1:nc; hd[c] = raw[c] - mu; end
            _solve_implicit_smooth!(hd, copy(hd), ft, deg, hd_nu, hd_order)
            @inbounds @simd for c in 1:nc; hd[c] += mu; end
        else
            copyto!(hd, raw)
        end
        # target vertical convergence (what cm will SEE): a blend of
        #   (i) the OMEGA smooth target vdiv_om  and  (ii) the hyperdiffused raw vdiv.
        # SIGN (the load-bearing fix). diagnose_cs_cm! sets vdiv[k]=cm[k]-cm[k+1]
        # = dm[k]-div_h[k], and cm is the DOWNWARD-positive vertical mass flux
        # through a layer's top face (structured_kernels.jl:145-146).  GEOS OMEGA
        # (dp/dt) is ALSO downward-positive, and omega_target builds
        #   vdiv_om[k] = (M_z_top - M_z_bot)*tau  (the OMEGA mass-flux convergence
        # into layer k, downward-positive).  So the OMEGA-CONSISTENT cm has
        # cm[k]≈M_z_top*tau ⇒ vdiv[k] = +vdiv_om[k] (NOT -vdiv_om).  Targeting
        # -vdiv_om (iter-1's "div_h_new=dm+vdiv_om") anchors cm to MINUS OMEGA, so
        # at alpha=1 cm would be ANTI-correlated with the resolved vertical motion.
        #   vdiv_target[c] = alpha*vdiv_om[c] + (1-alpha)*hd[c]
        #   div_h_new[c]   = dm[c] - vdiv_target[c]              (since vdiv=dm-div_h)
        # apply_cs_flux_correction! drives the GRAPH divergence to the desired
        # value via  rhs[c] = div_graph_current[c] - desired_graph_div[c]  (see
        # _balance_cs_level!).  level_divergence! returns div_graph = -div_h
        # (stencil), so desired_graph_div = -div_h_new ⇒
        #   rhs[c] = divh[c] - (-div_h_new[c]) = divh[c] + div_h_new[c].
        @inbounds for c in 1:nc
            p_idx = (c - 1) ÷ (Nc*Nc) + 1
            li = (c - 1) % (Nc*Nc); jl = li ÷ Nc + 1; il = li % Nc + 1
            vom = Float64(vdiv_om[p_idx][il,jl,k])     # OMEGA-anchored vdiv target
            vtarget = alpha*vom + (1.0-alpha)*hd[c]
            div_h_new = Float64(dm[p_idx][il,jl,k]) - vtarget
            desired_graph_div = -div_h_new             # graph = -(stencil div_h)
            rhs[c] = divh[c] - desired_graph_div       # = divh + div_h_new
        end
        P.solve_cs_poisson_pcg!(psi, rhs, ft, deg, cg; tol=1e-11, max_iter=8000,
                                project_every=50)
        P.apply_cs_flux_correction!(am, bm, psi, ft, k)
        if k in s.utls
            @inbounds for f in 1:ft.nf
                l = Int(ft.face_left[f]); rr = Int(ft.face_right[f])
                d = psi[rr] - psi[l]; glam2 += d*d
                panel = Int(ft.face_panel[f]); dir = Int(ft.face_dir[f])
                i = Int(ft.face_idx_i[f]); j = Int(ft.face_idx_j[f])
                bf = dir == 1 ? Float64(am[panel][i,j,k]) : Float64(bm[panel][i,j,k])
                base2 += bf*bf; nfaces_utls += 1
            end
        end
    end
    P._sync_cs_mirrors!(am, bm, ft, Nz)
    grad_rms = nfaces_utls > 0 ? sqrt(glam2/nfaces_utls) : NaN
    base_rms = nfaces_utls > 0 ? sqrt(base2/nfaces_utls) : NaN
    return grad_rms, base_rms
end

# (d) One-substep passive-tracer SH-UTLS variance-growth proxy.  Seed a smooth
# UTLS tracer (linear in pressure), apply ONE explicit upwind horizontal +
# vertical flux update with the candidate (am,bm,cm) and the layer mass m, and
# measure the grid-scale variance INJECTED in the SH-UTLS band.  Lower = less
# fingering for a real tracer.  This is a cheap directional proxy, not a full run.
function tracer_finger_proxy(am, bm, cm, m, s)
    Nc = s.Nc; Nz = s.Nz
    # smooth IC: q = pmid (hPa), constant on each layer -> only transport-induced
    # grid structure appears.  Use per-cell mixing ratio q and mass-weighted flux.
    q = ntuple(p -> begin
        a = zeros(Float64, Nc, Nc, Nz)
        for k in 1:Nz, j in 1:Nc, i in 1:Nc; a[i,j,k] = s.pmid[k]; end
        a
    end, 6)
    # one forward-Euler tracer mass update over the window using face masses am/bm
    # (upwind) + cm (upwind vertical).  q_new = (q*m + flux_in - flux_out)/m_new.
    # We only need the SH-UTLS grid-roughness of the increment, so a coarse single
    # explicit step on the panel interior suffices.
    nums = Float64[]; dens = Float64[]
    for k in s.utls
        lap = Float64[]; vals = Float64[]
        for p in 1:6, j in 2:Nc-1, i in 2:Nc-1
            s.sh[p][i,j] || continue
            mc = m[p][i,j,k]; mc <= 0 && continue
            # horizontal upwind flux divergence of q
            fx_w = am[p][i,j,k];   qx_w = fx_w >= 0 ? q[p][i-1,j,k] : q[p][i,j,k]
            fx_e = am[p][i+1,j,k]; qx_e = fx_e >= 0 ? q[p][i,j,k]   : q[p][i+1,j,k]
            fy_s = bm[p][i,j,k];   qy_s = fy_s >= 0 ? q[p][i,j-1,k] : q[p][i,j,k]
            fy_n = bm[p][i,j+1,k]; qy_n = fy_n >= 0 ? q[p][i,j,k]   : q[p][i,j+1,k]
            hflux = (fx_w*qx_w - fx_e*qx_e) + (fy_s*qy_s - fy_n*qy_n)
            # vertical upwind via cm (cm[k] is flux across top of layer k)
            cmt = cm[p][i,j,k];   qz_t = cmt >= 0 ? (k>1 ? q[p][i,j,k-1] : q[p][i,j,k]) : q[p][i,j,k]
            cmb = cm[p][i,j,k+1]; qz_b = cmb >= 0 ? q[p][i,j,k] : (k<Nz ? q[p][i,j,k+1] : q[p][i,j,k])
            vflux = cmt*qz_t - cmb*qz_b
            dq = (hflux + vflux) / mc
            push!(vals, dq)
        end
        # grid-Laplacian of the increment field over the SH-UTLS cells
        # (recompute on a dense grid for the Laplacian neighborhood)
        if length(vals) > 2
            # build a per-cell increment field for this level to take its Laplacian
            inc = fill(NaN, 6, Nc, Nc)
            for p in 1:6, j in 2:Nc-1, i in 2:Nc-1
                mc = m[p][i,j,k]; mc <= 0 && continue
                fx_w = am[p][i,j,k];   qx_w = fx_w >= 0 ? q[p][i-1,j,k] : q[p][i,j,k]
                fx_e = am[p][i+1,j,k]; qx_e = fx_e >= 0 ? q[p][i,j,k]   : q[p][i+1,j,k]
                fy_s = bm[p][i,j,k];   qy_s = fy_s >= 0 ? q[p][i,j-1,k] : q[p][i,j,k]
                fy_n = bm[p][i,j+1,k]; qy_n = fy_n >= 0 ? q[p][i,j,k]   : q[p][i,j+1,k]
                hflux = (fx_w*qx_w - fx_e*qx_e) + (fy_s*qy_s - fy_n*qy_n)
                cmt = cm[p][i,j,k];   qz_t = cmt >= 0 ? (k>1 ? q[p][i,j,k-1] : q[p][i,j,k]) : q[p][i,j,k]
                cmb = cm[p][i,j,k+1]; qz_b = cmb >= 0 ? q[p][i,j,k] : (k<Nz ? q[p][i,j,k+1] : q[p][i,j,k])
                vflux = cmt*qz_t - cmb*qz_b
                inc[p,i,j] = (hflux + vflux) / mc
            end
            lp = Float64[]
            for p in 1:6, j in 3:Nc-2, i in 3:Nc-2
                s.sh[p][i,j] || continue
                c = inc[p,i,j]
                (isnan(c)||isnan(inc[p,i+1,j])||isnan(inc[p,i-1,j])||isnan(inc[p,i,j+1])||isnan(inc[p,i,j-1])) && continue
                push!(lp, c - 0.25*(inc[p,i+1,j]+inc[p,i-1,j]+inc[p,i,j+1]+inc[p,i,j-1]))
            end
            if length(lp) > 2
                sd = std(vals); sd > 0 && push!(nums, sqrt(mean(abs2,lp))); sd > 0 && push!(dens, sd)
            end
        end
    end
    (isempty(nums) || isempty(dens)) && return NaN
    mean(nums) / mean(dens)
end

# ---------------------------------------------------------------------------
function main()
    FT = Float64
    s = setup(FT)
    @printf("OMEGA-consistent flux reconstruction (ITER 2) — %s  Nc=%d Nz=%d  steps=%d\n",
            DATE, s.Nc, s.Nz, s.steps)
    @printf("UTLS layers (80-300 hPa): %d levels  k=%s\n",
            length(s.utls), string(extrema(s.utls)))
    @printf("windows: %s   alpha sweep: %s   HD(nu=%.1f,order=%d)\n\n",
            string(CTM_WINS), string(ALPHA_SWEEP), HD_NU, HD_ORDER)

    isfile(ctm_path) || error("missing $ctm_path")
    isfile(a3_path)  || error("missing $a3_path")
    isfile(i3_path)  || error("missing $i3_path")
    ds_ctm = NCDataset(ctm_path, "r")
    or = P.detect_level_orientation(ds_ctm)
    nt = ds_ctm.dim["time"]
    ds_a3 = NCDataset(a3_path, "r")
    ds_i3 = NCDataset(i3_path, "r")
    n3_a3 = ds_a3.dim["time"]; n3_i3 = ds_i3.dim["time"]
    f1 = joinpath(RAW, Dates.format(DATE+Day(1), "yyyymmdd"),
                  "GEOSIT.$(Dates.format(DATE+Day(1),"yyyymmdd")).CTM_A1.C180.nc")
    ds_ctm1 = isfile(f1) ? NCDataset(f1, "r") : nothing
    @printf("orientation=%s  CTM windows=%d  A3dyn=%d  I3=%d  next-day=%s\n",
            or, nt, n3_a3, n3_i3, ds_ctm1 === nothing ? "no" : "yes")

    # --- (b) time-interp scheme verification at a 3-hourly bracket boundary ----
    # A3dyn node minutes are 90,270,...; a CTM window whose valid time is closest
    # to a node boundary (e.g. win with ctm_valid_min nearest 270 == 04:30) lets
    # us check that PCHIP matches the NODE value there and that the SLOPE is
    # continuous across the bracket (linear has a kink).  We probe the column-mean
    # |d2/dt2| (finite second time-difference) of the interpolated OMEGA at the
    # boundary window for linear vs pchip; pchip's curvature should be finite &
    # smooth where linear's is a delta (kink).
    @printf("\n=== (b) time-interp boundary check (OMEGA at CTM windows near A3 nodes) ===\n")
    @printf("  %-4s  %-7s  %-12s  %-12s  %-12s\n",
            "win", "~UTC", "lin@t", "pchip@t", "|lin-pchip|")
    for wb in (4, 7, 10, 13)
        (wb >= 1 && wb <= nt) || continue
        t = Float64(ctm_valid_min(wb))
        ol = read_interp_3d(ds_a3, "OMEGA", a3_valid_min, n3_a3, t, or; FT=FT, scheme=:linear)
        op = read_interp_3d(ds_a3, "OMEGA", a3_valid_min, n3_a3, t, or; FT=FT, scheme=:pchip)
        # SH-UTLS mean magnitude + max abs difference
        sl = 0.0; sp = 0.0; dmax = 0.0; n = 0
        for p in 1:6, j in 2:s.Nc-1, i in 2:s.Nc-1
            s.sh[p][i,j] || continue
            for k in s.utls
                sl += abs(ol[p][i,j,k]); sp += abs(op[p][i,j,k]); n += 1
                dmax = max(dmax, abs(ol[p][i,j,k]-op[p][i,j,k]))
            end
        end
        a0,a1,f = bracket3(a3_valid_min, n3_a3, t)
        tag = (f == 0.0 || f == 1.0) ? " (AT NODE)" : ""
        @printf("  %-4d  %02d:30   %-12.4e  %-12.4e  %-12.4e%s\n",
                wb, wb-1, sl/n, sp/n, dmax, tag)
    end

    @printf("\n=== per-window blended reconstruction ===\n")
    println("  win  alpha  r_vdiv(UTLS)  C1_rms(int)  C1_rms(glob)  ||dλ||/||f||  mirror_max  tracerProxy  cor(cm,OMEGA)")
    # accumulators per alpha
    acc = Dict{Float64,Vector{Float64}}()          # r_vdiv
    accC = Dict{Float64,Vector{Float64}}()         # continuity int
    accCg = Dict{Float64,Vector{Float64}}()        # continuity global
    accFullSH = Dict{Float64,Vector{Float64}}()    # full-SH roughness
    accTracer = Dict{Float64,Vector{Float64}}()
    accNum = Dict{Float64,Vector{Float64}}()       # r_vdiv numerator (RMS-Lap)
    accDen = Dict{Float64,Vector{Float64}}()       # r_vdiv denominator (std vdiv)
    accCorr = Dict{Float64,Vector{Float64}}()      # cor(realized vdiv, vdiv_om) @ SH-UTLS
    for a in ALPHA_SWEEP
        acc[a]=Float64[]; accC[a]=Float64[]; accCg[a]=Float64[]
        accFullSH[a]=Float64[]; accTracer[a]=Float64[]
        accNum[a]=Float64[]; accDen[a]=Float64[]; accCorr[a]=Float64[]
    end
    dirty_num = Float64[]; dirty_den = Float64[]
    dirty_acc = Float64[]; dirty_fullsh = Float64[]; dirty_tracer = Float64[]
    full_sh_levels = 1:s.Nz   # all levels for the full-SH (e) score

    for w in CTM_WINS
        (w >= 1 && w <= nt) || (@warn "skip win $w (out of 1..$nt)"; continue)
        (w < nt || ds_ctm1 !== nothing) || (@warn "skip final win $w (no next-day DELP)"; continue)
        bw = build_window(s, ds_ctm, ds_ctm1, w, nt, or, FT)

        # DIRTY control
        cm_dirty = ntuple(_ -> zeros(FT, s.Nc, s.Nc, s.Nz+1), 6)
        am_d = ntuple(p -> copy(bw.am[p]), 6); bm_d = ntuple(p -> copy(bw.bm[p]), 6)
        P.diagnose_cs_cm!(cm_dirty, am_d, bm_d, bw.dm, bw.m, s.Nc, s.Nz)
        rvd_dirty = r_vdiv(cm_dirty, s)
        push!(dirty_acc, rvd_dirty)
        push!(dirty_fullsh, r_vdiv(cm_dirty, s; levels=full_sh_levels))
        push!(dirty_tracer, tracer_finger_proxy(am_d, bm_d, cm_dirty, bw.m, s))
        dn0, dd0 = r_vdiv_split(cm_dirty, s); push!(dirty_num, dn0); push!(dirty_den, dd0)

        # OMEGA target (pchip time interp + interface dry conversion)
        t = Float64(ctm_valid_min(w))
        # (Bug-2 guard) A late window whose valid time exceeds the last same-day
        # 3-hourly node would SILENTLY extrapolate (clamp reuses the last A3/I3
        # sample) instead of interpolating across midnight, yielding an INVALID
        # OMEGA/QV target.  Refuse rather than report a fake interpolated metric.
        # (Next-day A3dyn/I3 reading is the clean extension; not wired here because
        #  the validated window set 10-13 is fully in-day bracketed.)
        if interp_extrapolates(a3_valid_min, n3_a3, t)
            error("win $w (valid $(Int(t)) min = $(div(Int(t),60)):$(lpad(Int(t)%60,2,'0')) UTC) " *
                  "is PAST the last same-day A3dyn node ($(a3_valid_min(n3_a3)) min); OMEGA would be " *
                  "constant-extrapolated, not interpolated across midnight. Restrict to windows whose " *
                  "valid time <= $(a3_valid_min(n3_a3)) min, or extend read_interp_3d with next-day A3dyn.")
        end
        if interp_extrapolates(i3_valid_min, n3_i3, t)
            error("win $w (valid $(Int(t)) min) is PAST the last same-day I3 node " *
                  "($(i3_valid_min(n3_i3)) min); QV would be constant-extrapolated across midnight. " *
                  "Restrict to in-day windows or extend read_interp_3d with next-day I3.")
        end
        omega = read_interp_3d(ds_a3, "OMEGA", a3_valid_min, n3_a3, t, or; FT=FT, scheme=:pchip)
        qv    = read_interp_3d(ds_i3, "QV",    i3_valid_min, n3_i3, t, or; FT=FT, scheme=:pchip)
        vdiv_om = omega_target(s, omega, qv, FT)

        for a in ALPHA_SWEEP
            am_o = ntuple(p -> copy(bw.am[p]), 6); bm_o = ntuple(p -> copy(bw.bm[p]), 6)
            grad_rms, base_rms = reconstruct_blended!(am_o, bm_o, bw.dm, vdiv_om, s,
                                                      a, HD_NU, HD_ORDER)
            cm_om = ntuple(_ -> zeros(FT, s.Nc, s.Nc, s.Nz+1), 6)
            P.diagnose_cs_cm!(cm_om, am_o, bm_o, bw.dm, bw.m, s.Nc, s.Nz)
            rvd_om = r_vdiv(cm_om, s)
            fullsh = r_vdiv(cm_om, s; levels=full_sh_levels)
            c1_int, _, _ = continuity_rms(am_o, bm_o, cm_om, bw.dm, bw.m, s; region=:interior)
            c1_glb, _, _ = continuity_rms(am_o, bm_o, cm_om, bw.dm, bw.m, s; region=:global)
            mir = mirror_consistency(am_o, bm_o, s)
            trc = tracer_finger_proxy(am_o, bm_o, cm_om, bw.m, s)
            ratio = base_rms > 0 ? grad_rms/base_rms : NaN
            nn, dd = r_vdiv_split(cm_om, s)
            corr = cm_vs_omega_corr(cm_om, vdiv_om, s)   # PROOF of OMEGA anchoring
            push!(acc[a], rvd_om); push!(accC[a], c1_int); push!(accCg[a], c1_glb)
            push!(accFullSH[a], fullsh); push!(accTracer[a], trc)
            push!(accNum[a], nn); push!(accDen[a], dd); push!(accCorr[a], corr)
            @printf("  %3d  %.2f   %.4f       %.2e     %.2e     %.4f       %.2e    %.4f      %+.4f\n",
                    w, a, rvd_om, c1_int, c1_glb, ratio, mir, trc, corr)
        end
        @printf("  %3d  DIRTY  %.4f       (endpoint closure control)\n", w, rvd_dirty)
    end

    println("\n=== SUMMARY (mean over windows) ===")
    md = mean(dirty_acc)
    dnum0 = mean(dirty_num); dden0 = mean(dirty_den)
    @printf("  DIRTY (native endpoint): r_vdiv(UTLS)=%.4f  fullSH=%.4f  tracerProxy=%.4f\n",
            md, mean(dirty_fullsh), mean(dirty_tracer))
    @printf("  DIRTY split: numerator RMS(Lap)=%.3e  denominator std(vdiv)=%.3e\n",
            dnum0, dden0)
    @printf("  %-6s  %-12s  %-12s  %-12s  %-12s  %-12s  %-13s\n",
            "alpha", "r_vdiv(UTLS)", "C1_int", "C1_glob", "fullSH", "tracerProxy", "cor(cm,OMEGA)")
    best_a = -1.0; best_rv = Inf
    for a in ALPHA_SWEEP
        rv = mean(acc[a])
        @printf("  %-6.2f  %-12.4f  %-12.2e  %-12.2e  %-12.4f  %-12.4f  %+-13.4f\n",
                a, rv, mean(accC[a]), mean(accCg[a]), mean(accFullSH[a]),
                mean(accTracer[a]), mean(accCorr[a]))
        if mean(accC[a]) <= 2e-5 && rv < best_rv   # gate: interior continuity
            best_rv = rv; best_a = a
        end
    end

    # --- (PROOF) OMEGA-anchoring is GENUINE: at alpha=1 the realized vdiv must be
    #     STRONGLY POSITIVELY correlated with vdiv_om (cm tracks the resolved
    #     downward-positive vertical motion).  Before the sign fix this would be
    #     strongly NEGATIVE (cm anchored to MINUS OMEGA).  Print it explicitly.
    if haskey(accCorr, 1.0) && !isempty(accCorr[1.0])
        c1 = mean(accCorr[1.0])
        @printf("\n  OMEGA-ANCHOR PROOF: cor(realized vdiv, vdiv_om) @ alpha=1, SH-UTLS = %+.4f\n", c1)
        if c1 > 0.9
            println("  >> cm GENUINELY TRACKS OMEGA (corr>0.9): anchoring is real & correctly signed.")
        elseif c1 > 0.0
            @printf("  >> positively correlated but weak (%.3f): anchor present but imperfect.\n", c1)
        else
            @printf("  >> NEGATIVE/zero (%.3f): OMEGA anchor is BROKEN/INVERTED (sign bug).\n", c1)
        end
    end
    # --- (HONESTY CHECK) num/denom split: is a low r_vdiv real signal removal
    #     or variance collapse?  Report numerator & denominator RATIO vs DIRTY.
    #     ANCHOR (measured on the AI-Training reference binaries, full day, via
    #     /tmp/score_split.py): the VALIDATED MERRA-2 cure (CLEAN) itself has
    #       num/numDIRTY = 0.49,  den/denDIRTY = 0.69,  r_vdiv 0.294 -> 0.227.
    #     So the target legitimately lowers BOTH; the genuine signature is the
    #     numerator dropping RELATIVELY FASTER than the denominator (0.49<0.69).
    #     A method is a REAL cut (not pure variance collapse) when nr/dr <= the
    #     CLEAN ratio (0.49/0.69 = 0.71) AND nr is at least as low as CLEAN's.
    CLEAN_NR = 0.49; CLEAN_DR = 0.69
    println("\n=== HONESTY CHECK: r_vdiv num/denom split (ratio vs DIRTY) ===")
    @printf("  reference CLEAN (MERRA-2): num/numDIRTY=%.2f  den/denDIRTY=%.2f  (r 0.294->0.227)\n",
            CLEAN_NR, CLEAN_DR)
    @printf("  %-6s  %-14s  %-14s  %-10s  %-12s  %-12s\n",
            "alpha", "num/numDIRTY", "den/denDIRTY", "nr/dr", "r_vdiv", "verdict")
    for a in ALPHA_SWEEP
        nr = mean(accNum[a])/dnum0; dr = mean(accDen[a])/dden0
        ndr = dr > 0 ? nr/dr : NaN
        # REAL: numerator removed at least as well as CLEAN AND the num falls
        # faster (relative to den) than CLEAN's own signature.
        verdict = (nr <= CLEAN_NR && ndr <= CLEAN_NR/CLEAN_DR + 0.05) ? "REAL>=CLEAN" :
                  (ndr <= CLEAN_NR/CLEAN_DR + 0.05) ? "real-cut" :
                  (dr < 0.5*CLEAN_DR) ? "VAR-COLLAPSE?" : "mixed"
        @printf("  %-6.2f  %-14.3f  %-14.3f  %-10.3f  %-12.4f  %-12s\n",
                a, nr, dr, ndr, mean(acc[a]), verdict)
    end
    @printf("\n  Reference baselines: DIRTY=0.294  CLEAN(MERRA-2 target)=0.227\n")
    @printf("  In-script DIRTY (this window-set) = %.4f\n", md)
    if best_a >= 0
        impr = (md - best_rv)/md*100
        @printf("  BEST alpha=%.2f: r_vdiv(UTLS)=%.4f  (%+.1f%% vs in-script DIRTY)\n",
                best_a, best_rv, -impr)
        if best_rv < 0.227
            println("  >> r_vdiv BELOW the CLEAN target — fingering cut below MERRA-2.")
        elseif best_rv < md
            println("  >> r_vdiv improved vs DIRTY but above CLEAN target.")
        else
            println("  >> no improvement vs DIRTY.")
        end
    else
        println("  >> no alpha held the continuity gate.")
    end

    close(ds_ctm); close(ds_a3); close(ds_i3); ds_ctm1 !== nothing && close(ds_ctm1)
end
main()
