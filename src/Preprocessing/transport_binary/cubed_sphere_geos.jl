# ===========================================================================
# Native GEOS-IT/FP cubed-sphere → v4 transport binary preprocessing path.
#
# Source axis:  AbstractGEOSSettings (read native CTM_A1/CTM_I1 NetCDF)
# Target axis:  CubedSphereTargetGeometry, source mesh == target mesh
#               (passthrough — IdentityRegrid)
#
# Critical design choices:
#
#  1. **Global-mean dry-mass pin + column replay balance.** FV3's native
#     MFXC/MFYC are close to discretely conservative, but the dry endpoint
#     derived from GEOS moist PS/QV can carry a small global-mean drift. Pin the
#     output dry-air mass to one global target before balancing horizontal
#     fluxes, then apply the column Poisson correction so the offline binary has
#     a physically conservative endpoint contract.
#
#  2. **Raw dry endpoint mass + diagnosed cm.** FV3's pressure-fixer rule is
#     useful for closing a local vertical flux, but its implied dry endpoint
#     can go negative in GEOS-IT's very thin upper layers even when the raw
#     next-hour GEOS dry mass is healthy. The v4 contract therefore makes the
#     raw GEOS DELP_dry endpoint the written mass target. The native
#     horizontal fluxes are column-balanced to that target, then `cm` is
#     diagnosed from `(am, bm, dm)` so replay and endpoint positivity are both
#     checked against the same physical endpoint.
#
#  3. **Window-by-window loop**:
#
#       read_window!(settings, handles, date, win)         # raw GEOS endpoints
#       geos_native_to_face_flux!(am_v4, bm_v4, ...)       # face-stagger + panel halos
#       derive m_next_target from raw GEOS DELP_dry endpoint
#       pin global dry-mass mean, choose steps, scale fluxes, balance columns
#       diagnose cm from dm = (m_next_target - m_cur)/(2·steps)
#       write window, m_cur ← m_next_target when chaining
# ===========================================================================

"""
    _delp_pa_to_air_mass_kg!(m_kg, m_pa, cell_areas, inv_g) -> m_kg

In-place: convert pressure thickness in Pa to cell air mass in kg per
`m_kg[i, j, k] = m_pa[i, j, k] × cell_areas[i, j] × inv_g`. Cell areas are
in m² and apply identically to every CS panel by symmetry.
"""
function _delp_pa_to_air_mass_kg!(m_kg::AbstractArray{FT, 3},
                                  m_pa::AbstractArray{FT, 3},
                                  cell_areas::AbstractMatrix{FT},
                                  inv_g::FT) where {FT}
    Nx, Ny, Nz = size(m_kg)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        m_kg[i, j, k] = m_pa[i, j, k] * cell_areas[i, j] * inv_g
    end
    return m_kg
end

"""
    _ps_from_air_mass!(ps, m, area, g, Nc, Nz)

Set `ps[i,j] = (Σ_k m[i,j,k]) · g / area[i,j]` (Pa). Used to keep the
binary's stored `ps` consistent with the chained pressure-fixer mass.
"""
function _ps_from_air_mass!(ps::AbstractMatrix{FT},
                            m::AbstractArray{FT, 3},
                            cell_areas::AbstractMatrix{FT},
                            g::FT, Nc::Int, Nz::Int) where {FT}
    @inbounds for j in 1:Nc, i in 1:Nc
        s = zero(FT)
        for k in 1:Nz
            s += m[i, j, k]
        end
        ps[i, j] = s * g / cell_areas[i, j]
    end
    return ps
end

"""
    _evolve_mass_pressure_fixer!(m_next, m_cur, am_v4, bm_v4, ΔB, two_steps, Nc, Nz)

FV3 pressure-fixer mass evolution (restored from commit e648bf3f):

    pit       = Σ_k (am[i,j,k] − am[i+1,j,k] + bm[i,j,k] − bm[i,j+1,k])
    m_next[k] = m_cur[k] + two_steps · ΔB[k] · pit

This is the endpoint implied by `compute_cs_cm_pressure_fixer!`'s closure
`cm[k+1]−cm[k] = C_k − ΔB[k]·pit`, so the window replay closes to roundoff.
The `two_steps` factor cancels the per-window flux scaling (`am ∝ 1/steps`), so
`m_next` is independent of the chosen substep count. Globally mass-conserving
(Σ_cells pit = 0 on the closed sphere), so the dry-mass pin is a no-op here.
"""
function _evolve_mass_pressure_fixer!(
        m_next::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        m_cur::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        am_v4::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        bm_v4::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        ΔB::AbstractVector,
        two_steps::FT, Nc::Int, Nz::Int) where {FT}
    @inbounds for p in 1:CS_PANEL_COUNT
        am = am_v4[p]; bm = bm_v4[p]
        m  = m_cur[p]; mn = m_next[p]
        for j in 1:Nc, i in 1:Nc
            pit = zero(FT)
            for k in 1:Nz
                pit += (am[i, j, k] - am[i + 1, j, k]) +
                       (bm[i, j, k] - bm[i, j + 1, k])
            end
            for k in 1:Nz
                mn[i, j, k] = m[i, j, k] + two_steps * FT(ΔB[k]) * pit
            end
        end
    end
    return nothing
end

"""
    _smooth_cs_residual_panels!(field, niter, w, Nc, Nz)

In-place horizontal Jacobi smoothing of the moisture-source residual, applied
INDEPENDENTLY and IDENTICALLY to each vertical level of every panel. The stencil
is the 4-neighbour interior average with weight `w`; panel-edge cells average
only their in-panel neighbours (no cross-panel exchange — the SH-UTLS fingering
lives in panel interiors, and a missing seam neighbour leaves the gate identity
untouched). Because the operator is LINEAR and LEVEL-INDEPENDENT,
`Σ_k smooth(rₖ) = smooth(Σ_k rₖ)`; with a zero column-integral residual this is
`smooth(0) = 0`, so the column closure (and thus surface pressure) is preserved
exactly while only the grid-scale per-layer structure is damped.
"""
function _smooth_cs_residual_panels!(field::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                     niter::Int, w::FT, Nc::Int, Nz::Int) where FT
    niter <= 0 && return nothing
    scratch = Array{FT}(undef, Nc, Nc)
    @inbounds for p in 1:CS_PANEL_COUNT
        f = field[p]
        for k in 1:Nz
            for _ in 1:niter
                for j in 1:Nc, i in 1:Nc
                    scratch[i, j] = f[i, j, k]
                end
                for j in 1:Nc, i in 1:Nc
                    s = zero(FT); n = 0
                    i > 1  && (s += scratch[i - 1, j]; n += 1)
                    i < Nc && (s += scratch[i + 1, j]; n += 1)
                    j > 1  && (s += scratch[i, j - 1]; n += 1)
                    j < Nc && (s += scratch[i, j + 1]; n += 1)
                    f[i, j, k] = (one(FT) - w) * scratch[i, j] + w * (s / FT(n))
                end
            end
        end
    end
    return nothing
end

"""
    _smooth_cs_columns!(field, niter, w, Nc)

In-place horizontal Jacobi low-pass of a per-panel 2D column field (e.g. the
column dry-mass drift), GLOBAL-SUM-PRESERVING (a uniform offset restores the
total after smoothing, so a zero-sum input stays zero-sum and the global dry
mass lands exactly on the analyzed target). Per-panel (no cross-panel exchange);
keeps the LARGE-SCALE part of the field and damps grid scales.
"""
function _smooth_cs_columns!(field::NTuple{CS_PANEL_COUNT, Array{FT, 2}},
                             niter::Int, w::FT, Nc::Int) where FT
    niter <= 0 && return nothing
    total_before = 0.0
    @inbounds for p in 1:CS_PANEL_COUNT, v in field[p]
        total_before += Float64(v)
    end
    scratch = Array{FT}(undef, Nc, Nc)
    @inbounds for p in 1:CS_PANEL_COUNT
        f = field[p]
        for _ in 1:niter
            copyto!(scratch, f)
            for j in 1:Nc, i in 1:Nc
                s = zero(FT); n = 0
                i > 1  && (s += scratch[i - 1, j]; n += 1)
                i < Nc && (s += scratch[i + 1, j]; n += 1)
                j > 1  && (s += scratch[i, j - 1]; n += 1)
                j < Nc && (s += scratch[i, j + 1]; n += 1)
                f[i, j] = (one(FT) - w) * scratch[i, j] + w * (s / FT(n))
            end
        end
    end
    total_after = 0.0
    @inbounds for p in 1:CS_PANEL_COUNT, v in field[p]
        total_after += Float64(v)
    end
    offset = FT((total_before - total_after) / (CS_PANEL_COUNT * Nc * Nc))
    @inbounds for p in 1:CS_PANEL_COUNT, idx in eachindex(field[p])
        field[p][idx] += offset
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Env-gated timing/diagnostic accumulator for OMEGA-based preparation.
# (set `ATMOS_OMEGA_TIMING=1`). Counts per-window prepares, omega Poisson solves,
# and CG iterations so the build-cost diagnosis is measurable without touching
# the production hot path when the env var is unset.
const _OMEGA_TIMING = Base.RefValue(false)
# Per-level Poisson parallelism for OMEGA target reconstruction.
# `true` in the single-day-per-process (`--day`) path so the level solve grabs
# the full thread pool (the validated/production usage, ~5.0× speedup). The
# multi-day driver sets it `false` BEFORE its `Threads.@threads` day loop so the
# inner per-level loop runs SERIAL — otherwise each day-worker re-grabs the whole
# pool (oversubscription / severe multi-day regression). Serial uses
# `scratches[1]` so it is bit-identical to the parallel path.
const _OMEGA_LEVEL_PARALLEL = Base.RefValue(true)

"""
    OmegaRegularization

Controls the scale-selective OMEGA prior used by `:omega_regularized`.

`pressure_taper_hpa = (outer_top, inner_top, inner_bottom, outer_bottom)`
defines a smooth pressure window: the correction is zero outside the outer
bounds and fully active between the inner bounds. `smoothing_steps` conservative
graph-diffusion sweeps define the low-pass field; OMEGA contributes only the
remaining high-pass difference from the endpoint-balanced vertical flux.
`max_relative_flux_correction` caps the RMS X/Y face-flux increment separately
at every level. `max_bottom_flux_correction` is a hard fidelity gate on the
bottom three model layers, where surface-source transport must remain native.
"""
Base.@kwdef struct OmegaRegularization
    pressure_taper_hpa::NTuple{4, Float64} = (50.0, 80.0, 300.0, 350.0)
    smoothing_steps::Int = 3
    smoothing_fraction::Float64 = 0.10
    max_relative_flux_correction::Float64 = 0.10
    max_bottom_flux_correction::Float64 = 0.01
end

struct OmegaRegularizationScratch{FT}
    omega_cm::NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    delta::Vector{Float64}
    lowpass::Vector{Float64}
    next::Vector{Float64}
    pressure_hpa::Vector{Float64}
    active_levels::BitVector
end

@inline _uses_omega(closure::Symbol) =
    closure === :omega_full_replacement || closure === :omega_regularized

function _validate_omega_regularization(options::OmegaRegularization)
    p0, p1, p2, p3 = options.pressure_taper_hpa
    0.0 <= p0 < p1 <= p2 < p3 ||
        throw(ArgumentError("OMEGA pressure taper must satisfy 0 ≤ outer_top < inner_top ≤ inner_bottom < outer_bottom; got $(options.pressure_taper_hpa)"))
    options.smoothing_steps >= 1 ||
        throw(ArgumentError("OMEGA smoothing_steps must be ≥ 1; got $(options.smoothing_steps)"))
    # A degree-four graph has λmax ≤ 8. Keeping fraction ≤ 1/8 makes every
    # eigenvalue of (I - fraction*L) non-negative, so the derived high-pass
    # cannot amplify a checkerboard through an odd-step sign reversal.
    0.0 < options.smoothing_fraction <= 0.125 ||
        throw(ArgumentError("OMEGA smoothing_fraction must lie in (0, 0.125]; got $(options.smoothing_fraction)"))
    0.0 < options.max_relative_flux_correction <= 1.0 ||
        throw(ArgumentError("OMEGA max_relative_flux_correction must lie in (0, 1]; got $(options.max_relative_flux_correction)"))
    0.0 < options.max_bottom_flux_correction <= 1.0 ||
        throw(ArgumentError("OMEGA max_bottom_flux_correction must lie in (0, 1]; got $(options.max_bottom_flux_correction)"))
    return options
end

@inline function _omega_pressure_weight(p_hpa::Float64,
                                        bounds::NTuple{4, Float64})
    p0, p1, p2, p3 = bounds
    if p_hpa <= p0 || p_hpa >= p3
        return 0.0
    elseif p_hpa < p1
        x = (p_hpa - p0) / (p1 - p0)
        return 0.5 - 0.5 * cospi(x)
    elseif p_hpa <= p2
        return 1.0
    else
        x = (p_hpa - p2) / (p3 - p2)
        return 0.5 + 0.5 * cospi(x)
    end
end

function _smooth_cs_graph_conservative!(lowpass::Vector{Float64},
                                        next::Vector{Float64},
                                        ft::CSGlobalFaceTable,
                                        steps::Int,
                                        fraction::Float64)
    for _ in 1:steps
        copyto!(next, lowpass)
        @inbounds for f in 1:ft.nf
            left = Int(ft.face_left[f])
            right = Int(ft.face_right[f])
            exchange = fraction * (lowpass[right] - lowpass[left])
            next[left] += exchange
            next[right] -= exchange
        end
        lowpass, next = next, lowpass
    end
    return lowpass
end
mutable struct _OmegaTimingState
    prepares::Int          # `_geos_prepare_window_for_steps!` calls (omega path)
    solves::Int            # per-level Poisson solves issued
    cg_iters::Int          # total CG iterations across all solves
    recon_time::Float64    # wall seconds inside `_reconstruct_omega_target!`
end
const _OMEGA_TIMING_STATE = _OmegaTimingState(0, 0, 0, 0.0)
function _reset_omega_timing!()
    s = _OMEGA_TIMING_STATE
    s.prepares = 0; s.solves = 0; s.cg_iters = 0; s.recon_time = 0.0
    return s
end

# ---------------------------------------------------------------------------
# OMEGA target reconstruction shared by the regularized and diagnostic modes.
#
# The diagnosed cm[k+1]=cm[k]+div_h[k]-dm[k] forces the grid-noisy MFXC↔DELP
# residual M into cm, so the per-layer vertical convergence vdiv=cm[k]-cm[k+1]
# (==div_h) is grid-rough at the SH-UTLS → "fingering". GEOS A3dyn archives
# OMEGA, the model's RESOLVED vertical pressure velocity, ~2x smoother than
# div_h(MFXC). We build a SMOOTH physical vertical-convergence target vdiv_om
# from OMEGA (DOWNWARD-positive, same sign as cm; dry-corrected by I3 QV), then
# per level solve for a least-norm horizontal flux POTENTIAL λ so the NEW
# horizontal convergence is div_h_new[k] = dm[k] − vdiv_om[k]; the telescoped cm
# then gives EXACTLY vdiv[k] = dm[k]−div_h_new[k] = +vdiv_om[k] (smooth, cm
# TRACKS OMEGA), while continuity holds BY CONSTRUCTION (the correction lives in
# continuity's null space → the replay gate passes, and Σ_k vdiv_om = 0 ⇒
# cm[Nz+1]=0).  alpha=1 (pure OMEGA) only; the hyperdiffusive-fallback blend in
# the prototype is not productionized.
#
# Validated at the binary level (r_vdiv 0.197 ≈ MERRA-2 CLEAN 0.227, continuity
# 4e-10, cor(cm,OMEGA)=+1.00). See
# scripts/diagnostics/fingerfix_proto_omega-consistent-flux-reconstruction.jl.
# ---------------------------------------------------------------------------

# --- Monotone-cubic (PCHIP / Fritsch-Carlson) 3-hourly→hourly time interp -----
# Uniform 3-hourly node spacing ⇒ the interior PCHIP slope is the harmonic-mean
# limited secant; C1 curve, no kink across a bracket boundary, monotone (no
# over/undershoot). At a node it returns that node exactly. Same scheme as the
# validated prototype.
@inline function _pchip_slope(dm1::Float64, d0::Float64)
    (dm1 == 0.0 || d0 == 0.0 || sign(dm1) != sign(d0)) && return 0.0
    return 2.0 / (1.0 / dm1 + 1.0 / d0)
end
@inline function _pchip_eval(y1::Float64, y2::Float64, y3::Float64, y4::Float64,
                             f::Float64)
    d12 = y2 - y1; d23 = y3 - y2; d34 = y4 - y3
    m2 = _pchip_slope(d12, d23)
    m3 = _pchip_slope(d23, d34)
    h00 = (1 + 2f) * (1 - f)^2
    h10 = f * (1 - f)^2
    h01 = f^2 * (3 - 2f)
    h11 = f^2 * (f - 1)
    return h00 * y2 + h10 * m2 + h01 * y3 + h11 * m3
end

# CTM_A1 window w (1..24) valid minute; A3dyn / I3 3-hourly node valid minutes.
# The A3dyn / I3 node valid-minute formulas are GLOBAL: they extend to node
# indices ≤ 0 (previous UTC day) and > n3 (next UTC day) at the same uniform
# 180-min spacing, so `_a3_valid_min(0)` = −90 (prev-day 22:30) and
# `_a3_valid_min(n3+1)` = next-day 01:30. This lets the day-edge PCHIP brackets
# span midnight when the adjacent-day handles are open.
@inline _ctm_valid_min(w::Int) = (w - 1) * 60 + 30
@inline _a3_valid_min(a::Int) = (a - 1) * 180 + 90
@inline _i3_valid_min(a::Int) = (a - 1) * 180

# Map a GLOBAL node index `g` (may be ≤0 = prev day, or >n3 = next day) to the
# dataset + 1-based local index that holds it. Returns `nothing` when the
# required adjacent-day dataset is absent (archive edge) so the caller can clamp
# back into today's range (the legacy constant-extrapolation fallback).
@inline function _resolve_global_node(g::Int, n3::Int, today, prev, next)
    if 1 <= g <= n3
        return (today, g)
    elseif g <= 0
        prev === nothing && return nothing
        n_prev = prev.dim["time"]
        loc = g + n_prev
        return loc >= 1 ? (prev, loc) : nothing
    else # g > n3
        next === nothing && return nothing
        n_next = next.dim["time"]
        loc = g - n3
        return loc <= n_next ? (next, loc) : nothing
    end
end

# 4-node PCHIP stencil over the GLOBAL node axis for target minute `t`, clamping
# each stencil index to the range of nodes actually available (today plus any
# present adjacent days). Returns global stencil `(gm1, g0, g1, gp2)`, the local
# fraction `f` between `g0` and `g1`, and `atnode` (g0 == g1, exact hit).
function _pchip_bracket_global(valid_min::Function, n3::Int, t::Float64,
                               gmin::Int, gmax::Int)
    # Largest global node index with valid_min ≤ t, searched over [gmin, gmax].
    a = gmin
    for g in gmin:gmax
        valid_min(g) <= t && (a = g)
    end
    g0 = clamp(a, gmin, gmax); g1 = clamp(a + 1, gmin, gmax)
    t0 = Float64(valid_min(g0)); t1 = Float64(valid_min(g1))
    f = (t1 == t0) ? 0.0 : clamp((t - t0) / (t1 - t0), 0.0, 1.0)
    gm1 = clamp(g0 - 1, gmin, gmax); gp2 = clamp(g1 + 1, gmin, gmax)
    return (gm1, g0, g1, gp2), f, (g0 == g1)
end

"""
    _read_geos_omega_qv_pchip!(omega, qv, handles, win, Nc, Nz, FT)

Read A3dyn OMEGA and I3 QV at the CTM window `win`'s valid time via monotone
cubic (PCHIP) interpolation of the 3-hourly nodes, level-flipped to TOA-first.
Day-edge windows whose valid time lies outside the same-day node span (win 1 is
BEFORE the first A3dyn node 01:30; win 23/24 are PAST the last A3dyn node 22:30 /
last I3 node 21:00) bracket ACROSS midnight into the previous/next day's nodes
when `handles.prev_a3dyn`/`next_a3dyn`/`prev_i3`/`next_i3` are open — removing the
former constant-extrapolation discontinuity at the day boundary. When an adjacent
handle is absent (first/last day of the archive) that edge clamps to the nearest
same-day node (the legacy bounded constant-extrapolation). Fills `omega`/`qv`
(NTuple{6,Array{FT,3}}).
"""
function _read_geos_omega_qv_pchip!(omega::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                    qv::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                    handles::GEOSDayHandles, win::Int,
                                    Nc::Int, Nz::Int) where FT
    handles.a3dyn === nothing &&
        error("OMEGA-based cm closure needs A3dyn OMEGA; set include_vdiff_fields=true")
    handles.i3 === nothing &&
        error("OMEGA-based cm closure needs I3 QV; set include_vdiff_fields=true")
    or = handles.orientation
    n3_a3 = handles.a3dyn.dim["time"]
    n3_i3 = handles.i3.dim["time"]
    t = Float64(_ctm_valid_min(win))
    _read_pchip_field_xday!(omega, "OMEGA", _a3_valid_min, n3_a3, t, or, FT,
                            handles.a3dyn, handles.prev_a3dyn, handles.next_a3dyn)
    _read_pchip_field_xday!(qv, "QV", _i3_valid_min, n3_i3, t, or, FT,
                            handles.i3, handles.prev_i3, handles.next_i3)
    return nothing
end

# Read one PCHIP-interpolated field, with the 4-node stencil resolved across the
# previous/today/next-day datasets. `gmin`/`gmax` bound the global stencil to the
# nodes that are actually on disk: today (1..n3) is always present, prev extends
# down to `1 - n_prev` when `prev` is open, next up to `n3 + n_next` when `next`
# is open.
function _read_pchip_field_xday!(out::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                 var::AbstractString, valid_min::Function,
                                 n3::Int, t::Float64, or::Symbol, ::Type{FT},
                                 today, prev, next) where FT
    gmin = prev === nothing ? 1 : 1 - prev.dim["time"]
    gmax = next === nothing ? n3 : n3 + next.dim["time"]
    (nodes, f, atnode) = _pchip_bracket_global(valid_min, n3, t, gmin, gmax)
    if atnode && f == 0.0
        (ds, loc) = _resolve_global_node(nodes[2], n3, today, prev, next)
        y = _read_panels_3d(ds[var], loc, or; FT = FT)
        for p in 1:CS_PANEL_COUNT; copyto!(out[p], y[p]); end
        return out
    end
    ys = ntuple(4) do s
        (ds, loc) = _resolve_global_node(nodes[s], n3, today, prev, next)
        _read_panels_3d(ds[var], loc, or; FT = FT)
    end
    y1, y2, y3, y4 = ys
    @inbounds for p in 1:CS_PANEL_COUNT
        o = out[p]; a = y1[p]; b = y2[p]; c = y3[p]; d = y4[p]
        for idx in eachindex(o)
            o[idx] = FT(_pchip_eval(Float64(a[idx]), Float64(b[idx]),
                                    Float64(c[idx]), Float64(d[idx]), f))
        end
    end
    return out
end

"""
    _omega_vdiv_target!(vdiv_om, omega, qv, cell_areas, g, tau, Nc, Nz)

Build the OMEGA-derived smooth per-layer vertical mass-convergence target
(downward-positive, same sign as cm), INTERFACE-consistent dry conversion:
qv_ifc[k]=0.5(qv[k-1]+qv[k]); the dry interface pressure velocity is
omega_ifc·(1−qv_ifc); the per-layer convergence is the telescoped interface
difference ·area/g·tau. Because omega_dry_ifc[1]=omega_dry_ifc[Nz+1]=0,
Σ_k vdiv_om = 0 exactly (matches cm[1]=cm[Nz+1]=0). `tau = MFDT/2`.
"""
function _omega_vdiv_target!(vdiv_om::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                             omega::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                             qv::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                             cell_areas::AbstractMatrix,
                             g::FT, tau::FT, Nc::Int, Nz::Int) where FT
    @inbounds for p in 1:CS_PANEL_COUNT
        om = omega[p]; q = qv[p]; vo = vdiv_om[p]
        for j in 1:Nc, i in 1:Nc
            a = FT(cell_areas[i, j])
            for k in 1:Nz
                om_top = (k == 1)  ? zero(FT) : FT(0.5) * (om[i, j, k - 1] + om[i, j, k])
                om_bot = (k == Nz) ? zero(FT) : FT(0.5) * (om[i, j, k] + om[i, j, k + 1])
                qv_top = (k == 1)  ? zero(FT) : FT(0.5) * (q[i, j, k - 1] + q[i, j, k])
                qv_bot = (k == Nz) ? zero(FT) : FT(0.5) * (q[i, j, k] + q[i, j, k + 1])
                od_top = om_top * (one(FT) - qv_top)
                od_bot = om_bot * (one(FT) - qv_bot)
                vo[i, j, k] = (a / g) * (od_top - od_bot) * tau
            end
        end
    end
    return vdiv_om
end

"""
    _regularize_omega_target!(target, native_cm, omega_vdiv, m_cur, m_next, grid, g,
                              vdiv_scale, options, scratch)

Build a conservative, scale-selective OMEGA vertical-convergence target.

The endpoint-balanced `native_cm` remains the large-scale reference. OMEGA
contributes only the horizontal high-pass part of its interface-flux difference
from `native_cm`, and only inside the configured pressure taper. Constructing
the blend on interfaces (rather than independently on layers) preserves zero
top/surface flux and therefore `sum(target; dims=level) == 0` per column.
"""
function _regularize_omega_target!(
        target::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        native_cm::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        omega_vdiv::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        m_cur::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        m_next::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
        grid::CubedSphereTargetGeometry,
        g::FT,
        vdiv_scale::Float64,
        options::OmegaRegularization,
        scratch::OmegaRegularizationScratch{FT}) where FT
    Nc = grid.Nc
    Nz = size(target[1], 3)
    nc = CS_PANEL_COUNT * Nc * Nc
    ft = grid.face_table
    omega_cm = scratch.omega_cm
    active_levels = scratch.active_levels
    length(active_levels) == Nz ||
        throw(DimensionMismatch("OMEGA active-level mask has length $(length(active_levels)); expected $Nz"))
    fill!(active_levels, false)

    # Telescope the OMEGA convergence into a downward-positive interface flux.
    @inbounds for p in 1:CS_PANEL_COUNT
        oc = omega_cm[p]
        vo = omega_vdiv[p]
        fill!(view(oc, :, :, 1), zero(FT))
        for j in 1:Nc, i in 1:Nc
            accum = 0.0
            for k in 1:Nz
                accum -= vdiv_scale * Float64(vo[i, j, k])
                oc[i, j, k + 1] = FT(accum)
            end
        end
    end

    fill!(scratch.pressure_hpa, 0.0)
    # Top and surface remain native (both zero). Interior interfaces receive the
    # UTLS-tapered high-pass OMEGA-minus-native increment.
    previous_interface_active = false
    @inbounds for k_ifc in 1:(Nz + 1)
        # Pressure varies horizontally, so an interface is active when at least
        # one cell has non-zero taper weight. Entirely inactive interfaces are
        # copied from the native closure exactly: no graph smoother and, below,
        # no Poisson solve on layers bounded by two inactive interfaces.
        interface_active = any(p -> _omega_pressure_weight(
            p, options.pressure_taper_hpa) > 0.0, scratch.pressure_hpa)
        if interface_active
            for p in 1:CS_PANEL_COUNT, j in 1:Nc, i in 1:Nc
                c = i + (j - 1) * Nc + (p - 1) * Nc * Nc
                scratch.delta[c] = Float64(omega_cm[p][i, j, k_ifc]) -
                                   Float64(native_cm[p][i, j, k_ifc])
            end
            copyto!(scratch.lowpass, scratch.delta)
            lowpass = _smooth_cs_graph_conservative!(
                scratch.lowpass, scratch.next, ft, options.smoothing_steps,
                options.smoothing_fraction)
            for p in 1:CS_PANEL_COUNT, j in 1:Nc, i in 1:Nc
                c = i + (j - 1) * Nc + (p - 1) * Nc * Nc
                weight = _omega_pressure_weight(scratch.pressure_hpa[c],
                                                options.pressure_taper_hpa)
                highpass = scratch.delta[c] - lowpass[c]
                omega_cm[p][i, j,k_ifc] =
                    FT(Float64(native_cm[p][i, j, k_ifc]) + weight * highpass)
            end
        else
            for p in 1:CS_PANEL_COUNT
                copyto!(view(omega_cm[p], :, :, k_ifc),
                        view(native_cm[p], :, :, k_ifc))
            end
        end
        k_ifc > 1 &&
            (active_levels[k_ifc - 1] = previous_interface_active || interface_active)
        previous_interface_active = interface_active
        if k_ifc <= Nz
            for p in 1:CS_PANEL_COUNT, j in 1:Nc, i in 1:Nc
                c = i + (j - 1) * Nc + (p - 1) * Nc * Nc
                scratch.pressure_hpa[c] +=
                    0.5 * (Float64(m_cur[p][i, j, k_ifc]) +
                           Float64(m_next[p][i, j, k_ifc])) * Float64(g) /
                    Float64(grid.mesh.cell_areas[i, j]) / 100.0
            end
        end
    end

    @inbounds for p in 1:CS_PANEL_COUNT, k in 1:Nz, j in 1:Nc, i in 1:Nc
        target[p][i, j, k] = omega_cm[p][i, j, k] - omega_cm[p][i, j, k + 1]
    end
    return target
end

"""
    _reconstruct_omega_target!(am, bm, dm, vdiv_om, grid, vdiv_scale; tol, max_iter)

After the column balance (so div_h closes the column: Σ_k div_h = Σ_k dm), apply
a per-level Poisson flux-potential correction so the NEW horizontal convergence
is div_h_new[k] = dm[k] − vdiv_scale·vdiv_om[k]. Structurally identical to
`_balance_cs_level!`: drive the graph divergence (= −div_h) toward
−(dm − vdiv) by solving L·ψ = (div_current − desired) and applying the flux
correction. `vdiv_scale = source_steps_per_met/steps` matches the per-substep
flux scaling without mutating the stored base-scaled `vdiv_om` (so the adaptive
loop can re-prepare at a different `steps` without F32 drift). The realized
telescoped cm then has vdiv[k] = dm[k] − div_h_new[k] = +vdiv_scale·vdiv_om[k]
UP TO a per-level global constant (the unrealizable mean removed before the
solve; zero grid-scale signature so r_vdiv is unchanged, global-mean part
reconciled by the dry-mass pin). Continuity holds per column to roundoff. Returns
the maximum increment and post-solve residual, the maximum global and local
relative corrections, and the global RMS relative correction for every level.
The local relative correction is diagnostic only; the per-level RMS values feed
the hard lower-layer fidelity gate.
"""
# Single-level OMEGA-consistent flux-potential correction. Independent per level
# (touches only `am[:,:,k]`/`bm[:,:,k]` and the supplied per-thread `scratch`),
# so the Nz levels can be solved concurrently. Returns the per-level correction
# magnitude, post-residual, CG iteration count, and global/local relative
# corrections for the gate and timing reductions.
@inline function _reconstruct_omega_level!(k::Int,
                                           am::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                           bm::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                           dm::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                           vdiv_om::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                           ft::CSGlobalFaceTable,
                                           degree::Vector{Int},
                                           scratch::CSPoissonScratch,
                                           vdiv_scale::Float64,
                                           Nc::Int, nc::Int;
                                           tol::Float64, max_iter::Int,
                                           max_relative_correction::Float64) where FT
    div = scratch.div
    rhs = scratch.rhs
    psi = scratch.psi
    cg_scratch = (r = scratch.r, p = scratch.p, Ap = scratch.Ap, z = scratch.z)
    @inbounds begin
        # Current graph divergence at level k (= −div_h).
        fill!(div, 0.0)
        for f in 1:ft.nf
            panel = Int(ft.face_panel[f]); dir = Int(ft.face_dir[f])
            i = Int(ft.face_idx_i[f]); j = Int(ft.face_idx_j[f])
            flux = dir == 1 ? Float64(am[panel][i, j, k]) : Float64(bm[panel][i, j, k])
            div[Int(ft.face_left[f])]  += flux
            div[Int(ft.face_right[f])] -= flux
        end
        # Desired graph divergence: −div_h_new = −(dm − vdiv_om).
        # rhs[c] = div_current[c] − desired_graph_div[c] = div[c] + dh_new[c].
        # A horizontal flux divergence is globally mean-zero per level, so ONLY
        # the null-space (mean-zero) part of dh_new is realizable as a flux
        # correction. Subtract its per-level global mean EXPLICITLY (rather than
        # leaning on the solver's internal mean-zero projection): the realized
        # vdiv = vdiv_scale·vdiv_om + (a per-level global CONSTANT). That constant
        # has zero grid-scale Laplacian, so the r_vdiv fingering metric is
        # UNCHANGED; the dropped global-mean part is the net per-level mass
        # tendency that cm carries, reconciled to ~0 by the dry-mass pin (so the
        # column bottom residual handed to diagnose_cs_cm! is the global-mean
        # drift only, not a per-column leak). [Codex P1: make this intentional.]
        dh_sum = 0.0
        for c in 1:nc
            p_idx = (c - 1) ÷ (Nc * Nc) + 1
            li = (c - 1) % (Nc * Nc); jl = li ÷ Nc + 1; il = li % Nc + 1
            dh_new = Float64(dm[p_idx][il, jl, k]) -
                     vdiv_scale * Float64(vdiv_om[p_idx][il, jl, k])
            rhs[c] = div[c] + dh_new
            dh_sum += dh_new
        end
        dh_mean = dh_sum / nc
        @simd for c in 1:nc
            rhs[c] -= dh_mean
        end
        _, cg_iter = solve_cs_poisson_pcg!(psi, rhs, ft, degree, cg_scratch;
                              tol = tol, max_iter = max_iter, project_every = 50)
        correction2 = 0.0
        base2 = 0.0
        for f in 1:ft.nf
            left = Int(ft.face_left[f]); right = Int(ft.face_right[f])
            d = psi[right] - psi[left]
            panel = Int(ft.face_panel[f]); dir = Int(ft.face_dir[f])
            i = Int(ft.face_idx_i[f]); j = Int(ft.face_idx_j[f])
            base = dir == 1 ? Float64(am[panel][i, j, k]) : Float64(bm[panel][i, j, k])
            correction2 += d * d
            base2 += base * base
        end
        requested_relative = if correction2 == 0.0
            0.0
        elseif base2 > 0.0
            sqrt(correction2 / base2)
        else
            Inf
        end
        applied_scale = requested_relative > max_relative_correction ?
            max_relative_correction / requested_relative : 1.0
        if applied_scale < 1.0
            @simd for c in 1:nc
                psi[c] *= applied_scale
            end
        end
        # Report the largest local change against a non-singular characteristic
        # flux. This is diagnostic only: clipping individual levels independently
        # would destroy the vertically integrated face-flux closure.
        base_rms = sqrt(base2 / ft.nf)
        applied2 = 0.0
        max_inc = 0.0
        max_local_relative = 0.0
        for f in 1:ft.nf
            left = Int(ft.face_left[f]); right = Int(ft.face_right[f])
            delta = psi[right] - psi[left]
            panel = Int(ft.face_panel[f]); dir = Int(ft.face_dir[f])
            i = Int(ft.face_idx_i[f]); j = Int(ft.face_idx_j[f])
            base = dir == 1 ? Float64(am[panel][i, j, k]) : Float64(bm[panel][i, j, k])
            characteristic = max(abs(base), base_rms)
            magnitude = abs(delta)
            max_inc = max(max_inc, magnitude)
            applied2 += delta * delta
            local_relative = characteristic > 0.0 ? magnitude / characteristic : 0.0
            max_local_relative = max(max_local_relative, local_relative)
        end
        apply_cs_flux_correction!(am, bm, psi, ft, k)
        fill!(div, 0.0)
        for f in 1:ft.nf
            panel = Int(ft.face_panel[f]); dir = Int(ft.face_dir[f])
            i = Int(ft.face_idx_i[f]); j = Int(ft.face_idx_j[f])
            flux = dir == 1 ? Float64(am[panel][i, j, k]) : Float64(bm[panel][i, j, k])
            div[Int(ft.face_left[f])]  += flux
            div[Int(ft.face_right[f])] -= flux
        end
        max_post = 0.0
        for c in 1:nc
            p_idx = (c - 1) ÷ (Nc * Nc) + 1
            li = (c - 1) % (Nc * Nc); jl = li ÷ Nc + 1; il = li % Nc + 1
            dh_new = Float64(dm[p_idx][il, jl, k]) -
                     vdiv_scale * Float64(vdiv_om[p_idx][il, jl, k])
            r = abs(div[c] - (-dh_new))
            r > max_post && (max_post = r)
        end
    end
    applied_relative = base2 > 0.0 ? sqrt(applied2 / base2) : 0.0
    return (max_inc, max_post, cg_iter, applied_relative, max_local_relative)
end

function _reconstruct_omega_target!(am::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                        bm::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                        dm::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                        vdiv_om::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                        grid::CubedSphereTargetGeometry,
                                        vdiv_scale::Float64;
                                        tol::Float64 = 1e-11,
                                        max_iter::Int = 8000,
                                        max_relative_correction::Float64 = Inf,
                                        active_levels::Union{Nothing, AbstractVector{Bool}} = nothing) where FT
    ft = grid.face_table
    degree = grid.cell_degree
    Nc = ft.Nc
    nc = ft.nc
    Nz = size(dm[1], 3)
    active_levels === nothing || length(active_levels) == Nz ||
        throw(DimensionMismatch("active_levels has length $(length(active_levels)); expected $Nz"))
    levels = active_levels === nothing ? (1:Nz) : findall(active_levels)

    # Each level is an independent Poisson solve; give every thread its own
    # CSPoissonScratch so the per-level div/rhs/psi/CG buffers never alias.
    # `apply_cs_flux_correction!` writes only level k (incl. its mirror entries),
    # so the cross-panel mirror sync is deferred to a single pass at the end.
    # The CG is a deterministic sequential solve on a per-level RHS, so the
    # written am/bm/cm are BIT-IDENTICAL to the serial loop regardless of the
    # thread schedule.
    nthread = Threads.maxthreadid()
    scratches = Vector{CSPoissonScratch}(undef, nthread)
    scratches[1] = grid.poisson_scratch
    for t in 2:nthread
        scratches[t] = CSPoissonScratch(nc)
    end

    inc_by_level = zeros(Float64, Nz)
    post_by_level = zeros(Float64, Nz)
    relative_by_level = zeros(Float64, Nz)
    local_relative_by_level = zeros(Float64, Nz)
    iter_by_level = zeros(Int, Nz)
    # Per-level parallelism only when the level solve owns the pool. The
    # multi-day driver clears `_OMEGA_LEVEL_PARALLEL` before its day `@threads`
    # so this runs SERIAL (no oversubscription); single-day `--day` runs keep
    # it set and use the full pool. Serial uses `scratches[1]`, so the written
    # am/bm/cm are bit-identical regardless of path.
    use_threads = _OMEGA_LEVEL_PARALLEL[] && Threads.maxthreadid() > 1
    if use_threads
        Threads.@threads :static for active_idx in eachindex(levels)
            k = levels[active_idx]
            mi, mp, ci, rel, local_rel = _reconstruct_omega_level!(
                k, am, bm, dm, vdiv_om, ft, degree,
                scratches[Threads.threadid()], vdiv_scale, Nc, nc;
                tol = tol, max_iter = max_iter,
                max_relative_correction = max_relative_correction)
            inc_by_level[k] = mi
            post_by_level[k] = mp
            iter_by_level[k] = ci
            relative_by_level[k] = rel
            local_relative_by_level[k] = local_rel
        end
    else
        for k in levels
            mi, mp, ci, rel, local_rel = _reconstruct_omega_level!(
                k, am, bm, dm, vdiv_om, ft, degree,
                scratches[1], vdiv_scale, Nc, nc;
                tol = tol, max_iter = max_iter,
                max_relative_correction = max_relative_correction)
            inc_by_level[k] = mi
            post_by_level[k] = mp
            iter_by_level[k] = ci
            relative_by_level[k] = rel
            local_relative_by_level[k] = local_rel
        end
    end
    max_inc = maximum(inc_by_level)
    max_post = maximum(post_by_level)
    if _OMEGA_TIMING[]
        _OMEGA_TIMING_STATE.solves += length(levels)
        _OMEGA_TIMING_STATE.cg_iters += sum(iter_by_level)
    end
    _sync_cs_mirrors!(am, bm, ft, Nz)
    return (max_increment = max_inc, max_post_residual = max_post,
            max_relative_correction = maximum(relative_by_level),
            max_local_relative_correction = maximum(local_relative_by_level),
            relative_correction_by_level = relative_by_level)
end

function _cs_total_air_mass(panels_m::NTuple{CS_PANEL_COUNT, <:AbstractArray})
    total = 0.0
    for p in 1:CS_PANEL_COUNT
        total += sum(Float64, panels_m[p])
    end
    return total
end

function _cs_total_area(cell_areas::AbstractMatrix)
    return CS_PANEL_COUNT * sum(Float64, cell_areas)
end

"""
    _pin_cs_global_air_mass!(panels_m, cell_areas, g, target_kg)

Apply a uniform dry-surface-pressure offset to a cubed-sphere dry-air mass
state so its global mass equals `target_kg`. The column offset is distributed
vertically in proportion to each column's existing dry layer mass, preserving
the endpoint's vertical shape while removing the nonphysical global mean.
"""
function _pin_cs_global_air_mass!(panels_m::NTuple{CS_PANEL_COUNT, <:AbstractArray{FT, 3}},
                                  cell_areas::AbstractMatrix,
                                  g::FT,
                                  target_kg::Real) where FT
    current = _cs_total_air_mass(panels_m)
    area_total = _cs_total_area(cell_areas)
    delta_kg = Float64(target_kg) - current
    delta_ps = delta_kg * Float64(g) / area_total
    Nz = size(panels_m[1], 3)

    @inbounds for p in 1:CS_PANEL_COUNT
        m = panels_m[p]
        for j in axes(m, 2), i in axes(m, 1)
            delta_col = delta_ps * Float64(cell_areas[i, j]) / Float64(g)
            col = 0.0
            for k in 1:Nz
                col += Float64(m[i, j, k])
            end
            if col > 0.0
                for k in 1:Nz
                    m[i, j, k] = FT(Float64(m[i, j, k]) + delta_col * Float64(m[i, j, k]) / col)
                end
            else
                per_layer = delta_col / Nz
                for k in 1:Nz
                    m[i, j, k] = FT(Float64(m[i, j, k]) + per_layer)
                end
            end
        end
    end

    final = _cs_total_air_mass(panels_m)
    return (before_kg = current,
            after_kg = final,
            target_kg = Float64(target_kg),
            delta_ps_pa = delta_ps,
            residual_kg = final - Float64(target_kg))
end

_validate_geos_native_panel_convention(::GEOSNativePanelConvention) = nothing
function _validate_geos_native_panel_convention(conv)
    error("GEOS-CS passthrough requires panel_convention=`geos_native` on " *
          "the target geometry; got $(typeof(conv)).")
end

_geos_next_endpoint_available(handles::GEOSDayHandles) =
    handles.next_ctm_i1 !== nothing
_geos_next_endpoint_available(handles::GEOSFPNativeDayHandles) =
    handles.next_ctm !== nothing

abstract type AbstractGEOSCSResolutionStrategy end
struct GEOSCSIdentityStrategy <: AbstractGEOSCSResolutionStrategy end
struct GEOSCSBlockCoarsenStrategy{R} <: AbstractGEOSCSResolutionStrategy end

_geos_cs_resolution_strategy(settings::AbstractGEOSSettings, grid::CubedSphereTargetGeometry) =
    _geos_cs_resolution_strategy(Val(settings.Nc), Val(grid.Nc))

_geos_cs_resolution_strategy(::Val{N}, ::Val{N}) where {N} =
    GEOSCSIdentityStrategy()

function _geos_cs_resolution_strategy(::Val{Nsrc}, ::Val{Ndst}) where {Nsrc, Ndst}
    (Nsrc > Ndst && Nsrc % Ndst == 0) ||
        throw(ArgumentError(
            "GEOS-CS conversion supports native passthrough or nested block coarsening only; " *
            "source Nc=$(Nsrc), target Nc=$(Ndst)."))
    return GEOSCSBlockCoarsenStrategy{Nsrc ÷ Ndst}()
end

_geos_cs_strategy_name(::GEOSCSIdentityStrategy) = "identity"
_geos_cs_strategy_name(::GEOSCSBlockCoarsenStrategy{R}) where {R} =
    "nested_block_coarsen_$(R)x$(R)"

function _coarsen_sum3!(dst::AbstractArray{FT, 3},
                        src::AbstractArray{FT, 3},
                        ::Val{R}) where {FT, R}
    Nc = size(dst, 1)
    Nz = size(dst, 3)
    @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
        s = zero(FT)
        for jj in ((j - 1) * R + 1):(j * R), ii in ((i - 1) * R + 1):(i * R)
            s += src[ii, jj, k]
        end
        dst[i, j, k] = s
    end
    return dst
end

function _coarsen_area_weighted2!(dst::AbstractMatrix{FT},
                                  src::AbstractMatrix{FT},
                                  src_area::AbstractMatrix{FT},
                                  ::Val{R}) where {FT, R}
    Nc = size(dst, 1)
    @inbounds for j in 1:Nc, i in 1:Nc
        num = zero(FT)
        den = zero(FT)
        for jj in ((j - 1) * R + 1):(j * R), ii in ((i - 1) * R + 1):(i * R)
            a = src_area[ii, jj]
            num += src[ii, jj] * a
            den += a
        end
        dst[i, j] = num / den
    end
    return dst
end

function _coarsen_area_weighted3!(dst::AbstractArray{FT, 3},
                                  src::AbstractArray{FT, 3},
                                  src_area::AbstractMatrix{FT},
                                  ::Val{R}) where {FT, R}
    Nc = size(dst, 1)
    Nz = size(dst, 3)
    @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
        num = zero(FT)
        den = zero(FT)
        for jj in ((j - 1) * R + 1):(j * R), ii in ((i - 1) * R + 1):(i * R)
            a = src_area[ii, jj]
            num += src[ii, jj, k] * a
            den += a
        end
        dst[i, j, k] = num / den
    end
    return dst
end

function _coarsen_xface_sum!(dst::AbstractArray{FT, 3},
                             src::AbstractArray{FT, 3},
                             ::Val{R}) where {FT, R}
    Nc = size(dst, 2)
    Nz = size(dst, 3)
    @inbounds for k in 1:Nz, j in 1:Nc, i in 1:(Nc + 1)
        fi = (i - 1) * R + 1
        s = zero(FT)
        for fj in ((j - 1) * R + 1):(j * R)
            s += src[fi, fj, k]
        end
        dst[i, j, k] = s
    end
    return dst
end

function _coarsen_yface_sum!(dst::AbstractArray{FT, 3},
                             src::AbstractArray{FT, 3},
                             ::Val{R}) where {FT, R}
    Nc = size(dst, 1)
    Nz = size(dst, 3)
    @inbounds for k in 1:Nz, j in 1:(Nc + 1), i in 1:Nc
        fj = (j - 1) * R + 1
        s = zero(FT)
        for fi in ((i - 1) * R + 1):(i * R)
            s += src[fi, fj, k]
        end
        dst[i, j, k] = s
    end
    return dst
end

function _geos_strategy_workspace(::GEOSCSIdentityStrategy,
                                  settings::AbstractGEOSSettings,
                                  grid::CubedSphereTargetGeometry,
                                  ::Type{FT}, Nz_native::Int,
                                  Nz_out::Int) where FT
    return nothing
end

function _geos_strategy_workspace(::GEOSCSBlockCoarsenStrategy{R},
                                  settings::AbstractGEOSSettings,
                                  grid::CubedSphereTargetGeometry,
                                  ::Type{FT}, Nz_native::Int,
                                  Nz_out::Int) where {FT, R}
    source_mesh = source_grid(settings; FT = FT)
    Nsrc = settings.Nc
    Nc = grid.Nc
    panels_3d_src() = ntuple(_ -> zeros(FT, Nsrc, Nsrc, Nz_native), CS_PANEL_COUNT)
    panels_xface_src() = ntuple(_ -> zeros(FT, Nsrc + 1, Nsrc, Nz_native), CS_PANEL_COUNT)
    panels_yface_src() = ntuple(_ -> zeros(FT, Nsrc, Nsrc + 1, Nz_native), CS_PANEL_COUNT)
    panels_2d_dst() = ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT)
    panels_3d_dst(nlev) = ntuple(_ -> zeros(FT, Nc, Nc, nlev), CS_PANEL_COUNT)
    return (
        source_mesh = source_mesh,
        source_cell_areas = source_mesh.cell_areas,
        fine_m_kg = panels_3d_src(),
        fine_am_v4 = panels_xface_src(),
        fine_bm_v4 = panels_yface_src(),
        cmfmc_native = settings.include_convection ? panels_3d_dst(Nz_native + 1) : nothing,
        dtrain_native = settings.include_convection ? panels_3d_dst(Nz_native) : nothing,
        surface = (settings.include_surface || settings.include_vdiff_fields) ? (
            pblh = panels_2d_dst(),
            ustar = panels_2d_dst(),
            hflux = panels_2d_dst(),
            t2m = panels_2d_dst(),
        ) : nothing,
        vdiff_native = settings.include_vdiff_fields ? (
            u = panels_3d_dst(Nz_native),
            v = panels_3d_dst(Nz_native),
            t = panels_3d_dst(Nz_native),
            qv = panels_3d_dst(Nz_native),
        ) : nothing,
        vdiff_weights_native = settings.include_vdiff_fields ?
            panels_3d_dst(Nz_native) : nothing,
    )
end

function _geos_fluxes_to_target!(::GEOSCSIdentityStrategy, _ws,
                                 am_v4, bm_v4, raw, grid,
                                 Nc::Int, Nz::Int, flux_scale)
    geos_native_to_face_flux!(am_v4, bm_v4, raw.am, raw.bm,
                              grid.mesh.connectivity, Nc, Nz, flux_scale)
    return nothing
end

function _geos_fluxes_to_target!(::GEOSCSBlockCoarsenStrategy{R}, ws,
                                 am_v4, bm_v4, raw, _grid,
                                 Nc::Int, Nz::Int, flux_scale) where R
    Nsrc = Nc * R
    geos_native_to_face_flux!(ws.fine_am_v4, ws.fine_bm_v4, raw.am, raw.bm,
                              ws.source_mesh.connectivity, Nsrc, Nz, flux_scale)
    for p in 1:CS_PANEL_COUNT
        _coarsen_xface_sum!(am_v4[p], ws.fine_am_v4[p], Val(R))
        _coarsen_yface_sum!(bm_v4[p], ws.fine_bm_v4[p], Val(R))
    end
    return nothing
end

function _geos_seed_mass!(::GEOSCSIdentityStrategy, _ws, m_cur, raw,
                          cell_areas, inv_g, Nc::Int, Nz::Int)
    for p in 1:CS_PANEL_COUNT
        _delp_pa_to_air_mass_kg!(m_cur[p], raw.m[p], cell_areas, inv_g)
    end
    return nothing
end

function _geos_seed_mass!(::GEOSCSBlockCoarsenStrategy{R}, ws, m_cur, raw,
                          _cell_areas, inv_g, _Nc::Int, _Nz::Int) where R
    for p in 1:CS_PANEL_COUNT
        _delp_pa_to_air_mass_kg!(ws.fine_m_kg[p], raw.m[p], ws.source_cell_areas, inv_g)
        _coarsen_sum3!(m_cur[p], ws.fine_m_kg[p], Val(R))
    end
    return nothing
end

function _geos_target_mass!(::GEOSCSIdentityStrategy, _ws, m_next, raw,
                            cell_areas, inv_g, Nc::Int, Nz::Int)
    for p in 1:CS_PANEL_COUNT
        _delp_pa_to_air_mass_kg!(m_next[p], raw.m_next[p], cell_areas, inv_g)
    end
    return nothing
end

function _geos_target_mass!(::GEOSCSBlockCoarsenStrategy{R}, ws, m_next, raw,
                            _cell_areas, inv_g, _Nc::Int, _Nz::Int) where R
    for p in 1:CS_PANEL_COUNT
        _delp_pa_to_air_mass_kg!(ws.fine_m_kg[p], raw.m_next[p], ws.source_cell_areas, inv_g)
        _coarsen_sum3!(m_next[p], ws.fine_m_kg[p], Val(R))
    end
    return nothing
end

_geos_surface_payload!(::GEOSCSIdentityStrategy, _ws, raw) = raw.surface

function _geos_surface_payload!(::GEOSCSBlockCoarsenStrategy{R}, ws, raw) where R
    raw.surface === nothing && return nothing
    for p in 1:CS_PANEL_COUNT
        _coarsen_area_weighted2!(ws.surface.pblh[p], raw.surface.pblh[p], ws.source_cell_areas, Val(R))
        _coarsen_area_weighted2!(ws.surface.ustar[p], raw.surface.ustar[p], ws.source_cell_areas, Val(R))
        _coarsen_area_weighted2!(ws.surface.hflux[p], raw.surface.hflux[p], ws.source_cell_areas, Val(R))
        _coarsen_area_weighted2!(ws.surface.t2m[p], raw.surface.t2m[p], ws.source_cell_areas, Val(R))
    end
    return ws.surface
end

_geos_vdiff_native_target!(::GEOSCSIdentityStrategy, _ws, raw) = raw.vdiff
_geos_vdiff_native_weights!(::GEOSCSIdentityStrategy, _ws, raw,
                            _cell_areas, _inv_g) = raw.m

function _geos_vdiff_native_target!(::GEOSCSBlockCoarsenStrategy{R}, ws, raw) where R
    raw.vdiff === nothing && return nothing
    for p in 1:CS_PANEL_COUNT
        _coarsen_area_weighted3!(ws.vdiff_native.u[p], raw.vdiff.u[p],
                                 ws.source_cell_areas, Val(R))
        _coarsen_area_weighted3!(ws.vdiff_native.v[p], raw.vdiff.v[p],
                                 ws.source_cell_areas, Val(R))
        _coarsen_area_weighted3!(ws.vdiff_native.t[p], raw.vdiff.t[p],
                                 ws.source_cell_areas, Val(R))
        _coarsen_area_weighted3!(ws.vdiff_native.qv[p], raw.vdiff.qv[p],
                                 ws.source_cell_areas, Val(R))
    end
    return ws.vdiff_native
end

function _geos_vdiff_native_weights!(::GEOSCSBlockCoarsenStrategy{R}, ws, raw,
                                     _cell_areas, inv_g) where R
    for p in 1:CS_PANEL_COUNT
        _delp_pa_to_air_mass_kg!(ws.fine_m_kg[p], raw.m[p],
                                  ws.source_cell_areas, inv_g)
        _coarsen_sum3!(ws.vdiff_weights_native[p], ws.fine_m_kg[p], Val(R))
    end
    return ws.vdiff_weights_native
end

_geos_cmfmc_native_target!(::GEOSCSIdentityStrategy, _ws, raw) = raw.cmfmc
_geos_dtrain_native_target!(::GEOSCSIdentityStrategy, _ws, raw) = raw.dtrain

function _geos_cmfmc_native_target!(::GEOSCSBlockCoarsenStrategy{R}, ws, raw) where R
    raw.cmfmc === nothing && return nothing
    for p in 1:CS_PANEL_COUNT
        _coarsen_area_weighted3!(ws.cmfmc_native[p], raw.cmfmc[p],
                                 ws.source_cell_areas, Val(R))
    end
    return ws.cmfmc_native
end

function _geos_dtrain_native_target!(::GEOSCSBlockCoarsenStrategy{R}, ws, raw) where R
    raw.dtrain === nothing && return nothing
    for p in 1:CS_PANEL_COUNT
        _coarsen_area_weighted3!(ws.dtrain_native[p], raw.dtrain[p],
                                 ws.source_cell_areas, Val(R))
    end
    return ws.dtrain_native
end

function _geos_cmfmc_payload!(workspace)
    workspace.cmfmc_v4 === nothing && return nothing
    native = _geos_cmfmc_native_target!(workspace.strategy, workspace.strategy_ws,
                                        workspace.raw)
    native === nothing && return nothing
    for p in 1:CS_PANEL_COUNT
        apply_vertical!(workspace.cmfmc_v4[p], native[p], workspace.plan,
                        ConvectionInterfaceFlux())
    end
    return workspace.cmfmc_v4
end

function _geos_dtrain_payload!(workspace)
    workspace.dtrain_v4 === nothing && return nothing
    native = _geos_dtrain_native_target!(workspace.strategy, workspace.strategy_ws,
                                         workspace.raw)
    native === nothing && return nothing
    for p in 1:CS_PANEL_COUNT
        apply_vertical!(workspace.dtrain_v4[p], native[p], workspace.plan,
                        ConvectionTendencyField())
    end
    return workspace.dtrain_v4
end

function _geos_vdiff_payload!(workspace)
    workspace.vdiff_v4 === nothing && return nothing
    native = _geos_vdiff_native_target!(workspace.strategy, workspace.strategy_ws,
                                        workspace.raw)
    native === nothing && return nothing
    weights = _geos_vdiff_native_weights!(workspace.strategy, workspace.strategy_ws,
                                          workspace.raw, workspace.cell_areas,
                                          workspace.inv_g)
    for p in 1:CS_PANEL_COUNT
        apply_vertical!(workspace.vdiff_v4.u[p], native.u[p], workspace.plan,
                        IntensiveCenterField(), weights[p])
        apply_vertical!(workspace.vdiff_v4.v[p], native.v[p], workspace.plan,
                        IntensiveCenterField(), weights[p])
        apply_vertical!(workspace.vdiff_v4.t[p], native.t[p], workspace.plan,
                        IntensiveCenterField(), weights[p])
        apply_vertical!(workspace.vdiff_v4.qv[p], native.qv[p], workspace.plan,
                        IntensiveCenterField(), weights[p])
    end
    return workspace.vdiff_v4
end

mutable struct GEOSCubedSphereWindowWorkspace{FT, ST, SW, RAW, CA, VP, CV, DV, VD, VO, OR} <:
               AbstractWindowWorkspace{CubedSphereTargetGeometry, FT}
    strategy    :: ST
    strategy_ws :: SW
    raw         :: RAW
    plan        :: VP
    am_native_v4 :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    bm_native_v4 :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    m_native_kg  :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    am_v4       :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    bm_v4       :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    cm_v4       :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    dm_v4       :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    m_cur       :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    m_next_target :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    ps_cur      :: NTuple{CS_PANEL_COUNT, Array{FT, 2}}
    cmfmc_v4    :: CV
    dtrain_v4   :: DV
    vdiff_v4    :: VD
    g           :: FT
    inv_g       :: FT
    cell_areas  :: CA
    base_flux_scale :: FT
    flux_scale  :: FT
    source_steps_per_met :: Int
    steps_current :: Int
    steps_schedule :: Vector{Int}
    adaptive_substeps :: Bool
    substep_cfl_target :: Float64
    min_steps_per_window :: Int
    max_steps_per_window :: Int
    chain_mass  :: Bool
    global_mass_pin_enabled :: Bool
    global_mass_target_kg :: Float64
    balance_mode :: Symbol
    # Vertical-flux (cm) closure: `:endpoint_balanced` (default) closes cm from
    # the column-balanced horizontal fluxes against the raw GEOS DELP_dry
    # endpoint tendency (`diagnose_cs_cm!`); `:pressure_fixer` keeps the native
    # horizontal fluxes UNBALANCED and closes cm by construction via the FV3
    # ΔB-distributed rule (`compute_cs_cm_pressure_fixer!`), chaining the mass
    # `m_next = m_cur + 2·steps·ΔB·pit` (avoids dumping the column moisture-source
    # term into cm — the SH-UTLS "fingering"). See module header + commit e648bf3f.
    cm_closure :: Symbol
    ΔB :: Vector{FT}          # B[k+1]-B[k], TOA-first, length Nz, Σ ΔB = 1
    # Raw (pinned) GEOS DELP_dry endpoint, preserved across the adaptive
    # refinement loop. `:moisture_filtered` balances + diagnoses against THIS
    # (the faithful analyzed endpoint) while `m_next_target` holds the filtered
    # endpoint the replay gate checks. For `:endpoint_balanced`/`:pressure_fixer`
    # it is unused (they read `m_next_target` directly).
    m_next_delp :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    # Horizontal Jacobi sweeps applied to the moisture-source residual in the
    # `:moisture_filtered` closure (0 ⇒ equivalent to `:endpoint_balanced`, up to
    # the dm = (m_next−m_cur)/(2·steps) round-trip in F32).
    smooth_iters :: Int
    # OMEGA closure scratch (else `nothing`): OMEGA/QV PCHIP read
    # buffers + the smooth OMEGA-derived per-layer vertical-convergence target
    # vdiv_om (downward-positive, Σ_k=0). Populated per window in `ingest_window!`.
    omega_buf :: VO
    qv_buf    :: VO
    vdiv_om   :: VO
    vdiv_target :: VO
    omega_regularization :: OmegaRegularization
    omega_regularization_scratch :: OR
end

const _GEOS_ADAPTIVE_SUBSTEP_MAX_REFINEMENTS = 8

function allocate_window_workspace(grid::CubedSphereTargetGeometry,
                                   settings::AbstractGEOSSettings,
                                   vertical,
                                   ::Type{FT};
                                   dt_met_seconds::Real,
                                   chain_mass::Bool = true,
                                   cache = nothing,
                                   adaptive_substeps::Bool = false,
                                   substep_cfl_target::Real = 0.95,
                                   min_steps_per_window::Integer = 1,
                                   max_steps_per_window::Integer = typemax(Int),
                                   windows_per_day::Integer = 0,
                                   global_mass_pin::Bool = false,
                                   global_mass_target_kg::Real = NaN,
                                   balance_mode::Symbol = :column,
                                   cm_closure::Symbol = :endpoint_balanced,
                                   smooth_iters::Integer = 8,
                                   omega_regularization::OmegaRegularization = OmegaRegularization()) where FT
    Nc = grid.Nc
    Nz = vertical.Nz
    Nz_native = vertical.Nz_native
    plan = if hasproperty(vertical, :plan)
        vertical.plan
    else
        Nz == Nz_native ||
            error("GEOS vertical setup with Nz=$(Nz), Nz_native=$(Nz_native) must carry a `plan` field.")
        vc_identity = HybridSigmaPressure(FT.(vertical.merged_vc.A),
                                          FT.(vertical.merged_vc.B))
        plan_vertical(IdentityVertical(), vc_identity)
    end
    strategy = _geos_cs_resolution_strategy(settings, grid)
    strategy_ws = _geos_strategy_workspace(strategy, settings, grid, FT,
                                           Nz_native, Nz)
    npanel = CS_PANEL_COUNT

    g = FT(STANDARD_GRAVITY)
    inv_g = inv(g)
    cell_areas = grid.mesh.cell_areas
    steps_per_met = round(Int, FT(dt_met_seconds) / FT(settings.mass_flux_dt))
    dt_factor = FT(settings.mass_flux_dt / 2)
    flux_scale = dt_factor / g
    target = Float64(substep_cfl_target)
    isfinite(target) && target > 0 ||
        error("substep_cfl_target must be finite and > 0; got $(substep_cfl_target)")
    min_steps = Int(min_steps_per_window)
    max_steps = Int(max_steps_per_window)
    1 <= min_steps <= max_steps ||
        error("invalid adaptive substep bounds: min=$(min_steps), max=$(max_steps)")
    schedule_len = Int(windows_per_day)
    schedule_len >= 0 ||
        error("windows_per_day must be non-negative, got $(windows_per_day)")
    balance_mode in (:column, :per_layer) ||
        error("GEOS-CS balance_mode must be :column or :per_layer; got $(balance_mode)")
    cm_closure in (:endpoint_balanced, :pressure_fixer, :moisture_filtered,
                   :pfix_corrected, :omega_full_replacement, :omega_regularized) ||
        error("GEOS-CS cm_closure must be :endpoint_balanced, :pressure_fixer, " *
              ":moisture_filtered, :pfix_corrected, :omega_full_replacement, " *
              "or :omega_regularized; got $(cm_closure)")
    if _uses_omega(cm_closure)
        global_mass_pin ||
            error("GEOS-CS OMEGA-based cm closure requires global_mass_pin=true " *
                  "to remove the unrealizable global column-mass mode")
        settings.include_vdiff_fields ||
            error("GEOS-CS OMEGA-based cm closure needs A3dyn OMEGA + I3 QV; " *
                  "set [source].include_vdiff_fields=true")
        grid.Nc == settings.Nc ||
            error("GEOS-CS OMEGA-based cm closure requires the native " *
                  "passthrough (target Nc == source Nc) only; got target Nc=$(grid.Nc), " *
                  "source Nc=$(settings.Nc).")
        Nz == Nz_native ||
            error("GEOS-CS OMEGA-based cm closure requires the identity " *
                  "vertical transform (Nz == Nz_native) only; got Nz=$(Nz), " *
                  "Nz_native=$(Nz_native). Use [vertical].transform=\"identity\" " *
                  "(the validated full-L72 build).")
    end
    cm_closure === :omega_regularized &&
        _validate_omega_regularization(omega_regularization)
    # ΔB[k] = B_interface[k+1] − B_interface[k] (TOA-first; Σ ΔB = 1 by hybrid
    # sigma-pressure construction). The merged_vc is the target coordinate (same
    # source the identity plan above is built from). Used by :pressure_fixer cm.
    Bifc = vertical.merged_vc.B
    length(Bifc) == Nz + 1 ||
        error("GEOS-CS ΔB needs $(Nz+1) interface B coefficients, got $(length(Bifc))")
    ΔB = FT[FT(Bifc[k + 1] - Bifc[k]) for k in 1:Nz]

    am_native_v4 = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz_native), npanel)
    bm_native_v4 = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz_native), npanel)
    m_native_kg = ntuple(_ -> zeros(FT, Nc, Nc, Nz_native), npanel)
    am_v4 = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), npanel)
    bm_v4 = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), npanel)
    cm_v4 = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), npanel)
    dm_v4 = ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel)
    m_cur = ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel)
    m_next_target = ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel)
    m_next_delp = ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel)
    ps_cur = ntuple(_ -> zeros(FT, Nc, Nc), npanel)
    cmfmc_v4 = settings.include_convection ?
        ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), npanel) : nothing
    dtrain_v4 = settings.include_convection ?
        ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel) : nothing
    vdiff_v4 = settings.include_vdiff_fields ? (
        u  = ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel),
        v  = ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel),
        t  = ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel),
        qv = ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel),
    ) : nothing
    # OMEGA-consistent closure scratch: OMEGA/QV read buffers + the smooth
    # vertical-convergence target. Identity passthrough (Nc==settings.Nc,
    # Nz==Nz_native) is enforced above, so all three are target-shaped.
    omega_buf = _uses_omega(cm_closure) ?
        ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel) : nothing
    qv_buf = _uses_omega(cm_closure) ?
        ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel) : nothing
    vdiv_om = _uses_omega(cm_closure) ?
        ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel) : nothing
    vdiv_target = _uses_omega(cm_closure) ?
        ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel) : nothing
    omega_regularization_scratch = if cm_closure === :omega_regularized
        nc = npanel * Nc * Nc
        OmegaRegularizationScratch(
            ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), npanel),
            zeros(Float64, nc), zeros(Float64, nc), zeros(Float64, nc),
            zeros(Float64, nc), falses(Nz))
    else
        nothing
    end
    raw = allocate_raw_window(settings; FT = FT, Nz = Nz_native)

    return GEOSCubedSphereWindowWorkspace{
        FT, typeof(strategy), typeof(strategy_ws), typeof(raw), typeof(cell_areas),
        typeof(plan), typeof(cmfmc_v4), typeof(dtrain_v4), typeof(vdiff_v4),
        typeof(vdiv_om), typeof(omega_regularization_scratch)}(
            strategy, strategy_ws, raw, plan,
            am_native_v4, bm_native_v4, m_native_kg,
            am_v4, bm_v4, cm_v4, dm_v4,
            m_cur, m_next_target, ps_cur, cmfmc_v4, dtrain_v4, vdiff_v4,
            g, inv_g, cell_areas,
            flux_scale, flux_scale, steps_per_met, steps_per_met,
            fill(steps_per_met, schedule_len), Bool(adaptive_substeps),
            target, min_steps, max_steps, chain_mass,
            Bool(global_mass_pin), Float64(global_mass_target_kg),
            balance_mode, cm_closure, ΔB, m_next_delp, Int(smooth_iters),
            omega_buf, qv_buf, vdiv_om, vdiv_target,
            omega_regularization, omega_regularization_scratch)
end

function _geos_pin_global_mass_if_needed!(workspace::GEOSCubedSphereWindowWorkspace{FT},
                                          panels_m::NTuple{CS_PANEL_COUNT, <:AbstractArray{FT, 3}},
                                          label::AbstractString) where FT
    workspace.global_mass_pin_enabled || return nothing
    if !isfinite(workspace.global_mass_target_kg)
        workspace.global_mass_target_kg = _cs_total_air_mass(panels_m)
        @info @sprintf("  GEOS global dry-mass pin target initialized from %s: %.9e kg",
                       label, workspace.global_mass_target_kg)
        return (before_kg = workspace.global_mass_target_kg,
                after_kg = workspace.global_mass_target_kg,
                target_kg = workspace.global_mass_target_kg,
                delta_ps_pa = 0.0,
                residual_kg = 0.0)
    end
    stats = _pin_cs_global_air_mass!(panels_m, workspace.cell_areas,
                                     workspace.g, workspace.global_mass_target_kg)
    abs(stats.delta_ps_pa) > 1e-10 &&
        @debug @sprintf("  GEOS global dry-mass pin %s: Δps=%+.6e Pa residual=%+.6e kg",
                        label, stats.delta_ps_pa, stats.residual_kg)
    return stats
end

function _scale_cs_flux_panels!(panels, factor)
    for p in 1:CS_PANEL_COUNT
        panels[p] .*= factor
    end
    return panels
end

function _geos_prepare_window_for_steps!(workspace::GEOSCubedSphereWindowWorkspace{FT},
                                         grid::CubedSphereTargetGeometry,
                                         steps::Int) where FT
    Nc = grid.Nc
    Nz = size(workspace.m_cur[1], 3)
    for p in 1:CS_PANEL_COUNT
        apply_vertical!(workspace.am_v4[p], workspace.am_native_v4[p],
                        workspace.plan, MassFluxField())
        apply_vertical!(workspace.bm_v4[p], workspace.bm_native_v4[p],
                        workspace.plan, MassFluxField())
    end
    if steps != workspace.source_steps_per_met
        factor = FT(workspace.source_steps_per_met / steps)
        _scale_cs_flux_panels!(workspace.am_v4, factor)
        _scale_cs_flux_panels!(workspace.bm_v4, factor)
    end
    workspace.flux_scale = workspace.base_flux_scale *
                           (workspace.source_steps_per_met / steps)

    for p in 1:CS_PANEL_COUNT
        fill!(workspace.cm_v4[p], zero(FT))
    end

    if workspace.cm_closure === :pressure_fixer
        # Keep native horizontal fluxes UNBALANCED; close cm by construction via
        # the FV3 ΔB-distributed rule, and chain the implied endpoint mass
        # `m_next = m_cur + 2·steps·ΔB·pit` (replay closes to roundoff). This
        # avoids dumping the column moisture-source term into cm. The raw-DELP
        # `m_next_target` set during ingest is replaced here by the pressure-
        # fixer endpoint so the chained mass / dm / ps stay self-consistent.
        compute_cs_cm_pressure_fixer!(workspace.cm_v4, workspace.am_v4,
                                      workspace.bm_v4, workspace.ΔB, Nc, Nz)
        _evolve_mass_pressure_fixer!(workspace.m_next_target, workspace.m_cur,
                                     workspace.am_v4, workspace.bm_v4,
                                     workspace.ΔB, FT(2 * steps), Nc, Nz)
        fill_cs_window_mass_tendency!(workspace.dm_v4, workspace.m_cur,
                                      workspace.m_next_target, steps)
        return (max_pre_residual = 0.0, max_post_residual = 0.0,
                max_cg_iter = 0, mode = :pressure_fixer)
    end

    if workspace.cm_closure === :pfix_corrected
        # "Spatially-resolved dry-mass correction" (generalizes the scalar global
        # mass pin). Native fluxes (smooth pit) → pressure-fixer cm; correct the
        # column drift toward the analyzed dry-PS with a ZERO-SUM SPATIAL LOW-PASS
        # — only the LARGE-SCALE drift (which prevents the pressure-fixer blow-up
        # and carries the global-mass target), leaving the grid-scale residual as
        # a small bounded mass perturbation NOT advected into cm. cm stays at the
        # smooth pressure-fixer floor.  `pit_eff = pit_native + δ_smooth/(2·steps)`;
        # `cm[k+1]=cm[k]+conv−ΔB·pit_eff` and `m_next=m_cur+2·steps·ΔB·pit_eff` ⇒
        # the replay gate closes by construction; global Σ(ΔB·δ_smooth)=Σδ lands
        # the global dry mass on the analyzed (pinned) target.
        # DIAGNOSTIC-ONLY LIMITATION: when δ_smooth ≠ 0 this leaves a nonzero
        # SURFACE flux cm[Nz+1] = −δ_smooth/(2·steps) (the closed-bottom boundary
        # is violated), and chain_mass=true accumulates negative UTLS mass. The
        # tracer analysis summarized in docs/src/preprocessing/geos_native_cs.md showed it
        # makes ~164–280 hPa WORSE. NOT a production closure.
        two_steps = FT(2 * steps)
        pit = ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT)
        δ   = ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT)
        @inbounds for p in 1:CS_PANEL_COUNT
            am = workspace.am_v4[p]; bm = workspace.bm_v4[p]
            mc = workspace.m_cur[p]; md = workspace.m_next_delp[p]
            for j in 1:Nc, i in 1:Nc
                pp = zero(FT); cur = zero(FT); ana = zero(FT)
                for k in 1:Nz
                    pp  += (am[i, j, k] - am[i + 1, j, k]) +
                           (bm[i, j, k] - bm[i, j + 1, k])
                    cur += mc[i, j, k]; ana += md[i, j, k]
                end
                pit[p][i, j] = pp
                δ[p][i, j]   = ana - (cur + two_steps * pp)   # column drift to analyzed
            end
        end
        _smooth_cs_columns!(δ, workspace.smooth_iters, FT(0.5), Nc)   # large-scale, Σ-preserving
        @inbounds for p in 1:CS_PANEL_COUNT
            am = workspace.am_v4[p]; bm = workspace.bm_v4[p]; cm = workspace.cm_v4[p]
            mc = workspace.m_cur[p]; mt = workspace.m_next_target[p]
            for j in 1:Nc, i in 1:Nc
                pe = pit[p][i, j] + δ[p][i, j] / two_steps
                cm[i, j, 1] = zero(FT); acc = zero(FT)
                for k in 1:Nz
                    conv = (am[i, j, k] - am[i + 1, j, k]) +
                           (bm[i, j, k] - bm[i, j + 1, k])
                    acc += conv - FT(workspace.ΔB[k]) * pe
                    cm[i, j, k + 1] = acc
                    mt[i, j, k] = mc[i, j, k] + two_steps * FT(workspace.ΔB[k]) * pe
                end
            end
        end
        fill_cs_window_mass_tendency!(workspace.dm_v4, workspace.m_cur,
                                      workspace.m_next_target, steps)
        return (max_pre_residual = 0.0, max_post_residual = 0.0,
                max_cg_iter = 0, mode = :pfix_corrected)
    end

    if workspace.cm_closure === :moisture_filtered
        # Balance native fluxes to the raw GEOS DELP_dry endpoint (faithful winds;
        # column closes so per-column `pit = Σ_k dm_dry`). Then split the endpoint
        # tendency dm_dry = ΔB·pit (smooth, ps-driven hybrid expansion) + residual
        # (the column moisture-source term that carries the SH-UTLS grid noise),
        # smooth ONLY the residual horizontally, and recombine. The residual has
        # zero column integral, and the per-level-identical linear smoother
        # preserves that ⇒ surface pressure is conserved per column EXACTLY; only
        # the grid-scale per-layer mass redistribution (the fingering) is damped.
        # `m_next_target` is set to the filtered endpoint so the replay gate
        # (m_evolved = m_cur − 2·steps·(div_h + Δcm)) closes by construction.
        #
        # The zero-column-integral property REQUIRES the column balance: it pins
        # `Σ_k div_h == Σ_k dm_dry` per column, so the residual `dm_dry − ΔB·pit`
        # integrates to exactly zero and the per-level smoother conserves PS per
        # column. The per-layer balance only closes the column up to a global
        # constant, which would let the smoother shift column dry mass — so this
        # closure forces the column balance regardless of `balance_mode`.
        bal_diag = balance_cs_column_mass_fluxes!(
            workspace.am_v4, workspace.bm_v4, workspace.m_cur,
            workspace.m_next_delp, grid.face_table, grid.cell_degree, steps,
            grid.poisson_scratch)
        # dm_dry = (m_next_delp − m_cur) / (2·steps)
        fill_cs_window_mass_tendency!(workspace.dm_v4, workspace.m_cur,
                                      workspace.m_next_delp, steps)
        # residual ← dm_dry − ΔB·pit (per-half-step convergence pit from balanced
        # fluxes; identical convention to `_evolve_mass_pressure_fixer!`).
        @inbounds for p in 1:CS_PANEL_COUNT
            am = workspace.am_v4[p]; bm = workspace.bm_v4[p]; dm = workspace.dm_v4[p]
            for j in 1:Nc, i in 1:Nc
                pit = zero(FT)
                for k in 1:Nz
                    pit += (am[i, j, k] - am[i + 1, j, k]) +
                           (bm[i, j, k] - bm[i, j + 1, k])
                end
                for k in 1:Nz
                    dm[i, j, k] -= FT(workspace.ΔB[k]) * pit
                end
            end
        end
        _smooth_cs_residual_panels!(workspace.dm_v4, workspace.smooth_iters,
                                    FT(0.5), Nc, Nz)
        # dmc ← ΔB·pit + smooth(residual); filtered endpoint m_next_target.
        two_steps = FT(2 * steps)
        @inbounds for p in 1:CS_PANEL_COUNT
            am = workspace.am_v4[p]; bm = workspace.bm_v4[p]
            dm = workspace.dm_v4[p]; mc = workspace.m_cur[p]
            mt = workspace.m_next_target[p]
            for j in 1:Nc, i in 1:Nc
                pit = zero(FT)
                for k in 1:Nz
                    pit += (am[i, j, k] - am[i + 1, j, k]) +
                           (bm[i, j, k] - bm[i, j + 1, k])
                end
                for k in 1:Nz
                    dm[i, j, k] += FT(workspace.ΔB[k]) * pit
                    mt[i, j, k] = mc[i, j, k] + two_steps * dm[i, j, k]
                end
            end
        end
        diagnose_cs_cm!(workspace.cm_v4, workspace.am_v4, workspace.bm_v4,
                        workspace.dm_v4, workspace.m_cur, Nc, Nz)
        return bal_diag
    end

    if _uses_omega(workspace.cm_closure)
        # Both OMEGA modes first column-balance the native horizontal fluxes to
        # the analyzed dry endpoint. `:omega_full_replacement` then replaces the full
        # per-layer convergence target. `:omega_regularized` keeps that diagnosed
        # native target at resolved scales and outside the UTLS, adding only the
        # pressure-tapered high-pass OMEGA discrepancy with a per-level flux cap.
        # The final cm is always diagnosed from the realized horizontal fluxes,
        # so continuity remains exact even when the regularized target is capped.
        bal_diag = balance_cs_column_mass_fluxes!(
            workspace.am_v4, workspace.bm_v4, workspace.m_cur,
            workspace.m_next_target, grid.face_table, grid.cell_degree, steps,
            grid.poisson_scratch)
        fill_cs_window_mass_tendency!(workspace.dm_v4, workspace.m_cur,
                                      workspace.m_next_target, steps)
        global_dm = sum(sum(Float64, panel) for panel in workspace.dm_v4)
        global_mass = sum(sum(Float64, panel) for panel in workspace.m_cur)
        global_tendency_rel = abs(2 * steps * global_dm) / global_mass
        global_tendency_rel <= replay_tolerance(FT) ||
            error("OMEGA reconstruction requires a globally closed endpoint mass " *
                  "tendency; relative residual $(global_tendency_rel) exceeds " *
                  "replay tolerance $(replay_tolerance(FT))")
        # vdiv_om was built at base tau=mass_flux_dt/2 (i.e. steps=source_steps_per_met).
        # Pass the per-substep scale (= source_steps_per_met/steps, same as the flux
        # rescale) so dm and vdiv share units, WITHOUT mutating the stored array
        # (the adaptive loop may re-prepare at another `steps`).
        vdiv_scale = workspace.source_steps_per_met / steps
        _OMEGA_TIMING[] && (_OMEGA_TIMING_STATE.prepares += 1)
        _t_recon = _OMEGA_TIMING[] ? time() : 0.0
        target = if workspace.cm_closure === :omega_regularized
            # Preserve endpoint-balanced transport as the resolved-scale reference.
            # OMEGA only supplies a capped, UTLS-local grid-scale correction.
            diagnose_cs_cm!(workspace.cm_v4, workspace.am_v4, workspace.bm_v4,
                            workspace.dm_v4, workspace.m_cur, Nc, Nz)
            _regularize_omega_target!(
                workspace.vdiv_target, workspace.cm_v4, workspace.vdiv_om,
                workspace.m_cur, workspace.m_next_target, grid, workspace.g,
                Float64(vdiv_scale),
                workspace.omega_regularization,
                workspace.omega_regularization_scratch)
        else
            workspace.vdiv_om
        end
        target_scale = workspace.cm_closure === :omega_regularized ? 1.0 : Float64(vdiv_scale)
        correction_cap = workspace.cm_closure === :omega_regularized ?
            workspace.omega_regularization.max_relative_flux_correction : Inf
        recon = _reconstruct_omega_target!(workspace.am_v4, workspace.bm_v4,
                                               workspace.dm_v4, target,
                                               grid, target_scale;
                                               max_relative_correction = correction_cap,
                                               active_levels =
                                                   workspace.cm_closure === :omega_regularized ?
                                                   workspace.omega_regularization_scratch.active_levels :
                                                   nothing)
        bottom_max = maximum(@view recon.relative_correction_by_level[(Nz - 2):Nz])
        if workspace.cm_closure === :omega_regularized &&
           bottom_max > workspace.omega_regularization.max_bottom_flux_correction
            error("OMEGA regularization altered a bottom-three-layer horizontal " *
                  "flux by $(bottom_max) RMS, exceeding the configured fidelity " *
                  "gate $(workspace.omega_regularization.max_bottom_flux_correction)")
        end
        _OMEGA_TIMING[] && (_OMEGA_TIMING_STATE.recon_time += time() - _t_recon)
        diagnose_cs_cm!(workspace.cm_v4, workspace.am_v4, workspace.bm_v4,
                        workspace.dm_v4, workspace.m_cur, Nc, Nz)
        return (bal_diag..., omega_max_increment = recon.max_increment,
                omega_max_post_residual = recon.max_post_residual,
                omega_max_relative_correction = recon.max_relative_correction,
                omega_max_local_relative_correction =
                    recon.max_local_relative_correction,
                omega_max_bottom_relative_correction = bottom_max,
                omega_global_mass_tendency_rel = global_tendency_rel,
                mode = workspace.cm_closure)
    end

    bal_diag = if workspace.balance_mode === :per_layer
        balance_cs_global_mass_fluxes!(
            workspace.am_v4, workspace.bm_v4, workspace.m_cur,
            workspace.m_next_target, grid.face_table, grid.cell_degree, steps,
            grid.poisson_scratch)
    else
        balance_cs_column_mass_fluxes!(
            workspace.am_v4, workspace.bm_v4, workspace.m_cur,
            workspace.m_next_target, grid.face_table, grid.cell_degree, steps,
            grid.poisson_scratch)
    end
    fill_cs_window_mass_tendency!(workspace.dm_v4, workspace.m_cur,
                                  workspace.m_next_target, steps)
    diagnose_cs_cm!(workspace.cm_v4, workspace.am_v4, workspace.bm_v4,
                    workspace.dm_v4, workspace.m_cur, Nc, Nz)
    return bal_diag
end

function _geos_select_steps_for_window!(workspace::GEOSCubedSphereWindowWorkspace,
                                        grid::CubedSphereTargetGeometry,
                                        win::Int)
    policy = SubstepSchedulePolicy(
        adaptive_substeps = workspace.adaptive_substeps,
        substep_cfl_target = workspace.substep_cfl_target,
        min_steps_per_window = workspace.min_steps_per_window,
        max_steps_per_window = workspace.max_steps_per_window)
    steps = initial_substeps(policy, workspace.source_steps_per_met)
    bal_diag = nothing
    positivity = nothing
    prepared_steps = 0
    if workspace.adaptive_substeps
        for _ in 1:_GEOS_ADAPTIVE_SUBSTEP_MAX_REFINEMENTS
            bal_diag = _geos_prepare_window_for_steps!(workspace, grid, steps)
            prepared_steps = steps
            positivity = verify_substep_positivity_cs!(
                workspace.m_cur, workspace.am_v4, workspace.bm_v4,
                workspace.cm_v4; cfl_limit = workspace.substep_cfl_target,
                m_next = workspace.m_next_target)
            next_steps = next_substeps(policy, steps, positivity.ratio)
            next_steps == steps && break
            steps = next_steps
        end
        if prepared_steps != steps
            bal_diag = _geos_prepare_window_for_steps!(workspace, grid, steps)
        end
    else
        bal_diag = _geos_prepare_window_for_steps!(workspace, grid, steps)
    end
    workspace.steps_current = steps
    1 <= win <= length(workspace.steps_schedule) ||
        throw(ArgumentError("GEOS steps_schedule length $(length(workspace.steps_schedule)) " *
                            "cannot record window $(win)."))
    workspace.steps_schedule[win] = steps
    if _OMEGA_TIMING[] && _uses_omega(workspace.cm_closure)
        s = _OMEGA_TIMING_STATE
        @info @sprintf("  [OMEGA_TIMING] win %2d steps=%-4d prepares=%d solves=%d cg_iters=%d recon=%.3fs (%.4fs/window)",
                       win, steps, s.prepares, s.solves, s.cg_iters, s.recon_time, s.recon_time)
        _reset_omega_timing!()
    end
    return (steps = steps, balance = bal_diag, positivity = positivity)
end

function ingest_window!(workspace::GEOSCubedSphereWindowWorkspace{FT},
                        reader::GEOSNativeReader{FT},
                        win::Int,
                        grid::CubedSphereTargetGeometry,
                        settings::AbstractGEOSSettings,
                        vertical) where FT
    Nc = grid.Nc
    Nz = vertical.Nz
    Nz_native = vertical.Nz_native
    read_window!(workspace.raw, reader, win)
    workspace.flux_scale = workspace.base_flux_scale
    workspace.steps_current = workspace.source_steps_per_met

    _geos_fluxes_to_target!(workspace.strategy, workspace.strategy_ws,
                            workspace.am_native_v4, workspace.bm_native_v4,
                            workspace.raw, grid, Nc, Nz_native,
                            workspace.flux_scale)
    for p in 1:CS_PANEL_COUNT
        apply_vertical!(workspace.am_v4[p], workspace.am_native_v4[p],
                        workspace.plan, MassFluxField())
        apply_vertical!(workspace.bm_v4[p], workspace.bm_native_v4[p],
                        workspace.plan, MassFluxField())
    end

    if win == 1 || !workspace.chain_mass
        if workspace.chain_mass && win == 1 && reader.seed !== nothing
            for p in 1:CS_PANEL_COUNT
                size(reader.seed[p]) == (Nc, Nc, Nz) ||
                    error("seed_m[$p] shape $(size(reader.seed[p])) ≠ ($Nc, $Nc, $Nz)")
                copyto!(workspace.m_cur[p], reader.seed[p])
            end
        else
            _geos_seed_mass!(workspace.strategy, workspace.strategy_ws,
                             workspace.m_native_kg, workspace.raw,
                             workspace.cell_areas, workspace.inv_g, Nc, Nz_native)
            for p in 1:CS_PANEL_COUNT
                apply_vertical!(workspace.m_cur[p], workspace.m_native_kg[p],
                                workspace.plan, MassField())
            end
        end
        _geos_pin_global_mass_if_needed!(workspace, workspace.m_cur,
                                         "window $(win) start")
        for p in 1:CS_PANEL_COUNT
            _ps_from_air_mass!(workspace.ps_cur[p], workspace.m_cur[p],
                               workspace.cell_areas, workspace.g, Nc, Nz)
        end
    end

    _geos_target_mass!(workspace.strategy, workspace.strategy_ws,
                       workspace.m_native_kg, workspace.raw,
                       workspace.cell_areas, workspace.inv_g, Nc, Nz_native)
    for p in 1:CS_PANEL_COUNT
        apply_vertical!(workspace.m_next_target[p], workspace.m_native_kg[p],
                        workspace.plan, MassField())
    end
    _geos_pin_global_mass_if_needed!(workspace, workspace.m_next_target,
                                     "window $(win) endpoint")
    # Preserve the pinned raw GEOS DELP_dry endpoint; `:moisture_filtered`
    # balances + diagnoses against it while overwriting `m_next_target` with the
    # filtered endpoint inside the adaptive loop.
    for p in 1:CS_PANEL_COUNT
        copyto!(workspace.m_next_delp[p], workspace.m_next_target[p])
    end
    # OMEGA-based closures read A3dyn OMEGA + I3 QV (PCHIP time-interp to this
    # window's valid time) and build the smooth vdiv_om target at the BASE flux
    # scaling (tau = mass_flux_dt/2). `_geos_prepare_window_for_steps!` rescales
    # it by source_steps_per_met/steps to match the per-substep flux scaling.
    if _uses_omega(workspace.cm_closure)
        _read_geos_omega_qv_pchip!(workspace.omega_buf, workspace.qv_buf,
                                   reader.handles, win, Nc, Nz)
        tau_base = FT(settings.mass_flux_dt / 2)
        _omega_vdiv_target!(workspace.vdiv_om, workspace.omega_buf, workspace.qv_buf,
                            workspace.cell_areas, workspace.g, tau_base, Nc, Nz)
    end
    _geos_select_steps_for_window!(workspace, grid, win)
    return nothing
end

function drain_ready_windows!(workspace::GEOSCubedSphereWindowWorkspace{FT},
                              contract::CubedSphereContract{FT},
                              win::Int,
                              grid::CubedSphereTargetGeometry,
                              settings::AbstractGEOSSettings,
                              steps_per_met::Int) where FT
    steps = workspace.steps_current
    contract.steps_per_window = steps
    contract_diag = verify_window!((m_cur = workspace.m_cur,
                                     am = workspace.am_v4,
                                     bm = workspace.bm_v4,
                                     cm = workspace.cm_v4,
                                     m_next = workspace.m_next_target),
                                    contract, win)

    for p in 1:CS_PANEL_COUNT
        copyto!(workspace.dm_v4[p], workspace.m_next_target[p])
    end
    convert_cs_mass_target_to_delta!(workspace.dm_v4, workspace.m_cur)

    full_window_scale = FT(2 * steps)
    _scale_cs_flux_panels!(workspace.am_v4, full_window_scale)
    _scale_cs_flux_panels!(workspace.bm_v4, full_window_scale)
    _scale_cs_flux_panels!(workspace.cm_v4, full_window_scale)

    surface_payload = (settings.include_surface || settings.include_vdiff_fields) ?
        _geos_surface_payload!(workspace.strategy, workspace.strategy_ws,
                               workspace.raw) : nothing
    cmfmc_payload = settings.include_convection ? _geos_cmfmc_payload!(workspace) : nothing
    dtrain_payload = settings.include_convection ? _geos_dtrain_payload!(workspace) : nothing
    vdiff_payload = settings.include_vdiff_fields ? _geos_vdiff_payload!(workspace) : nothing
    window_nt = (m = workspace.m_cur, am = workspace.am_v4,
                 bm = workspace.bm_v4, cm = workspace.cm_v4,
                 ps = workspace.ps_cur, dm = workspace.dm_v4,
                 surface = surface_payload,
                 cmfmc = cmfmc_payload,
                 dtrain = dtrain_payload,
                 vdiff = vdiff_payload)
    ready = ReadyWindow{CubedSphereTargetGeometry, FT}(win, window_nt)
    return PreverifiedWindow(ready, contract_diag)
end

function advance_window!(workspace::GEOSCubedSphereWindowWorkspace,
                         grid::CubedSphereTargetGeometry)
    workspace.chain_mass || return nothing
    Nc = grid.Nc
    Nz = size(workspace.m_cur[1], 3)
    for p in 1:CS_PANEL_COUNT
        copyto!(workspace.m_cur[p], workspace.m_next_target[p])
        _ps_from_air_mass!(workspace.ps_cur[p], workspace.m_cur[p],
                           workspace.cell_areas, workspace.g, Nc, Nz)
    end
    return nothing
end

struct GEOSReplayStats
    worst_replay_rel :: Float64
    worst_replay_abs :: Float64
    worst_replay_win :: Int
end

GEOSReplayStats() = GEOSReplayStats(0.0, 0.0, 0)

struct GEOSSplitSubstepStats
    max_steps_xy :: Int
    max_steps_z  :: Int
    win_xy       :: Int
    win_z        :: Int
    ratio_xy     :: Float64
    ratio_z      :: Float64
end

GEOSSplitSubstepStats() = GEOSSplitSubstepStats(0, 0, 0, 0, 0.0, 0.0)

struct GEOSCSUnifiedDriverContext{G, S, V}
    grid             :: G
    settings         :: S
    vertical         :: V
    steps_per_met    :: Int
    replay_stats     :: Base.RefValue{GEOSReplayStats}
    split_stats      :: Base.RefValue{GEOSSplitSubstepStats}
end

GEOSCSUnifiedDriverContext(grid, settings, vertical, steps_per_met::Integer) =
    GEOSCSUnifiedDriverContext{typeof(grid), typeof(settings), typeof(vertical)}(
        grid, settings, vertical, Int(steps_per_met), Ref(GEOSReplayStats()),
        Ref(GEOSSplitSubstepStats()))

function _geos_required_split_steps(workspace::GEOSCubedSphereWindowWorkspace,
                                    current_steps::Integer,
                                    ratio::Real)
    r = Float64(ratio)
    if isfinite(r)
        scaled = Float64(current_steps) * r / workspace.substep_cfl_target
        raw = scaled <= typemax(Int) ? ceil(Int, scaled) :
              workspace.max_steps_per_window
    else
        raw = workspace.max_steps_per_window
    end
    return min(max(raw, workspace.min_steps_per_window),
               workspace.max_steps_per_window)
end

function driver_ingest_window!(workspace::GEOSCubedSphereWindowWorkspace{FT},
                               reader::GEOSNativeReader{FT},
                               win::Int,
                               ctx::GEOSCSUnifiedDriverContext) where FT
    return ingest_window!(workspace, reader, win, ctx.grid, ctx.settings, ctx.vertical)
end

function driver_drain_ready_windows!(workspace::GEOSCubedSphereWindowWorkspace{FT},
                                     contract::CubedSphereContract{FT},
                                     win::Int,
                                     ctx::GEOSCSUnifiedDriverContext) where FT
    ready_diag = drain_ready_windows!(workspace, contract, win, ctx.grid,
                                      ctx.settings, ctx.steps_per_met)
    replay = ready_diag.contract.replay
    stats = ctx.replay_stats[]
    if stats.worst_replay_win == 0 || replay.max_rel_err > stats.worst_replay_rel
        ctx.replay_stats[] = GEOSReplayStats(replay.max_rel_err,
                                             replay.max_abs_err,
                                             win)
    end
    positivity = ready_diag.contract.positivity
    xy_steps = _geos_required_split_steps(workspace, workspace.steps_current,
                                          positivity.ratio_xy)
    z_steps = _geos_required_split_steps(workspace, workspace.steps_current,
                                         positivity.ratio_z)
    split = ctx.split_stats[]
    ctx.split_stats[] = GEOSSplitSubstepStats(
        max(split.max_steps_xy, xy_steps),
        max(split.max_steps_z, z_steps),
        xy_steps > split.max_steps_xy ? win : split.win_xy,
        z_steps > split.max_steps_z ? win : split.win_z,
        xy_steps > split.max_steps_xy ? Float64(positivity.ratio_xy) : split.ratio_xy,
        z_steps > split.max_steps_z ? Float64(positivity.ratio_z) : split.ratio_z,
    )
    return ready_diag
end

function driver_flush_final_windows!(::GEOSCubedSphereWindowWorkspace,
                                     ::GEOSNativeReader,
                                     ::CubedSphereContract,
                                     ::GEOSCSUnifiedDriverContext)
    return ()
end

function driver_before_close_writer!(workspace::GEOSCubedSphereWindowWorkspace,
                                     _reader::GEOSNativeReader,
                                     _contract::CubedSphereContract,
                                     writer::CubedSphereBinaryWriter,
                                     _ctx::GEOSCSUnifiedDriverContext)
    set_streaming_steps_per_window_schedule!(writer.inner, workspace.steps_schedule)
    return nothing
end

function driver_after_write_window!(workspace::GEOSCubedSphereWindowWorkspace,
                                    _reader::GEOSNativeReader,
                                    _ready::ReadyWindow,
                                    ctx::GEOSCSUnifiedDriverContext)
    return advance_window!(workspace, ctx.grid)
end

function _process_day_geos_cs_unified(date::Date,
                                      grid::CubedSphereTargetGeometry,
                                      settings::AbstractGEOSSettings,
                                      vertical;
                                      out_path::AbstractString,
                                      dt_met_seconds::Real,
                                      FT::Type{<:AbstractFloat},
                                      mass_basis::Symbol,
                                      replay_tol::Real,
                                      positivity_cfl_limit::Real,
                                      require_substep_positivity::Bool,
                                      adaptive_substeps::Bool,
                                      substep_cfl_target::Real,
                                      min_steps_per_window::Integer,
                                      max_steps_per_window::Integer,
                                      chain_mass::Bool,
                                      seed_m::Union{Nothing, NTuple{6, <:AbstractArray}},
                                      global_mass_pin::Bool,
                                      global_mass_target_kg::Real,
                                      balance_mode::Symbol,
                                      cm_closure::Symbol = :endpoint_balanced,
                                      smooth_iters::Integer = 8,
                                      omega_regularization::OmegaRegularization = OmegaRegularization())
    _OMEGA_TIMING[] = get(ENV, "ATMOS_OMEGA_TIMING", "0") in ("1", "true", "yes")
    Nc     = grid.Nc
    npanel = CS_PANEL_COUNT
    Nz     = vertical.Nz
    vc     = vertical.merged_vc
    panel_convention = "geos_native"

    steps_per_met = round(Int, FT(dt_met_seconds) / FT(settings.mass_flux_dt))
    reader_seed = seed_m === nothing ? nothing :
        ntuple(p -> Array{FT, 3}(seed_m[p]), CS_PANEL_COUNT)
    reader = open_reader(settings, date, FT;
                         seed = reader_seed,
                         chain_mass = chain_mass,
                         next_day_handle = true,
                         # Only OMEGA-based closures read the prev/next-day
                         # A3dyn+I3 handles (cross-midnight PCHIP); every other
                         # closure leaves them `nothing` (no extra opens).
                         adjacent_omega = _uses_omega(cm_closure))
    driver_started = false
    inner_writer = nothing
    tmp_path = out_path * ".tmp"

    try
        nw = windows_per_day(reader)
        workspace = allocate_window_workspace(grid, settings, vertical, FT;
                                               dt_met_seconds = dt_met_seconds,
                                               chain_mass = chain_mass,
                                               adaptive_substeps = adaptive_substeps,
                                               substep_cfl_target = substep_cfl_target,
                                               min_steps_per_window = min_steps_per_window,
                                               max_steps_per_window = max_steps_per_window,
                                               windows_per_day = nw,
                                               global_mass_pin = global_mass_pin,
                                               global_mass_target_kg = global_mass_target_kg,
                                               balance_mode = balance_mode,
                                               cm_closure = cm_closure,
                                               smooth_iters = smooth_iters,
                                               omega_regularization = omega_regularization)

        @info "GEOS → CS: $(date), source=$(settings) → $(out_path) [unified]"
        @info "  source_C=$(settings.Nc) target_C=$Nc  strategy=$(_geos_cs_strategy_name(workspace.strategy))"
        @info "  Nz=$Nz  windows=$nw  steps_per_met=$steps_per_met  flux_scale=$(workspace.flux_scale)"
        @info "  GEOS horizontal balance: $(workspace.balance_mode)   cm closure: $(workspace.cm_closure)"
        global_mass_pin &&
            @info @sprintf("  GEOS global dry-mass pin ENABLED: target=%s",
                           isfinite(Float64(global_mass_target_kg)) ?
                           @sprintf("%.9e kg", Float64(global_mass_target_kg)) :
                           "first window start")
        adaptive_substeps &&
            @info "  Adaptive substeps: target CFL=$(Float64(substep_cfl_target)) bounds=$(Int(min_steps_per_window)):$(Int(max_steps_per_window))"
        @info "  Level orientation: $(reader.handles.orientation)  (next-day endpoint: $(_geos_next_endpoint_available(reader.handles)))"

        mkpath(dirname(out_path))
        isfile(tmp_path) && rm(tmp_path; force = true)

        inner_writer = open_streaming_cs_transport_binary(
            tmp_path, Nc, npanel, Nz, nw, vc;
            FT = FT,
            dt_met_seconds = dt_met_seconds,
            steps_per_window = steps_per_met,
            flux_kind = :full_window_mass_amount,
            mass_basis = mass_basis,
            include_flux_delta = true,
            include_surface    = settings.include_surface || settings.include_vdiff_fields,
            include_cmfmc      = settings.include_convection,
            include_dtrain     = settings.include_convection,
            include_gchp_vdiff = settings.include_vdiff_fields,
            panel_convention   = panel_convention,
            cs_definition      = _cs_definition_tag(grid),
            cs_coordinate_law  = _cs_coordinate_law_tag(grid),
            cs_center_law      = _cs_center_law_tag(grid),
            longitude_offset_deg = longitude_offset_deg(cs_definition(grid.mesh)),
            extra_header = Dict{String, Any}(
                "preprocessor" => "geos_native_to_cs",
                "preprocessor_contract" => "plan41_variable_substeps",
                "runtime_substep_contract" => "binary_schedule",
                "runtime_flux_scaling" => "full_window_flux_divided_by_2x_steps_per_window",
                "cfl_definition" => "palindrome_outgoing_sum_over_min_endpoint_mass",
                "geos_mass_endpoint" => global_mass_pin ?
                    "dry_endpoint_global_mean_pinned" : "raw_dry_endpoint",
                "geos_horizontal_balance" => workspace.cm_closure === :pressure_fixer ?
                    "none_native_unbalanced" :
                    (workspace.balance_mode === :per_layer ?
                        "per_layer_poisson_to_endpoint" : "column_poisson_to_endpoint"),
                "geos_horizontal_balance_mode" => workspace.cm_closure === :pressure_fixer ?
                    "none" : String(workspace.balance_mode),
                "geos_cm_closure" => String(workspace.cm_closure),
                "geos_vertical_flux" =>
                    workspace.cm_closure === :pressure_fixer ?
                        "fv3_pressure_fixer_native_horizontal_chained_mass" :
                    workspace.cm_closure === :pfix_corrected ?
                        "fv3_pressure_fixer_native_horizontal_plus_zerosum_spatial_lowpass_drift_correction" :
                    workspace.cm_closure === :moisture_filtered ?
                        "diagnosed_from_balanced_horizontal_and_filtered_endpoint_moisture_residual_smoothed" :
                    workspace.cm_closure === :omega_full_replacement ?
                        "omega_full_replacement_with_per_level_horizontal_potential" :
                    workspace.cm_closure === :omega_regularized ?
                        "omega_utls_highpass_regularized_with_per_level_correction_cap" :
                        "diagnosed_from_balanced_horizontal_and_endpoint",
                "geos_global_mass_pin_enabled" => global_mass_pin,
                "geos_global_mass_pin_target_kg" => isfinite(workspace.global_mass_target_kg) ?
                    workspace.global_mass_target_kg : "first_window_start",
                "source_Nc" => settings.Nc,
                "geos_cs_resolution_strategy" => _geos_cs_strategy_name(workspace.strategy),
                "source_steps_per_window" => steps_per_met,
                "adaptive_substeps" => adaptive_substeps,
                "substep_cfl_target" => Float64(substep_cfl_target),
                "positivity_cfl_limit" => Float64(positivity_cfl_limit),
                "recommended_substeps_are_minimum" => true,
                "require_substep_positivity" => require_substep_positivity,
                "include_gchp_vdiff" => settings.include_vdiff_fields,
                "gchp_vdiff_source_fields" => settings.include_vdiff_fields ?
                    "A3dyn:U,V + I3:T + CTM_I1:QV + A1:PBLH,USTAR,HFLUX,T2M" : "none",
                "gchp_vdiff_sampling" => settings.include_vdiff_fields ?
                    "A3/I3 held constant over 3 hourly windows; QV uses left CTM_I1 endpoint" : "none",
                "vertical_transform" => String(Symbol(get(vertical, :vertical_mapping_method, :identity))),
                "vertical_Nz_native" => vertical.Nz_native,
                "vertical_Nz_output" => vertical.Nz,
                # Diagnostic-only key: emitted ONLY for the smoothing closures so
                # production `:endpoint_balanced` headers stay byte-for-byte identical.
                (workspace.cm_closure in (:moisture_filtered, :pfix_corrected) ?
                    ("geos_moisture_filter_smooth_iters" => workspace.smooth_iters,) :
                    ())...,
                (workspace.cm_closure === :omega_regularized ?
                    ("geos_omega_pressure_taper_hpa" =>
                         collect(workspace.omega_regularization.pressure_taper_hpa),
                     "geos_omega_smoothing_steps" =>
                         workspace.omega_regularization.smoothing_steps,
                     "geos_omega_smoothing_fraction" =>
                         workspace.omega_regularization.smoothing_fraction,
                     "geos_omega_max_relative_flux_correction" =>
                         workspace.omega_regularization.max_relative_flux_correction,
                     "geos_omega_max_bottom_flux_correction" =>
                         workspace.omega_regularization.max_bottom_flux_correction) :
                    ())...,
            ),
        )
        writer = CubedSphereBinaryWriter(inner_writer, DryBasis();
                                         Nc = Nc, npanel = npanel,
                                         final_path = out_path)
        window_contract = CubedSphereContract{FT}(
            replay_tol = replay_tol,
            positivity_cfl_limit = positivity_cfl_limit,
            require_substep_positivity = require_substep_positivity,
            steps_per_window = steps_per_met,
        )
        ctx = GEOSCSUnifiedDriverContext(grid, settings, vertical, steps_per_met)

        t_start = time()
        driver_started = true
        driver_result = run_unified_preprocessor_day!(
            UnifiedPreprocessorDay(reader, workspace, window_contract, writer;
                                   context = ctx))
        elapsed = time() - t_start
        stats = ctx.replay_stats[]
        @info @sprintf("  Done in %.1fs (%.2fs/window). Worst replay: rel=%.2e abs=%.2e at win=%d",
                       elapsed, elapsed / nw, stats.worst_replay_rel,
                       stats.worst_replay_abs, stats.worst_replay_win)
        split_stats = ctx.split_stats[]
        @info @sprintf("  Substep diagnostic: stored=%d..%d; hypothetical split max xy=%d at win=%d (ratio=%.3f), z=%d at win=%d (ratio=%.3f)",
                       minimum(workspace.steps_schedule),
                       maximum(workspace.steps_schedule),
                       split_stats.max_steps_xy, split_stats.win_xy,
                       split_stats.ratio_xy,
                       split_stats.max_steps_z, split_stats.win_z,
                       split_stats.ratio_z)

        final_m = chain_mass ? ntuple(p -> copy(workspace.m_cur[p]), npanel) : nothing
        set_end_of_day_seed!(reader, final_m)

        return (
            elapsed = elapsed,
            worst_replay_rel = stats.worst_replay_rel,
            worst_replay_abs = stats.worst_replay_abs,
            worst_replay_win = stats.worst_replay_win,
            out_path = driver_result.out_path,
            steps_per_window_by_window = copy(workspace.steps_schedule),
            final_m = final_m,
            global_mass_target_kg = workspace.global_mass_target_kg,
        )
    finally
        if !driver_started
            if inner_writer !== nothing
                try
                    close_streaming_transport_binary!(inner_writer)
                catch err
                    @warn("Unified GEOS-CS: failed to close writer during cleanup",
                          exception = (err, catch_backtrace()))
                end
            end
            close_reader!(reader)
            isfile(tmp_path) && rm(tmp_path; force = true)
        end
    end
end

"""
    process_day(date, grid::CubedSphereTargetGeometry,
                settings::AbstractGEOSSettings, vertical;
                out_path,
                dt_met_seconds = 3600.0,
                FT = Float64,
                mass_basis = :dry,
                replay_tol = replay_tolerance(FT),
                seed_m = nothing,
                next_day_hour0 = nothing,
                chain_mass = true) -> NamedTuple

Build a v4 cubed-sphere transport binary at `out_path` from one UTC day of
native GEOS data. Source mesh and target mesh must match (CS passthrough).

Stored mass targets the raw GEOS dry endpoint (`DELP_dry`) transformed to the
output vertical grid. The native horizontal fluxes are column-balanced to that
endpoint, then `cm` is diagnosed so the replay and positivity contracts are
checked against the same endpoint the runtime will see.

For multi-day preprocessing with `chain_mass = true`, `seed_m` carries the
raw endpoint from the previous day so adjacent daily binaries share a boundary
mass: pass `nothing` (default) on day 1 to seed from raw GEOS DELP_dry, and on
day N+1 pass the `final_m` returned by day N's `process_day`. With
`chain_mass = false`, `seed_m` is ignored and every window reinitializes from
raw GEOS mass.

When `chain_mass = true`, the returned NamedTuple includes
`final_m::NTuple{6, Array{FT, 3}}`, the raw-endpoint state at the END of the
last window. With `chain_mass = false`, `final_m` is `nothing`.

`next_day_hour0` is part of the inherited topology-dispatch contract but
unused — the GEOS reader handles next-day endpoints internally via
`next_ctm_i1`.
"""
function process_day(date::Date,
                     grid::CubedSphereTargetGeometry,
                     settings::AbstractGEOSSettings,
                     vertical;
                     out_path::AbstractString,
                     dt_met_seconds::Real = 3600.0,
                     FT::Type{<:AbstractFloat} = Float64,
                     mass_basis::Symbol = :dry,
                     replay_tol::Real = replay_tolerance(FT),
                     positivity_cfl_limit::Real = 0.95,
                     require_substep_positivity::Bool = true,
                     adaptive_substeps::Bool = false,
                     substep_cfl_target::Real = positivity_cfl_limit,
                     min_steps_per_window::Integer = 1,
                     max_steps_per_window::Integer = typemax(Int),
                     chain_mass::Bool = true,
                     seed_m::Union{Nothing, NTuple{6, <:AbstractArray}} = nothing,
                     global_mass_pin::Bool = false,
                     global_mass_target_kg::Real = NaN,
                     balance_mode::Symbol = :column,
                     cm_closure::Symbol = :endpoint_balanced,
                     smooth_iters::Integer = 8,
                     omega_regularization::OmegaRegularization = OmegaRegularization(),
                     next_day_hour0 = nothing)
    # Reject configurations the path cannot honor:
    mass_basis === :dry ||
        error("GEOS-CS passthrough only supports mass_basis=:dry; got $(mass_basis). " *
              "GEOS MFXC/MFYC are already dry; the chained pressure-fixer is dry-basis.")
    _validate_geos_native_panel_convention(grid.mesh.convention)
    return _process_day_geos_cs_unified(
        date, grid, settings, vertical;
        out_path = out_path,
        dt_met_seconds = dt_met_seconds,
        FT = FT,
        mass_basis = mass_basis,
        replay_tol = replay_tol,
        positivity_cfl_limit = positivity_cfl_limit,
        require_substep_positivity = require_substep_positivity,
        adaptive_substeps = adaptive_substeps,
        substep_cfl_target = substep_cfl_target,
        min_steps_per_window = min_steps_per_window,
        max_steps_per_window = max_steps_per_window,
        chain_mass = chain_mass,
        seed_m = seed_m,
        global_mass_pin = global_mass_pin,
        global_mass_target_kg = global_mass_target_kg,
        balance_mode = balance_mode,
        cm_closure = cm_closure,
        smooth_iters = smooth_iters,
        omega_regularization = omega_regularization,
    )
end
