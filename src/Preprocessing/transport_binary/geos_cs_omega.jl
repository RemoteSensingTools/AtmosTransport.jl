# GEOS native CS preprocessing: OMEGA-consistent vertical-flux target (PCHIP time interpolation, regularization, reconstruction).
# Split from cubed_sphere_geos.jl (refactor phase 4); included by Preprocessing.jl in this order.

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
# scripts/diagnostics/heritage/fingerfix_proto_omega-consistent-flux-reconstruction.jl.
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
