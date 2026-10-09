# GEOS native CS preprocessing: window workspace, per-window preparation and substep selection, ingest/drain/advance.
# Split from cubed_sphere_geos.jl (refactor phase 4); included by Preprocessing.jl in this order.

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
