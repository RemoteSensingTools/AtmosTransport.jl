# ===========================================================================
# Native GEOS-IT/FP cubed-sphere → v4 transport binary preprocessing path.
#
# Source axis:  AbstractGEOSSettings (read native CTM_A1/CTM_I1 NetCDF)
# Target axis:  CubedSphereTargetGeometry, the native mesh (passthrough,
#               IdentityRegrid) or an integer-factor block coarsening of it
#
# Files, in include order: geos_cs_mass_helpers.jl, geos_cs_omega.jl,
# geos_cs_resolution.jl, geos_cs_window.jl, then this file (driver context,
# hooks, process_day).
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

# The balance a GEOS closure applies: the pressure-fixer closures keep the
# native fluxes; the moisture-filtered and OMEGA closures always balance the
# column (their constructions need column closure, `geos_cs_window.jl`).
function _geos_effective_balance(cm_closure::Symbol, balance_mode::Symbol)
    cm_closure in (:pressure_fixer, :pfix_corrected) && return "none"
    (cm_closure === :moisture_filtered || _uses_omega(cm_closure)) && return "column"
    return String(balance_mode)
end

_geos_balance_description(balance::AbstractString) =
    balance == "none" ? "none_native_unbalanced" : balance * "_poisson_to_endpoint"

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
    write_replay_on  :: Bool
    replay_stats     :: Base.RefValue{GEOSReplayStats}
    split_stats      :: Base.RefValue{GEOSSplitSubstepStats}
end

GEOSCSUnifiedDriverContext(grid, settings, vertical, steps_per_met::Integer;
                           write_replay_on::Bool = true) =
    GEOSCSUnifiedDriverContext{typeof(grid), typeof(settings), typeof(vertical)}(
        grid, settings, vertical, Int(steps_per_met), write_replay_on,
        Ref(GEOSReplayStats()), Ref(GEOSSplitSubstepStats()))

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
                                      ctx.settings, ctx.steps_per_met;
                                      write_replay_on = ctx.write_replay_on)
    replay = ready_diag.contract.replay
    stats = ctx.replay_stats[]
    if ctx.write_replay_on &&
            (stats.worst_replay_win == 0 || replay.max_rel_err > stats.worst_replay_rel)
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

        # Resolved once: the header records it and the gate uses it.
        write_replay_on = write_replay_check_enabled()
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
            planet_radius      = grid.mesh.radius,
            extra_header = _with_replay_record(Dict{String, Any}(
                "preprocessor" => "geos_native_to_cs",
                "preprocessor_contract" => "plan41_variable_substeps",
                "runtime_substep_contract" => "binary_schedule",
                "runtime_flux_scaling" => "full_window_flux_divided_by_2x_steps_per_window",
                "cfl_definition" => "palindrome_outgoing_sum_over_min_endpoint_mass",
                "geos_mass_endpoint" => global_mass_pin ?
                    "dry_endpoint_global_mean_pinned" : "raw_dry_endpoint",
                "geos_horizontal_balance" => _geos_balance_description(
                    _geos_effective_balance(workspace.cm_closure, workspace.balance_mode)),
                "geos_horizontal_balance_mode" => _geos_effective_balance(workspace.cm_closure,
                                                                          workspace.balance_mode),
                "horizontal_balance" => _geos_effective_balance(workspace.cm_closure,
                                                                workspace.balance_mode),
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
            ), write_replay_on),
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
        write_replay_on || @info "  Write-time CS replay gate SKIPPED (disabled for this run)"
        ctx = GEOSCSUnifiedDriverContext(grid, settings, vertical, steps_per_met;
                                         write_replay_on)

        t_start = time()
        driver_started = true
        driver_result = run_unified_preprocessor_day!(
            UnifiedPreprocessorDay(reader, workspace, window_contract, writer;
                                   context = ctx))
        elapsed = time() - t_start
        stats = ctx.replay_stats[]
        @info write_replay_on ?
            @sprintf("  Done in %.1fs (%.2fs/window). Worst replay: rel=%.2e abs=%.2e at win=%d",
                     elapsed, elapsed / nw, stats.worst_replay_rel,
                     stats.worst_replay_abs, stats.worst_replay_win) :
            @sprintf("  Done in %.1fs (%.2fs/window). Replay gate skipped.", elapsed, elapsed / nw)
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
native GEOS data. The target mesh is either the native mesh (CS passthrough)
or a nested block coarsening of it (native `Nc` an integer multiple of the
target `Nc`); coarsening sums masses and face fluxes over each block and
area-weights the physics fields.

Stored mass targets the raw GEOS dry endpoint (`DELP_dry`) transformed to the
output vertical grid. With the default column balance (`[numerics]
balance_mode`) and `cm_closure = :endpoint_balanced`, the horizontal fluxes are
column-balanced to that endpoint, then `cm` is diagnosed so the replay and
positivity contracts are checked against the same endpoint the runtime will
see.

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
                     horizontal_balance::Union{Nothing, AbstractHorizontalBalance} = nothing,
                     balance_mode::Union{Nothing, Symbol} = nothing,
                     cm_closure::Symbol = :endpoint_balanced,
                     smooth_iters::Integer = 8,
                     omega_regularization::OmegaRegularization = OmegaRegularization(),
                     next_day_hour0 = nothing)
    # Reject configurations the path cannot honor:
    mass_basis === :dry ||
        error("GEOS-CS passthrough only supports mass_basis=:dry; got $(mass_basis). " *
              "GEOS MFXC/MFYC are already dry; the chained pressure-fixer is dry-basis.")
    _validate_geos_native_panel_convention(grid.mesh.convention)
    if balance_mode !== nothing        # deprecated keyword: `:column` or `:per_layer`
        Base.depwarn("process_day(...; balance_mode) for GEOS is deprecated; " *
                     "pass horizontal_balance = ColumnBalance() or LayerBalance()", :process_day)
        from_symbol = resolve_horizontal_balance(Dict("balance_mode" => String(balance_mode)))
        horizontal_balance === nothing || horizontal_balance === from_symbol || throw(ArgumentError(
            "balance_mode = $(repr(balance_mode)) conflicts with horizontal_balance = $(horizontal_balance)"))
        horizontal_balance = from_symbol
    end
    horizontal_balance isa LayerBalance && cm_closure !== :endpoint_balanced &&
        @warn "balance_mode = \"per_layer\" has no effect with geos_cm_closure = " *
              "$(repr(cm_closure)); the header records the balance the closure applies" maxlog = 1
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
        balance_mode = effective_horizontal_balance(horizontal_balance, ColumnBalance()) isa
                       LayerBalance ? :per_layer : :column,
        cm_closure = cm_closure,
        smooth_iters = smooth_iters,
        omega_regularization = omega_regularization,
    )
end
