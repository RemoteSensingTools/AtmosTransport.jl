# Reduced-Gaussian spectral preprocessing: window synthesis, mass pin and next-day end point, Poisson balance, driver hooks, process_day.
# Split from reduced_transport_helpers.jl (refactor phase 4); included by Preprocessing.jl in this order.

# =========================================================================
# RG window synthesis and process_day — mirrors the LL path
# =========================================================================

"""
    synthesize_and_merge_window!(work, merged, hour, spec, grid, vertical,
                                 settings, ps_offsets, win_idx;
                                 qv_ws=nothing, thermo_path="")

Spectral synthesis → native fields → mass fix → dry conversion → level merge
for one window. Results are left in `merged.m_merged`, `merged.hflux_merged`,
`merged.cm_merged` and `work.sp` (surface pressure). No allocation.

When `qv_ws` is provided and `settings.mass_basis == :dry`, loads QV from the
thermo file and interpolates it to RG cells; the window is then finished by
[`pin_convert_merge_window!`](@ref).
"""
function synthesize_and_merge_window!(work::ReducedTransformWorkspace,
                                      merged::ReducedMergeWorkspace{FT},
                                      hour::Int,
                                      spec,
                                      grid::ReducedGaussianTargetGeometry,
                                      vertical,
                                      settings,
                                      ps_offsets::Vector{Float64},
                                      win_idx::Int;
                                      qv_ws::Union{ReducedQVWorkspace, Nothing}=nothing,
                                      thermo_path::String="") where FT
    spectral_to_native_fields!(work,
        spec.lnsp_all[hour], spec.vo_by_hour[hour], spec.d_by_hour[hour],
        spec.T, vertical.level_range, vertical.ab, grid, settings.half_dt)
    dry = settings.mass_basis == :dry && qv_ws !== nothing
    dry && load_rg_qv!(qv_ws, thermo_path, win_idx, vertical.Nz_native)
    pin_convert_merge_window!(work, merged, vertical, settings, ps_offsets, win_idx, dry ? qv_ws : nothing)
    return nothing
end

"""
    synthesize_next_day_hour0!(work, merged, next_day_hour0, date, grid, vertical,
                               settings, ps_offsets; qv_ws=nothing)

The next day's 00 UTC state, the end point of the day's last window, through
the same synthesis, pin (offset stored in `ps_offsets[end]`), dry conversion
(the next day's thermo file, time index 1) and merge as every window.
"""
function synthesize_next_day_hour0!(work::ReducedTransformWorkspace,
                                    merged::ReducedMergeWorkspace,
                                    next_day_hour0,
                                    date::Date,
                                    grid::ReducedGaussianTargetGeometry,
                                    vertical,
                                    settings,
                                    ps_offsets::Vector{Float64};
                                    qv_ws::Union{ReducedQVWorkspace, Nothing}=nothing)
    spectral_to_native_fields!(work, next_day_hour0.lnsp, next_day_hour0.vo, next_day_hour0.d,
                               next_day_hour0.T, vertical.level_range, vertical.ab, grid,
                               settings.half_dt)
    dry = settings.mass_basis == :dry && qv_ws !== nothing
    if dry
        path = joinpath(settings.thermo_dir,
                        "era5_thermo_ml_$(Dates.format(date + Day(1), "yyyymmdd")).nc")
        isfile(path) || error("Thermo file not found for the next-day endpoint: $path")
        qv_ws.qv_ll .= read_qv_from_thermo(path, 1, qv_ws.Nx_ll, qv_ws.Ny_ll, vertical.Nz_native;
                                           FT = Float64)
        _interpolate_ll_to_rg!(qv_ws)
    end
    pin_convert_merge_window!(work, merged, vertical, settings, ps_offsets, length(ps_offsets),
                              dry ? qv_ws : nothing)
    return nothing
end

"""
    pin_convert_merge_window!(work, merged, vertical, settings, ps_offsets, slot, qv_ws)

Finish a synthesized window: the global mass fix (offset stored in
`ps_offsets[slot]`), the dry-basis conversion when `qv_ws` holds the window's
humidity, and the level merge.

With humidity the fix pins the global dry surface pressure, as the lat-lon and
cubed-sphere paths do: the vertical-flux closure needs the same global dry
mass at both ends of a window. Without humidity it pins the total surface
pressure with the climatological `qv_global_climatology`. The horizontal
fluxes follow the pinned pressure: each is wind × Δp at the face, so it scales
by the ratio of the face Δp after and before the pin.
"""
function pin_convert_merge_window!(work::ReducedTransformWorkspace,
                                   merged::ReducedMergeWorkspace,
                                   vertical,
                                   settings,
                                   ps_offsets::Vector{Float64},
                                   slot::Int,
                                   qv_ws::Union{ReducedQVWorkspace, Nothing})
    if settings.mass_fix_enable
        ab = vertical.ab
        ps_offsets[slot] = qv_ws === nothing ?
            pin_global_mean_ps!(work.sp, work.cell_areas; target_ps_dry_pa = settings.target_ps_dry_pa,
                                qv_global = settings.qv_global_climatology) :
            pin_global_mean_ps_using_qv!(work.sp, work.cell_areas, ab.dA, ab.dB, qv_ws.qv_cell;
                                         target_ps_dry_pa = settings.target_ps_dry_pa)
        compute_reduced_dp_and_mass!(work.dp, work.m_arr, work.sp, work.cell_areas, ab.dA, ab.dB)
        rescale_reduced_fluxes_to_ps!(work.hflux_arr, work.lnsp, work.sp, work.face_left,
                                      work.face_right, ab.dA, ab.dB)
        @. work.lnsp = log(work.sp)
    end
    qv_ws === nothing || apply_dry_basis_reduced!(work, qv_ws.qv_cell)
    merge_reduced_window!(merged, work, vertical)
    return nothing
end

"""
    rescale_reduced_fluxes_to_ps!(hflux, lnsp_old, sp_new, face_left, face_right, dA, dB)

Scale the face mass fluxes `hflux[f, k]` (wind × Δp at the face) from the
surface pressure `exp.(lnsp_old)` to `sp_new`: the face pressure is the
geometric mean of the two cells' (as in `compute_reduced_horizontal_fluxes!`)
and Δp = |dA_k + dB_k p|. Polar stub faces (a zero neighbour) are set to zero.
"""
function rescale_reduced_fluxes_to_ps!(hflux::AbstractMatrix{Float64},
                                       lnsp_old::AbstractVector{Float64},
                                       sp_new::AbstractVector{Float64},
                                       face_left::AbstractVector{<:Integer},
                                       face_right::AbstractVector{<:Integer},
                                       dA::AbstractVector, dB::AbstractVector)
    @inbounds for k in axes(hflux, 2), f in axes(hflux, 1)
        left, right = face_left[f], face_right[f]
        if left == 0 || right == 0
            hflux[f, k] = 0.0
            continue
        end
        p_old = exp((lnsp_old[left] + lnsp_old[right]) / 2)
        p_new = sqrt(sp_new[left] * sp_new[right])
        dp_old = abs(dA[k] + dB[k] * p_old)
        dp_old > 0 && (hflux[f, k] *= abs(dA[k] + dB[k] * p_new) / dp_old)
    end
    return hflux
end

"""
    balance_window!(hflux_work, m_cur_work, m_next_work, cm_work, div_scratch,
                    dm_target_work, buf, slot, m_next, work, cL, steps_per_window;
                    tol, max_iter)

Poisson-balance the horizontal fluxes in buffer `slot` using the
compressed Laplacian `cL` for the CG solver.  Mathematically identical
to the old face-indexed CG but ~16-27× faster because the compressed
MatVec iterates over ~4×ncells entries instead of nfaces (millions).

All scratch arrays are preallocated Float64 buffers — no per-call allocation.
Returns a diagnostics NamedTuple.
"""
function balance_window!(hflux_work::Matrix{Float64},
                         m_cur_work::Matrix{Float64},
                         m_next_work::Matrix{Float64},
                         cm_work::Matrix{Float64},
                         div_scratch::Matrix{Float64},
                         dm_target_work::Matrix{Float64},
                         buf::SlidingWindowBuffer{FT},
                         slot::Int,
                         m_next::AbstractMatrix{FT},
                         work::ReducedTransformWorkspace,
                         cL::CompressedLaplacian,
                         steps_per_window::Int;
                         tol::Float64 = 1e-14,
                         max_iter::Int = 20000) where FT

    copyto!(hflux_work, buf.hflux[slot])
    copyto!(m_cur_work, buf.m[slot])
    copyto!(m_next_work, m_next)

    scratch = (psi = work.balance_psi, rhs = work.balance_rhs,
               r = work.balance_r, p = work.balance_p,
               Ap = work.balance_Ap, z = work.balance_z)
    replay_layout = faceindexed_replay_layout(work.face_left, work.face_right)

    diag = balance_compressed_horizontal_fluxes!(
        hflux_work, m_cur_work, m_next_work,
        work.face_left, work.face_right, cL,
        steps_per_window, scratch; tol=tol, max_iter=max_iter)

    # Store balanced hflux back as FT.
    buf.hflux[slot] .= FT.(hflux_work)

    # Recompute cm from balanced hflux using the explicit-dm closure. The
    # Δb × pit closure does not hold on the dry basis: dm[k] = dB[k] Σ dm
    # holds only for moist hybrid coordinates, and the humidity profile
    # breaks it by ~27%.
    size(dm_target_work) == size(m_cur_work) ||
        error("balance_window!: dm_target_work shape $(size(dm_target_work)) " *
              "!= m_cur_work shape $(size(m_cur_work)).")
    inv_scale = 1.0 / (2 * max(Int(steps_per_window), 1))
    @. dm_target_work = (m_next_work - m_cur_work) * inv_scale
    recompute_cm_from_dm_target!(replay_layout, div_scratch, cm_work,
                                 m_cur_work, dm_target_work, hflux_work)
    buf.cm[slot] .= FT.(cm_work)

    return diag
end

function _verify_rg_balanced_window!(window_contract,
                                     m_cur_work::Matrix{Float64},
                                     hflux_work::Matrix{Float64},
                                     cm_work::Matrix{Float64},
                                     m_next_work::Matrix{Float64},
                                     win_idx::Int;
                                     write_replay_on::Bool,
                                     accumulate::Bool = true)
    if write_replay_on
        diag = verify_window!((m_cur = m_cur_work,
                               hflux = hflux_work,
                               cm = cm_work,
                               m_next = m_next_work),
                              window_contract, win_idx)
    else
        stub = verify_boundary_stub_flux_rg(hflux_work,
                                             window_contract.face_left,
                                             window_contract.face_right;
                                             tol = window_contract.boundary_stub_tol)
        stub.violated &&
            error("Boundary-stub flux gate FAILED for RG window $(win_idx): " *
                  "hflux=$(stub.worst_flux) on face=$(stub.worst_face) " *
                  "level=$(stub.worst_level) where " *
                  "face_left=$(window_contract.face_left[stub.worst_face]) " *
                  "face_right=$(window_contract.face_right[stub.worst_face]); " *
                  "runtime face-indexed advection (`Advection/sweeps.jl`) will silently " *
                  "discard this flux.")

        m_shape = size(m_cur_work)
        if window_contract._outgoing_h === nothing ||
           size(window_contract._outgoing_h) != m_shape
            window_contract._outgoing_h = Array{Float64}(undef, m_shape)
            window_contract._bad_h = Array{Bool}(undef, m_shape)
        end
        positivity = verify_substep_positivity_rg!(m_cur_work, hflux_work, cm_work,
                                                    window_contract.face_left,
                                                    window_contract.face_right;
                                                    cfl_limit =
                                                        window_contract.positivity_cfl_limit,
                                                    outgoing_h =
                                                        window_contract._outgoing_h,
                                                    bad_h =
                                                        window_contract._bad_h)
        diag = (replay = nothing, positivity = positivity)
    end

    accumulate && update_accumulator!(window_contract, diag.positivity, win_idx)
    return diag
end

mutable struct RGSpectralUnifiedDriverContext{G, S, V, SP, P, N}
    grid              :: G
    settings          :: S
    vertical          :: V
    spec              :: SP
    substep_policy    :: P
    date              :: Date
    next_day_hour0    :: N          # next day's 00 UTC spectral state, or nothing
    steps_per_window  :: Int
    write_replay_on   :: Bool
    worst_pre_raw     :: Float64
    worst_post_proj   :: Float64
    worst_post_raw    :: Float64
    worst_iter        :: Int
    worst_replay_rel  :: Float64
    worst_replay_abs  :: Float64
    worst_replay_win  :: Int
    worst_replay_idx  :: Tuple{Int, Int}
end

driver_windows_per_day(::Nothing, ctx::RGSpectralUnifiedDriverContext) =
    ctx.spec.n_times

function driver_ingest_window!(workspace::ReducedGaussianSpectralWindowWorkspace,
                               ::Nothing,
                               win::Int,
                               ctx::RGSpectralUnifiedDriverContext)
    slot = win == 1 ? workspace.cur : workspace.nxt
    t_synth = ingest_window!(workspace, slot, win, ctx.spec.hours[win],
                             ctx.spec, ctx.grid, ctx.vertical, ctx.settings)
    should_log_window(win, ctx.spec.n_times) &&
        @info @sprintf("    Window %2d/%d (hour %02d): synth %.2fs  offset=%+.3f Pa",
                       win, ctx.spec.n_times, ctx.spec.hours[win], t_synth,
                       workspace.ps_offsets[win])
    return nothing
end

function _rg_unified_record_diag!(ctx::RGSpectralUnifiedDriverContext,
                                  ready_diag,
                                  win::Int)
    diag = ready_diag.balance
    ctx.worst_pre_raw = max(ctx.worst_pre_raw, diag.max_pre_raw_residual)
    ctx.worst_post_proj = max(ctx.worst_post_proj, diag.max_post_projected)
    ctx.worst_post_raw = max(ctx.worst_post_raw, diag.max_post_raw_residual)
    ctx.worst_iter = max(ctx.worst_iter, diag.max_cg_iter)

    contract_diag = ready_diag.contract
    if ctx.write_replay_on && contract_diag.replay.max_rel_err > ctx.worst_replay_rel
        ctx.worst_replay_rel = contract_diag.replay.max_rel_err
        ctx.worst_replay_abs = contract_diag.replay.max_abs_err
        ctx.worst_replay_win = win
        ctx.worst_replay_idx = contract_diag.replay.worst_idx
    end
    return nothing
end

function driver_drain_ready_windows!(workspace::ReducedGaussianSpectralWindowWorkspace,
                                     contract,
                                     win::Int,
                                     ctx::RGSpectralUnifiedDriverContext)
    win == 1 && return ()
    written_win = win - 1
    t_bal = time()
    ready_diag = drain_ready_windows!(workspace, contract, written_win,
                                      ctx.steps_per_window;
                                      write_replay_on = ctx.write_replay_on,
                                      substep_policy = ctx.substep_policy)
    t_bal = time() - t_bal
    _rg_unified_record_diag!(ctx, ready_diag, written_win)
    diag = ready_diag.balance
    should_log_window(written_win, ctx.spec.n_times) &&
        @info @sprintf("    Window %2d/%d: wrote (bal %.2fs pre_raw=%.2e post_proj=%.2e iter=%d)",
                       written_win, ctx.spec.n_times, t_bal,
                       diag.max_pre_raw_residual, diag.max_post_projected,
                       diag.max_cg_iter)
    return (PreverifiedWindow(ready_diag.ready, ready_diag.contract;
                              accumulated = true),)
end

function driver_flush_final_windows!(workspace::ReducedGaussianSpectralWindowWorkspace,
                                     ::Nothing,
                                     contract,
                                     ctx::RGSpectralUnifiedDriverContext)
    Nt = ctx.spec.n_times
    # The last window ends at the next day's 00 UTC; without it the window keeps
    # its own mass at both ends (zero tendency).
    m_next = workspace.buf.m[workspace.cur]
    if ctx.next_day_hour0 !== nothing
        synthesize_next_day_hour0!(workspace.work, workspace.merged, ctx.next_day_hour0, ctx.date,
                                   ctx.grid, ctx.vertical, ctx.settings, workspace.ps_offsets;
                                   qv_ws = workspace.qv_ws)
        fill_buffer_slot!(workspace.buf, workspace.nxt, workspace.merged, workspace.work.sp)
        m_next = workspace.buf.m[workspace.nxt]
    end
    t_bal = time()
    ready_diag = flush_final_windows!(workspace, contract, Nt,
                                      ctx.steps_per_window;
                                      write_replay_on = ctx.write_replay_on,
                                      substep_policy = ctx.substep_policy,
                                      m_next)
    t_bal = time() - t_bal
    _rg_unified_record_diag!(ctx, ready_diag, Nt)
    diag = ready_diag.balance
    @info @sprintf("    Window %2d/%d (last): bal %.2fs  pre_raw=%.2e post_proj=%.2e iter=%d",
                   Nt, Nt, t_bal, diag.max_pre_raw_residual,
                   diag.max_post_projected, diag.max_cg_iter)
    return (PreverifiedWindow(ready_diag.ready, ready_diag.contract;
                              accumulated = true),)
end

function driver_before_close_writer!(workspace::ReducedGaussianSpectralWindowWorkspace,
                                     ::Nothing,
                                     contract,
                                     writer,
                                     ::RGSpectralUnifiedDriverContext)
    set_streaming_steps_per_window_schedule!(writer.inner, workspace.steps_schedule)
    writer.inner.header["ps_offsets_pa_per_window"] = workspace.ps_offsets[1:end - 1]
    writer.inner.header["ps_offsets_next_day_hour0_pa"] = workspace.ps_offsets[end]
    set_contract_steps_schedule!(contract, workspace.steps_schedule)
    return nothing
end

"""
    process_day(date, grid::ReducedGaussianTargetGeometry, settings, vertical;
                next_day_hour0=nothing, positivity_cfl_limit=0.95,
                require_substep_positivity=true)

Streaming one-day preprocessing for reduced-Gaussian targets.

Uses a 2-window sliding buffer: at any time only two windows' worth of
`(m, hflux, cm, ps)` are held in memory.  Each window is Poisson-balanced
and written to disk before the next pair is computed.  This reduces peak
memory from `O(Nt)` to `O(1)` and enables O160/O320 binary generation.

Pipeline per window:
  spectral synthesis → mass fix → level merge → (wait for next window) →
  Poisson balance using (m_cur, m_next) → cm recomputation → stream-write
"""
function process_day(date::Date,
                     grid::ReducedGaussianTargetGeometry,
                     settings::ERA5SpectralSettings,
                     vertical;
                     next_day_hour0=nothing,
                     positivity_cfl_limit::Real = 0.95,
                     require_substep_positivity::Bool = true,
                     substep_policy =
                         SubstepSchedulePolicy(
                             adaptive_substeps = false,
                             substep_cfl_target = positivity_cfl_limit),
                     run_cache = nothing)
    FT = settings.output_float_type
    get(settings, :horizontal_balance, nothing) isa ColumnBalance && throw(ArgumentError(
        "reduced-Gaussian preprocessing balances each layer (ring Poisson CG); " *
        "[numerics] balance_mode = \"column\" is not implemented for it"))
    settings.include_qv && throw(ArgumentError(
        "reduced-Gaussian preprocessing does not support output.include_qv=true; " *
        "set include_qv=false (humidity may still be used internally for dry-basis conversion)"))
    mesh = grid.mesh
    nc = ncells(mesh)
    nf = nfaces(mesh)
    Nz = vertical.Nz
    steps_per_met = exact_steps_per_window(settings.met_interval, settings.dt)
    date_str = Dates.format(date, "yyyymmdd")

    vo_d_path = joinpath(settings.spectral_dir, "era5_spectral_$(date_str)_vo_d.gb")
    lnsp_path = joinpath(settings.spectral_dir, "era5_spectral_$(date_str)_lnsp.gb")

    if !isfile(vo_d_path) || !isfile(lnsp_path)
        @warn "Missing GRIB files for $date_str, skipping"
        return nothing
    end

    t_day = time()
    @info "  Reading spectral data for $date_str..."
    spec = read_day_spectral(vo_d_path, lnsp_path;
                             T_target=settings.T_target,
                             cache_dir=settings.spectral_cache_dir)
    @info @sprintf("  Spectral data read: T=%d, %d hours (%.1fs)",
                   spec.T, spec.n_times, time() - t_day)

    Nt = spec.n_times
    mkpath(settings.out_dir)
    bin_path = output_binary_path(date, settings.out_dir, settings.min_dp, FT)

    workspace = allocate_window_workspace(grid, settings, vertical, spec, date, FT;
                                          cache = run_cache)
    work = workspace.work
    buf = workspace.buf
    ps_offsets = workspace.ps_offsets
    write_replay_on = write_replay_check_enabled()
    write_replay_on ||
        @info "  Write-time replay gate SKIPPED (ATMOSTR_NO_WRITE_REPLAY_CHECK=1)"
    window_contract = ReducedGaussianContract{FT}(
        replay_tol = replay_tolerance(FT),
        positivity_cfl_limit = positivity_cfl_limit,
        require_substep_positivity = require_substep_positivity,
        steps_per_window = steps_per_met,
        face_left = work.face_left,
        face_right = work.face_right,
    )

    # Open the streaming binary writer
    vc_merged = vertical.merged_vc
    transport_grid = AtmosGrid(mesh, vc_merged, CPU(); FT=FT, radius=mesh.radius)
    sample_window = (m = buf.m[1], hflux = buf.hflux[1],
                     cm = buf.cm[1], ps = buf.ps[1])

    # Declare the canonical window_constant contract
    # explicitly. The sliding-window payload stores already balanced,
    # window-constant fluxes and no endpoint deltas, so delta semantics must
    # be `none`. Poisson metadata still records how those fluxes were built.
    rg_contract = canonical_window_constant_contract(
        steps_per_window     = steps_per_met,
        humidity_sampling    = :none,
        source_flux_sampling = :window_start_endpoint,
        include_flux_delta   = false,
    )

    # Stage to `.tmp`; the driver renames it to `bin_path` after every gate
    # passed and deletes it on failure, so a failed day never touches `bin_path`.
    tmp_path = bin_path * ".tmp"
    rm(tmp_path; force = true)
    writer = open_streaming_transport_binary(
        tmp_path, transport_grid, Nt, sample_window;
        FT = FT,
        dt_met_seconds       = settings.met_interval,
        half_dt_seconds      = settings.half_dt,
        steps_per_window     = steps_per_met,
        source_flux_sampling = rg_contract.source_flux_sampling,
        air_mass_sampling    = rg_contract.air_mass_sampling,
        flux_sampling        = rg_contract.flux_sampling,
        flux_kind            = rg_contract.flux_kind,
        humidity_sampling    = rg_contract.humidity_sampling,
        delta_semantics      = rg_contract.delta_semantics,
        mass_basis           = Symbol(settings.mass_basis),
        extra_header = _with_replay_record(Dict{String, Any}(
            "preprocessor"     => "preprocess_transport_binary.jl",
            "source_type"      => "era5_spectral",
            "target_type"      => "reduced_gaussian",
            "gaussian_number"  => grid.gaussian_number,
            "poisson_balanced" => true,
            "horizontal_balance" => balance_tag(LayerBalance()),
            "mass_fix_enabled" => settings.mass_fix_enable,
            "mass_fix_target_ps_dry_pa" => settings.target_ps_dry_pa,
            "mass_fix_qv_global_climatology" => settings.qv_global_climatology,
            "mass_fix_qv_mode" => settings.mass_basis == :dry ? "native_hourly_qv" : "global_qv_climatology",
            # Poisson fields via extra_header — _transport_common_header
            # doesn't accept these directly. Single source
            # of truth: the contract.
            "poisson_balance_target_scale"     => rg_contract.poisson_balance_target_scale,
            "poisson_balance_target_semantics" => rg_contract.poisson_balance_target_semantics,
        ), write_replay_on))

    bytes_per_window = writer.elems_per_window * sizeof(eltype(writer.pack_buffer))
    expected_total = writer.header_bytes + Nt * bytes_per_window
    @info @sprintf("  Output: %s (%.2f GB, %d windows, %d cells, %d faces)",
                   basename(bin_path), expected_total / 1e9, Nt, nc, nf)

    log_mass_fix_configuration(settings)
    @info "  Streaming: synthesize → balance → write (2-window sliding buffer)..."

    window_writer = ReducedGaussianBinaryWriter(
        writer,
        mass_basis_from_symbol(Symbol(settings.mass_basis));
        final_path = bin_path)
    ctx = RGSpectralUnifiedDriverContext(
        grid, settings, vertical, spec, substep_policy, date, next_day_hour0,
        steps_per_met, write_replay_on,
        0.0, 0.0, 0.0, 0, 0.0, 0.0, 0, (0, 0))
    run_unified_preprocessor_day!(
        UnifiedPreprocessorDay(nothing, workspace, window_contract, window_writer;
                               context = ctx);
        close_reader = false)

    if settings.mass_fix_enable
        day_offsets = @view ps_offsets[1:Nt]
        @info @sprintf("  Mass-fix offsets (Pa) min/max/mean: %+.3f / %+.3f / %+.3f",
                       minimum(day_offsets), maximum(day_offsets), sum(day_offsets) / Nt)
    end

    if write_replay_on
        @info @sprintf("  Write-time RG replay gate: max rel=%.3e abs=%.3e win=%d cell=%s",
                       ctx.worst_replay_rel, ctx.worst_replay_abs,
                       ctx.worst_replay_win, ctx.worst_replay_idx)
    end

    @info @sprintf("  Poisson balance summary: pre_raw=%.3e  post_proj=%.3e  post_raw=%.3e  max_cg_iter=%d",
                   ctx.worst_pre_raw, ctx.worst_post_proj,
                   ctx.worst_post_raw, ctx.worst_iter)

    actual = filesize(bin_path)
    @info @sprintf("  Done: %s (%.2f GB, %.1fs)", basename(bin_path),
                   actual / 1e9, time() - t_day)
    actual == expected_total ||
        @warn @sprintf("File size mismatch: expected %d bytes, got %d", expected_total, actual)

    return bin_path
end
