# Reduced-Gaussian spectral preprocessing: two-slot window buffer, window workspace, ingest/drain/flush with adaptive substeps.
# Split from reduced_transport_helpers.jl (refactor phase 4); included by ReducedGaussian.jl in this order.

"""
    SlidingWindowBuffer{FT}

Two-slot circular buffer for streaming RG preprocessing.  Only two windows'
worth of `(m, hflux, cm, ps)` are kept in memory at any time, enabling
O160/O320 binary generation without OOM.
"""
struct SlidingWindowBuffer{FT}
    m     :: Vector{Matrix{FT}}     # length-2 circular buffer
    hflux :: Vector{Matrix{FT}}
    fidelity_samples :: Vector{Matrix{FT}}   # (fidelity samples, Nz)
    cm    :: Vector{Matrix{FT}}
    ps    :: Vector{Vector{FT}}
end

function allocate_sliding_window_buffer(nc::Int, nf::Int, Nz::Int, ::Type{FT};
                                        fidelity_samples::Int) where FT
    SlidingWindowBuffer{FT}(
        [zeros(FT, nc, Nz)     for _ in 1:2],
        [zeros(FT, nf, Nz)     for _ in 1:2],
        [zeros(FT, fidelity_samples, Nz) for _ in 1:2],
        [zeros(FT, nc, Nz + 1) for _ in 1:2],
        [zeros(FT, nc)         for _ in 1:2],
    )
end

"""
    fill_buffer_slot!(buf, slot, merged, ps_vec, FT)

Copy merged results into the given slot (1 or 2) of the sliding buffer.
"""
function fill_buffer_slot!(buf::SlidingWindowBuffer{FT},
                           slot::Int,
                           merged::ReducedMergeWorkspace{FT},
                           ps_vec::AbstractVector) where FT
    copyto!(buf.m[slot],     merged.m_merged)
    copyto!(buf.hflux[slot], merged.hflux_merged)
    copyto!(buf.fidelity_samples[slot], merged.fidelity_merged)
    buf.ps[slot] .= ps_vec   # broadcast handles Float64→FT conversion in-place
    return nothing
end

mutable struct ReducedGaussianSpectralWindowWorkspace{FT, TW, MW, BW, QW, CL, CW}
    work           :: TW
    merged         :: MW
    buf            :: BW
    qv_ws          :: QW
    thermo_path    :: String
    ps_offsets     :: Vector{Float64}
    cL             :: CL
    hflux_work     :: Matrix{Float64}
    m_cur_work     :: Matrix{Float64}
    m_next_work    :: Matrix{Float64}
    cm_work        :: Matrix{Float64}
    div_scratch    :: Matrix{Float64}
    dm_target_work :: Matrix{Float64}
    steps_schedule :: Vector{Int}
    balance_weights :: CW          # column-balance weights on the output levels
    fidelity_capture :: Ref{Any}
    dt_met_seconds :: Float64
    cur            :: Int
    nxt            :: Int
end

_rg_compressed_laplacian_cache_key(grid::ReducedGaussianTargetGeometry) =
    Symbol("rg_compressed_laplacian_", grid.gaussian_number)

function _get_or_build_rg_compressed_laplacian!(cache,
                                                grid::ReducedGaussianTargetGeometry,
                                                work::ReducedTransformWorkspace,
                                                nc::Int,
                                                nf::Int)
    cache_key = _rg_compressed_laplacian_cache_key(grid)
    t_cl = time()
    cL, reused = cached_or_build!(cache, cache_key) do
        build_compressed_laplacian(work.face_left, work.face_right, nc)
    end
    total_entries = cL.row_ptr[end] - 1
    avg_neighbors = total_entries / nc
    @info @sprintf("  Compressed Laplacian: %s%d unique entries (avg %.1f neighbors/cell) from %d faces (%.0f× compression, %.2fs)",
                   reused ? "reused " : "", total_entries, avg_neighbors, nf,
                   nf / max(total_entries, 1), time() - t_cl)
    return cL
end

function allocate_window_workspace(grid::ReducedGaussianTargetGeometry,
                                   settings,
                                   vertical,
                                   spec,
                                   date::Date,
                                   ::Type{FT};
                                   cache = nothing) where FT
    mesh = grid.mesh
    nc = ncells(mesh)
    nf = nfaces(mesh)
    Nz_native = vertical.Nz_native
    Nz = vertical.Nz

    work = allocate_reduced_transform_workspace(grid, spec.T, Nz_native)
    ns = nsamples(work.fidelity_sampler)
    merged = allocate_reduced_merge_workspace(grid, Nz_native, Nz, FT; fidelity_samples = ns)
    buf = allocate_sliding_window_buffer(nc, nf, Nz, FT; fidelity_samples = ns)
    ps_offsets = zeros(Float64, spec.n_times + 1)   # the last entry: the next day's 00 UTC

    thermo_path = ""
    qv_ws = nothing
    if settings.mass_basis == :dry
        date_str = Dates.format(date, "yyyymmdd")
        thermo_path = joinpath(settings.thermo_dir,
                               "era5_thermo_ml_$(date_str).nc")
        isfile(thermo_path) ||
            error("Thermo file not found for dry-basis conversion: $thermo_path")
        qv_ws = allocate_reduced_qv_workspace(grid, Nz_native, thermo_path;
                                              settings = settings)
        @info "  Dry-basis: QV from $thermo_path → $(qv_ws.Nx_ll)×$(qv_ws.Ny_ll) LL → $(nc) RG cells"
    end

    cL = _get_or_build_rg_compressed_laplacian!(cache, grid, work, nc, nf)

    hflux_work = zeros(Float64, nf, Nz)
    m_cur_work = zeros(Float64, nc, Nz)
    m_next_work = zeros(Float64, nc, Nz)
    cm_work = zeros(Float64, nc, Nz + 1)
    div_scratch = zeros(Float64, nc, Nz)
    dm_target_work = zeros(Float64, nc, Nz)

    balance_weights = spectral_column_weights(grid, settings, vertical)

    return ReducedGaussianSpectralWindowWorkspace{
        FT, typeof(work), typeof(merged), typeof(buf), typeof(qv_ws), typeof(cL),
        typeof(balance_weights)}(
            work, merged, buf, qv_ws, thermo_path, ps_offsets, cL,
            hflux_work, m_cur_work, m_next_work, cm_work, div_scratch,
            dm_target_work, fill(exact_steps_per_window(settings.met_interval,
                                                         settings.dt),
                                  spec.n_times),
            balance_weights,
            Ref{Any}(nothing),
            Float64(settings.met_interval),
            1, 2)
end

function ingest_window!(workspace::ReducedGaussianSpectralWindowWorkspace,
                        slot::Int,
                        win_idx::Int,
                        hour::Int,
                        spec,
                        grid::ReducedGaussianTargetGeometry,
                        vertical,
                        settings)
    t0 = time()
    synthesize_and_merge_window!(workspace.work, workspace.merged, hour, spec,
                                 grid, vertical, settings,
                                 workspace.ps_offsets, win_idx;
                                 qv_ws = workspace.qv_ws,
                                 thermo_path = workspace.thermo_path)
    fill_buffer_slot!(workspace.buf, slot, workspace.merged, workspace.work.sp)
    return time() - t0
end

"""
    _verify_rg_stored_window!(contract, workspace, m_next, win_idx; write_replay_on, accumulate)

Run the RG window contract on the arrays the writer stores: the current
buffer slot's `m`, `hflux` and `cm` in the output float type and the `m_next`
endpoint, not the Float64 balance work arrays they were rounded from.
"""
function _verify_rg_stored_window!(contract,
                                   workspace::ReducedGaussianSpectralWindowWorkspace,
                                   m_next::AbstractMatrix,
                                   win_idx::Int;
                                   write_replay_on::Bool,
                                   accumulate::Bool)
    slot = workspace.cur
    return _verify_rg_balanced_window!(contract, workspace.buf.m[slot],
                                       workspace.buf.hflux[slot], workspace.buf.cm[slot],
                                       m_next, win_idx;
                                       write_replay_on, accumulate)
end

function _finish_reduced_window!(workspace::ReducedGaussianSpectralWindowWorkspace{FT},
                                  contract, win_idx::Int, m_next;
                                  write_replay_on::Bool, substep_policy) where FT
    steps = initial_substeps(substep_policy, workspace.steps_schedule[win_idx])
    old_steps = workspace.steps_schedule[win_idx]
    source_steps = old_steps
    workspace.fidelity_capture[] = capture_faceindexed_flux_samples(
        workspace.work.fidelity_sampler.faces, workspace.buf.hflux[workspace.cur],
        workspace.buf.fidelity_samples[workspace.cur])
    diag = nothing
    contract_diag = nothing
    while true
        old_steps == steps ||
            rescale_substep_amounts!((workspace.buf.hflux[workspace.cur],),
                                     old_steps, steps)
        old_steps = steps
        workspace.steps_schedule[win_idx] = steps
        contract.steps_per_window = steps
        diag = balance_window!(workspace.hflux_work, workspace.m_cur_work,
                               workspace.m_next_work, workspace.cm_work,
                               workspace.div_scratch, workspace.dm_target_work,
                               workspace.buf, workspace.cur, m_next,
                               workspace.work, workspace.cL, steps;
                               weights = workspace.balance_weights)
        contract_diag = _verify_rg_stored_window!(
            contract, workspace, m_next, win_idx;
            write_replay_on = write_replay_on, accumulate = false)
        next_steps = next_substeps(substep_policy, steps,
                                   contract_diag.positivity.ratio)
        next_steps == steps && break
        steps = next_steps
    end
    update_accumulator!(contract, contract_diag.positivity, win_idx)
    amount_seconds = 0.5 * workspace.dt_met_seconds / steps
    initial_seconds = 0.5 * workspace.dt_met_seconds / source_steps
    accepted_capture = scale_captured_flux_samples(
        workspace.fidelity_capture[], amount_seconds / initial_seconds)
    flux_evidence = faceindexed_wind_evidence(
        ReducedGaussianTargetGeometry, accepted_capture,
        workspace.buf.hflux[workspace.cur]; method = "reduced_spectral_uv_dp",
        amount_seconds)
    omega_evidence = NoSourceOmegaEvidence{
        ReducedGaussianTargetGeometry,FT,SourceHasNoOmega}(SourceHasNoOmega())
    ready = ReadyWindow{ReducedGaussianTargetGeometry, FT}(
        win_idx,
        (m = workspace.buf.m[workspace.cur],
         hflux = workspace.buf.hflux[workspace.cur],
         cm = workspace.buf.cm[workspace.cur],
         ps = workspace.buf.ps[workspace.cur]),
        FluxFidelityEvidenceBundle(flux_evidence, omega_evidence,
                                   NoColumnClosureRecord()))
    return (ready = ready, balance = diag, contract = contract_diag)
end

function drain_ready_windows!(workspace::ReducedGaussianSpectralWindowWorkspace,
                              contract, win_idx::Int, steps_per_window::Int;
                              write_replay_on::Bool, substep_policy)
    result = _finish_reduced_window!(workspace, contract, win_idx,
                                     workspace.buf.m[workspace.nxt];
                                     write_replay_on, substep_policy)
    workspace.cur, workspace.nxt = workspace.nxt, workspace.cur
    return result
end

function flush_final_windows!(workspace::ReducedGaussianSpectralWindowWorkspace,
                              contract, win_idx::Int, steps_per_window::Int;
                              write_replay_on::Bool, substep_policy,
                              m_next=workspace.buf.m[workspace.cur])
    return _finish_reduced_window!(workspace, contract, win_idx, m_next;
                                   write_replay_on, substep_policy)
end
