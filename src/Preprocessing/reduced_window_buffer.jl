# Reduced-Gaussian spectral preprocessing: two-slot window buffer, window workspace, ingest/drain/flush with adaptive substeps.
# Split from reduced_transport_helpers.jl (refactor phase 4); included by Preprocessing.jl in this order.

"""
    SlidingWindowBuffer{FT}

Two-slot circular buffer for streaming RG preprocessing.  Only two windows'
worth of `(m, hflux, cm, ps)` are kept in memory at any time, enabling
O160/O320 binary generation without OOM.
"""
struct SlidingWindowBuffer{FT}
    m     :: Vector{Matrix{FT}}     # length-2 circular buffer
    hflux :: Vector{Matrix{FT}}
    cm    :: Vector{Matrix{FT}}
    ps    :: Vector{Vector{FT}}
end

function allocate_sliding_window_buffer(nc::Int, nf::Int, Nz::Int, ::Type{FT}) where FT
    SlidingWindowBuffer{FT}(
        [zeros(FT, nc, Nz)     for _ in 1:2],
        [zeros(FT, nf, Nz)     for _ in 1:2],
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
    copyto!(buf.cm[slot],    merged.cm_merged)
    buf.ps[slot] .= ps_vec   # broadcast handles Float64→FT conversion in-place
    return nothing
end

mutable struct ReducedGaussianSpectralWindowWorkspace{FT, TW, MW, BW, QW, CL}
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
    cached = cache === nothing ? nothing : get(cache, cache_key, nothing)
    t_cl = time()
    if cached !== nothing
        cL = cached
        total_entries = cL.row_ptr[end] - 1
        avg_neighbors = total_entries / nc
        @info @sprintf("  Compressed Laplacian: reused %d unique entries (avg %.1f neighbors/cell) from %d faces (%.0f× compression, %.2fs)",
                       total_entries, avg_neighbors, nf,
                       nf / max(total_entries, 1), time() - t_cl)
        return cL
    end

    cL = build_compressed_laplacian(work.face_left, work.face_right, nc)
    cache === nothing || (cache[cache_key] = cL)
    total_entries = cL.row_ptr[end] - 1
    avg_neighbors = total_entries / nc
    @info @sprintf("  Compressed Laplacian: %d unique entries (avg %.1f neighbors/cell) from %d faces (%.0f× compression, %.2fs)",
                   total_entries, avg_neighbors, nf,
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
    merged = allocate_reduced_merge_workspace(grid, Nz_native, Nz, FT)
    buf = allocate_sliding_window_buffer(nc, nf, Nz, FT)
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

    return ReducedGaussianSpectralWindowWorkspace{
        FT, typeof(work), typeof(merged), typeof(buf), typeof(qv_ws), typeof(cL)}(
            work, merged, buf, qv_ws, thermo_path, ps_offsets, cL,
            hflux_work, m_cur_work, m_next_work, cm_work, div_scratch,
            dm_target_work, fill(exact_steps_per_window(settings.met_interval,
                                                         settings.dt),
                                  spec.n_times),
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

function drain_ready_windows!(workspace::ReducedGaussianSpectralWindowWorkspace{FT},
                              contract,
                              win_idx::Int,
                              steps_per_window::Int;
                              write_replay_on::Bool,
                              substep_policy) where FT
    _ = steps_per_window
    steps = initial_substeps(substep_policy, workspace.steps_schedule[win_idx])
    old_steps = workspace.steps_schedule[win_idx]
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
                               workspace.buf, workspace.cur,
                               workspace.buf.m[workspace.nxt],
                               workspace.work, workspace.cL, steps)
        contract_diag = _verify_rg_balanced_window!(
            contract, workspace.m_cur_work, workspace.hflux_work, workspace.cm_work,
            workspace.m_next_work, win_idx; write_replay_on = write_replay_on,
            accumulate = false)
        next_steps = next_substeps(substep_policy, steps,
                                   contract_diag.positivity.ratio)
        next_steps == steps && break
        steps = next_steps
    end
    update_accumulator!(contract, contract_diag.positivity, win_idx)
    ready = ReadyWindow{ReducedGaussianTargetGeometry, FT}(
        win_idx,
        (m = workspace.buf.m[workspace.cur],
         hflux = workspace.buf.hflux[workspace.cur],
         cm = workspace.buf.cm[workspace.cur],
         ps = workspace.buf.ps[workspace.cur]))
    workspace.cur, workspace.nxt = workspace.nxt, workspace.cur
    return (ready = ready, balance = diag, contract = contract_diag)
end

function flush_final_windows!(workspace::ReducedGaussianSpectralWindowWorkspace{FT},
                              contract,
                              win_idx::Int,
                              steps_per_window::Int;
                              write_replay_on::Bool,
                              substep_policy,
                              m_next = workspace.buf.m[workspace.cur]) where FT
    _ = steps_per_window
    steps = initial_substeps(substep_policy, workspace.steps_schedule[win_idx])
    old_steps = workspace.steps_schedule[win_idx]
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
                               workspace.work, workspace.cL, steps)
        contract_diag = _verify_rg_balanced_window!(
            contract, workspace.m_cur_work, workspace.hflux_work, workspace.cm_work,
            workspace.m_next_work, win_idx; write_replay_on = write_replay_on,
            accumulate = false)
        next_steps = next_substeps(substep_policy, steps,
                                   contract_diag.positivity.ratio)
        next_steps == steps && break
        steps = next_steps
    end
    update_accumulator!(contract, contract_diag.positivity, win_idx)
    ready = ReadyWindow{ReducedGaussianTargetGeometry, FT}(
        win_idx,
        (m = workspace.buf.m[workspace.cur],
         hflux = workspace.buf.hflux[workspace.cur],
         cm = workspace.buf.cm[workspace.cur],
         ps = workspace.buf.ps[workspace.cur]))
    return (ready = ready, balance = diag, contract = contract_diag)
end
