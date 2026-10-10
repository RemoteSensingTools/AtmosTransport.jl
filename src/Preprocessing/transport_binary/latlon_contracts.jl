# Shared structured lat-lon replay, Poisson-balance, and v4 writer helpers.

"""
    LLWorst

Type-stable shape of the LL positivity accumulator. Same role as `CSWorst`
but with `(i, j, k)` cell coordinates (no panel axis). The
`LatLonContract.worst` field is typed against this alias so accessor
calls inside the accumulator/summarizer chain stay statically typed.
"""
const LLWorst = @NamedTuple{ratio::Float64, direction::Symbol, win::Int,
                              location::NTuple{3, Int}}

"""
    next_day_merged_fields(next_day_hour0, date, grid, vertical, settings,
                           transform, merged, qv, ps_offsets)

Process the next day's hour-0 spectral and humidity fields so the current day's
final window can form forward deltas and carry a consistent mass-fix offset.
Returns copies of the merged `(m, am, bm, cm, qv)` fields (`qv` is `nothing`
unless `settings.include_qv`).

Returns `nothing` early when `next_day_hour0 === nothing` (no next-day data
was loaded, e.g. at the end of the processed range).
"""
function next_day_merged_fields(next_day_hour0,
                                date::Date,
                                grid::LatLonTargetGeometry,
                                vertical,
                                settings,
                                transform::SpectralTransformWorkspace,
                                merged::MergeWorkspace{FT},
                                qv::AbstractQVWorkspace{FT},
                                ps_offsets::AbstractVector{<:Real}) where FT
    next_day_hour0 === nothing && return nothing
    Nx = size(transform.sp, 1)
    Ny = size(transform.sp, 2)
    @info "  Computing next day hour 0 for last-window delta..."

    spectral_to_native_fields!(
        transform.m_arr, transform.am_arr, transform.bm_arr, transform.cm_arr, transform.sp,
        transform.u_cc, transform.v_cc, transform.u_stag, transform.v_stag, transform.dp,
        next_day_hour0.lnsp, next_day_hour0.vo, next_day_hour0.d,
        next_day_hour0.T, vertical.level_range, vertical.ab, grid, settings.half_dt,
        transform.P_buf, transform.fft_buf, transform.field_2d,
        transform.P_buf_t, transform.fft_buf_t, transform.fft_out_t,
        transform.u_spec_t, transform.v_spec_t, transform.field_2d_t,
        transform.bfft_plans)

    read_next_day_qv!(qv, date, settings, Nx, Ny, vertical.Nz_native)
    apply_mass_fix_if_needed!(qv, transform, grid, vertical, settings, ps_offsets, length(ps_offsets))
    apply_dry_basis_if_needed!(settings.mass_basis, transform, qv)
    merge_native_window!(merged, transform, qv, vertical, settings)

    return (m=copy(merged.m_merged),
            am=copy(merged.am_merged),
            bm=copy(merged.bm_merged),
            cm=copy(merged.cm_merged),
            qv=settings.include_qv ? copy(qv.qv_merged) : nothing)
end

"""
    fill_window_mass_tendency!(dm_dt_buf, storage, last_hour_next, win_idx, steps_per_window)

Fill the cell-mass target used by the Poisson horizontal-flux balance step.

The stored `am/bm/cm` fields are half-sweep transport amounts. A full Strang
substep applies the horizontal fluxes twice, so the Poisson target must be the
forward window mass difference divided by `2 * steps_per_window`.
"""
function fill_window_mass_tendency!(dm_dt_buf::Array{FT, 3},
                                    storage::WindowStorage{FT},
                                    last_hour_next,
                                    win_idx::Int,
                                    steps_per_window::Int) where FT
    Nt = length(storage.all_m)
    scale = poisson_balance_target_scale(steps_per_window, FT)

    if win_idx < Nt
        dm_dt_buf .= (storage.all_m[win_idx + 1] .- storage.all_m[win_idx]) .* scale
    elseif last_hour_next !== nothing
        dm_dt_buf .= (last_hour_next.m .- storage.all_m[win_idx]) .* scale
    else
        fill!(dm_dt_buf, zero(FT))
    end

    return nothing
end

"""
    verify_storage_continuity_ll!(storage, last_hour_next, steps_per_window, ::Type{FT})

Write-time replay gate for structured LL storage.
Iterates every window k and asserts

    m[k] − 2·steps·(∇·am + ∇·bm + ∂_k cm) ≈ m[k+1]    (k < Nt)
    m[Nt] − 2·steps·(∇·am + ∇·bm + ∂_k cm) ≈ last_hour_next.m   (k == Nt, if available)
    m[Nt] − 2·steps·(∇·am + ∇·bm + ∂_k cm) ≈ m[Nt]              (otherwise zero-tendency fallback)

to within a Poisson-balance tolerance floor derived from `FT` (roughly
`1e-10` for `Float64`, `1e-4` for `Float32`). Errors loudly with a per-window
diagnostic if the contract is violated — this was the gate that would have
caught the dry-basis Δb×pit closure bug before it reached the runtime.

Bypass with env var `ATMOSTR_NO_WRITE_REPLAY_CHECK=1` for diagnostic runs.
"""
function verify_storage_continuity_ll!(storage::WindowStorage{FT},
                                        last_hour_next,
                                        steps_per_window::Int,
                                        ::Type{FT}) where FT
    if !write_replay_check_enabled()
        @info "  Write-time replay gate SKIPPED (ATMOSTR_NO_WRITE_REPLAY_CHECK=1)"
        return nothing
    end
    Nt = length(storage.all_m)
    Nt == 0 && return nothing
    tol_rel = replay_tolerance(FT)
    div_scratch = Array{Float64}(undef, size(storage.all_m[1]))
    layout = structured_replay_layout()
    run_replay_gate(Nt; tol_rel=tol_rel,
                    summary_label="  Write-time replay gate",
                    failure_prefix="Write-time replay gate") do k
        m_next = if k < Nt
            storage.all_m[k + 1]
        elseif last_hour_next !== nothing
            last_hour_next.m
        else
            storage.all_m[k]
        end
        verify_window_continuity(layout, div_scratch,
                                 storage.all_m[k],
                                 storage.all_cm[k],
                                 m_next,
                                 steps_per_window,
                                 storage.all_am[k],
                                 storage.all_bm[k])
    end
    return nothing
end

"""
    verify_storage_contract_ll!(storage, last_hour_next, contract)

Run the typed LL window contract across balanced per-day storage. This is the
P2a production bridge: the legacy `apply_poisson_balance!` loop still owns the
numerics, but the post-balance write gate now uses `LatLonContract`'s typed
policy and accumulator state.

If `ATMOSTR_NO_WRITE_REPLAY_CHECK=1` is set, only the replay gate is skipped;
the positivity scan still runs so diagnostic binaries cannot silently bypass
the runtime CFL contract.
"""
function verify_storage_contract_ll!(storage::WindowStorage{FT},
                                      last_hour_next,
                                      contract) where FT
    Nt = length(storage.all_m)
    Nt == 0 && return contract

    write_replay_on = write_replay_check_enabled()
    write_replay_on ||
        @info "  Write-time replay gate SKIPPED (ATMOSTR_NO_WRITE_REPLAY_CHECK=1)"

    worst_rel = 0.0
    worst_abs = 0.0
    worst_win = 0
    worst_idx = (0, 0, 0)

    for win_idx in 1:Nt
        m_next = if win_idx < Nt
            storage.all_m[win_idx + 1]
        elseif last_hour_next !== nothing
            last_hour_next.m
        else
            storage.all_m[win_idx]
        end

        if write_replay_on
            diag = verify_window!((m_cur = storage.all_m[win_idx],
                                   am = storage.all_am[win_idx],
                                   bm = storage.all_bm[win_idx],
                                   cm = storage.all_cm[win_idx],
                                   m_next = m_next),
                                  contract, win_idx)
            if diag.replay.max_rel_err > worst_rel
                worst_rel = diag.replay.max_rel_err
                worst_abs = diag.replay.max_abs_err
                worst_win = win_idx
                worst_idx = diag.replay.worst_idx
            end
            update_accumulator!(contract, diag.positivity, win_idx)
        else
            positivity = verify_substep_positivity_ll!(storage.all_m[win_idx],
                                                        storage.all_am[win_idx],
                                                        storage.all_bm[win_idx],
                                                        storage.all_cm[win_idx];
                                                        cfl_limit =
                                                            contract.positivity_cfl_limit)
            update_accumulator!(contract, positivity, win_idx)
        end
    end

    if write_replay_on
        @info @sprintf("  Write-time LL replay gate: max rel=%.3e abs=%.3e win=%d cell=%s",
                       worst_rel, worst_abs, worst_win, worst_idx)
    end

    return contract
end

"""
    apply_poisson_balance!(storage, last_hour_next, steps_per_window, contract=nothing)

Close each stored window against its forward mass endpoint.

By default this applies only a column-integrated horizontal correction, then
diagnoses the vertical mass flux from the explicit endpoint mass tendency. This
keeps the ERA layer winds anchored to the spectral U/V fields while satisfying
the zero top/bottom `cm` replay contract.

`balance = LayerBalance()` (`[numerics] balance_mode = "per_layer"`) restores
the older per-layer Poisson correction for controlled comparisons.
"""
function apply_poisson_balance!(storage::WindowStorage{FT},
                                last_hour_next,
                                steps_per_window::Int,
                                contract = nothing;
                                balance::Union{Nothing, AbstractHorizontalBalance} = nothing) where FT
    Nx, Ny, Nz = size(storage.all_m[1])
    dm_dt_buf = Array{FT}(undef, Nx, Ny, Nz)
    div_scratch = Array{Float64}(undef, Nx, Ny, Nz)
    replay_layout = structured_replay_layout()

    apply_horizontal_balance = effective_horizontal_balance(balance, ColumnBalance()) isa LayerBalance
    poisson_ws = LLPoissonWorkspace(Nx, Ny)
    if apply_horizontal_balance
        @info "  Applying horizontal Poisson mass-flux balance (legacy opt-in)..."
    else
        @info "  Applying column mass-balance correction; diagnosing cm from endpoint mass tendency..."
    end

    worst_column_pre = 0.0
    worst_column_post = 0.0
    worst_column_delta = 0.0
    for win_idx in eachindex(storage.all_m)
        fill_window_mass_tendency!(dm_dt_buf, storage, last_hour_next, win_idx, steps_per_window)
        if apply_horizontal_balance
            balance_mass_fluxes!(storage.all_am[win_idx], storage.all_bm[win_idx], dm_dt_buf, poisson_ws)
            @views storage.all_bm[win_idx][:, 1, :] .= zero(FT)
            @views storage.all_bm[win_idx][:, Ny + 1, :] .= zero(FT)
        else
            col_diag = balance_column_mass_fluxes!(storage.all_am[win_idx],
                                                   storage.all_bm[win_idx],
                                                   storage.all_m[win_idx],
                                                   dm_dt_buf, poisson_ws)
            worst_column_pre = max(worst_column_pre, col_diag.max_pre_residual)
            worst_column_post = max(worst_column_post, col_diag.max_post_residual)
            worst_column_delta = max(worst_column_delta, col_diag.max_face_delta)
        end
        # Dry-basis fix (2026-04-22): use explicit-dm closure, not
        # the hybrid Δb×pit one. The Δb×pit closure assumes
        # dm[k] = dB[k] × Σ_k dm[k], which holds under moist hybrid coords
        # but is violated by ~27% under dry basis because qv[k] varies with
        # level. That mismatch caused the 0.75% day-boundary air_mass jump
        # observed on F64 probe; see plan39_reconnect.md memory entry.
        recompute_cm_from_dm_target!(replay_layout, div_scratch,
                                     storage.all_cm[win_idx], storage.all_m[win_idx], dm_dt_buf,
                                     storage.all_am[win_idx], storage.all_bm[win_idx])
        @views storage.all_cm[win_idx][:, :, 1] .= zero(FT)
        @views storage.all_cm[win_idx][:, :, Nz + 1] .= zero(FT)
    end

    if !apply_horizontal_balance
        @info @sprintf("  Column balance summary: pre=%.3e post=%.3e max_face_delta=%.3e kg",
                       worst_column_pre, worst_column_post, worst_column_delta)
    end

    # Write-time replay gate. Under the
    # `:window_constant` contract, starting from `storage.all_m[k]` and
    # integrating the stored fluxes (am, bm, cm) over one window via
    # palindrome continuity must reproduce `storage.all_m[k+1]` (or
    # `last_hour_next.m` for k=Nt). P2a routes production callers through the
    # typed contract so the positivity gate and run-level policy are active
    # too; the legacy helper remains for direct/debug callers.
    if contract === nothing
        verify_storage_continuity_ll!(storage, last_hour_next, steps_per_window, FT)
    else
        verify_storage_contract_ll!(storage, last_hour_next, contract)
    end
    @info "  Continuity closure complete for $(length(storage.all_m)) windows"

    return nothing
end

function apply_poisson_balance!(storage::WindowStorage{FT},
                                last_hour_next,
                                steps_schedule::Vector{Int},
                                contract,
                                substep_policy::SubstepSchedulePolicy;
                                balance::Union{Nothing, AbstractHorizontalBalance} = nothing,
                                write_replay_on::Bool = write_replay_check_enabled()) where FT
    contract === nothing &&
        throw(ArgumentError("adaptive LL Poisson balance requires a LatLonContract"))
    Nt = length(storage.all_m)
    length(steps_schedule) == Nt ||
        throw(ArgumentError("LL steps_schedule length $(length(steps_schedule)) != Nt=$(Nt)"))
    Nx, Ny, Nz = size(storage.all_m[1])
    dm_dt_buf = Array{FT}(undef, Nx, Ny, Nz)
    div_scratch = Array{Float64}(undef, Nx, Ny, Nz)
    replay_layout = structured_replay_layout()

    apply_horizontal_balance = effective_horizontal_balance(balance, ColumnBalance()) isa LayerBalance
    poisson_ws = LLPoissonWorkspace(Nx, Ny)
    if apply_horizontal_balance
        @info "  Applying horizontal Poisson mass-flux balance (legacy opt-in)..."
    else
        mode = substep_policy.adaptive_substeps ? "adaptive" : "fixed"
        @info "  Applying column mass-balance correction with $(mode) substep schedule..."
    end

    write_replay_on || @info "  Write-time replay gate SKIPPED (disabled for this run)"

    worst_column_pre = 0.0
    worst_column_post = 0.0
    worst_column_delta = 0.0
    worst_replay_rel = 0.0
    worst_replay_abs = 0.0
    worst_replay_win = 0
    worst_replay_idx = (0, 0, 0)

    for win_idx in 1:Nt
        steps = initial_substeps(substep_policy, steps_schedule[win_idx])
        old_steps = steps_schedule[win_idx]
        while true
            old_steps == steps ||
                rescale_substep_amounts!(
                    (storage.all_am[win_idx], storage.all_bm[win_idx]),
                    old_steps, steps)
            old_steps = steps
            steps_schedule[win_idx] = steps
            contract.steps_per_window = steps

            fill_window_mass_tendency!(dm_dt_buf, storage, last_hour_next,
                                       win_idx, steps)
            if apply_horizontal_balance
                balance_mass_fluxes!(storage.all_am[win_idx],
                                     storage.all_bm[win_idx],
                                     dm_dt_buf, poisson_ws)
                @views storage.all_bm[win_idx][:, 1, :] .= zero(FT)
                @views storage.all_bm[win_idx][:, Ny + 1, :] .= zero(FT)
            else
                col_diag = balance_column_mass_fluxes!(storage.all_am[win_idx],
                                                       storage.all_bm[win_idx],
                                                       storage.all_m[win_idx],
                                                       dm_dt_buf, poisson_ws)
                worst_column_pre = max(worst_column_pre, col_diag.max_pre_residual)
                worst_column_post = max(worst_column_post, col_diag.max_post_residual)
                worst_column_delta = max(worst_column_delta, col_diag.max_face_delta)
            end
            recompute_cm_from_dm_target!(replay_layout, div_scratch,
                                         storage.all_cm[win_idx],
                                         storage.all_m[win_idx], dm_dt_buf,
                                         storage.all_am[win_idx],
                                         storage.all_bm[win_idx])
            @views storage.all_cm[win_idx][:, :, 1] .= zero(FT)
            @views storage.all_cm[win_idx][:, :, Nz + 1] .= zero(FT)

            m_next = if win_idx < Nt
                storage.all_m[win_idx + 1]
            elseif last_hour_next !== nothing
                last_hour_next.m
            else
                storage.all_m[win_idx]
            end

            diag = if write_replay_on
                verify_window!((m_cur = storage.all_m[win_idx],
                                am = storage.all_am[win_idx],
                                bm = storage.all_bm[win_idx],
                                cm = storage.all_cm[win_idx],
                                m_next = m_next),
                               contract, win_idx)
            else
                positivity = verify_substep_positivity_ll!(
                    storage.all_m[win_idx], storage.all_am[win_idx],
                    storage.all_bm[win_idx], storage.all_cm[win_idx];
                    cfl_limit = contract.positivity_cfl_limit)
                (replay = nothing, positivity = positivity)
            end

            next_steps = next_substeps(substep_policy, steps,
                                       diag.positivity.ratio)
            if next_steps == steps
                update_accumulator!(contract, diag.positivity, win_idx)
                if write_replay_on &&
                        (worst_replay_win == 0 ||
                         diag.replay.max_rel_err > worst_replay_rel)
                    worst_replay_rel = diag.replay.max_rel_err
                    worst_replay_abs = diag.replay.max_abs_err
                    worst_replay_win = win_idx
                    worst_replay_idx = diag.replay.worst_idx
                end
                break
            end
            steps = next_steps
        end
    end

    set_contract_steps_schedule!(contract, steps_schedule)
    if !apply_horizontal_balance
        @info @sprintf("  Column balance summary: pre=%.3e post=%.3e max_face_delta=%.3e kg",
                       worst_column_pre, worst_column_post, worst_column_delta)
    end
    if write_replay_on
        @info @sprintf("  Write-time LL replay gate: max rel=%.3e abs=%.3e win=%d cell=%s",
                       worst_replay_rel, worst_replay_abs, worst_replay_win,
                       worst_replay_idx)
    end
    if !isempty(steps_schedule)
        @info @sprintf("  LL substep schedule: min=%d max=%d%s",
                       minimum(steps_schedule), maximum(steps_schedule),
                       all(==(first(steps_schedule)), steps_schedule) ?
                           " (constant)" : " (per-window)")
    end
    @info "  Continuity closure complete for $(Nt) windows"
    return nothing
end

"""
    compute_window_deltas!(merged, storage, win_idx, last_hour_next)

Form the forward-in-time `dam`, `dbm`, `dcm`, and `dm` payloads for one window.
"""
function compute_window_deltas!(merged::MergeWorkspace{FT},
                                storage::WindowStorage{FT},
                                win_idx::Int,
                                last_hour_next) where FT
    Nt = length(storage.all_m)

    if win_idx < Nt
        merged.dam_merged .= storage.all_am[win_idx + 1] .- storage.all_am[win_idx]
        merged.dbm_merged .= storage.all_bm[win_idx + 1] .- storage.all_bm[win_idx]
        merged.dcm_merged .= storage.all_cm[win_idx + 1] .- storage.all_cm[win_idx]
        merged.dm_merged  .= storage.all_m[win_idx + 1]  .- storage.all_m[win_idx]
    elseif last_hour_next !== nothing
        merged.dam_merged .= last_hour_next.am .- storage.all_am[win_idx]
        merged.dbm_merged .= last_hour_next.bm .- storage.all_bm[win_idx]
        merged.dcm_merged .= last_hour_next.cm .- storage.all_cm[win_idx]
        merged.dm_merged  .= last_hour_next.m  .- storage.all_m[win_idx]
    else
        fill!(merged.dam_merged, zero(FT))
        fill!(merged.dbm_merged, zero(FT))
        fill!(merged.dcm_merged, zero(FT))
        fill!(merged.dm_merged, zero(FT))
    end

    return nothing
end

function fill_qv_endpoints!(storage::WindowStorage{FT}, last_hour_next) where FT
    isempty(storage.all_qv_start) && return nothing
    Nt = length(storage.all_qv_start)

    for win_idx in 1:Nt-1
        storage.all_qv_end[win_idx] = copy(storage.all_qv_start[win_idx + 1])
    end

    if last_hour_next !== nothing && hasproperty(last_hour_next, :qv) && last_hour_next.qv !== nothing
        storage.all_qv_end[Nt] = copy(last_hour_next.qv)
    else
        storage.all_qv_end[Nt] = copy(storage.all_qv_start[Nt])
    end

    return nothing
end

"""
    write_window!(io, win_idx, storage, settings, merged, last_hour_next) -> Int64

Write one window's payload blocks to the output stream in v4 on-disk order.
"""
function write_window!(io::IO,
                       win_idx::Int,
                       storage::WindowStorage{FT},
                       settings,
                       merged::MergeWorkspace{FT},
                       last_hour_next) where FT
    bytes_written = Int64(0)
    bytes_written += write_array!(io, storage.all_m[win_idx])
    bytes_written += write_array!(io, storage.all_am[win_idx])
    bytes_written += write_array!(io, storage.all_bm[win_idx])
    bytes_written += write_array!(io, storage.all_cm[win_idx])
    bytes_written += write_array!(io, storage.all_ps[win_idx])
    if settings.include_qv
        bytes_written += write_array!(io, storage.all_qv_start[win_idx])
        bytes_written += write_array!(io, storage.all_qv_end[win_idx])
    end

    compute_window_deltas!(merged, storage, win_idx, last_hour_next)
    bytes_written += write_array!(io, merged.dam_merged)
    bytes_written += write_array!(io, merged.dbm_merged)
    bytes_written += write_array!(io, merged.dcm_merged)
    bytes_written += write_array!(io, merged.dm_merged)

    # TM5 convection sections (order must match
    # _transport_push_optional_sections! in TransportBinary.jl:557-578).
    if settings.tm5_convection_enable
        bytes_written += write_array!(io, storage.all_entu[win_idx])
        bytes_written += write_array!(io, storage.all_detu[win_idx])
        bytes_written += write_array!(io, storage.all_entd[win_idx])
        bytes_written += write_array!(io, storage.all_detd[win_idx])
    end
    if _settings_include_surface(settings)
        bytes_written += write_array!(io, storage.all_pblh[win_idx])
        bytes_written += write_array!(io, storage.all_t2m[win_idx])
        bytes_written += write_array!(io, storage.all_ustar[win_idx])
        bytes_written += write_array!(io, storage.all_hflux[win_idx])
    end

    return bytes_written
end

# ===========================================================================
# Per-window LL transport-binary contract surface.
#
# Mirrors `cubed_sphere_contracts.jl` for the structured lat-lon
# topology. Today the LL preprocessor calls only `verify_window_continuity_ll`
# (the replay gate); there is no analogue of `verify_substep_positivity_cs!`
# for LL fluxes, so an LL binary that drives a cell mass negative mid-sweep
# can pass replay and only break later inside the runtime CFL scan. This
# surface closes that asymmetry: LL gets the same per-substep positivity
# gate, the same worst-window accumulator, and the same
# `require_substep_positivity` escape-hatch policy as CS. The gate is
# intentionally NOT yet wired into the LL `process_day` orchestrator.
#
# LL array shapes (confirmed from `mass_support.jl` and the storage struct
# in `latlon_workspaces.jl`):
#
#     m  :: (Nx, Ny, Nz)
#     am :: (Nx + 1, Ny, Nz)     # periodic in x: am[1,j,k] == am[Nx+1,j,k]
#     bm :: (Nx, Ny + 1, Nz)     # bm[:, 1, :] = bm[:, Ny+1, :] = 0 at poles
#     cm :: (Nx, Ny, Nz + 1)     # cm[:, :, 1] = cm[:, :, Nz+1] = 0 (TOA/sfc)
#
# So the positivity kernel is identical to a single CS panel modulo the
# missing panel loop. The diagnostic NamedTuple uses `(i, j, k)` instead
# of `(panel, i, j, k)`.
#
# LL/RG positivity probe note (DESIGN.md "Open design questions"):
# the design asks for an explicit yes-with-gate or no-with-stub answer to
# "do LL fluxes have a substep-positivity contract?". The gate IS shipped
# here because (a) the kernel is essentially free to implement once the CS
# kernel exists and (b) running it on a representative ERA5 day is the
# cheapest way to answer the question once-and-for-all when the
# unified driver wires the call into `process_day`. Until then, the gate
# exists as a contract type with full test coverage but is not invoked by
# the production orchestrator, so it cannot regress any current path.
# ===========================================================================

"""
    verify_substep_positivity_ll!(m, am, bm, cm; cfl_limit = 0.95)

Per-substep horizontal+vertical positivity scan for a structured LL
window. Mirrors `verify_substep_positivity_cs!` but operates on a single
LL window (no panel dimension).

For every cell `(i, j, k)`:
  1. `m > 0`. A non-positive cell mass is reported with `ratio = Inf`
     and short-circuits this cell's CFL ratios; the runtime divides by
     `m` and would `Inf`/`NaN` otherwise.
  2. Outgoing mass per substep, per direction, ≤ `cfl_limit * m`.

`NaN`/`Inf` cell mass and `NaN`/`Inf` fluxes are flagged as `ratio = Inf`
(see CS round-2 fix in `cubed_sphere_contracts.jl`).

Returns `(direction, ratio, location, ok)` with:
  - `direction :: Union{Symbol, Nothing}` — `:x` / `:y` / `:z` /
    `nothing` (no inspection).
  - `ratio :: Float64` — worst `outgoing / m`.
  - `location :: NTuple{3, Int}` — `(i, j, k)`.
  - `ok :: Bool` — `ratio ≤ cfl_limit`.
"""
function verify_substep_positivity_ll!(m::AbstractArray{FT, 3},
                                        am::AbstractArray,
                                        bm::AbstractArray,
                                        cm::AbstractArray;
                                        cfl_limit::Real = 0.95) where FT
    _validate_cfl_limit(cfl_limit, "verify_substep_positivity_ll!")
    Nx, Ny, Nz = size(m)
    size(am) == (Nx + 1, Ny, Nz) ||
        error("verify_substep_positivity_ll!: am shape $(size(am)) " *
              "incompatible with m $(size(m)); expected ($(Nx + 1), $(Ny), $(Nz)).")
    size(bm) == (Nx, Ny + 1, Nz) ||
        error("verify_substep_positivity_ll!: bm shape $(size(bm)) " *
              "incompatible with m $(size(m)); expected ($(Nx), $(Ny + 1), $(Nz)).")
    size(cm) == (Nx, Ny, Nz + 1) ||
        error("verify_substep_positivity_ll!: cm shape $(size(cm)) " *
              "incompatible with m $(size(m)); expected ($(Nx), $(Ny), $(Nz + 1)).")

    worst_dir = nothing
    worst_ratio = 0.0
    worst_loc = (0, 0, 0)
    for (dir, F_lo_view, F_hi_view) in (
        (:x, view(am, 1:Nx,     1:Ny,     1:Nz),
             view(am, 2:Nx + 1, 1:Ny,     1:Nz)),
        (:y, view(bm, 1:Nx,     1:Ny,     1:Nz),
             view(bm, 1:Nx,     2:Ny + 1, 1:Nz)),
        (:z, view(cm, 1:Nx,     1:Ny,     1:Nz),
             view(cm, 1:Nx,     1:Ny,     2:Nz + 1)),
    )
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            mi = m[i, j, k]
            fl = F_lo_view[i, j, k]
            fh = F_hi_view[i, j, k]
            # NaN-aware short-circuit: matches the CS round-2 fix so
            # `NaN`-mass cells, `NaN`/`Inf` fluxes, and `m ≤ 0` are all
            # surfaced consistently. NaN comparisons return `false`, so
            # `mi <= 0` alone would let `NaN`-mass cells slip through.
            if !isfinite(mi) || mi <= zero(FT) ||
               !isfinite(fl) || !isfinite(fh)
                if !isinf(worst_ratio)
                    worst_ratio = Inf
                    worst_dir = dir
                    worst_loc = (i, j, k)
                end
                continue
            end
            outgoing = max(zero(FT), -fl) + max(zero(FT), fh)
            ratio = outgoing / mi
            if ratio > worst_ratio
                worst_ratio = ratio
                worst_dir = dir
                worst_loc = (i, j, k)
            end
        end
    end
    return (direction = worst_dir, ratio = Float64(worst_ratio),
            location = worst_loc, ok = worst_ratio <= cfl_limit)
end

"""
    verify_ll_window_contract!(m_cur, am, bm, cm, m_next, steps_per_window, win_idx;
                               replay_tol, positivity_cfl_limit = 0.95,
                               div_scratch = nothing)

Single canonical per-window LL binary contract check. Runs the replay
gate (errors on failure) followed by the per-substep positivity scan
(`verify_substep_positivity_ll!`, returns a diagnostic).

`div_scratch` may be pre-allocated by the caller (workspace-owned
scratch) to suppress the per-window `Array{Float64}` the
default-allocating `verify_window_continuity_ll` would otherwise
produce. Default `nothing` → allocate locally.

Returns `(; replay, positivity)`. Positivity is non-fatal here —
callers aggregate the worst window across the loop and pass it to
`summarize_ll_positivity_status` where the run-level
`require_substep_positivity` policy decides whether to error or warn.
"""
function verify_ll_window_contract!(m_cur::AbstractArray{FT, 3},
                                     am::AbstractArray,
                                     bm::AbstractArray,
                                     cm::AbstractArray,
                                     m_next::AbstractArray,
                                     steps_per_window::Integer,
                                     win_idx::Integer;
                                     replay_tol::Real,
                                     positivity_cfl_limit::Real = 0.95,
                                     div_scratch::Union{Nothing, AbstractArray{Float64, 3}} = nothing) where FT
    _validate_replay_tol(replay_tol, "verify_ll_window_contract!")
    _validate_cfl_limit(positivity_cfl_limit, "verify_ll_window_contract!")
    if div_scratch === nothing
        div_scratch = Array{Float64}(undef, size(m_cur))
    else
        size(div_scratch) == size(m_cur) ||
            error("verify_ll_window_contract!: div_scratch shape " *
                  "$(size(div_scratch)) != m_cur shape $(size(m_cur)).")
    end
    replay = verify_window_continuity(structured_replay_layout(),
                                       div_scratch, m_cur, cm, m_next,
                                       steps_per_window, am, bm)
    replay.max_rel_err <= replay_tol ||
        error("Write-time replay gate FAILED for LL window $(win_idx): " *
              "rel=$(replay.max_rel_err) > tol=$(replay_tol) at cell " *
              "$(replay.worst_idx) (abs=$(replay.max_abs_err) kg). Stored LL " *
              "fluxes do not integrate to the target mass endpoint under " *
              "palindrome continuity.")
    positivity = verify_substep_positivity_ll!(m_cur, am, bm, cm;
                                                cfl_limit = positivity_cfl_limit)
    return (replay = replay, positivity = positivity)
end

"""
    init_ll_positivity_accumulator() -> NamedTuple

Zero-valued state for accumulating the worst per-window LL positivity
diagnostic across a preprocessing loop. Pair with
`update_ll_positivity_accumulator`.
"""
init_ll_positivity_accumulator() = LLWorst((0.0, :none, 0, (0, 0, 0)))

"""
    update_ll_positivity_accumulator(worst, diag, win_idx) -> NamedTuple

Return an updated LL accumulator from a fresh per-window diagnostic.
"""
function update_ll_positivity_accumulator(worst::LLWorst, diag::NamedTuple,
                                            win_idx::Integer)
    diag.ratio > worst.ratio || return worst
    return LLWorst((Float64(diag.ratio),
                    diag.direction === nothing ? :none : Symbol(diag.direction),
                    Int(win_idx),
                    NTuple{3, Int}(diag.location)))
end

"""
    summarize_ll_positivity_status(worst; cfl_limit, steps_per_window,
                                   require_substep_positivity = true,
                                   quarantine_path = nothing)

Post-loop summary helper for the LL positivity accumulator. Logs the
worst-window outcome, and if it exceeds `cfl_limit`:
* deletes `quarantine_path` (if given) so a downstream consumer cannot
  pick up the half-written binary;
* errors when `require_substep_positivity = true`, otherwise warns.

The error/warn message includes a recommended `steps_per_window` value
that would satisfy the gate, computed from the observed worst ratio.
The "no representable rescue" branch from CS round-3 is mirrored here.
"""
function summarize_ll_positivity_status(worst::LLWorst;
                                          cfl_limit::Real,
                                          steps_per_window::Integer,
                                          require_substep_positivity::Bool = true,
                                          quarantine_path::Union{Nothing, AbstractString} = nothing)
    msg = @sprintf("max outgoing/m=%.3f dir=%s win=%d cell=%s (limit=%.2f)",
                   worst.ratio, worst.direction, worst.win,
                   worst.location, cfl_limit)
    if worst.ratio <= cfl_limit
        @info "  Per-substep positivity gate: $msg"
        return nothing
    end

    max_safe_factor       = typemax(Int) ÷ max(Int(steps_per_window), 1)
    useful_recommendation = isfinite(worst.ratio) &&
                             isfinite(cfl_limit) && cfl_limit > 0 &&
                             worst.ratio / cfl_limit <= max_safe_factor

    detail = if useful_recommendation
        recommended = max(Int(steps_per_window),
                          ceil(Int, worst.ratio / cfl_limit) * Int(steps_per_window))
        "Per-substep positivity contract violated: $msg. " *
        "The binary stores fluxes that violate positivity at the recorded " *
        "`steps_per_window=$(steps_per_window)`. Re-run with " *
        "`steps_per_window=$(recommended)` (or higher) on the source " *
        "preprocessing config, or set `[numerics].require_substep_positivity = false` " *
        "to suppress."
    else
        "Per-substep positivity contract violated, and no representable " *
        "`steps_per_window` can rescue it: $msg. " *
        "The observed ratio is `Inf`/`NaN` (m<=0, NaN-mass/flux, or Inf-flux) " *
        "or finite but pathologically large; the configured `cfl_limit` may " *
        "also be non-positive. Either the source data has been corrupted " *
        "upstream of preprocessing, the pressure-fixer / endpoint-balance " *
        "pass drove a cell negative, or `[numerics].positivity_cfl_limit` is " *
        "misconfigured (must satisfy `0 < cfl_limit <= 1`). Investigate the " *
        "source field at the reported cell location, or set " *
        "`[numerics].require_substep_positivity = false` to record the " *
        "violation as a warning and keep the binary for diagnostic inspection."
    end
    if require_substep_positivity
        quarantine_path === nothing || (isfile(quarantine_path) && rm(quarantine_path; force = true))
        error(detail)
    else
        @warn detail
    end
    return nothing
end

# ---------------------------------------------------------------------------
# `LatLonContract{FT}` — typed Axis-3 concrete for the structured LL
# topology. Mirrors `CubedSphereContract{FT}` so the unified driver
# can dispatch the same `verify_window!` / `update_accumulator!` /
# `summarize_status!` trait surface on either topology.
# ---------------------------------------------------------------------------

"""
    LatLonContract{FT} <: AbstractWindowContract{LatLonTargetGeometry, FT}

Typed nominal owning an LL preprocessor's per-window gate policy and
worst-window positivity accumulator. Construction validates
`positivity_cfl_limit ∈ (0, 1]` and `steps_per_window ≥ 1` so an
invalid TOML value fails before any window runs.
"""
mutable struct LatLonContract{FT} <:
                AbstractWindowContract{LatLonTargetGeometry, FT}
    replay_tol                  :: Float64
    positivity_cfl_limit        :: Float64
    require_substep_positivity  :: Bool
    steps_per_window            :: Int
    worst                       :: LLWorst
    # Lazy scratch (codex round-2): allocated on first `verify_window!`
    # call from the window's mass shape and reused thereafter.
    # Eliminates the per-window `Array{Float64}(undef, size(m_cur))`
    # allocation that the wrapper would otherwise produce.
    _div_scratch                :: Union{Nothing, Array{Float64, 3}}

    function LatLonContract{FT}(;
                                 replay_tol::Real,
                                 positivity_cfl_limit::Real = 0.95,
                                 require_substep_positivity::Bool = true,
                                 steps_per_window::Integer) where FT
        isfinite(replay_tol) && replay_tol > 0 ||
            error("LatLonContract: replay_tol = $(replay_tol); " *
                  "must be finite and > 0 (Inf would silently disable replay; " *
                  "NaN would fail every window late).")
        isfinite(positivity_cfl_limit) && 0 < positivity_cfl_limit ≤ 1 ||
            error("LatLonContract: positivity_cfl_limit = " *
                  "$(positivity_cfl_limit); must be in (0, 1].")
        steps_per_window ≥ 1 ||
            error("LatLonContract: steps_per_window = " *
                  "$(steps_per_window); must be ≥ 1.")
        return new(Float64(replay_tol),
                   Float64(positivity_cfl_limit),
                   require_substep_positivity,
                   Int(steps_per_window),
                   init_ll_positivity_accumulator(),
                   nothing)
    end
end

@inline contract_replay_tolerance(c::LatLonContract)   = c.replay_tol
@inline contract_cfl_limit(c::LatLonContract)          = c.positivity_cfl_limit
@inline contract_require_positivity(c::LatLonContract) = c.require_substep_positivity

function verify_window!(window, contract::LatLonContract, win_idx::Integer)
    # Lazy-allocate `_div_scratch` on first call (or reallocate on a
    # shape change). Subsequent calls reuse the buffer.
    m_shape = size(window.m_cur)
    if contract._div_scratch === nothing || size(contract._div_scratch) != m_shape
        contract._div_scratch = Array{Float64}(undef, m_shape)
    end
    return verify_ll_window_contract!(window.m_cur, window.am, window.bm,
                                       window.cm, window.m_next,
                                       contract.steps_per_window, Int(win_idx);
                                       replay_tol           = contract.replay_tol,
                                       positivity_cfl_limit = contract.positivity_cfl_limit,
                                       div_scratch          = contract._div_scratch)
end

function update_accumulator!(contract::LatLonContract, positivity_diag,
                              win_idx::Integer)
    contract.worst = update_ll_positivity_accumulator(contract.worst,
                                                       positivity_diag,
                                                       Int(win_idx))
    return nothing
end

function summarize_status!(contract::LatLonContract;
                            quarantine_path::Union{Nothing, AbstractString} = nothing)
    return summarize_ll_positivity_status(contract.worst;
                                            cfl_limit = contract.positivity_cfl_limit,
                                            steps_per_window = contract.steps_per_window,
                                            require_substep_positivity =
                                                contract.require_substep_positivity,
                                            quarantine_path = quarantine_path)
end

"""
    write_day_binary!(bin_path, header_json, storage, settings, merged, last_hour_next)

Write the padded header and all window payloads for one daily binary file.
Returns the total number of bytes written.
"""
function write_day_binary!(bin_path::String,
                           header_json,
                           storage::WindowStorage{FT},
                           settings,
                           merged::MergeWorkspace{FT},
                           last_hour_next) where FT
    @info "  Writing binary..."
    bytes_written = Int64(0)

    open(bin_path, "w") do io
        hdr_buf = zeros(UInt8, HEADER_SIZE)
        copyto!(hdr_buf, 1, Vector{UInt8}(header_json), 1, length(header_json))
        write(io, hdr_buf)
        bytes_written += HEADER_SIZE

        for win_idx in eachindex(storage.all_m)
            bytes_written += write_window!(io, win_idx, storage, settings, merged, last_hour_next)
        end

        flush(io)
    end

    return bytes_written
end
