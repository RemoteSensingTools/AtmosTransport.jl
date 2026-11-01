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

    spectral_to_native_fields!(transform, next_day_hour0.lnsp, next_day_hour0.vo,
                                next_day_hour0.d, next_day_hour0.T, vertical, grid, settings)

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
    scale = FT(poisson_balance_target(:substep_mass_amount, [max(Int(steps_per_window), 1)]).scale)

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
    apply_poisson_balance!(storage, last_hour_next, steps_schedule, contract, substep_policy;
                           write_replay_on=true, weights=MassWeightedColumn())

Balance each LL window against its forward mass endpoint, diagnose explicit-dm
vertical closure, and apply the replay and positivity contract. Fixed schedules
also supply one step count per window.
"""
function apply_poisson_balance!(storage::WindowStorage{FT},
                                last_hour_next,
                                steps_schedule::Vector{Int},
                                contract,
                                substep_policy::SubstepSchedulePolicy;
                                write_replay_on::Bool = true,
                                weights::AbstractColumnWeights = MassWeightedColumn(),
                                rebuild_rates = nothing,
                                poisson_workspace = nothing,
                                window_offset::Int = 0) where FT
    contract === nothing &&
        throw(ArgumentError("adaptive LL Poisson balance requires a LatLonContract"))
    Nt = length(storage.all_m)
    length(steps_schedule) == Nt ||
        throw(ArgumentError("LL steps_schedule length $(length(steps_schedule)) != Nt=$(Nt)"))
    Nx, Ny, Nz = size(storage.all_m[1])
    dm_dt_buf = Array{FT}(undef, Nx, Ny, Nz)
    div_scratch = Array{Float64}(undef, Nx, Ny, Nz)
    replay_layout = structured_replay_layout()

    poisson_ws = poisson_workspace === nothing ? LLPoissonWorkspace(Nx, Ny) : poisson_workspace
    mode = substep_policy.adaptive_substeps ? "adaptive" : "fixed"
    @info "  Applying column mass-balance correction with $(mode) substep schedule..."

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
            if rebuild_rates !== nothing
                rebuild_rates(win_idx,steps)
            else
                old_steps == steps ||
                rescale_substep_amounts!(
                    (storage.all_am[win_idx], storage.all_bm[win_idx]),
                    old_steps, steps)
            end
            old_steps = steps
            steps_schedule[win_idx] = steps
            contract.steps_per_window = steps

            fill_window_mass_tendency!(dm_dt_buf, storage, last_hour_next,
                                       win_idx, steps)
            col_diag = balance_column_mass_fluxes!(storage.all_am[win_idx],
                                                   storage.all_bm[win_idx],
                                                   storage.all_m[win_idx],
                                                   dm_dt_buf, poisson_ws; weights)
            worst_column_pre = max(worst_column_pre, col_diag.max_pre_residual)
            worst_column_post = max(worst_column_post, col_diag.max_post_residual)
            worst_column_delta = max(worst_column_delta, col_diag.max_face_delta)
            recompute_cm_from_dm_target!(replay_layout, div_scratch,
                                         storage.all_cm[win_idx],
                                         storage.all_m[win_idx], dm_dt_buf,
                                         storage.all_am[win_idx],
                                         storage.all_bm[win_idx];
                                         residual_weight = ColumnResidualWeight(weights))
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
                               contract, window_offset+win_idx)
            else
                positivity = verify_substep_positivity_ll!(
                    storage.all_m[win_idx], storage.all_am[win_idx],
                    storage.all_bm[win_idx], storage.all_cm[win_idx];
                    cfl_limit = contract.positivity_cfl_limit, m_next)
                (replay = nothing, positivity = positivity)
            end

            next_steps = next_substeps(substep_policy, steps,
                                       diag.positivity.ratio)
            if next_steps == steps
                update_accumulator!(contract, diag.positivity, window_offset+win_idx)
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
    @info @sprintf("  Column balance summary: pre=%.3e post=%.3e max_face_delta=%.3e kg",
                   worst_column_pre, worst_column_post, worst_column_delta)
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

    if settings_include_surface(settings)
        bytes_written += write_array!(io, storage.all_pblh[win_idx])
        bytes_written += write_array!(io, storage.all_t2m[win_idx])
        bytes_written += write_array!(io, storage.all_ustar[win_idx])
        bytes_written += write_array!(io, storage.all_hflux[win_idx])
    end

    return bytes_written
end

# ===========================================================================
# Per-window LL transport-binary contract surface (mirrors
# `cubed_sphere_contracts.jl`): the replay gate and the per-substep positivity
# gate, with the same worst-window accumulator and `require_substep_positivity`
# policy as cubed sphere.
#
# LL array shapes (`latlon_workspaces.jl`):
#
#     m  :: (Nx, Ny, Nz)
#     am :: (Nx + 1, Ny, Nz)     # periodic in x: am[1,j,k] == am[Nx+1,j,k]
#     bm :: (Nx, Ny + 1, Nz)     # bm[:, 1, :] = bm[:, Ny+1, :] = 0 at poles
#     cm :: (Nx, Ny, Nz + 1)     # cm[:, :, 1] = cm[:, :, Nz+1] = 0 (TOA/sfc)
#
# so an LL window is one cubed-sphere panel block without the panel axis, and
# the positivity gate is the cubed-sphere one.
# ===========================================================================

"""
    verify_substep_positivity_ll!(m, am, bm, cm; cfl_limit = DEFAULT_POSITIVITY_CFL_LIMIT, m_next = nothing)

Per-substep positivity contract of a structured lat-lon window, the same as
the cubed-sphere one (`verify_substep_positivity_cs!`, here on one block). One
stored flux amount is one application in the runtime palindrome
`X-Y-Z-Z-Y-X`, so each direction acts twice per substep. For every cell
`(i, j, k)`:

  1. `m > 0` (and `m_next > 0` when given); non-finite masses or fluxes and
     non-positive masses are reported with `ratio = Inf`.
  2. The palindrome outflow `2 (out_x + out_y + out_z)` must not exceed
     `cfl_limit * m_ref`, with `m_ref = min(m, m_next)` when `m_next` is given
     and `m` otherwise. `out_d` is the cell's outflow through its two `d` faces
     for one application.

This bounds the outflow of a whole substep by the mass the cell has at either
end of the window. The earlier lat-lon gate checked each direction's single
application against `m` alone, so it admitted substeps the cubed-sphere gate
rejects.

Returns `(direction, ratio, ratio_xy, ratio_z, location, ok)` as the
cubed-sphere gate does, with `location = (i, j, k)`.
"""
function verify_substep_positivity_ll!(m::AbstractArray{FT, 3},
                                        am::AbstractArray,
                                        bm::AbstractArray,
                                        cm::AbstractArray;
                                        cfl_limit::Real = DEFAULT_POSITIVITY_CFL_LIMIT,
                                        m_next::Union{Nothing, AbstractArray} = nothing) where FT
    validate_cfl_limit(cfl_limit, "verify_substep_positivity_ll!")
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
    m_next === nothing || size(m_next) == size(m) ||
        error("verify_substep_positivity_ll!: m_next shape $(size(m_next)) != m shape $(size(m)).")
    d = verify_substep_positivity_cs!((m,), (am,), (bm,), (cm,); cfl_limit,
                                      m_next = m_next === nothing ? nothing : (m_next,))
    return (direction = d.direction, ratio = d.ratio, ratio_xy = d.ratio_xy,
            ratio_z = d.ratio_z, location = Base.tail(d.location), ok = d.ok)
end

"""
    verify_ll_window_contract!(m_cur, am, bm, cm, m_next, steps_per_window, win_idx;
                               replay_tol, positivity_cfl_limit = DEFAULT_POSITIVITY_CFL_LIMIT,
                               replay_scratch = nothing)

Single canonical per-window LL binary contract check. Runs the replay
gate (errors on failure) followed by the per-substep positivity scan
(`verify_substep_positivity_ll!`, returns a diagnostic).

`replay_scratch` (a `ReplayScratch` of the mass shape) may be pre-allocated
by the caller (workspace-owned scratch) to suppress the per-window
allocation. Default `nothing` → allocate locally.

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
                                     positivity_cfl_limit::Real = DEFAULT_POSITIVITY_CFL_LIMIT,
                                     replay_scratch::Union{Nothing, ReplayScratch} = nothing) where FT
    validate_replay_tol(replay_tol, "verify_ll_window_contract!")
    validate_cfl_limit(positivity_cfl_limit, "verify_ll_window_contract!")
    if replay_scratch === nothing
        replay_scratch = ReplayScratch(size(m_cur))
    else
        size(replay_scratch) == size(m_cur) ||
            error("verify_ll_window_contract!: replay_scratch shape " *
                  "$(size(replay_scratch)) != m_cur shape $(size(m_cur)).")
    end
    replay = verify_window_continuity(structured_replay_layout(),
                                       replay_scratch, m_cur, cm, m_next,
                                       steps_per_window, am, bm)
    replay.max_rel_err <= replay_tol ||
        error("Write-time replay gate FAILED for LL window $(win_idx): " *
              "rel=$(replay.max_rel_err) > tol=$(replay_tol) at cell " *
              "$(replay.worst_idx) (abs=$(replay.max_abs_err) kg). Stored LL " *
              "fluxes do not integrate to the target mass endpoint under " *
              "palindrome continuity.")
    positivity = verify_substep_positivity_ll!(m_cur, am, bm, cm;
                                                cfl_limit = positivity_cfl_limit, m_next)
    return (replay = replay, positivity = positivity)
end

"""
    init_ll_positivity_accumulator() -> NamedTuple

Zero-valued state for accumulating the worst per-window LL positivity
diagnostic across a preprocessing loop. Pair with
`update_ll_positivity_accumulator`.
"""
init_ll_positivity_accumulator() = init_positivity_accumulator(Val(3))

"""
    update_ll_positivity_accumulator(worst, diag, win_idx) -> NamedTuple

Return an updated LL accumulator from a fresh per-window diagnostic.
"""
function update_ll_positivity_accumulator(worst::LLWorst, diag::NamedTuple,
                                            win_idx::Integer)
    return update_positivity_accumulator(worst, diag, win_idx)
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
    return summarize_positivity_status(worst; budget_label="palindrome outgoing/m_ref",
                                        cfl_limit, steps_per_window,
                                        require_substep_positivity, quarantine_path)
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
    # Eliminates the per-window replay-scratch allocation that the
    # wrapper would otherwise produce.
    _replay_scratch             :: Union{Nothing, ReplayScratch{Array{Float64, 3}}}

    function LatLonContract{FT}(;
                                 replay_tol::Real,
                                 positivity_cfl_limit::Real = DEFAULT_POSITIVITY_CFL_LIMIT,
                                 require_substep_positivity::Bool = true,
                                 steps_per_window::Integer) where FT
        validate_replay_tol(replay_tol, "LatLonContract")
        validate_cfl_limit(positivity_cfl_limit, "LatLonContract")
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

function verify_window!(window, contract::LatLonContract, win_idx::Integer)
    # Lazy-allocate `_replay_scratch` on first call (or reallocate on a
    # shape change). Subsequent calls reuse the buffer.
    m_shape = size(window.m_cur)
    if contract._replay_scratch === nothing || size(contract._replay_scratch) != m_shape
        contract._replay_scratch = ReplayScratch(m_shape)
    end
    return verify_ll_window_contract!(window.m_cur, window.am, window.bm,
                                       window.cm, window.m_next,
                                       contract.steps_per_window, Int(win_idx);
                                       replay_tol           = contract.replay_tol,
                                       positivity_cfl_limit = contract.positivity_cfl_limit,
                                       replay_scratch       = contract._replay_scratch)
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
