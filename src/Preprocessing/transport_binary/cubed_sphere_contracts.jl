# Cubed-sphere replay helpers shared by spectral and regrid CS preprocessors.

"""
    CSWorst

Type-stable shape of the CS positivity accumulator. Pin the field
types here so accesses inside `update_cs_positivity_accumulator` /
`summarize_cs_positivity_status` and the `CubedSphereContract.worst`
field are statically typed (the Julia-style reviewer flagged the
`worst :: NamedTuple` field as dynamically dispatched).
"""
const CSWorst = @NamedTuple{ratio::Float64, direction::Symbol, win::Int,
                              location::NTuple{4, Int}}

"""
    fill_cs_window_mass_tendency!(dm_panels, m_cur, m_next, steps_per_window)

Fill the CS Poisson-balance target for one window.

`m_cur` and `m_next` are explicit endpoint masses. The stored CS horizontal
fluxes are half-sweep amounts under Strang splitting, so the target is
`(m_next - m_cur) / (2 * steps_per_window)` per panel cell.
"""
@inline function fill_cs_window_mass_tendency!(dm_panels::NTuple{NP, <:AbstractArray{FT, 3}},
                                               m_cur::NTuple{NP, <:AbstractArray{FT, 3}},
                                               m_next::NTuple{NP, <:AbstractArray{FT, 3}},
                                               steps_per_window::Int) where {FT, NP}
    inv_two_steps = one(FT) / FT(2 * steps_per_window)
    for p in 1:NP
        @inbounds for idx in eachindex(dm_panels[p])
            dm_panels[p][idx] = (m_next[p][idx] - m_cur[p][idx]) * inv_two_steps
        end
    end
    return nothing
end

"""
    convert_cs_mass_target_to_delta!(m_target, m_cur)

Convert an in-place CS endpoint target into the on-disk `dm` payload.

The CS writer stores forward endpoint differences. Call this only after all
balance and replay checks that still need the absolute target endpoint.
"""
@inline function convert_cs_mass_target_to_delta!(m_target::NTuple{NP, <:AbstractArray{FT, 3}},
                                                  m_cur::NTuple{NP, <:AbstractArray{FT, 3}}) where {FT, NP}
    for p in 1:NP
        @inbounds for idx in eachindex(m_target[p])
            m_target[p][idx] -= m_cur[p][idx]
        end
    end
    return nothing
end

"""
    verify_write_replay_cs!(m_cur, am, bm, cm, m_next, steps_per_window, tol_rel, win_idx;
                            div_scratch = nothing)

Run the CS write-time replay gate for one window and return its diagnostic.

The check integrates the stored panel-local fluxes from `m_cur` under the
runtime palindrome-continuity contract and verifies that the result matches the
explicit endpoint `m_next`. A failure here means the binary would produce a
runtime day-boundary or window-boundary mass inconsistency.

`div_scratch` may be pre-allocated by the caller (workspace-owned
scratch or contract-owned lazy scratch) so the gate
doesn't allocate the panel-shared `div_h` per call. Default
`nothing` → allocate locally. Shape must match `size(m_cur[1])`.
"""
function verify_write_replay_cs!(m_cur::NTuple{NP, <:AbstractArray{FT, 3}},
                                 am::NTuple{NP, <:AbstractArray},
                                 bm::NTuple{NP, <:AbstractArray},
                                 cm::NTuple{NP, <:AbstractArray},
                                 m_next::NTuple{NP, <:AbstractArray},
                                 steps_per_window::Int,
                                 tol_rel::Real,
                                 win_idx::Int;
                                 div_scratch::Union{Nothing, AbstractArray{Float64, 3}} = nothing) where {FT, NP}
    _validate_replay_tol(tol_rel, "verify_write_replay_cs!")
    NP >= 1 ||
        error("verify_write_replay_cs!: requires at least one panel.")
    panel_shape = size(m_cur[1])
    if div_scratch === nothing
        div_scratch = Array{Float64}(undef, panel_shape)
    else
        size(div_scratch) == panel_shape ||
            error("verify_write_replay_cs!: div_scratch shape " *
                  "$(size(div_scratch)) != panel shape $(panel_shape).")
    end
    layout = structured_replay_layout()
    worst_rel = 0.0
    worst_abs = 0.0
    worst_idx = (0, 0, 0, 0)
    for p in 1:NP
        diag_p = verify_window_continuity(layout, div_scratch,
                                           m_cur[p], cm[p], m_next[p],
                                           steps_per_window, am[p], bm[p])
        if diag_p.max_rel_err > worst_rel
            worst_rel = diag_p.max_rel_err
            worst_abs = diag_p.max_abs_err
            worst_idx = (p, diag_p.worst_idx...)
        end
    end
    diag = (max_abs_err = worst_abs, max_rel_err = worst_rel, worst_idx = worst_idx)
    diag.max_rel_err <= tol_rel ||
        error("Write-time replay gate FAILED for CS window $(win_idx): " *
              "rel=$(diag.max_rel_err) > tol=$(tol_rel) at cell $(diag.worst_idx) " *
              "(abs=$(diag.max_abs_err) kg). Stored CS fluxes do not integrate to " *
              "the target mass endpoint under palindrome continuity.")
    return diag
end

"""
    verify_substep_positivity_cs!(m, am, bm, cm; cfl_limit = 0.95,
                                  halo_width = 0, m_next = nothing)

Verify the per-substep horizontal+vertical positivity contract that the runtime's
`_cs_static_palindrome_subcycle_count` depends on. For every interior cell on every panel:

  1. The cell air mass itself must be positive (`m > 0`), a preprocessing
     invariant enforced here (the runtime CFL scan skips such cells). A
     non-positive cell mass is reported with `ratio = Inf` regardless of flux
     magnitude.
  2. The combined Strang-palindrome outgoing budget
     `2 * (out_x + out_y + out_z)` must not exceed `cfl_limit * m_ref`, where
     `m_ref = min(m, m_next)` when `m_next` is supplied by a caller that wants
     endpoint tightening, and `m_ref = m` otherwise. The factor of 2 is
     required because the CS runtime applies the
     direction sequence `X-Y-Z-Z-Y-X` for every met substep.

Returns a NamedTuple `(direction, ratio, ratio_xy, ratio_z, location, ok)`:
* `direction :: Union{Symbol, Nothing}` — the dominant contributor among
  `:x`, `:y`, `:z`, or `nothing` when no cell was inspected.
* `ratio :: Float64` — worst observed palindrome outgoing budget over the
  reference mass, or `Inf` if any inspected cell had invalid mass or flux.
* `ratio_xy :: Float64` — worst observed horizontal-only palindrome outgoing
  budget over the reference mass.
* `ratio_z :: Float64` — worst observed vertical-only palindrome outgoing
  budget over the reference mass.
* `location :: NTuple{4, Int}` — `(panel, i, j, k)` of the worst cell.
* `ok :: Bool` — `true` iff `ratio <= cfl_limit`.

The replay gate (`verify_write_replay_cs!`) only checks endpoint continuity. A
binary that drives a cell mass negative mid-sweep can still pass replay because
the cell re-fills from inflow before the window ends — but the runtime cannot
recover. This gate is the actual contract the runtime depends on.

`halo_width` defaults to `0` (panel arrays are stored unhaloed at preprocess
time); pass `> 0` to scan only the interior of a haloed buffer.
"""
function verify_substep_positivity_cs!(m::NTuple{NP, <:AbstractArray{FT, 3}},
                                       am::NTuple{NP, <:AbstractArray},
                                       bm::NTuple{NP, <:AbstractArray},
                                       cm::NTuple{NP, <:AbstractArray};
                                       cfl_limit::Real = 0.95,
                                       halo_width::Integer = 0,
                                       m_next::Union{Nothing, NTuple{NP, <:AbstractArray}} = nothing) where {FT, NP}
    _validate_cfl_limit(cfl_limit, "verify_substep_positivity_cs!")
    Hp = Int(halo_width)
    # All buffers share the same interior extent `(Nc, Nc, Nz)`; derive from m.
    Nc = size(m[1], 1) - 2Hp
    Nz = size(m[1], 3)
    iL = Hp + 1
    iH = Hp + Nc
    worst_dir = nothing
    worst_ratio = 0.0
    worst_ratio_xy = 0.0
    worst_ratio_z = 0.0
    worst_loc = (0, 0, 0, 0)
    for p in 1:NP
        m_p = m[p]
        m_next_p = m_next === nothing ? nothing : m_next[p]
        m_int = view(m_p, iL:iH, iL:iH, 1:Nz)
        m_next_int = m_next_p === nothing ? nothing :
                     view(m_next_p, iL:iH, iL:iH, 1:Nz)
        am_lo = view(am[p], iL    :iH,     iL:iH,     1:Nz)
        am_hi = view(am[p], iL + 1:iH + 1, iL:iH,     1:Nz)
        bm_lo = view(bm[p], iL:iH,     iL    :iH,     1:Nz)
        bm_hi = view(bm[p], iL:iH,     iL + 1:iH + 1, 1:Nz)
        cm_lo = view(cm[p], iL:iH, iL:iH, 1    :Nz)
        cm_hi = view(cm[p], iL:iH, iL:iH, 2:Nz + 1)
        for k in 1:Nz, j in 1:Nc, i in 1:Nc
            mi = m_int[i, j, k]
            mi_next = m_next_int === nothing ? mi : m_next_int[i, j, k]
            ax_l = am_lo[i, j, k]
            ax_h = am_hi[i, j, k]
            by_l = bm_lo[i, j, k]
            by_h = bm_hi[i, j, k]
            cz_l = cm_lo[i, j, k]
            cz_h = cm_hi[i, j, k]
            if !isfinite(mi) || mi <= zero(FT) ||
               !isfinite(mi_next) || mi_next <= zero(FT) ||
               !isfinite(ax_l) || !isfinite(ax_h) ||
               !isfinite(by_l) || !isfinite(by_h) ||
               !isfinite(cz_l) || !isfinite(cz_h)
                if !isinf(worst_ratio)
                    worst_ratio = Inf
                    worst_ratio_xy = Inf
                    worst_ratio_z = Inf
                    worst_dir = :x
                    worst_loc = (p, i, j, k)
                end
                continue
            end
            out_x = max(zero(FT), -ax_l) + max(zero(FT), ax_h)
            out_y = max(zero(FT), -by_l) + max(zero(FT), by_h)
            out_z = max(zero(FT), -cz_l) + max(zero(FT), cz_h)
            m_ref = min(mi, mi_next)
            outgoing_xy = FT(2) * (out_x + out_y)
            outgoing_z = FT(2) * out_z
            outgoing = outgoing_xy + outgoing_z
            ratio_xy = outgoing_xy / m_ref
            ratio_z = outgoing_z / m_ref
            ratio = outgoing / m_ref
            worst_ratio_xy = max(worst_ratio_xy, Float64(ratio_xy))
            worst_ratio_z = max(worst_ratio_z, Float64(ratio_z))
            if ratio > worst_ratio
                worst_ratio = ratio
                worst_dir = out_x >= out_y && out_x >= out_z ? :x :
                            out_y >= out_z ? :y : :z
                worst_loc = (p, i, j, k)
            end
        end
    end
    return (direction = worst_dir, ratio = Float64(worst_ratio),
            ratio_xy = Float64(worst_ratio_xy),
            ratio_z = Float64(worst_ratio_z),
            location = worst_loc, ok = worst_ratio <= cfl_limit)
end

"""
    verify_cs_window_contract!(m_cur, am, bm, cm, m_next, steps_per_window, win_idx;
                               replay_tol, positivity_cfl_limit = 0.95, halo_width = 0)

Single canonical per-window CS binary contract check. Runs the replay gate
(`verify_write_replay_cs!`, errors on failure) followed by the per-substep
positivity scan (`verify_substep_positivity_cs`, returns a diagnostic). Every
CS-producing preprocessor (spectral, regrid, GEOS-native) should call this so
no path can silently skip a gate.

Returns `(; replay, positivity)` with both diagnostics. Positivity is non-fatal
here — callers aggregate the worst window and pass it to
`summarize_cs_positivity_status` after the loop, where the run-level
`require_substep_positivity` policy decides whether to error or warn.
"""
function verify_cs_window_contract!(m_cur::NTuple{NP, <:AbstractArray{FT, 3}},
                                    am::NTuple{NP, <:AbstractArray},
                                    bm::NTuple{NP, <:AbstractArray},
                                    cm::NTuple{NP, <:AbstractArray},
                                    m_next::NTuple{NP, <:AbstractArray},
                                    steps_per_window::Int,
                                    win_idx::Int;
                                    replay_tol::Real,
                                    positivity_cfl_limit::Real = 0.95,
                                    halo_width::Integer = 0,
                                    div_scratch::Union{Nothing, AbstractArray{Float64, 3}} = nothing) where {FT, NP}
    _validate_replay_tol(replay_tol, "verify_cs_window_contract!")
    _validate_cfl_limit(positivity_cfl_limit, "verify_cs_window_contract!")
    replay = verify_write_replay_cs!(m_cur, am, bm, cm, m_next,
                                     steps_per_window, replay_tol, win_idx;
                                     div_scratch = div_scratch)
    positivity = verify_substep_positivity_cs!(m_cur, am, bm, cm;
                                               cfl_limit = positivity_cfl_limit,
                                               halo_width = halo_width,
                                               m_next = m_next)
    return (replay = replay, positivity = positivity)
end

"""
    init_cs_positivity_accumulator() -> NamedTuple

Zero-valued state for accumulating the worst per-window positivity diagnostic
across a preprocessing loop. Pair with `update_cs_positivity_accumulator!`.
"""
init_cs_positivity_accumulator() = CSWorst((0.0, :none, 0, (0, 0, 0, 0)))

"""
    update_cs_positivity_accumulator(worst, diag, win_idx) -> NamedTuple

Return an updated accumulator from a fresh per-window diagnostic.
"""
function update_cs_positivity_accumulator(worst::CSWorst, diag::NamedTuple,
                                            win_idx::Int)
    diag.ratio > worst.ratio || return worst
    return CSWorst((Float64(diag.ratio),
                    diag.direction === nothing ? :none : Symbol(diag.direction),
                    Int(win_idx),
                    NTuple{4, Int}(diag.location)))
end

"""
    summarize_cs_positivity_status(worst; cfl_limit, steps_per_window,
                                   require_substep_positivity = true,
                                   quarantine_path = nothing)

Post-loop summary helper. Logs the worst-window outcome, and if it exceeds
`cfl_limit`:
* deletes `quarantine_path` (if given) so a downstream consumer cannot pick up
  the half-written binary;
* errors when `require_substep_positivity = true`, otherwise warns.

The error message includes a recommended `steps_per_window` value that would
satisfy the gate, computed from the observed worst ratio.
"""
function summarize_cs_positivity_status(worst::CSWorst;
                                        cfl_limit::Real,
                                        steps_per_window::Int,
                                        require_substep_positivity::Bool = true,
                                        quarantine_path::Union{Nothing, AbstractString} = nothing)
    msg = @sprintf("max palindrome outgoing/m_ref=%.3f dir=%s win=%d cell=%s (limit=%.2f)",
                   worst.ratio, worst.direction, worst.win,
                   worst.location, cfl_limit)
    if worst.ratio <= cfl_limit
        @info "  Per-substep positivity gate: $msg"
        return nothing
    end

    # A useful integer step recommendation requires that (a) the observed
    # ratio is finite, (b) the configured `cfl_limit` is a sensible positive
    # finite divisor, and (c) `ceil(Int, ratio / cfl_limit) * steps_per_window`
    # fits in `Int`. Any of these failing — `Inf`/`NaN` ratio from m<=0 or
    # NaN-mass/flux, an invalid `cfl_limit = 0` that produces `Inf` from the
    # divide, or a finite-but-pathologically-large ratio like `1e308` — would
    # throw `InexactError` from `ceil(Int, ...)` BEFORE the intended
    # error/warn path runs, which would re-break the
    # `require_substep_positivity = false` escape hatch exactly when an
    # operator needs it most. Branch *before* the rescue arithmetic.
    max_safe_factor       = typemax(Int) ÷ max(steps_per_window, 1)
    useful_recommendation = isfinite(worst.ratio) &&
                             isfinite(cfl_limit) && cfl_limit > 0 &&
                             worst.ratio / cfl_limit <= max_safe_factor

    detail = if useful_recommendation
        recommended = max(steps_per_window,
                          ceil(Int, worst.ratio / cfl_limit) * steps_per_window)
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

# ===========================================================================
# Typed CS contract concrete.
#
# `CubedSphereContract{FT}` wraps the existing CS gate state (policy fields
# + worst-window accumulator) in a typed nominal that participates in the
# unified `AbstractWindowContract{G, FT}` dispatch surface. The struct holds
# the policy at construction time and validates
# `positivity_cfl_limit` in its inner constructor so an invalid TOML value
# errors before any window runs.
#
# The trait-surface methods (`verify_window!`, `update_accumulator!`,
# `summarize_status!`) delegate to the existing NamedTuple-based helpers
# above so the per-window math is bit-exact identical to today's path —
# the typed surface is additive scaffolding for the unified driver.
# ===========================================================================

"""
    CubedSphereContract{FT} <: AbstractWindowContract{CubedSphereTargetGeometry, FT}

Typed nominal owning a CS preprocessor's per-window gate policy and
worst-window positivity accumulator.

Fields:

  - `replay_tol::Float64` — relative replay tolerance.
  - `positivity_cfl_limit::Float64` — per-substep positivity CFL gate.
    Must satisfy `0 < limit ≤ 1`; validated at construction.
  - `require_substep_positivity::Bool` — whether `summarize_status!`
    errors (`true`) or warns (`false`) on a positivity violation.
  - `steps_per_window::Int` — for the recommended-steps message in the
    summary's escape-hatch detail. Must be ≥ 1.
  - `halo_width::Int` — passed through to `verify_substep_positivity_cs!`.
  - `worst::NamedTuple` — mutable accumulator (initially zero).

Construct with explicit kwargs; defaults match the CS round-2/round-3
production policy.
"""
mutable struct CubedSphereContract{FT} <:
                AbstractWindowContract{CubedSphereTargetGeometry, FT}
    replay_tol                  :: Float64
    positivity_cfl_limit        :: Float64
    require_substep_positivity  :: Bool
    steps_per_window            :: Int
    halo_width                  :: Int
    worst                       :: CSWorst
    # Lazy scratch (codex round-2): allocated on first `verify_window!`
    # call from the window's panel shape and reused thereafter. Eliminates
    # the `Array{Float64}(undef, panel_shape)` per-window allocation the
    # original path produced inside `verify_window_continuity_cs`.
    _div_scratch                :: Union{Nothing, Array{Float64, 3}}

    function CubedSphereContract{FT}(;
                                      replay_tol::Real,
                                      positivity_cfl_limit::Real = 0.95,
                                      require_substep_positivity::Bool = true,
                                      steps_per_window::Integer,
                                      halo_width::Integer = 0) where FT
        isfinite(replay_tol) && replay_tol > 0 ||
            error("CubedSphereContract: replay_tol = $(replay_tol); " *
                  "must be finite and > 0 (Inf would silently disable replay; " *
                  "NaN would fail every window late).")
        isfinite(positivity_cfl_limit) && 0 < positivity_cfl_limit ≤ 1 ||
            error("CubedSphereContract: positivity_cfl_limit = " *
                  "$(positivity_cfl_limit); must be in (0, 1].")
        steps_per_window ≥ 1 ||
            error("CubedSphereContract: steps_per_window = " *
                  "$(steps_per_window); must be ≥ 1.")
        halo_width ≥ 0 ||
            error("CubedSphereContract: halo_width = $(halo_width); " *
                  "must be ≥ 0.")
        return new(Float64(replay_tol),
                   Float64(positivity_cfl_limit),
                   require_substep_positivity,
                   Int(steps_per_window),
                   Int(halo_width),
                   init_cs_positivity_accumulator(),
                   nothing)
    end
end

# Convenience accessor shims so the abstract trait surface answers.
@inline contract_replay_tolerance(c::CubedSphereContract)    = c.replay_tol
@inline contract_cfl_limit(c::CubedSphereContract)           = c.positivity_cfl_limit
@inline contract_require_positivity(c::CubedSphereContract)  = c.require_substep_positivity

"""
    verify_window!(window, contract::CubedSphereContract, win_idx::Int;
                   write_replay_on = true) -> (; replay, positivity)

Run the per-window CS contract on a NamedTuple `window` with fields
`m_cur`, `am`, `bm`, `cm`, `m_next` (each a 6-tuple of panel arrays).
Delegates to `verify_cs_window_contract!`; the replay gate throws on
violation, the positivity gate is non-fatal here. With
`write_replay_on = false` (`ATMOSTR_NO_WRITE_REPLAY_CHECK=1`) only the
positivity gate runs and the replay diagnostic is zero.
"""
function verify_window!(window, contract::CubedSphereContract, win_idx::Integer;
                        write_replay_on::Bool = true)
    if !write_replay_on
        positivity = verify_substep_positivity_cs!(window.m_cur, window.am, window.bm, window.cm;
                                                   cfl_limit  = contract.positivity_cfl_limit,
                                                   halo_width = contract.halo_width,
                                                   m_next     = window.m_next)
        return (replay = (max_rel_err = 0.0, max_abs_err = 0.0, worst_idx = (0, 0, 0, 0)),
                positivity = positivity)
    end
    # Lazy-allocate `_div_scratch` on first call (or reallocate on a
    # shape change — should never happen in production but keeps the
    # invariant local). Subsequent calls reuse the buffer; the contract
    # owns the scratch for the lifetime of the run.
    panel_shape = size(window.m_cur[1])
    if contract._div_scratch === nothing || size(contract._div_scratch) != panel_shape
        contract._div_scratch = Array{Float64}(undef, panel_shape)
    end
    return verify_cs_window_contract!(window.m_cur, window.am, window.bm,
                                       window.cm, window.m_next,
                                       contract.steps_per_window, Int(win_idx);
                                       replay_tol           = contract.replay_tol,
                                       positivity_cfl_limit = contract.positivity_cfl_limit,
                                       halo_width           = contract.halo_width,
                                       div_scratch          = contract._div_scratch)
end

"""
    update_accumulator!(contract::CubedSphereContract, positivity_diag, win_idx::Int)

Fold one window's positivity diagnostic into the CS contract's
worst-window accumulator. Mutates `contract.worst`.
"""
function update_accumulator!(contract::CubedSphereContract, positivity_diag,
                              win_idx::Integer)
    contract.worst = update_cs_positivity_accumulator(contract.worst,
                                                       positivity_diag,
                                                       Int(win_idx))
    return nothing
end

"""
    summarize_status!(contract::CubedSphereContract;
                       quarantine_path::Union{Nothing, AbstractString} = nothing)

Run the CS positivity post-loop summary using the contract's worst
accumulator and policy fields. May log, warn, or error depending on
the accumulator state and `require_substep_positivity`.
"""
function summarize_status!(contract::CubedSphereContract;
                            quarantine_path::Union{Nothing, AbstractString} = nothing)
    return summarize_cs_positivity_status(contract.worst;
                                           cfl_limit = contract.positivity_cfl_limit,
                                           steps_per_window = contract.steps_per_window,
                                           require_substep_positivity =
                                               contract.require_substep_positivity,
                                           quarantine_path = quarantine_path)
end
