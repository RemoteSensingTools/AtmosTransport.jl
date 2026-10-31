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
    fill_cs_mass_delta_payload!(m_target, m_cur, m_target)
    return nothing
end

"""
    verify_write_replay_cs!(m_cur, am, bm, cm, m_next, steps_per_window, tol_rel, win_idx;
                            replay_scratch = nothing)

Run the CS write-time replay gate for one window and return its diagnostic.

The check integrates the stored panel-local fluxes from `m_cur` under the
runtime palindrome-continuity contract and verifies, cell by cell, that the
result matches the explicit endpoint `m_next` (the metric of
`verify_window_continuity`). A failure here means the binary would produce a
runtime day-boundary or window-boundary mass inconsistency.

`replay_scratch` (a `ReplayScratch`) may be pre-allocated by the caller
(workspace-owned scratch or contract-owned lazy scratch) so the gate
doesn't allocate the panel-shared work arrays per call. Default
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
                                 replay_scratch::Union{Nothing, ReplayScratch} = nothing) where {FT, NP}
    validate_replay_tol(tol_rel, "verify_write_replay_cs!")
    NP >= 1 ||
        error("verify_write_replay_cs!: requires at least one panel.")
    panel_shape = size(m_cur[1])
    if replay_scratch === nothing
        replay_scratch = ReplayScratch(panel_shape)
    else
        size(replay_scratch) == panel_shape ||
            error("verify_write_replay_cs!: replay_scratch shape " *
                  "$(size(replay_scratch)) != panel shape $(panel_shape).")
    end
    diag = verify_window_continuity_cs(m_cur, am, bm, cm, m_next,
                                       steps_per_window; scratch=replay_scratch)
    diag.max_rel_err <= tol_rel ||
        error("Write-time replay gate FAILED for CS window $(win_idx): " *
              "rel=$(diag.max_rel_err) > tol=$(tol_rel) at cell $(diag.worst_idx) " *
              "(abs=$(diag.max_abs_err) kg). Stored CS fluxes do not integrate to " *
              "the target mass endpoint under palindrome continuity.")
    return diag
end

"""
    verify_cs_window_contract!(m_cur, am, bm, cm, m_next, steps_per_window, win_idx;
                               replay_tol, positivity_cfl_limit = DEFAULT_POSITIVITY_CFL_LIMIT,
                               replay_scratch = nothing)

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
                                    positivity_cfl_limit::Real = DEFAULT_POSITIVITY_CFL_LIMIT,
                                    replay_scratch::Union{Nothing, ReplayScratch} = nothing) where {FT, NP}
    validate_replay_tol(replay_tol, "verify_cs_window_contract!")
    validate_cfl_limit(positivity_cfl_limit, "verify_cs_window_contract!")
    replay = verify_write_replay_cs!(m_cur, am, bm, cm, m_next,
                                     steps_per_window, replay_tol, win_idx;
                                     replay_scratch = replay_scratch)
    positivity = verify_substep_positivity_cs!(m_cur, am, bm, cm;
                                               cfl_limit = positivity_cfl_limit,
                                               m_next = m_next)
    return (replay = replay, positivity = positivity)
end

"""
    init_cs_positivity_accumulator() -> NamedTuple

Zero-valued state for accumulating the worst per-window positivity diagnostic
across a preprocessing loop. Pair with `update_cs_positivity_accumulator!`.
"""
init_cs_positivity_accumulator() = init_positivity_accumulator(Val(4))

"""
    update_cs_positivity_accumulator(worst, diag, win_idx) -> NamedTuple

Return an updated accumulator from a fresh per-window diagnostic.
"""
function update_cs_positivity_accumulator(worst::CSWorst, diag::NamedTuple,
                                            win_idx::Int)
    return update_positivity_accumulator(worst, diag, win_idx)
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
    return summarize_positivity_status(worst; budget_label="palindrome outgoing/m_ref",
                                        cfl_limit, steps_per_window,
                                        require_substep_positivity, quarantine_path)
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
    worst                       :: CSWorst
    # Lazy scratch (codex round-2): allocated on first `verify_window!`
    # call from the window's panel shape and reused thereafter. Eliminates
    # the per-window replay-scratch allocation the original path produced
    # inside `verify_window_continuity_cs`.
    _replay_scratch             :: Union{Nothing, ReplayScratch{Array{Float64, 3}}}

    function CubedSphereContract{FT}(;
                                      replay_tol::Real,
                                      positivity_cfl_limit::Real = DEFAULT_POSITIVITY_CFL_LIMIT,
                                      require_substep_positivity::Bool = true,
                                      steps_per_window::Integer) where FT
        validate_replay_tol(replay_tol, "CubedSphereContract")
        validate_cfl_limit(positivity_cfl_limit, "CubedSphereContract")
        steps_per_window ≥ 1 ||
            error("CubedSphereContract: steps_per_window = " *
                  "$(steps_per_window); must be ≥ 1.")
        return new(Float64(replay_tol),
                   Float64(positivity_cfl_limit),
                   require_substep_positivity,
                   Int(steps_per_window),
                   init_cs_positivity_accumulator(),
                   nothing)
    end
end

# Convenience accessor shims so the abstract trait surface answers.

"""
    verify_window!(window, contract::CubedSphereContract, win_idx::Int;
                   write_replay_on = true) -> (; replay, positivity)

Run the per-window CS contract on a NamedTuple `window` with fields
`m_cur`, `am`, `bm`, `cm`, `m_next` (each a 6-tuple of panel arrays).
Delegates to `verify_cs_window_contract!`; the replay gate throws on
violation, the positivity gate is non-fatal here. With
`write_replay_on = false` (`[numerics] write_replay_check = false`) only the
positivity gate runs and the replay diagnostic is zero.
"""
function verify_window!(window, contract::CubedSphereContract, win_idx::Integer;
                        write_replay_on::Bool = true)
    if !write_replay_on
        positivity = verify_substep_positivity_cs!(window.m_cur, window.am, window.bm, window.cm;
                                                   cfl_limit  = contract.positivity_cfl_limit,
                                                   m_next     = window.m_next)
        return (replay = (max_rel_err = 0.0, max_abs_err = 0.0, worst_idx = (0, 0, 0, 0)),
                positivity = positivity)
    end
    # Lazy-allocate `_replay_scratch` on first call (or reallocate on a
    # shape change — should never happen in production but keeps the
    # invariant local). Subsequent calls reuse the buffer; the contract
    # owns the scratch for the lifetime of the run.
    panel_shape = size(window.m_cur[1])
    if contract._replay_scratch === nothing || size(contract._replay_scratch) != panel_shape
        contract._replay_scratch = ReplayScratch(panel_shape)
    end
    return verify_cs_window_contract!(window.m_cur, window.am, window.bm,
                                       window.cm, window.m_next,
                                       contract.steps_per_window, Int(win_idx);
                                       replay_tol           = contract.replay_tol,
                                       positivity_cfl_limit = contract.positivity_cfl_limit,
                                       replay_scratch       = contract._replay_scratch)
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

"""
    fill_cs_mass_delta_payload!(dm_payload, m_cur, m_next)

Fill the on-disk forward endpoint-difference payload without mutating the
absolute next endpoint. The ERA5 writer's sliding-window loop swaps
`m_next` into `m_cur` after each write, so in-place conversion of the endpoint
would make the following window start from `m_next - m_cur`.
"""
function fill_cs_mass_delta_payload!(dm_payload::NTuple{NP, <:AbstractArray{FT, 3}},
                                      m_cur::NTuple{NP, <:AbstractArray{FT, 3}},
                                      m_next::NTuple{NP, <:AbstractArray{FT, 3}}) where {FT, NP}
    for p in 1:NP
        @inbounds for idx in eachindex(dm_payload[p])
            dm_payload[p][idx] = m_next[p][idx] - m_cur[p][idx]
        end
    end
    return dm_payload
end
