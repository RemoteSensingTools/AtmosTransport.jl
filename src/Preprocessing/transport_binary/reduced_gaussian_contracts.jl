# ===========================================================================
# Per-window RG transport-binary contract surface.
#
# Mirrors `cubed_sphere_contracts.jl` and `latlon_contracts.jl` for the
# face-indexed reduced-Gaussian topology: the replay gate
# (`verify_window_continuity_rg`), the per-substep positivity gate, the
# worst-window accumulator and the `require_substep_positivity` policy, as for
# CS. The RG `process_day` checks every balanced window through
# `_verify_rg_balanced_window!`.
#
# RG array shapes (as in `SlidingWindowBuffer` and `verify_window_continuity_rg`):
#
#     m_cur      :: (nc, Nz)             # cell-centered mass
#     hflux      :: (nf, Nz)             # face mass-flux per substep
#     cm         :: (nc, Nz + 1)         # vertical interface flux
#     face_left  :: Vector{Int32}        # `face_left[f] = c` ⇒ cell c is on
#                                        #   the left side of face f
#     face_right :: Vector{Int32}        # same on the right side
#
# Sign convention: `hflux[f, k] > 0` means flux goes from
# `face_left[f]` to `face_right[f]`. So for an interior face touching
# cell `c` on BOTH sides:
#
#     face_left[f]  == c → outgoing = max(0,  hflux[f, k])
#     face_right[f] == c → outgoing = max(0, -hflux[f, k])
#
# Boundary stubs (`face_left[f] ≤ 0` or `face_right[f] ≤ 0`) — the
# south/north pole singularities of the RG mesh — are SKIPPED entirely.
# This matches the runtime face-indexed advection in
# `src/Operators/Advection/StrangSplitting.jl`:
#
#     if left > 0 && right > 0
#         # accumulate flux to both cells
#     end
#     # else: skip — no mass enters or leaves through the pole singularity
#
# If the contract counted boundary-stub outflow against the interior
# cell, a binary with `(face_left=0, face_right=1, hflux=-0.99)` would
# trip the positivity gate, but the runtime would never apply that flux
# (the cell mass is unchanged by the face). The codex review of
# `3796526` (round-1) caught this asymmetry; the round-2 fix in this
# file moves the both-interior check above the sign branch.
#
# Non-zero flux on a boundary stub is its OWN explicit invariant
# violation. The writer that produces such a value is silently writing
# data the runtime will discard, which usually indicates a sign-flip
# bug in the boundary masking. `verify_boundary_stub_flux_rg` returns a
# diagnostic for the worst boundary-stub flux, and the wrapper
# `verify_rg_window_contract!` errors hard when this fires (no
# `require_*` escape hatch — there's no legitimate operational reason
# for a boundary-stub flux to be non-zero).
#
# Direction reported in the diagnostic is `:h` (horizontal) or `:z`
# (vertical); RG faces aren't axis-aligned, so there is no separate `:x`/
# `:y` decomposition. The RG runtime subcycles its horizontal pass (mixed
# x/y) and its vertical pass independently, so the contract is checked per
# pass (CS instead shares one palindrome count across X, Y and Z).
# ===========================================================================

"""
    RGWorst

Type-stable shape of the RG positivity accumulator. Same role as
`CSWorst`/`LLWorst` but with `(cell, level)` coordinates (no panel or
explicit x/y axis).
"""
const RGWorst = @NamedTuple{ratio::Float64, direction::Symbol, win::Int,
                              location::NTuple{2, Int}}

"""
    verify_substep_positivity_rg!(m, hflux, cm, face_left, face_right;
                                  cfl_limit = 0.95,
                                  outgoing_h = nothing, bad_h = nothing)

Per-substep horizontal+vertical positivity scan for a face-indexed RG
window. Mirrors `verify_substep_positivity_cs!` / `..._ll!` but operates
on the face-indexed RG mass-flux representation.

For every cell `(c, k)`:
  1. `m > 0`. A non-positive cell mass is reported with `ratio = Inf`.
  2. Horizontal outgoing mass per substep ≤ `cfl_limit * m`. Only
     interior faces (`face_left > 0 && face_right > 0`) contribute,
     matching the runtime advection in `StrangSplitting.jl:279`.
     Boundary stubs are not counted as outflow here — see
     `verify_boundary_stub_flux_rg` for the separate "non-zero flux on
     a boundary stub" invariant.
  3. Vertical outgoing mass per substep ≤ `cfl_limit * m`. Same as
     CS / LL: `max(0, -cm[c, k]) + max(0, cm[c, k+1])`.

`NaN`/`Inf` cell mass and `NaN`/`Inf` fluxes are flagged as `ratio = Inf`
(matches the CS round-2 fix).

`outgoing_h` and `bad_h` can be passed in as workspace-owned scratch
to suppress per-window allocation once this is wired into the
unified driver. Default `nothing` → allocate locally.

Returns `(direction, ratio, location, ok)` with:
  - `direction :: Union{Symbol, Nothing}` — `:h` / `:z` / `nothing`.
  - `ratio :: Float64` — worst `outgoing / m`.
  - `location :: NTuple{2, Int}` — `(cell, level)`.
  - `ok :: Bool` — `ratio ≤ cfl_limit`.
"""
function verify_substep_positivity_rg!(m::AbstractMatrix{FT},
                                        hflux::AbstractMatrix,
                                        cm::AbstractMatrix,
                                        face_left::AbstractVector{<:Integer},
                                        face_right::AbstractVector{<:Integer};
                                        cfl_limit::Real = 0.95,
                                        outgoing_h::Union{Nothing, AbstractMatrix{Float64}} = nothing,
                                        bad_h::Union{Nothing, AbstractMatrix{Bool}} = nothing) where FT
    _validate_cfl_limit(cfl_limit, "verify_substep_positivity_rg!")
    nc, Nz = size(m)
    size(hflux, 2) == Nz ||
        error("verify_substep_positivity_rg!: hflux level count " *
              "$(size(hflux, 2)) != m level count $(Nz).")
    size(cm) == (nc, Nz + 1) ||
        error("verify_substep_positivity_rg!: cm shape $(size(cm)) " *
              "incompatible with m $(size(m)); expected ($(nc), $(Nz + 1)).")
    nf = size(hflux, 1)
    length(face_left)  == nf ||
        error("verify_substep_positivity_rg!: face_left length " *
              "$(length(face_left)) != nfaces $(nf).")
    length(face_right) == nf ||
        error("verify_substep_positivity_rg!: face_right length " *
              "$(length(face_right)) != nfaces $(nf).")

    # Scratch buffers — pre-allocated by the caller (eventually plumbed
    # through workspace state) or auto-allocated here as a fallback.
    if outgoing_h === nothing
        outgoing_h = zeros(Float64, nc, Nz)
    else
        size(outgoing_h) == (nc, Nz) ||
            error("verify_substep_positivity_rg!: outgoing_h scratch " *
                  "shape $(size(outgoing_h)) != expected ($(nc), $(Nz)).")
        fill!(outgoing_h, 0.0)
    end
    if bad_h === nothing
        bad_h = falses(nc, Nz)
    else
        size(bad_h) == (nc, Nz) ||
            error("verify_substep_positivity_rg!: bad_h scratch " *
                  "shape $(size(bad_h)) != expected ($(nc), $(Nz)).")
        fill!(bad_h, false)
    end

    # Accumulate per-cell horizontal outflow once for all levels. Only
    # interior faces contribute, matching the runtime semantics. A NaN
    # or Inf flux at any interior face is propagated into the affected
    # cells' bad-flux flag so the worst-ratio scan reports the
    # contamination.
    @inbounds for f in 1:nf
        cL = Int(face_left[f])
        cR = Int(face_right[f])
        cL_interior = 1 ≤ cL ≤ nc
        cR_interior = 1 ≤ cR ≤ nc
        both_interior = cL_interior && cR_interior
        # Skip boundary stubs (one side is a pole / non-interior). The
        # runtime advection skips these faces entirely — counting them
        # as outflow here would falsely trip the positivity gate on
        # binaries whose runtime mass evolution is unaffected.
        both_interior || continue
        for k in 1:Nz
            h = hflux[f, k]
            if !isfinite(h)
                bad_h[cL, k] = true
                bad_h[cR, k] = true
                continue
            end
            if h > zero(h)
                outgoing_h[cL, k] += Float64(h)
            elseif h < zero(h)
                outgoing_h[cR, k] -= Float64(h)
            end
        end
    end

    worst_dir = nothing
    worst_ratio = 0.0
    worst_loc = (0, 0)
    @inbounds for k in 1:Nz, c in 1:nc
        mi = m[c, k]
        if !isfinite(mi) || mi <= zero(FT)
            if !isinf(worst_ratio)
                worst_ratio = Inf
                worst_dir = :h
                worst_loc = (c, k)
            end
            continue
        end
        # Horizontal first.
        if bad_h[c, k]
            if !isinf(worst_ratio)
                worst_ratio = Inf
                worst_dir = :h
                worst_loc = (c, k)
            end
        else
            ratio_h = outgoing_h[c, k] / Float64(mi)
            if ratio_h > worst_ratio
                worst_ratio = ratio_h
                worst_dir = :h
                worst_loc = (c, k)
            end
        end
        # Vertical second.
        fl = cm[c, k]
        fh = cm[c, k + 1]
        if !isfinite(fl) || !isfinite(fh)
            if !isinf(worst_ratio)
                worst_ratio = Inf
                worst_dir = :z
                worst_loc = (c, k)
            end
            continue
        end
        outgoing_z = max(zero(FT), -fl) + max(zero(FT), fh)
        ratio_z = Float64(outgoing_z) / Float64(mi)
        if ratio_z > worst_ratio
            worst_ratio = ratio_z
            worst_dir = :z
            worst_loc = (c, k)
        end
    end

    return (direction = worst_dir, ratio = worst_ratio,
            location = worst_loc, ok = worst_ratio <= cfl_limit)
end

"""
    verify_boundary_stub_flux_rg(hflux, face_left, face_right;
                                 tol = 0.0) -> NamedTuple

Explicit-invariant scan: any non-zero `hflux` value on a boundary stub
(`face_left ≤ 0` or `face_right ≤ 0`) is a contract violation. The
runtime advection silently discards such fluxes
(`StrangSplitting.jl:279`), so a writer that produces them is emitting
data the runtime cannot apply — almost always a sign-flip or boundary-
masking bug in preprocessing.

Returns `(violated, worst_flux, worst_face, worst_level)`:
  - `violated :: Bool` — `true` iff any |flux| > tol on a boundary stub.
  - `worst_flux :: Float64` — signed value of the worst-magnitude
    violation, or `0.0` if none.
  - `worst_face :: Int` — face index of the worst violation, or `0`.
  - `worst_level :: Int` — k-index of the worst violation, or `0`.

`tol` is the absolute tolerance below which a "near-zero" stub flux is
permitted. Default 0.0 (strict) — RG writers should explicitly zero
boundary stubs.
"""
function verify_boundary_stub_flux_rg(hflux::AbstractMatrix,
                                        face_left::AbstractVector{<:Integer},
                                        face_right::AbstractVector{<:Integer};
                                        tol::Real = 0.0)
    # Codex round-2: validate `tol` directly here too. The struct-level
    # constructor checks `boundary_stub_tol` on `ReducedGaussianContract`,
    # but this helper is exported and can be called bypassing the
    # contract. An `Inf` or `NaN` `tol` would silently disable the gate
    # (`abs(h) > NaN` is always `false`, so no face ever crosses the
    # threshold); validating here closes the bypass.
    isfinite(tol) && tol ≥ 0 ||
        error("verify_boundary_stub_flux_rg: tol = $(tol); " *
              "must be finite and ≥ 0 (Inf/NaN would silently disable " *
              "the gate; negative tol is meaningless).")
    nf = size(hflux, 1)
    length(face_left)  == nf ||
        error("verify_boundary_stub_flux_rg: face_left length " *
              "$(length(face_left)) != nfaces $(nf).")
    length(face_right) == nf ||
        error("verify_boundary_stub_flux_rg: face_right length " *
              "$(length(face_right)) != nfaces $(nf).")
    Nz = size(hflux, 2)
    worst_abs = Float64(tol)
    worst_flux = 0.0
    worst_face = 0
    worst_level = 0
    @inbounds for f in 1:nf
        cL = Int(face_left[f])
        cR = Int(face_right[f])
        is_stub = cL <= 0 || cR <= 0
        is_stub || continue
        for k in 1:Nz
            h = hflux[f, k]
            ah = isfinite(h) ? abs(Float64(h)) : Inf
            if ah > worst_abs
                worst_abs   = ah
                worst_flux  = Float64(h)   # `Float64(NaN)` / `Float64(Inf)` propagate naturally
                worst_face  = f
                worst_level = k
            end
        end
    end
    return (violated   = worst_face != 0,
            worst_flux = worst_flux,
            worst_face = worst_face,
            worst_level = worst_level)
end

"""
    verify_rg_window_contract!(m_cur, hflux, cm, m_next, face_left, face_right,
                               steps_per_window, win_idx;
                               replay_tol, positivity_cfl_limit = 0.95,
                               div_scratch = nothing,
                               outgoing_h = nothing, bad_h = nothing,
                               boundary_stub_tol = 0.0)

Single canonical per-window RG binary contract check. Runs three gates
in order:

  1. **Boundary-stub flux gate** — errors hard if any boundary stub
     (`face_left ≤ 0` / `face_right ≤ 0`) carries non-zero `hflux`
     above `boundary_stub_tol`. No `require_*` escape hatch: such
     fluxes are silently discarded by the runtime
     (`StrangSplitting.jl:279`), so emitting them is always a writer
     bug.
  2. **Replay gate** — `verify_window_continuity_rg`; errors on
     failure.
  3. **Per-substep positivity scan** —
     `verify_substep_positivity_rg!`, returns a non-fatal diagnostic;
     the run-level accumulator + `summarize_rg_positivity_status`
     decides fatal-vs-warn.

`div_scratch`, `outgoing_h`, `bad_h` may be pre-allocated by the
caller (workspace-owned scratch) to suppress per-window
allocation. Default `nothing` → allocate locally.

Returns `(; replay, positivity)`. Boundary-stub failure does not
return; it errors out before the replay gate so a broken writer
cannot silently emit a binary the runtime would partially evaluate.
"""
function verify_rg_window_contract!(m_cur::AbstractMatrix{FT},
                                     hflux::AbstractMatrix,
                                     cm::AbstractMatrix,
                                     m_next::AbstractMatrix,
                                     face_left::AbstractVector{<:Integer},
                                     face_right::AbstractVector{<:Integer},
                                     steps_per_window::Integer,
                                     win_idx::Integer;
                                     replay_tol::Real,
                                     positivity_cfl_limit::Real = 0.95,
                                     div_scratch::Union{Nothing, AbstractMatrix{Float64}} = nothing,
                                     outgoing_h::Union{Nothing, AbstractMatrix{Float64}} = nothing,
                                     bad_h::Union{Nothing, AbstractMatrix{Bool}} = nothing,
                                     boundary_stub_tol::Real = 0.0) where FT
    # Codex round-3: validate every policy knob at the wrapper boundary
    # (`replay_tol` / `positivity_cfl_limit` / `boundary_stub_tol`). The
    # `ReducedGaussianContract` constructor validates too, but the
    # wrapper is exported and CS production paths call analogous CS
    # wrappers directly — same defense-in-depth pattern.
    _validate_replay_tol(replay_tol, "verify_rg_window_contract!")
    _validate_cfl_limit(positivity_cfl_limit, "verify_rg_window_contract!")
    isfinite(boundary_stub_tol) && boundary_stub_tol ≥ 0 ||
        error("verify_rg_window_contract!: boundary_stub_tol = " *
              "$(boundary_stub_tol); must be finite and ≥ 0.")
    stub = verify_boundary_stub_flux_rg(hflux, face_left, face_right;
                                          tol = boundary_stub_tol)
    stub.violated &&
        error("Boundary-stub flux gate FAILED for RG window $(win_idx): " *
              "hflux=$(stub.worst_flux) on face=$(stub.worst_face) " *
              "level=$(stub.worst_level) where face_left=$(face_left[stub.worst_face]) " *
              "face_right=$(face_right[stub.worst_face]); runtime advection " *
              "(`StrangSplitting.jl:279`) will silently discard this flux. " *
              "Either the writer's boundary-masking logic dropped a zero, " *
              "or `face_left`/`face_right` connectivity has the wrong sign.")
    if div_scratch === nothing
        div_scratch = Array{Float64}(undef, size(m_cur))
    else
        size(div_scratch) == size(m_cur) ||
            error("verify_rg_window_contract!: div_scratch shape " *
                  "$(size(div_scratch)) != m_cur shape $(size(m_cur)).")
    end
    replay = verify_window_continuity_rg(m_cur, hflux, cm, m_next,
                                          face_left, face_right,
                                          div_scratch, steps_per_window)
    replay.max_rel_err <= replay_tol ||
        error("Write-time replay gate FAILED for RG window $(win_idx): " *
              "rel=$(replay.max_rel_err) > tol=$(replay_tol) at cell " *
              "$(replay.worst_idx) (abs=$(replay.max_abs_err) kg). Stored RG " *
              "fluxes do not integrate to the target mass endpoint under " *
              "palindrome continuity.")
    positivity = verify_substep_positivity_rg!(m_cur, hflux, cm,
                                                face_left, face_right;
                                                cfl_limit = positivity_cfl_limit,
                                                outgoing_h = outgoing_h,
                                                bad_h = bad_h)
    return (replay = replay, positivity = positivity)
end

"""
    init_rg_positivity_accumulator() -> NamedTuple

Zero-valued state for accumulating the worst per-window RG positivity
diagnostic across a preprocessing loop.
"""
init_rg_positivity_accumulator() = RGWorst((0.0, :none, 0, (0, 0)))

"""
    update_rg_positivity_accumulator(worst, diag, win_idx) -> NamedTuple

Return an updated RG accumulator from a fresh per-window diagnostic.
"""
function update_rg_positivity_accumulator(worst::RGWorst, diag::NamedTuple,
                                            win_idx::Integer)
    diag.ratio > worst.ratio || return worst
    return RGWorst((Float64(diag.ratio),
                    diag.direction === nothing ? :none : Symbol(diag.direction),
                    Int(win_idx),
                    NTuple{2, Int}(diag.location)))
end

"""
    summarize_rg_positivity_status(worst; cfl_limit, steps_per_window,
                                   require_substep_positivity = true,
                                   quarantine_path = nothing)

Post-loop summary helper for the RG positivity accumulator. Mirrors
`summarize_cs_positivity_status` (CS round-2 + round-3 escape-hatch
semantics).
"""
function summarize_rg_positivity_status(worst::RGWorst;
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
# `ReducedGaussianContract{FT}` — typed Axis-3 concrete for the RG topology.
# Mirrors `CubedSphereContract{FT}` and `LatLonContract{FT}`.
#
# The contract carries the per-grid face connectivity (`face_left`/
# `face_right`) so the abstract `verify_window!` call site doesn't need
# to know it. The `process_day` orchestrator constructs the contract
# once per run with the grid's connectivity vectors and reuses it for
# every window.
# ---------------------------------------------------------------------------

"""
    ReducedGaussianContract{FT} <: AbstractWindowContract{ReducedGaussianTargetGeometry, FT}

Typed nominal owning an RG preprocessor's per-window gate policy and
worst-window positivity accumulator. Holds the face connectivity
(`face_left` / `face_right`) so the per-window call site doesn't need
to thread it through every call.

Construction validates `replay_tol`, `positivity_cfl_limit ∈ (0, 1]`,
`steps_per_window ≥ 1`, `boundary_stub_tol ≥ 0`, and
`length(face_left) == length(face_right)`.

`boundary_stub_tol` (default `0.0`) is the absolute tolerance for the
boundary-stub flux gate (`verify_boundary_stub_flux_rg`). Default is
strict; tighten only with caution.
"""
mutable struct ReducedGaussianContract{FT} <:
                AbstractWindowContract{ReducedGaussianTargetGeometry, FT}
    replay_tol                  :: Float64
    positivity_cfl_limit        :: Float64
    require_substep_positivity  :: Bool
    steps_per_window            :: Int
    boundary_stub_tol           :: Float64
    face_left                   :: Vector{Int32}
    face_right                  :: Vector{Int32}
    worst                       :: RGWorst
    # Lazy scratch (codex round-2): allocated on first `verify_window!`
    # call from the window's `(nc, Nz)` shape and reused thereafter.
    # Eliminates per-window `Array{Float64}(undef, nc, Nz)` allocations
    # for the replay divergence, the positivity horizontal-outflow
    # accumulator, and the NaN/Inf flag mask.
    _div_scratch                :: Union{Nothing, Matrix{Float64}}
    _outgoing_h                 :: Union{Nothing, Matrix{Float64}}
    _bad_h                      :: Union{Nothing, Matrix{Bool}}

    function ReducedGaussianContract{FT}(;
                                          replay_tol::Real,
                                          positivity_cfl_limit::Real = 0.95,
                                          require_substep_positivity::Bool = true,
                                          steps_per_window::Integer,
                                          boundary_stub_tol::Real = 0.0,
                                          face_left::AbstractVector{<:Integer},
                                          face_right::AbstractVector{<:Integer}) where FT
        isfinite(replay_tol) && replay_tol > 0 ||
            error("ReducedGaussianContract: replay_tol = $(replay_tol); " *
                  "must be finite and > 0 (Inf would silently disable replay; " *
                  "NaN would fail every window late).")
        isfinite(positivity_cfl_limit) && 0 < positivity_cfl_limit ≤ 1 ||
            error("ReducedGaussianContract: positivity_cfl_limit = " *
                  "$(positivity_cfl_limit); must be in (0, 1].")
        steps_per_window ≥ 1 ||
            error("ReducedGaussianContract: steps_per_window = " *
                  "$(steps_per_window); must be ≥ 1.")
        isfinite(boundary_stub_tol) && boundary_stub_tol ≥ 0 ||
            error("ReducedGaussianContract: boundary_stub_tol = " *
                  "$(boundary_stub_tol); must be finite and ≥ 0.")
        length(face_left) == length(face_right) ||
            error("ReducedGaussianContract: face_left length " *
                  "$(length(face_left)) != face_right length " *
                  "$(length(face_right)).")
        return new(Float64(replay_tol),
                   Float64(positivity_cfl_limit),
                   require_substep_positivity,
                   Int(steps_per_window),
                   Float64(boundary_stub_tol),
                   Vector{Int32}(face_left),
                   Vector{Int32}(face_right),
                   init_rg_positivity_accumulator(),
                   nothing, nothing, nothing)
    end
end

@inline contract_replay_tolerance(c::ReducedGaussianContract)   = c.replay_tol
@inline contract_cfl_limit(c::ReducedGaussianContract)          = c.positivity_cfl_limit
@inline contract_require_positivity(c::ReducedGaussianContract) = c.require_substep_positivity

function verify_window!(window, contract::ReducedGaussianContract, win_idx::Integer)
    # Lazy-allocate the three scratch buffers on first call (or
    # reallocate on a shape change). Subsequent calls reuse them.
    m_shape = size(window.m_cur)
    if contract._div_scratch === nothing || size(contract._div_scratch) != m_shape
        contract._div_scratch = Array{Float64}(undef, m_shape)
        contract._outgoing_h  = Array{Float64}(undef, m_shape)
        contract._bad_h       = Array{Bool}(undef, m_shape)
    end
    return verify_rg_window_contract!(window.m_cur, window.hflux,
                                       window.cm, window.m_next,
                                       contract.face_left, contract.face_right,
                                       contract.steps_per_window, Int(win_idx);
                                       replay_tol           = contract.replay_tol,
                                       positivity_cfl_limit = contract.positivity_cfl_limit,
                                       boundary_stub_tol    = contract.boundary_stub_tol,
                                       div_scratch          = contract._div_scratch,
                                       outgoing_h           = contract._outgoing_h,
                                       bad_h                = contract._bad_h)
end

function update_accumulator!(contract::ReducedGaussianContract, positivity_diag,
                              win_idx::Integer)
    contract.worst = update_rg_positivity_accumulator(contract.worst,
                                                       positivity_diag,
                                                       Int(win_idx))
    return nothing
end

function summarize_status!(contract::ReducedGaussianContract;
                            quarantine_path::Union{Nothing, AbstractString} = nothing)
    return summarize_rg_positivity_status(contract.worst;
                                            cfl_limit = contract.positivity_cfl_limit,
                                            steps_per_window = contract.steps_per_window,
                                            require_substep_positivity =
                                                contract.require_substep_positivity,
                                            quarantine_path = quarantine_path)
end
