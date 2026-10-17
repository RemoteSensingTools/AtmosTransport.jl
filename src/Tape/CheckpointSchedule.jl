# ---------------------------------------------------------------------------
# Checkpoint schedules for tape reduction.
#
# `AbstractCheckpointSchedule` lets the user trade compute for tape storage
# during the reverse pass. Two concrete schedules:
#
#   * `FullCheckpoint()` (default) — every step lives on the tape; zero
#     recomputation, peak tape disk ~ nsteps × per-step panel bytes.
#
#   * `StrideCheckpoint(K)` — save full `panels_m` snapshots every K steps;
#     during the reverse pass, replay each K-step window from its left-edge
#     checkpoint into a throwaway window tape, walk that window in reverse,
#     drop it, move to the previous window. Peak in-memory ops × K instead
#     of × nsteps; peak window-tape disk ~ K × per-step panel bytes.
#
#   * `RevolveCheckpoint()` — recursively bisect the step range. Peak snapshot
#     storage is logarithmic in the number of steps; replay work is
#     O(nsteps log nsteps). This is a bisection schedule, not optimal binomial
#     Revolve.
# ---------------------------------------------------------------------------

abstract type AbstractCheckpointSchedule end

"""
    FullCheckpoint()

Default tape schedule. Every forward step contributes records to a single
tape; the reverse pass walks the tape once with no recomputation. Peak
storage is proportional to `nsteps`.
"""
struct FullCheckpoint <: AbstractCheckpointSchedule end

"""
    StrideCheckpoint(K)

Save a full `panels_m` checkpoint every `K` forward steps and recompute
the in-window tape lazily during the reverse pass. `K` must be a positive
integer; `K = 1` is degenerate (every step is a checkpoint and the
recompute factor matches `FullCheckpoint`); `K >= nsteps` collapses to a
single window (also degenerate but cheap on tape).

Peak in-memory ops count and per-window tape disk are both ~`K` times
the per-step cost; peak checkpoint memory is `cld(nsteps, K)` full
`panels_m` copies.
"""
struct StrideCheckpoint <: AbstractCheckpointSchedule
    K::Int
    function StrideCheckpoint(K::Integer)
        K >= 1 || throw(ArgumentError(
            "StrideCheckpoint stride must be >= 1; got $(K)"))
        return new(Int(K))
    end
end

"""
    RevolveCheckpoint()

Logarithmic-memory checkpoint schedule. The driver bisects the step
range recursively: at each level it propagates forward from a saved
state to the midpoint, reverses the upper half, restores the saved
state, then reverses the lower half. Peak snapshot memory is the
recursion depth ~ `ceil(log2(nsteps))` state-pair copies; total
recomputation work is ~`O(nsteps × log nsteps)` because each step is
re-propagated once per recursion level on the path from root to
leaf.

This is a tractable bisection variant of Griewank-Walther Revolve —
it does NOT compute optimal binomial splits. For `nsteps = 2880`
(C180 / 14-day campaign) memory caps at ~12 simultaneous snapshots,
which is the production target. A future commit can promote this to
the optimal Revolve schedule (`RevolveCheckpoint(snapshots)` with a
user-supplied snapshot cap) without changing the public API for
single-call inversions.

Trade-off vs `StrideCheckpoint(K)`:

| Scheme                         | Memory          | Forward work  |
|--------------------------------|-----------------|---------------|
| `FullCheckpoint`               | O(nsteps)       | O(nsteps)     |
| `StrideCheckpoint(K)`          | O(nsteps/K)     | ~2 × nsteps   |
| `RevolveCheckpoint()` (bisect) | O(log nsteps)   | O(nsteps × log nsteps) |

**Parity.** Every recorded step starts with production's step-start halo
refresh (`fill_cs_step_start_halos!`), so the halos a step reads are a
function of its interior state and a window that restarts at a bisection
point records the same states as `FullCheckpoint`. Footprints agree with
`FullCheckpoint` to rounding (tested at `atol = 1e-12`) for every supported
scheme, including monotone PPM with `ImplicitVerticalDiffusion` and
`CMFMCConvection` or `TM5Convection`. An earlier O(1e-7) Revolve drift in
that configuration came from the recorders omitting that refresh (the
first X sweep after convection read stale tracer halos in the full tape but
fresh ones in a restarted window; issue #60), not from limiter flips.
"""
struct RevolveCheckpoint <: AbstractCheckpointSchedule end

"""
    checkpoint_window_count(schedule, nsteps) -> Int

Number of forward windows for `schedule` covering `nsteps` integration
steps. `FullCheckpoint` is 1 (the existing single tape); `StrideCheckpoint(K)`
is `cld(nsteps, K)`.
"""
checkpoint_window_count(::FullCheckpoint, nsteps::Integer) = 1
checkpoint_window_count(s::StrideCheckpoint, nsteps::Integer) = cld(Int(nsteps), s.K)

"""
    checkpoint_window_range(schedule, window_index, nsteps) -> UnitRange{Int}

Step range covered by `window_index` (1-indexed) for `schedule` and a run
of `nsteps` integration steps.
"""
checkpoint_window_range(::FullCheckpoint, w::Integer, nsteps::Integer) = 1:Int(nsteps)
function checkpoint_window_range(s::StrideCheckpoint, w::Integer, nsteps::Integer)
    nw = checkpoint_window_count(s, nsteps)
    1 <= w <= nw || throw(BoundsError(1:nw, w))
    lo = (Int(w) - 1) * s.K + 1
    hi = min(Int(w) * s.K, Int(nsteps))
    return lo:hi
end
