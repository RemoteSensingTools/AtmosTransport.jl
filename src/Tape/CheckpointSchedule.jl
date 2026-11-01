# ---------------------------------------------------------------------------
# Checkpoint schedules for tape reduction.
#
# `AbstractCheckpointSchedule` lets the user trade compute for tape storage
# during the reverse pass (the `checkpoint` of an `AdjointTape`). Two concrete
# schedules:
#
#   * `FullCheckpoint()` (default) — every step lives on the tape; zero
#     recomputation, peak tape ~ nsteps × per-step panel bytes.
#
#   * `StrideCheckpoint(K)` — save the replay state every K steps; during the
#     reverse pass, re-record each K-step window from its left-edge checkpoint
#     into the call's tape pool, walk that window in reverse for every adjoint
#     start, rewind the pool, move to the previous window. Peak tape ~ K ×
#     per-step panel bytes plus the checkpoints; one extra forward run.
#
# An optimal binomial (Griewank–Walther) schedule is not implemented: no case
# needs logarithmic snapshot memory yet.
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

Save the replay state (tracer and air mass with halos, and the persistent
operator state) every `K` forward steps and re-record each `K`-step window
from its checkpoint during the reverse pass. `K` must be a positive integer;
`K = 1` checkpoints every step; `K >= nsteps` is a single window.

The tape holds one window at a time (~`K` times the per-step cost; the call's
tape pool is rewound and reused for every window), plus `cld(nsteps, K)`
checkpoints. Each window is recorded once and walked for every adjoint start
(the objectives of a Jacobian), so the extra cost is one forward run.

**Parity.** Every recorded step starts with production's step-start halo
refresh (`fill_cs_step_start_halos!`), so the halos a step reads are a
function of its interior state and a window that restarts at a checkpoint
records the same states as `FullCheckpoint`: the footprints agree bit for bit
on one CPU thread for every supported scheme and storage (issue #60 was the
missing refresh).
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
