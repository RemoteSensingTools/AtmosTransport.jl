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
    StrideCheckpoint(K; resident_tail = 0, snapshots = DeviceCSTapeStorage())

Save the replay state (tracer and air mass with halos, and the persistent
operator state) every `K` forward steps and re-record each `K`-step window
from its checkpoint during the reverse pass. `K` must be a positive integer;
`K = 1` checkpoints every step; `K >= nsteps` is a single window.

The tape holds one window at a time (~`K` times the per-step cost; the call's
tape pool is rewound and reused for every window), plus `cld(nsteps, K)`
checkpoints. Each window is recorded once and walked for every adjoint start
(the objectives of a Jacobian), so the extra cost is one forward run.

`resident_tail` keeps the final steps recorded during the forward run,
avoiding their recomputation; the head uses K-step windows. `snapshots` may
be `PinnedHostCSTapeStorage()` to move checkpoint tracer and air masses to
host memory; persistent operator state retains its backend types. Both pools
and every checkpoint are reserved before recording.

**Parity.** Every recorded step starts with production's step-start halo
refresh (`fill_cs_step_start_halos!`), so the halos a step reads are a
function of its interior state and a window that restarts at a checkpoint
records the same states as `FullCheckpoint`: the footprints agree bit for bit
on one CPU thread for every supported scheme and storage (issue #60 was the
missing refresh).
"""
struct StrideCheckpoint{S <: AbstractCSTapeStorage} <: AbstractCheckpointSchedule
    K::Int
    resident_tail::Int
    snapshots::S
    function StrideCheckpoint(K::Integer; resident_tail::Integer = 0,
                              snapshots::S = DeviceCSTapeStorage()) where {S <: AbstractCSTapeStorage}
        K >= 1 || throw(ArgumentError(
            "StrideCheckpoint stride must be >= 1; got $(K)"))
        resident_tail >= 0 || throw(ArgumentError("resident_tail must be >= 0"))
        snapshots isa Union{DeviceCSTapeStorage, PinnedHostCSTapeStorage} ||
            throw(ArgumentError("stride snapshots require device or pinned host storage"))
        return new{S}(Int(K), Int(resident_tail), snapshots)
    end
end

"""
    StrideSearch(K = nothing, resident_tail = nothing, snapshots = nothing)

Constrain AutoTape's stride search, leaving unspecified fields automatic.
AutoTape resolves this request to a concrete `StrideCheckpoint` before recording.
"""
struct StrideSearch
    K::Union{Nothing, Int}
    resident_tail::Union{Nothing, Int}
    snapshots::Union{Nothing, DeviceCSTapeStorage, PinnedHostCSTapeStorage}
    function StrideSearch(K = nothing, resident_tail = nothing, snapshots = nothing)
        K === nothing || K >= 1 || throw(ArgumentError("stride must be >= 1"))
        resident_tail === nothing || resident_tail >= 0 || throw(ArgumentError("resident_tail must be >= 0"))
        return new(K, resident_tail, snapshots)
    end
end

"""
    checkpoint_window_count(schedule, nsteps) -> Int

Number of forward windows for `schedule` covering `nsteps` integration
steps. `FullCheckpoint` is 1 (the existing single tape); `StrideCheckpoint(K)`
is `cld(nsteps, K)`.
"""
checkpoint_window_count(::FullCheckpoint, nsteps::Integer) = 1
checkpoint_window_count(s::StrideCheckpoint, nsteps::Integer) =
    cld(max(0, Int(nsteps) - s.resident_tail), s.K) + (s.resident_tail > 0)

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
    head = max(0, Int(nsteps) - s.resident_tail)
    w == nw && s.resident_tail > 0 && return (head + 1):Int(nsteps)
    hi = min(Int(w) * s.K, head)
    return lo:hi
end
