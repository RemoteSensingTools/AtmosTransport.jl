# ---------------------------------------------------------------------------
# Checkpoint schedules for tape reduction.
#
# `AbstractCheckpointSchedule` lets the user trade compute for tape storage
# during the reverse pass (the `checkpoint` of an `AdjointTape`). Three
# concrete schedules:
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
#   * `MetWindowCheckpoint()` — the same driver with segments that follow the
#     met windows of the step schedule (one segment per met window, or
#     `split_steps`-bounded pieces of it), so a segment never straddles two
#     met windows. Peak tape ~ the longest segment; one extra forward run.
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
    MetWindowCheckpoint(; split_steps = 0, snapshots = PinnedHostCSTapeStorage())

Checkpoint at the met-window boundaries of the step schedule
(`CSStepSchedule.steps_per_window`) and re-record one segment at a time during
the reverse pass, with the driver of [`StrideCheckpoint`](@ref). Each segment
covers exactly one met window, whatever its number of steps (the MERRA-2
transport binaries have 2–3 steps per window at C24 and 8–11 at C90), so the
met inputs of one window are all a segment's tape refers to.

`split_steps > 0` splits a met window of `n` steps into `cld(n, split_steps)`
segments of nearly equal length (lengths differ by at most one step, longer
ones first), none longer than `split_steps` and none straddling a window
boundary: the bound on the tape when one window's tape does not fit.
`split_steps = 0` (default) does not split. A run that ends inside
a met window ends its last segment there.

The tape holds one segment (the call's pool is reserved for the largest
per-type record count over the segments), plus one snapshot per segment.
`snapshots` holds the snapshots' tracer and air masses:
`PinnedHostCSTapeStorage()` (default; a month of C90 L72 Float32 snapshots is
about 24 GB) or `DeviceCSTapeStorage()`; persistent operator state keeps its backend types.
Every segment boundary is a step boundary, so the footprints and gradients
equal `FullCheckpoint`'s bit for bit on one CPU thread (see
[`StrideCheckpoint`](@ref), Parity). There is no resident tail. Takes
`StateTape()` only (an `AirMassTape` is one recording for the whole run).
"""
struct MetWindowCheckpoint{S <: AbstractCSTapeStorage} <: AbstractCheckpointSchedule
    split_steps::Int
    snapshots::S
    function MetWindowCheckpoint(; split_steps::Integer = 0,
                                 snapshots::S = PinnedHostCSTapeStorage()) where {S <: AbstractCSTapeStorage}
        split_steps >= 0 || throw(ArgumentError(
            "MetWindowCheckpoint split_steps must be >= 0 (0: one segment per met window); got $(split_steps)"))
        snapshots isa Union{DeviceCSTapeStorage, PinnedHostCSTapeStorage} ||
            throw(ArgumentError("met-window snapshots require device or pinned host storage"))
        return new{S}(Int(split_steps), snapshots)
    end
end

"""
    checkpoint_window_count(schedule, nsteps) -> Int

Number of forward windows for `schedule` covering `nsteps` integration
steps. `FullCheckpoint` is 1 (the existing single tape); `StrideCheckpoint(K)`
is `cld(nsteps, K)`. [`MetWindowCheckpoint`](@ref) segments follow the met
windows, not a step count.
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

"""
    _met_window_segments(schedule::MetWindowCheckpoint, step_windows) -> Vector{UnitRange{Int}}

The segments of `schedule` over the steps whose met windows are
`step_windows`, in step order, in one pass: each run of equal window indices
is one segment, or `cld(n, split_steps)` nearly equal pieces of its `n` steps
(the first `n % pieces` one step longer).
"""
function _met_window_segments(s::MetWindowCheckpoint, step_windows::AbstractVector{<:Integer})
    segments = UnitRange{Int}[]
    n = length(step_windows)
    lo = 1
    for hi in 1:n
        if hi < n
            step_windows[hi + 1] == step_windows[hi] && continue
            step_windows[hi + 1] > step_windows[hi] || throw(ArgumentError(
                "step_windows must be nondecreasing; window $(step_windows[hi + 1]) follows " *
                "window $(step_windows[hi]) at step $(hi + 1)"))
        end
        _push_window_segments!(segments, lo, hi, s.split_steps)
        lo = hi + 1
    end
    return segments
end

function _push_window_segments!(segments, lo::Int, hi::Int, split_steps::Int)
    len = hi - lo + 1
    pieces = split_steps == 0 ? 1 : cld(len, split_steps)
    base, longer = divrem(len, pieces)
    for k in 1:pieces
        width = base + (k <= longer)
        push!(segments, lo:(lo + width - 1))
        lo += width
    end
    return segments
end
