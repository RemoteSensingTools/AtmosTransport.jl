# ---------------------------------------------------------------------------
# The adjoint tape of a reverse-pass call as one typed value:
#
#   AdjointTape(content, storage, checkpoint)
#
#   * content    — what each record stages and how long a recording lives
#                  (`StateTape()`: what each operator's reverse reads from
#                  this forward run, recorded by every forward run;
#                  `AirMassTape()`: the air mass only, recorded once and
#                  reused by every later forward run of the call);
#   * storage    — where the staged states live (`DeviceCSTapeStorage()`,
#                  `PinnedHostCSTapeStorage()`, `MmapCSTapeStorage(dir)`);
#   * checkpoint — how much of the run is taped at once (`FullCheckpoint()`,
#                  `StrideCheckpoint(K)`).
#
# `AirMassTape` keeps one tape for the whole run, so it takes `FullCheckpoint`;
# Lin–Rood takes device storage only; `AirMassTape` needs a scheme whose
# reverse reads no tracer (checked where the scheme is known, in Adjoints).
# The value is an immutable descriptor: each call opens its own runtime pool
# from it, so one `AdjointTape` serves any number of calls. The drivers
# dispatch on its concrete type once per call.
# ---------------------------------------------------------------------------

"""
    AbstractTapeContent

What the records of an adjoint tape stage, and how long a recording serves:
the `content` of an [`AdjointTape`](@ref). [`StateTape`](@ref) (the default)
records every forward run; [`AirMassTape`](@ref) records the air mass once and
reuses it.
"""
abstract type AbstractTapeContent end

"""
    StateTape()

Every forward run records what each operator's reverse reads of its state:
the air mass entering every advection sweep, diffusion solve, convection call
and air-mass reset, and for a monotone or CW84 PPM sweep the limiter branch of
every cell (one byte per cell, `PPMLimiterCodes`), whose reverse replays the
forward's branches. A Lin–Rood tape also stages the tracer mass entering its
horizontal operators and FV3 vertical sweeps.

The records share what is the same: the palindrome center's air mass is the
one the next vertical sweep stages, and a convection call's air mass is the
next step's first sweep's (or a following air-mass reset's) slot, since the
convection reverse reads interior cells only.
"""
struct StateTape <: AbstractTapeContent end

"""
    AirMassTape()

The tape of a scheme whose reverse reads the air mass only, never the tracer
(`tape_dependence(scheme) isa AirMassDependence`: upwind, unlimited slopes and
unlimited PPM): recorded by the first forward run of a call and reused by
every later one, whatever its tracer and emissions (every adjoint start of a
call, every evaluation of `cs_surface_flux_4dvar_optimize`). Each reuse first
checks that the met inputs (a checksum of their contents), the time windows,
the scheme and operators, the grid, the taped steps and the pool's generation
are those the tape was recorded with; a difference is an `ArgumentError` that
names it. Takes `FullCheckpoint`. A scheme whose reverse reads the tracer
(limited PPM, Lin–Rood) is an `ArgumentError` before any forward run.
"""
struct AirMassTape <: AbstractTapeContent end

"""
    AdjointTape(; content = StateTape(), storage = DeviceCSTapeStorage(),
                checkpoint = FullCheckpoint())
    AdjointTape(content, storage, checkpoint)

The tape of a cubed-sphere reverse pass, passed as `tape` to
`cs_surface_emission_footprint`, `cs_surface_emission_footprint_from_seed`,
`cs_surface_flux_jacobian`, `cs_surface_flux_4dvar` and
`cs_surface_flux_4dvar_optimize`:

- `content` — what each record stages: [`StateTape`](@ref) (default; recorded
  by every forward run) or [`AirMassTape`](@ref) (recorded once per call and
  reused, for schemes whose reverse reads no tracer);
- `storage` — where the staged states live: [`DeviceCSTapeStorage`](@ref)
  (default), [`PinnedHostCSTapeStorage`](@ref) or
  [`MmapCSTapeStorage`](@ref)`(dir)`;
- `checkpoint` — [`FullCheckpoint`](@ref) (default; the whole run on one tape)
  or [`StrideCheckpoint`](@ref)`(K)` (one `K`-step window at a time,
  re-recorded from checkpoints; one extra forward run).

Every storage works with every schedule and gives the same gradients bit for
bit on one CPU thread. The Lin–Rood tape supports `DeviceCSTapeStorage` only;
`AirMassTape` takes `FullCheckpoint` only (an `ArgumentError` here).
Each call opens the memory its tapes need from this descriptor, reuses it for
every window (and, in `cs_surface_flux_4dvar_optimize`, every evaluation), and
frees it when it returns.
"""
struct AdjointTape{C <: AbstractTapeContent, S <: AbstractCSTapeStorage,
                   K <: AbstractCheckpointSchedule}
    content::C
    storage::S
    checkpoint::K
    function AdjointTape(content::C, storage::S, checkpoint::K) where {C <: AbstractTapeContent,
                                                                      S <: AbstractCSTapeStorage,
                                                                      K <: AbstractCheckpointSchedule}
        _check_tape_schedule(content, checkpoint)
        return new{C, S, K}(content, storage, checkpoint)
    end
end

AdjointTape(; content::AbstractTapeContent = StateTape(),
            storage::AbstractCSTapeStorage = DeviceCSTapeStorage(),
            checkpoint::AbstractCheckpointSchedule = FullCheckpoint()) =
    AdjointTape(content, storage, checkpoint)

# An air-mass tape is one recording reused for the whole run; a stride
# schedule re-records its windows from checkpoints in every reverse pass.
_check_tape_schedule(::AbstractTapeContent, ::AbstractCheckpointSchedule) = nothing
_check_tape_schedule(::AirMassTape, ::StrideCheckpoint) = throw(ArgumentError(
    "AirMassTape keeps one recorded tape for the whole run and reuses it; StrideCheckpoint " *
    "re-records each window from its checkpoint in every reverse pass. Use FullCheckpoint() " *
    "with AirMassTape(), or StateTape() with StrideCheckpoint."))

"""
    AutoTape(; storage = nothing, checkpoint = nothing, content = nothing,
             budget_bytes = nothing, scratch_dir = nothing)

Resolve and reserve an `AdjointTape` before recording, from the step schedule
and the available backend and host memory. Supplied axes constrain the search;
`budget_bytes` replaces the measured device tape and snapshot budget and its
safety margin; on CPU it also replaces the shared host budget.
`scratch_dir` enables the mmap candidate. With no fitting reservation, throw
[`TapeReservationError`](@ref). The default content is `StateTape`.
"""
struct AutoTape{S, K, C}
    storage::S
    checkpoint::K
    content::C
    budget_bytes::Union{Nothing, Int}
    scratch_dir::Union{Nothing, String}
end

function AutoTape(; storage::Union{Nothing, AbstractCSTapeStorage} = nothing,
                   checkpoint::Union{Nothing, AbstractCheckpointSchedule, StrideSearch} = nothing,
                   content::Union{Nothing, AbstractTapeContent} = nothing,
                   budget_bytes::Union{Nothing, Integer} = nothing,
                   scratch_dir::Union{Nothing, AbstractString} = nothing)
    budget_bytes === nothing || budget_bytes >= 0 || throw(ArgumentError("budget_bytes must be >= 0"))
    scratch_dir === nothing || !isempty(strip(scratch_dir)) || throw(ArgumentError("scratch_dir must be nonempty"))
    storage === nothing || storage isa MmapCSTapeStorage || scratch_dir === nothing ||
        throw(ArgumentError("scratch_dir requires automatic or mmap storage"))
    storage isa MmapCSTapeStorage && scratch_dir !== nothing && storage.dir != scratch_dir &&
        throw(ArgumentError("scratch_dir disagrees with MmapCSTapeStorage.dir"))
    content === nothing || checkpoint === nothing || _check_tape_schedule(content, checkpoint)
    return AutoTape(storage, checkpoint, content,
                    budget_bytes === nothing ? nothing : Int(budget_bytes),
                    scratch_dir === nothing ? nothing : String(scratch_dir))
end

"""
    TapeReservationError(estimate_bytes, budget_bytes, msg)

No permitted tape candidate could be reserved before recording. The estimate
and available budget are in bytes; `msg` describes the failed choices.
"""
struct TapeReservationError <: Exception
    estimate_bytes::Int
    budget_bytes::Int
    msg::String
end
Base.showerror(io::IO, e::TapeReservationError) = print(io, "TapeReservationError: ", e.msg,
    " (estimate ", e.estimate_bytes, " bytes, budget ", e.budget_bytes, " bytes)")

"""
    TapeReservationExceeded(capacity_bytes, requested_bytes)

Recording requested more memory than was reserved before the forward run.
No storage spill or unreserved pool growth is allowed.
"""
struct TapeReservationExceeded <: Exception
    capacity_bytes::Int
    requested_bytes::Int
end
Base.showerror(io::IO, e::TapeReservationExceeded) = print(io,
    "TapeReservationExceeded: requested ", e.requested_bytes,
    " bytes from a reservation of ", e.capacity_bytes, " bytes")

_check_tape_schedule(::AirMassTape, ::StrideSearch) =
    throw(ArgumentError("AirMassTape requires FullCheckpoint"))

_check_tape_schedule(::AbstractTapeContent, ::StrideSearch) = nothing
