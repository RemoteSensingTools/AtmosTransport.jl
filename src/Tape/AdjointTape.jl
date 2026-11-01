# ---------------------------------------------------------------------------
# The adjoint tape of a reverse-pass call as one typed value:
#
#   AdjointTape(content, storage, checkpoint)
#
#   * content    — what each record stages (`FullStateTape()`: the state each
#                  operator read, as the forward run had it);
#   * storage    — where the staged states live (`DeviceCSTapeStorage()`,
#                  `PinnedHostCSTapeStorage()`, `MmapCSTapeStorage(dir)`);
#   * checkpoint — how much of the run is taped at once (`FullCheckpoint()`,
#                  `StrideCheckpoint(K)`).
#
# Every combination is valid (Lin–Rood: device storage only). The value is an
# immutable descriptor: each call opens its own runtime pool from it, so one
# `AdjointTape` serves any number of calls. The drivers dispatch on its
# concrete type once per call.
# ---------------------------------------------------------------------------

"""
    AbstractTapeContent

What the records of an adjoint tape stage: the `content` of an
[`AdjointTape`](@ref). [`FullStateTape`](@ref) is the only content today.
"""
abstract type AbstractTapeContent end

"""
    FullStateTape()

Each record stages the full state its operator's reverse reads: the air mass
entering every advection sweep, diffusion solve, convection call and air-mass
reset, plus the tracer mass entering a limited-PPM (monotone, CW84) or FV3
vertical sweep, whose reverse replays the forward's limiter branches.
"""
struct FullStateTape <: AbstractTapeContent end

"""
    AdjointTape(; content = FullStateTape(), storage = DeviceCSTapeStorage(),
                checkpoint = FullCheckpoint())
    AdjointTape(content, storage, checkpoint)

The tape of a cubed-sphere reverse pass, passed as `tape` to
`cs_surface_emission_footprint`, `cs_surface_emission_footprint_from_seed`,
`cs_surface_flux_jacobian`, `cs_surface_flux_4dvar` and
`cs_surface_flux_4dvar_optimize`:

- `content` — what each record stages ([`FullStateTape`](@ref));
- `storage` — where the staged states live: [`DeviceCSTapeStorage`](@ref)
  (default), [`PinnedHostCSTapeStorage`](@ref) or
  [`MmapCSTapeStorage`](@ref)`(dir)`;
- `checkpoint` — [`FullCheckpoint`](@ref) (default; the whole run on one tape)
  or [`StrideCheckpoint`](@ref)`(K)` (one `K`-step window at a time,
  re-recorded from checkpoints; one extra forward run).

Every storage works with every schedule and gives the same gradients bit for
bit on one CPU thread. The Lin–Rood tape supports `DeviceCSTapeStorage` only.
Each call opens the memory its tapes need from this descriptor, reuses it for
every window (and, in `cs_surface_flux_4dvar_optimize`, every evaluation), and
frees it when it returns.
"""
struct AdjointTape{C <: AbstractTapeContent, S <: AbstractCSTapeStorage,
                   K <: AbstractCheckpointSchedule}
    content::C
    storage::S
    checkpoint::K
end

AdjointTape(; content::AbstractTapeContent = FullStateTape(),
            storage::AbstractCSTapeStorage = DeviceCSTapeStorage(),
            checkpoint::AbstractCheckpointSchedule = FullCheckpoint()) =
    AdjointTape(content, storage, checkpoint)
