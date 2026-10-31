# Tape

The adjoint tape of the cubed-sphere reverse pass: the `AdjointTape` value a
caller chooses, the runtime pools a reverse-pass call records into, the tape
record types, and the checkpoint schedules.

A tape is the list of records the forward pass writes so the reverse pass
can replay them backwards. This module owns what each record stages (the
content), where the staged panel snapshots live (device memory, pinned host
memory, or an mmap file on disk), the record structs that the reverse loop
dispatches on, and the schedules that trade recomputation for tape size. It
contains no transport physics and no reverse-loop logic; recording lives in
`../Adjoints/Footprint/TapeRecording.jl`, the runtime of a call in
`../Adjoints/Footprint/TapeRuntime.jl`, and the reverse walk in
`../Adjoints/Footprint/ReverseLoop.jl`.

## How It Is Loaded

- `Tape` is a real submodule. [`../AtmosTransport.jl`](../AtmosTransport.jl)
  includes `Tape/Tape.jl` and then `Adjoints/Adjoints.jl`, so `Tape` loads
  first. It imports nothing from other AtmosTransport modules and no package
  (the mmap tier calls POSIX `mmap`, `munmap` and `fallocate`, the pinned-host
  arena `malloc` and `free`, directly).
- [`../Adjoints/Adjoints.jl`](../Adjoints/Adjoints.jl) pulls the names in with
  `using ..Tape: ...` and re-exports the user-facing ones (`AdjointTape`, the
  content, storage and checkpoint types, `CSTapeSlot`).
- `Tape` also exports several `_`-prefixed internals (`_stage_panels`,
  `_tape_panels`, `_open_tape_pool`, the `_CS*Record` types, ...). Because of
  `using .Tape`, they are reachable as `AtmosTransport.<name>`.
  `AtmosTransport` itself does not re-export any `Tape` name.

## Entry Points

- `AdjointTape(; content, storage, checkpoint)` in
  [`AdjointTape.jl`](AdjointTape.jl): the `tape` keyword of every reverse-pass
  entry point (`cs_surface_emission_footprint`, `..._from_seed`,
  `cs_surface_flux_jacobian`, `cs_surface_flux_4dvar`,
  `cs_surface_flux_4dvar_optimize`), an immutable descriptor composed of
  three axes, each a type:
  - content (`AbstractTapeContent`): `FullStateTape()`, the full state each
    operator's reverse reads (the only content today);
  - storage (`AbstractCSTapeStorage`): `DeviceCSTapeStorage()` (default),
    `PinnedHostCSTapeStorage()`, `MmapCSTapeStorage(dir)` (a scratch
    directory, required: no default, never `/tmp` implicitly);
  - checkpoint (`AbstractCheckpointSchedule`, [`CheckpointSchedule.jl`](CheckpointSchedule.jl)):
    `FullCheckpoint()` (default), `StrideCheckpoint(K)`, plus
    `checkpoint_window_count` and `checkpoint_window_range`.
- Runtime pools (one per reverse-pass call, or per
  `cs_surface_flux_4dvar_optimize` call for all its evaluations; the
  Footprint drivers own them):
  - `_open_tape_pool(storage, nbytes)`: `_DeviceSlabPool` (device arrays),
    `_PinnedHostArena` (host chunks, [`HostTapeArena.jl`](HostTapeArena.jl)),
    `_MmapTapeFile` (one reserved, mapped scratch file of `nbytes`,
    [`MmapTapeStorage.jl`](MmapTapeStorage.jl)); `_needs_tape_size(storage)`
    says whether the driver must count `nbytes` first (mmap only);
  - `_rewind_tape_pool!(pool)` before each further tape (a stride window, a
    later 4D-Var evaluation): the next slots reuse the pool's memory, and the
    previous tape's slots become stale; `_holds_tape_bytes(pool, nbytes)` says
    whether a rewound pool can hold the next tape (an mmap file holds at most
    its reservation);
  - `_release_tape_pool!(pool)` when the call returns: device arrays and read
    caches are freed (`unsafe_free!` on a GPU), pinned chunks unregistered and
    freed, the mmap file unmapped and closed, now rather than at a later GC.
- Slot API: `_stage_panels(pool, panels)` allocates and fills a slot;
  `_tape_panels(slot, role)` returns readable panels; `stage_panels!` refills.
  Every slot carries the generation of its pool; using a slot after a rewind
  or release throws `StaleTapeSlotError`. The read role (`AirMassRead()`,
  `TracerMassRead()`, subtypes of `AbstractTapeReadRole`) names the
  reverse-kernel operand the read feeds. Device slots (`CSTapeSlot`) return
  their arrays; host slots (`HostCSTapeSlot`, pinned host and mmap) copy into
  one read cache per role and panel shape, on the device for GPU panels, so
  two reads for one kernel never share a buffer.
- Backend hooks, with methods for GPU arrays in the CUDA and AMDGPU
  extensions: `_free_tape_array!` (explicit release), `_tape_synchronize`
  (device↔host copies wait), `_pin_host_chunk` / `_unpin_host_chunk` (each
  arena chunk is registered once).
- Records in [`TapeRecords.jl`](TapeRecords.jl): `_CSSweepRecord`,
  `_CSHaloRecord`, `_CSMidpointRecord` (step and emission interval),
  `_CSStepEndRecord`, `_CSDiffusionRecord`, `_CSConvectionRecord` (with the
  interval it integrated and the forward call's CFL sub-step count),
  `_CSAirMassResetRecord`, handled by the reverse loop. The adjoint's tape
  recorder pushes them from the production step's recorder hooks.

## File Map

- [`Tape.jl`](Tape.jl) — module definition, export list, include order
  (`TapeStorage` -> `HostTapeArena` -> `TapeRecords` -> `MmapTapeStorage` ->
  `CheckpointSchedule` -> `AdjointTape`)
- [`TapeStorage.jl`](TapeStorage.jl) — storage descriptors, read roles,
  `StaleTapeSlotError`, the pool lifecycle (`_open_tape_pool`,
  `_rewind_tape_pool!`, `_release_tape_pool!`, generation checks), the
  backend hooks, the device pool `_DeviceSlabPool` and `CSTapeSlot`, the host
  pools' slots (`HostCSTapeSlot`, 64-byte aligned views, `_tape_slot_bytes`)
  and their role- and shape-keyed read caches, `_bytes_per_panel_tuple`
- [`HostTapeArena.jl`](HostTapeArena.jl) — `_PinnedHostArena`, the pool of
  `PinnedHostCSTapeStorage`: chunks of `malloc`ed host memory (64 MiB doubling
  to 1 GiB), each registered with the backend once and reused across rewinds
- [`TapeRecords.jl`](TapeRecords.jl) — record structs pushed by the forward
  recorder
- [`MmapTapeStorage.jl`](MmapTapeStorage.jl) — `MmapCSTapeStorage(dir)` and
  its pool `_MmapTapeFile` (file-system and free-space checks, unlinked file,
  `fallocate` reservation retried on EINTR, one POSIX mapping unmapped on
  release)
- [`CheckpointSchedule.jl`](CheckpointSchedule.jl) — schedule types and
  stride window arithmetic
- [`AdjointTape.jl`](AdjointTape.jl) — `AbstractTapeContent`,
  `FullStateTape`, `AdjointTape`

## Common Tasks

- Adding a storage: subtype `AbstractCSTapeStorage` (or
  `AbstractHostCSTapeStorage` if reads copy into device caches) with an
  immutable descriptor, implement `_open_tape_pool(storage, nbytes)` returning
  a pool with `generation` and `released` fields, and for the pool
  `_allocate_tape_slot`, `stage_panels!`, `_tape_panels(slot, role)`,
  `_rewind_tape_pool!` and `_release_tape_pool!` (plus `_needs_tape_size` and
  `_holds_tape_bytes` if it reserves its size up front).
- Adding a tape content: subtype `AbstractTapeContent` and add the record
  methods that dispatch on it (`_record_sweep!`, `_staged_tuples_per_sweep` in
  `../Adjoints/Footprint/TapeRecording.jl`); the reverse loop dispatches on
  the records.
- Adding a record type: define it in [`TapeRecords.jl`](TapeRecords.jl),
  add a branch in `_walk_window_reverse!`
  (`../Adjoints/Footprint/ReverseLoop.jl`).
- Changing GPU staging: edit the tape hooks in
  `../../ext/AtmosTransportCUDAExt.jl` and `../../ext/AtmosTransportAMDGPUExt.jl`.
- The drivers that use `FullCheckpoint` / `StrideCheckpoint` and the runtime
  that opens and releases the pools are in `../Adjoints/Footprint/`
  (`StrideCheckpoint.jl`, `TapeRuntime.jl`), not here.

## Invariants

- `AdjointTape` and the storage descriptors are immutable values with no
  runtime state: each call opens its own pool, so one `AdjointTape` serves
  any number of calls. A pool is owned by exactly one call (or one
  optimization) and never shared.
- `_CSSweepRecord` leaves its scheme parameter unconstrained to avoid a
  `Tape` -> `Adjoints` dependency. The supported-scheme check
  (`CSAdjointSupportedScheme`) happens where records are built and consumed.
- `_CSLinRoodHorizRecord` is defined in `../Adjoints/LinRoodTape.jl` and is
  handled directly by the reverse loop. Its arrays are device copies outside
  the pool, so the Lin–Rood tape is device-only (the fenced
  `_linrood_validate_tape_storage(tape.storage)`), and the drivers free them
  explicitly once every start has walked the tape.
- Every recorder ends each step with `_CSStepEndRecord(step)` (absolute
  step, after convection). It carries no payload; the reverse loop adds the
  4D-Var observation forcing of that step there and does nothing without a
  forcing. `CSTapeByteEstimate.step_records` counts these markers.
- A rewind or release advances the pool's generation; every slot checks its
  generation on use, so a stale record throws `StaleTapeSlotError` instead of
  reading another tape's data or freed memory (a host-side compare per read).
- Host slot views never leave their pool: staging copies into them and
  reading copies out of them into a read cache (also on the CPU), so nothing
  can touch a view after the release unmapped or freed its memory.
- An mmap tape never grows while it records: the driver counts the bytes of
  the largest tape on the replay (`_CSCountingRecorder`, in
  `_tape_slot_bytes`; once per runtime, so an optimization counts once), the
  file checks `diskstat(dir).available` against them plus 10 % (re-measuring
  for up to 5 s after a GC when short, since XFS frees an earlier tape's
  unlinked blocks lazily) and reserves them with `fallocate` (retried on
  EINTR) before the first record (zero fill off Linux). A full disk is an
  `ArgumentError` with the required and available bytes; a file system that
  cannot reserve blocks for in-place writes (no `fallocate`, ZFS) is an
  `ArgumentError` naming it. Growth past the reservation is an error (a
  counting bug), never a sparse extension. With the extent reserved on ext4 or
  XFS, writes through the mapping cannot hit a full disk (no SIGBUS); the
  off-Linux zero fill does not guarantee that on copy-on-write or compressing
  file systems (APFS, btrfs with compression).
- An `_MmapTapeFile` maps its extent once; slots are views into that mapping,
  each panel on a 64-byte boundary. `_release_tape_pool!` unmaps and closes
  it, so the unlinked file's blocks return to the file system when the call
  returns (XFS reports them free about a second later), not at a later GC.
- The pinned-host arena allocates its host memory in chunks and registers
  each chunk with the backend once (CUDA: `cuMemHostRegister`; AMDGPU: a
  registered host wrapper), dispatching on the pool's read template (the
  source backend's array type); staging is one synchronous device→host copy
  into pinned memory, reading one synchronous host→device copy into the role's
  cache. Chunks are reused in order after a rewind. On release the backend is
  synchronized first, then each chunk leaves the arena and is unregistered and
  freed (also on an error or interrupt part-way, never twice; an arena dropped
  without release is released by its finalizer). A chunk too small for a slot
  is replaced after its successor is allocated.
- One reverse-pass call records all its tapes into one pool: Stride windows
  rewind it. Tapes are not persisted (nothing in `src/` or `scripts/` read a
  saved tape back).
- A host-side read stays valid until the next read with the same role. A
  reverse kernel that takes two staged states reads them with two roles:
  the limited-PPM sweep record reads `m` with `AirMassRead()` and `rm` with
  `TracerMassRead()` (one shared cache gave the kernel `m === rm`, issue #57).
- Only staged slots are copied. Flux tuples (`_CSSweepRecord.panels_flux`)
  and forcing/operator objects are kept by reference, so callers must not
  change them before the reverse pass ends. The steps of a met window
  captured by `cs_adjoint_inputs` share one flux tuple, so their records
  reference the same arrays.
- Every recorded step starts with production's step-start halo refresh, so
  a window that restarts at a step boundary records the states of the full
  tape: `StrideCheckpoint` reproduces `FullCheckpoint` bit for bit on one CPU
  thread for every split-sweep scheme and storage, with diffusion,
  convection and air-mass resets (the bitwise gate), and to `atol = 1e-12`
  with Lin–Rood and several threads (issue #60 was that refresh missing from
  the recorders). There is no logarithmic-memory schedule: the bisection
  `RevolveCheckpoint` was removed (2.0–2.1× Stride, no device-peak benefit);
  an optimal binomial schedule is for when a case needs it.

## Related Docs And Tests

- API reference: [`../../docs/src/api/adjoints.md`](../../docs/src/api/adjoints.md);
  support matrix: [`../../docs/src/theory/adjoint_status.md`](../../docs/src/theory/adjoint_status.md)
- [`../../test/core/test_cs_adjoint_tape_bitwise.jl`](../../test/core/test_cs_adjoint_tape_bitwise.jl):
  the bitwise gate of every storage × checkpoint schedule on rough fields
  (single-threaded subprocess, recorded reference digests, limiter-branch
  coverage), including a Jacobian under Stride and a 4D-Var optimization
  whose evaluations share one pool; the full matrix is the opt-in
  [`../../test/diagnostic/test_cs_adjoint_tape_bitwise_full.jl`](../../test/diagnostic/test_cs_adjoint_tape_bitwise_full.jl)
- [`../../test/core/test_cs_tape_storage.jl`](../../test/core/test_cs_tape_storage.jl):
  device pool reuse, the pinned-host arena (chunks pinned once, reused,
  unregistered on release), the mmap reservation, free-space and overrun
  errors, stale slots, read caches, accepted and rejected `tape` values
- [`../../test/core/test_cs_stride_checkpoint.jl`](../../test/core/test_cs_stride_checkpoint.jl):
  Stride parity against `FullCheckpoint`, window arithmetic
- [`../../test/core/test_cs_4dvar_single_sweep.jl`](../../test/core/test_cs_4dvar_single_sweep.jl):
  one pool per call and per optimization (opens, rewinds, releases; one mmap
  count per optimization), one recording per stride window for all
  objectives
- [`../../test/core/test_public_api_surface.jl`](../../test/core/test_public_api_surface.jl):
  `CSTapeSlot` is reachable but not exported at top level

## Depends on

No packages (POSIX `mmap`, `munmap`, `fallocate`, `malloc` and `free`
through `ccall`). Tape contains storage and records; it imports no adjoint or
inversion implementation.

## Used by

Adjoints.Footprint, Inversion.Solver and the CUDA/AMDGPU extensions.

## Shared helpers owned here

`AdjointTape` and its axes, storage pools, read roles, staging/reading hooks,
backend hooks, record types and checkpoint schedules are explicitly exported
by Tape.jl. Extensions import the backend hooks directly from Tape.

## Forward<->adjoint mirrors

The forward recorder stores inputs; the reverse walk reads them in reverse
order. Air-mass and tracer-mass read roles have separate caches so a staged
read cannot overwrite the other operand of a limited-sweep transpose.

The pinned-host arena and the mmap file share the host slot type, its read
caches and the synchronous copy hooks in TapeStorage.jl; the GPU extensions
add only the backend hooks (free, synchronize, pin a chunk).
