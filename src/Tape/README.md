# Tape

Storage policies, tape record types, and checkpoint schedules for the
cubed-sphere adjoint reverse pass.

A tape is the list of records the forward pass writes so the reverse pass
can replay them backwards. This module owns where staged panel snapshots
live (device memory, host memory, or an mmap file on disk), the record
structs that the reverse loop dispatches on, and the schedules that trade
recomputation for tape size. It contains no transport physics and no
reverse-loop logic; recording lives in `../Adjoints/Footprint/TapeRecording.jl` and
the reverse walk lives in `../Adjoints/Footprint/ReverseLoop.jl`.

## How It Is Loaded

- `Tape` is a real submodule. [`../AtmosTransport.jl`](../AtmosTransport.jl)
  includes `Tape/Tape.jl` and then `Adjoints/Adjoints.jl`, so `Tape` loads
  first. It imports nothing from other AtmosTransport modules and no package
  (the mmap tier calls POSIX `mmap`, `munmap` and `fallocate` directly).
- [`../Adjoints/Adjoints.jl`](../Adjoints/Adjoints.jl) pulls the names in with
  `using ..Tape: ...` and re-exports the user-facing ones (storage types,
  slot types, schedules).
- `Tape` also exports several `_`-prefixed internals (`_stage_panels`,
  `_tape_panels`, `_allocate_tape_slot`, the `_CS*Record` types, ...).
  Because of `using .Tape`, they are reachable as `AtmosTransport.<name>`.
  `AtmosTransport` itself does not re-export any `Tape` name.

## Entry Points

- Storage policies (`AbstractCSTapeStorage` subtypes):
  - `DeviceCSTapeStorage()` in [`TapeStorage.jl`](TapeStorage.jl): snapshots
    stay on the source backend.
  - `PinnedHostCSTapeStorage()` in [`TapeStorage.jl`](TapeStorage.jl): host
    copies read back through device read caches, one per read role and panel
    shape. The generic method allocates plain `Array`s; the CUDA extension
    overrides `_allocate_tape_slot` and `stage_panels!` for `CuArray` panels
    to use `CUDA.pin` memory. The AMDGPU extension provides the same hooks
    for `ROCArray` panels using registered host arrays and ROCm synchronization.
  - `MmapCSTapeStorage(dir)` in [`MmapTapeStorage.jl`](MmapTapeStorage.jl):
    a scratch directory, required (no default, never `/tmp` implicitly).
    Each reverse-pass call opens one anonymous `_MmapTapeFile` there, sized
    to its largest tape, after checking the free space; nothing is persisted.
- Storage lifecycle of one reverse-pass call (the drivers in
  `../Adjoints/Footprint/`): `_validate_tape_storage(tape_storage)` before
  any forward run; `_needs_tape_size(tape_storage)` says whether the driver
  must count the tape bytes first; `_open_tape_storage(tape_storage, nbytes)`
  returns the storage it records into (a fresh device or pinned-host storage
  for `:device` / `:pinned_host`, a reserved `_MmapTapeFile` for
  `MmapCSTapeStorage`, a pre-constructed storage as is); `_rewind_tape_storage!`
  before each re-recorded window; `_release_opened_tape_storage!` when the call
  ends, which releases what the call opened (`_release_tape_storage!`) but never
  a caller's pre-constructed storage, which the call neither sizes nor
  releases.
- Slot API: `_stage_panels(storage, panels)` allocates and fills a slot;
  `_tape_panels(slot, role)` returns readable panels; `stage_panels!` refills.
  The read role (`AirMassRead()`, `TracerMassRead()`, subtypes of
  `AbstractTapeReadRole`) names the reverse-kernel operand the read feeds.
  `PinnedHostCSTapeStorage` and the `_MmapTapeFile` of an
  `MmapCSTapeStorage` (the `AbstractHostCSTapeStorage` policies) keep one
  device read cache per role and panel shape, so two reads for one kernel
  never share a buffer.
- Records in [`TapeRecords.jl`](TapeRecords.jl): `_CSSweepRecord`,
  `_CSHaloRecord`, `_CSMidpointRecord` (step and emission interval),
  `_CSStepEndRecord`, `_CSDiffusionRecord`, `_CSConvectionRecord` (with the
  interval it integrated and the forward call's CFL sub-step count),
  `_CSAirMassResetRecord`, handled by the reverse loop. The adjoint's tape
  recorder pushes them from the production step's recorder hooks.
- Schedules in [`CheckpointSchedule.jl`](CheckpointSchedule.jl):
  `FullCheckpoint()`, `StrideCheckpoint(K)`, `RevolveCheckpoint()`, plus
  `checkpoint_window_count` and `checkpoint_window_range` (Full and Stride
  only).

## File Map

- [`Tape.jl`](Tape.jl) — module definition, export list, include order
  (`TapeStorage` -> `TapeRecords` -> `MmapTapeStorage` -> `CheckpointSchedule`)
- [`TapeStorage.jl`](TapeStorage.jl) — abstract policies, read roles, the
  role- and shape-keyed device read caches of host-side storage
  (`_tape_read_cache!`), device and pinned-host policies, slot types,
  staging hooks, the storage lifecycle of a call (`_validate_tape_storage`,
  `_open_tape_storage`, no-op `_rewind_tape_storage!` and
  `_release_tape_storage!`, `_release_opened_tape_storage!`),
  `_bytes_per_panel_tuple`
- [`TapeRecords.jl`](TapeRecords.jl) — record structs pushed by the forward
  recorder
- [`MmapTapeStorage.jl`](MmapTapeStorage.jl) — on-disk policy:
  `MmapCSTapeStorage(dir)`, the per-call `_MmapTapeFile` (file-system and
  free-space checks, unlinked file, `fallocate` reservation, one POSIX
  mapping unmapped on release),
  `_tape_slot_bytes`, 64-byte aligned slot views into the mapping (returned on
  the CPU, copied into the role- and shape-keyed device caches on a GPU)
- [`CheckpointSchedule.jl`](CheckpointSchedule.jl) — schedule types and
  stride window arithmetic

## Common Tasks

- Adding a storage policy: subtype `AbstractCSTapeStorage` (or
  `AbstractHostCSTapeStorage` if reads copy into device caches), implement
  `_allocate_tape_slot`, `stage_panels!`, `_tape_panels(slot, role)`, and the
  lifecycle methods it needs (`_open_tape_storage`, `_needs_tape_size`,
  `_rewind_tape_storage!`, `_release_tape_storage!`).
- Adding a record type: define it in [`TapeRecords.jl`](TapeRecords.jl),
  add a branch in `_walk_window_reverse!`
  (`../Adjoints/Footprint/ReverseLoop.jl`).
- Changing GPU staging: edit `../../ext/AtmosTransportCUDAExt.jl`, which
  overrides the pinned-host and mmap hooks for `CuArray` panels.
- The driver code that uses `StrideCheckpoint` / `RevolveCheckpoint` is in
  `../Adjoints/Footprint/` (`StrideCheckpoint.jl`, `stride_checkpoint_*.jl`,
  `revolve_checkpoint.jl`), not here.

## Invariants

- `_CSSweepRecord` leaves its scheme parameter unconstrained to avoid a
  `Tape` -> `Adjoints` dependency. The supported-scheme check
  (`CSAdjointSupportedScheme`) happens where records are built and consumed.
- `_CSLinRoodHorizRecord` is defined in `../Adjoints/LinRoodTape.jl` and is
  handled directly by the reverse loop.
- Every recorder ends each step with `_CSStepEndRecord(step)` (absolute
  step, after convection). It carries no payload; the reverse loop adds the
  4D-Var observation forcing of that step there and does nothing without a
  forcing. `CSTapeByteEstimate.step_records` counts these markers.
- An mmap tape never grows while it records: the driver counts the bytes of
  the largest tape of the call on the replay (`_CSCountingRecorder`, in
  `_tape_slot_bytes`), the file checks `diskstat(dir).available` against them
  plus 10 % (re-measuring for up to 5 s after a GC when short, since XFS frees
  an earlier tape's unlinked blocks lazily) and reserves them with `fallocate`
  before the first record (zero fill off Linux). A full disk is an
  `ArgumentError` with the required and available bytes; a file system that
  cannot reserve blocks for in-place writes (no `fallocate`, ZFS) is an
  `ArgumentError` naming it. Growth past the reservation is an error (a
  counting bug), never a sparse extension. With the extent reserved on ext4 or
  XFS, writes through the mapping cannot hit a full disk (no SIGBUS); the
  off-Linux zero fill does not guarantee that on copy-on-write or compressing
  file systems (APFS, btrfs with compression).
- An `_MmapTapeFile` maps its extent once; slots are views into that mapping,
  each panel on a 64-byte boundary. `_release_tape_storage!` unmaps and closes
  it, so the unlinked file's blocks return to the file system when the call
  returns (XFS reports them free about a second later), not at a later GC.
  Views are invalid after the release.
- One reverse-pass call records all its tapes into one storage: Stride
  windows and Revolve base steps rewind it, so a pre-constructed storage
  serves every schedule. Tapes are not persisted: nothing in `src/` or
  `scripts/` read a saved tape back (the records' `ops` were never saved),
  so the manifest, the loader and per-window tape directories were removed.
- On the CPU, `_tape_panels` returns views of the mapping without copying. On
  the GPU, it copies into a device cache keyed by read role and shape.
- A host-side read stays valid until the next read with the same role. A
  reverse kernel that takes two staged states reads them with two roles:
  the limited-PPM sweep record reads `m` with `AirMassRead()` and `rm` with
  `TracerMassRead()` (one shared cache gave the kernel `m === rm`, issue #57).
- Only staged slots are copied. Flux tuples (`_CSSweepRecord.panels_flux`)
  and forcing/operator objects are kept by reference, so callers must not
  change them before the reverse pass ends. The steps of a met window
  captured by `cs_adjoint_inputs` share one flux tuple, so their records
  reference the same arrays.
- `RevolveCheckpoint` is plain bisection, not optimal binomial Revolve.
  Every recorded step starts with production's step-start halo refresh, so a
  window that restarts at a step boundary records the states of the full
  tape: `StrideCheckpoint` and `RevolveCheckpoint` reproduce
  `FullCheckpoint` bit for bit on one CPU thread for every split-sweep scheme,
  with diffusion, convection and air-mass resets (the bitwise gate), and to
  `atol = 1e-12` with Lin–Rood and several threads. (The former ~1e-7 Revolve drift with
  monotone PPM plus convection was that refresh missing from the recorders,
  issue #60, not limiter flips.)

## Related Docs And Tests

- API reference: [`../../docs/src/api/adjoints.md`](../../docs/src/api/adjoints.md);
  support matrix: [`../../docs/src/theory/adjoint_status.md`](../../docs/src/theory/adjoint_status.md)
- [`../../test/core/test_cs_adjoint_tape_bitwise.jl`](../../test/core/test_cs_adjoint_tape_bitwise.jl):
  the bitwise gate of every storage × checkpoint schedule on rough fields
  (single-threaded subprocess, recorded reference digests, limiter-branch
  coverage); the full matrix is the opt-in
  [`../../test/diagnostic/test_cs_adjoint_tape_bitwise_full.jl`](../../test/diagnostic/test_cs_adjoint_tape_bitwise_full.jl)
- [`../../test/core/test_cs_tape_storage.jl`](../../test/core/test_cs_tape_storage.jl):
  mmap reservation, free-space and overrun errors, slot staging and read
  caches, accepted and rejected `tape_storage` values
- [`../../test/core/test_cs_stride_checkpoint.jl`](../../test/core/test_cs_stride_checkpoint.jl):
  Stride and Revolve parity against `FullCheckpoint`, window arithmetic
- [`../../test/core/test_public_api_surface.jl`](../../test/core/test_public_api_surface.jl):
  `CSTapeSlot` is reachable but not exported at top level

## Depends on

No packages (POSIX `mmap`, `munmap` and `fallocate` through `ccall`). Tape
contains storage and records; it imports no adjoint or inversion
implementation.

## Used by

Adjoints.Footprint, Inversion.Solver and the CUDA/AMDGPU extensions.

## Shared helpers owned here

Storage policies, read roles, staging/reading hooks, record types and
checkpoint schedules are explicitly exported by Tape.jl. Extensions import
these hooks directly from Tape.

## Forward<->adjoint mirrors

The forward recorder stores inputs; the reverse walk reads them in reverse
order. Air-mass and tracer-mass read roles have separate caches so a staged
read cannot overwrite the other operand of a limited-sweep transpose.

Pinned-host and mmap policies share the host synchronization hooks in
TapeStorage.jl. GPU extension methods retain their pinning and device-sync setup.
