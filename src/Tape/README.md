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
  first. It imports nothing from other AtmosTransport modules (only `Mmap`,
  `TOML`, `Dates`).
- [`../Adjoints/Adjoints.jl`](../Adjoints/Adjoints.jl) pulls the names in with
  `using ..Tape: ...` and re-exports the user-facing ones (storage types,
  schedules, `finalize_tape!`, `load_mmap_tape`, `get_record`).
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
  - `MmapCSTapeStorage(; dir, cleanup_on_finalize)` in
    [`MmapTapeStorage.jl`](MmapTapeStorage.jl): appends payloads to
    `records.bin` and writes `manifest.toml` on `finalize_tape!`.
- Symbol resolution: `_tape_storage(:device | :pinned_host | :mmap)`.
  `_resolve_tape_path` and `_build_window_storage` turn the public
  `tape_storage` / `tape_path` kwargs into one storage per tape or window.
- Slot API: `_stage_panels(storage, panels)` allocates and fills a slot;
  `_tape_panels(slot, role)` returns readable panels; `stage_panels!` refills.
  The read role (`AirMassRead()`, `TracerMassRead()`, subtypes of
  `AbstractTapeReadRole`) names the reverse-kernel operand the read feeds.
  `PinnedHostCSTapeStorage` and `MmapCSTapeStorage` (the
  `AbstractHostCSTapeStorage` policies) keep one device read cache per role
  and panel shape, so two reads for one kernel never share a buffer.
- Persistence: `finalize_tape!`, `load_mmap_tape(dir)`,
  `get_record(storage, id)` in [`MmapTapeStorage.jl`](MmapTapeStorage.jl).
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
  staging hooks, `_resolve_tape_path`, no-op `finalize_tape!` fallback,
  `_bytes_per_panel_tuple`
- [`TapeRecords.jl`](TapeRecords.jl) — record structs pushed by the forward
  recorder
- [`MmapTapeStorage.jl`](MmapTapeStorage.jl) — on-disk policy: cursor
  reservation, mmap reads (views on the CPU, the role- and shape-keyed device
  caches on a GPU), manifest writer and loader, `_build_window_storage`
- [`CheckpointSchedule.jl`](CheckpointSchedule.jl) — schedule types and
  stride window arithmetic

## Common Tasks

- Adding a storage policy: subtype `AbstractCSTapeStorage` (or
  `AbstractHostCSTapeStorage` if reads copy into device caches), implement
  `_allocate_tape_slot`, `stage_panels!`, `_tape_panels(slot, role)`, and (if
  it holds resources) `finalize_tape!`; add a `_tape_storage(::Val{:name})`
  method.
- Adding a record type: define it in [`TapeRecords.jl`](TapeRecords.jl),
  add a branch in `_walk_window_reverse!`
  (`../Adjoints/Footprint/ReverseLoop.jl`).
- Changing the on-disk format: bump `_MMAP_TAPE_FORMAT_VERSION`; the loader
  rejects any other version.
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
- Mmap manifests describe one finalised `records.bin`. The recording
  constructor deletes any old manifest before truncating the binary.
  `load_mmap_tape` rejects tapes that are not finalised, use another version
  or byte order, or whose file size differs from `meta.total_bytes`.
- The manifest is written only by an explicit `finalize_tape!`. The GC
  finalizer only closes the file and deletes owned temp dirs.
- Each mmap record stores its own eltype. The loader accepts only
  `Float32`, `Float64`, and `Float16`. On the CPU, `_tape_panels` returns
  mmap views without copying. On the GPU, it copies into a device cache
  keyed by read role and shape.
- A host-side read stays valid until the next read with the same role. A
  reverse kernel that takes two staged states reads them with two roles:
  the limited-PPM sweep record reads `m` with `AirMassRead()` and `rm` with
  `TracerMassRead()` (one shared cache gave the kernel `m === rm`, issue #57).
- Only staged slots are copied. Flux tuples (`_CSSweepRecord.panels_flux`)
  and forcing/operator objects are kept by reference, so callers must not
  change them before the reverse pass ends.
- `RevolveCheckpoint` is plain bisection, not optimal binomial Revolve.
  Every recorded step starts with production's step-start halo refresh, so a
  window that restarts at a step boundary records the states of the full
  tape: `StrideCheckpoint` and `RevolveCheckpoint` agree with
  `FullCheckpoint` within the tested `atol = 1e-12` for every scheme family,
  with diffusion and convection. (The former ~1e-7 Revolve drift with
  monotone PPM plus convection was that refresh missing from the recorders,
  issue #60, not limiter flips.)

## Related Docs And Tests

- API reference: [`../../docs/src/api/adjoints.md`](../../docs/src/api/adjoints.md);
  support matrix: [`../../docs/src/theory/adjoint_status.md`](../../docs/src/theory/adjoint_status.md)
- [`../../test/core/test_cs_tape_mmap_roundtrip.jl`](../../test/core/test_cs_tape_mmap_roundtrip.jl):
  slot staging, manifest, reload, `get_record`, byte estimate
- [`../../test/core/test_cs_tape_path.jl`](../../test/core/test_cs_tape_path.jl):
  `tape_path` handling and per-window subdirectories
- [`../../test/core/test_cs_stride_checkpoint.jl`](../../test/core/test_cs_stride_checkpoint.jl):
  Stride and Revolve parity against `FullCheckpoint`, window arithmetic
- [`../../test/core/test_persistence_hardening.jl`](../../test/core/test_persistence_hardening.jl):
  a stale manifest cannot validate a re-recorded tape
- [`../../test/core/test_public_api_surface.jl`](../../test/core/test_public_api_surface.jl):
  `CSTapeSlot` is reachable but not exported at top level

## Depends on

Standard libraries Mmap, TOML and Dates. Tape contains storage and records;
it imports no adjoint or inversion implementation.

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
