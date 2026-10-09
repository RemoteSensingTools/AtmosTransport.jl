# Tape

Storage policies, tape record types, and checkpoint schedules for the
cubed-sphere adjoint reverse pass.

A tape is the list of records the forward pass writes so the reverse pass
can replay them backwards. This module owns where staged panel snapshots
live (device memory, host memory, or an mmap file on disk), the record
structs that the reverse loop dispatches on, and the schedules that trade
recomputation for tape size. It contains no transport physics and no
reverse-loop logic; recording lives in `../Footprint/TapeRecording.jl` and
the reverse walk lives in `../Footprint/ReverseLoop.jl`.

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
    copies plus one shared device read cache. The generic method allocates
    plain `Array`s; the CUDA extension overrides `_allocate_tape_slot` and
    `stage_panels!` for `CuArray` panels to use `CUDA.pin` memory.
  - `MmapCSTapeStorage(; dir, cleanup_on_finalize)` in
    [`MmapTapeStorage.jl`](MmapTapeStorage.jl): appends payloads to
    `records.bin` and writes `manifest.toml` on `finalize_tape!`.
- Symbol resolution: `_tape_storage(:device | :pinned_host | :mmap)`.
  `_resolve_tape_path` and `_build_window_storage` turn the public
  `tape_storage` / `tape_path` kwargs into one storage per tape or window.
- Slot API: `_stage_panels(storage, panels)` allocates and fills a slot;
  `_tape_panels(slot)` returns readable panels; `stage_panels!` refills.
- Persistence: `finalize_tape!`, `load_mmap_tape(dir; readonly = true)`,
  `get_record(storage, id)` in [`MmapTapeStorage.jl`](MmapTapeStorage.jl).
- Records in [`TapeRecords.jl`](TapeRecords.jl): `_CSSweepRecord`,
  `_CSHaloRecord`, `_CSMidpointRecord`, `_CSDiffusionRecord`,
  `_CSConvectionRecord`, and the union `_CSTapeOp`.
- Schedules in [`CheckpointSchedule.jl`](CheckpointSchedule.jl):
  `FullCheckpoint()`, `StrideCheckpoint(K)`, `RevolveCheckpoint()`, plus
  `checkpoint_window_count` and `checkpoint_window_range` (Full and Stride
  only).

## File Map

- [`Tape.jl`](Tape.jl) — module definition, export list, include order
  (`TapeStorage` -> `TapeRecords` -> `MmapTapeStorage` -> `CheckpointSchedule`)
- [`TapeStorage.jl`](TapeStorage.jl) — abstract policy, device and
  pinned-host policies, slot types, staging hooks, `_resolve_tape_path`,
  no-op `finalize_tape!` fallback, `_bytes_per_panel_tuple`
- [`TapeRecords.jl`](TapeRecords.jl) — record structs pushed by the forward
  recorder and the `_CSTapeOp` union
- [`MmapTapeStorage.jl`](MmapTapeStorage.jl) — on-disk policy: cursor
  reservation, per-shape device read cache, manifest writer and loader,
  `_build_window_storage`
- [`CheckpointSchedule.jl`](CheckpointSchedule.jl) — schedule types and
  stride window arithmetic

## Common Tasks

- Adding a storage policy: subtype `AbstractCSTapeStorage`, implement
  `_allocate_tape_slot`, `stage_panels!`, `_tape_panels`, and (if it holds
  resources) `finalize_tape!`; add a `_tape_storage(::Val{:name})` method.
- Adding a record type: define it in [`TapeRecords.jl`](TapeRecords.jl),
  add it to `_CSTapeOp`, and add a branch in `_walk_window_reverse!`
  (`../Footprint/ReverseLoop.jl`).
- Changing the on-disk format: bump `_MMAP_TAPE_FORMAT_VERSION`; the loader
  rejects any other version.
- Changing GPU staging: edit `../../ext/AtmosTransportCUDAExt.jl`, which
  overrides the pinned-host and mmap hooks for `CuArray` panels.
- The driver code that uses `StrideCheckpoint` / `RevolveCheckpoint` is in
  `../Footprint/StrideCheckpoint.jl`, not here.

## Invariants

- `_CSSweepRecord` leaves its scheme parameter unconstrained to avoid a
  `Tape` -> `Adjoints` dependency. The supported-scheme check
  (`CSAdjointSupportedScheme`) happens where records are built and consumed.
- `_CSLinRoodHorizRecord` is defined in `../Adjoints/LinRoodTape.jl` and is
  not part of `_CSTapeOp`; the reverse loop handles it in its own branch.
- Mmap manifests describe one finalised `records.bin`. The recording
  constructor deletes any old manifest before truncating the binary.
  `load_mmap_tape` rejects tapes that are not finalised, use another version
  or byte order, or whose file size differs from `meta.total_bytes`.
- The manifest is written only by an explicit `finalize_tape!`. The GC
  finalizer only closes the file and deletes owned temp dirs.
- Each mmap record stores its own eltype. The loader accepts only
  `Float32`, `Float64`, and `Float16`. On the CPU, `_tape_panels` returns
  mmap views without copying. On the GPU, it copies into a device cache
  keyed by shape.
- Only staged slots are copied. Flux tuples (`_CSSweepRecord.panels_flux`)
  and forcing/operator objects are kept by reference, so callers must not
  change them before the reverse pass ends.
- `RevolveCheckpoint` is plain bisection, not optimal binomial Revolve.
  With monotone PPM plus diffusion or convection it can differ from
  `FullCheckpoint` by about 1e-7, because halo refills at bisection points
  can flip limiter branches. `StrideCheckpoint` agrees with it within the
  tested `atol = 1e-12`.

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
