"""
    Tape

Tape storage policies (device, pinned-host, mmap on disk), tape record types,
and checkpoint schedules for the AtmosTransport.jl cubed-sphere adjoint
pipeline.

`Tape` imports nothing from other AtmosTransport modules.
`src/AtmosTransport.jl` loads it before `Adjoints`, which imports these names
with `using ..Tape: ...`. The forward recorder, reverse loop and checkpoint
drivers (`src/Footprint/`) and the 4D-Var code (`src/Inversion/`) are files
included into the `Adjoints` module, not separate modules.

Module dependency order:
    Tape  →  Adjoints (includes Footprint/ and Inversion/ files)
"""
module Tape

export AbstractCSTapeStorage,
       DeviceCSTapeStorage,
       PinnedHostCSTapeStorage,
       MmapCSTapeStorage,
       CSTapeSlot,
       PinnedHostCSTapeSlot,
       MmapCSTapeSlot,
       _tape_storage, _tape_panels,
       _resolve_tape_path, _build_window_storage,
       _allocate_tape_slot, stage_panels!, _stage_panels,
       _after_tape_stage!, _after_tape_read!,
       _sync_pinned_tape_storage!,
       _sync_mmap_tape_storage!,
       _mmap_prepare_for_panels!,
       _ensure_tape_read_cache!,
       _bytes_per_panel_tuple,
       finalize_tape!,
       load_mmap_tape, get_record,
       _CSSweepRecord, _CSHaloRecord, _CSMidpointRecord,
       _CSDiffusionRecord, _CSConvectionRecord,
       _CSTapeOp,
       AbstractCheckpointSchedule, FullCheckpoint, StrideCheckpoint,
       RevolveCheckpoint,
       checkpoint_window_count, checkpoint_window_range

include("TapeStorage.jl")
include("TapeRecords.jl")
include("MmapTapeStorage.jl")
include("CheckpointSchedule.jl")

end # module Tape
