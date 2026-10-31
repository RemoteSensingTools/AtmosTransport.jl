"""
    Tape

Tape storage policies (device, pinned-host, mmap on disk), tape record types,
and checkpoint schedules for the AtmosTransport.jl cubed-sphere adjoint
pipeline.

`Tape` imports nothing from other AtmosTransport modules.
`src/AtmosTransport.jl` loads it before `Adjoints`, which imports these names
with `using ..Tape: ...`. The forward recorder, reverse loop and checkpoint
drivers (`src/Adjoints/Footprint/`) and the 4D-Var code (`src/Adjoints/Inversion/`) are child modules of `Adjoints`.

Module dependency order:
    Tape  →  Adjoints core → Footprint → Inversion
"""
module Tape

export AbstractCSTapeStorage,
       AbstractHostCSTapeStorage,
       DeviceCSTapeStorage,
       PinnedHostCSTapeStorage,
       MmapCSTapeStorage,
       CSTapeSlot,
       PinnedHostCSTapeSlot,
       MmapCSTapeSlot,
       _MmapTapeFile,
       _tape_panels,
       _validate_tape_storage, _needs_tape_size, _open_tape_storage,
       _rewind_tape_storage!, _release_tape_storage!, _release_opened_tape_storage!,
       _tape_slot_bytes,
       _allocate_tape_slot, stage_panels!, _stage_panels,
       _after_tape_stage!, _after_tape_read!,
       _mmap_prepare_for_panels!,
       _register_tape_read_backend!, _tape_read_cache!,
       AbstractTapeReadRole, AirMassRead, TracerMassRead,
       _bytes_per_panel_tuple,
       _CSSweepRecord, _CSHaloRecord, _CSMidpointRecord, _CSStepEndRecord,
       _CSDiffusionRecord, _CSConvectionRecord, _CSAirMassResetRecord,
       AbstractCheckpointSchedule, FullCheckpoint, StrideCheckpoint,
       RevolveCheckpoint,
       checkpoint_window_count, checkpoint_window_range

include("TapeStorage.jl")
include("TapeRecords.jl")
include("MmapTapeStorage.jl")
include("CheckpointSchedule.jl")

end # module Tape
