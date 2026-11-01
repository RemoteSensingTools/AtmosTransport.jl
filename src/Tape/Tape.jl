"""
    Tape

The adjoint tape of the AtmosTransport.jl cubed-sphere reverse pass:
`AdjointTape` (content × storage × checkpointing), the storage descriptors
(device, pinned host, mmap on disk) and the runtime pools a call records into,
tape record types, and checkpoint schedules.

`Tape` imports nothing from other AtmosTransport modules.
`src/AtmosTransport.jl` loads it before `Adjoints`, which imports these names
with `using ..Tape: ...`. The forward recorder, reverse loop and checkpoint
drivers (`src/Adjoints/Footprint/`) and the 4D-Var code (`src/Adjoints/Inversion/`) are child modules of `Adjoints`.

Module dependency order:
    Tape  →  Adjoints core → Footprint → Inversion
"""
module Tape

using KernelAbstractions: get_backend
using KernelAbstractions: CPU as KATapeCPU

export AbstractTapeContent, StateTape, AirMassTape, AdjointTape, AutoTape, TapeReservationError, TapeReservationExceeded,
       AbstractCSTapeStorage,
       AbstractHostCSTapeStorage,
       DeviceCSTapeStorage,
       PinnedHostCSTapeStorage,
       MmapCSTapeStorage,
       CSTapeSlot,
       HostCSTapeSlot,
       StaleTapeSlotError,
       _DeviceSlabPool, _PinnedHostArena, _MmapTapeFile,
       _open_tape_pool, _holds_tape_bytes,
       _rewind_tape_pool!, _release_tape_pool!,
       _tape_panels, _tape_slot_bytes,
       _allocate_tape_slot, stage_panels!, _stage_panels, _stage_computed,
       _free_tape_array!, _free_tape_panels!, _tape_synchronize,
       _pin_host_chunk, _unpin_host_chunk,
       _register_tape_read_backend!, _tape_read_cache!,
       AbstractTapeReadRole, AirMassRead, TracerMassRead, LimiterCodeRead,
       _bytes_per_panel_tuple,
       _CSSweepRecord, _CSHaloRecord, _CSMidpointRecord, _CSStepEndRecord,
       _CSDiffusionRecord, _CSConvectionRecord, _CSAirMassResetRecord,
       AbstractCheckpointSchedule, FullCheckpoint, StrideCheckpoint, StrideSearch,
       checkpoint_window_count, checkpoint_window_range

include("TapeStorage.jl")
include("HostTapeArena.jl")
include("TapeRecords.jl")
include("MmapTapeStorage.jl")
include("CheckpointSchedule.jl")
include("AdjointTape.jl")
include("TapeReservation.jl")

end # module Tape

using KernelAbstractions: get_backend
using KernelAbstractions: CPU as KATapeCPU
