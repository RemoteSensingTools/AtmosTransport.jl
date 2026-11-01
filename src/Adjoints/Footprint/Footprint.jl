module Footprint
using ...Architectures: zeros_like

using KernelAbstractions: @Const, @index, @kernel, @localmem, @synchronize, get_backend, synchronize
using KernelAbstractions: CPU as KACPU
using ...Grids: AtmosGrid, CubedSphereMesh, HybridSigmaPressure
using ...Operators.Advection: CSAdjointFV3Vertical, CSAdjointLinearScheme, CSAdjointNonlinearScheme,
    CSAdjointWorkspace, CSLinRoodStyle, CSSplitSweepStyle, NoLimiter, PPMScheme,
    _adjoint_fill_panel_halos!, _adjoint_scheme_sweep!, cs_advection_style, fill_panel_halos!,
    needs_column_scratch, record_ppm_limiter_codes!
using ...Operators.Diffusion: DiffusionWorkspace, NoDiffusion, surface_flux_center!,
    surface_flux_coupling
using ...Operators.Convection.CMFMC: CMFMCConvection, CMFMCWorkspace, invalidate_cmfmc_cache!
using ...Operators.Convection: NoConvection
import ...MetDrivers: current_time, uses_binary_substep_contract
using ...Architectures: CPU
using ...State: CubedSphereFaceFluxState, CubedSphereState, DryBasis
using ...MetDrivers: ConvectionForcing
using ...Operators.SurfaceFlux: NoSurfaceFlux, SurfaceFluxOperator, SurfaceFluxSource
using ...Operators.Chemistry: NoChemistry
using ...Operators.StepPlans: window_clock, window_step_dt, AirMassResetStage, ConvectionStage,
    StepOperators, StepPlan, StepTiming, TransportStage, WindowEndAirMass, WindowStartAirMass,
    _cs_advection_workspace_for, execute_step_stages!, execute_window_advance!, execute_window_end!,
    execute_window_start!, operator_state, reset_operator_state!, restore_operator_state!
import ...Operators.StepPlans: air_mass_reset_target
using ...Operators: AbstractStepRecorder, NoStepRecorder
import ...Operators: record_air_mass_reset!, record_center!, record_chemistry!, record_convection!,
    record_diffusion!, record_emission!, record_halo!, record_linrood_horizontal!, record_sweep!
using ...Parameters: DEFAULT_CFL_LIMIT
using ...Tape: AbstractCheckpointSchedule, AbstractTapeContent, AdjointTape, AirMassRead,
    AirMassTape, FullCheckpoint, LimiterCodeRead, StateTape, StrideCheckpoint, TracerMassRead,
    _CSAirMassResetRecord, _CSConvectionRecord, _CSDiffusionRecord, _CSHaloRecord,
    _CSMidpointRecord, _CSStepEndRecord, _CSSweepRecord, _DeviceSlabPool, _bytes_per_panel_tuple,
    _free_tape_array!, _free_tape_panels!, _holds_tape_bytes, _needs_tape_size, _open_tape_pool,
    _release_tape_pool!, _rewind_tape_pool!, _stage_computed, _stage_panels, _tape_panels,
    _tape_slot_bytes, checkpoint_window_count, checkpoint_window_range
import ..Adjoints: require_adjoint_support
using ..Adjoints: AbstractCSFootprintObjective, AbstractStepForcing, AirMassDependence,
    CSAdjointLinRoodScheme, CSAdjointSupportedScheme, CSCellAverageObjective,
    CSPlacedPointObjective, CSRetrievalColumnObjective, CSSeedObjective, NoStepForcing,
    TracerDependence, _CSLinRoodHorizRecord, _accumulate_surface_footprint!,
    _apply_cs_convection_adjoint!,
    _apply_cs_diffusion_adjoint!, _apply_cs_linrood_horizontal_adjoint!, _cell_average_needs_4dvar,
    _copy_panel_tuple, _inject_step_forcing!, _linrood_validate_tape_storage,
    _placed_objective_needs_met, _require_cs_convection_workspace, _retrieval_objective_needs_met,
    _seed_objective!, _validate_cs_diffusion_inputs, _validate_objective, _validate_step_forcing,
    _zero_surface_rates, evaluate_objective, tape_dependence

export CSFootprintResult, CSTapeByteEstimate, run_cs_footprint_forward,
    cs_surface_emission_footprint, cs_surface_emission_footprint_from_seed, cs_tape_byte_estimate,
    CSStepSchedule

include("FootprintResult.jl")
include("StepReplay.jl")
include("ReplaySupport.jl")
include("TapeRecorder.jl")
include("TapeRecording.jl")
include("ReverseLoop.jl")
include("TapeRuntime.jl")
include("StrideCheckpoint.jl")
include("FootprintAPI.jl")

end # module Footprint
