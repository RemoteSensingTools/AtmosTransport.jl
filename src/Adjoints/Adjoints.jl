"""
    Adjoints

Prototype adjoint and footprint utilities.

The shipped path is a kernelized cubed-sphere reverse pass for tracer mass
with fixed meteorology/air-mass evolution. It accumulates surface-emission
footprints (and the initial-condition sensitivity) from an adjoint seed or
observation forcing without perturbing each surface cell.

The forward model is the production step by construction: the adjoint
replays the production `StepPlan` (`CSStepSchedule`: physics cadence, met
windows, air-mass resets) with the production operators on a single-tracer
state, records its tape through the recorder hooks of the production step
(`Operators.AbstractStepRecorder`) and walks the tape backwards with each
operator's adjoint. `require_adjoint_support` rejects a plan containing an
operator without an adjoint before anything is recorded.
"""
module Adjoints

using ..Grids: canonical_seams

using ..Grids: edge_interior_ij, edge_halo_ij, corner_source_ij, cs_seam_axis, cs_seam_sign, cs_seam_face_index, cs_seam

using KernelAbstractions: @Const, @atomic, @index, @kernel, get_backend, synchronize
# NCDatasets is loaded here (not inside ObservationsIO.jl) so the
# NetCDF readers/writers under `src/Inversion/` (ObservationsIO.jl,
# DeparturesIO.jl) share one import.
import NCDatasets

using ..Grids: CubedSphereMesh, reciprocal_edge,
    EDGE_NORTH, EDGE_SOUTH, EDGE_EAST, EDGE_WEST,
    panel_cell_center_lonlat,
    panel_convention, cs_definition, cs_definition_tag
using ..Operators.Advection: CSAdvectionWorkspace, NoLimiter,
    CSSplitSweepStyle, CSLinRoodStyle, cs_advection_style, fill_cs_step_start_halos!,
    MonotoneLimiter, CW84Limiter, PPMScheme, SameAsHorizontal, SlopesScheme, UpwindScheme,
    LinRoodPPMScheme, NoAdvection, FV3ScalarProfile, AbstractLimiter,
    fill_panel_halos!, strang_split_cs!, copy_corners!,
    _cs_static_palindrome_subcycle_count,
    _sweep_z_panel!, _sweep_cs_horizontal!,
    _CSInteriorFlux,
    apply_linrood_horizontal_adjoint_single_panel!,
    apply_linrood_update_adjoint!, _share_lr_seam_faces!,
    _init_q_buf_kernel!, _ppm_y_face_kernel!, _ppm_x_face_kernel!,
    _ppm_y_face_from_q_kernel!, _ppm_x_face_from_q_kernel!,
    _pre_advect_y_kernel!, _pre_advect_x_kernel!, _linrood_update_kernel!,
    _copy_interior!, _accumulate_linrood_halo_adjoint!,
    strang_split_linrood_ppm!, CSLinRoodAdvectionWorkspace, LinRoodWorkspace,
    fv_tp_2d_cs!, _sweep_z_panels!, _cs_flux_x_interior, _cs_flux_y_interior,
    _advection_mass_floor, _ppm_edge_weights,  # shared with the forward fluxes (AdvectionAdjoint.jl)
    _wrap_periodic, _HaloPaddedAxis, _validate_halo_for_scheme,
    needs_column_scratch, _fv3_adjoint_column_scratch, _sweep_z_panels_fv3_adjoint!
using ..Operators.Diffusion: NoDiffusion, ImplicitVerticalDiffusion, DiffusionWorkspace,
    apply_vertical_diffusion_vmr!, _dkg_transfer_ratios,
    surface_flux_coupling, surface_flux_center!
using ..Operators.Convection: CMFMCConvection, _require_edge_cloud_base, CMFMCWorkspace,
    NoConvection, TM5Convection, TM5Workspace,
    CMFMCMatrixConvection, CMFMCMatrixWorkspace,
    invalidate_cmfmc_cache!, invalidate_cmfmc_matrix_cache!,
    _get_or_compute_n_sub!, _ensure_tm5_scratch!,
    _tm5_diagnose_cloud_dims, _tm5_build_conv1!, _tm5_lu!, _tm5_factorize!,
    _tm5_identity_pivots, _tm5_column_sum, _tm5_restore_column_mass!,
    _launch_cmfmc_matrix_derivation!,
    _cmfmc_tiny, _cmfmc_cloud_base, _cmfmc_dtrain_array
using ..State: AbstractCubedSphereField,
    LocalHoltslagBovilleKzField, AbstractCSDkgField, PrecomputedCSDkgField, GCHPNonlocalPBLField,
    field_value, panel_field, update_field!
using ..MetDrivers: ConvectionForcing
import ..MetDrivers: current_time, uses_binary_substep_contract
using ..Architectures: _kahan_add, CPU
using ..Grids: AtmosGrid, HybridSigmaPressure
using ..State: CubedSphereState, CubedSphereFaceFluxState
using ..Operators: AbstractAdvectionScheme, apply!
using ..Operators.SurfaceFlux: NoSurfaceFlux, SurfaceFluxOperator, SurfaceFluxSource
using ..Operators.Chemistry: AbstractChemistryOperator, NoChemistry
using ..Operators.Convection: CMFMCEdgeCloudBase
# The step plan the adjoint replays and the recorder hooks its tape recorder
# implements (Operators/StepPlan, Operators/StepRecording.jl).
using ..Operators.StepPlans: StepPlan, StepOperators, StepTiming, AbstractStepStage,
    TransportStage, ConvectionStage, ChemistryBeforeStage, ChemistryAfterStage,
    AirMassResetStage, WindowStartAirMass, WindowEndAirMass, plan_stages,
    execute_window_advance!, execute_window_start!, execute_step_stages!,
    execute_window_end!, operator_state, restore_operator_state!, reset_operator_state!,
    _cs_advection_workspace_for
import ..Operators.StepPlans: air_mass_reset_target
using ..Operators: AbstractStepRecorder, NoStepRecorder
import ..Operators: record_halo!, record_sweep!, record_center!, record_diffusion!,
    record_emission!, record_linrood_horizontal!, record_convection!,
    record_air_mass_reset!, record_chemistry!
using ..Parameters: DEFAULT_CFL_LIMIT, DEFAULT_LINROOD_PPM_ORDER, STANDARD_GRAVITY
using ..State: AbstractMassBasis, DryBasis
# Observation binding and vertical placement (Inversion/): the cell locator and
# the placement rules of the forward observation sampler.
using ..Output: cell_locator, locate, CellLocation,
    VerticalPlacementSpec, VerticalPlacement, VERTICAL_PLACEMENT_CODES, ObservationTerrain,
    SoundingRequest, SiteRequest, UnplacedWithoutHeight, resolve_placement, check_terrain,
    check_surface_height_layout, check_pressure_basis,
    AbstractPressureBasis, PlacementColumn, sample_column_layer!, needs_column_humidity,
    AbstractLayerTemperature, ConstantLayerTemperature, ProfileLayerTemperature,
    ObservationCounters, _handle_unplaced, _count_unplaced!, _unplaced_example, _warn_below_ground,
    vertical_placement_codes_description,
    AbstractUnplacedPolicy, ErrorOnUnplaced, SkipUnplaced, placement_pressures!,
    RetrievalColumn, retrieval_layer_count, retrieval_column_weights!, retrieval_prior_offset,
    retrieval_profile!, retrieval_model_error_variance, mole_fraction_unit_scale, _mole_fraction_scale,
    AbstractRetrievalMapping, LayerRebin, HatLevelProjection, tm5_layer_column, retrieval_mapping_label,
    _parse_retrieval_mapping, retrieval_pressure_count, read_oco2_retrieval_columns, oco2_level_form,
    _nc_variable, _raw_block,
    _PRESSURE_UNITS_PA,
    _pressure_unit_key

# Tape storage policies + record types live in src/Tape/
# (loaded before Adjoints in src/AtmosTransport.jl). Re-imported here so
# call sites continue to use the unqualified names.
using ..Tape: AbstractCSTapeStorage,
              DeviceCSTapeStorage, PinnedHostCSTapeStorage, MmapCSTapeStorage,
              CSTapeSlot, PinnedHostCSTapeSlot, MmapCSTapeSlot,
              _tape_storage, _tape_panels,
              _resolve_tape_path, _validate_tape_path, _build_window_storage,
              _allocate_tape_slot, stage_panels!, _stage_panels,
              _after_tape_stage!, _after_tape_read!,
              _sync_pinned_tape_storage!, _sync_mmap_tape_storage!,
              _mmap_prepare_for_panels!,
              _register_tape_read_backend!, _tape_read_cache!,
              AbstractTapeReadRole, AirMassRead, TracerMassRead,
              _bytes_per_panel_tuple, finalize_tape!,
              load_mmap_tape, get_record,
              _CSSweepRecord, _CSHaloRecord, _CSMidpointRecord, _CSStepEndRecord,
              _CSDiffusionRecord, _CSConvectionRecord, _CSAirMassResetRecord, _CSTapeOp,
              AbstractCheckpointSchedule, FullCheckpoint, StrideCheckpoint,
              RevolveCheckpoint,
              checkpoint_window_count, checkpoint_window_range

# Split-sweep PPM adjoints exist for the default vertical reconstruction only;
# the FV3 vertical profile has an adjoint as Lin–Rood's vertical sweep (below).
const CSAdjointLinearScheme = Union{UpwindScheme, SlopesScheme{NoLimiter},
                                    PPMScheme{NoLimiter, SameAsHorizontal}}
# Limiters whose PPM adjoint replays the forward branches from taped tracer masses.
const _PPMTapedLimiter = Union{MonotoneLimiter, CW84Limiter}
const CSAdjointNonlinearScheme = PPMScheme{<:_PPMTapedLimiter, SameAsHorizontal}
# The FV3 `kord = 8` vertical sweep (`vertical = "fv3_kord8"` or
# `"fv3_kord8_signed"`). Its records stage the tracer mass entering the sweep;
# the reverse replays the forward profile there and applies the transpose at the
# forward's limiter branches (`vertical_fv3_profile_adjoint.jl`). Only Lin–Rood's
# vertical sweeps record it.
const CSAdjointFV3Vertical = PPMScheme{<:AbstractLimiter, <:FV3ScalarProfile}
# Sweeps whose record stages the tracer mass as well as the air mass.
const _RmTapedSweepScheme = Union{CSAdjointNonlinearScheme, CSAdjointFV3Vertical}
# LinRoodPPMScheme (ORD 5 and 7) is supported via its own horizontal
# tape record (`_CSLinRoodHorizRecord`, LinRoodTape.jl) and the kernel
# adjoints in `src/Operators/Advection/linrood_adjoint_*.jl`; its vertical
# sweeps are `_CSSweepRecord`s of `scheme.vertical` (upwind or the FV3 profile).
# The reverse loop (`_walk_window_reverse!` in `src/Footprint/ReverseLoop.jl`)
# handles the horizontal record in its own branch alongside the
# `_CSSweepRecord`, `_CSHaloRecord`, `_CSDiffusionRecord`,
# `_CSConvectionRecord`, and `_CSMidpointRecord` cases.
const CSAdjointLinRoodScheme = LinRoodPPMScheme{<:Any, <:Union{UpwindScheme, CSAdjointFV3Vertical}}
const CSAdjointSupportedScheme = Union{CSAdjointLinearScheme,
                                        CSAdjointNonlinearScheme,
                                        CSAdjointLinRoodScheme}

"""
    require_adjoint_support(scheme) -> scheme

Return the advection `scheme` if the cubed-sphere reverse pass has an adjoint
for it, otherwise throw an `ArgumentError` that names the missing piece and
lists the supported choices. Dispatches on the union of supported schemes that
the footprint and 4D-Var entry points also accept, so a configuration is
rejected with an explanation before any tape is recorded.

Supported: `UpwindScheme()`, `SlopesScheme(NoLimiter())`, `PPMScheme` with
`NoLimiter`, `MonotoneLimiter` or `CW84Limiter` and the default vertical
reconstruction, and `LinRoodPPMScheme` (ORD 5 or 7) with any of its vertical
sweeps: upwind or the FV3 `kord = 8` profile (`vertical = "fv3_kord8"` or
`"fv3_kord8_signed"`). Unsupported: limited `SlopesScheme`, the split-sweep
`PPMScheme` with the FV3 vertical profile, `NoAdvection`, and every other
scheme or limiter.
"""
require_adjoint_support(scheme::CSAdjointSupportedScheme) = scheme
require_adjoint_support(scheme) = throw(ArgumentError(
    _adjoint_gap(scheme) * " Advection with a cubed-sphere adjoint: " *
    "UpwindScheme(), SlopesScheme(NoLimiter()), PPMScheme with NoLimiter, " *
    "MonotoneLimiter or CW84Limiter and the default vertical, or " *
    "LinRoodPPMScheme(5 or 7) with upwind or FV3 kord-8 vertical sweeps."))

# The part of `scheme` that has no adjoint, for `require_adjoint_support`.
_adjoint_gap(s::SlopesScheme) =
    "SlopesScheme with $(nameof(typeof(s.limiter))) has no adjoint; only the unlimited " *
    "SlopesScheme(NoLimiter()) does."
_adjoint_gap(::PPMScheme{<:AbstractLimiter, <:FV3ScalarProfile}) =
    "PPMScheme with the FV3 vertical profile (vertical = \"fv3_kord8\" or " *
    "\"fv3_kord8_signed\") has no split-sweep adjoint; the profile's adjoint " *
    "covers the vertical sweeps of scheme = \"linrood\"."
_adjoint_gap(s::PPMScheme) = "PPMScheme with $(nameof(typeof(s.limiter))) has no adjoint."
_adjoint_gap(::NoAdvection) =
    "NoAdvection (scheme = \"none\") has no record in the cubed-sphere tape."
_adjoint_gap(s) = "$(nameof(typeof(s))) has no cubed-sphere adjoint."

# Footprint-objective types + seeding/eval kernels.
include("ObjectiveSeeding.jl")


# CSFootprintResult + CSTapeByteEstimate.
include("../Footprint/FootprintResult.jl")

# Observation operators as sparse linear functionals (`CSLinearFunctional`)
# and the end-of-step adjoint forcing of the reverse walk (`NoStepForcing`,
# `CSStepForcing`, `_inject_step_forcing!`); `_seed_objective!` above injects
# one functional through the same kernel.
include("../Footprint/StepForcing.jl")


# Control / observation / 4D-Var result types (`CSTimeSampling`,
# `CSObservation`, `CSObservationStencil`, `CS4DVarResult`, ...).
include("../Inversion/Observations.jl")

# The column meteorology of placed point observations
# (`CSObservationColumnMet`) and the per-step sampling of every objective
# (`_sample_objective`: value, linear functional and level), used by the
# forward replay `_observe_cs_forward`.
include("../Inversion/ObservationColumns.jl")

# On-disk observation IO (CSObservationRecord / CSObservationSet, read/write
# to the v2 NetCDF schema documented in `schemas/cs_observations_v2.toml`).
include("../Inversion/ObservationsIO.jl")

# `bind_to_mesh` bridge from the on-disk `CSObservationSet` to a 4D-Var-ready
# `Vector{CSObservation}`: time sampling, containing cell, and the vertical
# placement of point records (`CSPlacedPointObjective`) or column means.
include("../Inversion/ObservationBinding.jl")

# On-disk departures (forward-pass simulated values and sampling stencils
# paired with their originating observation records).
include("../Inversion/DeparturesIO.jl")
# OCO-2 / OCO-3 v11 MIP 10-second files → observation schema v2.
include("../Inversion/OCO2MIPObservations.jl")

# Surface-flux background-error covariance B and its
# spectral square root B^(1/2) for the preconditioned 4D-Var path.
# Self-contained (depends only on FFTW + base) so include order is not
# load-bearing; placed here so the export block below can re-export
# the public surface.
include("../Inversion/Covariance.jl")

# Preconditioner (χ ↔ x change of variables) with
# Linear and LogNormal optim types. Depends on Covariance.jl for
# `apply_B_half!` / `apply_B_half_adjoint!` / `apply_B_half_inverse!`,
# so loaded immediately after.
include("../Inversion/Preconditioning.jl")


"""
    CSAdjointWorkspace(mesh, prototype; column_scratch = false)

Scratch storage for CS adjoint sweeps. `lambda_A` is one halo-padded panel
used as the per-panel transpose output. `seam_seeds` caches both neighbors'
output-seed difference before any panel transpose overwrites those values.
`column_scratch` is the per-column working storage of the FV3 vertical sweep's
reverse pass (`Nc × Nc × (Nz+1) × 5`); it is empty unless built with
`column_scratch = true`, which a tape with FV3-profile vertical sweeps needs
(`needs_column_scratch(scheme)`).
"""
struct CSAdjointWorkspace{FT, A <: AbstractArray{FT, 3}, A4 <: AbstractArray{FT, 4}}
    lambda_A::A
    seam_seeds::A
    column_scratch::A4
end

function CSAdjointWorkspace(mesh::CubedSphereMesh,
                            prototype::AbstractArray{FT, 3};
                            column_scratch::Bool = false) where {FT <: AbstractFloat}
    N = mesh.Nc + 2 * mesh.Hp
    Nz = size(prototype, 3)
    lambda_A = similar(prototype, FT, N, N, Nz)
    seam_seeds = similar(prototype, FT, mesh.Nc, Nz, 12)
    scratch = _fv3_adjoint_column_scratch(lambda_A, mesh.Nc, Nz, column_scratch)
    return CSAdjointWorkspace{FT, typeof(lambda_A), typeof(scratch)}(lambda_A, seam_seeds, scratch)
end

_copy_panel_tuple(panels) = ntuple(p -> copy(panels[p]), 6)

function _zero_panel_tuple_like(panels)
    return ntuple(p -> begin
        a = similar(panels[p])
        fill!(a, zero(eltype(a)))
        a
    end, 6)
end

function _zero_surface_rates(mesh::CubedSphereMesh, prototype::AbstractArray{FT}) where {FT}
    return ntuple(_ -> begin
        a = similar(prototype, FT, mesh.Nc, mesh.Nc)
        fill!(a, zero(FT))
        a
    end, 6)
end

function _validate_step_sequences(panels_am_steps, panels_bm_steps, panels_cm_steps)
    nsteps = length(panels_am_steps)
    nsteps > 0 || throw(ArgumentError("footprint generation requires at least one step"))
    length(panels_bm_steps) == nsteps ||
        throw(ArgumentError("panels_bm_steps length $(length(panels_bm_steps)) does not match panels_am_steps length $nsteps"))
    length(panels_cm_steps) == nsteps ||
        throw(ArgumentError("panels_cm_steps length $(length(panels_cm_steps)) does not match panels_am_steps length $nsteps"))
    return nsteps
end

function _validate_emission_rates(rates, nsteps::Int, mesh::CubedSphereMesh,
                                  name::AbstractString)
    rates === nothing && return nothing
    length(rates) == nsteps || throw(ArgumentError(
        "$name length $(length(rates)) does not match nsteps $nsteps"))
    @inbounds for step in 1:nsteps, p in 1:6
        size(rates[step][p]) == (mesh.Nc, mesh.Nc) || throw(DimensionMismatch(
            "$name step $step panel $p has shape $(size(rates[step][p])); " *
            "expected $((mesh.Nc, mesh.Nc))"))
    end
    return nothing
end


# Split-sweep advection adjoint kernels.
include("AdvectionAdjoint.jl")
include("CubedSphereSeams.jl")

# CS halo-exchange adjoint kernels.
include("HaloAdjoint.jl")





# Vertical-diffusion adjoint kernels.
include("DiffusionAdjoint.jl")

# CMFMC, TM5 and CMFMC-matrix convection adjoint kernels.
include("ConvectionAdjoint.jl")


# Tape storage policies + staging API live in
# `src/Tape/TapeStorage.jl` and are imported via `using ..Tape` at
# the top of this module.


# Tape record types (_CSSweepRecord, _CSHaloRecord,
# _CSMidpointRecord, _CSDiffusionRecord, _CSConvectionRecord,
# _CSStepEndRecord) and the
# _CSTapeOp union live in `src/Tape/TapeRecords.jl` and are imported
# via `using ..Tape` at the top of this module. The `S <: CSAdjointSupportedScheme`
# type constraint on `_CSSweepRecord.scheme` is relaxed to plain `S`
# there; the constraint is now enforced at the
# `_record_sweep!`/`_adjoint_scheme_sweep!` call sites below.

# LinRoodPPMScheme tape record and its reverse: `_CSLinRoodHorizRecord`,
# `_apply_cs_linrood_horizontal_adjoint!`.
include("LinRoodTape.jl")

# The adjoint's forward model: `CSStepSchedule` and the replay of the
# production step plan (`_CSReplay`).
include("../Footprint/StepReplay.jl")

# The setup guard: `adjoint_support` and `require_adjoint_support`.
include("AdjointSupport.jl")

# Tape and counting recorders (the production step's recorder hooks) and the
# record builders / tape-size estimate.
include("../Footprint/TapeRecorder.jl")
include("../Footprint/TapeRecording.jl")

# Reverse walk + forward replays.
include("../Footprint/ReverseLoop.jl")

# Full, Stride and Revolve checkpoint drivers, one implementation for every
# scheme family: they save and restore replay snapshots, record windows of
# the replay and walk them with `_walk_window_reverse!`.
include("../Footprint/StrideCheckpoint.jl")
include("../Footprint/revolve_checkpoint.jl")


# User-facing footprint API.
include("../Footprint/FootprintAPI.jl")

# Surface-flux Jacobian + aggregation.
include("../Inversion/Jacobian.jl")

# 4D-Var cost + gradient evaluation.
include("../Inversion/CostGradient.jl")

# 4D-Var optimizer backends (`CSGradientDescent`, `CSLBFGS` via Optim.jl)
# and the `cs_surface_flux_4dvar_optimize` entry point.
include("../Inversion/Optimizer.jl")






export AbstractCSFootprintObjective
export CSLayerMeanObjective, CSColumnMeanObjective, CSPlacedPointObjective, CSRetrievalColumnObjective
export CSCellAverageObjective, CELL_AVERAGE_PLACEMENT_CODE
export AbstractFlaggedRecordPolicy, DropFlaggedRecords, CosampleFlaggedRecords
export AbstractCellAveraging, NoCellAveraging, CellAverage, inflate_cell_model_error
export read_oco2_mip_observations, convert_oco2_mip_observations
export CSSeedObjective, CSFootprintResult
export CSSurfaceFluxWindow, CSSurfaceFluxJacobianResult
export CSTimeSampling, CSObservation, CSSurfaceFluxControl, CS4DVarResult, CS4DVarSolveResult
export CSObservationStencil, COLUMN_MEAN_PLACEMENT_CODE, FIXED_LEVEL_PLACEMENT_CODE
export RETRIEVAL_COLUMN_PLACEMENT_CODE
export CSObservationColumnMet
export AbstractObservationKind, PointObservationKind, ColumnObservationKind
export CSObservationRecord, CSObservationSet, read_observations, write_observations
export AbstractOutOfRangePolicy, RejectOutOfRange, SkipOutOfRange, ClampOutOfRange
export bind_to_mesh
export CSDepartureRecord, CSDepartureSet, build_departure_set
export read_departures, write_departures
export AbstractCSSurfaceFluxCovariance
export DiagonalCSCovariance, IsotropicGaussianCSCovariance
export apply_B_half!, apply_B_half_adjoint!, apply_B_half_inverse!
export AbstractCSOptimType, LinearOptimType, LogNormalOptimType
export CSSurfaceFluxPreconditioner
export apply_preconditioner!, apply_preconditioner_inverse!
export apply_preconditioner_tangent!, apply_preconditioner_adjoint!
export CSAdjointWorkspace, CSTapeSlot, DeviceCSTapeStorage, PinnedHostCSTapeStorage
export MmapCSTapeStorage, MmapCSTapeSlot
export PinnedHostCSTapeSlot, CSTapeByteEstimate
export finalize_tape!
export load_mmap_tape, get_record
export AbstractCheckpointSchedule, FullCheckpoint, StrideCheckpoint, RevolveCheckpoint
export evaluate_objective, run_cs_footprint_forward
export cs_surface_emission_footprint, cs_surface_emission_footprint_from_seed
export cs_tape_byte_estimate
export cs_surface_flux_jacobian
export cs_surface_flux_4dvar, cs_surface_flux_4dvar_optimize
export AbstractCSOptimizer, CSGradientDescent, CSLBFGS, cs_surface_flux_4dvar_solve
export require_adjoint_support, adjoint_support
export CSStepSchedule
export CSIterationLog, CSIterationLogEntry

end # module Adjoints
