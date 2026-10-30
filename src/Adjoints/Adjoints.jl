"""Cubed-sphere discrete adjoint core; replay and inversion live in its child modules."""
module Adjoints

using KernelAbstractions: @Const, @index, @kernel, get_backend, synchronize
using ..Grids: CubedSphereMesh
using ..Operators.Advection: AbstractLimiter, CSAdjointFV3Vertical, CSAdjointLinearScheme,
    CSAdjointNonlinearScheme, CSAdjointWorkspace, FV3ScalarProfile, LinRoodPPMScheme, NoAdvection,
    PPMScheme, SlopesScheme, UpwindScheme, _accumulate_linrood_halo_adjoint!,
    _adjoint_fill_panel_halos!, _share_lr_seam_faces!,
    apply_linrood_horizontal_adjoint_single_panel!, apply_linrood_update_adjoint!
using ..Operators.Diffusion: ImplicitVerticalDiffusion, LocalHoltslagBovilleKzField, NoDiffusion,
    _vertical_diffusion_cs_single_adjoint_kernel!,
    _vertical_diffusion_cs_single_dkg_adjoint_kernel!, check_diffusion_workspace
using ..Operators.Convection.CMFMC: CMFMCConvection, CMFMCEdgeCloudBase, CMFMCMatrixConvection,
    CMFMCMatrixWorkspace, CMFMCWorkspace, _cmfmc_cs_panel_column_single_adjoint_kernel!,
    _cmfmc_dtrain_array, _cmfmc_matrix_synthetic_forcing, _get_or_compute_n_sub!,
    _refresh_cmfmc_matrix_rates!, _require_edge_cloud_base, invalidate_cmfmc_cache!,
    invalidate_cmfmc_matrix_cache!
using ..Operators.Convection.TM5: TM5Convection, TM5Workspace, _ensure_tm5_scratch!,
    _tm5_cs_panel_column_adjoint_kernel!
using ..Operators.Convection: NoConvection
using ..State: AbstractCSDkgField, AbstractCubedSphereField, GCHPNonlocalPBLField, panel_field,
    update_field!
using ..MetDrivers: ConvectionForcing
import ..MetDrivers: current_time
using ..Operators: AbstractAdvectionScheme
using ..Operators.SurfaceFlux: NoSurfaceFlux, SurfaceFluxOperator
using ..Operators.Chemistry: AbstractChemistryOperator, NoChemistry
using ..Operators.StepPlans: AbstractStepStage, AirMassResetStage, ChemistryAfterStage,
    ChemistryBeforeStage, ConvectionStage, StepOperators, StepPlan, TransportStage, operator_state,
    plan_stages, restore_operator_state!
using ..Observations: AbstractPressureBasis, AbstractRetrievalMapping, AbstractUnplacedPolicy,
    ErrorOnUnplaced, RetrievalColumn, VERTICAL_PLACEMENT_CODES, VerticalPlacement
using ..Tape: AbstractCheckpointSchedule, CSTapeSlot, DeviceCSTapeStorage, FullCheckpoint,
    MmapCSTapeSlot, MmapCSTapeStorage, PinnedHostCSTapeSlot, PinnedHostCSTapeStorage,
    RevolveCheckpoint, StrideCheckpoint, finalize_tape!, get_record, load_mmap_tape

export AbstractCSFootprintObjective, CSLayerMeanObjective, CSColumnMeanObjective,
    CSPlacedPointObjective, CSRetrievalColumnObjective, CSCellAverageObjective, CSSeedObjective,
    CSAdjointWorkspace, CSTapeSlot, DeviceCSTapeStorage, PinnedHostCSTapeStorage, MmapCSTapeStorage,
    MmapCSTapeSlot, PinnedHostCSTapeSlot, finalize_tape!, load_mmap_tape, get_record,
    AbstractCheckpointSchedule, FullCheckpoint, StrideCheckpoint, RevolveCheckpoint,
    evaluate_objective, require_adjoint_support, adjoint_support

# Sweeps whose record stages the tracer mass as well as the air mass.
const _RmTapedSweepScheme = Union{CSAdjointNonlinearScheme, CSAdjointFV3Vertical}
# LinRoodPPMScheme (ORD 5 and 7) is supported via its own horizontal
# tape record (`_CSLinRoodHorizRecord`, LinRoodTape.jl) and the kernel
# adjoints in `src/Operators/Advection/linrood_adjoint_*.jl`; its vertical
# sweeps are `_CSSweepRecord`s of `scheme.vertical` (upwind or the FV3 profile).
# The reverse loop (`_walk_window_reverse!` in `src/Adjoints/Footprint/ReverseLoop.jl`)
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



include("ObjectiveSeeding.jl")
include("StepForcing.jl")
include("PanelArrays.jl")
include("DiffusionAdjoint.jl")
include("ConvectionAdjoint.jl")
include("LinRoodTape.jl")
include("AdjointSupport.jl")

include("Footprint/Footprint.jl")
using .Footprint
include("Inversion/Inversion.jl")
using .Inversion

export CELL_AVERAGE_PLACEMENT_CODE, AbstractFlaggedRecordPolicy, DropFlaggedRecords,
    CosampleFlaggedRecords, AbstractCellAveraging, NoCellAveraging, CellAverage,
    inflate_cell_model_error, read_oco2_mip_observations, convert_oco2_mip_observations,
    CSFootprintResult, CSSurfaceFluxWindow, CSSurfaceFluxJacobianResult, CSTimeSampling,
    CSObservation, CSSurfaceFluxControl, CS4DVarResult, CS4DVarSolveResult, CSObservationStencil,
    COLUMN_MEAN_PLACEMENT_CODE, FIXED_LEVEL_PLACEMENT_CODE, RETRIEVAL_COLUMN_PLACEMENT_CODE,
    CSObservationColumnMet, AbstractObservationKind, PointObservationKind, ColumnObservationKind,
    CSObservationRecord, CSObservationSet, read_observations, write_observations,
    AbstractOutOfRangePolicy, RejectOutOfRange, SkipOutOfRange, ClampOutOfRange, bind_to_mesh,
    CSDepartureRecord, CSDepartureSet, build_departure_set, read_departures, write_departures,
    AbstractCSSurfaceFluxCovariance, DiagonalCSCovariance, IsotropicGaussianCSCovariance,
    apply_B_half!, apply_B_half_adjoint!, AbstractCSOptimType, LinearOptimType, LogNormalOptimType,
    CSSurfaceFluxPreconditioner, apply_preconditioner!, apply_preconditioner_adjoint!,
    CSTapeByteEstimate, run_cs_footprint_forward, cs_surface_emission_footprint,
    cs_surface_emission_footprint_from_seed, cs_tape_byte_estimate, cs_surface_flux_jacobian,
    cs_surface_flux_4dvar, cs_surface_flux_4dvar_optimize, AbstractCSOptimizer, CSGradientDescent,
    CSLBFGS, cs_surface_flux_4dvar_solve, CSStepSchedule, CSIterationLog, CSIterationLogEntry

end # module Adjoints
