"""
    Adjoints

Prototype adjoint and footprint utilities.

The shipped path here is a kernelized CS split-sweep reverse pass for tracer
mass with fixed meteorology/air-mass evolution. It accumulates
surface-emission footprints from an adjoint seed without perturbing each
surface cell. The optional vertical-diffusion slot mirrors the runtime
surface-source palindrome: half diffusion, midpoint emissions, half diffusion.
The optional convection slot transposes the CS `CMFMCConvection` and
`TM5Convection` column operators.
"""
module Adjoints

using KernelAbstractions: @Const, @atomic, @index, @kernel, get_backend, synchronize
# NCDatasets is loaded here (not inside ObservationsIO.jl) so the
# observation IO can share the dependency with any
# future read/write helpers under `src/Inversion/`.
import NCDatasets

using ..Grids: CubedSphereMesh, reciprocal_edge,
    EDGE_NORTH, EDGE_SOUTH, EDGE_EAST, EDGE_WEST,
    panel_cell_center_lonlat,
    panel_convention, cs_definition, cs_definition_tag
using ..Operators.Advection: CSAdvectionWorkspace, NoLimiter,
    MonotoneLimiter, PPMScheme, SameAsHorizontal, SlopesScheme, UpwindScheme,
    LinRoodPPMScheme,
    fill_panel_halos!, strang_split_cs!, copy_corners!,
    _cs_static_palindrome_subcycle_count,
    _sweep_z_panel!, _sweep_cs_horizontal!,
    _CSInteriorFlux, _cs_seam_axis, _cs_seam_sign, _cs_seam_face_index,
    apply_linrood_horizontal_adjoint_single_panel!,
    apply_linrood_update_adjoint!, _share_lr_seam_faces!,
    _init_q_buf_kernel!, _ppm_y_face_kernel!, _ppm_x_face_kernel!,
    _ppm_y_face_from_q_kernel!, _ppm_x_face_from_q_kernel!,
    _pre_advect_y_kernel!, _pre_advect_x_kernel!, _linrood_update_kernel!,
    _copy_interior!, _accumulate_linrood_halo_adjoint!,
    strang_split_linrood_ppm!, CSLinRoodAdvectionWorkspace, LinRoodWorkspace,
    fv_tp_2d_cs!, _sweep_z!, _cs_flux_x_interior, _cs_flux_y_interior
using ..Operators.Diffusion: NoDiffusion, ImplicitVerticalDiffusion,
    apply_vertical_diffusion_vmr!, _dkg_transfer_ratios
using ..Operators.Convection: CMFMCConvection, _require_edge_cloud_base, CMFMCWorkspace,
    NoConvection, TM5Convection, TM5Workspace,
    CMFMCMatrixConvection, CMFMCMatrixWorkspace,
    invalidate_cmfmc_cache!, invalidate_cmfmc_matrix_cache!,
    _get_or_compute_n_sub!, _ensure_tm5_scratch!,
    _tm5_diagnose_cloud_dims, _tm5_build_conv1!, _tm5_lu!, _tm5_factorize!,
    _tm5_identity_pivots, _tm5_column_sum, _tm5_restore_column_mass!,
    _launch_cmfmc_matrix_derivation!
using ..State: AbstractCubedSphereField,
    LocalHoltslagBovilleKzField, AbstractCSDkgField, PrecomputedCSDkgField, GCHPNonlocalPBLField,
    field_value, panel_field, update_field!
using ..MetDrivers: ConvectionForcing, current_time
using ..Output: cell_locator, locate                   # observation binding (Inversion/)

# Tape storage policies + record types live in src/Tape/
# (loaded before Adjoints in src/AtmosTransport.jl). Re-imported here so
# call sites continue to use the unqualified names.
using ..Tape: AbstractCSTapeStorage,
              DeviceCSTapeStorage, PinnedHostCSTapeStorage, MmapCSTapeStorage,
              CSTapeSlot, PinnedHostCSTapeSlot, MmapCSTapeSlot,
              _tape_storage, _tape_panels,
              _resolve_tape_path, _build_window_storage,
              _allocate_tape_slot, stage_panels!, _stage_panels,
              _after_tape_stage!, _after_tape_read!,
              _sync_pinned_tape_storage!, _sync_mmap_tape_storage!,
              _mmap_prepare_for_panels!,
              _ensure_tape_read_cache!,
              _bytes_per_panel_tuple, finalize_tape!,
              load_mmap_tape, get_record,
              _CSSweepRecord, _CSHaloRecord, _CSMidpointRecord,
              _CSDiffusionRecord, _CSConvectionRecord, _CSTapeOp,
              AbstractCheckpointSchedule, FullCheckpoint, StrideCheckpoint,
              RevolveCheckpoint,
              checkpoint_window_count, checkpoint_window_range

# PPM adjoints exist for the default vertical reconstruction only; the FV3
# vertical profile (`FV3ScalarProfile`) has no adjoint yet.
const CSAdjointLinearScheme = Union{UpwindScheme, SlopesScheme{NoLimiter},
                                    PPMScheme{NoLimiter, SameAsHorizontal}}
const CSAdjointNonlinearScheme = Union{PPMScheme{MonotoneLimiter, SameAsHorizontal}}
# LinRoodPPMScheme is supported via its own
# horizontal tape record (`_CSLinRoodHorizRecord`) and the kernel
# adjoints shipped in `src/Operators/Advection/linrood_adjoint_kernels.jl`.
# The reverse-loop dispatch arm in `_collect_surface_footprints`
# handles the new record type alongside the existing
# `_CSSweepRecord`, `_CSHaloRecord`, `_CSDiffusionRecord`,
# `_CSConvectionRecord`, and `_CSMidpointRecord` cases. ORD=5 only.
const CSAdjointLinRoodScheme = LinRoodPPMScheme{<:Any, UpwindScheme}   # upwind vertical only
const CSAdjointSupportedScheme = Union{CSAdjointLinearScheme,
                                        CSAdjointNonlinearScheme,
                                        CSAdjointLinRoodScheme}

# Footprint-objective types + seeding/eval kernels.
include("ObjectiveSeeding.jl")


# CSFootprintResult + CSTapeByteEstimate.
include("../Footprint/FootprintResult.jl")


# Control / observation / 4D-Var result types.
include("../Inversion/Observations.jl")

# On-disk observation IO (CSObservationRecord /
# CSObservationSet, read/write to the v1 NetCDF schema documented in
# `schemas/cs_observations_v1.toml`). Loaded after Observations.jl so
# the in-memory 4D-Var types and the raw observation-record types
# coexist in the Adjoints namespace.
include("../Inversion/ObservationsIO.jl")

# `bind_to_mesh` bridge from the on-disk
# `CSObservationSet` to a 4D-Var-ready `Vector{CSObservation}`. Loaded
# after Observations.jl + ObservationsIO.jl because it references
# `CSObservation` and `CSObservationRecord` directly; uses
# `CSColumnMeanObjective` defined in ObjectiveSeeding.jl above.
include("../Inversion/ObservationBinding.jl")

# On-disk departures (forward-pass simulated values
# paired with their originating observations). Loaded after
# ObservationBinding.jl because the `build_departure_set` helper
# consumes both `CSObservationSet` and `Vector{CSObservation}`.
include("../Inversion/DeparturesIO.jl")

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
    CSAdjointWorkspace(mesh, prototype)

Scratch storage for CS adjoint sweeps. `lambda_A` is one halo-padded panel
used as the per-panel transpose output. `seam_seeds` caches both neighbors'
output-seed difference before any panel transpose overwrites those values.
"""
struct CSAdjointWorkspace{FT, A <: AbstractArray{FT, 3}}
    lambda_A::A
    seam_seeds::A
end

function CSAdjointWorkspace(mesh::CubedSphereMesh,
                            prototype::AbstractArray{FT, 3}) where {FT <: AbstractFloat}
    N = mesh.Nc + 2 * mesh.Hp
    lambda_A = similar(prototype, FT, N, N, size(prototype, 3))
    seam_seeds = similar(prototype, FT, mesh.Nc, size(prototype, 3), 12)
    return CSAdjointWorkspace{FT, typeof(lambda_A)}(lambda_A, seam_seeds)
end

struct _SingleSurfaceRatePerturbation{FT}
    step::Int
    panel::Int
    i::Int
    j::Int
    rate_delta::FT
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


@kernel function _add_surface_rates_kernel!(panel_rm, @Const(panel_rate),
                                            dt, Hp, Nz)
    i, j = @index(Global, NTuple)
    @inbounds panel_rm[i + Hp, j + Hp, Nz] += panel_rate[i, j] * dt
end

@kernel function _add_surface_perturbation_kernel!(panel_rm, dt_rate,
                                                   i, j, Nz)
    _ = @index(Global)
    @inbounds panel_rm[i, j, Nz] += dt_rate
end

function _add_surface_rates!(panels_rm, rates::NTuple{6}, dt, mesh::CubedSphereMesh)
    Hp = mesh.Hp
    Nz = size(panels_rm[1], 3)
    @inbounds for p in 1:6
        panel_rm = panels_rm[p]
        panel_rate = rates[p]
        size(panel_rate) == (mesh.Nc, mesh.Nc) || throw(DimensionMismatch(
            "surface rate panel $p has shape $(size(panel_rate)); expected $((mesh.Nc, mesh.Nc))"))
        backend = get_backend(panel_rm)
        kernel! = _add_surface_rates_kernel!(backend, (16, 16))
        kernel!(panel_rm, panel_rate, eltype(panel_rm)(dt), Int32(Hp), Int32(Nz);
                ndrange = (mesh.Nc, mesh.Nc))
        synchronize(backend)
    end
    return nothing
end

function _add_surface_perturbation!(panels_rm, perturb::_SingleSurfaceRatePerturbation,
                                    dt, mesh::CubedSphereMesh)
    Hp = mesh.Hp
    Nz = size(panels_rm[1], 3)
    panel_rm = panels_rm[perturb.panel]
    backend = get_backend(panel_rm)
    kernel! = _add_surface_perturbation_kernel!(backend, 1)
    kernel!(panel_rm,
            eltype(panel_rm)(perturb.rate_delta) * eltype(panel_rm)(dt),
            Int32(Hp + perturb.i), Int32(Hp + perturb.j), Int32(Nz);
            ndrange = 1)
    synchronize(backend)
    return nothing
end

# Split-sweep advection adjoint kernels.
include("AdvectionAdjoint.jl")
include("CubedSphereSeams.jl")

# CS halo-exchange adjoint kernels.
include("HaloAdjoint.jl")





# Vertical-diffusion adjoint kernels.
include("DiffusionAdjoint.jl")

# CMFMC + TM5 convection adjoint kernels.
include("ConvectionAdjoint.jl")


function _convection_forcing_at(convection_forcing, step::Int, nsteps::Int)
    convection_forcing_is_vector = convection_forcing isa AbstractVector
    if convection_forcing === nothing
        return nothing
    elseif convection_forcing_is_vector
        length(convection_forcing) == nsteps || throw(ArgumentError(
            "convection_forcing length $(length(convection_forcing)) does not match nsteps $nsteps"))
        return convection_forcing[step]
    else
        return convection_forcing
    end
end

# Tape storage policies + staging API live in
# `src/Tape/TapeStorage.jl` and are imported via `using ..Tape` at
# the top of this module.


# Tape record types (_CSSweepRecord, _CSHaloRecord,
# _CSMidpointRecord, _CSDiffusionRecord, _CSConvectionRecord) and the
# _CSTapeOp union live in `src/Tape/TapeRecords.jl` and are imported
# via `using ..Tape` at the top of this module. The `S <: CSAdjointSupportedScheme`
# type constraint on `_CSSweepRecord.scheme` is relaxed to plain `S`
# there; the constraint is now enforced at the
# `_record_sweep!`/`_adjoint_scheme_sweep!` call sites below.

# LinRoodPPMScheme tape record + forward/reverse
# integration. Defines `_CSLinRoodHorizRecord`,
# `_record_cs_linrood_tape`, `_apply_cs_linrood_horizontal_adjoint!`.
include("LinRoodTape.jl")

# Tape-recorder kernels.
include("../Footprint/TapeRecording.jl")



# Reverse-loop driver + forward-replay helpers.
include("../Footprint/ReverseLoop.jl")


# Strided checkpoint driver. Depends on the linear-scheme
# `_record_cs_mass_tape` + `_walk_window_reverse!` defined above; the
# FootprintAPI below dispatches on the `checkpoint` kwarg to choose between
# the existing `_collect_surface_footprints` path (FullCheckpoint, no
# behaviour change) and `_collect_surface_footprints_stride`.
include("../Footprint/StrideCheckpoint.jl")


# User-facing footprint API.
include("../Footprint/FootprintAPI.jl")

# Surface-flux Jacobian + aggregation.
include("../Inversion/Jacobian.jl")

# 4D-Var cost + gradient evaluation.
include("../Inversion/CostGradient.jl")

# Prototype gradient-descent optimizer shim.
include("../Inversion/Optimizer.jl")






export AbstractCSFootprintObjective
export CSLayerMeanObjective, CSColumnMeanObjective, CSSeedObjective, CSFootprintResult
export CSSurfaceFluxWindow, CSSurfaceFluxJacobianResult
export CSObservation, CSSurfaceFluxControl, CS4DVarResult, CS4DVarSolveResult
export CSObservationRecord, CSObservationSet, read_observations, write_observations
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
export CSIterationLog, CSIterationLogEntry

end # module Adjoints
