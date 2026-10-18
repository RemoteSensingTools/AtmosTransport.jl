# ---------------------------------------------------------------------------
# User-facing surface-emission footprint API.
#
#   * `run_cs_footprint_forward` — forward-only entry returning the
#     scalar value of an objective at final time.
#   * `cs_surface_emission_footprint` — main reverse-mode entry point for a
#     final-time objective.
#   * `cs_surface_emission_footprint_from_seed` — variant that takes an
#     explicit final-time adjoint seed (`dJ/drm_final`) instead of
#     constructing it from one of the built-in objectives.
#   * `_cs_surface_footprints` — the one checkpoint dispatch behind both
#     entry points, `cs_surface_flux_jacobian` (several objectives on one
#     tape) and `cs_surface_flux_4dvar` (a zero start plus the observation
#     step forcing). Under `FullCheckpoint` it records the tape via
#     `_record_cs_adjoint_tape` once and walks `_collect_surface_footprints`
#     per adjoint start; `StrideCheckpoint` and `RevolveCheckpoint` go to
#     `_collect_surface_footprints_stride` / `_collect_surface_footprints_revolve`.
# ---------------------------------------------------------------------------

"""
    run_cs_footprint_forward(..., objective; kwargs...) -> scalar

Run the CS PPM transport path forward and return `objective` at the final
time. Optional `emission_rates[t][panel][i, j]` entries are midpoint surface
emission rates in model-storage units per second (dry mixing ratio × carrier
air mass per second, integrated over the cell). Convert physical species-mass
inventories and per-area rates before calling this low-level API.
Each step is the production cubed-sphere step of a run with surface
fluxes: the step-start halo refresh, the transport palindrome, and the
optional convection column operator (`CMFMCConvection()`, `TM5Convection()`
or `CMFMCMatrixConvection`) after it. If `diffusion_op` is supplied, the
palindrome center follows its `surface_flux_coupling`:
`V(dt/2) -> emissions -> V(dt/2)` (`SplitSurfaceFluxCoupling`) or
`emissions -> V(dt)` (`DiffusiveSurfaceFluxBoundary`); diffusion requires a
panel-native `DiffusionWorkspace` with filled `layer_thickness`. `flux_scale` multiplies all transport fluxes and is shared
with the tape/reverse APIs so forward finite differences exercise the same
model trajectory.
"""
function run_cs_footprint_forward(panels_rm0, panels_m0,
                                  panels_am_steps,
                                  panels_bm_steps,
                                  panels_cm_steps,
                                  mesh::CubedSphereMesh,
                                  objective::AbstractCSFootprintObjective;
                                  scheme = PPMScheme(NoLimiter()),
                                  dt = one(eltype(panels_rm0[1])),
                                  flux_scale = one(eltype(panels_rm0[1])),
                                  cfl_limit = DEFAULT_CFL_LIMIT,
                                  emission_rates = nothing,
                                  diffusion_op = NoDiffusion(),
                                  diffusion_workspace = nothing,
                                  diffusion_meteo = nothing,
                                  convection_op = NoConvection(),
                                  convection_forcing = nothing,
                                  convection_workspace = nothing)
    FT = eltype(panels_rm0[1])
    return _run_cs_footprint_forward(panels_rm0, panels_m0,
                                     panels_am_steps, panels_bm_steps, panels_cm_steps,
                                     mesh, objective;
                                     scheme = scheme,
                                     dt = FT(dt),
                                     flux_scale = FT(flux_scale),
                                     cfl_limit = cfl_limit,
                                     emission_rates = emission_rates,
                                     diffusion_op = diffusion_op,
                                     diffusion_workspace = diffusion_workspace,
                                     diffusion_meteo = diffusion_meteo,
                                     convection_op = convection_op,
                                     convection_forcing = convection_forcing,
                                     convection_workspace = convection_workspace)
end

"""
    cs_surface_emission_footprint(..., objective; kwargs...) -> CSFootprintResult

Generate reverse-mode footprints for a scalar final-time objective with
respect to surface-emission rates at each prior model step.

This is a kernelized prototype VJP generator for tests and diagnostics.
Supported CS schemes are the split-sweep `UpwindScheme()`,
`SlopesScheme(NoLimiter())`, `PPMScheme(NoLimiter())` and monotone
`PPMScheme()`, and `LinRoodPPMScheme` (ORD 5 or 7) with upwind or FV3
`kord = 8` vertical transport. The limited PPM paths, including the FV3
vertical sweeps, store tracer branch states from the base trajectory; pass
`base_emission_rates` when differentiating around nonzero surface emissions.
Optional `ImplicitVerticalDiffusion` support transposes the Backward-Euler
column solve in kernels on CPU/GPU and uses the runtime's surface-flux
coupling at the palindrome center (`V(dt/2) -> emissions -> V(dt/2)`, or
`emissions -> V(dt)` with `DiffusiveSurfaceFluxBoundary`). Optional `CMFMCConvection` support transposes
the well-mixed sub-cloud, updraft, and tendency passes; optional
`TM5Convection` and `CMFMCMatrixConvection` support replays the same
column matrix and applies the transposed LU solve after each reverse
transport step.

`tape_storage` selects the per-tape-slot storage policy (`:device`,
`:pinned_host`, or `:mmap`). When `tape_storage = :mmap`, the optional
`tape_path` kwarg points the on-disk tape at a user-owned directory
instead of a temporary one — required for tapes that must persist past
the call (manual inspection, partial-run debug, multi-session
campaigns). `tape_path` is created on demand and preserved past
`finalize_tape!`; it is rejected for non-`:mmap` storage. Under
`StrideCheckpoint` each window writes a `window_NNNNN/` subdirectory;
under `RevolveCheckpoint` each base-case step writes a `step_NNNNN/`
subdirectory. The LinRood scheme is pinned to `:device` and rejects
`tape_path` unconditionally.
"""
function cs_surface_emission_footprint(panels_rm0, panels_m0,
                                       panels_am_steps,
                                       panels_bm_steps,
                                       panels_cm_steps,
                                       mesh::CubedSphereMesh,
                                       objective::AbstractCSFootprintObjective;
                                       scheme::CSAdjointSupportedScheme = PPMScheme(NoLimiter()),
                                       dt = one(eltype(panels_rm0[1])),
                                       epsilon = nothing,
                                       flux_scale = one(eltype(panels_rm0[1])),
                                       cfl_limit = DEFAULT_CFL_LIMIT,
                                       base_emission_rates = nothing,
                                       diffusion_op = NoDiffusion(),
                                       diffusion_workspace = nothing,
                                       diffusion_meteo = nothing,
                                       convection_op = NoConvection(),
                                       convection_forcing = nothing,
                                       convection_workspace = nothing,
                                       tape_storage = :device,
                                       tape_path::Union{Nothing, AbstractString} = nothing,
                                       checkpoint::AbstractCheckpointSchedule = FullCheckpoint())
    return only(_cs_objective_footprints(
        panels_rm0, panels_m0, panels_am_steps, panels_bm_steps, panels_cm_steps,
        mesh, (objective,);
        scheme, dt, epsilon, flux_scale, cfl_limit, base_emission_rates,
        diffusion_op, diffusion_workspace, diffusion_meteo,
        convection_op, convection_forcing, convection_workspace,
        tape_storage, tape_path, checkpoint))
end

# The footprints of several final-time objectives on one trajectory, with the
# keywords (and defaults) of `cs_surface_emission_footprint`. Under
# `FullCheckpoint` the tape is recorded once and walked once per objective;
# `StrideCheckpoint` and `RevolveCheckpoint` run their schedule per objective.
# `cs_surface_flux_jacobian` calls it with all of its objectives.
function _cs_objective_footprints(panels_rm0, panels_m0,
                                  panels_am_steps,
                                  panels_bm_steps,
                                  panels_cm_steps,
                                  mesh::CubedSphereMesh,
                                  objectives;
                                  scheme::CSAdjointSupportedScheme = PPMScheme(NoLimiter()),
                                  dt = one(eltype(panels_rm0[1])),
                                  epsilon = nothing,
                                  flux_scale = one(eltype(panels_rm0[1])),
                                  cfl_limit = DEFAULT_CFL_LIMIT,
                                  base_emission_rates = nothing,
                                  diffusion_op = NoDiffusion(),
                                  diffusion_workspace = nothing,
                                  diffusion_meteo = nothing,
                                  convection_op = NoConvection(),
                                  convection_forcing = nothing,
                                  convection_workspace = nothing,
                                  tape_storage = :device,
                                  tape_path::Union{Nothing, AbstractString} = nothing,
                                  checkpoint::AbstractCheckpointSchedule = FullCheckpoint())
    _ = epsilon  # compatibility keyword of the finite-difference prototype
    Nz = size(panels_rm0[1], 3)
    starts = [_ObjectiveStart(objective) for objective in objectives]
    foreach(start -> _validate_start(start, mesh, Nz), starts)
    return _cs_surface_footprints(panels_rm0, panels_m0,
                                  panels_am_steps, panels_bm_steps, panels_cm_steps,
                                  mesh, starts;
                                  scheme, checkpoint, dt, flux_scale, cfl_limit,
                                  base_emission_rates,
                                  diffusion_op, diffusion_workspace, diffusion_meteo,
                                  convection_op, convection_forcing, convection_workspace,
                                  tape_storage, tape_path)
end

"""
    cs_surface_emission_footprint_from_seed(final_adjoint_rm, panels_m0,
                                            panels_am_steps, panels_bm_steps,
                                            panels_cm_steps, mesh; kwargs...)

General surface-emission footprint entry point. `final_adjoint_rm` is an
`NTuple{6}` of halo-padded adjoint tracer-mass arrays containing
`dJ/drm_final` for any scalar objective or observation operator. The reverse
pass and surface-gradient accumulation use the same CPU/GPU kernels as
`cs_surface_emission_footprint`.
"""
function cs_surface_emission_footprint_from_seed(final_adjoint_rm::NTuple{6},
                                                 panels_m0,
                                                 panels_am_steps,
                                                 panels_bm_steps,
                                                 panels_cm_steps,
                                                 mesh::CubedSphereMesh;
                                                 scheme::CSAdjointSupportedScheme = PPMScheme(NoLimiter()),
                                                 dt = one(eltype(panels_m0[1])),
                                                 flux_scale = one(eltype(panels_m0[1])),
                                                 cfl_limit = DEFAULT_CFL_LIMIT,
                                                 base_panels_rm0 = nothing,
                                                 base_emission_rates = nothing,
                                                 diffusion_op = NoDiffusion(),
                                                 diffusion_workspace = nothing,
                                                 diffusion_meteo = nothing,
                                                 convection_op = NoConvection(),
                                                 convection_forcing = nothing,
                                                 convection_workspace = nothing,
                                                 tape_storage = :device,
                                                 tape_path::Union{Nothing, AbstractString} = nothing,
                                                 checkpoint::AbstractCheckpointSchedule = FullCheckpoint())
    # The tape is recorded around `base_panels_rm0`, zero when unspecified; it
    # matters only for the limited schemes, whose reverse replays branches of
    # the base trajectory.
    tape_rm0 = base_panels_rm0 === nothing ?
        _zero_panel_tuple_like(panels_m0) :
        base_panels_rm0
    return only(_cs_surface_footprints(
        tape_rm0, panels_m0, panels_am_steps, panels_bm_steps, panels_cm_steps,
        mesh, [_SeedStart(final_adjoint_rm)];
        scheme, checkpoint, dt, flux_scale, cfl_limit, base_emission_rates,
        diffusion_op, diffusion_workspace, diffusion_meteo,
        convection_op, convection_forcing, convection_workspace,
        tape_storage, tape_path))
end

# ---------------------------------------------------------------------------
# The one checkpoint dispatch behind every reverse pass
# ---------------------------------------------------------------------------

"""
    _cs_surface_footprints(tape_rm0, panels_m0, panels_am_steps, panels_bm_steps,
                           panels_cm_steps, mesh, starts; scheme, checkpoint, dt,
                           flux_scale, cfl_limit, base_emission_rates,
                           diffusion_op, diffusion_workspace, diffusion_meteo,
                           convection_op, convection_forcing,
                           convection_workspace, tape_storage, tape_path,
                           forcing = NoStepForcing())
        -> Vector{CSFootprintResult}

One `CSFootprintResult` per adjoint start in `starts` (`_ObjectiveStart`,
`_SeedStart`, `_ZeroStart`), on the tape recorded around `tape_rm0` and
`base_emission_rates`. Validates the inputs once, then dispatches on
`checkpoint` and the scheme family to the Full, Stride or Revolve driver.
`forcing` is added at every end-of-step marker of the reverse walk; the
single-sweep 4D-Var gradient passes its observation forcing with
`[_ZeroStart()]`. The public footprint entry points, `cs_surface_flux_jacobian`
and `cs_surface_flux_4dvar` all come through here.
"""
function _cs_surface_footprints(tape_rm0, panels_m0,
                                panels_am_steps,
                                panels_bm_steps,
                                panels_cm_steps,
                                mesh::CubedSphereMesh,
                                starts::AbstractVector;
                                scheme::CSAdjointSupportedScheme,
                                checkpoint::AbstractCheckpointSchedule,
                                dt,
                                flux_scale,
                                cfl_limit,
                                base_emission_rates,
                                diffusion_op,
                                diffusion_workspace,
                                diffusion_meteo,
                                convection_op,
                                convection_forcing,
                                convection_workspace,
                                tape_storage,
                                tape_path::Union{Nothing, AbstractString},
                                forcing::AbstractStepForcing = NoStepForcing())
    FT = eltype(panels_m0[1])
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps, panels_cm_steps)
    _validate_emission_rates(base_emission_rates, nsteps, mesh,
                             "base_emission_rates")
    _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace, nsteps)
    _require_cs_convection_workspace(convection_op, convection_workspace)
    _validate_step_forcing(forcing, nsteps, size(panels_m0[1], 3))
    _require_checkpoint_supported(scheme, checkpoint)
    # `_require_tape_path_supported` runs BEFORE `_resolve_tape_path`
    # so an invalid `(scheme, tape_path)` combination leaves no
    # `mkpath`'d directory or empty `records.bin` behind.
    _require_tape_path_supported(scheme, tape_path)
    resolved_tape_path = _resolve_tape_path(tape_storage, tape_path)
    return _collect_footprints(checkpoint, scheme, tape_rm0, panels_m0,
                               panels_am_steps, panels_bm_steps, panels_cm_steps,
                               mesh, starts, FT(dt);
                               cfl_limit, flux_scale = FT(flux_scale),
                               base_emission_rates,
                               diffusion_op, diffusion_workspace, diffusion_meteo,
                               convection_op, convection_forcing, convection_workspace,
                               tape_storage, tape_path = resolved_tape_path, forcing)
end

# FullCheckpoint: one tape for all starts. The storage is constructed here so
# it is finalized deterministically after the last reverse walk. With
# `tape_path !== nothing` this is load-bearing — the user's directory keeps
# `records.bin` + `manifest.toml` after the call returns rather than relying
# on GC to flush the manifest. A pre-constructed `AbstractCSTapeStorage` is
# used as is (`_build_window_storage` returns it unchanged) and not finalized
# here, so the caller's manual lifecycle is preserved. When the caller
# supplied `tape_path`, manifest/close failures propagate (a saved tape
# without a valid manifest defeats the purpose of `tape_path`); temp-dir
# tapes keep warn-and-continue, as `cleanup_on_finalize` deletes them anyway.
function _collect_footprints(::FullCheckpoint, scheme, tape_rm0, panels_m0,
                             panels_am_steps, panels_bm_steps, panels_cm_steps,
                             mesh::CubedSphereMesh, starts, dt;
                             cfl_limit, flux_scale, base_emission_rates,
                             diffusion_op, diffusion_workspace, diffusion_meteo,
                             convection_op, convection_forcing, convection_workspace,
                             tape_storage, tape_path, forcing)
    storage = _build_window_storage(tape_storage, tape_path)
    owns_storage = !(tape_storage isa AbstractCSTapeStorage)
    strict_finalize = tape_path !== nothing
    try
        ops, final_m = _record_cs_adjoint_tape(tape_rm0, panels_m0,
                                               panels_am_steps, panels_bm_steps,
                                               panels_cm_steps, mesh, scheme;
                                               cfl_limit, flux_scale, dt,
                                               base_emission_rates,
                                               diffusion_op, diffusion_workspace,
                                               diffusion_meteo,
                                               convection_op, convection_forcing,
                                               convection_workspace,
                                               tape_storage = storage)
        return [_collect_surface_footprints(_initial_adjoint_state(start, final_m, mesh),
                                            ops, panels_m0, mesh,
                                            _result_objective(start), dt;
                                            diffusion_workspace, diffusion_meteo,
                                            convection_workspace, forcing)
                for start in starts]
    finally
        owns_storage && finalize_tape!(storage; quiet = true,
                                       strict = strict_finalize)
    end
end

# StrideCheckpoint and RevolveCheckpoint run their schedule once per start.
# The linear mass-tape drivers take neither the tracer state nor the base
# emissions (their reverse does not depend on the tracer); the monotone-PPM
# and Lin-Rood drivers take both.
function _collect_footprints(checkpoint::StrideCheckpoint, scheme::CSAdjointLinearScheme,
                             _tape_rm0, panels_m0,
                             panels_am_steps, panels_bm_steps, panels_cm_steps,
                             mesh::CubedSphereMesh, starts, dt;
                             base_emission_rates, kwargs...)
    return [_collect_surface_footprints_stride(panels_m0, panels_am_steps,
                                               panels_bm_steps, panels_cm_steps,
                                               mesh, scheme, checkpoint, start, dt;
                                               kwargs...)
            for start in starts]
end

function _collect_footprints(checkpoint::StrideCheckpoint, scheme, tape_rm0, panels_m0,
                             panels_am_steps, panels_bm_steps, panels_cm_steps,
                             mesh::CubedSphereMesh, starts, dt; kwargs...)
    return [_collect_surface_footprints_stride(tape_rm0, panels_m0, panels_am_steps,
                                               panels_bm_steps, panels_cm_steps,
                                               mesh, scheme, checkpoint, start, dt;
                                               kwargs...)
            for start in starts]
end

function _collect_footprints(checkpoint::RevolveCheckpoint, scheme::CSAdjointLinearScheme,
                             _tape_rm0, panels_m0,
                             panels_am_steps, panels_bm_steps, panels_cm_steps,
                             mesh::CubedSphereMesh, starts, dt;
                             base_emission_rates, kwargs...)
    return [_collect_surface_footprints_revolve(panels_m0, panels_am_steps,
                                                panels_bm_steps, panels_cm_steps,
                                                mesh, scheme, checkpoint, start, dt;
                                                kwargs...)
            for start in starts]
end

function _collect_footprints(checkpoint::RevolveCheckpoint, scheme, tape_rm0, panels_m0,
                             panels_am_steps, panels_bm_steps, panels_cm_steps,
                             mesh::CubedSphereMesh, starts, dt; kwargs...)
    return [_collect_surface_footprints_revolve(tape_rm0, panels_m0, panels_am_steps,
                                                panels_bm_steps, panels_cm_steps,
                                                mesh, scheme, checkpoint, start, dt;
                                                kwargs...)
            for start in starts]
end
