# ---------------------------------------------------------------------------
# User-facing surface-emission footprint API.
#
#   * `run_cs_footprint_forward` — forward-only entry returning the
#     scalar value of an objective at final time.
#   * `cs_surface_emission_footprint` — main reverse-mode entry point.
#     Under `FullCheckpoint` it builds the tape via
#     `_record_cs_adjoint_tape`, seeds the adjoint from the objective and
#     walks `_collect_surface_footprints`; `StrideCheckpoint` and
#     `RevolveCheckpoint` dispatch to `_collect_surface_footprints_stride`
#     / `_collect_surface_footprints_revolve`.
#   * `cs_surface_emission_footprint_from_seed` — variant that takes an
#     explicit final-time adjoint seed (`dJ/drm_final`) instead of
#     constructing it from one of the built-in objectives; same schedule
#     dispatch.
# ---------------------------------------------------------------------------

"""
    run_cs_footprint_forward(..., objective; kwargs...) -> scalar

Run the CS PPM transport path forward and return `objective` at the final
time. Optional `emission_rates[t][panel][i, j]` entries are midpoint surface
emission rates in model-storage units per second (dry mixing ratio × carrier
air mass per second, integrated over the cell). Convert physical species-mass
inventories and per-area rates before calling this low-level API.
If `diffusion_op` is supplied, the helper applies
`V(dt/2) -> emissions -> V(dt/2)` at the control midpoint and requires a
panel-native `DiffusionWorkspace` with filled `layer_thickness`. If
`convection_op=CMFMCConvection()` or `TM5Convection()` is supplied, the
helper applies the corresponding CS convection column operator after each
transport step. `flux_scale` multiplies all transport fluxes and is shared
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
                                  cfl_limit = 0.95,
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
`PPMScheme()`, and `LinRoodPPMScheme` (ORD 5 or 7) with upwind vertical
transport. The limited PPM path stores tracer branch states from the
base trajectory; pass `base_emission_rates` when differentiating around
nonzero surface emissions.
Optional `ImplicitVerticalDiffusion` support transposes the Backward-Euler
column solve in kernels on CPU/GPU and uses the same midpoint placement as
surface-flux runtime transport. Optional `CMFMCConvection` support transposes
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
                                       cfl_limit = 0.95,
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
    FT = eltype(panels_rm0[1])
    dt_ft = FT(dt)
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps, panels_cm_steps)
    Nz = size(panels_rm0[1], 3)
    _validate_objective(objective, mesh, Nz)
    _validate_emission_rates(base_emission_rates, nsteps, mesh,
                             "base_emission_rates")
    _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace, nsteps)
    _require_cs_convection_workspace(convection_op, convection_workspace)
    _require_checkpoint_supported(scheme, checkpoint)
    # `_require_tape_path_supported` runs BEFORE `_resolve_tape_path`
    # so an invalid `(scheme, tape_path)` combination leaves no
    # `mkpath`'d directory or empty `records.bin` behind.
    _require_tape_path_supported(scheme, tape_path)
    resolved_tape_path = _resolve_tape_path(tape_storage, tape_path)

    if checkpoint isa StrideCheckpoint
        if scheme isa CSAdjointLinearScheme
            return _collect_surface_footprints_stride(
                panels_m0,
                panels_am_steps, panels_bm_steps, panels_cm_steps,
                mesh, scheme, checkpoint, objective, dt_ft;
                cfl_limit = cfl_limit,
                flux_scale = FT(flux_scale),
                diffusion_op = diffusion_op,
                diffusion_workspace = diffusion_workspace,
                diffusion_meteo = diffusion_meteo,
                convection_op = convection_op,
                convection_forcing = convection_forcing,
                convection_workspace = convection_workspace,
                tape_storage = tape_storage,
                tape_path = resolved_tape_path)
        else
            # CSAdjointNonlinearScheme or CSAdjointLinRoodScheme —
            # both stride drivers take `panels_rm0` and
            # `base_emission_rates` (meaningless to the linear-mass
            # driver) and dispatch on the scheme type at the stride-
            # driver method level. LinRood additionally requires
            # `tape_storage = :device`; the LinRood method validates
            # that up front.
            return _collect_surface_footprints_stride(
                panels_rm0, panels_m0,
                panels_am_steps, panels_bm_steps, panels_cm_steps,
                mesh, scheme, checkpoint, objective, dt_ft;
                cfl_limit = cfl_limit,
                flux_scale = FT(flux_scale),
                base_emission_rates = base_emission_rates,
                diffusion_op = diffusion_op,
                diffusion_workspace = diffusion_workspace,
                diffusion_meteo = diffusion_meteo,
                convection_op = convection_op,
                convection_forcing = convection_forcing,
                convection_workspace = convection_workspace,
                tape_storage = tape_storage,
                tape_path = resolved_tape_path)
        end
    end

    if checkpoint isa RevolveCheckpoint
        if scheme isa CSAdjointLinearScheme
            return _collect_surface_footprints_revolve(
                panels_m0,
                panels_am_steps, panels_bm_steps, panels_cm_steps,
                mesh, scheme, checkpoint, objective, dt_ft;
                cfl_limit = cfl_limit,
                flux_scale = FT(flux_scale),
                diffusion_op = diffusion_op,
                diffusion_workspace = diffusion_workspace,
                diffusion_meteo = diffusion_meteo,
                convection_op = convection_op,
                convection_forcing = convection_forcing,
                convection_workspace = convection_workspace,
                tape_storage = tape_storage,
                tape_path = resolved_tape_path)
        else
            return _collect_surface_footprints_revolve(
                panels_rm0, panels_m0,
                panels_am_steps, panels_bm_steps, panels_cm_steps,
                mesh, scheme, checkpoint, objective, dt_ft;
                cfl_limit = cfl_limit,
                flux_scale = FT(flux_scale),
                base_emission_rates = base_emission_rates,
                diffusion_op = diffusion_op,
                diffusion_workspace = diffusion_workspace,
                diffusion_meteo = diffusion_meteo,
                convection_op = convection_op,
                convection_forcing = convection_forcing,
                convection_workspace = convection_workspace,
                tape_storage = tape_storage,
                tape_path = resolved_tape_path)
        end
    end

    # FullCheckpoint path: construct the tape storage here so we can
    # finalize it deterministically after the reverse walk. With
    # `tape_path !== nothing` this is load-bearing — the user's
    # directory keeps `records.bin` + `manifest.toml` after the call
    # returns rather than relying on GC to flush the manifest. When
    # the caller passes a pre-constructed `AbstractCSTapeStorage`,
    # `_build_window_storage` returns it unchanged and we skip our own
    # `finalize_tape!` so the caller's manual lifecycle (existing
    # mmap-roundtrip test pattern) is preserved.
    storage = _build_window_storage(tape_storage, resolved_tape_path)
    owns_storage = !(tape_storage isa AbstractCSTapeStorage)
    # When the caller supplied `tape_path`, manifest/close failures
    # must propagate — a saved tape without a valid manifest defeats
    # the entire purpose of `tape_path`. For temp-dir usage (no path)
    # keep the existing warn-and-continue behaviour: the directory
    # gets nuked by `cleanup_on_finalize` anyway.
    strict_finalize = resolved_tape_path !== nothing
    try
        ops, final_m = _record_cs_adjoint_tape(panels_rm0, panels_m0,
                                               panels_am_steps, panels_bm_steps,
                                               panels_cm_steps, mesh, scheme;
                                               cfl_limit = cfl_limit,
                                               flux_scale = FT(flux_scale),
                                               dt = dt_ft,
                                               base_emission_rates = base_emission_rates,
                                               diffusion_op = diffusion_op,
                                               diffusion_workspace = diffusion_workspace,
                                               diffusion_meteo = diffusion_meteo,
                                               convection_op = convection_op,
                                               convection_forcing = convection_forcing,
                                               convection_workspace = convection_workspace,
                                               tape_storage = storage)
        lambda_panels = ntuple(p -> begin
            a = similar(final_m[p])
            fill!(a, zero(FT))
            a
        end, 6)
        _seed_objective!(lambda_panels, objective, final_m, mesh)
        return _collect_surface_footprints(lambda_panels, ops, panels_m0, mesh, objective, dt_ft;
                                           diffusion_workspace = diffusion_workspace,
                                           diffusion_meteo = diffusion_meteo,
                                           convection_workspace = convection_workspace)
    finally
        owns_storage && finalize_tape!(storage; quiet = true,
                                       strict = strict_finalize)
    end
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
                                                 cfl_limit = 0.95,
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
    # From-seed stride. The objective-driven stride
    # drivers accept a `final_adjoint_seed` kwarg that
    # bypasses `_seed_objective!` and `_validate_objective` (since
    # `CSSeedObjective` would throw); reuse them by threading
    # `final_adjoint_seed = final_adjoint_rm` through. The objective
    # field on the returned `CSFootprintResult` is `CSSeedObjective()`
    # for metadata parity with the FullCheckpoint from-seed path.
    _require_checkpoint_supported(scheme, checkpoint)
    FT = eltype(panels_m0[1])
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps,
                                      panels_cm_steps)
    _validate_emission_rates(base_emission_rates, nsteps, mesh,
                             "base_emission_rates")
    _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace, nsteps)
    _require_cs_convection_workspace(convection_op, convection_workspace)
    _require_tape_path_supported(scheme, tape_path)
    resolved_tape_path = _resolve_tape_path(tape_storage, tape_path)

    if checkpoint isa StrideCheckpoint
        if scheme isa CSAdjointLinearScheme
            return _collect_surface_footprints_stride(
                panels_m0,
                panels_am_steps, panels_bm_steps, panels_cm_steps,
                mesh, scheme, checkpoint, CSSeedObjective(), FT(dt);
                cfl_limit = cfl_limit,
                flux_scale = FT(flux_scale),
                diffusion_op = diffusion_op,
                diffusion_workspace = diffusion_workspace,
                diffusion_meteo = diffusion_meteo,
                convection_op = convection_op,
                convection_forcing = convection_forcing,
                convection_workspace = convection_workspace,
                tape_storage = tape_storage,
                tape_path = resolved_tape_path,
                final_adjoint_seed = final_adjoint_rm)
        else
            # Nonlinear PPM or LinRood — both accept `panels_rm0` and
            # `base_emission_rates`. The nonlinear/LinRood propagation
            # pass needs an initial rm state; the from-seed flow uses
            # `base_panels_rm0` for the base trajectory, falling back
            # to zero when unspecified (matching the FullCheckpoint
            # from-seed contract below).
            tape_rm0 = base_panels_rm0 === nothing ?
                _zero_panel_tuple_like(panels_m0) :
                base_panels_rm0
            return _collect_surface_footprints_stride(
                tape_rm0, panels_m0,
                panels_am_steps, panels_bm_steps, panels_cm_steps,
                mesh, scheme, checkpoint, CSSeedObjective(), FT(dt);
                cfl_limit = cfl_limit,
                flux_scale = FT(flux_scale),
                base_emission_rates = base_emission_rates,
                diffusion_op = diffusion_op,
                diffusion_workspace = diffusion_workspace,
                diffusion_meteo = diffusion_meteo,
                convection_op = convection_op,
                convection_forcing = convection_forcing,
                convection_workspace = convection_workspace,
                tape_storage = tape_storage,
                tape_path = resolved_tape_path,
                final_adjoint_seed = final_adjoint_rm)
        end
    end

    if checkpoint isa RevolveCheckpoint
        if scheme isa CSAdjointLinearScheme
            return _collect_surface_footprints_revolve(
                panels_m0,
                panels_am_steps, panels_bm_steps, panels_cm_steps,
                mesh, scheme, checkpoint, CSSeedObjective(), FT(dt);
                cfl_limit = cfl_limit,
                flux_scale = FT(flux_scale),
                diffusion_op = diffusion_op,
                diffusion_workspace = diffusion_workspace,
                diffusion_meteo = diffusion_meteo,
                convection_op = convection_op,
                convection_forcing = convection_forcing,
                convection_workspace = convection_workspace,
                tape_storage = tape_storage,
                tape_path = resolved_tape_path,
                final_adjoint_seed = final_adjoint_rm)
        else
            tape_rm0 = base_panels_rm0 === nothing ?
                _zero_panel_tuple_like(panels_m0) :
                base_panels_rm0
            return _collect_surface_footprints_revolve(
                tape_rm0, panels_m0,
                panels_am_steps, panels_bm_steps, panels_cm_steps,
                mesh, scheme, checkpoint, CSSeedObjective(), FT(dt);
                cfl_limit = cfl_limit,
                flux_scale = FT(flux_scale),
                base_emission_rates = base_emission_rates,
                diffusion_op = diffusion_op,
                diffusion_workspace = diffusion_workspace,
                diffusion_meteo = diffusion_meteo,
                convection_op = convection_op,
                convection_forcing = convection_forcing,
                convection_workspace = convection_workspace,
                tape_storage = tape_storage,
                tape_path = resolved_tape_path,
                final_adjoint_seed = final_adjoint_rm)
        end
    end

    tape_rm0 = base_panels_rm0 === nothing ?
        _zero_panel_tuple_like(panels_m0) :
        base_panels_rm0
    storage = _build_window_storage(tape_storage, resolved_tape_path)
    owns_storage = !(tape_storage isa AbstractCSTapeStorage)
    strict_finalize = resolved_tape_path !== nothing
    try
        ops, _ = _record_cs_adjoint_tape(tape_rm0, panels_m0,
                                         panels_am_steps, panels_bm_steps,
                                         panels_cm_steps, mesh, scheme;
                                         cfl_limit = cfl_limit,
                                         flux_scale = FT(flux_scale),
                                         dt = FT(dt),
                                         base_emission_rates = base_emission_rates,
                                         diffusion_op = diffusion_op,
                                         diffusion_workspace = diffusion_workspace,
                                         diffusion_meteo = diffusion_meteo,
                                         convection_op = convection_op,
                                         convection_forcing = convection_forcing,
                                         convection_workspace = convection_workspace,
                                         tape_storage = storage)
        lambda_panels = _copy_panel_tuple(final_adjoint_rm)
        return _collect_surface_footprints(lambda_panels, ops, panels_m0, mesh,
                                           CSSeedObjective(), FT(dt);
                                           diffusion_workspace = diffusion_workspace,
                                           diffusion_meteo = diffusion_meteo,
                                           convection_workspace = convection_workspace)
    finally
        owns_storage && finalize_tape!(storage; quiet = true,
                                       strict = strict_finalize)
    end
end
