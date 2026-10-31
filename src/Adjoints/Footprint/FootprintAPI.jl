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
#     entry points and `cs_surface_flux_jacobian` (several objectives on one
#     recording). It builds the replay of the production step plan
#     (`_cs_replay_for_call`, which runs the setup guard) and hands it to the
#     Full, Stride or Revolve driver (`_collect_footprints`). The 4D-Var
#     gradient records during its observation forward run instead and calls
#     the drivers' phases itself (`StrideCheckpoint.jl`).
# ---------------------------------------------------------------------------

"""
    run_cs_footprint_forward(..., objective; kwargs...) -> scalar

Run the cubed-sphere production step plan forward and return `objective` at
the final time. Optional `emission_rates[t][panel][i, j]` entries are surface
emission rates in model-storage units per second (dry mixing ratio × carrier
air mass per second, integrated over the cell) that the surface-flux operator
applies at the palindrome center of step `t`. Convert physical species-mass
inventories and per-area rates before calling this low-level API.

Each step is the production step of `schedule` (a [`CSStepSchedule`](@ref);
default: one step of `dt` per met window and the full operator suite every
step, as `TransportModel.step!`): the transport block — step-start halo
refresh, palindrome and its center — and, at the plan's cadence, the
convection column operator (`CMFMCConvection()`, `TM5Convection()` or
`CMFMCMatrixConvection`) and the air-mass resets. With `surface_flux = true`
(the default) the plan carries a surface-flux operator (zero rates when
`emission_rates = nothing`) and the palindrome center follows the diffusion
operator's `surface_flux_coupling`: `V(dt/2) -> emissions -> V(dt/2)`
(`SplitSurfaceFluxCoupling`) or `emissions -> V(dt)`
(`DiffusiveSurfaceFluxBoundary`); `surface_flux = false` replays a
configuration without surface sources, whose center is one solve `V(dt)`.
Diffusion requires a panel-native `DiffusionWorkspace` with filled
`layer_thickness`. `flux_scale` multiplies all transport fluxes and is shared
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
                                  dt = nothing,
                                  schedule = nothing,
                                  surface_flux::Bool = true,
                                  flux_scale = one(eltype(panels_rm0[1])),
                                  cfl_limit = DEFAULT_CFL_LIMIT,
                                  emission_rates = nothing,
                                  diffusion_op = NoDiffusion(),
                                  diffusion_workspace = nothing,
                                  diffusion_meteo = nothing,
                                  convection_op = NoConvection(),
                                  convection_forcing = nothing,
                                  convection_workspace = nothing)
    return _run_cs_footprint_forward(panels_rm0, panels_m0,
                                     panels_am_steps, panels_bm_steps, panels_cm_steps,
                                     mesh, objective;
                                     scheme, dt, schedule, surface_flux, flux_scale,
                                     cfl_limit, emission_rates,
                                     diffusion_op, diffusion_workspace, diffusion_meteo,
                                     convection_op, convection_forcing, convection_workspace)
end

"""
    cs_surface_emission_footprint(..., objective; kwargs...) -> CSFootprintResult

Generate reverse-mode footprints for a scalar final-time objective with
respect to surface-emission rates at each prior model step.

The tape is recorded from the production step plan of `schedule` (a
[`CSStepSchedule`](@ref); default: one step of `dt` per met window, the full
operator suite every step), with a surface-flux operator in the plan (zero
rates unless `base_emission_rates` is given), and walked backwards with each
operator's adjoint. `require_adjoint_support` rejects a plan with an operator
the tape cannot transpose before anything is recorded.

Supported CS schemes are the split-sweep `UpwindScheme()`,
`SlopesScheme(NoLimiter())`, `PPMScheme(NoLimiter())` and monotone
`PPMScheme()`, and `LinRoodPPMScheme` (ORD 5 or 7) with upwind or FV3
`kord = 8` vertical transport. The limited PPM paths, including the FV3
vertical sweeps, store tracer branch states from the base trajectory; pass
`base_emission_rates` when differentiating around nonzero surface emissions.
Optional `ImplicitVerticalDiffusion` support transposes the Backward-Euler
column solve in kernels on CPU/GPU and uses the runtime's surface-flux
coupling at the palindrome center (`V(dt/2) -> emissions -> V(dt/2)`, or
`emissions -> V(dt)` with `DiffusiveSurfaceFluxBoundary`). Optional
`CMFMCConvection` support transposes the well-mixed sub-cloud, updraft, and
tendency passes; optional `TM5Convection` and `CMFMCMatrixConvection` support
replays the same column matrix and applies the transposed LU solve, at the
plan's physics cadence (every step, or once per met window over the window).

The replay uses the caller's convection workspace with the persistent state
of a fresh run (its ledger carry and CFL cache are reset when a replay
starts) and the diffusion workspaces' `factors` as scratch; do not pass a live
production model's workspaces. `cs_adjoint_inputs(sim)` captures every input
of a `DrivenSimulation` configuration, with fresh workspaces.

`tape_storage` selects where the tape's staged states live: `:device` (the
default), `:pinned_host`, a pre-constructed `DeviceCSTapeStorage()` or
`PinnedHostCSTapeStorage()` (the caller's: neither sized nor released by the
call), or `MmapCSTapeStorage(dir)`, a scratch file in `dir` (required) that
is sized and reserved before the tape is recorded and unmapped and closed
when the call returns. `checkpoint` (`FullCheckpoint()`,
`StrideCheckpoint(K)` or `RevolveCheckpoint()`) trades recomputation for tape
size; every storage works with every schedule. The Lin–Rood tape is device-only.
"""
function cs_surface_emission_footprint(panels_rm0, panels_m0,
                                       panels_am_steps,
                                       panels_bm_steps,
                                       panels_cm_steps,
                                       mesh::CubedSphereMesh,
                                       objective::AbstractCSFootprintObjective;
                                       scheme::CSAdjointSupportedScheme = PPMScheme(NoLimiter()),
                                       dt = nothing,
                                       schedule = nothing,
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
                                       checkpoint::AbstractCheckpointSchedule = FullCheckpoint())
    return only(_cs_objective_footprints(
        panels_rm0, panels_m0, panels_am_steps, panels_bm_steps, panels_cm_steps,
        mesh, (objective,);
        scheme, dt, schedule, flux_scale, cfl_limit, base_emission_rates,
        diffusion_op, diffusion_workspace, diffusion_meteo,
        convection_op, convection_forcing, convection_workspace,
        tape_storage, checkpoint))
end

# The footprints of several final-time objectives on one trajectory, with the
# keywords (and defaults) of `cs_surface_emission_footprint`. Every checkpoint
# schedule records (and recomputes) once for all objectives.
# `cs_surface_flux_jacobian` calls it with all of its objectives.
function _cs_objective_footprints(panels_rm0, panels_m0,
                                  panels_am_steps,
                                  panels_bm_steps,
                                  panels_cm_steps,
                                  mesh::CubedSphereMesh,
                                  objectives;
                                  scheme::CSAdjointSupportedScheme = PPMScheme(NoLimiter()),
                                  dt = nothing,
                                  schedule = nothing,
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
                                  checkpoint::AbstractCheckpointSchedule = FullCheckpoint())
    Nz = size(panels_rm0[1], 3)
    starts = [_ObjectiveStart(objective) for objective in objectives]
    foreach(start -> _validate_start(start, mesh, Nz), starts)
    return _cs_surface_footprints(panels_rm0, panels_m0,
                                  panels_am_steps, panels_bm_steps, panels_cm_steps,
                                  mesh, starts;
                                  scheme, checkpoint, dt, schedule, flux_scale, cfl_limit,
                                  base_emission_rates,
                                  diffusion_op, diffusion_workspace, diffusion_meteo,
                                  convection_op, convection_forcing, convection_workspace,
                                  tape_storage)
end

"""
    cs_surface_emission_footprint_from_seed(final_adjoint_rm, panels_m0,
                                            panels_am_steps, panels_bm_steps,
                                            panels_cm_steps, mesh; kwargs...)

General reverse-pass entry point. `final_adjoint_rm` is an `NTuple{6}` of
halo-padded adjoint tracer-mass arrays containing `dJ/drm_final` for any
scalar objective or observation operator; only its interior is read (the halos
of the final state are not prognostic, so halo entries are ignored). The
reverse pass and surface-gradient accumulation use the same CPU/GPU kernels as
`cs_surface_emission_footprint`.

With `surface_flux = true` (the default) the step plan carries a surface-flux
operator and the result holds the emission footprints. `surface_flux = false`
replays a configuration without surface sources (palindrome center `V(dt)`):
the result's `footprints` are empty and `initial_adjoint` is the
initial-condition sensitivity `dJ/drm_0`, which every result also carries.
"""
function cs_surface_emission_footprint_from_seed(final_adjoint_rm::NTuple{6},
                                                 panels_m0,
                                                 panels_am_steps,
                                                 panels_bm_steps,
                                                 panels_cm_steps,
                                                 mesh::CubedSphereMesh;
                                                 scheme::CSAdjointSupportedScheme = PPMScheme(NoLimiter()),
                                                 dt = nothing,
                                                 schedule = nothing,
                                                 surface_flux::Bool = true,
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
                                                 checkpoint::AbstractCheckpointSchedule = FullCheckpoint())
    # The tape is recorded around `base_panels_rm0`, zero when unspecified; it
    # matters only for the limited schemes, whose reverse replays branches of
    # the base trajectory.
    tape_rm0 = base_panels_rm0 === nothing ?
        zeros_like(ntuple(p -> panels_m0[p], 6)) :
        base_panels_rm0
    return only(_cs_surface_footprints(
        tape_rm0, panels_m0, panels_am_steps, panels_bm_steps, panels_cm_steps,
        mesh, [_SeedStart(final_adjoint_rm)];
        scheme, checkpoint, dt, schedule, surface_flux, flux_scale, cfl_limit,
        base_emission_rates,
        diffusion_op, diffusion_workspace, diffusion_meteo,
        convection_op, convection_forcing, convection_workspace,
        tape_storage))
end

# ---------------------------------------------------------------------------
# The one checkpoint dispatch behind every reverse pass
# ---------------------------------------------------------------------------

"""
    _cs_surface_footprints(tape_rm0, panels_m0, panels_am_steps, panels_bm_steps,
                           panels_cm_steps, mesh, starts; scheme, checkpoint, dt,
                           schedule, surface_flux, flux_scale, cfl_limit,
                           base_emission_rates, diffusion_op, diffusion_workspace,
                           diffusion_meteo, convection_op, convection_forcing,
                           convection_workspace, tape_storage)
        -> Vector{CSFootprintResult}

One `CSFootprintResult` per adjoint start in `starts` (`_ObjectiveStart`,
`_SeedStart`), on the replay of every step of the step plan recorded around
`tape_rm0` and `base_emission_rates`. Validates the inputs and the plan's
adjoint support once, then dispatches on `checkpoint` to the Full, Stride or
Revolve driver. The public footprint entry points and
`cs_surface_flux_jacobian` come through here.
"""
function _cs_surface_footprints(tape_rm0, panels_m0,
                                panels_am_steps,
                                panels_bm_steps,
                                panels_cm_steps,
                                mesh::CubedSphereMesh,
                                starts::AbstractVector;
                                scheme::CSAdjointSupportedScheme,
                                checkpoint::AbstractCheckpointSchedule,
                                dt = nothing,
                                schedule = nothing,
                                surface_flux::Bool = true,
                                flux_scale,
                                cfl_limit,
                                base_emission_rates,
                                diffusion_op,
                                diffusion_workspace,
                                diffusion_meteo,
                                convection_op,
                                convection_forcing,
                                convection_workspace,
                                tape_storage)
    _require_tape_storage_supported(scheme, tape_storage)
    replay, step_dt = _cs_replay_for_call(tape_rm0, panels_m0, panels_am_steps, panels_bm_steps,
                                          panels_cm_steps, mesh; scheme, dt, schedule,
                                          surface_flux, base_emission_rates, flux_scale,
                                          cfl_limit, diffusion_op, diffusion_workspace,
                                          diffusion_meteo, convection_op, convection_forcing,
                                          convection_workspace)
    return _collect_footprints(checkpoint, replay, starts, step_dt; tape_storage)
end
