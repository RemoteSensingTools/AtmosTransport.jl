# ---------------------------------------------------------------------------
# Reverse-loop driver + paired forward replay helpers.
#
# `_collect_surface_footprints` allocates the per-step footprints and
# calls `_walk_window_reverse!`, which walks the recorded tape in reverse,
# dispatching on each record type to call the appropriate per-physics
# adjoint kernel (advection, halo, diffusion, convection, LinRood
# horizontal) and accumulating per-step surface footprints.
#
# `_run_cs_footprint_forward` and `_run_cs_observations_forward` are the
# forward-replay helpers used by FD identity tests, 4D-Var simulation
# evaluation, and Jacobian aggregation. Both advance each step with
# `_replay_cs_step!`, which composes the step as production does (step-start
# halo refresh, palindrome with the coupled center, convection) without
# writing to a tape.
# ---------------------------------------------------------------------------

function _collect_surface_footprints(lambda_panels, ops, panels_m0,
                                     mesh::CubedSphereMesh,
                                     objective::AbstractCSFootprintObjective,
                                     dt;
                                     diffusion_workspace = nothing,
                                     diffusion_meteo = nothing,
                                     convection_workspace = nothing)
    FT = eltype(lambda_panels[1])
    nsteps = count(op -> op isa _CSMidpointRecord, ops)
    footprints = Vector{typeof(_zero_surface_rates(mesh, panels_m0[1]))}(undef, nsteps)
    @inbounds for step in 1:nsteps
        footprints[step] = _zero_surface_rates(mesh, panels_m0[1])
    end

    ws = CSAdjointWorkspace(mesh, lambda_panels[1];
                            column_scratch = any(_needs_column_scratch, ops))
    _walk_window_reverse!(footprints, lambda_panels, ops, mesh, ws, dt;
                          diffusion_meteo = diffusion_meteo,
                          convection_workspace = convection_workspace)

    lag_steps = [nsteps - step for step in 1:nsteps]
    A2 = typeof(footprints[1][1])
    return CSFootprintResult{FT, typeof(objective), A2}(
        objective, footprints, lag_steps, FT(dt), zero(FT), FT(NaN))
end

# Whether a tape record's reverse needs `CSAdjointWorkspace.column_scratch`
# (the FV3 vertical profile's sweeps).
_needs_column_scratch(op::_CSSweepRecord) = needs_column_scratch(op.scheme)
_needs_column_scratch(op) = false

# Extracted from the original `_collect_surface_footprints`
# body so that the stride / revolve checkpoint drivers can call it once per
# window without re-allocating `footprints` or seeding `lambda_panels` from
# scratch. Behaviour for the `FullCheckpoint` path is unchanged: one call
# over the whole `ops` list does the same work as before.
#
# Caller responsibilities:
#   * `footprints[step]` must be pre-allocated for every `step` index that
#     appears as `op.step` in a `_CSMidpointRecord` inside `ops`.
#   * `lambda_panels` is mutated in place — chain calls in window-reverse
#     order so the adjoint state flows from final to initial across windows.
function _walk_window_reverse!(footprints, lambda_panels, ops,
                               mesh::CubedSphereMesh, ws, dt;
                               diffusion_meteo = nothing,
                               convection_workspace = nothing)
    for op in Iterators.reverse(ops)
        if op isa _CSSweepRecord
            # A record's air mass and tracer mass are read with different
            # roles: host-side tape storage returns one device cache per
            # role, so the kernel never gets `m === rm` (issue #57).
            if op.panels_rm === nothing
                _adjoint_scheme_sweep!(lambda_panels, _tape_panels(op.panels_m, AirMassRead()),
                                       op.panels_flux, op.direction, op.scheme,
                                       mesh, ws, op.flux_scale)
            else
                _adjoint_scheme_sweep!(lambda_panels, _tape_panels(op.panels_m, AirMassRead()),
                                       _tape_panels(op.panels_rm, TracerMassRead()),
                                       op.panels_flux, op.direction, op.scheme, mesh, ws,
                                       op.flux_scale)
            end
        elseif op isa _CSHaloRecord
            _adjoint_fill_panel_halos!(lambda_panels, mesh; dir=op.dir)
        elseif op isa _CSMidpointRecord
            _accumulate_surface_footprint!(footprints[op.step], lambda_panels, dt, mesh)
        elseif op isa _CSDiffusionRecord
            _apply_cs_diffusion_adjoint!(lambda_panels, _tape_panels(op.panels_m, AirMassRead()), op.op,
                                         op.workspace, op.dt,
                                         diffusion_meteo, mesh)
        elseif op isa _CSConvectionRecord
            _apply_cs_convection_adjoint!(lambda_panels, _tape_panels(op.panels_m, AirMassRead()),
                                          op.forcing, op.op, op.dt,
                                          convection_workspace, mesh)
        elseif op isa _CSLinRoodHorizRecord
            # LinRood horizontal substep reverse.
            # `lambda_panels` holds the tracer-rm adjoint propagated
            # through the tape. The substep produces an internal
            # `lambda_m_panels` (via the `c = rm / m` chain rule and
            # the donor-cell α denominator), but the
            # cs_surface_emission_footprint design — like the existing
            # `_record_cs_mass_tape` / `_record_cs_tracer_tape` paths —
            # treats meteorology (air mass evolution) as a FIXED
            # tape input. Mass-flux Jacobians are read off the per-
            # record `panels_m` / `panels_am` / `panels_bm` snapshots
            # rather than propagated via dynamic `lambda_m`, so we seed
            # `lambda_m_new = 0` for each substep and discard its
            # output. This matches the existing PPM reverse-loop
            # contract where there is no `lambda_panels_m` at all.
            lambda_m_panels = ntuple(p -> begin
                a = similar(lambda_panels[p]); fill!(a, zero(eltype(a))); a
            end, Val(6))
            _apply_cs_linrood_horizontal_adjoint!(lambda_panels,
                                                  lambda_m_panels, op, mesh)
        else
            throw(ArgumentError("unknown CS adjoint tape operation $(typeof(op))"))
        end
    end
    return nothing
end

# One step of the adjoint's forward model: the production cubed-sphere step,
# `strang_split!` followed by the `TransportModel.step!` convection block, of a
# run with a surface-flux operator. A footprint is the derivative with respect
# to the surface emissions, so the palindrome center is always
# `surface_flux_center!` with the diffusion operator's coupling; `emit!(τ)`
# adds the step's surface sources integrated over `τ`. The tape recorders
# (`TapeRecording.jl`, `LinRoodTape.jl`) apply and record the same sequence.
function _replay_cs_step!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                          mesh::CubedSphereMesh, scheme, ws, emit!;
                          dt, flux_scale, cfl_limit,
                          diffusion_op, diffusion_workspace, diffusion_meteo,
                          convection_op, convection_forcing, convection_workspace)
    diffuse! = τ -> apply_vertical_diffusion_vmr!(
        panels_rm, panels_m, diffusion_op, diffusion_workspace, τ, diffusion_meteo;
        halo_width = mesh.Hp)
    coupling = surface_flux_coupling(diffusion_op)
    midpoint! = () -> surface_flux_center!(diffuse!, emit!, coupling, dt)
    _replay_cs_transport!(cs_advection_style(scheme), panels_rm, panels_m,
                          panels_am, panels_bm, panels_cm, mesh, scheme, ws,
                          midpoint!; flux_scale, cfl_limit)
    _apply_cs_convection_forward!(panels_rm, panels_m, convection_forcing,
                                  convection_op, dt, convection_workspace, mesh)
    return nothing
end

# Split-sweep schemes: production `strang_split!` refreshes the X halos of m
# and rm at every step start (`fill_cs_step_start_halos!`), then runs the
# palindrome. Without the refresh the first X sweep reads rm halos from before
# the previous step's convection.
function _replay_cs_transport!(::CSSplitSweepStyle, panels_rm, panels_m,
                               panels_am, panels_bm, panels_cm,
                               mesh::CubedSphereMesh, scheme, ws, midpoint!;
                               flux_scale, cfl_limit)
    fill_cs_step_start_halos!(panels_rm, panels_m, mesh)
    strang_split_cs!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                     mesh, scheme, ws;
                     flux_scale = flux_scale,
                     cfl_limit = cfl_limit,
                     midpoint! = midpoint!)
    return nothing
end

# Lin-Rood: the unsplit horizontal operator (`fv_tp_2d_cs!`) refills the edge
# and corner halos it reads, so production's step-start fills are never read
# and the replay omits them. The split-sweep `strang_split_cs!` has no face
# kernels for `LinRoodPPMScheme`.
function _replay_cs_transport!(::CSLinRoodStyle, panels_rm, panels_m,
                               panels_am, panels_bm, panels_cm,
                               mesh::CubedSphereMesh, scheme, ws, midpoint!;
                               flux_scale, cfl_limit)
    _linrood_run_forward_step!(panels_rm, panels_m, panels_am, panels_bm, panels_cm,
                               mesh, scheme, ws, midpoint!; flux_scale = flux_scale)
    return nothing
end

function _run_cs_footprint_forward(panels_rm0, panels_m0,
                                   panels_am_steps,
                                   panels_bm_steps,
                                   panels_cm_steps,
                                   mesh::CubedSphereMesh,
                                   objective::AbstractCSFootprintObjective;
                                   scheme = PPMScheme(NoLimiter()),
                                   dt,
                                   flux_scale = one(eltype(panels_rm0[1])),
                                   cfl_limit = DEFAULT_CFL_LIMIT,
                                   emission_rates = nothing,
                                   perturbation = nothing,
                                   diffusion_op = NoDiffusion(),
                                   diffusion_workspace = nothing,
                                   diffusion_meteo = nothing,
                                   convection_op = NoConvection(),
                                   convection_forcing = nothing,
                                   convection_workspace = nothing)
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps, panels_cm_steps)
    Nz = size(panels_rm0[1], 3)
    _validate_objective(objective, mesh, Nz)

    panels_rm = _copy_panel_tuple(panels_rm0)
    panels_m = _copy_panel_tuple(panels_m0)
    fill_panel_halos!(panels_rm, mesh; dir = 0)
    fill_panel_halos!(panels_m, mesh; dir = 0)
    ws = CSAdvectionWorkspace(mesh, panels_rm[1]; column_scratch = needs_column_scratch(scheme))

    if emission_rates !== nothing && length(emission_rates) != nsteps
        throw(ArgumentError("emission_rates length $(length(emission_rates)) does not match nsteps $nsteps"))
    end
    _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace, nsteps)
    _require_cs_convection_workspace(convection_op, convection_workspace)

    @inbounds for step in 1:nsteps
        diffusion_op_step = _diffusion_sequence_at(diffusion_op, step, nsteps,
                                                   "diffusion_op")
        diffusion_ws_step = diffusion_op_step isa NoDiffusion ? nothing :
            _diffusion_sequence_at(diffusion_workspace, step, nsteps,
                                   "diffusion_workspace")
        emit! = τ -> begin
            emission_rates !== nothing &&
                _add_surface_rates!(panels_rm, emission_rates[step], τ, mesh)
            perturbation !== nothing && perturbation.step == step &&
                _add_surface_perturbation!(panels_rm, perturbation, τ, mesh)
            nothing
        end
        _replay_cs_step!(panels_rm, panels_m, panels_am_steps[step],
                         panels_bm_steps[step], panels_cm_steps[step],
                         mesh, scheme, ws, emit!;
                         dt, flux_scale, cfl_limit,
                         diffusion_op = diffusion_op_step,
                         diffusion_workspace = diffusion_ws_step,
                         diffusion_meteo,
                         convection_op,
                         convection_forcing = _step_convection_forcing(
                             convection_op, convection_forcing, step, nsteps),
                         convection_workspace)
    end

    return evaluate_objective(objective, panels_rm, panels_m, mesh)
end

function _run_cs_observations_forward(panels_rm0, panels_m0,
                                      panels_am_steps,
                                      panels_bm_steps,
                                      panels_cm_steps,
                                      mesh::CubedSphereMesh,
                                      observations;
                                      scheme = PPMScheme(NoLimiter()),
                                      dt,
                                      flux_scale = one(eltype(panels_rm0[1])),
                                      cfl_limit = DEFAULT_CFL_LIMIT,
                                      emission_rates = nothing,
                                      diffusion_op = NoDiffusion(),
                                      diffusion_workspace = nothing,
                                      diffusion_meteo = nothing,
                                      convection_op = NoConvection(),
                                      convection_forcing = nothing,
                                      convection_workspace = nothing)
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps, panels_cm_steps)
    Nz = size(panels_rm0[1], 3)
    @inbounds for obs in observations
        1 <= obs.step <= nsteps || throw(ArgumentError(
            "CSObservation step $(obs.step) is outside 1:$nsteps"))
        _validate_objective(obs.objective, mesh, Nz)
    end

    panels_rm = _copy_panel_tuple(panels_rm0)
    panels_m = _copy_panel_tuple(panels_m0)
    fill_panel_halos!(panels_rm, mesh; dir = 0)
    fill_panel_halos!(panels_m, mesh; dir = 0)
    ws = CSAdvectionWorkspace(mesh, panels_rm[1]; column_scratch = needs_column_scratch(scheme))

    if emission_rates !== nothing && length(emission_rates) != nsteps
        throw(ArgumentError("emission_rates length $(length(emission_rates)) does not match nsteps $nsteps"))
    end
    _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace, nsteps)
    _require_cs_convection_workspace(convection_op, convection_workspace)

    obs_by_step = [Int[] for _ in 1:nsteps]
    @inbounds for idx in eachindex(observations)
        push!(obs_by_step[observations[idx].step], idx)
    end
    FT = eltype(panels_rm0[1])
    simulated = fill(FT(NaN), length(observations))

    @inbounds for step in 1:nsteps
        diffusion_op_step = _diffusion_sequence_at(diffusion_op, step, nsteps,
                                                   "diffusion_op")
        diffusion_ws_step = diffusion_op_step isa NoDiffusion ? nothing :
            _diffusion_sequence_at(diffusion_workspace, step, nsteps,
                                   "diffusion_workspace")
        emit! = τ -> begin
            emission_rates !== nothing &&
                _add_surface_rates!(panels_rm, emission_rates[step], τ, mesh)
            nothing
        end
        _replay_cs_step!(panels_rm, panels_m, panels_am_steps[step],
                         panels_bm_steps[step], panels_cm_steps[step],
                         mesh, scheme, ws, emit!;
                         dt, flux_scale, cfl_limit,
                         diffusion_op = diffusion_op_step,
                         diffusion_workspace = diffusion_ws_step,
                         diffusion_meteo,
                         convection_op,
                         convection_forcing = _step_convection_forcing(
                             convection_op, convection_forcing, step, nsteps),
                         convection_workspace)
        for obs_idx in obs_by_step[step]
            simulated[obs_idx] = evaluate_objective(
                observations[obs_idx].objective, panels_rm, panels_m, mesh)
        end
    end

    return simulated
end
