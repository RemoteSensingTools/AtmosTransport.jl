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
# evaluation, and Jacobian aggregation. They share the same Strang-
# palindrome-with-emissions structure as the production runtime, but
# without writing to a tape.
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

    ws = CSAdjointWorkspace(mesh, lambda_panels[1])
    _walk_window_reverse!(footprints, lambda_panels, ops, mesh, ws, dt;
                          diffusion_meteo = diffusion_meteo,
                          convection_workspace = convection_workspace)

    lag_steps = [nsteps - step for step in 1:nsteps]
    A2 = typeof(footprints[1][1])
    return CSFootprintResult{FT, typeof(objective), A2}(
        objective, footprints, lag_steps, FT(dt), zero(FT), FT(NaN))
end

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
            if op.panels_rm === nothing
                _adjoint_scheme_sweep!(lambda_panels, _tape_panels(op.panels_m),
                                       op.panels_flux, op.direction, op.scheme,
                                       mesh, ws, op.flux_scale)
            else
                _adjoint_scheme_sweep!(lambda_panels, _tape_panels(op.panels_m),
                                       _tape_panels(op.panels_rm), op.panels_flux,
                                       op.direction, op.scheme, mesh, ws,
                                       op.flux_scale)
            end
        elseif op isa _CSHaloRecord
            _adjoint_fill_panel_halos!(lambda_panels, mesh; dir=op.dir)
        elseif op isa _CSMidpointRecord
            _accumulate_surface_footprint!(footprints[op.step], lambda_panels, dt, mesh)
        elseif op isa _CSDiffusionRecord
            _apply_cs_diffusion_adjoint!(lambda_panels, _tape_panels(op.panels_m), op.op,
                                         op.workspace, op.dt,
                                         diffusion_meteo, mesh)
        elseif op isa _CSConvectionRecord
            _apply_cs_convection_adjoint!(lambda_panels, _tape_panels(op.panels_m),
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

function _run_cs_footprint_forward(panels_rm0, panels_m0,
                                   panels_am_steps,
                                   panels_bm_steps,
                                   panels_cm_steps,
                                   mesh::CubedSphereMesh,
                                   objective::AbstractCSFootprintObjective;
                                   scheme = PPMScheme(NoLimiter()),
                                   dt,
                                   flux_scale = one(eltype(panels_rm0[1])),
                                   cfl_limit = 0.95,
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
    ws = CSAdvectionWorkspace(mesh, panels_rm[1])

    if emission_rates !== nothing && length(emission_rates) != nsteps
        throw(ArgumentError("emission_rates length $(length(emission_rates)) does not match nsteps $nsteps"))
    end
    _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace, nsteps)
    _require_cs_convection_workspace(convection_op, convection_workspace)

    @inbounds for step in 1:nsteps
        midpoint! = nothing
        diffusion_op_step = _diffusion_sequence_at(diffusion_op, step, nsteps,
                                                   "diffusion_op")
        diffusion_ws_step = diffusion_op_step isa NoDiffusion ? nothing :
            _diffusion_sequence_at(diffusion_workspace, step, nsteps,
                                   "diffusion_workspace")
        needs_midpoint = !(diffusion_op_step isa NoDiffusion) ||
                         emission_rates !== nothing ||
                         (perturbation !== nothing && perturbation.step == step)
        if needs_midpoint
            midpoint! = () -> begin
                if !(diffusion_op_step isa NoDiffusion)
                    apply_vertical_diffusion_vmr!(
                        panels_rm, panels_m, diffusion_op_step, diffusion_ws_step,
                        dt / 2, diffusion_meteo; halo_width = mesh.Hp)
                end
                emission_rates !== nothing &&
                    _add_surface_rates!(panels_rm, emission_rates[step], dt, mesh)
                perturbation !== nothing && perturbation.step == step &&
                    _add_surface_perturbation!(panels_rm, perturbation, dt, mesh)
                if !(diffusion_op_step isa NoDiffusion)
                    apply_vertical_diffusion_vmr!(
                        panels_rm, panels_m, diffusion_op_step, diffusion_ws_step,
                        dt / 2, diffusion_meteo; halo_width = mesh.Hp)
                end
                nothing
            end
        end
        if scheme isa LinRoodPPMScheme
            # LinRood uses a 3-phase unsplit horizontal + Z sweep
            # composition; the standard split-sweep `strang_split_cs!`
            # doesn't have face kernels for LinRoodPPMScheme.
            _linrood_run_forward_step!(panels_rm, panels_m,
                panels_am_steps[step], panels_bm_steps[step],
                panels_cm_steps[step], mesh, scheme, ws, midpoint!;
                flux_scale = flux_scale)
        else
            strang_split_cs!(panels_rm, panels_m,
                             panels_am_steps[step], panels_bm_steps[step], panels_cm_steps[step],
                             mesh, scheme, ws;
                             flux_scale = flux_scale,
                             cfl_limit = cfl_limit,
                             midpoint! = midpoint!)
        end
        if !(convection_op isa NoConvection)
            forcing_step = _convection_forcing_at(convection_forcing, step, nsteps)
            forcing_step === nothing && throw(ArgumentError(
                "convection_op=$(typeof(convection_op)) requires `convection_forcing`"))
            _apply_cs_convection_forward!(panels_rm, panels_m, forcing_step,
                                          convection_op, dt,
                                          convection_workspace, mesh)
        end
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
                                      cfl_limit = 0.95,
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
    ws = CSAdvectionWorkspace(mesh, panels_rm[1])

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
        midpoint! = nothing
        diffusion_op_step = _diffusion_sequence_at(diffusion_op, step, nsteps,
                                                   "diffusion_op")
        diffusion_ws_step = diffusion_op_step isa NoDiffusion ? nothing :
            _diffusion_sequence_at(diffusion_workspace, step, nsteps,
                                   "diffusion_workspace")
        needs_midpoint = !(diffusion_op_step isa NoDiffusion) ||
                         emission_rates !== nothing
        if needs_midpoint
            midpoint! = () -> begin
                if !(diffusion_op_step isa NoDiffusion)
                    apply_vertical_diffusion_vmr!(
                        panels_rm, panels_m, diffusion_op_step, diffusion_ws_step,
                        dt / 2, diffusion_meteo; halo_width = mesh.Hp)
                end
                emission_rates !== nothing &&
                    _add_surface_rates!(panels_rm, emission_rates[step], dt, mesh)
                if !(diffusion_op_step isa NoDiffusion)
                    apply_vertical_diffusion_vmr!(
                        panels_rm, panels_m, diffusion_op_step, diffusion_ws_step,
                        dt / 2, diffusion_meteo; halo_width = mesh.Hp)
                end
                nothing
            end
        end
        if scheme isa LinRoodPPMScheme
            # LinRood uses a 3-phase unsplit horizontal + Z sweep
            # composition; the standard split-sweep `strang_split_cs!`
            # doesn't have face kernels for LinRoodPPMScheme.
            _linrood_run_forward_step!(panels_rm, panels_m,
                panels_am_steps[step], panels_bm_steps[step],
                panels_cm_steps[step], mesh, scheme, ws, midpoint!;
                flux_scale = flux_scale)
        else
            strang_split_cs!(panels_rm, panels_m,
                             panels_am_steps[step], panels_bm_steps[step], panels_cm_steps[step],
                             mesh, scheme, ws;
                             flux_scale = flux_scale,
                             cfl_limit = cfl_limit,
                             midpoint! = midpoint!)
        end
        if !(convection_op isa NoConvection)
            forcing_step = _convection_forcing_at(convection_forcing, step, nsteps)
            forcing_step === nothing && throw(ArgumentError(
                "convection_op=$(typeof(convection_op)) requires `convection_forcing`"))
            _apply_cs_convection_forward!(panels_rm, panels_m, forcing_step,
                                          convection_op, dt,
                                          convection_workspace, mesh)
        end
        for obs_idx in obs_by_step[step]
            simulated[obs_idx] = evaluate_objective(
                observations[obs_idx].objective, panels_rm, panels_m, mesh)
        end
    end

    return simulated
end
