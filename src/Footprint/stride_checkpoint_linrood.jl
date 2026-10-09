# Strided checkpoint driver for the Lin-Rood horizontal tape.
# Split from StrideCheckpoint.jl (refactor phase 4); included by Adjoints.jl in this order.

# ---------------------------------------------------------------------------
# Strided checkpoint driver for the LinRood horizontal
# tape (LinRoodPPMScheme).
#
# LinRood differs from the nonlinear PPM tracer tape in two key ways:
#
# 1. **Storage policy is fixed to `:device`.** The `_CSLinRoodHorizRecord`
#    struct holds raw `NTuple{6, P}` references rather than per-policy
#    slots, so `_record_cs_linrood_tape` rejects non-`:device` storage
#    via `_linrood_validate_tape_storage`. Per-window mmap finalization
#    is therefore a no-op for LinRood; storage construction returns a
#    plain `DeviceCSTapeStorage`.
#
# 2. **Horizontal substep is unsplit.** Each step runs one
#    `_record_linrood_horizontal_substep!` (which itself allocates
#    per-substep face / q_buf scratch — those go out of scope at the
#    substep's function exit, so peak scratch memory is bounded by
#    one substep, independent of `nsteps`). Setting `record_ops =
#    false` further elides the per-substep `panels_rm_tape` /
#    `panels_m_tape` / `panels_q_buf_phase{2,3}` snapshots that would
#    otherwise be retained for the reverse pass.
#
# Structurally identical to `_propagate_tracer_checkpoints` /
# `_collect_surface_footprints_stride(::CSAdjointNonlinearScheme)`
# in stride_checkpoint_ppm.jl — same `(rm, m)`-pair propagation, same `_walk_window_reverse!`
# reverse loop, same sliced per-step kwargs.
# ---------------------------------------------------------------------------

function _propagate_linrood_checkpoints(panels_rm0, panels_m0,
                                        panels_am_steps,
                                        panels_bm_steps,
                                        panels_cm_steps,
                                        mesh::CubedSphereMesh,
                                        scheme::CSAdjointLinRoodScheme,
                                        schedule::StrideCheckpoint;
                                        cfl_limit,
                                        flux_scale,
                                        dt,
                                        base_emission_rates = nothing,
                                        diffusion_op = NoDiffusion(),
                                        diffusion_workspace = nothing,
                                        diffusion_meteo = nothing,
                                        convection_op = NoConvection(),
                                        convection_forcing = nothing,
                                        convection_workspace = nothing)
    nsteps = length(panels_am_steps)
    nw = checkpoint_window_count(schedule, nsteps)

    initial_rm = _copy_panel_tuple(panels_rm0)
    initial_m  = _copy_panel_tuple(panels_m0)
    fill_panel_halos!(initial_rm, mesh; dir = 0)
    fill_panel_halos!(initial_m,  mesh; dir = 0)

    rm_checkpoints = Vector{typeof(initial_rm)}(undef, nw + 1)
    m_checkpoints  = Vector{typeof(initial_m)}(undef, nw + 1)
    rm_checkpoints[1] = initial_rm
    m_checkpoints[1]  = initial_m

    current_rm = initial_rm
    current_m  = initial_m
    @inbounds for w in 1:nw
        window_range = checkpoint_window_range(schedule, w, nsteps)
        _, current_rm, current_m = _record_cs_linrood_tape(
            current_rm, current_m,
            _slice_step_kwarg(panels_am_steps, window_range),
            _slice_step_kwarg(panels_bm_steps, window_range),
            _slice_step_kwarg(panels_cm_steps, window_range),
            mesh, scheme;
            cfl_limit = cfl_limit,
            flux_scale = flux_scale,
            dt = dt,
            base_emission_rates = _slice_step_kwarg(base_emission_rates, window_range),
            diffusion_op = _slice_step_kwarg(diffusion_op, window_range),
            diffusion_workspace = _slice_step_kwarg(diffusion_workspace, window_range),
            diffusion_meteo = diffusion_meteo,
            convection_op = convection_op,
            convection_forcing = _slice_step_kwarg(convection_forcing, window_range),
            convection_workspace = convection_workspace,
            tape_storage = :device,
            step_offset = first(window_range) - 1,
            record_ops = false)
        rm_checkpoints[w + 1] = current_rm
        m_checkpoints[w + 1]  = current_m
    end
    return rm_checkpoints, m_checkpoints
end

"""
    _collect_surface_footprints_stride(panels_rm0, panels_m0, ...,
                                       scheme::CSAdjointLinRoodScheme, ...)

Strided-checkpoint driver for the LinRood horizontal tape. Storage is
fixed to `:device` (see `_linrood_validate_tape_storage`); passing
`tape_storage = :mmap` raises an `ArgumentError` deep in the recorder
on the first window's reverse-pass call. Reject explicitly up front
so the failure mode is loud and stride-aware.
"""
function _collect_surface_footprints_stride(panels_rm0, panels_m0,
                                            panels_am_steps,
                                            panels_bm_steps,
                                            panels_cm_steps,
                                            mesh::CubedSphereMesh,
                                            scheme::CSAdjointLinRoodScheme,
                                            schedule::StrideCheckpoint,
                                            objective::AbstractCSFootprintObjective,
                                            dt;
                                            cfl_limit,
                                            flux_scale,
                                            base_emission_rates = nothing,
                                            diffusion_op = NoDiffusion(),
                                            diffusion_workspace = nothing,
                                            diffusion_meteo = nothing,
                                            convection_op = NoConvection(),
                                            convection_forcing = nothing,
                                            convection_workspace = nothing,
                                            tape_storage = :device,
                                            tape_path::Union{Nothing, AbstractString} = nothing,
                                            final_adjoint_seed = nothing)
    FT = eltype(panels_m0[1])
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps,
                                       panels_cm_steps)
    Nz = size(panels_m0[1], 3)
    final_adjoint_seed === nothing && _validate_objective(objective, mesh, Nz)
    _validate_emission_rates(base_emission_rates, nsteps, mesh, "base_emission_rates")
    _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace, nsteps)
    _require_cs_convection_workspace(convection_op, convection_workspace)
    tape_storage isa AbstractCSTapeStorage && throw(ArgumentError(
        "StrideCheckpoint requires `tape_storage` to be a Symbol " *
        "(:device for LinRood), not a pre-constructed " *
        "$(typeof(tape_storage)); the stride driver builds and " *
        "finalize_tape!s one storage instance per window."))
    # LinRood's reverse path only accepts `:device`; rather than wait
    # for the recorder to throw inside the first window, surface it
    # here with a stride-aware diagnostic.
    tape_storage === :device || throw(ArgumentError(
        "LinRoodPPMScheme + StrideCheckpoint requires tape_storage = :device " *
        "(got $(repr(tape_storage))). The `_CSLinRoodHorizRecord` struct " *
        "holds device-resident panel tuples directly; mmap eviction is " *
        "reserved for a Plan 26 follow-up that refactors the LinRood tape."))
    # `tape_path` only makes sense for `:mmap` storage. LinRood is
    # already pinned to `:device` above, so any non-nothing path is
    # incompatible — reject loudly here instead of silently ignoring
    # the kwarg.
    tape_path === nothing || throw(ArgumentError(
        "LinRoodPPMScheme does not support tape_path: storage is fixed " *
        "to :device. Use StrideCheckpoint with the nonlinear PPM scheme " *
        "(tape_storage = :mmap) if disk-backed tapes are required."))

    rm_checkpoints, m_checkpoints = _propagate_linrood_checkpoints(
        panels_rm0, panels_m0,
        panels_am_steps, panels_bm_steps, panels_cm_steps,
        mesh, scheme, schedule;
        cfl_limit = cfl_limit, flux_scale = flux_scale, dt = dt,
        base_emission_rates = base_emission_rates,
        diffusion_op = diffusion_op,
        diffusion_workspace = diffusion_workspace,
        diffusion_meteo = diffusion_meteo,
        convection_op = convection_op,
        convection_forcing = convection_forcing,
        convection_workspace = convection_workspace)

    nw = checkpoint_window_count(schedule, nsteps)
    final_m = m_checkpoints[nw + 1]

    lambda_panels = _build_stride_lambda_panels(final_m, final_adjoint_seed,
                                                objective, mesh, FT)

    footprints = [_zero_surface_rates(mesh, panels_m0[1]) for _ in 1:nsteps]
    ws = CSAdjointWorkspace(mesh, lambda_panels[1])

    @inbounds for w in nw:-1:1
        window_range = checkpoint_window_range(schedule, w, nsteps)
        storage_w = _tape_storage(tape_storage)
        ops_window, _, _ = _record_cs_linrood_tape(
            rm_checkpoints[w], m_checkpoints[w],
            _slice_step_kwarg(panels_am_steps, window_range),
            _slice_step_kwarg(panels_bm_steps, window_range),
            _slice_step_kwarg(panels_cm_steps, window_range),
            mesh, scheme;
            cfl_limit = cfl_limit,
            flux_scale = flux_scale,
            dt = dt,
            base_emission_rates = _slice_step_kwarg(base_emission_rates, window_range),
            diffusion_op = _slice_step_kwarg(diffusion_op, window_range),
            diffusion_workspace = _slice_step_kwarg(diffusion_workspace, window_range),
            diffusion_meteo = diffusion_meteo,
            convection_op = convection_op,
            convection_forcing = _slice_step_kwarg(convection_forcing, window_range),
            convection_workspace = convection_workspace,
            tape_storage = storage_w,
            step_offset = first(window_range) - 1)
        try
            _walk_window_reverse!(footprints, lambda_panels, ops_window, mesh, ws, dt;
                                  diffusion_meteo = diffusion_meteo,
                                  convection_workspace = convection_workspace)
        finally
            finalize_tape!(storage_w; quiet = true,
                           strict = tape_path !== nothing)
        end
    end

    lag_steps = [nsteps - step for step in 1:nsteps]
    A2 = typeof(footprints[1][1])
    return CSFootprintResult{FT, typeof(objective), A2}(
        objective, footprints, lag_steps, FT(dt), zero(FT), FT(NaN))
end
