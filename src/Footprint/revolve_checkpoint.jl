# RevolveCheckpoint driver (recursive bisection) of the footprint reverse pass.
# Split from StrideCheckpoint.jl (refactor phase 4); included by Adjoints.jl in this order.

# ---------------------------------------------------------------------------
# RevolveCheckpoint driver.
#
# Recursive-bisection variant of Griewank-Walther Revolve. The reverse
# pass walks the step range via depth-first bisection:
#
#   reverse_range!(state, lo, hi):
#       if hi == lo:   return
#       if hi == lo+1: record one step from state, walk reverse, drop
#       else:
#           mid = (lo + hi) ÷ 2
#           state_at_mid = propagate(state, lo+1:mid)  # record_ops=false
#           reverse_range!(state_at_mid, mid, hi)
#           reverse_range!(state,        lo, mid)
#
# Snapshot memory is the recursion depth — `ceil(log2(nsteps))` copies
# of the per-scheme state tuple (panels_m for linear; (panels_rm,
# panels_m) for nonlinear / LinRood). Recorder calls always copy
# their inputs internally, so the snapshot at a frame's `lo` lives in
# that frame's local `state` binding without an explicit save/restore
# step.
#
# Scope (this commit): bisection only. Optimal binomial splits
# (Griewank-Walther Algorithm 799) are a future promotion behind the
# same `RevolveCheckpoint` API.
# ---------------------------------------------------------------------------

function _collect_surface_footprints_revolve(panels_m0,
                                             panels_am_steps,
                                             panels_bm_steps,
                                             panels_cm_steps,
                                             mesh::CubedSphereMesh,
                                             scheme::CSAdjointLinearScheme,
                                             ::RevolveCheckpoint,
                                             objective::AbstractCSFootprintObjective,
                                             dt;
                                             cfl_limit,
                                             flux_scale,
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
    _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace, nsteps)
    _require_cs_convection_workspace(convection_op, convection_workspace)
    tape_storage isa AbstractCSTapeStorage && throw(ArgumentError(
        "RevolveCheckpoint requires `tape_storage` to be a Symbol " *
        "(:device, :pinned_host, or :mmap), not a pre-constructed " *
        "$(typeof(tape_storage)); the driver builds and " *
        "finalize_tape!s one storage instance per base-case step."))
    tape_path !== nothing && tape_storage !== :mmap && throw(ArgumentError(
        "tape_path requires tape_storage = :mmap; got " *
        "tape_storage = $(repr(tape_storage))"))

    # Initial state (halo-filled copy of input).
    initial_m = _copy_panel_tuple(panels_m0)
    fill_panel_halos!(initial_m, mesh; dir = 0)

    # Forward propagation to compute the final state (used for the
    # objective seed). The propagation pass runs the same kernels as
    # the per-step record but with `record_ops = false`, so the final
    # state matches the FullCheckpoint forward path bit-for-bit.
    _, final_m = _record_cs_mass_tape(
        initial_m,
        panels_am_steps, panels_bm_steps, panels_cm_steps,
        mesh, scheme;
        cfl_limit = cfl_limit, flux_scale = flux_scale, dt = dt,
        diffusion_op = diffusion_op,
        diffusion_workspace = diffusion_workspace,
        convection_op = convection_op,
        convection_forcing = convection_forcing,
        tape_storage = :device,
        step_offset = 0,
        record_ops = false)

    lambda_panels = _build_stride_lambda_panels(final_m, final_adjoint_seed,
                                                objective, mesh, FT)

    footprints = [_zero_surface_rates(mesh, panels_m0[1]) for _ in 1:nsteps]
    ws = CSAdjointWorkspace(mesh, lambda_panels[1])

    # Recursive bisection driver. `state_m` is the panels_m state at
    # step `lo`; the function reverses steps [lo+1, hi] in place
    # against `lambda_panels`, accumulating into `footprints[]`.
    function reverse_range!(state_m, lo, hi)
        if hi == lo
            return nothing
        elseif hi - lo == 1
            storage_w = _build_window_storage(tape_storage, tape_path,
                                              _revolve_step_subdir(hi))
            ops_window, _ = _record_cs_mass_tape(
                state_m,
                _slice_step_kwarg(panels_am_steps, hi:hi),
                _slice_step_kwarg(panels_bm_steps, hi:hi),
                _slice_step_kwarg(panels_cm_steps, hi:hi),
                mesh, scheme;
                cfl_limit = cfl_limit,
                flux_scale = flux_scale,
                dt = dt,
                diffusion_op = _slice_step_kwarg(diffusion_op, hi:hi),
                diffusion_workspace = _slice_step_kwarg(diffusion_workspace, hi:hi),
                convection_op = convection_op,
                convection_forcing = _slice_step_kwarg(convection_forcing, hi:hi),
                tape_storage = storage_w,
                step_offset = lo)
            try
                _walk_window_reverse!(footprints, lambda_panels, ops_window, mesh, ws, dt;
                                      diffusion_meteo = diffusion_meteo,
                                      convection_workspace = convection_workspace)
            finally
                finalize_tape!(storage_w; quiet = true,
                           strict = tape_path !== nothing)
            end
        else
            mid = (lo + hi) ÷ 2
            # Propagate state from lo to mid (no record). The recorder
            # copies its input internally, so the original `state_m`
            # in this frame still points to the lo state — usable for
            # the second recursive call below.
            _, state_m_at_mid = _record_cs_mass_tape(
                state_m,
                _slice_step_kwarg(panels_am_steps, (lo + 1):mid),
                _slice_step_kwarg(panels_bm_steps, (lo + 1):mid),
                _slice_step_kwarg(panels_cm_steps, (lo + 1):mid),
                mesh, scheme;
                cfl_limit = cfl_limit,
                flux_scale = flux_scale,
                dt = dt,
                diffusion_op = _slice_step_kwarg(diffusion_op, (lo + 1):mid),
                diffusion_workspace = _slice_step_kwarg(diffusion_workspace, (lo + 1):mid),
                convection_op = convection_op,
                convection_forcing = _slice_step_kwarg(convection_forcing, (lo + 1):mid),
                tape_storage = :device,
                step_offset = lo,
                record_ops = false)
            reverse_range!(state_m_at_mid, mid, hi)
            reverse_range!(state_m,        lo, mid)
        end
        return nothing
    end

    reverse_range!(initial_m, 0, nsteps)

    lag_steps = [nsteps - step for step in 1:nsteps]
    A2 = typeof(footprints[1][1])
    return CSFootprintResult{FT, typeof(objective), A2}(
        objective, footprints, lag_steps, FT(dt), zero(FT), FT(NaN))
end

function _collect_surface_footprints_revolve(panels_rm0, panels_m0,
                                             panels_am_steps,
                                             panels_bm_steps,
                                             panels_cm_steps,
                                             mesh::CubedSphereMesh,
                                             scheme::CSAdjointNonlinearScheme,
                                             ::RevolveCheckpoint,
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
        "RevolveCheckpoint requires `tape_storage` to be a Symbol " *
        "(:device, :pinned_host, or :mmap), not a pre-constructed " *
        "$(typeof(tape_storage)); the driver builds and " *
        "finalize_tape!s one storage instance per base-case step."))
    tape_path !== nothing && tape_storage !== :mmap && throw(ArgumentError(
        "tape_path requires tape_storage = :mmap; got " *
        "tape_storage = $(repr(tape_storage))"))

    initial_rm = _copy_panel_tuple(panels_rm0)
    initial_m  = _copy_panel_tuple(panels_m0)
    fill_panel_halos!(initial_rm, mesh; dir = 0)
    fill_panel_halos!(initial_m,  mesh; dir = 0)

    # Forward propagation to compute the final state for seeding.
    _, _, final_m = _record_cs_tracer_tape(
        initial_rm, initial_m,
        panels_am_steps, panels_bm_steps, panels_cm_steps,
        mesh, scheme;
        cfl_limit = cfl_limit, flux_scale = flux_scale, dt = dt,
        base_emission_rates = base_emission_rates,
        diffusion_op = diffusion_op,
        diffusion_workspace = diffusion_workspace,
        diffusion_meteo = diffusion_meteo,
        convection_op = convection_op,
        convection_forcing = convection_forcing,
        convection_workspace = convection_workspace,
        tape_storage = :device,
        step_offset = 0,
        record_ops = false)

    lambda_panels = _build_stride_lambda_panels(final_m, final_adjoint_seed,
                                                objective, mesh, FT)

    footprints = [_zero_surface_rates(mesh, panels_m0[1]) for _ in 1:nsteps]
    ws = CSAdjointWorkspace(mesh, lambda_panels[1])

    function reverse_range!(state_rm, state_m, lo, hi)
        if hi == lo
            return nothing
        elseif hi - lo == 1
            storage_w = _build_window_storage(tape_storage, tape_path,
                                              _revolve_step_subdir(hi))
            ops_window, _, _ = _record_cs_tracer_tape(
                state_rm, state_m,
                _slice_step_kwarg(panels_am_steps, hi:hi),
                _slice_step_kwarg(panels_bm_steps, hi:hi),
                _slice_step_kwarg(panels_cm_steps, hi:hi),
                mesh, scheme;
                cfl_limit = cfl_limit,
                flux_scale = flux_scale,
                dt = dt,
                base_emission_rates = _slice_step_kwarg(base_emission_rates, hi:hi),
                diffusion_op = _slice_step_kwarg(diffusion_op, hi:hi),
                diffusion_workspace = _slice_step_kwarg(diffusion_workspace, hi:hi),
                diffusion_meteo = diffusion_meteo,
                convection_op = convection_op,
                convection_forcing = _slice_step_kwarg(convection_forcing, hi:hi),
                convection_workspace = convection_workspace,
                tape_storage = storage_w,
                step_offset = lo)
            try
                _walk_window_reverse!(footprints, lambda_panels, ops_window, mesh, ws, dt;
                                      diffusion_meteo = diffusion_meteo,
                                      convection_workspace = convection_workspace)
            finally
                finalize_tape!(storage_w; quiet = true,
                           strict = tape_path !== nothing)
            end
        else
            mid = (lo + hi) ÷ 2
            _, state_rm_at_mid, state_m_at_mid = _record_cs_tracer_tape(
                state_rm, state_m,
                _slice_step_kwarg(panels_am_steps, (lo + 1):mid),
                _slice_step_kwarg(panels_bm_steps, (lo + 1):mid),
                _slice_step_kwarg(panels_cm_steps, (lo + 1):mid),
                mesh, scheme;
                cfl_limit = cfl_limit,
                flux_scale = flux_scale,
                dt = dt,
                base_emission_rates = _slice_step_kwarg(base_emission_rates, (lo + 1):mid),
                diffusion_op = _slice_step_kwarg(diffusion_op, (lo + 1):mid),
                diffusion_workspace = _slice_step_kwarg(diffusion_workspace, (lo + 1):mid),
                diffusion_meteo = diffusion_meteo,
                convection_op = convection_op,
                convection_forcing = _slice_step_kwarg(convection_forcing, (lo + 1):mid),
                convection_workspace = convection_workspace,
                tape_storage = :device,
                step_offset = lo,
                record_ops = false)
            reverse_range!(state_rm_at_mid, state_m_at_mid, mid, hi)
            reverse_range!(state_rm,         state_m,         lo, mid)
        end
        return nothing
    end

    reverse_range!(initial_rm, initial_m, 0, nsteps)

    lag_steps = [nsteps - step for step in 1:nsteps]
    A2 = typeof(footprints[1][1])
    return CSFootprintResult{FT, typeof(objective), A2}(
        objective, footprints, lag_steps, FT(dt), zero(FT), FT(NaN))
end

function _collect_surface_footprints_revolve(panels_rm0, panels_m0,
                                             panels_am_steps,
                                             panels_bm_steps,
                                             panels_cm_steps,
                                             mesh::CubedSphereMesh,
                                             scheme::CSAdjointLinRoodScheme,
                                             ::RevolveCheckpoint,
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
        "RevolveCheckpoint requires `tape_storage` to be a Symbol " *
        "(:device for LinRood), not a pre-constructed " *
        "$(typeof(tape_storage)); the driver builds and " *
        "finalize_tape!s one storage instance per base-case step."))
    tape_storage === :device || throw(ArgumentError(
        "LinRoodPPMScheme + RevolveCheckpoint requires tape_storage = :device " *
        "(got $(repr(tape_storage))). The `_CSLinRoodHorizRecord` struct " *
        "holds device-resident panel tuples directly; mmap eviction is " *
        "reserved for a Plan 26 follow-up that refactors the LinRood tape."))
    tape_path === nothing || throw(ArgumentError(
        "LinRoodPPMScheme does not support tape_path: storage is fixed " *
        "to :device. Use RevolveCheckpoint with the nonlinear PPM scheme " *
        "(tape_storage = :mmap) if disk-backed tapes are required."))

    initial_rm = _copy_panel_tuple(panels_rm0)
    initial_m  = _copy_panel_tuple(panels_m0)
    fill_panel_halos!(initial_rm, mesh; dir = 0)
    fill_panel_halos!(initial_m,  mesh; dir = 0)

    _, _, final_m = _record_cs_linrood_tape(
        initial_rm, initial_m,
        panels_am_steps, panels_bm_steps, panels_cm_steps,
        mesh, scheme;
        cfl_limit = cfl_limit, flux_scale = flux_scale, dt = dt,
        base_emission_rates = base_emission_rates,
        diffusion_op = diffusion_op,
        diffusion_workspace = diffusion_workspace,
        diffusion_meteo = diffusion_meteo,
        convection_op = convection_op,
        convection_forcing = convection_forcing,
        convection_workspace = convection_workspace,
        tape_storage = :device,
        step_offset = 0,
        record_ops = false)

    lambda_panels = _build_stride_lambda_panels(final_m, final_adjoint_seed,
                                                objective, mesh, FT)

    footprints = [_zero_surface_rates(mesh, panels_m0[1]) for _ in 1:nsteps]
    ws = CSAdjointWorkspace(mesh, lambda_panels[1])

    function reverse_range!(state_rm, state_m, lo, hi)
        if hi == lo
            return nothing
        elseif hi - lo == 1
            storage_w = _tape_storage(tape_storage)
            ops_window, _, _ = _record_cs_linrood_tape(
                state_rm, state_m,
                _slice_step_kwarg(panels_am_steps, hi:hi),
                _slice_step_kwarg(panels_bm_steps, hi:hi),
                _slice_step_kwarg(panels_cm_steps, hi:hi),
                mesh, scheme;
                cfl_limit = cfl_limit,
                flux_scale = flux_scale,
                dt = dt,
                base_emission_rates = _slice_step_kwarg(base_emission_rates, hi:hi),
                diffusion_op = _slice_step_kwarg(diffusion_op, hi:hi),
                diffusion_workspace = _slice_step_kwarg(diffusion_workspace, hi:hi),
                diffusion_meteo = diffusion_meteo,
                convection_op = convection_op,
                convection_forcing = _slice_step_kwarg(convection_forcing, hi:hi),
                convection_workspace = convection_workspace,
                tape_storage = storage_w,
                step_offset = lo)
            try
                _walk_window_reverse!(footprints, lambda_panels, ops_window, mesh, ws, dt;
                                      diffusion_meteo = diffusion_meteo,
                                      convection_workspace = convection_workspace)
            finally
                finalize_tape!(storage_w; quiet = true,
                           strict = tape_path !== nothing)
            end
        else
            mid = (lo + hi) ÷ 2
            _, state_rm_at_mid, state_m_at_mid = _record_cs_linrood_tape(
                state_rm, state_m,
                _slice_step_kwarg(panels_am_steps, (lo + 1):mid),
                _slice_step_kwarg(panels_bm_steps, (lo + 1):mid),
                _slice_step_kwarg(panels_cm_steps, (lo + 1):mid),
                mesh, scheme;
                cfl_limit = cfl_limit,
                flux_scale = flux_scale,
                dt = dt,
                base_emission_rates = _slice_step_kwarg(base_emission_rates, (lo + 1):mid),
                diffusion_op = _slice_step_kwarg(diffusion_op, (lo + 1):mid),
                diffusion_workspace = _slice_step_kwarg(diffusion_workspace, (lo + 1):mid),
                diffusion_meteo = diffusion_meteo,
                convection_op = convection_op,
                convection_forcing = _slice_step_kwarg(convection_forcing, (lo + 1):mid),
                convection_workspace = convection_workspace,
                tape_storage = :device,
                step_offset = lo,
                record_ops = false)
            reverse_range!(state_rm_at_mid, state_m_at_mid, mid, hi)
            reverse_range!(state_rm,         state_m,         lo, mid)
        end
        return nothing
    end

    reverse_range!(initial_rm, initial_m, 0, nsteps)

    lag_steps = [nsteps - step for step in 1:nsteps]
    A2 = typeof(footprints[1][1])
    return CSFootprintResult{FT, typeof(objective), A2}(
        objective, footprints, lag_steps, FT(dt), zero(FT), FT(NaN))
end
