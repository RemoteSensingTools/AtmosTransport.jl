# Strided checkpoint driver for the nonlinear (monotone) PPM tape.
# Included by Adjoints.jl after StrideCheckpoint.jl, whose shared helpers it uses.

# ---------------------------------------------------------------------------
# Strided checkpoint driver for the nonlinear PPM
# tracer tape.
#
# Differs from the linear-scheme driver (StrideCheckpoint.jl) in three ways:
#
# 1. **Two state checkpoints per window.** The tracer tape's reverse
#    walk needs the rm-state replayed alongside panels_m, so the
#    propagation pass saves `(panels_rm, panels_m)` snapshots — not
#    just `panels_m`. `_record_cs_tracer_tape` is called with
#    `record_ops = false`, which still runs the forward diffusion,
#    emission, and convection kernels (those genuinely mutate
#    panels_rm and panels_m forward in the tracer tape, unlike the
#    mass tape).
#
# 2. **`base_emission_rates` is load-bearing.** The tracer-tape
#    recorder accepts a per-step emission rate and adds it to
#    panels_rm at the midpoint. Stride needs to thread the sliced
#    `base_emission_rates` into each window.
#
# 3. **3-tuple from `_record_cs_tracer_tape`.** The recorder returns
#    `(ops, panels_rm, panels_m)`; we use all three.
#
# The reverse walk reuses `_walk_window_reverse!` — the per-record
# dispatch is identical between mass and tracer tapes.
# ---------------------------------------------------------------------------

function _propagate_tracer_checkpoints(panels_rm0, panels_m0,
                                       panels_am_steps,
                                       panels_bm_steps,
                                       panels_cm_steps,
                                       mesh::CubedSphereMesh,
                                       scheme::CSAdjointNonlinearScheme,
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
    initial_m = _copy_panel_tuple(panels_m0)
    fill_panel_halos!(initial_rm, mesh; dir = 0)
    fill_panel_halos!(initial_m, mesh; dir = 0)

    rm_checkpoints = Vector{typeof(initial_rm)}(undef, nw + 1)
    m_checkpoints  = Vector{typeof(initial_m)}(undef, nw + 1)
    rm_checkpoints[1] = initial_rm
    m_checkpoints[1]  = initial_m

    current_rm = initial_rm
    current_m  = initial_m
    @inbounds for w in 1:nw
        window_range = checkpoint_window_range(schedule, w, nsteps)
        _, current_rm, current_m = _record_cs_tracer_tape(
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
                                       scheme::CSAdjointNonlinearScheme, ...)

Strided-checkpoint driver for the nonlinear-PPM (monotone-limited)
tracer tape. Returns a `CSFootprintResult` that matches the
`FullCheckpoint` path to bit / floating-point accuracy on the same
`(panels_rm0, panels_m0)` inputs; the same per-window mmap-lifetime
discipline as the linear driver applies (`try / finally` finalize).

`base_emission_rates` is sliced per window. `convection_forcing` and
`diffusion_workspace` are likewise sliced if they are
`AbstractVector`s.
"""
function _collect_surface_footprints_stride(panels_rm0, panels_m0,
                                            panels_am_steps,
                                            panels_bm_steps,
                                            panels_cm_steps,
                                            mesh::CubedSphereMesh,
                                            scheme::CSAdjointNonlinearScheme,
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
        "(:device, :pinned_host, or :mmap), not a pre-constructed " *
        "$(typeof(tape_storage)); the stride driver builds and " *
        "finalize_tape!s one storage instance per window."))
    tape_path !== nothing && tape_storage !== :mmap && throw(ArgumentError(
        "tape_path requires tape_storage = :mmap; got " *
        "tape_storage = $(repr(tape_storage))"))

    rm_checkpoints, m_checkpoints = _propagate_tracer_checkpoints(
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
        storage_w = _build_window_storage(tape_storage, tape_path,
                                          _stride_window_subdir(w))
        ops_window, _, _ = _record_cs_tracer_tape(
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
