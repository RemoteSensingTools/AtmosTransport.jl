# ---------------------------------------------------------------------------
# Strided checkpoint driver for `cs_surface_emission_footprint`.
#
# `_collect_surface_footprints_stride` is the entry point used when the
# `checkpoint` kwarg is a `StrideCheckpoint(K)`. It runs a forward
# checkpoint pass that saves `panels_m` at every K-step boundary,
# discarding ops as it goes, and then walks the reverse pass one
# window at a time — re-recording each window's tape from its
# saved left-edge checkpoint, walking those window ops backwards,
# and dropping them before moving to the previous window.
#
# This file holds the linear-scheme mass-tape driver and the helpers the
# others share; the monotone-PPM tracer tape is in stride_checkpoint_ppm.jl,
# the Lin-Rood horizontal tape in stride_checkpoint_linrood.jl, and the
# RevolveCheckpoint driver in revolve_checkpoint.jl.
# ---------------------------------------------------------------------------

# Per-step kwargs (`diffusion_op`, `diffusion_workspace`,
# `convection_forcing`) are either Vectors of length `nsteps` or a
# scalar broadcast value (handled by `_diffusion_sequence_at` /
# `_convection_forcing_at`). When recording one window, slice
# Vector inputs to the window's step range; pass scalar inputs
# through unchanged so the downstream `_*_sequence_at` calls keep
# their broadcast semantics.
_slice_step_kwarg(v::AbstractVector, range::AbstractUnitRange) = view(v, range)
_slice_step_kwarg(v, ::AbstractUnitRange) = v

# Per-window / per-base-case subdirectory under a user-supplied
# `tape_path`. Five zero-padded digits keep alphabetical sort agreeing
# with numeric order through nsteps = 99999, which dominates any
# physically-motivated CS run length. The leading tag distinguishes
# Stride (`window_*`) from Revolve (`step_*`) subdirs so a single
# `tape_path` tree can host both schedules without collisions.
_stride_window_subdir(w::Integer) = "window_" * lpad(w, 5, '0')
_revolve_step_subdir(step::Integer) = "step_" * lpad(step, 5, '0')

# Lambda construction at the start of the reverse pass.
# - `final_adjoint_seed = nothing` (default): zero-allocate panels
#   shaped like `final_m` and seed via the objective. Used by
#   `cs_surface_emission_footprint` (objective-driven path).
# - `final_adjoint_seed::NTuple{6}` (A.3d from-seed stride): caller
#   supplies the final-time adjoint directly; the stride driver
#   copies it instead of running `_seed_objective!`. The objective
#   field is still threaded through for `CSFootprintResult` metadata,
#   but `_seed_objective!` / `_validate_objective` are bypassed so
#   `CSSeedObjective()` works without tripping its own validation
#   reject (`ObjectiveSeeding.jl:75-79`).
function _build_stride_lambda_panels(final_m, ::Nothing,
                                     objective, mesh, ::Type{FT}) where {FT}
    lambda_panels = ntuple(p -> begin
        a = similar(final_m[p])
        fill!(a, zero(FT))
        a
    end, 6)
    _seed_objective!(lambda_panels, objective, final_m, mesh)
    return lambda_panels
end

function _build_stride_lambda_panels(_final_m, final_adjoint_seed::NTuple{6},
                                     _objective, _mesh, ::Type{FT}) where {FT}
    return _copy_panel_tuple(final_adjoint_seed)
end

"""
    _propagate_mass_checkpoints(panels_m0, panels_am_steps, panels_bm_steps,
                                panels_cm_steps, mesh, scheme, schedule;
                                cfl_limit, flux_scale, dt,
                                diffusion_op, diffusion_workspace,
                                convection_op, convection_forcing)

Forward pass that emits the per-window left-edge checkpoints of
`panels_m`. Returns `Vector{NTuple{6, A}}` of length
`checkpoint_window_count(schedule, nsteps) + 1`: index `w` is
`panels_m` at the start of window `w`, and the final index is the
post-run state.

Implementation note: the forward propagation is driven by
`_record_cs_mass_tape` with `record_ops = false` so the recorder
runs the same `_sweep_x/y/z_panel!` + `fill_panel_halos!` kernels as
the FullCheckpoint path but skips every `_record_sweep!` /
`_stage_panels` / `push!(ops, ...)` site. This bounds peak
propagation memory to the live `panels_m` tuple (plus the
`dummy_rm` workspace inside the recorder), independent of `nsteps`,
while keeping the forward kernel path bit-identical to the
recording path.
"""
function _propagate_mass_checkpoints(panels_m0,
                                     panels_am_steps,
                                     panels_bm_steps,
                                     panels_cm_steps,
                                     mesh::CubedSphereMesh,
                                     scheme::CSAdjointLinearScheme,
                                     schedule::StrideCheckpoint;
                                     cfl_limit,
                                     flux_scale,
                                     dt,
                                     diffusion_op = NoDiffusion(),
                                     diffusion_workspace = nothing,
                                     convection_op = NoConvection(),
                                     convection_forcing = nothing)
    nsteps = length(panels_am_steps)
    nw = checkpoint_window_count(schedule, nsteps)

    initial = _copy_panel_tuple(panels_m0)
    fill_panel_halos!(initial, mesh; dir = 0)

    checkpoints = Vector{typeof(initial)}(undef, nw + 1)
    checkpoints[1] = initial

    current = initial
    @inbounds for w in 1:nw
        window_range = checkpoint_window_range(schedule, w, nsteps)
        _, current = _record_cs_mass_tape(
            current,
            _slice_step_kwarg(panels_am_steps, window_range),
            _slice_step_kwarg(panels_bm_steps, window_range),
            _slice_step_kwarg(panels_cm_steps, window_range),
            mesh, scheme;
            cfl_limit = cfl_limit,
            flux_scale = flux_scale,
            dt = dt,
            diffusion_op = _slice_step_kwarg(diffusion_op, window_range),
            diffusion_workspace = _slice_step_kwarg(diffusion_workspace, window_range),
            convection_op = convection_op,
            convection_forcing = _slice_step_kwarg(convection_forcing, window_range),
            tape_storage = :device,
            step_offset = first(window_range) - 1,
            record_ops = false)
        checkpoints[w + 1] = current
    end
    return checkpoints
end

"""
    _collect_surface_footprints_stride(panels_m0, panels_am_steps, ...,
                                       mesh, scheme, schedule, objective, dt;
                                       cfl_limit, flux_scale,
                                       diffusion_op, diffusion_workspace,
                                       diffusion_meteo,
                                       convection_op, convection_forcing,
                                       convection_workspace,
                                       tape_storage)

Strided-checkpoint variant of `_collect_surface_footprints`. Returns a
`CSFootprintResult` whose `footprints[step]` array agrees with the
`FullCheckpoint` path to bit accuracy on the linear-scheme paths
(`UpwindScheme`, `SlopesScheme(NoLimiter())`, `PPMScheme(NoLimiter())`),
since both routes call the same forward kernels in the same order
and the reverse pass adjoint is exact (not stochastic).

Each window's tape is built with the requested `tape_storage` policy
and dropped after its reverse walk completes. For `tape_storage = :mmap`,
this means peak disk is one window's worth of `records.bin`, not the
whole run's.
"""
function _collect_surface_footprints_stride(panels_m0,
                                            panels_am_steps,
                                            panels_bm_steps,
                                            panels_cm_steps,
                                            mesh::CubedSphereMesh,
                                            scheme::CSAdjointLinearScheme,
                                            schedule::StrideCheckpoint,
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
    # Objective validation is bypassed in the from-seed path because
    # `_validate_objective(::CSSeedObjective)` throws unconditionally
    # (`ObjectiveSeeding.jl:75`); from-seed callers supply the lambda
    # directly and only need the objective field for `CSFootprintResult`
    # metadata.
    final_adjoint_seed === nothing && _validate_objective(objective, mesh, Nz)
    _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace, nsteps)
    _require_cs_convection_workspace(convection_op, convection_workspace)
    # The stride driver constructs a fresh tape storage per window
    # via `_tape_storage(tape_storage)` and finalize_tape!s it after
    # each window's reverse walk. If the caller passes an
    # *already-constructed* `AbstractCSTapeStorage`, `_tape_storage`
    # is identity — every window would share the same storage, and
    # the second window would throw "MmapCSTapeStorage ... is
    # finalised; cannot allocate new slot" because we finalize after
    # the first reverse walk. Reject the misuse up front with a
    # tape-aware diagnostic.
    tape_storage isa AbstractCSTapeStorage && throw(ArgumentError(
        "StrideCheckpoint requires `tape_storage` to be a Symbol " *
        "(:device, :pinned_host, or :mmap), not a pre-constructed " *
        "$(typeof(tape_storage)); the stride driver builds and " *
        "finalize_tape!s one storage instance per window. Pass " *
        "`tape_storage = :mmap` (or similar) instead."))
    tape_path !== nothing && tape_storage !== :mmap && throw(ArgumentError(
        "tape_path requires tape_storage = :mmap; got " *
        "tape_storage = $(repr(tape_storage))"))

    checkpoints = _propagate_mass_checkpoints(
        panels_m0, panels_am_steps, panels_bm_steps, panels_cm_steps,
        mesh, scheme, schedule;
        cfl_limit = cfl_limit, flux_scale = flux_scale, dt = dt,
        diffusion_op = diffusion_op,
        diffusion_workspace = diffusion_workspace,
        convection_op = convection_op,
        convection_forcing = convection_forcing)

    nw = checkpoint_window_count(schedule, nsteps)
    final_m = checkpoints[nw + 1]

    lambda_panels = _build_stride_lambda_panels(final_m, final_adjoint_seed,
                                                objective, mesh, FT)

    footprints = [_zero_surface_rates(mesh, panels_m0[1]) for _ in 1:nsteps]
    ws = CSAdjointWorkspace(mesh, lambda_panels[1])

    @inbounds for w in nw:-1:1
        window_range = checkpoint_window_range(schedule, w, nsteps)
        # Construct the per-window tape storage explicitly so we can
        # finalize it deterministically after the reverse walk. For
        # `:mmap`, this is load-bearing: each window opens its own
        # temp dir + `records.bin`, and leaning on GC could leave
        # multiple per-window dirs live simultaneously on tmpfs.
        # `finalize_tape!` is a generic no-op for `:device` /
        # `:pinned_host` so the call site is uniform. When `tape_path`
        # is supplied, `_build_window_storage` roots each window's
        # `records.bin` under `joinpath(tape_path, "window_NNNNN")`
        # and leaves the directory in place past `finalize_tape!`
        # (user-owned).
        storage_w = _build_window_storage(tape_storage, tape_path,
                                          _stride_window_subdir(w))
        ops_window, _ = _record_cs_mass_tape(
            checkpoints[w],
            _slice_step_kwarg(panels_am_steps, window_range),
            _slice_step_kwarg(panels_bm_steps, window_range),
            _slice_step_kwarg(panels_cm_steps, window_range),
            mesh, scheme;
            cfl_limit = cfl_limit,
            flux_scale = flux_scale,
            dt = dt,
            diffusion_op = _slice_step_kwarg(diffusion_op, window_range),
            diffusion_workspace = _slice_step_kwarg(diffusion_workspace, window_range),
            convection_op = convection_op,
            convection_forcing = _slice_step_kwarg(convection_forcing, window_range),
            tape_storage = storage_w,
            step_offset = first(window_range) - 1)
        try
            _walk_window_reverse!(footprints, lambda_panels, ops_window, mesh, ws, dt;
                                  diffusion_meteo = diffusion_meteo,
                                  convection_workspace = convection_workspace)
        finally
            # Drop the window-scoped tape (mmap close + manifest emit +
            # temp-dir cleanup for `MmapCSTapeStorage`; no-op for
            # `:device` / `:pinned_host` whose state is plain arrays
            # released by GC).
            finalize_tape!(storage_w; quiet = true,
                           strict = tape_path !== nothing)
        end
    end

    lag_steps = [nsteps - step for step in 1:nsteps]
    A2 = typeof(footprints[1][1])
    return CSFootprintResult{FT, typeof(objective), A2}(
        objective, footprints, lag_steps, FT(dt), zero(FT), FT(NaN))
end

# Schedule dispatcher used by `cs_surface_emission_footprint` and
# `cs_surface_emission_footprint_from_seed`. A.3a (linear-mass),
# A.3b (nonlinear-PPM tracer), and A.3c (LinRood horizontal) are all
# wired up. The `_from_seed` entry still rejects non-Full schedules
# pending its own follow-up commit (separate seed-driven driver).
function _require_checkpoint_supported(scheme, schedule::AbstractCheckpointSchedule)
    schedule isa FullCheckpoint && return nothing
    scheme isa CSAdjointLinearScheme && return nothing
    scheme isa CSAdjointNonlinearScheme && return nothing
    scheme isa CSAdjointLinRoodScheme && return nothing
    throw(ArgumentError(
        "checkpoint=$(schedule) is not yet supported for scheme " *
        "$(nameof(typeof(scheme)))."))
end

# Per-scheme `tape_path` compatibility. LinRood is pinned to
# `:device` storage (`_linrood_validate_tape_storage` inside
# `_record_cs_linrood_tape`), so a non-nothing `tape_path` is
# meaningless and would only produce empty `records.bin` /
# `manifest.toml` side effects under the user's tree before the
# recorder throws. Reject here, BEFORE `_resolve_tape_path` runs
# `mkpath`, so an invalid request leaves the filesystem untouched.
function _require_tape_path_supported(scheme,
                                       tape_path::Union{Nothing, AbstractString})
    tape_path === nothing && return nothing
    scheme isa CSAdjointLinRoodScheme && throw(ArgumentError(
        "tape_path is not supported with LinRoodPPMScheme; the LinRood " *
        "reverse tape is :device-only. Omit tape_path or pick a non-" *
        "LinRood scheme. (Mmap eviction for LinRood is reserved for a " *
        "Plan 26 follow-up that refactors the LinRood tape.)"))
    return nothing
end
