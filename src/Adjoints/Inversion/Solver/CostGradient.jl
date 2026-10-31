# ---------------------------------------------------------------------------
# 4D-Var cost and gradient evaluation.
#
# Contains:
#
#   * Control-window validation (`_validate_control_windows`) and
#     conversion (`_surface_rates_from_controls`).
#   * Gradient accumulation (`_add_window_gradient!`) that walks a
#     `CSFootprintResult` and sums per-step contributions back into the
#     control's window.
#   * Diagonal background term (`_background_cost_and_gradient!`) and the
#     four kernels it dispatches to (scalar vs array sigma, value
#     vs zero background).
#   * Gradient-descent step helpers (`_gradient_step_control`,
#     `_gradient_step_controls`, `_control_gradient_norm`,
#     `_gradient_step_panel_kernel!`).
#   * Vector flatten helpers (`_observation_vector`, `_control_vector`).
#   * `cs_surface_flux_4dvar` — the main cost/gradient entry point: one
#     forward run that samples the observations and records the tape, and
#     one reverse pass with every observation's weighted departure injected
#     at its sampled steps (`CSCostAndGradient`); the forward run alone for a
#     `CSCostOnly` evaluation.
# ---------------------------------------------------------------------------

function _validate_control_windows(controls, nsteps::Int)
    for control in controls
        for step in control.window.steps
            1 <= step <= nsteps || throw(ArgumentError(
                "surface-flux control $(control.window.name) references step $step, " *
                "but the run has $nsteps steps"))
        end
    end
    return nothing
end

# Cross-check that every panel array carried by a control block matches
# the mesh's `(Nc, Nc)` surface grid before any rate-assembly / gradient
# kernel runs. The kernels (`_add_weighted_footprint_kernel!`,
# `_add_background_gradient_*_kernel!`) read `arr[i, j]` under
# `@inbounds` with `i, j ∈ 1..Nc`, so a wrong-sized panel either
# silently ignores cells (too big) or reads OOB (too small). The
# `CSSurfaceFluxControl` constructor cross-validates `value` vs
# `background`/`sigma` shapes but does not see the mesh — this is
# the mesh-aware gate.
function _validate_control_shapes(controls, mesh::CubedSphereMesh,
                                  name::AbstractString)
    expected = (mesh.Nc, mesh.Nc)
    @inbounds for control in controls
        for p in 1:6
            size(control.value[p]) == expected || throw(DimensionMismatch(
                "$name $(control.window.name) value panel $p has shape " *
                "$(size(control.value[p])); expected $expected for mesh.Nc=$(mesh.Nc)"))
        end
        if control.background !== nothing
            for p in 1:6
                size(control.background[p]) == expected || throw(DimensionMismatch(
                    "$name $(control.window.name) background panel $p has shape " *
                    "$(size(control.background[p])); expected $expected for mesh.Nc=$(mesh.Nc)"))
            end
        end
        if control.sigma isa NTuple{6}
            for p in 1:6
                size(control.sigma[p]) == expected || throw(DimensionMismatch(
                    "$name $(control.window.name) sigma panel $p has shape " *
                    "$(size(control.sigma[p])); expected $expected for mesh.Nc=$(mesh.Nc)"))
            end
        end
    end
    return nothing
end

function _surface_rates_from_controls(controls, nsteps::Int,
                                      mesh::CubedSphereMesh, prototype)
    rates = [_zero_surface_rates(mesh, prototype) for _ in 1:nsteps]
    @inbounds for control in controls
        window = control.window
        for (idx, step) in enumerate(window.steps)
            weight = eltype(control.value[1])(window.weights[idx])
            for p in 1:6
                backend = get_backend(rates[step][p])
                kernel! = _add_weighted_footprint_kernel!(backend, (16, 16))
                kernel!(rates[step][p], control.value[p], weight;
                        ndrange = size(rates[step][p]))
            end
        end
    end
    # The launches above are queued in order; one wait covers them all.
    synchronize(get_backend(prototype))
    return rates
end

function _add_window_gradient!(gradient, result::CSFootprintResult,
                               window::CSSurfaceFluxWindow, scale;
                               ignore_future::Bool = false)
    nsteps = length(result.footprints)
    FT = eltype(gradient[1])
    @inbounds for (idx, step) in enumerate(window.steps)
        if !(1 <= step <= nsteps)
            if ignore_future && step > nsteps
                continue
            end
            throw(ArgumentError(
                "surface-flux window $(window.name) references step $step, " *
                "but the footprint has $nsteps steps"))
        end
        weight = FT(scale) * FT(window.weights[idx])
        for p in 1:6
            backend = get_backend(gradient[p])
            kernel! = _add_weighted_footprint_kernel!(backend, (16, 16))
            kernel!(gradient[p], result.footprints[step][p], weight;
                    ndrange = size(gradient[p]))
        end
    end
    synchronize(get_backend(gradient[1]))
    return nothing
end

@kernel function _add_background_gradient_scalar_kernel!(grad, @Const(value),
                                                        @Const(background),
                                                        invvar)
    i, j = @index(Global, NTuple)
    @inbounds grad[i, j] += (value[i, j] - background[i, j]) * invvar
end

@kernel function _add_background_gradient_zero_scalar_kernel!(grad,
                                                             @Const(value),
                                                             invvar)
    i, j = @index(Global, NTuple)
    @inbounds grad[i, j] += value[i, j] * invvar
end

@kernel function _add_background_gradient_array_kernel!(grad, @Const(value),
                                                       @Const(background),
                                                       @Const(sigma))
    i, j = @index(Global, NTuple)
    @inbounds begin
        s = sigma[i, j]
        grad[i, j] += (value[i, j] - background[i, j]) / (s * s)
    end
end

@kernel function _add_background_gradient_zero_array_kernel!(grad,
                                                            @Const(value),
                                                            @Const(sigma))
    i, j = @index(Global, NTuple)
    @inbounds begin
        s = sigma[i, j]
        grad[i, j] += value[i, j] / (s * s)
    end
end

function _background_cost_and_gradient!(gradient, control::CSSurfaceFluxControl)
    control.sigma === nothing && return zero(eltype(control.value[1]))
    FT = eltype(control.value[1])
    cost = zero(FT)
    @inbounds for p in 1:6
        value = control.value[p]
        backend = get_backend(value)
        if control.sigma isa Real
            sigma = FT(control.sigma)
            invvar = inv(sigma * sigma)
            if control.background === nothing
                cost += FT(0.5) * invvar * sum(abs2, value)
                kernel! = _add_background_gradient_zero_scalar_kernel!(backend, (16, 16))
                kernel!(gradient[p], value, invvar; ndrange = size(value))
            else
                background = control.background[p]
                cost += FT(0.5) * invvar * sum(abs2, value .- background)
                kernel! = _add_background_gradient_scalar_kernel!(backend, (16, 16))
                kernel!(gradient[p], value, background, invvar; ndrange = size(value))
            end
        else
            sigma = control.sigma[p]
            if control.background === nothing
                cost += FT(0.5) * sum(abs2, value ./ sigma)
                kernel! = _add_background_gradient_zero_array_kernel!(backend, (16, 16))
                kernel!(gradient[p], value, sigma; ndrange = size(value))
            else
                background = control.background[p]
                cost += FT(0.5) * sum(abs2, (value .- background) ./ sigma)
                kernel! = _add_background_gradient_array_kernel!(backend, (16, 16))
                kernel!(gradient[p], value, background, sigma; ndrange = size(value))
            end
        end
    end
    synchronize(get_backend(gradient[1]))
    return cost
end

@kernel function _gradient_step_panel_kernel!(dst, @Const(value), @Const(gradient),
                                              step)
    i, j = @index(Global, NTuple)
    @inbounds dst[i, j] = value[i, j] - step * gradient[i, j]
end

function _gradient_step_control(control::CSSurfaceFluxControl, gradient, step)
    FT = eltype(control.value[1])
    stepped = ntuple(p -> begin
        dst = similar(control.value[p])
        backend = get_backend(dst)
        kernel! = _gradient_step_panel_kernel!(backend, (16, 16))
        kernel!(dst, control.value[p], gradient[p], FT(step);
                ndrange = size(dst))
        dst
    end, 6)
    synchronize(get_backend(stepped[1]))
    return CSSurfaceFluxControl(control.window, stepped;
                                background = control.background,
                                sigma = control.sigma)
end

function _gradient_step_controls(controls, gradients, step)
    length(controls) == length(gradients) || throw(DimensionMismatch(
        "controls length $(length(controls)) does not match gradients length $(length(gradients))"))
    # Untyped comprehension so the result element type is inferred from
    # `_gradient_step_control`'s concrete return — previously
    # `Any[...]`, which forced runtime dispatch on every `cost_fn(...)`
    # call inside the optimizer's line-search loop.
    return [_gradient_step_control(controls[i], gradients[i], step)
            for i in eachindex(controls)]
end

function _control_gradient_norm(gradients)
    isempty(gradients) && return 0.0
    FT = eltype(gradients[1][1])
    total = zero(FT)
    @inbounds for grad in gradients, p in 1:6
        total += sum(abs2, grad[p])
    end
    return sqrt(total)
end

function _observation_vector(obs::CSObservation)
    return CSObservation[obs]
end
function _observation_vector(observations)
    return CSObservation[observations...]
end

function _control_vector(control::CSSurfaceFluxControl)
    return CSSurfaceFluxControl[control]
end
function _control_vector(controls)
    return CSSurfaceFluxControl[controls...]
end

# Preconditioner-list normalization (mirrors `_control_vector`).
# Accepts either a single `CSSurfaceFluxPreconditioner` (returned as a
# length-1 vector, later broadcast to every control by
# `_align_preconditioners_to_controls`) or a `Vector` of them aligned
# 1-to-1 with the control vector.
_preconditioner_vector(prec::CSSurfaceFluxPreconditioner) =
    CSSurfaceFluxPreconditioner[prec]
_preconditioner_vector(precs::AbstractVector{<:CSSurfaceFluxPreconditioner}) =
    CSSurfaceFluxPreconditioner[precs...]

# Broadcast a scalar (length-1) preconditioner vector across every
# control, or pass through an already-aligned vector. The two cases
# the caller wants are:
#   * 1 preconditioner, N controls → use that preconditioner for all.
#   * N preconditioners, N controls → use as-is.
# Any other combination is an alignment error.
function _align_preconditioners_to_controls(preconditioner_vec, n_controls::Int)
    n_prec = length(preconditioner_vec)
    if n_prec == n_controls
        return preconditioner_vec
    elseif n_prec == 1
        return fill(preconditioner_vec[1], n_controls)
    else
        throw(ArgumentError(
            "preconditioner length $n_prec does not match controls " *
            "length $n_controls (only a single preconditioner or one " *
            "preconditioner per control is accepted)"))
    end
end

# Build a fresh `CSSurfaceFluxControl` whose `.value` is `T(χ)` —
# the physical-space image of the χ-space `chi_control.value`. The
# downstream forward run + observation-gradient kernels are agnostic
# to whether their input came from the user directly (unconditioned
# mode) or from a preconditioner forward (preconditioned mode); this
# helper produces input of the right shape for the existing rate-
# assembly path. `.background` and `.sigma` are not carried through
# because the preconditioned background term comes from `0.5 ‖χ‖²`
# instead of the per-control diagonal kernel.
function _preconditioned_physical_control(chi_control::CSSurfaceFluxControl,
                                          prec::CSSurfaceFluxPreconditioner)
    chi_value = chi_control.value
    x_value = ntuple(p -> similar(chi_value[p]), 6)
    apply_preconditioner!(x_value, prec, chi_value)
    return CSSurfaceFluxControl(chi_control.window, x_value)
end

# 0.5 · ‖χ‖² in panel-tuple form. The χ-space regularization term
# replaces the per-control diagonal background when running
# preconditioned.
function _half_chi_squared_norm(chi_controls)
    isempty(chi_controls) && return 0.0
    FT = eltype(chi_controls[1].value[1])
    total = zero(FT)
    @inbounds for control in chi_controls, p in 1:6
        total += sum(abs2, control.value[p])
    end
    return FT(0.5) * total
end

# Convert a physical-space gradient `∂J_obs/∂x` to a χ-space gradient
# `∂J/∂χ = T'(χ)^T ∂J_obs/∂x + χ`, in-place on `chi_gradient`. The
# `+ χ` term comes from differentiating the `0.5 ‖χ‖²` regularization.
function _physical_gradient_to_chi!(chi_gradient,
                                    prec::CSSurfaceFluxPreconditioner,
                                    chi_control::CSSurfaceFluxControl,
                                    physical_control::CSSurfaceFluxControl,
                                    physical_gradient)
    apply_preconditioner_adjoint!(chi_gradient, prec,
                                  physical_control.value, physical_gradient)
    chi_value = chi_control.value
    @inbounds for p in 1:6
        @. chi_gradient[p] += chi_value[p]
    end
    return chi_gradient
end

# ---------------------------------------------------------------------------
# The observation forward run and reverse pass of one evaluation + 4D-Var entry
# ---------------------------------------------------------------------------

# The forward run of an evaluation (a checkpoint driver's run on the
# observation replay). A cost-only evaluation keeps nothing. A gradient tapes
# through the last step of an observation with `assimilate = true`: the
# surface-pressure screen, applied after the run, only removes observations,
# so this bounds the reverse pass, and co-samples after it do not extend the
# tape. Without such an observation there is no reverse pass to tape for.
_observation_run(::CSCostOnly, _checkpoint, _replay, _tape_storage, _observations) = _UntapedRun()
function _observation_run(::CSCostAndGradient, checkpoint, replay, tape_storage, observations)
    taped_steps = maximum(obs -> obs.assimilate ? obs.step : 0, observations)
    taped_steps == 0 && return _UntapedRun()
    return _begin_taped_run(checkpoint, replay, tape_storage, taped_steps)
end

# Only a gradient needs the observations' linear functionals.
_builds_functionals(::CSCostAndGradient) = true
_builds_functionals(::CSCostOnly) = false

# ∂J_obs/∂x of each control into `gradients`: one reverse pass of the forward
# run, from a zero adjoint state through the last assimilated step, adding each
# step's forcing at the step's end. Footprint `t` is then ∂J_obs/∂E_t for every
# step through that step; each control's window collects its steps (window
# steps after it do not affect the cost, `ignore_future`). Without any
# assimilated observation the observation gradient is zero.
_add_observation_gradients!(_gradients, ::CSCostOnly, _run, _replay, _dt, _controls, _departures) = nothing
function _add_observation_gradients!(gradients, ::CSCostAndGradient, run, replay, dt, controls, departures)
    (; observations, functionals, scales, active, mesh, prototype) = departures
    isempty(active) && return nothing
    FT = eltype(prototype)
    forcing = _assemble_step_forcing(functionals[active], scales[active], mesh, size(prototype, 3), prototype)
    footprint = _forced_reverse(run, replay, dt; forcing,
                                nsteps = maximum(idx -> observations[idx].step, active))
    @inbounds for idx in eachindex(controls)
        _add_window_gradient!(gradients[idx], footprint, controls[idx].window, one(FT);
                              ignore_future = true)
    end
    return nothing
end

# The χ-space gradient ∂J/∂χ = T'(χ)ᵀ ∂J_obs/∂x + χ of each control
# (preconditioned mode); none for a cost-only evaluation.
_chi_gradients(::CSCostOnly, _preconditioners, _controls, _physical_controls, _physical_gradients) = nothing
function _chi_gradients(::CSCostAndGradient, preconditioners, controls, physical_controls,
                        physical_gradients)
    gradients = [zeros_like(control.value) for control in controls]
    @inbounds for idx in eachindex(controls)
        _physical_gradient_to_chi!(gradients[idx], preconditioners[idx], controls[idx],
                                   physical_controls[idx], physical_gradients[idx])
    end
    return gradients
end

# The reported gradients: every control's, or none for a cost-only evaluation.
_reported_gradients(::CSCostAndGradient, gradients, _template) = gradients
_reported_gradients(::CSCostOnly, _gradients, template) = typeof(template)[]

# Count the observations left out of the cost by the surface-pressure screen of
# averaging-kernel columns (`SkipUnplaced`); co-samples (`assimilate = false`)
# are counted when they are bound. The screened set depends on the meteorology
# only, so it is the same in every evaluation of an inversion; the message is
# logged once per set (keyed by the record ids).
function _log_unassimilated(observations, stencils)
    ids = Int64[observations[i].id for i in eachindex(stencils)
                if observations[i].assimilate && !stencils[i].assimilated]
    isempty(ids) && return nothing
    @info "cs_surface_flux_4dvar: $(length(ids)) averaging-kernel column observations are not assimilated: " *
          "the model surface pressure differs from the retrieval's by more than " *
          "max_surface_pressure_mismatch_pa (unplaced_records = \"skip\"); their stencils are marked " *
          "assimilated = false" _id = (:retrieval_surface_pressure_screen, hash(ids)) maxlog = 1
    return nothing
end

"""
    cs_surface_flux_4dvar(..., observations, controls; scheme, kwargs...) -> CS4DVarResult

Evaluate the prototype CS surface-flux 4D-Var cost and gradient. Controls are
named `CSSurfaceFluxControl`s over `CSSurfaceFluxWindow`s. Observations are
scalar `CSObservation`s sampled after the model steps of their
`CSTimeSampling` (one step, or a weighted time window; after convection, the
state runtime output sees). `CSPlacedPointObjective` observations choose their
layer, and `CSRetrievalColumnObjective` observations (averaging-kernel
columns) build their weights, at each sampled state from `column_met`, a
`CSObservationColumnMet` (required when there are any). An averaging-kernel
column whose surface-pressure mismatch exceeds its limit is an error, or,
under `SkipUnplaced`, is reported with its simulated value and residual but
left out of the cost and the gradient (its stencil is not `assimilated`).
Co-samples (`CSObservation(...; assimilate = false)`: flagged records bound
with `flagged_records = "cosample"`, the members of cell averages) are
simulated and reported the same way, never in the cost or the gradient, and
do not extend the taped run: adding them leaves the cost and the reverse
pass's forcing unchanged, bit for bit.

One cost-and-gradient evaluation is one forward run and one reverse pass
(with `FullCheckpoint`; the other schedules add their recomputation, and a
`MmapCSTapeStorage` adds the counting run that sizes its file), whatever the
number of observations, as in TM5-4DVAR:

1. the forward run gives the simulated values `Hᵢ(x) = Σₛ ωᵢₛ hᵢ(rₛ)`, each
   observation's stencil (`CSObservationStencil`: steps, weights, levels) and
   its linear functional (weights from the replayed air mass at each sampled
   step, times `ωᵢₛ`), and records the tape of its `checkpoint` schedule
   through the last step of an observation with `assimilate = true` (the
   whole tape with `FullCheckpoint`, the K-step checkpoints with
   `StrideCheckpoint(K)`);
2. each observation contributes `(Hᵢ(x) − yᵢ)/σᵢ²` times its functional's
   weights to the adjoint forcing of each sampled step;
3. one reverse pass from the last assimilated step (after the
   surface-pressure screen) back to the first, which adds each step's forcing
   at that step's end, gives `∂J_obs/∂E` for every step; each control's
   window collects its steps.

The run replays the production step plan of `schedule` (a
[`CSStepSchedule`](@ref AtmosTransport.Adjoints.Footprint.CSStepSchedule);
default: one step of `dt` per met window, the full
operator suite every step), with the controls' rates in its surface-flux
operator. Steps after the last assimilated step do not affect the cost and
get a zero gradient. `checkpoint` (`FullCheckpoint()`, `StrideCheckpoint(k)`,
`RevolveCheckpoint()`) and `tape_storage` select how the tape is stored, as in
`cs_surface_emission_footprint`.

An averaging-kernel column the surface-pressure screen leaves out after the
last assimilated step is taped but not walked: the tape (and the reservation
of a `MmapCSTapeStorage`) covers the steps through the last observation with
`assimilate = true`.

`evaluation = CSCostOnly()` computes the cost and the simulated values only:
the forward run without a tape and no reverse pass (the result's `gradients`
are empty; `checkpoint` and `tape_storage` are validated but not used); the
default `CSCostAndGradient()` computes both.

`scheme` is required: inversion callers must name the advection operator whose
forward and adjoint they intend to use; there is no hidden numerical default.

Two modes:

- **Unconditioned** (default, `preconditioner = nothing`). Each control's
  `.value` is the physical-space control directly. Optional diagonal
  background terms come from `.background` + `.sigma` and are added per
  control via `_background_cost_and_gradient!`.

- **Preconditioned** (`preconditioner` non-`nothing`). Each control's
  `.value` is the preconditioned-space variable `χ`. The function:
  (1) computes `x_k = T(χ_k)` per control via the preconditioner;
  (2) runs the forward simulation with `x`;
  (3) reverses through the existing observation-gradient path to get
      `∂J_obs/∂x_k` per control;
  (4) applies `T'(χ_k)^T` to get `∂J_obs/∂χ_k` and adds `χ_k` (the
      derivative of the `0.5 ‖χ‖²` regularization).
  The reported cost is `0.5 ‖χ‖² + observation_cost`, the reported
  gradient is `∂J/∂χ`, and the reported controls are the original
  `χ`-space inputs. Per-control `.background` and `.sigma` are
  ignored in this mode — the background term comes from `0.5 ‖χ‖²`,
  and the background `x_b` is carried by the preconditioner itself.

`preconditioner` accepts either a single `CSSurfaceFluxPreconditioner`
(broadcast to every control — the same preconditioner is used for
each of them) or a `Vector{<:CSSurfaceFluxPreconditioner}` aligned
1-to-1 with the control vector. Any other length mismatch throws
`ArgumentError`.
"""
function cs_surface_flux_4dvar(panels_rm0, panels_m0,
                               panels_am_steps,
                               panels_bm_steps,
                               panels_cm_steps,
                               mesh::CubedSphereMesh,
                               observations,
                               controls;
                               scheme::CSAdjointSupportedScheme,
                               dt = nothing,
                               schedule = nothing,
                               flux_scale = one(eltype(panels_rm0[1])),
                               cfl_limit = DEFAULT_CFL_LIMIT,
                               diffusion_op = NoDiffusion(),
                               diffusion_workspace = nothing,
                               diffusion_meteo = nothing,
                               convection_op = NoConvection(),
                               convection_forcing = nothing,
                               convection_workspace = nothing,
                               tape_storage = :device,
                               checkpoint::AbstractCheckpointSchedule = FullCheckpoint(),
                               evaluation::AbstractCS4DVarEvaluation = CSCostAndGradient(),
                               preconditioner = nothing,
                               column_met = nothing)
    FT = eltype(panels_rm0[1])
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps, panels_cm_steps)
    observation_vec = _observation_vector(observations)
    control_vec = _control_vector(controls)
    isempty(observation_vec) && throw(ArgumentError("at least one CSObservation is required"))
    isempty(control_vec) && throw(ArgumentError("at least one CSSurfaceFluxControl is required"))
    _validate_control_windows(control_vec, nsteps)
    _validate_control_shapes(control_vec, mesh, "surface-flux control")
    _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace, nsteps)
    _require_cs_convection_workspace(convection_op, convection_workspace)
    # Storage checks before the forward run, not after it.
    _require_tape_storage_supported(scheme, tape_storage)

    seen = Set{Symbol}()
    for control in control_vec
        name = control.window.name
        name in seen && throw(ArgumentError("duplicate surface-flux control name $name"))
        push!(seen, name)
    end

    # Preconditioned mode normalization. `chi_controls` holds the user's
    # χ-space inputs (returned in the result); `physical_controls` is
    # what feeds the rate-assembly + observation-gradient path.
    preconditioner_vec = preconditioner === nothing ?
        nothing : _preconditioner_vector(preconditioner)
    if preconditioner_vec !== nothing
        preconditioner_vec = _align_preconditioners_to_controls(
            preconditioner_vec, length(control_vec))
        physical_controls = [
            _preconditioned_physical_control(control_vec[idx],
                                             preconditioner_vec[idx])
            for idx in eachindex(control_vec)
        ]
        _validate_control_shapes(physical_controls, mesh,
                                 "preconditioned physical control")
    else
        physical_controls = control_vec
    end

    emission_rates = _surface_rates_from_controls(physical_controls, nsteps, mesh, panels_rm0[1])
    # (1) One forward run: the simulated values and each observation's linear
    # functional, and, for a gradient, the tape of its reverse pass (recorded
    # while the observations are sampled).
    replay, step_dt = _observation_replay(
        panels_rm0, panels_m0, panels_am_steps, panels_bm_steps, panels_cm_steps,
        mesh, observation_vec;
        scheme, dt, schedule,
        flux_scale = FT(flux_scale),
        cfl_limit, emission_rates,
        diffusion_op, diffusion_workspace, diffusion_meteo,
        convection_op, convection_forcing, convection_workspace,
        column_met)
    physical_gradients = [zeros_like(control.value) for control in physical_controls]
    run = _observation_run(evaluation, checkpoint, replay, tape_storage, observation_vec)
    observed = try
        sampled = _observe_cs_forward!(run, replay, mesh, observation_vec; column_met,
                                       build_functionals = _builds_functionals(evaluation))

        # (2) Weighted departures (H(x) − y)/σ² scale each functional's
        # weights. An observation that is not assimilated (a co-sample, or an
        # averaging-kernel column screened out by its surface-pressure
        # mismatch) keeps its residual but adds nothing to the cost or the
        # forcing.
        residuals = Vector{FT}(undef, length(observation_vec))
        scales = Vector{FT}(undef, length(observation_vec))
        observation_cost = zero(FT)
        @inbounds for obs_idx in eachindex(observation_vec)
            obs = observation_vec[obs_idx]
            residual = sampled.simulated[obs_idx] - FT(obs.value)
            sigma = FT(obs.sigma)
            residuals[obs_idx] = residual
            sampled.stencils[obs_idx].assimilated || (scales[obs_idx] = zero(FT); continue)
            observation_cost += FT(0.5) * (residual / sigma)^2
            scales[obs_idx] = residual / (sigma * sigma)
        end
        active = findall(s -> s.assimilated, sampled.stencils)
        _log_unassimilated(observation_vec, sampled.stencils)

        # (3) The reverse pass of the forward run. In preconditioned mode this
        # is `∂J_obs/∂x` per control; in unconditioned mode it doubles as the
        # final reported gradient `∂J/∂x`.
        _add_observation_gradients!(physical_gradients, evaluation, run, replay, step_dt, physical_controls,
                                    (; observations = observation_vec, sampled.functionals, scales, active,
                                     mesh, prototype = panels_rm0[1]))
        (; sampled.simulated, residuals, observation_cost, sampled.stencils)
    finally
        _end_taped_run!(run)
    end

    # Either: (a) apply T'^T to physical gradient and add χ for the
    # preconditioned background term, or (b) use the existing
    # per-control diagonal background.
    if preconditioner_vec === nothing
        gradients = physical_gradients
        background_cost = zero(FT)
        @inbounds for idx in eachindex(control_vec)
            background_cost += FT(_background_cost_and_gradient!(gradients[idx],
                                                                  control_vec[idx]))
        end
    else
        gradients = _chi_gradients(evaluation, preconditioner_vec, control_vec, physical_controls,
                                   physical_gradients)
        background_cost = FT(_half_chi_squared_norm(control_vec))
    end
    reported_controls = control_vec
    gradients = _reported_gradients(evaluation, gradients, physical_gradients[1])

    gradient_by_name = Dict{Symbol, eltype(gradients)}()
    @inbounds for idx in eachindex(gradients)
        gradient_by_name[reported_controls[idx].window.name] = gradients[idx]
    end
    A2 = typeof(physical_gradients[1][1])
    return CS4DVarResult{FT, A2}(
        observed.observation_cost + background_cost,
        observed.observation_cost,
        background_cost,
        observed.simulated,
        observed.residuals,
        gradients,
        gradient_by_name,
        reported_controls,
        observation_vec,
        observed.stencils)
end
