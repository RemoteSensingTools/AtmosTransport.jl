# ---------------------------------------------------------------------------
# Reverse walk + forward replays.
#
# `_walk_window_reverse!` walks a recorded tape backwards, dispatching on each
# record type to the per-physics adjoint kernel (advection, halo, diffusion,
# convection, air-mass reset, Lin-Rood horizontal), injecting the step forcing
# at end-of-step markers and accumulating per-step surface footprints at the
# emission records. `_initial_adjoint_state` builds the adjoint state the walk
# starts from (`_ObjectiveStart`, `_SeedStart`, `_ZeroStart`).
#
# `_run_cs_footprint_forward` replays the production step plan without a tape
# to evaluate a final-time objective. Observation sampling lives in
# Inversion.ObservationBinding/ForwardSampling.jl.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The adjoint state at the end of the taped run, before the reverse walk:
#
#   _ObjectiveStart(objective) — ∂J/∂rm_final of a final-time objective
#                                (`_seed_objective!`);
#   _SeedStart(seed)           — an explicit `dJ/drm_final` (the from-seed
#                                entry point);
#   _ZeroStart()               — zero; every sensitivity enters through the
#                                step forcing (the single-sweep 4D-Var
#                                gradient).
#
# The Full and Stride drivers build the starting state with
# `_initial_adjoint_state` from the final air mass of their forward pass.
# ---------------------------------------------------------------------------

struct _ObjectiveStart{O <: AbstractCSFootprintObjective}
    objective::O
end

struct _SeedStart{S <: NTuple{6}}
    seed::S
end

struct _ZeroStart end

_validate_start(start::_ObjectiveStart, mesh::CubedSphereMesh, Nz::Int) =
    _validate_objective(start.objective, mesh, Nz)
_validate_start(::_ObjectiveStart{<:CSPlacedPointObjective}, ::CubedSphereMesh, ::Int) =
    _placed_objective_needs_met()
_validate_start(::_ObjectiveStart{<:CSRetrievalColumnObjective}, ::CubedSphereMesh, ::Int) =
    _retrieval_objective_needs_met()
_validate_start(::_ObjectiveStart{<:CSCellAverageObjective}, ::CubedSphereMesh, ::Int) =
    _cell_average_needs_4dvar()
_validate_start(::Union{_SeedStart, _ZeroStart}, ::CubedSphereMesh, ::Int) = nothing

# The objective recorded in the `CSFootprintResult`.
_result_objective(start::_ObjectiveStart) = start.objective
_result_objective(::Union{_SeedStart, _ZeroStart}) = CSSeedObjective()

function _initial_adjoint_state(start::_ObjectiveStart, final_m, mesh::CubedSphereMesh)
    lambda_panels = zeros_like(final_m)
    _seed_objective!(lambda_panels, start.objective, final_m, mesh)
    return lambda_panels
end

# Only the interior of the final state is prognostic: the packed palindrome
# returns stale halos (`strang_split_cs_mt_pingpong!`), so the halo entries of
# a seed are ignored.
function _initial_adjoint_state(start::_SeedStart, _final_m, mesh::CubedSphereMesh)
    lambda_panels = _copy_panel_tuple(start.seed)
    Nc, Hp = mesh.Nc, mesh.Hp
    low, inner, high = 1:Hp, Hp+1:Hp+Nc, Hp+Nc+1:Nc+2Hp
    for a in lambda_panels, (I, J) in ((low, :), (high, :), (inner, low), (inner, high))
        view(a, I, J, :) .= zero(eltype(a))
    end
    return lambda_panels
end

_initial_adjoint_state(::_ZeroStart, final_m, ::CubedSphereMesh) =
    zeros_like(final_m)

# Walk `ops` backwards from `lambda_panels` (mutated in place; chain calls in
# window-reverse order so the adjoint state flows from final to initial across
# windows). `footprints[step]` must exist for every step that has an emission
# record in `ops`.
#
# `forcing` is added at each `_CSStepEndRecord(step)`. Writing `λ_s` for the
# adjoint state at the end of step `s` (after its window-end stages) and `M_s`
# for the tangent-linear step operator, the walk computes
#
#     λ_s = M_{s+1}ᵀ λ_{s+1} + f_s,     f_s = Σ_{i: sᵢ = s} (Hᵢ(x) − yᵢ)/σᵢ² · ∂hᵢ/∂rm_s,
#
# from the starting state (zero for 4D-Var) at the last step back to the
# first. At each step's emission record it reads `τ · λ` (the adjoint state
# where the step's emissions enter, partway through transposing `M_s`; `τ` the
# emission interval) as the emission footprint. One walk thus gives the
# gradient of every observation at once. `NoStepForcing` (the default) adds
# nothing.
function _walk_window_reverse!(footprints, lambda_panels, ops,
                               mesh::CubedSphereMesh, ws;
                               convection_workspace = nothing,
                               forcing::AbstractStepForcing = NoStepForcing())
    for op in Iterators.reverse(ops)
        if op isa _CSStepEndRecord
            _inject_step_forcing!(lambda_panels, forcing, op.step, mesh)
        elseif op isa _CSSweepRecord
            _reverse_sweep!(lambda_panels, op, mesh, ws)
        elseif op isa _CSHaloRecord
            _adjoint_fill_panel_halos!(lambda_panels, mesh; dir=op.dir)
        elseif op isa _CSMidpointRecord
            _accumulate_surface_footprint!(footprints[op.step], lambda_panels, op.dt, mesh)
        elseif op isa _CSDiffusionRecord
            _apply_cs_diffusion_adjoint!(lambda_panels, _tape_panels(op.panels_m, AirMassRead()), op.op,
                                         op.workspace, op.dt,
                                         _RecordedClock(op.time), mesh)
        elseif op isa _CSConvectionRecord
            _apply_cs_convection_adjoint!(lambda_panels, _tape_panels(op.panels_m, AirMassRead()),
                                          op.forcing, op.op, op.dt,
                                          convection_workspace, mesh; n_sub = op.substeps)
        elseif op isa _CSAirMassResetRecord
            _apply_air_mass_reset_adjoint!(lambda_panels, op)
        elseif op isa _CSLinRoodHorizRecord
            # LinRood horizontal substep reverse. `lambda_panels` holds the
            # tracer-mass adjoint. Meteorology (the air-mass evolution) is a
            # fixed tape input: the mass-flux Jacobians are read off the
            # record's `panels_m` / `panels_am` / `panels_bm`, so the substep's
            # internal air-mass adjoint starts from zero and is discarded, as
            # in the split-sweep reverse, which has no air-mass adjoint at all.
            lambda_m_panels = zeros_like(lambda_panels)
            _apply_cs_linrood_horizontal_adjoint!(lambda_panels,
                                                  lambda_m_panels, op, mesh)
        else
            throw(ArgumentError("unknown CS adjoint tape operation $(typeof(op))"))
        end
    end
    # The reverse kernels and device copies above are issued on the task's one
    # ordered queue without host barriers between them; wait once, so the
    # walk returns finished adjoint state and footprints.
    synchronize(get_backend(lambda_panels[1]))
    return nothing
end

# Reverse of a sweep record. Its air mass and its tracer part (the limiter
# codes of a limited PPM sweep, the tracer mass of an FV3-profile sweep) are
# read with different roles: host-side tape storage returns one device cache
# per role, so the kernel never gets one buffer for both operands (issue #57).
_reverse_sweep!(lambda_panels, op::_CSSweepRecord{<:Any, <:Any, Nothing}, mesh::CubedSphereMesh, ws) =
    _adjoint_scheme_sweep!(lambda_panels, _tape_panels(op.panels_m, AirMassRead()),
                           op.panels_flux, op.direction, op.scheme, mesh, ws, op.flux_scale)
_reverse_sweep!(lambda_panels, op::_CSSweepRecord, mesh::CubedSphereMesh, ws) =
    _adjoint_scheme_sweep!(lambda_panels, _tape_panels(op.panels_m, AirMassRead()),
                           _tape_panels(op.panels_tracer, _tracer_read_role(op.scheme)),
                           op.panels_flux, op.direction, op.scheme, mesh, ws, op.flux_scale)

_tracer_read_role(::CSAdjointNonlinearScheme) = LimiterCodeRead()
_tracer_read_role(::CSAdjointFV3Vertical) = TracerMassRead()

# The clock a diffusion record was taken at, for `update_field!` of the
# operator's Kz field in the reverse solve.
struct _RecordedClock
    time::Float64
end
current_time(c::_RecordedClock) = c.time

# Reverse of an air-mass reset. Preserving the tracer mass leaves it unchanged
# (identity). Preserving the mixing ratio scales the tracer mass by
# `new_m / old_m` where `old_m > eps` (zero elsewhere), a diagonal map that is
# its own transpose. The halo refresh that follows the reset has its own
# `_CSHaloRecord`.
_apply_air_mass_reset_adjoint!(_lambda, op::_CSAirMassResetRecord{Nothing}) = nothing

@kernel function _preserve_vmr_reset_adjoint_kernel!(lambda, @Const(old_m), @Const(new_m))
    I = @index(Global, Cartesian)
    @inbounds begin
        FT = eltype(lambda)
        old = old_m[I]
        lambda[I] = old > eps(FT) ? lambda[I] / old * new_m[I] : zero(FT)
    end
end

function _apply_air_mass_reset_adjoint!(lambda_panels, op::_CSAirMassResetRecord)
    op.mode === :preserve_vmr || throw(ArgumentError(
        "unknown air-mass reset mode $(repr(op.mode)) on the adjoint tape"))
    old_m = _tape_panels(op.panels_old_m, AirMassRead())
    new_m = _tape_panels(op.panels_new_m, TracerMassRead())
    for p in 1:6
        backend = get_backend(lambda_panels[p])
        _preserve_vmr_reset_adjoint_kernel!(backend, 256)(lambda_panels[p], old_m[p], new_m[p];
                                                         ndrange = size(lambda_panels[p]))
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Building a replay from the keyword API
# ---------------------------------------------------------------------------

# Production runs the palindrome with `DEFAULT_CFL_LIMIT`; the adjoint replays
# production and accepts no other limit.
function _require_production_cfl_limit(cfl_limit)
    cfl_limit == DEFAULT_CFL_LIMIT || throw(ArgumentError(
        "cfl_limit = $(cfl_limit): the cubed-sphere adjoint replays the production step, " *
        "whose palindrome budget is DEFAULT_CFL_LIMIT = $(DEFAULT_CFL_LIMIT)"))
    return nothing
end

"""
    _cs_replay_for_call(panels_rm0, panels_m0, am_steps, bm_steps, cm_steps, mesh;
                        scheme, dt = nothing, schedule = nothing,
                        nsteps = length(am_steps), surface_flux = true,
                        emission_rates = nothing, base_emission_rates = nothing,
                        flux_scale, cfl_limit,
                        diffusion_op, diffusion_workspace, diffusion_meteo,
                        convection_op, convection_forcing, convection_workspace)
        -> (replay, dt)

Validate the keyword inputs of an adjoint entry point, check that every
operator of its step plan has an adjoint (`require_adjoint_support`) and build
the replay of steps `1:nsteps`. `dt` is the step length (the schedule's when
one is given).
"""
function _cs_replay_for_call(panels_rm0, panels_m0, panels_am_steps, panels_bm_steps,
                             panels_cm_steps, mesh::CubedSphereMesh;
                             scheme,
                             dt = nothing,
                             schedule = nothing,
                             nsteps::Int = length(panels_am_steps),
                             surface_flux::Bool = true,
                             emission_rates = nothing,
                             base_emission_rates = nothing,
                             flux_scale = one(eltype(panels_m0[1])),
                             cfl_limit = DEFAULT_CFL_LIMIT,
                             diffusion_op = NoDiffusion(),
                             diffusion_workspace = nothing,
                             diffusion_meteo = nothing,
                             convection_op = NoConvection(),
                             convection_forcing = nothing,
                             convection_workspace = nothing)
    FT = eltype(panels_m0[1])
    total = _validate_step_sequences(panels_am_steps, panels_bm_steps, panels_cm_steps)
    1 <= nsteps <= total || throw(ArgumentError("nsteps = $nsteps outside 1:$total"))
    emission_rates !== nothing && base_emission_rates !== nothing && throw(ArgumentError(
        "pass emission_rates or base_emission_rates, not both"))
    rates = emission_rates === nothing ? base_emission_rates : emission_rates
    _validate_emission_rates(rates, total, mesh, "emission rates")
    _require_production_cfl_limit(cfl_limit)
    _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace, total)
    _require_cs_convection_workspace(convection_op, convection_workspace)
    convection_op isa NoConvection || _validate_convection_forcing(convection_forcing, total)
    schedule, step_dt = _call_schedule(schedule, dt, FT)
    replay = _build_cs_replay(panels_rm0, panels_m0, panels_am_steps, panels_bm_steps,
                              panels_cm_steps, mesh; scheme, schedule, nsteps,
                              flux_scale, surface_flux, emission_rates = rates,
                              diffusion_op, diffusion_workspace, diffusion_meteo,
                              convection_op, convection_forcing, convection_workspace)
    require_adjoint_support(replay)
    return replay, step_dt
end

function _validate_convection_forcing(forcing, nsteps::Int)
    forcing === nothing && throw(ArgumentError(
        "an active convection operator requires `convection_forcing`"))
    forcing isa AbstractVector && length(forcing) != nsteps && throw(ArgumentError(
        "convection_forcing length $(length(forcing)) does not match nsteps $nsteps"))
    return nothing
end

# ---------------------------------------------------------------------------
# Forward replays
# ---------------------------------------------------------------------------

function _run_cs_footprint_forward(panels_rm0, panels_m0,
                                   panels_am_steps,
                                   panels_bm_steps,
                                   panels_cm_steps,
                                   mesh::CubedSphereMesh,
                                   objective::AbstractCSFootprintObjective;
                                   scheme = PPMScheme(NoLimiter()),
                                   kwargs...)
    Nz = size(panels_rm0[1], 3)
    _validate_objective(objective, mesh, Nz)
    replay, _ = _cs_replay_for_call(panels_rm0, panels_m0, panels_am_steps, panels_bm_steps,
                                    panels_cm_steps, mesh; scheme, kwargs...)
    _replay_steps!(replay, 1:_replay_nsteps(replay))
    return evaluate_objective(objective, _replay_rm(replay), _replay_m(replay), mesh)
end
