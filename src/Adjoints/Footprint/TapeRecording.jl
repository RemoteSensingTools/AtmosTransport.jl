# ---------------------------------------------------------------------------
# Forward-tape recording for the CS adjoint pipeline.
#
# The tape is written by the production step itself: `_record_cs_tape!` runs
# the replay (`StepReplay.jl`) with a `_CSTapeRecorder` (`TapeRecorder.jl`),
# whose hooks stage each operator's input as production calls it. This file
# keeps the record builders the recorder uses and the tape-size estimate:
#
#   * `_record_sweep!` — pushes a `_CSSweepRecord` for one advection sweep,
#     by tape content and scheme (`FullStateTape`: air mass only for linear
#     schemes; air and tracer mass for the schemes whose reverse replays
#     limiter branches).
#   * `_center_diffusion_records` — diffusion solves per palindrome center.
#   * `cs_tape_byte_estimate` — counts the tape a recording would write, by
#     replaying the transport with a counting recorder.
#   * `_record_cs_tape!` — records a range of steps into a tape pool (the
#     Stride driver's windows).
#
# Loaded into the `Adjoints` module from `Adjoints.jl`.
# ---------------------------------------------------------------------------

# Full-panel tuples a sweep record of the tape content stages: the air mass,
# plus the tracer mass for the schemes whose reverse replays limiter branches.
_staged_tuples_per_sweep(::FullStateTape, ::CSAdjointLinearScheme) = 1
_staged_tuples_per_sweep(::FullStateTape, ::_RmTapedSweepScheme) = 2

# The record holds the state ENTERING the sweep: the reverse kernels rebuild
# each face coefficient from the donor air mass the forward kernel divided by
# (issue #10), so the recorder hook runs before the sweep. A linear sweep's
# reverse needs only the air mass; the recorder passes the tracer mass to
# every scheme, and the linear method ignores it.
function _record_sweep!(ops, ::FullStateTape, direction::Symbol, scheme::CSAdjointLinearScheme,
                        panels_m, _panels_rm, panels_flux, flux_scale, pool)
    push!(ops, _CSSweepRecord(direction, scheme, _stage_panels(pool, panels_m), nothing,
                              panels_flux, flux_scale))
    return nothing
end

function _record_sweep!(ops, ::FullStateTape, direction::Symbol, scheme::_RmTapedSweepScheme,
                        panels_m, panels_rm, panels_flux, flux_scale, pool)
    push!(ops, _CSSweepRecord(direction, scheme, _stage_panels(pool, panels_m),
                              _stage_panels(pool, panels_rm), panels_flux, flux_scale))
    return nothing
end

# Diffusion records per step of a palindrome center with `coupling`, counted
# by running the center composition itself so the estimate cannot drift from
# the recorders.
function _center_diffusion_records(coupling)
    count = Ref(0)
    surface_flux_center!(_ -> (count[] += 1), _ -> nothing, coupling, 1)
    return count[]
end

"""
    _record_cs_tape!(replay, steps, content, pool) -> ops

Run `steps` of `replay` with a tape recorder staging `content` into the tape
pool `pool` (`_open_tape_pool`) and return the tape (one record per hook of the
production step, plus an end-of-step marker after each step). If the
recording throws, the records it holds outside the pool are freed first
(`_release_records!`; a Lin–Rood substep that throws between its first and
last hook leaves its partial copies to the GC).
"""
function _record_cs_tape!(replay::_CSReplay, steps::AbstractUnitRange,
                          content::AbstractTapeContent, pool)
    recorder = _CSTapeRecorder(content, pool)
    try
        _replay_steps!(replay, steps; recorder)
    catch
        _release_records!(recorder.ops)
        rethrow()
    end
    return recorder.ops
end

"""
    cs_tape_byte_estimate(panels_m0, panels_am_steps, panels_bm_steps,
                          panels_cm_steps, mesh, scheme;
                          cfl_limit = DEFAULT_CFL_LIMIT,
                          flux_scale = one(eltype(panels_m0[1])),
                          diffusion_op = NoDiffusion(),
                          convection_op = NoConvection(),
                          schedule = nothing, surface_flux = true, dt = nothing)
        -> CSTapeByteEstimate

Size the tape (`FullStateTape` content) that `cs_surface_emission_footprint`
records for the given problem: per-record-class counts and the total
`state_bytes`, the raw panel-data cost of the tape, independent of storage. The transport
records (sweeps, halos, Lin-Rood horizontal records, palindrome centers and
air-mass resets) are counted by replaying the transport with a counting
recorder; convection and diffusion do not change the air mass, so their
records are counted from the step plan.

The same `state_bytes` figure applies to every `storage` of an `AdjointTape`,
because none of them compresses the payload:

  * `DeviceCSTapeStorage()` — `state_bytes` is the device-resident cost; the
    reverse loop holds the full tape on the backend.
  * `PinnedHostCSTapeStorage()` — `state_bytes` is the pinned host cost
    (allocated in chunks of up to 1 GiB, each slot padded to 64 bytes); at
    most two staged panel sets (one per read role: air mass and tracer mass)
    are mirrored on the device at a time.
  * `MmapCSTapeStorage(dir)` — `state_bytes` is the size of the scratch file
    with `FullCheckpoint` (`StrideCheckpoint` reserves its largest window),
    which needs 10 % more free space in `dir`; it is reserved before
    recording. The read caches hold one `NTuple{6, T}` per read role and
    distinct shape signature on top of that.

`bytes_per_state` is the size of one full-panel state tuple. Lin-Rood's
`state_bytes` also includes the face-panel tuples of its horizontal records,
whose dimensions differ from a full state. Halo, midpoint, reset and
end-of-step records carry only scalar metadata and are counted in
`total_records` but NOT in `state_bytes`.
"""
function cs_tape_byte_estimate(panels_m0, panels_am_steps, panels_bm_steps,
                               panels_cm_steps, mesh::CubedSphereMesh, scheme;
                               cfl_limit = DEFAULT_CFL_LIMIT,
                               flux_scale = one(eltype(panels_m0[1])),
                               diffusion_op = NoDiffusion(),
                               convection_op = NoConvection(),
                               schedule = nothing,
                               surface_flux::Bool = true,
                               dt = nothing)
    require_adjoint_support(scheme)
    _require_production_cfl_limit(cfl_limit)
    FT = eltype(panels_m0[1])
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps, panels_cm_steps)
    schedule, _ = _call_schedule(schedule, dt, FT)
    zero_rm = _zero_panel_tuple_like(panels_m0)
    # Transport records: the replay without diffusion and convection (neither
    # changes the air mass the sweeps and their subcycle counts see).
    replay = _build_cs_replay(zero_rm, panels_m0, panels_am_steps, panels_bm_steps,
                              panels_cm_steps, mesh; scheme, schedule, nsteps,
                              flux_scale, surface_flux)
    counter = _CSCountingRecorder(FullStateTape())
    _replay_steps!(replay, 1:nsteps; recorder = counter)

    # Records of the stripped operators, from the plan.
    steps = replay.steps
    diffusion_records = 0
    diffusion_states = 0
    for s in 1:nsteps
        op = _sequence_at(diffusion_op, s)
        op isa NoDiffusion && continue
        diffusion_records += surface_flux ? _center_diffusion_records(surface_flux_coupling(op)) : 1
        diffusion_states += 1
    end
    convection_records = convection_op isa NoConvection ? 0 :
        _plan_stage_count(replay.plan, steps, ConvectionStage)
    bytes_per_state = _bytes_per_panel_tuple(panels_m0)
    state_records = counter.state + diffusion_states + convection_records
    state_bytes = counter.state_bytes + (diffusion_states + convection_records) * bytes_per_state
    # Lin-Rood horizontal records count as sweep records.
    sweep_records = counter.sweep + counter.linrood
    total_records = sweep_records + counter.halo + counter.midpoint + diffusion_records +
                    convection_records + counter.reset + counter.step
    return CSTapeByteEstimate(nsteps, sweep_records, counter.halo, counter.midpoint,
                              diffusion_records, convection_records, counter.step,
                              state_records, total_records, bytes_per_state, state_bytes)
end

# How many times the steps run a stage of type `T` (window phases count at the
# windows' first and last steps).
function _plan_stage_count(plan::StepPlan, steps::_ResolvedSchedule, ::Type{T}) where T
    n(stages) = count(s -> s isa T, stages)
    total = 0
    for s in eachindex(steps.window)
        if steps.first[s]
            steps.window[s] > 1 && (total += n(plan.window_advance))
            total += n(plan.window_start)
        end
        total += n(plan.step)
        steps.last[s] && (total += n(plan.window_end))
    end
    return total
end
