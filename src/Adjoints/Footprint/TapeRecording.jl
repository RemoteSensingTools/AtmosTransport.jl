# ---------------------------------------------------------------------------
# Forward-tape recording for the CS adjoint pipeline.
#
# The tape is written by the production step itself: `_record_cs_tape!` runs
# the replay (`StepReplay.jl`) with a `_CSTapeRecorder` (`TapeRecorder.jl`),
# whose hooks stage each operator's input as production calls it. This file
# keeps the recording entry point and the tape-size estimate:
#
#   * `_record_cs_tape!` — records a range of steps into a tape pool (the
#     Stride driver's windows).
#   * `_center_diffusion_records` — diffusion solves per palindrome center.
#   * `cs_tape_byte_estimate` — counts the tape a recording would write, by
#     replaying the transport with a counting recorder.
#
# Loaded into the `Adjoints` module from `Adjoints.jl`.
# ---------------------------------------------------------------------------

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
production step, plus an end-of-step marker after each step), finished with
`_finish_recording!`. If the recording throws, the records it holds outside
the pool are freed first (`_release_records!`; a Lin–Rood substep that throws
between its first and last hook leaves its partial copies to the GC).
"""
function _record_cs_tape!(replay::_CSReplay, steps::AbstractUnitRange,
                          content::AbstractTapeContent, pool)
    recorder = _CSTapeRecorder(content, pool, replay.advection)
    try
        _replay_steps!(replay, steps; recorder)
        _finish_recording!(recorder)
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

Size the tape that `cs_surface_emission_footprint` records for the given
problem (`StateTape` and `AirMassTape` stage the same records): per-record-class
counts and the total `state_bytes`, the raw panel-data cost of the tape,
independent of storage. The transport records (sweeps, halos, Lin-Rood
horizontal records, palindrome centers and air-mass resets) are counted by
replaying the transport with the counting recorder, which runs the tape
recorder's own hooks; convection and diffusion do not change the air mass, so
their records, and the air masses they stage, are counted from the step plan.

What a split-sweep tape stages per step (one palindrome pass, `S` the bytes
of one air-mass state): `6S` for the six sweeps' air masses (the palindrome
center's diffusion records share the reverse half's first vertical sweep's,
and a convection record the next step's first sweep's or a `:preserve_vmr`
reset's), plus `6S / sizeof(FT)` of limiter codes for a monotone or CW84
PPM sweep; `2S` per `:preserve_vmr` reset, and `S` for a convection record
that has no later air mass to share (the last one of the run, one before a
`:preserve_tracer_mass` reset).

The same `state_bytes` figure applies to every `storage` of an `AdjointTape`,
because none of them compresses the payload:

  * `DeviceCSTapeStorage()` — `state_bytes` is the device-resident cost; the
    reverse loop holds the full tape on the backend.
  * `PinnedHostCSTapeStorage()` — `state_bytes` is the pinned host cost
    (allocated in chunks of up to 1 GiB, each slot padded to 64 bytes); at
    most two staged panel sets (one per read role: the air mass, and the
    limiter codes or tracer mass) are mirrored on the device at a time.
  * `MmapCSTapeStorage(dir)` — `state_bytes` is the size of the scratch file
    with `FullCheckpoint` (`StrideCheckpoint` reserves its largest window),
    which needs 10 % more free space in `dir`; it is reserved before
    recording. The read caches hold one `NTuple{6, T}` per read role and
    distinct shape signature on top of that.

`state_records` counts the staged six-panel tuples (air masses, limiter
codes, tracer masses), `bytes_per_state` is the size of one air-mass state.
Lin-Rood's `state_bytes` also includes the face-panel tuples of its
horizontal records, whose dimensions differ from a full state. Halo,
midpoint, reset and end-of-step records carry only scalar metadata and are
counted in `total_records` but NOT in `state_bytes`.
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
    counter = _CSCountingRecorder(StateTape(), scheme)
    _replay_steps!(replay, 1:nsteps; recorder = counter)
    _finish_recording!(counter)

    # Records of the stripped operators, from the plan. A split-sweep center
    # shares its air mass with the next sweep, which the transport replay
    # stages already; a Lin–Rood center stages its own.
    style = counter.style
    steps = replay.steps
    diffusion_records = 0
    diffusion_states = 0
    for s in 1:nsteps
        op = _sequence_at(diffusion_op, s)
        op isa NoDiffusion && continue
        diffusion_records += surface_flux ? _center_diffusion_records(surface_flux_coupling(op)) : 1
        _shares_air_mass(style) || (diffusion_states += 1)
    end
    convection_records, convection_states = convection_op isa NoConvection ? (0, 0) :
        (_plan_stage_count(replay.plan, steps, ConvectionStage),
         _staged_convection_records(style, replay.plan, steps))
    bytes_per_state = _bytes_per_panel_tuple(panels_m0)
    state_records = counter.state + diffusion_states + convection_states
    state_bytes = counter.state_bytes + (diffusion_states + convection_states) * bytes_per_state
    # Lin-Rood horizontal records count as sweep records.
    sweep_records = counter.sweep + counter.linrood
    total_records = sweep_records + counter.halo + counter.midpoint + diffusion_records +
                    convection_records + counter.reset + counter.step
    return CSTapeByteEstimate(nsteps, sweep_records, counter.halo, counter.midpoint,
                              diffusion_records, convection_records, counter.step,
                              state_records, total_records, bytes_per_state, state_bytes)
end

# The stages step `s` runs, in order (window phases at the windows' first and
# last steps).
function _stages_of_step(plan::StepPlan, steps::_ResolvedSchedule, s::Int)
    head = steps.first[s] ?
        (steps.window[s] > 1 ? (plan.window_advance..., plan.window_start...) : plan.window_start) : ()
    return (head..., plan.step..., (steps.last[s] ? plan.window_end : ())...)
end

# How many times the steps run a stage of type `T`.
_plan_stage_count(plan::StepPlan, steps::_ResolvedSchedule, ::Type{T}) where T =
    sum(s -> count(stage -> stage isa T, _stages_of_step(plan, steps, s)), eachindex(steps.window);
        init = 0)

# Convection records that stage their own air mass (the recorder's
# `record_convection!`): every one on a Lin–Rood tape; on a split-sweep tape
# those that no later sweep or `:preserve_vmr` reset shares, i.e. the ones
# followed by a `:preserve_tracer_mass` reset or by the end of the run.
_staged_convection_records(::CSLinRoodStyle, plan::StepPlan, steps::_ResolvedSchedule) =
    _plan_stage_count(plan, steps, ConvectionStage)
function _staged_convection_records(::CSSplitSweepStyle, plan::StepPlan, steps::_ResolvedSchedule)
    staged = 0
    pending = false
    for s in eachindex(steps.window), stage in _stages_of_step(plan, steps, s)
        if stage isa ConvectionStage
            staged += pending
            pending = true
        elseif stage isa TransportStage
            pending = false
        elseif stage isa AirMassResetStage
            staged += pending && stage.mode !== :preserve_vmr
            pending = false
        end
    end
    return staged + pending
end
