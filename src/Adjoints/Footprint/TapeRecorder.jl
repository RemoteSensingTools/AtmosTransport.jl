# ---------------------------------------------------------------------------
# Tape recorders: the step recorders (`Operators.AbstractStepRecorder`) the
# adjoint passes to the production step.
#
#   * `_CSTapeRecorder` stages each hook's input in the call's tape pool and
#     pushes one tape record per hook, so the tape holds exactly the operators
#     production ran, in production order.
#   * `_CSCountingRecorder` counts the records and staged bytes the tape
#     recorder writes (`cs_tape_byte_estimate`, and the extent an mmap tape
#     reserves before recording).
#
# Both run the same hooks (`_AbstractCSRecorder`), which decide what a record
# stages, so the count is the tape by construction; they differ only in what
# staging and pushing a record do (`_stage!`, `_stage_codes!`, `_record!`).
#
# What a record stages (`StateTape` and `AirMassTape`; the replay's advection
# style `L` decides what records share):
#
#   * a sweep: the air mass entering it; for a monotone or CW84 PPM sweep the
#     limiter branch of every cell (`record_ppm_limiter_codes!`, one byte per
#     cell) instead of the tracer mass; for an FV3-profile sweep (Lin–Rood's
#     vertical) the tracer mass;
#   * split sweeps (`CSSplitSweepStyle`) share two air masses that are
#     already on the tape:
#       - the palindrome center's air mass (staged for its diffusion records)
#         is the one the reverse half's first Z sweep enters with: the
#         midpoint writes only the tracer and no halo fill runs in between,
#         so that sweep record reads the center's slot;
#       - a convection record reads only interior cells, and the air mass's
#         interior does not change until the next step's first X sweep (the
#         step-start halo fill changes halos only) or a following air-mass
#         reset: the record takes the X sweep's slot, or a `:preserve_vmr`
#         reset's old air mass (identical, halos included). It stages its own
#         slot only before a `:preserve_tracer_mass` reset (which replaces
#         the air mass) and at the end of a recording (`_finish_recording!`);
#   * Lin–Rood (`CSLinRoodStyle`) records stage every state they read; its
#     horizontal records are device copies outside the pool (`LinRoodTape.jl`).
#
# Every hook the cubed-sphere step reaches is implemented; a hook a recorder
# does not implement is a `MethodError`, never a skipped record. The reverse
# walk (`ReverseLoop.jl`) dispatches on the records.
# ---------------------------------------------------------------------------

abstract type _AbstractCSRecorder <: AbstractStepRecorder end

# Hooks may observe arrays whose producing kernels are only queued (the
# palindrome's halo exchanges run with `sync = false`). The device pool
# stages with device copies queued behind those kernels, so it needs no host
# barrier; host pools copy into host memory and finish them first.
_sync_panels(::_DeviceSlabPool, _panels) = nothing
_sync_panels(_pool, panels) = synchronize(get_backend(first(panels)))

mutable struct _CSTapeRecorder{C <: AbstractTapeContent, S, L} <: _AbstractCSRecorder
    ops                 :: Vector{Any}
    const content       :: C
    const storage       :: S        # the tape pool the call opened
    const style         :: L        # the replay's advection style: what records share
    current_step        :: Int
    time                :: Float64  # clock of the current step (`current_time` of the replay)
    center_m            :: Any      # staged air mass of the current palindrome center
    center_pending      :: Bool     # the next sweep record takes `center_m` (split sweeps)
    convection_pending  :: Bool     # a convection record awaits its air-mass slot
    convection_index    :: Int      # its index in `ops`
    convection_m        :: Any      # the air-mass panels it read (staged if no slot comes)
    codes_scratch       :: Any      # `UInt8` panels the limiter codes are computed in, or nothing
    diffusion_workspace :: Any      # the caller's diffusion workspace of the current step
    linrood             :: Dict{Symbol, Any}   # phases of the Lin-Rood operator being recorded
end

"""
    _CSTapeRecorder(content, pool, scheme)

A tape recorder staging `content` into the tape pool `pool` for a replay with
advection `scheme` (whose style decides which records share a slot).
"""
_CSTapeRecorder(content::AbstractTapeContent, pool, scheme) =
    _CSTapeRecorder(Any[], content, pool, cs_advection_style(scheme), 0, 0.0, nothing, false,
                    false, 0, nothing, nothing, nothing, Dict{Symbol, Any}())

# ---------------------------------------------------------------------------
# Counting recorder (tape byte estimate, mmap reservations)
# ---------------------------------------------------------------------------

mutable struct _CSCountingRecorder{C <: AbstractTapeContent, L} <: _AbstractCSRecorder
    const content :: C
    const style   :: L
    sweep        :: Int
    halo         :: Int
    midpoint     :: Int
    diffusion    :: Int
    convection   :: Int
    reset        :: Int
    step         :: Int      # end-of-step records
    linrood      :: Int
    state        :: Int      # staged six-panel tuples (air and tracer masses, limiter codes)
    state_bytes  :: Int
    slot_bytes   :: Int      # the staged tuples' bytes in an mmap tape (`_tape_slot_bytes`)
    current_step :: Int
    time         :: Float64
    center_m           :: Any
    center_pending     :: Bool
    convection_pending :: Bool
    convection_index   :: Int
    convection_m       :: Any
    diffusion_workspace :: Any
    slot_bytes_through_step :: Vector{Int}    # `slot_bytes` at the end of each counted step
    layouts::Vector{Vector{Any}}
    pending_slot_bytes      :: Vector{Int}    # bytes a recording ending after the step adds (`_finish_recording!`)
end

"""
    _CSCountingRecorder(content, scheme)

Counts what a `_CSTapeRecorder(content, pool, scheme)` stages, step by step.
"""
_CSCountingRecorder(content::AbstractTapeContent, scheme) =
    _CSCountingRecorder(content, cs_advection_style(scheme), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0.0, nothing, false, false, 0, nothing, nothing, Int[], Vector{Any}[], Int[])

# ---------------------------------------------------------------------------
# What staging and recording do
# ---------------------------------------------------------------------------

_sync!(r::_CSTapeRecorder, panels) = _sync_panels(r.storage, panels)
_sync!(::_CSCountingRecorder, _panels) = nothing

_stage!(r::_CSTapeRecorder, panels) = _stage_panels(r.storage, panels)
_stage!(r::_CSCountingRecorder, panels) =
    (_count_layout!(r, panels, eltype(first(panels)));
     _count_state!(r, _bytes_per_panel_tuple(panels), _tape_slot_bytes(panels)); nothing)

function _count_state!(r::_CSCountingRecorder, bytes::Integer, slot_bytes::Integer)
    r.state += 1
    r.state_bytes += bytes
    r.slot_bytes += slot_bytes
    return nothing
end

# The limiter branch codes of a limited PPM sweep, computed from the state
# entering it, in place of its tracer mass: one byte per cell (a device slot
# is written in place; a host slot through the recorder's scratch panels).
function _stage_codes!(r::_CSTapeRecorder, direction::Symbol, scheme, panels_m, panels_rm)
    scratch = _codes_scratch!(r, panels_m)
    return _stage_computed(codes -> record_ppm_limiter_codes!(codes, panels_m, panels_rm, direction, scheme),
                           r.storage, scratch)
end
# A device slot is the output buffer: template its shape from the air mass
# and request UInt8 directly, without allocating a redundant scratch state.
function _stage_codes!(r::_CSTapeRecorder{C, S}, direction::Symbol, scheme, panels_m, panels_rm) where
        {C <: AbstractTapeContent, S <: _DeviceSlabPool}
    return _stage_computed(codes -> record_ppm_limiter_codes!(codes, panels_m, panels_rm, direction, scheme),
                           r.storage, panels_m, UInt8)
end
_stage_codes!(r::_CSCountingRecorder, _direction, _scheme, panels_m, _panels_rm) =
    (_count_layout!(r, panels_m, UInt8);
     _count_state!(r, _bytes_per_panel_tuple(UInt8, panels_m), _tape_slot_bytes(UInt8, panels_m)); nothing)

function _codes_scratch!(r::_CSTapeRecorder, panels_m)
    shared = _tape_code_scratch!(r.storage, panels_m)
    shared === nothing || return shared
    scratch = r.codes_scratch
    scratch isa NTuple{6} && all(p -> size(scratch[p]) == size(panels_m[p]), 1:6) && return scratch
    scratch === nothing || _free_tape_panels!(scratch)
    return r.codes_scratch = ntuple(p -> similar(panels_m[p], UInt8), 6)
end

_record!(r::_CSTapeRecorder, op) = (push!(r.ops, op); length(r.ops))
_record!(r::_CSCountingRecorder, op) = (_count_record!(r, op); 0)
_count_record!(r, ::_CSSweepRecord) = (r.sweep += 1; nothing)
_count_record!(r, ::_CSHaloRecord) = (r.halo += 1; nothing)
_count_record!(r, ::_CSMidpointRecord) = (r.midpoint += 1; nothing)
_count_record!(r, ::_CSDiffusionRecord) = (r.diffusion += 1; nothing)
_count_record!(r, ::_CSConvectionRecord) = (r.convection += 1; nothing)
_count_record!(r, ::_CSAirMassResetRecord) = (r.reset += 1; nothing)
_count_record!(r, ::_CSStepEndRecord) = (r.step += 1; nothing)

# The convection record `i` with the air-mass slot `slot`.
function _set_convection_slot!(r::_CSTapeRecorder, i::Int, slot)
    old = r.ops[i]
    r.ops[i] = _CSConvectionRecord(old.op, old.forcing, slot, old.dt, old.substeps)
    return nothing
end
_set_convection_slot!(::_CSCountingRecorder, _i::Int, _slot) = nothing

# Which records share a slot: split sweeps share the palindrome center's and a
# convection call's air mass; Lin–Rood records stage their own.
_shares_air_mass(::CSSplitSweepStyle) = true
_shares_air_mass(::CSLinRoodStyle) = false

# ---------------------------------------------------------------------------
# The hooks (both recorders)
# ---------------------------------------------------------------------------

function _begin_recorded_step!(r::_AbstractCSRecorder, s::Int, diffusion_workspace, time)
    r.current_step = s
    r.time = Float64(time)
    r.diffusion_workspace = diffusion_workspace
    return nothing
end

# End-of-step marker, after the window-end stages: the state runtime output and
# the observation sampler see. The reverse walk injects the observation
# forcing of the step here, before transposing the step's last operator.
_end_recorded_step!(r::_CSTapeRecorder, s::Int) = (_record!(r, _CSStepEndRecord(s)); nothing)
function _end_recorded_step!(r::_CSCountingRecorder, s::Int)
    _record!(r, _CSStepEndRecord(s))
    push!(r.slot_bytes_through_step, r.slot_bytes)
    push!(r.pending_slot_bytes, r.convection_pending ? _tape_slot_bytes(r.convection_m) : 0)
    return nothing
end

"""
    _finish_recording!(recorder)

End a recording: a convection record still waiting for an air-mass slot
stages its own (the recording may end with a convection call), and the
recorder's scratch panels are freed. Every recording calls it after its last
step (`_record_cs_tape!`, the Full driver).
"""
function _finish_recording!(r::_AbstractCSRecorder)
    _flush_convection!(r)
    r.center_pending = false
    _free_recorder_scratch!(r)
    return r
end
_free_recorder_scratch!(::_CSCountingRecorder) = nothing
function _free_recorder_scratch!(r::_CSTapeRecorder)
    scratch = r.codes_scratch
    r.codes_scratch = nothing
    scratch === nothing || _free_tape_panels!(scratch)
    return nothing
end

record_halo!(r::_AbstractCSRecorder, dir) = (_record!(r, _CSHaloRecord(dir)); nothing)

# The record holds the state ENTERING the sweep: the reverse kernels rebuild
# each face coefficient from the donor air mass the forward kernel divided by
# (issue #10), so the recorder hook runs before the sweep.
function record_sweep!(r::_AbstractCSRecorder, direction::Symbol, scheme, panels_m, panels_rm,
                       panels_flux, flux_scale)
    _sync!(r, panels_m)
    m_slot = r.center_pending ? r.center_m : _stage!(r, panels_m)
    r.center_pending = false
    _bind_convection!(r, m_slot)
    tracer = _stage_sweep_tracer!(r, direction, scheme, panels_m, _single_tracer(panels_rm))
    _record!(r, _CSSweepRecord(direction, scheme, m_slot, tracer, panels_flux, flux_scale))
    return nothing
end

# What a sweep record stages of the tracer: nothing for the linear schemes,
# the limiter codes of a limited PPM sweep, the tracer mass entering an
# FV3-profile sweep.
_stage_sweep_tracer!(_r, _direction, ::CSAdjointLinearScheme, _panels_m, _panels_rm) = nothing
_stage_sweep_tracer!(r, direction, scheme::CSAdjointNonlinearScheme, panels_m, panels_rm) =
    _stage_codes!(r, direction, scheme, panels_m, panels_rm)
_stage_sweep_tracer!(r, _direction, ::CSAdjointFV3Vertical, _panels_m, panels_rm) = _stage!(r, panels_rm)

# The palindrome center leaves the air mass unchanged: one staged state serves
# both half solves of a split center (and, for split sweeps, the next sweep).
record_center!(r::_AbstractCSRecorder, _panels_m, ::NoDiffusion) =
    (r.center_m = nothing; r.center_pending = false; nothing)
function record_center!(r::_AbstractCSRecorder, panels_m, _diffusion_op)
    _sync!(r, panels_m)
    r.center_m = _stage!(r, panels_m)
    r.center_pending = _shares_air_mass(r.style)
    return nothing
end

record_diffusion!(::_AbstractCSRecorder, ::NoDiffusion, _workspace, _dt) = nothing
function record_diffusion!(r::_AbstractCSRecorder, diffusion_op, _workspace, dt)
    # The reverse solve reads the layer thickness of the caller's workspace of
    # this step (the replay workspace holds the same values only until the
    # next step loads its own) and evaluates the Kz field at the forward's
    # clock.
    _record!(r, _CSDiffusionRecord(diffusion_op, r.diffusion_workspace, r.center_m, dt, r.time))
    return nothing
end

record_emission!(r::_AbstractCSRecorder, dt) = (_record!(r, _CSMidpointRecord(r.current_step, dt)); nothing)

# Convection does not change the air mass. Its record waits for the next
# air-mass slot with the same interior (split sweeps) or stages its own now.
function record_convection!(r::_AbstractCSRecorder, op, forcing, panels_m, dt, workspace)
    _flush_convection!(r)
    i = _record!(r, _CSConvectionRecord(op, forcing, nothing, dt, _convection_substeps(op, workspace)))
    r.convection_pending = true
    r.convection_index = i
    r.convection_m = panels_m
    _shares_air_mass(r.style) || _flush_convection!(r)
    return nothing
end

# Give the waiting convection record the air-mass slot `slot`.
function _bind_convection!(r::_AbstractCSRecorder, slot)
    r.convection_pending || return nothing
    _set_convection_slot!(r, r.convection_index, slot)
    r.convection_pending = false
    r.convection_m = nothing
    return nothing
end

# Stage the waiting convection record's own air mass (before a reset that
# replaces it, at the end of a recording).
function _flush_convection!(r::_AbstractCSRecorder)
    r.convection_pending || return nothing
    _sync!(r, r.convection_m)
    _bind_convection!(r, _stage!(r, r.convection_m))
    return nothing
end

# The CFL sub-step count the forward call used: `CMFMCConvection` caches it per
# met window from the air mass of the window's first call.
_convection_substeps(::CMFMCConvection, workspace::CMFMCWorkspace) = workspace.cached_n_sub[]
_convection_substeps(_op, _workspace) = 1

function record_air_mass_reset!(r::_AbstractCSRecorder, mode::Symbol, state, target)
    if mode === :preserve_vmr
        _sync!(r, state.air_mass)
        old_m = _stage!(r, state.air_mass)
        # The reset runs after this hook: the air mass is still the one a
        # waiting convection record read, halos included.
        _bind_convection!(r, old_m)
        _record!(r, _CSAirMassResetRecord(mode, old_m, _stage!(r, target)))
    else
        _flush_convection!(r)
        _record!(r, _CSAirMassResetRecord(mode, nothing, nothing))
    end
    # The reset ends with the air- and tracer-mass halo refresh (dir 1).
    _record!(r, _CSHaloRecord(1))
    return nothing
end

record_chemistry!(::_AbstractCSRecorder, ::NoChemistry, _dt) = nothing
record_chemistry!(::_AbstractCSRecorder, op, _dt) = throw(ArgumentError(
    "chemistry $(nameof(typeof(op))) has no cubed-sphere adjoint record"))

# ---------------------------------------------------------------------------
# Lin-Rood horizontal operator: the phases of one `fv_tp_2d_cs!` call build one
# `_CSLinRoodHorizRecord` (see `LinRoodTape.jl`), whose arrays are device
# copies outside the pool.
# ---------------------------------------------------------------------------

function record_linrood_horizontal!(r::_CSTapeRecorder, ::Val{:start}, panels_rm, panels_m)
    _sync_panels(r.storage, panels_m)
    empty!(r.linrood)
    r.linrood[:rm] = map(copy, panels_rm)
    r.linrood[:m] = map(copy, panels_m)
    return nothing
end
function record_linrood_horizontal!(r::_CSTapeRecorder, ::Val{:phase2}, q_buf)
    _sync_panels(r.storage, q_buf)
    r.linrood[:q2] = map(copy, Tuple(q_buf))
    return nothing
end
function record_linrood_horizontal!(r::_CSTapeRecorder, ::Val{:phase3}, q_buf)
    _sync_panels(r.storage, q_buf)
    r.linrood[:q3] = map(copy, Tuple(q_buf))
    return nothing
end
function record_linrood_horizontal!(r::_CSTapeRecorder, ::Val{:faces}, fx_in, fy_in)
    _sync_panels(r.storage, fx_in)
    r.linrood[:fx_in] = map(copy, Tuple(fx_in))
    r.linrood[:fy_in] = map(copy, Tuple(fy_in))
    return nothing
end
function record_linrood_horizontal!(r::_CSTapeRecorder, ::Val{:finish}, fx_out, am, bm,
                                    ::Val{ORD}) where ORD
    _sync_panels(r.storage, fx_out)
    parts = r.linrood
    rm = parts[:rm]
    FT = eltype(rm[1])
    fx_out_tape = map(copy, Tuple(fx_out))
    am_tape = map(copy, Tuple(am))
    bm_tape = map(copy, Tuple(bm))
    A3 = typeof(rm[1]); A3x = typeof(fx_out_tape[1]); A3y = typeof(parts[:fy_in][1])
    push!(r.ops, _CSLinRoodHorizRecord{FT, A3, A3x, A3y, A3, ORD}(
        rm, parts[:m], parts[:q2], parts[:q3], parts[:fx_in], fx_out_tape, parts[:fy_in],
        am_tape, bm_tape, one(FT)))
    empty!(r.linrood)
    return nothing
end

# A Lin-Rood horizontal record stages four full-panel states (rm, m and the
# two predictor-buffer phases) and five face-panel tuples (fx_in, fx_out,
# fy_in and the interior face fluxes am, bm), outside the pool: they count
# toward `state_bytes`, not toward the slot bytes an mmap tape reserves.
function record_linrood_horizontal!(r::_CSCountingRecorder, ::Val{:start}, panels_rm, panels_m)
    _count_layout!(r, panels_rm; pooled = false)
    _count_layout!(r, panels_m; pooled = false)
    r.linrood += 1
    bytes = _bytes_per_panel_tuple(panels_m)
    r.state += 2
    r.state_bytes += 2bytes
    return nothing
end
record_linrood_horizontal!(r::_CSCountingRecorder, ::Val{:phase2}, q_buf) =
    (_count_layout!(r, Tuple(q_buf); pooled = false);
     r.state += 1; r.state_bytes += _bytes_of(Tuple(q_buf)); nothing)
record_linrood_horizontal!(r::_CSCountingRecorder, ::Val{:phase3}, q_buf) =
    (_count_layout!(r, Tuple(q_buf); pooled = false);
     r.state += 1; r.state_bytes += _bytes_of(Tuple(q_buf)); nothing)
record_linrood_horizontal!(r::_CSCountingRecorder, ::Val{:faces}, fx_in, fy_in) =
    (_count_layout!(r, Tuple(fx_in); pooled = false); _count_layout!(r, Tuple(fy_in); pooled = false);
     r.state_bytes += _bytes_of(Tuple(fx_in)) + _bytes_of(Tuple(fy_in)); nothing)
record_linrood_horizontal!(r::_CSCountingRecorder, ::Val{:finish}, fx_out, am, bm, ::Val) =
    (_count_layout!(r, Tuple(fx_out); pooled = false); _count_layout!(r, Tuple(am); pooled = false);
     _count_layout!(r, Tuple(bm); pooled = false);
     r.state_bytes += _bytes_of(Tuple(fx_out)) + _bytes_of(Tuple(am)) + _bytes_of(Tuple(bm)); nothing)

_bytes_of(panels) = sum(p -> sizeof(eltype(p)) * length(p), panels)

function _count_layout!(r::_CSCountingRecorder, panels, T = eltype(first(panels)); pooled::Bool = true)
    while length(r.layouts) < r.current_step
        push!(r.layouts, Any[])
    end
    push!(r.layouts[r.current_step], (panels, T, pooled))
    return nothing
end
