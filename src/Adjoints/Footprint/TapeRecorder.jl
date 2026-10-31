# ---------------------------------------------------------------------------
# Tape recorders: the step recorders (`Operators.AbstractStepRecorder`) the
# adjoint passes to the production step.
#
#   * `_CSTapeRecorder` stages each hook's input in tape storage and pushes one
#     tape record per hook, so the tape holds exactly the operators production
#     ran, in production order.
#   * `_CSCountingRecorder` counts the records and staged bytes a tape
#     recorder would write (`cs_tape_byte_estimate`, and the extent an mmap
#     tape reserves before recording).
#
# Both implement every hook the cubed-sphere step reaches; a hook they do not
# implement is a `MethodError`, never a skipped record. The reverse walk
# (`ReverseLoop.jl`) dispatches on the records.
# ---------------------------------------------------------------------------

# Hooks may observe arrays whose producing kernels are only queued (the
# palindrome's halo exchanges run with `sync = false`). Device tape storage
# stages with device copies queued behind those kernels, so it needs no host
# barrier; host-side storage copies into host memory and finishes them first.
_sync_panels(::DeviceCSTapeStorage, _panels) = nothing
_sync_panels(_storage, panels) = synchronize(get_backend(first(panels)))

mutable struct _CSTapeRecorder{S} <: AbstractStepRecorder
    ops                 :: Vector{Any}
    storage             :: S
    step                :: Int
    time                :: Float64  # clock of the current step (`current_time` of the replay)
    center_m            :: Any      # staged air mass of the current palindrome center
    diffusion_workspace :: Any      # the caller's diffusion workspace of the current step
    linrood             :: Dict{Symbol, Any}   # phases of the Lin-Rood operator being recorded
end

_CSTapeRecorder(storage) = _CSTapeRecorder(Any[], storage, 0, 0.0, nothing, nothing,
                                           Dict{Symbol, Any}())

function _begin_recorded_step!(r::_CSTapeRecorder, s::Int, diffusion_workspace, time)
    r.step = s
    r.time = Float64(time)
    r.diffusion_workspace = diffusion_workspace
    return nothing
end

# End-of-step marker, after the window-end stages: the state runtime output and
# the observation sampler see. The reverse walk injects the observation
# forcing of the step here, before transposing the step's last operator.
_end_recorded_step!(r::_CSTapeRecorder, s::Int) = (push!(r.ops, _CSStepEndRecord(s)); nothing)

record_halo!(r::_CSTapeRecorder, dir) = (push!(r.ops, _CSHaloRecord(dir)); nothing)

function record_sweep!(r::_CSTapeRecorder, direction::Symbol, scheme, panels_m, panels_rm,
                       panels_flux, flux_scale)
    _sync_panels(r.storage, panels_m)
    _record_sweep!(r.ops, direction, scheme, panels_m, _single_tracer(panels_rm),
                   panels_flux, flux_scale, r.storage)
    return nothing
end

# The palindrome center leaves the air mass unchanged: one staged state serves
# both half solves of a split center.
record_center!(r::_CSTapeRecorder, _panels_m, ::NoDiffusion) = (r.center_m = nothing; nothing)
function record_center!(r::_CSTapeRecorder, panels_m, _diffusion_op)
    _sync_panels(r.storage, panels_m)
    r.center_m = _stage_panels(r.storage, panels_m)
    return nothing
end

record_diffusion!(::_CSTapeRecorder, ::NoDiffusion, _workspace, _dt) = nothing
function record_diffusion!(r::_CSTapeRecorder, diffusion_op, _workspace, dt)
    # The reverse solve reads the layer thickness of the caller's workspace of
    # this step (the replay workspace holds the same values only until the
    # next step loads its own) and evaluates the Kz field at the forward's
    # clock.
    push!(r.ops, _CSDiffusionRecord(diffusion_op, r.diffusion_workspace, r.center_m, dt, r.time))
    return nothing
end

record_emission!(r::_CSTapeRecorder, dt) = (push!(r.ops, _CSMidpointRecord(r.step, dt)); nothing)

function record_convection!(r::_CSTapeRecorder, op, forcing, panels_m, dt, workspace)
    _sync_panels(r.storage, panels_m)
    push!(r.ops, _CSConvectionRecord(op, forcing, _stage_panels(r.storage, panels_m), dt,
                                     _convection_substeps(op, workspace)))
    return nothing
end

# The CFL sub-step count the forward call used: `CMFMCConvection` caches it per
# met window from the air mass of the window's first call.
_convection_substeps(::CMFMCConvection, workspace::CMFMCWorkspace) = workspace.cached_n_sub[]
_convection_substeps(_op, _workspace) = 1

function record_air_mass_reset!(r::_CSTapeRecorder, mode::Symbol, state, target)
    if mode === :preserve_vmr
        _sync_panels(r.storage, state.air_mass)
        push!(r.ops, _CSAirMassResetRecord(mode, _stage_panels(r.storage, state.air_mass),
                                           _stage_panels(r.storage, target)))
    else
        push!(r.ops, _CSAirMassResetRecord(mode, nothing, nothing))
    end
    # The reset ends with the air- and tracer-mass halo refresh (dir 1).
    push!(r.ops, _CSHaloRecord(1))
    return nothing
end

record_chemistry!(::_CSTapeRecorder, ::NoChemistry, _dt) = nothing
record_chemistry!(::_CSTapeRecorder, op, _dt) = throw(ArgumentError(
    "chemistry $(nameof(typeof(op))) has no cubed-sphere adjoint record"))

# Lin-Rood horizontal operator: the phases of one `fv_tp_2d_cs!` call build one
# `_CSLinRoodHorizRecord` (see `LinRoodTape.jl`).
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

# ---------------------------------------------------------------------------
# Counting recorder (tape byte estimate)
# ---------------------------------------------------------------------------

mutable struct _CSCountingRecorder <: AbstractStepRecorder
    sweep        :: Int
    halo         :: Int
    midpoint     :: Int
    diffusion    :: Int
    convection   :: Int
    reset        :: Int
    step         :: Int
    linrood      :: Int
    state        :: Int      # staged full-panel tuples
    state_bytes  :: Int
    bytes_per_state :: Int
    slot_bytes   :: Int      # the staged states' bytes in an mmap tape (`_tape_slot_bytes`)
    slot_bytes_through_step :: Vector{Int}    # `slot_bytes` at the end of each counted step
end

_CSCountingRecorder() = _CSCountingRecorder(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Int[])

_bytes_of(panels) = sum(p -> sizeof(eltype(p)) * length(p), panels)

function _count_state!(r::_CSCountingRecorder, panels)
    r.state += 1
    bytes = _bytes_of(panels)
    r.state_bytes += bytes
    r.slot_bytes += _tape_slot_bytes(panels)
    r.bytes_per_state = bytes
    return nothing
end

_begin_recorded_step!(::_CSCountingRecorder, _s, _diffusion_ws, _time) = nothing
function _end_recorded_step!(r::_CSCountingRecorder, _s)
    r.step += 1
    push!(r.slot_bytes_through_step, r.slot_bytes)
    return nothing
end
record_halo!(r::_CSCountingRecorder, _dir) = (r.halo += 1; nothing)
function record_sweep!(r::_CSCountingRecorder, _direction, scheme, panels_m, _rm, _flux, _scale)
    r.sweep += 1
    for _ in 1:_staged_tuples_per_sweep(scheme)
        _count_state!(r, panels_m)
    end
    return nothing
end
record_center!(::_CSCountingRecorder, _m, ::NoDiffusion) = nothing
record_center!(r::_CSCountingRecorder, panels_m, _op) = _count_state!(r, panels_m)
record_diffusion!(::_CSCountingRecorder, ::NoDiffusion, _ws, _dt) = nothing
record_diffusion!(r::_CSCountingRecorder, _op, _ws, _dt) = (r.diffusion += 1; nothing)
record_emission!(r::_CSCountingRecorder, _dt) = (r.midpoint += 1; nothing)
function record_convection!(r::_CSCountingRecorder, _op, _forcing, panels_m, _dt, _ws)
    r.convection += 1
    _count_state!(r, panels_m)
    return nothing
end
# Counts what the tape recorder stages: the air mass and the target (whose
# element type may differ from the state's).
function record_air_mass_reset!(r::_CSCountingRecorder, mode, state, target)
    r.reset += 1
    r.halo += 1
    if mode === :preserve_vmr
        _count_state!(r, state.air_mass); _count_state!(r, target)
    end
    return nothing
end
record_chemistry!(::_CSCountingRecorder, ::NoChemistry, _dt) = nothing
record_chemistry!(r::_CSCountingRecorder, op, dt) = record_chemistry!(_CSTapeRecorder(nothing), op, dt)
# A Lin-Rood horizontal record stages four full-panel states (rm, m and the
# two predictor-buffer phases) and five face-panel tuples (fx_in, fx_out,
# fy_in and the interior face fluxes am, bm).
function record_linrood_horizontal!(r::_CSCountingRecorder, ::Val{:start}, panels_rm, panels_m)
    r.linrood += 1
    _count_state!(r, panels_m); _count_state!(r, panels_m)
    return nothing
end
record_linrood_horizontal!(r::_CSCountingRecorder, ::Val{:phase2}, q_buf) =
    (r.state += 1; r.state_bytes += _bytes_of(Tuple(q_buf)); nothing)
record_linrood_horizontal!(r::_CSCountingRecorder, ::Val{:phase3}, q_buf) =
    (r.state += 1; r.state_bytes += _bytes_of(Tuple(q_buf)); nothing)
record_linrood_horizontal!(r::_CSCountingRecorder, ::Val{:faces}, fx_in, fy_in) =
    (r.state_bytes += _bytes_of(Tuple(fx_in)) + _bytes_of(Tuple(fy_in)); nothing)
record_linrood_horizontal!(r::_CSCountingRecorder, ::Val{:finish}, fx_out, am, bm, ::Val) =
    (r.state_bytes += _bytes_of(Tuple(fx_out)) + _bytes_of(Tuple(am)) + _bytes_of(Tuple(bm)); nothing)
