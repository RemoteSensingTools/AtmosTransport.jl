# ---------------------------------------------------------------------------
# Window sources: the meteorology the replay reads, one met window at a time.
#
# The replay (`StepReplay.jl`) reads the met inputs of a step through the
# met of the step's window (`CSWindowMet`): its face fluxes, convection
# forcing, diffusion operator and layer thickness, and air-mass reset
# targets. It acquires a window from its source at the window's first step
# (or at a checkpoint restart inside the window) and releases it when the
# window's data can no longer be read:
#
#   * a window only untaped runs have read is released when the replay
#     acquires the next one;
#   * a recording run retains its windows (its records reference their met)
#     until the driver releases them, after `_release_records!`
#     (`_release_replay_windows!`, also in the drivers' `finally` blocks).
#     The Stride/MetWindow driver checks that every window a segment's
#     records reference is still current (`require_current`) before it walks
#     the segment.
#
# Two kinds of records outlive their windows' leases: a Stride resident tail
# (its windows are released after the first re-recorded head segment; the
# tail has been walked by then) and the tape a runtime holds for
# `AirMassTape` (released when the call ends). Both are harmless while every
# window stays resident, and a streamed source rejects both (with Full and
# fixed-K Stride) before the forward run (`_require_tape_supported`).
#
# The data of a leased window is read-only to the replay, except that a reset
# refreshes the halos of its target in place (a function of the interior). A
# refilled slot therefore holds a target's halos as captured where the
# resident target already holds refreshed ones; the one read before the
# refresh (a `:preserve_vmr` reset record) is multiplied by an adjoint state
# the next halo record has zeroed (see `_window_reset_targets` in
# Models.Runner), so streamed and resident results agree bit for bit.
#
# `ResidentCSWindowSource` holds every window of the run, built once from the
# per-step inputs of the adjoint entry points; releasing one of its windows
# does nothing. `HostCSWindowSource` keeps every window in host memory and a
# few on the replay's backend (device slots), refilled as the replay moves.
# ---------------------------------------------------------------------------

"""
    AbstractCSWindowSource

The met windows a cubed-sphere adjoint replay reads (`met_source` of the
adjoint entry points). A source implements `window_count`, `window_steps`,
`window_layout`, `acquire_window!` and `release_window!`; `prefetch_window!`,
`close` and `streaming` have defaults (no-op, no-op, `false`). The entry
points build a `ResidentCSWindowSource` from their per-step arguments when no
source is given.
"""
abstract type AbstractCSWindowSource end

"""
    AbstractStreamedCSWindowSource <: AbstractCSWindowSource

A window source that keeps a bounded number of met windows on the replay's
backend and refills them (`streaming(source) == true`). Only
`MetWindowCheckpoint` tapes, which record one met window per segment, run on
it: `FullCheckpoint`, `StrideCheckpoint`, `AirMassTape` and an unconstrained
`AutoTape()` are rejected before the forward run.
"""
abstract type AbstractStreamedCSWindowSource <: AbstractCSWindowSource end

"""
    CSWindowMet

Everything the replay reads for one met window: `window` (the absolute
schedule window), `steps` (the window's steps of the run), `fluxes` (one
`CubedSphereFaceFluxState` for window-constant fluxes, or one per step of
`steps` when they change within the window), `convection_forcing` (or
`nothing`), `diffusion_op` (`NoDiffusion()` or the window's operator),
`layer_thickness` (six panels [m] or `nothing`), and the air-mass reset
targets `start_air_mass` and `end_air_mass` (six halo-padded panels, or
`nothing` without the reset).
"""
struct CSWindowMet{FX, CF, DO, DZ, MS, ME}
    window             :: Int
    steps              :: UnitRange{Int}
    fluxes             :: FX
    convection_forcing :: CF
    diffusion_op       :: DO
    layer_thickness    :: DZ
    start_air_mass     :: MS
    end_air_mass       :: ME
end

# The face fluxes of step `s` of the window.
_fluxes_at(met::CSWindowMet, s::Int) = _fluxes_at(met.fluxes, met, s)
_fluxes_at(fluxes::CubedSphereFaceFluxState, _met, _s) = fluxes
_fluxes_at(fluxes::AbstractVector, met, s) = fluxes[s - first(met.steps) + 1]

"""
    CSWindowLease

One acquisition of a window from its source (`acquire_window!`), returned to
it with `release_window!`. `met` is valid until then. A streamed source's
lease also names the device slot holding `met` and the slot's fill count
(`generation`) at acquisition: `require_current(lease)` checks that the slot
has not been refilled since.
"""
struct CSWindowLease{M <: CSWindowMet, S}
    met        :: M
    slot       :: S        # the device slot holding `met`, or `nothing` (resident)
    generation :: Int
end
CSWindowLease(met::CSWindowMet) = CSWindowLease(met, nothing, 0)

"""
    StaleWindowSlotError

The device slot of a window lease was refilled with another window (or freed
by `close`) after the lease was acquired: whatever still reads the lease's
met would read another window. `require_current` throws it.
"""
struct StaleWindowSlotError <: Exception
    msg::String
end
Base.showerror(io::IO, e::StaleWindowSlotError) = print(io, "StaleWindowSlotError: ", e.msg)

# ---------------------------------------------------------------------------
# The interface (the methods every source shares)
# ---------------------------------------------------------------------------

"""
    window_steps(source) -> Vector{Int}

The number of steps of each window of `source` (the replay checks them
against its schedule for the steps it runs).
"""
function window_steps end

"""
    window_layout(source) -> CSWindowMet

A met with the array types and shapes of `source`'s windows (its contents are
not a window's): what the replay is built from (the types of its reset-target
fields, whether it needs a diffusion workspace) and what its setup checks
read (operator types, shapes against the replay's templates), without
acquiring a window. A streamed source's windows must all share the layout's
piece types (`HostCSWindowSource` checks it when it is built).
"""
function window_layout end

"""
    prefetch_window!(source, w)

A hint that window `w` will be acquired next. The replay gives it after each
window it acquires in an untaped run, and the Stride/MetWindow driver before
it re-records a segment. Every source loads synchronously at
`acquire_window!` for now, so it does nothing.
"""
prefetch_window!(::AbstractCSWindowSource, ::Int) = nothing

"""
    require_current(lease) -> lease

Throw a `StaleWindowSlotError` if the device slot of `lease` has been
refilled (or freed) since it was acquired; resident leases are always current.
"""
require_current(lease::CSWindowLease) = (_require_current_slot(lease.slot, lease); lease)
_require_current_slot(::Nothing, _lease) = nothing

"""
    close(source::AbstractCSWindowSource)

Free what `source` holds on the device (a streamed source's slots); no
window can be acquired afterwards. Does nothing for a resident source.
"""
Base.close(::AbstractCSWindowSource) = nothing

"""
    streaming(source) -> Bool

Whether `source` keeps only a bounded number of windows on the replay's
backend (`AbstractStreamedCSWindowSource`): the tape guards then accept
`MetWindowCheckpoint` only, and Lin–Rood tapes are sized from one window.
"""
streaming(::AbstractCSWindowSource) = false
streaming(::AbstractStreamedCSWindowSource) = true

# ---------------------------------------------------------------------------
# ResidentCSWindowSource
# ---------------------------------------------------------------------------

"""
    ResidentCSWindowSource(am_steps, bm_steps, cm_steps, step_windows; flux_scale,
                           convection_forcing, diffusion_op, layer_thickness,
                           start_air_mass, end_air_mass, layer_thickness_like,
                           air_mass_like)

Every met window of a run in memory: the per-step inputs of the adjoint entry
points (face fluxes, convection forcing, diffusion operator and layer
thickness: one value or one per step) grouped by the window of each step
(`step_windows`, the windows `1:W` of the run's steps), and each window's
reset targets (`start_air_mass[w]`, `end_air_mass[w]`, or `nothing`). Each
window's met is built and validated once: the convection forcing, diffusion
operator and layer thickness are constant within a window, the layer
thickness has the panel shapes, element type and backend of
`layer_thickness_like` (when given), and the reset targets the shapes and
element types of `air_mass_like`.
"""
struct ResidentCSWindowSource{V <: AbstractVector} <: AbstractCSWindowSource
    windows :: V
end

function ResidentCSWindowSource(am, bm, cm, step_windows::AbstractVector{Int};
                                flux_scale, convection_forcing = nothing,
                                diffusion_op = NoDiffusion(), layer_thickness = nothing,
                                start_air_mass = nothing, end_air_mass = nothing,
                                layer_thickness_like = nothing, air_mass_like)
    fluxes = _replay_fluxes(am, bm, cm, length(step_windows), flux_scale)
    windows = map(enumerate(_window_ranges(step_windows))) do (w, steps)
        met = CSWindowMet(w, steps, _window_fluxes(fluxes[steps]),
                          _window_value(convection_forcing, steps, w, "convection_forcing",
                                        "the convection forcing"),
                          _window_value(diffusion_op, steps, w, "diffusion_op", "the diffusion operator"),
                          _window_value(layer_thickness, steps, w, "diffusion_layer_thickness",
                                        "the diffusion layer thickness"),
                          _window_target(start_air_mass, w), _window_target(end_air_mass, w))
        _check_window_met(met, layer_thickness_like, air_mass_like)
        met
    end
    return ResidentCSWindowSource(windows)
end

# The interface of a source: its number of met windows (a method of
# `MetDrivers.window_count`) and the acquisition and release of one window.
window_count(src::ResidentCSWindowSource) = length(src.windows)
window_steps(src::ResidentCSWindowSource) = [length(met.steps) for met in src.windows]
window_layout(src::ResidentCSWindowSource) = first(src.windows)

function acquire_window!(src::ResidentCSWindowSource, w::Int)
    1 <= w <= window_count(src) || throw(ArgumentError(
        "window $w outside the source's windows 1:$(window_count(src))"))
    return CSWindowLease(src.windows[w])
end
release_window!(::ResidentCSWindowSource, ::CSWindowLease) = nothing

# The steps of each window `1:W` (consecutive steps of one window).
function _window_ranges(step_windows::AbstractVector{Int})
    ranges = UnitRange{Int}[]
    for (s, w) in enumerate(step_windows)
        w > length(ranges) ? push!(ranges, s:s) : (ranges[w] = first(ranges[w]):s)
    end
    return ranges
end

# Face fluxes of steps `1:nsteps`. The production palindrome reads unscaled
# fluxes (`flux_scale = 1`), so a `flux_scale ≠ 1` of the low-level API scales
# them once here; the tape records reference these arrays for the reverse
# pass. Consecutive steps given the same arrays (a met window's fluxes, which
# `cs_adjoint_inputs` shares across the window's steps) share one flux state,
# so a `flux_scale ≠ 1` scales each set once.
_step_fluxes(am, bm, cm, scale) =
    isone(scale) ? CubedSphereFaceFluxState{DryBasis}(am, bm, cm) :
    CubedSphereFaceFluxState{DryBasis}(map(a -> scale .* a, am), map(b -> scale .* b, bm),
                                       map(c -> scale .* c, cm))

function _replay_fluxes(am, bm, cm, nsteps::Int, scale)
    previous = Ref{Any}(nothing)
    return map(1:nsteps) do s
        shared = s > 1 && am[s] === am[s - 1] && bm[s] === bm[s - 1] && cm[s] === cm[s - 1]
        previous[] = shared ? previous[] : _step_fluxes(am[s], bm[s], cm[s], scale)
    end
end

# A window's flux states: one when its steps share it, else one per step.
_window_fluxes(states::AbstractVector) = all(f -> f === first(states), states) ? first(states) : states

# The value of a per-step input in window `w`: one per met window, as
# production refreshes the convection forcing, the diffusion operator and its
# layer thickness at the window's first step only.
function _window_value(values::AbstractVector, steps::UnitRange{Int}, w::Int, name, what)
    for s in (first(steps) + 1):last(steps)
        values[s] === values[s - 1] || throw(ArgumentError(
            "$name changes at step $s inside met window $w; production refreshes $what once per window"))
    end
    return values[first(steps)]
end
_window_value(value, _steps, _w, _name, _what) = value

_window_target(::Nothing, ::Int) = nothing
function _window_target(targets, w::Int)
    w <= length(targets) || throw(ArgumentError(
        "the schedule has air-mass reset targets for $(length(targets)) windows; window $w needs one"))
    target = targets[w]
    target isa NTuple{6, AbstractArray} || throw(ArgumentError("air-mass reset targets require six panels"))
    return target
end

function _check_window_met(met::CSWindowMet, layer_thickness_like, air_mass_like)
    dz = met.layer_thickness
    if layer_thickness_like !== nothing && dz !== nothing
        for (src, dst) in zip(dz, layer_thickness_like)
            size(src) == size(dst) && eltype(src) === eltype(dst) &&
                typeof(get_backend(src)) === typeof(get_backend(dst)) || throw(ArgumentError(
                "diffusion_layer_thickness at step $(first(met.steps)): panels of $(typeof(src)) with shape " *
                "$(size(src)); expected $(eltype(dst)) panels of shape $(size(dst)) on the " *
                "state's backend"))
        end
    end
    for panels in (met.start_air_mass, met.end_air_mass)
        panels === nothing && continue
        all(p -> size(panels[p]) == size(air_mass_like[p]) &&
            eltype(panels[p]) === eltype(air_mass_like[p]), 1:6) ||
            throw(ArgumentError("air-mass reset targets must match the initial air-mass panel shapes and element types"))
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Copies of a window's met: Adapt traverses each piece (flux state,
# convection forcing, diffusion operator, panel tuples) through the
# `Adapt.adapt_structure` methods that move a model to the GPU, so every
# array a piece holds is reached in the same order for a host copy, a device
# slot and the list of its arrays.
# ---------------------------------------------------------------------------

# Fresh host arrays.
struct _HostCopy end
Adapt.adapt_storage(::_HostCopy, a::AbstractArray) = Array(a)

# Fresh arrays on the backend of `like`, with the contents of the source.
struct _BackendCopy{A <: AbstractArray}
    like :: A
end
Adapt.adapt_storage(c::_BackendCopy, a::AbstractArray) = copyto!(similar(c.like, eltype(a), size(a)), a)

# The arrays of a piece, in traversal order. Adapt does not walk into a
# vector of objects (a per-step flux vector): pass a tuple of them.
struct _ArrayLeaves
    arrays :: Vector{Any}
end
Adapt.adapt_storage(c::_ArrayLeaves, a::AbstractArray{<:Number}) = (push!(c.arrays, a); a)
Adapt.adapt_storage(::_ArrayLeaves, a::AbstractArray) = throw(ArgumentError(
    "the arrays of a $(typeof(a)) are not reached through Adapt; pass a tuple of its elements"))

_host_copy(x) = Adapt.adapt(_HostCopy(), x)
_array_leaves(x) = (leaves = _ArrayLeaves(Any[]); Adapt.adapt(leaves, x); leaves.arrays)

"""
    _host_window_met(w, steps, am, bm, cm; convection_forcing, diffusion_op,
                     layer_thickness, start_air_mass, end_air_mass) -> CSWindowMet

Window `w` (steps `steps` of the run) in host memory: copies of the window's
per-step face fluxes `am[k], bm[k], cm[k]` (consecutive steps given the same
arrays share one copy, as in `ResidentCSWindowSource`), its convection
forcing, diffusion operator and layer thickness. The reset targets are taken
as given (the capture keeps them in host memory, shared between windows
where they are equal bit for bit).
"""
function _host_window_met(w::Int, steps::UnitRange{Int}, am, bm, cm; convection_forcing,
                          diffusion_op, layer_thickness, start_air_mass, end_air_mass)
    states = _replay_fluxes(am, bm, cm, length(steps), true)
    copies = IdDict{Any, Any}()
    fluxes = _window_fluxes([get!(() -> _host_copy(f), copies, f) for f in states])
    return CSWindowMet(w, steps, fluxes, _host_copy(convection_forcing), _host_copy(diffusion_op),
                       _host_copy(layer_thickness), start_air_mass, end_air_mass)
end

# The distinct flux states of a window, in step order, and each step's state
# (none for window-constant fluxes).
_flux_states(fluxes::CubedSphereFaceFluxState) = Any[fluxes]
_flux_states(fluxes::AbstractVector) = unique(objectid, fluxes)
_flux_index(::CubedSphereFaceFluxState, _states) = Int[]
_flux_index(fluxes::AbstractVector, states) = [findfirst(x -> x === f, states) for f in fluxes]

# Every piece of a window's met, as Adapt traverses it.
_met_pieces(met::CSWindowMet) = (Tuple(_flux_states(met.fluxes)), met.convection_forcing, met.diffusion_op,
                                 met.layer_thickness, met.start_air_mass, met.end_air_mass)

# Free the arrays of met pieces their caller owns (the capture's device copies
# of a window, once the window is in host memory).
_free_met_arrays!(x) = (foreach(_free_tape_array!, _array_leaves(x)); nothing)

# Bytes of the distinct arrays of `x` not yet in `seen`.
function _new_array_bytes!(seen::IdDict{Any, Nothing}, x)
    bytes = 0
    for a in _array_leaves(x)
        haskey(seen, a) && continue
        seen[a] = nothing
        bytes += sizeof(eltype(a)) * length(a)
    end
    return bytes
end

# The host memory a source may hold: `budget` bytes, or by default the host
# (or cgroup) memory limit less AutoTape's host margin: a period that could
# not fit even an otherwise idle host fails before its capture fills memory.
# The current headroom (`_host_tape_budget`) is no default: it counts only
# inactive page cache as reclaimable, which on a host whose cache fills its
# cgroup leaves next to nothing. Shared hosts set the budget explicitly.
function _host_window_budget(budget)
    budget === nothing || return Int(budget)
    limit = last(_host_tape_memory())
    return max(0, limit - max(cld(limit, 10), 8 << 30))
end

function _require_host_budget(bytes::Integer, budget::Integer, what)
    bytes <= budget || throw(ArgumentError(
        "$what needs $(round(bytes / 2^30; digits = 2)) GiB of host memory; host_budget_bytes is " *
        "$(round(budget / 2^30; digits = 2)) GiB. Shorten the period, raise host_budget_bytes, or keep " *
        "the windows on the device (residency = :device)."))
    return nothing
end

# ---------------------------------------------------------------------------
# HostCSWindowSource: every window in host memory, `device_slots` on the
# replay's backend
# ---------------------------------------------------------------------------

# A window of a host source: its met (host arrays), the distinct flux states
# of its steps with each step's state (`index`, empty for window-constant
# fluxes) and the arrays a fill copies, in slot order.
struct _HostWindow{M <: CSWindowMet}
    met          :: M
    index        :: Vector{Int}
    state_leaves :: Vector{Vector{Any}}
    leaves       :: Vector{Any}          # convection forcing, diffusion operator, layer thickness, targets
end

function _HostWindow(met::CSWindowMet)
    states = _flux_states(met.fluxes)
    return _HostWindow(met, _flux_index(met.fluxes, states), [_array_leaves(f) for f in states],
                       _other_leaves(met))
end

_other_leaves(met::CSWindowMet) = _array_leaves((met.convection_forcing, met.diffusion_op, met.layer_thickness,
                                                  met.start_air_mass, met.end_air_mass))

# One device slot: buffers with the layout of the source's windows (from its
# first window), the window they hold (`0`: none), how often they have been
# filled (`generation`), the leases on it and when it was last acquired.
mutable struct _WindowSlot{F, CF, DO, DZ, MS, ME}
    const fluxes             :: Vector{F}
    const convection_forcing :: CF
    const diffusion_op       :: DO
    const layer_thickness    :: DZ
    const start_air_mass     :: MS
    const end_air_mass       :: ME
    const state_leaves       :: Vector{Vector{Any}}
    const leaves             :: Vector{Any}
    window     :: Int
    generation :: Int
    refcount   :: Int
    last_use   :: Int
    met        :: Any       # the met of `window`
end

function _WindowSlot(first_window::_HostWindow, nstates::Int, like::AbstractArray)
    met = first_window.met
    device = _BackendCopy(like)
    state = first(_flux_states(met.fluxes))
    fluxes = [Adapt.adapt(device, state) for _ in 1:nstates]
    slot = _WindowSlot(fluxes, Adapt.adapt(device, met.convection_forcing), Adapt.adapt(device, met.diffusion_op),
                       Adapt.adapt(device, met.layer_thickness), Adapt.adapt(device, met.start_air_mass),
                       Adapt.adapt(device, met.end_air_mass), [_array_leaves(f) for f in fluxes],
                       Any[], 0, 0, 0, 0, nothing)
    append!(slot.leaves, _array_leaves((slot.convection_forcing, slot.diffusion_op, slot.layer_thickness,
                                        slot.start_air_mass, slot.end_air_mass)))
    return slot
end

_slot_met(slot::_WindowSlot, w::Int, steps::UnitRange{Int}, fluxes) =
    CSWindowMet(w, steps, fluxes, slot.convection_forcing, slot.diffusion_op, slot.layer_thickness,
                slot.start_air_mass, slot.end_air_mass)

function _require_current_slot(slot::_WindowSlot, lease::CSWindowLease)
    slot.generation == lease.generation || throw(StaleWindowSlotError(
        "window $(lease.met.window) was leased from fill $(lease.generation) of its device slot; the slot " *
        (slot.window == 0 ? "has been freed" : "now holds window $(slot.window) (fill $(slot.generation))") *
        ". A record or met reference outlived its window's lease."))
    return nothing
end

"""
    HostCSWindowSource(windows; device_like, device_slots = 2, host_budget_bytes = nothing)

A streamed window source: every met window of a run in host memory
(`windows`, one `CSWindowMet` of host arrays per window `1:W`, their steps
consecutive from step 1), and `device_slots` windows on the backend of
`device_like` (the replay's: the initial air mass). `acquire_window!(source,
w)` returns the slot holding window `w`, or refills the least recently used
slot without a lease from the host copy (one copy per array, in place;
`generation` counts the fills); `release_window!` synchronizes the backend
when a slot's last lease ends, so a refill never overwrites data a queued
kernel still reads. Results are bitwise those of the resident windows: the
slots hold copies of the same arrays, and the replay reads them through the
same operators.

`cs_adjoint_inputs(sim; residency = :host)` builds it. Every window must have
the layout of the first: the types of its pieces and the shapes of their
arrays (its flux states, one per window or one per step, those of the first
window's); a slot holds as many flux states as the window with the most. The
diffusion operator of every window is validated as the entry points validate
a per-step one. With `host_budget_bytes`, the host bytes of the windows (each
array once) must fit it. The source is reusable across calls (each 4D-Var
evaluation of an optimization); `close(source)` frees its slots. Arrays an
operator holds outside its `Adapt` traversal (the host staging copy of a
window Kz cache's panels) stay with each window's host copy, are not copied
into the slots (the replay does not read them) and are not counted in
`host_bytes` (the host bytes of the windows' arrays, each counted once).
"""
mutable struct HostCSWindowSource{S <: _WindowSlot} <: AbstractStreamedCSWindowSource
    const windows    :: Vector{_HostWindow}
    const slots      :: Vector{S}
    const host_bytes :: Int
    clock            :: Int
    closed           :: Bool
end

function HostCSWindowSource(windows::AbstractVector{<:CSWindowMet}; device_like::AbstractArray,
                            device_slots::Integer = 2, host_budget_bytes = nothing)
    isempty(windows) && throw(ArgumentError("a window source needs at least one window"))
    device_slots >= 1 || throw(ArgumentError("device_slots must be >= 1; got $device_slots"))
    hosts = _HostWindow[_HostWindow(met) for met in windows]
    seen = IdDict{Any, Nothing}()
    bytes = 0
    for (w, host) in enumerate(hosts)
        met = host.met
        met.window == w || throw(ArgumentError("window $w of the source is numbered $(met.window)"))
        first(met.steps) == (w == 1 ? 1 : last(hosts[w - 1].met.steps) + 1) && !isempty(met.steps) ||
            throw(ArgumentError("window $w has steps $(met.steps); the windows' steps must be consecutive from step 1"))
        all(a -> get_backend(a) isa KACPU, Iterators.flatten((host.leaves, host.state_leaves...))) ||
            throw(ArgumentError("window $w of a HostCSWindowSource holds device arrays; pass host copies " *
                                "(cs_adjoint_inputs(sim; residency = :host))"))
        _check_host_layout(host, first(hosts), w)
        _validate_cs_diffusion_inputs(met.diffusion_op, met.layer_thickness, 1)
        bytes += _new_array_bytes!(seen, _met_pieces(met))
    end
    host_budget_bytes === nothing || _require_host_budget(bytes, host_budget_bytes, "the met windows")
    nstates = maximum(host -> length(host.state_leaves), hosts)
    slots = [_WindowSlot(first(hosts), nstates, device_like) for _ in 1:device_slots]
    return HostCSWindowSource(hosts, slots, bytes, 0, false)
end

# A window fills the slot buffers built from the first window: the same piece
# types and array shapes.
function _check_host_layout(host::_HostWindow, first_window::_HostWindow, w::Int)
    met, ref = host.met, first_window.met
    ref_state = first(_flux_states(ref.fluxes))
    states = _flux_states(met.fluxes)
    ok = all(f -> typeof(f) === typeof(ref_state), states) &&
         all(leaves -> _same_shapes(leaves, first(first_window.state_leaves)), host.state_leaves) &&
         typeof((met.convection_forcing, met.diffusion_op, met.layer_thickness, met.start_air_mass,
                 met.end_air_mass)) ===
         typeof((ref.convection_forcing, ref.diffusion_op, ref.layer_thickness, ref.start_air_mass,
                 ref.end_air_mass)) &&
         _same_shapes(host.leaves, first_window.leaves)
    ok || throw(ArgumentError(
        "window $w of the source differs in layout from window 1 (types of its flux states, convection " *
        "forcing, diffusion operator, layer thickness or reset targets, or the shapes of their arrays); " *
        "every window fills the same device slots"))
    return nothing
end
_same_shapes(a::Vector{Any}, b::Vector{Any}) =
    length(a) == length(b) && all(i -> size(a[i]) == size(b[i]) && eltype(a[i]) === eltype(b[i]), eachindex(a))

window_count(src::HostCSWindowSource) = length(src.windows)
window_steps(src::HostCSWindowSource) = [length(host.met.steps) for host in src.windows]

function window_layout(src::HostCSWindowSource)
    _require_source_open(src)
    slot = first(src.slots)
    return _slot_met(slot, 0, 1:0, first(slot.fluxes))
end

function acquire_window!(src::HostCSWindowSource, w::Int)
    _require_source_open(src)
    1 <= w <= window_count(src) || throw(ArgumentError(
        "window $w outside the source's windows 1:$(window_count(src))"))
    src.clock += 1
    i = findfirst(slot -> slot.window == w, src.slots)
    slot = i === nothing ? _refill!(src, w) : src.slots[i]
    slot.refcount += 1
    slot.last_use = src.clock
    return CSWindowLease(slot.met, slot, slot.generation)
end

# Fill the least recently used slot without a lease with window `w`. The slot
# counts as empty while it is written, so a failed copy leaves no window.
function _refill!(src::HostCSWindowSource, w::Int)
    free = filter(slot -> slot.refcount == 0, src.slots)
    isempty(free) && throw(ArgumentError(
        "all $(length(src.slots)) device slots of the window source are leased (windows " *
        "$(join([slot.window for slot in src.slots], ", "))); window $w needs one: raise device_slots"))
    slot = argmin(slot -> slot.last_use, free)
    host = src.windows[w]
    slot.window = 0
    slot.generation += 1
    slot.met = nothing
    for (dst, src_leaves) in zip(slot.state_leaves, host.state_leaves)
        foreach(copyto!, dst, src_leaves)
    end
    foreach(copyto!, slot.leaves, host.leaves)
    met = host.met
    fluxes = isempty(host.index) ? first(slot.fluxes) : [slot.fluxes[k] for k in host.index]
    slot.met = _slot_met(slot, w, met.steps, fluxes)
    slot.window = w
    return slot
end

function release_window!(src::HostCSWindowSource, lease::CSWindowLease)
    slot = lease.slot
    slot.refcount > 0 && slot.generation == lease.generation || throw(ArgumentError(
        "release of a lease of window $(lease.met.window) that its source does not hold"))
    # The version-1 fence: the backend has finished every queued use of the
    # slot before the slot can be refilled.
    slot.refcount == 1 && synchronize(get_backend(first(first(slot.state_leaves))))
    slot.refcount -= 1
    return nothing
end

function Base.close(src::HostCSWindowSource)
    src.closed && return nothing
    leased = [slot.window for slot in src.slots if slot.refcount > 0]
    isempty(leased) || throw(ArgumentError("close of a window source with leased windows $(leased)"))
    synchronize(get_backend(first(first(first(src.slots).state_leaves))))
    for slot in src.slots
        slot.window = 0
        slot.generation += 1
        slot.met = nothing
        foreach(_free_tape_array!, slot.leaves)
        foreach(leaves -> foreach(_free_tape_array!, leaves), slot.state_leaves)
    end
    src.closed = true
    return nothing
end

_require_source_open(src::HostCSWindowSource) =
    src.closed && throw(ArgumentError("the window source is closed"))
