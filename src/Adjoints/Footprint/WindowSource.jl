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
#
# Two kinds of records outlive their windows' leases: a Stride resident tail
# (its windows are released after the first re-recorded head segment; the
# tail has been walked by then) and the tape a runtime holds for
# `AirMassTape` (released when the call ends). Both are harmless while every
# window stays resident. The setup checks (`require_adjoint_support`,
# `_tape_identity`) acquire every window once.
#
# The data of a leased window is read-only to the replay, except that a reset
# refreshes the halos of its target in place (a function of the interior).
#
# `ResidentCSWindowSource` holds every window of the run, built once from the
# per-step inputs of the adjoint entry points; releasing one of its windows
# does nothing.
# ---------------------------------------------------------------------------

abstract type AbstractCSWindowSource end

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
it with `release_window!`. `met` is valid until then.
"""
struct CSWindowLease{M <: CSWindowMet}
    met :: M
end

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
        window_fluxes = fluxes[steps]
        met = CSWindowMet(w, steps,
                          all(f -> f === first(window_fluxes), window_fluxes) ? first(window_fluxes) :
                                                                                window_fluxes,
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
