"""
    AbstractSurfaceFluxSource

Supertype for all per-tracer surface emission sources consumed by the
surface-flux operator and stored in a [`PerTracerFluxMap`](@ref).

Concrete subtypes:

- [`SurfaceFluxSource`](@ref) — a single static per-cell rate array.
- [`TimeVaryingSurfaceFluxSource`](@ref) — a time series of per-cell
  rate slices plus a sorted `times` vector; the operator linearly
  interpolates to the simulation clock at `apply!` time.

Every subtype carries a `tracer_name :: Symbol` field used by
`flux_for` to key the map.
"""
abstract type AbstractSurfaceFluxSource end

"""
    SurfaceFluxSource{RateT}

A single-tracer surface source: a `tracer_name` plus a `cell_mass_rate`
array supplying model-storage amount added **per cell per second** to the
surface layer.

- `tracer_name :: Symbol` — matches a name in `CellState.tracer_names`.
- `cell_mass_rate :: RateT` — one of:
  - a 2D `(Nx, Ny)` array for structured grids
  - a 1D `(Nc,)` array for face-indexed grids
  - an `NTuple{6}` of 2D `(Nc, Nc)` arrays for cubed-sphere panels
  The rates are already area-integrated. For dry-VMR tracers, file-based
  physical fluxes in kg species/s are converted by the source builder to
  dry-air-equivalent storage units before reaching this struct. The surface
  flux kernel applies `rm_surface += rate × dt` without multiplying by cell
  area.

# Why per-cell rates (not kg/m²/s)

The prognostic tracer is stored per cell, so a per-cell rate × dt is the
natural unit. A per-area variant would need to multiply by cell area before
entering the operator.

# Provenance

The source lives with `SurfaceFluxOperator`; model and simulation layers only
compose and schedule the operator.

# Fields
- `tracer_name :: Symbol`
- `cell_mass_rate :: RateT` — backend-agnostic; `Adapt.adapt` converts
  the array between host and device transparently via
  `Adapt.adapt_structure`.
"""
struct SurfaceFluxSource{RateT, CompT} <: AbstractSurfaceFluxSource
    tracer_name    :: Symbol
    cell_mass_rate :: RateT
    compensation   :: CompT   # Kahan compensation; same shape as cell_mass_rate
end

# Outer constructor: allocate zero compensation matching the rate shape.
SurfaceFluxSource(name::Symbol, rate::RateT) where {RateT} =
    SurfaceFluxSource(name, rate, _alloc_flux_comp(rate))

# Adapt hook: carry both the rate and the compensation to the device.
function Adapt.adapt_structure(to, source::SurfaceFluxSource)
    cell_mass_rate = Adapt.adapt(to, source.cell_mass_rate)
    compensation   = Adapt.adapt(to, source.compensation)
    return SurfaceFluxSource{typeof(cell_mass_rate), typeof(compensation)}(
        source.tracer_name, cell_mass_rate, compensation)
end

# Allocate a zero compensation array matching the shape of a static rate.
_alloc_flux_comp(rate::AbstractArray) = zero(rate)
_alloc_flux_comp(rates::NTuple{6})    = ntuple(p -> zero(rates[p]), Val(6))

# Allocate a zero compensation array matching ONE time-slice of a rate series
# (drops the trailing time dimension). Uses `similar` to match the source's
# array type and backend — preserves device placement for GPU-resident series.
_alloc_flux_comp_from_series(s::AbstractArray{FT}) where {FT} =
    fill!(similar(s, size(s)[1:end-1]...), zero(FT))
_alloc_flux_comp_from_series(s::NTuple{6}) =
    ntuple(p -> _alloc_flux_comp_from_series(s[p]), Val(6))

# =========================================================================
# Temporal reconstruction schemes — how a time-varying source maps its
# stored slices onto the simulation clock. Dispatch-selected; every scheme
# resolves to the SAME `(i0, i1, w0, w1)` two-slice blend the interp kernels
# consume, so adding a scheme never touches the kernels.
# =========================================================================

"""
    AbstractFluxTemporalScheme

How a [`TimeVaryingSurfaceFluxSource`](@ref) reconstructs a per-step rate
from its stored time slices at `t = current_time(meteo)` (and step `dt`).
Concrete schemes implement

    _flux_temporal_weights(scheme, times, t, dt) -> (i0, i1, w0, w1)

returning two bracketing slice indices and blend weights (`w0 + w1 == 1`)
applied as `w0·series[i0] + w1·series[i1]`, together with

    _flux_temporal_segments(scheme, times, t, dt) -> [(i0, i1, w0, w1, frac), …]
    _slices_read_at(scheme, times, τ) -> (first, last)

the pieces of one step `[t, t + dt]`, and the first and last slice index that
an evaluation at the instant `τ` reads (non-decreasing in `τ`; the device
window buffer of a time-varying source holds the slices between those of a
window's start and end). New reconstructions (e.g. mass-flux-integral
matching, higher-order in time) are added as new subtypes and these methods —
no kernel or operator change.
"""
abstract type AbstractFluxTemporalScheme end

"""
    StepwiseFlux()

Piecewise-constant in time: emit the value of the slice *block* containing
`t` (the largest slice time `≤ t`), with no blending. This holds each
3-hourly CAMS value constant across its block — the "change the flux in 3h
blocks" option.
"""
struct StepwiseFlux <: AbstractFluxTemporalScheme end

"""
    LinearInterpFlux()

Linear interpolation between the two bracketing slices, point-evaluated at
the step's `current_time(meteo)`. HEMCO-like treatment of instantaneous
3-hourly fields. Default scheme.
"""
struct LinearInterpFlux <: AbstractFluxTemporalScheme end

"""
    ConservativeMeanFlux()

Window/step integral-conserving: evaluate the linear reconstruction at the
step centre `t + dt/2`. Summed over the substeps of a met window, this
midpoint quadrature reproduces the exact trapezoidal time-integral of a
piecewise-linear flux — i.e. the emitted mass per window matches the source's
own integral, independent of the substep count.
"""
struct ConservativeMeanFlux <: AbstractFluxTemporalScheme end

"""
    _flux_temporal_weights(scheme, times, t, dt) -> (i0, i1, w0, w1)

Resolve the two-slice blend for `scheme` at time `t` (seconds since run
start) with step length `dt` (seconds). All schemes funnel through the same
`(i0, i1, w0, w1)` contract so the interp kernels are scheme-agnostic.
"""
@inline _flux_temporal_weights(::LinearInterpFlux, times, t, dt) =
    _time_interp_bracket(times, t)

@inline _flux_temporal_weights(::ConservativeMeanFlux, times, t, dt) =
    _time_interp_bracket(times, t + dt / 2)

@inline function _flux_temporal_weights(::StepwiseFlux, times, t, dt)
    n = length(times)
    n == 0 && throw(ArgumentError("_flux_temporal_weights: empty times vector"))
    t <= times[1] && return (1, 1, 1.0, 0.0)
    k = searchsortedlast(times, t)   # largest k with times[k] <= t
    return (k, k, 1.0, 0.0)
end

# -------------------------------------------------------------------------
# Exact time-integral over a step [t, t+dt]: split at every flux time knot
# inside the step so each sub-interval is integrated exactly (the source
# reconstruction is smooth — linear or constant — within a knot interval).
# Returns a vector of `(i0, i1, w0, w1, dt_frac)` segments with
# `Σ dt_frac == 1`; the operator applies the two-slice blend with `dt·dt_frac`
# per segment, so the emitted mass reproduces the scheme's intended integral
# regardless of how the step straddles the source cadence.
#
# `LinearInterpFlux` is a *point* interpolation at the step's current time, so
# it is deliberately a single full-step segment (no knot split). `StepwiseFlux`
# and `ConservativeMeanFlux` split at knots and integrate exactly (piecewise
# constant / piecewise linear, respectively).
# -------------------------------------------------------------------------
const _FluxSegment = Tuple{Int, Int, Float64, Float64, Float64}

@inline _flux_temporal_segments(::LinearInterpFlux, times, t, dt) =
    _FluxSegment[(_time_interp_bracket(times, t)..., 1.0)]

_flux_temporal_segments(s::StepwiseFlux, times, t, dt) =
    _knot_split_segments(times, t, dt, m -> _flux_temporal_weights(s, times, m, dt))

_flux_temporal_segments(::ConservativeMeanFlux, times, t, dt) =
    _knot_split_segments(times, t, dt, m -> _time_interp_bracket(times, m))

# Build the knot-split segments of [t, t+dt], evaluating `point_weights(mid)`
# (an (i0,i1,w0,w1) tuple) at each sub-interval midpoint.
function _knot_split_segments(times, t, dt, point_weights)
    t0 = Float64(t); t1 = t0 + Float64(dt)
    segs = _FluxSegment[]
    dt > 0 || (push!(segs, (point_weights(t0)..., 1.0)); return segs)
    lo = t0
    @inbounds for k in eachindex(times)
        tk = Float64(times[k])
        if tk > lo + eps(t1) && tk < t1 - eps(t1)
            mid = 0.5 * (lo + tk)
            push!(segs, (point_weights(mid)..., (tk - lo) / dt))
            lo = tk
        end
    end
    push!(segs, (point_weights(0.5 * (lo + t1))..., (t1 - lo) / dt))
    return segs
end

# -------------------------------------------------------------------------
# The slices one met window reads. Every segment above evaluates the scheme's
# point weights at one time τ inside its step: the step start (linear) or a
# sub-interval midpoint (stepwise, conservative). The slice in effect and both
# bracket indices are non-decreasing in τ, so the slices of all steps inside
# [t0, t1] lie between the first index read at t0 and the last index read at
# t1. The window end is padded because the last step can end slightly after
# t1: the runner's clock is exact Float64, but its step length is the storage
# float type (a Float32 5400/7 s step is 2.3e-8 relative too long), and
# `_knot_split_segments` splits at a slice time inside that overshoot.
# -------------------------------------------------------------------------

# The first and last slice index read at the instant τ.
_slices_read_at(::StepwiseFlux, times, τ) = (k = first(_flux_temporal_weights(StepwiseFlux(), times, τ, 0.0)); (k, k))
_slices_read_at(::Union{LinearInterpFlux, ConservativeMeanFlux}, times, τ) =
    (b = _time_interp_bracket(times, τ); (b[1], b[2]))

# The overshoot is at most eps(Float32)/2 of one step plus the clock's Float64
# rounding; 8 eps(Float32) of the window length and 4 eps(t1) cover it.
_window_read_end(t0::Float64, t1::Float64) = t1 + 8 * eps(Float32) * (t1 - t0) + 4 * eps(abs(t1))

"""
    _flux_window_slices(scheme, times, t0, t1) -> UnitRange{Int}

Slice indices that `_flux_temporal_segments(scheme, times, t, dt)` can reference
(as `i0` or `i1`, including a zero-weight `i1`) for any step `[t, t + dt]`
inside the met window `[t0, t1]` (seconds since the run start): the slices
from the first one read at `t0` to the last one read at the padded window end.

- [`StepwiseFlux`](@ref): the slices in effect over the window.
- [`LinearInterpFlux`](@ref), [`ConservativeMeanFlux`](@ref): from the left
  bracket slice at `t0` to the right bracket slice at the window end.

The range is contiguous and may hold one slice more than the window's steps
read (a slice that starts exactly at `t1`, or the right bracket of a linear
step that starts before it). Every time-varying source copies these slices to
its device `FluxWindowBuffer` before the window's steps
(`stage_surface_flux_window!`).
"""
function _flux_window_slices(scheme::AbstractFluxTemporalScheme, times, t0::Real, t1::Real)
    a, b = Float64(t0), Float64(t1)
    b >= a || throw(ArgumentError("_flux_window_slices: window end $(t1) precedes its start $(t0)"))
    return first(_slices_read_at(scheme, times, a)):last(_slices_read_at(scheme, times, _window_read_end(a, b)))
end

# Slice indices that a list of segments references.
_segment_slices(segments) = minimum(first, segments):maximum(s -> s[2], segments)

"""
    flux_temporal_scheme(name) -> AbstractFluxTemporalScheme

Map a config string to a temporal scheme. `"stepwise"`/`"block"` →
[`StepwiseFlux`](@ref); `"linear"`/`"interp"` → [`LinearInterpFlux`](@ref);
`"conservative"`/`"window_mean"`/`"integral"` → [`ConservativeMeanFlux`](@ref).
"""
function flux_temporal_scheme(name::AbstractString)
    s = lowercase(strip(name))
    s in ("stepwise", "block", "piecewise_constant") && return StepwiseFlux()
    s in ("linear", "interp", "interpolate") && return LinearInterpFlux()
    s in ("conservative", "window_mean", "integral", "mass_conserving") &&
        return ConservativeMeanFlux()
    throw(ArgumentError("unknown surface-flux temporal_scheme \"$(name)\"; " *
        "expected \"stepwise\", \"linear\", or \"conservative\""))
end

# =========================================================================
# FluxWindowBuffer — the slices of the current met window, on the device
# =========================================================================

"""
    FluxWindowBuffer

The time slices of a [`TimeVaryingSurfaceFluxSource`](@ref) that the current
met window reads, held on the compute backend (CPU arrays on the CPU, device
arrays on CUDA or Metal), while the full series stays in host memory. The
surface-flux kernels read rates from this buffer; nothing else changes, so
results equal those of a device-resident series bit for bit.

The buffer is a ring of `capacity` slots: slice `k` goes to slot
`mod1(k, capacity)`, so a contiguous range of at most `capacity` slices fills
distinct slots, and a slice that the next window reads again stays where it
is. `stage_surface_flux_window!` copies the slices of a window
(`_flux_window_slices`) that are not resident yet; the operator stages
the slices of each step the same way, which copies nothing after the window
was staged. The capacity grows to the longest range staged (the largest
window's slice count: 2 or 3 slices for daily or hourly series and hourly met
windows) and never with the run length; it starts at zero, so a source that
never emits allocates no slots.

# Fields
- `slots` — `NTuple` of `(…, capacity)` panel arrays, or one `(…, capacity)`
  array, matching the series without its time dimension.
- `resident :: Vector{Int}` — the slice index held by each slot, `0` if none
  (host).
- `uploads :: Int` — slices copied into this buffer so far.
"""
mutable struct FluxWindowBuffer{B}
    slots    :: B
    resident :: Vector{Int}
    uploads  :: Int
end

# An empty buffer on the series' backend.
FluxWindowBuffer(series) = FluxWindowBuffer(_alloc_slice_slots(series, 0), Int[], 0)

_capacity(buffer::FluxWindowBuffer) = length(buffer.resident)

# Slot of slice `k` (it must be resident).
@inline _buffer_slot(buffer::FluxWindowBuffer, k::Integer) = mod1(k, length(buffer.resident))

# `capacity` slots shaped like one slice of `like` (a series or a buffer's slots).
_alloc_slice_slots(like::AbstractArray, capacity::Integer) =
    similar(like, size(like)[1:end-1]..., capacity)
_alloc_slice_slots(like::Tuple, capacity::Integer) = map(a -> _alloc_slice_slots(a, capacity), like)

# Copy slice `k` of `series` into `slot`: one contiguous block per array (the
# time dimension is the slowest), so a host-to-device copy is one transfer.
function _copy_slice!(slots::AbstractArray, slot::Integer, series::AbstractArray, k::Integer)
    n = length(series) ÷ size(series, ndims(series))
    copyto!(slots, (slot - 1) * n + 1, series, (k - 1) * n + 1, n)
    return slots
end
function _copy_slice!(slots::Tuple, slot::Integer, series::Tuple, k::Integer)
    foreach((a, s) -> _copy_slice!(a, slot, s, k), slots, series)
    return slots
end

"""
    _stage_flux_slices!(buffer, series, ks::UnitRange{Int}) -> buffer

Make slices `ks` of `series` resident in `buffer`, copying only those that are
not. A range longer than the capacity first replaces the slots by `length(ks)`
empty ones on the same backend.
"""
function _stage_flux_slices!(buffer::FluxWindowBuffer, series, ks::UnitRange{Int})
    if length(ks) > _capacity(buffer)
        buffer.slots = _alloc_slice_slots(buffer.slots, length(ks))
        buffer.resident = zeros(Int, length(ks))
    end
    for k in ks
        slot = _buffer_slot(buffer, k)
        buffer.resident[slot] == k && continue
        _copy_slice!(buffer.slots, slot, series, k)
        buffer.resident[slot] = k
        buffer.uploads += 1
    end
    return buffer
end

# Adapt hook: the slots move to the target backend with their contents, so the
# residency map stays valid. Slots already there are shared, with the buffer.
function Adapt.adapt_structure(to, buffer::FluxWindowBuffer)
    slots = Adapt.adapt(to, buffer.slots)
    slots === buffer.slots && return buffer
    return FluxWindowBuffer(slots, copy(buffer.resident), 0)
end

# =========================================================================
# TimeVaryingSurfaceFluxSource — 3-hourly (or any cadence) emission series
# =========================================================================

"""
    TimeVaryingSurfaceFluxSource{RateT, T, S, CompT, BufT}

A single-tracer surface source whose per-cell rate advances through a
time series of slices with the simulation clock. Used to carry the
CAMS/LMDZ 3-hourly diurnal cycle (instead of collapsing it to a monthly
mean), the daily and hourly `cs_native` biosphere drivers, and to match
GeosChem.

- `tracer_name :: Symbol` — matches a name in `CellState.tracer_names`.
- `cell_mass_rate_series :: RateT` — a stack of static rate slices, in
  the same per-cell model-storage units as [`SurfaceFluxSource`](@ref).
  For cubed-sphere this is an `NTuple{6}` of `(Nc, Nc, ntime)` arrays
  (one stacked panel per face). The trailing dimension is time. The series
  stays where it was built (host memory for every loader): `Adapt` does not
  move it, so a multi-year series costs no device memory.
- `times :: T` — slice times in **seconds since run start**, length
  `ntime`, sorted strictly ascending. Stays a **host** `Vector` even
  after Adapt-to-device so the operator can `searchsortedlast` it on the
  host before launching the interpolated kernel.
- `scheme :: S` — an [`AbstractFluxTemporalScheme`](@ref) selecting how the
  stored slices are reconstructed onto the simulation clock
  ([`StepwiseFlux`](@ref), [`LinearInterpFlux`](@ref), or
  [`ConservativeMeanFlux`](@ref)). A singleton, carried through Adapt
  unchanged.
- `compensation :: CompT` — Kahan compensation, one slice in shape, on the
  compute backend.
- `window_buffer :: BufT` — a `FluxWindowBuffer`: the slices the
  current met window reads, on the compute backend. `Adapt` moves it (and
  `compensation`) to the device.

The operator resolves the segments `(i0, i1, w0, w1, frac)` of
`_flux_temporal_segments(scheme, times, current_time(meteo), dt)` on the host,
makes their slices resident in `window_buffer` (the runner has staged the
window's slices before its first step, see `stage_surface_flux_window!`),
then blends the two slices in the kernel. End slices are clamped (constant
extrapolation) outside `[times[1], times[end]]`.
"""
struct TimeVaryingSurfaceFluxSource{RateT, T <: AbstractVector{<:Real},
                                    S <: AbstractFluxTemporalScheme,
                                    CompT, BufT <: FluxWindowBuffer} <: AbstractSurfaceFluxSource
    tracer_name           :: Symbol
    cell_mass_rate_series :: RateT
    times                 :: T
    scheme                :: S
    compensation          :: CompT   # Kahan compensation; shape = one time-slice of series
    window_buffer         :: BufT    # the current window's slices, on the compute backend
end

# Outer constructors: allocate zero compensation and an empty window buffer
# from the series shape.
function TimeVaryingSurfaceFluxSource(name::Symbol, series::RateT,
                                      times::T, scheme::S) where {RateT, T, S}
    isempty(times) && throw(ArgumentError(
        "time-varying surface source $(name) requires at least one time"))
    all(t -> isfinite(t), times) || throw(ArgumentError(
        "time-varying surface source $(name) times must be finite"))
    all(>(0), diff(times)) || throw(ArgumentError(
        "time-varying surface source $(name) times must be strictly increasing"))
    ntime = length(times)
    if series isa AbstractArray
        size(series, ndims(series)) == ntime || throw(DimensionMismatch(
            "time-varying surface source $(name) has $(ntime) times but series trailing dimension $(size(series, ndims(series)))"))
    elseif series isa Tuple
        isempty(series) && throw(ArgumentError(
            "time-varying surface source $(name) series must not be empty"))
        for (panel, values) in pairs(series)
            size(values, ndims(values)) == ntime || throw(DimensionMismatch(
                "time-varying surface source $(name) panel $(panel) has $(size(values, ndims(values))) slices; expected $(ntime)"))
        end
    end
    comp = _alloc_flux_comp_from_series(series)
    buffer = FluxWindowBuffer(series)
    return TimeVaryingSurfaceFluxSource{RateT, T, S, typeof(comp), typeof(buffer)}(
        name, series, times, scheme, comp, buffer)
end

TimeVaryingSurfaceFluxSource(name::Symbol, series, times) =
    TimeVaryingSurfaceFluxSource(name, series, times, LinearInterpFlux())

# Adapt hook: compensation and window buffer go to the device; the series,
# times and scheme stay where they are.
function Adapt.adapt_structure(to, source::TimeVaryingSurfaceFluxSource)
    compensation = Adapt.adapt(to, source.compensation)
    buffer       = Adapt.adapt(to, source.window_buffer)
    return TimeVaryingSurfaceFluxSource{typeof(source.cell_mass_rate_series),
                                        typeof(source.times), typeof(source.scheme),
                                        typeof(compensation), typeof(buffer)}(
        source.tracer_name, source.cell_mass_rate_series, source.times, source.scheme,
        compensation, buffer)
end

"""
    stage_surface_flux_window!(op_or_source, t0, t1)

Copy the time slices that the met window `[t0, t1]` (seconds since the run
start) reads from each time-varying source's host series into its device
`FluxWindowBuffer`: the slices of
`_flux_window_slices(scheme, times, t0, t1)` that are not resident
yet. `DrivenSimulation` calls it on the model's surface-flux operator when a
window becomes current (at construction and after each window advance, on the
task that launches the window's kernels), so the copies are ordered before
the window's first surface-flux kernel. Static sources, other source types and
[`NoSurfaceFlux`](@ref) have nothing to stage. Returns `nothing`.
"""
stage_surface_flux_window!(::AbstractSurfaceFluxSource, t0, t1) = nothing

function stage_surface_flux_window!(src::TimeVaryingSurfaceFluxSource, t0::Real, t1::Real)
    _stage_flux_slices!(src.window_buffer, src.cell_mass_rate_series,
                        _flux_window_slices(src.scheme, src.times, t0, t1))
    return nothing
end

function _check_surface_source_compatibility(state::CubedSphereState,
                                             source::TimeVaryingSurfaceFluxSource)
    tracer_index(state, source.tracer_name) === nothing &&
        throw(ArgumentError("surface source tracer $(source.tracer_name) is not present in model state"))

    series = source.cell_mass_rate_series
    series isa NTuple{6} || throw(ArgumentError(
        "cubed-sphere time-varying surface source $(source.tracer_name) must provide an NTuple{6} " *
        "of panel series, got $(typeof(series))"))

    ntime = length(source.times)
    issorted(source.times) || throw(ArgumentError(
        "time-varying surface source $(source.tracer_name) requires ascending `times`"))

    Hp = state.halo_width
    @inbounds for p in 1:6
        panel = state.air_mass[p]
        expected = (size(panel, 1) - 2Hp, size(panel, 2) - 2Hp, ntime)
        size(series[p]) == expected || throw(ArgumentError(
            "cubed-sphere time-varying surface source $(source.tracer_name) panel $p has shape " *
            "$(size(series[p])) but the expected interior series shape is $(expected)"))
    end
    return nothing
end

"""
    _time_interp_bracket(times, t) -> (i0, i1, w0, w1)

Return the two bracketing slice indices `i0 ≤ i1` and the linear
interpolation weights `w0, w1` (with `w0 + w1 == 1`) for time `t`
against the sorted `times` vector. Clamps to the end slices (constant
extrapolation) when `t` is outside `[times[1], times[end]]`:

- `t ≤ times[1]`   → `(1, 1, 1, 0)`
- `t ≥ times[end]` → `(end, end, 1, 0)`
- otherwise        → `(k, k+1, 1-frac, frac)` where `k =
  searchsortedlast(times, t)` and `frac` is the position of `t` in
  `[times[k], times[k+1]]`.
"""
@inline function _time_interp_bracket(times::AbstractVector, t::Real)
    n = length(times)
    n == 0 && throw(ArgumentError("_time_interp_bracket: empty times vector"))
    n == 1 && return (1, 1, 1.0, 0.0)
    t <= times[1]   && return (1, 1, 1.0, 0.0)
    t >= times[end] && return (n, n, 1.0, 0.0)
    k = searchsortedlast(times, t)          # times[k] <= t < times[k+1]
    t0 = Float64(times[k]); t1 = Float64(times[k + 1])
    span = t1 - t0
    frac = span > 0 ? (Float64(t) - t0) / span : 0.0
    return (k, k + 1, 1.0 - frac, frac)
end

# =========================================================================
# Surface-slice helpers used by the operator kernel shell.
# =========================================================================

"""
    _surface_shape(rm) -> Tuple

Return the expected shape of a surface source's `cell_mass_rate` for the
given tracer mass array `rm`. For 3D structured `(Nx, Ny, Nz)` tracers,
this is `(Nx, Ny)`; for 2D face-indexed `(Nc, Nz)` tracers, `(Nc,)`.
"""
@inline _surface_shape(rm::AbstractArray{<:Any, 3}) = (size(rm, 1), size(rm, 2))
@inline _surface_shape(rm::AbstractArray{<:Any, 2}) = (size(rm, 1),)

"""
    _check_surface_source_compatibility(state, source)

Validate that `source.tracer_name` is present in `state`, and that
`size(source.cell_mass_rate)` matches the state's surface slice shape.
Throws `ArgumentError` on mismatch. Used at DrivenSimulation construction
and at SurfaceFluxOperator construction.
"""
function _check_surface_source_compatibility(state, source::SurfaceFluxSource)
    tracer_index(state, source.tracer_name) === nothing &&
        throw(ArgumentError("surface source tracer $(source.tracer_name) is not present in model state"))
    rm = get_tracer(state, source.tracer_name)
    size(source.cell_mass_rate) == _surface_shape(rm) ||
        throw(ArgumentError("surface source $(source.tracer_name) has shape $(size(source.cell_mass_rate)) but tracer surface shape is $(_surface_shape(rm))"))
    return nothing
end

function _check_surface_source_compatibility(state::CubedSphereState, source::SurfaceFluxSource)
    tracer_index(state, source.tracer_name) === nothing &&
        throw(ArgumentError("surface source tracer $(source.tracer_name) is not present in model state"))

    rates = source.cell_mass_rate
    rates isa NTuple{6} || throw(ArgumentError(
        "cubed-sphere surface source $(source.tracer_name) must provide an NTuple{6} " *
        "of panel rates, got $(typeof(rates))"))

    Hp = state.halo_width
    @inbounds for p in 1:6
        panel = state.air_mass[p]
        expected = (size(panel, 1) - 2Hp, size(panel, 2) - 2Hp)
        size(rates[p]) == expected || throw(ArgumentError(
            "cubed-sphere surface source $(source.tracer_name) panel $p has shape $(size(rates[p])) " *
            "but the interior panel shape is $(expected)"))
    end
    return nothing
end
