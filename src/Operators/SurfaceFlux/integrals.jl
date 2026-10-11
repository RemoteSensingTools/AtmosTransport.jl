# ===========================================================================
# Emission totals of a surface source, computed from the same temporal
# reconstruction the surface-flux operator applies (`_flux_temporal_weights`,
# `_flux_temporal_segments` in `sources.jl`). The runner's start-up log and
# the budget diagnostics use these functions instead of re-deriving the
# per-scheme arithmetic.
#
# Notation. A source carries per-cell model-storage rates r_c (kg dry-air
# equivalent s⁻¹ = physical kg species s⁻¹ × M_air / M_species). A
# time-varying source stores slices r_{c,k} at times t_k (seconds since the
# run start). The global rate of slice k is R_k = Σ_c r_{c,k}. Every scheme
# blends at most two slices linearly, so the global rate of a blend
# w₀ r_{c,i₀} + w₁ r_{c,i₁} is w₀ R_{i₀} + w₁ R_{i₁}. With Float64 storage the
# per-cell operator and these global totals differ only by summation order;
# Float32 storage adds the operator's rounding of each increment.
# ===========================================================================

"""
    total_emission_rate(src::AbstractSurfaceFluxSource, t) -> Float64

Global emission rate of `src` at the instant `t` (seconds since the run start):
the per-cell model-storage rates summed over all cells, in kg dry-air
equivalent s⁻¹ (physical kg species s⁻¹ × `DRY_AIR_MOLAR_MASS / M_species`).

- [`SurfaceFluxSource`](@ref): ``R = \\sum_c r_c``, independent of `t`.
- [`TimeVaryingSurfaceFluxSource`](@ref): the scheme's reconstruction at `t`,
  ``R(t) = w_0 R_{i_0} + w_1 R_{i_1}`` with ``R_k = \\sum_c r_{c,k}`` and
  `(i₀, i₁, w₀, w₁) = _flux_temporal_weights(scheme, times, t, 0)`. That is the
  slice in effect (largest slice time ≤ `t`) for [`StepwiseFlux`](@ref) and
  linear interpolation between the bracketing slices for
  [`LinearInterpFlux`](@ref) and [`ConservativeMeanFlux`](@ref). Outside
  `[times[1], times[end]]` the end slices hold.

Cell sums are accumulated in Float64 (pairwise) whatever the storage float
type, so a Float32 source gives its total to Float64 accuracy. The sums run on
the host: a source adapted to a device (CUDA, Metal) is copied to host memory
first, the whole rate field of a static source or one slice at a time of a
series, because Metal has no Float64. Host sources are summed in place. The
runner logs `total_emission_rate(source, 0.0)`, the rate at the run start.
"""
function total_emission_rate end

total_emission_rate(src::SurfaceFluxSource, t::Real) = _cell_total(src.cell_mass_rate)

function total_emission_rate(src::TimeVaryingSurfaceFluxSource, t::Real)
    i0, i1, w0, w1 = _flux_temporal_weights(src.scheme, src.times, Float64(t), 0.0)
    return _blended_rate!(Dict{Int, Float64}(), src, i0, i1, w0, w1)
end

"""
    emission_segments(src::AbstractSurfaceFluxSource, t, dt) -> Vector{Tuple{Float64, Float64}}

The pieces in which one surface-flux step `[t, t + dt]` adds `src`: pairs
`(R, τ)` of global storage rate (kg s⁻¹) and duration (s) with ``\\sum \\tau = dt``.
The step adds ``\\sum R\\,\\tau``, the global total of what
[`apply_surface_flux!`](@ref) adds cell by cell.

- [`SurfaceFluxSource`](@ref): one piece `(R, dt)`.
- [`TimeVaryingSurfaceFluxSource`](@ref): one piece per segment of
  `_flux_temporal_segments(scheme, times, t, dt)`.
  [`LinearInterpFlux`](@ref) gives one piece, the rate interpolated at the step
  start `t`. [`StepwiseFlux`](@ref) and [`ConservativeMeanFlux`](@ref) split the
  step at every slice time inside it and evaluate each piece at its midpoint.
  This integrates the piecewise-constant or piecewise-linear reconstruction
  exactly, so their pieces over one long step and over many short steps give
  the same integral.

Within a piece the rate is constant (`StepwiseFlux`, `LinearInterpFlux`) or is
the piece mean of a linear reconstruction (`ConservativeMeanFlux`). Budget
diagnostics that integrate a sink alongside the source (radioactive decay over
a met window, for example) need these pieces rather than only their sum.
"""
function emission_segments end

emission_segments(src::SurfaceFluxSource, t::Real, dt::Real) =
    [(total_emission_rate(src, t), Float64(dt))]

function emission_segments(src::TimeVaryingSurfaceFluxSource, t::Real, dt::Real)
    totals = Dict{Int, Float64}()
    return [(_blended_rate!(totals, src, i0, i1, w0, w1), Float64(dt) * frac)
            for (i0, i1, w0, w1, frac) in
                _flux_temporal_segments(src.scheme, src.times, Float64(t), Float64(dt))]
end

"""
    integrated_emission(src::AbstractSurfaceFluxSource, t0, t1, dt) -> Float64

Global storage mass (kg dry-air equivalent) that the surface-flux operator adds
from `src` over the run's steps of length `dt` between `t0` and `t1` (seconds
since the run start): the steps start at ``t_n = t_0 + n\\,dt``,
``n = 0, …, N - 1``, with ``N = (t_1 - t_0)/dt`` an integer (to rounding), and

```math
E = \\sum_{n=0}^{N-1} \\sum_{(R, \\tau) \\in \\mathrm{segments}(t_n,\\, dt)} R\\,\\tau ,
```

the pieces of [`emission_segments`](@ref). A [`SurfaceFluxSource`](@ref) gives
``E = R\\,N\\,dt``.

For [`StepwiseFlux`](@ref) and [`ConservativeMeanFlux`](@ref), `E` is the exact
integral of the reconstruction over `[t0, t1]` and does not depend on `dt` (up
to rounding); one step of length `t1 - t0` gives the same value. For
[`LinearInterpFlux`](@ref), `E` is the step-start Riemann sum of the
interpolated rate and depends on `dt`: pass the run's advection step.

Slice totals are summed once per call in Float64 on the host (as for
[`total_emission_rate`](@ref), a device source is copied one slice at a time)
and the step sum is compensated (Neumaier), so a run-long integral carries
about one rounding of the total. Throws `ArgumentError` unless `dt`, `t0` and `t1` are finite,
`dt > 0`, `t1 ≥ t0`, and `(t1 - t0) / dt` is an integer to a relative
`√eps(Float64)`.
"""
function integrated_emission end

function integrated_emission(src::SurfaceFluxSource, t0::Real, t1::Real, dt::Real)
    n = _emission_step_count(t0, t1, dt)
    return total_emission_rate(src, t0) * (n * Float64(dt))
end

function integrated_emission(src::TimeVaryingSurfaceFluxSource, t0::Real, t1::Real, dt::Real)
    n = _emission_step_count(t0, t1, dt)
    t0_s, dt_s = Float64(t0), Float64(dt)
    totals = Dict{Int, Float64}()
    s = c = 0.0
    for step in 0:(n - 1)
        for (i0, i1, w0, w1, frac) in
                _flux_temporal_segments(src.scheme, src.times, t0_s + step * dt_s, dt_s)
            s, c = _neumaier_add(s, c, _blended_rate!(totals, src, i0, i1, w0, w1) * (dt_s * frac))
        end
    end
    return s + c
end

# Number of whole steps of length `dt` in `[t0, t1]`. The runner's step is
# `window_dt / steps_per_window`, so `N·dt` matches the span only to rounding.
function _emission_step_count(t0::Real, t1::Real, dt::Real)
    span, step = Float64(t1) - Float64(t0), Float64(dt)
    step > 0 && isfinite(step) ||
        throw(ArgumentError("integrated_emission: step dt must be positive and finite, got $(dt)"))
    isfinite(span) || throw(ArgumentError("integrated_emission: t0 = $(t0) and t1 = $(t1) must be finite"))
    span >= 0 || throw(ArgumentError("integrated_emission: t1 = $(t1) precedes t0 = $(t0)"))
    n = round(Int, span / step)
    isapprox(n * step, span; rtol = sqrt(eps(Float64))) || throw(ArgumentError(
        "integrated_emission: [$(t0), $(t1)] is not a whole number of steps dt = $(dt)"))
    return n
end

# Σ_c over a static rate field: one array, or one array per cubed-sphere panel.
_cell_total(rate::AbstractArray) = _host_total(rate)
_cell_total(rates::Tuple) = sum(_cell_total, rates)

# R_k: Σ_c over slice k (the trailing dimension) of a rate series.
_slice_total(series::AbstractArray, k::Integer) = _host_total(selectdim(series, ndims(series), k))
_slice_total(series::Tuple, k::Integer) = sum(panel -> _slice_total(panel, k), series)

# Pairwise Float64 sum of `values`, taken on the host. A device array (CUDA,
# Metal) is copied to host memory first: Metal has no Float64, and these totals
# are start-up and budget diagnostics, not per-step work. Host arrays are summed
# in place, so CPU totals are unchanged by the dispatch.
_host_total(values::AbstractArray) = sum(Float64, _host_values(get_backend(values), values))

_host_values(::KA.CPU, values) = values
_host_values(::KA.Backend, values) = Array(values)

# w₀ R_{i₀} + w₁ R_{i₁}, with the slice totals memoised in `totals`.
_blended_rate!(totals::Dict{Int, Float64}, src::TimeVaryingSurfaceFluxSource, i0, i1, w0, w1) =
    w0 * _global_slice_rate!(totals, src, i0) + w1 * _global_slice_rate!(totals, src, i1)

_global_slice_rate!(totals::Dict{Int, Float64}, src::TimeVaryingSurfaceFluxSource, k::Integer) =
    get!(() -> _slice_total(src.cell_mass_rate_series, k), totals, Int(k))
