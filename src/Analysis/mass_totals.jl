# ===========================================================================
# Global tracer-mass series.
#
# A run writes the global total of every tracer at every snapshot
# (`<tracer>_total_mass`, a compensated Float64 sum of the tracer storage,
# independent of the storage precision). Two diagnostics are built on these
# series:
#
# - the drift of one run against its initial value: for a tracer without
#   sources or sinks the total is invariant, so any change is arithmetic error;
# - the difference of a run from its twin, a run of the same configuration in
#   another precision (Float32 against Float64), frame by frame.
#
# Both work on plain vectors of the same frames; reading and aligning them
# (`daily_snapshot_series`, matching times) is the caller's step.
# ===========================================================================

"""
    drift_statistics(series) -> (; drift, increment_mean, increment_std, t, n)

Change of a global-total series ``s_1, \\dots, s_N`` (for example a
`<tracer>_total_mass` series, ``N \\ge 2``) relative to its initial value
``s_1 > 0``:

- `drift` ``= (s_N - s_1) / s_1``, the net relative change;
- the ``n = N - 1`` relative increments ``d_k = (s_{k+1} - s_k) / s_1``, with
  their mean `increment_mean` ``\\bar d`` and sample standard deviation
  `increment_std` ``\\sigma_d`` (normalised by ``n - 1``);
- `t` ``= \\bar d / (\\sigma_d / \\sqrt{n})``, the mean increment in units of
  its standard error.

Without sources or sinks the total is invariant, so `drift` is accumulated
arithmetic error, and `t` tells a bias (a mean per-step error, ``|t| \\gtrsim
3``) from unbiased noise (``|t| \\lesssim 2``). With uniformly spaced frames
the increments are per snapshot interval.

Degenerate cases: with ``n = 1`` the standard deviation and `t` are NaN; with
``\\sigma_d = 0`` (every increment equal), `t` is 0 when the increments are 0,
as for a `Float64` total that stays constant to the last bit, and ``\\pm\\infty``
otherwise.

Replaces `increment_stats` and the drift column of
`float32_conservation_attribution.py`. (That script returned ``t = 0`` for
any ``\\sigma_d = 0`` or NaN.)
"""
function drift_statistics(series::AbstractVector{<:Real})
    N = length(series)
    N >= 2 || throw(ArgumentError("drift_statistics needs at least 2 values; got $(N)"))
    s = Float64.(series)
    all(isfinite, s) || throw(ArgumentError("drift_statistics: the series holds non-finite values"))
    s1 = s[1]
    s1 > 0 || throw(ArgumentError("drift_statistics: the initial value must be positive; got $(s1)"))
    d = [(s[k + 1] - s[k]) / s1 for k in 1:N-1]
    n = length(d)
    m = mean(d)
    sd = n >= 2 ? std(d; corrected = true) : NaN
    t = sd > 0 ? m / (sd / sqrt(n)) :
        sd == 0 ? (m == 0 ? 0.0 : copysign(Inf, m)) : NaN
    return (; drift = (s[N] - s1) / s1, increment_mean = m, increment_std = sd, t, n)
end

"""
    twin_relative_difference(candidate, reference) -> (; raw, net)

Frame-by-frame relative difference of a global total from its twin:
`candidate` and `reference` hold the same total (for example
`<tracer>_total_mass`) at the same frames, from two runs that differ in one
setting, typically a `Float32` run and its `Float64` twin.

- `raw[k]` ``= (c_k - r_k) / r_k``, NaN where ``r_k = 0``;
- `net[k]` ``=`` `raw[k] - raw[1]` when ``r_1 \\ne 0``: the difference
  accumulated after the first frame. The first frame carries the rounding of
  the initial state to the run precision (`Float32(400e-6)` is 2.53e-8 below
  `400e-6`), a constant offset that is removed. When ``r_1 = 0``, for a tracer
  that starts from zero mass, `net = raw`, relative to the reference total,
  which is then the mass emitted since the first frame.

Both series must start at the same frame, the start of the compared span:
`net` is net of whatever difference that frame holds, so for the offset of the
initial state it must be the initial snapshot of both runs. Non-finite totals
are an error.

Replaces `f32_minus_f64` of `float32_conservation_attribution.py` (which is
`net` at the last common frame; its ``(c_k - r_k)/(r_k - r_1)`` for a zero
start is `raw[k]` exactly) and the `rel`/`net` series of
`compare_float32_float64_runs.py`, with the same Float64 operations.
"""
function twin_relative_difference(candidate::AbstractVector{<:Real}, reference::AbstractVector{<:Real})
    length(candidate) == length(reference) ||
        throw(DimensionMismatch("candidate has $(length(candidate)) frames but reference $(length(reference))"))
    isempty(reference) && throw(ArgumentError("twin_relative_difference: the series are empty"))
    (all(isfinite, candidate) && all(isfinite, reference)) ||
        throw(ArgumentError("twin_relative_difference: the series hold non-finite values"))
    raw = [r == 0 ? NaN : (Float64(c) - Float64(r)) / Float64(r)
           for (c, r) in zip(candidate, reference)]
    net = !iszero(first(reference)) ? raw .- raw[1] : copy(raw)
    return (; raw, net)
end
