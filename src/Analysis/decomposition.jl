# ===========================================================================
# Seasonal, trend and interannual decompositions of time series.
#
# Both decompositions are linear least-squares projections solved through the
# Moore-Penrose pseudoinverse (`pinv`). A design matrix can be rank deficient
# (a calendar month never observed in a polar band, a calendar day seen in a
# single year), and the pseudoinverse then returns the minimum-norm solution,
# which is what `numpy.linalg.lstsq` returns; an unpivoted QR solve (`A \ B`)
# would not. Being linear projections, both commute with sums of series and
# remove constant offsets (tracer carriers) exactly.
# ===========================================================================

"""
    doy_linear_detrend(x, dates) -> Array{Float64}

Interannual variability of daily series: for each calendar day, fit and
remove a straight line across years.

`dates` holds one date per entry of the last dimension of `x` (time); all other
dimensions of `x` are independent series. Samples are grouped by calendar day
(month and day of month), with 29 February folded onto 28 February, so in a
leap year that key has two samples of the same year. For the samples
``i \\in I_d`` of day ``d``, in years ``y_i``,

```math
x_i \\approx \\alpha_d + \\beta_d\\,(y_i - \\bar y_d), \\qquad
r_i = x_i - \\alpha_d - \\beta_d\\,(y_i - \\bar y_d),
```

with ``\\bar y_d`` the mean year of the samples; the least-squares fit gives
``\\alpha_d`` the mean of the samples and
``\\beta_d = \\sum (y_i - \\bar y_d)\\,x_i / \\sum (y_i - \\bar y_d)^2``. The
result is the residual ``r``. When all samples of a day fall in one year the
minimum-norm solution has ``\\beta_d = 0``, and a day seen once has residual 0.

This removes the mean seasonal cycle and any linear drift of it over the record
in one step. Detrending the whole series first and deseasonalising afterwards
does not do the same: a seasonal cycle much larger than the trend is not
orthogonal to a linear fit over a finite window, and the leakage leaves a
spurious sawtooth in the residual.

Missing values are not skipped: a NaN sample makes the fit of its calendar day
NaN for that series, so all of that series' samples on that day become NaN.
"""
function doy_linear_detrend(x::AbstractArray{<:Real}, dates::AbstractVector{<:TimeType})
    ndims(x) >= 1 || throw(ArgumentError("doy_linear_detrend needs at least one dimension"))
    nt = size(x, ndims(x))
    length(dates) == nt ||
        throw(DimensionMismatch("$(length(dates)) dates for a last dimension of length $(nt)"))
    X = reshape(Array{Float64}(x), :, nt)             # (series, time), a copy
    out = copy(X)
    keys = map(_calendar_day_key, dates)
    for key in unique(keys)
        members = findall(==(key), keys)
        years = Float64[year(dates[i]) for i in members]
        A = hcat(years .- mean(years), ones(length(members)))
        Xm = X[:, members]
        coef = Xm * transpose(pinv(A))                # (series, 2)
        out[:, members] .= Xm .- coef * transpose(A)
    end
    return reshape(out, size(x))
end

# Calendar day of a date, 29 February folded onto 28 February.
function _calendar_day_key(d::TimeType)
    m, dd = month(d), day(d)
    return (m == 2 && dd == 29) ? (2, 28) : (m, dd)
end

"""
    seasonal_trend_decomposition(series, seen; min_months = 24,
                                 min_years_per_month = 5, first_month = 1)
        -> (; anomaly, seasonal, detrended)

Split monthly series into a mean seasonal cycle, a linear trend and
interannual anomalies, fitting only the observed months.

`series` has size `(nloc, nmonth)` or `(nloc, nmonth, nseries)`: locations
(latitude bands, regions or cells), consecutive calendar months starting in
calendar month `first_month`, and optionally several series (observations
and models) fitted on the same mask. `seen` (`nloc × nmonth`, Bool) marks the
observed months; `series` must be finite wherever `seen` is true and is not
read elsewhere.

For each location with at least `min_months` observed months, each series is
reduced to its mean over the observed months, ``v_k = x_k - \\bar x``, and
fitted on the observed months ``k`` by

```math
v_k \\approx \\sum_{m=1}^{12} c_m\\,[\\mathrm{month}(k) = m] + \\beta\\,t_k,
\\qquad t_k = \\frac{k - (n+1)/2}{12}\\ \\text{years},
```

twelve calendar-month intercepts and a linear trend in time centred on the
record, solved by the pseudoinverse (minimum norm, so a calendar month never
observed gets ``c_m = 0``). Outputs, NaN wherever not computed:

- `anomaly` (size of `series`): the residual ``v_k - \\hat v_k`` on observed months;
- `detrended` (size of `series`): ``v_k - \\beta t_k`` on observed months, minus
  its mean over them;
- `seasonal` (`nloc × 12 [× nseries]`, indexed by calendar month): ``c_m``,
  set to NaN for calendar months observed in fewer than `min_years_per_month`
  years, then reduced by the mean of the remaining months.

Adding a constant to a series leaves all three outputs unchanged, and the
decomposition of a sum of series is the sum of their decompositions.
"""
function seasonal_trend_decomposition(series::AbstractArray{<:Real}, seen::AbstractMatrix{Bool};
                                      min_months::Integer = 24,
                                      min_years_per_month::Integer = 5,
                                      first_month::Integer = 1)
    ndims(series) in (2, 3) ||
        throw(ArgumentError("series must have size (nloc, nmonth) or (nloc, nmonth, nseries), got $(size(series))"))
    nloc, nt, ns = size(series, 1), size(series, 2), size(series, 3)
    size(seen) == (nloc, nt) ||
        throw(DimensionMismatch("seen has size $(size(seen)); expected $((nloc, nt))"))
    1 <= first_month <= 12 || throw(ArgumentError("first_month must be in 1:12, got $(first_month)"))
    min_months >= 1 || throw(ArgumentError("min_months must be positive, got $(min_months)"))
    min_years_per_month >= 1 ||
        throw(ArgumentError("min_years_per_month must be positive, got $(min_years_per_month)"))

    calendar_month = [mod(first_month + k - 2, 12) + 1 for k in 1:nt]
    t = [(k - (nt + 1) / 2) / 12 for k in 1:nt]
    design = zeros(Float64, nt, 13)
    for k in 1:nt
        design[k, calendar_month[k]] = 1.0
        design[k, 13] = t[k]
    end

    anomaly = fill(NaN, nloc, nt, ns)
    detrended = fill(NaN, nloc, nt, ns)
    seasonal = fill(NaN, nloc, 12, ns)
    x = reshape(series, nloc, nt, ns)
    for b in 1:nloc
        observed = findall(view(seen, b, :))
        length(observed) < min_months && continue
        values = Float64[x[b, k, s] for k in observed, s in 1:ns]
        all(isfinite, values) ||
            throw(ArgumentError("series is not finite at an observed month of location $(b)"))
        values .-= mean(values; dims = 1)
        D = design[observed, :]
        coef = pinv(D) * values                       # (13, ns)
        anomaly[b, observed, :] .= values .- D * coef
        counts = zeros(Int, 12)
        for k in observed
            counts[calendar_month[k]] += 1
        end
        for s in 1:ns
            d = values[:, s] .- coef[13, s] .* t[observed]
            detrended[b, observed, s] .= d .- mean(d)
            cycle = coef[1:12, s]
            cycle[counts .< min_years_per_month] .= NaN
            valid = .!isnan.(cycle)
            any(valid) && (cycle .-= mean(cycle[valid]))
            seasonal[b, :, s] .= cycle
        end
    end
    if ndims(series) == 2
        return (; anomaly = anomaly[:, :, 1], seasonal = seasonal[:, :, 1],
                detrended = detrended[:, :, 1])
    end
    return (; anomaly, seasonal, detrended)
end
