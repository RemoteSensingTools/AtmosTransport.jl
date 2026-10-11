# ===========================================================================
# Skill statistics of a model against a reference.
#
# Both functions take the model first and the reference second, use only the
# pairs where both are finite, and use population (1/n) moments.
# ===========================================================================

"""
    skill_metrics(model, reference; min_n = 3, degenerate_std = 1e-12)
        -> (; r, rmse, bias, amplitude_ratio, n)

Unweighted agreement of `model` with `reference` (arrays of equal size) over
the `n` pairs where both are finite:

- `r`: Pearson correlation; NaN when `n < min_n` or either population standard
  deviation is below `degenerate_std` (in the units of the data);
- `rmse` ``= \\sqrt{\\tfrac1n \\sum (x_i - y_i)^2}``;
- `bias` ``= \\tfrac1n \\sum (x_i - y_i)``, model minus reference;
- `amplitude_ratio` ``= \\sigma_x / \\sigma_y``, population standard deviations
  (``\\sigma^2 = \\tfrac1n \\sum (x_i - \\bar x)^2``).

With `n = 0` every statistic is NaN.
"""
function skill_metrics(model::AbstractArray{<:Real}, reference::AbstractArray{<:Real};
                       min_n::Integer = 3, degenerate_std::Real = 1e-12)
    size(model) == size(reference) ||
        throw(DimensionMismatch("model has size $(size(model)) but reference $(size(reference))"))
    finite = [(Float64(x), Float64(y)) for (x, y) in zip(model, reference) if isfinite(x) && isfinite(y)]
    n = length(finite)
    n == 0 && return (; r = NaN, rmse = NaN, bias = NaN, amplitude_ratio = NaN, n = 0)
    x = first.(finite)
    y = last.(finite)
    sx = std(x; corrected = false)
    sy = std(y; corrected = false)
    r = (n < min_n || sx < degenerate_std || sy < degenerate_std) ? NaN : clamp(cor(x, y), -1.0, 1.0)
    d = x .- y
    return (; r, rmse = sqrt(mean(abs2, d)), bias = mean(d), amplitude_ratio = sx / sy, n)
end

"""
    weighted_fit_stats(model, reference, weights)
        -> (; r2, slope, bias, rmse, ref_mean, ref_std)

Weighted agreement of `model` (``y``) with `reference` (``x``), for example two
fields on one grid with cell areas as weights. Only the cells where both
values are finite enter; their weights must be finite and non-negative and
are normalised, ``w_i \\leftarrow w_i / \\sum w``. With the weighted means
``\\bar x = \\sum w x``, ``\\bar y = \\sum w y`` and moments
``s_{xx} = \\sum w (x - \\bar x)^2``, ``s_{yy}``, ``s_{xy}``:

- `r2` ``= s_{xy}^2 / (s_{xx} s_{yy})``, NaN unless both moments are positive;
- `slope` ``= s_{xy} / s_{xx}``, the regression of the model on the reference
  (NaN when ``s_{xx} = 0``);
- `bias` ``= \\bar y - \\bar x``;
- `rmse` ``= \\sqrt{\\sum w (y - x)^2}``;
- `ref_mean` ``= \\bar x`` and `ref_std` ``= \\sqrt{s_{xx}}``.

When no cell has finite values (or the selected weights sum to 0) every
statistic is NaN.
"""
function weighted_fit_stats(model::AbstractArray{<:Real}, reference::AbstractArray{<:Real},
                            weights::AbstractArray{<:Real})
    (size(model) == size(reference) == size(weights)) ||
        throw(DimensionMismatch("model $(size(model)), reference $(size(reference)) and " *
                                "weights $(size(weights)) must have one size"))
    sel = [i for i in eachindex(model, reference, weights) if isfinite(model[i]) && isfinite(reference[i])]
    nan = (; r2 = NaN, slope = NaN, bias = NaN, rmse = NaN, ref_mean = NaN, ref_std = NaN)
    isempty(sel) && return nan
    w = Float64[weights[i] for i in sel]
    all(a -> isfinite(a) && a >= 0, w) ||
        throw(ArgumentError("weights must be finite and non-negative where model and reference are finite"))
    total = sum(w)
    total > 0 || return nan
    w ./= total
    x = Float64[reference[i] for i in sel]
    y = Float64[model[i] for i in sel]
    mx = sum(w .* x)
    my = sum(w .* y)
    sxx = sum(w .* (x .- mx) .^ 2)
    syy = sum(w .* (y .- my) .^ 2)
    sxy = sum(w .* (x .- mx) .* (y .- my))
    r2 = (sxx > 0 && syy > 0) ? sxy^2 / (sxx * syy) : NaN
    slope = sxx > 0 ? sxy / sxx : NaN
    return (; r2, slope, bias = my - mx, rmse = sqrt(sum(w .* (y .- x) .^ 2)),
            ref_mean = mx, ref_std = sqrt(sxx))
end
