# ===========================================================================
# Tools for one atmospheric column.
#
# Layers are ordered top first (layer 1 at the model top), the order of
# AtmosTransport snapshots (`lev` positive down). Profiles from files stored
# surface first (GEOS-Chem) are reversed by the caller before use.
# ===========================================================================

"""
    column_band_weights(m, bands) -> Matrix{Float64}

Weights that average a column profile over bands of column air mass.

`m` holds the air mass of each layer of one column, top first, in any mass
unit (kg m⁻² or kg per cell); it must be finite and non-negative with a
positive total. `bands` is a collection of `(lower, upper)` fractions of the
column air mass counted from the surface, ``0 \\le`` lower ``<`` upper
``\\le 1``: `(0, 0.1)` is the lowest 10 % of the air mass, `(0.9, 1)` the top
10 % and `(0, 1)` the whole column. A `NamedTuple` of pairs works and keeps
its order. The result is a `length(m) × length(bands)` matrix.

With the column mass ``M = \\sum_k m_k`` and the fraction of the column below
and at the top of layer ``k``,

```math
f^{\\rm bot}_k = \\frac{1}{M} \\sum_{j > k} m_j, \\qquad
f^{\\rm top}_k = f^{\\rm bot}_k + \\frac{m_k}{M},
```

the weight of layer ``k`` in band ``(a, b)`` is the air mass of the part of the
layer inside the band,

```math
w_k = \\max\\big(0,\\ \\min(b, f^{\\rm top}_k) - \\max(a, f^{\\rm bot}_k)\\big)\\, M,
```

so that ``\\sum_k w_k = (b - a) M`` and the mass-weighted band mean of a mixing
ratio is ``\\sum_k w_k q_k / \\sum_k w_k``. ``M`` is summed top to bottom and
``\\sum_{j>k} m_j`` is formed as the running sum from the surface minus
``m_k``, the arithmetic of the Python benchmark this replaces.
"""
function column_band_weights(m::AbstractVector{<:Real}, bands)
    edges = _mass_fraction_bands(bands)
    w = Matrix{Float64}(undef, length(m), length(edges))
    return _column_band_weights!(w, m, edges)
end

"""
    column_band_weights!(w, m, bands) -> w

In-place [`column_band_weights`](@ref): `w` is `length(m) × length(bands)`.
"""
column_band_weights!(w::AbstractMatrix{<:AbstractFloat}, m::AbstractVector{<:Real}, bands) =
    _column_band_weights!(w, m, _mass_fraction_bands(bands))

function _column_band_weights!(w, m, edges::Vector{NTuple{2, Float64}})
    L = length(m)
    size(w) == (L, length(edges)) ||
        throw(DimensionMismatch("weights have size $(size(w)); expected $((L, length(edges)))"))
    total = 0.0
    for k in 1:L
        mk = Float64(m[k])
        (isfinite(mk) && mk >= 0) ||
            throw(ArgumentError("layer air mass must be finite and non-negative, got $(mk) in layer $(k)"))
        total += mk
    end
    total > 0 || throw(ArgumentError("column air mass is zero"))
    from_surface = 0.0
    for k in L:-1:1
        mk = Float64(m[k])
        from_surface += mk
        below = from_surface - mk
        f_bot = below / total
        f_top = (below + mk) / total
        for (j, (a, b)) in enumerate(edges)
            w[k, j] = max(min(b, f_top) - max(a, f_bot), 0.0) * total
        end
    end
    return w
end

function _mass_fraction_bands(bands)
    edges = NTuple{2, Float64}[]
    for band in bands
        length(band) == 2 ||
            throw(ArgumentError("a column-mass band is a (lower, upper) pair, got $(band)"))
        a, b = Float64(band[1]), Float64(band[2])
        0 <= a < b <= 1 ||
            throw(ArgumentError("column-mass band ($(a), $(b)) must satisfy 0 ≤ lower < upper ≤ 1"))
        push!(edges, (a, b))
    end
    isempty(edges) && throw(ArgumentError("no column-mass bands given"))
    return edges
end

"""
    logp_interpolate(p, p_mid, q; outside = NaN)

Value of the column profile `q` at the pressure `p`, linear in log pressure.

`p_mid` holds the layer mid pressures of one column, strictly increasing (top
first) and positive; `q` the layer values. `p` is one target pressure (the
result is a `Float64`) or a vector of them (the result is a
`Vector{Float64}`), in the unit of `p_mid`. For a target with
``p_k \\le p \\le p_{k+1}``,

```math
q(p) = q_k\\,(1 - w) + q_{k+1}\\,w, \\qquad
w = \\frac{\\ln p - \\ln p_k}{\\ln p_{k+1} - \\ln p_k}.
```

A target outside ``[p_1, p_L]`` (above the top mid level or below the
lowest one) gets `outside`: there is no extrapolation and no clamping, so a
zonal mean on pressure levels averages only the columns that reach the level.
"""
logp_interpolate(p::Real, p_mid::AbstractVector{<:Real}, q::AbstractVector{<:Real};
                 outside::Real = NaN) =
    only(logp_interpolate!(Vector{Float64}(undef, 1), [p], p_mid, q; outside))

logp_interpolate(p::AbstractVector{<:Real}, p_mid::AbstractVector{<:Real},
                 q::AbstractVector{<:Real}; outside::Real = NaN) =
    logp_interpolate!(Vector{Float64}(undef, length(p)), p, p_mid, q; outside)

"""
    logp_interpolate!(out, p, p_mid, q; outside = NaN) -> out

In-place [`logp_interpolate`](@ref) for a vector of target pressures `p`.
"""
function logp_interpolate!(out::AbstractVector{<:AbstractFloat}, p::AbstractVector{<:Real},
                           p_mid::AbstractVector{<:Real}, q::AbstractVector{<:Real};
                           outside::Real = NaN)
    L = length(p_mid)
    L >= 1 || throw(ArgumentError("empty pressure profile"))
    length(q) == L || throw(DimensionMismatch("$(length(q)) values for $(L) pressure levels"))
    length(out) == length(p) ||
        throw(DimensionMismatch("output has length $(length(out)) for $(length(p)) target pressures"))
    (isfinite(p_mid[1]) && p_mid[1] > 0) ||
        throw(ArgumentError("mid pressures must be positive and finite, got $(p_mid[1]) at the top"))
    for k in 2:L
        (isfinite(p_mid[k]) && p_mid[k] > p_mid[k-1]) ||
            throw(ArgumentError("mid pressures must increase strictly downward (top first); " *
                                "level $(k) has $(p_mid[k]) after $(p_mid[k-1])"))
    end
    lp = Float64[log(Float64(x)) for x in p_mid]
    for (i, target) in enumerate(p)
        (isfinite(target) && target > 0) ||
            throw(ArgumentError("target pressure must be positive and finite, got $(target)"))
        x = log(Float64(target))
        k = searchsortedlast(lp, x)            # number of levels with ln p_k ≤ ln p
        out[i] = if k == 0 || x > lp[L]
            outside
        elseif k == L                          # exactly at the lowest mid level
            Float64(q[L])
        else
            w = (x - lp[k]) / (lp[k+1] - lp[k])
            Float64(q[k]) * (1 - w) + Float64(q[k+1]) * w
        end
    end
    return out
end
