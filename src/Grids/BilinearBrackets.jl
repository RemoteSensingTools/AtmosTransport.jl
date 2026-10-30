# Coordinate brackets for bilinear interpolation. Keep the clamped uniform-axis
# IC and half-cell DEM policies separate: their indexing and arithmetic differ.

"""
    bilinear_bracket(val, arr) -> (lo, w)

Find the 1-based bracket index `lo` and fractional weight `w ∈ [0, 1]`
such that `val ≈ arr[lo] + w × (arr[lo+1] − arr[lo])`. Uses binary
search on a **strictly increasing, non-periodic** array (e.g. latitude).

Clamps: if `val ≤ arr[1]`, returns `(1, 0.0)` (extrapolate to first value).
If `val ≥ arr[end]`, returns `(N, 0.0)` (extrapolate to last value).
"""
function bilinear_bracket(val::Float64, arr::Vector{Float64})
    N = length(arr)
    N == 1 && return (1, 0.0)
    val <= arr[1] && return (1, 0.0)
    val >= arr[N] && return (N, 0.0)
    lo, hi = 1, N
    while hi - lo > 1
        mid = (lo + hi) >> 1
        if arr[mid] <= val
            lo = mid
        else
            hi = mid
        end
    end
    denom = arr[hi] - arr[lo]
    w = denom > 0 ? (val - arr[lo]) / denom : 0.0
    return lo, w
end

"""
    uniform_periodic_bilinear_bracket(val, arr) -> (ilo, w)

Periodic (longitude) version of `bilinear_bracket`. `arr` must be
strictly increasing with uniform spacing `Δ = arr[2] − arr[1]` over a
360° domain (e.g. `[-179.5, -178.5, ..., 179.5]` or `[0.5, 1.5, ..., 359.5]`).

Returns 1-based index `ilo` and fractional weight `w` for the bracket
surrounding `mod(val − arr[1], 360)` in the periodic domain. Wraps both
`val` and the index modulo `length(arr)`.

The caller should also compute `ihi = (ilo == N) ? 1 : ilo + 1` for the
upper bracket (periodic wrap at the last cell).
"""
function uniform_periodic_bilinear_bracket(val::Float64, arr::Vector{Float64})
    N = length(arr)
    N == 1 && return (1, 0.0)
    Δ = arr[2] - arr[1]   # assumed uniform spacing [degrees]
    Δ > 0 || throw(ArgumentError("longitude coordinate must be strictly increasing"))
    u = mod(val - arr[1], 360.0) / Δ   # fractional index into the periodic domain
    ilo0 = floor(Int, u)
    return mod1(ilo0 + 1, N), u - ilo0  # (1-indexed, fractional weight)
end

# Bracketing indices and the weight of the upper one along latitude; a point
# within half a spacing outside the outermost centre takes that row, farther
# away it is outside the DEM (`nothing`).
"""Bracket a nonperiodic axis, accepting at most half a spacing beyond its centres."""
function half_cell_bilinear_bracket(lat::Vector{Float64}, φ::Float64)
    n = length(lat)
    if φ < lat[1]
        return lat[1] - φ <= (lat[2] - lat[1]) / 2 ? (1, 1, 0.0) : nothing
    elseif φ > lat[n]
        return φ - lat[n] <= (lat[n] - lat[n - 1]) / 2 ? (n, n, 0.0) : nothing
    end
    j = min(searchsortedlast(lat, φ), n - 1)
    return j, j + 1, (φ - lat[j]) / (lat[j + 1] - lat[j])
end

# Same along longitude, with `λ` taken into [lon[1], lon[1] + 360).
"""Bracket longitude on nonuniform centres, with periodic wrap or half-cell regional coverage."""
function longitude_bilinear_bracket(lon::Vector{Float64}, λ::Float64, periodic::Bool)
    n = length(lon)
    x = mod(λ - lon[1], 360.0) + lon[1]
    if x <= lon[n]
        i = min(searchsortedlast(lon, x), n - 1)
        return i, i + 1, (x - lon[i]) / (lon[i + 1] - lon[i])
    end
    spacing = (lon[n] - lon[1]) / (n - 1)
    periodic && return n, 1, (x - lon[n]) / (lon[1] + 360.0 - lon[n])
    return x - lon[n] <= spacing / 2 ? (n, n, 0.0) :
           lon[1] + 360.0 - x <= spacing / 2 ? (1, 1, 0.0) : nothing
end

export bilinear_bracket, uniform_periodic_bilinear_bracket,
       half_cell_bilinear_bracket, longitude_bilinear_bracket
