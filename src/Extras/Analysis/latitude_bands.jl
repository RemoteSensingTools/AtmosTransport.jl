# ===========================================================================
# Latitude-band means.
#
# A band set assigns every cell centre latitude to one band (or to none).
# `BandAssignment` stores that assignment with the cell areas once per grid;
# `band_means` then forms area-weighted means of any number of fields on the
# grid. Missing values (NaN) are left out of both the weighted sum and the
# area it is divided by, so a band mean is the mean over the cells that have
# data.
# ===========================================================================

"""
    AbstractLatitudeBands

A partition of latitudes into bands. A subtype implements
[`nbands`](@ref), [`band_edges`](@ref) and [`band_index`](@ref)`(bands, lat)`,
which returns the band of a cell centre latitude in `1:nbands(bands)`, or `0`
for a latitude in no band (such a cell enters no mean).
"""
abstract type AbstractLatitudeBands end

"""
    EqualAreaBands(n)

`n` latitude bands of equal area. On a sphere the area south of latitude φ is
proportional to ``1 + \\sin\\varphi``, so equal areas mean edges equally spaced
in ``\\sin\\varphi``:

```math
s_k = \\frac{2k}{n} - 1, \\qquad k = 0, \\dots, n,
```

evaluated as `(k / n) * 2 - 1` in Float64 (the rounding of the Python
diagnostics this replaces). Band ``b \\in 1{:}n`` holds the latitudes with
``s_{b-1} \\le \\sin\\varphi < s_b``, where ``\\sin\\varphi`` is evaluated as
`sin(deg2rad(φ))` and compared in IEEE arithmetic (so `-0.0` lies with `0.0`,
as in NumPy `digitize`); the north pole (``\\sin\\varphi = 1``) belongs to band `n`.
Every latitude in ``[-90°, 90°]`` is in exactly one band.
"""
struct EqualAreaBands <: AbstractLatitudeBands
    n::Int
    function EqualAreaBands(n::Integer)
        n >= 1 || throw(ArgumentError("EqualAreaBands needs at least one band, got $(n)"))
        return new(Int(n))
    end
end

"""
    nbands(bands::AbstractLatitudeBands) -> Int

Number of bands.
"""
nbands(b::EqualAreaBands) = b.n

"""
    band_edges(bands::AbstractLatitudeBands) -> Vector{Float64}

The `nbands(bands) + 1` band edges, ascending, in the coordinate the band set
is defined in: ``\\sin\\varphi`` for [`EqualAreaBands`](@ref).
"""
band_edges(b::EqualAreaBands) = Float64[k / b.n * 2 - 1 for k in 0:b.n]

"""
    band_index(bands::AbstractLatitudeBands, lat) -> Int

Band of the latitude `lat` (degrees north) in `1:nbands(bands)`, or `0` when
the latitude is in no band. A latitude outside ``[-90°, 90°]`` (or NaN) is an
`ArgumentError`.
"""
function band_index(b::EqualAreaBands, lat::Real)
    -90 <= lat <= 90 || throw(ArgumentError("latitude $(lat) is outside [-90, 90] degrees"))
    n = b.n
    s = sin(deg2rad(Float64(lat)))
    edge(k) = k / n * 2 - 1                    # the edges of `band_edges`, strictly increasing
    # i = number of edges ≤ s (IEEE comparison, so -0.0 counts as 0.0, as in
    # NumPy `digitize`), from an arithmetic first guess corrected against the
    # exact edges, without allocating.
    i = clamp(floor(Int, (s + 1) / 2 * n) + 1, 0, n + 1)
    while i <= n && edge(i) <= s
        i += 1
    end
    while i > 0 && edge(i - 1) > s
        i -= 1
    end
    return clamp(i, 1, n)
end

"""
    BandAssignment(bands, lat, area)

The band of every cell of a grid and the cell areas, built once and reused by
[`band_means`](@ref). `lat` (cell centre latitudes, degrees north) and `area`
(cell areas, or any non-negative weights) have the shape of the grid, for
example `(Xdim, Ydim, nf)` for a cubed sphere read with NCDatasets or
`(Nx, Ny)` for a lat-lon grid. Areas must be finite and non-negative.
"""
struct BandAssignment{B <: AbstractLatitudeBands, N}
    bands::B
    shape::NTuple{N, Int}
    band::Vector{Int}       # band of each cell in linear order; 0 = in no band
    area::Vector{Float64}   # weight of each cell in linear order

    function BandAssignment(bands::B, shape::NTuple{N, Int}, band::Vector{Int},
                            area::Vector{Float64}) where {B <: AbstractLatitudeBands, N}
        (length(band) == length(area) == prod(shape)) ||
            throw(DimensionMismatch("$(length(band)) band indices and $(length(area)) areas for a grid of shape $(shape)"))
        all(b -> 0 <= b <= nbands(bands), band) ||
            throw(ArgumentError("band indices must lie in 0:$(nbands(bands))"))
        all(a -> isfinite(a) && a >= 0, area) ||
            throw(ArgumentError("cell areas must be finite and non-negative"))
        return new{B, N}(bands, shape, band, area)
    end
end

function BandAssignment(bands::AbstractLatitudeBands, lat::AbstractArray{<:Real},
                        area::AbstractArray{<:Real})
    size(lat) == size(area) ||
        throw(DimensionMismatch("latitudes have size $(size(lat)) but areas $(size(area))"))
    return BandAssignment(bands, size(lat), _band_indices(bands, lat), Float64[a for a in vec(area)])
end

_band_indices(bands::AbstractLatitudeBands, lat) = Int[band_index(bands, φ) for φ in vec(lat)]

"""
    band_means(field, lat, area, bands) -> Array{Float64}
    band_means(field, assignment::BandAssignment) -> Array{Float64}

Area-weighted mean of `field` in each latitude band,

```math
\\bar x_b = \\frac{\\sum_{c \\in b,\\ x_c \\ne \\mathrm{NaN}} a_c\\, x_c}
                 {\\sum_{c \\in b,\\ x_c \\ne \\mathrm{NaN}} a_c},
```

with cell areas ``a_c``. NaN marks a missing value: such a cell is left out of
the numerator and the denominator, so a band whose cells are all NaN has the
mean NaN. Infinite values are not missing and propagate.

The leading dimensions of `field` must have the shape of the grid (the shape
of `lat`); trailing dimensions (time, series) are looped, and the result has
size `(nbands, trailing...)`: a `Vector` for a single field. Sums run in
Float64 over the cells in linear (column-major) order, the order of the Python
`numpy.bincount` diagnostics this replaces, so the same inputs give the same
bits. Build a [`BandAssignment`](@ref) once when averaging many fields on one
grid.
"""
band_means(field::AbstractArray, lat::AbstractArray, area::AbstractArray,
           bands::AbstractLatitudeBands) = band_means(field, BandAssignment(bands, lat, area))

function band_means(field::AbstractArray, a::BandAssignment)
    trailing = _trailing_size(field, a)
    out = Array{Float64}(undef, nbands(a.bands), trailing...)
    return band_means!(out, field, a)
end

"""
    band_means!(out, field, assignment::BandAssignment) -> out

In-place [`band_means`](@ref): `out` has size `(nbands, trailing...)`, where
`trailing` are the dimensions of `field` after the grid dimensions.
"""
function band_means!(out::AbstractArray{<:AbstractFloat}, field::AbstractArray,
                     a::BandAssignment)
    trailing = _trailing_size(field, a)
    nb = nbands(a.bands)
    size(out) == (nb, trailing...) ||
        throw(DimensionMismatch("output has size $(size(out)); expected $((nb, trailing...))"))
    ncell = length(a.band)
    x = reshape(field, ncell, :)
    y = reshape(out, nb, :)
    numerator = zeros(Float64, nb)
    denominator = zeros(Float64, nb)
    @inbounds for col in axes(x, 2)
        fill!(numerator, 0.0)
        fill!(denominator, 0.0)
        for c in 1:ncell
            b = a.band[c]
            v = Float64(x[c, col])
            (b == 0 || isnan(v)) && continue
            numerator[b] += v * a.area[c]
            denominator[b] += a.area[c]
        end
        for b in 1:nb
            y[b, col] = numerator[b] / denominator[b]
        end
    end
    return out
end

function _trailing_size(field::AbstractArray, a::BandAssignment{B, N}) where {B, N}
    (ndims(field) >= N && size(field)[1:N] == a.shape) ||
        throw(DimensionMismatch("field has size $(size(field)); its leading dimensions " *
                                "must be the grid shape $(a.shape)"))
    return size(field)[N+1:end]
end
