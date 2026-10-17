# ===========================================================================
# Regridding by quantity kind.
#
# Every regridder (`ConservativeRegridding.Regridder`, `IdentityRegrid`,
# `NestedCubedSphereRegrid`) maps an intensive field: a value per area or per
# point, d = W s ./ dst_areas. A per-cell total is regridded through its
# density. The kind of the field is a dispatch tag from `Quantities`, so the
# choice is explicit at every call site.
# ===========================================================================

"""
    apply_regridder!(dst, r, src, kind::QuantityKind; scratch = nothing) -> dst

Regrid `src` onto `dst` with regridder `r` according to the kind of field
(see `Quantities`). Axis 1 of `dst` and `src` is the flat horizontal index;
trailing axes (levels, times, …) are regridded column by column.

- `IntensiveCellField()`: a value per area or per point (`ps`, `T`, mixing
  ratios, flux densities). Same as `apply_regridder!(dst, r, src)`:
  ``d_t = \\sum_s W_{ts} s_s / A^{\\rm dst}_t``.
- `ExtensiveCellField()`: a per-cell total (`m` in kg, a flux in kg s⁻¹ per
  cell). Regridded as a density and multiplied back,
  ```math
  d_t = A^{\\rm dst}_t \\; \\mathcal{R}\\!\\left(\\frac{s}{A^{\\rm src}}\\right)_t
      = \\sum_s \\frac{W_{ts}}{A^{\\rm src}_s}\\, s_s ,
  ```
  with `A^src = r.src_areas` and `A^dst = r.dst_areas`. Since
  ``\\sum_t W_{ts} = A^{\\rm src}_s`` for a target that covers the source,
  ``\\sum_t d_t = \\sum_s s_s`` whatever polygon areas the regridder uses.
  Regridding a per-cell total as intensive instead distorts it in proportion
  to the source cell area (the ~12× polar mass deficit of early C180 ERA5
  binaries).
- `HorizontalVectorField()`, `HorizontalFluxField()`: not scalar regrids;
  throw an `ArgumentError`. Winds are regridded component by component as
  intensive and rotated (`rotate_winds_to_panel_local!`); face fluxes are
  rebuilt from regridded winds (`reconstruct_cs_fluxes!`).

`src` is not modified. The density ``s/A^{\\rm src}`` of one horizontal column
at a time is formed in the element type of `src` and stored in `scratch`; the
product with ``A^{\\rm dst}`` is stored in the element type of `dst`. Both
steps are broadcasts, not scalar loops.

`scratch` is a vector of length `length(r.src_areas)` with element type
`eltype(src)` that does not share memory with `src` or `dst`; its contents are
overwritten. With `nothing` (the default) an `ExtensiveCellField` regrid
allocates one such vector per call, `similar(src, n_src)`. Callers that regrid
repeatedly pass a reusable buffer, as the cubed-sphere preprocessing workspace
does (`density_scratch`), so its steady state allocates nothing. The other kinds
accept and ignore `scratch`, so one call site serves every kind.
"""
apply_regridder!(dst::AbstractArray, r, src::AbstractArray, ::IntensiveCellField;
                 scratch = nothing) =
    apply_regridder!(dst, r, src)

function apply_regridder!(dst::AbstractVector, r, src::AbstractVector, ::ExtensiveCellField;
                          scratch = nothing)
    density = _density_scratch(scratch, dst, src, length(r.src_areas))
    _density!(density, src, r.src_areas)
    apply_regridder!(dst, r, density)
    return _times_areas!(dst, r.dst_areas)
end

function apply_regridder!(dst::AbstractArray, r, src::AbstractArray, ::ExtensiveCellField;
                          scratch = nothing)
    n_dst, n_src = length(r.dst_areas), length(r.src_areas)
    size(dst, 1) == n_dst ||
        throw(DimensionMismatch("dst leading dim $(size(dst, 1)) ≠ regridder n_dst $n_dst"))
    size(src, 1) == n_src ||
        throw(DimensionMismatch("src leading dim $(size(src, 1)) ≠ regridder n_src $n_src"))
    dst_flat = _flat_columns(dst, n_dst)
    src_flat = _flat_columns(src, n_src)
    size(dst_flat, 2) == size(src_flat, 2) || throw(DimensionMismatch(
        "trailing shape mismatch: dst $(size(dst_flat, 2)) ≠ src $(size(src_flat, 2))"))
    density = _density_scratch(scratch, dst, src, n_src)
    for col in axes(src_flat, 2)
        dst_col = view(dst_flat, :, col)
        _density!(density, view(src_flat, :, col), r.src_areas)
        apply_regridder!(dst_col, r, density)
        _times_areas!(dst_col, r.dst_areas)
    end
    return dst
end

apply_regridder!(::AbstractArray, ::Any, ::AbstractArray, ::HorizontalVectorField;
                 scratch = nothing) =
    throw(ArgumentError("HorizontalVectorField is not a scalar regrid; \
use component-wise IntensiveCellField regrid + rotate_winds_to_panel_local!"))

apply_regridder!(::AbstractArray, ::Any, ::AbstractArray, ::HorizontalFluxField;
                 scratch = nothing) =
    throw(ArgumentError("HorizontalFluxField is not a scalar regrid; \
fluxes are reconstructed from regridded winds via reconstruct_cs_fluxes!"))

# The per-column density buffer: allocated on the device of `src` when the
# caller passes none, else checked against the contract of `apply_regridder!`.
_density_scratch(::Nothing, dst, src, n_src::Int) = similar(src, eltype(src), n_src)

function _density_scratch(scratch::AbstractVector, dst, src, n_src::Int)
    length(scratch) == n_src || throw(DimensionMismatch(
        "scratch length $(length(scratch)) ≠ regridder n_src $n_src"))
    eltype(scratch) === eltype(src) || throw(ArgumentError(
        "scratch element type $(eltype(scratch)) ≠ eltype(src) = $(eltype(src)); " *
        "the density is formed in the element type of src"))
    (Base.mightalias(scratch, src) || Base.mightalias(scratch, dst)) && throw(ArgumentError(
        "scratch must not share memory with src or dst"))
    return scratch
end

function _density!(density::AbstractVector, src::AbstractVector, areas::AbstractVector)
    length(src) == length(areas) || throw(DimensionMismatch(
        "src length $(length(src)) ≠ regridder n_src $(length(areas))"))
    density .= src ./ areas
    return density
end

_times_areas!(x::AbstractVector, areas::AbstractVector) = (x .*= areas)

"""
    regrid_conservation_report(r, src, dst, kind::QuantityKind) -> (; src_total, dst_total, rel_error)

Global totals of a field before (`src`) and after (`dst`) regridding with `r`,
and the relative error ``|T_{\\rm dst} - T_{\\rm src}| / |T_{\\rm src}|`` (0 when
both totals are 0, `Inf` when only `src_total` is). Axis 1 is the flat
horizontal index, of length `length(r.src_areas)` for `src` and
`length(r.dst_areas)` for `dst` (else a `DimensionMismatch`, for both kinds);
totals include every trailing column.

- `ExtensiveCellField()`: ``T = \\sum_c x_c``.
- `IntensiveCellField()`: ``T = \\sum_c x_c A_c``, with `r.src_areas` for `src`
  and `r.dst_areas` for `dst`.

Sums are taken in Float64 in a fixed order. Vector and flux kinds throw an
`ArgumentError`.
"""
function regrid_conservation_report(r, src::AbstractArray, dst::AbstractArray,
                                    kind::Union{IntensiveCellField, ExtensiveCellField})
    size(src, 1) == length(r.src_areas) || throw(DimensionMismatch(
        "src leading dim $(size(src, 1)) ≠ regridder n_src $(length(r.src_areas))"))
    size(dst, 1) == length(r.dst_areas) || throw(DimensionMismatch(
        "dst leading dim $(size(dst, 1)) ≠ regridder n_dst $(length(r.dst_areas))"))
    src_total = _regrid_total(src, r.src_areas, kind)
    dst_total = _regrid_total(dst, r.dst_areas, kind)
    rel_error = if iszero(src_total)
        iszero(dst_total) ? 0.0 : Inf
    else
        abs(dst_total - src_total) / abs(src_total)
    end
    return (; src_total, dst_total, rel_error)
end

regrid_conservation_report(r, src::AbstractArray, dst::AbstractArray, kind::QuantityKind) =
    throw(ArgumentError("regrid_conservation_report needs a scalar cell field " *
                        "(IntensiveCellField or ExtensiveCellField); got $(typeof(kind))"))

_regrid_total(x::AbstractArray, ::AbstractVector, ::ExtensiveCellField) = sum(Float64, x)

# Leading dimensions are checked by `regrid_conservation_report`.
function _regrid_total(x::AbstractArray, areas::AbstractVector, ::IntensiveCellField)
    n = length(areas)
    return sum(k -> Float64(x[k]) * Float64(areas[mod1(k, n)]), 1:length(x))
end

"""
    source_coverage(r) -> Vector{Float64}

The area of each source cell that the destination cells of `r` cover,
``C_s = \\sum_t W_{ts}``, in the flat source order of `r.src_areas`. When the
destination tiles the source, ``C_s = A^{\\rm src}_s`` to the round-off of the
polygon intersections, and an `ExtensiveCellField()` regrid keeps
``\\sum_s s_s``. `IdentityRegrid` and `NestedCubedSphereRegrid` cover every
source cell exactly and return `r.src_areas`.

A `ReducedGaussianMesh` destination does not tile the sphere where adjacent
rings have different longitude counts: its cell edges are great circles, so
the latitude two such rings share is two different polylines, and the sliver
between them belongs to no destination cell. There ``C_s < A^{\\rm src}_s``
(by 17 % for 5° source cells on the boundary of a 12-cell and a 24-cell ring;
the octahedral grids change the count at every ring). A per-cell total divided
by ``C_s`` instead of ``A^{\\rm src}_s`` before an intensive regrid, and
multiplied by `r.dst_areas` after it, keeps its total on such destinations:
``\\sum_t \\sum_s W_{ts}\\, s_s / C_s = \\sum_s s_s``.
"""
source_coverage(r::Regridder) = Float64.(vec(Array(sum(r.intersections; dims = 1))))
source_coverage(r::Union{IdentityRegrid, NestedCubedSphereRegrid}) = r.src_areas
