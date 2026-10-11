# ===========================================================================
# Nested cubed spheres: the block kernels and the nested regridder.
#
# Two cubed-sphere meshes with the same definition (coordinate law, centre
# law, panel convention, longitude offset) and radius, whose edge sizes
# differ by an integer factor R, are nested: every coarse cell (i, j) of a
# panel is exactly the union of the R × R fine cells
#
#     ((i − 1)R + ii, (j − 1)R + jj),   ii, jj = 1 … R,
#
# because every coordinate law places the corners of a C<N> panel at
# x_s = f((s − 1)/N), so the fine corners at s = (s_c − 1)R + 1 are the
# coarse corners. Remapping between nested meshes needs no polygon
# intersection: restriction sums blocks and prolongation splits a coarse
# cell by area fraction.
#
# - `coarsen_sum!`, `coarsen_area_weighted!`: the one copy of the block
#   kernels on panel arrays `(Nc, Nc[, Nz])`, used by the transport-binary
#   coarseners in Preprocessing (GEOS native → coarser target, nested binary
#   coarsener) and by the nested regridder.
# - `meshes_nested`, `NestedCubedSphereRegrid{D}`: what `build_regridder`
#   returns for a nested pair (after the identity check, before CR.jl).
# ===========================================================================

# ---------------------------------------------------------------------------
# Block kernels
# ---------------------------------------------------------------------------

function _check_block_shapes(dst::AbstractArray, src::AbstractArray, ratio::Integer)
    ratio >= 1 || throw(ArgumentError("block ratio must be positive; got $(ratio)"))
    size(src, 1) == ratio * size(dst, 1) && size(src, 2) == ratio * size(dst, 2) &&
        size(src, 3) == size(dst, 3) ||
        throw(DimensionMismatch(
            "fine array $(size(src)) is not a $(ratio)×$(ratio) block refinement of $(size(dst))"))
    return nothing
end

"""
    coarsen_sum!(dst, src, ratio::Int) -> dst

Block-sum the fine panel array `src` onto the coarse panel array `dst`, for an
extensive quantity (kg per cell, kg per substep, …):

```math
d_{i,j,k} = \\sum_{jj=1}^{R} \\sum_{ii=1}^{R} s_{(i-1)R+ii,\\,(j-1)R+jj,\\,k}
```

with `R = ratio`. `dst` and `src` are both 2-D `(Nc, Nc)` or both 3-D
`(Nc, Nc, Nz)` with `size(src) = (R·Nc, R·Nc[, Nz])`, of the same element type
`FT`. The sum accumulates in `FT`, `ii` fastest, which is the order of the two
earlier kernel copies (`Preprocessing`'s nested binary coarsener and GEOS block
coarsener), so their outputs are unchanged bit for bit.

See also [`coarsen_area_weighted!`](@ref) for intensive fields.
"""
function coarsen_sum!(dst::AbstractArray{FT, N}, src::AbstractArray{FT, N},
                      ratio::Int) where {FT, N}
    N in (2, 3) || throw(ArgumentError("coarsen_sum! takes 2-D or 3-D panel arrays; got $(N)-D"))
    _check_block_shapes(dst, src, ratio)
    @inbounds for k in 1:size(dst, 3), j in 1:size(dst, 2), i in 1:size(dst, 1)
        total = zero(FT)
        i0 = (i - 1) * ratio
        j0 = (j - 1) * ratio
        for jj in 1:ratio, ii in 1:ratio
            total += src[i0 + ii, j0 + jj, k]
        end
        dst[i, j, k] = total
    end
    return dst
end

"""
    coarsen_area_weighted!(dst, src, area, ratio::Int) -> dst

Area-weighted block mean of the fine panel array `src` onto the coarse panel
array `dst`, for an intensive quantity (K, m s⁻¹, kg kg⁻¹, kg m⁻² s⁻¹, …):

```math
d_{i,j,k} = \\mathrm{fl}_{FT}\\!\\left(
  \\frac{\\sum_{f \\in B(i,j)} A_f\\, s_{f,k}}{\\sum_{f \\in B(i,j)} A_f}\\right),
\\qquad B(i,j) = \\{((i-1)R+ii,\\,(j-1)R+jj) : 1 \\le ii, jj \\le R\\},
```

with `R = ratio` and `A = area`, the `(R·Nc, R·Nc)` fine cell areas of one
panel (every panel has the same areas). Numerator and denominator accumulate
in Float64 (`ii` fastest) and the quotient is rounded once to
`FT = eltype(dst)`, so a Float32 result is the correctly rounded block mean
up to the Float64 accumulation error, and a uniform field stays uniform.
`area` may be Float64 while the data are Float32. `dst`/`src` are both 2-D
`(Nc, Nc)` or both 3-D `(Nc, Nc, Nz)`; the 3-D result equals the 2-D kernel
applied level by level, bit for bit.

The denominator is the block sum of fine areas, not the coarse mesh area:
nested meshes agree to round-off only (relative 6.3e-13 at C180 → C90 on the
GMAO grid), and the block sum keeps constants exact. [`NestedCubedSphereRegrid`](@ref)
divides by the coarse mesh area instead, so that its extensive path conserves
exactly.
"""
function coarsen_area_weighted!(dst::AbstractArray{<:Real, N}, src::AbstractArray{<:Real, N},
                                area::AbstractMatrix{<:Real}, ratio::Int) where N
    N in (2, 3) || throw(ArgumentError(
        "coarsen_area_weighted! takes 2-D or 3-D panel arrays; got $(N)-D"))
    _check_block_shapes(dst, src, ratio)
    size(area) == (size(src, 1), size(src, 2)) || throw(DimensionMismatch(
        "area $(size(area)) does not match the fine panel $((size(src, 1), size(src, 2)))"))
    FT = eltype(dst)
    @inbounds for k in 1:size(dst, 3), j in 1:size(dst, 2), i in 1:size(dst, 1)
        numerator = 0.0
        denominator = 0.0
        i0 = (i - 1) * ratio
        j0 = (j - 1) * ratio
        for jj in 1:ratio, ii in 1:ratio
            a = Float64(area[i0 + ii, j0 + jj])
            numerator += Float64(src[i0 + ii, j0 + jj, k]) * a
            denominator += a
        end
        dst[i, j, k] = FT(numerator / denominator)
    end
    return dst
end

# ---------------------------------------------------------------------------
# Nesting test
# ---------------------------------------------------------------------------

# Largest unit-sphere distance between a coarse corner and the fine corner
# that should coincide with it, accepted as nesting. The corners of nested
# meshes agree to ~1e-15 (round-off of the coordinate law); corners of meshes
# that do not nest are misplaced by a sizeable fraction of a cell width, which
# is about 1.6/Nc on the unit sphere (1.6e-4 at Nc = 10⁴).
const _NESTED_CORNER_TOLERANCE = 1e-12

function _corners_nest(fine::CubedSphereMesh, coarse::CubedSphereMesh, ratio::Int)
    for p in 1:6, j in 1:(coarse.Nc + 1), i in 1:(coarse.Nc + 1)
        xc, yc, zc = cs_corner_xyz(coarse, i, j, p)
        xf, yf, zf = cs_corner_xyz(fine, (i - 1) * ratio + 1, (j - 1) * ratio + 1, p)
        hypot(xc - xf, yc - yf, zc - zf) <= _NESTED_CORNER_TOLERANCE || return false
    end
    return true
end

"""
    meshes_nested(a, b) -> Bool

True when `a` and `b` are cubed-sphere meshes with the same definition (tag,
coordinate law, centre law, panel convention, longitude offset) and radius,
different edge sizes `Nc` with an integer ratio, and corners that coincide:
corner `(i, j)` of each panel of the coarser mesh is corner
`((i − 1)R + 1, (j − 1)R + 1)` of the finer one, to `1e-12` on the unit sphere
(evaluated, not assumed). Every fine cell then lies in exactly one coarse
cell. Meshes with equal `Nc` are equivalent ([`meshes_equivalent`](@ref)),
not nested; meshes of other types are never nested.
"""
meshes_nested(::Any, ::Any) = false

function meshes_nested(a::CubedSphereMesh, b::CubedSphereMesh)
    a.Nc == b.Nc && return false
    _same_cs_definition(a, b) || return false
    fine, coarse = a.Nc > b.Nc ? (a, b) : (b, a)
    fine.Nc % coarse.Nc == 0 || return false
    return _corners_nest(fine, coarse, fine.Nc ÷ coarse.Nc)
end

# ---------------------------------------------------------------------------
# Nested regridder
# ---------------------------------------------------------------------------

"""
    AbstractNestingDirection

Direction of a [`NestedCubedSphereRegrid`](@ref): [`Restriction`](@ref)
(fine → coarse) or [`Prolongation`](@ref) (coarse → fine).
"""
abstract type AbstractNestingDirection end

"""
    Restriction <: AbstractNestingDirection

Fine → coarse remap between nested cubed spheres: area-weighted block sum.
"""
struct Restriction <: AbstractNestingDirection end

"""
    Prolongation <: AbstractNestingDirection

Coarse → fine remap between nested cubed spheres: each coarse cell is split
among its `R × R` fine cells by area fraction.
"""
struct Prolongation <: AbstractNestingDirection end

"""
    NestedCubedSphereRegrid{D<:AbstractNestingDirection}(ratio, src_areas, dst_areas)
    NestedCubedSphereRegrid(src_mesh::CubedSphereMesh, dst_mesh::CubedSphereMesh)

Exact conservative remap between nested cubed spheres ([`meshes_nested`](@ref)),
returned by [`build_regridder`](@ref) for such a pair. `ratio` is the integer
ratio `R` of the edge sizes; `src_areas` and `dst_areas` are the cell areas
[m²] of the two meshes in the flat cell order of [`apply_regridder!`](@ref),
`(p − 1)Nc² + i + (j − 1)Nc`. The mesh constructor evaluates them in Float64
from the mesh corners (Girard's theorem), whatever the mesh precision, so
Float32 and Float64 meshes give the same regridder.

`apply_regridder!(dst, r, src)` remaps an intensive field (per area or per
point) as a conservative regridder with intersection matrix `W` does,
``d_t = \\sum_s W_{ts}\\, s_s / A^{\\rm dst}_t``. For nested meshes `W` needs no
polygon clipping: fine cell `f` lies in coarse cell `c(f)`.

- `Restriction` (fine → coarse), ``W_{cf} = A_f``:
  ```math
  d_c = \\frac{1}{A_c} \\sum_{f \\in c} A_f\\, s_f
  ```
- `Prolongation` (coarse → fine), ``W_{fc} = A_c\\, A_f / \\sum_{f' \\in c} A_{f'}``:
  ```math
  d_f = s_c\\, \\frac{A_c}{\\sum_{f' \\in c} A_{f'}}
  ```

Both satisfy ``\\sum_t W_{ts} = A^{\\rm src}_s``, so the extensive path of
[`apply_regridder!`](@ref) (`ExtensiveCellField`: divide by `src_areas`,
remap, multiply by `dst_areas`) conserves the total exactly up to rounding: a
block sum for restriction, a split by area fraction ``m_f = m_c A_f / \\sum A_{f'}``
for prolongation. The coarse mesh area and the block sum of fine areas differ
only by round-off (6.3e-13 relative at C180/C90 on the GMAO grid), so a
uniform intensive field stays uniform to that level, and the result equals a
CR.jl `Regridder` of the same pair to about 1e-12. Sums run in Float64 in a
fixed order (`ii` fastest within a block); the result is rounded once to the
element type of `dst`.

Nothing is cached on disk: the regridder is built from the meshes in O(Nc²).
It has no intersection matrix, so it cannot be saved as weights, and
`normalize` does not scale its areas.
"""
struct NestedCubedSphereRegrid{D <: AbstractNestingDirection}
    ratio     :: Int
    src_areas :: Vector{Float64}
    dst_areas :: Vector{Float64}

    function NestedCubedSphereRegrid{D}(ratio::Integer, src_areas::AbstractVector{<:Real},
                                        dst_areas::AbstractVector{<:Real}) where {D <: AbstractNestingDirection}
        ratio >= 2 || throw(ArgumentError("nesting ratio must be at least 2; got $(ratio)"))
        fine, coarse = _fine_coarse_lengths(D, length(src_areas), length(dst_areas))
        Nc = _panel_edge(coarse)
        Nc > 0 && fine == 6 * (ratio * Nc)^2 || throw(DimensionMismatch(
            "$(length(src_areas)) source and $(length(dst_areas)) destination cells are " *
            "not two cubed spheres nested by a factor $(ratio) ($(D))"))
        return new{D}(Int(ratio), Vector{Float64}(src_areas), Vector{Float64}(dst_areas))
    end
end

_fine_coarse_lengths(::Type{Restriction}, n_src, n_dst) = (n_src, n_dst)
_fine_coarse_lengths(::Type{Prolongation}, n_src, n_dst) = (n_dst, n_src)

# Edge size of a six-panel flat vector of length n (0 when n is not 6Nc²).
function _panel_edge(n::Integer)
    Nc = isqrt(n ÷ 6)
    return 6 * Nc^2 == n ? Nc : 0
end

# Cell areas of every panel in the flat regridder order, in Float64 whatever
# the mesh precision: a Float32 mesh stores Float32-rounded areas, so its
# areas are recomputed by the (Float64) mesh constructor.
_float64_cell_areas(m::CubedSphereMesh{Float64}) = m.cell_areas
_float64_cell_areas(m::CubedSphereMesh) =
    CubedSphereMesh(; FT = Float64, Nc = m.Nc, Hp = 0, definition = cs_definition(m),
                    radius = Float64(m.radius)).cell_areas
_flat_float64_cell_areas(m::CubedSphereMesh) = repeat(vec(_float64_cell_areas(m)), 6)

function NestedCubedSphereRegrid(src::CubedSphereMesh, dst::CubedSphereMesh)
    meshes_nested(src, dst) || throw(ArgumentError(
        "C$(src.Nc) and C$(dst.Nc) meshes are not nested (same definition and radius, " *
        "integer Nc ratio, coinciding corners)"))
    D = src.Nc > dst.Nc ? Restriction : Prolongation
    ratio = max(src.Nc, dst.Nc) ÷ min(src.Nc, dst.Nc)
    return NestedCubedSphereRegrid{D}(ratio, _flat_float64_cell_areas(src),
                                      _flat_float64_cell_areas(dst))
end

"""
    nested_regrid_or_nothing(src, dst) -> Union{NestedCubedSphereRegrid, Nothing}

Helper used inside `build_regridder` after the identity check: the nested
regridder when [`meshes_nested`](@ref)`(src, dst)`, else `nothing`.
"""
nested_regrid_or_nothing(src, dst) =
    meshes_nested(src, dst) ? NestedCubedSphereRegrid(src, dst) : nothing

# Flat indices of coarse cell (i, j) of panel p and of its fine cell (ii, jj).
@inline _flat_cell(p, i, j, N) = (p - 1) * N * N + i + (j - 1) * N
@inline _flat_fine_cell(p, i, j, ii, jj, R, Nf) =
    _flat_cell(p, (i - 1) * R + ii, (j - 1) * R + jj, Nf)

function _nested_apply!(dst::AbstractVector, r::NestedCubedSphereRegrid{Restriction},
                        src::AbstractVector)
    R = r.ratio
    Nc = _panel_edge(length(r.dst_areas))
    Nf = R * Nc
    A_fine, A_coarse = r.src_areas, r.dst_areas
    @inbounds for p in 1:6, j in 1:Nc, i in 1:Nc
        acc = 0.0
        for jj in 1:R, ii in 1:R
            f = _flat_fine_cell(p, i, j, ii, jj, R, Nf)
            acc += Float64(src[f]) * A_fine[f]
        end
        c = _flat_cell(p, i, j, Nc)
        dst[c] = acc / A_coarse[c]
    end
    return dst
end

function _nested_apply!(dst::AbstractVector, r::NestedCubedSphereRegrid{Prolongation},
                        src::AbstractVector)
    R = r.ratio
    Nc = _panel_edge(length(r.src_areas))
    Nf = R * Nc
    A_coarse, A_fine = r.src_areas, r.dst_areas
    @inbounds for p in 1:6, j in 1:Nc, i in 1:Nc
        block_area = 0.0
        for jj in 1:R, ii in 1:R
            block_area += A_fine[_flat_fine_cell(p, i, j, ii, jj, R, Nf)]
        end
        c = _flat_cell(p, i, j, Nc)
        value = Float64(src[c]) * (A_coarse[c] / block_area)
        for jj in 1:R, ii in 1:R
            dst[_flat_fine_cell(p, i, j, ii, jj, R, Nf)] = value
        end
    end
    return dst
end

function _check_nested_lengths(dst, r::NestedCubedSphereRegrid, src)
    size(dst, 1) == length(r.dst_areas) ||
        throw(DimensionMismatch("dst leading dim $(size(dst, 1)) ≠ regridder n_dst $(length(r.dst_areas))"))
    size(src, 1) == length(r.src_areas) ||
        throw(DimensionMismatch("src leading dim $(size(src, 1)) ≠ regridder n_src $(length(r.src_areas))"))
    return nothing
end

function apply_regridder!(dst::AbstractVector, r::NestedCubedSphereRegrid, src::AbstractVector)
    _check_nested_lengths(dst, r, src)
    return _nested_apply!(dst, r, src)
end

function apply_regridder!(dst::AbstractArray, r::NestedCubedSphereRegrid, src::AbstractArray)
    _check_nested_lengths(dst, r, src)
    dst_flat = reshape(dst, length(r.dst_areas), :)
    src_flat = reshape(src, length(r.src_areas), :)
    size(dst_flat, 2) == size(src_flat, 2) || throw(DimensionMismatch(
        "trailing shape mismatch: dst $(size(dst_flat, 2)) ≠ src $(size(src_flat, 2))"))
    for col in axes(dst_flat, 2)
        _nested_apply!(view(dst_flat, :, col), r, view(src_flat, :, col))
    end
    return dst
end
