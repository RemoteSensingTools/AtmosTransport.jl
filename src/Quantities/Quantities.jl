"""
    Quantities

Dispatch traits that classify a meteorological field by its
*regrid-and-conservation semantics*. Only `Preprocessing` uses them: the CS
regrid helpers (`regrid_2d_to_cs_panels!`, `regrid_3d_to_cs_panels!`, which
call `apply_regridder!`) take a kind argument so extensive/intensive
distinctions stay explicit at the type level.

The four kinds are not interchangeable. Confusing one for another is the
exact mistake that produced ~12× polar mass deficits in C180 ERA5
binaries (the regrid path treated extensive `m` as intensive, distorting
the spatial distribution proportional to source-cell area variation).

| Kind | Examples | Regrid handling |
|---|---|---|
| `IntensiveCellField` | `ps`, `qv`, `T`, mixing ratios, cell-center winds | Pass-through to `apply_regridder!`. |
| `ExtensiveCellField` | `m` (kg/cell), tracer mass, accumulated emissions mass | Convert via density: `src ./ src_areas` → regrid → `× dst_areas`. |
| `HorizontalVectorField` | cell-center `(u, v)` viewed as a tangent vector | Regrid components as intensive, then *rotate* into the target panel-local basis at every CS panel. |
| `HorizontalFluxField` | window-summed `am`, `bm`, accumulated face-flux integrals | Directed integrals across faces; correct handling needs face-aware regridding (MAPL `CONSERVE_HFLUX`-style) or reconstruction from regridded winds + pressure. The pipeline reconstructs from winds (`reconstruct_cs_fluxes!`). |

`IntensiveCellField` and `ExtensiveCellField` cover all scalar cell fields
and are the only kinds the regrid helpers accept. Passing
`HorizontalVectorField` or `HorizontalFluxField` makes them throw an
`ArgumentError`, because those fields need richer handling than scalar
regridding can give.

The trait types are small singletons. Use them as a final positional
argument on regrid helpers, not as wrapper types around arrays — staying
out of the array data type keeps the runtime hot loops untouched and
lets us introduce these dispatch tags incrementally.
"""
module Quantities

"""
    QuantityKind

Marker supertype for the four field-classification traits.
"""
abstract type QuantityKind end

"""
    IntensiveCellField <: QuantityKind

Per-area or point-value cell field: `ps`, `qv`, `T`, mixing ratios,
cell-center wind components, etc. Conservative area-weighted regridding
gives a correct destination-cell average without unit conversion.
"""
struct IntensiveCellField <: QuantityKind end

"""
    ExtensiveCellField <: QuantityKind

Per-cell total: `m` in kg/cell, tracer mass, accumulated emissions mass.
Conservative regridding requires conversion to density (intensive) on the
source grid, then re-multiplication by destination cell area to recover
the extensive total. Without this conversion, the regridded field is
distorted in proportion to source-grid area variation — at the LL pole,
cells are 230× smaller than at the equator, producing severe spatial
errors in CS polar cells.
"""
struct ExtensiveCellField <: QuantityKind end

"""
    HorizontalVectorField <: QuantityKind

Tangent-plane vector field defined at cell centers: cell-center
`(u, v)` winds. The components are individually intensive, but at every
CS panel the vector itself must be re-expressed in the target panel-local
basis. This requires a rotation step on top of component-wise regridding:
`rotate_winds_to_panel_local!` (Preprocessing) projects the regridded
east/north components onto panel-local face normals built from
`panel_cell_local_tangent_basis` (Grids). The scalar regrid helpers reject
this kind with an `ArgumentError`.
"""
struct HorizontalVectorField <: QuantityKind end

"""
    HorizontalFluxField <: QuantityKind

Directed face-flux integral: window-summed `am`, `bm`, accumulated mass
flux across a face over a time window. Correct regridding requires
face-aware operators (MAPL `CONSERVE_HFLUX`) or reconstruction from
regridded winds + pressure. The pipeline reconstructs face fluxes from
regridded cell-center winds and a target-grid pressure thickness
(`reconstruct_cs_fluxes!`); the scalar regrid helpers reject this kind with
an `ArgumentError`.
"""
struct HorizontalFluxField <: QuantityKind end

export QuantityKind, IntensiveCellField, ExtensiveCellField,
       HorizontalVectorField, HorizontalFluxField

end # module Quantities
