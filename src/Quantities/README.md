# Quantities

Dispatch traits that classify a horizontal field by how it must be regridded.

The traits are zero-size singleton types. Callers pass one as the last
positional argument of a regrid helper; the traits never wrap arrays.
`src/AtmosTransport.jl` loads this folder after `Architectures` and
`SectionTimer` and before `Parameters`. It depends on nothing else in the
package.

## Entry Points

All four traits are defined in [`Quantities.jl`](Quantities.jl) as subtypes of
`QuantityKind`:

| Trait | Examples | Regrid rule |
|---|---|---|
| `IntensiveCellField` | `ps`, `qv`, `T`, mixing ratios | pass straight to `apply_regridder!` |
| `ExtensiveCellField` | `m` (kg/cell), tracer mass | divide by source area, regrid, multiply by destination area |
| `HorizontalVectorField` | cell-centre `(u, v)` | not a scalar regrid: regrid components as intensive, then rotate to panel-local axes |
| `HorizontalFluxField` | window-summed `am`, `bm` | not a scalar regrid: face fluxes are rebuilt from regridded winds |

## File Map

- [`Quantities.jl`](Quantities.jl): module docstring with the taxonomy, the
  four types, and exports

## Where It Is Used

`Regridding` implements the per-trait rule and `Preprocessing` passes the
traits:

- `apply_regridder!(dst, r, src, kind)` in
  [`../Regridding/quantity_regrid.jl`](../Regridding/quantity_regrid.jl)
  implements it for every regridder (`ConservativeRegridding.Regridder`,
  `IdentityRegrid`, `NestedCubedSphereRegrid`).
  `HorizontalVectorField` and `HorizontalFluxField` throw `ArgumentError`
  there. Winds go through `rotate_winds_to_panel_local!`, and face fluxes
  through `reconstruct_cs_fluxes!`.
- `regrid_3d_to_cs_panels!` and `regrid_2d_to_cs_panels!` in
  [`../Preprocessing/cs_transport_helpers.jl`](../Preprocessing/cs_transport_helpers.jl)
  take `kind::QuantityKind = IntensiveCellField()` and pass it on.
- The call sites in `../Preprocessing/transport_binary/`
  (`cubed_sphere_regrid.jl`, `cubed_sphere_spectral.jl`,
  `era5_n320_regrid.jl`, `merra2_latlon_regrid.jl`) tag `m` with
  `ExtensiveCellField()` and surface or state fields with
  `IntensiveCellField()`.

## Conventions

- The default trait is `IntensiveCellField()`. A per-cell total passed without
  `ExtensiveCellField()` is regridded as a density. The result is then distorted
  in proportion to the variation of source-cell area, which is largest at the
  lat-lon poles. Always tag `m` and other per-cell totals.
- `ExtensiveCellField` reads `regridder.src_areas` and `regridder.dst_areas`;
  a `ConservativeRegridding.Regridder`, an `IdentityRegrid` and a
  `NestedCubedSphereRegrid` all carry them.
- Vertical layer-merging uses a separate trait family:
  `AbstractFieldKind` (`MassField`, `IntensiveCenterField`, `PressureFluxField`,
  ...) in
  [`../Preprocessing/vertical_transforms.jl`](../Preprocessing/vertical_transforms.jl).
  The two families are not interchangeable.

## Tests And Docs

- [`../../test/regridding/test_quantity_regrid.jl`](../../test/regridding/test_quantity_regrid.jl)
  tests every trait through `apply_regridder!`, conservation totals computed
  in the test, and that the CS regrid helpers give the same
  bits as before the rule moved to `Regridding`.
  [`../../test/core/test_cs_face_fluxes.jl`](../../test/core/test_cs_face_fluxes.jl)
  calls `regrid_3d_to_cs_panels!` with the default trait.
  [`../../test/core/test_vertical_transforms.jl`](../../test/core/test_vertical_transforms.jl)
  covers the vertical `AbstractFieldKind` family.
- API page: [`../../docs/src/api/infrastructure.md`](../../docs/src/api/infrastructure.md)
