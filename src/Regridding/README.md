# Regridding

Conservative regridding between `LatLonMesh`, `CubedSphereMesh`, and
`ReducedGaussianMesh`, built on ConservativeRegridding.jl (CR.jl).

A regridder is a sparse matrix of spherical polygon intersection areas, with
the source and destination cell areas. It is built once per mesh pair, can be
cached on disk, and is applied to fields whose first axis is the flattened
horizontal index.

`src/AtmosTransport.jl` loads this folder after `Adjoints` and before
`Preprocessing`, `Visualization`, and `Models`. It imports only `Grids`: mesh
types, panel conventions, cubed-sphere definition accessors, and
`panel_cell_corner_lonlat`. Its consumers are `Preprocessing` (met data to
target grids), `Models.InitialConditionIO` (initial conditions and surface-flux
inventories at run setup), and `Visualization` (cubed-sphere to lon-lat
rasters). Operators and the stepping loop do not call it.

## Entry Points

- `build_regridder(src, dst; normalize=false, cache_dir=nothing, kwargs...)`
  in [`weights_io.jl`](weights_io.jl). It returns an `IdentityRegrid` when the
  meshes are equivalent. Otherwise it returns a `ConservativeRegridding.Regridder`,
  loaded from or saved to `cache_dir` when one is given. Exported at the top
  level.
- `apply_regridder!(dst, regridder, src)` in [`weights_io.jl`](weights_io.jl)
  and [`identity_regrid.jl`](identity_regrid.jl). Axis 1 is the flat
  horizontal index; trailing axes are looped column by column. Exported at the
  top level.
- `save_regridder(path, r)` and `load_regridder(path)` round-trip a regridder
  through JLD2.
- `save_esmf_weights(path, r; src_shape, dst_shape, src_grid_name, dst_grid_name)`
  writes an ESMF offline-weights NetCDF file (`S`, `row`, `col`, `frac_a`,
  `frac_b`, `area_a`, `area_b`).
- `IdentityRegrid` and `meshes_equivalent(src, dst)` are in
  [`identity_regrid.jl`](identity_regrid.jl).
- `cubed_sphere_face_corners(mesh)` in [`treeify_meshes.jl`](treeify_meshes.jl)
  returns six `(Nc+1) × (Nc+1)` Float64 unit-sphere corner matrices.

## File Map

- [`Regridding.jl`](Regridding.jl): module, imports, exports, and a docstring
  with the workflow and supported-mesh table
- [`treeify_meshes.jl`](treeify_meshes.jl): `GOCore.best_manifold` and
  `Trees.treeify` methods, the spatial trees CR.jl searches. Lat-lon uses a
  quadtree. Cubed sphere uses six panel quadtrees, with
  `YReversedCellBasedGrid` for left-handed panels. Reduced Gaussian uses one
  tree per ring sector.
- [`identity_regrid.jl`](identity_regrid.jl): `IdentityRegrid`,
  `meshes_equivalent`, `identity_regrid_or_nothing`, and a `copyto!`
  `apply_regridder!`
- [`weights_io.jl`](weights_io.jl): cache key (`_hash_mesh!`,
  `_regridder_cache_key`), `build_regridder`, JLD2 save/load, ESMF export,
  and `apply_regridder!` for `Regridder`

## Invariants

- The cache key is the SHA-1 of the string
  `atmostransport_regridder_v<_REGRIDDER_CACHE_VERSION>` (currently 3), the
  `normalize` flag, and `_hash_mesh!` of source and destination. `_hash_mesh!`
  writes geometry only:
  - lat-lon: `Nx`, `Ny`, `λᶠ`, `φᶠ`, `radius`
  - cubed sphere: `Nc`, definition, coordinate-law, and centre-law tags,
    convention type, longitude offset, `radius`
  - reduced Gaussian: `nlon_per_ring`, `latitudes`, `lat_faces`, `radius`

  Meshes with different radii never share a cache file. Halo width and
  precision are not in the key.
- Cubed-sphere weights do not depend on precision: `best_manifold` and the
  corners are Float64 for every mesh `FT`, so Float32 and Float64 runs share
  them. Lat-lon and reduced-Gaussian meshes keep their coordinates in mesh
  `FT`, so their keys and polygons can differ between precisions.
- `meshes_equivalent` compares the same fields `_hash_mesh!` writes; meshes of
  different types are never equivalent. `build_regridder` checks it before the
  cache, so identical meshes cost one `copyto!`.
- `IdentityRegrid` stores `mesh` and its cell areas as `src_areas` and
  `dst_areas` (flat cell order of `apply_regridder!`), so density ↔ cell-total
  conversions work as with a `Regridder`. It has no `intersections`: it cannot
  be saved as weights, and `normalize` does not scale its areas.
- Extra CR.jl `kwargs` together with `cache_dir` throw `ArgumentError`.
  `load_regridder` errors on a `format_version` mismatch, and `save_regridder`
  writes a temp file and then `mv`s it into place.
- `normalize=false` (the default) stores raw intersection areas in m², and
  CR.jl `regrid!` divides by the destination area (the xESMF convention).
- Flat index: lat-lon `i + (j-1)Nx`; cubed sphere `(p-1)Nc² + i + (j-1)Nc`
  in file order (GEOS-native panels 4 and 5 are flipped only inside the
  tree); reduced Gaussian `ring_offsets[j] + i - 1`, rings south to north,
  longitudes from 0°.
- Reduced-Gaussian rings are split into at most four sectors, so no sector
  spans a hemisphere. Polar face latitudes are clamped 0.001° from the poles,
  and each ring needs at least 3 cells.
- `build_regridder(src, dst)` calls `ConservativeRegridding.Regridder(dst, src)`:
  CR.jl takes the destination first.

## Common Tasks

- New mesh type: add `best_manifold`, `Trees.treeify`, `_hash_mesh!`, and a
  `meshes_equivalent` method over the same fields.
- Geometry change that invalidates old caches: bump
  `_REGRIDDER_CACHE_VERSION` in [`weights_io.jl`](weights_io.jl).
- Cache directory: this module has no default.
  - Preprocessing reads `regridder_cache_dir`
    (`../Preprocessing/target_geometry.jl`).
  - Runtime surface fluxes use `_regrid_cache_dir()`
    (`../Models/initial_conditions/surface_flux.jl`, overridable with
    `ATMOSTR_REGRID_CACHE_DIR`).
  - Both default to `~/.cache/AtmosTransport/cr_regridding`.
  - The cubed-sphere initial-condition path and `Visualization` use no disk
    cache.

## Tests And Docs

- [`../../test/core/test_identity_regrid.jl`](../../test/core/test_identity_regrid.jl):
  equivalence and passthrough
- [`../../test/core/test_float32_conservation.jl`](../../test/core/test_float32_conservation.jl):
  Float32 meshes give Float64 corners and manifold
- [`../../test/core/test_binary_planet_radius.jl`](../../test/core/test_binary_planet_radius.jl):
  the runtime source mesh shares the destination radius
- The `test/regridding/` tier is part of the default `test/runtests.jl`
  selection: `test_conservation.jl`, `test_cubed_sphere_corners.jl`,
  `test_serialization.jl` (cache hit), `test_transpose.jl`,
  `test_reduced_gaussian_stub.jl`
- API page: [`../../docs/src/api/regridding.md`](../../docs/src/api/regridding.md)
