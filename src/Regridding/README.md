# Regridding

Conservative regridding between `LatLonMesh`, `CubedSphereMesh`, and
`ReducedGaussianMesh`, built on ConservativeRegridding.jl (CR.jl).

A regridder is a sparse matrix of spherical polygon intersection areas, with
the source and destination cell areas. It is built once per mesh pair, can be
cached on disk, and is applied to fields whose first axis is the flattened
horizontal index. Equivalent meshes get an `IdentityRegrid` and nested cubed
spheres a `NestedCubedSphereRegrid`, which need no intersection matrix.
Fields are regridded by quantity kind: intensive fields directly, per-cell
totals through their density. The folder also holds the one copy of the
cubed-sphere block kernels used by the transport-binary coarseners.

`src/AtmosTransport.jl` loads this folder after `Adjoints` and before
`Preprocessing`, `Visualization`, and `Models`. It imports `Grids` (mesh
types, panel conventions, cubed-sphere definition accessors,
`panel_cell_corner_lonlat`, `cs_corner_xyz`), `Quantities` (the quantity-kind
traits) and `AtomicFiles` (`atomic_publish` for cached weights). Its consumers
are `Preprocessing` (met data to target grids, block coarsening),
`Models.InitialConditionIO` (initial conditions and surface-flux inventories at
run setup), and `Visualization` (cubed-sphere to lon-lat rasters). Operators
and the stepping loop do not call it.

## Entry Points

- `build_regridder(src, dst; normalize=false, cache_dir=nothing, kwargs...)`
  in [`weights_io.jl`](weights_io.jl), in this order:
  1. `IdentityRegrid` when `meshes_equivalent(src, dst)`;
  2. `NestedCubedSphereRegrid` when `meshes_nested(src, dst)`;
  3. a `ConservativeRegridding.Regridder`, loaded from or saved to
     `cache_dir` when one is given.

  Exported at the top level.
- `apply_regridder!(dst, regridder, src)` in [`weights_io.jl`](weights_io.jl),
  [`identity_regrid.jl`](identity_regrid.jl) and
  [`cs_nesting.jl`](cs_nesting.jl): an intensive field (value per area or per
  point). Axis 1 is the flat horizontal index; trailing axes are looped column
  by column. Exported at the top level.
- `apply_regridder!(dst, regridder, src, kind::QuantityKind)` in
  [`quantity_regrid.jl`](quantity_regrid.jl), for any regridder:
  - `IntensiveCellField()`: the three-argument call;
  - `ExtensiveCellField()`: `src ./ src_areas` (formed per column in the
    element type of `src`, in the keyword `scratch` or a vector allocated per
    call), regrid, `× dst_areas`;
  - `HorizontalVectorField()`, `HorizontalFluxField()`: `ArgumentError`.
- `source_coverage(r)` in [`quantity_regrid.jl`](quantity_regrid.jl): the
  area of each source cell that the destination covers, `C_s = Σ_t W_ts`
  (`r.src_areas` for `IdentityRegrid` and `NestedCubedSphereRegrid`). The
  static surface-flux sources divide their per-cell rates by it, so their
  totals survive destinations that do not tile the source.
- `default_regrid_cache_dir()` in [`weights_io.jl`](weights_io.jl):
  `ATMOSTR_REGRID_CACHE_DIR` if set and not empty, else
  `~/.cache/AtmosTransport/cr_regridding`.
- `meshes_nested(a, b)`, `NestedCubedSphereRegrid{Restriction|Prolongation}`
  (`AbstractNestingDirection`), `coarsen_sum!(dst, src, ratio)` and
  `coarsen_area_weighted!(dst, src, area, ratio)` in
  [`cs_nesting.jl`](cs_nesting.jl).
- `save_regridder(path, r)` and `load_regridder(path)` round-trip a regridder
  through JLD2.
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
  `meshes_equivalent`, `_same_cs_definition` (shared with `meshes_nested`),
  `identity_regrid_or_nothing`, and a `copyto!` `apply_regridder!`
- [`cs_nesting.jl`](cs_nesting.jl): the block kernels `coarsen_sum!`,
  `coarsen_area_weighted!`, `coarsen_xface_sum!`, `coarsen_yface_sum!`, `meshes_nested` (with the corner check),
  the nesting directions, `NestedCubedSphereRegrid` and its
  `apply_regridder!`, `nested_regrid_or_nothing`
- [`weights_io.jl`](weights_io.jl): cache key (`_hash_mesh!`,
  `_regridder_cache_key`), `default_regrid_cache_dir`, `build_regridder`,
  JLD2 save/load, and `apply_regridder!` for `Regridder`
- [`quantity_regrid.jl`](quantity_regrid.jl): `apply_regridder!` by quantity
  kind and `source_coverage`

## Invariants

- A `ReducedGaussianMesh` destination does not tile the sphere where adjacent
  rings have different longitude counts: its cells have great-circle edges,
  so the latitude two such rings share is two different polylines and the
  sliver between them belongs to no cell. On the octahedral O24 grid (counts
  20, 24, 28, … per ring) 6.3e-4 of the sphere is uncovered, and a 1° (0.5°)
  source cell loses up to 6.4 % (13.5 %) of its area; the regridder's
  `dst_areas` there are the intersection row sums. An extensive regrid through
  `r.src_areas` loses those slivers; divide by `source_coverage(r)` instead to
  keep the total.

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
- `meshes_nested` requires the same cubed-sphere definition and radius
  (`_same_cs_definition`), different `Nc` with an integer ratio `R`, and
  evaluates the nesting: coarse corner `(i, j)` of every panel must be fine
  corner `((i − 1)R + 1, (j − 1)R + 1)` to `1e-12` on the unit sphere. Fine
  cell `((i − 1)R + ii, (j − 1)R + jj)` then lies in coarse cell `(i, j)` in
  file order, for both panel conventions.
- `NestedCubedSphereRegrid` remaps an intensive field as a conservative
  regridder with intersections `W` does, `d = W s ./ dst_areas`, with
  `W_cf = A_f` for restriction and `W_fc = A_c A_f / Σ_{f'∈c} A_f'` for
  prolongation. Both give `Σ_t W_ts = src_areas[s]`, so the extensive path is
  an exact block sum (restriction) or split by area fraction (prolongation).
  Its areas are Float64 (`CubedSphereMesh(FT = Float64)` areas) for every mesh
  precision. Built in O(Nc²), never cached; `normalize` and CR.jl `kwargs` do
  not apply to it, and it has no `intersections`.
- Nothing is cached for nested pairs, and the cached content of every other
  pair is unchanged, so `_REGRIDDER_CACHE_VERSION` stays 3; v3 files of nested
  pairs are no longer read.
- Block kernels (`coarsen_sum!`, `coarsen_area_weighted!`) take 2-D `(Nc, Nc)`
  or 3-D `(Nc, Nc, Nz)` panel arrays and an `Int` ratio; blocks are summed
  `ii` fastest. Sums accumulate in the array element type (bit-identical to
  the two earlier copies); area-weighted means accumulate numerator and
  denominator in Float64, divide by the block sum of fine areas (constants
  stay exact) and round once. Float64 areas are accepted with Float32 data.
- `apply_regridder!(…, ExtensiveCellField())` never modifies `src`. Its
  density buffer is the keyword `scratch` (length `n_src`, element type
  `eltype(src)`, not aliasing `src` or `dst`); without one it allocates one
  horizontal column per call (`similar(src, n_src)`). The preprocessing
  workspace passes `density_scratch`, and matrices are not reshaped, so warm
  `regrid_2d/3d_to_cs_panels!` calls allocate nothing.
- `IdentityRegrid` stores `mesh` and its cell areas as `src_areas` and
  `dst_areas` (flat cell order of `apply_regridder!`), so density ↔ cell-total
  conversions work as with a `Regridder`. It has no `intersections`: it cannot
  be saved as weights, and `normalize` does not scale its areas.
- Extra CR.jl `kwargs` together with `cache_dir` throw `ArgumentError`.
  `load_regridder` errors on a `format_version` mismatch, and `save_regridder`
  publishes through `AtomicFiles.atomic_publish` (sibling staging file, mode
  `0o666 & ~umask`, one replacing rename when complete), so a shared cache is
  not owner-only and readers never see it missing.
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
- Cache directory: `build_regridder` caches only when given `cache_dir`.
  - `default_regrid_cache_dir()` ([`weights_io.jl`](weights_io.jl)) is the
    shared default: `ATMOSTR_REGRID_CACHE_DIR` if set and not empty, else
    `~/.cache/AtmosTransport/cr_regridding`. Runtime surface fluxes use it
    (`../Models/initial_conditions/surface_flux_regridding.jl`).
  - Preprocessing reads `regridder_cache_dir` from its config
    (`../Preprocessing/target_geometry.jl`), with the same default path.
  - The cubed-sphere initial-condition path and `Visualization` use no disk
    cache.
- New nested-mesh pair: `meshes_nested` and the regridder handle cubed
  spheres only; another topology would add its own `meshes_nested` method and
  regridder type, checked in `build_regridder` before CR.jl.

## Tests And Docs

- [`../../test/core/test_identity_regrid.jl`](../../test/core/test_identity_regrid.jl):
  equivalence and passthrough
- [`../../test/core/test_float32_conservation.jl`](../../test/core/test_float32_conservation.jl):
  Float32 meshes give Float64 corners and manifold
- [`../../test/core/test_binary_planet_radius.jl`](../../test/core/test_binary_planet_radius.jl):
  the runtime source mesh shares the destination radius
- The `test/regridding/` tier is part of the default `test/runtests.jl`
  selection: `test_conservation.jl`, `test_cubed_sphere_corners.jl`,
  `test_serialization.jl` (identical weights after save/load and cache hit),
  `test_reduced_gaussian_stub.jl`,
  `test_quantity_regrid.jl` (extensive totals to 1e-13; the CS regrid helpers
  bit-identical to the pre-move code), `test_nested_cs_regrid.jl` (nesting
  test; C4 ↔ C8 conservation to 1e-13 and agreement with CR.jl to 1e-12; the
  block and face kernels against the independent Set A reference)
- API page: [`../../docs/src/api/regridding.md`](../../docs/src/api/regridding.md)

## Depends on

Grids, Quantities and AtomicFiles; ConservativeRegridding, GeometryOps,
GeometryOpsCore, GeoInterface, JLD2, LinearAlgebra, SHA and StaticArrays.

## Used by

Preprocessing, Models.InitialConditionIO and Visualization.

## Shared helpers owned here

The exported regridder constructors, application, persistence and coverage
interfaces, and `coarsen_sum!`, `coarsen_area_weighted!`,
`coarsen_xface_sum!`, `coarsen_yface_sum!`. Face restriction lives in
`cs_nesting.jl`: each coarse boundary face sums its fine faces in array
precision and the original order; fine faces inside coarse cells are omitted.

## Forward<->adjoint mirrors

No transport-adjoint kernels. Restriction/prolongation share nesting geometry;
conservative regridding and block restriction retain their own sum orders.

`meshes_equivalent` compares horizontal regrid geometry. Models' runtime
compatibility check additionally requires equal CS halo width and compares
representations of definitions/conventions; it stays separate to preserve its
acceptance policy (including signed-zero longitude offsets).
