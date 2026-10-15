# Regridding

Whenever the preprocessor source mesh and target mesh differ, mass
fields move through a **conservative regridder** built on top of
[ConservativeRegridding.jl](https://github.com/JuliaGeo/ConservativeRegridding.jl).
This page covers the public surface, the JLD2 weights cache, and the
mass-consistency correction the preprocessor applies after every
regrid.

## Public API

```julia
regridder = build_regridder(src_mesh, dst_mesh;
                             normalize = false,
                             cache_dir = nothing)   # opt-in caching
apply_regridder!(dst, regridder, src)                         # intensive field
apply_regridder!(dst, regridder, src, ExtensiveCellField())   # per-cell totals
regrid_conservation_report(regridder, src, dst, ExtensiveCellField())
source_coverage(regridder)                                    # covered area per source cell
```

- **`build_regridder`** chooses, in this order: `IdentityRegrid` for
  equivalent meshes, `NestedCubedSphereRegrid` for nested cubed spheres
  (below), else a `ConservativeRegridding.Regridder` carrying
  `(intersections, dst_areas, src_areas)`.
- **`apply_regridder!(dst, r, src)`** does an in-place column-by-column regrid
  of an intensive field (a value per area or per point,
  ``d_t = \sum_s W_{ts} s_s / A^{\rm dst}_t``) for any n-D source where the
  leading dimension is the horizontal one (3-D `(ncell_src, Nz)` →
  `(ncell_dst, Nz)`, 2-D `(ncell_src,)` → `(ncell_dst,)`).
- **`apply_regridder!(dst, r, src, kind)`** takes the quantity kind from
  `Quantities`. `IntensiveCellField()` is the call above.
  `ExtensiveCellField()` regrids a per-cell total `m` through its density,
  ``d_t = A^{\rm dst}_t \, \mathcal{R}(m / A^{\rm src})_t = \sum_s W_{ts} m_s / A^{\rm src}_s``,
  so ``\sum_t d_t = \sum_s m_s`` whatever polygon areas the regridder uses.
  The density of one column is formed in a vector of length `ncell_src`,
  allocated per call unless the caller passes a reusable one as
  `scratch = buf` (the cubed-sphere preprocessing workspace does, so its
  regrids allocate nothing after the first call). Vector and face-flux kinds
  throw.
- **`regrid_conservation_report`** returns `(; src_total, dst_total,
  rel_error)` with Float64 sums; the leading dimensions of `src` and `dst`
  must match the regridder's cell counts.
- **`source_coverage(r)`** returns the area of each source cell that the
  destination covers, ``C_s = \sum_t W_{ts}``. It equals the source polygon
  area when the destination tiles the source. A reduced-Gaussian destination
  whose adjacent rings have different longitude counts leaves great-circle
  slivers uncovered (6.3e-4 of the sphere on O24), so the runtime surface-flux
  sources regrid their per-cell rates as ``\sum_s W_{ts} m_s / C_s``, which
  keeps ``\sum_s m_s`` on every destination.
- `normalize = true` scales weights so the largest intersection gets
  weight 1 — useful for diagnostic plots, **never** for mass fields.
- `cache_dir = nothing` (the default) skips the JLD2 cache and
  rebuilds the weights every call. Pass a directory to opt in;
  `default_regrid_cache_dir()` gives the shared default
  (`ATMOSTR_REGRID_CACHE_DIR`, else `~/.cache/AtmosTransport/cr_regridding`).
  The preprocessing TOMLs set it via `[grid] regridder_cache_dir`.

`build_regridder` and the cache live in `src/Regridding/weights_io.jl`, the
quantity kinds in `src/Regridding/quantity_regrid.jl`, and the nested
cubed-sphere pieces in `src/Regridding/cs_nesting.jl`.

## Supported topology pairs

The same generic API covers every (source, target) pair:

| Source ↓  Target → | LatLon | Reduced Gaussian | CubedSphere |
|---|---|---|---|
| **LatLon** | identity (or LL→LL regrid) | **per-ring conservative** (LL→RG specifically uses the per-ring R-tree workaround below) | conservative |
| **Reduced Gaussian** | conservative (generic path) | identity (when the rings match) | conservative |
| **CubedSphere** | conservative | conservative | identity |

Same-mesh paths (LL → LL with identical `(Nx, Ny)`, CS → CS with
identical `Nc` and panel convention, etc.) automatically resolve to
**`IdentityRegrid`** through the structural-equivalence check
described below.

## `IdentityRegrid` — zero-overhead passthrough

```julia
# build_regridder returns an IdentityRegrid when meshes_equivalent(src, dst);
# direct construction is also fine:
regridder = IdentityRegrid(mesh)
apply_regridder!(dst, regridder, src)   # equivalent to copyto!(dst, src)
```

Used heavily by the **GEOS-CS → CS passthrough** preprocessing path
where the source and target meshes are structurally equivalent (same
type, same shape parameters, same panel convention). The
equivalence check is `meshes_equivalent(src, dst)` in
`src/Regridding/identity_regrid.jl`, not a Julia object-identity
(`===`) test, so two independently constructed meshes with the same
parameters still resolve to the identity path. Type-stable, zero
allocation, no `if` branch in the hot loop. Any future grid type
that defines `meshes_equivalent` for itself plugs into
`IdentityRegrid` without code changes.

## Nested cubed spheres

Two cubed spheres with the same definition (coordinate law, centre law, panel
convention, longitude offset) and radius whose edge sizes differ by an integer
factor ``R`` are nested: coarse cell ``(i, j)`` of a panel is the union of
the fine cells ``((i-1)R + ii, (j-1)R + jj)``, ``1 \le ii, jj \le R``.
`meshes_nested(a, b)` checks the definitions and evaluates the corners
(coarse corner ``(i, j)`` = fine corner ``((i-1)R+1, (j-1)R+1)`` to 1e-12 on
the unit sphere). For such a pair `build_regridder` returns a
`NestedCubedSphereRegrid` with the implicit intersection matrix

- restriction (fine → coarse): ``W_{cf} = A_f``, so
  ``d_c = \sum_{f \in c} A_f s_f / A_c``;
- prolongation (coarse → fine): ``W_{fc} = A_c A_f / \sum_{f' \in c} A_{f'}``, so
  ``d_f = s_c A_c / \sum_{f' \in c} A_{f'}``.

Both satisfy ``\sum_t W_{ts} = A^{\rm src}_s``: per-cell totals are block-summed
or split by area fraction exactly, and the result equals the polygon
regridder to about 1e-12. Areas are the Float64 mesh areas. Nothing is cached
on disk.

The same module holds the block kernels of the transport-binary coarseners:
`coarsen_sum!` (extensive amounts, summed in the data precision) and
`coarsen_area_weighted!` (intensive fields,
``\mathrm{fl}\big(\sum A_f s_f / \sum A_f\big)`` with Float64 sums and one
rounding).

## The JLD2 weights cache

Building the conservative-regridder weights for a real grid pair is
the single most expensive thing the preprocessor does on day 1
(seconds for low resolution, minutes for C180 / O320). The cache
makes day 2 onward effectively free.

- **Cache directory:** opt-in via the `cache_dir` kwarg on
  `build_regridder` (the preprocessing TOMLs set
  `[grid] regridder_cache_dir = "~/.cache/AtmosTransport/cr_regridding"`).
  Runtime surface-flux sources regridded from lat-lon files always use the
  cache, in `default_regrid_cache_dir()`: `ATMOSTR_REGRID_CACHE_DIR` if set,
  else `~/.cache/AtmosTransport/cr_regridding`. Cache files are published
  atomically with mode `0o666 & ~umask`.
- **Cache key:** SHA-1 of a tuple containing:
  - source and destination mesh **type** + shape parameters
    (`(Nx, Ny)`, `Nc`, `nlon_per_ring`, panel convention, …),
  - source and destination radii (Earth radius is stable, but the
    cache key still includes it),
  - `normalize` flag.
- **Cache file:** `regridder_<SHA1>.jld2`, JLD2-loaded (not memory-
  mapped) on subsequent calls. JLD2 load is ~10 ms; rebuilding the
  weights is seconds-to-minutes, so the cache is the difference
  between "instant" and "wait" on warm runs.

The cache key intentionally does **not** include source / target
data — it's purely geometric. Two preprocessing runs over different
days share the same regridder JLD2.

## Per-ring regrid for `LatLon → ReducedGaussian`

The generic `ConservativeRegridding.MultiTreeWrapper` path has a
known degeneracy on reduced-Gaussian meshes (some ring boundaries
get pruned because of LCM segmentation overlap). For the **LL → RG**
case specifically, `weights_io.jl` builds **one R-tree per ring**
and stitches the results; the reverse direction (RG → anything)
uses the generic conservative path. The per-ring workaround is the
reason the RG-target preprocessing takes a bit longer to build than
equivalent LL or CS targets at first run.

## Mass closure across topologies

Conservative regridding preserves total mass exactly. The earlier
"per-level mass correction" helper
(`_enforce_perlevel_mass_consistency!`) has been **removed** — it
was a band-aid for a regrid path that did not respect the
extensive / intensive distinction.

The current path uses `regrid_3d_to_cs_panels!` with an
`ExtensiveCellField()` flag on both sides of the regrid. The
extensive-on-both-sides contract converts through density, so the
spatial distribution is preserved by construction and the per-level
mass identity holds to floating-point tolerance without any
post-regrid correction.

If you are extending the regridding subsystem with a new mesh pair
and the write-time replay gate fails because the per-level totals
drift: the first check is whether you are using
`ExtensiveCellField()` consistently — not whether you need to
re-introduce a post-regrid correction.

## What's next

- [Conventions cheat sheet](@ref) — units, replay tolerances, level
  orientation, panel conventions.
- *Tutorials* — once the bundle artifact lands, a Literate tutorial
  will demo a non-trivial cross-topology regrid (ERA5 spectral → CS).
