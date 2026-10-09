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
apply_regridder!(dst, regridder, src)
```

- **`build_regridder`** returns a wrapped `ConservativeRegridding.Regridder`
  carrying `(intersections, dst_areas, src_areas)`.
- **`apply_regridder!`** does an in-place column-by-column regrid for
  any n-D source where the leading dimension is the horizontal one
  (3-D `(ncell_src, Nz)` → `(ncell_dst, Nz)`, 2-D `(ncell_src,)` →
  `(ncell_dst,)`).
- `normalize = true` scales weights so the largest intersection gets
  weight 1 — useful for diagnostic plots, **never** for mass fields.
- `cache_dir = nothing` (the default) skips the JLD2 cache and
  rebuilds the weights every call. Pass an absolute path
  (e.g. `~/.cache/AtmosTransport/cr_regridding`) to opt in. The
  preprocessing TOMLs set this via `[grid] regridder_cache_dir`.

Both functions live in `src/Regridding/weights_io.jl`.

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

## The JLD2 weights cache

Building the conservative-regridder weights for a real grid pair is
the single most expensive thing the preprocessor does on day 1
(seconds for low resolution, minutes for C180 / O320). The cache
makes day 2 onward effectively free.

- **Cache directory:** opt-in via the `cache_dir` kwarg on
  `build_regridder` (the preprocessing TOMLs set
  `[grid] regridder_cache_dir = "~/.cache/AtmosTransport/cr_regridding"`).
  Runtime surface-flux sources regridded from lat-lon files always use the
  cache, in `ATMOSTR_REGRID_CACHE_DIR` if set, else
  `~/.cache/AtmosTransport/cr_regridding`.
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
