# Regridding API

These are the offline conservative-regridding construction, persistence, and
application helpers. See [Regridding](@ref) for the physical
contract.

- `build_regridder` returns an `IdentityRegrid` for equivalent meshes, a
  `NestedCubedSphereRegrid` (`Restriction` or `Prolongation`) for nested cubed
  spheres, and a ConservativeRegridding.jl `Regridder` otherwise.
- `apply_regridder!(dst, r, src)` maps an intensive field;
  `apply_regridder!(dst, r, src, kind)` takes the quantity kind
  (`IntensiveCellField()`, `ExtensiveCellField()` for per-cell totals),
  `source_coverage` gives the covered area of each source cell.
- `coarsen_sum!`, `coarsen_area_weighted!`, `coarsen_xface_sum!` and
  `coarsen_yface_sum!` are the cubed-sphere cell and face restriction
  kernels of the transport-binary coarseners.
- `default_regrid_cache_dir` is the shared weight cache.

```@autodocs
Modules = [AtmosTransport.Regridding]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```
