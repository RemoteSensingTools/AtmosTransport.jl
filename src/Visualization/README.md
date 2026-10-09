# Visualization

Makie-free data layer for AtmosTransport NetCDF snapshots. It opens a
snapshot, reads one horizontal field per time in the topology the snapshot
stores (reduced-Gaussian snapshots are already lon-lat rasters), and rasterizes
it to lon-lat.

The plotting functions here are stubs that throw until a Makie backend is
loaded. Their real methods live in
[`../../ext/AtmosTransportMakieExt.jl`](../../ext/AtmosTransportMakieExt.jl),
which loads when `Makie` is present (`[extensions]` in `Project.toml`).

`src/AtmosTransport.jl` loads this folder after `Preprocessing` and before
`Models`. It imports from `Grids` (`LatLonMesh`, `CubedSphereMesh`, panel
conventions), `Regridding` (`build_regridder`, `apply_regridder!`), and
`Parameters` (`IFS_EARTH_RADIUS`). `open_snapshot`, `fieldview`, `mapplot`,
and `movie` are exported at the top level.

## Entry Points

All in [`Visualization.jl`](Visualization.jl):

- `open_snapshot(path) -> SnapshotDataset` reads the metadata and picks a
  topology:
  - cubed sphere if the file has `Xdim` and `nf` dimensions
  - otherwise lon/lat; reduced Gaussian if the `grid_type` attribute contains
    `reduced_gaussian`

  The returned object holds no open file handle.
- `available_variables`, `snapshot_times`, `snapshot_topology`, and
  `frame_indices(snapshot, times=:all)` query an open snapshot
- `fieldview(snapshot, variable; transform=:column_mean, time=1, level=nothing, unit=:native) -> HorizontalField`
  - `time` is a frame index (integer) or a snapshot hour (the nearest frame is
    taken)
  - `unit=:ppm` multiplies the values by 1e6
  - Cubed-sphere transforms:
    - `:column_mean` uses the stored `<var>_column_mean` if present; otherwise
      it computes an air-mass-weighted mean from full levels
    - `:column_sum`
    - `:level_slice` (requires `level`)
    - `:surface_slice` (defaults to level `nlevel`, the last level)
  - Lat-lon and reduced-Gaussian files support `:column_mean` only
- `as_raster(field; resolution=(360, 181), cache=SnapshotRegridCache()) -> RasterField`
  - lat-lon and reduced-Gaussian values pass through unchanged
  - cubed-sphere fields are conservatively regridded to lon-lat. The
    regridder is kept in `SnapshotRegridCache`, keyed on
    `(Nc, resolution, definition)`; the definition (coordinate and center
    laws, panel convention, longitude offset) comes from the snapshot's
    `cs_*` attributes, or the convention's default for older files.
- `robust_colorrange(fields; trim=(0.01, 0.99))` returns a color range from
  quantiles of the finite values
- `PlotSpec(variable; transform, level, title, unit)` describes one panel of
  `movie_grid`
- Makie stubs that throw `ArgumentError` until a backend is loaded: `mapplot`,
  `mapplot!`, `snapshot_grid`, `movie`, `movie_grid`,
  `catrine_map_curtains`, `catrine_map_curtains_3way`

## File Map

- [`Visualization.jl`](Visualization.jl): the whole module. It holds the
  snapshot topology types, `SnapshotDataset`, `HorizontalField`, and
  `RasterField`, plus the NetCDF readers, cubed-sphere column reductions,
  cubed-sphere to lon-lat rasterization, and the Makie stubs.
- Related: [`../../ext/AtmosTransportMakieExt.jl`](../../ext/AtmosTransportMakieExt.jl)
  has the Makie methods for every stub, including the CATRINE
  map-and-curtain comparison products

## Conventions

- Array layouts:
  - lat-lon and reduced-Gaussian fields are `(lon, lat)`
  - a cubed-sphere horizontal field is `(Xdim, Ydim, nf)`
  - cubed-sphere 3-D file variables are `(Xdim, Ydim, nf, lev, time)`
- Reduced-Gaussian snapshots are read from the writer's nearest-neighbour
  lon/lat diagnostic raster (`regridding` attribute), not from native cells.
- Cubed-sphere level selection uses the original model level indices. If a
  variable stores only selected levels (`lev_selected` dimension), a level that
  was not written raises `ArgumentError`, and `:column_mean` is refused unless
  a stored column mean exists.
- Cubed-sphere rasterization rebuilds the mesh from the `Nc` and
  `panel_convention` (`gnomonic` or `geos_native`) attributes. It uses the
  default definition for that convention, puts both meshes on
  `IFS_EARTH_RADIUS`, and keeps no disk cache.
- Snapshot times are read as raw Float64 hours (`ds["time"].var[:]`), skipping
  CF time decoding.
- To plot, load a Makie backend such as CairoMakie before calling the
  plotting functions.

## Tests And Docs

- [`../../test/core/test_visualization_snapshots.jl`](../../test/core/test_visualization_snapshots.jl):
  lat-lon and cubed-sphere field views and rasters
- [`../../test/core/test_netcdf_stream.jl`](../../test/core/test_netcdf_stream.jl):
  streamed snapshots read back through `open_snapshot` / `fieldview`
- [`../../test/core/test_public_api_surface.jl`](../../test/core/test_public_api_surface.jl):
  top-level exports
- API page: [`../../docs/src/api/output_visualization.md`](../../docs/src/api/output_visualization.md)
