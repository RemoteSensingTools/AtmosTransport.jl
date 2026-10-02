# Output

`Output` owns the runtime snapshot contract and NetCDF file schema.

Runners should only decide when to sample state. They should call
`capture_snapshot(model; time_hours, halo_width, fields)` and
`write_snapshot_netcdf(path, frames, grid; mass_basis, options)` instead of
writing topology-specific NetCDF files directly.

## Files

- `snapshots.jl` defines `SnapshotFrame`, `SnapshotWriteOptions`, model-state
  capture, and compensated Float64 signed tracer totals.
- `selected_snapshots.jl` captures requested layers and backend column reductions.
- `snapshot_totals.jl` computes compensated signed totals without retaining full
  host tracer volumes on CUDA; Metal copies bounded slabs for CPU Float64 sums.
- `netcdf_stream.jl` appends single-file runtime output and records completed
  snapshots. Its owner must close the stream on every exit.
- `diagnostics.jl` derives VMR, column means, and mass-per-area fields.
- `netcdf_schema.jl` defines topology-specific dimensions, coordinates, and metadata.
- `netcdf_writer.jl` writes topology-specific payload variables through one public API.
- `observations/observation_sources.jl` defines the typed `[[output.observations.sources]]`
  descriptors (`OCO2LiteSource`, `ObsPackSource`, `TableSource`), the singleton
  mode / grouping / table-format types and quality filters they dispatch on, and
  their validation. The `_SOURCE_TYPES`, `_OBSERVATION_MODES`, `_SITE_GROUPINGS`
  and `_TABLE_FORMATS` tables are the parser's choices; the editor schema must
  match them (`test/core/test_observation_schema.jl`).
- `observations/observation_requests.jl` defines point events (`SoundingRequest`),
  station series (`SiteRequest`), site schedules (`EveryWindow`, `TimeRange`,
  `TimeList`), and the per-run `ObservationSet`.
- `observations/observation_readers.jl` turns sources into requests:
  `expand_observation_paths` (date tokens + file-name wildcards),
  `read_observation_requests` per source type and mode (OCO-2 Lite and MIP 10-s
  files, ObsPack records or stations, CSV/TOML/NetCDF tables with schedules),
  and `build_observation_set` (time lists to point events, transported-span
  filter, sites merged by id, skip counts).
- `observations/observation_output_spec.jl` parses `[output.observations]` into
  `ObservationOutputSpec` (or `NoObservationOutput`), rejecting unknown keys, and
  resolves `_soundings` / `_sites` file paths with the snapshot day template.
- `observations/cell_locator.jl` maps an observation (lon, lat) to its containing
  cell per topology (`cell_locator`, `locate` → `CellLocation` with native cell
  index, halo-aware device column, centre and area). Lat-lon and reduced
  Gaussian use the face arrays; the cubed sphere uses `lonlat_to_panel_xy`.
- `observations/observation_gather.jl` is the only device work of observation
  sampling: a pure gather kernel (`gather_columns!`, `gather_field!`) that copies
  air mass and selected tracer storage for a batch of columns into compact
  `(level, observation)` buffers, plus the host Float64 reconstruction of
  interface pressures, hypsometric layer heights (`AbstractLayerTemperature`
  sources), intake layer index, mixing-ratio profiles, and column means.
- `observations/observation_netcdf.jl` holds the append-only point-event and
  station NetCDF sinks (`SoundingNetCDFStream`, `SiteNetCDFStream`). Writes are
  queued and flushed whenever the shared NetCDF lock (`with_netcdf_lock`) is
  free, so the run never waits for a background daily snapshot write; every
  flush is synced before its `completed_*` attribute, and `close` drains.
- `observations/observation_sampler.jl` is the runtime sampler: at every
  met-window end it gathers the bracketing point events plus all sites in one
  launch, blends masses linearly between the two window ends (or takes the
  nearest end), reconstructs pressures, heights, intake layers, and mixing
  ratios on the host, and appends to the streams; daily files rotate without
  blocking. `NoObservationSampler` is the default that keeps runs without
  observation output unchanged.

All runtime NetCDF writes, including `write_snapshot_netcdf` and the snapshot
stream, take the shared lock because netcdf-c is not thread-safe and daily
snapshot files are written on a background task.

## Topology Contract

- LL writes CF lon/lat coordinates, bounds, cell areas, full per-level fields, and column diagnostics.
- RG writes authoritative native `cell` variables, quadrilateral cell bounds,
  plus a diagnostic lon/lat raster for quick plots.
- CS writes native `(Xdim, Ydim, nf, lev, time)` fields with `lons`, `lats`, corners, cell area, and a `cubed_sphere` mapping variable.

Every selected tracer also writes `<tracer>_total_mass(time)` as Float64. The
value is captured before spatial output conversion and is the authoritative
global sum of model storage. ATMSNAP carries it in the JSON header so its
Float32 spatial payload does not erase small signed residuals.

To add a topology, implement schema and payload methods for the new mesh type.
Do not special-case the runner.

Single-file NetCDF runs retain only the current frame, file handle, and schema
metadata. Daily output permits one owned background write. `capture_snapshot`
without `fields` still returns full native storage for existing callers and
ATMSNAP output. All selected tracers keep their independent Float64 totals,
even when no layer or column diagnostic is requested.
