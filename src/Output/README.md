# Output

`Output` owns the runtime snapshot contract and NetCDF file schema.

Runners should only decide when to sample state. They should call
`capture_snapshot(model; time_hours, halo_width, fields)` and
`write_snapshot_netcdf(path, frames, grid; mass_basis, options, time_origin)` instead of
writing topology-specific NetCDF files directly.

## Files

- `Output.jl` assembles the module (imports, include order, exports).
- `runtime_output.jl` defines the `[output]` schedule, partition and
  layer-selection types, runtime output specification, and the resolved
  `physics_*`/time-origin metadata carried to every sink.
- `binary_writer.jl` writes the ATMSNAP snapshot binary (header with
  the mesh definition and radius, Float32 payload).
- `snapshots.jl` defines `SnapshotFrame`, `SnapshotWriteOptions`, and
  model-state capture.
- `selected_snapshots.jl` captures requested layers and backend column reductions.
- `snapshot_totals.jl` records compensated Float64 signed tracer totals through
  `Architectures._compensated_total`: in place on CPU and CUDA, in bounded host
  slabs on Metal.
- `netcdf_stream.jl` appends single-file runtime output and records completed
  snapshots. Its owner must close the stream on every exit.
- `diagnostics.jl` derives VMR, column means, and mass-per-area fields.
- `netcdf_schema.jl` defines topology-specific dimensions, coordinates, and metadata.
  Two pieces are public so that other NetCDF writers (generated `cs_native`
  input files) share them instead of copying them:
  `define_cs_horizontal_geometry!(ds, mesh)` writes the cubed-sphere horizontal
  block (`Xdim`/`Ydim`/`Xcorner`/`Ycorner`/`nf`, `lons`, `lats`, corners,
  `cell_area`, the `cubed_sphere` mapping with `semi_major_axis = mesh.radius`,
  and the matching global definition attributes) without any level or time
  axis, and `define_provenance_attributes!(ds)` writes the automated provenance
  attributes (`creation_date`, `framework_commit`, `framework_dirty`, runtime,
  host, user) and returns `(; creation_date, git_commit, git_dirty)`. The
  snapshot writer builds its CS layout from the former and then adds `lev` and
  `time`; snapshot and observation files take their provenance from the latter.
- `netcdf_writer.jl` writes topology-specific payload variables through one public API.
- `observations/observation_sources.jl` defines the typed `[[output.observations.sources]]`
  descriptors (`OCO2LiteSource`, `ObsPackSource`, `TableSource`), the singleton
  mode / grouping / table-format types and quality filters they dispatch on, and
  their validation. The `_SOURCE_TYPES`, `_OBSERVATION_MODES`, `_SITE_GROUPINGS`
  and `_TABLE_FORMATS` tables are the parser's choices; the editor schema must
  match them (`test/core/test_observation_schema.jl`).
- `observations/observation_requests.jl` defines point events (`SoundingRequest`),
  station series (`SiteRequest`), site schedules (`EveryWindow`, `TimeRange`,
  `TimeList`), and the per-run `ObservationSet`. Requests carry what the
  record says about its height (`intake_height_m` above ground, `altitude_m`,
  `elevation_m`, `pressure_pa`; NaN when absent).
- `observations/observation_readers.jl` turns sources into requests:
  `expand_observation_paths` (date tokens + file-name wildcards),
  `read_observation_requests` per source type and mode (OCO-2 Lite and MIP 10-s
  files, ObsPack records or stations, CSV/TOML/NetCDF tables with schedules),
  and `build_observation_set` (time lists to point events, transported-span
  filter, sites merged by id, skip counts). The height above ground of a row
  is resolved row by row (`record_height_above_ground`: `intake_height`, else
  `altitude − elevation`); an altitude without elevation is kept for
  placement.
- `observations/observation_output_spec.jl` parses `[output.observations]` into
  `ObservationOutputSpec` (or `NoObservationOutput`), rejecting unknown keys, and
  resolves `_soundings` / `_sites` file paths with the snapshot day template.
  The placement keys (`_VERTICAL_PLACEMENT_KEYS`) go through the shared
  `_parse_vertical_placement(cfg, label)`, which any table that places records
  in the column can reuse.
- `observations/observation_placement.jl` places requests in the column: the
  static ground interfaces `AbstractModelSurfaceHeight` (`ModelSurfaceHeight`,
  per-cell model surface height) and `AbstractPointElevation` (DEM; the
  concrete lat-lon type and both readers live in `GriddedIO`), bundled as
  `ObservationTerrain`; the `[output.observations]` placement settings
  (`VerticalPlacementSpec`, `ErrorOnUnplaced`/`SkipUnplaced`,
  `AboveGroundPlacement`/`AbsoluteAltitudeAbove`, the ordered
  `HeightAboveGroundCoordinate`/`PressureCoordinate` of `vertical_coordinates`,
  and the pressure basis `TotalFromHumidity`/`DryInterfaces`); the per-source
  rule for records without any vertical coordinate (`missing_height_rule`:
  `LowestLayerWithoutHeight` for OCO-2 and tables, `UnplacedWithoutHeight` for
  ObsPack); `resolve_placement` (the first configured coordinate that places
  the record: height above ground = record height, else altitude − DEM; or
  pressure; else the missing-height rule; `nothing` when a record cannot be
  placed; the model surface height is never a stand-in for the true ground)
  into a `VerticalPlacement` with a `VERTICAL_PLACEMENT_CODES` code;
  `placement_layer` / `pressure_layer_index`, the layer of a placement in one
  sampled column; and `placement_pressures!` / `check_pressure_basis`, the
  total (from the specific humidity) or dry interfaces pressure placement uses.
- `observations/cell_locator.jl` maps an observation (lon, lat) to its containing
  cell per topology (`cell_locator`, `locate` → `CellLocation` with native cell
  index, halo-aware device column, centre and area). Lat-lon and reduced
  Gaussian use the face arrays; the cubed sphere uses `lonlat_to_panel_xy`.
- `observations/observation_gather.jl` is the only device work of observation
  sampling: a pure gather kernel (`gather_columns!`, `gather_field!`) that copies
  air mass and selected tracer storage for a batch of columns into compact
  `(level, observation)` buffers, plus the host Float64 reconstruction of
  interface pressures (air-mass interfaces, and total interfaces from the dry
  air mass and specific humidity), hypsometric layer heights (`AbstractLayerTemperature`
  sources), intake layer index (a `NaN` height is an error, never the lowest
  layer), mixing-ratio profiles, and column means.
- `observations/retrieval_column.jl` is the averaging-kernel (retrieval)
  column operator, as pure host Float64 functions of one column:
  `RetrievalColumn{M}` (a retrieval's pressures from the surface up, pressure
  weights summing to 1, column averaging kernel, prior in mol/mol, optional
  model-profile uncertainty `profile_sigma`, and the mapping `M`; validated),
  `stretch_model_interfaces!` (the model column mapped linearly onto the
  retrieval's surface and top), the projection ``P`` of the mapping
  (`LayerRebin`: TM5-4DVAR's mass-conserving overlap rebin R,
  `retrieval_rebin_matrix`; `HatLevelProjection`: the air-mass-weighted
  projection onto the hat functions of levels, OCO-2/OCO-3;
  `retrieval_projection_matrix`), `retrieval_column_weights!`
  (``H_k = Σᵢ hᵢ aᵢ Pᵢₖ``), `retrieval_prior_offset`
  (``Σᵢ hᵢ(1 − aᵢ) x_a,ᵢ``), `retrieval_profile!` (``P x``),
  `retrieval_column_value`, `retrieval_model_error_variance`
  (``Σᵢ (hᵢ aᵢ σᵢ)²``), `tm5_layer_column` (TM5's 19 layers of a level
  retrieval), plus `mole_fraction_unit_scale` for files that state ppm or
  ppb. Model interfaces are top first (total pressures, e.g. from
  `placement_pressures!`); the hat projection also takes the layers' air
  masses. The 4D-Var's `CSRetrievalColumnObjective` and the forward sampler
  (OCO-2 sources with `averaging_kernel_tracers`, whose reader then reads the
  kernel fields with `read_oco2_retrieval_columns`) both use it.
- `observations/observation_netcdf.jl` holds the append-only point-event and
  station NetCDF sinks (`SoundingNetCDFStream`, `SiteNetCDFStream`). Writes are
  queued and flushed whenever the shared NetCDF lock (`with_netcdf_lock`) is
  free, so the run never waits for a background daily snapshot write; every
  flush is synced before its `completed_*` attribute, and `close` drains.
- `observations/observation_sampler.jl` is the runtime sampler. When it is
  built it locates and places every request (requests that cannot be placed
  fail the build or are skipped and counted, per `unplaced_records`) and keeps
  only the terrain's provenance, not the DEM; at every
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
- CS writes native `(Xdim, Ydim, nf, lev, time)` fields with `lons`, `lats`,
  corners, cell area, and a `cubed_sphere` mapping variable, all from
  `define_cs_horizontal_geometry!`.

Every selected tracer also writes `<tracer>_total_mass(time)` as Float64. The
value is captured before spatial output conversion and is the authoritative
global sum of model storage. ATMSNAP carries it in the JSON header so its
Float32 spatial payload does not erase small signed residuals.

Runtime NetCDF time coordinates use CF `hours since <run start>` with the
`proleptic_gregorian` calendar. ATMSNAP records the same units and calendar in
its JSON header. Both formats also record the typed runtime physics block:
advection and limiter, vertical remap, convection, diffusion, surface sources,
chemistry/decay, mass basis, met source, and transport-binary provenance.

To add a topology, implement schema and payload methods for the new mesh type.
Do not special-case the runner.

Single-file NetCDF runs retain only the current frame, file handle, and schema
metadata. Daily output permits one owned background write. `capture_snapshot`
without `fields` still returns full native storage for existing callers and
ATMSNAP output. All selected tracers keep their independent Float64 totals,
even when no layer or column diagnostic is requested.
