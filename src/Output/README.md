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
  mode / grouping / table-format types they dispatch on, and their validation.
- `observations/observation_output_spec.jl` parses `[output.observations]` into
  `ObservationOutputSpec` (or `NoObservationOutput`), rejecting unknown keys, and
  resolves `_soundings` / `_sites` file paths with the snapshot day template.
- `observations/observation_sampler.jl` defines the runtime sampler interface
  (`observe_window_boundary!`, `begin_observation_day!`, `finish_observations!`) and
  the `NoObservationSampler` default that keeps runs without observation output
  unchanged.

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
