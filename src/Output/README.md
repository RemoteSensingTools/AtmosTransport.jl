# Output

`Output` owns the runtime snapshot contract and NetCDF file schema.

Runners should only decide when to sample state. They should call
`capture_snapshot(model; time_hours, halo_width, fields)` and
`write_snapshot_netcdf(path, frames, grid; mass_basis, options, time_origin)` instead of
writing topology-specific NetCDF files directly.

## Files

- `Output.jl` assembles the module (imports, include order, exports).
- `runtime_output.jl` parses `[output]` into a `RuntimeOutputSpec`: the
  schedule, partition, format and layer-selection types, the `[output]` key
  tables (`OUTPUT_KEYS`, `SNAPSHOT_SCHEDULE_KEYS`, ...; alias keys in
  precedence order) with their choice tables, `check_output_targets`, and the
  resolved `physics_*`/time-origin metadata carried to every sink. Scalars and
  choices are read with the ConfigChecks helpers (`config_bool`, `config_int`,
  `config_float`, `parse_choice`).
- `binary_reader.jl` defines `read_snapshot_binary` for the conversion script.
- `binary_writer.jl` writes the ATMSNAP snapshot binary (header with
  the mesh definition and radius, Float32 payload).
- `snapshots.jl` defines `SnapshotFrame`, `SnapshotWriteOptions`, and
  model-state capture.
- `selected_snapshots.jl` captures requested layers and backend column reductions.
- `snapshot_totals.jl` records compensated Float64 signed tracer totals through
  `Architectures.compensated_total`: in place on CPU and CUDA, in bounded host
  slabs on Metal.
- `netcdf_stream.jl` appends single-file runtime output and records completed
  snapshots. Its owner must close the stream on every exit.
- `diagnostics.jl` derives VMR, column means, and mass-per-area fields.
- `netcdf_schema.jl` defines topology-specific dimensions, coordinates, and metadata.
  Geometry and provenance are public so that other NetCDF writers (generated
  `cs_native` input files) share them instead of copying them:
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

Observation sources, retrievals and sampling live in
[`../Observations/`](../Observations/README.md).

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

Explicit snapshot schedules sort a copy of the supplied hours, including
`Vector{Float64}` inputs, and reject non-finite hours in the constructor.

Geometry convention tags come from Grids and mass-basis tags from State.
Observation placement uses `AbstractPlacementCoordinate`; Grids owns
`AbstractVerticalCoordinate` for model vertical grids.

## Depends on

- Grids: meshes, cell areas, CS geometry and definition/convention tags.
- State: `basis_symbol`, `mass_basis`, tracer access.
- Parameters: `EARTH_RADIUS` (ATMSNAP files without a recorded radius).
- Architectures: `compensated_total`, `CPU` and the shared backend trait
  `_total_accumulator_type` (column sums on Metal).
- ConfigChecks: `config_bool`, `config_int`, `config_float`, `parse_choice`,
  `parse_float_type`.
- Package root: `expand_data_path`, `source_revision`.
- External: NCDatasets, HDF5_jll (error-handler silencing in
  `_create_netcdf_dataset`), JSON3 and Mmap (ATMSNAP), KernelAbstractions
  (column-sum kernel).

## Used by

- Models.DrivenRunner: capture, `NetCDFSnapshotStream`/`append_snapshot!`,
  batch writers, `runtime_output_spec` and its accessors, the key tables and
  `check_output_targets`.
- Observations.Sampling: the lock, path, partition and NetCDF primitives below.
- GriddedIO: `define_cs_horizontal_geometry!`, `define_provenance_attributes!`.
- Diagnostics and Extras/Analysis: `runtime_output_spec`, output paths,
  `with_netcdf_lock`.
- Scripts: `read_snapshot_binary`/`write_snapshot_netcdf`
  (`scripts/postprocess/binary_to_netcdf.jl`), `column_mean_mixing_ratio`.

## Shared helpers owned here

`with_netcdf_lock`, `define_cs_horizontal_geometry!`,
`define_provenance_attributes!`, `define_product_attributes!`, `check_output_targets` and the exported
output-key tables. Observations shares the existing lock, path and NetCDF
schema primitives so lock nesting and written metadata stay unchanged.
Within Output, `_column_ratio` is the one column-mean quotient (public
`column_mean_mixing_ratio` and the writers), `_time_units`/`_TIME_CALENDAR`
the one time-unit spelling of NetCDF and ATMSNAP, and `_physics_attribute_key`
the one `physics_` key check of every sink.

# shared: _NETCDF_IO_LOCK _parse_tracer_names _output_partition _substitute_day_template _insert_suffix_before_extension _has_day_token _ensure_parent_dir _create_netcdf_dataset _tracer_units _payload_fill_value _grid_type_string

## Forward<->adjoint mirrors

None. Snapshot reductions preserve their existing accumulation order.

Selected CS snapshot interiors use State.panel_interior, cropped to the
requested mesh extent after the existing shape/halo checks.
