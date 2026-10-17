# Observation Sampling

Samples model columns at met-window ends for point events (satellite soundings,
flask records) and fixed stations. Writes append-only `<path>_soundings.nc` and
`<path>_sites.nc` files.

These files are included into the `Output` module by [`../Output.jl`](../Output.jl)
in this order: sources, requests, readers, cell locator, placement, output
spec, gather, NetCDF, sampler. The runner glue lives in
[`../../Models/runner/observations.jl`](../../Models/runner/observations.jl). See
also the parent overview [`../README.md`](../README.md) and the file schema in
[`../../../docs/src/config/output_schema.md`](../../../docs/src/config/output_schema.md).
The inversion observation path under [`../../Inversion/`](../../Inversion/) is a
separate system.

## Run Sequence

1. `observation_output_spec(output_cfg)` builds the spec.
2. `build_observation_sampler(spec, state, grid; origin, window_seconds, halo_width,
   terrain, has_humidity)` reads the sources into an `ObservationSet`
   (`build_observation_set`), locates every request, and places it in the
   column (`resolve_placement`, with the static ground fields of `terrain`, an
   `ObservationTerrain` the runner reads through `GriddedIO`). Requests that
   cannot be placed fail the build or are skipped and counted
   (`unplaced_records`). `check_pressure_basis` then checks that the pressure
   basis can serve the pressure-placed requests (`has_humidity`: the binaries
   carry specific humidity).
3. Per input binary, `begin_observation_day!` opens or rotates the files.
4. At t = 0 and after every window, `observe_window_boundary!` samples the state
   (with the window end's specific humidity when pressure placement needs it).
5. At the end of the run, `finish_observations!` then `close`.

## Entry Points

- Config: `observation_output_spec(output_cfg; partition)` returns
  `NoObservationOutput` or `ObservationOutputSpec`, and `observation_output_path`
  resolves file paths ([`observation_output_spec.jl`](observation_output_spec.jl)).
  `observation_source_from_cfg` builds one source table entry.
  `_parse_vertical_placement(cfg, label)` parses the placement keys
  (`_VERTICAL_PLACEMENT_KEYS`) of any config table into a
  `VerticalPlacementSpec`.
- Sources, in [`observation_sources.jl`](observation_sources.jl): kinds
  `OCO2LiteSource`, `ObsPackSource`, `TableSource`; modes `SoundingMode`,
  `SiteMode`; groupings `SiteCodeGrouping`, `LocationGrouping`; table formats
  (`AutoTableFormat`, `CSVTableFormat`, `TOMLTableFormat`, `NetCDFTableFormat`);
  quality filters `QualityFlagFilter`, `QualityFlagValues`, `NoQualityFilter`
- Requests, in [`observation_requests.jl`](observation_requests.jl):
  `SoundingRequest`, `SiteRequest`, the schedules `EveryWindow`, `TimeRange`,
  `TimeList`, and `ObservationSet`
- Readers, in [`observation_readers.jl`](observation_readers.jl):
  `expand_observation_paths`, `read_observation_requests`, `build_observation_set`
- Cell locator, in [`cell_locator.jl`](cell_locator.jl):
  `cell_locator(mesh; halo_width)`, `locate(loc, lon, lat)` (returns
  `CellLocation` or `nothing`), `ncolumns`, `isvalid_lonlat`
- Vertical placement, in [`observation_placement.jl`](observation_placement.jl):
  the ground interfaces `AbstractModelSurfaceHeight` (`NoModelSurfaceHeight`,
  `ModelSurfaceHeight`) and `AbstractPointElevation` (`NoPointElevation`; the
  lat-lon DEM lives in `GriddedIO`), `ObservationTerrain`,
  `VerticalPlacementSpec` with `ErrorOnUnplaced`/`SkipUnplaced`,
  `AboveGroundPlacement`/`AbsoluteAltitudeAbove`, the ordered coordinates
  `AbstractVerticalCoordinate` (`HeightAboveGroundCoordinate`,
  `PressureCoordinate`) and the pressure basis `AbstractPressureBasis`
  (`TotalFromHumidity`, `DryInterfaces`), the source trait
  `missing_height_rule` (`LowestLayerWithoutHeight`, `UnplacedWithoutHeight`),
  `resolve_placement(spec, rule, request, cell, terrain)` → `VerticalPlacement`
  (`VERTICAL_PLACEMENT_CODES`), `placement_layer` / `pressure_layer_index`,
  and `placement_pressures!` / `check_pressure_basis`
- Gather and host reconstruction, in [`observation_gather.jl`](observation_gather.jl):
  `ObservationGatherBuffers`, `gather_columns!`, `gather_field!`,
  `interface_pressures!`, `total_interface_pressures!`, `layer_heights_agl!`,
  `intake_layer_index`,
  `mixing_ratio_profile!`, `column_mean_vmr`, and the layer-temperature types
  `ConstantLayerTemperature`, `SurfaceLapseTemperature`, `ProfileLayerTemperature`
- NetCDF, in [`observation_netcdf.jl`](observation_netcdf.jl):
  `SoundingNetCDFStream` with `append_soundings!`, `SiteNetCDFStream` with
  `append_site_record!`, `write_summary_attributes!`,
  `check_observation_tracer_names`
- Sampler, in [`observation_sampler.jl`](observation_sampler.jl):
  `build_observation_sampler`, `begin_observation_day!`,
  `observe_window_boundary!`, `finish_observations!`, `samples_observations`,
  `NoObservationSampler`

## File Map

- [`observation_sources.jl`](observation_sources.jl) — source descriptors, the choice tables (`_SOURCE_TYPES`, `_OBSERVATION_MODES`, `_SITE_GROUPINGS`, `_TABLE_FORMATS`, `_QUALITY_FILTERS`), source-table parsing
- [`observation_requests.jl`](observation_requests.jl) — request and schedule types, `ObservationSet`, time-list expansion into point events
- [`observation_readers.jl`](observation_readers.jl) — path templates and wildcards, CF time decoding, OCO-2 Lite / ObsPack / CSV-TOML-NetCDF readers, site merging, `build_observation_set`
- [`observation_output_spec.jl`](observation_output_spec.jl) — `[output.observations]` keys, the shared vertical-placement parser, strict ISO-8601 UTC parsing, time-interpolation types, output paths
- [`cell_locator.jl`](cell_locator.jl) — `CellLocation` and per-topology containing-cell lookup
- [`observation_placement.jl`](observation_placement.jl) — vertical placement: ground interfaces, placement settings, vertical-coordinate and pressure-basis types, the rules (first configured coordinate: height above ground = record height → altitude − DEM, or pressure; then the missing-height rule), codes, layer choice, pressure-basis interfaces
- [`observation_gather.jl`](observation_gather.jl) — device gather kernel and host twin, pressure (air-mass and total) and height reconstruction, height-method and `interp_flag` codes
- [`observation_netcdf.jl`](observation_netcdf.jl) — queued, lock-aware sounding and site NetCDF streams and their schemas
- [`observation_sampler.jl`](observation_sampler.jl) — `ObservationSampler`, linear or nearest-window time handling, daily file rotation, counters

## Common Tasks

- To add a source kind:
  1. Subtype `AbstractObservationSource` and register it in `_SOURCE_TYPES`.
  2. Implement `source_keys`, `_parse_source`, `source_kind`, `source_mode`, and
     `_read_requests_file!(soundings, sites, source, path, source_index, origin,
     stats, window)`.
  3. Update [`../../../schemas/atmos_transport_run.schema.json`](../../../schemas/atmos_transport_run.schema.json).
     `test_observation_schema.jl` fails until the schema lists each choice.
- To add a mode, grouping, table format, quality filter, or time interpolation,
  extend its NamedTuple choice table and the schema in the same way.
- To add an `[output.observations]` key, add it to `_OBSERVATION_OUTPUT_KEYS`
  (placement keys to `_VERTICAL_PLACEMENT_KEYS`), add a field to
  `ObservationOutputSpec` (or `VerticalPlacementSpec` for placement keys), parse
  it in `observation_output_spec` (or `_parse_vertical_placement`), and add it
  to the schema.
- To add a vertical coordinate, subtype `AbstractVerticalCoordinate`, register
  it in `_VERTICAL_COORDINATES`, give it a `vertical_coordinate_label`, a
  `_place_by` method (the placement, or `nothing`), and a `_coordinate_text`
  for the unplaced message, and list it in the schema.
- A new source kind also needs a `missing_height_rule` method: whether its
  records without any vertical coordinate are column/surface products
  (`LowestLayerWithoutHeight`) or cannot be placed (`UnplacedWithoutHeight`).
- To add an output variable:
  1. Define it in `_define_sounding_schema!` or `_define_site_schema!`.
  2. Write it in `_write_soundings!` or `_write_site_record!`.
  3. Add its name to `_OBSERVATION_FIXED_NAMES`, or the suffix to
     `_OBSERVATION_TRACER_SUFFIXES`, so tracer-name collisions are caught.
- To support a new topology, add a locator with `cell_locator`, `locate`, and
  `ncolumns`, a `gather_columns!` method for its state type, and `_npanel`.

## Invariants

- State is sampled only at t = 0 and at met-window ends:
  - `linear` (the default) blends the air and tracer masses of the two bracketing
    ends and then divides.
  - `nearest_window` takes the end within half a window.
  - `interp_flag` records how each row was made: 0 bracketed, 1 one-sided,
    2 nearest window.
- Point events before the first sampled boundary (normally t = 0) are dropped.
  So are events after the last window end, except those exactly at it. Both are
  counted in the summary attributes.
- Sampling reads the containing cell, with no horizontal interpolation:
  - `locate` returns `nothing` only outside a regional lat-lon mesh.
  - Invalid coordinates throw.
  - LL/RG locators require `halo_width = 0`. On the cubed sphere, `column`
    includes the halo offset.
- The gather is the only device work. It copies values without arithmetic. All
  derived quantities are Float64 on the host.
- Pressure reconstruction needs `B[1] == 0`:
  - `p_half[1] = A[1]` and `p_half[k+1] = p_half[k] + g * m[k] / area`.
  - On `DryBasis` these are dry pressures, and the variables are named `ps_dry`,
    `p_half_dry`, and `air_mass_per_area_dry`. Heights and the written
    pressures always use them.
  - Pressure placement compares the record's total pressure with the
    interfaces of `pressure_basis`: by default (`TotalFromHumidity`) the total
    pressures `p_half[k+1] = p_half[k] + g * m[k] / (area * (1 - q[k]))` from
    the binary's specific humidity on `DryBasis` (the moist air mass's
    interfaces on `MoistBasis`); `DryInterfaces` (`"dry"`) uses the dry ones.
  - On `DryBasis` with pressure-placed requests, `TotalFromHumidity` needs
    humidity: the build fails without it (`has_humidity = false`), and every
    window end must pass it. The runner passes `qv_start` at the run start and
    `qv_end` at window ends (lat-lon, reduced Gaussian), else the window's GCHP
    VDIFF `qv` (cubed sphere). Linear interpolation blends it like the masses.
    Only pressure-placed rows read it, so the other rows do not change with
    the basis.
- Mixing ratio is tracer storage divided by air mass, and NaN where air mass ≤ 0.
  The column mean is Σ tracer / Σ air.
- Layer `k = 1` is the top:
  - Heights are hypsometric with `R_DRY_AIR`.
  - The layer temperature is the runner's per-layer field when one is passed
    (CS GCHP VDIFF `t`). Otherwise it is `layer_height_temperature_kelvin`
    (default 280 K).
  - A non-positive height selects the lowest layer; a NaN height is an error.
- Vertical placement (issues #56 and #20) is resolved once per request at
  build time:
  - A record is placed by the first of `vertical_coordinates` it gives: its
    height above the true ground (`intake_height`, `altitude − elevation`,
    else `altitude −` DEM ground), measured up from the model surface; or its
    pressure. The default order is height, then pressure.
  - A record that gives none of the configured coordinates is unplaced if it
    states any vertical coordinate; one without any follows its source's
    `missing_height_rule`.
  - `absolute_altitude_above_agl_m` re-places height placements aloft by
    pressure only when `pressure` is a coordinate.
  - The model surface height is never used as the true ground; it is only the
    reference of `absolute_altitude_above_agl_m`.
  - A request is never sampled in the lowest layer for lack of a height unless
    its source declares `LowestLayerWithoutHeight`; the row then records
    `vertical_placement = lowest_layer`.
  - The sampler keeps only the terrain provenance, not the DEM.
  - Undeclared fill values never place a record: ObsPack `pressure` ≤ 0
    (`valid_pressure`), `elevation` / `site_elevation` below −450 m
    (`valid_elevation`) and negative `intake_height` (`valid_intake_height`)
    become NaN at read time with a warning per file.
  - A placement more than `BELOW_GROUND_TOLERANCE_M` (100 m) below the ground
    (`below_ground`) is sampled in the lowest layer and reported in a warning.
  - ObsPack station series key altitude-only rows by altitude (0.1 m); a
    `site_code` dataset whose records stray more than 1° from the site
    position is warned about as a moving platform.
- Each NetCDF write is queued and runs under `with_netcdf_lock`. If the lock is
  busy, the write stays queued for a later call. Each flush is synced before
  `completed_soundings` / `completed_times` is updated. A failed write poisons the
  stream, and `close` drains the queue and is idempotent.
- A file is created only for a mode that has requests. No path may be opened
  twice in one run.
- Site ids are unique:
  - A repeated id must keep its location. Time lists are merged; any other
    schedules must be equal.
  - `TimeList` sites become point events in the `_soundings` file.
- Config times must be strict ISO-8601 UTC (`_ISO_UTC_RE`).

## Related Tests

Under [`../../../test/core/`](../../../test/core/): `test_observation_output_spec.jl`,
`test_observation_sources.jl`, `test_observation_schedules.jl`,
`test_observation_schema.jl`, `test_observation_cell_locator.jl`,
`test_observation_gather.jl`, `test_observation_placement.jl`,
`test_gridded_io_surface_height.jl`, `test_observation_sampler.jl`,
`test_observation_runner.jl`, `test_async_output_lifetime.jl` (sampler close).
