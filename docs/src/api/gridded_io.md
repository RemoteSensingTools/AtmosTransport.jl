# GriddedIO API

`AtmosTransport.GriddedIO` owns the conventions of native gridded input files
(surface-flux inventories, land fractions, initial-condition fields). Each rule
is a type the caller names, and the methods dispatch on it:

- **Coordinates.** [`find_coordinate`](@ref AtmosTransport.GriddedIO.find_coordinate)
  finds a coordinate by variable name only (never by `units` or `long_name`,
  which real files mislabel). [`latlon_orientation`](@ref AtmosTransport.GriddedIO.latlon_orientation),
  [`orient`](@ref AtmosTransport.GriddedIO.orient) and
  [`orient_coordinates`](@ref AtmosTransport.GriddedIO.orient_coordinates) put a
  field into ascending latitude and longitudes starting at the first value `≥ 0`.
- **Fill values.** [`read_raw`](@ref AtmosTransport.GriddedIO.read_raw) with
  [`MaskedAsZero`](@ref AtmosTransport.GriddedIO.MaskedAsZero),
  [`MaskedAsNaN`](@ref AtmosTransport.GriddedIO.MaskedAsNaN) or
  [`FillValueOnly`](@ref AtmosTransport.GriddedIO.FillValueOnly), straight into
  the caller's element type.
- **Source meshes.** [`source_latlon_mesh`](@ref AtmosTransport.GriddedIO.source_latlon_mesh)
  derives cell faces from a file's centres by a face rule.
  [`SnappedFaces`](@ref AtmosTransport.GriddedIO.SnappedFaces) (the default):
  the snapping rule for regular global axes, faces at the centres ± half a cell
  for longitudes centred on 0° or ±180°, midpoint faces with pole and
  truncation rules for pole-centred, truncated and Gaussian latitudes.
  [`UniformSpanFaces`](@ref AtmosTransport.GriddedIO.UniformSpanFaces): the
  uniform mesh the runtime surface-flux and initial-condition loaders used
  before they moved to `SnappedFaces`; the runtime no longer uses it.
  Bounds variables are never used.
- **Time axes.** [`decode_time`](@ref AtmosTransport.GriddedIO.decode_time)
  classifies an axis by its raw `units` before any CFTime call and decodes it
  with [`CFDecoder`](@ref AtmosTransport.GriddedIO.CFDecoder),
  [`MonthsSince`](@ref AtmosTransport.GriddedIO.MonthsSince),
  [`DecimalYears`](@ref AtmosTransport.GriddedIO.DecimalYears) or
  [`AuxiliaryTimeVariable`](@ref AtmosTransport.GriddedIO.AuxiliaryTimeVariable);
  [`MonthKey`](@ref AtmosTransport.GriddedIO.MonthKey),
  [`TimeInterval`](@ref AtmosTransport.GriddedIO.TimeInterval) and
  [`seconds_since`](@ref AtmosTransport.GriddedIO.seconds_since) carry the result.
  Origins are validated, never normalised. `decode_time` returns proleptic
  Gregorian `DateTime`s, so a `360_day` 30 February is an error there;
  [`decode_month_keys`](@ref AtmosTransport.GriddedIO.decode_month_keys) gives
  the months of any axis in its own calendar.
- **Staging.** [`stage_input`](@ref AtmosTransport.GriddedIO.stage_input)
  decompresses `.nc.gz` and single-member `.nc.tar.gz` inputs into a cache.
- **Flux units.** [`parse_flux_units`](@ref AtmosTransport.GriddedIO.parse_flux_units)
  reads every units string in use into [`FluxUnits`](@ref AtmosTransport.GriddedIO.FluxUnits);
  [`mass_rate_factor`](@ref AtmosTransport.GriddedIO.mass_rate_factor) converts
  to kg of a species per second with the molar masses of `Parameters`.
- **`cs_native` flux files** (the runtime contract of
  `surface_flux.kind = "cs_native"`, defined once): layout
  [`CS_NATIVE_FLUX_DIMS`](@ref AtmosTransport.GriddedIO.CS_NATIVE_FLUX_DIMS);
  the writer [`open_cs_native_writer`](@ref AtmosTransport.GriddedIO.open_cs_native_writer),
  [`write_slices!`](@ref AtmosTransport.GriddedIO.write_slices!),
  [`publish!`](@ref AtmosTransport.GriddedIO.publish!) and
  [`abandon!`](@ref AtmosTransport.GriddedIO.abandon!) (start-stamped intervals
  with `time_bnds`, mesh geometry and cell areas, no fill values, atomic
  publication); the streaming reader
  [`cs_native_flux_info`](@ref AtmosTransport.GriddedIO.cs_native_flux_info) and
  [`read_cs_native_flux!`](@ref AtmosTransport.GriddedIO.read_cs_native_flux!),
  the runtime's loader, panel by panel into the caller's arrays; the whole-file
  reader [`read_cs_native_flux`](@ref AtmosTransport.GriddedIO.read_cs_native_flux);
  the cell-area gate [`check_cell_area`](@ref AtmosTransport.GriddedIO.check_cell_area).
- **Lat-lon flux files** for `kind = "file"`:
  [`write_latlon_flux`](@ref AtmosTransport.GriddedIO.write_latlon_flux), static
  fields only, on meshes whose faces the runtime rebuilds exactly from the
  written centres with `SnappedFaces`.
- **Observation ground fields** (`[output.observations]` vertical placement):
  the per-grid model surface height,
  [`write_model_surface_height`](@ref AtmosTransport.GriddedIO.write_model_surface_height)
  and [`read_model_surface_height`](@ref AtmosTransport.GriddedIO.read_model_surface_height)
  (grid-checked, into `Output.ModelSurfaceHeight`), and a lat-lon DEM for
  point lookups, [`read_point_elevation`](@ref AtmosTransport.GriddedIO.read_point_elevation)
  → [`LatLonPointElevation`](@ref AtmosTransport.GriddedIO.LatLonPointElevation).

The rules were measured on the TRENDY v14, GFED5, FLUXCOM-X, CAMS, GridFED and
EDGAR files; `src/GriddedIO/README.md` lists them with their invariants.

```@autodocs
Modules = [AtmosTransport.GriddedIO]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```
