# GriddedIO

The one owner of the conventions of native gridded input files: surface-flux
inventories, land fractions and initial-condition fields. It decides how a
file's coordinates are found, which values are fill, how a latitude-longitude
field is oriented, which cell faces its coordinates imply, how its time axis is
decoded, how a compressed input is staged, and how its flux units are read.
It also owns the schema of the surface-flux files the model writes and the
runtime reads (`cs_native` on the cubed sphere, lat-lon for `kind = "file"`),
with the one writer and the one reader of each.

Every rule was measured on the files the model reads (TRENDY v14 S3, GFED5,
FLUXCOM-X, CAMS/LMDZ, GridFED, EDGAR); the measurements are in
`docs/memos/2026-10-10_python_flux_port_p0_facts.md` and the design in
`docs/memos/2026-10-10_python_flux_port_design.md` (Section 2.4). The rules are
types: the caller names an axis, a fill policy, a time decoder or a substance,
and the methods dispatch on it.

`src/AtmosTransport.jl` loads this folder after `Regridding` and before
`Preprocessing`. It imports `Parameters` (`SPECIES_MOLAR_MASS`,
`CARBON_MOLAR_MASS`, `DAYS_PER_JULIAN_YEAR`), `Grids` (`LatLonMesh`,
`CubedSphereMesh`, `cell_area`), `Output` (`define_cs_horizontal_geometry!`,
`define_provenance_attributes!`) and `AtomicFiles`, plus NCDatasets (and its
CFTime), Dates and SHA. It never imports `Preprocessing` or `Models`, so the runtime loaders
(`Models.InitialConditionIO`) and the surface-flux builder can both use it.

## File Map

| File | Contents |
|---|---|
| [`GriddedIO.jl`](GriddedIO.jl) | Module, imports, exports, includes. |
| [`coordinates.jl`](coordinates.jl) | `AbstractAxis` (`LongitudeAxis`, `LatitudeAxis`, `TimeAxis`, `VerticalAxis`) and `find_coordinate`; `LatLonOrientation`, `latlon_orientation`, `orient`, `orient_coordinates`; `AbstractFillPolicy` (`MaskedAsZero`, `MaskedAsNaN`, `FillValueOnly`) and `read_raw([T,] ds, name, policy, idx...)`. |
| [`source_meshes.jl`](source_meshes.jl) | `source_latlon_mesh(lon, lat, rule; radius)` and `source_latlon_mesh(ds, rule; radius)` with the face rules `AbstractFaceRule` (`SnappedFaces`: the runtime rule, with snapping, centre-on-0/180, midpoint and pole rules on `LatLonMesh(λᶠ, φᶠ; radius)`; `UniformSpanFaces`: the legacy uniform-span reconstruction). |
| [`time_axes.jl`](time_axes.jl) | `TimeInterval`, `MonthKey`, `interval`, `month_keys`, `seconds_since`, `check_last_month`; `AbstractTimeDecoder` (`CFDecoder`, `MonthsSince`, `DecimalYears`, `AuxiliaryTimeVariable`) with the stamp positions `MonthStart`, `MidMonth`, `MonthEnd`; `DEFAULT_TIME_DECODERS`, `decode_time`, `decode_time_bounds`, `decode_slice_intervals` (the interval each slice of a file stands for), `known_slice_intervals`, `stamp_intervals` (the regular step of a series of stamps over several files) and `STAMP_SPACING_TOLERANCE`, `decode_month_keys`, and bounds/calendar-aware `decode_month_seconds`; the flux writers' time axis (contiguous intervals → `time`, `time_bnds`). |
| [`staging.jl`](staging.jl) | `stage_input(path; staging_dir)`: `.nc.gz` and single-member `.nc.tar.gz` decompressed through the system `gzip`/`tar`, cached, published atomically. |
| [`flux_units.jl`](flux_units.jl) | `FluxUnits{S,A,T}` with `AbstractSubstance` (`TargetSpecies`, `Carbon`, `Species`), `AbstractAreaBasis` (`PerSquareMetre`, `PerCell`), `AbstractTimeBasis` (`PerSecond`, `PerDay`, `PerMonth`, `PerYear`); `parse_flux_units`, `kg_per_unit`, `period_seconds`, `mass_rate_factor`, `canonical_units_string`. |
| [`cs_native_flux.jl`](cs_native_flux.jl) | The `cs_native` flux file, the runtime contract: `CS_NATIVE_FLUX_DIMS`; the writer `CSNativeFluxWriter` (`open_cs_native_writer`, `write_slices!`, `publish!`, `abandon!`); the streaming reader `cs_native_flux_info` / `read_cs_native_flux!` (the runtime's `kind = "cs_native"` loader, panel by panel into the caller's arrays); the whole-file reader `read_cs_native_flux` (the builder's readback); the cell-area gate `check_cell_area`. |
| [`latlon_flux.jl`](latlon_flux.jl) | `write_latlon_flux`: static CF lat-lon flux files for `kind = "file"`, on meshes the runtime rebuilds exactly from their coordinates; `check_latlon_cell_area`: the gate on a lat-lon file's `cell_area`/`area` (units m², within 1 % of the areas its coordinates define, norm-based; positive and finite wherever the flux is nonzero). |

## Invariants

### Coordinates

- Found **by variable name only** (`lon`, `longitude`, `x`; `lat`, `latitude`,
  `y`; `time`, `t`; `lev`, `level`, `plev`, `z`, `hybrid`, `nhym`, first match
  wins). Attributes are never consulted: FLUXCOM-X labels `lon` as
  `degrees_north`/`latitude`, GFED5 NRT coordinates have no attributes. The
  found variable must be 1-D and, when the caller passes `length`, that long.
- Orientation (`latlon_orientation`) is the runtime's: latitude ascending; when
  a longitude is negative, the axis is rolled to start at its first value
  `≥ 0` and the coordinates are taken modulo 360°. `orient` copies, never
  computes, so oriented fields are bitwise the input in the new order.

### Fill policies

| Policy | Fill marked by | Fill becomes |
|---|---|---|
| `MaskedAsZero` | `_FillValue`, `missing_value` | `0.0` (the runtime's `nomissing(…, 0)`, bit for bit) |
| `MaskedAsNaN` | `_FillValue`, `missing_value` | `NaN` (flux sources that count valid cells) |
| `FillValueOnly` | `_FillValue` only | `NaN` (IBIS land fraction: its `missing_value = 0` is a valid value) |

VISIT, VISIT-UT and iMAPLE mark fill only through `missing_value = -99999`, so
flux reads use a masking policy. A NaN `_FillValue` matches NaN values.
`read_raw(T, ds, name, policy, idx...)` converts each stored value once,
straight to `T` (Float64 by default), so a Float32 reader never holds a Float64
copy, and the values equal `T.(nomissing(ds[name][idx...], zero(T)))` bit for
bit.

### Source meshes

`source_latlon_mesh(lon, lat, rule; radius)` takes a face rule
(`AbstractFaceRule`): `SnappedFaces()` (the default, the rules below) or
`UniformSpanFaces()`, the legacy uniform-span reconstruction. Runtime surface
fluxes and file initial conditions use the default `SnappedFaces`. `UniformSpanFaces`
builds a uniform mesh whose spacing is the mean spacing of the centres, with
faces half a spacing outside the first and last centre and extents within
`1e-3` of a cell of ±90° or a 360° span snapped to them; its faces carry the
round-off of Float32-stored centres, and it does not bound pole-centred,
truncated or Gaussian rows. `SnappedFaces`:

- **Regular global axes (snapping rule).** `res = 180/n` or `360/n`, origin
  `-90`/`+90` (ascending/descending latitude), `-180` (a negative longitude) or
  `0`. Every centre must sit within `1e-6` cells (plus `2 eps(T) max|x| / res`
  for the storage precision `T`; GridFED's Float32 centres sit 3e-5 cells off)
  of its cell-centred position, else `ArgumentError` (half-cell offset, wrong
  resolution). Faces are the correctly rounded integer ratios
  `(x₀ n + s j 180) / n`; outer faces are exactly ±90 and ±180 (or 0/360).
- **Longitudes with a centre on 0° or ±180°** (CLASSIC, CLM, ELM, JSBACH, the
  FATES grids): faces at the centres ± `res/2`, first face not ±180 or 0. A
  wrapped axis (CLM-FATES) is unwrapped, so column `i` stays file column `i`.
- **Other latitudes** (pole-centred, truncated, Gaussian, regional): midpoints
  of adjacent centres; an outer face is the pole when the outer centre is
  closer to it than the adjacent spacing (CLM, ELM, iMAPLE, JSBACH), else the
  centre minus half the adjacent spacing (FATES south: -60.6316°). Such meshes
  have `Δφ = NaN`.
- **Regional longitudes:** uniform faces from the first centre minus to the last
  plus half the mean spacing.
- `lat_bnds`/`lon_bnds` are never read.
- On exactly stored regular grids and regional grids the faces equal those of
  `UniformSpanFaces` bit for bit. On Float32 centres `UniformSpanFaces` takes
  its faces from the centres' span and carries their round-off (GridFED 0.1°:
  west face 7.45e-10° instead of 0).

### Time axes

- `decode_time` reads the raw variable's `units` **before any CFTime call**
  and picks the first decoder of its list that accepts the class:
  `<day|hour|minute|second> since` → `CFDecoder`; `months since` →
  `MonthsSince` (never CFTime, which decodes it with a fixed 30.44-day month);
  no units, `years since …` or unparseable → `DecimalYears` /
  `AuxiliaryTimeVariable`, named per product (the default list,
  `DEFAULT_TIME_DECODERS = (CFDecoder(), MonthsSince())`, refuses such axes).
- `CFDecoder` on `standard`/`gregorian`/`proleptic_gregorian`/`julian`/no
  calendar is NCDatasets' own decoding: `seconds_since` of it equals the
  runtime's slice times bit for bit. On `noleap`/`365_day`/`all_leap`/
  `366_day`/`360_day` the calendar date and time of day are carried over to a
  `DateTime` (values rounded to the millisecond). Every `noleap`/`365_day`
  date exists in the proleptic Gregorian calendar of `DateTime`; an
  `all_leap`/`366_day` or `360_day` stamp on 29 February of a common year or
  on 30 February is an `ArgumentError` from `decode_time` that names the date.
  A trailing `" UTC"` on the origin is dropped.
- `decode_month_keys` gives the month of each stamp in the axis's own
  calendar (CFTime's year and month of the calendar date, never a `DateTime`),
  so such axes still have month keys (30 February of `360_day` is February):
  the monthly keys of the flux builders. Elsewhere it equals
  `month_keys(decode_time(...))`.
- Origins are validated, never normalised: a `Y-M[-D][ h:m[:s]]` origin with a
  month outside 1:12, a day outside 1:31 (for `CFDecoder`, beyond the month's
  length in the axis's calendar: `2001-02-29` in `standard`, `2000-02-30` in
  `proleptic_gregorian`) or an invalid time of day is an `ArgumentError`
  (CFTime would read `days since 2000-13-01` as 2001-01-01 in the standard
  calendar). A `months since` origin must have that form; other
  `<unit> since` origins go to CFTime, which parses or rejects them.
- `MonthsSince`: month = origin month + `⌊v + 1e-6⌋`; `DecimalYears`:
  round-to-nearest month with a residual `≤ 1e-6` month;
  `AuxiliaryTimeVariable`: stamps from another variable (CARDAMOM's `Time`,
  end-of-month days since 2003-01-01, `MonthEnd()`). All three return month
  starts; the stamp position is named per product, and a wrong one shifts the
  axis by a month, which `check_last_month` catches. Fill values in a time axis are an error. `decode_time` reads no
  time bounds, and the axis length is never used to guess its meaning.
- `decode_slice_intervals` gives the interval each slice of one file stands
  for: `known_slice_intervals` (the CF bounds, `decode_time_bounds`, or for a
  `months since` axis the month of each stamp) when the axis defines them,
  else `stamp_intervals` of its stamps. A series spread over several files
  calls `stamp_intervals` on the sorted stamps of all its files. There each
  slice holds for the series' regular step: for month starts, until the next
  stamp when that is one month later or one year later from 1 January to 1
  January, the last stamp one month (one year after a yearly step); for other
  stamps, until the next stamp when the spacing is the smallest spacing
  (within `STAMP_SPACING_TOLERANCE`, 1 s), the last stamp that step. A slice
  before a longer spacing holds for one step, so the gap stays uncovered. A
  single stamp (also a month start: one 1 January 00Z stamp may be a day, a
  month or a year), stamps about a month or a year apart that are not month
  starts (mid-month stamps), and stamps that do not increase strictly are an
  `ArgumentError`. The time-varying surface-flux loaders check their run
  against these intervals.

### Staging

- Uncompressed paths are returned unchanged. `.gz`, `.tar.gz`/`.tgz` (exactly
  one member) are decompressed with the system `gzip`/`tar` into
  `staging_dir` (required; no `/tmp` default) as `<sha1(path, size,
  mtime)>-<name>`, checked for a NetCDF signature and published with
  `atomic_publish`. A cached file is reused until its source changes.

### Flux units

- One grammar: `<amount>[<substance>]<factors>[, <qualifier>]`, ignoring case,
  whitespace, `.`/`*` separators and LaTeX exponent markup. Amounts `kg`, `g`,
  `tonnes`, `mol`, `mmol`, `µmol`; substance `C` or a species; factors an area
  (`m-2`, `/m2`, `cell-1`; none means per cell) and a time (`s-1`, `d-1`,
  `month-1`, `yr-1` and slash/`per` forms). `Tonnes` alone is the EDGAR
  convention, tonnes per cell and year. The only qualifier is SDGVM's
  `per unit land area occupied by the PFT`.
- Strings that name no substance are of the `TargetSpecies` unless the source
  declares one (`substance = :carbon` for TRENDY's `kg m-2 s-1`). Molar units
  need a substance.
- `kg_per_unit`: carbon → species by `M(species) / CARBON_MOLAR_MASS` (one
  carbon atom; for `kgC` and `:co2` exactly
  `SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS`), mol → kg by
  `SPECIES_MOLAR_MASS`, units of another species throw. `period_seconds`: 1,
  86400, the calendar month containing the interval, or
  `DAYS_PER_JULIAN_YEAR × 86400`. `mass_rate_factor = kg_per_unit /
  period_seconds`; for per-second units it equals the runtime's species factor
  bit for bit, and `period_seconds` of GridFED's and EDGAR's units equal the
  runtime's divisors bit for bit.
- `canonical_units_string(:co2) == "kg CO2 m-2 s-1"` parses back to kg of
  `Species(:co2)` per m² per second.

### `cs_native` flux files

- Layout: each series has Julia dimensions `CS_NATIVE_FLUX_DIMS = ("Xdim",
  "Ydim", "nf", "time")` (NetCDF `(time, nf, Ydim, Xdim)`), `Xdim = Ydim = Nc`,
  six panels in the mesh's panel convention. Every live input (the TRENDY,
  GFED5, FLUXCOM-X, SIF and CATRINE files of `config/runs`) uses it.
- Values: kg of a species or of carbon per m² per second, positive into the
  atmosphere. The reader accepts exactly the units the runtime always accepted
  (`kg CO2 m-2 s-1`, `kg m-2 s-1`, `kgC m-2 s-1`, `kg/m2/s`, …): kg (not g or
  mol), per m², per second. Fill values (`_FillValue`, `missing_value`) and
  non-finite values are errors; the writer stores no `_FillValue` and refuses
  non-finite values.
- Time: start-stamped slices, strictly increasing, decoded with `decode_time`
  (default decoders). The writer takes contiguous `TimeInterval`s and stores
  `time` (interval starts) and `time_bnds` (start, stop) as Float64
  `"hours since <first day> 00:00:00"`, `calendar = "proleptic_gregorian"`;
  the stamps are decoded back and must round-trip.
- Geometry: the writer calls `Output.define_cs_horizontal_geometry!`
  (`lons`/`lats`, corners, `cell_area = mesh.cell_areas`, the `cubed_sphere`
  mapping with `semi_major_axis = mesh.radius`).
- Cell-area gate (`check_cell_area`): a stored `cell_area`, `(Nc, Nc)` or
  `(Nc, Nc, 6)`, must satisfy `isapprox(panel, mesh.cell_areas; rtol = 1e-4)`
  for every panel (norm-based), the runtime's gate since `cs_native` exists.
  The element-wise gate of the design (error above 1e-3, warning above 1e-9,
  `semi_major_axis` check) is scheduled for P11.
- Chunks `(Nc, Nc, 1, chunk_time)`, deflate with shuffle: `chunk_time = 1`
  for daily and `24` for hourly files (P0 read benchmarks).
- Publication is atomic: the writer works on an `AtomicFiles` staging file and
  `publish!` renames it onto the path only when every slice of every series has
  been written; any failure removes the staging file and leaves a previous
  file in place. The global attributes carry the provenance block, the species
  and, when given, the build configuration text and its SHA-1.
- Reading: `cs_native_flux_info` applies the checks that need no values
  (layout, time axis, units, cell area) and selects the slices;
  `read_cs_native_flux!` reads them panel by panel (the chunk layout) straight
  into the caller's arrays, converting each value once to their element type,
  with one panel-slab read buffer as its only array allocation (tested with
  `@allocated`), so the runtime holds a series once. `read_cs_native_flux`
  returns the whole series in one array (the builder's readback).
- The runtime (`Models.InitialConditionIO`) adds only the slice times in
  seconds since the run start, the products density × `mesh.cell_areas` ×
  `mass_rate_factor(units, species)` × `scale` × the storage factor (in place,
  in the run's float type), and the temporal scheme.

### Lat-lon flux files

- `write_latlon_flux` writes static CF `(lon, lat)` Float32 densities (kg of
  the species per m² per second, positive into the atmosphere, no
  `_FillValue`), `lon`/`lat` centres from the mesh (the source-mesh rule
  rebuilds the mesh faces from them), `lon_bnds`/`lat_bnds` and the mesh
  `cell_area`, atomically. There is no time axis: `kind = "file"` reads one
  static field per tracer and is not a time-varying kind.
- The runtime rebuilds the faces of a `kind = "file"` source from the oriented
  centres with `SnappedFaces`, not from bounds. The writer refuses a mesh whose
  faces that rule does not reproduce bit for bit, cell by cell up to the 360°
  shift of the orientation, because the runtime would otherwise regrid the
  field on different cells.

## Tests

`test/core/test_gridded_io_coordinates.jl`, `test_gridded_io_source_meshes.jl`,
`test_gridded_io_time_axes.jl`, `test_gridded_io_staging.jl`,
`test_flux_units.jl`, `test_cs_native_flux_schema.jl` (writer → reader round
trip, atomic publication, reader checks, the cell-area gate against a frozen
copy of the runtime gate, the streaming reader and its one-slab allocation
bound, and the migrated `cs_native` loader against a frozen copy of the old
one, bit for bit), `test_latlon_flux_writer.jl` (static files, the runtime
reading them, and the face gate). Each freezes
the runtime code it replaces and compares bit for bit where the behaviour is
preserved.
