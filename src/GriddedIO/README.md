# GriddedIO

The one owner of the conventions of native gridded input files: surface-flux
inventories, land fractions and initial-condition fields. It decides how a
file's coordinates are found, which values are fill, how a latitude-longitude
field is oriented, which cell faces its coordinates imply, how its time axis is
decoded, how a compressed input is staged, and how its flux units are read.

Every rule was measured on the files the model reads (TRENDY v14 S3, GFED5,
FLUXCOM-X, CAMS/LMDZ, GridFED, EDGAR); the measurements are in
`docs/memos/2026-10-10_python_flux_port_p0_facts.md` and the design in
`docs/memos/2026-10-10_python_flux_port_design.md` (Section 2.4). The rules are
types: the caller names an axis, a fill policy, a time decoder or a substance,
and the methods dispatch on it.

`src/AtmosTransport.jl` loads this folder after `Regridding` and before
`Preprocessing`. It imports `Parameters` (`SPECIES_MOLAR_MASS`,
`CARBON_MOLAR_MASS`, `DAYS_PER_JULIAN_YEAR`), `Grids` (`LatLonMesh`) and
`AtomicFiles` (`atomic_publish`), plus NCDatasets (and its CFTime), Dates and
SHA. It never imports `Preprocessing` or `Models`, so the runtime loaders
(`Models.InitialConditionIO`) and the surface-flux builder can both use it.

## File Map

| File | Contents |
|---|---|
| [`GriddedIO.jl`](GriddedIO.jl) | Module, imports, exports, includes. |
| [`coordinates.jl`](coordinates.jl) | `AbstractAxis` (`LongitudeAxis`, `LatitudeAxis`, `TimeAxis`, `VerticalAxis`) and `find_coordinate`; `LatLonOrientation`, `latlon_orientation`, `orient`, `orient_coordinates`; `AbstractFillPolicy` (`MaskedAsZero`, `MaskedAsNaN`, `FillValueOnly`) and `read_raw([T,] ds, name, policy, idx...)`. |
| [`source_meshes.jl`](source_meshes.jl) | `source_latlon_mesh(lon, lat, rule; radius)` and `source_latlon_mesh(ds, rule; radius)` with the face rule `AbstractFaceRule` (`SnappedFaces`: snapping rule, centre-on-0/180 rule, midpoint and pole rules on `LatLonMesh(λᶠ, φᶠ; radius)`). |
| [`time_axes.jl`](time_axes.jl) | `TimeInterval`, `MonthKey`, `interval`, `month_keys`, `seconds_since`, `check_last_month`; `AbstractTimeDecoder` (`CFDecoder`, `MonthsSince`, `DecimalYears`, `AuxiliaryTimeVariable`) with the stamp positions `MonthStart`, `MidMonth`, `MonthEnd`; `DEFAULT_TIME_DECODERS`, `decode_time` and `decode_month_keys`. |
| [`staging.jl`](staging.jl) | `stage_input(path; staging_dir)`: `.nc.gz` and single-member `.nc.tar.gz` decompressed through the system `gzip`/`tar`, cached, published atomically. |
| [`flux_units.jl`](flux_units.jl) | `FluxUnits{S,A,T}` with `AbstractSubstance` (`TargetSpecies`, `Carbon`, `Species`), `AbstractAreaBasis` (`PerSquareMetre`, `PerCell`), `AbstractTimeBasis` (`PerSecond`, `PerDay`, `PerMonth`, `PerYear`); `parse_flux_units`, `kg_per_unit`, `period_seconds`, `mass_rate_factor`, `canonical_units_string`. |

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
(`AbstractFaceRule`); `SnappedFaces()`, the default, applies these rules:

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
  the runtime loaders' uniform mesh (`Models.InitialConditionIO`'s
  `_build_source_latlon_mesh`) bit for bit. On Float32 centres that mesh takes
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
  axis by a month, which `check_last_month` catches. Fill values in a time axis are an error. Time bounds are not read,
  and the axis length is never used to guess its meaning.

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

## Tests

`test/core/test_gridded_io_coordinates.jl`, `test_gridded_io_source_meshes.jl`,
`test_gridded_io_time_axes.jl`, `test_gridded_io_staging.jl`,
`test_flux_units.jl`. Each freezes
the runtime code it replaces and compares bit for bit where the behaviour is
preserved.
