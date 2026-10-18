# TOML schema

This page is the **canonical reference** for the TOML configs that
drive both the runtime (`scripts/run_transport.jl`) and the
preprocessor (`scripts/preprocessing/preprocess_transport_binary.jl`).
Per-block, per-key, with type, default, and what it does.

The two config families (run vs. preprocessing) live in different
directories and are consumed by different code paths; this page
keeps them separate.

## Run config (`config/runs/*.toml`)

Editors with Taplo (Even Better TOML in VS Code) check run configs in
`config/runs` and `config/examples` against
`schemas/atmos_transport_run.schema.json` (`.taplo.toml`): every key has a
description on hover, and unknown keys are marked, as `validate_config`
rejects them. The archived directories `config/runs/completed_experiments`
and `config/runs/likely_legacy` are not checked.

Consumed by `scripts/run_transport.jl`, which parses the TOML and forwards
the resulting Julia dictionary to `run_driven_simulation(cfg)`.

The runtime infers the **target topology** from the binary header's
`grid_type` field at load time; a
`[grid]` block in the run config is therefore unnecessary and
ignored.

`validate_config(cfg)` checks runtime table shapes, input path existence,
precision/backend compatibility, and window bounds without opening binary
readers or allocating model state. It also parses the physics sections,
`air_mass_reset_mode`, `physics_cadence`, `prefetch_windows`, `[logging]`,
`[checks]` and `[output]` as the run does, so their errors are reported
together; decay half-lives must name a tracer, and an enabled `[output]` needs
both a path and snapshot times (`hours = []` means deliberately none). Nested
tracer `init` and `surface_flux` values must be tables. Shape errors are
returned before value checks; a successful result is not a full physics or
binary validation. See
[Run with real meteorology](@ref Run-with-real-meteorology) for an example.

Unknown tables and keys are errors when the config is checked (also at the
start of every run), with a suggestion when a known key is close
(`[diffussion]` → `[diffusion]`, `order` → `ppm_order`); so is a
`[output.fields.per_tracer.<name>]` table for a tracer the run does not carry.
Known keys that the chosen kind or another setting leaves unread are logged as
warnings, such as `value` without `[diffusion] kind = "constant"`, flat
`[tracers.<name>]` keys next to an `init` table, a `surface_flux` table
without `kind`, or `start_hour` without an interval key. A surface-flux kind
that is not one of the named sources is an error; use `kind = "file"` for a
generic NetCDF file. When the run starts and has read every binary's header,
snapshot hours that do not fall on a met-window end are an error (such an
hour would never be written, and every later snapshot would be lost with it),
and `format = "binary_mmap"` is rejected on lat-lon and reduced-Gaussian
grids.

### `[input]` — which transport binaries to load

Two valid shapes (mutually exclusive):

```toml
# Shape A — explicit list
[input]
binary_paths = [
    "~/data/.../era5_transport_20211201.bin",
    "~/data/.../era5_transport_20211202.bin",
]

# Shape B — folder + date range
[input]
folder       = "~/data/.../era5_ll72x37_dec2021_f32/"
start_date   = "2021-12-01"
end_date     = "2021-12-03"
file_pattern = "{YYYYMMDD}"   # optional; default scans for date stamp
```

Path expansion + continuity validation lives in
`src/Models/BinaryPathExpander.jl`.
Shape B asserts that the resolved binaries form a contiguous date
sequence; gaps fail at expansion time, not at first window-load.
Shape A preserves the explicit list's order after expanding paths. It does
not sort entries or validate date continuity; provide them chronologically.

#### `[input]` checks of the binaries

- `validate_replay = true` replays every binary's continuity when it is
  opened: the stored fluxes must carry each window's air mass to the next
  in every cell, within `replay_tolerance` (1e-12 in Float64, 1e-5 in
  Float32, of the summed magnitudes of the cell's budget terms). Off
  by default (it doubles binary load time); the preprocessor already checks
  every binary it writes. Use it for binaries of unknown origin.
- `expected_nlevel`, `required_preprocessor_contract` and
  `require_adaptive_substeps` reject a first binary whose header differs.
- The run origin is 00:00 UTC of the day of window 1 of the first binary.
  `start_date`, that binary's `date` header and a `YYYYMMDD` (or
  `YYYY-MM-DD`) date in its file name each name this day when present, and
  every run requires them to agree, with or without output. Output needs at
  least one of them for its CF time origin; time-varying surface fluxes need
  `start_date`.

#### `[input.staging]` — rolling NVMe staging (opt-in)

This value must be a TOML table even when staging is disabled. A scalar such
as `staging = false` is rejected during configuration preflight; use
`[input.staging] enabled = false` on separate TOML lines.

Transport binaries (~15 GB/day) usually live on NFS-mounted storage. Cold,
strided per-window reads over NFS make the window prefetch IO-bound (measured
on a C180 GEOS 6-day run: `prefetch_fetch_wait` 96 s → 5 s, wall 265 s → 160 s,
−40 %, when reading from local NVMe instead). For long (multi-month/year) runs
the dataset exceeds RAM, so every new day is a cold NFS read mid-run.

Enable rolling staging to copy a small look-ahead window of upcoming day-files
onto fast local disk and evict processed days, bounding local use to
`lookahead_days + 1 + keep_behind_days` files regardless of run length:

```toml
[input.staging]
enabled          = true                       # default false ⇒ read NAS directly
dir              = "/temp1/atmostransport_stage"   # local NVMe directory (required)
lookahead_days   = 2                          # days kept staged ahead (default 2)
keep_behind_days = 0                          # processed days retained (default 0)
cleanup_on_exit  = true                       # remove staged files at run end (default true)
```

Default off ⇒ bit-identical to a non-staged run. Copies run on a background
task (overlapping GPU transport); a copy failure transparently falls back to the
NAS path. Each active run owns its staging directory; another run targeting
that directory falls back to the source paths. Use distinct directories when
concurrent runs should both stage inputs.

With `cleanup_on_exit = false`, retained copies can be reused on a later run.
Reuse requires matching source path, size, modification/change times, and inode
metadata in a `.source.toml` sidecar. This rejects equal-sized rewritten inputs
and old copies without metadata. These checks are filesystem identity checks,
not content checksums: source binaries must remain immutable during a run.
Cleanup removes this run's staged files and metadata, preserving unrelated
files in the directory. Completed staged binaries and metadata sidecars are
published through `AtomicFiles`, so readers never open a partial copy.
Implementation: `src/Models/InputStaging.jl`.

### `[architecture]` — backend selection

```toml
[architecture]
use_gpu = true                # default: false
backend = "auto"              # default: "auto" if use_gpu else "cpu"
```

| `backend` | Effect |
|---|---|
| `"cpu"` | CPU only. **Conflicts with `use_gpu = true` and errors at config-load time** (see `src/Architectures.jl`). |
| `"cuda"` | NVIDIA CUDA via `CUDA.jl` (must be installed). |
| `"metal"` | Apple Silicon Metal via `Metal.jl`. F32 only. |
| `"auto"` | Auto-detects an available GPU backend (CUDA → Metal); errors if none is available. |
| (omitted) | If `backend` is absent, the runtime picks `CPU()` when `use_gpu = false` and auto-detects `GPU(:cuda)` or `GPU(:metal)` when `use_gpu = true`. |

The resolved `CPU()` or `GPU(:cuda|:metal)` architecture is stored on the grid
and also controls model-array adaptation, synchronization, and runtime checks.
Optional GPU packages load on demand without injecting modules into `Main`.

### `[numerics]` — precision

```toml
[numerics]
float_type = "Float32"        # default: "Float64"; one of "Float32" / "Float64"
```

Both precisions are supported on CPU and CUDA. Their speed and memory costs
depend on the GPU and enabled operators; the V100 experiments in
[Validation status](@ref) cover both. Metal requires Float32. Mixing with the
binary's `on_disk_float_type` is allowed: the runtime casts on load, which
does not recover precision already lost in the stored forcing.

### `[run]` — runtime knobs

```toml
[run]
start_window = 1              # default: 1 — first window to process
stop_window  = 24             # default: nothing — uses the binary's full range
air_mass_reset_mode = "preserve_tracer_mass"
# physics_cadence = "window"  # default; "substep" runs convection and chemistry
                              # every advection substep (cadence comparisons)
# prefetch_windows = true     # default; false loads each met window when needed
```

`stop_window` is the inclusive last window; setting it lets you
run a partial day for smoke tests with a single input file. Multi-file runs
require complete window ranges so no forcing is skipped between files.
The index is local to each input file, not cumulative across the run. For two
daily binaries with 24 windows each, omit `stop_window`; the result covers
48 hours while each file supplies windows 1–24. Output times are cumulative.
Both indices must be integers: Boolean and floating-point values are rejected.
`start_window` must be at least 1, and `stop_window` must not precede it.
Cubed-sphere runs currently require `start_window = 1`.
`air_mass_reset_mode` is one of
`"none"`, `"preserve_vmr"`, or `"preserve_tracer_mass"`. GPU runs with at
least two Julia threads load the next met window on a second thread while the
current one runs; `prefetch_windows = false` turns that off (for debugging; the
results are the same). Advection belongs in the separate `[advection]` table.

### `[tracers.<name>]` — per-tracer setup

Each tracer gets its own block. The name is what shows up in the
output NetCDF; `[tracers.co2_bl]` writes `co2_bl`,
`co2_bl_column_mean`, etc.

```toml
[tracers.co2_bl.init]
kind        = "bl_enhanced"   # initial-condition kind; see table below
background  = 4.0e-4          # uniform background dry VMR (mol/mol)
enhancement = 1.0e-4          # extra dry VMR in lowest n_layers (LL only)
n_layers    = 3
```

Initial-condition kinds (LL/RG builders in `src/Models/InitialConditionIO.jl`,
CS builder in `src/Models/initial_conditions/cubed_sphere.jl`):

| Kind | LL | RG | CS | Required keys |
|---|---|---|---|---|
| `"uniform"` | yes | yes | yes | `background` |
| `"latitude_step"` | yes | yes | yes | optional `south_value`, `north_value`, `split_lat_deg` |
| `"bl_enhanced"` | yes | **no** | **no** | `background`, `enhancement`, `n_layers` (LL-only; RG/CS path errors at IC build) |
| `"gaussian_blob"` | yes | yes | yes | `background`, `lon0_deg`, `lat0_deg`, `sigma_lon_deg`, `sigma_lat_deg`, `amplitude` |
| `"file"` / `"netcdf"` | yes | yes | yes | `file`, `variable`, optional `humidity_variable`, `time_index`, `vertical_order` |
| `"file_field"` | yes | yes | yes | `file`, `variable`, optional `humidity_variable`, `time_index`, `vertical_order` |
| `"catrine_co2"` | yes | yes | yes | optional `file`, `variable`, `humidity_variable`, `time_index` overrides for the built-in defaults |
| `"pressure_layer"` | no | no | yes | `lowest_layer = true` or `psurf_fraction`; optional `total_molecules` |
| `"cs_native"` | no | no | yes | `file`, `variable`; optional `humidity_variable`, `time_index`, `vertical_order`, `clamp_negative` (default `false`) |

The file kinds (`file`, `netcdf`, `file_field`, `catrine_co2`) read a
lon-lat-level field of layer-mean dry VMR. A file with hybrid coefficients
`ap` and `bp` is oriented by its interface pressures and conservatively
remapped, whatever its level count; `Psurf` is not needed or read. In each
column the source interfaces are `ap + bp * ps_target`, using the transport
binary's surface pressure, and the target interfaces are
`A_target + B_target * ps_target`. The remap preserves layer pressure
integrals. It is terrain-following near the surface, becomes absolute-pressure
mapping where `bp = 0`, and extends the nearest end layer if a pressure-level
source does not span the target top or surface. This avoids making an initial
VMR depend on topographic differences or the preprocessing dry-mass pin; see
the equations and CATRINE measurements in
[`src/Models/initial_conditions/README.md`](https://github.com/RemoteSensingTools/AtmosTransport/blob/main/src/Models/initial_conditions/README.md).

The dry-air weighting uses source specific humidity only when it is explicitly
selected with `humidity_variable = "name"`; without that key, source humidity
is not loaded. If the transport binary
stores dry surface pressure, its window-start endpoint humidity is used to
recover total pressure; on native cubed-sphere physics binaries VDIFF `qv` is
used when endpoints are absent. A legacy dry-pressure binary with no humidity
cannot determine total pressure, so the runtime keeps dry `ps` for hybrid
interface placement and emits one warning. This approximation is typically
0.5--2 % low in humid columns. Current binaries record this choice explicitly
as `surface_pressure_basis = "total"` or `"dry"`; keyless legacy binaries are
inferred once when opened and warn that regeneration is required.

A file without `ap` and `bp` is copied level by level, so it must have the
model's level count, and its vertical coordinate must say which end is the
top: pressure units or CF `positive = "down"` (values grow toward the
surface), height units or `positive = "up"`; otherwise set
`vertical_order = "toa_first"` or `"surface_first"`.

Native cubed-sphere ICs keep their cell-for-cell horizontal mapping. When their
level count matches the binary, `ap`/`bp` are used when present to infer level
orientation and must match the binary coefficients within numeric tolerance
before the exact vertical copy. Without them, `vertical_order` controls the
copy. When the level count differs, `ap` and `bp` are required and the same conservative
hybrid remap, dry-carrier humidity weighting, and typed surface-pressure basis
described above are used column by column. In both IC paths, `ap` has pressure
units: an explicit units attribute must be `Pa`; a missing attribute is accepted
only for all-zero pure-sigma coefficients or magnitudes that unambiguously
indicate pascals. Ambiguous hPa-sized values are rejected.

`pressure_layer` places tracer in one level per column. With
`lowest_layer = true`, it uses the bottom level and ignores `psurf_fraction`.
Otherwise it chooses the level whose logarithmic pressure midpoint is nearest
`psurf_fraction * surface_pressure`, using the binary's hybrid coefficients and
per-column surface pressure. The fraction defaults to `0.5` and must be in
`(0, 1]`; equal distances select the first level. One uniform dry VMR across
the selected cells is normalized to the positive `total_molecules`
(default `1e22`), using their summed dry-air mass. Other levels start at zero.

Surface-flux emission is configured as a nested sub-table under each
tracer. Each tracer that emits gets one `[tracers.<name>.surface_flux]`
block with a `kind` selector and per-kind keys:

```toml
[tracers.co2_fossil.surface_flux]
kind       = "gridfed_fossil_co2"   # one of the registered source kinds
time_index = 12                     # positional slice index (1-based)
# month = 12                        # decoded calendar month when time exists
scale      = 1.0                    # optional multiplicative scaling

[tracers.sf6.surface_flux]
kind = "edgar_sf6"

[tracers.ocean.surface_flux]
kind     = "file"
file     = "/path/to/air_sea_flux.nc"
variable = "co2_flux_ocean"
species  = "co2"                    # required: the name "ocean" implies no species
# regridding = "conservative"       # default; "bilinear" on lat-lon and reduced Gaussian
```

Registered surface-flux source kinds (`_SURFACE_FLUX_KINDS` in
`src/Models/initial_conditions/surface_flux.jl`): `lmdz_co2`, `gridfed_fossil_co2`,
`edgar_sf6`, `zhang_rn222`, plus a generic `file` for arbitrary
NetCDF sources and `cs_native` for time-varying fluxes already on the native
cubed-sphere grid. Any other kind is an error. There is no `edgar_co2` kind — use
`gridfed_fossil_co2` for the GridFED-derived fossil CO₂ inventory.
A `file` source is one static field (a 2-D variable, or slice `time_index` of a
3-D one); it is not a time-varying kind. `GriddedIO.write_latlon_flux` writes
such files.
`species` names the species whose kilograms the flux counts: `"co2"`, `"sf6"`
or `"rn222"`, the entries of `Parameters.SPECIES_MOLAR_MASS`. Its molar mass
converts physical kg species s⁻¹ to the model's storage units (dry VMR × dry-air
mass: × `DRY_AIR_MOLAR_MASS / M_species`). Tracers named `co2`, `co2_natural`,
`co2_fossil` or `fossil_co2` (CO₂), `sf6` or `rn222` may omit it; any other
tracer with a flux must set it, and another value is an error that lists the
known species.

The flux variable's `units` attribute decides the conversion to
kg species m⁻² s⁻¹ (`GriddedIO.parse_flux_units`): kg, g, tonnes or mol; of
the species, unnamed, or of carbon (× M(species)/M(C), 3.66419 for CO₂); per m²
or per cell (÷ the cell area); per second, day, calendar month or Julian year.
Units that name another species than `species` are an error. The shipped
inventories use `kgC m-2 s-1` (CAMS/LMDZ), `kgCO2/month/m2` (GridFED, each
month's length), `Tonnes` (EDGAR, per cell and year) and `kg/m2/s` (Zhang
Rn-222). Time-varying lat-lon sources must be per m²; `cs_native` files must be
kg (of the species or of carbon) m⁻² s⁻¹.

**Regridding.** A static lat-lon inventory is put onto the model mesh by
`regridding`:

- `"conservative"` (the default on every topology): the per-cell rate, the
  flux times the file's `cell_area` or `area` variable (else the
  latitude-circle area of the source cell), is regridded as an extensive
  field, so the model's total equals the file's own total to rounding, for
  net-uptake fields and files without an area variable alike. A file's
  area variable must be in m² and agree within 1 % with the cell areas its
  coordinates define (`GriddedIO.check_latlon_cell_area`), else the run stops
  with an error: areas in km² or a land fraction would scale the total. A
  cell with a nonzero flux must also have a positive, finite area: masked
  (zero) rows would drop their emission. On
  reduced-Gaussian targets, whose great-circle cells leave slivers between
  rings uncovered, each source cell's amount goes to the target cells it
  overlaps, so the total is kept there too. On a target that does not cover
  the sphere, a source cell covered only in part puts its whole amount into
  the covered part (see [Operators](@ref Operator-concepts)).
- `"bilinear"` (lat-lon and reduced-Gaussian targets, only when set): the
  flux density sampled at each target cell centre times the cell area. The
  emitting and the uptake cells are then rescaled separately to the file's
  emission and uptake totals, so the total is the file's and no cell changes
  sign. When the sample misses all of the file's emission or all of its
  uptake (one uptake cell between the sampled centres), the run stops with an
  error. The two factors keep the totals, not the cell values, and can be far
  from 1 (3.53 for the GridFED December 2021 uptake on a 5° grid); they are
  logged, and a factor more than 10 % from 1 is a warning. Bilinear sampling
  still aliases point sources cell by cell.

Cubed-sphere targets always regrid conservatively (`"bilinear"` there logs a
warning and is ignored). The equations are in the surface-flux section of
[Operators](@ref Operator-concepts).

**Time-varying emission** (cubed-sphere). A source whose inventory has
sub-monthly time slices (e.g. the LMDZ/CAMS biospheric flux) can drive the
diurnal cycle instead of a monthly mean:

```toml
[tracers.co2_natural.surface_flux]
kind            = "lmdz_co2"
file_pattern    = "$ATMOSTRANSPORT_DATA_ROOT/catrine/Emissions/LMDZ_fluxes/z_cams_l_cams55_{YYYYMM}_FT24r2_ra_sfc_3h_co2_flux.nc"
year            = 2022
time_varying    = true            # advance through the inventory's time slices
temporal_scheme = "stepwise"      # how slices are applied between sample times
```

`file_pattern` expands `{YYYYMM}` to all twelve months of `year` (or the
run-start year when `year` is omitted). Each file's time axis is decoded by
`GriddedIO.decode_time` into `DateTime` stamps on the run's proleptic
Gregorian clock:

- `days`, `hours`, `minutes` or `seconds since <origin>` in the calendars
  `standard`/`gregorian`, `proleptic_gregorian` and `julian` (the instants
  themselves), and `noleap`/`365_day`, `all_leap`/`366_day` and `360_day` (each
  stamp's calendar date and time of day, carried over). Every `noleap` date
  exists on the run's clock; an `all_leap` or `360_day` stamp on 29 February
  of a common year or on 30 February does not, and is an error that names it.
- `months since <origin>` as the first instant of each month.
- The origin must be a valid instant: a month outside 1–12, a day outside
  1–31 (for `days`…`seconds since`, beyond the month in the axis's calendar)
  or an invalid time of day is an error, never rolled over
  (`days since 2000-13-01` is not 2001-01-01).
- Axes without such units are refused.

For a span that crosses calendar
years, use `files = ["/path/to/month1.nc", "/path/to/month2.nc", ...]` in
chronological order. GridFED supports the same time-varying path; its twelve
monthly totals are converted using the actual number of days in each month.

`temporal_scheme` (default `"stepwise"` for LMDZ and GridFED) is one of:

- `"stepwise"` — hold each slice piecewise-constant until the next sample.
  This matches GEOS-Chem/HEMCO's exact CAMS treatment (verified against
  `EmisCO2_Total`), so use it to reproduce GC.
- `"linear"` — linearly interpolate between the two bracketing slices,
  evaluated at the start of each step.
- `"conservative"` — integrate that linear interpolation exactly over each
  step, so the emitted mass matches the source's own time integral for any
  number of substeps.

Slices are indexed by **absolute** time since the run's `start_date`, so a
multi-day run advances through the inventory correctly (a per-day clock would
replay the first day's slices — the cause of the historical co2_natural
+1 Pg/month surplus, now fixed).

When `[input]` gives `start_date`, only the slices the run can use are read:
from the last slice at or before the run start to the first slice at or after
the run end. The run end is the end of the binaries' windows (read from every
binary's header). Every listed file's time axis is
read, but flux data only for those slices. Every temporal scheme blends only
the two slices around a time, so results are unchanged. A binary that runs past
that end stops the run with an error before it starts. Time-varying sources
require `start_date`.

**Time coverage.** The run must lie inside the time the slices stand for.
A slice stands for the interval of its CF time bounds when the time axis has
them; GridFED files without a time axis cover their year, a `months since`
axis its months. Stamps without bounds are read as one series over **all**
the listed files (merged and sorted; `GriddedIO.stamp_intervals`): a slice
holds until the next stamp when that is the series' regular step (one month
or one year from 1 January for month starts, else the smallest spacing), and
the last stamp holds for one step. Daily files with one 00Z stamp each are
thus a daily series; a slice before a longer spacing (Jan 1 then Mar 1, a
missing day) holds for one step, leaving a gap. One stamp in all the files
(it may stand for a day, a month or a year) and stamps about a month or a year
apart that are not month starts (mid-month stamps) have no known period and
stop the run with an error asking for CF time bounds.
A part of the run, or a gap between or inside files, that no slice covers
stops the run before it starts, with an error naming the uncovered dates: outside its
slices a source holds its first or last slice, so a December CAMS file in a
January run would emit its last 3-hourly value for weeks. Set
`hold_outside_coverage = true` to hold the end slices on purpose, e.g. for a
climatology:

```toml
[tracers.co2_natural.surface_flux]
kind                  = "lmdz_co2"
files                 = ["/path/to/z_cams_l_cams55_202112_FT24r2_ra_sfc_3h_co2_flux.nc"]
time_varying          = true
hold_outside_coverage = true      # default false: a run outside the files is an error
```

The loaded series stays in host memory; a GPU run copies
only the slices of the current met window to the device (see
[Operators](@ref Operator-concepts)), so the device memory of a time-varying
source does not grow with the run length.

For an already aligned GEOS-native cubed-sphere inventory, use
`kind = "cs_native"`, `time_varying = true`, `file`, and `variable`.
The file follows the `cs_native` schema of `AtmosTransport.GriddedIO`
(`GriddedIO.cs_native_flux_info` and `GriddedIO.read_cs_native_flux!` are the
loader; new files come from `GriddedIO.open_cs_native_writer`):

- the variable has NetCDF dimensions `(time, nf, Ydim, Xdim)`; grid resolution
  and panel order must match the meteorology;
- its `units` are kg of the species or of carbon per m² per second
  (`kg CO2 m-2 s-1`, `kg m-2 s-1`, `kgC m-2 s-1`; carbon is converted to CO₂
  by `SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS`); gram, molar, per-cell and
  missing units are errors;
- values are start-stamped slices on a strictly increasing time axis (decoded
  by `GriddedIO.decode_time`), with no fill or non-finite values; they are read
  panel by panel into the run's own arrays (`GriddedIO.read_cs_native_flux!`);
  `time_bnds` (written by `GriddedIO.open_cs_native_writer`), else the
  stamps, set the time coverage above;
- a stored `cell_area` must agree with the runtime mesh areas
  (`isapprox(…; rtol = 1e-4)` per panel), so a file built on another cube
  geometry is refused.

The loader multiplies by native mesh cell areas and converts to model storage
units with the tracer's species. It defaults to `temporal_scheme = "stepwise"`;
signed uptake is retained. The TRENDY ensemble example is
`config/runs/trendy_v14_s3_all_models_npp_rh_c30_2014_2024.toml`.

### `[advection]`, `[diffusion]`, `[convection]`, `[chemistry]`

Each operator has a selector and per-kind options. See
[Operators](@ref Operator-concepts) and [Advection schemes](@ref) for what each
selector means; relevant config keys:

```toml
[advection]
scheme    = "linrood"           # "upwind" | "slopes" | "ppm" | "linrood" | "none"
ppm_order = 7                   # cubed-sphere LinRoodPPM only; ∈ {5, 7}.
                                # Setting ppm_order with scheme = "ppm" errors.
# vertical = "fv3_kord8"        # scheme = "ppm": "same_as_horizontal" (default);
                                # scheme = "linrood": "fv3_kord8" (default);
                                # explicit "upwind" retains first-order vertical;
                                # either: "fv3_kord8" (FV3 scalar_profile,
                                # positive definite, the GCHP tracer profile)
                                # | "fv3_kord8_signed" (for tracers that go
                                # negative); FV3 options are cubed sphere only
# limiter = "cw84"              # scheme = "ppm" only: "monotone" (default) |
                                # "cw84" (complete Colella–Woodward PPM; keeps
                                # tracers non-negative in sweeps where no cell
                                # exports more than its air mass). "none"
                                # (unlimited PPM) is accepted only by the
                                # 4D-Var inversion driver
# check_binary_cfl = true       # cubed sphere, upwind | slopes | ppm: recompute
                                # the CFL subcycle count each step and stop if
                                # the binary's substep schedule asks for fewer
                                # (default false; slower)

[diffusion]
kind  = "constant"              # "none" | "constant" |
                                # "tm5_beljaars_viterbo_local_kz" (CS-only;
                                #   local Kz from the binary's pblh, ustar,
                                #   pbl_hflux and t2m) |
                                # "geoschem_holtslag_boville_vdiff" (CS-only;
                                #   local Kz; requires include_gchp_vdiff=true binary) |
                                # "geoschem_nonlocal_vdiff" (CS-only; GEOS-Chem's
                                #   non-local PBL scheme incl. counter-gradient
                                #   transport of fresh emissions; needs VDIFF +
                                #   :pbl_eflux, e.g. the MERRA-2 GEOS-Chem archive;
                                #   always S(dt)->V(dt)) |
                                # "tm5_dkg" (CS-only; exact TM5 dry-air
                                #   interface exchange — requires a
                                #   binary built with include_tm5_diffusion=true)
value = 1.0                     # m²/s — broadcast Kz when kind="constant"
surface_flux_boundary = false   # LL/CS: true selects S(dt)->V(dt).
                                # false selects V/2->S->V/2; RG requires false.

[convection]
kind = "cmfmc"                  # "none" | "cmfmc" | "cmfmc_matrix" | "tm5"
                                # cmfmc_matrix = TM5 LU solver on GEOS CMFMC
                                # rates; tm5 = TM5 entrainment (:entu/:detu/
                                # :entd/:detd payload)
cloud_base = "cmfmc"            # cmfmc only: "cmfmc" = lowest layer with updraft
                                # inflow; "dqrcu" = GEOS-Chem's cloud base from the
                                # binary's :cmfmc_cloud_base (CS-only)

# Collaborative-LU knobs (cmfmc_matrix and tm5). use_collab_lu is REQUIRED for
# lmax_conv / n_merge to take effect — setting them without it is a hard error.
use_collab_lu = true            # batched/collaborative column LU (fast path)
lmax_conv     = 0               # 0 = full column; >0 retains that many bottom layers
n_merge       = 1               # 1 = no aggregation; >1 merges adjacent vertical
                                # layers within each column (a numerical approximation;
                                # not additive in the tracer — use 1 for tagged-tracer
                                # budgets, anomaly transport and adjoints; see
                                # "Level merging" in operators_on_binaries.md)

# TM5 and CMFMC-matrix: per-topology legacy column-tile budget in GiB.
# Collaborative GPU runs defer this global scratch allocation. A CPU or
# unsupported Float64 fallback allocates the tile when first needed.
tile_workspace_gib = 1.0

[chemistry]
kind = "decay"                  # currently only first-order decay
  [chemistry.half_lives_seconds]
  rn222 = 330350.4              # per-tracer half-lives (seconds)
```

For Float32 GPU matrix convection, each column's LU factors remain in shared
memory while tracers pass through in batches of six. This buffer size does not
limit the run to six tracers; 65-tracer runs are covered by V100 regression
tests. The effective vertical matrix depth must still fit the supported
1–85-level envelope. Float64 CUDA uses the same batched solver for unmerged
(`n_merge=1`) depths 1–73, with Float64 shared arrays and arithmetic. CPU and
unsupported Float64 configurations keep the legacy solver with a warning;
that fallback uses the full column without aggregation, so requested
`lmax_conv`/`n_merge` approximations do not apply there. There is no automatic
truncation or precision conversion. CS adjoint footprints
continue to require `use_collab_lu=false`, `lmax_conv=0`, and `n_merge=1`.
An eligible Float64 request now engages the collaborative solver instead of
falling back; a positive `lmax_conv` therefore selects that lower-atmosphere
region. Use the full-column legacy settings above to retain the previous solve.
Tracer batching requires neither truncation nor layer aggregation. Choose
`lmax_conv` and `n_merge` from scientific accuracy checks;
conservation alone does not establish that either approximation is acceptable.

With layer aggregation, the solver uses `L_super = fld(L, n_merge)`, where
`L` is the requested active depth (or `Nz` when `lmax_conv=0`). Its fine span is
`L_super*n_merge`, so a nondivisible choice omits up to `n_merge-1` additional
top layers. Cloud-top closure forces remaining updraft mass to detrain at the
active boundary; it does not recover the mixing excluded by that boundary.
Each fine layer then receives a share of its super-layer's tracer-mass change,
weighted by `|r_k|/Σ|r_j|` over the super-layer (air-mass weights when all its
values are zero, equal weights if its air mass is zero too). Non-negative tracers stay non-negative, layers the solve did
not change keep their values, and no cell moves by more than its super-layer's
change; the equations are in the
[level-merging section](@ref tm5-level-merging).
Conservation is to rounding, not exact arithmetic. The default `n_merge` is 1.

Topology checks happen while the runtime recipe is built: reduced-Gaussian
runs accept `upwind` or `none` advection and require midpoint surface-flux
splitting, while Lin-Rood and the binary-derived PBL closures are cubed-sphere
only. The runtime also rejects operator selections that the loaded binary does
not support (for example, `convection.kind = "cmfmc"` against a binary lacking
`:cmfmc` payload). These checks run before model allocation or the first
transport step. See
[Binary format](@ref Binary-format) for the capability surface.

### `[output]` — snapshots

```toml
[output]
format        = "netcdf"       # "netcdf" | "binary_mmap" (ATMSNAP)
path          = "~/data/.../my_run.nc"
cadence_hours = 3              # or hours = [0, 6, 12, ...]
split         = "single"       # "single" | "daily"
deflate_level = 0              # NetCDF4 deflate (0..9); 0 = no compression
shuffle       = true           # shuffle filter; only effective when deflate>0
```

`split = "single"` writes one file after the run. `split = "daily"` writes
one complete file per daily binary; use `{date}` or `{YYYYMMDD}` in `path`
for an explicit filename template, otherwise the date is inserted before the
suffix. Use the current `path`, `hours`, and `cadence_*` keys shown above. See
[Output schema](@ref) for the per-topology variable list the file actually
contains.

`format = "binary_mmap"` writes fast self-describing per-day **ATMSNAP** binary
files inline (skipping the NetCDF/HDF5 encode in the GPU run), to be converted
to NetCDF offline on CPU with `scripts/postprocess/binary_to_netcdf.jl`. This
is the throughput path for long multi-day runs. The ATMSNAP payload is **always
Float32 on disk**, including for `float_type = "Float64"` runs — the on-disk
spatial precision is independent of the compute precision. Each snapshot also
stores a compensated Float64 global tracer total in the ATMSNAP JSON header;
offline conversion copies it exactly to `<tracer>_total_mass(time)` in NetCDF.
This preserves signed mass-balance diagnostics even when large positive and
negative spatial values nearly cancel.

Optional field selection keeps production files small:

```toml
[output.fields]
tracers = ["co2_natural", "co2_fossil"]  # omit for all tracers
layers = "none"                          # "full" | "selected" | "none"
levels = [1, 32, 64]                     # used when layers = "selected"
column_mean = true
column_mass_per_area = false
air_mass_layers = "none"
air_mass = false
air_mass_per_area = false
column_air_mass_per_area = true

[output.fields.per_tracer.co2_natural]
layers = "selected"
column_mean = true
```

Defaults match the historical writer: all tracers, full per-level tracer VMR,
column means, column tracer mass per area, stored air mass, layer air mass per
area, and column air mass per area.

### `[output.observations]` — sampling at observation points

Instead of (or in addition to) gridded snapshots, a run can sample tracers at
observation points. A step-by-step guide with a runnable example
(`config/examples/observation_sampling_oco2mip.toml`) and Python recipes for
OCO averaging kernels is in `docs/memos/2026-10-02_observation_sampling_guide.md`. **Point events** are sampled once at their own time:
satellite soundings, ObsPack flask, continuous, or aircraft records, and
station time lists. **Station series** are written at every met-window end
their schedule allows. Sampling uses the model cell containing each point and
happens at met-window ends, where convection and chemistry have been applied.
Every choice below maps to a Julia type; the editor schema
(`schemas/atmos_transport_run.schema.json`, used by Taplo / Even Better TOML)
shows that type when you hover over a value.

```toml
[output.observations]
enabled = true
path = "~/data/AtmosTransport/output/obs_{YYYYMMDD}.nc"   # -> obs_<date>_soundings.nc, obs_<date>_sites.nc
time_interpolation = "linear"            # LinearWindowInterpolation | "nearest_window" (NearestWindowSampling)
tracers = ["co2_natural", "co2_fossil"]  # omit for all tracers; must exist in [tracers]
write_profile_for_sites = true           # also write full station profiles (e.g. TCCON sites)
layer_height_temperature_kelvin = 280.0  # ConstantLayerTemperature for intake heights
# start_time = "2021-12-02T00:00:00"     # required only when [input].start_date is absent
deflate_level = 0
dem_file = "~/data/AtmosTransport/static/dem_1min.nc"  # true ground of altitude-only records (LatLonPointElevation)
dem_variable = "z"
unplaced_records = "error"               # ErrorOnUnplaced | "skip" (SkipUnplaced)
# surface_height_file = "~/data/AtmosTransport/static/surface_height_c90_merra2.nc"  # model ground (ModelSurfaceHeight)
# absolute_altitude_above_agl_m = 3000.0 # AbsoluteAltitudeAbove (needs surface_height_file); omitted: AboveGroundPlacement
vertical_coordinates = ["height_above_ground", "pressure"]  # preference order (default); HeightAboveGroundCoordinate, PressureCoordinate
pressure_basis = "total"                 # TotalFromHumidity (default) | "dry" (DryInterfaces, for binaries without humidity)

[[output.observations.sources]]
kind = "oco2_lite"                       # OCO2LiteSource: Lite files or OCO-2 v11 MIP 10-s averages
path = "/kiwi-data/Data/model/OCO2MIP/observation_input/OCO2_b11.2_10sec_GOOD_r2.nc4"
quality_filter = "none"                  # NoQualityFilter; default "flag_max" = QualityFlagFilter("xco2_quality_flag", 0)
# averaging_kernel_tracers = ["co2"]     # opt-in: write co2_ak_column with each sounding's averaging kernel
# averaging_kernel_mapping = "hat"       # "hat" (the 20 levels, default) | "tm5" (TM5's 19 layers)

[[output.observations.sources]]
kind = "obspack"                         # ObsPackSource
mode = "soundings"                       # SoundingMode: every record at its own time and intake height
path = "~/data/obspack/data/nc/co2_*.nc"

[[output.observations.sources]]
kind = "table"                           # TableSource: your own CSV / TOML / NetCDF list
mode = "sites"                           # SiteMode: station series per site schedule
path = "~/data/AtmosTransport/observations/tccon_sites.csv"
```

| Key | Choices → type |
|---|---|
| `kind` | `"oco2_lite"` → `OCO2LiteSource`, `"obspack"` → `ObsPackSource`, `"table"` → `TableSource` |
| `mode` | `"soundings"` → `SoundingMode` (point events), `"sites"` → `SiteMode` (station series); required for `obspack` and `table` |
| `quality_filter` | `oco2_lite` only: `"flag_max"` (default) → `QualityFlagFilter(quality_variable, quality_flag_max)` keeps flags `<= max`; `"flag_values"` → `QualityFlagValues(quality_variable, quality_flag_values)` keeps listed flags (MIP `assimilate_flag`: 0 not assimilated, 1 assimilated, 2 withheld, so `[1]` selects the assimilated set); `"none"` → `NoQualityFilter`. Keys of the other filters are rejected. |
| `averaging_kernel_tracers` | `oco2_lite` only, opt-in: array of sampled tracer names → `AveragingKernelColumns`. The reader also reads each sounding's retrieval (`pressure_levels`, or `sigma_levels` × `psurf`; `pressure_weight`, `xco2_averaging_kernel`, `co2_profile_apriori`) and the soundings file gets `<tracer>_ak_column`, the averaging-kernel column of the sampled profile with the 4D-Var's operator, on the interface pressures of `pressure_basis` (total pressures need the binaries' humidity). Name the tracer the retrieval measures (total CO2): the prior term belongs to it. Omitted → `NoAveragingKernels`, output unchanged |
| `averaging_kernel_mapping` | with `averaging_kernel_tracers` only: `"hat"` (default; the 20 levels as given, `HatLevelProjection`) or `"tm5"` (TM5-4DVar's 19 layers, `tm5_layer_column`) |
| `split` (in `[output]`) | `"single"` → `SingleOutputFile`, `"daily"` → `DailyOutputFiles`; observation files follow it |
| `site_grouping` | `obspack` sites only: `"site_code"` → `SiteCodeGrouping`, `"location"` → `LocationGrouping` |
| `format` | `table` only: `"auto"`, `"csv"`, `"toml"`, `"netcdf"` → `AutoTableFormat`, `CSVTableFormat`, `TOMLTableFormat`, `NetCDFTableFormat` |
| `time_interpolation` | `"linear"` → `LinearWindowInterpolation`, `"nearest_window"` → `NearestWindowSampling` |
| `surface_height_file` | per-grid model surface height (m above sea level) → `ModelSurfaceHeight`, read by `GriddedIO.read_model_surface_height`; omitted → `NoModelSurfaceHeight`. Reference of `absolute_altitude_above_agl_m`, and written per row; never used as the true ground |
| `dem_file`, `dem_variable` | lat-lon DEM (ground or sea surface, m above sea level) → `GriddedIO.LatLonPointElevation`; omitted → `NoPointElevation`. `dem_variable` (default `"z"`) only with `dem_file` |
| `unplaced_records` | `"error"` (default) → `ErrorOnUnplaced`, `"skip"` → `SkipUnplaced` |
| `absolute_altitude_above_agl_m` | number ≥ 0 → `AbsoluteAltitudeAbove(h₀)`; omitted → `AboveGroundPlacement`. Needs `surface_height_file` and `"height_above_ground"` in `vertical_coordinates` |
| `vertical_coordinates` | non-empty array, in preference order, of `"height_above_ground"` → `HeightAboveGroundCoordinate` and `"pressure"` → `PressureCoordinate`, each at most once; default `["height_above_ground", "pressure"]`. Without `"height_above_ground"`, `dem_file` and `absolute_altitude_above_agl_m` are rejected |
| `pressure_basis` | `"total"` (default) → `TotalFromHumidity`, `"dry"` → `DryInterfaces`: the interface pressures a pressure-placed record is compared with (below); only with `"pressure"` in `vertical_coordinates` |

Source paths are templates: `{YYYYMMDD}` (or `{date}`), `{YYMMDD}`, `{YYYY}`,
`{MM}`, `{DD}` are substituted per run day, and `*` / `?` wildcards expand in
the file name. A path without date tokens must match at least one file.

**Station tables.** A `table` source with `mode = "sites"` reads `id`, `lat`,
`lon`, optional `elevation` (m asl) and `intake_height` (m above ground) or
`altitude` (m asl, converted with `elevation`, else as described under
"Vertical placement" below). Each row's schedule follows from its keys:

| Row keys | Schedule | Output |
|---|---|---|
| none | `EveryWindow` | every met-window end, `_sites` file |
| `start_time`, `end_time` | `TimeRange` | window ends inside the range; NaN outside |
| `times` | `TimeList` | one point event per listed UTC time, `_soundings` file |

In CSV, list several times in one `times` cell separated by `;`. Times are
ISO-8601 UTC strings or TOML date-times (no offset, or `Z`); in NetCDF tables
`time`, `start_time`, and `end_time` are decoded from their CF units, and bare
numbers are rejected elsewhere. A site id that repeats (across rows, daily
files, or sources) must keep its location; its time lists are merged, and any
other schedules must agree. Unknown CSV/TOML columns are an error, so a
misspelt `intake_height` cannot silently select the lowest layer.
ObsPack station series (`mode = "sites"`) are for fixed stations: one site
per distinct height in a dataset (`intake_height`, else the altitude of rows
that give only an altitude, rounded to 0.1 m), at the site position. With
`site_grouping = "site_code"` a dataset whose records stray more than 1° from
that position (aircraft, ship) is reported as a moving platform; sample those
with `mode = "soundings"`. TOML site
tables can declare `#:schema .../schemas/observation_sites.schema.json` for
editor help; `config/examples/observation_sites_demo.toml` shows all three
schedules. Point-event tables (`mode = "soundings"`) read `id`, `time`, `lat`,
`lon` and the same optional height columns.

**Vertical placement.** Every request is placed in the model column once,
when the sampler is built, and the layer is chosen at every sample from that
sample's column. The column's interface pressures come from the air mass,
``p_{1/2} = p_\mathrm{top}``, ``p_{k+1/2} = p_{k-1/2} + g\,m_k/A`` (dry
pressures on a dry-basis run; these are the pressures written to the files),
and its interface heights above the model surface from the hypsometric
equation,

```math
z_{N+1/2} = 0, \qquad
z_{k-1/2} = z_{k+1/2} + \frac{R_d T_k}{g} \ln\frac{p_{k+1/2}}{p_{k-1/2}},
```

with ``T_k`` the GCHP VDIFF layer temperature on cubed-sphere binaries that
carry it, else `layer_height_temperature_kelvin`. A record at height ``h``
above the ground is sampled in the layer ``k`` with
``z_{k+1/2} \le h < z_{k-1/2}`` (the lowest layer for ``h \le 0``, the top
layer above the column). A record is placed by the first of the
`vertical_coordinates` it gives, in their order: `"height_above_ground"`
(codes 0 and 1 below) and `"pressure"` (code 2). The default,
`["height_above_ground", "pressure"]`, places a record by its height above
the ground when it has one and by its pressure otherwise;
`["pressure", "height_above_ground"]` prefers the pressure; a single
coordinate uses only that one. The rule that placed the record is written per
row as `vertical_placement`:

| Code | `vertical_placement` | Layer |
|---|---|---|
| 0 | `record_height` | ``h`` = the record's `intake_height`, else `altitude − elevation` (row by row; for ObsPack station series the station's `site_elevation` when the rows have none) |
| 1 | `altitude_minus_dem` | ``h`` = `altitude` − DEM ground elevation at the point (`dem_file`, bilinear) |
| 2 | `pressure` | the record's ObsPack `pressure` ``p``: the layer with ``p_{k-1/2} \le p < p_{k+1/2}`` of the `pressure_basis` interfaces (below) |
| 3 | `absolute_altitude` | only with `absolute_altitude_above_agl_m` (below) |
| 4 | `lowest_layer` | a record with no vertical coordinate at all, from a source whose records are column or surface products (`missing_height_rule`: OCO-2 soundings, table rows without height columns) |

Placing by height above the true ground keeps a record in the same part of
the model column as the real sample. The model's interface heights are
measured from the model surface ``z_s`` (the cell mean of the orography), so
the record sits at ``z_s + h`` and its layer does not depend on ``z_s``: a
mountain station (Mauna Loa, ground 3397 m, intake 40 m) whose cell has a
model surface near 2000 m stays in the model's lowest layer instead of being
sampled 1.4 km above the model surface, and an aircraft 500 m above a valley
floor is sampled 500 m above the model ground. The true ground is the
record's own `elevation`, else the DEM; the model surface height is **never**
used in its place (that would place the record by its absolute altitude, the
error this rule avoids).

A record that none of the coordinates place cannot be placed: one with an
`altitude` but no ground elevation (no `elevation`, no `dem_file` value at the
point) and no `pressure`, one that gives only a coordinate the run does not
use (an `intake_height` with `vertical_coordinates = ["pressure"]`), or an
ObsPack record without any vertical coordinate. With
`unplaced_records = "error"` (default) the sampler refuses
to build and names the first records; with `"skip"` they are dropped and
counted in the `n_unplaced_soundings` / `n_unplaced_sites` file attributes.
They are never sampled in the lowest layer. Undeclared fill values do not
count as coordinates: an ObsPack `pressure` at or below 0, an `elevation`
(or `site_elevation`) below −450 m and a negative `intake_height` are read
as absent, with one warning per file (the JMA aircraft-flask rows of the MIP
ObsPack give `elevation` = `intake_height` = −1000 at 5 km altitude; they are
placed by `altitude −` DEM). A record placed more than 100 m below the ground
(a wrong elevation, DEM value or altitude) is sampled in the lowest layer and
reported in a warning with the first records; its height is in
`placement_height_agl`. In the OCO-2 MIP v5.0.1 daily
ObsPack, 16.5% of the 2021-12 to 2023-12 records (97% of the aircraft-insitu
records, all AirCore records) have fill `intake_height` and `elevation` and the
files have no `pressure`, so sampling them needs a `dem_file`.

`absolute_altitude_above_agl_m = h₀` re-places records at least ``h_0`` above
the true ground (free-tropospheric aircraft and AirCore samples, whose air
does not follow the terrain) by their ObsPack `pressure` when they have one
and `"pressure"` is one of the `vertical_coordinates` (code 2), else by their
altitude ``z_\mathrm{obs}`` above sea level, as the
layer holding ``z_\mathrm{obs} - z_s`` (code 3, `absolute_altitude`; needs
`surface_height_file`). ``z_\mathrm{obs}`` is the record's `altitude`, else
its ground elevation (the record's, else the DEM value at the point) plus its
`intake_height`; a record with neither keeps its height above the ground
(code 0). Records below ``h_0`` keep their height above the
ground. Whether and where to blend from terrain-following to absolute
placement aloft is still open; the default places every record by its height
above the ground. With ``h_0 = 0`` every record that
carries an altitude is placed by altitude above sea level (or pressure), as
the GEOS-Chem ObsPack diagnostic places every record
(`ObsPack/obspack_mod.F90`: the level where ``\Phi_s/g + \sum_l \Delta z_l``
first reaches the ObsPack altitude). ObsPack `pressure` must declare its
units (`Pa`, `hPa`, `mbar`, `kPa` and their spelled-out forms, any case);
a missing or other unit is an error.

**Pressure basis.** A record's pressure is the total (ambient) pressure of
dry air and water vapour. The air mass of a dry-basis run gives the dry
interface pressures above, the weight of the dry air above each interface;
the total pressure there is larger by the weight of the water vapour above
it (4.9 hPa at the surface for 50 kg m⁻² of precipitable water, 1 hPa for
10 kg m⁻²). With `pressure_basis = "total"` (default, `TotalFromHumidity`) a
pressure-placed record is compared with the total interface pressures. A
layer with dry air mass ``m_k`` and specific humidity ``q_k`` (kg of vapour
per kg of moist air) holds ``m_k/(1 - q_k)`` of moist air, so hydrostatically

```math
p_{1/2} = p_\mathrm{top}, \qquad
p_{k+1/2} = p_{k-1/2} + \frac{g\,m_k}{A\,(1 - q_k)},
```

with ``p_\mathrm{top}`` = `A_ifc[1]`, the pure-pressure top interface of the
met model's hybrid coordinate (a total pressure). This inverts the
preprocessor's dry conversion ``Δp_\mathrm{dry} = (1 - q)\,Δp``. On merged
or remapped levels the binary carries a dry-mass-weighted ``q``, so the
inverse is not exact there: on the golden `pre_ll72` binary (ERA5 L137 merged
to 34 levels) the rebuilt total surface pressure differs from the stored one
by −0.012 Pa on average and 0.125 Pa at most, where the dry interfaces are
169 Pa low on average and 700 Pa at worst.

The humidity is the binary's layer-mean specific humidity at the sampled
boundary:

- lat-lon binaries preprocessed with `output.include_qv = true`: `qv_start`
  at the run start and `qv_end` at each window end;
- cubed-sphere binaries with the GCHP VDIFF payload (`include_vdiff_fields`):
  VDIFF `qv` is a window-start value, so a window end takes the `qv` of the
  next window (window 1 of the next binary after a binary's last window),
  and the run start the first window's. The run's last window end has no
  next window and takes the ended window's `qv`, one window old: on the
  golden MERRA-2 C90 L72 binary the interfaces rebuilt with it are up to
  23 Pa (99th percentile 11 Pa) from the met model's total interfaces
  ``A + B\,p_s``, against at most 0.23 Pa with the next window's `qv`;
- reduced-Gaussian binaries (the preprocessor refuses `include_qv`) and
  cubed-sphere binaries without the VDIFF payload (for example the ERA5
  N320 → cubed-sphere products) carry none.

With linear time interpolation the humidity is blended between the two
window ends like the air mass. A moist-basis run's air mass already gives
total pressures. A dry-basis run whose binaries do not all carry humidity
(every binary's header is checked before transport) cannot build them:
building the sampler then fails if any record is placed by pressure, and
`pressure_basis = "dry"` (`DryInterfaces`) compares with the dry interfaces
instead. Those lie at lower pressure, so a record can land one or more layers
too low (closer to the surface, where the layers are thinnest). On the GEOS
L72 levels with ``p_s`` = 1010 hPa and
``q = q_s\,(p_\mathrm{mid}/p_s)^\lambda``, ``\lambda`` set by the
precipitable water, the dry interfaces put a pressure uniformly spaced
between 900 and 1010 hPa in a lower layer 25% of the time, between 500 and
900 hPa 6% and between 200 and 500 hPa 0.4% for a tropical column
(``q_s`` = 18 g/kg, 50 kg m⁻²), and 5%, 1% and 0% for ``q_s`` = 4 g/kg and
10 kg m⁻²; an independent construction of the tropical case gave 29%, 6.5%
and 0.36%. The files keep the dry interfaces as `p_half_dry`; the attribute
`pressure_basis` records the basis used.

**External ground datasets.** Both are static and external, never part of a
transport binary, and read through `GriddedIO`:

- `dem_file`: a lat-lon NetCDF file with 1-D longitude/latitude centre
  coordinates and a 2-D elevation variable (`dem_variable`, `units = "m"`)
  giving the **ground or sea surface** in metres above sea level. Over the
  ocean the ground is the sea surface (0 m): clip a bathymetric product at
  0 m over the sea. A file going below −450 m (deeper than the Dead Sea shore)
  is rejected as bathymetry or an undeclared fill value, one going above
  9000 m as an undeclared fill value. The variable may be stored on
  `(lon, lat)` or `(lat, lon)` (told apart by dimension name); a
  gridline-registered global grid (−180…180 inclusive) has its repeated seam
  column dropped. Fill values become
  "no ground" (the record is then unplaced or pressure-placed). It is read
  whole as Float32 (about 0.9 GB for a global 1-arc-minute grid), used while
  the sampler is built, and released afterwards. A suitable global choice is
  NOAA ETOPO 2022 at 60 arc-seconds, *surface* variant (ice-sheet surface, not
  the `bed` variant; `ETOPO_2022_v1_60s_N90W180_surface.nc` from NOAA NCEI),
  clipped at 0 m (`z = max(z, 0)`), which also sets land below sea level
  (Dead Sea, Caspian shore) to 0 m. On 34 days of the 2022 MIP ObsPack it
  places every aircraft and AirCore record; the `bed` variant puts the
  Greenland and Antarctic ground up to 3 km too low (Summit 236 m instead of
  3203 m).
- `surface_height_file`: the per-grid model surface height, written by
  `GriddedIO.write_model_surface_height` and checked against the run's mesh
  (cell centres, cubed-sphere areas, finite values, metres) when read.
  `scripts/preprocessing/build_model_surface_height.jl --binary <a binary of
  the grid> --source <surface geopotential file> --variable PHIS --out <file>`
  divides the met source's surface geopotential (MERRA-2/GEOS `PHIS`, ERA5
  `z`) by the standard gravity and regrids it conservatively onto the binary's
  mesh; use the met source the binary was built from. For C90 from MERRA-2
  `PHIS` the result matches GEOS-Chem's own C90 `Met_PHIS` diagnostic to
  0.03 m RMS (0.5 m at most). It is only needed for
  `absolute_altitude_above_agl_m`; otherwise it is written per row as
  `model_surface_height` for diagnostics.

Each output row records the placement, the height used
(`placement_height_agl`), the model surface height, the chosen layer, its
bottom and top heights, and the model surface pressure.

**Time and files.** The run origin is the start of window 1 of the first
binary: `[input].start_date` at 00:00 UTC, or `start_time` with an explicit
`binary_paths` list; `validate_config` enforces it and the run refuses an
origin on another day than the first binary's date label. Only point events
inside the transported span are sampled (an event exactly at the final window
end is kept, so two chained runs sharing a boundary both write it). The file
partition follows `[output].split`; daily files use the day index when a
binary name carries no date, and two binaries resolving to the same daily file
(snapshots or observations) fail before transport starts. A `_soundings` file exists only when there are
point events and a `_sites` file only when there are station series.
Observation rows are queued and written whenever the shared NetCDF lock is
free, so a background daily snapshot write never stalls the run. All sources
are read at startup, keeping only records inside the run span; reading the
whole-mission MIP file takes a few seconds and about 2 GB of transient memory. Averaging kernels are applied offline.
See [Output schema](@ref) for the variable tables.

Observation sampling adds one small gather per window and does not change
transport: the transported state and gridded snapshot output are unchanged
with or without it (tested on lat-lon runs; on lat-lon and reduced-Gaussian
runs it makes the runner step window by window, as snapshots do).

### `[logging]` — how much the run reports

```toml
[logging]
level = "info"        # "warn" (concise) | "info" | "debug" (verbose) | "error";
                      # omitted: the current logger's level (info on the command line)
progress = "bar"      # "bar" (default) | "lines" | "none"
progress_interval_windows = 24   # with progress = "lines": one line per 24 windows
```

`level` is the minimum level of the run's log messages, validation included.
`"warn"` keeps warnings and errors only (the end-of-run timing summary is an
info message); `"debug"` adds the debug messages. The run keeps the current
logger and only sets its level; without `level` the logger is left as it is
(info for the default console logger). `progress = "bar"` redraws a progress bar with status and
timing lines, for interactive terminals. `"lines"` writes one log line every
`progress_interval_windows` met windows and at the end (window count, elapsed
time, ETA, the current status and the time split), for batch jobs whose log
files would otherwise fill with bar redraws; lat-lon and reduced-Gaussian runs
then step window by window. `"none"` writes no progress.
The table is optional; the defaults reproduce the earlier output.

### `[checks]` — runtime checks of the model state

```toml
[checks]
nan_check = true                 # default
nan_check_interval_windows = 1   # default: after every met window
```

With `nan_check` (on by default), the run stops with an error naming the
field, the met window, the iteration and the model time as soon as the air
mass or a tracer holds a NaN or Inf (interior cells). The check runs
reductions on the run's backend without copying fields to the host: two per
check on lat-lon and reduced-Gaussian grids, one per panel for the air mass
and for the tracers on the cubed sphere (about 0.1 ms each on an L40S). The
window count runs over the whole run, across input files.
`nan_check_interval_windows` checks after every n-th window instead. In Julia
the check is `Callback(NaNChecker(), WindowInterval(n))`, passed to
`DrivenSimulation(...; callbacks)` with any other callbacks (see
[Callbacks and runtime checks](@ref)).

### Multi-threaded execution

```bash
julia --threads=2 --project=. scripts/run_transport.jl <cfg.toml>
```

Some preprocessing kernels (spectral synthesis, regridding) and
some host-side workspace operations parallelize across threads.
GPU runs also prefetch the next meteorological window when ≥2 Julia threads
are available, unless `[run] prefetch_windows = false`. Startup reads the first window once and creates two independent
device buffers, so loading the next window cannot overwrite active forcing.
Daily snapshot writes run on an owned background task and overlap the next
day's transport. The runner drains both tasks before closing their resources,
including when stepping fails. Single-file NetCDF appends each snapshot without
retaining earlier frames.

For multi-day or multi-year runs over NFS, opt into rolling local-NVMe input
staging via `[input.staging]` (above). There is no `[buffering]` TOML block.
Give GPU runs `--threads=2` (or more) to enable prefetch.

## Preprocessing config (`config/preprocessing/*.toml`)

Consumed by `scripts/preprocessing/preprocess_transport_binary.jl`, which
calls the unified `process_day` preprocessing entry point. Source-specific
`[preprocessing]` keys live in the met-source descriptor
(`config/met_sources/*.toml`); the flux-construction keys of the MERRA-2 and
ERA5 N320 sources (`column_balance_weights`, `face_fluxes`, `face_lengths`,
`face_interpolation`, `wind_regrid`, and `flux_thickness`) and the
`column_balance_weights` of the GEOS sources are described in
[Data sources](data_sources.md).

The preprocessing config has a different shape from the run config:
the **target topology IS specified here** because that's the act of
producing a binary for that topology.

### `[input]` (spectral source) or `[source]` (native source)

The preprocessor source-axis dispatch reads either:

```toml
# ERA5 spectral source
[input]
spectral_dir = "~/data/AtmosTransport/met/era5/0.5x0.5/spectral_hourly"
thermo_dir   = "~/data/AtmosTransport/met/era5/0.5x0.5/physics"
coefficients = "config/era5_L137_coefficients.toml"
```

```toml
# GEOS-IT native source
[source]
toml     = "config/met_sources/geosit.toml"     # source descriptor
root_dir = "~/data/AtmosTransport/met/geosit/C180/raw_catrine"
```

For the GEOS path, `[source].toml` points to the **source
descriptor** (a separate TOML in `config/met_sources/`) that
declares collection mappings, `mass_flux_dt_seconds`, and
`level_orientation`. See [GEOS native cubed-sphere](@ref) for the
descriptor schema details.

A relative `[source].toml` is relative to the AtmosTransport package root. A
relative hybrid-coefficient table (`coefficients`, `coefficients_file`,
`target_coefficients`) is looked up in the working directory first and then in
the package root, where `config/era5_L137_coefficients.toml` and
`config/geos_L72_coefficients.toml` ship, so the shipped configs work from any
working directory; a table found in neither place is an error that names both
locations.

### `[output]`

```toml
[output]
directory  = "~/data/AtmosTransport/met/.../preprocessed/"
mass_basis = "dry"             # default: "dry"; "moist" supported but not recommended for the runtime
include_qv = false             # LL spectral path only — writes paired :qv_start/:qv_end endpoints.
                               # Native GEOS and CS/RG writers ignore this key today.
```

### `[grid]` — target topology

```toml
# Lat-lon
[grid]
type   = "latlon"
nlon   = 144
nlat   = 73
echlevs              = "ml137_tropo34"
level_top            = 1
level_bot            = 137
merge_min_thickness_Pa = 1000.0

# Cubed sphere
[grid]
type                = "cubed_sphere"
Nc                  = 180
panel_convention    = "geos_native"             # or "gnomonic"
definition          = "gmao_equal_distance"                    # optional; inferred from convention if omitted
regridder_cache_dir = "~/.cache/AtmosTransport/cr_regridding"

# Reduced Gaussian (synthetic — picks a standard ECMWF reduced-Gaussian grid)
[grid]
type            = "synthetic_reduced_gaussian"
gaussian_number = 90
nlon_mode       = "octahedral"                  # ECMWF O-grid distribution
```

### `[vertical]`

```toml
[vertical]
coefficients = "config/geos_L72_coefficients.toml"
```

Per-source defaults are baked into the source-descriptor TOML; this
key is the per-run override.

For native ERA5 N320, named interface-selection presets use:

```toml
[vertical]
coefficients_file = "config/era5_L137_coefficients.toml"
transform = "level_selection"
preset = "ml137_66L"
```

#### [ERA5 vertical level presets](@id era5-vertical-level-presets)

The same preset names work for `[vertical] preset` (native ERA5 N320) and
for `[grid] echlevs` (spectral ERA5). Each preset lists the ERA5 L137
interfaces (half levels) that are kept; the native layers between two kept
interfaces are summed into one layer. An unknown name is an error that lists
the available presets.

The indices follow TM5 and ECMWF: `e = 0` is the model top and `e = 137` is
the surface. For native hybrid coefficients ``A_e, B_e`` and surface pressure
``p_s``, selecting interface ``e`` retains

```math
p_{e+1/2} = A_e + B_e p_s.
```

This is TM5's `at(l) = a_ec(echlevs(l-1))` convention. Older
AtmosTransport versions instead selected coefficient `137 - e`, reflecting
the grid about the middle of the coefficient array.

| Preset | Layers | Selection |
|---|---|---|
| `ml137_tropo34a` | 34 | TM5 cy3-4dvar `levels/ml137/tropo34a`, the CAMS73 34-layer selection |
| `ml137_tropo34` | 34 | TM5-MP r1112 `levels/ml137/tropo34` (an older TM5 selection) |
| `ml137_66L` | 66 | the selection of the ERA5 N320 → C90 configs |
| `ml137_cfl94` | 94 | native layers 1-24 (from the top) in groups of 12, 25-53 in groups of 4, native below |
| `ml137_cfl85` | 85 | as `ml137_cfl94`, with layers 54-71 in pairs |
| `ml137_full` | 137 | all native layers |

The AtmosTransport-native `ml137_66L`, `ml137_cfl94`, and `ml137_cfl85`
recipes use transformed indices so this numbering correction leaves their
grids bit-identical. The development aliases `ml137_94L` and `ml137_85L`
were removed; use the `cfl` names shown above.

The two `tropo34` presets are different grids. They share only the
surface, the model top and the interface above the lowest layer. At a
surface pressure of 1013.25 hPa, `ml137_tropo34a` puts interface 10
(counted from the surface) at 558.3 hPa and interface 20 at 211.6 hPa;
`ml137_tropo34` puts them at 459.6 hPa and 112.1 hPa. Below 200 hPa,
`ml137_tropo34a` has 20 layers and `ml137_tropo34` has 15. Use the preset
whose named TM5 source is the intended comparison.

### `[numerics]`

The numerics block has **different keys** on the spectral and native
paths:

```toml
# ERA5 spectral preprocessing
[numerics]
float_type   = "Float32"     # "Float32" or "Float64"
dt           = 900.0         # advection sub-step (s)
met_interval = 3600.0        # window cadence (s); 1 hour for ERA5
cs_balance_tol = 1e-14       # CS Poisson balance tolerance
cs_balance_project_every = 50 # CS PCG mean-zero projection cadence; 1 = every iteration
face_fluxes = "vector"        # panel_average, vector, or line_integral
face_lengths = "edge"         # edge or cell_centerline (panel_average only)
face_interpolation = "fv3"    # linear, cubic, or fv3 (vector only)
wind_regrid = "cartesian"     # cartesian or scalar
flux_time_sampling = "window_mean" # window_mean or window_start
```

The five flux keys apply to spectral cubed-sphere targets. They share the
typed implementations used by native MERRA-2 and ERA5 N320 preprocessing;
lat-lon and reduced-Gaussian spectral targets do not construct cube faces.

```toml
# Native (GEOS-IT) preprocessing
[numerics]
float_type     = "Float32"
dt_met_seconds = 3600.0      # window cadence (s); 1 hour for GEOS-IT
```

`mass_flux_dt` for the GEOS path lives in the **source descriptor's**
`[preprocessing].mass_flux_dt_seconds` (`config/met_sources/geosit.toml`,
default `450.0` — the FV3 dynamics step); there is **no per-run
`[numerics].mass_flux_dt` override** today.

#### `write_replay_check` — write-time replay gate

Every writer replays each window's stored fluxes and refuses to keep a binary
whose air mass does not reach the next window's (the write-time gate in
[Binary format](@ref)). `write_replay_check = false` skips that gate for
diagnostic binaries; the binary records `write_replay_check = false` in its
header, the inspector marks it, and the runtime warns when it opens it. The
positivity gate still runs. Default `true`.

#### `column_balance_weights` — horizontal mass-flux balance

Every path that applies a Poisson balance closes a window's horizontal mass
fluxes against its endpoint mass tendency before diagnosing `cm`, with one
solve for the column-integrated divergence; the correction is distributed over
the layers by `column_balance_weights` (`"hybrid_mass"` by default on every
target;
[Data sources](data_sources.md) lists where each path sets it), so layer winds
keep their vertical structure and `cm` carries the vertical motion. GEOS
`pressure_fixer` and `pfix_corrected` keep the native horizontal fluxes and
record `horizontal_balance = "none"`. ERA5 spectral configs set the weights
here, for lat-lon, reduced-Gaussian and cubed-sphere targets alike:

```toml
[numerics]
column_balance_weights = "hybrid_mass"    # or "hybrid_b", "mass"
```

Binary headers record the balance as `horizontal_balance = "column"` (GEOS:
the balance its `geos_cm_closure` applies, `"none"` for the pressure-fixer
closures). The per-layer balance and its key `balance_mode` were removed: with
no independent vertical flux, balancing each layer to its own mass change left
almost no vertical mass flux ([Vertical transport](../theory/vertical_transport.md)
section 2). A config that sets `balance_mode` is an error. Older binaries
balanced per layer record `horizontal_balance = "per_layer"`; every
reduced-Gaussian binary written before this change is one of them.

#### `geos_cm_closure` — GEOS native CS vertical-flux closure

How the vertical mass flux `cm` is diagnosed when regridding GEOS native fields
to cubed-sphere:

```toml
[numerics]
geos_cm_closure = "endpoint"   # default; "omega_regularized" is experimental
```

- `"endpoint"` (default) — diagnose `cm` from the endpoint-`DELP` mass tendency.
  Closes the explicit-`dm` continuity gate exactly, but injects the intrinsic
  native MFXC↔DELP residual as grid-scale noise that shows up as SH-UTLS
  "fingering" in adv-only tracers.
- `"omega_regularized"` — retain endpoint-balanced transport at resolved scales
  and use A3dyn `OMEGA` only for a pressure-local, horizontal-high-pass
  correction. The default taper is 50--350 hPa and each level's horizontal
  flux correction is capped at 10% RMS. This avoids the local-enhancement loss
  seen when OMEGA replaces the full vertical-convergence field, but remains an
  experimental native-cube pathway pending multi-day tracer validation.
- `"omega_full_replacement"` — diagnostic all-level/all-scale OMEGA replacement. It
  closes mass exactly and reduces the SH-UTLS roughness metric, but suppresses
  physically resolved local XCO2 enhancements; do not use it for production.
  The historical `"omega"` and `"omega_consistent"` aliases now resolve to the
  safe regularized mode.
- `:pressure_fixer`, `:moisture_filtered`, `:pfix_corrected` are diagnostic-only
  (they fail the replay gate or drift `ps`) and are warned at use.

The regularized controls are explicit and recorded in the binary header:

```toml
[numerics.omega_regularization]
pressure_taper_hpa = [50.0, 80.0, 300.0, 350.0]
smoothing_steps = 3
smoothing_fraction = 0.10
max_relative_flux_correction = 0.10
max_bottom_flux_correction = 0.01
```

The RMS cap limits broad circulation changes. The bottom-layer gate aborts
preprocessing if any of the lowest three levels changes by more than 1% RMS;
this protects fresh surface emissions from artificial OMEGA-driven PBL export.
The largest local edge increment is also reported as a diagnostic, using
`max(abs(native_face_flux), level_rms_native_flux)` as its denominator.
OMEGA modes also require `[mass_fix].enable = true`; without global endpoint
mass closure, a per-level horizontal Poisson solve cannot realize the global
column tendency.

### `[logging]` — preprocessing log level

A preprocessing config may set `[logging] level` like a run config: `"warn"`
keeps the per-window log of the writers quiet, `"debug"` adds debug messages.
Without it the current logger's level applies (info for the command-line
scripts).

### `[mass_fix]` — global dry-mass pin

ERA5 spectral path:

```toml
[mass_fix]
enable                = true
qv_global_climatology = 0.00247
```

pins the global-mean surface pressure so that the dry air mass stays at
`Parameters.GLOBAL_MEAN_DRY_SURFACE_PRESSURE` (98,325 Pa). LL spectral runs
without it drift by tens of Pa per window. The target is one package constant,
not a config choice; `target_ps_dry_pa` and `target_total_kg` are rejected as
stale keys.

Native sources (GEOS, MERRA-2 and ERA5 N320 to the cubed sphere):

```toml
[mass_fix]
enable           = true
mode             = "target_ps_dry"  # or "initial_endpoint" (GEOS only)
```

`target_ps_dry` (the default) pins the global dry air mass of every window
endpoint to `Parameters.GLOBAL_MEAN_DRY_SURFACE_PRESSURE` times the binary
header sphere's area divided by `g`. The native correction is distributed as
`Delta B_k (1-q_k)`, so pure-pressure layers are unchanged.
`initial_endpoint` pins to the dry mass at the start of the first window
instead, carried across the days of one preprocessing run; only the GEOS native
path implements it, and the other sources reject it.

## Where to read next

- [Output schema](@ref) — what the snapshot NetCDF actually contains
  per topology.
- [Data sources](@ref) — ERA5 / GEOS access, credentials, and recommended
  local layout.
- [Run with real meteorology](@ref) — move from the synthetic tutorial to
  preprocessed forcing.
