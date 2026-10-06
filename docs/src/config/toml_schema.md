# TOML schema

This page is the **canonical reference** for the TOML configs that
drive both the runtime (`scripts/run_transport.jl`) and the
preprocessor (`scripts/preprocessing/preprocess_transport_binary.jl`).
Per-block, per-key, with type, default, and what it does.

The two config families (run vs. preprocessing) live in different
directories and are consumed by different code paths; this page
keeps them separate.

## Run config (`config/runs/*.toml`)

Consumed by `scripts/run_transport.jl`, which parses the TOML and forwards
the resulting Julia dictionary to `run_driven_simulation(cfg)`.

The runtime infers the **target topology** from the binary header's
`grid_type` field at load time; a
`[grid]` block in the run config is therefore unnecessary and
ignored.

`validate_config(cfg)` checks runtime table shapes, input path existence,
precision/backend compatibility, and window bounds without opening binary
readers or allocating model state. Nested tracer `init` and `surface_flux`
values must be tables. Shape errors are returned before value checks; a
successful result is not a full physics or binary validation. See
[Run with real meteorology](@ref Run-with-real-meteorology) for an example.

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
files in the directory. Implementation: `src/Models/InputStaging.jl`.

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
`"none"`, `"preserve_vmr"`, or `"preserve_tracer_mass"`. Advection belongs
in the separate `[advection]` table.

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

Initial-condition kinds (declared in `src/Models/InitialConditionIO.jl`):

| Kind | LL | RG | CS | Required keys |
|---|---|---|---|---|
| `"uniform"` | yes | yes | yes | `background` |
| `"latitude_step"` | yes | yes | yes | optional `south_value`, `north_value`, `split_lat_deg` |
| `"bl_enhanced"` | yes | **no** | **no** | `background`, `enhancement`, `n_layers` (LL-only; RG/CS path errors at IC build) |
| `"gaussian_blob"` | yes | yes | yes | `background`, `lon0_deg`, `lat0_deg`, `sigma_lon_deg`, `sigma_lat_deg`, `amplitude` |
| `"file"` / `"netcdf"` | yes | yes | yes | `file`, `variable`, optional `time_index` |
| `"file_field"` | yes | yes | yes | `file`, `variable` |
| `"catrine_co2"` | yes | yes | yes | optional `file`, `variable`, `time_index` overrides for the built-in defaults |
| `"pressure_layer"` | no | no | yes | `lowest_layer = true` or `psurf_fraction`; optional `total_molecules` |
| `"cs_native"` | no | no | yes | `file`, `variable`; optional `time_index`, `vertical_order`, `clamp_negative` (default `false`) |

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
time_index = 12                     # month index (1..12) for monthly inventories
scale      = 1.0                    # optional multiplicative scaling

[tracers.sf6.surface_flux]
kind = "edgar_sf6"
```

Registered surface-flux source kinds (full list in
`src/Models/InitialConditionIO.jl`): `lmdz_co2`, `gridfed_fossil_co2`,
`edgar_sf6`, `zhang_rn222`, plus a generic `file` for arbitrary
NetCDF sources and `cs_native` for time-varying fluxes already on the native
cubed-sphere grid. There is no `edgar_co2` kind — use
`gridfed_fossil_co2` for the GridFED-derived fossil CO₂ inventory.
Known tracer names carry built-in molar masses; for a custom tracer, set
`molar_mass_kg_mol` inside its `surface_flux` table.

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
run-start year when `year` is omitted). For a span that crosses calendar
years, use `files = ["/path/to/month1.nc", "/path/to/month2.nc", ...]` in
chronological order. GridFED supports the same time-varying path; its twelve
monthly totals are converted using the actual number of days in each month.

`temporal_scheme` (default `"stepwise"` for LMDZ and GridFED) is one of:

- `"stepwise"` — hold each slice piecewise-constant until the next sample.
  This matches GEOS-Chem/HEMCO's exact CAMS treatment (verified against
  `EmisCO2_Total`), so use it to reproduce GC.
- `"linear"` — linearly interpolate between adjacent slices.
- `"conservative"` — window-mean each interval (mass-conserving over the
  window, but smears the diurnal cycle).

Slices are indexed by **absolute** time since the run's `start_date`, so a
multi-day run advances through the inventory correctly (a per-day clock would
replay the first day's slices — the cause of the historical co2_natural
+1 Pg/month surplus, now fixed).

For an already aligned GEOS-native cubed-sphere inventory, use
`kind = "cs_native"`, `time_varying = true`, `file`, and `variable`.
The NetCDF variable must have dimensions `(time,nf,Ydim,Xdim)` and contain
mass flux density in kg species m⁻² s⁻¹. Grid resolution and panel order must
match the meteorology. The loader multiplies by native mesh cell areas and
converts to model storage units using the tracer molar mass. It defaults to
`temporal_scheme = "stepwise"`; signed uptake is retained. The TRENDY ensemble
example is `config/runs/trendy_v14_s3_all_models_npp_rh_c30_2014_2024.toml`.

### `[advection]`, `[diffusion]`, `[convection]`, `[chemistry]`

Each operator has a selector and per-kind options. See
[Operators](@ref Operator-concepts) and [Advection schemes](@ref) for what each
selector means; relevant config keys:

```toml
[advection]
scheme    = "linrood"           # "upwind" | "slopes" | "ppm" | "linrood" | "none"
ppm_order = 7                   # cubed-sphere LinRoodPPM only; ∈ {5, 7}.
                                # Setting ppm_order with scheme = "ppm" errors.

[diffusion]
kind  = "constant"              # "none" | "constant" |
                                # "tm5_beljaars_viterbo_local_kz" |
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
                                # layers within each column (a numerical approximation)

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
Redistribution follows each tracer's prior fine-layer profile, using uniform
weights when its old super-layer total is zero. Conservation is to rounding,
not exact arithmetic. The default `n_merge` is 1.

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

[[output.observations.sources]]
kind = "oco2_lite"                       # OCO2LiteSource: Lite files or OCO-2 v11 MIP 10-s averages
path = "/kiwi-data/Data/model/OCO2MIP/observation_input/OCO2_b11.2_10sec_GOOD_r2.nc4"
quality_filter = "none"                  # NoQualityFilter; default "flag_max" = QualityFlagFilter("xco2_quality_flag", 0)

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
| `split` (in `[output]`) | `"single"` → `SingleOutputFile`, `"daily"` → `DailyOutputFiles`; observation files follow it |
| `site_grouping` | `obspack` sites only: `"site_code"` → `SiteCodeGrouping`, `"location"` → `LocationGrouping` |
| `format` | `table` only: `"auto"`, `"csv"`, `"toml"`, `"netcdf"` → `AutoTableFormat`, `CSVTableFormat`, `TOMLTableFormat`, `NetCDFTableFormat` |
| `time_interpolation` | `"linear"` → `LinearWindowInterpolation`, `"nearest_window"` → `NearestWindowSampling` |

Source paths are templates: `{YYYYMMDD}` (or `{date}`), `{YYMMDD}`, `{YYYY}`,
`{MM}`, `{DD}` are substituted per run day, and `*` / `?` wildcards expand in
the file name. A path without date tokens must match at least one file.

**Station tables.** A `table` source with `mode = "sites"` reads `id`, `lat`,
`lon`, optional `elevation` (m asl) and `intake_height` (m above ground) or
`altitude` (m asl, converted with `elevation`). Each row's schedule follows
from its keys:

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
misspelt `intake_height` cannot silently select the lowest layer. TOML site
tables can declare `#:schema .../schemas/observation_sites.schema.json` for
editor help; `config/examples/observation_sites_demo.toml` shows all three
schedules. Point-event tables (`mode = "soundings"`) read `id`, `time`, `lat`,
`lon` and the same optional height columns.

**Intake layer.** The model has no orography, so the intake layer is chosen
by height above the model's own surface, from hypsometric layer heights
(GCHP VDIFF temperatures on cubed-sphere binaries that carry them, else the
constant above). Each output row records the chosen layer, its bottom and top
heights, and the model surface pressure, so mountain sites show their
representativeness gap.

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

### Multi-threaded execution

```bash
julia --threads=2 --project=. scripts/run_transport.jl <cfg.toml>
```

Some preprocessing kernels (spectral synthesis, regridding) and
some host-side workspace operations parallelize across threads.
GPU runs also prefetch the next meteorological window when ≥2 Julia threads
are available. Startup reads the first window once and creates two independent
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
calls the unified `process_day` preprocessing entry point.

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

Available ERA5 presets are `ml137_tropo34`, `ml137_66L`, `ml137_cfl85`
(`ml137_85L`), `ml137_cfl94` (`ml137_94L`), and `ml137_full`.

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
```

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

### `[mass_fix]` — global PS pinning (spectral path only)

```toml
[mass_fix]
enable                = true
target_ps_dry_pa      = 98726.0
qv_global_climatology = 0.00247
```

The GEOS native CS path doesn't apply mass fix (the FV3 dynamical
core's mass flux is already conservative). LL spectral runs without
it drift by tens of Pa per window.

## Where to read next

- [Output schema](@ref) — what the snapshot NetCDF actually contains
  per topology.
- [Data sources](@ref) — ERA5 / GEOS access, credentials, and recommended
  local layout.
- [Run with real meteorology](@ref) — move from the synthetic tutorial to
  preprocessed forcing.
