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

#### `[input.staging]` — rolling NVMe staging (opt-in)

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
| `"cpu"` | CPU only. **Conflicts with `use_gpu = true` and errors at config-load time** (see `src/Architectures.jl:120-121`). |
| `"cuda"` | NVIDIA CUDA via `CUDA.jl` (must be installed). |
| `"metal"` | Apple Silicon Metal via `Metal.jl`. F32 only. |
| `"auto"` | Auto-detects an available GPU backend (CUDA → Metal); errors if none is available. |
| (omitted) | If `backend` is absent, the runtime picks `CPUBackend` when `use_gpu = false` and auto-detects a GPU backend when `use_gpu = true`. |

The CLI eagerly loads a requested CUDA or Metal backend before AtmosTransport.
The package runtime can also load an optional backend on demand without
injecting modules into `Main`.

### `[numerics]` — precision

```toml
[numerics]
float_type = "Float32"        # default: "Float64"; one of "Float32" / "Float64"
```

Float32 is the recommended default for L40S / consumer-GPU
production runs; F64 needs an A100-class card or CPU. Mixing
with the binary's `on_disk_float_type` is allowed (the runtime
casts on load).

### `[run]` — runtime knobs

```toml
[run]
start_window = 1              # default: 1 — first window to process
stop_window  = 24             # default: nothing — uses the binary's full range
air_mass_reset_mode = "preserve_tracer_mass"
```

`start_window` selects the first window for LL/RG runs. The cubed-sphere
runner currently requires `start_window = 1`; other values are rejected.
`stop_window` is the inclusive last window; setting it lets you
run a partial day for smoke tests. Partial window ranges require one input
file; a run spanning multiple files must consume every file's full window
range so it cannot carry state across a gap in the forcing. `air_mass_reset_mode` is one of
`"none"`, `"preserve_vmr"`, or `"preserve_tracer_mass"`. Advection belongs
in `[advection]`; the legacy `[run].scheme` spelling is accepted only when an
`[advection]` table is absent.

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
| `"cs_native"` | no | no | yes | `file`, `variable`; optional `time_index`, `vertical_order` |

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
`edgar_sf6`, `zhang_rn222`, `cs_native`, plus a generic `file` for arbitrary
regular-latitude/longitude NetCDF sources. There is no `edgar_co2` kind — use
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

An already aligned panel-native flux-density series can bypass horizontal
regridding:

```toml
[tracers.co2_sif_gpp.surface_flux]
kind              = "cs_native"
file              = "/path/to/hourly_c180_flux.nc"
variable          = "CO2_FLUX"
molar_mass_kg_mol = 0.0440095
time_varying      = true
temporal_scheme   = "stepwise"
```

The NetCDF variable must use `(time,nf,Ydim,Xdim)` dimensions on disk, match
the transport cube resolution and GEOS-native panel convention, and contain a
per-area, per-second species mass flux. The loader performs no horizontal
interpolation: it multiplies the density by the runtime mesh's exact cell
areas and converts the physical species rate to the dry-air storage basis.

### `[advection]`, `[diffusion]`, `[convection]`, `[chemistry]`

Each operator has a selector and per-kind options. See
[Operators](@ref) and [Advection schemes](@ref) for what each
selector means; relevant config keys:

```toml
[advection]
scheme    = "linrood"           # "upwind" | "slopes" | "ppm" | "linrood" | "none"
ppm_order = 7                   # cubed-sphere LinRoodPPM only; ∈ {5, 7}.
                                # Setting ppm_order with scheme = "ppm" errors.

[diffusion]
kind  = "constant"              # "none" | "constant" |
                                # "tm5_beljaars_viterbo_local_kz" |
                                # "pbl" (legacy alias for the above; CS-only) |
                                # "geoschem_holtslag_boville_vdiff" (CS-only;
                                #   requires include_gchp_vdiff=true binary) |
                                # "precomputed_kz" (CS-only; prefers the exact
                                #   TM5 :dkg interface-exchange payload; legacy
                                #   :kz binaries remain readable — requires a
                                #   binary built with include_tm5_diffusion=true)
value = 1.0                     # m²/s — broadcast Kz when kind="constant"
surface_flux_boundary = false   # true: S(dt)->V(dt); false: V/2->S->V/2

[convection]
kind = "cmfmc"                  # "none" | "cmfmc" | "cmfmc_matrix" | "tm5"
                                # cmfmc_matrix = TM5 LU solver on GEOS CMFMC
                                # rates; tm5 = TM5 entrainment (:entu/:detu/
                                # :entd/:detd payload)

# Collaborative-LU knobs (cmfmc_matrix and tm5). use_collab_lu is REQUIRED for
# lmax_conv / n_merge to take effect — setting them without it is a hard error.
use_collab_lu = true            # batched/collaborative column LU (fast path)
lmax_conv     = 0               # 0 = full column; >0 truncates convection above
                                # that level (cloud-top closure keeps it mass-safe)
n_merge       = 2               # merge n adjacent columns per LU solve; 1 is
                                # bit-exact, 2 is the best-accuracy production
                                # merge (the historical n=2 blow-up was a
                                # clipping bug, since fixed)

# TM5-only — per-topology budget for the column-tile workspace,
# in binary GiB. Default 1.0 (fits production through C720/L137 on
# H100). Set lower on memory-tight GPUs; setting it higher beyond
# the topology's total cells is a no-op.
tile_workspace_gib = 1.0

[chemistry]
kind = "decay"                  # currently only first-order decay
  [chemistry.half_lives_seconds]
  rn222 = 330350.4              # per-tracer half-lives (seconds)
```

The runtime **rejects** at load time any operator selection that
the binary doesn't support (e.g. `convection.kind = "cmfmc"`
against a binary lacking `:cmfmc` payload). See
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
suffix. Legacy `snapshot_file`, `snapshot_hours`, and
`snapshot_interval_hours` are still accepted. See [Output schema](@ref) for
the per-topology variable list the file actually contains.

`format = "binary_mmap"` writes fast self-describing per-day **ATMSNAP** binary
files inline (skipping the NetCDF/HDF5 encode in the GPU run), to be converted
to NetCDF offline on CPU with `scripts/postprocess/binary_to_netcdf.jl`. This
is the throughput path for long multi-day runs. The ATMSNAP payload is **always
Float32 on disk**, including for `float_type = "Float64"` runs — the on-disk
snapshot precision is independent of the compute precision (the F64 benefit is
in the in-run transport; the F64 mass-balance check comes from the runtime
budget log, not the snapshot file).

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

### Multi-threaded execution

```bash
julia --threads=2 --project=. scripts/run_transport.jl <cfg.toml>
```

Some preprocessing kernels (spectral synthesis, regridding) and
some host-side workspace operations parallelize across threads.
The runtime also overlaps I/O with compute when ≥2 threads are
available: each met window is **prefetched** on a background task
while the current window integrates (`_start_window_prefetch!` in
`DrivenSimulation.jl`), and the per-day snapshot write runs on a
spawned task (`pending_write` in `DrivenRunner.jl`) so disk output
overlaps the next day's transport. For multi-day/-year runs over
NFS, opt into rolling local-NVMe input staging via `[input.staging]`
(above). There is no `[buffering]` TOML block — these overlaps are
automatic; give the run `--threads=2` (or more) to enable them.

## Preprocessing config (`config/preprocessing/*.toml`)

Consumed by `scripts/preprocessing/preprocess_transport_binary.jl`, which
calls the unified `process_day` preprocessing entry point.

The preprocessing config has a different shape from the run config:
the **target topology IS specified here** because that's the act of
producing a binary for that topology.

### `[input]` (legacy spectral) or `[source]` (native)

The preprocessor source-axis dispatch reads either:

```toml
# ERA5 spectral (legacy path)
[input]
spectral_dir = "~/data/AtmosTransport/met/era5/0.5x0.5/spectral_hourly"
thermo_dir   = "~/data/AtmosTransport/met/era5/0.5x0.5/physics"
coefficients = "config/era5_L137_coefficients.toml"
```

```toml
# GEOS-IT native (current path)
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
include_qv = false             # LL spectral path only — controls writing the :qv payload.
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
definition          = "gmao"                    # optional; inferred from convention if omitted
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

### `[numerics]`

The numerics block has **different keys** on the spectral and native
paths:

```toml
# Spectral (legacy ERA5) preprocessing
[numerics]
float_type   = "Float32"     # "Float32" or "Float64"
dt           = 900.0         # advection sub-step (s)
met_interval = 3600.0        # window cadence (s); 1 hour for ERA5
cs_balance_tol = 1e-14       # CS Poisson balance tolerance
cs_balance_project_every = 50 # CS PCG mean-zero projection cadence; 1 = legacy
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
- [Data sources](@ref) — ERA5 / GEOS access, credentials, recommended
  local layout, and the quickstart bundle.
- [First run](@ref) — the smallest end-to-end invocation pattern.

Preflight validation rejects unknown top-level tables, misspelled architecture,
numerics and run options, and unknown physics keys or kinds before allocating
model state. For example, `[advection] scheem = "ppm"` fails instead of silently
selecting upwind. Topology and payload checks follow when the binary header is
available; reduced-Gaussian slopes/PPM requests fail during recipe construction.
