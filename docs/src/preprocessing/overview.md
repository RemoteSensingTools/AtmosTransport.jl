# [Preprocessing overview](@id Preprocessing-overview)

The preprocessor turns **raw meteorological input** (ERA5 spectral GRIB,
GEOS-IT / GEOS-FP native NetCDF, …) into the **transport binary**
(`format_version = 4`) the runtime consumes. It runs offline, once per
day per (source, target) combination; the runtime then memory-maps the
result.

!!! tip "Preprocessing is the expensive stage"
    Spectral synthesis, vertical transforms, regridding, mass balancing, and
    replay verification happen once before a run. Runtime then reads a flat,
    self-describing file with fixed-stride windows. Measure a representative
    day on the intended storage and thread count before estimating a campaign.

## One unified driver, three dispatch axes

Every preprocessing path goes through one entry point — the unified
driver `run_unified_preprocessor_day!` in
`src/Preprocessing/transport_binary/driver.jl`. The driver dispatches
on three orthogonal axes, each a typed abstraction:

| Axis | Abstract type | Concrete examples |
| --- | --- | --- |
| **Source** | `AbstractMetReader` | `ERA5SpectralReader`, `GEOSNativeReader` (with chained mass policy) |
| **Vertical transform** | `AbstractVerticalTransform` | `IdentityVertical`, `MergeAbovePressure`, `MergeLayersThinnerThan`, `PressureOverlap` |
| **Target topology** | `AbstractTargetGeometry` | `LatLonTargetGeometry`, `ReducedGaussianTargetGeometry`, `CubedSphereTargetGeometry` |

The driver pairs each concrete source / vertical / target combination
with a typed contract surface from
`src/Preprocessing/transport_binary/window_contracts.jl`:

| Trait | Role |
| --- | --- |
| `AbstractWindowContract{G, FT}` | Per-window mass-balance + positivity verification |
| `AbstractWindowWorkspace{G, FT}` | Pre-allocated scratch for the topology |
| `AbstractBinaryWriter{G, FT, Basis}` | On-disk schema for that target + basis |
| `PreprocessorRunCache` | Run-level cache for regridders, compressed Laplacians, etc. |

The `G` type parameter pins topology at compile time; the `Basis` parameter
pins the writer's on-disk label to the runtime's mass basis. Constructor and
writer validations reject incompatible combinations before promotion.

```mermaid
flowchart LR
    RAW[Raw meteo<br/>GRIB / NetCDF]
    READER[AbstractMetReader<br/>ingest_window!]
    VT[AbstractVerticalTransform<br/>apply_vertical!]
    DRIVER[run_unified_preprocessor_day!]
    BAL[Mass-fix +<br/>Poisson / column balance]
    GATE[Write-time<br/>replay gate + positivity gate]
    WRITER[AbstractBinaryWriter]
    BIN[transport binary]
    RAW --> READER --> VT --> DRIVER --> BAL --> GATE --> WRITER --> BIN
```

### Supported (source × target) combinations

| Source ↓  Target → | LatLon | Reduced Gaussian | CubedSphere |
| --- | --- | --- | --- |
| **Spectral ERA5** | ✅ unified driver | ✅ unified driver | ✅ unified driver |
| **GEOS-IT native** | — | — | ✅ unified driver (production) |
| **GEOS-FP native** | — | — | ✅ unified driver |
| **MERRA-2 native LL winds** | — | — | 🟡 wind-derived CS writer; convection/PBL fields from the GEOS-Chem archive; files staged externally |
| **LL transport binary → CS** (regrid passthrough) | — | — | 🟡 `regrid_ll_binary_to_cs` (functional; not yet on the unified driver) |

The source-target implementations share the typed preprocessing contract and
canonical CLI. The MERRA-2 path uses its dedicated wind-derived CS writer;
its unified OPeNDAP download execution is not yet wired.
`regrid_ll_binary_to_cs` still owns its own loop and is the remaining
migration item.

## Run from the CLI

The canonical entry point:

```bash
julia --project=. -t8 scripts/preprocessing/preprocess_transport_binary.jl \
    <preprocessing-config.toml> --day 2021-12-01

# Or a date range:
julia --project=. -t8 scripts/preprocessing/preprocess_transport_binary.jl \
    <preprocessing-config.toml> --start 2021-12-01 --end 2021-12-03
```

The CLI accepts:

- **`--day YYYY-MM-DD`** on every source path.
- **`--start YYYY-MM-DD --end YYYY-MM-DD`** on the native GEOS-source
  paths. The spectral path can also take `--day`; if neither flag is
  given, the spectral path processes every day for which spectral
  input is on disk.

`-t8` enables 8 Julia threads — the spectral synthesis path
parallelizes naturally per latitude row, so threads pay off. The
script reads the TOML, picks the source / vertical / target based on
the configuration, and dispatches into `run_unified_preprocessor_day!`.

## What a preprocessing config contains

This spectral ERA5 example follows
`config/preprocessing/era5_ll72x37_advresln_dec2021_f32.toml`. Native-source
configs use a different vertical-selection interface, described below:

```toml
# Where the raw met data lives
[input]
spectral_dir = "~/data/AtmosTransport/met/era5/0.5x0.5/spectral_hourly"
thermo_dir   = "~/data/AtmosTransport/met/era5/0.5x0.5/physics"
coefficients = "config/era5_L137_coefficients.toml"

# Where the output binary goes + on what basis
[output]
directory  = "~/data/AtmosTransport/met/era5/ll72x37_advresln/transport_binary_dec2021_f32"
mass_basis = "dry"          # binary header: mass_basis = :dry

# Target topology (drives the target axis)
[grid]
type = "latlon"             # "latlon" | "reduced_gaussian" | "cubed_sphere"
nlon = 72
nlat = 37

# The spectral ERA5 path selects/merges levels in [grid].
level_top = 1
level_bot = 137
echlevs = "ml137_tropo34"
merge_min_thickness_Pa = 1000.0

# Numerics
[numerics]
float_type        = "Float32"        # binary's on_disk_float_type
dt                = 900.0            # advection sub-step (s)
met_interval      = 3600.0           # window cadence (s); 1 h for ERA5/GEOS-IT
substep_schedule  = "adaptive_cfl"   # "constant" | "adaptive_cfl"
substep_cfl_target = 0.95            # palindrome-budget CFL target
min_steps_per_window = 2
max_steps_per_window = 16

# Optional: pin global-mean dry surface pressure (recommended for ERA5)
[mass_fix]
enable                = true
target_ps_dry_pa      = 98726.0
qv_global_climatology = 0.00247
```

Native ERA5 can reuse the named L137 interface-selection presets:

```toml
[vertical]
coefficients_file = "config/era5_L137_coefficients.toml"
transform = "level_selection"
preset = "ml137_66L"
```

The N320 reader still synthesizes all 137 native levels. The transform is
applied on the target grid: layer mass and horizontal mass fluxes are summed
over each group, while thermodynamic fields are pressure-mass weighted.

For GEOS-native preprocessing, use `[source] toml = "config/met_sources/geosit.toml"`
and `root_dir` instead of the spectral `[input]` block. That path reads a
`[vertical]` table (`transform`, `threshold_pa`, `coefficients`); do not copy
those keys into a spectral configuration and expect the same selection. See
[GEOS native cubed-sphere](geos_native_cs.md) for the full schema.

## What the unified driver does, conceptually

For every date in the requested range:

1. **Build the run-level cache** (`PreprocessorRunCache`) — once per
   run, not once per day. The LL→CS regridder and the RG compressed
   Laplacian both live here.
2. **Allocate the topology workspace** (`allocate_window_workspace`) —
   one instance per day; reused across all 24 windows.
3. **Open the day's source readers** — `ERA5SpectralReader` opens GRIB
   handles, `GEOSNativeReader` opens hourly NetCDF.
4. **Per window** (typically 24 hourly windows per day):
    - **`ingest_window!`** — read the source data into the workspace's
      raw buffers. For chained sources (`GEOSNativeReader`) the
      previous window's endpoint feeds into the current window's seed.
    - **`apply_vertical!`** — fold raw vertical fields through the
      configured `AbstractVerticalTransform`.
    - **`drain_ready_windows!`** — once enough source windows are
      ready, balance horizontal fluxes (spectral: Poisson; GEOS-native:
      column balance against the raw next-hour dry endpoint), then
      diagnose `cm` from the balanced fluxes and the explicit `dm`.
    - **`verify_window!`** — write-time replay gate
      (`‖m_evolved − m_stored_{n+1}‖ / ‖m_stored_{n+1}‖ ≤ tol`) plus
      per-substep positivity gate
      (`2·(out_x + out_y + out_z) / m_start < safety`). Failures abort
      the run rather than producing a known-bad file.
    - **`write_window!`** — stream the window to the binary.
5. **`flush_final_windows!`** — handle the last window's closure
   (next-day hour-0 endpoint or zero-tendency fallback) and patch the
   header (`steps_per_window_by_window`, `poisson_balance_target_scale_by_window`).
6. **Cross-day chaining** — the day's final mass endpoint is threaded
   into the next day's seed so window boundaries close.

## Where to read next

- [ERA5 spectral path](spectral_era5.md) — the LL / RG / CS spectral pipeline
  (Holton synthesis, vertical merging, mass-fix, Poisson balance).
- [GEOS native cubed-sphere](geos_native_cs.md) — the GEOS-IT / GEOS-FP CS
  passthrough (column balance + raw dry endpoint, dry-basis conversion,
  GCHP convection + VDIFF wiring, adaptive substep schedule).
- [Regridding](regridding.md) — conservative weights, `IdentityRegrid`,
  extensive/intensive contract.
- [Conventions cheat sheet](conventions.md) — units, replay tolerances,
  level orientation, panel conventions — the one-page reference for
  debugging an unexpected binary.
