# Data sources

This page covers where to obtain the raw meteorological input the
preprocessor needs, how to authenticate against each source, and
the recommended local layout under `~/data/AtmosTransport/met/`.

If you only want to learn the runtime first, start with the
[Quickstart](@ref). It generates small current-format forcing locally and does
not require a data account or a large download.

## ERA5 (ECMWF Reanalysis 5)

ERA5 supports LL, reduced-Gaussian and CS preprocessing. The repository has a
unified downloader; choose its recipe together with a matching preprocessor.
The older split spectral/thermo layout and the newer native N320 layout are
not interchangeable merely by changing the input directory.

### Choose a download route

| Recipe | Local workflow |
|---|---|
| `config/downloads/era5_arco.toml` | Native core/surface downloads from the ARCO mirror. Pair with the ARCO-aware N320 source descriptor. |
| `config/downloads/era5_convection_only.toml` | Additional CDS convection stream for a workflow that requires it. |
| `config/downloads/era5_native_daily.toml` | CDS native model-level core, convection and surface streams, requested by day. |

Preview one day without downloading it:

```bash
julia --project=. scripts/downloads/download_data.jl \
    config/downloads/era5_arco.toml --start 2021-12-01 --end 2021-12-01 --dry-run
```

Remove `--dry-run` to download after reviewing the output paths and requested
streams. `config/met_sources/era5_n320_arco.toml` explicitly sources surface
pressure from the downloaded single-level field because the assembled ARCO
core lacks spectral `lnsp`. That descriptor currently disables optional
surface/convection/diffusion payloads: downloading the surface files alone
does not enable runtime physics. Read its comments and the
[preprocessing guide](@ref Preprocessing-overview) before assembling a campaign.

CDS requests need credentials and accepted dataset terms; the public ARCO
route does not use a CDS token. Check the selected recipe and its source
descriptor for the exact file layout. Existing split-file spectral examples
use the layout listed below instead of the native N320 bundle.

### Split-file spectral input

| File | Variable | Format |
|---|---|---|
| `era5_spectral_YYYYMMDD_lnsp.gb` | log surface pressure spectral coefficients | GRIB |
| `era5_spectral_YYYYMMDD_vo_d.gb` | vorticity and divergence spectral coefficients | GRIB |
| `era5_thermo_ml_YYYYMMDD.nc` | model-level specific humidity and temperature | NetCDF |

The thermo file supplies humidity for dry-air conversion. These examples use
ERA5's 137 model levels, with surface pressure defining the hybrid coordinate.
The files must already match the spectral reader's naming and variables;
raw native bundles may require preparation rather than only renaming.

### Credentials

Get a free CDS account, then drop your Personal Access Token (PAT)
in `~/.cdsapirc`:

```text
url: https://cds.climate.copernicus.eu/api
key: <YOUR-PAT>
```

CDS migrated to single PAT-style keys in September 2024; the older
`<UID>:<API-KEY>` format is no longer accepted. Get your PAT from your
CDS account profile page once you're logged in. The `base_url` in
`config/met_sources/era5.toml` matches this. Tools that read the
CDS API will pick up the `~/.cdsapirc` file automatically.

### Datasets

| CDS dataset name | Use |
|---|---|
| `reanalysis-era5-complete` | Model-level spectral fields (VO, D, LNSP) — the AtmosTransport spectral preprocessor input |
| `reanalysis-era5-single-levels` | Surface fields (PS, 2T, 10U, 10V, …) |
| `reanalysis-era5-pressure-levels` | Pressure-level diagnostics (not used by the preprocessor) |

The maintained downloader is `scripts/downloads/download_data.jl`; its recipes
reference the descriptors in `config/met_sources/`. The native daily recipe
includes the core and optional-physics request definitions.

### Recommended local layout

```
~/data/AtmosTransport/met/era5/
└── 0.5x0.5/
    ├── spectral_hourly/                    # CDS reanalysis-era5-complete output
    │   ├── era5_spectral_20211201_lnsp.gb
    │   ├── era5_spectral_20211201_vo_d.gb
    │   └── …
    └── physics/
        └── era5_thermo_ml_20211201.nc      # CDS reanalysis-era5-complete with q
```

The preprocessing `[input].spectral_dir` and `thermo_dir` keys point at these
folders. They are not runtime `[input]` keys: the forward runtime consumes
transport binaries. See [Preprocessing overview](@ref).

## GEOS-IT (NASA GMAO Integrated Tropospheric)

GEOS-IT is the primary native cubed-sphere source. C180 (~50 km) is
the production/debug resolution. GEOS-FP native C720 hourly CTM files
are wired through the same source contract, with optional 0.25°
surface/convection fallback files attached into the preprocessed
binary.

### Per-day file set

For each `YYYYMMDD`:

| File | Cadence | Variables |
|---|---|---|
| `GEOSIT.YYYYMMDD.CTM_A1.C180.nc` | hourly (window-averaged) | `MFXC`, `MFYC`, `DELP` |
| `GEOSIT.YYYYMMDD.CTM_I1.C180.nc` | hourly (instantaneous) | `PS`, `QV` |
| `GEOSIT.YYYYMMDD.A1.C180.nc` | hourly | `PBLH`, `USTAR`, `HFLUX`, `T2M` *(only with `include_surface`)* |
| `GEOSIT.YYYYMMDD.A3mstE.C180.nc` | 3-hourly | `CMFMC` *(only with `include_convection`)* |
| `GEOSIT.YYYYMMDD.A3dyn.C180.nc` | 3-hourly | `DTRAIN` *(only with `include_convection`)*; `U`, `V` *(with `include_vdiff_fields`)* |
| `GEOSIT.YYYYMMDD.I3.C180.nc` | 3-hourly | `T` *(only with `include_vdiff_fields`)* |

The GCHP Holtslag-Boville VDIFF data contract additionally requires
`include_vdiff_fields = true` in the preprocessing TOML and the
`config/downloads/geosit_c180_gchp_vdiff.toml` download recipe.

The preprocessor needs **next-day hour 0** for the last window's
forward-flux endpoint; download `[start, end+1]` for production
runs.

### Access

| Source | URL pattern | Auth |
|---|---|---|
| **AWS S3 (primary)** — `s3://geos-chem/GEOS_C180/GEOS_IT/...` | public bucket; requester-pays NOT required | none — use `aws s3 cp --no-sign-request` |
| WashU HTTP archive (fallback) | `http://geoschemdata.wustl.edu/ExtData/GEOS_C180/GEOS_IT/...` | none |

The canonical descriptor is `config/met_sources/geosit.toml` (line
50-60), including the bucket name and the WashU base URL.

### Recommended local layout

The downloader's canonical layout puts each day's collections under
a per-day subdirectory:

```
~/data/AtmosTransport/met/geosit/
└── C180/
    └── daily/
        └── raw/
            └── 20211201/
                ├── GEOSIT.20211201.CTM_A1.C180.nc
                ├── GEOSIT.20211201.CTM_I1.C180.nc
                ├── GEOSIT.20211201.A3mstE.C180.nc       # if convection
                ├── GEOSIT.20211201.A3dyn.C180.nc        # if convection or VDIFF
                └── GEOSIT.20211201.I3.C180.nc           # if GCHP VDIFF
```

The `[source].root_dir` key in the GEOS preprocessing TOML points at
the parent directory containing the `YYYYMMDD/` per-day folders
(`~/data/AtmosTransport/met/geosit/C180/daily/raw` in this layout).
The preprocessor's file resolver also accepts a flat directory of
all NetCDFs (no per-day subdir) — that's the `raw_catrine` layout
the project's own configs use historically.

## GEOS-FP, MERRA-2 (status)

**GEOS-FP (C720).** The active descriptor is native cubed-sphere:

- `config/met_sources/geosfp.toml` — the **native C720 hourly CTM**
  product (`GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.*.nc4`).
- `config/downloads/geosfp_c720.toml` — the **native C720**
  cubed-sphere download descriptor; the **`src/Downloads/sources/geosfp.jl`**
  downloader pulls from the WashU HTTP archive (NOT the
  GEOS-IT-style AWS S3 path) into a local directory.

`GEOSSettings{:geosfp}` opens 24 hourly native CTM files plus the
next-day 00Z endpoint. The WashU archive names the hourly averaged
files with `HH30` timestamps; test fixtures may also use `HH00`. When
`[source] include_surface = true` or
`include_convection = true`, set `[source] physics_dir` to a directory
containing `GEOSFP.YYYYMMDD.{A1,A3mstE,A3dyn}.025x03125.nc` files (or
pre-regridded CS equivalents) and the preprocessor embeds `PBLH`,
`USTAR`, `HFLUX`, `T2M`, `CMFMC`, and `DTRAIN` in the transport binary.

**MERRA-2.** `MERRA2Settings` and the wind-derived CS writer are implemented.
They read native 0.5° × 0.625° PS/QV/U/V fields, derive mass fluxes, and write
CS transport binaries through the canonical preprocessing CLI. MERRA-2 has no
native MFXC/MFYC, so this is deliberately separate from `AbstractGEOSSettings`.
Two archive layouts are supported (`[preprocessing] layout`):

- `"nasa"` (default) — GES DISC `M2I3NVASM` / `M2T3NVASM` files under
  `root_dir/{M2I3NVASM,M2T3NVASM}/YYYY/MM/`; mass fluxes only. See
  `config/preprocessing/merra2_c180_dec2021_f32.toml`. The unified
  `OPeNDAPProtocol.execute!` downloader is still unavailable, so these files
  must be staged separately with NASA Earthdata credentials.
- `"geoschem"` — the GEOS-Chem-processed files that GEOS-Chem and GCHP read,
  `root_dir/YYYY/MM/MERRA2.YYYYMMDD.{I3,A3dyn,A3mstE,A1}.05x0625.nc4`,
  publicly mirrored at `s3://gcgrid/GEOS_0.5x0.625/MERRA2/`
  (`aws s3 sync --no-sign-request`). Same values as the GES DISC files, with
  levels stored surface first. This layout can also write dry `cmfmc`/`dtrain`
  (`include_convection`), GEOS-Chem's convective cloud base from A3mstC DQRCU
  (`include_convective_cloud_base`), the A1 PBL surface fields
  (`include_surface`), and the VDIFF fields plus the latent heat flux
  (`include_vdiff_fields`). See `config/met_sources/merra2_geoschem.toml` and
  `config/preprocessing/merra2_geoschem_c90_l72_f32.toml`; run with
  `[convection] kind = "cmfmc", cloud_base = "dqrcu"` and
  `[diffusion] kind = "geoschem_nonlocal_vdiff"` for GEOS-Chem's physics.

`[preprocessing] column_balance_weights` (MERRA-2 and ERA5 N320) spreads the column
mass-budget correction of the horizontal fluxes over the levels:
- `"mass"` (default): by layer air mass;
- `"hybrid_b"`: by `ΔB`, as in TM5;
- `"hybrid_mass"`: by air mass in hybrid layers only.

The hybrid options leave the pure-pressure stratospheric layers untouched.
See [Vertical transport](../theory/vertical_transport.md) section 2 and
`config/met_sources/merra2_geoschem_hybrid{b,mass}.toml`.

Face-flux construction (MERRA-2 and ERA5 N320; see
[Vertical transport](../theory/vertical_transport.md) section 1):
- `face_fluxes`:
  - `"panel_average"` (default): averages panel-local wind components, with
    one-sided panel seams;
  - `"vector"`: combines the winds as 3-D vectors, projects them onto the true
    face normals with true face lengths, and interpolates along the edge at
    panel seams.
  - `"line_integral"` (ERA5 N320 only): integrates `(V · N) Δp` along each face
    from the N320 winds and surface pressure (bilinear interpolation of the
    Cartesian wind components, 16 midpoints per face). TM5 integrates the
    spectral winds along its cell edges in the same spirit. Each cube cell's
    convergence is then that of the source flow; the cell-centre methods
    smooth it at the cube grid scale.
- `face_lengths = "cell_centerline"` (default) or `"edge"`: the length used by
  `panel_average`.
- `face_interpolation` (`vector` only): `"linear"` (default, the two adjacent
  cells), `"cubic"` (FV3's fourth-order stencil across interior faces) or
  `"fv3"` (`"cubic"` plus the filter along the face of GCHP's A → D → C
  restaggering).
- `flux_thickness = "moist"` (default) or `"dry_mass"` (MERRA-2 only): the
  layer thickness in the fluxes.
- `wind_regrid = "scalar"` (default) or `"cartesian"`: regrid `u` and `v` as
  two scalars, or the wind as a vector, as GCHP does. The scalar regrid is
  off by about 5% poleward of 88°.
- `flux_time_sampling` (ERA5 N320 only): `"window_start"` (default) holds the
  instantaneous winds of each hour over the following hourly window;
  `"window_mean"` uses the mean of the face fluxes at the start and end of the
  window, the trapezoidal rule for `∫ u Δp dt`, so the fluxes are centred in
  time. The binary header records it as `source_flux_sampling`.

The GEOS sources reject these keys. `config/met_sources/merra2_geoschem_hm_gchp.toml`
and `config/met_sources/era5_n320_arco_diffusion_hb_gchp.toml` select GCHP's
construction; the ERA5 one uses `hybrid_b`, because its 66-level grid has
hybrid layers up to 82 hPa.
`config/met_sources/era5_n320_arco_diffusion_li.toml` selects `hybrid_b`,
`line_integral` and `window_mean`; with
`config/preprocessing/era5_n320_arco_diffusion_to_c90_l117_li.toml` it keeps
ERA5's native levels below about 8 hPa and merges the 27 thinner ones above into
7 (`[vertical] transform = "merge_layers_thinner_than"`, 100 Pa), 117 levels in
all.

`[numerics] dt_met_seconds = 3600` splits every 3-hour MERRA-2 block into three
hourly windows (endpoint mass, PS, QV and T linear in time, 3-hour mean winds),
so the hourly A1 boundary-layer fields are used as archived; `10800` writes one
window per block with A1 averaged over it.

Neither layout carries DELP, so the level order of each day's files is
detected from the inst3 QV profile (moist end = surface).

## Try the runtime without external data

The maintained quickstart creates a small, deterministic version-4 transport
binary in the repository's ignored `data/quickstart/` directory:

```bash
julia --project=. examples/generate_synthetic_quickstart.jl
julia --project=. scripts/run_transport.jl config/examples/minimal_template.toml
```

This is the supported smoke test and tutorial path.

## A note on disk space

Binary size scales with grid cells, vertical levels, windows, precision, and
optional physics sections. Generate one representative day, inspect its
`payload_sections`, and size campaign storage from that file rather than from
a different source or physics configuration.

## Where to read next

- [Quickstart](@ref) — a zero-download, runnable walkthrough.
- [TOML schema](@ref) — runtime input and operator configuration.
- [Preprocessing overview](@ref) — the unified `process_day` dispatch.
