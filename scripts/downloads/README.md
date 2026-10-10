# Download Scripts

## Unified entry point

All downloads are now driven by TOML configs via a single script:

```bash
julia --project=. scripts/downloads/download_data.jl config/downloads/<recipe>.toml \
    [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--dry-run] [--verify]
```

Download recipe TOMLs are in `config/downloads/`. They reference met source
definitions in `config/met_sources/`. See `docs/src/config/data_sources.md` for
the maintained data-layout guidance.

### Available recipes

| Recipe | Source | Chunk | Description |
|--------|--------|-------|-------------|
| `era5_arco.toml` | ERA5-ARCO | Daily | **Default ERA5 core+surface.** Native spectral GRIB + single_level netCDF from Google ARCO-ERA5 (GCS, no MARS queue, ~78 MB/s) |
| `era5_convection_only.toml` | ERA5 (CDS) | Daily | Convective mass flux (235009-012) — the one field ARCO lacks; daily requests retain normal CDS priority |
| `era5_native_monthly.toml` | ERA5 (CDS) | Monthly | Legacy all-fields CDS path (core/convection/surface); MARS-queue-bound |
| `geosfp_c720.toml` | GEOS-FP | Per-file | C720 cubed-sphere CTM mass fluxes from WashU |
| `geosit_c180.toml` | GEOS-IT | Per-file | C180 cubed-sphere from AWS S3 |
| `merra2.toml` | MERRA-2 | Per-day | OPeNDAP download (not yet implemented) |

**ERA5 default:** use `era5_arco.toml` (queue-free GCS) for core+surface and
`era5_convection_only.toml` (CDS) for convection. Preprocess the ARCO core with
`config/met_sources/era5_n320_arco.toml` (`arco_surface_pressure=true` — PS comes
from the single_level `sp` netCDF, not spectral `lnsp`). The CDS
`era5_native_*` recipes remain for fallback but are gated by the MARS queue.
For a full multi-year bulk pull, run several `download_data.jl` invocations in
parallel with disjoint `[time]` ranges — the per-file download manifest keeps
concurrent processes from colliding.

### Examples

```bash
# Preview what would be downloaded (no network calls)
julia --project=. scripts/downloads/download_data.jl \
    config/downloads/geosit_c180.toml --dry-run

# Download one month of ERA5
julia --project=. scripts/downloads/download_data.jl \
    config/downloads/era5_native_monthly.toml \
    --start 2021-12-01 --end 2021-12-31

# Check existing files for completeness
julia --project=. scripts/downloads/download_data.jl \
    config/downloads/geosit_c180.toml --verify
```

## Heritage scripts

`download_era5_surface_netcdf.py` (monthly CDS surface NetCDF) is kept for
reference in [`heritage/`](heritage/README.md); it is not maintained.
