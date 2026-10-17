# Preprocessing Configs

Preprocessing TOMLs are consumed by:

```bash
julia --project=. scripts/preprocessing/preprocess_transport_binary.jl <config.toml> --day YYYY-MM-DD
julia --project=. scripts/preprocessing/preprocess_transport_binary.jl <config.toml> --start YYYY-MM-DD --end YYYY-MM-DD
```

The preprocessor writes one transport binary per day. Runtime configs under
`config/runs/` then point `[input].folder` at that output directory.

| Group | Purpose | Start Here |
|---|---|---|
| `era5_*transport_binary*.toml` | ERA5 spectral to lat-lon, reduced-Gaussian, or cubed-sphere transport binaries. | Match the target grid in the filename. |
| `geosit_*` | GEOS-IT native cubed-sphere preprocessing. | `geosit_c180_native_dec2021_f32.toml` |
| `geosfp_*` | GEOS-FP native cubed-sphere preprocessing. | `geosfp_c720_native_to_cs180.toml` |
| `likely_legacy/` | Older descriptors preserved for provenance. | Avoid for new work. |

## Important knobs

The unified CLI supports several source readers, and their settings do not all
live in the same TOML section:

| Path | Section and key | Notes |
|---|---|---|
| ERA5 spectral | `[input].spectral_dir`, `thermo_dir`, `coefficients` | Directories and hybrid coefficients for the spectral source. |
| ERA5 spectral | `[surface].enable` | Include PBL surface fields when the selected config supports them. |
| Native GEOS | `[source].include_surface` | Include PBL surface fields. |
| Native GEOS | `[source].include_convection` | Include CMFMC/DTRAIN convection forcing. |
| Native GEOS | `[source].include_vdiff_fields` | Include the inputs needed by the GCHP-style VDIFF closure. |
| ERA5 N320 | `[source].toml` | The source descriptor owns `include_surface`, `include_convection` (the four TM5 convection fields), and `include_tm5_diffusion` under its `[preprocessing]` table. |
| All paths | `[output].directory` | Runtime configs point `[input].folder` here. |

ERA5 spectral configs carry no convection. The former `[tm5_convection]`
table (lat-lon physics-file route) was removed and is rejected; build TM5
convection with an ERA5 N320 source or attach it with
`scripts/preprocessing/attach_catrine_tm5_convection_cs.jl`.

For native GEOS, the dynamics time step used to interpret archived mass fluxes
is `mass_flux_dt_seconds` in the selected source descriptor under
`config/met_sources/`; it is not a top-level preprocessing-run switch. Start
from a maintained config in this directory instead of combining keys from
different source families.

## Data Roots

Production preprocessing configs normally use `$ATMOSTRANSPORT_DATA_ROOT`,
which defaults to `~/data/AtmosTransport`. Set it once instead of editing many
TOMLs:

```bash
export ATMOSTRANSPORT_DATA_ROOT=/scratch/$USER/AtmosTransport
```
