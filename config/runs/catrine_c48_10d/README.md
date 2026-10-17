# Catrine C48 10-day test configs

Three physics-ladder configs for `scripts/run_transport.jl` at C48, 10 days
(Dec 1–10 2021), ERA5 source met.

`scripts/run_transport.jl` is the unified driven-simulation runner
(plan 40 Commit 6c). It dispatches on `grid_type` from the binary
header, so the same CLI handles LL, RG, and CS binaries. Physics,
diffusion, and convection are composed from TOML. The older
panel-native, advection-only harness is kept for reference at
`scripts/benchmarks/heritage/run_cs_transport.jl`.

| Config | Physics | Runs today? |
|---|---|---|
| `advonly.toml` | advection | ✅ yes |
| `advdiff.toml` | + `ImplicitVerticalDiffusion` (constant Kz) | ✅ yes |
| `advdiffconv.toml` | + `TM5Convection` | ⚠️ needs TM5 sections in binary |

## Prerequisites

### 1. C48 transport binaries (required for all three configs)

Neither ERA5 nor GEOS-IT ship C48 binaries. Generate from existing
LL 720×361 v4 F64 binaries via:

```bash
for d in 01 02 03 04 05 06 07 08 09 10; do
  julia -t16 --project=. scripts/preprocessing/regrid_ll_transport_binary_to_cs.jl \
    --input  ~/data/AtmosTransport/met/era5/ll720x361_v4/transport_binary_v2_tropo34_dec2021_f64/era5_transport_202112${d}_merged1000Pa_float64.bin \
    --output ~/data/AtmosTransport/met/era5/cs_c48/transport_binary_v2_tropo34_dec2021_f64/era5_transport_202112${d}_merged1000Pa_float64.bin \
    --Nc 48
done
```

Runtime ≈ 2–4 min per day at Nc=48. Total ≈ 30 min for 10 days.

### 2. (Only for `advdiffconv.toml`) TM5 convection sections in the binary

`TM5Convection` reads entu/detu/entd/detd from the transport binary.
No lat-lon route produces them any more: the ERA5 spectral preprocessor's
`[tm5_convection]` table (the lat-lon physics-file route) was removed (issue
#55), and readers reject lat-lon binaries that carry its sections and their
LL→CS regrids. Options:

- **(a) Attach TM5 convection on the cubed sphere**: build a C48 binary from
  native ERA5 (an ERA5 N320 met source with `include_convection = true`) or
  attach the TM5-converted CATRINE archive with
  `scripts/preprocessing/attach_catrine_tm5_convection_cs.jl`.
- **(b) Switch to `kind = "cmfmc"`** if you have a GEOS-style C48 binary
  with native CMFMC+DTRAIN sections. No C48 GEOS-IT binary exists today;
  would need either a C180→C48 coarsen or C48 preprocessing from source.

With a TM5-less plain C48 binary, `kind = "tm5"` now fails at recipe
validation before the run starts.

## How `run_transport.jl` wires CS physics

The TOML → operator mapping is:

```toml
[advection]  scheme = "upwind" | "slopes" | "ppm" | "linrood"
             ppm_order = 5 | 7   # only with scheme = "linrood"
[diffusion]  kind   = "none" | "constant"          (+ value: m²/s for constant)
[convection] kind   = "none" | "tm5" | "cmfmc"
```

Under the hood:

```julia
scheme     = UpwindScheme() | SlopesScheme() | PPMScheme() | LinRoodPPMScheme(order)
diffusion  = NoDiffusion()
              | ImplicitVerticalDiffusion(kz_field = CubedSphereField(
                    ntuple(_ -> ConstantField{FT,3}(value), 6)))
convection = NoConvection() | TM5Convection() | CMFMCConvection()
```

All three are attached at model construction:

```julia
model = TransportModel(state, fluxes, grid, scheme;
                        diffusion  = diffusion,
                        convection = convection)
```

and `DrivenSimulation` auto-threads the per-window convection forcing
from the binary into the operator each window.

## Remaining gaps (not blocked by the three configs above)

| Gap | Notes |
|---|---|
| Profile/precomputed/derived Kz from TOML | `ImplicitVerticalDiffusion` supports them; need per-kind TOML schema + builder dispatch. Constant Kz covers today's needs. |
| `[tracers.*.surface_flux]` (emissions) for CS | `build_surface_flux_source` now dispatches on CS grids (plan 40 Commit 1d) and `run_driven_simulation` picks it up from TOML (plan 40 Commit 6a via `build_surface_flux_sources`). Fossil tracer gets kg/s-per-cell emissions when `[tracers.co2_fossil.surface_flux]` is present. |
| TM5 sections for C48 | Blocker for `advdiffconv.toml` specifically; see Prerequisite #2 above. |

**Resolved (plan 40 Commit 1c / 2, 2026-04-24):** file-based CS initial
conditions now work via the unified
`build_initial_mixing_ratio` + `pack_initial_tracer_mass` pipeline
(`src/Models/InitialConditionIO.jl`). The C48 configs use
`kind = "catrine_co2"`, which conservatively LL→CS-regrids the Catrine
NetCDF (`~/data/AtmosTransport/catrine/InitialConditions/startCO2_202112010000.nc`)
and log-pressure-remaps column-by-column — no longer a flat-411 stub.
