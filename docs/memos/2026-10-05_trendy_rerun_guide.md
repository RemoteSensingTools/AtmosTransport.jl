# Re-running the TRENDY v14 S3 transport experiments

This guide is for group members who want to transport TRENDY v14 S3 land
fluxes (GPP, Ra, Rh, NEE) through AtmosTransport again. Everything needed is
on `main`: run configurations, flux-preparation scripts, campaign controllers
and plotting tools. This page explains what was run before, what has changed,
and how to set up a new run.

**What changed since the last runs.** All earlier TRENDY runs used ERA5
meteorology *without* convection. A C90 ERA5 archive *with* TM5 convection now
covers 2014-01-01 to 2025-12-31. On C90, a re-run should use it (Section 2).

## 1. What was run before

All of these runs used Float32 and no convection.

| Experiment | Config (`config/runs/`) | Grid, period | Tracers | Status, output |
|---|---|---|---|---|
| 23 models, hourly GPP and TER | `trendy_v14_s3_gpp_ter_c90_2021_advdiff_batch01..04.toml`, launched by `scripts/heritage/run_trendy_s3_2021_batches.sh` | C90 L137, 2021 | 46, in 4 batches | Done: `/kiwi-data/Data/groupMembers/cfranken/AtmosTransport/output/trendy_v14_s3_gpp_ter_c90_2021_advdiff/` |
| 3 models: GPP, TER, NEE | `trendy_v14_s3_c90_2018_2024.toml` | C90 L66, 2018-05 to 2024-12 | 9 | Done: `/temp1/cfranken/sif_gpp_iav/output/trendy_c90/` |
| 3 models: Ra, Rh | `trendy_v14_s3_ra_rh_c90_2018_2024.toml` | C90 L66, 2018-05 to 2024-12 | 6 | Not run |
| 3 models: GPP, Ra, Rh, NEE | `trendy_v14_s3_gpp_ra_rh_c90_2014_2024.toml` | C90 L66, 2014-09 to 2024-12 | 12 | Not run (output folder is empty) |
| 3 models: GPP, Ra, Rh, NEE | `trendy_v14_s3_gpp_ra_rh_c30_2014_2024.toml` | C30 L66, 2014-09 to 2024-12 | 12 | Done: `/temp1/cfranken/sif_gpp_iav/output/trendy_oco2era_c30/` |
| 23 models: NPP, Rh | `trendy_v14_s3_all_models_npp_rh_c30_2014_2024.toml`, launched by `scripts/run_trendy_npp_rh_campaign.py` | C30 L66, 2014-09 to 2024-12 | 46 | Done; TEM NPP missed the budget tolerance (0.0127% vs 0.01%). `/temp1/cfranken/trendy_allmodels_c30_20260908/` |
| 23 models × 11 TransCom land regions | `scripts/run_transcom_campaign.py` | C30 L66, 2014-09 to 2024-12 | 47 per region | Stopped in region 1 (budget guard). `/temp1/cfranken/trendy_transcom_c30_20260908/` |

The three models in the smaller runs are CLM, JULES-ES and ORCHIDEE.
`trendy_v14_s3_all_models_gpp_ter_c90_2021_advdiff.toml` holds all 46 tracers
in one job; the four batch configs replaced it so that each job fits in GPU
memory.

Background reading:

- Flux method: `TRENDYV14_S3_C90_FLUX_DRIVERS_2021_ATBD.md`,
  `TRENDYV14_S3_GPP_TER_C90_2021_ATBD.md`
- 23-model C30 run and OCO-2 ranking: `TRENDY_S3_ALL_MODELS_C30_20260908.md`
- TransCom regional split: `TRENDY_TRANSCOM_C30_20260908.md`
- 3-model runs and the SIF comparison: `SIF_GPP_IAV_XCO2_2018_2025.md`

## 2. Meteorology

| Use | Path | Notes |
|---|---|---|
| **C90 with convection** (recommended) | `/kiwi-data/Data/groupMembers/cfranken/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour_v3/` | 2014-01-01 to 2025-12-31, one file per day, ~3.1 GB/day. ERA5 winds plus TM5 convection and TM5 diffusion. |
| C90 without convection | `.../n320_to_c90/transport_binary_v4_l66_f32_no_convection/` | What the earlier C90 runs used. Use it only to compare with them. |
| C30 without convection | `/kiwi-data/Data/groupMembers/spandey/atmos/met/c30_meteo/` | No C30 archive with convection exists. |

Do **not** use `.../transport_binary_v4_l66_f32_tm5_convection_1deg_3hour/`
(no `_v3`) or the C30 `*_fullphysics_experimental` sets under
`/temp2/catrine-runs/met/`. Their convection is upside down (see
`2026-10-03_tm5_attach_level_order.md`).

The met files belong to the `cfranken` group. If you get a permission error,
ask Christian.

Earlier C90 runs needed a frozen "format-4" copy of the code to read these
binaries. Current `main` reads them directly.

## 3. One-time setup

```bash
git clone https://github.com/RemoteSensingTools/AtmosTransport.jl.git
cd AtmosTransport.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia -e 'using Pkg; Pkg.activate("gpu-env"); Pkg.develop(path="."); Pkg.add("CUDA")'
```

Run on wurst, which has two L40S GPUs. Check which GPU is free with
`nvidia-smi`, then select it with `export CUDA_VISIBLE_DEVICES=0` (or `1`).

The flux scripts need Python 3 with `numpy`, `scipy`, `netCDF4` and
`matplotlib`.

Long runs should not use your working checkout. Run them from a separate
checkout or `git worktree` pinned to a commit, so later edits cannot change a
running job. Record the commit hash with the output.

## 4. Flux drivers

All drivers are daily (or hourly) surface fluxes on the native cubed-sphere
grid. Units are kg CO2 m-2 s-1 per total cell area, positive into the
atmosphere.

| Contents | Path | Variables |
|---|---|---|
| CLM, JULES-ES, ORCHIDEE; C90; daily 2014-01-01 to 2024-12-31 | `/temp1/cfranken/sif_gpp_iav/fluxes/trendy_v2/TRENDYv14_S3_<MODEL>_gpp_ter_nee_daily_co2flux_c90_2014_2024.nc` | `GPP_CO2_FLUX`, `RA_CO2_FLUX`, `RH_CO2_FLUX`, `TER_CO2_FLUX`, `NEE_CO2_FLUX` |
| Same 3 models on C30 | `/temp1/cfranken/sif_gpp_iav/fluxes/trendy_c30/` | same |
| 23 models; C30; daily 2014-01-01 to 2024-12-31 | `/temp1/cfranken/trendy_allmodels_c30_20260908/fluxes/TRENDYv14_S3_<MODEL>_npp_rh_daily_co2flux_c30_2014_2024.nc` (with per-model JSON audits and `manifest.json`) | `NPP_CO2_FLUX` (= Ra − GPP for positive Ra and GPP magnitudes), `RH_CO2_FLUX` |
| 23 models; C90; hourly, 2021 only | `/kiwi-data/Data/groupMembers/cfranken/AtmosTransport/fluxes/TRENDYv14/S3/C90/2021/` | `GPP_CO2_FLUX`, `TER_CO2_FLUX` |
| GFED5 fire; C90; daily 2014-01-01 to 2024-12-31 (same grid and time axis as the C90 TRENDY files) | `/temp1/cfranken/sif_gpp_iav/fluxes/gfed5/GFED5_fire_daily_co2flux_c90_2014_2024.nc` | `FIRE_CO2_FLUX` |
| GFED5 fire; C30 | `/temp1/cfranken/sif_gpp_iav/fluxes/gfed5/GFED5_fire_daily_co2flux_c30_2014_2024.nc` | `FIRE_CO2_FLUX` |

GFED5 combines GFED5.1 (to 2022) with GFED5 NRT (from 2023), made by
`scripts/preprocessing/prepare_gfed5_fire_flux.py`. Annual totals in the C90
file agree with native GFED to about 2e-10 relative (2.9–4.0 Pg C/yr); the
driver is stored in Float32.

`/temp1` is a local disk on wurst. Copy files elsewhere if you run on another
machine.

The TRENDY source files are in `/kiwi-data/Data/model/TRENDYv14/S3/<MODEL>/`.
Of the 25 model folders, ISBA-CTRIP and OCN have no usable GPP or respiration
files, which leaves 23 models.

### Making new drivers

There are **no** C90 multi-year drivers for all 23 models yet. To make them,
run the multi-year script:

```bash
python3 scripts/preprocessing/prepare_trendy_s3_multiyear_c90.py \
    --models CABLE-POP CARDAMOM CLASSIC CLM <further models> \
    --start-year 2014 --end-year 2024 \
    --outdir /temp1/$USER/trendy/fluxes_c90 \
    --staging-dir /temp1/$USER/trendy/staging
```

Replace `<further models>` with the rest of the model names (see the first
point below for LPJ-GUESS). Check these points first. Each one comes from
reading the scripts:

- **LPJ-GUESS is not handled.** LPJ-GUESS has no monthly `rh`, only an
  annual `arh`. The multi-year script reads `rh` with the monthly reader only.
  `prepare_trendy_s3_npp_rh.py` has the annual fallback
  (`read_annual_as_months`). Port that branch first, or leave LPJ-GUESS out.
- **The NPP/Rh script only makes C30.** `prepare_trendy_s3_npp_rh.py` has the
  C30 grid size and file names hard-coded. For C90 NPP/Rh you must edit it, or
  form NPP from the multi-year script's output as `RA_CO2_FLUX + GPP_CO2_FLUX`:
  the stored fluxes are positive to the atmosphere, so `GPP_CO2_FLUX` is
  already negative.
- **No audit files.** The multi-year script writes no JSON audit, no manifest
  and has no `--skip-existing`. Check the annual budgets it prints for each
  model.
- **The default staging directory is the system temp folder.** `/tmp` on
  wurst is small; always pass `--staging-dir`.
- **TRENDY v14 ends in December 2024.** The last half month is held at the
  December value.
- **Land-fraction rules are per model.** The table in
  `prepare_trendy_s3_gpp_ter_c90.py` says which models get a land-fraction
  factor. Review it if a model's files are replaced.

## 5. Configure a C90 run with convection

A ready-to-run config is on `main`:
`config/runs/trendy_v14_s3_gpp_ra_rh_gfed5_c90_conv_2014_2024.toml`. It has 13
tracers: CLM, JULES-ES and ORCHIDEE × GPP/Ra/Rh/NEE, plus GFED5 fire. It runs
2014-09-01 to 2024-12-31 in Float64. Met and flux paths are absolute.

Output goes to
`$ATMOSTRANSPORT_DATA_ROOT/output/trendy_gfed5_c90_conv/`. If that variable is
unset, it defaults to `~/data/AtmosTransport`. Change the `path` if you want
the output elsewhere.

Compared with the earlier no-convection config
(`trendy_v14_s3_gpp_ra_rh_c90_2014_2024.toml`), it changes these sections:

```toml
[numerics]
float_type = "Float64"          # see Section 7

[input]
folder = "/kiwi-data/Data/groupMembers/cfranken/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour_v3"

[convection]                    # was kind = "none"
kind = "tm5"
tile_workspace_gib = 0.25
use_collab_lu = true
lmax_conv = 66
n_merge = 1
```

Keep `[diffusion] kind = "tm5_dkg"` on. The convection block matches the
CATRINE protocol run on the same archive. Leave `"_float32"` in
`file_pattern`: it is part of the met file names and does not set the run
precision.

Each tracer is one model component:

```toml
[tracers.co2_clm_gpp]
  [tracers.co2_clm_gpp.init]
  kind = "uniform"
  background = 1.3e-3           # 1300 ppm carrier (dry mole fraction)

  [tracers.co2_clm_gpp.surface_flux]
  kind = "cs_native"
  file = "/temp1/cfranken/sif_gpp_iav/fluxes/trendy_v2/TRENDYv14_S3_CLM_gpp_ter_nee_daily_co2flux_c90_2014_2024.nc"
  variable = "GPP_CO2_FLUX"
  molar_mass_kg_mol = 0.0440095
  time_varying = true
  temporal_scheme = "stepwise"
```

To add models, add one block per model and component, and list every tracer
in `[output.fields] tracers`.

**Carriers.** Every tracer starts from a uniform carrier, so uptake never
drives it negative. Subtract the carrier before analysis. Sizes used for the
10-year window:

| Component | Carrier | Reason |
|---|---|---|
| GPP, NPP (uptake) | 1300 ppm | GPP draws down up to ~770 ppm; NPP up to ~400 ppm |
| Ra, Rh, TER (emission) | 10 ppm | only increases |
| NEE | 200 ppm | drifts −30 to −60 ppm |
| GFED5 fire (emission) | 10 ppm | rises ~16 ppm over the window |

Transport NEE as its own tracer rather than differencing GPP and TER. In
Float32 the difference of two large carriers loses precision, and the
monotone-limited PPM transport is not exactly linear, so a sum of separately
transported components only approximates a directly transported total; the
NEE tracer is the reference. The TRENDY NEE has no fire in it; the land
response including fire is approximately `(NEE − 200) + (fire − 10)` ppm (add
a combined NEE + fire tracer if the sum must be exact).

## 6. Smoke test, then the full run

First copy the config, set `end_date = "2014-09-03"` and a separate output
`path`, and run three days:

```bash
export CUDA_VISIBLE_DEVICES=1
julia --project=gpu-env scripts/run_transport.jl smoke.toml
```

Check that one file exists per day, every tracer's column mean is finite, and
uptake tracers sit below their carrier and emission tracers above it. In the
test run, after 3 days the GPP tracers were 0.5–0.8 ppm below 1300 ppm (global
mean) and Ra/Rh 0.2–0.4 ppm above 10 ppm. A few cells of the emission-only
tracers dipped up to 0.015 ppm below 10 ppm; treat differences that small as
noise.

Give the full run a new, empty output directory: daily NetCDF files are
created with clobber, so a rerun into the same `path` overwrites earlier days,
and an interrupted rerun leaves a mix of new and stale files.

Then start the full run so it survives logging out:

```bash
nohup setsid julia --project=gpu-env scripts/run_transport.jl \
    config/runs/trendy_v14_s3_gpp_ra_rh_gfed5_c90_conv_2014_2024.toml > run.log 2>&1 &
```

Before the first time step, the runner opens every met file header. In the
earlier 7- and 10-year runs this took 15–30 minutes with no visible progress.
Do not kill the job during this phase.

**Measured cost.** These smoke runs (2014-09-01 to 09-03) were done on wurst
(L40S) on 2026-10-05:

| Run | Time per simulated day (after day 1) | Estimate for 3,775 days | Peak GPU memory |
|---|---|---|---|
| 12 TRENDY tracers, Float32 | 13 s | ~14 h | 19.6 GB |
| 12 TRENDY tracers, Float64 | 32 s | ~34 h | 38.6 GB |
| The 13-tracer config above (with GFED5), Float64 | 39 s | ~40 h | ~42 GB |

Startup comes on top. The estimates assume the 3-day rate holds for the whole
run. For comparison, the 25-month CATRINE run (Float64, 4 tracers) averaged
17 s per day on the same archive.

The smoke test of the 13-tracer config also checked:

- The 12 TRENDY tracers were bitwise identical to the 12-tracer run.
- The fire tracer's mass gain matched the GFED5 flux for 1–3 September to
  8.5e-8 relative. The neighbouring days would be off by 0.6–2%, so the flux
  dates line up.

**GPU memory limits the number of tracers per job.** The 13-tracer Float64
config already uses ~42 GB of the 46 GB card, so it needs a GPU to itself and
has no room for more tracers. For all 23 models with two components each
(for example NPP and Rh, 46 tracers), plan 4 jobs of about 12 tracers in
Float64; with this config's four components (92 tracers), about 8. Float32
needs about half the memory per tracer, but the per-job limit has not been
measured. A shortened smoke run loads only its own days of flux, so it checks
correctness, not capacity: size batches from the full-run estimate (about
1.47 GB of flux per tracer in Float64 plus the measured base), or test the
largest batch once with the full `start_date`/`end_date` and stop it after
the first day.

## 7. Float32 or Float64

- **Float32 loses tracer mass.** The loss is proportional to each tracer's
  background. A no-flux test on this C90 archive with convection (December
  2021) lost 1.4e-8 of the tracer mass per hour. The TRENDY smoke test shows
  the same rate: after 72 hours every Float32 tracer was 1.0e-6 of its
  carrier below its Float64 twin (−1.3e-3 ppm for GPP at 1300 ppm). Assuming
  the rate stays constant, 10.3 years (about 90,000 hours) is about 0.13%.
  That is roughly 1.6 ppm on a 1300 ppm GPP carrier and 0.25 ppm on a 200 ppm
  NEE carrier. The earlier C30 runs without convection lost much less: the
  worst tracer (TEM NPP) was off by 0.02 ppm.
- **Float64 conserves mass** to about 1e-16 on the same test.
- **Cost:** Float64 is 2.5 times slower and needs twice the GPU memory per
  tracer (Section 6).

Use Float64 when budgets or trends matter. Float32 is acceptable for spatial
patterns and detrended anomalies, as long as carriers stay small.

## 8. Checking and analysing output

The output has one NetCDF per day: the column-mean dry CO2 mole fraction of
every tracer on the native cubed sphere (`nf × Ydim × Xdim`). Subtract each
tracer's carrier, for example:

    NEE response = NEE − 200 ppm  ≈  (GPP − 1300) + (Ra − 10) + (Rh − 10) ppm

(the component sum is approximate; see section 5).

Scripts on `main`:

| Script | Purpose |
|---|---|
| `scripts/diagnostics/check_trendy_npp_rh_run.py` | Checks completeness, finite values and each tracer's global budget against the flux integral. Written for the 23-model C30 run folder (expects `fluxes/manifest.json`, `co2_<model>_<npp\|rh>` names, start 2014-09-01); adapt it for other layouts. |
| `scripts/diagnostics/rank_trendy_hovmoller.py` | Ranks TRENDY + GFED5 runs against OCO-2 seasonal and interannual Hovmöllers. |
| `scripts/diagnostics/heritage/plot_c30_trendy_hovmoller.py` | Hovmöller and zonal growth-rate plots for the 12-tracer C30 run. |
| `scripts/diagnostics/heritage/plot_trendy_allmodels_nee_decomposition.py` | Splits each model's NEE anomaly into NPP, Rh and fire; scores it against the OCO-2 growth-rate anomaly. |
| `scripts/visualization/plot_trendy_s3_xco2_hovmoller_atlas.py` | Daily XCO2 Hovmöller atlas of the ensemble. |
| `scripts/visualization/animate_trendy_s3_xco2_ensemble.py` | Per-model XCO2 animations against ensemble references. |

These scripts were written for earlier campaigns, and new paths alone are not
enough for this run: `rank_trendy_hovmoller.py` hard-codes the 23-model C30
NPP/Rh file names, tracer names, fire variable and the 2015–2024 interval; the
atlas and animation scripts expect the 2021 batch layout and
`co2_trendy_<model>_{gpp,ter}` tracers on 400 ppm carriers. Adapt them to this
run's tracers before use.

## 9. Pitfalls

- **Wrong met folder.** Only the `_v3` C90 archive has correct convection.
- **`/tmp` is small on wurst.** Use `/temp1/<you>/` for staging, runs and
  output.
- **GPU memory is set by the flux files, not the tracer fields.** Each
  `cs_native` tracer keeps the decoded time series of its flux variable on the
  GPU, for the slices of the run period (all slices when `[input]` has no
  `start_date`). For the full run (3775 daily slices, 2014-09-01 to
  2024-12-31) that is about 0.73 GB per tracer in Float32 and 1.47 GB in
  Float64. The totals in section 6 were measured when every slice (2014-01-01
  onwards) was loaded; the full run needs slightly less, a short smoke run
  much less. The 2021 hourly files are 1.7 GB per tracer,
  which is why the 23-model 2021 run had to be split into 4 batches. Section
  6 gives measured totals; plan batches from them.
- **Sharing output.** If group members cannot read your files, run
  `chmod -R g+rX <run folder>`.
