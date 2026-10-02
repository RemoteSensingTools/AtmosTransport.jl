# TRENDYv14 S3: all-model NPP and Rh transport on C30

Campaign requested 2026-09-08. Configuration:
`config/runs/trendy_v14_s3_all_models_npp_rh_c30_2014_2024.toml`.

- 23 models, 46 tracers, one continuous transport integration.
- September 1, 2014 through December 31, 2024, matching the earlier OCO-2
  comparison. TRENDYv14 stops at 2024; this does not extend to later OCO-2 data.
- C30/L66 ERA5 meteorology from the existing Spandey archive. All 3,775 daily
  files exist. PPM advection, precomputed TM5 DKG diffusion and surface fluxes;
  no convection payload is available. The input archive labels its nested
  C90-to-C30 operator restriction experimental.
- FP32 transport and meteorology, explicitly requested by the user for wurst.
- Physical GPU 1 (second L40S), selected by UUID
  `GPU-d67e323b-b699-9138-7640-24f4774841e3`.

## Components and interpretation

Each model has `co2_<model>_npp` and `co2_<model>_rh`. The NPP tracer is driven
by **Ra-GPP**, the atmospheric sign of the requested GPP-Ra component; Rh is
positive to the atmosphere. Their carrier-subtracted sum gives the
fire-free NEE response. No fossil, ocean or fire tracer is included.

Uniform initial carriers are 1,300 ppm for NPP and 10 ppm for Rh. Subtract
these when analyzing component XCO2 responses. Maximum integrated global
NPP drawdown across the ensemble is 403.36 ppm. Output contains daily native
C30 column means and compensated Float64 global storage totals. Spatial
column means use the model's FP32 precision.

Storage totals are VMR times dry-air mass. To compare storage changes against
kg CO2 flux integrals, multiply the latter by `0.02896546 / 0.0440095`.

## Flux preparation and checks

`scripts/preprocessing/prepare_trendy_s3_npp_rh.py` uses the previous
ensemble's remapper and source interpretation, preserved in
`prepare_trendy_s3_gpp_ter_c90.py`. It applies the documented land fractions,
clips negative gross component inputs as before, remaps conservatively using
0.25-degree overlap sampling, and interpolates each gross component to daily
means with exact monthly renormalization. It forms Ra-GPP after interpolation
without clipping the net flux. Daily values are constant within each UTC day.
The final half-month uses endpoint persistence because January 2025 is absent.

LPJ-GUESS uses annual `arh` expanded to monthly values before the same daily
interpolation. ISBA-CTRIP and OCN have no usable local component inputs and are
excluded, matching the earlier 23-model ensemble. All other models are present.

All 23 preparations completed. Maximum relative source/remapped integral
error was `4.29e-15`; maximum daily/monthly mean error was `3.11e-15` before
FP32 storage. Compared with the previous three-model daily drivers, Rh is
identical and NPP agrees with GPP+Ra within `1.40e-7` of the peak flux.
Per-model JSON audits record sources, units, clipping, land treatment, annual
budgets and conservation errors; `fluxes/manifest.json` records inclusion.

## Execution and results

Run root: `/temp1/cfranken/trendy_allmodels_c30_20260908`.
Transport source is an isolated snapshot of main commit
`1a97ff50` plus the campaign scripts/configuration, the native flux-loader port,
and the checkout's Julia Manifest. Full revision and file hashes are recorded
in `provenance.json`.
Subsequent edits in the working checkout do not affect this run.

The detached controller `scripts/run_trendy_npp_rh_campaign.py` first runs
all 46 tracers over three days, checks finite positive columns and integrated
flux budgets, then launches the full integration only if that check passes.
After transport it validates all 3,775 expected daily outputs and writes
`validation.json`. It records failures and does not continue after a failed
stage. It uses rolling local meteorology staging with three days of lookahead.

The three-day, 46-tracer smoke integration completed in FP32 and used about
4.5 GiB on GPU 1. All output columns were finite and positive. Maximum global
budget residual was `3.876e-5 ppm` (`0.0196%` of the corresponding three-day
source change). The checker uses `1e-4` relative tolerance plus one FP32 ULP
of each initial carrier, so small source increments are judged at the actual
carrier precision. Its first relative-only threshold failed for some uptake
tracers; that original report is retained as
`smoke_validation_relative_only.json`. Revalidation with the documented FP32
tolerance passed, without repeating or changing the transport integration.
Existing initialization and time-varying flux tests also passed (209 checks,
in addition to the 82 native-loader checks).

Production started at 2026-09-08 16:46 UTC. Controller PID `3576159`, transport
PID `3576163`; consult `status.json` for subsequent completion or failure.

Live state: `status.json`. Logs: `logs/campaign.log`, `logs/smoke.log`,
`logs/smoke_validation.log`, `logs/transport.log`, `logs/validation.log`.
Smoke audit: `smoke_validation.json`. Production files:
`output/trendy_allmodels_c30_YYYYMMDD.nc`.

The first setup attempt failed before integration because `git archive` omits
the ignored Manifest; it was copied from the working checkout before restart.
The original setup logs are retained with a `setup_attempt1_` prefix.
The second attempt found that current main lacked the earlier native CS flux
reader. That reader was ported to `surface_flux_native.jl` and connected to the
existing source builder. Its 82 focused tests pass for FP32 and FP64; they
check spatial orientation, signs, molecular conversion, reference-time offsets
and rejection of invalid data. No transport kernels were changed. The second
attempt's logs are retained with a `setup_attempt2_` prefix.

## Queued OCO-2 intercomparison

The user requested anomaly Hovmöller rankings with the earlier GFED5 fire
enhancement, followed by separate seasonal-cycle rankings to assess whether
seasonal skill carries over to interannual variability.
`scripts/diagnostics/rank_trendy_hovmoller.py` is running detached with
`--wait`: it begins after production transport and final validation pass.
Its source snapshot, inputs, status, logs and eventual report are under
`/temp1/cfranken/trendy_allmodels_c30_20260908/intercomparison`.

The comparison uses the earlier OCO-2 cache, January 2015–December 2024,
40 equal-area latitude bands, and at least 2,000 soundings per band-month.
It fits seasonal means and linear trends jointly on the same observation
mask for every series. It ranks interannual residuals, mean seasonal cycles,
and detrended monthly variations separately by pooled correlation, with RMSE
and spatial/temporal diagnostics alongside. It compares seasonal and anomaly
ranks, and checks anomaly-rank sensitivity to omitting individual years.
All 3,653 needed GFED5 daily files exist and their grid matches exactly.
The four scientific-invariant tests and the complete reporting workflow on
a temporary synthetic ensemble with real observation gaps passed.

Production subsequently completed all 3,775 days, but the final budget gate
failed for TEM NPP only (45/46 tracer budgets passed). Its relative error is
0.012714%, compared with the approximately 0.010% threshold plus carrier ULP;
the accumulated global-mean residual is -0.02090 ppm. Completeness, finite
columns and positive carriers passed. The original failed validation report
and campaign status are preserved. The queued analysis correctly stopped.
After the user asked for the plots, diagnostic analysis was started with
`--allow-budget-failure`, which flags TEM as provisional in tables, plots
and the report without changing the tolerance or marking validation passed.

Diagnostic intercomparison completed at 2026-09-08 19:19 UTC. Results are in
`intercomparison/REPORT.md`, three Hovmöller PDF atlases (anomaly, seasonal,
detrended), and `seasonal_vs_anomaly_ranking.png`. CABLE-POP leads anomaly
correlation (0.768; RMSE 0.269 ppm), while LPX-Bern leads seasonal correlation
(0.969). Seasonal and anomaly ranks have Spearman correlation -0.484 and no
top-five overlap. These are descriptive rankings against total observed
XCO2, with TEM flagged provisional and the sampling/omitted-source limits
documented in the report. The analysis completed; the transport budget gate
remains failed and has not been relabeled as passed.
