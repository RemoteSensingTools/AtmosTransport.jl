# Diagnostics port, steps B1 and B2: measured facts (2026-10-11)

Measured record of steps B1 and B2 (B2a budget closure, B2b GEOS-Chem mass
balance, B2c Float32 diagnostics) of the Python → Julia port
(`2026-10-10_python_flux_port_design.md`, Sections 1c, 2.8, 4.3 and 5): every
Julia port against the reference named in design Section 4.3, and the test
files run on the final tree. Each item names the command and where its output
lives. No plans are recorded here.

## 0. Setting

- Code: worktree `AtmosTransportModel-diag`, branch `diag/b1b2`, uncommitted
  changes on `a5df02ae`. "Final tree" below means the worktree after the last
  code edit of the four ports (2026-10-11 08:11 PDT, `src/Analysis/run_context.jl`).
- Host: `wurst.gps.caltech.edu`, no GPU (`CUDA_VISIBLE_DEVICES=""`). Load
  average 35–40 on 128 cores during the reruns of 08:16–08:45 (other jobs).
- Julia 1.12.6 with the worktree's `test/Manifest.toml` (NCDatasets 0.14.15,
  NetCDF_jll 401.1000.101). `J` stands for
  `$HOME/.juliaup/bin/julia --heap-size-hint=12G --project=<worktree>/test`,
  `JR` for the same with `--project=<worktree>` (used for the B2c scripts, as
  in the B2c session).
- Python references: `/usr/bin/python3` 3.9.25, numpy 1.26.4, netCDF4 1.5.8.
  The Python scripts behind every reference are those of `a5df02ae`, either
  in the tree or copied next to the reference
  (`f32_attribution/python_original_a5df02ae.py`,
  `f32_f64/python_original_a5df02ae.py`, `gc_mass_balance/old_script/`).
- All inputs were opened read-only. Outputs are under
  `/temp1/cfranken/diag_port/` (abbreviated `DP` below), not in the
  repository:

| Path (under `DP`) | Content |
|---|---|
| `scripts/` | B1 validation scripts and the Python reference scripts |
| `trendy_ranking/`, `sif_control/`, `catrine_one_time/` | B1 Python references and the B1-session validation reports |
| `budget_closure/v2/` | B2a parity, CLI and drawdown runs of the B2a session |
| `gc_mass_balance/` | B2b full C90 run, C180 window, day-1 check, AtmosTransport-mode regression |
| `f32_attribution/`, `f32_f64/`, `f32_rerun_1011/` | B2c Python references, views and reruns |
| `integration_1011/` | reruns of this memo on the final tree: `b1/`, `b2a/`, `b2b/`, `b2c/`, driver scripts `revalidate_chain_{a,b,c}.sh`, status files |

The integration reruns use copies of the session scripts that differ only in
the output path (`integration_1011/b1/validate_*.jl`,
`integration_1011/b2a/validate_carrier_drawdown.jl`,
`integration_1011/b2c/rerun_attribution.jl`).

---

## 1. B1: `Analysis` foundations

Design 4.3 targets: TRENDY ranking monthly fields ≤ 1e-13, arrays and CSV
≤ 1e-10, identical ranks, CABLE-POP r = 0.76797; SIF control ≤ 1e-10 with
band arrays element by element and the memo values (0.644/0.244, ratio 0.38;
0.627/0.012; r = −0.002).

### 1.1 TRENDY ranking

Command: `$J DP/integration_1011/b1/validate_trendy_ranking.jl` (log
`integration_1011/b1/trendy_ranking.log`, report
`integration_1011/b1/trendy_ranking/validation.json`). Inputs: the 3653 daily
C30 snapshot files 2015-01-01 … 2024-12-31 of
`/temp1/cfranken/trendy_allmodels_c30_20260908/output/` (46 column means) and
of `/temp1/cfranken/sif_gpp_iav/output/gfed5_fire_c30/`. References:
`intercomparison/monthly_fields.npz` and `comparison_fields.npz` (converted
to `DP/trendy_ranking/python_reference.nc` by `DP/scripts/npz_to_nc_trendy.py`)
and `intercomparison/{anomaly,seasonal,detrended}_ranking.csv`.

| Quantity | Result |
|---|---|
| Monthly 40-band NEE and fire means (`daily_snapshot_series` → `BandAssignment` → `band_means`) | max relative difference 0; 100 % of values bitwise equal |
| `seasonal_trend_decomposition`, max abs difference | anomaly 5.68e-14 ppm, anomaly without fire 4.97e-14, seasonal 1.95e-14, detrended 3.93e-14; NaN masks identical; the same from the Julia and from the Python monthly fields |
| `skill_metrics` vs the ranking CSVs, max abs difference over r, rmse, bias, amplitude ratio, no-fire r | anomaly ≤ 8.5e-16, seasonal ≤ 8.9e-16, detrended ≤ 6.7e-16; `n` identical; ranks identical for all three |
| Winners | anomaly CABLE-POP r = 0.7679747570737526 (n = 4199); seasonal LPX-Bern r = 0.9687296344053349 (n = 424); detrended ELM r = 0.946088312727864 (n = 4199) |
| Read time | 3653 + 3653 frames in 212 s (B1 session: 206 s) |

### 1.2 SIF-GPP transport control

Command: `$J DP/integration_1011/b1/validate_sif_control.jl` (log
`integration_1011/b1/sif_control.log`). Inputs: 2802 daily files (2803 frames) of
`/temp1/cfranken/sif_gpp_iav/output/sif_gpp_clim_c90/` and
`/temp1/cfranken/sif_gpp_iav/sif_gpp_iav_c90_xco2_timeseries.nc`; reference:
a fresh run of the functions of `compare_sif_gpp_transport_control.py`
(`/usr/bin/python3 -I DP/scripts/sif_control_python_ref.py <worktree>` →
`DP/sif_control/python_reference.{json,nc,log}`). 2557 common days
2019-01-01 … 2025-12-31, no NaN.

| Number | Julia | Python | Difference |
|---|---|---|---|
| zonal flux RMS | 0.64370322529722 | 0.64370322529722 | 0 |
| zonal transport RMS | 0.244229062982601 | 0.244229062982601 | 0 |
| zonal ratio | 0.379412520218198 | 0.379412520218198 | 0 |
| global flux RMS | 0.627186976547674 | 0.627186976547675 | 4.4e-16 |
| global transport RMS | 0.0120690568275417 | 0.0120690568275414 | 2.8e-16 |
| zonal r | −0.00243352871566583 | −0.00243352871566582 | 1.3e-17 |

Band arrays, max absolute difference: the Hovmöller band means `ha` and `hc`
0 (bitwise equal); `da` 2.7e-15, `dc` 1.4e-14, `ga` 3.4e-14, `gc` 7.2e-14,
`gda` 2.8e-14, `gdc` 5.2e-14. Per-year
flux/transport RMS 2019–2025: 0.163/0.241, 0.256/0.236, 0.168/0.261,
0.190/0.254, 0.225/0.249, 0.251/0.239, 0.244/0.226. Every printed value equals
the B1-session report `DP/sif_control/validation.json`.

### 1.3 CATRINE column tools at one time

Command: `$J DP/integration_1011/b1/validate_catrine_one_time.jl` (log
`integration_1011/b1/catrine_one_time.log`). 2021-12-01 03Z, the ERA5 C90 run
of `/temp1/cfranken/catrine_protocol` against
`~/data/AtmosTransport/catrine-geoschem-runs/C90/2021/12/GEOSChem.CATRINE_inst.20211201_0300z.nc4`;
reference: the functions of `catrine_compare_vs_geoschem.py`
(`DP/scripts/catrine_one_time_python_ref.py`) and
`compare_c90_full/timeseries.npz`.

| Quantity | Result |
|---|---|
| Column-band fields (`column_band_weights`, 5 bands × 4 tracers × 2 models) | bitwise equal |
| `weighted_fit_stats` vs the Python `wstats` and `timeseries.npz` | r² 6.3e-16, slope 3.6e-16, rmse 2.4e-16, ref mean 3.0e-16, ref std 1.8e-16, bias 3.3e-13 relative |
| 4° zonal means on 17 pressure levels (`logp_interpolate`) | CO₂ and SF₆ ≤ 9.4e-16 relative; fossil CO₂ ≤ 5.8e-13 and Rn-222 ≤ 5.7e-13 relative (≤ 2.1e-13 of the field maximum); NaN masks identical |

### 1.4 `compare_cs_flux_files.jl` band assignment

`DP/edge_count.log` (B1 session; the script was not kept) records 0 cells
assigned to a different band by `Analysis.EqualAreaBands` than by the
script's former bands, for n = 40 and 60, on C4, C8, C30, C90, C180, C360 and
C720 in the gnomonic and GEOS-native panel conventions.
`test/core/test_compare_cs_flux_files.jl` passes (§5).

---

## 2. B2a: budget closure (`Analysis.budget_closure`)

Design 4.3 target: with the legacy constants injected by the harness, expected
and actual changes ≤ 1e-12 from the Python report; TEM NPP fails at
1.2714e-4; 45/46 pass; smoke max 1.96e-4.

References: `/temp1/cfranken/trendy_allmodels_c30_20260908/validation.json`
and `smoke_validation.json`, written by the retired
`check_trendy_npp_rh_run.py`. Runs: `run.toml` (46 tracers, 3775 daily files,
3776 frames 2014-09-01 … 2025-01-01) and `smoke.toml` (3 files, 4 frames).

Commands (`R=/temp1/cfranken/trendy_allmodels_c30_20260908`,
`O=DP/integration_1011/b2a`):

```bash
$J scripts/validation/flux_port_parity/budget_closure_parity.jl --config $R/smoke.toml \
   --reference $R/smoke_validation.json --json $O/smoke_parity.json
$J scripts/validation/flux_port_parity/budget_closure_parity.jl --config $R/run.toml \
   --reference $R/validation.json --json $O/full_parity.json
$J scripts/diagnostics/tracer_budget_closure.jl series $R/smoke.toml --report $O/smoke_cli.json
$J scripts/diagnostics/tracer_budget_closure.jl series $R/run.toml --report $O/full_cli.json
```

Variants: `legacy` = the harness emulator of the Python arithmetic
(`LegacyDailyFluxFileEmission`: flux-file `cell_area`, whole-day sums,
M_air = 0.02896546); `runtime` = the library default (the run's own sources,
current `DRY_AIR_MOLAR_MASS` = 0.0289644); `run constant` = the runtime
changes times 0.02896546 / 0.0289644.

| Run | Variant | Strict pass | Max relative error | Against the Python report |
|---|---|---|---|---|
| smoke | legacy | 46/46 | 1.96004e-4 (co2_elm_fates_npp); reference 1.96004e-4 | expected 3.767e-16, actual 0, tolerance 2.387e-9 (max relative differences); pass flags equal |
| smoke | runtime | 46/46 | 1.59392e-4 | |
| smoke | run constant | 46/46 | 1.95981e-4 | expected vs legacy ≤ 6.206e-8 |
| full | legacy | 45/46 (co2_tem_npp fails) | 1.27141e-4 (co2_tem_npp); reference 1.27141e-4, passed false | expected 7.066e-16, actual 0, tolerance 1.901e-11 (max relative differences); pass flags equal |
| full | runtime | 43/46 (co2_tem_npp, co2_clm_fates_npp, co2_elm_npp fail) | 1.63756e-4 (co2_tem_npp) | |
| full | run constant | 45/46 (co2_tem_npp fails) | 1.27155e-4 (co2_tem_npp) | expected vs legacy ≤ 5.484e-8 |

- All rows were run on the final tree (smoke 08:27; full parity 08:19–08:38;
  full CLI 08:33–08:42). Both
  parity logs equal the B2a-session logs `DP/budget_closure/v2/{smoke,full}_parity.log`
  line for line, and both parity JSONs differ from the B2a-session ones only
  in their timing fields.
- `tracer_budget_closure.jl series smoke.toml`: 46 tracers, strict and
  operational pass, max relative error 1.59392e-4, exit 0; the JSON report
  equals the B2a-session `budget_closure/v2/smoke_cli_final.json`.
- `tracer_budget_closure.jl series run.toml`: 46 tracers, 3776 frames in 3775
  files; the strict level fails for co2_tem_npp (relative error 1.6376e-4),
  co2_clm_fates_npp (1.3308e-4) and co2_elm_npp (1.1413e-4) and the
  operational level passes for all 46; max
  relative error 1.63756e-4; exit 1; 8 min 13 s wall, 1.56 GB peak RSS. The
  JSON report equals the B2a-session `budget_closure/v2/full_cli.json`.
- Full parity: 18 min 37 s wall, 1.60 GB peak RSS (`/usr/bin/time -v`).
- `carrier_drawdown` (`$J DP/integration_1011/b2a/validate_carrier_drawdown.jl`)
  against the `npp_max_global_drawdown_ppm` audits of
  `prepare_trendy_s3_npp_rh.py` (`$R/fluxes/*.json`), after rescaling by the
  predicted ratio 0.9999634048 (constant and air mass): worst
  |ratio/predicted − 1| = 5.484e-8 over the 23 NPP tracers (CLASSIC); Rh
  drawdown 0 ppm; 115 s. The JSON equals the B2a-session one apart from its
  timing field.

---

## 3. B2b: GEOS-Chem true mass balance (`catrine_true_mass_balance.jl --model geoschem`)

Design 4.3 target: hours identical; burdens and sources ≤ 1e-12; residuals
about 1e-13 × burden; the four summary lines to every printed digit.

Reference: `/temp1/cfranken/catrine_protocol/true_mass_balance/geoschem.csv`
and `.log` (retired `catrine_true_mass_balance_gc.py`). Archive:
`~/data/AtmosTransport/catrine-geoschem-runs/C90`. Comparison script:
`DP/gc_mass_balance/scripts/compare_csv.jl <ref.csv> <ref.log> <new.csv> <new.log> <report.json>`.

### 3.1 Full C90 archive (B2b session, 2026-10-11 00:39)

Command: `julia -t 8 --project=<worktree>/test scripts/diagnostics/catrine_true_mass_balance.jl --model geoschem --out DP/gc_mass_balance/geoschem.csv`
(defaults: 2021-12-01T03:00 … 2024-01-01T00:00, origin 2021-12-01T00:00);
40 min 2 s wall. Report `DP/gc_mass_balance/validation_c90_full.json`. This
run used the code before the B2b review fixes (§3.2).

| Column | Result |
|---|---|
| header, hours, rows | identical, 6088 rows |
| `dry_air_kg` | ≤ 8.0e-16 relative |
| burdens | ≤ 4.9e-15 relative (co2_fossil), co2_natural 2.9e-15, sf6 2.6e-15, rn222 2.2e-15 |
| cumulative sources | co2_fossil 2.9e-16, sf6 8.8e-16, rn222 0 (bitwise); co2_natural 3.07e-12 pointwise at hour 5922, where the cumulative source is 1.12e9 kg against a maximum of 2.75e13 kg; as a fraction of that maximum 5.7e-16 |
| residuals, max \|Δr\| / burden | co2_natural 2.9e-15, co2_fossil 4.9e-15, sf6 3.5e-15, rn222 7.7e-15 |
| final residual, relative difference | co2_natural 7.7e-11, co2_fossil 1.4e-11, sf6 9.0e-10, rn222 9.3e-14 |
| summary lines (r, r/∫E, r/air) | identical to every printed digit: co2_natural +2.2622e+10, +8.241e-04, +4.4236e-09; co2_fossil +5.6914e+09, +1.132e-04, +1.1129e-09; sf6 −5.8247e+01, −1.413e-05, −1.1390e-17; rn222 +7.4453e-05, +1.935e-05, +1.4559e-23 |

The co2_natural pointwise figure was recomputed from the two CSVs for this
memo with the Python standard library.

### 3.2 After the review fixes

- Day 1 (8 rows, 2021-12-01T03 … 2021-12-02T00): the B2b-session outputs
  before and after the fixes, `gc_day1.csv` and `gc_day1_final.csv`, are
  byte-identical. Rerun on the final tree,
  `julia -t 8 ... --model geoschem --end 2021-12-02T00:00 --out DP/integration_1011/b2b/gc_day1.csv`,
  is byte-identical to `gc_day1_final.csv`. Against the Python
  `DP/gc_mass_balance/ref_day1.csv`: hours identical; dry air, burdens and
  sources ≤ 2.2e-15 relative; residuals ≤ 2.2e-15 of the burden
  (`integration_1011/b2b/validation_day1.json`).
- C180 window 2021-12-20T03 … 2021-12-23T00 with the truncated file
  `C180/GEOSChem.CATRINE_inst.20211221_1200z.nc4`: both scripts list that file
  as unreadable and write 23 rows; hours identical; dry air, burdens and
  sources ≤ 3.6e-14 relative; residuals ≤ 3.5e-15 of the burden; summary lines
  identical to every printed digit (`gc_mass_balance/c180_window/validation_c180.json`).
  The CSVs before and after the fixes are byte-identical.
- AtmosTransport mode (B2b session, 00:11, code before the review fixes; not
  repeated): on the 25-month ERA5 fossil CO₂ run (6088 output times) the CSV
  of the `a5df02ae` script and of the new script are byte-identical and the
  logs differ only in the output path (`gc_mass_balance/at_regression/`).

---

## 4. B2c: Float32 diagnostics

### 4.1 `float32_conservation_attribution.jl`

Design 4.3 target: every printed digit (full_final F32−F64 −2.322e-09,
−2.003e-08, −3.747e-09, −1.805e-08); 16 configs pass `validate_config`.

Command: `cd DP/integration_1011/b2c && $JR rerun_attribution.jl` (log
`rerun_attribution.log`). Inputs:
`~/data/AtmosTransport/output/precision_attribution` and its symlink views
`DP/f32_attribution/views/{base,fixed,final,final2}`; references: the tables
printed by the `a5df02ae` Python script (`DP/f32_attribution/python_budget_*.txt`),
its full-precision values (`python_full_precision.json`) and the configs it
writes (`python_configs/`).

| Check | Result |
|---|---|
| `budget` tables, four run sets (views) | byte-identical to the Python tables |
| `budget` on the workspace with tag templates `merra2={set}_final_{ft}`, `era5=era5_{set}_final_{ft}` (and the base set) and prefix `precision` | identical to the Python tables of the views |
| full precision, 160 values | 129 bitwise; max relative difference: drift 0, F32−F64 0, increment mean 3.5e-16, increment std 2.1e-16, t 2.7e-16 |
| full_final F32−F64 | background −2.322460930860182e-9, co2_natural −2.0030146834791572e-8, sf6 −3.74705490732099e-9, fossil −1.805250558867306e-8 |
| `write-configs` with the real data root | 16 written; 16/16 pass `validate_config` when read back; 16/16 equal the Python-written configs as parsed Dicts; the Python-written configs pass too (16/16) |

### 4.2 `compare_float32_float64_runs.jl`

Design 4.3 target: 16 numbers ≤ 1e-12 against
`compare_f32_f64_dec2021/default/float32_vs_float64.json`, then the 761-day
JSON.

Commands (from `DP/integration_1011/b2c`; patterns quoted):

```bash
$JR <worktree>/scripts/diagnostics/compare_float32_float64_runs.jl \
   --f32 '/temp1/cfranken/catrine_protocol/output/merra2_full_f32/catrine_c90_merra2_202112*.nc' \
   --f64 '/home/cfranken/data/AtmosTransport/output/catrine_merra2_f64/catrine_c90_merra2_202112*.nc' \
   --out dec2021_default --label "MERRA-2 C90"
$JR ... --f32 '/temp1/cfranken/catrine_protocol/output/merra2_hybridmass_fv3_dec2021_mar2022/catrine_c90_merra2_202112*.nc' \
   --f64 '/temp1/cfranken/catrine_protocol/output/merra2_hybridmass_fv3_f64_dec2021/catrine_c90_merra2_202112*.nc' \
   --out dec2021_fv3 --label "MERRA-2 C90 FV3"
$JR ... --f32 '/temp1/cfranken/catrine_protocol/output/merra2_full_f32/catrine_c90_merra2_*.nc' \
   --f64 '/home/cfranken/data/AtmosTransport/output/catrine_merra2_f64/catrine_c90_merra2_*.nc' \
   --out full_761d --label "MERRA-2 C90"
```

Comparison: `python3 DP/f32_f64/compare_json.py <reference.json> <julia.json>`
over the four numbers per tracer (`final_total_rel_net_of_start`,
`final_total_rel_raw`, `final_column_mean_rms`, `final_column_mean_mean`).

| Pair | Reference | Days | Max relative difference (16 numbers) | Bitwise |
|---|---|---|---|---|
| December 2021, default | `/temp1/cfranken/catrine_protocol/compare_f32_f64_dec2021/default/float32_vs_float64.json` | 31 | 9.95e-15 (fossil column-mean mean) | 8 (all 8 totals) |
| December 2021, FV3 | `.../compare_f32_f64_dec2021/fv3/float32_vs_float64.json` | 31 | 1.58e-14 | 11 |
| 2021-12-01 … 2023-12-31 | `/home/cfranken/www/catrine/benchmark/figures/float32_vs_float64.json` | 761 | 8.87e-15 (fossil column-mean mean) | 12 |

The Julia JSON has the same keys plus `origin` and `column_mean_hour_utc`.
The 761-day comparison took 72 s on the final tree.

---

## 5. Tests and documentation build on the final tree

Each file run alone, `$J test/core/<file>.jl`, 08:16–08:28; logs in this
session's scratch directory. All passed:

| File | Passes | | File | Passes |
|---|---|---|---|---|
| `test_analysis_decomposition` | 66 | | `test_flux_port_parity_harness` | 117 |
| `test_analysis_bands_stats` | 91 | | `test_multiday_source_clock` | 11 |
| `test_analysis_snapshot_series` | 38 | | `test_public_config_and_cli` | 157 |
| `test_analysis_mass_totals` | 48 | | `test_runtests_runner` | 9 |
| `test_analysis_budget_closure` | 130 | | `test_output_snapshots` | 107 |
| `test_analysis_geoschem_archive` | 149 | | `test_public_api_surface` | 12 |
| `test_float32_float64_diagnostics` | 205 | | `test_public_docstrings` | 31 |
| `test_float32_conservation` | 27 | | `test_documentation_links` | 101 |
| `test_physical_constants` | 79 | | `test_readme_current` | 271 |
| `test_compare_cs_flux_files` | 412 | | `test_aqua` | 10 |

The strict documentation build (`ATMOSTR_DOCS_BUILD_ONLY=true`, `warnonly =
false`, `checkdocs = :exports`) passed, run as
`julia --project=<scratch env> docs/make.jl` with a scratch copy of
`docs/Project.toml` whose `[sources]` points at the worktree (no
`Pkg.develop`), at 08:20 and again at 08:45 after the last documentation
edits (`CHANGELOG.md` section, `docs/src/api/index.md` row);
`test_documentation_links`, `test_readme_current` and
`test_public_docstrings` passed again after those edits. Goldens and the full
suite were not run.
