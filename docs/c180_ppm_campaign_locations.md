# C180 PPM Campaign Locations

Date: 2026-05-04

## Durable Scripts

- PPM visualization script: `scripts/visualization/run_c180_3day_ppm_viz.sh`
- GEOS-IT native PPM visualization script: `scripts/visualization/run_c180_geosit_native_3day_ppm_viz.sh`
- ERA5-on-GEOS-native PPM visualization script: `scripts/visualization/run_c180_era5_geosnative_cfl85_3day_ppm_viz.sh`
- ERA5-vs-GEOS side-by-side movie script: `scripts/visualization/run_c180_era_geos_comparison_movies.sh`
- ERA5-vs-GEOS movie generator: `scripts/visualization/compare_c180_era_geos_movies.jl`
- Window-tendency comparison script: `scripts/diagnostics/compare_c180_window_tendencies.jl`
- Window-tendency summary plot script: `scripts/diagnostics/plot_c180_window_tendency_summary.py`
- IC/cumulative-increment comparison script: `scripts/diagnostics/compare_c180_era_geos_ic_divergence.jl`
- Original pre-PPM temp visualization script: `/tmp/tm5_smoke/run_viz_3day.sh`
- Original temp campaign runner: `/tmp/tm5_smoke/run_3day_campaign.sh`

## GEOS-IT Native C180/L72 PPM Campaign

This is the current GEOS-IT native campaign. It ignores the older converted
binaries and uses a fresh conversion from the original C180 GEOS-IT NetCDF
files. Runtime outputs were refreshed on 2026-05-03 after the ERA TM5
area-scaling fix, for a clean ERA-vs-GEOS comparison.

Raw source data:

- `/home/cfranken/data/AtmosTransport/met/geosit/C180/raw_catrine/YYYYMMDD/`
- Daily files include `GEOSIT.YYYYMMDD.CTM_A1.C180.nc`, `A1`, `A3mstE`, and `A3dyn`.
- `CTM_A1` carries dry `MFXC` and `MFYC`; these are converted directly, not reconstructed from winds.

Preprocessing:

- Config: `config/preprocessing/geosit_c180_native_dec2021_f32.toml`
- Output binary directory: `/temp1/c180_geosit_native_v4_dec2021_f32`
- Regenerated 2026-05-03 with the GEOS `MFXC/(2g)` speed-scale fix.
- Binaries:
  - `/temp1/c180_geosit_native_v4_dec2021_f32/geos_transport_20211202_float32.bin`
  - `/temp1/c180_geosit_native_v4_dec2021_f32/geos_transport_20211203_float32.bin`
  - `/temp1/c180_geosit_native_v4_dec2021_f32/geos_transport_20211204_float32.bin`
- Corrected Dec 2 speed-fix sample:
  `/temp1/c180_geosit_native_v4_dec2021_f32_speedfix_sample/geos_transport_20211202_float32.bin`
- Preprocess log: `/temp1/c180_geosit_native_3d/logs/preprocess_geosit_native_f32_20211202_20211204_speedfix.log`

GEOS-IT run configs:

- Adv only: `config/runs/geosit_c180_native_3day_ppm/advonly.toml`
- Adv + PBL diffusion: `config/runs/geosit_c180_native_3day_ppm/advdiff.toml`
- Full physics: `config/runs/geosit_c180_native_3day_ppm/fullphysics.toml`

GEOS-IT NetCDF outputs:

- Adv only: `/temp1/c180_geosit_native_3d/advonly_ppm.nc`
- Adv + PBL diffusion: `/temp1/c180_geosit_native_3d/advdiff_ppm.nc`
- Full physics: `/temp1/c180_geosit_native_3d/fullphysics_ppm.nc`

GEOS-IT logs:

- Adv only: `/temp1/c180_geosit_native_3d/logs/advonly_ppm_tm5areas_20260503T183011.log`
- Adv + PBL diffusion: `/temp1/c180_geosit_native_3d/logs/advdiff_ppm_tm5areas_20260503T183011.log`
- Full physics: `/temp1/c180_geosit_native_3d/logs/fullphysics_ppm_tm5areas_20260503T183011.log`
- Snapshot stats summary: `/temp1/c180_geosit_native_3d/logs/snapshot_stats_summary_speedfix.txt`

GEOS-IT visualizations:

- Visualization directory: `/tmp/tm5_smoke/viz_geosit_native_3d_ppm`
- Visualization summary: `/tmp/tm5_smoke/viz_geosit_native_3d_ppm_summary.txt`
- Visualization rerun log: `/temp1/c180_geosit_native_3d/logs/viz_geosit_native_3d_ppm_tm5areas_20260503T183011.log`
- Generated inventory: 24 PNG grids and 6 MP4 column-mean movies.
- Script: `scripts/visualization/run_c180_geosit_native_3day_ppm_viz.sh`

GEOS-IT diagnostics:

- Binary wind-speed diagnostics:
  - Pre-fix binary comparison: `/temp1/c180_speed_diagnostics`
  - Corrected Dec 2 sample comparison: `/temp1/c180_speed_diagnostics_speedfix`
  - Regenerated main binary comparison: `/temp1/c180_speed_diagnostics_main_regenerated`
- Flat advection-only, no surface flux, 2-day smoke output:
  `/temp1/c180_geosit_native_3d/debug_flat_ppm_reset_2day.nc`
- Flat adv+diff, no surface flux, 1-day smoke output after VMR diffusion fix:
  `/temp1/c180_geosit_native_3d/debug_flat_ppm_advdiff_1day.nc`

GEOS-IT notes:

- Preprocessing uses `chain_mass = false`; every hourly window starts from the raw GEOS endpoint mass.
- Runtime configs use `reset_air_mass_each_window = true` and preserve tracer VMR across air-mass resets.
- CTM dry mass fluxes are 450 s dynamics-step transport amounts reused over the hourly window; the GEOS passthrough test expects `MFXC/(2g)` per horizontal Strang half-sweep.
- Regenerated main binary wind diagnostic matches GEOS A3dyn U/V at sampled points with mean inferred/raw speed ratio 0.976.
- All three speed-fix campaign outputs are finite at every 6-hour snapshot.

## ERA5-on-CS C180/L137 PPM Campaign

This is the earlier L137 campaign retained for comparison. The PPM outputs
below were regenerated with the current code on 2026-05-03 after the stale
ERA adv+diff/full-physics outputs were identified, and refreshed again after
the TM5 runtime area-scaling fix.

## ERA5-on-GEOS-Native C180/CFL85 Build

This is the replacement ERA5 forcing path for a cleaner GEOS-IT comparison:
ERA5 is first built on LL720 with the CFL-oriented `ml137_cfl85` upper-layer
merge, TM5 convection, and raw PBL surface fields, then regridded to the same
GEOS-native C180 horizontal geometry as the GEOS-IT binaries.

Durable config/script:

- LL source config: `config/preprocessing/era5_ll720x361_cfl85_dec2021_f32_tm5_surface.toml`
- Build/regrid script: `scripts/preprocessing/build_era5_geos_c180_cfl85_tm5_surface.sh`

Output locations:

- LL source binaries: `/home/cfranken/data/AtmosTransport/met/era5/ll720x361_v4/transport_binary_v2_cfl85_dec2021_f32_tm5_surface`
- GEOS-native C180 binaries: `/temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps8`
- Logs: `/temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps8/logs`

Status as of 2026-05-04 08:02 PDT:

- LL source binaries complete for Dec 2-4:
  `/home/cfranken/data/AtmosTransport/met/era5/ll720x361_v4/transport_binary_v2_cfl85_dec2021_f32_tm5_surface/era5_transport_20211202_merged1000Pa_float32.bin`
  through
  `/home/cfranken/data/AtmosTransport/met/era5/ll720x361_v4/transport_binary_v2_cfl85_dec2021_f32_tm5_surface/era5_transport_20211204_merged1000Pa_float32.bin`
- GEOS-native C180 binaries complete for Dec 2-4:
  `/temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps8/era5_transport_20211202_merged1000Pa_float32.bin`
  through
  `/temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps8/era5_transport_20211204_merged1000Pa_float32.bin`
- Header checks for all three target binaries: C180, `panel_convention=geos_native`,
  `cs_definition=gmao_equal_distance`, 85 levels, dry basis, 8 steps/window,
  payloads include `m/am/bm/cm/ps/dm`, PBL surface, and TM5
  `entu/detu/entd/detd`.
- Regrid gates passed at 8 steps/window:
  - Dec 2 replay max relative error `8.493e-08`; positivity max outgoing/mass `0.780`.
  - Dec 3 replay max relative error `8.315e-08`; positivity max outgoing/mass `0.740`.
  - Dec 4 replay max relative error `5.207e-08`; positivity max outgoing/mass `0.730`.
  - Positivity limit was `0.95`.

Command used:

```bash
THREADS=16 START_DAY=03 END_DAY=04 STEPS_PER_WINDOW=8 \
  scripts/preprocessing/build_era5_geos_c180_cfl85_tm5_surface.sh
```

## ERA5-on-GEOS-Native C180/L85 PPM Campaign

This is the runtime campaign driven by the GEOS-native C180 ERA5 binaries
above. It uses the same horizontal geometry and 8 steps/window as the GEOS-IT
native campaign, but retains ERA5-derived L85 vertical structure and TM5
convection fields.

ERA5-on-GEOS-native run configs:

- Adv only: `config/runs/era5_geosnative_c180_cfl85_3day_ppm/advonly.toml`
- Adv + PBL diffusion: `config/runs/era5_geosnative_c180_cfl85_3day_ppm/advdiff.toml`
- Full physics: `config/runs/era5_geosnative_c180_cfl85_3day_ppm/fullphysics.toml`

ERA5-on-GEOS-native NetCDF outputs:

- Adv only: `/temp1/c180_era5_geosgrid_cfl85_3d/advonly_ppm.nc`
- Adv + PBL diffusion: `/temp1/c180_era5_geosgrid_cfl85_3d/advdiff_ppm.nc`
- Full physics: `/temp1/c180_era5_geosgrid_cfl85_3d/fullphysics_ppm.nc`

ERA5-on-GEOS-native logs:

- Adv only: `/temp1/c180_era5_geosgrid_cfl85_3d/logs/advonly_ppm_20260504T104148.log`
- Adv + PBL diffusion: `/temp1/c180_era5_geosgrid_cfl85_3d/logs/advdiff_ppm_20260504T104428.log`
- Full physics: `/temp1/c180_era5_geosgrid_cfl85_3d/logs/fullphysics_ppm_20260504T104727.log`

ERA5-on-GEOS-native visualizations:

- Visualization directory: `/tmp/tm5_smoke/viz_era5_geosnative_c180_cfl85_3d_ppm`
- Visualization summary: `/tmp/tm5_smoke/viz_era5_geosnative_c180_cfl85_3d_ppm_summary.txt`
- Visualization run log: `/temp1/c180_era5_geosgrid_cfl85_3d/logs/viz_era5_geosnative_c180_cfl85_3d_ppm_20260504T112327.log`
- Generated inventory: 24 PNG grids and 6 MP4 column-mean movies.
- Script: `scripts/visualization/run_c180_era5_geosnative_cfl85_3day_ppm_viz.sh`

ERA5-on-GEOS-native notes:

- Runtime uses the default continuous air-mass evolution; no
  `reset_air_mass_each_window` is set.
- All three outputs are 13-frame, 72-hour C180/L85 F32 GPU snapshots.
- Python/netCDF4 finite scan passed for `co2_natural`, `co2_fossil`,
  both column means, and `air_mass` across all 13 snapshots.

## ERA5-on-GEOS-Native vs GEOS-IT Native Movies

These are side-by-side comparison movies with common color ranges. Each frame
has three panels: ERA5-on-GEOS-native, GEOS-IT native, and `GEOS-IT - ERA5`.
The ERA/GEOS panels share one robust color range per movie; the difference
panel uses a symmetric robust range.

Comparison movie outputs:

- Directory: `/tmp/tm5_smoke/viz_era_geos_c180_comparison_movies`
- README: `/tmp/tm5_smoke/viz_era_geos_c180_comparison_movies/README.txt`
- Run log: `/temp1/c180_era5_geosgrid_cfl85_3d/logs/viz_era_geos_comparison_movies_allruns_20260504T114248.log`
- Script: `scripts/visualization/run_c180_era_geos_comparison_movies.sh`
- Generator: `scripts/visualization/compare_c180_era_geos_movies.jl`

Generated column-mean MP4s:

- `advonly_ppm_co2_natural_column_mean_era_geos_compare.mp4`
- `advonly_ppm_co2_fossil_column_mean_era_geos_compare.mp4`
- `advdiff_ppm_co2_natural_column_mean_era_geos_compare.mp4`
- `advdiff_ppm_co2_fossil_column_mean_era_geos_compare.mp4`
- `fullphysics_ppm_co2_natural_column_mean_era_geos_compare.mp4`
- `fullphysics_ppm_co2_fossil_column_mean_era_geos_compare.mp4`

## PPM Run Configs

- Adv only: `artifacts/c180_3d_ppm_advonly.toml`
- Adv + PBL diffusion: `artifacts/c180_3d_ppm_advdiff.toml`
- Full physics: `artifacts/c180_3d_ppm_fullphysics.toml`

## PPM NetCDF Outputs

- Adv only: `/temp1/c180_full137_3d/advonly_ppm.nc`
- Adv + PBL diffusion: `/temp1/c180_full137_3d/advdiff_ppm.nc`
- Full physics: `/temp1/c180_full137_3d/fullphysics_ppm.nc`

All three PPM outputs are 13-frame, 72-hour C180/L137 F32 GPU snapshots.
The 2026-05-03 TM5-area rerun is finite for all tracers/snapshots. ERA
adv+diff and full physics are no longer identical: full physics now has a
nonzero TM5 convection signal relative to adv+diff.

## PPM Logs

- Adv only: `/temp1/c180_full137_3d/logs/advonly_ppm_tm5areas_20260503T183011.log`
- Adv + PBL diffusion: `/temp1/c180_full137_3d/logs/advdiff_ppm_tm5areas_20260503T183011.log`
- Full physics: `/temp1/c180_full137_3d/logs/fullphysics_ppm_tm5areas_20260503T183011.log`

## PPM Visualizations

- Visualization directory: `/tmp/tm5_smoke/viz_3d_ppm`
- Visualization summary: `/tmp/tm5_smoke/viz_3d_ppm_summary.txt`
- Visualization rerun log: `/temp1/c180_full137_3d/logs/viz_3d_ppm_tm5areas_20260503T183011.log`
- Generated inventory: 24 PNG grids and 6 MP4 column-mean movies.

Naming pattern:

- PNG: `<run>_<tracer>_<slice>.png`
- MP4: `<run>_<tracer>_column_mean.mp4`

Runs:

- `advonly_ppm`
- `advdiff_ppm`
- `fullphysics_ppm`

Tracers:

- `co2_natural`
- `co2_fossil`

PNG slices:

- `surface`
- `mid_trop` (`level_slice --level 100`)
- `upper_trop` (`level_slice --level 70`)
- `column_mean`

## ERA5-vs-GEOS Comparison Diagnostics

Current comparison outputs:

- Window tendency diagnostics: `/temp1/c180_era_geos_window_tendencies`
- Window tendency run log: `/temp1/c180_era_geos_window_tendencies_tm5areas_20260503T183011.log`
- Refreshed IC/cumulative-increment diagnostics: `/temp1/c180_era_geos_ic_divergence`
- Refreshed IC/cumulative-increment run log: `/temp1/c180_era_geos_ic_divergence_tm5areas_20260503T183011.log`

Window tendency files:

- `metadata.csv`
- `all_sim_window_tendency_metrics.csv`
- `all_sim_window_tendency_summary.csv`
- `matched_era_geos_window_tendency_summary.csv`
- `matched_era_geos_mean_corr_6h.png`
- `matched_era_geos_mean_corr_24h.png`
- `matched_era_geos_mean_rmse_6h.png`
- `matched_era_geos_mean_rmse_24h.png`
- `within_path_mean_corr_6h.png`
- `within_path_mean_corr_24h.png`
- `within_path_mean_rmse_6h.png`
- `within_path_mean_rmse_24h.png`

These diagnostics compare `dCO2/dt` on a 180 x 90 common lat-lon grid for
6, 12, and 24 hour snapshot lags. Diagnostics include column mean, surface,
and pressure-matched 850, 500, and 250 hPa slices.

Post-TM5-fix comparison highlights:

- ERA adv+diff vs ERA full-physics 6 h tendency, natural CO2:
  column-mean RMSE `0.0061356 ppm/hr`, surface RMSE `0.19948 ppm/hr`.
- Matched ERA full-physics vs GEOS full-physics 6 h tendency, natural CO2:
  column-mean correlation `0.19798`, column-mean RMSE `0.22076 ppm/hr`.
- Final 72 h common-grid natural-column increment correlation:
  adv-only `0.85831`, adv+diff `0.87738`, full-physics `0.64477`.

## ERA5-vs-GEOS Binary Mass-Flux Audit

Current binary-level audit outputs:

- Audit script: `scripts/diagnostics/compare_c180_binary_mass_fluxes.jl`
- Audit directory: `/temp1/c180_binary_mass_flux_audit`
- Smoke-test directory: `/temp1/c180_binary_mass_flux_audit_smoke`

The audit streams all 2021-12-02 through 2021-12-04 C180 daily binaries and
compares ERA5 vs GEOS-IT implied horizontal mass-flux speeds at 850, 500, and
250 hPa on a 180 x 90 common lat-lon grid. It also writes edge/interior,
roughness, temporal-jump, and vertical `cm` absolute-rate summaries.

Audit files:

- `binary_metadata.csv`
- `global_speed_stats.csv`
- `common_grid_era_geosit_speed_metrics.csv`
- `common_grid_speed_roughness.csv`
- `common_grid_temporal_speed_jumps.csv`
- `edge_vs_interior_speed_stats.csv`
- `vertical_cm_abs_rate_stats.csv`
- `worst_pair_speed_map_*hpa.csv`
- `worst_pair_speed_map_*hpa.png`
- `common_grid_speed_rmse_corr_timeseries.png`
- `common_grid_speed_roughness_ratio_timeseries.png`

Binary-audit highlights:

- Headers agree on dry-mass `substep_mass_amount` semantics and
  window-start/end mass sampling, but the products differ structurally:
  ERA5 is C180/L137 equiangular gnomonic with 24 substeps per hour; GEOS-IT is
  native C180/L72 GMAO geometry with 8 substeps per hour.
- Horizontal implied speeds do not show a global scale-factor error. Mean
  common-grid ERA5 vs GEOS-IT correlations across 72 windows are `0.8896`
  at 850 hPa, `0.9604` at 500 hPa, and `0.9523` at 250 hPa.
- Mean common-grid speed bias (`GEOS-IT - ERA5`) is `-0.130 m/s` at 850 hPa,
  `-0.171 m/s` at 500 hPa, and `-2.294 m/s` at 250 hPa, so the largest
  systematic horizontal difference is upper-tropospheric ERA5 being faster.
- Common-grid neighbor roughness ratios stay near 1.0, so the audit did not
  find a strong ERA-only ringing/roughness excess in horizontal mass fluxes.
- Vertical `cm` differs by orders of magnitude because the paths use different
  closure rules: GEOS-IT uses FV3 pressure-fixer `cm`, while ERA5/LL-to-CS
  uses Poisson-balanced continuity `cm`. Treat vertical `cm` comparison as a
  closure-semantics diagnostic, not a direct source-wind agreement metric.

## ERA5-on-GEOS-Native Vector Projection Finding

The ERA5-on-GEOS-native C180 binaries generated before 2026-05-04 should be
treated as suspect for directional plume interpretation. The LL-to-CS path was
projecting geographic ERA winds onto local cubed-sphere tangent directions and
then reconstructing `am`/`bm` as if those components were face-normal
velocities. GEOS-native C180 cells are non-orthogonal, so tangent projections
inject cross-components into the stored face fluxes.

South Africa hotspot spot check:

- Nearest GEOS-native C180 cell to lon 27E, lat 26S:
  panel `1`, `i=162`, `j=31`, lon `26.959`, lat `-25.966`.
- Local tangent dot product `ex dot ey = 0.313`.
- ERA5 LL source wind, 2021-12-02 window 2 near 850 hPa:
  east `-2.761 m/s`, north `-2.032 m/s`, bearing `233.6 deg`.
- Old tangent-as-normal projection gives runtime-equivalent bearing
  `225.4 deg`, an `8.2 deg` direction error and `1.335x` speed error at that
  cell.
- Region lon `15..35E`, lat `35..15S`, 850 hPa:
  mean tangent non-orthogonality `0.291`, mean direction error `10.7 deg`,
  p95 direction error `21.8 deg`, max `28.1 deg` for winds above `1 m/s`.

Fix applied in code:

- `src/Preprocessing/cs_wind_rotation.jl` now projects geographic winds
  onto face normals derived from the panel tangent basis before reconstructing
  CS face fluxes.
- `rotate_panel_to_geographic!` now solves the inverse face-normal Gram system.
- `src/Preprocessing/spectral_synthesis.jl` comments now clarify the ERA5
  meridional pseudo-wind/cosine convention; the LL cosine convention was not
  the primary problem.
- `scripts/preprocessing/regrid_ll_transport_binary_to_cs.jl` now accepts the
  test-used `--cache-dir` option.

Implication: regenerate `/temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps8`
and rerun the ERA5-on-GEOS-native PPM simulations/movies before trusting the
ERA-vs-GEOS plume-direction differences.

### Active Regeneration Handoff

Started: 2026-05-04 17:22 PDT.

Detached session:

- `tmux` session: `era5_geosnative_regen_facefix_20260505T0023`
- Driver log:
  `/temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps8/logs/regenerate_all_after_face_normal_fix_20260505T0023_tmux.log`
- Command running inside the session:
  `THREADS=16 START_DAY=02 END_DAY=04 STEPS_PER_WINDOW=8 FLOAT_TYPE=Float32 ./scripts/preprocessing/build_era5_geos_c180_cfl85_tm5_surface.sh`

Expected regenerated binaries:

- `/temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps8/era5_transport_20211202_merged1000Pa_float32.bin`
- `/temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps8/era5_transport_20211203_merged1000Pa_float32.bin`
- `/temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps8/era5_transport_20211204_merged1000Pa_float32.bin`

Per-day regrid logs overwritten by the script:

- `/temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps8/logs/regrid_geosnative_20211202_steps8.log`
- `/temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps8/logs/regrid_geosnative_20211203_steps8.log`
- `/temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps8/logs/regrid_geosnative_20211204_steps8.log`

Return checks:

- `tmux attach -t era5_geosnative_regen_facefix_20260505T0023`
- `tmux list-sessions | rg era5_geosnative_regen_facefix_20260505T0023`
- `tail -f /temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps8/logs/regenerate_all_after_face_normal_fix_20260505T0023_tmux.log`
- `tail -f /temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps8/logs/regrid_geosnative_20211202_steps8.log`
- `ls -lh /temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps8/*.bin`

At handoff time, the session was running the 2021-12-02 LL-to-GEOS-native C180
regrid into:

- `/temp1/c180_era5_geosgrid_cfl85_tm5_surface_f32_steps8/era5_transport_20211202_merged1000Pa_float32.bin.tmp`

After completion, rerun the ERA5-on-GEOS-native PPM campaign and comparison
movies listed above; the existing NetCDFs and movies still reflect the old
tangent-as-normal projection.

## Old Slopes Outputs

These are the contaminated pre-fix/slopes outputs and should not be used for interpretation:

- `/temp1/c180_full137_3d/advonly.nc`
- `/temp1/c180_full137_3d/advdiff.nc`
- `/temp1/c180_full137_3d/fullphysics.nc`
- Old visualizations: `/tmp/tm5_smoke/viz_3d`

## Notes

- Standard split `scheme = "ppm"` passed the flat-field no-flux probe.
- `scheme = "linrood"` produced NaNs in the flat-field probe and should not be used for these diagnostics yet.
- The original TM5 Fortran path (`deps/tm5-cy3-4dvar/base/src/tm5_conv.F90`) expects `m(k)` in kg/m². Production TM5 workspaces now carry cell-area metrics and convert `state.air_mass / cell_area` before building the matrix.
