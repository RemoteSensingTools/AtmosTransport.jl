# Heritage visualization

Scripts in `heritage/` are not part of a maintained workflow. They are kept
for reference, as the method or evidence of a finished study, until they are
trimmed. They may hard-code the paths, dates and data of that study, and they
are not tested; check imports and inputs before running one. Retired scripts
are listed under "Retired scripts" in [`scripts/README.md`](../../README.md).

| Script | Purpose | Why it is here |
|---|---|---|
| `animate_fluxcom_sif_ter_xco2_panel.py` | 2x3 animation of the 2021 FLUXCOM-X, SIF-GPP and TER XCO2 contributions (400 ppm carriers removed) from daily split C90 output | Figures of the single 2021 FLUXCOM/SIF/TER C90 study; reads current output variables |
| `animate_jules_components_xco2.py` | Animates JULES-ES GPP-Ra, Rh and GFED5 fire XCO2 components on C30 (absolute or IAV anomaly); provides `backfit`, `interpolate_frames`, `make_mapper` | JULES-ES component study. Its input `c30_maps_jules_fire.npz` came from a scratch directory, not a tracked script. Imported by `animate_jules_flux_vs_xco2.py` and `animate_xco2_daily.py` |
| `animate_jules_flux_vs_xco2.py` | 3x2 animation of JULES-ES flux anomalies (GPP-Ra, Rh, NEE) above the XCO2 anomalies they produce | Same study and scratch `.npz` inputs; imports `animate_jules_components_xco2.py` |
| `animate_transcom_xco2_daily.py` | Animates daily TransCom-region SIF-GPP contributions (`co2_sif_<region>`) to transported XCO2 | Figures of the Dec-2021 SIF-GPP TransCom C180 pilot (`config/runs/sif_gpp_transcom_c180_dec2021_daily*.toml`); inputs come from `scripts/diagnostics/extract_cs_xco2*.jl` |
| `animate_trendy_s3_xco2_atlas.py` | 24-panel TRENDY S3 GPP/TER/NEE XCO2 ensemble atlases (multi-model mean plus 23 model departures) | 2021 TRENDY v14 S3 GPP/TER C90 study (ATBD v0.1, 2026-07-22). Imports `animate_trendy_s3_xco2_ensemble` from `../` (added to `sys.path`), whose 2021 tracer and file names differ from the 2014-2024 rerun |
| `animate_xco2_daily.py` | Two-panel daily XCO2 animation of the C30 JULES-ES NEE run vs its climatological-flux control | Reads a scratch `c30_daily_grids.npz`; imports `make_mapper` from `animate_jules_components_xco2.py`. The decomposition itself is `scripts/diagnostics/heritage/decompose_transport_vs_flux_iav.py` |
| `compare_c180_era_geos_movies.jl` | Side-by-side C180 ERA5, GEOS-IT and difference movies with shared colour ranges (env-configured dirs, runs, tracers) | Recipe for two-met-source difference movies using only exported Visualization names. Defaults point at finished `/temp1` C180 campaign directories |
| `ocean_co2_viz.py` | ECCO-Darwin ocean-CO2 December plots: two-panel MP4 (daily flux, `co2_ocean` XCO2 anomaly) and a three-panel two-week-mean PNG | Figures of the finished ECCO-Darwin Dec-2021 study; inputs come from `scripts/diagnostics/extract_cs_column_means.jl` and `scripts/preprocessing/eccodarwin_co2flux_to_latlon.py` |
| `plot_transcom_xco2_daily.py` | Static daily TransCom-tagged SIF-GPP XCO2 attribution plots | Static companion of `animate_transcom_xco2_daily.py` for the same pilot |
