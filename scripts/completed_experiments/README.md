# Completed experiments

Scripts from closed-loop investigations — each one corresponds to a
specific plan-era experiment that has since landed, been validated,
or been superseded. Kept for reference value in case a similar
question is asked again; not part of any active workflow. Most hard-code the
dates, binaries and output directories of their investigation, which may no
longer exist; repository-relative includes resolve from this folder.

If you're writing new visualization, start from the canonical scripts
in `scripts/visualization/` (animate_catrine_vs_geoschem.jl,
animate_gchp_v4_fullphys.jl, animate_era5_vs_geoschem.jl, etc.).

## Index

**Hybrid PE → direct cumsum PE (fix landed 2026-03-13):**
- `animate_hybrid_pe_vs_geoschem.jl`

**Mass fixer development (Invariant 11):**
- `animate_fixer_vs_nofixer.jl`

**Plan 14 vertical remap study:**
- `animate_vremap_vs_strang_vs_geoschem.jl`
- `animate_vremap_vs_geoschem_co2.jl`
- `animate_vremap_diff_vs_geoschem.jl`
- `animate_perremap_comparison.jl`

**Plan 13/14 scheme experiments:**
- `animate_linrood_vs_geoschem.jl`
- `animate_advonly_3way.jl`
- `animate_advonly_vs_ord7damp.jl`
- `animate_qspace_vs_geoschem_fast.jl`

**GCHP path validation (Plans 14-17):**
- `animate_gchp_flat_nsub.jl`
- `animate_gchp_vs_geoschem.jl`

**CATRINE variant explorations:**
- `animate_catrine_ord7damp.jl` / `animate_catrine_ord7damp_co2.jl`
- `animate_catrine_comparison.jl` / `_co2.jl` / `_natural_co2.jl`

**Other:**
- `animate_local_vs_nonlocal_pbl.jl` — PBL variant comparison.
- `animate_lmdz_emissions.jl` — LMDZ emissions visualization.
- `animate_era5_merged_vs_geoschem.jl` — ERA5 merged-level validation
  intermediate.

**GEOS native mass-flux UTLS fingering (2026-06; developer note
`GEOS_MASS_FLUX_UTLS_FINGERING.md`):**
- `cm_closure_headtohead.jl`, `fingerfix_proto_anisotropic-divergence-damping.jl`,
  `fingerfix_proto_cm-hyperdiffusion-via-potential.jl`,
  `fingerfix_proto_two-window-time-averaged-flux.jl`, `prototype_pfix_balanced.jl`,
  `prototype_remap_vs_cm.jl`, `compare_moisturefiltered_vs_endpoint.jl`
- `cx_implied_dp_vs_delp.jl`, `moisture_residual_I1_vs_A1.jl`, `moist_budget_IT_vs_FP.jl`,
  `iau_signature_M.jl`, `M_vs_DMDTANA_geosfp.jl`, `omega_midnight_pchip_proof.jl`
- `omega_vs_cm_driver_roughness.jl`, `merra2_vs_geos_divh_roughness.jl`,
  `wind_vs_mfxc_divh_roughness.jl`
- `finger_era5_vs_geos.py`, `finger_pfix_vs_endpoint.py`, `finger_route1_dec11.py`,
  `omega_tracer_finger_and_mass.py`

**C180 ERA5 vs GEOS winds and mass fluxes (2026-05; `docs/c180_ppm_campaign_locations.md`):**
- `compare_c180_binary_mass_fluxes.jl`, `compare_c180_binary_winds.jl`,
  `compare_c180_era_geos_ic_divergence.jl`, `compare_c180_window_tendencies.jl`,
  `plot_c180_window_tendency_summary.py`
- `compare_era_c180_binary_to_raw_uv.jl`, `compare_spectral_native_vs_cds_ll.jl`,
  `diagnose_south_africa_wind_profiles.jl`, `isolate_era_ll_to_cs_wind_error.jl`,
  `compare_geos_raw_winds_to_binary.py`
- `run_c180_3day_ppm_viz.sh`, `run_c180_era5_geosnative_cfl85_3day_ppm_viz.sh`,
  `run_c180_geosit_native_3day_ppm_viz.sh`, `run_c180_era_geos_comparison_movies.sh`

**December 2021 campaign checks against GEOS-Chem:**
- `compare_to_geoschem_dec1-5.py`, `meridional_curtains_dec1-5.py`,
  `meridional_curtains_fullphys_dec1-3.py`, `plot_column_mean_maps.py`,
  `animate_column_mean_3way_dec.py`, `tracer_mass_balance_vs_gc.py`,
  `tropical_pbl_diurnal_co2.py`, `_finish_era5_and_animate.sh`
- `verify_emissions_vs_geoschem.jl`, `verify_emissions_lmdz_timing.jl` (gating
  checks before the three-month campaign)

**Advection, diffusion, convection and full-physics profile plots (four-tracer
comparisons):**
- `plot_{advection,diffusion,fullphysics}_profiles_per_panel_percentiles{,_linear}.jl`
- `plot_convection_profiles_per_panel{,_overlay,_percentiles,_percentiles_linear}.jl`

**TM5 convection optimization rounds (2026-05; benchmarks):**
- `bench_tm5_alternatives.jl`, `bench_tm5_collab_lu.jl`, `bench_tm5_p4_bucketed.jl`,
  `bench_tm5_p6_lu_cache.jl`, `bench_tm5_p8_schur.jl`, `bench_tm5_p13_cache_bottom.jl`,
  `bench_tm5_p17_layout.jl`, `_probe_panel_depths.jl`

**Early scheme comparisons and probes:**
- `animate_4panel_comparison.jl`, `animate_comparison.jl`,
  `animate_linrood_comparison.jl`, `animate_tiedtke_vs_ras.jl`
- `advresln_vertical_redistribution.jl`, `animate_noise_growth.py`,
  `animate_wet_dry.py`, `mwe_cr_regridding.jl`, `linrood_la_footprint_c180.jl`
- `probe_f64_day_boundary.jl` (plan 39, dry-basis `cm` closure),
  `run_tm5_2day_demo.jl` (plan 24 demo)

**Campaign launchers:**
- `build_catrine_c30_2022_macbook_bundle.sh`, `finish_catrine_c30_2022_fullphysics.sh`
- `run_trendy_s3_2021_batches.sh`, `plot_trendy_allmodels_nee_decomposition.py`
