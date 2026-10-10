# Scripts

Command-line entry points and tools around the package. Library code lives in
`src/`; a script that needs a reusable function should call the library (or
move the function there), not copy it.

| Folder | Contents |
|---|---|
| [`run_transport.jl`](run_transport.jl) | The runtime entry point: `julia --project=. scripts/run_transport.jl <config.toml>` |
| [`preprocessing/`](preprocessing/) | Building transport binaries (`preprocess_transport_binary.jl`, the canonical CLI), regridding, coarsening and attaching fields, year-batch drivers, and preparing surface-flux inputs |
| [`downloads/`](downloads/) | `download_data.jl` with the recipes in `config/downloads/` |
| [`diagnostics/`](diagnostics/) | Inspection and comparison tools that take paths as arguments (`inspect_transport_binary.jl`, the CATRINE benchmark suite against GEOS-Chem, mass-balance and Float32 checks, XCO2 extraction and Hovmoller plots) |
| [`validation/`](validation/) | Checks of binaries and runs: replay continuity (`verify_cs_binary_continuity.jl`), cubed sphere vs lat-lon (`compare_cs_vs_ll.jl`), TM5 convection active layers (`diagnose_tm5_active_layers.jl`) |
| [`visualization/`](visualization/) | Plots and animations; `atmos_viz.jl` is the topology-aware snapshot CLI |
| [`benchmarks/`](benchmarks/) | Synthetic cubed-sphere advection and TM5 convection benchmarks; recorded experiments with their evidence in `benchmarks/results/` |
| [`postprocess/`](postprocess/), [`inversions/`](inversions/), [`checks/`](checks/) | Snapshot conversion, inversion prototypes, documentation checks |

A `heritage/` subfolder (in `scripts/` itself and in `benchmarks/`,
`diagnostics/`, `downloads/`, `preprocessing/`, `validation/` and
`visualization/`) holds scripts that are not part of a maintained workflow.
They are kept for reference until they are trimmed, may hard-code the paths
and data of their study, and are not tested. Each `heritage/README.md` gives
the purpose of every script and why it is kept.

The campaign launchers at the top level (`run_*_campaign.py`,
`run_catrine_c30_*.sh`, `profile_catrine_c30_2021_fullphysics.sh`,
`run_campaign5d.sh`) drive specific production campaigns; their defaults point
at the machines and directories of those campaigns.

## Retired scripts

The scripts below were removed from the tree; git history keeps them. Recover
one with `git show 7c515038:<path>`, where `<path>` is the folder heading plus
the file name (for example
`git show 7c515038:scripts/deprecated/run_cs_driven.jl`). The former
`completed_experiments/` and `deprecated/` folders are gone: kept scripts from
`completed_experiments/` moved to the `heritage/` subfolder of the folder that
fits their purpose (for example `diagnostics/heritage/`;
`finish_catrine_c30_2022_fullphysics.sh` went to `preprocessing/`), and the
`deprecated/` runner shims are replaced by `run_transport.jl`.

### `scripts/benchmarks/`

| Path | Purpose |
|---|---|
| `bench_adjoint_memory.jl` | CS adjoint tape-size estimate for several grids |

### `scripts/benchmarks/results/`

| Path | Purpose |
|---|---|
| `main_device_workspace_v100_20260905/compare.jl` | Exact-equality check of before/after outputs |
| `main_device_workspace_v100_20260905/construction.jl` | Model-construction order A/B in one process |
| `main_device_workspace_v100_20260905/profile.jl` | V100 2-hour real-input C90 run (6/32 tracers) |
| `main_direct_packing_v100_20260905/compare.jl` | Exact-equality check of pressure-init vs direct-packing outputs |
| `main_direct_packing_v100_20260905/halo_probe.jl` | Halo-packing probe on the C90 GEOS-native mesh |
| `main_direct_packing_v100_20260905/initialization.jl` | CPU A/B of IC construction vs direct packing |
| `main_direct_packing_v100_20260905/profile.jl` | V100 2-hour real-input C90 run (6/32 tracers) |
| `main_dkg_mass_v100_20260905/ablation.jl` | V100 operator ablation of the split-diffusion mass residual |
| `main_dkg_mass_v100_20260905/check_outputs.jl` | Full-day Dkg output checks vs the split baseline |
| `main_dkg_mass_v100_20260905/check_profile_outputs.jl` | 32-tracer Dkg profile output checks vs the split baseline |
| `main_dkg_mass_v100_20260905/check_totals.jl` | Recheck archived hourly mass totals (no GPU or met input) |
| `main_dkg_mass_v100_20260905/column_probe.jl` | Conservative Dkg diffusion on real C90 columns |
| `main_dkg_mass_v100_20260905/factorization_probe.jl` | Dkg column factorization accuracy on real C90 columns |
| `main_dkg_mass_v100_20260905/full_day.jl` | V100 full-day mass-drift run (Float32/Float64) |
| `main_dkg_mass_v100_20260905/production_columns.jl` | Dkg mass behaviour on 216 real columns with layer impulses |
| `main_dkg_mass_v100_20260905/profile.jl` | V100 full-day timing, before/after conservative Dkg |
| `main_dkg_mass_v100_20260905/weak_exchange.jl` | Synthetic weak-exchange check of Dkg mass diffusion |
| `main_dkg_parallel_v100_20260906/check_outputs.jl` | CPU check of before/after full-day outputs |
| `main_dkg_parallel_v100_20260906/launch_layout.jl` | Seven 2-D thread layouts for the Dkg kernel |
| `main_dkg_parallel_v100_20260906/parallel_tracers.jl` | Fused column loops vs separate factor/solve kernels |
| `main_dkg_parallel_v100_20260906/profile.jl` | V100 full-day timing, before/after parallel Dkg kernels |
| `main_dkg_parallel_v100_20260906/small_batches.jl` | Dkg kernel choice for 2-4 tracer batches |
| `main_f64_ppm_tiles_v100_20260906/check_outputs.jl` | CPU check of Float64 PPM tile-size day outputs |
| `main_f64_ppm_tiles_v100_20260906/profile.jl` | V100 Float64 full-day run per PPM tile (256, 32, 32x2) |
| `main_f64_ppm_tiles_v100_20260906/summarize.jl` | Recompute the day table from archived samples |
| `main_f64_ppm_tiles_v100_20260906/sweep_tiles.jl` | Float64 PPM tile-shape sweep (CUDA events) |
| `main_initial_reuse_20260906/check_outputs.jl` | CPU check of before/after full-day outputs |
| `main_initial_reuse_20260906/initialization.jl` | IC construction timing with buffer reuse (format-4 ERA5 input) |
| `main_initial_reuse_20260906/profile.jl` | V100 full-day timing, before/after buffer reuse |
| `main_input_v100_20260905/compare.jl` | Exact-equality check of output-port vs input-port outputs |
| `main_input_v100_20260905/profile.jl` | V100 2-hour real-input C90 run (6/32 tracers) |
| `main_output_v100_20260905/compare.jl` | Exact-equality check of baseline vs output-port outputs |
| `main_output_v100_20260905/profile.jl` | V100 2-hour real-input run |
| `main_ppm_day_v100_20260905/compare.jl` | Exact-equality check of before/after full-day PPM outputs |
| `main_ppm_day_v100_20260905/initialization_phases.jl` | CPU timing of IC construction phases |
| `main_ppm_day_v100_20260905/profile.jl` | V100 full-day PPM run, before/after |
| `main_ppm_tiles_v100_20260905/compare.jl` | Before/after PPM-tile output comparison |
| `main_ppm_tiles_v100_20260905/profile.jl` | V100 2-hour run, before/after the PPM tile change |
| `main_ppm_tiles_v100_20260905/sweep_blocks.jl` | Scalar PPM workgroup-size sweep (32-256) |
| `main_ppm_tiles_v100_20260905/sweep_tiles.jl` | 2-D PPM tile-shape sweep |
| `main_ppm_tiles_v100_20260905/sweep_trial.jl` | Rejected tracer-per-thread launch trial |
| `main_ppm_tiles_v100_20260905/trial_kernels.jl` | Rejected parallel-x PPM kernels used by the sweeps |
| `main_pressure_init_v100_20260905/compare.jl` | Exact-equality check of device-workspace vs pressure-init outputs |
| `main_pressure_init_v100_20260905/equivalence.jl` | New pressure-layer IC vs the method of commit 9544a3d3 |
| `main_pressure_init_v100_20260905/initialization_probe.jl` | CPU probe of pressure-layer IC (synthetic C90 L66) |
| `main_pressure_init_v100_20260905/profile.jl` | V100 2-hour run for the pressure-init change |
| `main_real_input_v100_20260905/profile.jl` | V100 2-window real-input run (PPM, tm5_dkg, collaborative convection) |
| `main_real_input_v100_20260905/verify.jl` | Checks of the six real-input output files |
| `main_release_adjoint_v100_20260906/fresh-resolve.jl` | Fresh-environment resolve of test/Project.toml compat |
| `main_release_adjoint_v100_20260906/gpu-checks.jl` | Selected GPU diagnostic tests from an isolated export |
| `main_release_adjoint_v100_20260906/gpu-transport.jl` | GPU transport check reusing committed test fixtures |
| `main_release_output_review_20260906/benchmark-smoke.jl` | Smoke run of benchmarking/run_benchmarks.jl in a temp env |
| `main_release_output_review_20260906/hdf5-compat.jl` | Output tests against alternative HDF5/NCDatasets versions |
| `main_split_mass_seams_v100_20260905/check_outputs.jl` | Completeness and finite-field check of split-seam outputs |
| `main_split_mass_seams_v100_20260905/check_profile_outputs.jl` | Initial-total and finite-field check of profile outputs |
| `main_split_mass_seams_v100_20260905/check_totals.jl` | Hourly totals check; writes drift_summary.csv |
| `main_split_mass_seams_v100_20260905/full_day.jl` | V100 full-day mass-drift run (Float32/Float64) |
| `main_split_mass_seams_v100_20260905/profile.jl` | V100 full-day timing, before/after the split seam change |

### `scripts/completed_experiments/`

| Path | Purpose |
|---|---|
| `_finish_era5_and_animate.sh` | One-shot finisher of the Dec-2021 ERA5 rerun (deficits, 3-way animation) |
| `_probe_panel_depths.jl` | Per-panel TM5 convection depth distributions (P13 cache sizing) |
| `advresln_vertical_redistribution.jl` | BL/FT/UT mass split of a BL tracer across 5/2.5/1 degree ERA5 runs |
| `animate_4panel_comparison.jl` | ERA5 vs GEOS-IT C180 CO2 GIF, surface and ~800 hPa (June 2023) |
| `animate_advonly_3way.jl` | Adv-only CO2 GIF: dry q, moist q, dry rm space |
| `animate_advonly_vs_ord7damp.jl` | GEOS-Chem vs ORD7+damp vs adv-only CO2 GIF |
| `animate_catrine_comparison.jl` | GEOS-Chem vs AT fossil-CO2 GIF (CATRINE D7.1) |
| `animate_catrine_comparison_co2.jl` | GEOS-Chem vs AT total-CO2 GIF (CATRINE D7.1) |
| `animate_catrine_comparison_natural_co2.jl` | GEOS-Chem vs AT natural-CO2 GIF |
| `animate_catrine_ord7damp.jl` | CATRINE ORD7+damp vs GEOS-Chem fossil-CO2 GIF |
| `animate_catrine_ord7damp_co2.jl` | CATRINE ORD7+damp vs GEOS-Chem total-CO2 GIF |
| `animate_column_mean_3way_dec.py` | Dec-campaign column-mean movie: GEOS-Chem, GEOS-IT-omega, ERA5 |
| `animate_comparison.jl` | ERA5 vs GEOS-IT C180 column-mean CO2 GIF (June 2023) |
| `animate_era5_merged_vs_geoschem.jl` | ERA5 68-level merged run vs GEOS-Chem C180 movie |
| `animate_fixer_vs_nofixer.jl` | Mass fixer vs no fixer vs GEOS-Chem CO2 GIF |
| `animate_gchp_flat_nsub.jl` | Flat-IC GIF: n_sub loop+remap vs single-step GCHP path |
| `animate_gchp_vs_geoschem.jl` | GEOS-Chem vs AT GCHP-path total-CO2 GIF |
| `animate_hybrid_pe_vs_geoschem.jl` | Hybrid-PE AT vs GEOS-Chem surface GIF (4 species) |
| `animate_linrood_comparison.jl` | Strang vs Lin-Rood full-physics fossil-CO2 GIF |
| `animate_linrood_vs_geoschem.jl` | Lin-Rood variant vs GEOS-Chem CO2 GIF |
| `animate_lmdz_emissions.jl` | 3-day LMDZ CO2 emission animation from a CS emission binary |
| `animate_local_vs_nonlocal_pbl.jl` | Local K vs non-local Holtslag-Boville PBL fossil-CO2 GIF |
| `animate_noise_growth.py` | Noise growth from a flat dry-VMR IC under moist transport |
| `animate_perremap_comparison.jl` | 1 vs 8 vertical remaps per hour GIF |
| `animate_qspace_vs_geoschem_fast.jl` | Fast q-space vs GEOS-Chem CO2 GIF |
| `animate_tiedtke_vs_ras.jl` | Tiedtke vs RAS convection fossil-CO2 GIF |
| `animate_vremap_diff_vs_geoschem.jl` | Lin-Rood + vertical remap + PBL diffusion vs GEOS-Chem GIF |
| `animate_vremap_vs_geoschem_co2.jl` | Vertical-remap AT vs GEOS-Chem CO2 GIF |
| `animate_vremap_vs_strang_vs_geoschem.jl` | GEOS-Chem vs Strang vs vertical-remap CO2 GIF |
| `animate_wet_dry.py` | Wet vs dry ERA5 transport animation |
| `bench_tm5_alternatives.jl` | CUDA alternatives to the per-thread TM5 convection LU |
| `bench_tm5_collab_lu.jl` | Collaborative TM5 convection LU prototype and benchmark |
| `bench_tm5_p13_cache_bottom.jl` | TM5 convection P13: cache only the bottom Schur block |
| `bench_tm5_p17_layout.jl` | TM5 convection P17: transposed LU-cache layout |
| `bench_tm5_p4_bucketed.jl` | TM5 convection P4: depth-bucketed collaborative LU |
| `bench_tm5_p6_lu_cache.jl` | TM5 convection P6: persistent per-window LU cache |
| `bench_tm5_p8_schur.jl` | TM5 convection P8: Hessenberg + Schur-complement LU split |
| `cm_closure_headtohead.jl` | Endpoint vs FV3 pressure-fixer cm closure on one GEOS-IT window |
| `compare_c180_binary_winds.jl` | C180 binary winds vs raw GEOS A3dyn and ERA5 spectral winds |
| `compare_c180_era_geos_ic_divergence.jl` | ERA5 vs GEOS-IT C180 run-output differences |
| `compare_era_c180_binary_to_raw_uv.jl` | ERA5-on-GEOS-native C180 winds vs CDS pressure-level u/v |
| `compare_geos_raw_winds_to_binary.py` | GEOS-IT C180 binary vs raw native MFXC/winds |
| `compare_moisturefiltered_vs_endpoint.jl` | `:endpoint_balanced` vs `:moisture_filtered` cm closures |
| `compare_spectral_native_vs_cds_ll.jl` | ERA5 spectral vs CDS vs LL-binary winds (C180 wind error) |
| `compare_to_geoschem_dec1-5.py` | SH-UTLS RMS and noise of ours minus GEOS-Chem, Dec 2-6 |
| `diagnose_south_africa_wind_profiles.jl` | ERA5 vs GEOS C180 wind profiles in the South Africa plume |
| `finger_pfix_vs_endpoint.py` | SH-UTLS tracer roughness, pfix_corrected vs endpoint closure |
| `finger_route1_dec11.py` | Route-1 SH-UTLS roughness: GEOS MFXC vs MERRA-2 vs ERA5 |
| `fingerfix_proto_anisotropic-divergence-damping.jl` | Fingering fix prototype: roughness-gated biharmonic div_h damping |
| `fingerfix_proto_two-window-time-averaged-flux.jl` | Fingering fix prototype: two-window time-averaged MFXC |
| `iau_signature_M.jl` | Tests whether GEOS-IT residual M is the IAU increment |
| `isolate_era_ll_to_cs_wind_error.jl` | Locates the ERA5 wind error in the LL to C180 pipeline |
| `M_vs_DMDTANA_geosfp.jl` | IAU test: GEOS-FP C720 residual M vs DMDTANA mass tendency |
| `meridional_curtains_dec1-5.py` | Lat-pressure curtains: GEOS-Chem vs GEOS MFXC vs MERRA-2 (adv-only) |
| `merra2_vs_geos_divh_roughness.jl` | SH-UTLS div_h roughness: native GEOS MFXC vs MERRA-2 winds |
| `moisture_residual_I1_vs_A1.jl` | Splits residual M into PS/DELP time-sampling parts |
| `mwe_cr_regridding.jl` | Minimal ConservativeRegridding.jl LL-to-reduced-grid mass check |
| `omega_midnight_pchip_proof.jl` | Checks that omega_consistent PCHIP targets cross midnight |
| `omega_vs_cm_driver_roughness.jl` | Is A3dyn OMEGA smoother than div_h(MFXC) at the SH-UTLS? |
| `plot_advection_profiles_per_panel_percentiles.jl` | Per-panel percentile profiles, adv-only GEOS-IT vs ERA5 (log axis) |
| `plot_advection_profiles_per_panel_percentiles_linear.jl` | Same, linear axis |
| `plot_c180_window_tendency_summary.py` | Heatmaps of ERA vs GEOS C180 window-tendency statistics |
| `plot_column_mean_maps.py` | Column-mean maps of the 4 CATRINE tracers vs GEOS-Chem |
| `plot_convection_profiles_per_panel.jl` | Per-panel mean profiles per convection scheme and tracer |
| `plot_convection_profiles_per_panel_overlay.jl` | Overlay of per-panel mean profiles, three convection schemes |
| `plot_convection_profiles_per_panel_percentiles_linear.jl` | Linear-axis variant of the heritage convection percentile plot |
| `plot_diffusion_profiles_per_panel_percentiles.jl` | Per-panel percentile profiles, diffusion-only (log axis) |
| `plot_diffusion_profiles_per_panel_percentiles_linear.jl` | Same, linear axis |
| `plot_fullphysics_profiles_per_panel_percentiles.jl` | Per-panel percentile profiles, full physics (log axis) |
| `plot_fullphysics_profiles_per_panel_percentiles_linear.jl` | Same, linear axis |
| `probe_f64_day_boundary.jl` | Plan 39 Float64 cross-day air-mass handoff probe |
| `prototype_pfix_balanced.jl` | Prototype pressure-fixer cm balanced to the analyzed PS |
| `prototype_remap_vs_cm.jl` | Prototype: flux advection + vertical remap vs cm advection |
| `run_c180_3day_ppm_viz.sh` | `atmos_viz.jl` loop over C180/L137 3-day PPM runs |
| `run_c180_era5_geosnative_cfl85_3day_ppm_viz.sh` | Same loop for ERA5-on-GEOS-native C180/L85 runs |
| `run_c180_era_geos_comparison_movies.sh` | Driver of `compare_c180_era_geos_movies.jl` |
| `run_c180_geosit_native_3day_ppm_viz.sh` | Same loop for GEOS-IT native C180/L72 runs |
| `run_tm5_2day_demo.jl` | Plan 24 two-day LL TM5 convection demo |
| `tracer_mass_balance_vs_gc.py` | Tracer burden ratio ours/GEOS-Chem plus dry-air drift |
| `tropical_pbl_diurnal_co2.py` | Tropical surface `co2_natural` diurnal cycle, Dec 2 2021 |
| `verify_emissions_lmdz_timing.jl` | lmdz CO2 flux timing vs GEOS-Chem and raw CAMS |
| `wind_vs_mfxc_divh_roughness.jl` | SH-UTLS div_h roughness: A3dyn winds vs native MFXC |

### `scripts/deprecated/`

| Path | Purpose |
|---|---|
| `run_cs_driven.jl` | Shim for the old CS CLI name; forwarded to `run_driven_simulation` |
| `run_cs_transport.jl` | Shim that included `benchmarks/run_cs_transport.jl` |
| `run_transport_binary.jl` | Shim for the old LL/RG CLI name; forwarded to `run_driven_simulation` |

### `scripts/diagnostics/`

| Path | Purpose |
|---|---|
| `check_mass_conservation.py` | Tracer mass drift in a no-flux run (4 CATRINE tracers) |
| `cm_sh_roughness_profile.jl` | SH cm roughness profile of a CS binary |
| `compare_convection_cadence_experiment.jl` | Per-window vs per-substep convection cadence vs n_merge |
| `compare_cross_grid_zonal_mean.py` | src_v2 LL vs RG zonal-mean column CO2 |
| `export_linrood_la_footprint_csv.jl` | C24 LA footprint `.bin` to CSV |
| `plot_linrood_la_footprint.py` | Matplotlib plot of the C24 LA footprint CSV |
| `publish_ll_rg_validation_plots.py` | LL/RG validation panels from src_v2 snapshots |
| `quick_viz.py` | Quick surface CO2 heatmaps from a lat-lon NetCDF |
| `scan_tm5_active_depth.jl` | Aggregates TM5 active-layer CSVs into `tm5_active_safety.toml` |
| `verify_snapshot_netcdf.py` | Raw-value checks of src_v2 snapshot NetCDFs |

### `scripts/preprocessing/`

| Path | Purpose |
|---|---|
| `generate_campaign5d_configs.jl` | Generated the CATRINE 5-day config matrix (configs are committed) |

### `scripts/visualization/`

| Path | Purpose |
|---|---|
| `animate_catrine_map_curtains.py` | Python CATRINE map + curtains; replaced by `animate_catrine_map_curtains_makie.jl` |
| `animate_catrine_vs_geoschem.jl` | Six-panel GEOS-Chem vs AT CATRINE CO2 animation |
| `animate_era5_vs_geoschem.jl` | Dec-2021 GEOS-Chem vs ERA5 Float64/Float32 animation |
| `animate_gchp_v4_fullphys.jl` | GEOS-Chem vs AT GCHP v4 full-physics animation |
| `cs_regrid_utils.jl` | Nearest-neighbour CS to lat-lon map and loaders for older animations |
| `plot_snapshot_grid.jl` | Multi-panel snapshot PNG; replaced by `atmos_viz.jl --kind grid` |
