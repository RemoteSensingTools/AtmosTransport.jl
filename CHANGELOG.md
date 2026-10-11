# Release notes

## Unreleased

### Breaking changes

- Run settings no longer come from environment variables. These are removed
  without a transition period; the package ignores them:

  | Removed variable | Use instead |
  |---|---|
  | `ATMOSTR_FORCE_PER_SUBSTEP_PHYSICS` | `[run] physics_cadence = "substep"` |
  | `ATMOSTR_REPLAY_CHECK` | `[input] validate_replay = true` |
  | `ATMOSTR_NO_REPLAY_CHECK` | nothing (the load-time check runs only when asked for) |
  | `ATMOSTR_ENABLE_HORIZONTAL_POISSON_BALANCE` | `[numerics] balance_mode = "per_layer"` |
  | `ATMOSTR_DISABLE_PREFETCH` | `[run] prefetch_windows = false` (new) |
  | `ATMOSTR_ASSERT_CS_BINARY_CFL` | `[advection] check_binary_cfl = true` (new) |
  | `ATMOSTR_NO_WRITE_REPLAY_CHECK` | `[numerics] write_replay_check = false` in the preprocessing config (new) |

  Environment variables now set only paths, metadata, profiling and
  diagnostics (see `docs/src/config/environment.md`). Julia code that calls a
  preprocessing writer directly (`process_day(date, grid, settings, vertical)`,
  `regrid_ll_binary_to_cs`, `process_merra2_to_cs_day`,
  `process_era5_n320_to_cs_day`) passes `write_replay_check = false` instead
  of setting the variable; `write_replay_check_enabled` is removed. The three
  new keys:

  - `[run] prefetch_windows` (default `true`): GPU runs with at least two
    Julia threads load the next met window on a second thread. The
    `DrivenSimulation` keyword of the same name fixes the choice for the whole
    simulation; the variable was read every window.
  - `[advection] check_binary_cfl` (default `false`): a cubed-sphere
    split-sweep step (`upwind`, `slopes`, `ppm`) that takes its subcycle count
    from the binary recomputes the CFL budget and stops if the binary asks for
    fewer. Typed as `BinaryCFLCheck()`/`NoBinaryCFLCheck()` on
    `CSAdvectionWorkspace` (the `binary_cfl_check` keyword of
    `TransportModel`); other schemes and grids reject `true`. The three
    operator copies of the check are now one dispatched helper.
  - `[numerics] write_replay_check` (default `true`) of a preprocessing config
    reaches every writer as the `write_replay_check` keyword.

- `[numerics] geos_balance_mode` is renamed `balance_mode` (an error names the
  new key), and `balance_mode` accepts only `"column"` and `"per_layer"` (the
  aliases `column_poisson`, `layer`, `layer_local` and `global` are gone). The
  shipped preprocessing configs use the new key. The GEOS writer's deprecated
  `process_day(...; balance_mode)` keyword is removed; pass
  `horizontal_balance = ColumnBalance()` or `LayerBalance()`.

- The advection keys `scheme`, `ppm_order`, `vertical` and `limiter` are read
  only from `[advection]`; in `[run]`, their old location, they are an error
  that names the new table. The 121 shipped run configs that used `[run]`
  moved them to `[advection]` (checked by parsing each file before and after);
  two of them, which set `[run] scheme` next to an `[advection]` table and
  were therefore rejected before, now run.

- `surface_flux.molar_mass_kg_mol` is replaced by `surface_flux.species`
  (`"co2"`, `"sf6"` or `"rn222"`); the molar mass comes from
  `Parameters.SPECIES_MOLAR_MASS`. The old key is an error that names the new
  one. Tracers named `co2`, `co2_natural`, `co2_fossil`, `fossil_co2`, `sf6` or
  `rn222` may still omit it; any other tracer with a flux must set it, and an
  unknown species is an error listing the known ones (also reported by
  `validate_config`). The 250 occurrences in the shipped run configs (245 in
  `config/runs/`, 5 in the golden-output configs) moved to `species`; every
  migrated tracer resolves the same molar mass, and so the same storage factor
  bit for bit (checked on all 461 tracer fluxes of those configs).
  `Models.InitialConditionIO.surface_flux_species` gives the resolution to
  scripts (`tracer_budget_closure.jl` uses it); the private
  `_tracer_molar_mass_kg_mol` and `_KNOWN_TRACER_MOLAR_MASS_KG_MOL` are removed.

- The 4D-Var inversion driver (`scripts/inversions/cs_4dvar.jl`) reads a
  required `[advection]` table (#16). It hard-coded unlimited PPM and ignored
  any `[advection]` table, even an invalid one, so the gradient was always
  that of `PPMScheme(NoLimiter())`. The table now goes through the forward
  run's `advection_spec` and `materialize`, with an `InversionAdvectionContext`
  that requires `scheme` and also accepts `limiter = "none"` (unlimited PPM);
  forward runs are unchanged and still reject `none`. Schemes without a
  cubed-sphere adjoint (limited slopes, the FV3 vertical profile, `none`) are
  rejected by the new `require_adjoint_support`. Unknown tables and keys are
  errors, including `[diffusion]` and `[convection]`, which the driver does
  not wire into the cost, and the documented but never written `[output]`
  path; keys the selected kind does not read are warnings. The shipped
  `example_synthetic.toml` states `scheme = "ppm"`, `limiter = "none"` and
  gives bit-identical results. `SyntheticExperimentConfig` of
  `scripts/inversions/synthetic_experiment.jl` gains a `scheme` field
  (default unchanged).

### Fixes

- `CSLBFGS` evaluated every control point at least twice (#17): Optim
  received separate value and gradient closures that each ran the full
  `cost_fn` (a forward run plus the adjoint footprints), and the initial
  probe and the final result were recomputed. Optim now gets one
  `OnceDifferentiable` whose value, gradient and combined methods share a
  cache of the last evaluation, so each point Optim visits costs one
  `cost_fn` call: 26 calls became 12 over an 11-iteration test inversion, with
  bitwise-identical costs, gradients and minimizer (single-threaded). The
  iteration log no longer records iteration 0 twice; its rows now match
  `cost_history`.
- Transport binaries (eager and streaming writers), the spectral day cache and
  cached regridders are published through `AtomicFiles`: a sibling staging
  file moved onto the target by one replacing rename (`rename(2)` on POSIX
  systems) on every Julia version. Before, `mv(…; force = true)` deleted the
  previous file before renaming on Julia 1.10, so a reader could find no
  file, and on every Julia version deleted it when the rename failed. A failed
  rename now leaves the previous file in place and removes the staging file.
- Time-varying surface fluxes without `[input].start_date` took their time
  origin from the file: the first slice for decoded dates, the CF units origin
  for raw numbers. A run on a mid-month binary then read the month's first
  days. They now require `start_date` (`validate_config` reports its absence).
  All 61 shipped configs with time-varying fluxes set it; the two golden cases
  that did not (`run_c90_merra2`, `run_c90_era5_tm5`, on 2022-01-15) now set
  it, so their 3-hourly LMDZ fluxes come from January 15 instead of January 1.
- Surface-flux files in per-second units must be in kg of the species per m²
  per second (`kg m-2 s-1`, `kg/m2/s`, `kgCO2/m2/s`, ...); other per-second
  units such as `mol m-2 s-1`, and files without units, were read unconverted
  and are now rejected. Every shipped flux file passes.
- With `binary_paths`, `[input] end_date` was reported as unread but still
  widened the span of loaded time-varying flux slices; the span is now the
  run's windows, also with `folder` (where every day through `end_date` has a
  binary, so nothing changes there).
- File-based initial conditions took their vertical order from the values
  of the level coordinate and copied levels one to one when the counts
  matched. A surface-first file with hybrid coefficients and the model's
  level count (the CATRINE start files have 79 levels) would have started
  upside down, a top-first file with hybrid coefficients would have been
  remapped wrongly, and a file without them but with another level count was
  zero-filled. Files with hybrid coefficients are now oriented by their
  interface pressures and always remapped in log pressure; files without
  them must match the model's level count and state their order (pressure
  or height units, a CF `positive` attribute, or `init.vertical_order`).
  No shipped configuration was affected: all use the CATRINE files on 34,
  72, 117 or 137 levels, which took the correct path, and their initial
  conditions are unchanged.
- ERA5 N320 preprocessing had two registration errors. The GRIB
  `reduced_gg` fields (specific humidity and the convective mass fluxes)
  were placed half a cell east: their first point is at 0°, while the mesh
  cells are centred at `(i − ½) Δλ`. They are now interpolated to the cell
  centres. And the N320 → cubed-sphere regridder left up to 0.3 % of the
  polar cells uncovered, so every intensive field was too small there
  (surface pressure about 3 hPa low and temperature 0.8 K cold poleward of
  89° on C90); results are now divided by the covered fraction. ERA5 N320
  binaries built before this fix carry both biases.
- The per-ring spectral synthesis (`spectral_to_ring!`) did not mirror the
  highest wavenumber `(nlon − 1)/2` of rings with an odd number of longitudes,
  so that wave entered at half amplitude. ERA5 N320 has 72 such rings (of 640).
  Fixed there and in the batched synthesis below that now serves u, v and T;
  the change is at most 6e-3 m/s and 2.5e-3 K. The Nyquist term `nlon/2` of
  even rings still enters with half weight, as before.
- `scripts/preprocessing/attach_catrine_tm5_convection_cs.jl` stored the
  legacy 1-degree TM5 convection upside down. The CATRINE files keep ERA5 L137
  surface first, and the script summed them through the binary's top-first
  `merge_map` without reversing, so updraft entrainment landed near the model
  top. The script now verifies the level order against ERA5 L137 coefficients,
  the 1-degree grid, the date, and the three-hour slots (from
  `timevalues_bounds`; the CF `time` variable is corrupt in the 2022-04-30 to
  2023-12-31 files), and fails if the updraft entrainment's mean level lies in
  the upper half of the column. Outputs carry attachment tag `..._v3`, the
  source level order, and the code revision. Every binary set written by
  earlier versions is upside down: the C90 L66 set for 2018-12 to 2019-12 and
  the C30 full-physics sets for 2021 and Dec 2021 to 2022. Run configs and
  driver scripts now point at new `_v3` folders. See
  `docs/memos/2026-10-03_tm5_attach_level_order.md`.
- Cubed-sphere cell areas and edge lengths were computed in the mesh
  precision. In `Float32` they erred by up to 1.3e-4 per cell at C90,
  6.8e-4 at C180 and 9.7e-3 at C720. Preprocessing with
  `float_type = "Float32"` used these areas wherever it converts between
  pressure and mass. They are now evaluated in `Float64` and rounded once;
  regenerate `Float32`-preprocessed C180 and finer binaries.
- The EDGAR tonnes-to-flux normalisation (`_lonlat_cell_areas_m2`) took the
  grid spacing from the first coordinate difference, which errs by ~1e-4 for
  coordinates stored in `Float32`. It now uses the full span.
- `[mass_fix] mode = "initial_endpoint"` silently skipped the global dry-mass
  pin for MERRA-2 and ERA5 N320 sources: their writers pin only to a finite
  target, and this mode passes none. Only the GEOS native path implements it
  (`supports_initial_endpoint_mass_pin`); other sources now refuse the mode.
  No shipped configuration used it.
- Skipping the write-time replay gate (then `ATMOSTR_NO_WRITE_REPLAY_CHECK=1`,
  now `[numerics] write_replay_check = false`) did not skip it in the GEOS
  cubed-sphere writer. It does now, through
  `verify_window!(…; write_replay_on)`, and its log no longer reports a worst
  replay window when the gate was skipped. With the gate skipped, the
  cubed-sphere regrid, ERA5 N320 and MERRA-2 writers ran the positivity gate
  against the window's start mass only; they now also pass the end mass, as
  with the gate on (only runs with the gate skipped are affected).
- The reduced-Gaussian spectral writer wrote straight to the final file, so a
  day that failed a gate deleted an existing binary of that day. It now
  stages to `<out>.tmp` like the other writers.
- Visualization regridded cubed-sphere snapshots on the panel convention's
  default mesh, ignoring the recorded coordinate law, center law and
  longitude offset (`cs_*` and `longitude_of_central_meridian` attributes).
  `CubedSphereSnapshotTopology` now carries the snapshot's definition.
- The MERRA-2 download recipe uses OPeNDAP, whose download step is not
  implemented: it failed at the first file, after creating the output tree,
  with a message naming a script that no longer exists. `download_data!` now
  refuses such recipes before doing anything unless `--dry-run` or
  `--verify` is given (`protocol_can_download`).
- `[numerics] geos_balance_mode` was ignored by the MERRA-2 and ERA5 N320
  writers, which balanced per layer only with the environment variable
  `ATMOSTR_ENABLE_HORIZONTAL_POISSON_BALANCE=1`; no header recorded the
  mode. `[numerics] balance_mode = "column" | "per_layer"` now selects it on
  every path (the old name was later removed; see Breaking changes), the
  LL-to-CS regrid script takes
  `--balance-mode`, and every transport-binary header records
  `horizontal_balance`. The environment variable is removed (see Breaking
  changes). Default results are unchanged.

### Numerical changes

- Carbon-mass fluxes (`kgC m-2 s-1`, the CAMS/LMDZ inventories) become CO₂ by
  the molar-mass ratio `M(CO2)/M(C)` = 44.0095/12.0107 ≈ 3.66419
  (`CARBON_MOLAR_MASS`, the IUPAC value consistent with `SPECIES_MOLAR_MASS.co2`),
  which conserves carbon atoms, instead of 44/12. These fluxes are 0.068 % lower
  than before; GEOS-Chem's HEMCO uses `CTOCO2 = 3.6667` (44/12).
  The Python flux preprocessors (TRENDY, FLUXCOM-X, SIF-GPP, GFED5) use the
  same ratio; drivers they wrote before are 0.068 % higher until regenerated.
- Lat-lon preprocessing uses the cubed-sphere positivity gate (owner's
  decision, 2026-10-10): the palindrome outflow `2 (out_x + out_y + out_z)`
  against `min(m, m_next)`, one shared implementation, instead of each
  direction's single application against `m`. Adaptive schedules (the
  default) choose more substeps on lat-lon binaries: on the ERA5 72×37
  lat-lon golden day (2021-12-01) two of 24 windows go from 4 to 5 substeps. A fixed
  schedule that passed the old gate can now fail it. Reduced-Gaussian
  binaries keep their per-direction gate.
- New cubed-sphere option `[advection] scheme = "ppm"`, `vertical = "fv3_kord8"`.
  The vertical sweep integrates FV3's `scalar_profile` parabola (`kord = 8`,
  positive definite; GCHP's tracer profile) over the swept fraction of the
  donor layer (`PPMScheme(; vertical = FV3ScalarProfile())`). Use
  `"fv3_kord8_signed"` for tracers that become negative.
  - Edges and limited parabolas match a transcription of `fv_mapz.F90`
    bit for bit.
  - In 1-D translation tests it is third order or better and 4–900 times
    more accurate than the default.
  - `Float32` mass drift is at or below the default's.
  - The default (`SameAsHorizontal`) is unchanged, and lat-lon runs and the
    adjoint reject the option.
  - `scheme = "linrood"` accepts the same `vertical` values. Its default stays
    upwind: `LinRoodPPMScheme(ORD; vertical = UpwindScheme())`.
  - `CSAdvectionWorkspace` gains `column_scratch` (`column_scratch = true`).
- `PPMScheme` was documented as third order. It is second order: the limited
  PPM edge sets a Russell–Lerner slope and the parabola is not integrated.
  The docs now say so.
- `Float32` runs now conserve global tracer mass to about 1e-6 of the burden
  per year. Over 3 days of ERA5 C90 transport without sources, the background
  tracer drifts +9e-9 instead of −9.7e-7 and SF₆ +2.1e-9 instead of −1.0e-6.
  A small advection bias (≈+1e-6 per year) and, with CMFMC convection, about
  −1e-6 per year for CO₂ remain. Changes:
  - TM5 convection and `dkg` diffusion return each column's rounding residual,
    computed with compensated sums, to its largest participating cell. Only
    rounding-sized residuals are returned, so a non-conserving matrix stays
    visible.
  - GEOS-Chem's non-local emission profile accounts for every upper-layer
    addition with `TwoSum`.
  - Cubed-sphere areas, edge lengths and corners, regridding geometry, the
    decay decrement `expm1(−λΔt)` and the model clock are evaluated in
    `Float64` and rounded once. The clock follows the window and step
    counters, so window ends are exact for any step count. Meshes of either
    precision now regrid on a `Float64` sphere and share cached weights
    (cache version 3, so every cached regridder is rebuilt once).
  - Lat-lon flux and initial-condition sources stored in `Float32` (GridFED)
    snap to exact global extents. Previously the regridder dropped 2.1e-6 of
    the GridFED flux in `Float64`.
  - `total_mass` and `total_air_mass` return compensated `Float64` sums,
    reduced on the device (shared with the snapshot totals).

  `Float32` global totals change at the 1e-6 level. `Float64` global totals
  change by about 1e-11, except GridFED emissions (+2.1e-6). See
  `docs/src/theory/float32_conservation.md`.
- The two copies of the cubed-sphere block kernels (nested binary coarsener
  and GEOS block coarsening) are one, `Regridding.coarsen_sum!` and
  `coarsen_area_weighted!`. Area-weighted means now accumulate numerator and
  denominator in `Float64` and round once (the coarsener's semantics); the
  GEOS path accumulated them in the data precision. Float32 GEOS-native
  builds whose target `Nc` is a divisor of the source `Nc`
  (`geosit_c90_dec2021_catrine_f32_fullL72.toml`,
  `geosit_c45_dec2021_catrine_f32_fullL72.toml`,
  `geosfp_c720_native_to_cs180.toml`) change in `pblh`, `ustar`, `hflux`,
  `t2m`, the vdiff `u/v/t/qv`, `cmfmc` and `dtrain` before vertical
  remapping: 1–5 Float32 ulp in about half the cells of positive fields,
  more on near-cancelling signed fields (the old result could be thousands of
  ulp from the correctly rounded mean). Cell sums, face sums, Float64 builds
  and same-`Nc` builds are bit-identical.

### Runtime and output

- Every number that more than one place used, or that is a physical constant,
  now lives in `src/Parameters` (owner's rule, 2026-10-10): physical and
  reference values in `PhysicalConstants.jl` (new: `VON_KARMAN`, U.S. Standard
  Atmosphere values, `HYDROSTATIC_REFERENCE_TEMPERATURE`, `DAYS_PER_JULIAN_YEAR`,
  the dry-mass target `GLOBAL_MEAN_DRY_SURFACE_PRESSURE` and
  `GLOBAL_MEAN_SPECIFIC_HUMIDITY`), shared defaults of numerical settings in the
  new `NumericalDefaults.jl` (Courant limits and targets, Poisson-balance
  tolerances, subcycle caps, met interval, background VMR, Lin–Rood PPM order,
  …) and fixed properties of ERA5, GEOS and MERRA-2 in the new
  `MetSourceConstants.jl`. Scheme parameters stay in their scheme's struct
  (`PBLPhysicsParameters` gains the u*, PBL-height and 2 m temperature floors
  and the entrainment fractions that five Kz kernels wrote inline). Forward and
  adjoint advection share one mass-floor definition each. Every constant is
  pinned by `test/core/test_physical_constants.jl`. Results are bit-identical
  (all golden cases).
- Logging and runtime checks after Oceananigans. `[logging]` sets the run's
  minimum log level (`"warn"` concise, `"info"`, `"debug"` verbose; without it
  the current logger's level applies, info for the command-line runner)
  and its progress output (`progress = "bar"` as before, `"lines"` for one
  log line every `progress_interval_windows` windows in batch logs, or
  `"none"`). `[checks] nan_check` (on by default) stops the run with an error
  naming the field, window and model time when the air mass or a tracer holds
  a NaN or Inf, checked after every `nan_check_interval_windows` windows.
  In Julia, `DrivenSimulation(...; callbacks)` takes `Callback(func,
  schedule)` with the schedules `IterationInterval`, `WindowInterval` and
  `WallTimeInterval`; `NaNChecker` is one such callback function. Results are
  unchanged.
- Faster cubed-sphere GPU runs, with unchanged results (runtime golden cases
  bit-identical): on CUDA the halo exchange fills all 24 panel edges in one
  kernel launch and the corners in a second, instead of 24 edge launches (6 for
  packed tracer fields) plus 6 corner launches, one per panel (C90 L72 on an
  L40S: 120 → 25 µs per air-mass exchange); the CMFMC convection CFL scan runs
  on the device instead of copying three fields to the host once per window;
  NetCDF output stacks the layer-resolved cubed-sphere fields into staging buffers of the output
  type, one per requested layer count, reused across the fields and snapshots
  of one write. Two-day warm runs on C90: transport time −7 % (MERRA-2) and
  −9 % (ERA5) from the kernel changes; with 3-hourly 3-D NetCDF output, the
  stacking change together with a type-stable tuple of the 24 edge operations
  cuts the MERRA-2 run-loop wall time by a further 16 % (11.6 → 9.7 s; run-loop
  host allocations 26.8 → 15.0 GiB per two days; end to end including setup
  29.9 → 27.3 s). The GPU paths were tested on CUDA (L40S:
  `test/diagnostic/test_cmfmc_cfl_gpu.jl`, `test_cs_halo_fill_gpu.jl`,
  `test_point_ops_gpu.jl` and the seam, Lin–Rood and PPM GPU diagnostics) and
  on Metal (Apple M5 Pro, Float32, `test/diagnostic/test_metal_kernels.jl`:
  point operations, halo exchange and CFL scan
  exact; a one-day C24 run agrees with the CPU to rounding and is about 10 %
  faster than before, with laptop timings varying by up to 40 %). On Metal,
  fused point operations are launched one after another, each on its
  `bound_context` (`fusion_policy`; the halo fills bind their panel arrays); a C90 halo
  exchange then takes 1.12–1.34 ms instead of 1.29–1.67 ms for packed tracers
  and 0.45 instead of 0.58 ms for a 3-D field with corners.
- Cubed-sphere GPU runs refill one host window in place for every met window
  they load (`load_transport_window!`) instead of allocating a new one and its
  padded copies; the window is then copied to the device as before. Results
  are unchanged (runtime golden cases bit-identical). Two-day warm MERRA-2 C90
  runs on an L40S (six runs each): run-loop host allocations 15.0 → 4.9 GiB,
  run-loop wall time 10.2 → 8.7 s, of which transport 7.2 → 5.3 s (less
  garbage collection during transport).
- Time-varying surface fluxes (cubed sphere) are read only for the run period
  when `[input]` gives `start_date`: from the last slice at or before the run
  start to the first slice at or after the run end. Every temporal scheme uses
  only the two slices around a time, so results are unchanged (two-day
  MERRA-2 C90 run with LMDZ, GridFED and three native-C90 sources:
  bit-identical NetCDF output). That run reads 10 instead of 6088 LMDZ slices
  and 2 instead of 36 GridFED slices, and takes 12.4 s instead of 25.9 s end to
  end (warm, L40S, six runs each; garbage collection 6.0 → 2.3 s). The
  loaded period ends with the binaries' windows (from every binary's header)
  or the end of `end_date`, whichever is later.
- New in `Architectures`: point operations (`AbstractPointOp`, `Fused`,
  `Sequence`, `launch!(op, ctx, backend)`), with a per-backend
  `fusion_policy` (one fused kernel by default, separate launches on Metal). A kernel body is written once as
  `apply_point!(op, ctx, I...)` and runs as one GPU kernel or as CPU loops;
  independent operations fuse into one launch by type.
- The end-of-run summary also reports garbage-collection time, JIT
  compilation time and allocated memory.
- `fill_panel_halos!` and `copy_corners!` reject a halo wider than the panel
  (`Hp > Nc`) and panels of unequal size. CMFMC convection stops with an
  explicit error when its CFL scan finds a non-finite ratio, such as NaN or Inf
  in `cmfmc` (previously an `InexactError`); layers with NaN or non-positive
  air mass are skipped by the scan, as before.
- New PPM option `[advection] limiter = "cw84"` (`PPMScheme(CW84Limiter())`):
  the complete Colella–Woodward (1984) PPM, with van Leer-limited edge values
  and a flux that integrates the limited parabola over the swept fraction.
  Every sweep in which no cell exports more than its air mass keeps
  non-negative tracers such as fossil CO₂ non-negative, where the default PPM
  undershoots (−0.16 on a [0, 1] box in a 1-D test). Lat-lon and cubed
  sphere, CPU and GPU, with the cubed-sphere adjoint. The default
  `scheme = "ppm"` is unchanged.
- `[run] physics_cadence = "window" | "substep"` selects whether, on binaries
  with a per-window physics contract, convection and chemistry run once per
  met window (default) or every advection substep. It replaces the
  environment variable `ATMOSTR_FORCE_PER_SUBSTEP_PHYSICS`, which was read on
  every time step and is removed. Unlike the variable, `"substep"` keeps the window-end reset to
  the binary's endpoint air mass, so a cadence comparison changes only where
  convection and chemistry run.
- Deprecated, for removal in the next minor release: `State.MetState`,
  `diagnose_cm_from_continuity_vc!` and `diagnose_cm_from_continuity_ka!`. No
  part of the package uses them. Removed: the exported generic function
  `Preprocessing.reset_workspace!`, which had no methods.
- Runs and preprocessing from `git archive` code snapshots record their commit:
  git writes it into `src/REVISION` on export (`export-subst`), and
  `source_revision()` reports it when the tree has no `.git`; edits made after
  the export are not detected. Snapshot outputs had
  `framework_commit = "unknown"` before. Cubed-sphere transport binaries now
  carry `git_commit` and `git_dirty` in their header.
- `scripts/diagnostics/catrine_benchmark_page.py` writes the CATRINE benchmark
  web page from a `catrine_compare_vs_geoschem.py` output: Chart.js charts with
  run check boxes and tracer, band and statistic selectors (daily and monthly
  agreement statistics, period table, global burdens, true mass balance), code
  versions, maps and animations. The page is self-contained (data embedded,
  Chart.js copied next to it) and keeps the selection in the URL.
- `scripts/diagnostics/catrine_true_mass_balance.jl` and
  `catrine_true_mass_balance_gc.py` compute the true global mass balance of a
  run (burden change minus the run's own applied emissions and decay) for
  AtmosTransport, from the run's output and its sources rebuilt by the model,
  and for GEOS-Chem, from its `SpeciesConcVV`, `Met_AD` and `Emis*` output;
  the benchmark page plots them.
- New `[output.observations]` contract for sampling tracer profiles at
  observation points (OCO-2 Lite soundings, NOAA ObsPack sites, generic point
  tables) from the containing model cell at met-window ends. The table is
  parsed by `observation_output_spec` and checked by `validate_config`, which
  rejects unknown keys and requires an absolute run origin. Both runners
  sample at t = 0 and at every met-window end: soundings are blended linearly
  between the two bracketing window ends (or taken from the nearest), sites
  get the intake-layer value from hypsometric heights. Output goes to
  append-only `_soundings` / `_sites` NetCDF files (see the output schema).
  Gridded snapshot output is unchanged.
- Point events (satellite soundings, ObsPack records, station time lists)
  carry an intake height and get `<tracer>_intake` from the layer containing
  it; station tables choose `EveryWindow`, `TimeRange`, or `TimeList`
  schedules per row and accept `altitude` with `elevation`; repeated site ids
  merge their time lists. OCO sources take a typed `quality_filter`
  (`"flag_max"`, `"flag_values"` for categorical flags such as the MIP
  `assimilate_flag`, or `"none"` to co-sample every record). The editor schema documents every choice with its Julia
  type and is checked against the parser by a test; site tables have their own
  schema (`schemas/observation_sites.schema.json`).
- User guide `docs/memos/2026-10-02_observation_sampling_guide.md` with a
  runnable OCO-2 MIP + TCCON example
  (`config/examples/observation_sampling_oco2mip.toml`, site table
  `config/examples/tccon_ggg2020_sites.csv`).
- All runtime NetCDF writes, including the background daily snapshot task,
  share one lock (`with_netcdf_lock`); netcdf-c is not thread-safe.
  Observation rows queue while the lock is busy instead of stalling the run.
- Daily output files (snapshots and observations) now use the day index when
  a binary name carries no date; previously every day wrote the same file.
  Two binaries that still resolve to one daily file fail before transport.
- `[output.fields].tracers = "name"` (a single string) no longer throws a
  `MethodError`; it selects that one tracer as documented.

- Scripts cleanup: 161 stale scripts were removed (git history keeps them;
  recover one with `git show 7c515038:<path>`) and 58 were moved into a
  `heritage/` subfolder of their folder, kept for reference but not
  maintained or tested. The deprecated runner shims
  `scripts/deprecated/run_cs_driven.jl`, `run_transport_binary.jl` and
  `run_cs_transport.jl` are gone; use `scripts/run_transport.jl`.
  `scripts/completed_experiments/` and `scripts/deprecated/` no longer exist.
  `scripts/README.md` lists every removed script with its purpose, and each
  `heritage/README.md` says why its scripts are kept.
- Run configs are checked before any binary is opened. `validate_config`
  (run at the start of every run) now also parses `[advection]`,
  `[diffusion]`, `[convection]`, `[chemistry]`, `air_mass_reset_mode`,
  `physics_cadence` and `[output]`, so these errors no longer appear only after
  the binaries are inspected. New errors: a decay half-life for a tracer the run
  does not carry (it failed at the first chemistry step), an enabled `[output]`
  with a path but no snapshot-schedule key or with snapshot times but no path
  (nothing was written; `hours = []` still means deliberately none), Lin–Rood
  `ppm_order` other than 5 or 7. Keys the run would ignore are logged as
  warnings: unknown tables and keys with a "did you mean" suggestion, known
  keys the chosen kind leaves unread (including flat `[tracers.<name>]` keys),
  and surface-flux tables without `kind` (no flux is emitted). Of the 246 run
  configs with an `[input]` table in `config/runs` and `config/examples`, only
  four that already failed at run time differ: they now fail at the check.
- `[input] validate_replay = true` replays every binary's continuity when the
  run opens it (the stored fluxes must carry each window's air mass to the
  next). It replaces the environment variable `ATMOSTR_REPLAY_CHECK`; both it
  and `ATMOSTR_NO_REPLAY_CHECK` are removed (the check runs only when asked
  for).
  The replay error messages now name the key.
- A transport binary written with the write-time replay gate skipped
  (`[numerics] write_replay_check = false`, or the new
  `--no-write-replay-check` of `regrid_ll_transport_binary_to_cs.jl`) records
  `write_replay_check = false` in its header; `binary_capabilities` reports
  it, `inspect_binary` marks it and `TransportBinaryDriver` warns when it
  opens such a binary. Binaries written with the gate on are unchanged.
- `docs/src/config/environment.md` lists every environment variable the
  package reads, and `test/core/test_environment_variables.jl` fails when
  `src/` reads one that is not listed there (or from outside the folder that
  owns it).
- Breaking: a surface-flux `kind` that is not a known source
  (`none`, `file`, `cs_native`, `lmdz_co2`, `gridfed_fossil_co2`, `edgar_sf6`,
  `zhang_rn222`) is an error, with a suggestion for a close name; it used to
  be read silently as a generic file, so a typo such as `gridfed` lost the
  GridFED unit conversion. Use `kind = "file"` for a generic NetCDF file. The
  two ocean-flux configs now say `kind = "file"` instead of
  `"eccodarwin_ocean_co2"` (same source, results unchanged).
- The two binary-format A/B configs `config/runs/binary_format_ab/c45_*` are
  retired: they set `[advection] order = 7`, which was never read, so they
  ran Lin–Rood PPM5 instead of the intended PPM7.
- Snapshot hours are checked against the met-window ends of every binary
  (now recorded by `binary_capabilities` as `nwindow` and `window_seconds`)
  when the run starts: an hour that is not a window end was never matched, and
  every later snapshot was lost with it. Snapshots are matched to window ends
  within half a window (at most half an hour, as before), so runs with
  sub-hour windows no longer take a snapshot one window early. The default
  snapshot span and the observation span count the windows of every binary,
  not the first binary's times the binary count. `format = "binary_mmap"` is
  rejected on lat-lon and reduced-Gaussian grids before the run instead of at
  the first write.
- The editor schema lists every key the runtime reads (new:
  `expected_nlevel`, `required_preprocessor_contract`, `convection.cloud_base`,
  surface-flux `regridding`, `init.clamp_negative`, top-level legacy `[init]`,
  and the deprecated aliases) and accepts `diffusion.kind =
  "geoschem_nonlocal_vdiff"`. A new test compares every schema choice with the
  parser's accepted values and every schema table with the known-key tables,
  both ways. New module `ConfigChecks` holds the strict Boolean and known-key
  helpers that `Output`, `Preprocessing` and `Models` each had a copy of.

### Surface fluxes and preprocessing

- ERA5 N320 preprocessing is about five times faster: a C90 day takes about 7
  minutes on 12 threads instead of 23–38.
  - The core GRIB file of a day (12–24 GB) is indexed once with ecCodes
    (`_core_messages`); every hourly window used to scan the whole file.
  - The spectral synthesis of all 137 levels runs as one BLAS product per
    zonal wavenumber with precomputed Legendre functions, followed by one
    inverse real FFT per ring and level (`ReducedSpectralSynthesis`); it
    recomputed the Legendre table for every ring, level and field before.
  - Spectral coefficients are decoded straight into the level cubes.
  - Humidity and surface pressure are unchanged; u, v and T agree to 1e-12
    relative, apart from the odd-ring fix above.
- ERA5 N320 option `[preprocessing] face_fluxes = "line_integral"` integrates
  `(V · N) Δp` along each cube face from the N320 winds and surface pressure
  (`LineIntegralFaceFluxes`; bilinear N320 values, midpoint rule), following
  TM5, which integrates the spectral winds along its cell edges, instead of
  interpolating cube-centre winds to the faces.
- ERA5 N320 option `[preprocessing] flux_time_sampling = "window_mean"` uses the
  mean of the face fluxes at the start and end of each hourly window instead of
  the winds at its start.
- `config/met_sources/era5_n320_arco_diffusion_li.toml` and
  `config/preprocessing/era5_n320_arco_diffusion_to_c90_l117_li.toml` combine
  both with `hybrid_b` on 117 levels (native below about 8 hPa).
- MERRA-2 preprocessing option `[preprocessing] face_fluxes = "vector"`
  builds face fluxes from the cell winds as 3-D vectors. They are projected
  onto the true face normals, use the faces' great-circle lengths, and are
  interpolated along the edge at panel seams, as FV3 does. The default
  (`"panel_average"`) is unchanged.
  - The default construction misplaces the seam winds by up to a quarter of a
    face length and uses cell centerline widths instead of face lengths.
  - For solid-body rotation the spurious divergence at seam cells drops from
    4e-3 to 5e-6 of the face flux, and in the interior from 1e-4 to 2e-7.
  - Against MERRA-2 OMEGA on 2022-07-15 (C90, p < 150 hPa), the
    vertical-velocity error at seam cells falls from 8.8e-3 to 1.3e-3 Pa/s.
  - Related opt-in keys: `face_lengths = "edge"` and
    `flux_thickness = "dry_mass"`.
- MERRA-2 preprocessing option `[preprocessing] column_balance_weights` sets
  how the column mass-budget correction of the horizontal fluxes is spread
  over levels. The options are `"mass"` (default, unchanged),
  `"hybrid_b"` (by ΔB, as in TM5) and `"hybrid_mass"` (by air mass in hybrid
  layers only). With `"mass"` the correction reaches the stratosphere as a
  vertically coherent `cm` mode; the hybrid options keep it out of the
  pure-pressure layers. In a four-month C90 run against GCHP, `"hybrid_mass"`
  cut the growth of the bias above 100 hPa by about two thirds. Near-surface
  RMSE changed by +0.4 % to −5 % depending on the tracer
  (`docs/src/theory/vertical_transport.md`). The ERA5 N320 source reads the
  same key, and the ERA5 spectral (`[numerics] column_balance_weights`),
  lat-lon → cubed-sphere (`--column-balance-weights`) and GEOS
  (`[preprocessing] column_balance_weights`) paths offer it too, with `"mass"`
  as the default; those three record it in the header only when it is not
  `"mass"`, so their default binaries are unchanged. Per-layer balance and
  the GEOS pressure-fixer closures reject other weights; lat-lon and
  reduced-Gaussian spectral targets accept only `"mass"`.

- Time-varying surface fluxes can span several files: `files = [...]` or a
  `file_pattern` with `{YYYYMM}` plus `year`. `gridfed_fossil_co2` joins the
  time-varying path (stepwise by default; monthly totals use each month's
  length), and an inventory's year is taken from the last year in its file
  name (`GCP-GridFEDv2024.0_2022.short.nc` is 2022).
- `kind = "cs_native"` with `time_varying = true` reads an already aligned
  GEOS-native `(time, nf, Ydim, Xdim)` flux-density series and converts it to
  per-cell storage rates without regridding.
- ERA5 N320 preprocessing to cubed-sphere grids can select a named L137
  level set on the target grid (`[vertical] transform = "level_selection"`,
  `preset = "ml137_66L"`), the recipe behind the C90 L66 ERA5 binaries, plus
  ARCO-ERA5 diffusion-only configs and a per-year C90 driver script.
- The ARCO C90/C180 and GEOS-IT OMEGA-regularized preprocessing configs now
  name the cubed-sphere definition `"gmao_equal_distance"`; the previous
  `"gmao"` was rejected by the parser.
- The inventory year in a surface-flux file name skips version markers and
  rejects ambiguous names. The default GridFED file
  `GCP-GridFEDv2024.0_2021.short.nc` was read as 2024, a leap year; it is now
  2021, so static February GridFED rates rise by 29/28.
- `cs_native` fluxes accept only kg m-2 s-1 units (of the species, or kgC
  converted by M(CO2)/M(C)), reject fill values, and must match the runtime mesh
  when the file carries `cell_area`.
- `[vertical].level_selection` presets are rejected on sources without 137
  native levels, and `[vertical].coefficients_file` is honoured as an alias of
  `coefficients`.
- Experimental `coarsen_nested_cs_transport_binary` restricts a cubed-sphere
  transport binary to a coarser nested grid (for example C90 to C30) by
  block-summing the transport operator; not yet validated as a replacement
  for preprocessing at the target resolution.
- `apply_regridder!(dst, r, src, kind)` regrids by quantity kind for every
  regridder: `IntensiveCellField()` as given, `ExtensiveCellField()` (per-cell
  totals) through the density, so totals are kept whatever polygon areas the
  regridder uses. It replaces Preprocessing's private `_density_correct!`;
  `regrid_2d_to_cs_panels!` / `regrid_3d_to_cs_panels!` give the same bits.
  The density buffer is the keyword `scratch` (one column, `eltype(src)`),
  allocated per call when not given; the cubed-sphere preprocessing workspace
  passes its own, and warm `regrid_2d/3d_to_cs_panels!` calls allocate
  nothing. `regrid_conservation_report(r, src, dst, kind)` returns both totals
  and the relative error and checks both leading dimensions against the
  regridder.
- `build_regridder` returns a `NestedCubedSphereRegrid{Restriction}` or
  `{Prolongation}` for cubed spheres of one definition and radius whose `Nc`
  differ by an integer factor (corners checked): exact block remapping in
  O(Nc²), equal to the polygon regridder to 1e-12, with nothing cached. Code
  that read `intersections` from such a pair's regridder must use
  `src_areas`/`dst_areas` and `apply_regridder!` instead.
- `Regridding.default_regrid_cache_dir()` is the one definition of the weight
  cache directory (`ATMOSTR_REGRID_CACHE_DIR`, else
  `~/.cache/AtmosTransport/cr_regridding`); the runtime's private
  `_regrid_cache_dir` is removed, and preprocessing targets without
  `[grid] regridder_cache_dir` use it too (before, a second hard-coded copy
  of the path that ignored `ATMOSTR_REGRID_CACHE_DIR`). Cached regridders are now published with
  mode `0o666 & ~umask` instead of `mktemp`'s owner-only `0o600`.
- `LatLonMesh(λᶠ, φᶠ; FT, radius)` builds a lat-lon mesh from explicit cell
  faces (the source grids of input files); `radius` is required. Longitude
  spacing must be uniform but may start at any face, and a sequence that wraps
  by −360° is unwrapped so data columns keep their order. Latitude faces may be
  descending (stored ascending; `Grids.latitude_row_order(φᶠ)` gives the row
  permutation) and non-uniform (pole-centred, truncated, Gaussian), in which
  case `Δφ = NaN` and `dx`, `dy` and the x-face lengths throw, and the lat-lon
  `write_transport_binary` refuses the mesh (its header records only the
  latitude interval). Areas come from the faces. The regular keyword
  constructor now builds its faces and calls this one; regular meshes are
  bit-identical.
- `Output.define_cs_horizontal_geometry!(ds, mesh)` (cubed-sphere `Xdim`/`Ydim`/`nf`,
  centres, corners, `cell_area`, `cubed_sphere` mapping with
  `semi_major_axis = mesh.radius`) and `Output.define_provenance_attributes!(ds)`
  are public, split from the snapshot writer's private blocks so other NetCDF
  writers share them. Snapshot files keep the same dimensions, data and
  attributes; the `lev` and `time` variables are now defined after the
  horizontal geometry variables.
- `Operators.total_emission_rate(src, t)`, `emission_segments(src, t, dt)` and
  `integrated_emission(src, t0, t1, dt)` give a surface source's global
  storage rate at an instant, the pieces one operator step applies, and the
  mass the operator adds over a run's steps, from the operator's own temporal
  reconstruction (equal to a stepped application to 1e-14 for every scheme).
  The runner's start-up line "Surface source … total model-storage rate" now
  prints `total_emission_rate(source, 0.0)`: the same number for `Float64`
  runs; for `Float32` runs the sum is accumulated in `Float64` (it differed by
  up to 5e-8 on the CATRINE C90 sources), and a linear or conservative series
  without a slice at the run start reports the interpolated run-start rate
  instead of its first slice. The sums run on the host: a source adapted to a
  device (CUDA, Metal) is copied to host memory first, so Metal, which has no
  Float64, gets its totals too. `tracer_budget_closure.jl` and
  `catrine_true_mass_balance.jl` use these functions instead of the private
  `_flux_temporal_segments`.
- New module `GriddedIO` (after `Regridding`, before `Preprocessing`): the one
  owner of the conventions of native gridded input files, measured on the
  TRENDY v14 S3 models, GFED5, FLUXCOM-X, CAMS, GridFED and EDGAR
  (`docs/memos/2026-10-10_python_flux_port_p0_facts.md`). `find_coordinate`
  finds coordinates by variable name only and checks their length;
  `latlon_orientation`/`orient`/`orient_coordinates` give ascending latitude
  and longitudes from the first value `≥ 0`; `read_raw` reads with the fill
  policies `MaskedAsZero`, `MaskedAsNaN` or `FillValueOnly`, converting each
  value once to the caller's element type (default Float64);
  `source_latlon_mesh` derives a file's cell faces (`SnappedFaces`: snapped
  regular axes, pole-centred, truncated and Gaussian latitudes, on
  `LatLonMesh(λᶠ, φᶠ; radius)`; bounds variables are never used);
  `decode_time` classifies a time axis by its units before any CFTime call
  (`CFDecoder`, `MonthsSince`, `DecimalYears`, `AuxiliaryTimeVariable`) and
  refuses invalid origins instead of normalising them (`months since
  2000-13-01`, `days since 2000-02-30` in the standard calendar);
  `decode_month_keys` gives the months of an axis in its own calendar, also
  where `DateTime` cannot hold a stamp (30 February of `360_day`);
  `stage_input` decompresses `.nc.gz` and single-member `.nc.tar.gz` inputs
  into a cache with atomic publication; `parse_flux_units` reads every units
  string in use into `FluxUnits`, converted by `kg_per_unit`,
  `period_seconds` and `mass_rate_factor` with the molar masses of
  `Parameters`.
- The `cs_native` flux file is defined once, in `GriddedIO`: the layout
  `CS_NATIVE_FLUX_DIMS`, the writer `open_cs_native_writer` / `write_slices!` /
  `publish!` / `abandon!` (contiguous start-stamped intervals with
  `time_bnds` in `hours since <first day>`, the mesh geometry and
  `cell_area`, units `canonical_units_string(species)`, chunks
  `(Nc, Nc, 1, chunk_time)`, no `_FillValue`, non-finite values refused,
  atomic publication), the streaming reader `cs_native_flux_info` /
  `read_cs_native_flux!`, which is now the runtime's `cs_native` loader (panel
  by panel into the source's own arrays, one panel-slab read buffer, as the
  former loader), and the whole-file reader `read_cs_native_flux`
  (`check_cell_area` is the former per-panel area gate, unchanged). Sources
  built from every `cs_native` file tested are bit-identical to the former
  loader. The reader is stricter: the NetCDF
  dimensions must be exactly `(time, nf, Ydim, Xdim)`, times must strictly
  increase, time axes are decoded by `decode_time`, and units in kg of
  another named species than the tracer's are an error (they were read with
  factor 1). `write_latlon_flux` writes static CF `(lon, lat)` flux files for
  `kind = "file"` (not a time-varying kind) and, until step P2b, refuses a mesh
  whose faces the runtime would not rebuild bit for bit from the written
  centres by its uniform face rule, now `GriddedIO.UniformSpanFaces`
  (pole-centred, snapped, truncated, Gaussian rows). `catrine_protocol_cs_fluxes.jl` writes through the
  `cs_native` writer: the same data, now with `time_bnds`, the geometry block
  and units `kg CO2|Rn222|SF6 m-2 s-1`; `compare_cs_flux_files.jl` takes the
  layout from `GriddedIO`.
- The runtime surface-flux and initial-condition loaders read their files
  through `GriddedIO`: coordinates by name with a check against the field's
  dimensions, orientation, fill values (read straight into the run's float
  type), the conservative-regridding source mesh
  (`source_latlon_mesh(…, UniformSpanFaces(); radius)`, the former uniform mesh),
  time axes and flux units. For every file the shipped configs read (102
  inventories: CAMS/LMDZ, GridFED, EDGAR, Zhang Rn-222, ECCO-Darwin, CATRINE and
  TRENDY `cs_native` series), the loaded fields, source meshes, slice times and
  storage factors are bit-identical to the former loaders. What changes:
  - Units go through one grammar (`GriddedIO.parse_flux_units`): grams, days,
    molar units with a named species and per-cell units with any time unit now
    convert; the conversion follows the units, not the kind (a
    `gridfed_fossil_co2` file is divided by the month length only if its units
    are per month, an `edgar_sf6` file by the cell area only if they are per
    cell). Molar units that name no species are moles of the tracer's
    species; missing and unknown units remain errors, now naming the variable
    and file. Units in kg of another
    named species than the tracer's (`surface_flux.species`) are an error;
    they were read with factor 1. Time-varying lat-lon sources need densities
    (per m²).
  - Time axes are classified by their units before any CFTime call
    (`GriddedIO.decode_time`): `months since` axes decode to month starts
    (CFTime used a fixed 30.44-day month), `noleap`/`365_day`,
    `all_leap`/`366_day` and `360_day` calendars are accepted (a stamp on a
    date the run's proleptic Gregorian clock lacks, such as 30 February, is an
    error), invalid origins are errors, `years since` and unlabelled axes are
    refused, and abbreviated units (`h since`, …) are decoded through
    `DateTime` (millisecond resolution).
  - The private helpers `_ic_find_coord`, `_build_source_latlon_mesh`,
    `_surface_flux_times_seconds`, `_parse_cf_time_origin`,
    `_normalize_units_string`, `_per_second_flux_scale`, `_days_in_month` and
    `_is_leap_year` are removed; the loaders `_load_file_surface_flux_field`
    and `_load_timevarying_surface_flux_field` take the `species` keyword.
- SIF-GPP, FLUXCOM, TRENDY v14, TRANSCOM, GFED, and CATRINE campaign
  preprocessing, run configs, visualization scripts, and ATBD memos.

### Coverage, benchmarks, and documentation

- Documentation corrections from an external review against the TM5 Fortran
  by pera-noaa (NOAA GML), GitHub issues #12, #15 and #20. Thank you.
  - `SlopesScheme` is no longer described as a port of TM5's slopes
    advection (#12). It uses TM5's Russell-Lerner face flux but re-diagnoses
    a minmod slope from the neighbouring cell means every sweep, while TM5
    carries the first moments `rxm`, `rym`, `rzm` as prognostic state and
    clamps them only to `|rxm| <= rm`. The README, *Validation status* (the
    row moves from parity to "differs"), *Advection schemes*, the operator
    pages, and the `SlopesScheme`, `MonotoneLimiter` and limiter docstrings
    are corrected, and the README no longer cites the `test_tm5_*` convection
    tests for advection. *Advection schemes* gives the measured difference
    (after 10 revolutions of a 1-D Gaussian the `SlopesScheme` L2 error is
    4.6 to 6.8 times TM5's at Courant numbers 0.05 to 0.75) and lists the
    missing prognostic-moment scheme as a known limitation; building one is
    postponed. Runs with `scheme = "slopes"`, including the README showcase,
    are diagnosed-slope (van Leer) results.
  - The replay gate is described by what it certifies (#15, part 1).
    *Binary format* has a new section, *What the replay gate certifies*:
    every preprocessor that builds binaries from meteorological data derives
    `cm` from the stored horizontal fluxes just before the gate, so the gate
    checks column budgets, the `cm` closure, storage rounding, substep counts
    and window chaining, but cannot detect rotational flux errors,
    level-flipped fluxes, a wrong vertical split (including `cm` close to
    zero), a flux through the model top and surface that is the same at
    every interface of a column, or large relative errors in thin cells. The gate's metric is the largest cell
    error divided by the largest cell mass; *Binary format*, *Mass
    conservation*, the ERA5 preprocessing page and the TM5/GCHP pages
    showed a norm ratio, and *Philosophy* claimed bit-level closure.
  - The adjoint pages list the current 4D-Var limits (#20):
    `cs_surface_flux_4dvar` runs one taped forward and reverse run per
    observation, so a gradient's cost grows linearly with the number of
    observations, and `bind_to_mesh` binds every observation to a column
    mean whatever its altitude (the schema's `alt` is above sea level and
    binaries carry no surface elevation). Both were documented only in
    `src/Inversion/README.md`.
- The Julia 1 CI job uploads line coverage of `src/` and `ext/` to Codecov.
  The upload authenticates with GitHub OIDC, so the repository holds no
  Codecov secret, and a failed upload does not fail CI. `codecov.yml` sets
  the status thresholds (project coverage may fall 2 percentage points below
  the base commit, coverage of the changed lines up to 20 points below it)
  and excludes tests, scripts, documentation, benchmarks, examples, and
  heritage folders. The hosted job runs only the
  CPU tiers, so the CUDA and Metal extensions count as uncovered.
- GPU benchmarks of the NVIDIA L40S (`Float32`), NVIDIA A100 (`Float32` and
  `Float64`), and Apple M5 Pro (Metal, `Float32`) lab machines reach the
  performance dashboard only through a lab lane started by hand on wurst
  (`benchmarking/lab/run_lab_benchmarks.jl`). Each lab machine's exact Julia
  version is pinned in `benchmarking/lab/machines.toml` (`julia_version`,
  1.12.6 on all three) and checked before anything runs. Every benchmark
  record carries an environment fingerprint (Julia version and sha256 of the
  resolved benchmark Manifest) in its raw `metadata` and its dashboard
  `extra`; the lab lane and the publish workflow refuse a lab run whose
  machines disagree on it, and the grouped dashboard shows it per point. A
  machine whose run fails still has its (partial) result files copied back
  before the failure is reported. The optional self-hosted CUDA and Metal jobs of the `Benchmarks`
  workflow, its `run_cuda`, `run_metal`, and `gpu_float_types` inputs, and
  `benchmarking/RUNNERS.md` are removed: self-hosted runners would run
  pull-request code on lab hardware of a public repository. The workflow's
  CPU job, which runs pull-request code, now has a read-only token; only the
  publish job writes (`contents`, `deployments`; no `pull-requests`).
- Benchmark harness: `--preset` loads a case matrix from
  `benchmarking/presets/` (the `lab` preset runs C90 and C180, L72 and L137,
  1 and 12 tracers: 44 cases per float type), and `--convection-depth`,
  `--device-label`, and `--commit` are new. GPU series names carry the
  device (`GPU-CUDA L40S`); CPU series names are unchanged. The `io` case
  now serializes every requested tracer; it moved one tracer whatever its
  name said. `adjoint` runs at one tracer only (the footprint of one scalar
  objective), so the `adjoint 4tr` CI series ends, and operators that ignore
  `--scheme` run once under the default `ppm`. `Operator Breakdown` series
  are named `Operator Breakdown / <case name> / <phase>` (the old names
  merged tracer counts, groups, and schemes), so those series start a new
  history, and the `io` and `adjoint` phase values are now per step: the
  adjoint value was divided by the repeat count twice, and the io value by
  `--steps` although a repeat is one round trip. The dashboard `extra` field
  is written as a JSON string; it was published as `[object Object]`. A
  case that throws is kept as `failed` in the raw JSON, and the run exits
  with an error after writing both files. The synthetic face mass fluxes
  are now steady (`benchmarking/src/synthetic_fields.jl`): the old fields
  disagreed at panel seams and had unbalanced top and bottom layers, so
  repeated steps drained edge cells by up to 11 % of their mass per step
  until the Courant number exploded (the first lab case, C180 L72, stopped
  at the 4096-subcycle limit). The new fields leave the air mass unchanged
  at a palindrome Courant number of about 0.1 on every grid, and the
  harness fails a case whose Courant number exceeds 0.5 or whose air mass
  drifts. The lab preset now runs C90, C180, C360, and C720 at L72 and
  L137 with 2 tracers. A typed `[limits]` table keeps `io` and `adjoint` at
  C180 and below, which gives 40 cases per float type.
  Before building a case, the harness estimates its device memory from a
  linear model of the model's arrays, fitted on CPU
  (`benchmarking/src/memory_model.jl`). A case that does not fit, or that
  runs out of memory, is recorded as `skipped_memory` with the required and
  available GiB instead of failing the run; skipped cases are not published,
  and the lab lane accepts and lists them. `--device-memory-gib` overrides
  the device query. Metal.jl 1.9 only prints a failed command buffer, so
  the harness captures standard output while a Metal case runs. A sentinel
  command buffer's completion handler marks the point where all of the
  case's handlers have reported. If every failed buffer ran out of memory,
  the case is `skipped_memory`; any other failure makes it `failed`. A
  kernel that never ran therefore cannot be published as ok. The CPU `io` case no longer copies its host payload to serialize
  it; on the CPU its estimate counts the read-back copy. The grouped dashboard draws one
  line per group, grid, levels, scheme, and device; lines of renamed groups
  no longer merge.
- The README opens with badges for CI, the stable and development manuals,
  the license, the Julia compatibility bound (read from `Project.toml`),
  Aqua, Codecov, and the performance dashboard.
- New manual page *Performance benchmarks* (section *Development*): what the
  synthetic cubed-sphere benchmarks measure, the metric definitions
  (time per step, cell-tracer updates per second), the lab machines and their
  roles, how results reach the dashboard, how to run a benchmark locally, and
  how to read the dashboard. `CONTRIBUTING.md` explains the benchmarks and
  coverage.

### Post-run diagnostics

- New module `AtmosTransport.Analysis` (loaded after `Models`): the shared
  numerics of post-run diagnostics. `daily_snapshot_series` concatenates the
  frames of snapshot files (explicit run origin; a repeated hour is read once);
  `EqualAreaBands` and `band_means` form area-weighted means on bands equally
  spaced in sin φ, leaving NaN cells out of numerator and denominator;
  `doy_linear_detrend` removes a per-calendar-day line across years (29
  February folded onto 28 February); `seasonal_trend_decomposition` fits 12
  calendar-month intercepts and a trend on the observed months (minimum norm);
  `skill_metrics`, `weighted_fit_stats`, `column_band_weights` and
  `logp_interpolate` complete it. On the TRENDY C30 ranking inputs the monthly
  band means equal the Python ones bit for bit and the decomposition agrees to
  6e-14 ppm with the same ranks; on the SIF-GPP control the printed statistics
  agree to every digit. `test/diagnostics/test_trendy_hovmoller.py` is ported
  to `test/core/test_analysis_decomposition.jl`;
  `scripts/diagnostics/catrine_station_timeseries.py` moves to `heritage/`;
  `compare_cs_flux_files.jl` takes its sin-latitude bands from `Analysis`
  (the same assignment as before on C4, C8, C30, C90, C180, C360 and C720 in
  both panel conventions).
- Budget closure in Julia. `Analysis.budget_closure(run)` checks every
  tracer of a finished run: the change of `<tracer>_total_mass` between the
  first and last snapshot against the storage the run's own surface sources
  add (`build_surface_flux_sources` + `integrated_emission`, over the run's
  own steps: each window's step count and the float-type step
  `FT(L)/FT(n)` from the run clock's step starts, so a Float32 window of
  3600 s in 7 steps adds 3599.99994 s), within
  ρ|E| + k·eps(Float32(carrier))·air mass (`BudgetTolerance`: strict 1e-4 and
  1 ulp, operational 1e-3 and 4 ulps), with finite values and, for a positive
  carrier, positive column means. `RunContext` builds the run's view with the
  runner's own functions (tracer specs, including the legacy single tracer of
  lat-lon runs, now read by the structured runner and `RunContext` through one
  function `_structured_tracer_specs`; grid; every binary's window layout; run
  start; the air mass of window 1 that the initial state is packed on; output
  file names; window ranges and tracer tables the runner rejects are errors,
  with the runner's multi-file range check shared as
  `_check_multifile_window_count`), and `carrier_drawdown`
  gives the largest cumulative loss of global mixing ratio before a run.
  `scripts/diagnostics/tracer_budget_closure.jl series <run.toml>` replaces
  `check_trendy_npp_rh_run.py` (JSON report, exit 1 when the gated level
  fails); `scripts/run_trendy_npp_rh_campaign.py` calls it. On the 23-model
  TRENDY C30 run (3775 days) the harness emulator of the Python arithmetic
  (`scripts/validation/flux_port_parity/budget_closure_parity.jl`) reproduces
  `validation.json` (expected changes to 7e-16, actual changes identical, TEM
  NPP failing at 1.2714e-4, 45 of 46 passing) and `smoke_validation.json`
  (max 1.96e-4). With the current `DRY_AIR_MOLAR_MASS` (the run used the
  pre-`d552d3e0` 0.02896546) every expected change is 3.66e-5 smaller and 43
  of 46 pass (TEM, CLM-FATES and ELM NPP fail); at the run's constant the
  runtime sources differ from the Python sums by at most 5.5e-8, so cell
  areas and integration explain none of the TEM excess. The NPP drawdowns
  equal the Python audits times DRY_AIR_MOLAR_MASS/0.02896546 to 5.5e-8. The
  single-file mode of `tracer_budget_closure.jl` read the bare-hours time axis
  of current snapshots as years and reported a gap of about −1; it now reads
  hours since the run start.

### Verification and remaining limits

- `test/core/test_slopes_tm5_reference.jl` checks `SlopesScheme` against an
  independent transcription of TM5's `dynamu` (`advectx.F90`) in
  `test/helpers/tm5_dynamu_reference.jl`: with the moment re-diagnosed by
  minmod the TM5 update equals `SlopesScheme()`, and with the moment set to
  zero `UpwindScheme()`, to rounding (non-negative and signed fields,
  eastward, westward and divergent flow). With the prognostic moment, TM5
  keeps 0.990 of a Gaussian's peak after one revolution against 0.892, and
  TM5's positivity clamp is not offset-equivariant while `SlopesScheme` is.
  The transcription itself keeps a uniform field exact, conserves mass and
  converges at second order.
- `test/core/test_replay_consistency.jl` has a *Replay gate scope* testset
  for the statements in *What the replay gate certifies*: the normalization
  by the largest cell mass, and that with `cm` re-diagnosed as the
  preprocessors do it a rotational flux error, an `am`/`bm` level flip and a
  column-uniform `cm` offset pass while a broken column budget fails.
- The golden-output harness (`test/golden/run_goldens.jl`) runs its cases in
  worker sessions instead of one Julia process per case: one process
  (`golden_worker.jl`) per group of cases with the same entry point,
  `[architecture]` table and threads, which includes the entry scripts once
  and calls their `main(args)` for each case, with the case's output and log
  messages in its `log.txt` and the logger, working directory and environment
  restored after it. A full check of the 27 cases takes 41 minutes instead of
  68 on an L40S (preprocessing 16 instead of 18, CPU runs 16 instead of 23,
  GPU runs 8 instead of 27), with outputs bitwise identical to the
  per-process reference. The harness watches each worker: it writes a
  case's `status.toml` as soon as the case finishes (an interrupted run
  keeps them), continues the remaining cases in a new worker when one
  crashes, stops a worker whose case runs longer than `--timeout` minutes
  (default 45) and fails that case, and starts a new worker when one reports
  a poisoned session (a CUDA `synchronize` or `reclaim` that fails after a
  case, the mark of a sticky device error). `status.toml` records the case's
  session and position (`[session]`, with `poisoned` when it applies) and
  the `error` of a failure; `check` prints wall times as `(own process)`,
  `(session case k of n)` or `failed` and does not compare them.
  `scripts/run_transport.jl` and
  `scripts/preprocessing/preprocess_transport_binary.jl` define
  `main(args)`, which runs in the calling Julia session with a logger scoped
  to the call (the scripts set the global logger before) and throws on error;
  the thread re-execution and GPU preload of `run_transport.jl` happen only
  when it runs as a script. `ATMOSTR_PROFILE_MODE` now follows the config's
  `[architecture]` instead of whether CUDA happens to be loaded: a CUDA run
  is profiled also on the first call of a session that included the script
  (CUDA is loaded for it), and a CPU or Metal run warns and runs without a
  profile. `preprocess_transport_binary.jl` accepts `-h`/`--help`.
  `test/core/test_golden_sessions.jl` tests the grouping, the worker's
  isolation, crashes before and between cases (also with exit status 0), a
  poisoned session, timeouts while loading, in a case and in the cleanup
  after a case, SIGKILL after an ignored SIGTERM, and `main(["--help"])` of
  both entry scripts as a session includes them.
- New core tests cover config parsing, cell location on every topology
  against independent references, source readers (including a real OCO-2
  Lite file check), the device gather, pressure and height
  reconstruction on GEOS L72, the sampler's time bookkeeping, and end-to-end
  lat-lon and cubed-sphere runs, including identical results across a
  binary handoff. An opt-in CUDA test covers the gather on an L40S.
- Station placement uses a constant temperature except on cubed-sphere
  binaries with GCHP VDIFF temperatures. Averaging kernels are applied
  offline.

## 0.4.0 — 2026-09-06

This release changes cubed-sphere numerical results and runtime/output
behavior. Keep the version, forcing binary, run TOML and dependency manifest
with reproducibility records; recheck conservation and scientific diagnostics
when upgrading from 0.3.0.

### Numerical changes

- Cubed-sphere split advection and Lin–Rood now share conservative face
  transfers across panel seams. Previous runs could accumulate tracer mass
  drift at those seams.
- Precomputed TM5 Dkg diffusion advances conservative tracer storage with
  bidiagonal factors. It preserves weak exchange into empty layers, isolated
  layers and signed totals more accurately. Its adjoint uses the matching
  transpose. This is a numerical correction, so old and new output need not
  match bit for bit.
- CUDA PPM launch tiles, separate Dkg factor/tracer launches and batched
  convection solves reduce work without lowering configured precision.
  CUDA collaborative Float64 convection supports full columns through 73
  levels within its shared-memory budget; Float32 retains its portable depth
  limit. Explicit `lmax_conv`/`n_merge` choices remain numerical approximations.
- Lin–Rood recording and reverse halo propagation use backend kernels and
  pass transporting CUDA adjoint checks with scalar indexing disabled.

### Runtime and output

- Input drivers and device workspaces are reused across daily files. Rolling
  staging owns its cache and checks retained source identity; failed prefetch
  and output tasks are consumed during cleanup.
- Single-file NetCDF output streams selected snapshots and records
  `completed_snapshots`. A failed run may leave partial output; file existence
  is not evidence of completion. Reopening a stream to resume is unsupported.
- Selected NetCDF output avoids retaining full tracer volumes on the host.
  Signed Float64 `<tracer>_total_mass` diagnostics are captured independently
  of spatial output precision. Metal uses bounded host slabs for Float64
  totals and column accumulation; CUDA retains device Float64 reductions.
- `SnapshotFrame` and `SnapshotWriteOptions` are part of the curated top-level
  API. Write and resource-close failures are preserved together rather than
  allowing one to mask the other.
- Runtime preflight reports malformed tables, including `[input.staging]`
  and tracer subtables, before opening binaries. Window indices must be
  integers. Multiple input files require their complete window ranges;
  omit `stop_window` for a full multi-day run. Nine obsolete 48-hour configs
  now live under `config/runs/likely_legacy/`.

### Dependencies and documentation

- Add explicit `FileWatching` and `HDF5_jll` dependencies for staging ownership
  and NetCDF error-handler handling. HDF5_jll compatibility accepts 1.14 and 2.
- Remove self-version bounds from test and benchmark environments so release
  bumps do not invalidate them. Julia 1.10 remains the minimum supported version.
- Add an executed emission-footprint tutorial with a finite-difference check
  and a beginner's guide to adjoints, observations, priors and inversion.
  Correct preprocessing configuration/balance explanations and distinguish
  hosted CPU CI from separate GPU verification.

### Verification and remaining limits

The branch records Julia 1.10/1.12 CPU test runs, strict documentation builds,
all ten maintained L40S GPU diagnostics, V100 adjoint/output checks, and
real-input Float32 C90/L66 comparisons. The matched L40S day benchmark improves
from 7.35 to 4.54 seconds for six tracers and from 47.47 to 13.84 seconds for
32 tracers. The 32-tracer 0.3.0 baseline uses its supported legacy solver,
because that release's collaborative solver is capped at six tracers. These
measurements include runtime setup and I/O and apply to the documented workload;
see [benchmark evidence](scripts/benchmarks/results/release_fp32_l40s_20260906/README.md).

The C90/L66 Float32 forward smoke test passed on an Apple M5 Pro (20 GPU
cores), with six and 32 tracers, full TM5 collaborative convection, Dkg
diffusion and column output. Warmed runs took 2.90 and 8.28 seconds; maximum
column relative L2 differences from CUDA were below `9e-8` and relative mass
drift below `5.7e-8`. See the [Metal verification record](docs/memos/release_readiness_20260906.md).
This does not establish coverage of Metal adjoints or every operator. Optimized,
clamped or reduced-column convection paths do not all have supported adjoints.
The shipped inversion CLI remains synthetic-only; real-data inversion assembly
and external TM5-4DVAR cross-validation are separate tasks. Numerical checks
and performance measurements are not a complete observational validation.

## 0.3.0

The preceding release is the baseline for the changes above. This changelog
was introduced for 0.4.0; it does not reconstruct earlier release histories.
