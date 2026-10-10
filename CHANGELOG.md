# Release notes

## Unreleased

### Fixes

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
- `ATMOSTR_NO_WRITE_REPLAY_CHECK=1` did not skip the write-time replay gate
  of the GEOS cubed-sphere writer. It does now, through
  `verify_window!(…; write_replay_on)`, and its log no longer reports a worst
  replay window when the gate was skipped. With the gate skipped, the
  cubed-sphere regrid, ERA5 N320 and MERRA-2 writers ran the positivity gate
  against the window's start mass only; they now also pass the end mass, as
  with the gate on (only runs with the variable set are affected).
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
  mode. `[numerics] balance_mode = "column" | "per_layer"` (old name
  accepted) now selects it on every path, the LL-to-CS regrid script takes
  `--balance-mode`, and every transport-binary header records
  `horizontal_balance`. The environment variable still works where it did
  (every path except GEOS) when the key is absent, with a deprecation
  warning. Default results are unchanged.

### Numerical changes

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

### Runtime and output

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
  every time step and is still honored, with a deprecation warning, when the
  key is absent. Unlike the variable, `"substep"` keeps the window-end reset to
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
  next). It replaces the environment variable `ATMOSTR_REPLAY_CHECK`, which
  still works for one release with a deprecation warning;
  `ATMOSTR_NO_REPLAY_CHECK` is removed (the check runs only when asked for).
  The replay error messages now name the key.
- A transport binary written with the write-time replay gate skipped
  (`ATMOSTR_NO_WRITE_REPLAY_CHECK=1`, or the new `--no-write-replay-check` of
  `regrid_ll_transport_binary_to_cs.jl`) records `write_replay_check = false`
  in its header; `binary_capabilities` reports it, `inspect_binary` marks it
  and `TransportBinaryDriver` warns when it opens such a binary. Binaries
  written with the gate on are unchanged. Every writer asks one resolver,
  `write_replay_check_enabled`, instead of reading the variable itself.
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
  (`docs/src/theory/vertical_transport.md`). Other met
  sources reject the key.

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
  converted by 44/12), reject fill values, and must match the runtime mesh
  when the file carries `cell_area`.
- `[vertical].level_selection` presets are rejected on sources without 137
  native levels, and `[vertical].coefficients_file` is honoured as an alias of
  `coefficients`.
- Experimental `coarsen_nested_cs_transport_binary` restricts a cubed-sphere
  transport binary to a coarser nested grid (for example C90 to C30) by
  block-summing the transport operator; not yet validated as a replacement
  for preprocessing at the target resolution.
- SIF-GPP, FLUXCOM, TRENDY v14, TRANSCOM, GFED, and CATRINE campaign
  preprocessing, run configs, visualization scripts, and ATBD memos.

### Verification and remaining limits

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
