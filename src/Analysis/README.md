# Analysis

Shared numerics for post-run diagnostics. Diagnostic command lines in
`scripts/diagnostics/` take these numerics from here instead of carrying their
own copies, and plotting scripts (which may stay in Python) read the NetCDF or
JSON those diagnostics write. `scripts/validation/compare_cs_flux_files.jl`
takes its sin-latitude bands from `EqualAreaBands`. Everything runs on host arrays and accumulates in
Float64 in a fixed order, so results do not depend on threading or on the
element type of the input.

`src/AtmosTransport.jl` loads this folder after `Models` and before
`Downloads`. The functions so far need NCDatasets, Dates, LinearAlgebra and
Statistics, the compensated sum of `Architectures`, the constants of
`Parameters` and the NetCDF lock of `Output`; the budget closure also builds
on the runtime's config parsing, grids, output naming and surface-flux sources
(`Models`, `MetDrivers`, `Output`, `Operators`). No earlier module depends on
`Analysis`. The names are
exported from `AtmosTransport.Analysis`, not from the top level:

```julia
using AtmosTransport.Analysis
```

This module is the diagnostics track of the Python → Julia port
(`docs/memos/2026-10-10_python_flux_port_design.md`, Sections 1c and 2.8).
Each function names the Python routine it replaces, and its tests pin values
computed by that routine.

## Entry Points

- `daily_snapshot_series(paths, variables; origin=nothing)` in
  [`snapshot_series.jl`](snapshot_series.jl): the frames of runtime snapshot
  files, concatenated in file order, as `(; times::Vector{DateTime},
  values::Dict{String, Array})` with time as the last dimension. Current files
  decode the simulation start from CF `hours since <run start>` units; `origin`
  is only needed for legacy files with bare `"hours"`. A repeated time is
  dropped (first occurrence wins); fill values become NaN.
  `keep_frame = t -> ...` reads only the frames whose time it accepts (one
  field per day from 3-hourly snapshots).
- `EqualAreaBands(n)`, `BandAssignment(bands, lat, area)`,
  `band_means(field, lat, area, bands)`, `band_means(field, assignment)` and
  `band_means!` in [`latitude_bands.jl`](latitude_bands.jl): area-weighted
  means on bands of equal area (edges equally spaced in sin φ). NaN cells are
  left out of the numerator and the denominator. `AbstractLatitudeBands`
  (`nbands`, `band_edges`, `band_index`) is the interface for other band sets;
  `band_index` may return 0 for a latitude in no band.
- `doy_linear_detrend(x, dates)` and
  `seasonal_trend_decomposition(series, seen; min_months, min_years_per_month,
  first_month)` in [`decomposition.jl`](decomposition.jl): a per-calendar-day
  linear fit across years (29 February folded onto 28 February), and a joint
  fit of 12 calendar-month intercepts plus a linear trend on the observed
  months, returning `(; anomaly, seasonal, detrended)`. Both solve through
  `pinv` (minimum norm).
- `skill_metrics(model, reference)`,
  `weighted_fit_stats(model, reference, weights)` and
  `weighted_difference_stats(model, reference, weights)` in
  [`skill.jl`](skill.jl): `(; r, rmse, bias, amplitude_ratio, n)` with
  population moments, the area-weighted `(; r2, slope, bias, rmse, ref_mean,
  ref_std)`, and the weighted `(; mean, rms, n)` of the per-cell difference
  (compensated sums; a Float32 field against its Float64 twin).
- `drift_statistics(series)` and `twin_relative_difference(candidate,
  reference)` in [`mass_totals.jl`](mass_totals.jl): the drift of a global
  total relative to its first value with the mean, sample standard deviation
  and t-value of its increments (a source-free tracer's arithmetic error), and
  the frame-by-frame `(; raw, net)` relative difference of a total from its
  twin run, net of the first frame unless the reference starts at zero.
- `column_band_weights(m, bands)` (and `!`) and `logp_interpolate(p, p_mid, q;
  outside = NaN)` (and `!`) in [`column_profiles.jl`](column_profiles.jl):
  per-layer air mass inside bands of column-mass fraction counted from the
  surface, and linear interpolation in ln p with no extrapolation.
- `GEOSChemCATRINEArchive(dir; tracers, interval)` in
  [`geoschem_archive.jl`](geoschem_archive.jl): the output of the GEOS-Chem
  CATRINE reference run, `GEOSChem.CATRINE_inst.YYYYMMDD_HHMMz.nc4` every 3 h,
  flat or under `YYYY/MM/`. `gc_path(a, t)` resolves a file, `gc_times(a,
  start, stop)` lists the output times, `with_gc_file(f, a, t)` opens one
  file (read into memory, decoded under the NetCDF lock) and returns `f(ds)`
  or `nothing` when the file is unreadable, `global_totals(a, t)` (or a vector
  of times, on all threads) returns Σ Met_AD and, per tracer, Σ χ·Met_AD and
  the emission rate (Σ E·Met_AREAM2)·M_air/M in dry-air-equivalent kg, and
  `unreadable_files(a)` lists the files that could not be read.
  `GEOSChemTracer` names a tracer's variables, GEOS-Chem molar mass and
  half-life; `CATRINE_GEOSCHEM_TRACERS` holds the four CATRINE tracers.
- `budget_closure(run; output_glob, tolerance, emission, tracers,
  require_complete)` in [`budget_closure.jl`](budget_closure.jl): the global
  budget of every tracer of a finished run, `ΔS = S(t₁) − S(t₀)` from
  `<tracer>_total_mass` of the run's snapshot files against the storage `E`
  its runtime sources added (`Operators.integrated_emission`), with the
  tolerance `ρ|E| + k·eps(FT(χ₀))·M`, where `FT` is the run's float type
  (`BudgetTolerance`: strict 1e-4 and
  1 ulp, operational 1e-3 and 4 ulps), finite values and a carrier that is
  never exhausted. Returns a `BudgetReport` of `TracerBudget`s;
  `report_dict` gives its JSON form. `carrier_drawdown(run)` is the largest
  cumulative loss of global mixing ratio that each tracer's sources cause over
  the run (a carrier must exceed it). The expected change is pluggable:
  `AbstractExpectedEmission` with `expected_storage_change(method, ctx, spec,
  t0, t1)`; the default is `RuntimeSurfaceSources()`.
- `RunContext(run_config)` and `run_snapshot_paths(ctx)` in
  [`run_context.jl`](run_context.jl): a run seen as the runner sees it, built
  with the runner's own functions (tracer specs, including the legacy
  single-tracer form of lat-lon runs, grid of the first binary,
  float type, run start, every met window with its step count and the
  float-type step length the runtime applies, run length, the initial air
  mass of window 1, decaying tracers; window ranges the runner rejects are
  errors), and the snapshot files it writes (a daily file only for binaries
  that take a snapshot).

## File Map

- [`Analysis.jl`](Analysis.jl): module definition, imports, includes and
  exports.
- [`snapshot_series.jl`](snapshot_series.jl): `daily_snapshot_series` and its
  NetCDF helpers.
- [`latitude_bands.jl`](latitude_bands.jl): band sets and band means.
- [`decomposition.jl`](decomposition.jl): `doy_linear_detrend`,
  `seasonal_trend_decomposition`.
- [`skill.jl`](skill.jl): `skill_metrics`, `weighted_fit_stats`,
  `weighted_difference_stats`.
- [`column_profiles.jl`](column_profiles.jl): `column_band_weights`,
  `logp_interpolate`.
- [`mass_totals.jl`](mass_totals.jl): `drift_statistics`,
  `twin_relative_difference`.
- [`geoschem_archive.jl`](geoschem_archive.jl): `GEOSChemTracer`,
  `CATRINE_GEOSCHEM_TRACERS`, `GEOSChemCATRINEArchive`, `gc_path`, `gc_times`,
  `with_gc_file`, `global_totals`, `unreadable_files`.
- [`run_context.jl`](run_context.jl): `RunContext`, `run_snapshot_paths`.
- [`budget_closure.jl`](budget_closure.jl): `BudgetTolerance`,
  `TracerBudget`, `BudgetReport`, `budget_closure`, `report_dict`,
  `carrier_drawdown`, `AbstractExpectedEmission`, `RuntimeSurfaceSources`,
  `expected_storage_change`.

## Python Routines Replaced

| Julia | Python |
|---|---|
| `EqualAreaBands`, `band_means` | the `numpy.bincount` band means of `rank_trendy_hovmoller.py` (`extract`), `run_transcom_campaign.py` (`summarize`), `plot_sif_gpp_iav_xco2.py` (`_sinlat_bands`) |
| `doy_linear_detrend` | `interannual` of `compare_sif_gpp_transport_control.py` |
| `seasonal_trend_decomposition` | `decompose` of `rank_trendy_hovmoller.py` |
| `skill_metrics` | `metrics` of `rank_trendy_hovmoller.py` |
| `weighted_fit_stats` | `wstats` of `catrine_compare_vs_geoschem.py` |
| `column_band_weights` | `band_sums` of `catrine_compare_vs_geoschem.py` |
| `logp_interpolate` | the interpolation in `zonal_on_pressure` of `catrine_compare_vs_geoschem.py` |
| `drift_statistics` | `increment_stats` and the drift column of `float32_conservation_attribution.py` (retired; now `float32_conservation_attribution.jl`) |
| `twin_relative_difference` | `f32_minus_f64` of `float32_conservation_attribution.py` and the `rel`/`net` series of `compare_float32_float64_runs.py` (both retired; now `.jl`) |
| `weighted_difference_stats` | the daily area-weighted mean and RMS of `compare_float32_float64_runs.py` |
| `GEOSChemCATRINEArchive`, `gc_path`, `global_totals` | `gc_path` and `one` of `catrine_true_mass_balance_gc.py` (retired; its mass balance is `catrine_true_mass_balance.jl --model geoschem`); `gc_path` of `catrine_compare_vs_geoschem.py` |
| `budget_closure`, `report_dict` | `check_trendy_npp_rh_run.py` (retired; now the `series` mode of `scripts/diagnostics/tracer_budget_closure.jl`) |
| `carrier_drawdown` | the `npp_max_global_drawdown_ppm` audit of `prepare_trendy_s3_npp_rh.py` |

A Python script is deleted in the commit whose Julia command line replaces
it and is listed under "Retired scripts" in `scripts/README.md`; the scripts
above not marked retired follow in steps B3 and B4 of the design.

## Invariants

- Band means skip NaN cells in the numerator **and** the denominator.
  `_sinlat_bands` in `plot_sif_gpp_iav_xco2.py` kept the area of NaN cells in
  the denominator, biasing a band with missing cells toward zero; the two
  agree when a field has no NaN.
- Band sums run in linear (column-major) cell order, the order of
  `numpy.bincount`; with the same band assignment the band means are
  bitwise equal to the Python ones.
- `EqualAreaBands` edges are `(k / n) * 2 - 1` and a latitude's coordinate is
  `sin(deg2rad(φ))`, as in the Python diagnostics (`np.linspace(-1, 1, n + 1)`
  differs from these edges by one ulp at some edges).
- The decompositions are linear and remove constants exactly: adding a carrier
  to a tracer series does not change anomaly, seasonal cycle or detrended
  series.
- Skill statistics take the model first and the reference second and use
  population (1/n) moments. (`wstats` in Python took the reference first.)
- `column_band_weights` and `logp_interpolate` take layers top first. Profiles
  stored surface first (GEOS-Chem) are reversed by the caller.
- `logp_interpolate` treats both end mid levels as inside the column. The
  Python `zonal_on_pressure` returned NaN for a target exactly equal to the
  top mid pressure; elsewhere the two agree.
- `weighted_difference_stats` forms the difference per cell before any sum and
  sums with Neumaier compensation, so a mean far below the spread of the
  differences keeps its digits (the `bias` of `weighted_fit_stats`, a
  difference of two means, does not). It leaves NaN cells out (the Python
  comparison propagated NaN); on C90 fields without NaN it agrees with NumPy's
  pairwise sums to ≤ 1.5e-14 relative.
- `twin_relative_difference` and the drift of `drift_statistics` repeat the
  Python Float64 operations, so they are bitwise equal to the Python values.
  `drift_statistics` returns `t = ±Inf` for equal non-zero increments, where
  the Python returned 0.
- `global_totals` reads the same Float32 values as the Python script (fill
  values as zero), forms the same Float64 products and converts emissions in
  the same order, `Σ · M_air / M`; it sums pairwise in linear-index order,
  where NumPy sums pairwise within blocks of 8192 elements. On the first day
  of the CATRINE C90 archive, re-summing in NumPy's order reproduces all 44
  Python values bit for bit; over all 6088 files the Julia dry-air masses and
  burdens differ from the Python ones by at most 5e-15 relative. `valid_range`
  is not applied (NCDatasets does not; GEOS-Chem writes no value outside it).
- netcdf-c is not thread-safe: `with_gc_file` decodes under
  `Output.with_netcdf_lock`, but reads the file bytes outside it, so threads
  overlap their (network) reads.
- `budget_closure` works in tracer storage (mixing ratio × carrier air mass),
  so molar masses cancel, and takes the expected change from the run's own
  sources: cell areas, unit factors, storage scale `DRY_AIR_MOLAR_MASS / M`
  and slice times are the runtime's. `check_trendy_npp_rh_run.py` used the
  flux file's `cell_area`, whole-day sums and `M_air = 0.02896546`; on the
  TRENDY C30 run its expected changes are 3.66e-5 larger (the constant) and
  differ by ≤ 6e-8 otherwise. A run made before commit `d552d3e0` applied the
  old constant, so its runtime-consistent budget is off by that 3.66e-5.
- The carrier term uses the run's initial air mass from the first window of
  the binary; the Python script used `total_mass(t₀) / carrier` (2.4e-9
  relative apart on the TRENDY run). The carrier is the `background` of a
  `uniform` initial condition, otherwise the initial mean `S(t₀)/M`.
- A linearly interpolated source is sampled at every step start, so its
  integral is summed window by window with each window's step count (real
  binaries vary it, e.g. 4 to 6 steps per hour); stepwise, conservative-mean
  and static sources are integrated exactly in one step.
- A tracer with a positive carrier fails when its snapshots have no
  `<tracer>_column_mean`: carrier exhaustion cannot be ruled out (the Python
  script required the variable too).
- Snapshot `time` is hours since `[input].start_date`. A daily total cannot
  resolve a decay sink, so tracers with a `[chemistry]` half-life are skipped
  (the single-file mode of `tracer_budget_closure.jl` handles rn222).

## Common Tasks

- Monthly band means from daily snapshots: read with
  `daily_snapshot_series`, build one `BandAssignment` from the file's `lats`
  and `cell_area`, call `band_means(values, assignment)` to get
  `(nbands, ntime)`, then average the days of each month.
- Score a model against observations on the same months: stack the band
  series as `(nband, nmonth, nseries)`, call `seasonal_trend_decomposition`
  with the observation mask, then `skill_metrics(model, obs)` on each output.
- Compare a Float32 run with its Float64 twin: read `<tracer>_total_mass`
  of both with `daily_snapshot_series`, match the times, and call
  `twin_relative_difference`; read the column means with `keep_frame` and
  call `weighted_difference_stats(f32, f64, cell_area)` per frame
  (`scripts/diagnostics/compare_float32_float64_runs.jl`).
- Gate a campaign on its budgets: `report = budget_closure("run.toml")`,
  then `report.passed` (strict) or `report.operational_passed`, and write
  `report_dict(report)` as JSON
  (`scripts/diagnostics/tracer_budget_closure.jl series <run.toml>`).
  Before the run, compare `carrier_drawdown("run.toml")` with each carrier.
- Add a band set: subtype `AbstractLatitudeBands` and implement `nbands`,
  `band_edges` and `band_index` (returning 0 for latitudes in no band, for
  example the cells at exactly 90° of `np.digitize` latitude bins).
