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
Statistics; the budget closure also builds on the runtime's config parsing,
grids, output naming and surface-flux sources (`Models`, `MetDrivers`,
`Output`, `Operators`), the compensated sum of `Architectures` and the
constants of `Parameters`. No earlier module depends on `Analysis`. The names
are
exported from `AtmosTransport.Analysis`, not from the top level:

```julia
using AtmosTransport.Analysis
```

This module is the diagnostics track of the Python → Julia port
(`docs/memos/2026-10-10_python_flux_port_design.md`, Sections 1c and 2.8).
Each function names the Python routine it replaces, and its tests pin values
computed by that routine.

## Entry Points

- `daily_snapshot_series(paths, variables; origin)` in
  [`snapshot_series.jl`](snapshot_series.jl): the frames of runtime snapshot
  files, concatenated in file order, as `(; times::Vector{DateTime},
  values::Dict{String, Array})` with time as the last dimension. `origin` is
  the simulation start (snapshots store `time` in bare `"hours"`); a repeated
  hour is dropped (first occurrence wins); fill values become NaN.
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
- `skill_metrics(model, reference)` and
  `weighted_fit_stats(model, reference, weights)` in [`skill.jl`](skill.jl):
  `(; r, rmse, bias, amplitude_ratio, n)` with population moments, and the
  area-weighted `(; r2, slope, bias, rmse, ref_mean, ref_std)`.
- `column_band_weights(m, bands)` (and `!`) and `logp_interpolate(p, p_mid, q;
  outside = NaN)` (and `!`) in [`column_profiles.jl`](column_profiles.jl):
  per-layer air mass inside bands of column-mass fraction counted from the
  surface, and linear interpolation in ln p with no extrapolation.
- `budget_closure(run; output_glob, tolerance, emission, tracers,
  require_complete)` in [`budget_closure.jl`](budget_closure.jl): the global
  budget of every tracer of a finished run, `ΔS = S(t₁) − S(t₀)` from
  `<tracer>_total_mass` of the run's snapshot files against the storage `E`
  its runtime sources added (`Operators.integrated_emission`), with the
  tolerance `ρ|E| + k·eps(Float32(χ₀))·M` (`BudgetTolerance`: strict 1e-4 and
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
- [`skill.jl`](skill.jl): `skill_metrics`, `weighted_fit_stats`.
- [`column_profiles.jl`](column_profiles.jl): `column_band_weights`,
  `logp_interpolate`.
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
- Gate a campaign on its budgets: `report = budget_closure("run.toml")`,
  then `report.passed` (strict) or `report.operational_passed`, and write
  `report_dict(report)` as JSON
  (`scripts/diagnostics/tracer_budget_closure.jl series <run.toml>`).
  Before the run, compare `carrier_drawdown("run.toml")` with each carrier.
- Add a band set: subtype `AbstractLatitudeBands` and implement `nbands`,
  `band_edges` and `band_index` (returning 0 for latitudes in no band, for
  example the cells at exactly 90° of `np.digitize` latitude bins).
