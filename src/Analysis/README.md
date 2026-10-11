# Analysis

Shared numerics for post-run diagnostics. Diagnostic command lines in
`scripts/diagnostics/` take these numerics from here instead of carrying their
own copies, and plotting scripts (which may stay in Python) read the NetCDF or
JSON those diagnostics write. `scripts/validation/compare_cs_flux_files.jl`
takes its sin-latitude bands from `EqualAreaBands`. Everything runs on host arrays and accumulates in
Float64 in a fixed order, so results do not depend on threading or on the
element type of the input.

`src/AtmosTransport.jl` loads this folder after `Models` and before
`Downloads`. The functions so far need only NCDatasets, Dates, LinearAlgebra
and Statistics; no earlier module depends on `Analysis`. The names are
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

The Python scripts stay until their Julia command lines replace them (steps
B3 and B4 of the design).

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

## Common Tasks

- Monthly band means from daily snapshots: read with
  `daily_snapshot_series`, build one `BandAssignment` from the file's `lats`
  and `cell_area`, call `band_means(values, assignment)` to get
  `(nbands, ntime)`, then average the days of each month.
- Score a model against observations on the same months: stack the band
  series as `(nband, nmonth, nseries)`, call `seasonal_trend_decomposition`
  with the observation mask, then `skill_metrics(model, obs)` on each output.
- Add a band set: subtype `AbstractLatitudeBands` and implement `nbands`,
  `band_edges` and `band_index` (returning 0 for latitudes in no band, for
  example the cells at exactly 90° of `np.digitize` latitude bins).
