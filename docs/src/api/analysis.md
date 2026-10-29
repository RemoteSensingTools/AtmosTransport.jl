# Analysis API

`Analysis` holds the shared numerics of post-run diagnostics: reading a time
series from daily snapshot files, area-weighted means on equal-area latitude
bands, the seasonal, trend and interannual decompositions used to compare
transported tracers with observations, skill statistics, and column profile
tools. Diagnostic command lines under `scripts/diagnostics/` take these
numerics from here, and plotting scripts read the files those command lines
write.
The names are exported from `AtmosTransport.Analysis`
(`using AtmosTransport.Analysis`), not from the top level.

The non-core implementation is in `src/Extras/Analysis/` and has no tests.
Mass-budget checks remain core in `src/Diagnostics/`: `RunContext`,
`budget_closure`, `carrier_drawdown`, their budget types and expected-emission
interface, `daily_snapshot_series`, `drift_statistics`, and
`twin_relative_difference`. These names are exported by
`AtmosTransport.Diagnostics` and re-exported by `AtmosTransport.Analysis` for
existing scripts. Core never imports Extras, and Extras use only public core
names. Each Extras folder is intended to become its own package.

Conventions shared by every function:

- arrays are read in NCDatasets order, for example `(Xdim, Ydim, nf, time)`
  for a cubed-sphere column mean, and time is the last dimension of a series;
- sums run in Float64 in a fixed (linear-index) order, so a result does not
  depend on the input's element type or on threading;
- NaN marks a missing value: band means and skill statistics leave it out,
  while the decompositions propagate it (and `seasonal_trend_decomposition`
  rejects it at observed months);
- statistics take the model first and the reference second and use population
  (1/n) moments;
- column profiles are ordered top first (layer 1 at the model top), as in
  AtmosTransport snapshots.

A typical Hovmöller comparison reads the daily column means with
`daily_snapshot_series`, forms `(nbands, ntime)` band means with one
`BandAssignment`, averages them to months, and passes the stacked model and
observation series to `seasonal_trend_decomposition` and `skill_metrics`.

The global totals a run writes at every snapshot (`<tracer>_total_mass`, a
compensated Float64 sum) feed two conservation diagnostics. `drift_statistics`
measures the change of one run's total against its initial value and tests
its per-snapshot increments for a mean (``t``); for a tracer without sources
this is pure arithmetic error. `twin_relative_difference` compares a run with
its twin in another precision frame by frame, net of the rounding of the
initial state. `weighted_difference_stats` gives the area-weighted mean and RMS
of a field difference, such as the column-mean difference of the two runs.
`scripts/diagnostics/float32_conservation_attribution.jl` and
`scripts/diagnostics/compare_float32_float64_runs.jl` are built on them; see
[Tracer Conservation in Single Precision](@ref).

`GEOSChemCATRINEArchive` reads the output of the GEOS-Chem (GCHP) CATRINE
reference run, one `GEOSChem.CATRINE_inst.YYYYMMDD_HHMMz.nc4` file per 3-hourly
output time, flat or under `YYYY/MM/`. `global_totals` turns a file into global
sums in GEOS-Chem's dry-air-equivalent storage units: the dry-air mass
``\sum \mathrm{Met\_AD}``, the burden ``\sum \chi \, \mathrm{Met\_AD}`` of each
tracer and its emission rate
``\left(\sum E \, \mathrm{Met\_AREAM2}\right) M_\mathrm{air} / M``, with
GEOS-Chem's molar masses (`Parameters.GEOSCHEM_SPECIES_MOLAR_MASS`,
`Parameters.DRY_AIR_MOLAR_MASS`). A missing, truncated or incomplete file
reads as `nothing` and is listed by `unreadable_files`. Files are read into
memory on all threads and decoded under the package's NetCDF lock.
`scripts/diagnostics/catrine_true_mass_balance.jl --model geoschem` builds
GEOS-Chem's true mass balance on it.

`budget_closure` checks the global budget of every tracer of a finished run.
The runtime conserves tracer storage ``S = \sum_c \chi_c m_c`` (dry mixing
ratio times carrier air mass, the unit of `<tracer>_total_mass`), and
transport drops out of its global sum, so between the first and last snapshot

```math
\Delta S = S(t_1) - S(t_0) = E(t_0, t_1),
```

with ``E`` the storage the tracer's surface sources added. ``E`` comes from the
run's own sources (`RuntimeSurfaceSources`: built by the runner's loader on the
run's grid and integrated with `Operators.integrated_emission`), so cell
areas, unit factors, the storage scale ``M_\mathrm{air}/M`` and slice times
are the run's and molar masses cancel. A tracer passes when
``|\Delta S - E| \le \rho |E| + k\,\varepsilon_{32}(\chi_0)\,M``, where
``\varepsilon_{32}(\chi_0)`` is the Float32 spacing of its initial carrier and
``M`` the run's initial air mass (`BudgetTolerance`: strict ``\rho = 10^{-4}``,
``k = 1``; operational ``\rho = 10^{-3}``, ``k = 4``), its values are finite
and, with a positive carrier, no column mean drops to zero. `RunContext`
holds the run as the runner builds it from its config. `carrier_drawdown`
gives, before a run, the largest cumulative loss of global mixing ratio each
tracer's sources cause, to compare with its carrier.
`scripts/diagnostics/tracer_budget_closure.jl series <run.toml>` writes the
JSON report (`report_dict`) and exits 1 when the gated level fails. Another
way to compute ``E`` plugs in as a subtype of `AbstractExpectedEmission`.

```@autodocs
Modules = [AtmosTransport.Analysis, AtmosTransport.Diagnostics]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```
