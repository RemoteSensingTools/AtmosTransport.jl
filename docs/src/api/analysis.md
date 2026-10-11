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
Modules = [AtmosTransport.Analysis]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```
