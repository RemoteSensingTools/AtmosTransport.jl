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

```@autodocs
Modules = [AtmosTransport.Analysis]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```
