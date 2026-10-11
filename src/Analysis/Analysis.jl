"""
    Analysis (src)

Shared numerics for post-run diagnostics: reading daily snapshot series,
equal-area latitude-band means, seasonal and interannual decompositions,
skill statistics, and column profile tools (column-mass band weights,
log-pressure interpolation).

Diagnostic command lines in `scripts/diagnostics/` take these numerics from
here instead of carrying their own copies, and plotting scripts read the
NetCDF or JSON those diagnostics write. Every function works on plain arrays
on the host and accumulates in Float64 in a fixed order, so a result does not
depend on threading or on the element type of the input.

`src/AtmosTransport.jl` loads this module after `Models` and before
`Downloads`, so that diagnostics can build on the runtime's own types; no
runtime module depends on it. The functions here use only NCDatasets, Dates,
LinearAlgebra and Statistics. See `src/Analysis/README.md` for the file map
and `docs/src/api/analysis.md` for the reference.
"""
module Analysis

using Dates: Dates, DateTime, TimeType, Millisecond, year, month, day
using LinearAlgebra: pinv
using NCDatasets: NCDataset, nomissing, dimnames
using Statistics: mean, std, cor

include("snapshot_series.jl")
include("latitude_bands.jl")
include("decomposition.jl")
include("skill.jl")
include("column_profiles.jl")

export daily_snapshot_series
export AbstractLatitudeBands, EqualAreaBands, BandAssignment
export nbands, band_edges, band_index, band_means, band_means!
export doy_linear_detrend, seasonal_trend_decomposition
export skill_metrics, weighted_fit_stats
export column_band_weights, column_band_weights!, logp_interpolate, logp_interpolate!

end # module Analysis
