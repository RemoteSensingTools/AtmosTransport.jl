"""
    Analysis

Non-core post-processing: latitude bands, decompositions, skill statistics,
column profiles and GEOS-Chem CATRINE archive reading. Conservation diagnostics
are imported and re-exported from core `Diagnostics`, preserving the API used
by scripts. Core modules never depend on this module; its core dependencies
are public names only. See `src/Extras/Analysis/README.md` and
`docs/src/api/analysis.md`.
"""
module Analysis

using Dates: Dates, DateTime, TimeType, Millisecond, Hour, @dateformat_str, year, month, day
using LinearAlgebra: pinv
using NCDatasets: NCDataset, NetCDFError, nomissing, dimnames
using Statistics: mean, std, cor
using ..Architectures: _neumaier_sum
using ..Parameters: DRY_AIR_MOLAR_MASS, GEOSCHEM_SPECIES_MOLAR_MASS, RN222_HALF_LIFE
using ..Output: with_netcdf_lock

# Conservation diagnostics remain core; preserve the Analysis API for scripts.
using ..Diagnostics: daily_snapshot_series, drift_statistics, twin_relative_difference,
                     RunContext, run_snapshot_paths, BudgetTolerance, TracerBudget,
                     BudgetReport, budget_closure, carrier_drawdown, report_dict,
                     AbstractExpectedEmission, RuntimeSurfaceSources, expected_storage_change

include("latitude_bands.jl")
include("decomposition.jl")
include("skill.jl")
include("column_profiles.jl")
include("geoschem_archive.jl")

export daily_snapshot_series
export AbstractLatitudeBands, EqualAreaBands, BandAssignment
export nbands, band_edges, band_index, band_means, band_means!
export doy_linear_detrend, seasonal_trend_decomposition
export skill_metrics, weighted_fit_stats, weighted_difference_stats
export column_band_weights, column_band_weights!, logp_interpolate, logp_interpolate!
export drift_statistics, twin_relative_difference
export GEOSChemTracer, CATRINE_GEOSCHEM_TRACERS, GEOSChemCATRINEArchive
export gc_path, gc_times, with_gc_file, global_totals, unreadable_files

export RunContext, run_snapshot_paths
export BudgetTolerance, TracerBudget, BudgetReport, budget_closure, carrier_drawdown, report_dict
export AbstractExpectedEmission, RuntimeSurfaceSources, expected_storage_change

end # module Analysis
