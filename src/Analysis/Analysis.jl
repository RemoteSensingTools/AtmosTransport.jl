"""
    Analysis (src)

Shared numerics for post-run diagnostics: reading daily snapshot series,
equal-area latitude-band means, seasonal and interannual decompositions,
skill statistics, column profile tools (column-mass band weights,
log-pressure interpolation), and the global budget closure of a finished run
(`budget_closure`, `carrier_drawdown`).

Diagnostic command lines in `scripts/diagnostics/` take these numerics from
here instead of carrying their own copies, and plotting scripts read the
NetCDF or JSON those diagnostics write. Every function works on plain arrays
on the host and accumulates in Float64 in a fixed order, so a result does not
depend on threading or on the element type of the input.

`src/AtmosTransport.jl` loads this module after `Models` and before
`Downloads`, so that diagnostics can build on the runtime's own types; no
runtime module depends on it. The functions here use NCDatasets, Dates,
LinearAlgebra and Statistics; the budget closure builds its view of a run
(`RunContext`) and the expected emissions with the runner's own config
parsing, grids, output naming and surface-flux sources.
See `src/Analysis/README.md` for the file map and `docs/src/api/analysis.md`
for the reference.
"""
module Analysis

using Dates: Dates, DateTime, TimeType, Millisecond, year, month, day
using LinearAlgebra: pinv
using NCDatasets: NCDataset, nomissing, dimnames
using Statistics: mean, std, cor

# Budget closure (`run_context.jl`, `budget_closure.jl`) builds on the runtime's own
# config parsing, grid, output naming and surface-flux sources.
using TOML
using ..Architectures: CPU, _neumaier_add
using ..Grids: Grids, CubedSphereMesh, floattype
using ..Parameters: DEFAULT_BACKGROUND_VMR
using ..MetDrivers: TransportBinaryDriver, TransportBinaryReader, binary_capabilities, driver_grid,
                    load_transport_window, steps_per_window_schedule
using ..Operators: AbstractFluxTemporalScheme, ConservativeMeanFlux, StepwiseFlux, SurfaceFluxSource,
                   TimeVaryingSurfaceFluxSource, emission_segments, integrated_emission
using ..Output: DailyOutputFiles, SingleOutputFile, output_enabled, output_path, output_path_for_day,
                runtime_output_spec, snapshot_hours
using ..Models: TransportTracerSpec, build_surface_flux_sources, expand_binary_paths,
                chemistry_spec, NoChemistrySpec, DecayChemistrySpec
using ..Models.DrivenRunner: _binary_date_label, _cfg_float_type, _check_multifile_window_count,
                             _layout_run_hours, _parse_tracer_specs, _run_reference_time,
                             _run_time_span, _run_window_index, _runtime_window_layout,
                             _structured_tracer_specs
using ..Models.InitialConditionIO: _init_kind

include("snapshot_series.jl")
include("latitude_bands.jl")
include("decomposition.jl")
include("skill.jl")
include("column_profiles.jl")
include("run_context.jl")
include("budget_closure.jl")

export daily_snapshot_series
export AbstractLatitudeBands, EqualAreaBands, BandAssignment
export nbands, band_edges, band_index, band_means, band_means!
export doy_linear_detrend, seasonal_trend_decomposition
export skill_metrics, weighted_fit_stats
export column_band_weights, column_band_weights!, logp_interpolate, logp_interpolate!

export RunContext, run_snapshot_paths
export BudgetTolerance, TracerBudget, BudgetReport, budget_closure, carrier_drawdown, report_dict
export AbstractExpectedEmission, RuntimeSurfaceSources, expected_storage_change

end # module Analysis
