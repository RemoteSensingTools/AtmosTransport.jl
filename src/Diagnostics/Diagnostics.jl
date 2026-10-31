"""
    Diagnostics

Core host-side run diagnostics: snapshot series, global mass drift and twin-run
comparisons, runtime-consistent budget closure and carrier drawdown. Loaded after
`Models`; independent of `Extras`. `SectionTimer` is loaded separately during
foundation setup. See `src/Diagnostics/README.md` for the file map.
"""
module Diagnostics

using Dates: Dates, DateTime, Millisecond
using NCDatasets: NCDataset, nomissing, dimnames
using Statistics: mean, std
using ..GriddedIO: decode_time, CFDecoder

# Budget closure (`run_context.jl`, `budget_closure.jl`) builds on the runtime's own
# config parsing, grid, output naming and surface-flux sources.
using TOML
using ..Architectures: CPU, _neumaier_add
using ..Grids: Grids, CubedSphereMesh, floattype
using ..MetDrivers: TransportBinaryDriver, TransportBinaryReader, binary_capabilities, driver_grid,
                    load_transport_window, steps_per_window_schedule
using ..Operators: AbstractFluxTemporalScheme, ConservativeMeanFlux, StepwiseFlux, SurfaceFluxSource,
                   TimeVaryingSurfaceFluxSource, emission_segments, integrated_emission
using ..Output: AbstractOutputFormat, NetCDFOutput, DailyOutputFiles, SingleOutputFile,
                output_enabled, output_path, output_path_for_day, output_format_label,
                runtime_output_spec, snapshot_hours
using ..Models: TransportTracerSpec, build_surface_flux_sources, expand_binary_paths,
                chemistry_spec, NoChemistrySpec, DecayChemistrySpec,
                AbstractInitialConditionSpec, UniformInitialConditionSpec
using ..Models: binary_date_label as _binary_date_label,
                configured_float_type as _cfg_float_type,
                check_multifile_window_count as _check_multifile_window_count,
                layout_run_hours as _layout_run_hours,
                parse_tracer_specs as _parse_tracer_specs,
                run_reference_time as _run_reference_time,
                run_time_origin as _run_time_origin,
                run_time_span as _run_time_span,
                run_window_index as _run_window_index,
                runtime_window_layout as _runtime_window_layout,
                structured_tracer_specs as _structured_tracer_specs,
                run_tracer_specs as _run_tracer_specs, uniform_background

include("snapshot_series.jl")
include("mass_totals.jl")
include("run_context.jl")
include("budget_closure.jl")

# budget_closure stores repr(emission) in its report. Preserve the existing
# serialized label when the type's implementation moves out of Analysis.
Base.show(io::IO, ::RuntimeSurfaceSources) =
    print(io, "AtmosTransport.Analysis.RuntimeSurfaceSources()")

export daily_snapshot_series, drift_statistics, twin_relative_difference
export RunContext, run_snapshot_paths
export BudgetTolerance, TracerBudget, BudgetReport, budget_closure, carrier_drawdown, report_dict
export AbstractExpectedEmission, RuntimeSurfaceSources, expected_storage_change

end # module Diagnostics
