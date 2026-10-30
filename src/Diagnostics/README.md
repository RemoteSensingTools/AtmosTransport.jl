# Diagnostics

Core host-side run diagnostics and runtime instrumentation. Core never imports
from `Extras`. Analysis re-exports the conservation API for existing scripts.

`Diagnostics.jl` defines `AtmosTransport.Diagnostics`, loaded after `Models`
and before Extras. It depends on `Architectures`, `Parameters`, `Grids`,
`MetDrivers`, `Operators`, `Output`, `GriddedIO`, `Models`, and the standard
libraries Dates, Statistics and TOML plus NCDatasets. `RunContext` uses the
runner's existing layout helpers without changing their logic.

[`../SectionTimer.jl`](../SectionTimer.jl) is the separate foundation module `SectionTimer`, loaded
right after `Architectures`, before every other module. Timing is off by default;
it depends only on `Printf`.

## Conservation API

- `RunContext`, `run_snapshot_paths`: reconstruct a run using the runtime's
  config parsing, binary layouts, grid and output schedule.
- `budget_closure`, `BudgetTolerance`, `TracerBudget`, `BudgetReport`,
  `report_dict`: compare snapshot storage changes with runtime surface sources.
- `carrier_drawdown`: check that a planned carrier covers cumulative uptake.
- `AbstractExpectedEmission`, `RuntimeSurfaceSources`, `expected_storage_change`:
  the expected-emission interface.
- `daily_snapshot_series`: read and concatenate snapshots with their CF origin,
  repeated boundary times and fill-value handling.
- `drift_statistics`, `twin_relative_difference`: conservation and precision
  comparisons of global mass totals.

Used by diagnostic scripts through `AtmosTransport.Analysis` re-exports or
`AtmosTransport.Diagnostics`. The tests `test_analysis_budget_closure.jl`,
`test_analysis_snapshot_series.jl`, `test_analysis_mass_totals.jl` and
`test_diagnostics_mass_balance.jl` retain the core conservation checks.
There are no forward/adjoint mirrors here.

## Entry Points

All in [`../SectionTimer.jl`](../SectionTimer.jl):

- `@section name expr` times `expr` and records the sample under the `Symbol`
  `name`
- `time_section(f, name)` is the function form, for do-blocks. It returns
  `f()` with a concrete type.
- `record_sample!(section, ns, bytes=0)` adds an externally measured sample
- `enable!(; timing=true, allocations=false, nvtx=false)`, `disable!()`, and
  `is_enabled()` control recording
- `maybe_enable_from_env!()` reads three environment variables:
  `ATMOSTR_TIMERS`, `ATMOSTR_ALLOC_TIMERS`, and `ATMOSTR_NVTX`. Each accepts
  `1`, `true`, `on`, or `yes`. It enables recording only when timing or NVTX
  is requested, so `ATMOSTR_ALLOC_TIMERS` alone does nothing.
- `report(io=stderr)` prints a table of n_calls, total, mean, p50, p95, max,
  and fraction, plus allocation columns when collected. `write_csv(path)`
  writes the same summary.

## File Map

- [`Diagnostics.jl`](Diagnostics.jl): conservation module, imports and exports
- [`run_context.jl`](run_context.jl): runtime layout and snapshot paths
- [`budget_closure.jl`](budget_closure.jl): budget checks and carrier drawdown
- [`snapshot_series.jl`](snapshot_series.jl): snapshot series and NetCDF helpers
- [`mass_totals.jl`](mass_totals.jl): global-total drift and twin-run comparisons

## How A Run Uses It

- `_run_driven_simulation` in
  [`../Models/Runner/DrivenRunner.jl`](../Models/Runner/DrivenRunner.jl) does four things:
  1. Calls `maybe_enable_from_env!()` before opening inputs.
  2. Calls `disable!()` in a `finally` block.
  3. Calls `report(stderr)`.
  4. If the run config has an output path, calls `write_csv` with that path,
     `.nc` removed, plus `.timings.csv`.
- Timed sections are placed in `../Models/TransportModel.jl`,
  the `../Models/driven_*.jl` files, and in `../Operators/Advection/`:
  `StrangSplitting.jl`, `strang_apply.jl`, `CubedSphereStrang.jl` and
  `cs_sweep_common.jl` (the profiled kernel launches).
- NVTX ranges come from
  [`../../ext/AtmosTransportNVTXExt.jl`](../../ext/AtmosTransportNVTXExt.jl).
  It adds `_nvtx_start(::AbstractString)` and `_nvtx_end(::NVTX.RangeId)`
  methods when `NVTX` is loaded. Otherwise the `::Any` fallbacks are no-ops.

## Conventions

- Times are wall clock at host call boundaries. GPU work is charged to a
  section only if the code inside synchronizes the backend before the section
  returns.
- Sections nest. For example, `:diffusion` in `StrangSplitting.jl` runs inside
  `:advection` in `TransportModel.jl`. The `frac%` column is relative to the
  sum of section totals, and the header line reports covered time against wall
  time separately.
- Every `enable!` and `disable!` advances an epoch. A sample whose section
  began in an earlier epoch is dropped.
- Shared state sits behind one `ReentrantLock`, so recording from several
  threads is safe.
- When everything is off, `@section` costs atomic flag loads and then runs
  `expr`.

## Tests And Docs

- [`../../test/core/test_section_timer.jl`](../../test/core/test_section_timer.jl):
  concurrent recording, report and CSV output, epoch rejection
- [`../../test/core/test_input_resource_lifetime.jl`](../../test/core/test_input_resource_lifetime.jl):
  runner input cleanup, prefetch drain and partial-file forcing guards
- API page: [`../../docs/src/api/infrastructure.md`](../../docs/src/api/infrastructure.md)

## Depends on

The public Models runtime-layout/tracer interface, MetDrivers, State, Grids,
GriddedIO, Operators.SurfaceFlux, Output, Parameters and Architectures.

## Used by

The package facade, Analysis re-exports, diagnostic scripts and budget tests.

## Shared helpers owned here

Exported `RunContext`, snapshot-series, budget-closure, drift and reporting APIs.
RunContext reads the runner's public aliases; diagnostic accumulated clocks
retain their original arithmetic rather than switching to StepPlans.window_clock.

## Forward<->adjoint mirrors

None. These host diagnostics compare forward transport budgets and snapshots.
