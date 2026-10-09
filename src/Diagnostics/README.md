# Diagnostics

Host-side runtime instrumentation.

The folder holds one file, [`SectionTimer.jl`](SectionTimer.jl), which defines
the module `SectionTimer`. There is no `Diagnostics` module. Timing is off by
default. `src/AtmosTransport.jl` loads it right after `Architectures`, before
every other module, so any later module can `using ..SectionTimer`. It
depends only on `Printf`.

## Entry Points

All in [`SectionTimer.jl`](SectionTimer.jl):

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

- [`SectionTimer.jl`](SectionTimer.jl): the whole module

## How A Run Uses It

- `_run_driven_simulation` in
  [`../Models/DrivenRunner.jl`](../Models/DrivenRunner.jl) does four things:
  1. Calls `maybe_enable_from_env!()` before opening inputs.
  2. Calls `disable!()` in a `finally` block.
  3. Calls `report(stderr)`.
  4. If the run config has an output path, calls `write_csv` with that path,
     `.nc` removed, plus `.timings.csv`.
- Timed sections are placed in `../Models/TransportModel.jl`,
  `../Models/DrivenSimulation.jl`, and in `../Operators/Advection/`:
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
  the runner disables instrumentation after a failed run
- API page: [`../../docs/src/api/infrastructure.md`](../../docs/src/api/infrastructure.md)
