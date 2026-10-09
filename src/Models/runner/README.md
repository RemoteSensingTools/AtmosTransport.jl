# Runner

Support code for `run_driven_simulation(cfg)`: config validation, model setup,
capability checks, output and observation glue, resource ownership, and progress
reporting.

These files are not a module. [`../DrivenRunner.jl`](../DrivenRunner.jl) includes
them in this order: `progress`, `configuration`, `summary`, `resources`, `output`,
`observations`, `model_setup`. `DrivenRunner.jl` keeps the public entry point and
the two transport loops: `_run_driven_simulation_structured` for lat-lon and
reduced Gaussian, and `_run_driven_simulation_cs`. The parent overview, including
how a TOML option reaches an operator, is [`../README.md`](../README.md).

## Entry Points

- `validate_config(cfg) -> (ok, errors)` in [`configuration.jl`](configuration.jl)
  is public. `run_driven_simulation` calls it first. It opens no binaries and
  allocates no state.
- `TransportTracerSpec` and `_parse_tracer_specs(cfg)` in
  [`configuration.jl`](configuration.jl) turn `[tracers.<name>]` into name,
  `init`, and `surface_flux` dicts. Legacy flat keys are mapped into those
  subtables.
- Model construction in [`model_setup.jl`](model_setup.jl):
  - `_make_structured_model` builds the lat-lon and reduced-Gaussian model.
  - `_initialize_cs_dry_state` builds the cubed-sphere state.
  - `_assert_gpu_residency!` checks where the state lives.
  - `_validate_capability_match(driver, recipe)` checks the physics against the
    binary.
- Output resources in [`output.jl`](output.jl) and [`resources.jl`](resources.jl):
  - `RunSnapshotOutput` holds the snapshot stream, the pending daily write, and
    the observation sampler.
  - `RunInputResources` holds the current driver and `DrivenSimulation`.
  - `_with_run_resource(f, resource)` closes the resource on every exit path.
- Observation glue in [`observations.jl`](observations.jl):
  - `_install_observation_sampler!`, `_begin_observation_binary!`, and
    `_observe_window_end!` drive the sampler.
  - `_check_observation_time_origin` and `_check_observation_tracers` are used
    by `validate_config`.
- Progress in [`progress.jl`](progress.jl): `RunProgressTimer`,
  `timed_io_read!`, `timed_transport!`, `timed_io_write!`, `tick_window!`,
  `set_progress_status!`, `summarize_progress!`
- Startup log in [`summary.jl`](summary.jl): `_log_runtime_summary`

## File Map

- [`configuration.jl`](configuration.jl) — tracer specs, `[numerics] float_type`,
  architecture, table-shape and window-bound checks, `validate_config`,
  multi-file window-range guard. It also holds the ANSI and advection-label
  helpers used by the summary.
- [`model_setup.jl`](model_setup.jl) — GPU residency assertion, flux allocation,
  LL/RG model build, per-tracer CS dry-state packing, convection-capability
  dispatch
- [`output.jl`](output.jl) — binary date labels, default output span, duplicate
  daily-path guard, `RunSnapshotOutput`, single-file stream, background daily
  writes
- [`observations.jl`](observations.jl) — run origin, sampler construction and
  ownership, per-binary day switch, window-end sampling with optional VDIFF
  temperature
- [`resources.jl`](resources.jl) — `_with_run_resource`, `RunInputResources`
  (drains window prefetch, closes the driver, calls `release_payload!`)
- [`progress.jl`](progress.jl) — progress bar with io_read / transport /
  io_write wall-clock accumulators
- [`summary.jl`](summary.jl) — diffusion and schedule labels, multi-line runtime
  summary

## Common Tasks

- To add a config check that needs no binary, put it in `validate_config`. Shape
  checks in `_check_config_table_shapes!` run first and stop validation. Wrap
  value checks in `_capture_config_error!` so all errors are reported together.
- To support a new convection operator, add a `_validate_convection_capability`
  method in [`model_setup.jl`](model_setup.jl). The `AbstractConvection`
  fallback throws.
- To change how tracers are initialised, edit `_make_structured_model` for
  LL/RG or `_initialize_cs_dry_state` for CS. The builders themselves live in
  [`../initial_conditions/`](../initial_conditions/README.md) and
  [`../InitialConditionIO.jl`](../InitialConditionIO.jl).
- To add an output resource, give it a `close` method and own it through
  `RunSnapshotOutput` or `_with_run_resource`, so failure paths close it too.
- To debug "daily output would be written for both ...", look at
  `_check_unique_day_paths`. Date labels are the first 8-digit token of each
  binary's basename (`_binary_date_label`).

## Invariants

- Every resource the runner opens is closed on success and on error.
  `_with_run_resource` rethrows the run error. If cleanup also fails, it throws
  both as a `CompositeException`. `close(::RunSnapshotOutput)` closes the
  observation sampler last.
- At most one background daily snapshot write is in flight
  (`_start_daily_output!` waits for the previous one). It writes host copies of
  frames and is not charged to the timer.
- `RunInputResources` finishes window prefetch before closing the driver, then
  calls `release_payload!` on its mmap.
- The binary's `mass_basis` sets the state basis. Initial tracer storage is dry
  VMR × air mass:
  - LL/RG pass no `qv`, so `pack_initial_tracer_mass` rejects moist binaries.
  - The CS runner rejects moist binaries explicitly.
- Multi-file runs must cover full windows of every file
  (`_check_multifile_window_range`). CS runs also require `start_window = 1`.
- Observation sampling has one clock:
  - Its origin is `[input].start_date` at 00:00 UTC, or
    `[output.observations].start_time` when there is no start date. The two
    must agree if both are set.
  - When the first binary's name carries a date label, the origin must fall on
    that date.
  - Sampling runs at t = 0 and at every met-window end. In the LL/RG loop it
    forces per-window stepping even without snapshots (`samples_observations`).
- Each binary's `DrivenSimulation` gets `start_time` equal to the accumulated run
  time in seconds. Time-varying surface sources therefore index slices from the
  run start, not the start of each file.

## Related Tests

Under [`../../../test/core/`](../../../test/core/): `test_public_config_and_cli.jl`
(`validate_config`), `test_input_resource_lifetime.jl` (input cleanup, multi-file
window ranges), `test_async_output_lifetime.jl` (background writes, sampler close),
`test_observation_runner.jl`, `test_observation_schedules.jl` (daily-path guard),
`test_cs_initial_state_packing.jl` (`_initialize_cs_dry_state`),
`test_multiday_source_clock.jl`, `test_cs_multifile_equivalence.jl`,
`test_fv3_vertical_profile.jl` (advection labels).
