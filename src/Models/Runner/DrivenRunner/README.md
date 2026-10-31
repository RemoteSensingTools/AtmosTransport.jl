# Configured Driven Runner

Support code for `run_driven_simulation(cfg)`: config validation, model setup,
capability checks, output and observation glue, resource ownership, and progress
reporting.

The `Models.Runner.DrivenRunner` module is also available through the legacy
`Models.DrivenRunner` alias. Its file and method loading order is unchanged.
`DrivenRunner.jl` includes `progress`, `logging`, `configuration`, `config_keys`,
`summary`, `resources`, `output`, `observations`, `model_setup`, then `run_layout`.
It retains the structured and CS transport loops.

## Entry Points

- `validate_config(cfg) -> (ok, errors)` in [`configuration.jl`](configuration.jl)
  is public. `run_driven_simulation` calls it first. It opens no binaries and
  allocates no state.
- `TransportTracerSpec` and `_parse_tracer_specs(cfg)` in
  [`configuration.jl`](configuration.jl) turn `[tracers.<name>]` into name,
  `init`, and `surface_flux` dicts. Legacy flat keys are mapped into those
  subtables. `_structured_tracer_specs(cfg)` adds the lat-lon and
  reduced-Gaussian fallback without `[tracers]` (one tracer
  `[run].tracer_name`, default `CO2`, initialised from `[init]`); the
  structured runner and `Analysis.RunContext` read tracers with it.
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
  - `_observation_terrain` reads the static ground fields of
    `[output.observations]` (`surface_height_file`, `dem_file`) through
    `GriddedIO` and passes them to `build_observation_sampler`.
  - `_boundary_humidity` hands the sampler the specific humidity of each
    sampled boundary when it places records by total pressure:
    `_WindowFieldHumidity` takes `qv_start` at the run start and `qv_end` at
    window ends (lat-lon binaries with `include_qv`);
    `_NextWindowVDIFFHumidity` (cubed sphere) reads the window-start VDIFF
    `qv` of the next window with `load_vdiff_qv_window!` (the next binary's
    window 1 after a binary's last window; the run's last window end falls
    back to the ended window's, marked `HACK`). `has_humidity` of the sampler
    requires every binary to carry humidity (`_binary_observation_humidity`
    of the header capabilities, checked before transport).
  - `_check_observation_time_origin`, `_check_observation_tracers` and
    `_check_observation_placement_files` are used by `validate_config`.
- Progress in [`progress.jl`](progress.jl): `RunProgressTimer` with its
  reporter (`ProgressBar`, `ProgressLines`, `NoProgress`), `timed_io_read!`,
  `timed_transport!`, `timed_io_write!`, `tick_window!`,
  `set_progress_status!`, `summarize_progress!`, `stop_compile_timing!` (the
  runners call it in `finally`, so compile timing is released on errors)
- Logging and checks in [`logging.jl`](logging.jl): `run_logging` (`[logging]`
  → `RunLogging`) and `run_checks` (`[checks]` → callbacks); the log level
  itself is applied by `LogLevels.with_log_level` around the whole run
- Startup log in [`summary.jl`](summary.jl): `_log_runtime_summary`

## File Map

- [`configuration.jl`](configuration.jl) — tracer specs, `[numerics] float_type`,
  architecture, table-shape and window-bound checks, `validate_config`,
  multi-file window-range guard; `validate_config` also parses the physics,
  run and output settings.
- [`config_keys.jl`](config_keys.jl) — the keys each runtime table may hold
  and `_config_key_findings`: unknown keys (with suggestions) are errors of
  `validate_config`, keys the chosen kind leaves unread are its warnings
- [`model_setup.jl`](model_setup.jl) — GPU residency assertion, flux allocation,
  LL/RG model build, per-tracer CS dry-state packing, convection-capability
  dispatch
- [`output.jl`](output.jl) — binary date labels, runtime window layout and span, duplicate
  daily-path guard, output path/schedule checks (snapshot hours on window
  ends), `RunSnapshotOutput`, single-file stream, background daily writes
- [`observations.jl`](observations.jl) — observation origin, sampler construction and
  ownership, per-binary day switch, window-end sampling with optional VDIFF
  temperature
- [`resources.jl`](resources.jl) — `_with_run_resource`, `RunInputResources`
  (drains window prefetch, closes the driver, calls `release_payload!`)
- [`progress.jl`](progress.jl) — progress reporters (bar, log lines or none)
  with io_read / transport / io_write wall-clock accumulators; the end-of-run
  summary adds GC time, JIT compilation time and allocated memory over the run
- [`logging.jl`](logging.jl) — `[logging]` (log level, progress reporter) and
  `[checks]` (the NaN checker callback)
- [`summary.jl`](summary.jl) — ANSI, advection, diffusion and schedule labels, multi-line runtime
  summary

- `DrivenRunner.jl`: library entry point, absolute run origin and transport loops.
- `DrivenSimulation.jl`: driven type, compatibility checks, storage allocation/scaling.
- `driven_window_state.jl`: payload copies, staging, prefetch and reusable buffers.
- `InputStaging.jl`: rolling local copies, identity checks, directory ownership.
- `BinaryPathExpander.jl`: explicit paths or folder/date-range binary selection.
- `run_layout.jl`: public aliases for runtime layout/clock/terrain and
  `run_tracer_specs` shared with Diagnostics and scripts.

## Common Tasks

- To add a config check that needs no binary, put it in `validate_config`. Shape
  checks in `_check_config_table_shapes!` run first and stop validation. Wrap
  value checks in `_capture_config_error!` so all errors are reported together.
- To support a new convection operator, add a `_validate_convection_capability`
  method in [`model_setup.jl`](model_setup.jl). The `AbstractConvection`
  fallback throws.
- To change how tracers are initialised, edit `_make_structured_model` for
  LL/RG or `_initialize_cs_dry_state` for CS. The builders themselves live in
  [`../../InitialConditions/`](../../InitialConditions/README.md) and
  [`../../InitialConditions/InitialConditionIO.jl`](../../InitialConditions/InitialConditionIO.jl).
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
  - LL/RG pass window-start `qv` when available for moist-basis packing.
  - The CS runner rejects moist binaries explicitly.
- Explicit `[run].Hp` and `halo_padding` must be integers ≥ 1; Boolean values
  are rejected. Input dates accept TOML dates or date strings, including
  `start_date` with explicit `binary_paths`; malformed dates fail preflight.
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

Under [`../../../../test/core/`](../../../../test/core/): `test_public_config_and_cli.jl`
(`validate_config`), `test_input_resource_lifetime.jl` (input cleanup, multi-file
window ranges), `test_async_output_lifetime.jl` (background writes, sampler close),
`test_observation_runner.jl`, `test_observation_schedules.jl` (daily-path guard),
`test_cs_initial_state_packing.jl` (`_initialize_cs_dry_state`),
`test_multiday_source_clock.jl`, `test_cs_multifile_equivalence.jl`,
`test_fv3_vertical_profile.jl` (advection labels).

Output owns the `[output]` key tables, format choices, per-tracer/column-mass
alias precedence and `check_output_targets`; this folder
reports findings and orchestrates the writers. Observation APIs are imported
from `AtmosTransport.Observations`.

## Depends on

Core, Physics, InitialConditionIO and the parent Runner; ConfigChecks, Architectures,
MetDrivers, Output, Observations, GriddedIO, State, Grids, Operators,
AtomicFiles, Parameters, LogLevels and SectionTimer.

## Used by

`scripts/run_transport.jl`, Diagnostics (and its Analysis facade), diagnostic,
inversion and flux-building scripts, runtime tests.

## Shared helpers owned here

`run_layout.jl` exports the public shared run interface, re-exported by Models.
Legacy qualified aliases still resolve to the same generic. Their original
names are documented here so the static checker resolves public aliases too:

# shared: _binary_date_label, _cfg_float_type, _check_multifile_window_count, _layout_run_hours, _parse_tracer_specs, _run_reference_time, _run_time_origin, _run_time_span, _run_window_index, _runtime_window_layout, _structured_tracer_specs, _runtime_output_origin, _window_end_hours, _observation_terrain

## Forward<->adjoint mirrors

The parent Runner stepping and capture use the same plan, refresh functions
and clock transition; resource cleanup drains prefetch before closing its input binary.
