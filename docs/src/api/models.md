# Models API


The `Models` module contains the runtime stepper —
`TransportModel`, `DrivenSimulation`, the IC pipeline, and the
recipe builders that turn a TOML config into a fully-wired runtime
object.

Use `step!(model, dt)` for a full model step; driven runs execute their
`StepPlan` phases. `build_initial_mixing_ratio` returns dry VMR and accepts
`surface_pressure`, `pressure_basis`, and `qv` for pressure remapping. Set
`mass_basis` when converting that VMR with `pack_initial_tracer_mass`.

The source is grouped under `Models/Core`, `Physics`, `InitialConditions` and
`Runner`; these are plain folders where the driven type, stepping and refresh
methods share the existing Models module. Nested module paths remain stable.

Diagnostics and scripts can use the public Models run interface:

| API | Contract |
|---|---|
| `run_tracer_specs(cfg, mesh)` | Runtime tracer specs, including the structured legacy form. |
| `run_reference_time(cfg)`, `run_time_origin(cfg, driver, path)` | Config reference time and agreement with the first binary's dates. |
| `runtime_output_origin(origin, driver; start_window=1)` | Absolute output origin after skipped windows. |
| `runtime_window_layout(layout, FT)`, `layout_run_hours(layout; ...)`, `window_end_hours(layout; ...)` | Runtime-rounded window lengths, run duration and output boundary hours, retaining each operation's existing rounding order. |
| `run_time_span(run_hours)`, `run_window_index(run_cfg, key, default)`, `check_multifile_window_count(...)` | Coverage span and window-bound validation. |
| `configured_float_type(cfg)`, `binary_date_label(path)` | Runtime precision and existing output filename label policy. |
| `uniform_background(init_cfg, FT=Float64)` | Runtime uniform initial VMR default. |
| `observation_terrain(placement, mesh)` | Terrain inputs for observation placement. |

Window-clock arithmetic remains in `Operators.StepPlans.window_clock` and
`window_step_dt`. Diagnostic accumulated clocks retain their current arithmetic.
`Models.InitialConditionIO.load_file_surface_flux_field` and
`conservative_surface_flux_rate` expose the inventory operations used by the
flux-building script. Native file order, Pa-unit and inventory-year policies
are owned by `GriddedIO/initial_condition_conventions.jl`.

```@autodocs
Modules = [
    AtmosTransport.Models,
    AtmosTransport.Models.InitialConditionIO,
    AtmosTransport.Models.BinaryPathExpander,
    AtmosTransport.Models.InputStaging,
    AtmosTransport.Models.DrivenRunner,
]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```
