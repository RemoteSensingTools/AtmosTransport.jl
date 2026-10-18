# Release notes

## Unreleased

### Breaking changes

- **Cubed-sphere adjoint replays the production step plan (owner decision
  D31; findings OF-114, OF-115).** A `StepPlan` (`Operators.StepPlans`) now
  describes the model step: the stages of window advance, window start, every
  step and window end, with the physics cadence, the binary window contract
  and the air-mass reset mode resolved once. `TransportModel.step!` and
  `DrivenSimulation` execute it (production results are bit-identical), and
  the cubed-sphere adjoint replays the same plan with the production
  operators on a single-tracer state, recording its tape through recorder
  hooks of the production step (`AbstractStepRecorder`; production passes the
  no-op `NoStepRecorder()`). The separate replay composition
  (`_replay_cs_step!`, the mass/tracer/Lin-Rood recorders, the per-family
  Stride drivers) is removed. Consequences for the adjoint API:
  - New `schedule = CSStepSchedule(...)` keyword on every entry point
    (`cs_surface_emission_footprint`, `..._from_seed`, `run_cs_footprint_forward`,
    `cs_surface_flux_jacobian`, `cs_surface_flux_4dvar`, `cs_tape_byte_estimate`):
    the plan, steps per met window, window length, start time, binary subcycle
    contract and air-mass reset targets; `CSStepSchedule(sim)` gives a
    `DrivenSimulation`'s, and `cs_adjoint_inputs(sim; tracer)` captures every
    adjoint input of a cubed-sphere driven configuration from production (per-
    step face fluxes after window loading and flux scaling, per-window
    convection forcing and diffusion geometry, emission rates, fresh
    workspaces). Without it the schedule is one step of `dt` per met
    window with the full operator suite every step (the former behaviour).
    Under `physics_cadence = :window` with a binary window contract,
    convection is recorded once per window over the window length (OF-114;
    the adjoint used to apply it every step).
  - `dt` defaults to `nothing` (one when no schedule is given; with a
    schedule its step dt). `cfl_limit` other than `DEFAULT_CFL_LIMIT` is an
    error (production uses the default).
  - `cs_surface_emission_footprint_from_seed(...; surface_flux = false)` and
    `run_cs_footprint_forward(...; surface_flux = false)` replay a
    configuration without surface sources: the palindrome center is one
    `V(dt)` solve, as in production (OF-115; the replay always built
    `V(dt/2) S V(dt/2)` with `S = 0`). `CSFootprintResult` gains
    `initial_adjoint`, `dJ/drm` at the initial state (positional
    construction changes); without a surface-flux operator `footprints` is
    empty.
  - Surface emissions go through the production surface-flux kernel (Kahan
    compensated) and convection through the production `apply!` with its
    ledger carry and the CMFMC CFL cache; checkpoints save this persistent
    state. Footprints and 4D-Var values change at rounding level; the
    forward replay, the tape recording and every checkpoint restart equal
    the production run bit for bit.
  - `require_adjoint_support(plan, ops...)`, `require_adjoint_support(sim)`
    and the trait `adjoint_support(x)`: every entry point rejects, before
    recording, a plan with a stage or operator without a record and an
    adjoint (for example exponential-decay chemistry, the collaborative TM5
    solve, clamped CMFMC), naming it.
  - Tape records: `_CSMidpointRecord` carries its emission interval,
    `_CSConvectionRecord` the forward call's CFL sub-step count,
    `_CSDiffusionRecord` the forward's clock (time-varying Kz is transposed
    at the time production evaluated it), and a new
    `_CSAirMassResetRecord` transposes the reset. Lin-Rood horizontal records
    are built from the production `fv_tp_2d_cs!`'s hooks;
    `cs_tape_byte_estimate` counts their face fluxes too.
  - `DrivenSimulation` gains a `plan` field and drops
    `physics_every_substep`.
  - `test/core/test_cs_adjoint_step_plan_parity.jl` is the parity matrix
    (replay, recording and checkpoint restarts == production bit for bit
    across schemes, cadences, convection, diffusion couplings, no-emission
    diffusion and reset modes; gradients vs. production finite differences).

- **Observation and departure files: schema v2 (issue #20, part 3; findings
  OF-84, OF-15).** `read_observations` and `read_departures` reject schema-v1
  files with an explanation (internal development: no reader for v1).
  Observation files store Float64 `lat`/`lon` (a Float32 point within
  rounding of a cell edge could bind to the neighbouring cell) and replace
  `alt` by `elevation_m`, `intake_height_m`, `altitude_m` and `pressure_pa`
  (NaN = absent), with new `kind` (`"point"` / `"column"`), `time_window_s`,
  `flag` and `model_sigma` (`schemas/cs_observations_v2.toml`). Departure
  files store Float64 coordinates, the record kind and window, the vertical
  placement (code, height above the model surface, pressure), `model_sigma`,
  and every sampled step with its weight and level as a ragged `sample`
  dimension, instead of `alt` and `step`, plus a `clamped` flag per row and
  an `n_clamped` attribute; `normalized_departure` divides by
  `√(value_sigma² + model_sigma²)` (`schemas/cs_departures_v2.toml`).
  `CSObservationRecord` and `CSDepartureRecord` change accordingly.
  `build_departure_set(set, result::CS4DVarResult, mesh, t_start, dt, nsteps)`
  (or with `observations, simulated, stencils`) joins observations to records
  by id, so filtered or skipped records need no pre-filtering of the set; the
  old `(set, observations, simulated, ...)` signature is gone.
- **4D-Var observations (issue #20, part 3).** `bind_to_mesh` binds point
  records to `CSPlacedPointObjective`s placed in the column (it used to drop
  the altitude and bind every record to a column mean) and returns a
  `Vector{CSObservation}` of mixed objective types; records with a nonzero
  `flag` are dropped. `CSObservation` gains `sampling::CSTimeSampling` and
  `id` (positional construction of the struct changes; `CSObservation(step,
  objective, value, sigma)` still builds an instantaneous observation).
  `CS4DVarResult` gains a `stencils` field (positional construction changes).
  `CSLinearFunctional` stores one step per entry (`steps`, instead of the
  scalar `step`).
- **OCO-2 follow-up of issue #20 (D30): struct fields.** `CSObservation`
  gains `assimilate` (positional construction of the struct changes; the
  `CSObservation(step or sampling, objective, value, sigma; id, assimilate =
  true)` constructors are unchanged for existing calls). `CSDepartureRecord`
  gains `flag` and `cell_average_id` (positional construction changes; the
  keyword constructor defaults them to 0). `Output.RetrievalColumn` is
  parametric in its mapping (`RetrievalColumn{LayerRebin}` by default, a new
  last field `mapping`); `Output.SoundingRequest` gains `retrieval`
  (`nothing` from the existing constructors) and `Output.SoundingBatch` an
  optional `retrieval_values`. In departure files `assimilated` is no longer
  part of the retrieval block: it is written whenever a row is not
  assimilated (co-samples) or is an averaging-kernel column.
- **Adjoint API (issue #20).** A user-defined `AbstractCSFootprintObjective`
  needs a `_linear_functional(objective, step, panels_m, mesh)` method (its
  weights as a `CSLinearFunctional`) to be used in `cs_surface_flux_4dvar`
  and as a footprint seed; validation, evaluation and seeding methods alone no
  longer suffice. The exported `CSTapeByteEstimate` gains a `step_records`
  field before `total_records` (positional construction changes), and
  `total_records` now includes one end-of-step record per step.
- **Lat-lon ERA5 physics-file TM5 route removed** (issue #55, pera-noaa). The
  `[tm5_convection]` table of ERA5 spectral preprocessing configs converted the
  ERA5 convection NetCDF to flat physics files and then to TM5 fields on the
  lat-lon grid. The files kept the CDS longitudes 0…360° while the route read
  them as −180…180°, so every TM5 field sat 180° from its column, and each
  hourly window used the mean of the previous hour. The route served only nine
  December-2021 test configs and was retired instead of fixed. A config with a
  `[tm5_convection]` table (even with `enable = false`) now fails with an error naming the supported
  routes: the ERA5 N320 → cubed-sphere preprocessor with
  `include_convection = true`, or `scripts/preprocessing/attach_catrine_tm5_convection_cs.jl`.
  Removed: the nine configs (`config/preprocessing/catrine5d/*_tm5.toml`,
  `config/preprocessing/era5_ll720x361_*tm5*.toml`),
  `scripts/preprocessing/convert_era5_physics_nc_to_bin.jl`, and from
  `Preprocessing` `convert_era5_physics_nc_to_bin`, `ERA5PhysicsBinaryReader`,
  `ERA5PhysicsBinaryHeader`, `open_era5_physics_binary`,
  `close_era5_physics_binary`, `get_era5_physics_field`,
  `tm5_native_fields_for_hour!`, `merge_tm5_field_3d!`,
  `TM5PreprocessingWorkspace`, `allocate_tm5_workspace`,
  `compute_tm5_merged_hour_on_source!`, `log_tm5_cleanup_stats`,
  `tm5_copy_or_regrid_ll!` and `resolve_tm5_convection_settings`
  (`reject_removed_tm5_convection_route` takes its place), the
  `test/real_data/test_tm5_catrine_1day.jl` check of that route, and the
  convert hint of `scripts/downloads/download_era5_physics.py` (its
  convection download stays). Lat-lon spectral binaries keep their header
  schema (`include_tm5conv = false`). No lat-lon route now writes TM5
  sections, so the lat-lon `kind = "tm5"` run configs
  (`config/runs/catrine5d/ll*_advdiffconv_*`,
  `config/runs/catrine_ll720_3d/{advdiffconv,fullphysics}.toml`) and run
  configs that read binaries of the removed route or their C48/C180 regrids
  fail on load until their inputs are rebuilt; they are kept for provenance.
- **TM5 convection provenance checked on load** (issue #54). ERA5 N320
  binaries with convection now record `tm5_convection_source` and
  `tm5_ec2tm_convention = "tm5_halflevel_v2"`, and the lat-lon → cubed-sphere
  regrid copies its source's record (`"unrecorded"` when it has none). Opening
  a binary rejects, with a regenerate message: ERA5-derived TM5 sections
  without the current convention (every ERA5 N320 binary built with
  `include_convection = true` before this fix, e.g. the December-2021 and
  2022–2023 C180 CATRINE sets), lat-lon binaries with TM5 sections from the
  ERA5 spectral preprocessor, and lat-lon → cubed-sphere regrids of those.
  Sections attached from the TM5-converted CATRINE archive
  (`convection_attachment`, all v3/v4 attached sets and the golden inputs,
  whatever other key they carry) and sections without a recorded source stay
  accepted.
  `binary_capabilities` gains `tm5_convection_provenance`, and
  `inspect_binary` prints it. New constants `TM5_CONVECTION_SOURCE_KEY`,
  `TM5_EC2TM_CONVENTION_KEY`, `TM5_EC2TM_CONVENTION`.

### Post-run diagnostics

- `scripts/diagnostics/catrine_benchmark_v05_site.py` builds the v0.5 CATRINE
  benchmark web site (GCHP against the v0.5 MERRA-2 and ERA5 runs) from the
  signoff comparison products: an overview (signoff state, key numbers,
  agreement table per tracer and air-mass band) and pages for time series,
  latitude–pressure sections, maps, the near-surface band, the stratosphere,
  the mass budget, NOAA stations and the methods. Charts use self-hosted D3
  (hover or tap for values; page scrolling is never captured on phones;
  drag-to-zoom with a reset on daily time axes); one control row per page is
  kept in the URL and data are loaded per view. Only finished comparison
  seasons are read; `--datasets auto` (default) shows MERRA-2 and adds ERA5
  once all 25 months are compared. The stations page shows GCHP and the runs
  at eight NOAA sites (nearest C90 cell; lowest layer, or the site's height
  through GCHP's height–dry-pressure profile) with NOAA CO₂ observations from
  a local ObsPack (CC0 datasets only, cited) and scores each against them
  (bias, RMSE, correlation). `catrine_benchmark_site/publish.sh` builds into a
  staging folder and copies it to the public one (README there).

### Testing

- `test/core/test_cs_4dvar_placed_observations.jl` (inversion topic, new)
  checks point observations placed in the column and time windows: parity
  with the forward sampler (`build_observation_sampler` in nearest-window
  mode and `bind_to_mesh` + `cs_surface_flux_4dvar` give the same layer and
  value for intake-height, altitude, DEM and pressure records); a 10 m intake
  and a 5 km aircraft sample in one cell read layers 10 and 5 with different
  values; total against dry pressure placement; layers that follow the
  replayed air mass step by step (checked against an independent replay); a
  window observation's value and gradient against the Δt-weighted sum of its
  single-step observations (`1e-12`); the single sweep against
  per-observation footprints for placed and window observations (`1e-12`;
  upwind, monotone PPM, Lin-Rood, TM5 convection, Full and Stride
  checkpoints); and central finite differences of the cost.
  `test_cs_observation_binding.jl`, `test_cs_observations_io.jl` and
  `test_cs_departures_io.jl` move to schema v2 and add Float64 edge points,
  window weights (including truncation at the run's ends), placement by
  intake height, altitude minus elevation or DEM and pressure, the unplaced
  policy, flags and model sigma, and v1 rejection; `test_cs_inversion_driver.jl`
  runs an observation file on `synthetic_hydrostatic` meteorology.
- `test/core/test_cs_4dvar_single_sweep.jl` (inversion topic) checks the
  single-sweep 4D-Var gradient against one `cs_surface_emission_footprint` per
  observation rebuilt in the test (upwind, unlimited slopes and PPM, monotone
  PPM, Lin-Rood; no physics, diffusion with both surface-flux couplings, TM5
  and CMFMC; observations at the first and last step, two in one column and
  step, two in one cell and step, at different steps), `StrideCheckpoint(2)`
  and `RevolveCheckpoint()` against `FullCheckpoint`, the `tape_path` layout
  (exactly one schedule's windows), one recording and one reverse pass per
  gradient (counted through a tape storage) for one and seven observations,
  the reused-tape Jacobian against per-objective footprints, and the step
  forcing's injection. The tape-layout tests of
  `test_cs_adjoint_production_parity.jl`, `test_subcycle_composition.jl` and
  `test_cs_tape_mmap_roundtrip.jl` now expect the end-of-step record that
  closes every step.
- `test/core/test_cs_adjoint_production_parity.jl` (adjoint_footprint topic)
  checks the cubed-sphere adjoint against production: replay vs.
  `TransportModel.step!` for both surface-flux couplings with TM5, CMFMC and
  no convection; footprints vs. production finite differences; Stride and
  Revolve vs. `FullCheckpoint` with convection for both limiters; the tape
  layout vs. `cs_tape_byte_estimate`; and limited-PPM `:pinned_host` and
  `:mmap` tapes vs. `:device` at every entry point. The Revolve test of
  `test_cs_stride_checkpoint.jl` with monotone PPM now includes CMFMC
  convection and asserts the `1e-12` parity of the other schedules instead of
  `atol = 1e-5, rtol = 1e-3`; the opt-in GPU block of
  `test_cs_ppm_adjoint_footprint.jl` compares limited-PPM `:pinned_host` and
  `:mmap` footprints with `:device`.
- **JET 0.12 allowed in the test environment.** On 2026-10-15 the General registry
  applied JET's rolling Julia-compatibility policy (General PR #171848), capping JET
  ≤ 0.11 to Julia 1.12; on Julia ≥ 1.13 only JET 0.12 resolves. With `[compat]`
  allowing only JET 0.9–0.11 the test environment could not be resolved on the
  current Julia release (`Unsatisfiable requirements detected for package JET`;
  every PR's "Julia 1" CI job failed at setup). `Project.toml` and
  `test/Project.toml` now allow `JET = "0.9, 0.10, 0.11, 0.12"`;
  `test/core/test_jet.jl` passes with JET 0.12.3 on Julia 1.13.1 and 1.13.2 (115
  reports, baseline 115) and skips with a message on a Julia version JET does not
  support yet instead of erroring.

### Numerical changes

- **Kz vertical diffusion updates tracer mass in flux form, with the column
  ledger (finding OF-113).** `ImplicitVerticalDiffusion` with a Kz field
  (lat-lon and reduced-Gaussian runs, `tm5_beljaars_viterbo_local_kz` and
  `geoschem_holtslag_boville_vdiff` on the cubed sphere) still solves the
  backward-Euler system for the new mixing ratio, but no longer writes back
  `m_k (g_k + c)`. In `Float32` that lost tracer mass with one sign: in the
  pure-pressure levels `Δt D/m` is the same in every column (exactly with
  isothermal layer thickness and uniform Kz, as in the golden configurations;
  approximately otherwise), so the rounding
  of the diagonal `1 + Δt (D_{k−½} + D_{k+½})/m_k` is too, and the solved
  departure from the column minimum is non-negative (diffusion alone: −2.0e-6
  of a background-free tracer per day in `run_ll72_upwind_cpu_f32`). The
  tracer mass is now updated from the interface fluxes
  `F_{k+½} = Δt D_{k+½} (g_k − g_{k+1})`, `rm_k ← (rm_k + F_{k−½}) − F_{k+½}`,
  so the column total telescopes, and the remaining rounding (which keeps one
  sign where fluxes keep one direction: −9.4e-9 of CO₂ per day) goes through
  the column mass ledger with persistent carry of the `dkg` solve. Diffusion's
  own change of each total in that run is now below 5e-10 per day (blob
  −1.5e-10, CO₂ +1.5e-11, fossil CO₂ −3.5e-10 of the emitted mass, Rn-222
  +4.6e-10). The solve and fluxes are unchanged and both forms agree in exact
  arithmetic, so the adjoint is unchanged; layers without exchange keep their
  mass bit for bit. The VMR-level reference `apply_vertical_diffusion!` keeps
  the matrix form. The flux form trades local accuracy for column
  conservation: per-layer rounding scales with `ε·Δt D/m` (`Float32`, strongly
  mixed columns: worst cell 1.4e-4 relative against 1.0e-5 for the matrix form,
  column error per solve 3e-8 against 5e-6; damped by the next mixing solve),
  which is why the golden fields below change by more than a few ulps.
  Golden changes (one day; `blob` total drift before → after, then the largest
  change of a field and of a column mean relative to its maximum):
  `run_ll72_upwind_cpu_f32` −2.0e-6 → +1.8e-8 (fields 2.5e-5, column means
  3.2e-6); `run_ll72_slopes_gpu_f32` −2.2e-6 → +1.3e-8 (2.5e-5, 3.4e-6);
  `run_c24_upwind_cpu_f32` −6.3e-7 → −1.1e-10 (1.6e-5, 1.5e-6);
  `run_c24_slopes_gpu_f32` −7.0e-7 → −1.1e-8 (1.7e-5, 1.3e-6);
  `run_c24_linrood_cpu_f32` −7.0e-7 → −5.9e-9 (column means 2.3e-4 and
  blob cells 2.6e-3 at isolated upper-level points where the nonlinear Lin–Rood
  limiter amplifies rounding; 99 % of cells within 8.7e-7, and the new run is
  closer to a `Float64` run: fossil CO₂ column means within 3.2e-7 instead of
  2.3e-4); `run_c90_era5_localkz_gpu_f32` (SF₆ without sources −2.6e-7 →
  −2.8e-10; fields 1.7e-5, column means 1.7e-5) and
  `run_c90_merra2_hbvdiff_gpu_f32` (uniform −6.1e-10 → +1.2e-9; fields
  8.3e-5, column means 8.5e-6). The uniform tracer stays at the advection
  level (at most 1.1e-8 per day). `Float64` cases (`run_ll72_ppm_cpu_f64`,
  `run_ll72_ppm_cw84_cpu_f64`, `run_o24_upwind_cpu_f64`, `run_c24_ppm_cpu_f64`,
  `run_c24_ppm_cw84_cpu_f64`) change by at most 1.7e-14 of a field's maximum
  (totals 1.9e-16); their drifts stay at 0 or ±1.5e-16. Cases with `tm5_dkg` or
  `geoschem_nonlocal_vdiff` diffusion and preprocessing are bit-identical. The
  kernel also fuses the mass/mixing-ratio conversions and launches all six
  cubed-sphere panels at once: on an L40S a C90 L72 call with ten `Float32`
  tracers takes 1.35 ms instead of 4.06 ms, a 1° L72 lat-lon call 2.70 ms
  instead of 5.89 ms. `DiffusionWorkspace` gains `exchange` and `elimination`
  column scratch, and lat-lon and reduced-Gaussian workspaces now hold a
  `ledger_carry`; the per-tracer reduced-Gaussian transport passes
  `tracer = idx`, as the per-tracer Lin–Rood path does
  (`test/core/test_diffusion_flux_form.jl`).
- **Pressure-placed observation samples use total pressures** (issue #20,
  finding OF-116). Observation sampling (`[output.observations]`) placed a
  record by its pressure (ObsPack `pressure`, a total pressure) against the
  column's dry interface pressures on dry-basis runs, which lie lower by the
  weight of the water vapour above each interface (4.9 hPa at the surface for
  50 kg m⁻² of precipitable water), so a record could land one or more layers
  too low.
  The default `pressure_basis = "total"` (`TotalFromHumidity`) now compares it
  with the total interface pressures
  `p[k+1] = p[k] + g·m[k] / (A·(1 − q[k]))`, built from the dry air mass and
  the binary's specific humidity, blended in time like the air mass:
  `qv_start` at the run start and `qv_end` at window ends on lat-lon binaries
  preprocessed with `output.include_qv = true`; on cubed-sphere binaries with
  the GCHP VDIFF payload the window-start VDIFF `qv` of the next window (read
  from the binary, or the next binary's window 1). The run's last window end
  has no next window and takes the ended window's `qv`: on the golden MERRA-2
  C90 L72 binary its interfaces are up to 23 Pa (99th percentile 11 Pa) from
  the met model's total interfaces, against ≤ 0.23 Pa with the next window's.
  Reduced-Gaussian binaries and cubed-sphere binaries without the VDIFF
  payload carry no humidity. On merged levels the binary's dry-mass-weighted
  `q` makes the inverse inexact: on the golden `pre_ll72` binary (ERA5 L137
  merged to 34 levels) the rebuilt total surface pressure is −0.012 Pa off on
  average and 0.125 Pa at most (dry interfaces: 169 Pa and 700 Pa low).
  Only rows with `vertical_placement = 2` change: their `intake_level`, layer
  heights and `<tracer>_intake`. On the GEOS L72 levels (ps = 1010 hPa,
  q = q_s (p_mid/ps)^λ with λ set by the precipitable water, pressures
  uniformly spaced) 25% of pressures between 900 and 1010 hPa, 6% between
  500 and 900 hPa and 0.4% between 200 and 500 hPa move up for a tropical
  column (q_s = 18 g/kg, 50 kg m⁻²; an independent construction gave 29%,
  6.5% and 0.36%), and 5%, 1% and 0% for q_s = 4 g/kg and 10 kg m⁻²; in the
  sampler test case two of
  three pressure-placed rows moved up one layer and their intake values by
  −2.2 ppm (the column's per-layer CO₂ gradient). Height-placed, lowest-layer
  and site rows, and the written `p_half_dry`, are bit-identical. A dry-basis
  run whose binaries do not all carry humidity (every binary's header is
  checked before transport), such as every reduced-Gaussian run, now fails to
  build a sampler with pressure-placed records unless `pressure_basis = "dry"`
  (`DryInterfaces`, the old comparison) is set. Moist-basis runs are unchanged (their air mass
  already gives total pressures). No golden case samples observations.
- **4D-Var gradients** (issue #20) change at the rounding level: the single
  reverse sweep sums the observations' contributions inside the adjoint state
  instead of adding one footprint per observation. Against the previous
  algorithm the gradients agree to `1e-12` relative in Float64 and `1e-5` in
  Float32 (`test/core/test_cs_4dvar_single_sweep.jl`); costs and simulated
  values are unchanged. Footprints from `cs_surface_emission_footprint` and
  `cs_surface_emission_footprint_from_seed`, and Jacobians, are unchanged bit
  for bit (single-threaded comparison with the previous main on 390 footprint,
  15 from-seed and 15 Jacobian cases: every scheme family, diffusion with both
  couplings, TM5 and CMFMC, three schedules, three tape storages). Forward
  results are unchanged.
- **Cubed-sphere adjoint footprints and 4D-Var gradients** change for runs
  with convection, with `DiffusiveSurfaceFluxBoundary`, or with limited-PPM
  `:pinned_host` / GPU `:mmap` tapes: the replays and tapes now follow the
  production step (issues #57 and #60, see Fixes). Forward results are
  unchanged.
- **ERA5 → TM5 convection conversion follows TM5's half levels** (issue #54,
  pera-noaa). ERA5 archives UDMF/DDMF level k as the flux through the top of
  layer k, with none through the ground (TM5 `tmm.F90`: "levels top down,
  surface implicit zero"). The ERA5 N320 preprocessor placed level k at the
  bottom of layer k, so level 137 became a flux through the ground, and
  `ec2tm_from_rates!` took the first interface with flux as the cloud-top
  layer, one layer too low; the column closure then moved the cloud-top
  detrainment to the layer of largest entrainment. Now level k is interface k
  and interface Nz + 1 is zero, and the cloud top is the layer above the first
  interface with flux, as in TM5's `ECconv_to_TMconv`. The fields now return
  the input fluxes at every interface and match an independent transcription
  of TM5's code to rounding; the closure moves only rounding. On 3000 deep
  ERA5 columns the old conversion lifted 14 % less boundary-layer tracer above
  500 hPa in a 3 h step and detrained about 4 levels too low; mass was
  conserved. `ec2tm_from_rates!` now rejects a non-zero flux through the
  ground (a packing error). ERA5 TM5-convection binaries must be regenerated;
  runs on attached CATRINE convection (the CATRINE benchmark sets) and all
  goldens are unaffected.

- **Column mass ledger carry** (OF-110). The `Float32` column mass ledgers of
  TM5 convection (and `cmfmc_matrix`, which uses the TM5 solve) and of `dkg`
  diffusion (and so of GEOS-Chem non-local VDIFF) add each column's rounding
  residual to its largest cell, where the part below the cell's spacing was
  lost, with one sign for tracers peaked at the surface: ERA5 C90 L117 fossil
  CO₂ lost 8.1e4 kg, 2.5e-9 of its burden, in one day of `dkg` diffusion.
  That part, about half an ulp at most, is now kept as a per-column,
  per-tracer carry in the operator workspace and added with the column's next
  residual. The same day's change is +57 kg instead, and with all operators
  fossil CO₂ loses 1.1e-10 instead of 2.7e-9 of its burden. Cells are still
  rounded once per solve, so runs with TM5 convection or `dkg` diffusion
  change at the rounding level (golden outputs of such runs change). With
  level merging (`n_merge > 1`) the ledger covers the whole fine-grid
  operation, including the split. The carry is not written to restarts (a
  restart drops about half an ulp per column, once), and adjoint forward
  replays keep none, so replays still reproduce recorded states bit for bit.
- **CMFMC column mass ledger** (OF-111). CMFMC convection's explicit update
  conserves column mass by construction but rounds each layer's new mass when
  writing it in `Float32`, and the lost sub-ulp changes had one sign: in one
  day of MERRA-2 C90 L72 convection SF₆ gained 1.3e-9 and background CO₂
  2.5e-9 of their burdens. A compensated column ledger with the same
  persistent carry now returns them on lat-lon, reduced-Gaussian and
  cubed-sphere grids (and in the adjoint's forward replay, without a carry):
  the same day changes SF₆ by −5.5e-12. The OF-110 carry also removes the gain
  of GEOS-Chem non-local VDIFF (SF₆ +4.5e-9 → −5.3e-12 per day); with all
  operators the day's SF₆ residual falls from +3.8e-9 to −6.6e-10 of the
  burden, the advection level. Columns with a flux through the model top or a
  layer without air, and the clamped variant, are not ledgered. Runs with
  CMFMC convection change at the rounding level.
- New internal state: `TM5Workspace.ledger_carry` and
  `CMFMCWorkspace.ledger_carry` (sized by the first `apply!`) and
  `DiffusionWorkspace.ledger_carry`. The cubed-sphere
  `apply_vertical_diffusion_vmr!` takes `tracer` for single-tracer panel
  tuples (used by the per-tracer Lin–Rood path); TM5, CMFMC and `dkg` kernels
  take the carry array as an argument. Packed cubed-sphere `dkg` diffusion now
  throws a `DimensionMismatch` unless the workspace was built for exactly the
  state's number of tracers (`DiffusionWorkspace(state)` or
  `DiffusionWorkspace(air_mass, Hp, Nt)`); before, the tracer count of the
  workspace was not checked on this path.
- The ledgers rely on IEEE arithmetic without fast-math reassociation (which
  would break `TwoSum`). They are tested on CPU and CUDA; the Metal checks
  (`test/diagnostic/test_metal_kernels.jl`) have not been run yet.
- **ERA5 spectral pole rows are ordinary cells** (issue #58, pera-noaa). The
  spectral preprocessor set the zonal mass flux of the two pole rows of every
  lat-lon target and cubed-sphere staging grid to zero, although these rows
  are cells centred at ±(90° − Δφ/2). A flow across the pole then had no path
  through the polar cap: the lat-lon column balance re-routed `bm` at every
  latitude (real ERA5 72 × 37: 4.5–10 % RMS in `bm` equatorward of 80°, 55 %
  poleward), and the cubed-sphere staging recovered `u = 0` in the pole rows
  (C24 poleward of 75°: 9 % in `am`/`bm`, 82 % in `cm`). Only the meridional
  faces on the poles stay closed. The narrow pole-row cells raise the
  lat-lon substep count (real ERA5 day, per hourly window: 72 × 37 4 → 7–10,
  360 × 181 43–63 → 113–177, 720 × 361 155–244 → 437–715; 2.0–2.9× more
  substeps, and about that much more lat-lon transport time). The same zeroing
  had been removed once before (bug archive, 2026-04-03); a regression test
  now keeps it out. ERA5 N320 and reduced-Gaussian targets are unaffected.
  ERA5 spectral lat-lon binaries now record `pole_row_zonal_flux = "open"` in
  their header.
- **Lat-lon → cubed-sphere regrid** (`regrid_ll_binary_to_cs`):
  - winds are recovered with each window's own source substep count
    (`steps_per_window_by_window`, through `flux_application_seconds`), not
    the header maximum; with an adaptive lat-lon schedule a window with fewer
    substeps was regridded too strong by the ratio;
  - the output substep count no longer copies the source's (which the
    lat-lon pole rows now set): by default each cubed-sphere window gets the
    smallest count whose per-substep Courant ratio meets the positivity
    limit, written as a per-window schedule; `steps_per_window` still fixes
    one count for all windows;
  - ERA5 spectral lat-lon sources without the `pole_row_zonal_flux = "open"`
    header key (written before the #58 fix) are refused before any output is
    opened. Their stored pole-row `am` is the column balance's correction,
    not zero and not the wind, so only the header identifies them;
    regenerate them;
  - with the adaptive schedule the `column_closure_*` header keys are
    evaluated again on the rescaled stored fluxes
    (`rescaled_column_closure_record`); they described the one-substep
    balance and were N times the stored amounts.
- **ERA5 humidity on lat-lon targets** (issue #63, pera-noaa). The thermo
  `q` is a set of point values at longitudes 0, Δ, … (index 1 at −180° after
  the roll), but the lat-lon and cubed-sphere-staging interpolation read it
  as cell-centred, half a source spacing (0.25°) east, and a 720 × 361 target
  copied the 720 × 361 archive without interpolating (also up to 0.25° off in
  latitude near the poles). `q` is now always interpolated bilinearly with
  weights built from the file's coordinates, which are read through
  `GriddedIO.find_coordinate`. Real ERA5: `q` changes by 1–2.5e-4 kg/kg RMS
  (up to 6e-3 at sharp gradients), the dry air mass by 2e-4 RMS (up to 0.6 %
  locally). The reduced-Gaussian path was already right and is unchanged.
- Golden outputs that change: `pre_ll72` (#58 and #63), `pre_c24` (#58 and
  #63) and the runs on them (`run_ll72_*`, `run_c24_*`). `pre_o24` and the
  N320, MERRA-2 and GEOS cases do not change.

- **Static surface fluxes regrid conservatively on every grid, with the
  file's totals** (#59). Lat-lon and reduced-Gaussian targets now default to
  `regridding = "conservative"` (before: bilinear). The bilinear default
  renormalised only positive totals of files with an area variable: net-uptake
  fields and area-less files (Zhang Rn-222, EDGAR SF₆) kept the sampling
  error (Rn-222 −0.95 % on LL 72×37 and −0.20 % on O24; EDGAR SF₆ +47 % above
  the conservative total), and a sampled net of the other sign flipped every
  cell. The conservative path now regrids the per-cell rate, flux × the file's
  `cell_area`/`area` (else the source mesh's latitude-circle area), divided by
  each source cell's covered area (`Regridding.source_coverage`), so every
  model total equals the file's own total to rounding on lat-lon,
  reduced-Gaussian and cubed-sphere grids. Since the file's area now sets the
  total, it must be in m² and agree within 1 % (norm-based) with the cell
  areas the file's coordinates define (`GriddedIO.check_latlon_cell_area`),
  and be positive and finite in every cell with a nonzero flux (masked zero
  rows would drop their emission); otherwise the build stops with an error.
  On a target that does not cover the sphere (not used by the shipped runs),
  a source cell covered only in part puts its whole amount into the covered
  part (documented in Operators). Before, the density was regridded
  over great-circle polygons (CAMS total −9.8e-5 on C24) and lost the slivers
  that reduced-Gaussian rings with different longitude counts leave uncovered
  (6.3e-4 of the sphere on O24). Explicit `regridding = "bilinear"` rescales
  emission and uptake separately (no cell changes sign) or stops with an
  error; the two factors are logged and a factor more than 10 % from 1 is a
  warning (GridFED December 2021 uptake on a 5° grid: 3.53). Runs with static sources change: the `co2`, `fossil` and `rn222`
  tracers of the `run_ll72_*` and `run_o24_*` golden cases (cell patterns, and
  the Rn-222 total) and of the `run_c24_*` cases (by up to 8e-5 of the
  largest cell rate). Time-varying lat-lon series (cubed sphere only) still
  regrid their densities per slice and are unchanged.

- **Runtime CFL subcycling runs whole palindromes' worth of passes**
  (issue #61). When the runtime pilot needs `n > 1` subcycles, a transport step
  now runs `(X Y Z)^n → midpoint → (Z Y X)^n` with every flux divided by `n`
  (`(H V)^n → midpoint → (V H)^n` on reduced Gaussian) instead of
  `X^n Y^n Z^n | Z^n Y^n X^n`. The new composition is symmetric (second
  order, same `6n` sweeps) and identical at `n = 1`. The old order subcycled
  each direction alone, so the passes of one leg still removed that
  direction's full-flux outflow before another direction refilled the cell:
  in exactly non-divergent flows air mass went negative (to −0.46 of a cell's
  mass at C12, n = 6) and uniform tracers were lost. All topologies now take
  one count from the palindrome budget
  `2 (out_x + out_y + out_z) / (n m_ref) ≤ cfl_limit` with
  `m_ref = min(m, m_end)`, `m_end = m + 2 (in − out)` the cell's mass after
  the step (the reference mass of the write-time positivity gate). Between
  `X Y Z` blocks a cell's mass lies between `m` and `m_end`, so every pass
  removes at most `cfl_limit / 2` of the mass the cell holds when it starts
  and air mass stays positive through the step. Budgeting against the
  step-start mass `m` alone (main's per-direction counts too) does not bound
  a cell that drains within the step: one cell of mass 1 among cells of 100
  with inflow 1.51 in x and outflow 2.0 in y got `n = 5`, its air mass went
  negative, and PPM gave a mixing ratio of −1.7e43 (now `n = 211`). A cell
  with outflow and `m_ref ≤ 0`, or a non-finite flux, throws an
  `ArgumentError` (main also failed, with an "exceeded max_n_sub" error);
  the lat-lon and
  reduced-Gaussian default `cfl_limit` is `DEFAULT_CFL_LIMIT` (0.95, was 1.0
  per direction), like the cubed sphere. The footprint/4D-Var tape recorders
  record the same composition. Only runs whose pilot returns `n > 1` change:
  cubed-sphere binaries with the `binary_schedule` contract run `n = 1`, and
  lat-lon binaries written with the default positivity gate (the same
  palindrome budget) stay below it. The reduced-Gaussian write-time gate still
  bounds horizontal and vertical outflow separately (each `≤ 0.95 m`), so a
  reduced-Gaussian binary near that gate can now subcycle at run time (static
  palindrome ratio up to 3.8, `n = 4`). Golden outputs do not change: their
  worst static ratios are below the budget (`pre_o24`: 0.61). The per-direction
  pass-count helpers (`_x/_y/_z_subcycling_pass_count`, the face-indexed
  `_sweep_*_face_subcycled!`) are removed; the corrected
  `operators_on_binaries.md` no longer claims the per-leg counts keep mass
  positive.
- **`PositivityLimiter` keeps downhill slopes** (issue #62). Its slope was
  `minmod(s_c, c_i)`, which is zero whenever the centred slope is negative, so
  the limiter acted as a sign test and treated rising and falling profiles
  differently (mirror asymmetry 0.25 in a 1-D test). It now keeps the sign
  and clips the magnitude to `|s| ≤ 2 c_i`, the bound that keeps both face
  values non-negative. Selectable only from Julia (`SlopesScheme(PositivityLimiter())`).
- **Lin–Rood `ppm_order = 7` uses one edge value per face** (issue #62). The
  seam value replaced the edges only at the panel-edge faces, so the face
  kernels rebuilt the cell next to a seam with a cross-seam fourth-order edge
  when they evaluated its interior face. ORD=7 now follows FV3's cube-edge
  values (`tp_core.F90`, `iord < 8`): the Putman–Lin seam average at the
  panel edge and one-sided `5/14, 11/14, −2/14` edges on the faces beside it,
  in the forward and adjoint kernels alike. Solid-body rotation errors fall
  by 10–31% (C8–C64). Runs with `ppm_order = 7` change; `ppm_order = 5` (the
  default, and all golden cases) is bit-identical.

### Fixes

- **Cubed-sphere adjoint: `:pinned_host` and GPU `:mmap` tapes handed the
  limited-PPM reverse sweep `m === rm`** (issue #57, pera-noaa). Host-side tape
  storage reads a staged state back through a device cache, and both policies
  kept one cache per panel shape. A `MonotoneLimiter` or `CW84Limiter` sweep
  record reads its air mass and tracer mass in one call, so the kernel got
  the tracer mass twice: footprints were off by about 80 % and 4D-Var
  gradients by about 20 % at every entry point (`FullCheckpoint`, Stride,
  Revolve, from-seed, `cs_surface_flux_4dvar`). `:device`, CPU `:mmap` (mmap
  views), unlimited schemes and Lin–Rood were not affected. Reads now name
  their operand, `_tape_panels(slot, AirMassRead())` or `TracerMassRead()`,
  and `PinnedHostCSTapeStorage` and `MmapCSTapeStorage` (the new
  `AbstractHostCSTapeStorage`) keep one device cache per read role and panel
  shape; `:pinned_host` and `:mmap` now agree with `:device` to rounding
  on the CPU and on CUDA (L40S). Internal API: the one-argument
  `_tape_panels(slot)` and `_ensure_tape_read_cache!` are replaced by
  `_tape_panels(slot, role)`, `_register_tape_read_backend!` and
  `_tape_read_cache!`; the mmap field `device_caches` became `read_caches`,
  and the pinned-host field `device_cache` became `read_template` and
  `read_caches`.
- **Cubed-sphere adjoint replays and tapes now follow the production step**
  (issue #60, pera-noaa). Two differences between the adjoint's forward model
  (`run_cs_footprint_forward`, the 4D-Var simulated values, the tape
  recorders) and `TransportModel.step!`:
  (1) With `DiffusiveSurfaceFluxBoundary` (`[diffusion] surface_flux_boundary
  = true`, which `geoschem_holtslag_boville_vdiff` needs) production runs
  `S(dt) → V(dt)` at the palindrome center, while the replays always ran
  `V(dt/2) → S(dt) → V(dt/2)`: the footprint differed from finite differences
  of production by about 9 %. (2) Production refreshes the X halos of the air
  mass and tracer mass at the start of every step; the replays and recorders
  did not, so from the second step on the first X sweep read tracer halos
  from before the previous step's convection (replay vs. production up to
  1e-3 relative after 12 steps, growing with the step count and the
  convective mass flux), and checkpoint windows, which restart with fresh
  halos, disagreed with `FullCheckpoint` by about 1e-2 for limited PPM. The
  center order is now one function, `surface_flux_center!(diffuse!, emit!,
  coupling, dt)`, used by production (through `palindrome_center!`, now the
  center of every topology's palindrome) and by the replays and recorders
  (which record one full-step diffusion after the midpoint for the diffusive
  boundary); every replayed and recorded split-sweep step starts with
  `fill_cs_step_start_halos!`, production's refresh, and records its
  `_CSHaloRecord(1)`. `cs_tape_byte_estimate` counts the new records. The
  replay now equals production to rounding, footprints match production
  finite differences to about 1e-10, and Stride and Revolve match
  `FullCheckpoint` to 1e-16. The documented "Revolve drift of about 1e-7 from
  limiter flips at bisection halo refills" was this missing refresh and is
  gone. Forward production results are unchanged bit for bit (cubed-sphere
  split-sweep and Lin–Rood, lat-lon and reduced-Gaussian palindromes, every
  center, Float32 and Float64); adjoint footprints and gradients change for
  runs with convection or the diffusive boundary.
- **Observation heights: altitude-only records are placed above the ground,
  never silently in the lowest layer** (issue #56). ObsPack records that give
  an `altitude` above sea level but no `intake_height` or `elevation` had a
  `NaN` height and were sampled in the lowest model layer without a message.
  This is the aircraft-insitu layout: in the OCO-2 MIP v5.0.1 daily ObsPack,
  16.5% of the 2021-12 to 2023-12 records fill both per row, including 97% of
  aircraft-insitu and all AirCore records. Heights are now resolved row by row
  (`intake_height`, else `altitude − elevation`). An altitude without an
  elevation becomes a height above the true ground through a DEM
  (`dem_file`), and the record is placed that far above the model surface.
  A record with no such height but a `pressure` is placed by pressure. The
  model surface height is never used as the true ground. Records that still
  cannot be placed, and ObsPack records with no vertical coordinate at all,
  stop the sampler build (`unplaced_records = "error"`, the default) or are
  skipped and counted (`"skip"`, `n_unplaced_*` attributes). A table row with
  an altitude but no elevation no longer fails at read time; it follows the
  same rules. `absolute_altitude_above_agl_m` optionally places
  free-tropospheric records by pressure or by altitude above sea level against
  the model surface height (`surface_height_file`); a record with only an
  intake height takes its ground from the DEM there. Point events with an intake height (or
  altitude and elevation) are sampled as before, except ObsPack rows whose
  file has an `intake_height` variable but a fill value on that row: they used
  `NaN` (lowest layer) and now use `altitude − elevation`.
  Undeclared fill values are not placed: an ObsPack `pressure` ≤ 0, an
  `elevation` (or `site_elevation`) below −450 m and a negative
  `intake_height` are ignored with a warning (the JMA aircraft-flask rows of
  the MIP ObsPack give elevation = intake_height = −1000 at 5 km altitude and
  were sampled in the lowest layer; they are now placed by altitude − DEM).
  Records placed more than 100 m below the ground are reported in a warning.
  In ObsPack station series (`mode = "sites"`), rows that give only an
  altitude are keyed by that altitude (one site per level of a tower that
  states its levels as altitudes) instead of forming one lowest-layer
  "surface" site; a dataset whose records stray more than 1° from the site
  position (a moving platform) is reported, since each of its heights becomes
  a site at that position.

### API and configuration changes

- **OCO-2/OCO-3 level retrievals, the MIP converter, co-sampling and cell
  averages** (issue #20 follow-up, decision D30).
  - *Hat projection.* A retrieval block may describe values on levels, the
    profile linear in pressure between them (`retrieval_mapping =
    "hat_levels"`, `Output.HatLevelProjection`; OCO-2/OCO-3: 20 levels "on
    layer boundaries"), besides layer means (`"layer_rebin"`,
    `Output.LayerRebin`, the default and the only form of earlier files).
    The model column, stretched onto the levels as before, is projected onto
    the levels' hat functions in dry air (the L2, Galerkin, projection
    ``c = M⁻¹ b``, ``M_ji = ∫φ_j φ_i dp_dry``, ``b_j = ∫φ_j x dp_dry``; one
    tridiagonal solve), then ``X = X_a + Σ h_j a_j (c_j − x_a,j)`` (the MIP
    protocol's eq. 2) with the file's weights, kernel, prior and prior column
    ``X_a`` (`xco2_apriori`; new optional `RetrievalColumn.prior_column` and
    observation-file variable `retrieval_prior_column`; without it ``X_a =
    Σ h x_a``, which for the 10-s averages differs by up to ~0.01 ppm). It
    conserves the model's column with the hat integrals as weights and
    reproduces profiles the retrieval can represent: a model equal to the
    retrieval's prior, on a fine column aligned with the levels, gives ``X_a``
    to 1e-5 ppm (on GEOS L72, 1e-5 to 2.1e-3 ppm; a plain hat average would
    be 0.0185 ppm low). The
    mapping is a type of the shared operator (`retrieval_column_weights!`,
    `retrieval_profile!`, `retrieval_column_value` take the layers' air
    masses; new `retrieval_projection_matrix`), used by the 4D-Var and the
    forward sampler; layer records give bit-identical results.
    `Output.tm5_layer_column` builds TM5-4DVar's 19-layer form (levels as
    interfaces, midpoint kernel and prior, weights normalized to 1) for
    comparisons. Observation files store `retrieval_mapping` per record.
  - *Converter.* `read_oco2_mip_observations` / `convert_oco2_mip_observations`
    and `scripts/preprocessing/oco2_mip_to_observations.jl` turn the OCO-2 and
    OCO-3 v11 MIP 10-second files into schema-v2 column records with a level
    block (one file per day or month), every record of the time range, with
    `flag = 1 + assimilate_flag` for records outside the experiment's data
    types (LN, LG, OG by default; `SAM` for OCO-3's data type 0) and
    assimilation flags (1 by default); `mapping = Output.LayerRebin()`
    (`--mapping tm5`) writes TM5's layer form. `write_observations` takes
    extra root `attributes`.
  - *Co-sampling.* `bind_to_mesh(...; flagged_records = "cosample")`
    (`CosampleFlaggedRecords`; driver key `[observations].flagged_records`)
    binds records with a nonzero flag as co-samples (`assimilate = false`):
    simulated and reported in `CS4DVarResult` and departure files (new
    `flag` column), never in the cost or the gradient (the cost and the
    reverse pass's forcing are bit for bit those without them), not
    screened, not extending the taped run. Default `"drop"` as before.
  - *Cell averages.* `bind_to_mesh(...; cell_averaging = CellAverage(;
    time_tolerance_s, correlation))` (driver table
    `[observations.cell_averaging]`) averages assimilated column records of
    one cell, step sampling and overpass into one observation
    (`CSCellAverageObjective`, `CELL_AVERAGE_PLACEMENT_CODE`) with 1/σ²
    weights, ``σ²_meas = ΣΣ w_i w_j ρ_ij σ_i σ_j/(Σw)²`` and the model error
    added once; the members stay as co-samples, and departure files have a
    row for each average (placement code −4, numbered by `cell_average_id`,
    with a row id above every record id; departure ids are checked unique;
    σ_meas, σ_mod and `cell_average_correlation`) besides the member rows
    (`cell_average_id`), so Σ normalized_departure² over assimilated rows is
    2 J_obs. `inflate_cell_model_error` is the alternative that keeps the
    records and multiplies each record's `model_sigma²` by the group size
    (not the profile-error term).
  - *Forward sampler.* `oco2_lite` sources take `averaging_kernel_tracers`
    (opt-in; `AveragingKernelColumns`) and `averaging_kernel_mapping`
    (`"hat"`, `"tm5"`): the reader also reads each sounding's retrieval
    (`Output.read_oco2_retrieval_columns`, Lite or MIP files) and the
    soundings file gets `<tracer>_ak_column` from the shared operator.
    Without the key the output is unchanged.
  - Docs: `docs/src/theory/adjoint_status.md` (level retrievals; OCO-2 MIP
    records), `schemas/cs_observations_v2.toml`, `schemas/cs_departures_v2.toml`,
    the TOML and output schema pages. Tests:
    `test/core/test_retrieval_hat_projection.jl`,
    `test/core/test_cs_4dvar_oco2_mip.jl`.
- **Averaging-kernel (retrieval) columns in the 4D-Var** (issue #20, part 4).
  Observation schema v2 gains an optional, NaN-padded retrieval block for
  column records (`retrieval_level` / `retrieval_interface` dimensions;
  `retrieval_pressure_pa`, `retrieval_pressure_weight`,
  `retrieval_averaging_kernel`, `retrieval_prior_profile`, optional
  `retrieval_profile_sigma`, `retrieval_surface_pressure_pa`), ordered from the
  surface up (`ordering = "surface_to_top"`, required) with stated units
  (pressures converted to Pa, the prior to mol/mol from `"ppm"`, `"ppb"`, …)
  and validated (strictly decreasing pressures, weights ≥ 0 summing to 1,
  finite values, a prior within [0, 1] mol/mol, the surface pressure equal to
  the first interface; only column records, which are then absolute column
  averages: value in (0, 1] mol/mol, value_sigma and model_sigma at most
  1 mol/mol; errors larger than the value, as for a low-signal retrieval, are
  accepted). `value`, `value_sigma` and `model_sigma` may state a `units`
  attribute: a dry mole-fraction unit (`"mol mol-1"`, `"ppm"`, `"ppb"`, …)
  is converted to mol/mol and must be stated on all three; files with
  retrieval blocks must state one (the writer states `"mol mol-1"`); files
  without units read as before. `CSObservationRecord` gains
  `retrieval::Union{Nothing, Output.RetrievalColumn}` (the 16-argument
  positional constructor still builds a record without one). `bind_to_mesh`
  binds a column record with a block to the new `CSRetrievalColumnObjective`
  and keeps column means for the others: at every sampled state the model's
  total interface pressures (dry air mass plus humidity; or the dry
  interfaces with `pressure_basis = "dry"`) are stretched onto the retrieval
  column and the dry mixing ratios rebinned onto its layers by pressure
  overlap, `X = Σᵢ hᵢ aᵢ (R x)ᵢ + Σᵢ hᵢ (1 − aᵢ) x_a,ᵢ`, TM5-4DVAR's operator
  (`averaging_kernels.F90`), with functional weights `Hₖ/max(mₖ, ε)` and the
  prior term as offset. The operator is shared in `Output`
  (`RetrievalColumn`, `stretch_model_interfaces!`, `retrieval_rebin_matrix`,
  `retrieval_column_weights!`, `retrieval_prior_offset`, `retrieval_profile!`,
  `retrieval_column_value`, `retrieval_model_error_variance`). New keyword
  `bind_to_mesh(...; max_surface_pressure_mismatch_pa = Inf)` (inversion
  driver: `[observations.placement].max_surface_pressure_mismatch_pa`): a
  record whose time-weighted model minus retrieval surface pressure exceeds
  it is an error under `unplaced_records = "error"` and, under `"skip"`, is
  reported but left out of the cost and the gradient (its
  `CSObservationStencil` has `assimilated = false`; counted in one log
  message); by default the stretch absorbs any mismatch, as in TM5. An
  uncertainty of the model profile on the retrieval layers
  (`retrieval_profile_sigma`; not the retrieval's a-priori uncertainty) adds
  the model-error term `Σᵢ (hᵢ aᵢ σᵢ)²` to σ² (fixed by the record, so the
  cost stays quadratic; analogous to, not the same as, TM5's `Σ H²σ²` on
  model layers). The inversion driver accepts
  `[observations.placement].pressure_basis` without `"pressure"` in
  `vertical_coordinates` (it selects the pressures of averaging-kernel
  columns; `Output._parse_vertical_placement(...; pressure_basis_for_columns
  = true)`); `[output.observations]` still rejects it there.
  `CSObservationStencil` gains `surface_pressure_mismatch_pa`,
  `retrieval_profile` and `assimilated` (the 9-argument constructor still
  works); new `RETRIEVAL_COLUMN_PLACEMENT_CODE` (−3). Departures v2: rows of
  averaging-kernel columns carry code −3, and files with such rows add
  `surface_pressure_mismatch_pa`, `retrieval_model_profile`
  (`retrieval_level`, surface first), `profile_model_sigma` and
  `assimilated`; `CSDepartureRecord` gains these fields (the 24-argument
  positional constructor still works) and `normalized_departure` divides by
  `√(value_sigma² + model_sigma² + profile_model_sigma²)`; the reader requires
  the profile's ordering and units and the mismatch's units. Files without
  retrieval blocks are written and read as before. The forward observation
  sampler does not apply averaging kernels yet. See
  `docs/src/theory/adjoint_status.md` (averaging-kernel columns) and
  `test/core/test_cs_4dvar_retrieval_columns.jl`.
- **Point observations placed in the column, and time windows, in the 4D-Var**
  (issue #20, part 3). `bind_to_mesh(set, mesh, t_start, dt; nsteps,
  tracer_filter, out_of_range_policy, placement, terrain)` turns each point
  record into a forward-sampler request and places it with the forward rule
  (`Output.resolve_placement` with an `Output.VerticalPlacementSpec`: ordered
  `vertical_coordinates`, height above the true ground measured from the model
  surface and/or pressure; pressure basis; altitude option; unplaced policy;
  and an `Output.ObservationTerrain` with the model surface height and a DEM)
  into a `CSPlacedPointObjective`; column records bind to column means. At
  every sampled state the layer is chosen from the replayed air mass and the
  new `CSObservationColumnMet` (model-top pressure, gravity, layer
  temperature, humidity; static or per step; `cs_surface_flux_4dvar(...;
  column_met)`) with `Output.sample_column_layer!`, which the forward sampler
  now also calls, so both choose the same layer; the observation is that
  layer's mixing ratio and its functional weight `1/max(m, eps)` there. A
  record with `time_window_s = W > 0` samples the steps overlapping
  `[t − W/2, t + W/2]` with weights proportional to the overlap (TM5-4DVAR's
  Δt/ΣΔt), truncated at the run's ends and renormalised (`CSTimeSampling`;
  window records need `nsteps`); `CSLinearFunctional` and the step forcing
  carry entries at several steps. `ClampOutOfRange` now keeps the part of an
  outside record's window inside the run (else the first or last step, as
  before), counts clamped and skipped records in an `@info`, and departure
  rows mark such records `clamped`. Record ids must be unique
  (`CSObservationSet`, `read_observations`, `bind_to_mesh`), and the optional
  vertical fields read a declared `_FillValue` (xarray's NaN) as absent. The
  inversion samples the end of the model step containing a record's time (or
  its window mean), the forward sampler window ends (linear or nearest):
  both read the same layer and value of the same state.
  Instantaneous observations give bit-identical costs and gradients to
  before (checked on 75 scheme/physics/checkpoint combinations, Float32, and
  the footprints and Jacobians). `CS4DVarResult.stencils`
  (`CSObservationStencil`) returns each observation's steps, weights,
  levels and placement. Observation `sigma` is
  `√(value_sigma² + model_sigma²)`.
- **Inversion driver: observation files and hydrostatic meteorology**
  (issue #20). `scripts/inversions/cs_4dvar.jl` reads `[observations].file`
  (schema v2) with `tracer` and `out_of_range_policy`, places its point
  records with `[observations.placement]` (the `[output.observations]`
  placement keys through the shared `_parse_vertical_placement`, plus
  `layer_height_temperature_kelvin`), and needs `[time].start` and the new
  `[meteo].source = "synthetic_hydrostatic"` (`interface_pressures_pa`,
  optional `specific_humidity` and `temperature_kelvin`; dry air mass
  `m = (1 − q)Δp A/g` per cell, no fluxes). Inline `[[observations.entries]]`
  stay. A 4D-Var on transport-binary meteorology is assembled with the Julia
  API.
- **Single-sweep 4D-Var gradient** (issue #20). `cs_surface_flux_4dvar` now
  computes the observation-term gradient with one taped forward run (to the
  last observation step) and one reverse pass, whatever the number of
  observations, as TM5-4DVAR does; before, every observation re-ran a taped
  forward run and its own reverse pass, so one gradient cost
  `O(N_obs)` adjoint runs. Each observation is a fixed sparse linear
  functional of the tracer mass after its step (`CSLinearFunctional`, weights
  from the replayed air mass: `1/max(m, eps)` for a layer, `1/max(Σm, eps)` for a column). Its
  weighted departure `(H(x) − y)/σ²` times those weights is added to the
  adjoint state at the end of its step, after convection and before the
  transposed convection: `λ_s = M_{s+1}ᵀ λ_{s+1} + Σ_{i: s_i = s} (H_i − y_i)/σ_i² w_i`.
  Window steps after the last observation get a zero gradient, as before.
  `cs_surface_flux_4dvar` gains `checkpoint` (`FullCheckpoint()`,
  `StrideCheckpoint(k)`, `RevolveCheckpoint()`) and `tape_path`; before, it
  supported only `FullCheckpoint` and no `tape_path`. Every tape recorder now
  ends each step with an end-of-step record (`_CSStepEndRecord`, no payload),
  where the reverse pass injects that step's forcing; `CSTapeByteEstimate`
  gains `step_records`, and `total_records` counts these records.
- `cs_surface_flux_jacobian` records the forward tape once and walks it once
  per objective (with `FullCheckpoint`, the default) instead of recording one
  tape per objective. Each walk is the one a separate
  `cs_surface_emission_footprint` call makes.
- Internally, `cs_surface_emission_footprint`,
  `cs_surface_emission_footprint_from_seed`, `cs_surface_flux_jacobian` and
  `cs_surface_flux_4dvar` share one checkpoint dispatch
  (`_cs_surface_footprints`, replacing the duplicated dispatch in
  `FootprintAPI.jl`); the Stride and Revolve drivers take an adjoint start
  (objective seed, explicit seed, or zero) and a step forcing.
  `_seed_objective!` zeroes the adjoint state and adds the objective's linear
  functional; the seeds are bit-identical, except that the column seed now
  carries the `eps` air-mass floor of the column objective's evaluation (it
  differs only for columns holding less than `eps` of air).
- **Time-varying surface fluxes must cover the run** (#59). A run that a
  time-varying source's files do not cover (before their first slice, after
  their last, or in a gap between files) now stops before it starts with an
  `ArgumentError` naming the uncovered dates. Before, the source silently held
  its first or last slice there: the three `catrine_c30_2021*` configs emitted
  the 2021-12-01 00:00 CAMS slice from January to November. A slice covers
  its CF time bounds (a `months since` stamp its month). Stamps without
  bounds are one series over all files of the source, merged and sorted
  (`GriddedIO.stamp_intervals`): each slice holds for the series' regular
  step (one month, or one year from 1 January, for month starts; else the
  smallest spacing), the last stamp for one step, and a slice before a
  longer spacing for one step, so a gap inside or between files (Jan 1 then
  Mar 1) is uncovered. Daily files with one 00Z stamp each are a daily
  series. A single stamp in all files (a day, a month or a year?) and stamps
  about a month or a year apart that are not month starts (mid-month) are an
  error asking for CF bounds. The new key
  `surface_flux.hold_outside_coverage = true` keeps the holding on purpose
  (e.g. climatologies). The CATRINE signoff chunk configs (635 time-varying
  sources) and those of `config/runs` and the golden configs whose files
  exist on wurst (179) keep their coverage.
- Removed `config/runs/catrine_c30_2021_fullphysics_experimental.toml`,
  `catrine_c30_2021_fullphysics_column_experimental.toml` and
  `catrine_c30_2021H1_fullphysics_column_experimental.toml`, with their
  launchers `scripts/run_catrine_c30_2021_fullphysics.sh`,
  `scripts/run_catrine_c30_2021H1_column.sh` and
  `scripts/profile_catrine_c30_2021_fullphysics.sh`: no January–November 2021
  CAMS files exist for them.
- `[output.observations]` takes `dem_file`, `dem_variable`,
  `unplaced_records`, `surface_height_file` and `absolute_altitude_above_agl_m`
  (`VerticalPlacementSpec`). Both ground datasets are external and static,
  read through `GriddedIO`: the DEM by `read_point_elevation` (ground or sea
  surface in m; a file below −450 m is rejected as bathymetry and one above
  9000 m as an undeclared fill; the axes are told apart by dimension name, so
  a `(lat, lon)`-stored DEM is read correctly; a gridline-registered global
  DEM, −180…180 inclusive, has its repeated seam column dropped), the per-grid
  model surface height by `read_model_surface_height` (written by
  `write_model_surface_height` and
  `scripts/preprocessing/build_model_surface_height.jl` from a met source's
  surface geopotential). `build_observation_sampler` takes
  `terrain = ObservationTerrain(...)` and keeps only its provenance.
- `[output.observations]` takes `vertical_coordinates` (issue #20): the
  vertical coordinates that may place a record, in preference order, from
  `"height_above_ground"` (`HeightAboveGroundCoordinate`) and `"pressure"`
  (`PressureCoordinate`), default `["height_above_ground", "pressure"]` (the
  previous rule, bit-identical). `["pressure", "height_above_ground"]` prefers
  a record's pressure; a single coordinate uses only that one, and a record
  that gives only the other one is unplaced (`unplaced_records`). An empty,
  repeating or unknown list is an error, as are `dem_file` and
  `absolute_altitude_above_agl_m` without `"height_above_ground"` and
  `pressure_basis` without `"pressure"`;
  `absolute_altitude_above_agl_m` re-places by pressure only when
  `"pressure"` is listed. And `pressure_basis` (`"total"` →
  `TotalFromHumidity`, default; `"dry"` → `DryInterfaces`), see Numerical
  changes. `VerticalPlacementSpec` gains `coordinates` and `pressure_basis`;
  `resolve_placement(spec::VerticalPlacementSpec, rule, request, cell, terrain)`
  replaces the `resolve_placement(altitude_placement, ...)` method; the shared
  parser `_parse_vertical_placement(cfg, label)` reads the placement keys
  (`_VERTICAL_PLACEMENT_KEYS`) of any table. `build_observation_sampler` takes
  `has_humidity` and `observe_window_boundary!` takes `humidity` (the specific
  humidity at the window end); new `total_interface_pressures!`,
  `placement_pressures!`, `check_pressure_basis`, `observation_needs_humidity`,
  the free layer rule `column_layer!` with its scratch `PlacementColumn`, and
  `MetDrivers.load_vdiff_qv_window!` (one window's VDIFF `qv`). Observation files add
  the attributes `vertical_coordinates` and `pressure_basis`, and the
  `vertical_placement_codes` text of code 2 names the pressure basis.
- Observation files add `altitude`, `pressure` (soundings),
  `vertical_placement`, `placement_height_agl` and `model_surface_height` per
  row or site, and the attributes `vertical_placement_codes`,
  `n_unplaced_soundings`, `n_unplaced_sites`, `unplaced_records`,
  `surface_height_file`, `dem_file`, `absolute_altitude_above_agl_m`.
- `SoundingRequest` gains `altitude_m` and `pressure_pa`, `SiteRequest`
  gains `altitude_m` (the old positional constructors remain);
  `SoundingBatch` takes `altitudes`, `pressures` and `placements`;
  `SiteNetCDFStream` takes the sites' placements. `intake_layer_index` throws
  on a `NaN` height instead of returning the lowest layer.
  `resolve_placement(option, missing_height_rule(source), request, cell, terrain)`
  places one request.

### Performance

- Cubed-sphere FV3 vertical sweep (`vertical = "fv3_kord8"`): the packed sweep
  launches several panels at once, up to all six, until a launch has about
  65 536 columns (`_fv3_sweep_panel_group`); it used to launch each panel
  separately, a C90 one-tracer launch running 8100 columns. Each column computes
  exactly what it did before, so results are bit-identical.
  `CSAdvectionWorkspace` takes `column_scratch_panels` (default from that rule;
  Lin-Rood workspaces use 1) and its column scratch holds one slot per tracer
  and panel of a launch.
- Cubed-sphere column operators launch once for all six panels instead of
  once per panel: the conservative Dkg diffusion solve (also the GEOS-Chem
  non-local VDIFF; one synchronization per call instead of six), the surface
  flux (static and time-varying sources, one launch per source and time
  segment), and the CMFMC convection substep. Each column runs the same
  arithmetic, so results are bit-identical.
- Multi-binary cubed-sphere runs keep their window storage across binaries:
  `DrivenSimulation(...; reuse_buffers_of = previous)` refills the previous
  simulation's device windows, host staging window, expected-air-mass and
  humidity buffers and convection forcing when the new driver has the same
  window layout (`MetDrivers.same_window_layout`), and skips the startup
  garbage collection and device-pool trim. The cubed-sphere runner passes the
  previous binary's simulation. Each day boundary used to allocate and zero
  new host and device windows and run a full collection (about 0.35 s per day
  at C90). Window contents are unchanged.
- The host staging window of a device run is page-locked
  (`Architectures.pin_host_memory!`; CUDA only), so each window upload is a
  direct DMA transfer (24 GB/s instead of 12 GB/s on an L40S, without
  occupying the prefetch thread). The prefetch waits for the upload on its own
  stream (`Architectures.synchronize_storage!`) before handing the window over;
  before, an upload staged through the driver's pageable path could in
  principle still be in flight when the stepping task used the window.
- `DrivenSimulation` copies window-constant fluxes (and scales them for
  `full_window_mass_amount` binaries) and the convection forcing into the
  model at the first substep of each window instead of every substep; the
  operators only read them. Interpolated fluxes, the expected air mass and
  humidity still change every substep. A C90 L72 step issued 36 fewer
  device copies (about 0.3 ms of a 4.4 ms one-tracer step).
- Cubed-sphere seam transfers: the transfers of a directional group are
  cached in one launch of `SeamTransferCache` point operations (`Fused`,
  twelve seams with the other group's inactive) instead of one launch per
  seam, and neither the cache nor the seam application synchronizes the host
  on a device (later work on the device queue sees their cells in order). A
  C90 step issues 20 fewer launches and 8 fewer synchronizations; results are
  bit-identical.
- Cubed-sphere transport binaries whose on-disk element type is the run's
  load windows on a GPU through raw window staging
  (`MetDrivers.RawWindowStaging`): four concurrent `pread`s of the window's
  contiguous payload into a page-locked host buffer (libuv thread pool; one
  `read!` off Unix), then per payload section an upload into a device buffer
  the size of the largest section and one unpacking launch (padded fields get
  zero halo cells, as before). This replaces a page-faulting copy of the file
  mapping into a padded host window (31 ms per C90 L72 window, row by row for
  padded fields) and the copy of that window to the device; the device windows
  are unchanged, and the device holds one extra section-sized buffer (14 MB at
  C90 L72). Binaries of another element type keep the host staging window.
  `same_window_layout` also compares the on-disk element type, and the
  cubed-sphere runner releases the previous binary's buffers before building
  new ones when the layouts differ.
- The packed cubed-sphere palindrome (`strang_split_cs_mt!`) queues its halo
  exchanges (`fill_panel_halos!(…; sync = false)`, new keyword) and
  synchronizes the host once at its end instead of after each of its 16
  exchanges per step; with the surface flux and the diffusion solve a C90 step
  now synchronizes 3 times. A window prefetch first waits for the stepping
  task's queued work, which may still read the window it refills. Results
  are bit-identical.
- `expected_air_mass!` on cubed-sphere windows with flux deltas computes all
  six panels in one launch (`ExpectedAirMass` point operations) instead of six
  copies and six broadcasts per substep.

### Other changes

- `merge_thin_levels` (`[vertical] transform = "merge_layers_thinner_than"`,
  spectral `merge_min_thickness_Pa`): its top-down pass could never change the
  result and is removed; the produced level sets are bit-identical (for
  level sets without a zero-thickness layer, which includes all ERA5 sets). The
  documentation now states the rule: groups close from the surface up, and
  the native layers left at the top form a top layer that can be thinner than
  the minimum (6.8 Pa for the L117 set at 100 Pa; folding it into an L116 set
  waits for the next regeneration).

## 0.5.0 — 2026-10-13

This is a breaking release. It changes numerical results in preprocessing and
at run time, removes configuration keys, environment variables and some Julia
API, and tightens the transport-binary header contract. Regenerate transport
binaries to get the corrected physics, and keep the version, forcing binary,
run TOML and dependency manifest with reproducibility records; pin 0.4.0 to
reproduce older runs. Most of the physics changes answer an external review
against the TM5 and IFS Fortran by pera-noaa (NOAA GML), GitHub issues #10–#20.
Thank you.

### Summary

- **One dry-air mass target** of 98,325 Pa for every met source with the
  default `[mass_fix] mode = "target_ps_dry"` (was 98,726 Pa, 0.41 % too
  much dry air; #14, PR #45); GEOS `mode = "initial_endpoint"` runs still pin
  to their first endpoint. Native pins use the ΔB(1−q) shape.
- **TM5 convection on the dry basis** (#19), **TM5 `echlevs` numbering
  fixed** (#18) and the **unpaired Nyquist harmonic dropped** on even rings
  (#13), all in PR #41. **TM5 level merging** (`n_merge > 1`) no longer
  amplifies near-cancelling mixed-sign layer pairs into dipoles with negative
  mixing ratios (OF-107, PR #51); the default `n_merge = 1` is unaffected.
- **GEOS-Chem non-local VDIFF emission deposit fixed** (OF-108, PR #53): a
  roundoff-triggered `qmincg` fallback had kept about 94 % of fresh emissions
  of zero-background tracers (Rn-222, fossil CO₂) in the surface layer
  instead of the boundary-layer profile. Rerun MERRA-2 runs that use
  `[diffusion] kind = "geoschem_nonlocal_vdiff"`.
- **Column-only Poisson balance**, which fixes the reduced-Gaussian sign error
  (#11, PR #36).
- **Per-cell replay gate** with tighter tolerances (#15, PR #39), and
  **write-time flux-fidelity certificates** in every binary the
  preprocessors write (schema 2: horizontal re-derivation; OMEGA comparison with NRMSE,
  correlation and a two-sided slope bound), plus a **checked per-cell column
  closure** recorded by the cubed-sphere column balances (#15 item 3,
  PR #49).
- **Conservative hybrid-coordinate remap** for file initial conditions
  (PR #45) and for `cs_native` initial conditions (PR #46).
- **GCHP-matched preprocessing defaults** (PR #49): vector/FV3 face fluxes
  with true edge lengths and Cartesian wind regridding, ERA5 `window_mean`
  sampling, `hybrid_mass` column weights on every target, dry N320 flux
  thickness, and one target-sphere radius (`EARTH_RADIUS`).
- **Runtime fidelity** (PR #46): symmetric Rn-222 decay; configured Lin–Rood
  defaults to the `fv3_kord8` vertical remap; surface-flux and IC source
  meshes from snapped file-coordinate faces; per-month fluxes decoded from CF
  time; outputs record the physics that ran, with CF time units.
- **Adjoint and inversion:** the Lin–Rood adjoint Z-sweep fix (#10, PR #37);
  the Lin–Rood adjoint now also transposes the FV3 `kord = 8` vertical
  sweeps, so footprints and 4D-Var use the forward default (OF-94, PR #50);
  the 4D-Var driver honours `[advection]` and L-BFGS evaluates each point once
  (#16, #17, PR #35).
- **Settings move from environment variables to typed TOML keys** (PR #25).
  Configs are validated strictly before any binary opens (PRs #24, #26).
- **Performance:** surface-flux series stay on the host (device memory
  18.9 → 2.9 GB for a ten-year C90 Float64 run, PR #42); CUDA halo exchange in
  2 launches instead of 30 and in-place window refills (PR #24); ERA5 N320
  preprocessing about 5× faster (PR #22).
- **New tooling:** the `Analysis` module (PR #38), the `GriddedIO` module
  (PR #33), observation sampling (PR #8), lab GPU benchmarks and Codecov
  (PR #30), golden sessions (PR #31), test tiers and a quick gate (PR #47),
  and the former orphan tests in CI (PR #48).

**Regenerate your transport binaries** to get the corrected physics.
Multi-file runs reject a mix of old (98,726 Pa) and new (98,325 Pa) dry-mass
targets.
Binaries without the new `surface_pressure_basis` key, and dry binaries with
TM5 convection but no `tm5_convection_mass_basis`, still open with a one-time
warning; these warnings become errors in the next minor release.

### Breaking changes

The details, with measurements, are in the sections further down.

#### Configuration (TOML)

| Removed or changed | Replacement / new rule | PR |
|---|---|---|
| `[mass_fix] target_ps_dry_pa`, `target_total_kg` | Removed and rejected. One target, `Parameters.GLOBAL_MEAN_DRY_SURFACE_PRESSURE = 98325.0` Pa. | #45 |
| `[mass_fix] mode = "initial_endpoint"` on MERRA-2 / ERA5 N320 | Refused (only GEOS native implements it; before, it silently skipped the pin). | #22 |
| `[numerics] geos_balance_mode` (and the per-layer balance) | Removed; a config that sets it is an error. The column balance is the only balance. | #36 |
| `[run] scheme`, `ppm_order`, `vertical`, `limiter` | Read only from `[advection]`; in `[run]` they are an error that names the new table. | #26 |
| `surface_flux.molar_mass_kg_mol` | `surface_flux.species = "co2" \| "sf6" \| "rn222"`; the molar mass comes from `Parameters.SPECIES_MOLAR_MASS`. Tracers named `co2`, `co2_natural`, `co2_fossil`, `fossil_co2`, `sf6`, `rn222` may omit it. | #33 |
| Unknown `surface_flux.kind` (was read as a generic file) | Error with a suggestion; use `kind = "file"` for a generic NetCDF file (`"eccodarwin_ocean_co2"` → `"file"`). | #24 |
| Time-varying surface fluxes without `[input] start_date` | `start_date` is required (the time origin came from the file). | #26 |
| Surface-flux units | One unit grammar (`GriddedIO.parse_flux_units`); missing or unknown units are errors; kg of another named species than the tracer's is an error (it was read with factor 1). | #26, #33 |
| Per-month surface-flux slices | Month lengths come from the file's CF time coordinate, bounds and calendar; a static `month` selects by decoded month; see *Surface fluxes and preprocessing*. | #46 |
| `cs_native` flux files | Units kg m⁻² s⁻¹ of the species (or kgC, converted), no fill values, dimensions exactly `(time, nf, Ydim, Xdim)`, strictly increasing time; the mesh must match when the file carries `cell_area`. | #8, #26, #33 |
| ERA5 L137 level presets `ml137_94L`, `ml137_85L` | Removed; use `ml137_cfl94`, `ml137_cfl85`. `echlevs` now uses TM5/ECMWF half-level numbering; `ml137_tropo34` is the TM5-MP r1112 list, new `ml137_tropo34a` the cy3-4dvar/CAMS73 list. | #41 |
| ERA5 source `bottom_up` level orientation | Rejected (it was accepted but ignored). | #41 |
| `[vertical].level_selection` on sources without 137 native levels | Rejected; `[vertical].coefficients_file` is accepted as an alias of `coefficients`. | #8 |
| `[architecture] backend` aliases `host`, `nvidia`, `apple`, `apple_metal`, `gpu`, and other case or separator spellings | Rejected, naming the canonical value (`cpu`, `cuda`, `metal`, `auto`). | #44 |
| Unknown tables and keys in a run config (ignored before) | Error with a "did you mean" suggestion; known keys that the chosen kind leaves unread are warnings. | #24, #26 |
| `[advection] prognostic_slopes` | Was ignored; now an error (unknown key), removed from the ten shipped configs that set it. | #26, #43 |
| `start_window` / `stop_window` beyond an input's window count | Error (`stop_window` was clamped). Lat-lon and reduced-Gaussian initial conditions use the air mass of `start_window`. | #43 |
| Snapshot hours that are not met-window ends | Error at start (such snapshots, and every later one, were silently lost). `format = "binary_mmap"` on lat-lon / reduced-Gaussian grids is rejected before the run. | #24 |
| Snapshot output without an absolute time origin | A run needs `[input] start_date`, a binary `date` header, or a `YYYYMMDD` date in the first binary's file name; where several are present they must agree. | #46 |
| File ICs without hybrid coefficients | Must match the model's level count and state their order (pressure or height units, CF `positive`, or `init.vertical_order`). The file's `Psurf` is no longer read. | #26, #45 |
| `cs_native` ICs whose level count differs from the binary | Need source `ap`/`bp` (conservative hybrid remap); optional `init.humidity_variable`. | #46 |
| Lat-lon meshes | Must span 360° of longitude. | #43 |
| `cmfmc_matrix` convection | Every binary must carry `cmfmc` and `dtrain`. | #43 |
| 4D-Var driver `scripts/inversions/cs_4dvar.jl` | Requires an `[advection]` table with `scheme`; unknown tables and keys are errors, including `[diffusion]`, `[convection]` and `[output]`; schemes without a cubed-sphere adjoint are rejected (`require_adjoint_support`). Lin–Rood defaults to `vertical = "fv3_kord8"`, as in forward runs. | #35, #43, #50 |
| Default `column_balance_weights` | `"hybrid_mass"` (was `"mass"`) on every target: cubed sphere, lat-lon and reduced Gaussian. `[numerics] column_balance_weights` is now accepted for lat-lon and reduced-Gaussian targets. The GEOS `pressure_fixer` / `pfix_corrected` closures keep the native fluxes and ignore the weights. A binary whose recorded weights differ is regenerated instead of reused. | #49 |
| Wind-derived cubed-sphere flux construction | Defaults to `face_fluxes = "vector"`, `face_lengths = "edge"`, `face_interpolation = "fv3"`, `wind_regrid = "cartesian"`, and (ERA5) `flux_time_sampling = "window_mean"`; ERA5 N320 `flux_thickness = "dry_mass"`. The historical choices stay available explicitly. | #49 |
| Met-source descriptors `merra2_geoschem_hm_gchp.toml`, `merra2_geoschem_hm_vs.toml`, `merra2_geoschem_hybridmass.toml` | Removed (equal to the new defaults); use `config/met_sources/merra2_geoschem.toml`. | #49 |
| Missing next-day endpoint in native preprocessing | The zero-tendency fallback applies only to the configured last day of a range; any other missing endpoint is an error. | #49 |
| Configured `scheme = "linrood"` | Defaults to `vertical = "fv3_kord8"` (was upwind); `vertical = "upwind"` keeps the old behavior. | #46 |

`validate_config` (run at the start of every run) parses `[advection]`,
`[diffusion]`, `[convection]`, `[chemistry]`, `air_mass_reset_mode`,
`physics_cadence` and `[output]` before any binary opens (PR #24).

#### Environment variables → typed config (PR #25)

These are removed without a transition period; the package ignores them.

| Removed variable | Use instead |
|---|---|
| `ATMOSTR_FORCE_PER_SUBSTEP_PHYSICS` | `[run] physics_cadence = "substep"` |
| `ATMOSTR_REPLAY_CHECK` | `[input] validate_replay = true` |
| `ATMOSTR_NO_REPLAY_CHECK` | nothing (the load-time check runs only when asked for) |
| `ATMOSTR_ENABLE_HORIZONTAL_POISSON_BALANCE` | nothing (the per-layer balance is removed) |
| `ATMOSTR_DISABLE_PREFETCH` | `[run] prefetch_windows = false` |
| `ATMOSTR_ASSERT_CS_BINARY_CFL` | `[advection] check_binary_cfl = true` |
| `ATMOSTR_NO_WRITE_REPLAY_CHECK` | `[numerics] write_replay_check = false` (preprocessing config); `--no-write-replay-check` of `regrid_ll_transport_binary_to_cs.jl` |

Environment variables now set only paths, metadata, profiling and
diagnostics; `docs/src/config/environment.md` lists them, and a test enforces
the list.

#### Julia API

Removed:
- `State.MetState`, `diagnose_cm_from_continuity_vc!` and
  `diagnose_cm_from_continuity_ka!` (call `diagnose_cm_from_continuity!` with
  explicit `Δb`), and `Preprocessing.reset_workspace!` (#43).
- `FileInitialConditionSource.psurf` (#45).
- The GEOS writer's `process_day(...; balance_mode)` keyword (#36).

Changed signatures and required keywords:
- `write_transport_binary(...)` requires `surface_pressure_basis =
  TotalSurfacePressure()` or `DrySurfacePressure()` (#45).
- `open_streaming_transport_binary` and `open_streaming_cs_transport_binary`
  require both `mass_basis` and `surface_pressure_basis` (#41, #45).
- The direct preprocessing writers (`process_day`, `regrid_ll_binary_to_cs`,
  `process_merra2_to_cs_day`, `process_era5_n320_to_cs_day`) take a
  `write_replay_check` keyword in place of the environment variable (#25).
- The replay-check API takes a `ReplayScratch` instead of a divergence array:
  `verify_window_continuity(layout, scratch, ...)`,
  `verify_window_continuity_rg(...; scratch)`, and `replay_scratch` in place
  of `div_scratch` in `verify_{ll,rg,cs}_window_contract!` and
  `verify_write_replay_cs!` (#39).
- `build_regridder` returns `NestedCubedSphereRegrid{Restriction|Prolongation}`
  for cubed spheres of one definition whose `Nc` differ by an integer factor;
  code that read `intersections` must use `src_areas`/`dst_areas` and
  `apply_regridder!` (#29).
- `apply_regridder!(dst, r, src, kind)` takes a quantity kind,
  `IntensiveCellField()` or `ExtensiveCellField()` (#29).
- `Visualization.open_snapshot` and `Analysis.daily_snapshot_series` decode
  CF time for current files (#46).
- `CSAdjointWorkspace` takes `column_scratch = true` for tapes with FV3
  vertical sweeps (#50); `cs_surface_flux_4dvar` requires the `scheme`
  keyword (#43).
- Symbol selectors are types (#44): `fieldview` takes `transform =
  ColumnMean()`, `ColumnSum()`, `LevelSlice()` or `SurfaceSlice()` and
  `unit = NativeUnit()` or `PPMUnit()`; `AllFrames()` replaces `:all`;
  `bind_to_mesh(...; out_of_range_policy)` takes `RejectOutOfRange()`,
  `SkipOutOfRange()`, `ClampOutOfRange()` or the strings; and
  `RuntimeOutputSpec.format` is `NetCDFOutput()` or `BinaryMmapOutput()`
  (its six-argument constructor is removed).
- The TM5 convection basis is the writer keyword `tm5_convection_mass_basis`;
  `extra_header` may set neither it nor `surface_pressure_basis` (#41, #45).
- Preprocessing certificate and column-closure internals changed (#49); see
  *API and configuration changes in detail*.

New top-level exports (13; see `docs/src/api/public_api.md`):
`Callback`, `IterationInterval`, `WindowInterval`, `WallTimeInterval`,
`NaNChecker` (#26); `AbstractSurfacePressureBasis`, `TotalSurfacePressure`,
`DrySurfacePressure`, `surface_pressure_basis` (#45); `AbstractCloudBase`,
`CMFMCEdgeCloudBase`, `ArchivedCloudBase` (#22); `observation_output_spec`
(#8). No top-level export was removed. New modules: `AtmosTransport.Analysis`
(#38), `AtmosTransport.GriddedIO` (#33), `ConfigChecks` (#24) and
`AtomicFiles` (#29).

#### Transport-binary header contract

`format_version` stays 4, and every current binary opens. New writers record
these keys, and readers act on them:

| Header key | Written by | Reader behavior for older binaries | PR |
|---|---|---|---|
| `surface_pressure_basis` (`"total"` / `"dry"`) | every LL/RG/CS writer, eager and streaming; preserved by transformations | Missing: one warning per session, basis inferred from window-1 `ps` and air mass. Becomes an error in the next minor release. LL→CS regridding rejects dry-`ps` sources. | #45 |
| `tm5_convection_mass_basis` | every writer with TM5 sections | Missing on a dry binary with TM5 sections: one warning (moist fluxes inflate convective rates by 1/(1−q), about 0.7–1.1 % globally). Becomes an error in the next minor release. A recorded mismatch is already an error. | #41 |
| `mass_fix_target_ps_dry_pa`, `mass_fix_enabled` | pinned writers | Missing: target inferred from window-1 dry mass. Multi-file runs reject mixed targets and mixed pinned/unpinned files. | #45 |
| `write_replay_check = false` | only when the write-time gate was skipped | The driver warns; `inspect_binary` marks it. | #24, #25 |
| `horizontal_balance` (`"column"` / `"none"`) | every writer | informational | #36 |
| `column_balance_weights` | MERRA-2 and ERA5 N320 always; other balancing paths when not `"mass"` | keyless means `"mass"` | #26, #49 |
| `planet_radius_m` | every writer; targets use `EARTH_RADIUS` = 6,371,000 m (was 6,371,229 m) | keyless means `EARTH_RADIUS` | #49 |
| flux-construction keys (`face_fluxes`, `face_lengths`, `face_interpolation`, `wind_regrid`, `flux_thickness`, `flux_time_sampling`) | effective choices | informational | #22, #49 |
| flux-fidelity certificate (`flux_fidelity_schema = 2`) | every preprocessing writer (not the generic `write_transport_binary`) | Pre-certificate binaries read as `FidelityUnavailable` (`legacy_header_missing`); any other schema is an error ("regenerate this binary"). A binary without the current schema is regenerated, not reused, by the preprocessor. | #49 |
| column closure (`column_closure_*`) | cubed-sphere column balances | Binaries without the keys load as "not recorded"; an incomplete record, or one that exceeds its tolerance, is refused. | #49 |
| `git_commit`, `git_dirty` | cubed-sphere writers (`src/REVISION` for `git archive` trees) | provenance | #22 |

#### Scripts and files

- 161 stale scripts were removed and 58 moved to `heritage/` folders (PR #24);
  recover a removed script with `git show 7c515038:<path>`. The
  `scripts/deprecated/` runner shims are gone; use
  `scripts/run_transport.jl`. `scripts/README.md` lists every removed script.
- Python diagnostics are replaced by Julia (PR #38):
  `check_trendy_npp_rh_run.py` → `tracer_budget_closure.jl series`,
  `catrine_true_mass_balance_gc.py` →
  `catrine_true_mass_balance.jl --model geoschem`, and the Float32 attribution
  and Float32-vs-Float64 scripts; `catrine_station_timeseries.py` moved to
  `heritage/`.
- Retired configs: `config/runs/binary_format_ab/c45_*` (#24), the per-layer
  GEOS twin `geosit_c180_dec2021_catrine_f32_reclosed.toml` and its two run
  configs (#36), and one L137 TM5 config moved to `likely_legacy` (#43).
- The `test/orphan/` tier and `--orphan` flag (#48) and `test/archived/`
  (#47) are gone.
- Benchmarks workflow: the self-hosted CUDA/Metal jobs and the inputs
  `run_cuda`, `run_metal`, `gpu_float_types` are removed; GPU benchmarks run
  in the lab lane (`benchmarking/lab/`, #30).

### Migration guide

1. **Regenerate transport binaries.**

   | Binary set | Why | Action |
   |---|---|---|
   | Every dry-pinned binary written before PR #45 (target 98,726–98,733 Pa) | 0.41 % excess dry air; runs cannot mix old and new targets | Regenerate. Until then, run only complete old-target spans. |
   | ERA5 binaries with TM5 convection (spectral ec2tm, N320→CS + TM5, CATRINE attach) before PR #41 | moist TM5 fluxes on dry air (+0.7–1.1 % rates) | Regenerate; check `tm5_convection_mass_basis = dry` with `inspect_transport_binary.jl`. |
   | `ml137_tropo34` binaries | mirrored TM5 level set (#18) | Regenerate (`ml137_tropo34` = TM5-MP r1112 or `ml137_tropo34a` = cy3-4dvar). |
   | CATRINE TM5-attached sets without the `_v3` tag | convection upside down (PR #9) | Regenerate with the current attach script. |
   | ERA5 N320 binaries before PR #22 | half-cell humidity/convection shift; polar under-coverage | Regenerate. |
   | Reduced-Gaussian (O-grid) binaries before PR #36 | wrong-sign per-layer balance | Regenerate. |
   | Float32-preprocessed C180 and finer | Float32 mesh geometry (PR #22) | Regenerate. |
   | Binaries without `surface_pressure_basis` | warn now; error in the next minor release | Regenerate. |
   | Wind-derived binaries before PR #49 | new flux-construction, `hybrid_mass`, dry N320 thickness, target radius and ERA5 surface-field defaults; no fidelity certificate | Regenerate to get the defaults (old binaries stay readable). |
   | Python-built flux drivers (TRENDY, FLUXCOM-X, SIF-GPP, GFED5) | carbon molar mass (+0.068 %) | Regenerate. |

   Inspect a binary with
   `julia --project=. scripts/diagnostics/inspect_transport_binary.jl <file.bin>`:
   it reports the mass basis, `surface_pressure_basis`, the dry-mass target
   and whether it is current, the TM5 basis and `write_replay_check`.

2. **Edit configurations.** Run the config once: `validate_config` errors
   name the replacement key. Then:
   - delete `[mass_fix] target_ps_dry_pa` / `target_total_kg` and
     `[numerics] geos_balance_mode`;
   - move `scheme`, `ppm_order`, `vertical`, `limiter` from `[run]` to
     `[advection]`;
   - replace `molar_mass_kg_mol` by `species` (or omit it for the standard
     tracer names);
   - set `[input] start_date` for time-varying fluxes and for snapshot output
     from undated binaries;
   - use `kind = "file"` for generic flux files and fix typos in
     `surface_flux.kind`;
   - rename `ml137_94L` → `ml137_cfl94` and `ml137_85L` → `ml137_cfl85`;
   - use the canonical `[architecture] backend` values `cpu`, `cuda`,
     `metal`, `auto`;
   - replace environment switches by the keys in the table above;
   - remove `prognostic_slopes`, and make snapshot hours met-window ends;
   - replace `merra2_geoschem_hm_gchp.toml`, `_hm_vs.toml` and
     `_hybridmass.toml` by `merra2_geoschem.toml`;
   - to keep the pre-0.5 flux construction for a wind-derived source, set it
     explicitly (for example `face_fluxes = "panel_average"`,
     `face_lengths = "cell_centerline"`, `face_interpolation = "linear"`,
     `wind_regrid = "scalar"`, `flux_time_sampling = "window_start"`,
     `column_balance_weights = "mass"`, and for ERA5 N320
     `flux_thickness = "moist"`);
   - forward Lin–Rood runs (and Lin–Rood inversions) that must reproduce
     v0.4 vertical transport: set `[advection] vertical = "upwind"`.
3. **Update Julia code.** Pass `surface_pressure_basis` (and `mass_basis` for
   the streaming writers) when writing binaries; replace `div_scratch` by
   `ReplayScratch` and `MetState` by explicit fields; use the `kind` argument
   of `apply_regridder!`; pass `write_replay_check = false` to direct writer
   calls instead of setting `ATMOSTR_NO_WRITE_REPLAY_CHECK`.
4. **Output consumers.** NetCDF `time` is CF-encoded (`hours since <run
   start>`), so NCDatasets returns `DateTime`; read `ds["time"].var[:]` for
   elapsed hours. Outputs carry `physics_*` global attributes.
5. **Reproducing v0.4 results.** Pin `AtmosTransport = "=0.4.0"` with the
   v0.4 binaries, run TOML and Manifest. v0.5.0 cannot reproduce v0.4
   numerics bit for bit: the target, conventions and gates changed.

### Known limitations

- The production binary archive is being regenerated, so the legacy-binary
  warnings (`surface_pressure_basis`, `tm5_convection_mass_basis`) remain
  warnings in 0.5.0; they become errors in the next minor release.
- The OMEGA certificate thresholds are calibrated on MERRA-2 C90 only
  (NRMSE 0.47, correlation 0.89, slope 0.87); no golden covers the GEOS OMEGA
  closures (`omega_regularized`, `omega_full_replacement`), so their baseline
  is unmeasured. Coarse unpinned Float32 builds can approach the `cm`
  closure limit (unpinned GEOS-IT C45 golden day: ratio 0.924).
- A run restarted from the previous chunk's final state differs from a
  continuous run in the GEOS-Chem non-local VDIFF path: near-surface Rn-222
  by up to 0.5 % of the field maximum in about 0.2 % of cells, decaying
  within a day (MERRA-2, two days, CPU); with diffusion off the difference is
  about 1500 times smaller. The cause is not yet identified.
- Adjoints: the split-sweep `PPMScheme` with the FV3 vertical profile has no
  adjoint (Lin–Rood with `fv3_kord8` has one). Adjoints, including the new
  FV3 transpose, are untested on Metal. Split-sweep PPM cubed-sphere
  footprints are not bit-reproducible with two or more threads (last-bit
  differences; Lin–Rood is deterministic). `cs_tape_byte_estimate`
  undercounts the Lin–Rood tape by four face-panel tuples per step. Atomic
  publication of memory-mapped tapes is not implemented. 4D-Var cost grows
  with the number of observations, and observations bind to column means
  (#20). TM5 prognostic slopes are not implemented (#12).
- TM5 level merging (`n_merge > 1`) is not additive in the tracer: tagged-
  tracer budgets, anomaly transport and adjoints need `n_merge = 1` (the
  convection adjoint rejects `n_merge ≠ 1`).
- Lin–Rood transport is far more sensitive to arithmetic precision than PPM.
  On the C24 golden blob day (maximum absolute difference of the blob field,
  normalized by the blob maximum), default Lin–Rood (PPM5, `fv3_kord8`
  vertical) differs by 2.0e-3 between CPU Float32 and CPU Float64, and by
  2.2e-3 (CUDA) and 2.3e-3 (Metal) between GPU Float32 and CPU Float32; split
  PPM with `fv3_kord8` differs by 7.6e-6 between CPU Float32 and Float64. The
  difference grows through the day and sits in undershoot cells near the model
  top; it is under investigation. Use Float64 or PPM where this matters.
- Metal runs Float32 only. A time-varying surface-flux source built in
  Julia with host `SubArray` series cannot be staged on Metal (Metal.jl 1.11
  has no offset `copyto!` from a host view, so it fails with a
  scalar-indexing error); the configured runner builds dense arrays.
- The experimental nested coarsener carries a `# HACK:` include
  (`src/Preprocessing/transport_binary/cubed_sphere_coarsen.jl`), and the
  `cs_native` cell-area gate is still norm-based
  (`src/GriddedIO/cs_native_flux.jl`, `TODO(P11)`).
- Per-Strang adaptive subcycling is not implemented, and the MERRA-2 OPeNDAP
  download recipe refuses to start.
- `ConservativeRegridding = "=0.2.0"` and `GeometryOps = "=0.1.38"` stay
  pinned exactly; newer versions removed an API this package uses.
- Golden outputs do not cover Float64 on GPU, Metal, observation sampling,
  adjoints or the TM5 attachment; the release procedure checks Float64 on
  GPU and Metal separately.
- Julia 1.10 remains supported (`julia = "1.10"`); CI runs the full suite on
  Julia 1.10 and the current 1.x release.

### Numerical changes

- Column balances use `hybrid_mass` weights by default on every target
  (cubed sphere, and now lat-lon and reduced Gaussian): the correction of a
  column's flux budget goes by air mass to hybrid layers only, and the
  column's remaining `cm` residual is folded with the same weights, so
  pure-pressure (stratospheric) layers receive none of it. A layer is hybrid
  when `ΔB_k ≥ 1e-6` (`HYBRID_LAYER_MIN_DB`; its thickness changes by at
  least 0.1 Pa when `p_s` changes by 1000 hPa): GEOS/MERRA-2 L72 layer 41
  (150–177 hPa) has `ΔB = 8.2e-9` (8.2e-4 Pa per 1000 hPa), a rounding
  residue of the table, and now counts as pure pressure (with `ΔB > 0` it
  took about 3 % of every correction at full mass weight); the smallest
  genuine increment of the shipped tables is ERA5 L137 layer 55,
  `ΔB = 7e-6` (0.7 Pa per 1000 hPa). The change is large in
  the stratosphere:
  - GEOS-IT C45 (`pre_geosit_c45`, `hybrid_mass` against `mass`, measured
    with the `ΔB > 0` mask):
    stratospheric RMS `cm` falls by 25–43 % between 5 and 130 hPa (0.57× at
    48 hPa); horizontal fluxes on faces above 10 % of the global maximum
    change by up to 17.8 % (p99 2 %; flux-weighted 1–2.7 % in the
    pure-pressure layers 1–40, 0.25–0.7 % in hybrid layers); tropospheric
    `cm` changes by less than 0.3 %. ERA5 spectral C24 (`pre_c24`):
    stratospheric RMS `cm` 0.65–0.72×.
  - The `ΔB` threshold alone: `pre_geosit_c45` layer-41 fluxes change by
    1.8–1.9 % RMS and `cm` at 177–288 hPa by 12 % to 5 % RMS;
    `pre_merra2_c90` by 0.25 % and 3.5 % to 1.4 %, and its OMEGA NRMSE
    improves from 0.4725 to 0.4712 (300–100 hPa: 0.302 to 0.297). ERA5 cases
    are bit-identical.
  - Lat-lon and reduced Gaussian (`hybrid_mass` against `mass`):
    `pre_ll72` stratospheric RMS `cm` 0.60–0.69× at 10–100 hPa and
    0.72–0.93× at 100–200 hPa, horizontal fluxes up to 11 % RMS per level
    (meridional, 28 hPa); `pre_o24` 0.70–0.83× and 0.83–0.96×, horizontal
    fluxes up to 1 % RMS per level.
  - Air masses, `ps` and `dm` are bit-identical in every case. Run a
    stratospheric tracer A/B before regenerating science sets.
- Wind-derived cubed-sphere preprocessing defaults to the GCHP-matched
  construction: Cartesian wind regridding, vector face fluxes with FV3
  interpolation and true edge lengths, and (ERA5) trapezoidal `window_mean`
  endpoint sampling. The ERA5 spectral path uses the same typed
  implementations as MERRA-2 and ERA5 N320 and supports source-grid line
  integrals. On `pre_c24` `am`, `bm` and `cm` change (largest `cm` change
  5.0e12 kg against a maximum of 7.4e12 kg); `m`, `ps` and `dm` are
  bit-identical. The historical construction remains available as
  `panel_average`/`cell_centerline`/`linear`/`scalar`/`window_start`.
- ERA5 N320 face fluxes use the dry pressure thickness by default
  (`flux_thickness = "dry_mass"`, `Δp_dry = (1 − q)|ΔA + ΔB p_s| = g m_dry/A`),
  consistent with the dry air masses; `"moist"` reproduces older files. On the
  L117 golden day only `am`, `bm` and `cm` change (largest 5.4e10 kg against
  3.3e12 kg); one window takes 13 instead of 14 substeps.
- Every preprocessing target mesh (lat-lon, reduced Gaussian, cubed sphere)
  uses `Parameters.EARTH_RADIUS` (6.371e6 m) and records it as
  `planet_radius_m`, so cell areas and air masses scale by
  `(6371000/6371229)² − 1 = −7.19e-5`. The IFS radius stays only in the
  source-side ERA5 spectral and N320 calculations.
- Reduced-Gaussian preprocessing now uses the same column-integrated Poisson
  balance as the lat-lon and cubed-sphere paths: one ring-graph solve per
  column, with each face correction spread over the layers by air mass. On the
  O24 golden day, equivalent omega at the 418 hPa interface is restored from
  6.9e-6 to 0.146 Pa/s RMS while every stored air-mass value and the global
  dry-air total remain bit-identical. Horizontal fluxes change by at most 6.7%
  of the previous global maximum, as expected from the reviewed prototype.
  Reported by pera-noaa (NOAA GML) in issue #11.
- Lat-lon preprocessing uses the cubed-sphere positivity gate (owner's
  decision, 2026-10-10): the palindrome outflow `2 (out_x + out_y + out_z)`
  against `min(m, m_next)`, one shared implementation, instead of each
  direction's single application against `m`. Adaptive schedules (the
  default) choose more substeps on lat-lon binaries: on the ERA5 72×37
  lat-lon golden day (2021-12-01) two of 24 windows go from 4 to 5 substeps. A fixed
  schedule that passed the old gate can now fail it. Reduced-Gaussian
  binaries keep their per-direction gate.
- Carbon-mass fluxes (`kgC m-2 s-1`, the CAMS/LMDZ inventories) become CO₂ by
  the molar-mass ratio `M(CO2)/M(C)` = 44.0095/12.0107 ≈ 3.66419
  (`CARBON_MOLAR_MASS`, the IUPAC value consistent with `SPECIES_MOLAR_MASS.co2`),
  which conserves carbon atoms, instead of 44/12. These fluxes are 0.068 % lower
  than before; GEOS-Chem's HEMCO uses `CTOCO2 = 3.6667` (44/12).
  The Python flux preprocessors (TRENDY, FLUXCOM-X, SIF-GPP, GFED5) use the
  same ratio; drivers they wrote before are 0.068 % higher until regenerated.
- New cubed-sphere option `[advection] scheme = "ppm"`, `vertical = "fv3_kord8"`.
  The vertical sweep integrates FV3's `scalar_profile` parabola (`kord = 8`,
  positive definite; GCHP's tracer profile) over the swept fraction of the
  donor layer (`PPMScheme(; vertical = FV3ScalarProfile())`). Use
  `"fv3_kord8_signed"` for tracers that become negative.
  - Edges and limited parabolas match a transcription of `fv_mapz.F90`
    bit for bit.
  - In 1-D translation tests it is third order or better and 4–900 times
    more accurate than the default.
  - `Float32` mass drift is at or below the default's.
  - The default (`SameAsHorizontal`) is unchanged, and lat-lon runs and the
    adjoint reject the option.
  - `scheme = "linrood"` accepts the same `vertical` values; configured runs
    default to `fv3_kord8` (see below), while the programmatic
    `LinRoodPPMScheme(ORD)` keeps `vertical = UpwindScheme()`.
  - `CSAdvectionWorkspace` gains `column_scratch` (`column_scratch = true`).
- Configured Lin-Rood transport now defaults its vertical remap to
  `fv3_kord8`, the closest GCHP match. Explicit `vertical = "upwind"` and
  `vertical = "fv3_kord8_signed"` settings are unchanged.
  `LinRoodPPMScheme(ORD)` and `LinRoodAdvectionSpec(order)` keep their
  upwind default. Of the CPU goldens, `run_c24_linrood_cpu_f32` changes in
  every tracer.
- The Lin-Rood adjoint now also transposes the FV3 `kord = 8` vertical sweeps
  (`vertical = "fv3_kord8"` and `"fv3_kord8_signed"`; OF-94); explicit upwind
  vertical sweeps are no longer required. Footprints and 4D-Var gradients use
  the same vertical transport as forward Lin-Rood runs.
  For fixed limiter branches the sweep is linear, `rm' = W(β) rm`. Each vertical
  record now also stages the tracer mass entering the sweep; the reverse pass
  replays the forward profile code on numbers that carry their partial
  derivatives, which recovers the forward's branches bit for bit, and applies
  `W(β)ᵀ` column by column on CPU and GPU (a frozen-limiter adjoint, as for the
  horizontal PPM). The Lin-Rood tape grows from ten to twelve full-panel tuples
  per step (C90 L72 Float64: 552 instead of 488 MB per step).
  `require_adjoint_support` accepts these schemes, and the inversion driver's
  Lin-Rood `vertical` default is now `fv3_kord8`, as in forward runs. The footprint API's finite-difference forward
  (`run_cs_footprint_forward`) now runs the scheme's vertical sweep; it ran
  upwind for every Lin-Rood vertical. Forward transport is bit-identical.
  `CSAdjointWorkspace` takes `column_scratch = true` for tapes with FV3 vertical
  sweeps. The split-sweep `PPMScheme` with the FV3 vertical profile still has
  no adjoint.
- `PPMScheme` was documented as third order. It is second order: the limited
  PPM edge sets a Russell–Lerner slope and the parabola is not integrated.
  The docs now say so.
- Rn-222 and other configured exponential decay now use exact half-window
  decay steps before and after each window's emissions. For constant emission
  rate `E`, decay rate `λ`, and window `T`, the discrete steady burden is
  `E T / (2 sinh(λT/2))`; its hourly Rn-222 bias relative to `E/λ` is
  -0.000237736%, down from -0.377204% with emit-then-decay ordering.
  Only constant decay rates are split, which covers every `[chemistry]`
  decay; `ExponentialDecay` with a time-varying rate and other chemistry
  keep the end-of-step order. On binaries with a per-window physics contract
  the half-steps bracket the whole met window and convection stays at its
  end. Rn-222 golden fields change by 0.09–0.38 % relative;
  `catrine_true_mass_balance.jl` replays the new order.
- Kz diffusion computes layer thicknesses from total surface pressure; on
  binaries that store dry `ps` (the GEOS native, MERRA-2 and ERA5 N320
  writers) it used the dry value. Total pressure is recovered per column as
  `p_s = (p_s,dry − Σ_k (1 − q_k) ΔA_k) / Σ_k (1 − q_k) ΔB_k` from
  window-start or VDIFF humidity (one warning without either); `B ≡ 0`
  coordinates are unchanged. `pressure_layer` initial conditions use the
  same recovery. Precomputed `dkg` diffusion reads no layer thickness and
  is unchanged, as are the CPU runtime goldens, which use it.
- Conservative surface-flux regridding (cubed sphere; lat-lon and
  reduced-Gaussian targets with `regridding = "conservative"`), the fallback
  cell areas of per-cell inventories without a `cell_area` or `area`
  variable, and cubed-sphere file initial conditions build their source
  cells from the file's coordinates with `GriddedIO.SnappedFaces` (snapped
  regular axes; midpoint faces and half-width polar caps for pole-centred
  rows) instead of the uniform mesh of `UniformSpanFaces`, which the runtime
  no longer uses. The uniform mesh misplaced the 143 pole-centred latitude
  rows of the CATRINE SF₆ initial state by up to 0.62° (mean 0.31°). In the
  `run_c90_era5_tm5` golden, SF₆ column means change by up to 0.87 % and
  single cells by up to 40 % (at East Asian surface hotspots near 38.5° N,
  127.6° E; at most 0.46 % poleward of 85° N). GridFED totals change by
  9e-16 relative; the regridded GridFED field of the `run_c90_merra2` golden
  changes by at most 3e-9 of its maximum (its Float32 0.1° coordinates gave
  the uniform mesh a west face of 7.45e-10° instead of 0°).
- Stepwise time-varying surface fluxes read from lat-lon files whose time
  coordinate has CF `bounds` switch to each slice at the start of its bounds
  instead of at its time stamp, so a monthly file stamped mid-month now
  switches on the first of each month, about half a month earlier.
  `stepwise` is the default for `lmdz_co2` and `gridfed_fossil_co2`; linear
  and conservative schemes and files without bounds still use the stamps.
  Golden results are unchanged.
- `Float32` runs now conserve global tracer mass to about 1e-6 of the burden
  per year. Over 3 days of ERA5 C90 transport without sources, the background
  tracer drifts +9e-9 instead of −9.7e-7 and SF₆ +2.1e-9 instead of −1.0e-6.
  A small advection bias (≈+1e-6 per year) and, with CMFMC convection, about
  −1e-6 per year for CO₂ remain. Changes:
  - TM5 convection and `dkg` diffusion return each column's rounding residual,
    computed with compensated sums, to its largest participating cell. Only
    rounding-sized residuals are returned, so a non-conserving matrix stays
    visible.
  - GEOS-Chem's non-local emission profile accounts for every upper-layer
    addition with `TwoSum`.
  - Cubed-sphere areas, edge lengths and corners, regridding geometry, the
    decay decrement `expm1(−λΔt)` and the model clock are evaluated in
    `Float64` and rounded once. The clock follows the window and step
    counters, so window ends are exact for any step count. Meshes of either
    precision now regrid on a `Float64` sphere and share cached weights
    (cache version 3, so every cached regridder is rebuilt once).
  - Lat-lon flux and initial-condition sources stored in `Float32` (GridFED)
    snap to exact global extents. Previously the regridder dropped 2.1e-6 of
    the GridFED flux in `Float64`.
  - `total_mass` and `total_air_mass` return compensated `Float64` sums,
    reduced on the device (shared with the snapshot totals).

  `Float32` global totals change at the 1e-6 level. `Float64` global totals
  change by about 1e-11, except GridFED emissions (+2.1e-6). See
  `docs/src/theory/float32_conservation.md`.
- The two copies of the cubed-sphere block kernels (nested binary coarsener
  and GEOS block coarsening) are one, `Regridding.coarsen_sum!` and
  `coarsen_area_weighted!`. Area-weighted means now accumulate numerator and
  denominator in `Float64` and round once (the coarsener's semantics); the
  GEOS path accumulated them in the data precision. Float32 GEOS-native
  builds whose target `Nc` is a divisor of the source `Nc`
  (`geosit_c90_dec2021_catrine_f32_fullL72.toml`,
  `geosit_c45_dec2021_catrine_f32_fullL72.toml`,
  `geosfp_c720_native_to_cs180.toml`) change in `pblh`, `ustar`, `hflux`,
  `t2m`, the vdiff `u/v/t/qv`, `cmfmc` and `dtrain` before vertical
  remapping: 1–5 Float32 ulp in about half the cells of positive fields,
  more on near-cancelling signed fields (the old result could be thousands of
  ulp from the correctly rounded mean). Cell sums, face sums, Float64 builds
  and same-`Nc` builds are bit-identical.

### API and configuration changes in detail

- Preprocessing column balances spread their correction with
  `column_balance_weights = "hybrid_mass"` by default on every target (owner
  decision OF-03; effect in Numerical changes). `[numerics]
  column_balance_weights` is now accepted for lat-lon and reduced-Gaussian
  targets (it was rejected earlier in this cycle; v0.4.0 had no
  `column_balance_weights` key), every path that balances records weights
  other than `mass` in its header (MERRA-2 and ERA5 N320 record every
  choice), and an existing binary whose recorded weights differ (absent
  means `mass`) is regenerated instead of reused. The GEOS `pressure_fixer`
  and `pfix_corrected` closures keep the native fluxes, so column weights do
  not apply to them: they are accepted and ignored (and not recorded) instead
  of rejected. The three shipped GEOS pressure-fixer configs
  (`geosit_c180_dec2021_catrine_f32_{fullL72_pressurefixer,merged_pressurefixer,pfixcorrected}.toml`)
  failed with the new default because `config/met_sources/geosit.toml`
  supplies it. `Preprocessing.validate_preprocessing_config(cfg)` runs every
  config-level check of `process_day(cfg)` without reading input, and a core
  test loads all shipped preprocessing configs through it from a scratch
  working directory (a config whose external input, such as the
  reduced-Gaussian `geometry_source_grib`, is absent is skipped and counted).
- Removed the met-source descriptors `merra2_geoschem_hm_gchp.toml`,
  `merra2_geoschem_hm_vs.toml` and `merra2_geoschem_hybridmass.toml`: with
  the new defaults they equal `config/met_sources/merra2_geoschem.toml`, which
  their preprocessing configs now use.
  They were added during this release cycle; v0.4.0 did not ship them.
- Native preprocessing (ERA5 N320, MERRA-2) errors when the next day's
  endpoint is missing on any day but the configured last day; the warned
  zero-tendency fallback applies only there. ERA5 N320 days are no longer
  day-threaded (their inner kernels use static scheduling).
- The preprocessing dry-mass target is now the single
  `Parameters.GLOBAL_MEAN_DRY_SURFACE_PRESSURE = 98325.0` Pa constant for
  every meteorological source. `[mass_fix].target_ps_dry_pa` and
  `target_total_kg` are removed and rejected. Transport binaries record
  `mass_fix_target_ps_dry_pa`; multi-file runs reject mixed targets, using the
  window-1 mass and header radius as a legacy-binary fallback. Existing
  production binaries pinned near 98,726–98,733 Pa must be regenerated.
- ERA5 L137 `echlevs` now uses TM5/ECMWF half-level numbering (`0` is the
  model top, `137` the surface). The ambiguous development aliases
  `ml137_94L` and `ml137_85L` are removed; use `ml137_cfl94` and
  `ml137_cfl85`.
- Run settings no longer come from environment variables. These are removed
  without a transition period; the package ignores them:

  | Removed variable | Use instead |
  |---|---|
  | `ATMOSTR_FORCE_PER_SUBSTEP_PHYSICS` | `[run] physics_cadence = "substep"` |
  | `ATMOSTR_REPLAY_CHECK` | `[input] validate_replay = true` |
  | `ATMOSTR_NO_REPLAY_CHECK` | nothing (the load-time check runs only when asked for) |
  | `ATMOSTR_ENABLE_HORIZONTAL_POISSON_BALANCE` | nothing (the per-layer balance it selected is removed; see below) |
  | `ATMOSTR_DISABLE_PREFETCH` | `[run] prefetch_windows = false` (new) |
  | `ATMOSTR_ASSERT_CS_BINARY_CFL` | `[advection] check_binary_cfl = true` (new) |
  | `ATMOSTR_NO_WRITE_REPLAY_CHECK` | `[numerics] write_replay_check = false` in the preprocessing config (new) |

  Environment variables now set only paths, metadata, profiling and
  diagnostics (see `docs/src/config/environment.md`). Julia code that calls a
  preprocessing writer directly (`process_day(date, grid, settings, vertical)`,
  `regrid_ll_binary_to_cs`, `process_merra2_to_cs_day`,
  `process_era5_n320_to_cs_day`) passes `write_replay_check = false` instead
  of setting the variable; `write_replay_check_enabled` (added and removed
  during this cycle) is gone. The three
  new keys:

  - `[run] prefetch_windows` (default `true`): GPU runs with at least two
    Julia threads load the next met window on a second thread. The
    `DrivenSimulation` keyword of the same name fixes the choice for the whole
    simulation; the variable was read every window.
  - `[advection] check_binary_cfl` (default `false`): a cubed-sphere
    split-sweep step (`upwind`, `slopes`, `ppm`) that takes its subcycle count
    from the binary recomputes the CFL budget and stops if the binary asks for
    fewer. Typed as `BinaryCFLCheck()`/`NoBinaryCFLCheck()` on
    `CSAdvectionWorkspace` (the `binary_cfl_check` keyword of
    `TransportModel`); other schemes and grids reject `true`. The three
    operator copies of the check are now one dispatched helper.
  - `[numerics] write_replay_check` (default `true`) of a preprocessing config
    reaches every writer as the `write_replay_check` keyword.
- The per-layer Poisson balance is removed from every preprocessing path
  (issue #11): balancing each layer to its own mass change leaves almost no
  vertical mass flux, because no path has an independent vertical flux (see
  Numerical changes). Every path that applies a Poisson balance uses the
  column balance; GEOS `pressure_fixer` and `pfix_corrected` keep their native
  horizontal fluxes and record `"none"`. Removed with it:
  the `[numerics]` keys `balance_mode` and `geos_balance_mode` (a config that
  sets either is an error; the 19 shipped preprocessing configs and the
  `pre_merra2_c90` golden config that set `balance_mode = "column"` no longer
  do), the `--balance-mode` option of
  `scripts/preprocessing/regrid_ll_transport_binary_to_cs.jl`, the
  `horizontal_balance` keyword of the writers (`process_day`,
  `regrid_ll_binary_to_cs`, `process_merra2_to_cs_day`,
  `process_era5_n320_to_cs_day`) and `balance` of `apply_poisson_balance!`,
  the types `AbstractHorizontalBalance`, `ColumnBalance` and `LayerBalance`
  with `balance_tag`, `resolve_horizontal_balance` and
  `effective_horizontal_balance`, the GEOS writer's `process_day(...;
  balance_mode)` keyword, and `config/preprocessing/geosit_c180_dec2021_catrine_f32_reclosed.toml`
  (the per-layer twin of `..._pinned_column.toml`; its output directory was
  empty) with its two diagnostic run configs
  `catrine_geosit_c180_v4_fullphys_reclosed_1d{,_f64_curry}.toml`. Headers
  keep `horizontal_balance` (`"column"`, or `"none"` for the GEOS closures that
  keep the native fluxes, `HORIZONTAL_BALANCE_TAG`).
  Of the removed names, only `geos_balance_mode`, the GEOS `process_day(...;
  balance_mode)` keyword and the `_reclosed` configs existed in v0.4.0; the
  others were added during this release cycle.
- The advection keys `scheme`, `ppm_order`, `vertical` and `limiter` are read
  only from `[advection]`; in `[run]`, their old location, they are an error
  that names the new table. The 121 shipped run configs that used `[run]`
  moved them to `[advection]` (checked by parsing each file before and after);
  two of them, which set `[run] scheme` next to an `[advection]` table and
  were therefore rejected before, now run.
- `surface_flux.molar_mass_kg_mol` is replaced by `surface_flux.species`
  (`"co2"`, `"sf6"` or `"rn222"`); the molar mass comes from
  `Parameters.SPECIES_MOLAR_MASS`. The old key is an error that names the new
  one. Tracers named `co2`, `co2_natural`, `co2_fossil`, `fossil_co2`, `sf6` or
  `rn222` may still omit it; any other tracer with a flux must set it, and an
  unknown species is an error listing the known ones (also reported by
  `validate_config`). The 250 occurrences in the shipped run configs (245 in
  `config/runs/`, 5 in the golden-output configs) moved to `species`; every
  migrated tracer resolves the same molar mass, and so the same storage factor
  bit for bit (checked on all 461 tracer fluxes of those configs).
  `Models.InitialConditionIO.surface_flux_species` gives the resolution to
  scripts (`tracer_budget_closure.jl` uses it); the private
  `_tracer_molar_mass_kg_mol` and `_KNOWN_TRACER_MOLAR_MASS_KG_MOL` are removed.
- The 4D-Var inversion driver (`scripts/inversions/cs_4dvar.jl`) reads a
  required `[advection]` table (#16). It hard-coded unlimited PPM and ignored
  any `[advection]` table, even an invalid one, so the gradient was always
  that of `PPMScheme(NoLimiter())`. The table now goes through the forward
  run's `advection_spec` and `materialize`, with an `InversionAdvectionContext`
  that requires `scheme` and also accepts `limiter = "none"` (unlimited PPM);
  forward runs are unchanged and still reject `none`. Schemes without a
  cubed-sphere adjoint (limited slopes, the FV3 vertical profile, `none`) are
  rejected by the new `require_adjoint_support`. Unknown tables and keys are
  errors, including `[diffusion]` and `[convection]`, which the driver does
  not wire into the cost, and the documented but never written `[output]`
  path; keys the selected kind does not read are warnings. The shipped
  `example_synthetic.toml` states `scheme = "ppm"`, `limiter = "none"` and
  gives bit-identical results. `SyntheticExperimentConfig` of
  `scripts/inversions/synthetic_experiment.jl` gains a `scheme` field
  (default unchanged).
- The replay-check API takes a `ReplayScratch` (per-cell divergence and
  flux-magnitude arrays) instead of one divergence array:
  `verify_window_continuity(layout, scratch, ...)`,
  `verify_window_continuity_rg(m_cur, hflux, cm, m_next, face_left, face_right,
  steps_per_window; scratch)` (the positional `div_scratch` is gone), and the
  `replay_scratch` keyword of `verify_ll_window_contract!`,
  `verify_rg_window_contract!`, `verify_cs_window_contract!` and
  `verify_write_replay_cs!` (was `div_scratch`). See Fixes for the test itself.
- Removed the deprecated, unused `State.MetState`,
  `diagnose_cm_from_continuity_vc!`, and
  `diagnose_cm_from_continuity_ka!` APIs. Call
  `diagnose_cm_from_continuity!` with explicit hybrid-coordinate `Δb`
  coefficients when diagnosing structured vertical mass fluxes.
- Preprocessing internals changed for the certificate and column-closure work
  (callers on other branches must follow): `FluxFidelityEvidenceBundle(flux,
  omega, closure)` takes a third, column-closure field;
  `SourceOmegaEvidence` and `source_omega_evidence_cs` take an
  `OmegaFidelityBaseline` (and cached `OmegaFidelityCells`) instead of
  `baseline_id`/`nrmse_limit`; `NativeFluxEvidence` holds
  `FluxBalanceSample`s; `balance_column_mass_fluxes!`, `balance_window!` and
  `apply_poisson_balance!` take `weights`; `recompute_cm_from_dm_target!`
  takes `residual_weight` (default: the layer mass, bit-identical);
  `balance_cs_column_mass_fluxes!` returns `final_column_residual_ratio` and
  `final_column_tolerance_{min,max,by_cell}` instead of
  `final_column_tolerance`; `diagnose_cs_cm!` accepts a per-cell
  `residual_tol` and a `context`; `CSPoissonScratch` has a `tol` buffer;
  `capture_structured_wind_flux_samples` takes a `StructuredFidelitySampler`
  and `capture_faceindexed_flux_samples` the sampled faces and values. In the
  header and its typed form, `omega_fidelity_limit` is replaced by
  `omega_fidelity_nrmse_limit`, `omega_fidelity_correlation_min`,
  `omega_fidelity_slope_min` and `omega_fidelity_slope_max`
  (`OmegaFidelityHeader.nrmse_limit`, `.correlation_min`, `.slope_min`,
  `.slope_max`; `OmegaFidelityBaseline` gains `slope_max`), and
  `FluxFidelityHeader` gains `time_sampling_by_window`. The per-sample
  balance-correction ratios `balance_correction_max_ratio` and
  `balance_correction_rms_ratio` (near-zero pre-balance faces dominated them)
  are replaced by `balance_correction_rms_relative`,
  `balance_correction_max_level_rms_relative` and
  `balance_correction_max_level` (`FluxFidelityHeader` fields of the same
  names). `TransportBinaryHeader` gains `column_closure`
  (`ColumnClosureHeader` or `NoColumnClosureHeader`), and
  `recheck_cs_column_closure` takes `flux_scale`;
  `scale_column_closure_record` is replaced by
  `full_window_column_closure_record`.
  Most of these names were introduced during this release cycle; for code
  written against v0.4.0 the changes that matter are the new arguments and
  results of `balance_column_mass_fluxes!`, `balance_window!`,
  `apply_poisson_balance!`, `recompute_cm_from_dm_target!`,
  `balance_cs_column_mass_fluxes!`, `diagnose_cs_cm!` and `CSPoissonScratch`.
- A surface-flux `kind` that is not a known source
  (`none`, `file`, `cs_native`, `lmdz_co2`, `gridfed_fossil_co2`, `edgar_sf6`,
  `zhang_rn222`) is an error, with a suggestion for a close name; it used to
  be read silently as a generic file, so a typo such as `gridfed` lost the
  GridFED unit conversion. Use `kind = "file"` for a generic NetCDF file. The
  two ocean-flux configs now say `kind = "file"` instead of
  `"eccodarwin_ocean_co2"` (same source, results unchanged).
- Removed the exported generic function `Preprocessing.reset_workspace!`,
  which had no methods.
- Snapshot `time` is a CF coordinate: units `hours since <run start>`
  (`YYYY-MM-DD HH:MM:SS`) with `calendar = "proleptic_gregorian"`, and in
  ATMSNAP headers `time_units` and `time_calendar` beside `times_hours`
  (`binary_to_netcdf.jl` restores the coordinate). The stored values are the
  same elapsed hours, but NCDatasets and xarray now decode `time` to dates;
  read `ds["time"].var[:]` for the numbers. Writing snapshots needs an
  absolute origin, `[input].start_date`, the first binary's `date` header or
  a `YYYYMMDD` (or `YYYY-MM-DD`) date in its file name; a run with output
  enabled (the default) and none of them is an error. With `[run]
  start_window`, the origin is the start of the first simulated window. The
  quickstart and `config/examples/minimal_template.toml` set `start_date`,
  and the synthetic quickstart binary has a `date` header.
- File initial conditions (lat-lon kinds and `cs_native`) must give the
  hybrid coefficients `ap` and `bp` together; a lat-lon file with only one
  of them was read as having none. `ap` must be in pascals: a `units`
  attribute must name Pa (`Pa`, `pascal`, `pascals`), and without one only
  all-zero coefficients or a largest magnitude above 2000 are accepted;
  hPa-sized values are refused instead of moving pressure interfaces by a
  factor of 100.
- `[architecture] backend` accepts exactly `"cpu"`, `"cuda"`, `"metal"` or
  `"auto"`. The aliases `host`, `nvidia`, `apple`, `apple_metal` and `gpu`,
  and spellings that differ only in case or separator (`"CPU"`,
  `"apple-metal"`), are errors that name the canonical value
  (`architecture.backend spelling "nvidia" was removed; use one of the
  canonical values "cpu", "cuda", "metal", or "auto" ("cuda" for this
  backend).`); any other value is an error listing the four. No shipped
  config used an alias. The table is parsed once into `CPURequest`,
  `CUDARequest`, `MetalRequest` or `AutoGPURequest`, and `GPU(::String)`
  accepts only `"cuda"` and `"metal"`.
- Julia selectors that were Symbols are types, without aliases. `fieldview`
  takes `transform = ColumnMean()`, `ColumnSum()`, `LevelSlice()` or
  `SurfaceSlice()` and `unit = NativeUnit()` or `PPMUnit()` (exported by
  `AtmosTransport.Visualization`); `frame_indices`, `snapshot_grid` and
  `movie_grid` select every frame with `AllFrames()` instead of `:all`;
  `PlotSpec` (and so `movie` and `snapshot_grid`) still parses the names.
  `CubedSphereSnapshotTopology` holds a panel-convention type.
  `bind_to_mesh(...; out_of_range_policy)` takes `RejectOutOfRange()`,
  `SkipOutOfRange()`, `ClampOutOfRange()` or the strings `"reject"`,
  `"skip"`, `"clamp"`; a Symbol is an error. `RuntimeOutputSpec.format` is
  `NetCDFOutput()` or `BinaryMmapOutput()` and its six-argument constructor
  is removed. Run outputs are unchanged.
- `[advection] prognostic_slopes`, which v0.4.0 ignored silently (there is
  no prognostic-moment scheme; `scheme = "slopes"` re-diagnoses its slopes),
  is an unknown key and therefore an error. The ten shipped configs that
  set it no longer do.
- `[run] start_window` or `stop_window` beyond the window count of any
  input binary is an error; `stop_window` was clamped to each binary's
  count. Lat-lon and reduced-Gaussian runs build their initial tracer
  mass on the air mass of `start_window` instead of window 1
  (cubed-sphere runs still require `start_window = 1`).
- Lat-lon transport refuses a mesh whose longitudes do not span 360°:
  the zonal boundary is periodic, so a regional mesh had its east and west
  edges joined silently.
- `cs_surface_flux_4dvar` requires the `scheme` keyword. It defaulted to
  `PPMScheme(NoLimiter())`, whatever scheme the caller's forward runs used.
- The transport-binary writers take their basis keywords explicitly.
  `open_streaming_transport_binary` and `open_streaming_cs_transport_binary`
  require `mass_basis` (v0.4.0 defaulted it to `:moist`, while
  `write_transport_binary` defaults to `:dry`); every writer
  (`write_transport_binary` and both streaming openers) requires
  `surface_pressure_basis = TotalSurfacePressure()` or
  `DrySurfacePressure()`, independent of `mass_basis`. The TM5 convection
  basis is the writer keyword `tm5_convection_mass_basis`; `extra_header`
  may set neither it nor `surface_pressure_basis`.
- ERA5 met sources accept only `[preprocessing] level_orientation =
  "top_down"`. `"bottom_up"` was accepted but never applied (every field
  and merge map is read top-first, so such an input would have been merged
  upside down); it is now an error. The six ERA5 N320 met-source files drop
  the redundant `level_orientation = "top_down"`.
- `regrid_ll_binary_to_cs` refuses a lat-lon binary that stores dry
  surface pressure: wind recovery needs total pressure, and the regrid path
  loads no humidity.
- Multi-file runs also reject binaries that mix explicitly pinned and
  unpinned dry mass (unpinned day chains, which drift by design, are
  accepted) or total and dry surface pressure. Writers record
  `mass_fix_target_ps_dry_pa` only when the pin is applied, next to one
  `mass_fix_enabled` flag. `binary_capabilities` and `inspect_binary`
  report `surface_pressure_basis`, `dry_mass_target_ps_pa`,
  `dry_mass_target_source` (`:header`, `:legacy_window1`,
  `:unpinned_window1` or `:unclassifiable`), `dry_mass_pinned` and
  `dry_mass_target_is_current` (within 0.1 Pa of
  `GLOBAL_MEAN_DRY_SURFACE_PRESSURE`), and `inspect_binary` flags an
  outdated target. Legacy moist binaries take their dry mass from
  `m (1 − qv_start)`; without humidity the target is unclassifiable
  (`MetDrivers.binary_dry_mass_target`).

### Fixes

- GEOS-Chem's non-local VDIFF (`[diffusion] kind = "geoschem_nonlocal_vdiff"`,
  OF-108) now spreads fresh surface emissions over the boundary layer in the
  columns where GEOS-Chem does. Its `qmincg` fallback keeps a column's
  emission `x` in the surface layer instead. It fired when any layer `k` had
  `q_k + c_k < 0`, with `c_k = x (f_k − δ_kN)` the counter-gradient increment,
  including layers with `c_k = 0`. Transport leaves roundoff negatives, as
  small as −3e-316, in tracers with a zero background. Over the first 2 h of
  the C90 MERRA-2 golden day, 94 % of the Rn-222 and fossil CO₂ column
  deposits therefore stayed in the surface layer; 2–3 % took the profile and
  3–4 % fell back as GEOS-Chem would. The choice also flipped with the sign of
  roundoff, which failed the Float64 GPU-vs-CPU release comparison. The
  fallback now requires `c_k < 0` and `q_k + c_k < 0` for some `k`: the
  redistribution makes a layer negative or a negative layer more negative.
  Every nonzero `c_k` lies in GEOS-Chem's bottom `npbl` layers, the range
  `vdiff_mod.F90` checks. GEOS-Chem also falls back on a layer that was
  negative before the step and that `c_k` does not lower; this test does
  not. Both placements add exactly `x` to the column. The equations are in
  the `ProfileDeposit` docstring and in
  `docs/src/for_tm5_gchp_users/operators_on_binaries.md`.
  Golden effect: the five `geoschem_nonlocal_vdiff` cases change in Rn-222
  and fossil CO₂ only; CO₂, SF₆ and the uniform tracer are bit-identical, as
  are all other cases. In `run_c90_merra2_cpu_f64` (24 h), Rn-222 changes by
  3.2 % (L1; largest cell 4.3 % of the field maximum) and fossil CO₂ by 3.6 %
  (2.0 %); column means by 0.79 % and 1.1 % (L1). The surface layer holds less
  of the tracer: 59.8 % instead of 60.8 % of the Rn-222 mass at 3 h (fossil
  CO₂: 44.9 % instead of 47.1 %), 29.7 % instead of 30.1 % at 24 h (30.0 %
  instead of 30.5 %). Total masses agree to 2e-16. In the Float32 GPU cases
  Rn-222 changes by 1.6–2.6 % and fossil CO₂ by 3.6–3.9 % (L1; total masses
  to 8e-9, Float32 rounding), except with the CW84 limiter
  (`run_c90_merra2_cw84_gpu_f32`): 0.03 % and 0.20 %. That case leaves fewer
  negative values (at 3 h, 0.5 % and 1.1 % of the cells, against 2–3 % and
  17–22 % in the other Float32 cases and 49 % and 48 % in Float64), so the
  old test rarely fired there.
  CATRINE MERRA-2 protocol runs with this scheme
  (`config/runs/catrine_protocol/catrine_c90_merra2_l72_protocol.toml` and
  the GCHP-matched MERRA-2 runs) therefore kept too much of the emission of
  zero-background tracers (Rn-222, fossil CO₂) in the surface layer. Tracers
  with a positive background everywhere (CO₂, SF₆) are unaffected.
- TM5 convection level merging (`n_merge > 1`, OF-107) no longer amplifies
  super-layers that hold near-cancelling values of both signs. The merged
  path gave each fine layer `R'_s r_k / R_s` of its super-layer's new tracer
  mass. These weights are unbounded when the fine values nearly cancel, as
  small advection undershoots and negative emission cells (0.47 % of the
  CATRINE fossil flux cells) make them. On the C90 golden columns, one
  hourly step grew a column's `Σ|r|` up to 84-fold while totals stayed
  conserved. Each fine layer now receives a share of the coarse change,
  `r'_k = r_k + (R'_s - R_s) |r_k| / Σ_{j∈s} |r_j|`. An all-zero
  super-layer uses air-mass weights, or equal weights if its air mass is
  zero too. The split conserves mass and keeps a uniform mixing ratio.
  Layers the solve leaves unchanged keep their values bit for bit, and no
  cell moves by more than `|R'_s - R_s|`. For single-sign super-layers the
  split equals the old proportional split and is evaluated as
  `R'_s (r_k / R_s)`, so non-negative tracers stay non-negative, also in
  Float32. The air-mass-weighted increment is linear but was rejected: on
  the same columns it made up to 35,000 negative cells per hourly step from
  non-negative Rn-222 and fossil fields. Merged solves still have no
  adjoint. The equations are in the TM5Convection docstring and in
  `docs/src/for_tm5_gchp_users/operators_on_binaries.md`.
  Golden effect: the 27 `n_merge = 1` cases are bit-identical, and
  `run_c90_era5_tm5_nmerge2_gpu_f32` changes in every tracer. Its dipoles
  are gone. At every output time of the 24-hour run, the Rn-222 maximum
  matches the `n_merge = 1` case to four digits; before, it reached
  8.4e-17 at 06 UTC, 163 times the `n_merge = 1` maximum. The most negative
  Rn-222 value at 24 h is −2.62e-20 (−2.62e-20 with `n_merge = 1`;
  −4.7e-19 before). Fossil CO₂ at 24 h ranges from −2.54e-6 to 4.27e-4
  (`n_merge = 1`: −2.57e-6 to 4.27e-4; before: −3.05e-3 to 3.18e-3). The
  largest difference from the `n_merge = 1` case falls from 163 to
  0.17 times its maximum for Rn-222 and from 9.1 to 0.11 for fossil CO₂.
  Column `Σ|r|` exceeds `|Σ r|` by 1.45e-3 (Rn-222) and 3.7e-3 (fossil
  CO₂) at 24 h, as with `n_merge = 1`; before, by 2.0e-2 and 0.25.
  Every earlier run with `n_merge ≥ 2` used the unbounded split, with or
  without `lmax_conv` clipping. Check its surface-tracer output for dipoles
  and negative values. The shipped configs that set it are
  `config/runs/campaign_winter2021/era5_4tracer_fullphys.toml` and
  `config/runs/fullphys_dec1-3/era5_fullphys_co2nat_dec1-3.toml` (both
  `n_merge = 2`, ERA5 L137), and
  `config/runs/convection_compare_4tracer/convonly_tm5_n3_era5_c180_3day.toml`
  (`n_merge = 3`). The timing configs
  `config/runs/era5_geosnative_c180_cfl85_3day_ppm/fullphysics_steps48_1day_timing_collab_n{3,4,5,15}.toml`
  also set it.
- Native preprocessing (GEOS, MERRA-2, ERA5 N320) records the configured
  dry-mass target `mass_fix_target_ps_dry_pa` exactly
  (`GLOBAL_MEAN_DRY_SURFACE_PRESSURE`, 98325 Pa). It was recomputed from the
  pinned air mass: 98325.00162169515 Pa on the GEOS-IT C45 golden (the GEOS
  writer divided by a Float32 `g`) and 98325.00000000001 Pa on the MERRA-2
  C90 and ERA5 N320 L117 goldens at 637036cf. GEOS `initial_endpoint` runs,
  which have no configured pressure, still record the pressure of the pinned
  mass. The pin and the payloads are unchanged.
- Relative hybrid-coefficient paths of preprocessing configs
  (`config/era5_L137_coefficients.toml`, `config/geos_L72_coefficients.toml`)
  resolve in the working directory first and then in the package root
  (`Preprocessing.resolve_coefficients_path`), as `[source].toml` already did;
  the shipped configs failed from any other working directory, e.g. under
  `Pkg.test`, which runs in `test/`. A missing table names both locations; a
  working-directory table that differs from the shipped one is used with a
  warning naming both files.
- A run's absolute time origin is checked before any output or forcing is set
  up, whether or not the run writes output: `[input].start_date`, the first
  binary's `date` header and a `YYYYMMDD` (or `YYYY-MM-DD`) date in its
  original, unstaged file name must agree wherever present, so a start date
  one day off a legacy `transport_20211201.bin` is an error rather than a
  silent shift of output times and time-varying surface fluxes. `RunContext`
  applies the same check. A delimited 8-digit file-name token that is not a
  calendar date does not date the run (warning).
- `GriddedIO.decode_time_bounds` converts `julian` bounds and `standard`
  bounds before 1582-10-15 as instants, exactly as `decode_time` converts the
  coordinate (Julian 2000-01-01 is 2000-01-14), so stepwise surface fluxes
  from such files switch at the right instants, not up to 13 days early.
- Write-time flux-fidelity certificate in the header of every transport binary
  the preprocessors write
  (#15, item 3). Wind-derived paths rebuild deterministic face samples from
  `u`, `v` and `Δp` before the column balance and fail a window before it is
  written when a sample exceeds both `64 ε max(F_scale, 1 kg)` and
  `256 ε` relative; the balance correction `Δ = F_balanced − F` is reported
  separately, flux-weighted: largest `|Δ|`, `RMS(Δ)/RMS(F)` over all samples
  and its largest per-level value with the level (goldens, global and
  largest per-level value, the latter at the lowest level in every case:
  0.39 % and 0.95 % for MERRA-2 C90, 0.29 % and 0.58 % for ERA5 N320 L117,
  2.3 % and 5.9 % for GEOS-IT C45, 3.0 % and 6.9 % for ERA5 C24, 3.7 % and
  9.4 % for the 72 × 37 lat-lon and 0.78 % and 1.7 % for O24). Where
  source OMEGA is open (MERRA-2; GEOS OMEGA closures), `ω_cm = g cm/(A Δt)`
  must beat a zero vertical flux, `NRMSE < 1`, with correlation `> 0.5` and
  slope `0.5 < s < 1.5` (MERRA-2 C90: 0.47, 0.89, 0.87; baselines
  `merra2_wind_omega_v3`, `geos_native_omega_v3`); a failure quarantines the
  staged file. The two-sided slope bound rejects a `cm` that is uniformly a
  factor 2 or more too large or too small (an amount or time-convention
  error), which NRMSE and correlation alone accept down to 1/20. Native GEOS fluxes have no independent source and are certified
  `not_applicable` (`native_fluxes_not_rederivable`, balance correction only);
  payload-only paths say `source_payload_only`. The checks evaluate only the
  sampled faces (lat-lon columns, at most 32 reduced-Gaussian latitude lines,
  N320 thickness at stencil cells), record each window's effective time
  sampling (`window_start_boundary_fallback` at an archive boundary) and are
  checked for consistency by the writer and the reader (coverage of every
  window and level, finite metrics, status/kind pairs). The certificate is one
  header block that the TM5 convection attachment carries over from its source
  (the flux payload is unchanged). No certificate can catch input errors
  (registration, level order, units): reference and candidate share the input
  fields, and the vector evaluator shares the production face geometry. See
  `docs/src/concepts/binary_format.md`. The block carries
  `flux_fidelity_schema = 2`; the reader refuses any other schema with a
  "regenerate this binary" error.
- Cubed-sphere column balances close in Float64 with a bounded Float32
  closure loop; every production caller checks the closure, the `cm` residual
  is checked before it is folded into the column (an out-of-tolerance or
  non-finite residual raises), seam faces weight both physical cells, and
  zero-sum, NaN or Inf weights raise. The Float32 closure tolerance is per
  cell, `τ_c = max(10 kg, 8 ε Σ_{f∂c} Σ_k |F_{f,k}|)`: the global maximum of
  that budget (8.9e8 kg on GEOS C45, 2.3e7 kg on MERRA-2 C90) would have
  accepted the measured pre-fix defects (2.9e8 kg, 6.0e6 kg). Headers record
  the closure (`column_closure_*`: largest projected residual, its ratio to
  `τ_c`, largest residual folded into `cm`); a residual dominated by the
  global mean of an unpinned run names `[mass_fix]`. The Float64 solve changed
  golden payloads by at most 2.7e-4 (`pre_c24`), 3.8e-5 (`pre_merra2_c90`),
  6.0e-8 (`pre_era5_n320_c90_l117`) and 2.8e-3 (`pre_geosit_c45`) relative;
  the per-cell rule changes none: every golden window closes at the first pass
  with largest residual/tolerance ratios of 0.0021 (MERRA-2 C90), 0.014
  (GEOS-IT C45), 0.0002 (ERA5 N320 L117) and 0.01 (ERA5 C24), and `cm`
  residual ratios of at most 0.22. The record describes the stored payload:
  GEOS stores full-window amounts, `2n` times the balanced fluxes, a Float32
  product that rounds when `2n` is not a power of two, so its closure is
  evaluated again after that scaling (GEOS-IT C45: 0.014, set by windows with
  `n = 9`, against 0.0012 before the scaling; largest projected residual
  3.4e5 kg per flux application). The `cm` keys record the residual folded
  into `cm` at its diagnosis. The reader decodes the record
  (`TransportBinaryHeader.column_closure`) and refuses a binary whose record
  is incomplete, covers other than `nwindow` windows or exceeds its
  tolerance; binaries without the keys load as "not recorded". Known limits:
  the GEOS OMEGA baseline is unmeasured (no golden covers `omega_regularized`
  or `omega_full_replacement`), and coarse unpinned Float32 runs can
  approach the `cm` limit (unpinned GEOS-IT C45 golden day: `cm` residual
  ratio 0.924).
- ERA5 surface fields (PBL height, friction velocity, heat fluxes, 2 m
  temperature) are remapped from cell faces derived from the file's own
  coordinates (`SnappedFaces`, half-width polar caps). The previous uniform
  mesh shifted every column 0.125° east and stretched latitudes by up to
  0.125° near the poles. On the L117 golden day the surface fields change
  locally (median 0.05–0.2 %, p99.9 4–5 %, largest 10.8 % (pblh), 9.8 %
  (ustar), 12.8 % (heat flux) of each field's maximum), as does the derived
  TM5 `dkg` (largest 35.7 % of its maximum, but only 0.06 % of values change
  by more than 1 %).
- Every preprocessing transport writer and the ERA5 physics NetCDF-to-binary
  converter write through a unique sibling staging file and publish
  atomically; a failed day removes its staging file. A certificate that fails
  when the header is patched closes the writer and abandons the inner staging
  file too.
- The restored ERA5 physics and hourly spectral downloaders stage outputs,
  validate complete inventories and the requested grid/representation, and
  exit nonzero rather than accepting partial downloads.
- `CSLBFGS` uses Optim's L-infinity gradient norm consistently for stopping,
  `gradient_norm_history`, and iteration logs. This explicitly preserves the
  existing `gradient_tolerance` convention while making every reported
  L-BFGS norm match the stopping criterion. `CSGradientDescent` continues to
  use its documented L2 norm.
- Transport binaries now record whether `ps` is total or dry pressure as the
  typed `surface_pressure_basis` header contract. Every eager and streaming
  LL/RG/CS writer emits it, binary transformations preserve it, and the file-IC
  hybrid remap consumes the parsed `TotalSurfacePressure` or
  `DrySurfacePressure` value. Keyless legacy files retain the previous
  window-1 mass/pressure inference with one regeneration warning.
- `CSLBFGS` evaluated every control point at least twice (#17): Optim
  received separate value and gradient closures that each ran the full
  `cost_fn` (a forward run plus the adjoint footprints), and the initial
  probe and the final result were recomputed. Optim now gets one
  `OnceDifferentiable` whose value, gradient and combined methods share a
  cache of the last evaluation, so each point Optim visits costs one
  `cost_fn` call: 26 calls became 12 over an 11-iteration test inversion, with
  bitwise-identical costs, gradients and minimizer (single-threaded). The
  iteration log no longer records iteration 0 twice; its rows now match
  `cost_history`.
- The transport-binary replay check (write time and `[input] validate_replay`)
  is a per-cell test (#15, item 2). It divided each cell's budget residual by
  the largest cell mass of the window, so in thin top-level cells (down to
  1.85e-5 of the largest mass on the LL72 golden) it accepted relative errors
  up to 5.4e-6 at the Float64 tolerance and 540 % at the Float32 one. Each
  cell's residual is now measured against the summed magnitudes of its own
  budget terms,
  `|m_cur| + |m_next| + 2n Σ_faces |F|` (the scale that bounds its rounding
  error), with tolerances `REPLAY_TOLERANCE_FLOAT64 = 1e-12` and
  `REPLAY_TOLERANCE_FLOAT32 = 1e-5` (were 1e-10 and 1e-4) now in
  `src/Parameters`. Stored binaries measure at most 6.5e-14 (Float64 goldens,
  15× margin) and 1.0e-6 (Float32: the goldens and samples of the C90 ERA5
  L117/L66 and MERRA-2 and the C180 GEOS-IT production binaries, 10× margin).
  A residual normalised by the cell's mass alone would leave Float32 only
  2.7× margin (3.7e-6 measured), because Float32 rounding grows with the
  fluxes through the cell. Also: NaN or Inf in the stored data now fails the
  check (it reported 0 before); the load-time check of a binary loaded in a
  different float type than stored uses the less precise type's tolerance
  (`replay_tolerance(FT, DiskFT)`; a Float32 binary opened in a Float64 run
  failed it before); and the reduced-Gaussian write-time contract checks the
  stored output-type arrays instead of the Float64 work arrays they are
  rounded from, like the lat-lon and cubed-sphere paths.
  `scripts/validation/verify_cs_binary_continuity.jl` defaults `--threshold`
  to `replay_tolerance` of the binary's storage type (was 1e-10). Payloads
  are unchanged; the check does not validate the winds themselves.
- The ERA5 level selector chose coefficient `137 - e` for a TM5 `echlevs`
  entry `e`, reflecting TM5 level selections about the middle of L137. It
  now chooses coefficient `e` as TM5 does. `ml137_tropo34` contains the
  TM5-MP r1112 list verbatim, and new `ml137_tropo34a` contains the TM5
  cy3-4dvar/CAMS73 list verbatim. AtmosTransport's `ml137_66L`,
  `ml137_cfl94`, and `ml137_cfl85` recipes were transformed to keep their
  existing grids bit-identical; `ml137_tropo34` binaries must be regenerated.
- ERA5 and legacy TM5-archive convection fluxes are now converted from their
  total-air (moist) basis to the transport binary's air-mass basis before use.
  This removes a 0.7–1.1% global high bias (up to about 2% in humid tropical
  columns) in TM5 convective rates for dry-air binaries. New binaries record
  `tm5_convection_mass_basis`; readers reject a recorded mismatch and warn for
  legacy dry TM5 binaries until the production archive is regenerated. All
  three ERA5 routes are covered: spectral ec2tm, N320 to cubed sphere, and the
  CATRINE TM5 attachment. Reported by pera-noaa (NOAA GML).
- Spectral synthesis now drops the top zonal wavenumber `m = N/2` on
  even-length latitude rings instead of giving it half weight. The per-ring
  limit `min(T, (N - 1) ÷ 2)` is shared by the lat-lon, reduced-Gaussian and
  batched ERA5 N320 synthesis paths, with single-harmonic coverage for even
  and odd rings. Reported by pera-noaa (NOAA GML) in issue #13.
- Native MERRA-2, GEOS and ERA5-N320 dry-mass pins now distribute a uniform
  surface-pressure correction by `Delta B_k (1-q_k)`, matching the spectral
  path and leaving pure-pressure layers unchanged. Native target mass uses the
  binary header sphere, removing the older approximately 7 Pa radius mismatch.
- File initial conditions with hybrid coefficients are now remapped as layer
  integrals in the surface-anchored hybrid coordinate: source interfaces use
  `ap + bp * ps_target`, with the binary's own column surface pressure. The
  file's `Psurf` is no longer read or required, and
  `FileInitialConditionSource.psurf` is removed. This conserves column
  integrals to round-off, makes terrain-following ICs insensitive to uniform
  dry-mass-pin shifts, and handles pressure-level (`bp = 0`) files by extending
  the nearest end layer when their coverage stops above or below the target
  surface. For the CATRINE C90 CO₂ file, the measured bottom-10%-of-column
  difference from the file on its own levels is −0.000153 ppm for a uniform
  400 Pa shift; the replaced absolute-pressure point interpolation produced
  1.2–1.5 ppm RMS local errors. Runtime goldens using file ICs change in their
  file-initialized tracers (`co2` and `sf6`); analytic and `cs_native` tracers
  are unchanged.
- Transport binaries (eager and streaming writers), the spectral day cache and
  cached regridders are published through `AtomicFiles`: a sibling staging
  file moved onto the target by one replacing rename (`rename(2)` on POSIX
  systems) on every Julia version. Before, `mv(…; force = true)` deleted the
  previous file before renaming on Julia 1.10, so a reader could find no
  file, and on every Julia version deleted it when the rename failed. A failed
  rename now leaves the previous file in place and removes the staging file.
- Time-varying surface fluxes without `[input].start_date` took their time
  origin from the file: the first slice for decoded dates, the CF units origin
  for raw numbers. A run on a mid-month binary then read the month's first
  days. They now require `start_date` (`validate_config` reports its absence).
  All 61 shipped configs with time-varying fluxes set it; the two golden cases
  that did not (`run_c90_merra2`, `run_c90_era5_tm5`, on 2022-01-15) now set
  it, so their 3-hourly LMDZ fluxes come from January 15 instead of January 1.
- Surface-flux files in per-second units must be in kg of the species per m²
  per second (`kg m-2 s-1`, `kg/m2/s`, `kgCO2/m2/s`, ...); other per-second
  units such as `mol m-2 s-1`, and files without units, were read unconverted
  and were rejected at first; with the unit grammar of `GriddedIO` (see
  *Surface fluxes and preprocessing*) molar units (moles of the tracer's
  species unless the units name one) and gram units convert, while missing
  and unknown units stay errors. Every shipped flux file passes.
- With `binary_paths`, `[input] end_date` was reported as unread but still
  widened the span of loaded time-varying flux slices; the span is now the
  run's windows, also with `folder` (where every day through `end_date` has a
  binary, so nothing changes there).
- File-based initial conditions took their vertical order from the values
  of the level coordinate and copied levels one to one when the counts
  matched. A surface-first file with hybrid coefficients and the model's
  level count (the CATRINE start files have 79 levels) would have started
  upside down, a top-first file with hybrid coefficients would have been
  remapped wrongly, and a file without them but with another level count was
  zero-filled. Files with hybrid coefficients are now oriented by their
  interface pressures and always remapped; files without them must match the
  model's level count and state their order (pressure or height units, a CF
  `positive` attribute, or `init.vertical_order`). No shipped configuration
  was affected by the ordering fix: all use the CATRINE files on 34, 72, 117
  or 137 levels, which already took the correctly oriented path. Their values
  now change for the separate conservative-remap fix above.
- ERA5 N320 preprocessing had two registration errors. The GRIB
  `reduced_gg` fields (specific humidity and the convective mass fluxes)
  were placed half a cell east: their first point is at 0°, while the mesh
  cells are centred at `(i − ½) Δλ`. They are now interpolated to the cell
  centres. And the N320 → cubed-sphere regridder left up to 0.3 % of the
  polar cells uncovered, so every intensive field was too small there
  (surface pressure about 3 hPa low and temperature 0.8 K cold poleward of
  89° on C90); results are now divided by the covered fraction. ERA5 N320
  binaries built before this fix carry both biases.
- The per-ring spectral synthesis (`spectral_to_ring!`) did not mirror the
  highest wavenumber `(nlon − 1)/2` of rings with an odd number of longitudes,
  so that wave entered at half amplitude. ERA5 N320 has 72 such rings (of 640).
  Fixed there and in the batched synthesis below that now serves u, v and T;
  the change is at most 6e-3 m/s and 2.5e-3 K. (At that step the Nyquist
  term `nlon/2` of even rings still entered with half weight; the even-ring
  entry above drops it.)
- `scripts/preprocessing/attach_catrine_tm5_convection_cs.jl` stored the
  legacy 1-degree TM5 convection upside down. The CATRINE files keep ERA5 L137
  surface first, and the script summed them through the binary's top-first
  `merge_map` without reversing, so updraft entrainment landed near the model
  top. The script now verifies the level order against ERA5 L137 coefficients,
  the 1-degree grid, the date, and the three-hour slots (from
  `timevalues_bounds`; the CF `time` variable is corrupt in the 2022-04-30 to
  2023-12-31 files), and fails if the updraft entrainment's mean level lies in
  the upper half of the column. Outputs carry attachment tag `..._v3`, the
  source level order, and the code revision. Every binary set written by
  earlier versions is upside down: the C90 L66 set for 2018-12 to 2019-12 and
  the C30 full-physics sets for 2021 and Dec 2021 to 2022. Run configs and
  driver scripts now point at new `_v3` folders. See
  `docs/memos/2026-10-03_tm5_attach_level_order.md`.
- Cubed-sphere cell areas and edge lengths were computed in the mesh
  precision. In `Float32` they erred by up to 1.3e-4 per cell at C90,
  6.8e-4 at C180 and 9.7e-3 at C720. Preprocessing with
  `float_type = "Float32"` used these areas wherever it converts between
  pressure and mass. They are now evaluated in `Float64` and rounded once;
  regenerate `Float32`-preprocessed C180 and finer binaries.
- The EDGAR tonnes-to-flux normalisation (`_lonlat_cell_areas_m2`) took the
  grid spacing from the first coordinate difference, which errs by ~1e-4 for
  coordinates stored in `Float32`. It now uses the full span.
- `[mass_fix] mode = "initial_endpoint"` silently skipped the global dry-mass
  pin for MERRA-2 and ERA5 N320 sources: their writers pin only to a finite
  target, and this mode passes none. Only the GEOS native path implements it
  (`supports_initial_endpoint_mass_pin`); other sources now refuse the mode.
  No shipped configuration used it.
- Skipping the write-time replay gate (then `ATMOSTR_NO_WRITE_REPLAY_CHECK=1`,
  now `[numerics] write_replay_check = false`) did not skip it in the GEOS
  cubed-sphere writer. It does now, through
  `verify_window!(…; write_replay_on)`, and its log no longer reports a worst
  replay window when the gate was skipped. With the gate skipped, the
  cubed-sphere regrid, ERA5 N320 and MERRA-2 writers ran the positivity gate
  against the window's start mass only; they now also pass the end mass, as
  with the gate on (only runs with the gate skipped are affected).
- The reduced-Gaussian spectral writer wrote straight to the final file, so a
  day that failed a gate deleted an existing binary of that day. It now
  stages to `<out>.tmp` like the other writers.
- Visualization regridded cubed-sphere snapshots on the panel convention's
  default mesh, ignoring the recorded coordinate law, center law and
  longitude offset (`cs_*` and `longitude_of_central_meridian` attributes).
  `CubedSphereSnapshotTopology` now carries the snapshot's definition.
- The MERRA-2 download recipe uses OPeNDAP, whose download step is not
  implemented: it failed at the first file, after creating the output tree,
  with a message naming a script that no longer exists. `download_data!` now
  refuses such recipes before doing anything unless `--dry-run` or
  `--verify` is given (`protocol_can_download`).
- `[numerics] geos_balance_mode` was ignored by the MERRA-2 and ERA5 N320
  writers, which balanced per layer only with the environment variable
  `ATMOSTR_ENABLE_HORIZONTAL_POISSON_BALANCE=1`; no header recorded the
  mode. The variable and the per-layer balance are removed (see Breaking
  changes), and every transport-binary header records
  `horizontal_balance`. The MERRA-2 and ERA5 N320 default results are unchanged.
- The Lin-Rood (`LinRoodPPMScheme`) adjoint tape recorded the air mass for its
  two Upwind Z half-sweeps after the sweep had updated it. The upwind adjoint
  divides each face flux by the donor mass entering the sweep, so footprints
  and 4D-Var gradients carried an error that grows with the square of the
  vertical Courant number. At vertical Courant 0.08 and horizontal Courant
  0.02 it was up to 8.6e-5 relative for a surface-layer objective and 5.7e-8
  for a column mean; purely downward flux gave none. Every tape now records, then applies, its Z sweeps
  through one helper (`_record_and_sweep_z!`), and the Lin-Rood footprint
  matches central differences to about 1e-11. Lin-Rood footprints change by
  the size of that error; forward transport, the split-sweep footprints and
  transport binaries are bit-for-bit unchanged.
  Reported by pera-noaa (NOAA GML) in #10. The new
  `test/core/test_cs_linrood_footprint_fd.jl` covers the upward-flux case,
  and the Lin-Rood finite-difference tolerances that hid the error are
  tightened (`test_cs_stride_checkpoint.jl` from `3e-4` to `1e-9`).
- `[convection] kind = "cmfmc_matrix"` requires `cmfmc` and `dtrain` in
  every input binary; `dtrain` was checked in the first binary only.
- `scripts/validation/verify_cs_binary_continuity.jl` scales
  `full_window_mass_amount` fluxes (GEOS native binaries) to
  per-application amounts before the replay; it replayed the stored
  full-window amounts.
- `cs_tape_byte_estimate` uses Lin-Rood's own record layout (ten
  full-panel states per step, twelve with FV3 vertical sweeps, plus six
  face-panel tuples); it applied the split-sweep layout to Lin-Rood. The
  realized Lin-Rood tape is still larger than the estimate by four
  face-panel tuples per step (see Known limitations).
- The GEOS-FP C720 download recipe reads from the GEOS-Chem S3 archive
  (`https://geos-chem.s3.amazonaws.com/GEOS_C720/GEOS_FP_Raw/Y<yyyy>/M<mm>/D<dd>/`)
  instead of the WashU archive, without the collection directory the URL
  builder inserted. The two C720 preprocessing configs read physics from
  `met/geosfp/C720/raw_025`.
- Staged input copies (`[input.staging]`) and their source stamps are
  published through `AtomicFiles`; they were moved from `.part` files with
  `mv(…; force = true)`.
- Streaming transport binaries are published with mode `0o666 & ~umask`
  (`0o644` under umask `022`), like the other writers' files; they were
  staged with `mktemp` and published owner-only (`0o600`).
- The conservative remap of file initial conditions conserves tracer in
  the dry-air carrier. Each target layer gets the source's dry-air mean
  `∫q (1 − q_v) dp / ∫(1 − q_v) dp`, so a uniform field stays uniform,
  when `init.humidity_variable = "<name>"` names the source's specific
  humidity (file kinds, `catrine_co2` and `cs_native`; it must share the
  field's dimensions); without it the weight is `dp`, and names such as
  `q`, `QV` or `hus` are not detected. Source interfaces use the binary's
  total surface pressure: on a binary that stores dry `ps` it is recovered
  from the window-start (else VDIFF) humidity; a legacy dry-pressure
  binary without humidity keeps dry `ps` and warns once (typically
  0.5–2 % low in humid columns). The remap is O(N_source + N_target) and
  allocation-free.

### Runtime and output

- Every number that more than one place used, or that is a physical constant,
  now lives in `src/Parameters` (owner's rule, 2026-10-10): physical and
  reference values in `PhysicalConstants.jl` (new: `VON_KARMAN`, U.S. Standard
  Atmosphere values, `HYDROSTATIC_REFERENCE_TEMPERATURE`, `DAYS_PER_JULIAN_YEAR`,
  the dry-mass target `GLOBAL_MEAN_DRY_SURFACE_PRESSURE` and
  `GLOBAL_MEAN_SPECIFIC_HUMIDITY`), shared defaults of numerical settings in the
  new `NumericalDefaults.jl` (Courant limits and targets, Poisson-balance
  tolerances, subcycle caps, met interval, background VMR, Lin–Rood PPM order,
  …) and fixed properties of ERA5, GEOS and MERRA-2 in the new
  `MetSourceConstants.jl`. Scheme parameters stay in their scheme's struct
  (`PBLPhysicsParameters` gains the u*, PBL-height and 2 m temperature floors
  and the entrainment fractions that five Kz kernels wrote inline). Forward and
  adjoint advection share one mass-floor definition each. Every constant is
  pinned by `test/core/test_physical_constants.jl`. Results are bit-identical
  (all golden cases).
- Logging and runtime checks after Oceananigans. `[logging]` sets the run's
  minimum log level (`"warn"` concise, `"info"`, `"debug"` verbose; without it
  the current logger's level applies, info for the command-line runner)
  and its progress output (`progress = "bar"` as before, `"lines"` for one
  log line every `progress_interval_windows` windows in batch logs, or
  `"none"`). `[checks] nan_check` (on by default) stops the run with an error
  naming the field, window and model time when the air mass or a tracer holds
  a NaN or Inf, checked after every `nan_check_interval_windows` windows.
  In Julia, `DrivenSimulation(...; callbacks)` takes `Callback(func,
  schedule)` with the schedules `IterationInterval`, `WindowInterval` and
  `WallTimeInterval`; `NaNChecker` is one such callback function. Results are
  unchanged.
- Faster cubed-sphere GPU runs, with unchanged results (runtime golden cases
  bit-identical): on CUDA the halo exchange fills all 24 panel edges in one
  kernel launch and the corners in a second, instead of 24 edge launches (6 for
  packed tracer fields) plus 6 corner launches, one per panel (C90 L72 on an
  L40S: 120 → 25 µs per air-mass exchange); the CMFMC convection CFL scan runs
  on the device instead of copying three fields to the host once per window;
  NetCDF output stacks the layer-resolved cubed-sphere fields into staging buffers of the output
  type, one per requested layer count, reused across the fields and snapshots
  of one write. Two-day warm runs on C90: transport time −7 % (MERRA-2) and
  −9 % (ERA5) from the kernel changes; with 3-hourly 3-D NetCDF output, the
  stacking change together with a type-stable tuple of the 24 edge operations
  cuts the MERRA-2 run-loop wall time by a further 16 % (11.6 → 9.7 s; run-loop
  host allocations 26.8 → 15.0 GiB per two days; end to end including setup
  29.9 → 27.3 s). The GPU paths were tested on CUDA (L40S:
  `test/diagnostic/test_cmfmc_cfl_gpu.jl`, `test_cs_halo_fill_gpu.jl`,
  `test_point_ops_gpu.jl` and the seam, Lin–Rood and PPM GPU diagnostics) and
  on Metal (Apple M5 Pro, Float32, `test/diagnostic/test_metal_kernels.jl`:
  point operations, halo exchange and CFL scan
  exact; a one-day C24 run agrees with the CPU to rounding and is about 10 %
  faster than before, with laptop timings varying by up to 40 %). On Metal,
  fused point operations are launched one after another, each on its
  `bound_context` (`fusion_policy`; the halo fills bind their panel arrays); a C90 halo
  exchange then takes 1.12–1.34 ms instead of 1.29–1.67 ms for packed tracers
  and 0.45 instead of 0.58 ms for a 3-D field with corners.
- Cubed-sphere GPU runs refill one host window in place for every met window
  they load (`load_transport_window!`) instead of allocating a new one and its
  padded copies; the window is then copied to the device as before. Results
  are unchanged (runtime golden cases bit-identical). Two-day warm MERRA-2 C90
  runs on an L40S (six runs each): run-loop host allocations 15.0 → 4.9 GiB,
  run-loop wall time 10.2 → 8.7 s, of which transport 7.2 → 5.3 s (less
  garbage collection during transport).
- Time-varying surface fluxes (cubed sphere) are read only for the run period
  (they now require `[input]` `start_date`): from the last slice at or before the run
  start to the first slice at or after the run end. Every temporal scheme uses
  only the two slices around a time, so results are unchanged (two-day
  MERRA-2 C90 run with LMDZ, GridFED and three native-C90 sources:
  bit-identical NetCDF output). That run reads 10 instead of 6088 LMDZ slices
  and 2 instead of 36 GridFED slices, and takes 12.4 s instead of 25.9 s end to
  end (warm, L40S, six runs each; garbage collection 6.0 → 2.3 s). The
  loaded period ends with the binaries' windows (from every binary's header);
  `end_date` no longer widens it (see Fixes).
- Time-varying surface-flux series (`TimeVaryingSurfaceFluxSource`: `cs_native`
  and the regridded LMDZ and GridFED series) stay in host memory on every
  backend. Their `Adapt` hook used to copy the whole run-span series to the
  device: 0.78 GB per tracer in Float32 for a C90 2014–2024 daily driver, 73 GB
  for 93 such tracers, more than one GPU holds. Each source now has a device
  window buffer (`FluxWindowBuffer`, a ring of slots) with the slices the
  current met window reads (`_flux_window_slices`, next to
  `_flux_temporal_segments`); `DrivenSimulation` copies the missing ones when
  a window becomes current (`stage_surface_flux_window!`, at construction and
  after each window advance, on the task that launches the window's kernels).
  The buffer holds two or three slices for daily or hourly series and hourly
  met windows, independent of the run length. The kernels read the same
  rates, so results are bit-identical: every scheme, Float32 and Float64,
  window lengths that do and do not align with slices, and multi-binary runs
  (`test/core/test_surface_flux_window_buffer.jl`, against the former
  full-series operator kept in `test/helpers/surface_flux_window_buffer.jl`). `total_emission_rate`,
  `emission_segments` and `integrated_emission` sum the host series in place.
  On the CPU the buffer is a host array, so the same code runs on every
  backend. The surface-flux operator also stages each step's own slices, which
  copies nothing after the window was staged and lets it run without a
  `DrivenSimulation`. Verified on CUDA (L40S): every GPU runtime golden case is
  bit-identical (the ten C90 cases use time-varying fluxes), and the opt-in
  `test/diagnostic/test_surface_flux_window_buffer_gpu.jl` matches the full
  device series bit for bit for every scheme. A three-day C90 Float64 run with
  four `cs_native` daily tracers whose series span 2014-09-01 to 2024-12-31
  (3775 slices, 5.9 GB) writes bit-identical output, including the 3-D tracer
  fields, and its device memory peaks at 2.9 GB instead of 18.9 GB: the old
  hook copied the series to the device again for every input file, and up to
  three copies were alive at once. Time per day is unchanged within noise
  (10–13 s), and on CUDA host staging/enqueue takes 0.02 s over the run's 72
  windows (DMA completion can be paid at a later synchronization). This timer
  is backend-dependent: Metal's optimized dense-host path avoids repeated
  global queue synchronizations, but its low-level private-storage transfers
  wait for completion one by one, so `surface_flux_upload` includes those
  per-transfer DMA waits. The series
  still take host memory (0.73 GB per tracer in Float32 and 1.47 GB in Float64
  for that period). The opt-in Metal diagnostic covers the Float32 window
  buffer against the full-series reference.
- New in `Architectures`: point operations (`AbstractPointOp`, `Fused`,
  `Sequence`, `launch!(op, ctx, backend)`), with a per-backend
  `fusion_policy` (one fused kernel by default, separate launches on Metal). A kernel body is written once as
  `apply_point!(op, ctx, I...)` and runs as one GPU kernel or as CPU loops;
  independent operations fuse into one launch by type.
- The end-of-run summary also reports garbage-collection time, JIT
  compilation time and allocated memory.
- `fill_panel_halos!` and `copy_corners!` reject a halo wider than the panel
  (`Hp > Nc`) and panels of unequal size. CMFMC convection stops with an
  explicit error when its CFL scan finds a non-finite ratio, such as NaN or Inf
  in `cmfmc` (previously an `InexactError`); layers with NaN or non-positive
  air mass are skipped by the scan, as before.
- New PPM option `[advection] limiter = "cw84"` (`PPMScheme(CW84Limiter())`):
  the complete Colella–Woodward (1984) PPM, with van Leer-limited edge values
  and a flux that integrates the limited parabola over the swept fraction.
  Every sweep in which no cell exports more than its air mass keeps
  non-negative tracers such as fossil CO₂ non-negative, where the default PPM
  undershoots (−0.16 on a [0, 1] box in a 1-D test). Lat-lon and cubed
  sphere, CPU and GPU, with the cubed-sphere adjoint. The default
  `scheme = "ppm"` is unchanged.
- `[run] physics_cadence = "window" | "substep"` selects whether, on binaries
  with a per-window physics contract, convection and chemistry run once per
  met window (default) or every advection substep. It replaces the
  environment variable `ATMOSTR_FORCE_PER_SUBSTEP_PHYSICS`, which was read on
  every time step and is removed. Unlike the variable, `"substep"` keeps the window-end reset to
  the binary's endpoint air mass, so a cadence comparison changes only where
  convection and chemistry run.
- Runs and preprocessing from `git archive` code snapshots record their commit:
  git writes it into `src/REVISION` on export (`export-subst`), and
  `source_revision()` reports it when the tree has no `.git`; edits made after
  the export are not detected. Snapshot outputs had
  `framework_commit = "unknown"` before. Cubed-sphere transport binaries now
  carry `git_commit` and `git_dirty` in their header.
- `scripts/diagnostics/catrine_benchmark_page.py` writes the CATRINE benchmark
  web page from a `catrine_compare_vs_geoschem.py` output: Chart.js charts with
  run check boxes and tracer, band and statistic selectors (daily and monthly
  agreement statistics, period table, global burdens, true mass balance), code
  versions, maps and animations. The page is self-contained (data embedded,
  Chart.js copied next to it) and keeps the selection in the URL.
- `scripts/diagnostics/catrine_true_mass_balance.jl` computes the true global
  mass balance of a run (burden change minus the run's own applied emissions
  and decay) for AtmosTransport, from the run's output and its sources rebuilt
  by the model, and with `--model geoschem` for GEOS-Chem, from its
  `SpeciesConcVV`, `Met_AD` and `Emis*` output; the benchmark page plots them.
- New `[output.observations]` contract for sampling tracer profiles at
  observation points (OCO-2 Lite soundings, NOAA ObsPack sites, generic point
  tables) from the containing model cell at met-window ends. The table is
  parsed by `observation_output_spec` and checked by `validate_config`, which
  rejects unknown keys and requires an absolute run origin. Both runners
  sample at t = 0 and at every met-window end: soundings are blended linearly
  between the two bracketing window ends (or taken from the nearest), sites
  get the intake-layer value from hypsometric heights. Output goes to
  append-only `_soundings` / `_sites` NetCDF files (see the output schema).
  Gridded snapshot output is unchanged.
- Point events (satellite soundings, ObsPack records, station time lists)
  carry an intake height and get `<tracer>_intake` from the layer containing
  it; station tables choose `EveryWindow`, `TimeRange`, or `TimeList`
  schedules per row and accept `altitude` with `elevation`; repeated site ids
  merge their time lists. OCO sources take a typed `quality_filter`
  (`"flag_max"`, `"flag_values"` for categorical flags such as the MIP
  `assimilate_flag`, or `"none"` to co-sample every record). The editor schema documents every choice with its Julia
  type and is checked against the parser by a test; site tables have their own
  schema (`schemas/observation_sites.schema.json`).
- User guide `docs/memos/2026-10-02_observation_sampling_guide.md` with a
  runnable OCO-2 MIP + TCCON example
  (`config/examples/observation_sampling_oco2mip.toml`, site table
  `config/examples/tccon_ggg2020_sites.csv`).
- All runtime NetCDF writes, including the background daily snapshot task,
  share one lock (`with_netcdf_lock`); netcdf-c is not thread-safe.
  Observation rows queue while the lock is busy instead of stalling the run.
- Daily output files (snapshots and observations) now use the day index when
  a binary name carries no date; previously every day wrote the same file.
  Two binaries that still resolve to one daily file fail before transport.
- `[output.fields].tracers = "name"` (a single string) no longer throws a
  `MethodError`; it selects that one tracer as documented.
- Scripts cleanup: 161 stale scripts were removed (git history keeps them;
  recover one with `git show 7c515038:<path>`) and 58 were moved into a
  `heritage/` subfolder of their folder, kept for reference but not
  maintained or tested. The deprecated runner shims
  `scripts/deprecated/run_cs_driven.jl`, `run_transport_binary.jl` and
  `run_cs_transport.jl` are gone; use `scripts/run_transport.jl`.
  `scripts/completed_experiments/` and `scripts/deprecated/` no longer exist.
  `scripts/README.md` lists every removed script with its purpose, and each
  `heritage/README.md` says why its scripts are kept.
- Run configs are checked before any binary is opened. `validate_config`
  (run at the start of every run) now also parses `[advection]`,
  `[diffusion]`, `[convection]`, `[chemistry]`, `air_mass_reset_mode`,
  `physics_cadence` and `[output]`, so these errors no longer appear only after
  the binaries are inspected. New errors: a decay half-life for a tracer the run
  does not carry (it failed at the first chemistry step), an enabled `[output]`
  with a path but no snapshot-schedule key or with snapshot times but no path
  (nothing was written; `hours = []` still means deliberately none), Lin–Rood
  `ppm_order` other than 5 or 7. Keys the run would ignore were logged as
  warnings at this step; since the strict-config change (PR #26) unknown
  tables and keys are errors with a "did you mean" suggestion, and only known
  keys the chosen kind leaves unread (including flat `[tracers.<name>]` keys)
  and surface-flux tables without `kind` (no flux is emitted) are warnings. Of the 246 run
  configs with an `[input]` table in `config/runs` and `config/examples`, only
  four that already failed at run time differ: they now fail at the check.
- `[input] validate_replay = true` replays every binary's continuity when the
  run opens it (the stored fluxes must carry each window's air mass to the
  next). It replaces the environment variable `ATMOSTR_REPLAY_CHECK`; both it
  and `ATMOSTR_NO_REPLAY_CHECK` are removed (the check runs only when asked
  for).
  The replay error messages now name the key.
- A transport binary written with the write-time replay gate skipped
  (`[numerics] write_replay_check = false`, or the new
  `--no-write-replay-check` of `regrid_ll_transport_binary_to_cs.jl`) records
  `write_replay_check = false` in its header; `binary_capabilities` reports
  it, `inspect_binary` marks it and `TransportBinaryDriver` warns when it
  opens such a binary. Binaries written with the gate on are unchanged.
- `docs/src/config/environment.md` lists every environment variable the
  package reads, and `test/core/test_environment_variables.jl` fails when
  `src/` reads one that is not listed there (or from outside the folder that
  owns it).
- The two binary-format A/B configs `config/runs/binary_format_ab/c45_*` are
  retired: they set `[advection] order = 7`, which was never read, so they
  ran Lin–Rood PPM5 instead of the intended PPM7.
- Snapshot hours are checked against the met-window ends of every binary
  (now recorded by `binary_capabilities` as `nwindow` and `window_seconds`)
  when the run starts: an hour that is not a window end was never matched, and
  every later snapshot was lost with it. Snapshots are matched to window ends
  within half a window (at most half an hour, as before), so runs with
  sub-hour windows no longer take a snapshot one window early. The default
  snapshot span and the observation span count the windows of every binary,
  not the first binary's times the binary count. `format = "binary_mmap"` is
  rejected on lat-lon and reduced-Gaussian grids before the run instead of at
  the first write.
- The editor schema lists every key the runtime reads (new:
  `expected_nlevel`, `required_preprocessor_contract`, `convection.cloud_base`,
  surface-flux `regridding`, `init.clamp_negative`, top-level legacy `[init]`,
  and the deprecated aliases) and accepts `diffusion.kind =
  "geoschem_nonlocal_vdiff"`. A new test compares every schema choice with the
  parser's accepted values and every schema table with the known-key tables,
  both ways. New module `ConfigChecks` holds the strict Boolean and known-key
  helpers that `Output`, `Preprocessing` and `Models` each had a copy of.
- Runtime snapshot output records the physics that ran, always, as global
  attributes of NetCDF files and as root fields of the ATMSNAP JSON header
  (`scripts/postprocess/binary_to_netcdf.jl` carries them over). The string
  values come from the typed runtime specifications and the instantiated
  surface sources, not from the TOML text: `physics_advection_scheme`,
  `physics_advection_limiter`, `physics_advection_order` (Lin–Rood),
  `physics_vertical_reconstruction`, `physics_convection` with
  `physics_convection_clamp` and `physics_convection_cloud_base` (CMFMC) or
  `physics_convection_solver`, `physics_convection_lmax` and
  `physics_convection_level_merge` (TM5, CMFMC matrix), `physics_diffusion`
  with `physics_diffusion_surface_flux_coupling` and, for constant Kz,
  `physics_diffusion_kz_m2_s`, `physics_surface_fluxes` (per tracer `static`
  or `time_varying:<scheme>`), `physics_chemistry`,
  `physics_decay_half_lives_seconds`, `physics_mass_basis`,
  `physics_cadence`, `physics_air_mass_reset_mode`, `physics_met_source`
  (the binary's `source_type`), `physics_binary_provenance` (format version
  and the header's preprocessor, git commit, dirty flag and generation
  fingerprint) and `physics_binary_inputs` (binary file names). The snapshot
  writers take them as `physics_attributes` (names must start with
  `physics_`), and `Output` exports `set_physics_attributes!` and
  `set_time_origin!`; `write_snapshot_netcdf`, `write_snapshot_binary` and
  `NetCDFSnapshotStream` take `physics_attributes` and `time_origin`, and
  `RuntimeOutputSpec` gains a `metadata` field. Data values are unchanged.
- `init.kind = "cs_native"` initial conditions with a level count other
  than the binary's are remapped column by column with the conservative
  hybrid remap of the lat-lon file initial conditions (the binary's typed
  total or dry surface pressure; optional dry-air weighting with the new
  `init.humidity_variable`); they need the source `ap` and `bp` and were
  rejected before. Files with the binary's level count are still copied
  cell for cell (bit-identical); if they carry `ap`/`bp`, those set the
  source level order (a contradicting `vertical_order` is an error) and must
  match the binary's coefficients (`ap` within max(1e-3 Pa, 1e-6 relative),
  `bp` within max(1e-8, 1e-6 relative)) or the file is refused. Without
  coefficients `vertical_order` still defaults to `surface_first`.
- Downloads report through the logger instead of writing to standard
  output. A download recipe's `[logging]` table takes `level` (`"debug"`,
  `"info"`, `"warn"`, `"error"`) and `progress` (`"lines"`, the default,
  `"bar"` or `"none"`); start and completion summaries remain leveled
  messages when progress is off. Python and AWS subprocess output is read
  concurrently and forwarded while the job runs; only the last 64 KiB of
  each stream is kept, and a job that exhausts its retries reports both
  tails, also at `level = "error"`.
- New exports `AbstractSurfacePressureBasis`, `TotalSurfacePressure`,
  `DrySurfacePressure` and `surface_pressure_basis` (of a reader or
  driver). A custom `AbstractMetDriver` is taken to carry total pressure
  unless it returns `DrySurfacePressure()`.
  `Operators.Diffusion.recover_total_surface_pressure!` recovers total from
  dry surface pressure on CPU, CUDA and Metal.

### Surface fluxes and preprocessing

- ERA5 N320 preprocessing is about five times faster: a C90 day takes about 7
  minutes on 12 threads instead of 23–38.
  - The core GRIB file of a day (12–24 GB) is indexed once with ecCodes
    (`_core_messages`); every hourly window used to scan the whole file.
  - The spectral synthesis of all 137 levels runs as one BLAS product per
    zonal wavenumber with precomputed Legendre functions, followed by one
    inverse real FFT per ring and level (`ReducedSpectralSynthesis`); it
    recomputed the Legendre table for every ring, level and field before.
  - Spectral coefficients are decoded straight into the level cubes.
  - Humidity and surface pressure are unchanged; u, v and T agree to 1e-12
    relative, apart from the odd-ring fix above.
- ERA5 N320 option `[preprocessing] face_fluxes = "line_integral"` integrates
  `(V · N) Δp` along each cube face from the N320 winds and surface pressure
  (`LineIntegralFaceFluxes`; bilinear N320 values, midpoint rule), following
  TM5, which integrates the spectral winds along its cell edges, instead of
  interpolating cube-centre winds to the faces.
- ERA5 N320 option `[preprocessing] flux_time_sampling = "window_mean"` uses the
  mean of the face fluxes at the start and end of each hourly window instead of
  the winds at its start.
- `config/met_sources/era5_n320_arco_diffusion_li.toml` and
  `config/preprocessing/era5_n320_arco_diffusion_to_c90_l117_li.toml` combine
  both with `hybrid_b` on 117 levels (native below about 8 hPa).
- MERRA-2 preprocessing `[preprocessing] face_fluxes = "vector"`
  builds face fluxes from the cell winds as 3-D vectors. They are projected
  onto the true face normals, use the faces' great-circle lengths, and are
  interpolated along the edge at panel seams, as FV3 does. It is now the
  default together with Cartesian wind regridding, edge lengths, and FV3
  interpolation; `"panel_average"` remains available explicitly.
  - The default construction misplaces the seam winds by up to a quarter of a
    face length and uses cell centerline widths instead of face lengths.
  - For solid-body rotation the spurious divergence at seam cells drops from
    4e-3 to 5e-6 of the face flux, and in the interior from 1e-4 to 2e-7.
  - Against MERRA-2 OMEGA on 2022-07-15 (C90, p < 150 hPa), the
    vertical-velocity error at seam cells falls from 8.8e-3 to 1.3e-3 Pa/s.
  - Related key: `flux_thickness = "dry_mass"` (opt-in for MERRA-2, the
    ERA5 N320 default).
- Preprocessing option `column_balance_weights` sets how the column
  mass-budget correction of the horizontal fluxes is spread over levels:
  `"mass"`, `"hybrid_b"` (by ΔB, as in TM5) or `"hybrid_mass"` (by air mass in
  hybrid layers only, the default on every target; see Numerical changes).
  With `"mass"` the correction reaches the stratosphere as a vertically
  coherent `cm` mode; the hybrid options keep it out of the pure-pressure
  layers. In a four-month C90 run against GCHP, `"hybrid_mass"` cut the
  growth of the bias above 100 hPa by about two thirds. Near-surface RMSE
  changed by +0.4 % to −5 % depending on the tracer
  (`docs/src/theory/vertical_transport.md`). MERRA-2, ERA5 N320 and GEOS set
  it in the met source's `[preprocessing]` table, ERA5 spectral configs in
  `[numerics]` (lat-lon, reduced Gaussian and cubed sphere alike), and the
  lat-lon → cubed-sphere regrid with `--column-balance-weights`. Headers
  record every choice but `"mass"`; the GEOS pressure-fixer closures, which
  keep the native fluxes, ignore it.
- Time-varying surface fluxes can span several files: `files = [...]` or a
  `file_pattern` with `{YYYYMM}` plus `year`. `gridfed_fossil_co2` joins the
  time-varying path (stepwise by default; monthly totals use each month's
  length), and an inventory's year is taken from the last year in its file
  name (`GCP-GridFEDv2024.0_2022.short.nc` is 2022). Files with a CF time
  coordinate now take their months and month lengths from it; the file-name
  year and ordinal months remain only for coordinate-less GridFED files
  (see the per-month entry below).
- `kind = "cs_native"` with `time_varying = true` reads an already aligned
  GEOS-native `(time, nf, Ydim, Xdim)` flux-density series and converts it to
  per-cell storage rates without regridding.
- ERA5 N320 preprocessing to cubed-sphere grids can select a named L137
  level set on the target grid (`[vertical] transform = "level_selection"`,
  `preset = "ml137_66L"`), the recipe behind the C90 L66 ERA5 binaries, plus
  ARCO-ERA5 diffusion-only configs and a per-year C90 driver script.
- The ARCO C90/C180 and GEOS-IT OMEGA-regularized preprocessing configs now
  name the cubed-sphere definition `"gmao_equal_distance"`; the previous
  `"gmao"` was rejected by the parser.
- The inventory year in a surface-flux file name skips version markers and
  rejects ambiguous names. The default GridFED file
  `GCP-GridFEDv2024.0_2021.short.nc` was read as 2024, a leap year; it is now
  2021, so static February GridFED rates rise by 29/28.
- `cs_native` fluxes accept only kg m-2 s-1 units (of the species, or kgC
  converted by M(CO2)/M(C)), reject fill values, and must match the runtime mesh
  when the file carries `cell_area`.
- `[vertical].level_selection` presets are rejected on sources without 137
  native levels, and `[vertical].coefficients_file` is honoured as an alias of
  `coefficients`.
- Experimental `coarsen_nested_cs_transport_binary` restricts a cubed-sphere
  transport binary to a coarser nested grid (for example C90 to C30) by
  block-summing the transport operator; not yet validated as a replacement
  for preprocessing at the target resolution.
- `apply_regridder!(dst, r, src, kind)` regrids by quantity kind for every
  regridder: `IntensiveCellField()` as given, `ExtensiveCellField()` (per-cell
  totals) through the density, so totals are kept whatever polygon areas the
  regridder uses. It replaces Preprocessing's private `_density_correct!`;
  `regrid_2d_to_cs_panels!` / `regrid_3d_to_cs_panels!` give the same bits.
  The density buffer is the keyword `scratch` (one column, `eltype(src)`),
  allocated per call when not given; the cubed-sphere preprocessing workspace
  passes its own, and warm `regrid_2d/3d_to_cs_panels!` calls allocate
  nothing. `regrid_conservation_report(r, src, dst, kind)` returns both totals
  and the relative error and checks both leading dimensions against the
  regridder.
- `build_regridder` returns a `NestedCubedSphereRegrid{Restriction}` or
  `{Prolongation}` for cubed spheres of one definition and radius whose `Nc`
  differ by an integer factor (corners checked): exact block remapping in
  O(Nc²), equal to the polygon regridder to 1e-12, with nothing cached. Code
  that read `intersections` from such a pair's regridder must use
  `src_areas`/`dst_areas` and `apply_regridder!` instead.
- `Regridding.default_regrid_cache_dir()` is the one definition of the weight
  cache directory (`ATMOSTR_REGRID_CACHE_DIR`, else
  `~/.cache/AtmosTransport/cr_regridding`); the runtime's private
  `_regrid_cache_dir` is removed, and preprocessing targets without
  `[grid] regridder_cache_dir` use it too (before, a second hard-coded copy
  of the path that ignored `ATMOSTR_REGRID_CACHE_DIR`). Cached regridders are now published with
  mode `0o666 & ~umask` instead of `mktemp`'s owner-only `0o600`.
- `LatLonMesh(λᶠ, φᶠ; FT, radius)` builds a lat-lon mesh from explicit cell
  faces (the source grids of input files); `radius` is required. Longitude
  spacing must be uniform but may start at any face, and a sequence that wraps
  by −360° is unwrapped so data columns keep their order. Latitude faces may be
  descending (stored ascending; `Grids.latitude_row_order(φᶠ)` gives the row
  permutation) and non-uniform (pole-centred, truncated, Gaussian), in which
  case `Δφ = NaN` and `dx`, `dy` and the x-face lengths throw, and the lat-lon
  `write_transport_binary` refuses the mesh (its header records only the
  latitude interval). Areas come from the faces. The regular keyword
  constructor now builds its faces and calls this one; regular meshes are
  bit-identical.
- `Output.define_cs_horizontal_geometry!(ds, mesh)` (cubed-sphere `Xdim`/`Ydim`/`nf`,
  centres, corners, `cell_area`, `cubed_sphere` mapping with
  `semi_major_axis = mesh.radius`) and `Output.define_provenance_attributes!(ds)`
  are public, split from the snapshot writer's private blocks so other NetCDF
  writers share them. Snapshot files keep the same dimensions, data and
  attributes; the `lev` and `time` variables are now defined after the
  horizontal geometry variables.
- `Operators.total_emission_rate(src, t)`, `emission_segments(src, t, dt)` and
  `integrated_emission(src, t0, t1, dt)` give a surface source's global
  storage rate at an instant, the pieces one operator step applies, and the
  mass the operator adds over a run's steps, from the operator's own temporal
  reconstruction (equal to a stepped application to 1e-14 for every scheme).
  The runner's start-up line "Surface source … total model-storage rate" now
  prints `total_emission_rate(source, 0.0)`: the same number for `Float64`
  runs; for `Float32` runs the sum is accumulated in `Float64` (it differed by
  up to 5e-8 on the CATRINE C90 sources), and a linear or conservative series
  without a slice at the run start reports the interpolated run-start rate
  instead of its first slice. The sums run on the host: a source adapted to a
  device (CUDA, Metal) is copied to host memory first, so Metal, which has no
  Float64, gets its totals too. `tracer_budget_closure.jl` and
  `catrine_true_mass_balance.jl` use these functions instead of the private
  `_flux_temporal_segments`.
- New module `GriddedIO` (after `Regridding`, before `Preprocessing`): the one
  owner of the conventions of native gridded input files, measured on the
  TRENDY v14 S3 models, GFED5, FLUXCOM-X, CAMS, GridFED and EDGAR
  (`docs/memos/2026-10-10_python_flux_port_p0_facts.md`). `find_coordinate`
  finds coordinates by variable name only and checks their length;
  `latlon_orientation`/`orient`/`orient_coordinates` give ascending latitude
  and longitudes from the first value `≥ 0`; `read_raw` reads with the fill
  policies `MaskedAsZero`, `MaskedAsNaN` or `FillValueOnly`, converting each
  value once to the caller's element type (default Float64);
  `source_latlon_mesh` derives a file's cell faces (`SnappedFaces`: snapped
  regular axes, pole-centred, truncated and Gaussian latitudes, on
  `LatLonMesh(λᶠ, φᶠ; radius)`; bounds variables are never used);
  `decode_time` classifies a time axis by its units before any CFTime call
  (`CFDecoder`, `MonthsSince`, `DecimalYears`, `AuxiliaryTimeVariable`) and
  refuses invalid origins instead of normalising them (`months since
  2000-13-01`, `days since 2000-02-30` in the standard calendar);
  `decode_month_keys` gives the months of an axis in its own calendar, also
  where `DateTime` cannot hold a stamp (30 February of `360_day`);
  `stage_input` decompresses `.nc.gz` and single-member `.nc.tar.gz` inputs
  into a cache with atomic publication; `parse_flux_units` reads every units
  string in use into `FluxUnits`, converted by `kg_per_unit`,
  `period_seconds` and `mass_rate_factor` with the molar masses of
  `Parameters`.
- The `cs_native` flux file is defined once, in `GriddedIO`: the layout
  `CS_NATIVE_FLUX_DIMS`, the writer `open_cs_native_writer` / `write_slices!` /
  `publish!` / `abandon!` (contiguous start-stamped intervals with
  `time_bnds` in `hours since <first day>`, the mesh geometry and
  `cell_area`, units `canonical_units_string(species)`, chunks
  `(Nc, Nc, 1, chunk_time)`, no `_FillValue`, non-finite values refused,
  atomic publication), the streaming reader `cs_native_flux_info` /
  `read_cs_native_flux!`, which is now the runtime's `cs_native` loader (panel
  by panel into the source's own arrays, one panel-slab read buffer, as the
  former loader), and the whole-file reader `read_cs_native_flux`
  (`check_cell_area` is the former per-panel area gate, unchanged). Sources
  built from every `cs_native` file tested are bit-identical to the former
  loader. The reader is stricter: the NetCDF
  dimensions must be exactly `(time, nf, Ydim, Xdim)`, times must strictly
  increase, time axes are decoded by `decode_time`, and units in kg of
  another named species than the tracer's are an error (they were read with
  factor 1). `write_latlon_flux` writes static CF `(lon, lat)` flux files for
  `kind = "file"` (not a time-varying kind) and refuses a mesh whose faces
  the runtime would not rebuild bit for bit from the written centres with
  `GriddedIO.SnappedFaces` (pole-centred, truncated and Gaussian rows,
  regional axes across 0°; snapped Float32 grids such as GridFED 0.1° are
  written). `catrine_protocol_cs_fluxes.jl` writes through the
  `cs_native` writer: the same data, now with `time_bnds`, the geometry block
  and units `kg CO2|Rn222|SF6 m-2 s-1`; `compare_cs_flux_files.jl` takes the
  layout from `GriddedIO`.
- The runtime surface-flux and initial-condition loaders read their files
  through `GriddedIO`: coordinates by name with a check against the field's
  dimensions, orientation, fill values (read straight into the run's float
  type), the conservative-regridding source mesh
  (`source_latlon_mesh(…; radius)`, now with `SnappedFaces` faces; see
  Numerical changes),
  time axes and flux units. For every file the shipped configs read (102
  inventories: CAMS/LMDZ, GridFED, EDGAR, Zhang Rn-222, ECCO-Darwin, CATRINE and
  TRENDY `cs_native` series), the loaded fields, source meshes, slice times and
  storage factors were bit-identical to the former loaders at that step; the
  source meshes then moved to `SnappedFaces`. What changes:
  - Units go through one grammar (`GriddedIO.parse_flux_units`): grams, days,
    molar units with a named species and per-cell units with any time unit now
    convert; the conversion follows the units, not the kind (a
    `gridfed_fossil_co2` file is divided by the month length only if its units
    are per month, an `edgar_sf6` file by the cell area only if they are per
    cell). Molar units that name no species are moles of the tracer's
    species; missing and unknown units remain errors, now naming the variable
    and file. Units in kg of another
    named species than the tracer's (`surface_flux.species`) are an error;
    they were read with factor 1. Time-varying lat-lon sources need densities
    (per m²).
  - Time axes are classified by their units before any CFTime call
    (`GriddedIO.decode_time`): `months since` axes decode to month starts
    (CFTime used a fixed 30.44-day month), `noleap`/`365_day`,
    `all_leap`/`366_day` and `360_day` calendars are accepted (a stamp on a
    date the run's proleptic Gregorian clock lacks, such as 30 February, is an
    error), invalid origins are errors, `years since` and unlabelled axes are
    refused, and abbreviated units (`h since`, …) are decoded through
    `DateTime` (millisecond resolution).
  - The private helpers `_ic_find_coord`, `_build_source_latlon_mesh`,
    `_surface_flux_times_seconds`, `_parse_cf_time_origin`,
    `_normalize_units_string`, `_per_second_flux_scale`, `_days_in_month` and
    `_is_leap_year` are removed; the loaders `_load_file_surface_flux_field`
    and `_load_timevarying_surface_flux_field` take the `species` keyword.
- SIF-GPP, FLUXCOM, TRENDY v14, TRANSCOM, GFED, and CATRINE campaign
  preprocessing, run configs, visualization scripts, and ATBD memos.
- Per-month surface-flux units (GridFED `kgCO2/month/m2`) are converted with
  the month each slice covers on the file's CF time coordinate: the midpoint
  of its `bounds`, or its stamp without bounds, and that month's length in
  the coordinate's `calendar` (`360_day`, `noleap`/`365_day` and
  `all_leap`/`366_day` included). Bounds that do not span exactly that
  calendar month are an error. Before, slice `k` (or `time_index`) was month
  `k` of the year from `year`, the time units or the file name, and
  time-varying files needed exactly 12 slices; partial-year, out-of-order and
  non-Gregorian files are now read correctly. A static `month = m` selects
  the one slice whose time stamp lies in month `m` (none or several is an
  error); `time_index` is a positional slice index. Only the coordinate-less
  twelve-slice GridFED layout keeps the ordinal rule; other per-month files
  without a time coordinate are rejected. New in `GriddedIO`:
  `decode_month_seconds` and `decode_time_bounds`. Golden results are
  unchanged.

### Coverage, benchmarks, and documentation

- Faster test runs while developing; the full suite stays the merge check.
  `test/groups.toml` sorts every `test/core/` file and the regridding suite
  into exactly one of 21 topics (for example `advection`, `convection`,
  `metdrivers_binary`, `preprocessing`, `inversion`), each with the
  repository paths it covers, and names a quick set. `test/runtests.jl`
  gains selectors that run a subset of the default tiers: `--quick`,
  `--topics=a,b`, and `--changed[=<git-ref>]` (default `origin/main`), which
  maps the branch's and the uncommitted changes to topics and runs the full
  default tiers when a changed `src/` file maps to no topic; `--list` prints
  the selection. Selectors combine with each other and with the tier flags.
  Without flags the runner runs `core/` and `regridding/` as before; an
  unknown argument is now an error instead of being ignored.
  `test/core/test_test_groups.jl` checks the groups file. The quick set
  (Aqua, JET, the metadata gates, the groups and keyword-call checks, and
  physics smokes) took 4 minutes in one process on a lab machine, where the
  full default run took 32.
- `scripts/checks/check_keyword_calls.jl` parses `src/` for methods with
  required keywords (and `@kwdef` fields without defaults) and reports calls
  in `src/`, `ext/`, `test/`, `scripts/`, `benchmarking/`, `examples/`,
  `docs/literate/` and `docs/scripts/` that omit one, such as the
  transport-binary writers' `surface_pressure_basis`; on the tree before
  `5e00e94a` it reports the nine call sites that commit fixed. It runs in the
  quick set as `test/core/test_keyword_calls.jl`.
- `scripts/dev/quick_gate.sh [<git-ref>]` runs `git diff --check`, the
  keyword-call check, and `--quick --changed=<ref>` in one Julia process, and
  exits non-zero on any failure.
- CI adds a `Quick tests (Julia 1)` job that runs the quick set for early
  feedback. The full-suite jobs on Julia 1.10 and 1 are unchanged and remain
  the merge requirement.
- The `test/archived/` tests are deleted: they included preprocessing scripts
  that `cc63b87f` removed and a configuration that `71ff14f9` removed, so they
  could not load.
- The `test/orphan/` tier is gone, and with it the `--orphan` runner flag:
  its 18 files, which CI never ran, now run in `core/`. Fourteen moved there
  and load the cached package; `test_cmfmc_convection.jl` lost the
  hand-expanded checks of the two-term CMFMC tendency that `dfb7a7be`
  replaced, and `test_convection_forcing.jl` expects the four-slot
  capability tuple (cloud base, `4a5daf38`). The unique checks of
  `test_convection_types.jl`, `test_current_time.jl`,
  `test_gchp_vdiff_binary_payload.jl` and `test_surface_flux_operator.jl`
  moved into `test_transport_model_convection.jl`,
  `test_multiday_source_clock.jl`, `test_cs_optional_2d_sections.jl` and
  `test_per_tracer_flux_map.jl`. Newly in CI: the lat-lon chemistry
  operators, the CS Poisson projection cadence, the cubed-sphere driven
  runtime (per-window step schedule, Lin-Rood with diffusion and TM5
  convection, rejection of obsolete binaries and partial-day carryover),
  the ERA5 surface reader (ARCO layout, split CDS ZIP, latent heat flux),
  the Kz and stepwise fields, pressure-overlap remap, QV longitude
  normalization, the lat-lon `run_transport` recipe, the TM5 sparsity
  above the effective cloud top, and the surface-flux map and palindrome
  emissions. `test_era5_n320_to_c180_pipeline.jl`, which skips without the
  N320 archive, moved to `test/real_data/`, and `test_tm5_vs_cmfmc_parity.jl`
  keeps only its uniform-mixing-ratio checks (its conservation checks are
  tighter elsewhere).
- Documentation corrections from an external review against the TM5 Fortran
  by pera-noaa (NOAA GML), GitHub issues #12, #15 and #20. Thank you.
  - `SlopesScheme` is no longer described as a port of TM5's slopes
    advection (#12). It uses TM5's Russell-Lerner face flux but re-diagnoses
    a minmod slope from the neighbouring cell means every sweep, while TM5
    carries the first moments `rxm`, `rym`, `rzm` as prognostic state and
    clamps them only to `|rxm| <= rm`. The README, *Validation status* (the
    row moves from parity to "differs"), *Advection schemes*, the operator
    pages, and the `SlopesScheme`, `MonotoneLimiter` and limiter docstrings
    are corrected, and the README no longer cites the `test_tm5_*` convection
    tests for advection. *Advection schemes* gives the measured difference
    (after 10 revolutions of a 1-D Gaussian the `SlopesScheme` L2 error is
    4.6 to 6.8 times TM5's at Courant numbers 0.05 to 0.75) and lists the
    missing prognostic-moment scheme as a known limitation; building one is
    postponed. Runs with `scheme = "slopes"`, including the README showcase,
    are diagnosed-slope (van Leer) results.
  - The replay gate is described by what it certifies (#15, part 1).
    *Binary format* has a new section, *What the replay gate certifies*:
    every preprocessor that builds binaries from meteorological data derives
    `cm` from the stored horizontal fluxes just before the gate, so the gate
    checks column budgets, the `cm` closure, storage rounding, substep counts
    and window chaining, but cannot detect rotational flux errors,
    level-flipped fluxes, a wrong vertical split (including `cm` close to
    zero), a flux through the model top and surface that is the same at
    every interface of a column. *Binary format*, *Mass conservation*, the
    ERA5 preprocessing page and the TM5/GCHP pages now show the per-cell
    budget-term metric and current tolerances, and *Philosophy* no longer
    claims bit-level closure.
  - The adjoint pages list the current 4D-Var limits (#20):
    `cs_surface_flux_4dvar` runs one taped forward and reverse run per
    observation, so a gradient's cost grows linearly with the number of
    observations, and `bind_to_mesh` binds every observation to a column
    mean whatever its altitude (the schema's `alt` is above sea level and
    binaries carry no surface elevation). Both were documented only in
    `src/Inversion/README.md`.
- The Julia 1 CI job uploads line coverage of `src/` and `ext/` to Codecov.
  The upload authenticates with GitHub OIDC, so the repository holds no
  Codecov secret, and a failed upload does not fail CI. `codecov.yml` sets
  the status thresholds (project coverage may fall 2 percentage points below
  the base commit, coverage of the changed lines up to 20 points below it)
  and excludes tests, scripts, documentation, benchmarks, examples, and
  heritage folders. The hosted job runs only the
  CPU tiers, so the CUDA and Metal extensions count as uncovered.
- GPU benchmarks of the NVIDIA L40S (`Float32`), NVIDIA A100 (`Float32` and
  `Float64`), and Apple M5 Pro (Metal, `Float32`) lab machines reach the
  performance dashboard only through a lab lane started by hand on wurst
  (`benchmarking/lab/run_lab_benchmarks.jl`). Each lab machine's exact Julia
  version is pinned in `benchmarking/lab/machines.toml` (`julia_version`,
  1.12.6 on all three) and checked before anything runs. Every benchmark
  record carries an environment fingerprint (Julia version and sha256 of the
  resolved benchmark Manifest) in its raw `metadata` and its dashboard
  `extra`; the lab lane and the publish workflow refuse a lab run whose
  machines disagree on it, and the grouped dashboard shows it per point. A
  machine whose run fails still has its (partial) result files copied back
  before the failure is reported. The optional self-hosted CUDA and Metal jobs of the `Benchmarks`
  workflow, its `run_cuda`, `run_metal`, and `gpu_float_types` inputs, and
  `benchmarking/RUNNERS.md` are removed: self-hosted runners would run
  pull-request code on lab hardware of a public repository. The workflow's
  CPU job, which runs pull-request code, now has a read-only token; only the
  publish job writes (`contents`, `deployments`; no `pull-requests`).
- Benchmark harness: `--preset` loads a case matrix from
  `benchmarking/presets/` (the `lab` preset runs C90 and C180, L72 and L137,
  1 and 12 tracers: 44 cases per float type), and `--convection-depth`,
  `--device-label`, and `--commit` are new. GPU series names carry the
  device (`GPU-CUDA L40S`); CPU series names are unchanged. The `io` case
  now serializes every requested tracer; it moved one tracer whatever its
  name said. `adjoint` runs at one tracer only (the footprint of one scalar
  objective), so the `adjoint 4tr` CI series ends, and operators that ignore
  `--scheme` run once under the default `ppm`. `Operator Breakdown` series
  are named `Operator Breakdown / <case name> / <phase>` (the old names
  merged tracer counts, groups, and schemes), so those series start a new
  history, and the `io` and `adjoint` phase values are now per step: the
  adjoint value was divided by the repeat count twice, and the io value by
  `--steps` although a repeat is one round trip. The dashboard `extra` field
  is written as a JSON string; it was published as `[object Object]`. A
  case that throws is kept as `failed` in the raw JSON, and the run exits
  with an error after writing both files. The synthetic face mass fluxes
  are now steady (`benchmarking/src/synthetic_fields.jl`): the old fields
  disagreed at panel seams and had unbalanced top and bottom layers, so
  repeated steps drained edge cells by up to 11 % of their mass per step
  until the Courant number exploded (the first lab case, C180 L72, stopped
  at the 4096-subcycle limit). The new fields leave the air mass unchanged
  at a palindrome Courant number of about 0.1 on every grid, and the
  harness fails a case whose Courant number exceeds 0.5 or whose air mass
  drifts. The lab preset now runs C90, C180, C360, and C720 at L72 and
  L137 with 2 tracers. A typed `[limits]` table keeps `io` and `adjoint` at
  C180 and below, which gives 40 cases per float type.
  Before building a case, the harness estimates its device memory from a
  linear model of the model's arrays, fitted on CPU
  (`benchmarking/src/memory_model.jl`). A case that does not fit, or that
  runs out of memory, is recorded as `skipped_memory` with the required and
  available GiB instead of failing the run; skipped cases are not published,
  and the lab lane accepts and lists them. `--device-memory-gib` overrides
  the device query. Metal.jl 1.9 only prints a failed command buffer, so
  the harness captures standard output while a Metal case runs. A sentinel
  command buffer's completion handler marks the point where all of the
  case's handlers have reported. If every failed buffer ran out of memory,
  the case is `skipped_memory`; any other failure makes it `failed`. A
  kernel that never ran therefore cannot be published as ok. The CPU `io` case no longer copies its host payload to serialize
  it; on the CPU its estimate counts the read-back copy. The grouped dashboard draws one
  line per group, grid, levels, scheme, and device; lines of renamed groups
  no longer merge.
- The README opens with badges for CI, the stable and development manuals,
  the license, the Julia compatibility bound (read from `Project.toml`),
  Aqua, Codecov, and the performance dashboard.
- New manual page *Performance benchmarks* (section *Development*): what the
  synthetic cubed-sphere benchmarks measure, the metric definitions
  (time per step, cell-tracer updates per second), the lab machines and their
  roles, how results reach the dashboard, how to run a benchmark locally, and
  how to read the dashboard. `CONTRIBUTING.md` explains the benchmarks and
  coverage.
- `CITATION.cff` and GitHub issue templates (bug report, feature request)
  are new, and the README labels each feature stable or
  preview/experimental, following *Validation status*.
- `LinRoodPPMScheme{5}` was documented as Huynh-constrained PPM. It uses
  Colella–Woodward fourth-order edges with extremum flattening, without
  Huynh's second constraint; `ORD = 7` adds the cubed-sphere face
  treatment. The docstrings say so.
- The JET hot-path baseline on Julia 1.12 and later is 115 reports (was
  153); the Julia 1.10 baseline stays 130. A core test checks that every
  `scripts/` path cited in `src/`, docs and configs exists.
- `config/runs/catrine5d/c180_full137_f32_pbl_tm5_smoke_gpu.toml` moved to
  `config/runs/likely_legacy/` (as
  `catrine5d_c180_full137_f32_pbl_tm5_smoke_gpu.toml`): no compatible
  binary remains at its input path.
- The golden-output harness gains `pre_geosit_c45`, its first GEOS native
  preprocessing case: one GEOS-IT C180 day (2021-12-01) coarsened to C45
  from staged, checksummed inputs, with default column weights. The
  reference set has 28 cases.
- Julia 1.10 remains the minimum supported version (`julia = "1.10"`).
  CI runs the full suite on Julia 1.10 and on the current release
  (`'1'`, Julia 1.13 since #32).

### Post-run diagnostics

- New module `AtmosTransport.Analysis` (loaded after `Models`): the shared
  numerics of post-run diagnostics. `daily_snapshot_series` concatenates the
  frames of snapshot files (run origin from the CF time units, or an explicit
  `origin` for legacy files; a repeated hour is read once);
  `EqualAreaBands` and `band_means` form area-weighted means on bands equally
  spaced in sin φ, leaving NaN cells out of numerator and denominator;
  `doy_linear_detrend` removes a per-calendar-day line across years (29
  February folded onto 28 February); `seasonal_trend_decomposition` fits 12
  calendar-month intercepts and a trend on the observed months (minimum norm);
  `skill_metrics`, `weighted_fit_stats`, `column_band_weights` and
  `logp_interpolate` complete it. On the TRENDY C30 ranking inputs the monthly
  band means equal the Python ones bit for bit and the decomposition agrees to
  6e-14 ppm with the same ranks; on the SIF-GPP control the printed statistics
  agree to every digit. `test/diagnostics/test_trendy_hovmoller.py` is ported
  to `test/core/test_analysis_decomposition.jl`;
  `scripts/diagnostics/catrine_station_timeseries.py` moves to `heritage/`;
  `compare_cs_flux_files.jl` takes its sin-latitude bands from `Analysis`
  (the same assignment as before on C4, C8, C30, C90, C180, C360 and C720 in
  both panel conventions).
- Budget closure in Julia. `Analysis.budget_closure(run)` checks every
  tracer of a finished run: the change of `<tracer>_total_mass` between the
  first and last snapshot against the storage the run's own surface sources
  add (`build_surface_flux_sources` + `integrated_emission`, over the run's
  own steps: each window's step count and the float-type step
  `FT(L)/FT(n)` from the run clock's step starts, so a Float32 window of
  3600 s in 7 steps adds 3599.99994 s), within
  ρ|E| + k·eps(Float32(carrier))·air mass (`BudgetTolerance`: strict 1e-4 and
  1 ulp, operational 1e-3 and 4 ulps), with finite values and, for a positive
  carrier, positive column means. `RunContext` builds the run's view with the
  runner's own functions (tracer specs, including the legacy single tracer of
  lat-lon runs, now read by the structured runner and `RunContext` through one
  function `_structured_tracer_specs`; grid; every binary's window layout; run
  start; the air mass of window 1 that the initial state is packed on; output
  file names; window ranges and tracer tables the runner rejects are errors,
  with the runner's multi-file range check shared as
  `_check_multifile_window_count`), and `carrier_drawdown`
  gives the largest cumulative loss of global mixing ratio before a run.
  `scripts/diagnostics/tracer_budget_closure.jl series <run.toml>` replaces
  `check_trendy_npp_rh_run.py` (JSON report, exit 1 when the gated level
  fails); `scripts/run_trendy_npp_rh_campaign.py` calls it. On the 23-model
  TRENDY C30 run (3775 days) the harness emulator of the Python arithmetic
  (`scripts/validation/flux_port_parity/budget_closure_parity.jl`) reproduces
  `validation.json` (expected changes to 7e-16, actual changes identical, TEM
  NPP failing at 1.2714e-4, 45 of 46 passing) and `smoke_validation.json`
  (max 1.96e-4). With the current `DRY_AIR_MOLAR_MASS` (the run used the
  pre-`d552d3e0` 0.02896546) every expected change is 3.66e-5 smaller and 43
  of 46 pass (TEM, CLM-FATES and ELM NPP fail); at the run's constant the
  runtime sources differ from the Python sums by at most 5.5e-8, so cell
  areas and integration explain none of the TEM excess. The NPP drawdowns
  equal the Python audits times DRY_AIR_MOLAR_MASS/0.02896546 to 5.5e-8. The
  single-file mode of `tracer_budget_closure.jl` read the bare-hours time axis
  of current snapshots as years and reported a gap of about −1; it now reads
  hours since the run start.
- `scripts/diagnostics/catrine_true_mass_balance_gc.py` is replaced by
  `catrine_true_mass_balance.jl --model geoschem`, which writes the same csv
  columns as the Python script and the AtmosTransport mode. The GEOS-Chem
  CATRINE output is read through `Analysis.GEOSChemCATRINEArchive`
  (`gc_path`, flat or `YYYY/MM/`; `gc_times`; `with_gc_file`;
  `global_totals`; `unreadable_files`): files are read into memory on all
  threads and decoded under the NetCDF lock, and an unreadable file is
  listed, its row skipped and its interval given the next readable rate, as
  before. New constants `GEOSCHEM_SPECIES_MOLAR_MASS` (GEOS-Chem species
  database) and `RN222_HALF_LIFE` (3.8235 d, exactly the `330350.4` s of the
  run configs). On the C90 archive (6088 files, 2021-12-01 03Z to
  2024-01-01 00Z) the hours equal the Python csv, dry-air masses and burdens
  agree to 5e-15 and cumulative sources to 1e-15 relative (co2_natural, whose
  cumulative source crosses zero, to 6e-16 of its largest value), residuals to
  8e-15 of the burden, and the four summary lines to every printed digit; on
  a C180 window with a truncated file the unreadable-file handling gives the
  same rows and summary. The AtmosTransport mode is unchanged (bit-identical
  csv and log on the 25-month ERA5 fossil CO₂ run) and is now tested on a
  tiny run, where a stable and a decaying tracer close to 1e-12.
- Float32 diagnostics in Julia. `scripts/diagnostics/float32_conservation_attribution.jl`
  replaces the Python script: `write-configs` builds the 16 attribution
  configs as Dicts, checks them with `validate_config` and writes them with
  `TOML.print` (equal, as parsed Dicts, to the Python ones); `budget` prints
  the same tables as the Python script, character for character, on the runs
  behind `docs/src/theory/float32_conservation.md`, with the run tag templates
  (per archive and precision) and snapshot file prefix as arguments. That
  page's Reproducing section now gives the command that prints its "after"
  rows (`full_final2` against `full_final`), and the SF₆ entry of the MERRA-2
  row "all operators, after" is corrected to −3.7e-9 (it was −3.747e-9 rounded
  twice).
  `scripts/diagnostics/compare_float32_float64_runs.jl` replaces the numbers of
  the Python comparison and writes `float32_vs_float64.json` (same keys; the
  16 values agree to 1e-14 on the December 2021 and the 761-day MERRA-2 twins)
  and `float32_vs_float64.nc`; the figures moved to
  `scripts/visualization/plot_float32_vs_float64.py`, which reads that file.
  New in `Analysis`: `drift_statistics`, `twin_relative_difference`,
  `weighted_difference_stats` (compensated sums), and `keep_frame` for
  `daily_snapshot_series`. Column means are now sampled at every snapshot of
  the requested UTC hour (the Python script took one frame per file, the
  initial state from the first file, and so skipped 00 UTC of the second day),
  and cells with fill values are left out of the daily statistics and counted.
- `Analysis.daily_snapshot_series` takes the run origin from the CF units;
  `origin` is now optional and needed only for legacy files whose `time`
  units are the bare `"hours"` (an `origin` given for a current file must
  match it, and all files must share one origin). `Visualization.open_snapshot`
  gives `DateTime`s in `snapshot_times`, `HorizontalField.time` and
  `RasterField.time` for current files, and `fieldview`/`frame_indices` also
  accept a `DateTime`; integer frame indices and numeric elapsed hours select
  frames as before, and legacy files keep numeric times. `SnapshotDataset`,
  `HorizontalField` and `RasterField` gain a time type parameter, and
  `SnapshotDataset` an `hours` field. `compare_float32_float64_runs.jl`
  checks the encoded origin against `--origin`, `compare_cs_vs_ll.jl`
  compares decoded times, and `catrine_true_mass_balance.jl` reads the stored
  hours.

### Verification and remaining limits

- `test/core/test_slopes_tm5_reference.jl` checks `SlopesScheme` against an
  independent transcription of TM5's `dynamu` (`advectx.F90`) in
  `test/helpers/tm5_dynamu_reference.jl`: with the moment re-diagnosed by
  minmod the TM5 update equals `SlopesScheme()`, and with the moment set to
  zero `UpwindScheme()`, to rounding (non-negative and signed fields,
  eastward, westward and divergent flow). With the prognostic moment, TM5
  keeps 0.990 of a Gaussian's peak after one revolution against 0.892, and
  TM5's positivity clamp is not offset-equivariant while `SlopesScheme` is.
  The transcription itself keeps a uniform field exact, conserves mass and
  converges at second order.
- `test/core/test_replay_consistency.jl` has a *Replay gate scope* testset
  for the statements in *What the replay gate certifies*: a thin cell's
  residual is measured against that cell's own budget terms, and with `cm`
  re-diagnosed as the preprocessors do it a rotational flux error, an
  `am`/`bm` level flip and a column-uniform `cm` offset pass while a broken
  column budget fails.
- The golden-output harness (`test/golden/run_goldens.jl`) runs its cases in
  worker sessions instead of one Julia process per case: one process
  (`golden_worker.jl`) per group of cases with the same entry point,
  `[architecture]` table and threads, which includes the entry scripts once
  and calls their `main(args)` for each case, with the case's output and log
  messages in its `log.txt` and the logger, working directory and environment
  restored after it. A full check of the 27 cases takes 41 minutes instead of
  68 on an L40S (preprocessing 16 instead of 18, CPU runs 16 instead of 23,
  GPU runs 8 instead of 27), with outputs bitwise identical to the
  per-process reference. The harness watches each worker: it writes a
  case's `status.toml` as soon as the case finishes (an interrupted run
  keeps them), continues the remaining cases in a new worker when one
  crashes, stops a worker whose case runs longer than `--timeout` minutes
  (default 45) and fails that case, and starts a new worker when one reports
  a poisoned session (a CUDA `synchronize` or `reclaim` that fails after a
  case, the mark of a sticky device error). `status.toml` records the case's
  session and position (`[session]`, with `poisoned` when it applies) and
  the `error` of a failure; `check` prints wall times as `(own process)`,
  `(session case k of n)` or `failed` and does not compare them.
  `scripts/run_transport.jl` and
  `scripts/preprocessing/preprocess_transport_binary.jl` define
  `main(args)`, which runs in the calling Julia session with a logger scoped
  to the call (the scripts set the global logger before) and throws on error;
  the thread re-execution and GPU preload of `run_transport.jl` happen only
  when it runs as a script. `ATMOSTR_PROFILE_MODE` now follows the config's
  `[architecture]` instead of whether CUDA happens to be loaded: a CUDA run
  is profiled also on the first call of a session that included the script
  (CUDA is loaded for it), and a CPU or Metal run warns and runs without a
  profile. `preprocess_transport_binary.jl` accepts `-h`/`--help`.
  `test/core/test_golden_sessions.jl` tests the grouping, the worker's
  isolation, crashes before and between cases (also with exit status 0), a
  poisoned session, timeouts while loading, in a case and in the cleanup
  after a case, SIGKILL after an ignored SIGTERM, and `main(["--help"])` of
  both entry scripts as a session includes them.
- New core tests cover config parsing, cell location on every topology
  against independent references, source readers (including a real OCO-2
  Lite file check), the device gather, pressure and height
  reconstruction on GEOS L72, the sampler's time bookkeeping, and end-to-end
  lat-lon and cubed-sphere runs, including identical results across a
  binary handoff. An opt-in CUDA test covers the gather on an L40S.
- Station placement uses a constant temperature except on cubed-sphere
  binaries with GCHP VDIFF temperatures. Averaging kernels are applied
  offline.

## 0.4.0 — 2026-09-06

This release changes cubed-sphere numerical results and runtime/output
behavior. Keep the version, forcing binary, run TOML and dependency manifest
with reproducibility records; recheck conservation and scientific diagnostics
when upgrading from 0.3.0.

### Numerical changes

- Cubed-sphere split advection and Lin–Rood now share conservative face
  transfers across panel seams. Previous runs could accumulate tracer mass
  drift at those seams.
- Precomputed TM5 Dkg diffusion advances conservative tracer storage with
  bidiagonal factors. It preserves weak exchange into empty layers, isolated
  layers and signed totals more accurately. Its adjoint uses the matching
  transpose. This is a numerical correction, so old and new output need not
  match bit for bit.
- CUDA PPM launch tiles, separate Dkg factor/tracer launches and batched
  convection solves reduce work without lowering configured precision.
  CUDA collaborative Float64 convection supports full columns through 73
  levels within its shared-memory budget; Float32 retains its portable depth
  limit. Explicit `lmax_conv`/`n_merge` choices remain numerical approximations.
- Lin–Rood recording and reverse halo propagation use backend kernels and
  pass transporting CUDA adjoint checks with scalar indexing disabled.

### Runtime and output

- Input drivers and device workspaces are reused across daily files. Rolling
  staging owns its cache and checks retained source identity; failed prefetch
  and output tasks are consumed during cleanup.
- Single-file NetCDF output streams selected snapshots and records
  `completed_snapshots`. A failed run may leave partial output; file existence
  is not evidence of completion. Reopening a stream to resume is unsupported.
- Selected NetCDF output avoids retaining full tracer volumes on the host.
  Signed Float64 `<tracer>_total_mass` diagnostics are captured independently
  of spatial output precision. Metal uses bounded host slabs for Float64
  totals and column accumulation; CUDA retains device Float64 reductions.
- `SnapshotFrame` and `SnapshotWriteOptions` are part of the curated top-level
  API. Write and resource-close failures are preserved together rather than
  allowing one to mask the other.
- Runtime preflight reports malformed tables, including `[input.staging]`
  and tracer subtables, before opening binaries. Window indices must be
  integers. Multiple input files require their complete window ranges;
  omit `stop_window` for a full multi-day run. Nine obsolete 48-hour configs
  now live under `config/runs/likely_legacy/`.

### Dependencies and documentation

- Add explicit `FileWatching` and `HDF5_jll` dependencies for staging ownership
  and NetCDF error-handler handling. HDF5_jll compatibility accepts 1.14 and 2.
- Remove self-version bounds from test and benchmark environments so release
  bumps do not invalidate them. Julia 1.10 remains the minimum supported version.
- Add an executed emission-footprint tutorial with a finite-difference check
  and a beginner's guide to adjoints, observations, priors and inversion.
  Correct preprocessing configuration/balance explanations and distinguish
  hosted CPU CI from separate GPU verification.

### Verification and remaining limits

The branch records Julia 1.10/1.12 CPU test runs, strict documentation builds,
all ten maintained L40S GPU diagnostics, V100 adjoint/output checks, and
real-input Float32 C90/L66 comparisons. The matched L40S day benchmark improves
from 7.35 to 4.54 seconds for six tracers and from 47.47 to 13.84 seconds for
32 tracers. The 32-tracer 0.3.0 baseline uses its supported legacy solver,
because that release's collaborative solver is capped at six tracers. These
measurements include runtime setup and I/O and apply to the documented workload;
see [benchmark evidence](scripts/benchmarks/results/release_fp32_l40s_20260906/README.md).

The C90/L66 Float32 forward smoke test passed on an Apple M5 Pro (20 GPU
cores), with six and 32 tracers, full TM5 collaborative convection, Dkg
diffusion and column output. Warmed runs took 2.90 and 8.28 seconds; maximum
column relative L2 differences from CUDA were below `9e-8` and relative mass
drift below `5.7e-8`. See the [Metal verification record](docs/memos/release_readiness_20260906.md).
This does not establish coverage of Metal adjoints or every operator. Optimized,
clamped or reduced-column convection paths do not all have supported adjoints.
The shipped inversion CLI remains synthetic-only; real-data inversion assembly
and external TM5-4DVAR cross-validation are separate tasks. Numerical checks
and performance measurements are not a complete observational validation.

## 0.3.0

The preceding release is the baseline for the changes above. This changelog
was introduced for 0.4.0; it does not reconstruct earlier release histories.
