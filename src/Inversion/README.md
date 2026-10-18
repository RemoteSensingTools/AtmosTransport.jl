# Inversion

Cubed-sphere surface-flux inversion: observations, Jacobians, the 4D-Var
cost and gradient, background covariance, preconditioning, and optimizers.

This folder builds on the footprint machinery in [`../Footprint/`](../Footprint/).
Controls are surface emission-rate fields grouped into named time windows.
Observations are scalar objectives (layer means, column means, in-situ
point samples placed in the column, and averaging-kernel columns of satellite
and TCCON retrievals) sampled after one model step or averaged over the steps
of a time window. Gradients come from one reverse-mode sweep
that carries every observation's weighted departure, not from finite
differences. Observations and departures can be read from and written to
versioned NetCDF schemas (v2).

## How It Is Loaded

`Inversion/` is **not a module**. [`../Adjoints/Adjoints.jl`](../Adjoints/Adjoints.jl)
`include`s these files into `module Adjoints`, and public names are exported
from the `Adjoints.jl` export block. The files load in two groups:

1. Early, right after `ObjectiveSeeding.jl`, `FootprintResult.jl` and
   `StepForcing.jl`: `Observations.jl` -> `ObservationColumns.jl` ->
   `ObservationsIO.jl` -> `ObservationBinding.jl` -> `DeparturesIO.jl` ->
   `OCO2MIPObservations.jl` -> `Covariance.jl` -> `Preconditioning.jl`.
2. Last, after `../Footprint/FootprintAPI.jl`: `Jacobian.jl` ->
   `CostGradient.jl` -> `Optimizer.jl`.

`NCDatasets` is imported in `Adjoints.jl`. `FFTW` (Covariance), `Optim`
(Optimizer), and `Dates` (ObservationBinding) are imported in the files that
use them. `bind_to_mesh` uses the forward observation sampler's cell lookup
(`Output.cell_locator` / `locate`) and placement rule
(`Output.resolve_placement`, `VerticalPlacementSpec`, `ObservationTerrain`,
the unplaced policy), and placed point observations choose their layer with
`Output.sample_column_layer!`. Averaging-kernel columns use the shared
operator of `Output` (`RetrievalColumn` with its mapping `LayerRebin` or
`HatLevelProjection`, `retrieval_column_weights!`, `retrieval_prior_offset`,
`retrieval_profile!`, `retrieval_model_error_variance`, in
`../Output/observations/retrieval_column.jl`) on the interface pressures of
`Output.placement_pressures!`, as the forward sampler does for OCO-2
soundings; the OCO-2 MIP converter reads the retrieval fields with
`Output.read_oco2_retrieval_columns`. Terrain files are read by the caller
(`GriddedIO`, loaded after `Adjoints`) and passed in as an
`ObservationTerrain`.

## Entry Points

- [`CostGradient.jl`](CostGradient.jl): `cs_surface_flux_4dvar(panels_rm0,
  panels_m0, panels_am_steps, panels_bm_steps, panels_cm_steps, mesh,
  observations, controls; scheme, ..., tape_storage, tape_path, checkpoint,
  preconditioner, column_met)` returns a `CS4DVarResult` (with one
  `CSObservationStencil` per observation). `scheme` is required so an
  inversion cannot silently choose a different forward/adjoint advection
  operator; `column_met` (a `CSObservationColumnMet`) is required when any
  observation is a `CSPlacedPointObjective` or a
  `CSRetrievalColumnObjective`.
- [`Optimizer.jl`](Optimizer.jl): `cs_surface_flux_4dvar_optimize(...;
  optimizer)` and `cs_surface_flux_4dvar_solve(opt, cost_fn, controls)`.
  Backends are `CSGradientDescent` and `CSLBFGS` (`Optim.LBFGS`).
- [`Jacobian.jl`](Jacobian.jl): `cs_surface_flux_jacobian(..., objectives,
  windows; kwargs...)` returns a `CSSurfaceFluxJacobianResult`; `kwargs` are
  those of `cs_surface_emission_footprint`.
- [`Covariance.jl`](Covariance.jl) / [`Preconditioning.jl`](Preconditioning.jl):
  `DiagonalCSCovariance`, `IsotropicGaussianCSCovariance`,
  `apply_B_half!` and its `_adjoint!` and `_inverse!` forms;
  `CSSurfaceFluxPreconditioner(covariance, background, optim_type)` with
  `LinearOptimType` or `LogNormalOptimType`; and `apply_preconditioner!`
  with its `_inverse!`, `_tangent!`, and `_adjoint!` forms.
- File IO: `read_observations` / `write_observations` (schema v2, with the
  optional retrieval block of averaging-kernel columns, layer or level
  records), then `bind_to_mesh(set, mesh, t_start, dt; nsteps, tracer_filter,
  out_of_range_policy, placement, terrain, max_surface_pressure_mismatch_pa,
  flagged_records, cell_averaging)` (records to `CSObservation`s; flagged
  records dropped or co-sampled, column records optionally averaged per
  cell), then `build_departure_set(set, result::CS4DVarResult, mesh, t_start,
  dt, nsteps)` and `read_departures` / `write_departures` (schema v2).
- OCO-2/OCO-3 v11 MIP 10-second files: `read_oco2_mip_observations(path;
  time_range, data_types, assimilate_flags, model_sigma, mapping)` (a
  `CSObservationSet` with level blocks, every record, flags for the ones not
  assimilated) and `convert_oco2_mip_observations(path, output; partition)`
  (daily or monthly observation files;
  `scripts/preprocessing/oco2_mip_to_observations.jl` is the CLI).
  `inflate_cell_model_error(set, mesh, t_start, dt; ...)` is the
  alternative to cell averaging that keeps every record.

## File Map

- [`Observations.jl`](Observations.jl) — window, time sampling
  (`CSTimeSampling`), observation (with `assimilate`: `false` for a
  co-sample), control, stencil (`CSObservationStencil`), result, and
  iteration-log types
- [`ObservationColumns.jl`](ObservationColumns.jl) — `CSObservationColumnMet`
  (model top, gravity, layer temperature, humidity; static or per step), the
  per-step layer of a placed point observation (`_placed_level`),
  `_sample_objective` (value, linear functional, level and averaging-kernel
  diagnostics of any objective on a replayed state; the averaging-kernel
  column's weights `H_k/max(m_k, ε)` on its cell's total interface
  pressures; a cell average's weighted members), and the surface-pressure
  screen of averaging-kernel columns (`_screen_surface_pressure`; co-samples
  are never screened)
- [`ObservationsIO.jl`](ObservationsIO.jl) — `CSObservationRecord` (with
  `PointObservationKind` / `ColumnObservationKind` and an optional
  `retrieval::Output.RetrievalColumn`), `CSObservationSet`, and the NetCDF
  reader/writer for `schemas/cs_observations_v2.toml`, including the optional
  NaN-padded retrieval block with each record's `retrieval_mapping` (units,
  ordering and padding checks; v1 files are rejected)
- [`ObservationBinding.jl`](ObservationBinding.jl) — `bind_to_mesh`: time to
  a `CSTimeSampling` (instantaneous or Δt/ΣΔt window), (lat, lon) to the
  containing `(panel, i, j)`, point records to `CSPlacedPointObjective`s with
  the forward placement rule, column records with a retrieval block to
  `CSRetrievalColumnObjective`s (pressure basis, surface-pressure limit and
  unplaced policy of the placement), other column records to column means;
  the out-of-range, unplaced and flagged-record policies (`DropFlaggedRecords`,
  `CosampleFlaggedRecords`), cell averaging (`CellAverage`,
  `CSCellAverageObjective`) and `inflate_cell_model_error`
- [`DeparturesIO.jl`](DeparturesIO.jl) — `CSDepartureRecord`,
  `CSDepartureSet`, `build_departure_set` (joins a 4D-Var result to the
  records by id), and strict v2 NetCDF IO for `schemas/cs_departures_v2.toml`
  (sampled steps, weights and levels as a ragged `sample` dimension; the
  optional block of averaging-kernel columns: surface-pressure mismatch,
  retrieval-level model profile, profile model sigma; `assimilated`, the
  record `flag` of co-samples and `cell_average_id`)
- [`OCO2MIPObservations.jl`](OCO2MIPObservations.jl) — the OCO-2/OCO-3 v11
  MIP 10-second files to schema v2: field mapping, units (ppm → mol/mol,
  hPa → Pa), flags from the MIP `assimilate_flag` and data types, the hat or
  TM5 mapping, daily or monthly files
- [`Covariance.jl`](Covariance.jl) — `B^(1/2) = D·L`, built per panel
  from the FFT spectral square root of a periodic Gaussian
- [`Preconditioning.jl`](Preconditioning.jl) — χ <-> x transforms and their
  tangent and adjoint
- [`Jacobian.jl`](Jacobian.jl) — per-window footprint aggregation
  (`_aggregate_surface_window`) and `cs_surface_flux_jacobian` (one tape,
  one reverse walk per objective under `FullCheckpoint`)
- [`CostGradient.jl`](CostGradient.jl) — control validation and rate
  assembly, diagonal background term, preconditioned path, and
  `cs_surface_flux_4dvar` (forward run, weighted departures of the
  assimilated observations, one taped run and one reverse pass with the step
  forcing)
- [`Optimizer.jl`](Optimizer.jl) — `AbstractCSOptimizer` backends and
  `cs_surface_flux_4dvar_optimize`

## Common Tasks

- Adding an optimizer: subtype `AbstractCSOptimizer` and implement
  `cs_surface_flux_4dvar_solve(opt, cost_fn, controls)`, where
  `cost_fn(controls)` returns a `CS4DVarResult`.
- Adding a covariance: subtype `AbstractCSSurfaceFluxCovariance{FT, A}`, add
  an `Nc` field (the preconditioner reads `covariance.Nc`), and implement
  `apply_B_half!`, `apply_B_half_adjoint!`, and `apply_B_half_inverse!`.
- Changing how observations reach the model: time, space and vertical
  mapping is in `bind_to_mesh`; sampling during the forward run (simulated
  values, stencils and each observation's linear functional) is
  `_observe_cs_forward` in `../Footprint/ReverseLoop.jl`, which calls
  `_sample_objective` once per sampled step; a new observation operator needs
  a `_sample_objective` method (or `evaluate_objective` and
  `_linear_functional` methods, `../Footprint/StepForcing.jl`) and a
  `_stencil_placement` method.
- Changing the vertical placement rule: edit `Output.resolve_placement` /
  `Output.sample_column_layer!`; the forward sampler and the inversion change
  together, and the parity test in `test_cs_4dvar_placed_observations.jl`
  guards them.
- Changing the averaging-kernel operator: edit
  `../Output/observations/retrieval_column.jl`; the TM5 reference values and
  the transcription in `test_cs_4dvar_retrieval_columns.jl` guard it.
- Running an end-to-end inversion: `scripts/inversions/cs_4dvar.jl` with
  `config/inversions/*.toml`.

## Conventions

- Cost: `J = Σ 0.5 ((H(x) − y)/σ)²` plus the background term. Residual and
  departure are `simulated − observed`. The departure file records this in
  its `departure_sign_convention` attribute.
- Gradient norm: each backend uses one convention consistently for its
  tolerance, `gradient_norm_history`, and `CSIterationLogEntry.gradient_norm`.
  `CSGradientDescent` uses the Euclidean (L2) norm; `CSLBFGS` uses Optim's
  L-infinity norm, preserving the established meaning of its
  `gradient_tolerance`. The norm is over all controls, panels, and cells, in
  physical x-space or preconditioned χ-space according to the active mode.
- Preconditioned mode (`preconditioner` set): `controls[k].value` is χ,
  `J = 0.5‖χ‖² + J_obs(T(χ))`, and the gradient is `χ + T'(χ)ᵀ ∇ₓJ_obs`.
  Per-control `background` and `sigma` are ignored in this mode.
- One gradient is one adjoint integration, whatever the number of
  observations. Each observation is a fixed linear functional of the tracer
  mass after the steps it samples (`CSLinearFunctional`, one step per entry,
  weights from the replayed air mass times the time-sampling weights).
  `cs_surface_flux_4dvar` runs the forward model once for the simulated
  values, stencils and functionals, scales each functional by
  `(H(x) − y)/σ²`, tapes one run to the last sampled step, and walks it back
  once, adding each step's scaled weights at the step's end-of-step marker
  (after convection, before its transpose). Control-window steps after the
  last sampled step get a zero gradient. `checkpoint`, `tape_storage` and
  `tape_path` apply to that one taped run. Instantaneous observations give
  bit-identical results to the single-step functionals they replace.
- `bind_to_mesh` represents step `k` = `[t_start + (k−1)dt, t_start + k dt)`
  by the state at its end. A record without a window samples the step
  containing its time; a record with `time_window_s = W` samples the steps
  overlapping `[t − W/2, t + W/2]`, weighted by their overlap, truncated at
  the run's ends and renormalised (TM5-4DVAR's Δt/ΣΔt mean; a step partly in
  the window counts its overlap). Window records need `nsteps`. The record
  time decides whether it is in the run; `ClampOutOfRange` keeps an outside
  record (the part of its window inside the run, else the first or last
  step; departure rows mark it `clamped`), and skipped and clamped records
  are counted in an `@info`. Records with a nonzero `flag` are dropped, or
  co-sampled with `flagged_records = "cosample"` (`assimilate = false`:
  simulated and reported, never in the cost or the gradient, never extending
  the taped run); record ids are unique (checked by `CSObservationSet`, `read_observations`
  and `bind_to_mesh`); `sigma` is `√(value_sigma² + model_sigma²)`;
  observations carry the record `id`.
- This time sampling differs from the forward sampler's (window ends,
  linearly blended or nearest): the inversion samples the end of the model
  step containing the record time, up to `dt` later, or the window mean.
  Parity with the sampler holds for the same sampled state.
- A placed point observation's layer is resolved at every sampled state from
  that state's replayed air mass (dry interfaces, hypsometric heights,
  total or dry interface pressures), never fixed at binding: the vertical
  stencil follows the meteorology, as TM5 recomputes it from `gph` per step,
  and every vertical coordinate (height and pressure) is in the adjoint.
- An averaging-kernel column (`CSRetrievalColumnObjective`) builds its
  weights at every sampled state: the cell's interface pressures of its
  pressure basis (total from the humidity by default) stretched onto the
  retrieval column and mapped onto its values by the mapping's projection
  (TM5-4DVAR's rebin for layers; for levels, the dry-air L2 projection onto
  the hats, from the replayed dry air mass), ``X = Σₖ Hₖ xₖ + Σᵢ hᵢ(1 − aᵢ) x_a,ᵢ``
  on the dry mixing ratios; its
  functional has weights `Hₖ/max(mₖ, ε)` on every level and the prior term as
  offset. Its model-error term `Σᵢ (hᵢ aᵢ σᵢ)²` (a record with
  `retrieval_profile_sigma`) is added to σ² at binding: it depends on the record
  only, so it is frozen for the inversion. A record whose time-weighted
  model minus retrieval surface pressure exceeds
  `max_surface_pressure_mismatch_pa` follows the unplaced policy at the
  forward replay: an error, or (`SkipUnplaced`) reported with
  `assimilated = false` in its stencil and left out of the cost and the
  gradient (the same records in every iteration; counted in one log
  message).
- Precision: the model `FT` comes from the panels. Window weights are
  stored as `Float64` and converted to `FT` when used.
  `IsotropicGaussianCSCovariance` floors each 1-D spectrum at
  `sqrt(eps(FT))` times its maximum, so `B^(-1/2)` stays finite in Float32.
- Backends: `IsotropicGaussianCSCovariance` uses CPU FFTW plans and `Matrix`
  scratch. `CSLBFGS` flattens controls into a host `Vector{FT}`.
  `DiagonalCSCovariance` uses broadcasts only. The isotropic covariance and
  `CSSurfaceFluxPreconditioner` reuse internal scratch buffers, so they are
  not thread-safe.
- Evaluation count: `CSLBFGS` gives Optim one objective whose value, gradient
  and combined methods share a cache of the last evaluation (`_CSCostCache`),
  so `cost_fn` (forward plus adjoint) runs once per point Optim visits. The
  initial point and the returned result reuse those evaluations.
- Covariance limitation: there is no correlation across panels; each panel
  is smoothed as if it were periodic.

## Related Docs And Tests

- [`../../docs/src/theory/adjoint_status.md`](../../docs/src/theory/adjoint_status.md),
  [`../../docs/src/for_tm5_gchp_users/adjoints.md`](../../docs/src/for_tm5_gchp_users/adjoints.md)
- All tests below are in [`../../test/core/`](../../test/core/).
  - Adjoint and gradient identities (TM5-4DVAR test equations):
    `test_adjoint_identity_model_space.jl`,
    `test_adjoint_identity_preconditioned.jl`, and
    `test_gradient_taylor_sweep.jl`.
  - Components: `test_cs_covariance.jl`, `test_cs_preconditioning.jl`,
    `test_cs_4dvar_preconditioned.jl`, `test_cs_optimizer_dispatch.jl`,
    `test_cs_lbfgs.jl`, and `test_cs_iteration_log.jl`.
  - IO and binding: `test_cs_observations_io.jl`,
    `test_cs_observation_binding.jl` (time windows, Float64 edge points,
    placement, unplaced policy), and `test_cs_departures_io.jl`.
  - Placed point and window observations: `test_cs_4dvar_placed_observations.jl`
    (parity with the forward sampler, 10 m vs 5 km samples, total vs dry
    pressure, per-step layers, window = Δt-weighted single steps, single
    sweep vs per-observation footprints, finite differences).
  - Averaging-kernel columns: `test_cs_4dvar_retrieval_columns.jl` (TM5
    Fortran reference values and transcription, rebin conservation,
    column-mean equivalence, the schema block, single sweep vs seeded
    footprints, finite differences with windows, the mismatch policy,
    departures).
  - Level retrievals and the OCO-2 MIP: `test_retrieval_hat_projection.jl`
    (the hat projection against an independent construction, column
    conservation, reproduction, TM5 mode against a transcription of TM5's
    Python, level records in schema v2, the converter on a synthetic and the
    real MIP file) and `test_cs_4dvar_oco2_mip.jl` (single sweep and finite
    differences for level columns, co-sampling bit for bit, cell averages,
    forward-sampler parity).
  - Driver: `test_cs_inversion_driver.jl` runs
    `config/inversions/example_synthetic.toml`, and an observation file on
    the `synthetic_hydrostatic` meteorology.
  - Finite-difference and optimizer testsets for the 4D-Var path:
    `test_cs_ppm_adjoint_footprint.jl`.
  - Single-sweep gradient: `test_cs_4dvar_single_sweep.jl` checks it against
    one footprint per observation (every scheme family, diffusion, TM5,
    CMFMC, Full/Stride/Revolve), counts one recording and one reverse pass
    per gradient, and checks the reused-tape Jacobian.
