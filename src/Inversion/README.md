# Inversion

Cubed-sphere surface-flux inversion: observations, Jacobians, the 4D-Var
cost and gradient, background covariance, preconditioning, and optimizers.

This folder builds on the footprint machinery in [`../Footprint/`](../Footprint/).
Controls are surface emission-rate fields grouped into named time windows.
Observations are scalar objectives (layer or column means) sampled after a
given model step. Gradients come from one reverse-mode sweep that carries
every observation's weighted departure, not from finite differences. Observations and departures can be read from and written
to versioned NetCDF schemas.

## How It Is Loaded

`Inversion/` is **not a module**. [`../Adjoints/Adjoints.jl`](../Adjoints/Adjoints.jl)
`include`s these files into `module Adjoints`, and public names are exported
from the `Adjoints.jl` export block. The files load in two groups:

1. Early, right after `ObjectiveSeeding.jl` and `FootprintResult.jl`:
   `Observations.jl` -> `ObservationsIO.jl` -> `ObservationBinding.jl` ->
   `DeparturesIO.jl` -> `Covariance.jl` -> `Preconditioning.jl`.
2. Last, after `../Footprint/FootprintAPI.jl`: `Jacobian.jl` ->
   `CostGradient.jl` -> `Optimizer.jl`.

`NCDatasets` is imported in `Adjoints.jl`. `FFTW` (Covariance), `Optim`
(Optimizer), and `Dates` (ObservationBinding) are imported in the files that
use them. `bind_to_mesh` uses `Output.cell_locator` / `locate`, the same cell
lookup as the forward observation sampler.

## Entry Points

- [`CostGradient.jl`](CostGradient.jl): `cs_surface_flux_4dvar(panels_rm0,
  panels_m0, panels_am_steps, panels_bm_steps, panels_cm_steps, mesh,
  observations, controls; scheme, ..., tape_storage, tape_path, checkpoint,
  preconditioner)` returns a `CS4DVarResult`. `scheme` is required so an
  inversion cannot silently choose a different forward/adjoint advection
  operator.
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
- File IO: `read_observations` / `write_observations`, then `bind_to_mesh`
  (records to `Vector{CSObservation}`), then `build_departure_set` and
  `read_departures` / `write_departures`.

## File Map

- [`Observations.jl`](Observations.jl) — window, observation, control,
  result, and iteration-log types
- [`ObservationsIO.jl`](ObservationsIO.jl) — `CSObservationRecord`,
  `CSObservationSet`, and NetCDF reader/writer for
  `schemas/cs_observations_v1.toml`
- [`ObservationBinding.jl`](ObservationBinding.jl) — `bind_to_mesh`: maps
  time to a step index and (lat, lon) to `(panel, i, j)`
- [`DeparturesIO.jl`](DeparturesIO.jl) — `CSDepartureRecord`,
  `CSDepartureSet`, and strict v1 NetCDF IO for
  `schemas/cs_departures_v1.toml`
- [`Covariance.jl`](Covariance.jl) — `B^(1/2) = D·L`, built per panel
  from the FFT spectral square root of a periodic Gaussian
- [`Preconditioning.jl`](Preconditioning.jl) — χ <-> x transforms and their
  tangent and adjoint
- [`Jacobian.jl`](Jacobian.jl) — per-window footprint aggregation
  (`_aggregate_surface_window`) and `cs_surface_flux_jacobian` (one tape,
  one reverse walk per objective under `FullCheckpoint`)
- [`CostGradient.jl`](CostGradient.jl) — control validation and rate
  assembly, diagonal background term, preconditioned path, and
  `cs_surface_flux_4dvar` (forward run, weighted departures, one taped run
  and one reverse pass with the step forcing)
- [`Optimizer.jl`](Optimizer.jl) — `AbstractCSOptimizer` backends and
  `cs_surface_flux_4dvar_optimize`

## Common Tasks

- Adding an optimizer: subtype `AbstractCSOptimizer` and implement
  `cs_surface_flux_4dvar_solve(opt, cost_fn, controls)`, where
  `cost_fn(controls)` returns a `CS4DVarResult`.
- Adding a covariance: subtype `AbstractCSSurfaceFluxCovariance{FT, A}`, add
  an `Nc` field (the preconditioner reads `covariance.Nc`), and implement
  `apply_B_half!`, `apply_B_half_adjoint!`, and `apply_B_half_inverse!`.
- Changing how observations reach the model: time and space mapping is in
  `bind_to_mesh`; sampling during the forward run (simulated values and each
  observation's linear functional) is `_observe_cs_forward` in
  `../Footprint/ReverseLoop.jl`; a new observation operator needs
  `evaluate_objective` and `_linear_functional` methods
  (`../Footprint/StepForcing.jl`).
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
  mass after its step (`CSLinearFunctional`, weights from the replayed air
  mass). `cs_surface_flux_4dvar` runs the forward model once for the
  simulated values and functionals, scales each functional by
  `(H(x) − y)/σ²`, tapes one run to the last observation step, and walks it
  back once, adding each step's scaled weights at the step's end-of-step
  marker (after convection, before its transpose). Window steps after the
  last observation get a zero gradient. `checkpoint`, `tape_storage` and
  `tape_path` apply to that one taped run.
- `bind_to_mesh` maps `[t_start + (k−1)dt, t_start + k dt)` to step `k`. It
  drops altitude (every observation becomes a `CSColumnMeanObjective`) and
  returns `CSObservation{_, Float64}`.
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
  - IO: `test_cs_observations_io.jl`, `test_cs_observation_binding.jl`, and
    `test_cs_departures_io.jl`.
  - Driver: `test_cs_inversion_driver.jl` runs
    `config/inversions/example_synthetic.toml`.
  - Finite-difference and optimizer testsets for the 4D-Var path:
    `test_cs_ppm_adjoint_footprint.jl`.
  - Single-sweep gradient: `test_cs_4dvar_single_sweep.jl` checks it against
    one footprint per observation (every scheme family, diffusion, TM5,
    CMFMC, Full/Stride/Revolve), counts one recording and one reverse pass
    per gradient, and checks the reused-tape Jacobian.
