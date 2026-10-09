# Adjoint status

This page marks the current boundary of the adjoint implementation: what can
be run and tested today, and which combinations are deliberately rejected.
The LinRood adjoint and TM5-style inversion scaffold are implemented and
covered by the core test suite, but real-data parity with TM5-4DVAR remains a
validation gap.

For a first introduction, start with [Learning adjoints and inversions](@ref Learning-adjoints)
and the [executed footprint tutorial](@ref First-emission-footprint). This page
is the support reference, not the starting lesson.

## What is shipped

### Forward operators (unchanged)

The forward transport model — advection (four schemes: `UpwindScheme`,
`SlopesScheme`, `PPMScheme`, `LinRoodPPMScheme`), convection (CMFMC +
TM5), implicit vertical diffusion (Beljaars–Viterbo / Holtslag–Boville
Kz fields), and surface flux sources — is implemented for the runtime's
supported topology/backend combinations and covered by the test suite documented in
[Conservation budgets](@ref).

### Tape, checkpoint, and reverse pass

`src/Adjoints/` ships a full discrete-adjoint pipeline for the
cubed-sphere split-sweep path, with a separate kernel-level pipeline
for `LinRoodPPMScheme` integrated through `cs_surface_emission_footprint`.

The shipped pieces, all exported through `AtmosTransport.Adjoints`:

| Surface | What it does |
| --- | --- |
| `cs_surface_emission_footprint` | Forward → tape → reverse pass driver for CS, returning `dJ/dE_t` at every surface step |
| `cs_surface_emission_footprint_from_seed` | Same, but accepts an explicit final `dJ/drm` seed for arbitrary observation operators |
| `cs_surface_flux_jacobian` | Batches layer/column objectives, aggregates per-step footprints into named user windows |
| `cs_surface_flux_4dvar` | 4D-Var cost / gradient with named controls, step-indexed scalar observations, diagonal background |
| `cs_surface_flux_4dvar_optimize` | Dependency-free gradient-descent driver for the above |
| `cs_surface_flux_4dvar_solve` | Production driver with optimizer choice (`CSGradientDescent` / `CSLBFGS`) |

The adjoint supports the following advection schemes (full union in
`CSAdjointSupportedScheme`):

- `UpwindScheme()`
- `SlopesScheme(NoLimiter())`
- `PPMScheme(NoLimiter())`
- `PPMScheme(MonotoneLimiter())` and `PPMScheme(CW84Limiter())` — via a stored tracer-branch tape around the base trajectory
- `LinRoodPPMScheme(; ppm_order = 5)`
- `LinRoodPPMScheme(; ppm_order = 7)`

The supporting kernel adjoints in
`src/Operators/Advection/linrood_adjoint_*.jl` are
transposition-tested (`test/core/test_linrood_kernel_adjoints.jl`) and
finite-difference VJP-tested via single-panel and cross-panel halo
compositions.

The opt-in `test/diagnostic/test_cs_transport_adjoint_gpu.jl` also exercises
transporting PPM and LinRood footprints on CUDA with scalar indexing disabled.
It checks Float32/Float64 CPU/GPU agreement, Float64 directional finite
differences, and full/stride/revolve checkpoint parity, plus the single-panel
recording and reverse wrappers. LinRood copy-back and halo-gradient updates
run on the array backend; the tape path does not require host scalar reads.

The CS reverse pass also supports the default, unclamped/full-column/unmerged
forms of `TM5Convection`, `CMFMCConvection`, and `CMFMCMatrixConvection`.
CMFMC and CMFMC-matrix have column adjoint-identity tests; the TM5/CMFMC
footprint paths are checked against finite-difference emission gradients in
`test_cs_ppm_adjoint_footprint.jl`.

### Checkpoint schedules

`src/Tape/CheckpointSchedule.jl` ships three checkpoint policies,
selectable via the `checkpoint` kwarg of `cs_surface_emission_footprint`:

| Policy | Memory | Recompute |
| --- | --- | --- |
| `FullCheckpoint` | O(N) state snapshots | none — fastest reverse pass |
| `StrideCheckpoint(stride)` | O(N/stride) | replays each stride forward |
| `RevolveCheckpoint()` | O(log N) | recursive bisection; each step may be replayed once per recursion level |

`RevolveCheckpoint()` has no snapshot-budget argument. It is a recursive
bisection variant rather than the optimal binomial Revolve: peak snapshots are
proportional to the recursion depth and total replay work is
`O(N log N)`. It is covered in `test/core/test_cs_stride_checkpoint.jl`.

### Tape storage backends

Split-sweep tape records can live on the device (the default), in pinned-host
memory, or on disk via mmap — selectable via the `tape_storage`
kwarg (`:device` / `:pinned_host` / `:mmap`). Storage backends are
defined in `src/Tape/TapeStorage.jl` and `src/Tape/MmapTapeStorage.jl`;
the mmap path is covered by `test_cs_tape_mmap_roundtrip.jl`. LinRood tape
records currently require `:device` storage.

### Inversion scaffold

`src/Inversion/` ships:

| Module | Surface |
| --- | --- |
| `Covariance.jl` | `DiagonalCSCovariance`, `IsotropicGaussianCSCovariance`, `apply_B_half!`, `apply_B_half_adjoint!`, `apply_B_half_inverse!` |
| `Preconditioning.jl` | `apply_preconditioner!`, `LinearOptimType`, `LogNormalOptimType` |
| `Optimizer.jl` | `AbstractCSOptimizer`, `CSGradientDescent`, `CSLBFGS` (via multiple-dispatch surfaces) |
| `Jacobian.jl` | Footprint-Jacobian assembly for batched observation operators |
| `CostGradient.jl` | 4D-Var cost / gradient evaluator with preconditioning |
| `Observations.jl`, `ObservationBinding.jl`, `ObservationsIO.jl` | Typed observation surface + on-disk schema |
| `DeparturesIO.jl` | Departures-file IO |

These are all on CI. Full inversion tests:
`test_cs_4dvar_preconditioned.jl`, `test_cs_lbfgs.jl`,
`test_cs_inversion_driver.jl`, `test_cs_inversion_truth_recovery.jl`,
`test_cs_iteration_log.jl`, `test_cs_observations_io.jl`,
`test_cs_observation_binding.jl`, `test_cs_departures_io.jl`,
`test_cs_covariance.jl`, `test_cs_preconditioning.jl`,
`test_cs_optimizer_dispatch.jl`.

## What is not yet shipped

| Item | Notes |
| --- | --- |
| **Optimal binomial Revolve** | `RevolveCheckpoint` ships as the bisection variant — logarithmic memory but not the Griewank–Walther optimal recompute count. Optimal binomial Revolve is the next refinement. |
| **Optimized/clamped convection adjoints** | CMFMC `clamp = true` and TM5/CMFMC-matrix collaborative, truncated, or merged solves are rejected by the footprint API until their exact branches are taped and transposed. |
| **TM5-4DVAR cross-validation** | Synthetic truth-recovery via `test_cs_inversion_truth_recovery.jl` is on CI; a side-by-side parity run against TM5-4DVAR on real data has not been published. |
| **Tangent-linear model** | Forward TL is not exposed as a separate driver. If you need it, use the reverse pass plus identity seeding. |

## Adjoint-readiness in the forward design

Three concrete forward-design choices that pay off in the adjoint:

- **Vertical diffusion** — generic Kz kernels expose the tridiagonal
  coefficients `(a, b, c)` and transpose the backward-Euler solve with its
  mass/VMR scaling. The precomputed Dkg mass path exposes the retention
  fractions of two conservative bidiagonal solves. Its reverse applies
  those passes' transposes in reverse order, preserving a constant total-mass
  gradient exactly for positive carrier masses.
- **Convection (CMFMC + TM5)** — `apply!` takes a `ConvectionForcing`
  carrier explicitly so the operator does not call `current_time`
  internally; this keeps the operator pure-functional in the time
  variable. The TM5 reverse pass rebuilds the same per-column matrix
  and solves with the transposed LU factors.
- **Advection** — the adjoint composes each operator's transpose in reverse
  order. The palindrome helps organize that reversal, but forward transport
  is not its own inverse. Stored states supply the limiter branches and
  reconstructed-face sensitivities needed by the reverse pass.

## How to use the adjoint today

The maintained command-line path is the inversion driver:

```bash
julia --project=. scripts/inversions/cs_4dvar.jl \
    config/inversions/example_synthetic.toml
```

That synthetic-only configuration owns observations, controls, covariance and
optimizer choices. It does not expose checkpoint or real-binary settings.
For the lower-level Julia API, start with the
[footprint tutorial](@ref First-emission-footprint), then
`test/core/test_cs_inversion_truth_recovery.jl` for a larger inversion assembly.
Use `RevolveCheckpoint()` (without arguments)
for recursive bisection, or `StrideCheckpoint(K)` for an explicit interval.
LinRood checkpoints currently require `tape_storage = :device`; split-sweep
schemes also support `:pinned_host` and `:mmap`.

## Where to read next

- [Validation status](@ref) — what the forward model has been
  validated against.
- [Conservation budgets](@ref) — the explicit verification tests
  the forward operators pass.
- [Adjoints on top of the binary](../for_tm5_gchp_users/adjoints.md) —
  TM5-4DVAR / GIGC-adjoint user perspective on how the pipeline maps
  to those workflows.
