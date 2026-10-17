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
| `cs_surface_flux_jacobian` | Batches layer/column objectives on one recorded tape (one reverse walk per objective), aggregates per-step footprints into named user windows |
| `cs_surface_flux_4dvar` | 4D-Var cost / gradient with named controls, step-indexed scalar observations, diagonal background; one taped run and one reverse pass per gradient, whatever the number of observations ([single-sweep gradient](@ref single-sweep-4dvar-gradient)) |
| `cs_surface_flux_4dvar_optimize` | Dependency-free gradient-descent driver for the above |
| `cs_surface_flux_4dvar_solve` | Production driver with optimizer choice (`CSGradientDescent` / `CSLBFGS`) |

The adjoint supports the following advection schemes (full union in
`CSAdjointSupportedScheme`):

- `UpwindScheme()`
- `SlopesScheme(NoLimiter())`
- `PPMScheme(NoLimiter())`
- `PPMScheme(MonotoneLimiter())` and `PPMScheme(CW84Limiter())` — via a stored tracer-branch tape around the base trajectory
- `LinRoodPPMScheme(5)` and `LinRoodPPMScheme(7)`, with either vertical
  sweep: FV3's `kord = 8` profile (`vertical = "fv3_kord8"`, the configured
  default, or `"fv3_kord8_signed"`) or upwind (`vertical = "upwind"`). The FV3
  sweeps are transposed at the limiter branches of the recorded state; see
  [the adjoint of the FV3 vertical sweep](@ref fv3-vertical-adjoint)

The split-sweep `PPMScheme` with the FV3 vertical profile has no adjoint;
neither do limited `SlopesScheme` and `NoAdvection`.
`require_adjoint_support(scheme)` returns a supported scheme and otherwise
throws an error that names the missing piece. The inversion driver applies it
to its `[advection]` table; a Lin–Rood table without `vertical` selects
`fv3_kord8` there as in forward runs.

The supporting kernel adjoints in
`src/Operators/Advection/linrood_adjoint_*.jl` are
transposition-tested (`test/core/test_linrood_kernel_adjoints.jl`) and
finite-difference VJP-tested via single-panel and cross-panel halo
compositions. The full LinRood footprint, including its two Upwind Z
half-sweeps, matches central differences to about `1e-11` relative on a smooth
base field. `test/core/test_cs_linrood_footprint_fd.jl` asserts `1e-9` with
upward vertical flux and a surface-layer objective, the case most sensitive to
which air mass the Z records hold. Every tape records the air mass entering a
Z sweep, because the upwind adjoint coefficient `F / m_donor` uses the
pre-sweep donor mass; the FV3 profile's records also hold the tracer mass
entering the sweep. `test/core/test_cs_linrood_fv3_adjoint.jl` checks the FV3
vertical sweeps' transpose in 256-bit arithmetic and the Lin–Rood footprints
with them against central differences (`1e-9`, upward and downward flux,
ORD 5 and 7, both profiles), and Full/Stride/Revolve parity.

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
selectable via the `checkpoint` kwarg of `cs_surface_emission_footprint`,
`cs_surface_emission_footprint_from_seed`, `cs_surface_flux_jacobian` and
`cs_surface_flux_4dvar`:

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
`test_cs_4dvar_single_sweep.jl`, `test_cs_4dvar_preconditioned.jl`, `test_cs_lbfgs.jl`,
`test_cs_inversion_driver.jl`, `test_cs_inversion_truth_recovery.jl`,
`test_cs_iteration_log.jl`, `test_cs_observations_io.jl`,
`test_cs_observation_binding.jl`, `test_cs_departures_io.jl`,
`test_cs_covariance.jl`, `test_cs_preconditioning.jl`,
`test_cs_optimizer_dispatch.jl`.

### [Single-sweep 4D-Var gradient](@id single-sweep-4dvar-gradient)

`cs_surface_flux_4dvar` computes the observation term of the cost and its
gradient with one taped forward run and one reverse pass, whatever the number
of observations, as TM5-4DVAR does.

**Observations as linear functionals.** An observation ``i`` is sampled after
model step ``s_i``, after convection: the state runtime output and the forward
observation sampler see. The meteorology is fixed, so the air mass ``m`` of
every step is known before any tracer is transported, and the observation
operator is a fixed sparse linear functional of the tracer mass
``\mathbf{r}_{s_i}`` at that step,

```math
H_i(x) = h_i(\mathbf{r}_{s_i}) = c_i + \sum_e w_{ie}\, r_{s_i}[e],
```

with weights from the replayed air mass at ``s_i``: ``w = 1/\max(m, \epsilon)``
on one cell for `CSLayerMeanObjective` (``\epsilon`` the machine epsilon, the
floor of its evaluation), ``w = 1/\sum_k m_k`` on every level of a column for
`CSColumnMeanObjective`, and ``c_i = 0`` for both (the offset carries
tracer-independent terms such as an averaging kernel's prior term). Its
gradient ``\partial h_i/\partial\mathbf{r}_{s_i}`` is the weight vector
``\mathbf{w}_i``.

**One reverse sweep.** Write ``M_s`` for the tangent-linear model of step
``s`` (the transport palindrome with the surface-flux center, then
convection) and ``\lambda_s`` for the adjoint state at the end of step ``s``.
The cost ``\mathcal{J}_\mathrm{obs} = \tfrac12\sum_i ((H_i(x) - y_i)/\sigma_i)^2``
has the gradient of a sum, so all observations share one recursion,

```math
\lambda_{N} = f_{N}, \qquad
\lambda_{s} = M_{s+1}^{T}\,\lambda_{s+1} + f_{s}, \qquad
f_{s} = \sum_{i:\,s_i = s} \frac{H_i(x) - y_i}{\sigma_i^{2}}\,\mathbf{w}_i ,
```

run from the last observation step ``N`` back to the first step. Every
recorder ends each step with an end-of-step marker (`_CSStepEndRecord`, after
the convection record); the reverse walk adds ``f_s`` there, before it
transposes that step's convection. This is the TM5-4DVAR structure: its
forcing enters just before the adjoint of the operator after which the forward
run sampled. Where the walk reaches the midpoint of step ``t`` (the
palindrome center, where the step's surface emissions enter) it reads
``\Delta t\,\lambda`` as ``\partial\mathcal{J}_\mathrm{obs}/\partial E_t``,
and each control window sums its steps.

The forcing entries are grouped by step, panel and cell; the injection
kernel gives each cell one work item, which adds the cell's entries in input
order, so the injection is deterministic without atomics. With
`FullCheckpoint` one evaluation therefore costs a forward run for ``H(x)``
and the weights, one taped forward run to the last observation step, and one
reverse pass; `StrideCheckpoint` adds its checkpoint propagation pass and
`RevolveCheckpoint` its ``O(\log N)`` recomputation, as for a footprint. Steps
after the last observation do not affect the cost and get a zero gradient.
`checkpoint`, `tape_storage` and `tape_path` choose how that one taped run is
stored. `test/core/test_cs_4dvar_single_sweep.jl` checks the gradient
against one footprint per observation (the previous algorithm, rebuilt in the
test) for upwind, unlimited slopes and PPM, monotone PPM and Lin–Rood with
diffusion of both surface-flux couplings, TM5 and CMFMC convection and the
three checkpoint schedules, and for CW84-limited PPM and Lin–Rood with the
FV3 vertical sweep
(``10^{-12}`` relative in Float64, ``10^{-5}`` in Float32), and counts tape
reads to confirm one recording and one reverse pass per gradient.

`cs_surface_flux_jacobian` likewise records its tape once and walks it once
per objective (with `FullCheckpoint`); each walk is the one a separate
`cs_surface_emission_footprint` call makes.

## What is not yet shipped

| Item | Notes |
| --- | --- |
| **Optimal binomial Revolve** | `RevolveCheckpoint` ships as the bisection variant — logarithmic memory but not the Griewank–Walther optimal recompute count. Optimal binomial Revolve is the next refinement. |
| **Optimized/clamped convection adjoints** | CMFMC `clamp = true` and TM5/CMFMC-matrix collaborative, truncated, or merged solves are rejected by the footprint API until their exact branches are taped and transposed. |
| **TM5-4DVAR cross-validation** | Synthetic truth-recovery via `test_cs_inversion_truth_recovery.jl` is on CI; a side-by-side parity run against TM5-4DVAR on real data has not been published. |
| **Vertical observation binding** | `bind_to_mesh` ignores the observation altitude and binds every record to a column mean (`CSColumnMeanObjective`), which is wrong for surface in-situ and aircraft data. Layer binding needs an above-ground height or a pressure in the observation schema (`alt` is above sea level and binaries carry no surface elevation). No averaging-kernel column objective exists: the 4D-Var path evaluates only `CSLayerMeanObjective` and `CSColumnMeanObjective`, and `cs_surface_emission_footprint_from_seed` gives footprints for other operators but no cost. |
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
optimizer choices, and an `[advection]` table that is parsed like a forward
run's (`limiter = "none"`, the unlimited PPM, is accepted only here). Unknown
tables and keys are errors. It does not expose checkpoint, diffusion,
convection or real-binary settings.
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
