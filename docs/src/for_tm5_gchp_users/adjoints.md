```@meta
CurrentModule = AtmosTransport
```

# Adjoints and surface-flux inversion

If you have used TM5-4DVAR or the GEOS-Chem adjoint, the goal is familiar:
run transport forward, seed a scalar observation objective, and propagate its
sensitivity backward to surface emissions. AtmosTransport uses the same Julia
operator implementations and accelerator backend for forward and reverse
work; it does not maintain a separate adjoint fork.

This subsystem currently targets cubed-sphere surface-emission footprints and
a compact 4D-Var scaffold. It is useful for method development and synthetic
inversions. Read [Adjoint status](@ref) before planning a production campaign,
because not every optimized forward-physics branch has a matching reverse
operator yet.

## Try the shipped inversion first

New to reverse sensitivities? Read [Learning adjoints and inversions](@ref Learning-adjoints)
and run the [checked footprint tutorial](@ref First-emission-footprint) first.

The repository contains a tiny, self-contained inversion with synthetic
meteorology:

```bash
julia --project=. scripts/inversions/cs_4dvar.jl \
    config/inversions/example_synthetic.toml
```

It builds a C3 cubed sphere with zero transport fluxes, two model steps,
unlimited PPM advection, one cell/layer observation, an isotropic covariance,
a linear preconditioner, and an L-BFGS solve. It prints the initial and final
cost. First-time compilation can take longer
than the small inversion itself.

!!! note "Two config interfaces"
    `scripts/run_transport.jl` reads the forward-run schema documented in
    [TOML schema](@ref). The inversion script above reads its own
    smaller schema. Its required `[advection]` table goes through the same
    parser as a forward run's (`limiter = "none"`, the unlimited PPM, is
    accepted only by the inversion), and a scheme without an adjoint is
    rejected. Any table or key the script does not read, such as
    `[diffusion]`, `[convection]` or `[output]`, is an error. A forward config
    does **not** accept an `[adjoint]` block. For programmatic work, call the
    APIs in `AtmosTransport.Adjoints` directly.

The second example, `config/inversions/example_c48.toml`, is a schema sketch
for a future real-meteorology driver. It is intentionally not runnable with
the current synthetic-only script.

## What the reverse pass computes

```mermaid
flowchart LR
    A[Forward CS transport] --> B[Record operator data]
    B --> C[Evaluate observation objective]
    C --> D[Seed final adjoint state]
    D --> E[Replay records in reverse]
    E --> F[Footprints dJ / dE at each surface step]
```

The footprint entry point is
`AtmosTransport.Adjoints.cs_surface_emission_footprint`. Its positional
inputs are panel tuples for initial tracer mass and air mass, step sequences
for the three mass-flux directions, a `CubedSphereMesh`, and an objective such
as `CSLayerMeanObjective` or `CSColumnMeanObjective`. It returns a
`CSFootprintResult`; `result.footprints[t]` contains the sensitivity to the
surface-emission rate at model step `t`.

The API accepts the following advection families:

- `UpwindScheme()`
- `SlopesScheme(NoLimiter())`
- `PPMScheme(NoLimiter())`
- monotone `PPMScheme()` and `PPMScheme(CW84Limiter())` using recorded
  limiter branches
- `LinRoodPPMScheme` with its dedicated horizontal tape and either vertical
  sweep: upwind, or FV3's `kord = 8` profile (`vertical = "fv3_kord8"`, the
  configured default, or `"fv3_kord8_signed"`), transposed at the limiter
  branches of the recorded state

`require_adjoint_support(scheme)` checks a scheme against this list and names
what is missing otherwise (for example the split-sweep `PPMScheme` with the FV3
vertical profile).

Optional vertical diffusion and supported convection operators can be passed
with the corresponding operator, forcing, and workspace keywords. Unsupported
physics combinations fail validation instead of silently using an incomplete
transpose.

## The forward model the adjoint transposes

A footprint is a derivative of the production model, so the adjoint's forward
model is the production step of a run with surface fluxes
(`TransportModel.step!`: the transport palindrome of `strang_split!`, then
convection). Write `x_n` for the tracer mass on the six panels, interiors and
halos, before step `n` of length `Δt`. Read operator products right to left:

```math
x_{n+1} = \mathcal{K}_n \, \mathcal{R}_n \, \mathcal{C}_n \, \mathcal{F}_n \, \mathcal{H} \, x_n ,
\qquad
\mathcal{F}_n = (\mathcal{Z}\mathcal{Y}\mathcal{X})^{s}, \quad
\mathcal{R}_n = (\mathcal{X}\mathcal{Y}\mathcal{Z})^{s}.
```

- ``\mathcal{H}`` is the step-start halo refresh (`fill_cs_step_start_halos!`):
  every edge halo cell, then every corner cell in the X-sweep orientation,
  becomes a copy of an interior cell of a neighbouring panel. The first X
  sweep of the step reads these halos, and the previous step's convection
  changed the interiors they mirror.
- ``\mathcal{F}_n`` and ``\mathcal{R}_n`` are the two halves of the transport
  palindrome with `s` CFL subcycles, every flux divided by `s`, with the halo
  exchanges between sweeps.
- ``\mathcal{K}_n`` is the convection column operator (TM5 or CMFMC).
- ``\mathcal{C}_n`` is the palindrome center. With surface-flux injection
  ``\mathcal{S}_n(\Delta t)`` (adds ``E_n \Delta t`` to the surface layer) and
  the backward-Euler diffusion solve ``\mathcal{V}(\tau) = (I - \tau D)^{-1}``,
  the diffusion operator's `surface_flux_coupling` selects

```math
\mathcal{C}_n =
\begin{cases}
\mathcal{V}(\Delta t/2)\,\mathcal{S}_n(\Delta t)\,\mathcal{V}(\Delta t/2) & \texttt{SplitSurfaceFluxCoupling} \\
\mathcal{V}(\Delta t)\,\mathcal{S}_n(\Delta t) & \texttt{DiffusiveSurfaceFluxBoundary}
\end{cases}
```

Backward Euler is not a semigroup, ``\mathcal{V}(\Delta t) \neq
\mathcal{V}(\Delta t/2)^2``, so the two couplings are different models that
differ at ``O((\Delta t\,D)^2)``; the adjoint must transpose the one the run
uses. One function, `surface_flux_center!`, defines this order for production
(through `palindrome_center!`), for the tape-free forward replays and for the
tape recorders.

The reverse pass applies the transposes in the opposite order,
``\mathcal{H}^T \mathcal{F}_n^T \mathcal{C}_n^T \mathcal{R}_n^T \mathcal{K}_n^T``.
``E_n`` enters only through ``\mathcal{S}_n``, so

```math
\frac{\partial J}{\partial E_n} = \Delta t \, \lambda_n^{S}\big|_{k = N_z},
```

where ``\lambda_n^{S}`` is the adjoint state at the position of
``\mathcal{S}_n``: the final seed propagated back through all later steps,
``\mathcal{K}_n^T``, ``\mathcal{R}_n^T``, and the part of the center after
the injection, ``\mathcal{V}(\Delta t/2)^T`` or ``\mathcal{V}(\Delta t)^T``.
The tape marks that position with its midpoint record. ``\mathcal{H}`` is a
linear copy, ``h = P\,i`` from interior to halo, so ``\mathcal{H}^T`` adds each
halo sensitivity into its source interior cell and zeroes the halo. Without
``\mathcal{H}`` the replayed X sweep would read halos copied before the
previous step's convection: a different operator from production, whose
error grew with the step count and the convective mass flux, and which
also made a checkpoint window that restarts with fresh halos disagree with
the full tape. Lin–Rood's horizontal operator refills the halos it reads, so
its steps need no ``\mathcal{H}``.
`test/core/test_cs_adjoint_production_parity.jl` checks the replay against
`TransportModel.step!` for both couplings with TM5 and CMFMC convection, and
the footprints against finite differences of the production model.

## From footprints to 4D-Var

The inversion layer composes the reverse pass rather than reimplementing it:

| API | Purpose |
| --- | --- |
| `cs_surface_emission_footprint` | One scalar objective to per-step emission sensitivities. |
| `cs_surface_flux_jacobian` | Batch objectives and aggregate steps into named control windows. |
| `cs_surface_flux_4dvar` | Evaluate observation/background cost and its control gradient, with one taped run and one reverse pass for all observations. |
| `cs_surface_flux_4dvar_optimize` | Optimize with `CSGradientDescent` or `CSLBFGS`. |
| `read_observations`, `bind_to_mesh` | Read point observations and bind them to CS column-mean objectives (altitude is ignored; see Current limits). |
| `read_departures`, `write_departures` | Exchange modeled-minus-observed diagnostics. |

As in TM5-4DVAR, one gradient is one adjoint integration. Each observation
is a fixed linear functional of the tracer mass after its model step (weights
from the replayed air mass); `cs_surface_flux_4dvar` runs the forward model
for the simulated values, tapes one run to the last observation step, and
walks that tape back once, adding every observation's weighted departure
`(H(x) − y)/σ²` at the end of its step, before the transposed convection of
that step (TM5 adds its forcing just before the adjoint of the operator after
which it sampled). See the [single-sweep gradient](@ref single-sweep-4dvar-gradient).
`cs_surface_flux_jacobian` records its tape once and walks it once per
objective.

Controls are `CSSurfaceFluxControl` values associated with
`CSSurfaceFluxWindow`s. A `CSSurfaceFluxPreconditioner` can apply either a
linear or log-normal transform using a diagonal or isotropic-Gaussian
covariance. See the working assembly in `scripts/inversions/cs_4dvar.jl` and
the exact keyword contracts in [Adjoints and checkpointing API](@ref).

## Tape and checkpoint choices

The forward pass records typed advection, halo, midpoint, diffusion,
convection, and end-of-step operations. Lin-Rood horizontal updates use an additional dedicated
record. The reverse loop dispatches on those record types and calls the
matching transpose kernel.

`tape_storage` controls where staged tape data lives:

| Value | Use |
| --- | --- |
| `:device` | Default and simplest choice for short runs. |
| `:pinned_host` | Move staged data to pinned host memory when device memory is limiting. Reads come back through one device buffer per operand (air mass, tracer mass). |
| `:mmap` | Store records under an on-disk directory for long or inspectable runs. |

With `:mmap`, pass `tape_path` when the files must persist beyond the API call.
After finalization, reopen the directory with `load_mmap_tape(path)` and read
records with `get_record`. An interrupted, unfinalized tape is rejected.
Lin-Rood currently supports device tape storage only.

The `checkpoint` keyword controls the memory/recompute trade-off:

| Policy | Stored state | Recompute behavior |
| --- | --- | --- |
| `FullCheckpoint()` | Proportional to the number of steps. | No window replay. |
| `StrideCheckpoint(k)` | Checkpoints every `k` steps. | Replays one `k`-step window at a time. |
| `RevolveCheckpoint()` | Logarithmic-depth bisection snapshots. | Recursively replays subranges. |

`RevolveCheckpoint()` is a bisection schedule inspired by Revolve, not the
optimal binomial Griewank-Walther schedule and not a user-specified snapshot
budget. All three schedules record the same states: every step starts from
the step-start halo refresh, so a window that restarts at a step boundary
sees the halos of the full tape. Stride and Revolve footprints agree with
`FullCheckpoint` to rounding (tested at `1e-12`) for every supported scheme,
including limited PPM with diffusion and convection. An earlier Revolve drift
of order `1e-7` in that configuration came from a missing step-start halo
refresh in the recorders, not from limiter flips.

## How this maps to established systems

| Task | TM5-4DVAR / GEOS-Chem adjoint | AtmosTransport |
| --- | --- | --- |
| Forward integration | Separate CTM driver | `run_driven_simulation` or the lower-level CS arrays |
| Reverse forcing | Observation operator, departures injected during one adjoint run | `CSObservation` objectives as linear functionals, injected at each step's end in one reverse pass, or an explicit seed |
| Observation IO | System-specific files | `CSObservationSet` NetCDF IO + `bind_to_mesh` |
| Background covariance | B or B¹ᐟ² implementation | `DiagonalCSCovariance` / `IsotropicGaussianCSCovariance` |
| Preconditioning | Hand-coded control transform | `CSSurfaceFluxPreconditioner` |
| Optimization | M1QN3 / L-BFGS variants | `CSGradientDescent` / `CSLBFGS` |
| Driver | Dedicated inversion executable | `scripts/inversions/cs_4dvar.jl` for the synthetic workflow |

## Current limits

- Observations lose their altitude. `bind_to_mesh` turns every record into
  an air-mass-weighted column mean (`CSColumnMeanObjective`) of its cell and
  ignores `alt`, so a surface site and an aircraft sample in the same cell
  and step get the same observation operator. That is wrong for surface
  in-situ and aircraft data. Layer binding needs an above-ground intake
  height or a pressure: the observation schema stores `alt` above sea level
  and transport binaries carry no surface elevation. The forward
  observation sampler (`[output.observations]`) already places intake
  heights in layers. There is no averaging-kernel column operator for
  satellite or TCCON columns either: `cs_surface_flux_4dvar` evaluates only
  `CSLayerMeanObjective` and `CSColumnMeanObjective`, and
  `cs_surface_emission_footprint_from_seed` gives footprints for other
  operators but no cost.
- CMFMC requires `clamp = false` in the footprint path.
- TM5 and CMFMC-matrix convection require the full, unmerged,
  non-collaborative solve (`use_collab_lu = false`, `lmax_conv = 0`,
  `n_merge = 1`).
- A tangent-linear driver is not exposed.
- A real-data TM5-4DVAR or GCHP-adjoint cross-validation has not been
  published.
- The supplied TOML inversion driver supports synthetic constant meteorology;
  real transport-binary ingestion still requires programmatic assembly.

For the test-by-test support matrix and the distinction between verified
kernels and externally validated workflows, continue with
[Adjoint status](@ref) and [Validation status](@ref).
