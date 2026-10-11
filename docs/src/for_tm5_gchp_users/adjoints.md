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

It builds a C3 cubed sphere with zero transport fluxes, two model steps, one cell/layer observation,
an isotropic covariance, a linear preconditioner, and an L-BFGS solve. It
prints the initial and final cost. First-time compilation can take longer
than the small inversion itself.

!!! note "Two config interfaces"
    `scripts/run_transport.jl` reads the forward-run schema documented in
    [TOML schema](@ref). The inversion script above reads its own
    smaller schema. A forward config does **not** accept an `[adjoint]` block.
    For programmatic work, call the APIs in `AtmosTransport.Adjoints` directly.

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
- monotone `PPMScheme()` using recorded limiter branches
- `LinRoodPPMScheme` with its dedicated horizontal tape

Optional vertical diffusion and supported convection operators can be passed
with the corresponding operator, forcing, and workspace keywords. Unsupported
physics combinations fail validation instead of silently using an incomplete
transpose.

## From footprints to 4D-Var

The inversion layer composes the reverse pass rather than reimplementing it:

| API | Purpose |
| --- | --- |
| `cs_surface_emission_footprint` | One scalar objective to per-step emission sensitivities. |
| `cs_surface_flux_jacobian` | Batch objectives and aggregate steps into named control windows. |
| `cs_surface_flux_4dvar` | Evaluate observation/background cost and its control gradient. |
| `cs_surface_flux_4dvar_optimize` | Optimize with `CSGradientDescent` or `CSLBFGS`. |
| `read_observations`, `bind_to_mesh` | Read point observations and bind them to CS column-mean objectives (altitude is ignored; see Current limits). |
| `read_departures`, `write_departures` | Exchange modeled-minus-observed diagnostics. |

Controls are `CSSurfaceFluxControl` values associated with
`CSSurfaceFluxWindow`s. A `CSSurfaceFluxPreconditioner` can apply either a
linear or log-normal transform using a diagonal or isotropic-Gaussian
covariance. See the working assembly in `scripts/inversions/cs_4dvar.jl` and
the exact keyword contracts in [Adjoints and checkpointing API](@ref).

## Tape and checkpoint choices

The forward pass records typed advection, halo, midpoint, diffusion, and
convection operations. Lin-Rood horizontal updates use an additional dedicated
record. The reverse loop dispatches on those record types and calls the
matching transpose kernel.

`tape_storage` controls where staged tape data lives:

| Value | Use |
| --- | --- |
| `:device` | Default and simplest choice for short runs. |
| `:pinned_host` | Move staged data to pinned host memory when device memory is limiting. |
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
budget. Use `StrideCheckpoint(k)` when bitwise parity with a full checkpoint is
more important than minimum storage; nonlinear limiter branches can differ at
roundoff level after recursive replay.

## How this maps to established systems

| Task | TM5-4DVAR / GEOS-Chem adjoint | AtmosTransport |
| --- | --- | --- |
| Forward integration | Separate CTM driver | `run_driven_simulation` or the lower-level CS arrays |
| Reverse forcing | Observation operator | `CSObservation` objectives or an explicit seed |
| Observation IO | System-specific files | `CSObservationSet` NetCDF IO + `bind_to_mesh` |
| Background covariance | B or B¹ᐟ² implementation | `DiagonalCSCovariance` / `IsotropicGaussianCSCovariance` |
| Preconditioning | Hand-coded control transform | `CSSurfaceFluxPreconditioner` |
| Optimization | M1QN3 / L-BFGS variants | `CSGradientDescent` / `CSLBFGS` |
| Driver | Dedicated inversion executable | `scripts/inversions/cs_4dvar.jl` for the synthetic workflow |

## Current limits

- One adjoint run per observation. `cs_surface_flux_4dvar` runs the forward
  model once for the simulated values, then for every observation re-runs it
  with a tape up to the observation's step and runs a separate reverse pass.
  The cost of one cost-and-gradient evaluation therefore grows linearly with
  the number of observations, where TM5-4DVAR injects all weighted
  departures into a single adjoint sweep. This is practical for small
  synthetic experiments, not for ObsPack (about `10⁵` records) or OCO-2
  (about `10⁶` soundings) volumes, where one gradient would need that many
  adjoint runs.
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
