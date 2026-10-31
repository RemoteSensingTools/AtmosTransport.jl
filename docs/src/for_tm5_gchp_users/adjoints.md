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
with the corresponding operator, forcing, and workspace keywords. The physics
cadence, the met windows and the air-mass resets of a driven run come in as a
`CSStepSchedule` (`schedule` keyword; `CSStepSchedule(sim)` for a
`DrivenSimulation`). `cs_adjoint_inputs(sim; tracer)` captures every input of
a cubed-sphere `DrivenSimulation` configuration from production itself — the
face fluxes of each step after window loading and flux scaling (one copy per
met window, shared by its steps, as production's steps read the same
arrays; one per step when the fluxes are interpolated within the window), the
convection forcing and diffusion layer thickness of each met window, the
tracer's emission rates, the schedule and fresh workspaces — so the adjoint
of a driven run needs no hand assembly. Every entry point checks the whole step plan before it
records anything (below): an operator without an adjoint is an error that
names it, never a silent approximation. `CSStepSchedule(sim)` also runs that
check and retains the production chemistry operator, so a schedule cannot hide
unsupported decay. Captured air-mass reset targets are computed on the model's
backend, including CUDA with `air_mass_reset_mode = :preserve_vmr`.

Capture requires fresh persistent operator state, in addition to `iteration == 0`.
It rejects nonzero Kahan compensation, convection or diffusion/Dkg ledger carries,
and a valid CMFMC CFL cache before consuming the simulation. A model continued
from a previous binary can retain these values even when its new simulation is
at iteration zero. Build a fresh model for capture; continuation from existing
operator state is not supported.

## The forward model the adjoint transposes

A footprint is a derivative of the production model, so the adjoint's forward
model is the production step, by construction rather than by a second
implementation kept in step with it.

**One description of the step.** A `StepPlan` holds the operators of a run's
step and when they run, in four phases:

| phase | when | stages |
| --- | --- | --- |
| `window_advance` | entering a met window after the first | air-mass reset to the new window (`air_mass_reset_mode`) |
| `window_start` | before the first step of each window | first decay half over the window (`:window` cadence) |
| `step` | every advection step | transport block; with `:substep` cadence also decay halves and convection over `Δt` |
| `window_end` | after the last step of each window | reset to the window's end mass (binary contract), convection and the closing chemistry over the window (`:window` cadence) |

`StepPlan(; physics_cadence, window_contract, air_mass_reset)` builds it.
`TransportModel.step!` executes the `step` stages of the substep plan,
`DrivenSimulation` executes all four phases of its plan (`sim.plan`), and the
cubed-sphere adjoint replays and records the same plan
(`CSStepSchedule(sim)`): the same stage functions, the same production
operators (the packed-tracer palindrome, the Kahan-compensated surface source,
the convection and diffusion operators with their ledger carries and the CMFMC
CFL cache), on a single-tracer state.

**Recording.** The production step calls an observation hook at every point
whose input the transpose needs: before each advection sweep, after each halo
exchange, at the palindrome center, inside Lin–Rood's horizontal operator,
after convection, before an air-mass reset. Production passes
`NoStepRecorder()`, for which every hook is an inlined no-op. The adjoint
passes a tape recorder that stages the hook's input and pushes one tape record
per hook. The tape therefore lists exactly the operators production ran, in
production order, and the reverse walk applies their transposes backwards.

Write `x_n` for the tracer mass on the six panels, interiors and halos, before
step `n` of length `Δt`. Read operator products right to left. The transport
block of a split-sweep step is

```math
\mathcal{T}_n = \mathcal{R}_n \, \mathcal{C}_n \, \mathcal{F}_n \, \mathcal{H},
\qquad
\mathcal{F}_n = (\mathcal{Z}\mathcal{Y}\mathcal{X})^{s}, \quad
\mathcal{R}_n = (\mathcal{X}\mathcal{Y}\mathcal{Z})^{s}.
```

- ``\mathcal{H}`` is the step-start halo refresh (`fill_cs_step_start_halos!`):
  every edge halo cell, then every corner cell in the X-sweep orientation,
  becomes a copy of an interior cell of a neighbouring panel. The first X
  sweep of the step reads these halos, and the operators of the previous step
  changed the interiors they mirror.
- ``\mathcal{F}_n`` and ``\mathcal{R}_n`` are the two halves of the transport
  palindrome with `s` subcycles (from the CFL budget, or one under a binary
  window contract), every flux divided by `s`, with the halo exchanges between
  sweeps.
- ``\mathcal{C}_n`` is the palindrome center. With surface-flux injection
  ``\mathcal{S}_n(\Delta t)`` (adds ``E_n \Delta t`` to the surface layer) and
  the backward-Euler diffusion solve ``\mathcal{V}(\tau) = (I - \tau D)^{-1}``,

```math
\mathcal{C}_n =
\begin{cases}
\mathcal{V}(\Delta t/2)\,\mathcal{S}_n(\Delta t)\,\mathcal{V}(\Delta t/2) & \texttt{SplitSurfaceFluxCoupling} \\
\mathcal{V}(\Delta t)\,\mathcal{S}_n(\Delta t) & \texttt{DiffusiveSurfaceFluxBoundary} \\
\mathcal{V}(\Delta t) & \text{no surface-flux operator}
\end{cases}
```

Backward Euler is not a semigroup, ``\mathcal{V}(\Delta t) \neq
\mathcal{V}(\Delta t/2)^2``, so the three centers are different models that
differ at ``O((\Delta t\,D)^2)``; the adjoint transposes the one the plan has.
Footprints and 4D-Var differentiate with respect to the emissions, so their
plans always carry a surface-flux operator (zero rates allowed);
`cs_surface_emission_footprint_from_seed(...; surface_flux = false)` replays a
configuration without surface sources, whose center is one solve
``\mathcal{V}(\Delta t)``, and returns its initial-condition sensitivity
(`result.initial_adjoint`; every result carries it).

**Physics cadence.** With `physics_cadence = :substep` (or without a binary
window contract) step `n` is ``x_{n+1} = \mathcal{K}_n(\Delta t)\,
\mathcal{T}_n x_n``, convection ``\mathcal{K}`` over the step. Under a
transport binary's per-window contract and `physics_cadence = :window` (the
default of driven runs) the steps of a met window `w` run the transport block
only, and after the window's last step

```math
x \leftarrow \mathcal{K}_w(\Delta t_w)\,\mathcal{M}_w\,x ,
```

where ``\mathcal{M}_w`` resets the air mass to the window's end mass (the
tracer mass is unchanged under `:preserve_tracer_mass`, scaled by
``m_\text{new}/m_\text{old}`` under `:preserve_vmr`) and ``\mathcal{K}_w``
is convection over the whole window ``\Delta t_w`` with the window's forcing
(installed at its first step) and, for `CMFMCConvection`, the CFL sub-step
count computed from the reset air mass. The tape records the convection at
that point with that interval, forcing and sub-step count; the observation
sampling point of the last step (its end-of-step record) follows the window
end. A checkpoint is a step boundary of the absolute schedule, so a Stride
window that restarts inside a met window runs that window's remaining steps
and its window-end stages exactly as the recording run did.

The reverse pass applies the transposes in the opposite order.
``E_n`` enters only through ``\mathcal{S}_n``, so

```math
\frac{\partial J}{\partial E_n} = \Delta t \, \lambda_n^{S}\big|_{k = N_z},
```

where ``\lambda_n^{S}`` is the adjoint state at the position of
``\mathcal{S}_n``: the final seed propagated back through all later
operators and the part of the center after the injection. The tape marks that
position with its emission record. ``\mathcal{H}`` is a linear copy,
``h = P\,i`` from interior to halo, so ``\mathcal{H}^T`` adds each halo
sensitivity into its source interior cell and zeroes the halo. Lin–Rood's
horizontal operator refills every halo cell it reads, so its step-start fills
are never read and have no record.

**Enforcement.** Three layers keep the adjoint the transpose of production:

1. *One source of truth*: the plan above; there is no second composition of
   the step in the adjoint.
2. *Setup guard*: `adjoint_support(x)` is `nothing` for every stage and
   operator the tape records and transposes, and a reason otherwise (the
   fallback for a new stage, operator or scheme). `require_adjoint_support(plan,
   ops)` — called by every entry point before recording, and available as
   `require_adjoint_support(sim)` for a `DrivenSimulation` — throws an error
   naming every unsupported piece, e.g. exponential-decay chemistry, the
   collaborative TM5 solve or clamped CMFMC.
3. *Parity matrix* (`test/core/test_cs_adjoint_step_plan_parity.jl`): for every
   supported scheme family, both cadences, TM5, CMFMC and CMFMC-matrix
   convection, both diffusion couplings, diffusion without surface sources and
   both reset modes, the adjoint's forward replay, its tape recording and a
   checkpoint restart inside a met window reproduce the production
   `DrivenSimulation` bit for bit; the captured fluxes are production's bit
   for bit, one copy per met window, and shared and per-step copies give the
   same footprints; Full and Stride footprints agree exactly; and
   footprints and the initial-condition sensitivity match central finite
   differences of the production simulation. A forward feature that
   changes the production step without a plan stage makes this test fail; a
   plan stage or operator without an adjoint makes the guard fail.

## From footprints to 4D-Var
## From footprints to 4D-Var

The inversion layer composes the reverse pass rather than reimplementing it:

| API | Purpose |
| --- | --- |
| `cs_surface_emission_footprint` | One scalar objective to per-step emission sensitivities. |
| `cs_surface_flux_jacobian` | Batch objectives and aggregate steps into named control windows. |
| `cs_surface_flux_4dvar` | Evaluate observation/background cost and its control gradient, with one forward run (which samples the observations and records the tape) and one reverse pass for all observations; `evaluation = CSCostOnly()` runs the forward run alone. |
| `cs_surface_flux_4dvar_optimize` | Optimize with `CSGradientDescent` or `CSLBFGS`; every evaluation records into one tape pool. |
| `read_observations`, `bind_to_mesh` | Read observation files (schema v2) and bind each record to its cell and time sampling: point records to `CSPlacedPointObjective`s placed in the column by the forward sampler's rule (height above ground and/or pressure), column records to column means. |
| `build_departure_set`, `read_departures`, `write_departures` | Exchange modeled-minus-observed diagnostics with each observation's sampled steps, levels and placement. |

As in TM5-4DVAR, one gradient is one adjoint integration. Each observation
is a fixed linear functional of the tracer mass after its model step (weights
from the replayed air mass); `cs_surface_flux_4dvar` runs the forward model
once, sampling the simulated values and recording the tape through the last
step of an observation with `assimilate = true`, and walks that tape back once
from the last assimilated step, adding every observation's weighted departure
`(H(x) − y)/σ²` at the end of its step, before the transposed convection of
that step (TM5 adds its forcing just before the adjoint of the operator after
which it sampled). See the [single-sweep gradient](@ref single-sweep-4dvar-gradient).
`cs_surface_flux_jacobian` records its tape (with `StrideCheckpoint`, each
window) once and walks it once per objective.

Controls are `CSSurfaceFluxControl` values associated with
`CSSurfaceFluxWindow`s. A `CSSurfaceFluxPreconditioner` can apply either a
linear or log-normal transform using a diagonal or isotropic-Gaussian
covariance. See the working assembly in `scripts/inversions/cs_4dvar.jl` and
the exact keyword contracts in [Adjoints and checkpointing API](@ref).

## Tape and checkpoint choices

The forward pass records typed advection, halo, emission, diffusion,
convection, air-mass reset and end-of-step operations through the production
step's recorder hooks. Lin-Rood horizontal updates use an additional dedicated
record. The reverse loop dispatches on those record types and calls the
matching transpose kernel.

Every reverse-pass entry point takes one `tape = AdjointTape(; content,
storage, checkpoint)` keyword (the inversion driver: `[adjoint.tape]`). The
`storage` controls where staged tape data lives:

| Value | Use |
| --- | --- |
| `DeviceCSTapeStorage()` | Default and simplest choice when the tape fits on the device. |
| `PinnedHostCSTapeStorage()` | Move staged data to pinned host memory when device memory is limiting. The host memory is allocated in large chunks, each pinned once; reads come back through one device buffer per operand (air mass, tracer mass). |
| `MmapCSTapeStorage(dir)` | Store the tape in a scratch file in `dir` (required) for tapes larger than memory. |

Each call opens the tape's memory once, reuses it for every stride window
(and, in `cs_surface_flux_4dvar_optimize`, every evaluation), and frees it
when it returns. An mmap tape is sized before it is recorded: the call counts
its largest tape on the forward replay (once per optimization), checks that
`dir` has that much space plus 10 % free (an `ArgumentError` with the
required and available bytes otherwise, raised after this sizing pass), and
reserves it with `fallocate` in one unlinked file that is unmapped and closed
when the call returns. Use a local file system that can reserve blocks (ext4,
XFS); others, such as ZFS or NFS without `fallocate`, are rejected by name.
Tapes are not persisted. Every storage works with every checkpoint policy.
Lin-Rood currently supports device tape storage only.

The `checkpoint` controls the memory/recompute trade-off:

| Policy | Stored state | Recompute behavior |
| --- | --- | --- |
| `FullCheckpoint()` | Proportional to the number of steps. | No window replay. |
| `StrideCheckpoint(k)` | Checkpoints every `k` steps and one `k`-step window. | Re-records one `k`-step window at a time (one extra forward run), walked for every objective. |

Both schedules record the same states: a checkpoint holds the tracer and air
mass with their halos and the persistent operator state (the surface sources'
Kahan compensation, the convection and Dkg diffusion ledger carries, the
CMFMC CFL cache), so a window that restarts at a step boundary, also inside a
met window, replays the recording run bit for bit. Stride footprints equal
`FullCheckpoint`'s (exactly in one thread; the halo and Lin–Rood transposes
accumulate with atomics when several threads run them) for every supported
scheme, including limited PPM with diffusion and convection. There is no
logarithmic-memory (Revolve) schedule; an optimal binomial one will be added
when a case needs it.

## How this maps to established systems

| Task | TM5-4DVAR / GEOS-Chem adjoint | AtmosTransport |
| --- | --- | --- |
| Forward integration | Separate CTM driver | `run_driven_simulation` or the lower-level CS arrays |
| Reverse forcing | Observation operator, departures injected during one adjoint run | `CSObservation` objectives as linear functionals, injected at each step's end in one reverse pass, or an explicit seed |
| Observation IO | System-specific files | `CSObservationSet` NetCDF IO (schema v2) + `bind_to_mesh` |
| Background covariance | B or B¹ᐟ² implementation | `DiagonalCSCovariance` / `IsotropicGaussianCSCovariance` |
| Preconditioning | Hand-coded control transform | `CSSurfaceFluxPreconditioner` |
| Optimization | M1QN3 / L-BFGS variants | `CSGradientDescent` / `CSLBFGS` |
| Driver | Dedicated inversion executable | `scripts/inversions/cs_4dvar.jl` for the synthetic workflow |

Point observations are placed as TM5-4DVAR places them in spirit, with
deliberate differences: by their height above the true ground measured up
from the model surface (TM5 compares sea-level altitudes with heights on the
model orography and never samples layer 1), or by pressure, in the forward
run and in the adjoint alike (TM5's adjoint ignores the pressure
coordinate). Windows are Δt/ΣΔt means as in TM5, with a step partly inside
the window weighted by its overlap; the sample is taken after the full step
(after convection). See [observation operators](@ref placed-observation-operator).

Satellite and TCCON columns with a retrieval block use TM5-4DVAR's
averaging-kernel operator (`averaging_kernels.F90`: the stretch of the model
column onto the retrieval column, the mass-conserving rebin of `inv_rebin`,
``X = \sum_i h_i a_i (R x)_i + \sum_i h_i (1 - a_i) x_{a,i}``), applied in the
model to the dry mixing ratios on total interface pressures rather than in
Python to the track file's moist profiles, with an explicit surface-pressure
mismatch limit (`max_surface_pressure_mismatch_pa`) and a model-error term
fixed by the record. See [averaging-kernel columns](@ref retrieval-column-operator).

OCO-2 and OCO-3 retrievals sit on 20 levels. TM5-4DVAR turns the levels into
the interfaces of 19 layers, interpolates the kernel and prior to the layer
midpoints and uses ``h = \Delta p/p_s`` (summing to 0.9999); AtmosTransport
projects the model profile onto the levels' hat functions with the file's
pressure weights instead, which conserves the column and keeps the
retrieval's representation. TM5's form is available for comparisons (with its
weights normalized to 1). Records not assimilated (withheld,
evaluation-only) can be co-sampled without entering the cost, which the MIP
asks for and TM5's MIP setup does differently (it assimilates
`assimilate_flag` 1 and 2). See [level retrievals](@ref hat-level-projection)
and [OCO-2 MIP records](@ref oco2-mip-records).

## Current limits

- The inversion driver binds observation files on its synthetic hydrostatic
  meteorology only; real transport binaries need the Julia API.
- CMFMC requires `clamp = false` in the footprint path.
- TM5 and CMFMC-matrix convection require the full, unmerged,
  non-collaborative solve (`use_collab_lu = false`, `lmax_conv = 0`,
  `n_merge = 1`).
- Chemistry has no adjoint record: a plan with exponential decay or other
  chemistry is rejected by the guard.
- The adjoint records one tracer; the per-step face fluxes and per-window
  convection forcing of a driven run are the caller's inputs
  (`CSStepSchedule(sim)` supplies the plan, windows and reset targets).
- A tangent-linear driver is not exposed.
- A real-data TM5-4DVAR or GCHP-adjoint cross-validation has not been
  published.
- The supplied TOML inversion driver supports synthetic constant meteorology;
  real transport-binary ingestion still requires programmatic assembly.

For the test-by-test support matrix and the distinction between verified
kernels and externally validated workflows, continue with
[Adjoint status](@ref) and [Validation status](@ref).
