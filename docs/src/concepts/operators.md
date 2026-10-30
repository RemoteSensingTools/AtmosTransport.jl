# [Operators](@id Operator-concepts)

Every physics process in AtmosTransport is implemented behind an
**abstract operator type** with a `No<Operator>` no-op default. The
runtime composes operators through a fixed Strang palindrome; users
swap implementations by changing one field in the TOML config (or by
implementing a new subtype and registering it).

## The five operator families

```mermaid
classDiagram
    class AbstractOperator
    class AbstractAdvectionScheme
    class AbstractDiffusion
    class AbstractConvection
    class AbstractSurfaceFluxOperator
    class AbstractChemistryOperator
    class UpwindScheme
    class SlopesScheme
    class PPMScheme
    class LinRoodPPMScheme
    class NoDiffusion
    class ImplicitVerticalDiffusion
    class NoConvection
    class CMFMCConvection
    class TM5Convection
    class NoSurfaceFlux
    class SurfaceFluxOperator
    class NoChemistry
    class ExponentialDecay
    class CompositeChemistry

    AbstractOperator <|-- AbstractAdvectionScheme
    AbstractOperator <|-- AbstractDiffusion
    AbstractOperator <|-- AbstractConvection
    AbstractOperator <|-- AbstractSurfaceFluxOperator
    AbstractOperator <|-- AbstractChemistryOperator
    AbstractAdvectionScheme <|-- UpwindScheme
    AbstractAdvectionScheme <|-- SlopesScheme
    AbstractAdvectionScheme <|-- PPMScheme
    AbstractAdvectionScheme <|-- LinRoodPPMScheme
    AbstractDiffusion <|-- NoDiffusion
    AbstractDiffusion <|-- ImplicitVerticalDiffusion
    AbstractConvection <|-- NoConvection
    AbstractConvection <|-- CMFMCConvection
    AbstractConvection <|-- TM5Convection
    AbstractSurfaceFluxOperator <|-- NoSurfaceFlux
    AbstractSurfaceFluxOperator <|-- SurfaceFluxOperator
    AbstractChemistryOperator <|-- NoChemistry
    AbstractChemistryOperator <|-- ExponentialDecay
    AbstractChemistryOperator <|-- CompositeChemistry
```

## The `apply!` contract

Each operator family's mutating entry point has a uniform shape, with
the second positional argument carrying whatever time-resolved data
that family needs:

| Family | Signature |
|---|---|
| Advection | `apply!(state, fluxes, grid, scheme, dt; workspace, cfl_limit, midpoint = NoMidpoint())` |
| Diffusion | `apply!(state, meteo, grid, op::AbstractDiffusion, dt; workspace)` |
| Convection | `apply!(state, forcing::ConvectionForcing, grid, op::AbstractConvection, dt; workspace)` |
| Surface flux | `apply!(state, meteo, grid, op::AbstractSurfaceFluxOperator, dt; workspace = nothing)` |
| Chemistry | `apply!(state, meteo, grid, op::AbstractChemistryOperator, dt; workspace = nothing)` |

Advection does not call diffusion or surface emissions itself. Between the two
halves of the Strang palindrome it calls `apply_midpoint!(midpoint, ...)`; the
default `NoMidpoint()` does nothing. `transport_step!` in `Operators.StepPlans`
builds a `TransportMidpoint` that applies diffusion and emissions there, so
diffusion and emission keywords belong to `transport_step!`, not to Advection.

Every method mutates `state` in place. Performance-sensitive implementations
keep reusable scratch buffers in workspaces. Each family provides an identity
operator (`NoAdvection`, `NoDiffusion`, `NoConvection`, `NoSurfaceFlux`, or
`NoChemistry`) so the runtime can compose one typed model without conditionally
calling absent physics.

Topology dispatch happens **inside** each `apply!` method, not on a
separate axis: an LL state and a CS state route through different
specialized kernels via Julia's multiple dispatch on the grid type.

## Advection

`AbstractAdvectionScheme` is the root; the concrete schemes live in
`src/Operators/Advection/Reconstruction/schemes.jl`:

| Subtype | Smooth-region reconstruction | Notes |
| --- | --- | --- |
| `UpwindScheme` | First order | Donor-cell; cheap, very diffusive. |
| `SlopesScheme{L}` | Second order | Russell–Lerner face flux with the slope re-diagnosed from neighbouring means every sweep (van Leer). Not TM5's prognostic-moment slopes scheme. Limiter parameter `L`. |
| `PPMScheme{L,V}` | Second order | Limited piecewise-parabolic edges set a Russell–Lerner slope. Limiter parameter `L`. `V = FV3ScalarProfile` (CS only) integrates FV3's `kord = 8` parabola in the vertical sweep. Supported on LL and CS split-sweep; RG supports upwind only. |
| `LinRoodPPMScheme` | Piecewise parabolic | FV3 horizontal cross terms (CS only), paired by default with the FV3 `kord = 8` vertical profile in configured runs; explicit `vertical = "upwind"` retains first-order vertical transport. `ppm_order=5` or `7` selects the horizontal edge-value family; both share the same interior reconstruction. |

Reconstruction order does not establish the temporal order or positivity of
the complete transport update. See [Advection schemes](@ref) for the boundary
treatment and [Validation status](@ref) for measured scheme coverage.

Limiter parameter `L` ranges over `NoLimiter`, `MonotoneLimiter`,
`PositivityLimiter` and, for `PPMScheme` only, `CW84Limiter` (the complete
Colella–Woodward PPM) — declared in the same file. `PPMScheme()` defaults
to `MonotoneLimiter()`. The monotone and CW84 limiters are signed and
constant-offset equivariant; only `PositivityLimiter` uses tracer zero as a
bound.

**TOML config** (preferred form):

```toml
[advection]
scheme = "slopes"     # "upwind" | "slopes" | "ppm" | "linrood"

# Cubed-sphere only: pick the Lin–Rood edge-value family (5 or 7).
# Only valid with scheme = "linrood"; setting ppm_order with
# scheme = "ppm" errors at config-parse time.
# scheme    = "linrood"
# ppm_order = 7
```

A `NoAdvection` identity operator is available for isolating other
operators (e.g. convection-only timing, regression). Select with
`[advection] scheme = "none"`. Diffusion-only runs are supported as a
single mass-conserving vertical solve. Surface fluxes require an advection
palindrome and are therefore rejected with `NoAdvection`.

## Diffusion

`AbstractDiffusion` is the root; concrete subtypes:

| Subtype | Use |
|---|---|
| `NoDiffusion()` | Identity no-op; default when `[diffusion]` is absent or `kind = "none"`. |
| `ImplicitVerticalDiffusion{FT, KzF, SFC}` | Backward-Euler vertical diffusion driven by an `AbstractTimeVaryingField` Kz. `SFC` chooses midpoint source splitting or source-before-full-solve ordering. |

Generic Kz diffusion uses a per-column backward-Euler tridiagonal solve for
the new mixing ratio and updates tracer mass in flux form (each interface flux
leaves one layer and enters the next), with a supported transpose in the CS
adjoint. The exact TM5 Dkg path instead uses
column-conservative bidiagonal factors directly on tracer storage, preserving
weak transfers and signed mass budgets. Its reverse applies the factor
transposes in reverse order. See [Adjoint status](@ref) for supported combinations.

**TOML config** (`[diffusion]` block):

```toml
[diffusion]
kind  = "constant"
value = 1.0      # Kz [m²/s]; broadcast to all (i, j, k)
```

`kind = "none"` (or omitting the block entirely) selects `NoDiffusion`.
Cubed-sphere binaries carrying PBL surface sections can use
`kind = "tm5_beljaars_viterbo_local_kz"`.
On lat-lon and cubed-sphere grids, `surface_flux_boundary = true` adds
configured surface mass before one full implicit vertical solve
(`S(dt) -> V(dt)`) instead of using the separate
midpoint split (`V(dt/2) -> S(dt) -> V(dt/2)`). The key name is historical:
this changes operator ordering, not the lower row of the tridiagonal system.
Reduced-Gaussian transport currently supports midpoint splitting only and
rejects `surface_flux_boundary = true` while building the runtime recipe.
Profile / derived / precomputed Kz fields exist in `src/State/Fields/` — see
[State & basis](@ref) for the full field-type list.

## Convection

`AbstractConvection` is the root; concrete subtypes:

| Subtype | Forcing carrier | Source |
| --- | --- | --- |
| `NoConvection()` | — | Identity no-op; default. |
| `CMFMCConvection()` | `ConvectionForcing.{cmfmc, dtrain}` | GCHP-style upwind moist convection; mass flux + optional detrainment. |
| `CMFMCMatrixConvection()` | `ConvectionForcing.{cmfmc, dtrain}` | Implicit matrix solve assembled from GEOS cloud-mass flux and detrainment. |
| `TM5Convection{FT}()` | `ConvectionForcing.tm5_fields.{entu, detu, entd, detd}` | TM5 Tiedtke-1989 four-field entrainment / detrainment with an implicit column solve. Parametric on `FT`. |

The convection facade re-exports the CMFMC and TM5 operator families from
`Operators.Convection.CMFMC` and `Operators.Convection.TM5`. Each family owns
its forward and adjoint kernels. `Convection.convection_workspace_for` builds
the runtime workspace; checkpoint storage uses the shared Operators interface.

The convection subtypes consume a `ConvectionForcing` carrier (declared in
`src/MetDrivers/ConvectionForcing.jl`) — different physics, identical
plumbing. `_refresh_forcing!` populates `model.convection_forcing`
at the first substep of each window by copying from the current met
window; the operator does not call `current_time` itself.

**TOML config** (`[convection]` block):

```toml
[convection]
kind = "cmfmc"     # or "cmfmc_matrix" / "tm5" / "none"
```

The runtime picks `:cmfmc` only against binaries whose header carries
the `:cmfmc` payload section (and `:dtrain` if requested); similarly
for `:tm5` requiring `:entu / :detu / :entd / :detd`. Asking for a
convection scheme the binary does not support is a **load-time
error**, not a silent fallback.

## Surface flux (sources)

`AbstractSurfaceFluxOperator` is the root; concrete subtypes:

| Subtype | Use |
|---|---|
| `NoSurfaceFlux()` | Identity no-op; default. |
| `SurfaceFluxOperator{M}` | Applies a `PerTracerFluxMap` of `SurfaceFluxSource`s to the bottom-most layer (`k = Nz`). |

`SurfaceFluxOperator` is materialized from
`[tracers.<name>.surface_flux]` blocks and is the path for emissions
inventories (EDGAR, GFED, GridFED, LMDz, …). See the worked CATRINE configs
(`config/runs/catrine_*.toml`) for examples.

### Inventories on the model grid

A static lat-lon inventory has flux densities ``F_s`` [kg m⁻² s⁻¹] on source
cells ``s`` with areas ``A_s``: the file's `cell_area` or `area` variable, or,
without one, the latitude-circle area
``R^2\,\Delta\lambda_s(\sin\varphi_{s+1/2}-\sin\varphi_{s-1/2})`` of the cell
on the model sphere. The file's own total is ``T = \sum_s F_s A_s``; for a
per-cell inventory (EDGAR tonnes per cell) ``F_s A_s`` is the stored amount
itself. The source is the per-cell rate ``r_t`` [kg s⁻¹] of model cell ``t``
(`surface_flux.regridding`):

- **Conservative** (default, every topology). The rate ``m_s = F_s A_s`` is
  regridded as an extensive field with the intersection areas ``W_{ts}`` of the
  source and model cells,

  ```math
  r_t = \sum_s \frac{W_{ts}}{C_s}\, m_s, \qquad C_s = \sum_t W_{ts}, \qquad
  \sum_t r_t = \sum_s m_s = T .
  ```

  ``C_s`` is the area of source cell ``s`` that model cells cover
  (`Regridding.source_coverage`). The regridder's cells have great-circle
  edges, so a lat-lon cell's polygon is not its latitude-circle cell:
  regridding the density ``F_s`` instead would conserve
  ``\sum_s F_s A^{\rm gc}_s``, which differs from ``T`` by up to
  ``5\times10^{-5}`` per 1° cell and ``7\times10^{-4}`` per 3.75°×1.875° cell (and
  by the file's radius convention, about ``7\times10^{-5}`` for the CAMS
  `area`). On reduced-Gaussian grids two rings with different longitude
  counts share a latitude as two different great-circle polylines, so a
  sliver between them belongs to no model cell and ``C_s`` is smaller than
  the source cell (on O24, 6.3e-4 of the sphere is uncovered and a 1° cell
  loses up to 6.4 %; the density regrid gave a December 2021 CAMS total
  1.2e-3 below the file's there). Dividing by ``C_s`` gives each source cell's whole amount
  to the model cells it overlaps. On a target that does not cover the sphere
  (a regional or latitude-truncated mesh; the shipped runs are global) this
  also holds for a source cell that the target covers only in part: its whole
  amount goes into the covered part, whose density is raised by
  ``A_s / C_s`` (+4.8 % in the boundary row of a 1° source on a lat-lon target
  ending at 74.75° N), and source cells that no model cell covers are dropped.
  Telling a sliver from a partial cover would need an area threshold, so this
  rule is kept and documented rather than special-cased.
- **Bilinear** (lat-lon and reduced-Gaussian, only when set). The density is
  sampled at the model cell centre ``x_t``, ``\tilde r_t = F(x_t)\,a_t``, with
  the model cell area ``a_t``. The emission and uptake parts are renormalised
  separately,

  ```math
  r_t = \tilde r_t \frac{P}{p}\ (\tilde r_t > 0), \qquad
  r_t = \tilde r_t \frac{N}{n}\ (\tilde r_t < 0), \qquad
  P = \sum_s \max(m_s, 0),\ N = \sum_s \min(m_s, 0),
  ```

  with ``p`` and ``n`` the same sums of ``\tilde r_t``. Both factors are
  positive, so no cell changes sign, and ``\sum_t r_t = P + N = T``.
  Renormalising the net total instead fails for fields with uptake: the
  factor ``T/\sum_t \tilde r_t`` is arbitrary when the net is near zero and
  negative when the sampled net has the other sign, which turns every source
  into a sink (10 of 20 random fossil-plus-uptake fields on a 72×46 grid).
  When ``p = 0 \ne P`` or ``n = 0 \ne N`` no factor exists, and the build is an
  error; one uptake cell that falls between the sampled centres is enough.
  The factors keep the two totals, not the cell values, and can be far from
  1: the uptake of the GridFED December 2021 file on a 5° LL72 grid has
  ``N/n = 3.53``, so every sampled uptake cell is multiplied by 3.53 (0.83 on
  2.5°, 2.38 on 1°). Both factors are logged, and a factor more than 10 % from
  1 is a warning. Point sources stay aliased: a 2° hotspot lands in whichever
  model cell's centre samples it.

Cubed-sphere grids always use the conservative path. Time-varying lat-lon
series (cubed sphere only) are regridded slice by slice as densities with one
regridder, so their slices conserve ``\sum_s F_s A^{\rm gc}_s``; their loader
reads no file areas.

### Time coverage of time-varying sources

A time-varying source evaluates its slices with clamped temporal schemes, so
outside ``[t_1, t_n]`` it holds its first or last slice. The loaders therefore
require the run ``[t_{\rm start}, t_{\rm end}]`` to lie inside the union of
the slices' intervals, taken over all files of the source together. A slice
with CF time bounds stands for its bounds, a `months since` stamp for its
month (`GriddedIO.known_slice_intervals`). The stamps without bounds of all
files are merged and sorted, and each holds for the regular step of that
series (`GriddedIO.stamp_intervals`):

- month starts step by one month, or by one year from a 1 January to the
  next: slice ``k`` holds ``[t_k, t_{k+1})``, the last slice one calendar
  month (one year after a yearly step);
- other stamps step by their smallest spacing ``\Delta``: slice ``k`` holds
  ``[t_k, t_{k+1})`` when ``t_{k+1} - t_k = \Delta`` (within 1 s), the last
  slice ``\Delta``;
- a slice before a longer spacing (Jan 1 then Mar 1, or a missing day) holds
  for one step only, so the gap stays uncovered.

So daily files with one 00Z stamp each are a daily series, and only the very
last stamp is extended. The period cannot be inferred, and the loaders stop
with an error that asks for CF bounds, for a single stamp in all the files (a
1 January 00Z stamp may be a day, a month or a year) and for stamps about a
month or a year apart that are not month starts (mid-month stamps: held from
the stamp or centred on it?). A part of the run that no slice covers,
including a gap between or inside files, is an `ArgumentError` naming the
dates, unless the source sets `hold_outside_coverage = true`. The runner's run span ends 1 s after its last
window for the rounding of its accumulated clock
(`RUN_SPAN_CLOCK_SLACK_SECONDS`); the files need not cover that second.

### Time-varying sources on a GPU

A `TimeVaryingSurfaceFluxSource` (cubed sphere; `time_varying = true`) stores
its per-cell rates as slices ``r_{c,k}`` at times ``t_k``. The series stays in
host memory on every backend. On the device each source holds a small window
buffer with the slices that the current met window ``[t_0, t_1]`` reads, and
the surface-flux kernels read the rates from it:

- `StepwiseFlux` reads the slices in effect over the window, ``k`` from
  ``\max\{k : t_k \le t_0\}`` to ``\max\{k : t_k \le t_1\}`` (clamped to the
  series).
- `LinearInterpFlux` and `ConservativeMeanFlux` read from the left bracket
  slice at ``t_0`` to the right bracket slice at ``t_1``.

``t_1`` is padded by
``8\,\varepsilon_\text{Float32}\,(t_1 - t_0) + 4\,\mathrm{ulp}(t_1)``, because
a Float32 step (for example 5400/7 s) can end slightly after the window. When a
window becomes current, `DrivenSimulation` copies the slices of this range that
the buffer does not hold yet, on the task that then launches the window's
kernels. The buffer is a ring of slots (slice ``k`` in slot
``\mathrm{mod1}(k, \text{capacity})``), so a slice that consecutive windows
share is copied once, and its capacity is the largest window's slice count
(two slices for stepwise, three for linear or conservative sources with daily
or hourly series and hourly met windows), independent of the run length. The
kernels see the same rates as with the
whole series on the device, so results are bit-identical; a C90 2014–2024
daily series (0.78 GB per tracer in Float32) now costs a few 0.19 MB slices of
device memory per tracer. The emission totals below sum the host series.

### Emission totals

Sources hold per-cell model-storage rates ``r_c`` (kg dry-air equivalent
s⁻¹). Logs and budget diagnostics use global totals that the surface-flux
module computes from the operator's own temporal reconstruction:

- `total_emission_rate(src, t)`: the global rate ``R(t) = \sum_c r_c(t)`` at
  the instant `t` (seconds since the run start).
- `emission_segments(src, t, dt)`: the pieces ``(R_s, \tau_s)`` in which one
  step ``[t, t + dt]`` adds the source, with ``\sum_s \tau_s = dt``.
- `integrated_emission(src, t0, t1, dt)`: the mass added over the steps
  ``t_n = t_0 + n\,dt``, with ``N = (t_1 - t_0)/dt`` steps:

```math
E = \sum_{n=0}^{N-1} \sum_s R_s^{(n)}\,\tau_s^{(n)} .
```

A time-varying source stores slices with totals ``R_k = \sum_c r_{c,k}``.
Every temporal scheme blends at most two slices,
``R = w_0 R_{i_0} + w_1 R_{i_1}``, so these totals equal the sum of the
operator's per-cell increments up to summation order
(`test/core/test_surface_flux_integrals.jl` checks this to 1e-14 relative).
`StepwiseFlux` and `ConservativeMeanFlux` split each step at every slice time
and evaluate each piece at its midpoint. This integrates the piecewise-constant
or piecewise-linear reconstruction exactly, so their ``E`` does not depend on
``dt``. `LinearInterpFlux` evaluates the rate at the step start, so its ``E``
is a Riemann sum that depends on ``dt``.

The runner logs `total_emission_rate(source, 0.0)`, the rate at the run start,
for every source. `Analysis.budget_closure` (the `series` mode of
`scripts/diagnostics/tracer_budget_closure.jl`), `Analysis.carrier_drawdown`
and `scripts/diagnostics/catrine_true_mass_balance.jl` rebuild a run's sources
and integrate them with these functions.

## Chemistry

`AbstractChemistryOperator` is the root; concrete subtypes:

| Subtype | Use |
|---|---|
| `NoChemistry()` | Identity no-op; default. |
| `ExponentialDecay` | Applies configured first-order loss rates to selected tracers. |
| `CompositeChemistry` | Applies a typed tuple of chemistry operators in sequence. |

Constant-rate exponential decay uses exact half-steps before and after the
transport/emissions window. Other chemistry, including exponential decay with
a time-varying rate, runs after the transport and convection blocks. The same operator interface covers `CellState` and
`CubedSphereState`; topology dispatch selects the storage-specific kernel
launch.

## Transport composition

StepPlan owns emission/diffusion composition. Advection calls the abstract
`apply_midpoint!` hook on its active buffers between palindrome halves; the
default `NoMidpoint` performs no work. StepPlan supplies `TransportMidpoint`,
including physics workspaces, tracer selection, timing and CS recorder hooks.
The adjoint replays the same plan and transposes its records in reverse order.
Direct advection callers compose physics through `StepPlans.transport_step!`.

The directional advection shell is a palindrome. Its center follows the
diffusion operator's surface-coupling policy:

```text
forward:   (X → Y → Z) × n   (fluxes / n; n = 1 unless CFL subcycling is needed)
center A:  V(dt/2) → S(dt) → V(dt/2)   (default symmetric Strang split)
center B:  S(dt) → V(dt)               (LL/CS source-before-full-solve policy)
           [without surface flux, the center is one V(dt); NoDiffusion is a no-op]
reverse:   (Z → Y → X) × n   (same n; the center runs once)
post:      apply!(convection)   (convection is outside the palindrome)
           apply!(chemistry)    (chemistry is outside the palindrome)
```

`V` is `apply_vertical_diffusion_vmr!` and `S` is `apply_surface_flux!`.
Center A is time-symmetric and lets fresh bottom-layer storage diffuse in both
half solves. Center B reproduces the supported GCHP-style placement of fresh
surface storage before one full solve; it is an ordering policy, not a
time-symmetric Strang center or a different tridiagonal boundary condition.
Reduced-Gaussian transport implements Center A only.

Convection and chemistry sit **outside** the palindrome for the same
reason: they are not commutative with advection at the per-substep
level, and their natural cadence is the met window rather than the
advection sub-step.

## Adding a new operator

The same recipe applies in every family:

1. Subtype the abstract root: `struct MyConvection <: AbstractConvection; … end`.
2. Provide a `No<Operator>` peer if one doesn't already cover your slot — usually it does.
3. Implement `apply!(state, …, op::MyConvection, dt; workspace)` for whichever grid types you support. Multiple dispatch on the grid type handles topology specialization.
4. Wire selection from TOML through `RuntimePhysicsSpecs.jl` and its topology-dispatched `materialize` methods.
5. Test that the `No<Operator>` path is bit-exact to the explicit no-op — this is the contract that lets future code skip the slot for free.

For an operator at the transport midpoint, extend StepPlan’s typed midpoint
composition in `src/Operators/StepPlan/midpoint.jl`, including recorder and
adjoint support. Advection supplies current buffers through `apply_midpoint!`;
keep its sweep drivers independent of the added physics.

## What's next

- [Binary format](@ref) — what the operator's input data looks like
  on disk, and how the runtime validates it.
