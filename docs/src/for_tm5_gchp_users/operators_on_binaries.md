```@meta
CurrentModule = AtmosTransport
```

# Operators on top of the binary

With the daily binary mmap'd and a typed `state.air_mass` on the same
basis, every physics operator becomes a small `apply!(state, ...,
op, dt)` call. This page maps the five operator families
(advection, diffusion, convection, surface flux, chemistry) onto the vocabulary
TM5 and GCHP users already have.

If you have only read the [Binary pipeline](binary_pipeline.md) page,
the relevant takeaway is that at *every* substep the runtime has on
hand:

- `m` — air mass per cell on the binary's basis (dry by default),
- `am`, `bm` — horizontal mass fluxes through x- and y-faces (or
  `hflux` on RG),
- `cm` — vertical mass flux at level interfaces (closed against
  `dm` at write time),
- `ps` — surface pressure per cell,
- optionally paired `qv_start`/`qv_end` endpoints, `cmfmc`/`dtrain`,
  or the four TM5 fields.

Operators consume these directly; no operator builds a flux field
from raw winds at run time.

## Transport composition

Each transport substep uses a directional advection palindrome with a
policy-selected center. Convection and chemistry are separate window-cadence
blocks:

```
transport shell:  X → Y → Z → CENTER → Z → Y → X
default CENTER:   V(dt/2) → S(dt) → V(dt/2)
LL/CS alternate:  S(dt) → V(dt)
convection: apply!(state, convection_forcing, grid, convection, dt)
chemistry:  apply!(state, meteo, grid, chemistry, dt)
```

Where:

- `X`, `Y`, `Z` are half-step directional sweeps,
- `V` is vertical diffusion (the Thomas implicit solve),
- `S` is the surface flux deposition,
- the default midpoint center is the time-symmetric Strang composition,
- the `surface_flux_boundary = true` LL/CS center is GCHP-style
  source-before-full-solve ordering and is not a symmetric Strang split.

For binary-scheduled drivers the full `step!` is bypassed; the
runtime calls `transport_step!` per substep and
`convection_chemistry_step!` once per window. This is the GCHP-style
"convection and chemistry after the dynamics step" placement, and is
*different from TM5's `xyzvvzyx` interleaving* — TM5 runs convection
inside the vertical-only legs. See the philosophy page for the
rationale.

```mermaid
flowchart LR
    X1[X half] --> Y1[Y half] --> Z1[Z half]
    Z1 --> V1[V dt/2]
    V1 --> S[S dt]
    S --> V2[V dt/2]
    V2 --> Z2[Z half]
    Z2 --> Y2[Y half] --> X2[X half]
    X2 --> CONV[Convection<br/>once per window]
    CONV --> CHEM[Chemistry]
```

## Advection

### Scheme selection by type

`AbstractAdvectionScheme` has four production subtypes:

| Scheme | What TM5/GCHP user it maps to | Grids | Multi-tracer fusion |
| --- | --- | --- | --- |
| `UpwindScheme` | First-order upwind (debugging only) | LL, RG, CS | yes |
| `SlopesScheme` | Russell-Lerner slopes / "TM5 sl_advection" | LL, CS | yes |
| `PPMScheme` | PPM with Colella-Woodward monotonicity (GCHP PPM-NV5) | LL, CS split-sweep | yes |
| `LinRoodPPMScheme{ORD}` (ORD ∈ {5,7}) | FV3 Lin-Rood PPM (GCHP dynamics) | CS only | **no** (per-tracer loop) |

Reduced Gaussian currently supports `UpwindScheme` only; requesting a
higher-order scheme on RG is rejected before transport begins.

The TOML `[advection].scheme` key parses to a Julia type at config
time; the runtime sees the typed value and dispatches at compile
time. *There is no runtime branch on the scheme name.* Subtyping is
the extension mechanism.

### Cubed-sphere palindrome and the positivity budget

The CS Strang split is a six-leg palindrome: `X → Y → Z → Z → Y → X`.
Each leg may be **subcycled** by an integer count `n_x`, `n_y`, `n_z`
chosen to keep tracer mass positive. The subcycle count comes from a
*palindrome positivity budget* tighter than per-direction CFL:

```
ratio = 2 · (out_x + out_y + out_z) / m_start
n_sub = ceil(ratio / safety_factor)
```

with `out_d = max(0,-a_lo) + max(0,a_hi)` (the Lin-Rood 1996
refinement, not `max(|a_lo|, |a_hi|)`). The factor of 2 is because
the palindrome traverses each direction twice. The preprocessor's positivity
gate uses the same budget against `min(m, m_next)`, for cubed-sphere and
lat-lon binaries alike. For binaries written
with adaptive substeps, the preprocessor bakes the verified count into
`steps_per_window_by_window[k]`. The binary-scheduled runtime executes exactly
that many transport substeps and fixes the inner directional subcycle count at
one; it neither re-pilots nor refines the stored schedule.

This is the analogue of TM5's CFL-driven substep, hardened against
the palindrome's specific stacking pattern.

### Multi-tracer fusion

For `SlopesScheme` and `PPMScheme` on the CS split-sweep path, each
direction's sweep packs **all tracers into a single kernel** —
`_cs_xsweep_mt_kernel!`, `_cs_ysweep_mt_kernel!`,
`_cs_zsweep_mt_kernel!`. The palindrome launches six kernels total
regardless of tracer count (one per leg of `X → Y → Z → Z → Y → X`).
On GCHP / TM5 the tracer loop is *inside* the PPM call; you launch
`6 · Nt` kernels per palindrome. For `Nt = 50` this is the
difference between 6 and 300 launches per palindrome.

```mermaid
flowchart LR
    subgraph GCHP/TM5
        L1[Outer: scheme step] --> L2[for tracer in 1..Nt]
        L2 --> L3[PPM x-flux] --> L4[PPM y-flux] --> L5[PPM z-flux]
    end
    subgraph AtmosTransport[AtmosTransport CS split-sweep]
        M1[X kernel<br/>all tracers] --> M2[Y kernel<br/>all tracers]
        M2 --> M3[Z kernel<br/>all tracers]
    end
```

The `LinRoodPPMScheme` path on CS does *not* fuse and runs the
per-tracer loop. This is a known asymmetry — the Lin-Rood
horizontal step is structurally per-tracer and the multi-tracer
fusion would require restructuring `fv_tp_2d_cs!` to loop tracers
inside the update kernel.

### Adaptive substeps

```mermaid
flowchart TD
    A[Window k starts] --> B[Read binary substep count<br/>steps_per_window_by_window[k]]
    B --> C[Fix inner directional<br/>subcycle count to 1]
    C --> D[Run one palindrome substep]
    D --> E[Advance substep counter]
    E --> F{Stored substeps done?}
    F -->|no| D
    F -->|yes| G[Window k complete]
```

The positivity budget is a write-time preprocessing gate. A v4 runtime trusts
that verified contract and does not alter the schedule; a failing product must
be regenerated with a safer preprocessing schedule.

## Diffusion

`ImplicitVerticalDiffusion` is the production diffusion operator. It
runs the standard tri-diagonal Thomas solve column-by-column. The
diffusivity profile is supplied by an `AbstractKzField`:

| Kz field | Source | Status |
| --- | --- | --- |
| `ProfileKzField` | Static profile from TOML | Production |
| `DerivedKzField` | Beljaars–Viterbo local Kz from `(ps, u, v, T, q, z0)` | Production |
| `WindowPBLKzField` | PBL-aware variant of Beljaars–Viterbo | Production |
| `LocalHoltslagBovilleKzField` | GEOS/VDIFF local Holtslag–Boville Kz | Supported when the v4 binary carries all VDIFF fields |

For TM5 users: `DerivedKzField` is the closest analogue to the
Holtslag–Boville-with-Beljaars-correction profile TM5 ships with.
For GCHP users: `LocalHoltslagBovilleKzField` follows the local GCHP/VDIFF
formulation almost line-for-line, with the same `(R_dry, cp_dry,
karman)` parameters and the same non-local counter-gradient term.

### Surface-flux and diffusion ordering

The diffusion operator pairs with a surface-flux operator through a typed
composition policy:

- `SplitSurfaceFluxCoupling` — two half diffusion steps bracket the surface
  source: `V(dt/2) → S(dt) → V(dt/2)`.
- `DiffusiveSurfaceFluxBoundary` — surface mass is injected first and then
  participates in one full implicit solve: `S(dt) → V(dt)`.

Despite its historical name, `DiffusiveSurfaceFluxBoundary` is an ordering
policy, not a distinct lower-boundary discretization. With fixed linear
diffusion coefficients, injecting the source before the implicit solve gives
`(I - dt*L) * rm_new = rm_old + dt*source`. This is algebraically equivalent
to adding that source to the backward-Euler right-hand side, including a
bottom-layer surface source; the equivalence does not require small `dt`.
Matching that placement alone does not establish full GCHP VDIFF parity.
On lat-lon and cubed-sphere grids, use
`surface_flux_boundary = true` for the supported GCHP-style placement.
Reduced-Gaussian transport currently supports the midpoint split only and
rejects that setting. Treat exact discrete VDIFF parity as unvalidated
until the full-model comparison listed in [Validation status](@ref) exists.

## Convection

Three production operators today:

### `TM5Convection`

Consumes the four TM5 fields from the binary (`:entu`, `:detu`,
`:entd`, `:detd`) and runs the TM5 column solve via a four-band
implicit operator. This is the path for ERA5-driven runs. The
algorithm follows `tm5/levitation.F90` closely; the column solve is
in `src/Operators/Convection/tm5_column_solve.jl` and the kernels
in `tm5_kernels.jl`.

### `CMFMCConvection`

Consumes `:cmfmc` (and optionally `:dtrain`) from the binary and
runs the GCHP-style upwind moist convection scheme. This is the
path for GEOS-driven runs. The CFL substep is chosen at met-window
start from the maximum cell-relative mass flux.

### `CMFMCMatrixConvection`

Consumes `:cmfmc` and `:dtrain`, derives TM5-style entrainment and detrainment
rates, and delegates to the implicit `TM5Convection` column solve. This offers
the GEOS forcing contract with matrix-solve numerics on LL, RG, and CS.

### Placement note (important for TM5 users)

We place convection **after** the FV transport block — once per met
window, between the palindrome's last `X` sweep and chemistry. TM5
interleaves convection inside the palindrome's vertical legs
(`xyzvvzyx` → convection inside the `vv`). The GCHP placement is
faithful to the corresponding GCHP CTM placement in `gchp_chunk_mod.F90`; the
TM5 placement is faithful to TM5-4DVAR's documented order.

For research questions where convective tracer transport in the
column matters (`222Rn` profiles, deep-convective venting), the
placement matters; we have a follow-up plan to support both
placements via a `[run].convection_placement` setting.

## Surface flux

`SurfaceFluxOperator` carries a `PerTracerFluxMap` — a typed mapping
from tracer name to an `AbstractSurfaceFluxSource`. Each source
supplies its surface flux as **kg/s per cell, already
area-integrated**.

For TM5/GCHP users this means:

- EDGAR/GFED/HEMCO `kg/m²/s` raster needs an explicit
  multiplication by cell area at TOML-build time. This is the
  preprocessor's job, not the runtime's.
- The runtime adds the flux mass to the bottom cell (`k = Nz`) at
  the surface step. With `DiffusiveSurfaceFluxBoundary` the
  diffusion solve then redistributes that mass upward.

```@example
# Conceptual layout of the per-tracer flux map (do not actually run)
# Each source struct knows how to read its data and inject into
# the appropriate tracer at apply! time.
```

## Composition rules and apply! contract

Every operator has the same `apply!` shape:

| Family | Signature |
| --- | --- |
| Advection | `apply!(state, fluxes, grid, scheme, dt; workspace, cfl_limit, diffusion_op, emissions_op, meteo)` |
| Diffusion | `apply!(state, meteo, grid, op::AbstractDiffusion, dt; workspace)` |
| Convection | `apply!(state, forcing::ConvectionForcing, grid, op::AbstractConvection, dt; workspace)` |
| SurfaceFlux | `apply!(state, sources::PerTracerFluxMap, grid, op::AbstractSurfaceFluxOperator, dt; workspace)` |

The second positional argument carries the time-resolved data the
operator needs from the binary. Workspaces are pre-allocated
parametric structs that own the scratch buffers and the per-backend
kernel objects.

## What's not yet a typed operator

A small list of known gaps you may run into if you are porting a TM5
or GCHP workflow:

- **Wet deposition** is not yet an operator family. Aerosol /
  scavenging users will need to add an `AbstractWetDeposition`.
- **Dry deposition** lives inside the surface flux operator today;
  it does not have its own resistance-based abstraction. GCHP-style
  deposition users will want a separate `AbstractDryDeposition`.
- **Photolysis / fast chemistry** is out of scope. Only first-order
  decay (`ExponentialDecay`) is shipped; this is enough for
  `222Rn → 222Pb` and similar half-life tracers.

The current validation boundary is summarized in
[Validation status](../theory/validation_status.md).

## Reading next

- For the sensitivity side (adjoint of every operator above), see
  [Adjoints on top of the binary](adjoints.md).
- For how the multi-tracer kernels actually launch on CPU, CUDA, and
  Metal from a single source, see
  [Kernel architecture](kernel_architecture.md).
- For the deep-physics treatment (palindrome derivation, monotonicity, and
  signed-tracer boundedness tradeoffs), see
  [Advection schemes](../theory/advection_schemes.md).
