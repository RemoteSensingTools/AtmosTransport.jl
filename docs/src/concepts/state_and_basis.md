# State & basis

Once the [Grids](@ref) are set, prognostic state lives in a
**state object** that carries air mass and one or more tracer-storage fields. There
are two state types — one for `LatLon` (structured) and `Reduced
Gaussian` (face-indexed / unstructured) sharing the same `CellState`
layout, and one for the panel-native cubed-sphere — but they share
the same accessor API and the same dry-basis contract.

## State types

| State | Topology | Storage shape per tracer |
|---|---|---|
| `CellState{Basis, A, Raw, Names}` | LatLon (structured), ReducedGaussian (face-indexed) | `(Nx, Ny, Nz)` (LL) or `(ncells, Nz)` (RG) |
| `CubedSphereState{Basis, A3, Raw4, Names}` | CubedSphere | `NTuple{6, (Nc + 2Hp, Nc + 2Hp, Nz)}` per panel (halo-padded) |

Both types are **GPU-aware**: their array fields are parametric, so
`Adapt.adapt_structure` ships them onto a `CuArray` (or another
backend) without copying any logic.

```mermaid
classDiagram
    class CellState {
      +air_mass     :: A
      +tracers_raw  :: Raw
      +tracer_names :: NTuple{Nt, Symbol}
      +mass_basis() :: AbstractMassBasis
    }
    class CubedSphereState {
      +air_mass     :: NTuple{6, A3}
      +tracers_raw  :: NTuple{6, Raw4}
      +tracer_names :: NTuple{Nt, Symbol}
      +halo_width   :: Int
    }
    class AbstractMassBasis
    class DryBasis
    class MoistBasis
    AbstractMassBasis <|-- DryBasis
    AbstractMassBasis <|-- MoistBasis
    CellState ..> AbstractMassBasis        : Basis param
    CubedSphereState ..> AbstractMassBasis : Basis param
```

### [`CellState{Basis, A, Raw, Names}`](@id CellState)

```julia
struct CellState{Basis <: AbstractMassBasis, A, Raw, Names}
    air_mass     :: A         # per-cell mass (kg) on the chosen basis
    tracers_raw  :: Raw       # packed χ × carrier-air-mass storage
    tracer_names :: Names     # NTuple{Nt, Symbol}
end
```

User-facing surface:
- `state.air_mass` — per-cell mass, shape `(Nx, Ny, Nz)` or `(ncells, Nz)`.
- `state.tracers.<name>` — non-allocating view of one tracer's conservative storage.
- `state.tracer_names` — tuple of symbols.
- `mass_basis(state)` — `DryBasis()` or `MoistBasis()`.

### [`CubedSphereState{Basis, A3, Raw4, Names}`](@id CubedSphereState)

```julia
struct CubedSphereState{Basis, A3, Raw4, Names}
    air_mass     :: NTuple{6, A3}     # per panel (Nc+2Hp, Nc+2Hp, Nz)
    tracers_raw  :: NTuple{6, Raw4}   # per panel (Nc+2Hp, Nc+2Hp, Nz, Nt)
    tracer_names :: Names
    halo_width   :: Int               # Hp from mesh
end
```

User-facing surface mirrors `CellState`. The halo padding is exposed in
`halo_width`; advection sweeps and panel-edge flux rotation use it.

## Signed tracer contract

Air mass is the positive carrier; tracer storage is a conservative signed
contribution:

```math
m_{air} > 0, \qquad r_t \in \mathbb{R}, \qquad q_t = r_t / m_{air} \in \mathbb{R}.
```

Negative `r_t` or `q_t` is valid. It can represent an anomaly relative to a
reference VMR, photosynthetic uptake, an adjoint variable, or one signed
component of a physical tracer. The runtime does not clamp tracer storage to
zero. Air-mass positivity is instead protected by the transport-binary CFL
contract and runtime subcycling; a non-positive carrier cell is invalid input,
not a reason to change tracer sign.

For a constant reference VMR `q0`, store `r_anomaly = r - q0 * air_mass` and
restore `q = r_anomaly / air_mass + q0` in diagnostics. Because air mass evolves,
restoring the reference in mass space requires the current `q0 * air_mass`, not
a fixed tracer-mass offset.

Offset equivariance is not the same as linear superposition. The default
monotone limiter chooses its branch from each transported field, so separately
transported GPP and respiration components need not sum exactly to a run that
transports their combined field. Use a linear scheme (`UpwindScheme` or an
unlimited reconstruction) when exact component additivity is required, or a
future frozen-coefficient/replay path that applies one combined-field limiter
decision to every component.

Snapshot capture computes a compensated Float64 sum of each signed storage
field. NetCDF exposes it as `<tracer>_total_mass`; ATMSNAP keeps the same value
in its JSON header while its large spatial payload remains Float32. Use this
scalar time series for conservation checks where positive and negative
components nearly cancel. It is model storage (`q × carrier-air-mass`), not
physical kilograms of the tracer species.

## The dry-basis contract

By default, `state.air_mass` carries **dry-air mass** and every tracer
is interpreted on a **dry-VMR contract**. This is the single most
important runtime invariant in the project: trace-gas VMRs are always
dry VMRs, including column averages.

A subtle point: **`state.tracers_raw` is model storage, not physical kilograms
of the chemical species.** On `DryBasis`, the stored quantity is

```math
r_m = \chi_{dry}\,m_{dry},
```

so it has a mass-like carrier unit and divides directly by dry-air mass to
recover dry mole fraction. This representation composes naturally with
mass-flux advection. The conversion happens at the boundaries:

- **Initial conditions.** A TOML entry
  ```toml
  [tracers.co2.init]
  kind       = "uniform"
  background = 4.0e-4
  ```
  is read as a *dry VMR* (`background`) and converted to storage through
  `pack_initial_tracer_mass`, which applies `χ × air_mass` on `DryBasis`.
- **In the loop.** Operators consume `state.tracers_raw` as a conservative
  mass-like field; the
  mass-conservation contract holds because both `air_mass` and the
  tracer-storage slices are pinged through the same flux divergence.
- **Snapshot output.** `<tracer>_column_mean` is
  `sum(r_m) / sum(m_dry)` — a dry VMR by construction.
- **Programmatic readout.** `mixing_ratio(state, :CO2)` (in
  `CellState.jl`) gives you the dry VMR directly:
  `get_tracer(state, :CO2) ./ state.air_mass`.

Physical surface inventories are different: a rate in kg species s⁻¹ is
multiplied by ``M_{dry\ air} / M_{species}`` before entering model storage.
The species comes from `surface_flux.species` (`co2`, `sf6`, `rn222`, with the
molar masses of `Parameters.SPECIES_MOLAR_MASS`); tracers named after a species
may omit it. The output variable
`<tracer>_column_mass_per_area` is the **model storage mass** per area and does
not undo that molecular-weight scaling.

What this means for the binary side:

- The transport binary's `mass_basis = :dry` header says the
  preprocessor already converted `DELP_moist → DELP_dry × (1 − qv)`.
- `state.air_mass` comes from that dry mass.
- Snapshot output writes `<tracer>_column_mean` etc. consistently on
  that dry basis.

Operators dispatch on the basis tag so a `MoistBasis` state would
automatically take a different code path; in practice the runtime is
exclusively dry today; the moist path exists for explicit experiments and future
diagnostic comparisons. **Mixing a moist binary with a dry runtime
contract is a load-time error**, not a silent corruption.

## Basis types

```
AbstractMassBasis  (Basis.jl)
├── DryBasis()    — air_mass is dry-air mass; tracers are dry VMR
└── MoistBasis()  — air_mass is total (moist) air mass; tracers are moist VMR
```

Query the basis at runtime with `mass_basis(state)`. The default
constructor `CellState(air_mass, ...)` produces a `DryBasis` state.

If you find yourself reaching for the moist basis, double-check by
looking at the binary header — under the dry-basis contract, the
preprocessor ships `mass_basis = :dry` and the runtime relies on it.

## Accessor API

All access goes through a small set of helpers in `src/State/Tracers.jl`.
**Tests should observe state through these helpers, not through input
arrays cached before construction** (per the project's testing rules).

### Reading a tracer

```julia
co2_storage = get_tracer(state, :CO2)     # non-allocating χ × air-mass view
ch4_storage = state.tracers.CH4            # property access; same view

co2_vmr  = mixing_ratio(state, :CO2)      # storage / carrier-air mass — dry VMR
```

`get_tracer` returns a view into the last dimension of `tracers_raw`
— **model storage**, not VMR or physical species mass. On `CellState` this is a 3-D / 2-D view of
the right shape; on `CubedSphereState` it returns one slice per panel.
For dry VMR use `mixing_ratio(state, name)` (or compute the ratio
yourself if you need a backend-specialized variant).

### Iterating

```julia
for (name, rm) in eachtracer(state)
    @info name vmr_range=extrema(rm ./ state.air_mass) storage_total=sum(rm)
end
```

### Counts and indices

```julia
ntracers(state)                 # how many tracers
tracer_index(state, :CO2)       # 1-based index, or nothing
```

### Writing

```julia
set_uniform_mixing_ratio!(state, :CO2, 4.0e-4)   # sets mass = χ × air_mass
```

For more involved initialization (loading from a NetCDF file, regridding
from a different mesh), the runtime's `InitialConditionIO.jl` is the
canonical entry point; user-facing IC kinds are described in the
[Run with real meteorology](@ref) and [Quickstart](@ref) pages.

## GPU residency

When `[architecture] use_gpu = true`, the runtime constructs
`state.air_mass` as a `CuArray` (or the equivalent for the chosen
backend) and asserts residency before stepping. The check fails loudly
if the dispatch chain accidentally fell back to CPU storage:

```text
[ Info: [gpu verified] backend=cuda backing=CuArray device=NVIDIA L40S
```

If you need to round-trip state to and from the host (e.g. for a
diagnostic), use `Array(state.air_mass)` — but **don't** do it inside
the simulation loop; that defeats the GPU.

## Time-varying meteorological fields

For physics blocks that need a time-varying input (e.g. Kz for
diffusion, surface fields for convection), the runtime exposes
`AbstractTimeVaryingField` subtypes. They share a small interface:
`field_value(f, idx)` (kernel-safe) and `update_field!(f, t)`
(host-side cache refresh, called once per met window). Concrete
types currently in the tree:

| Type | Use |
|---|---|
| `ConstantField{FT, N}` | Scalar broadcast to a fixed value. |
| `ProfileKzField{FT, V, N}` | Fixed vertical profile, uniform horizontally. |
| `PreComputedKzField{FT, N, A}` | Wrap a precomputed spatial Kz field. |
| `DerivedKzField{FT, SF, DELP, A, P}` | Beljaars-Viterbo Kz derived from meteorological fields. |
| `StepwiseField{FT, N, A, B, W}` | Piecewise-constant in time (read from binary). |
| `WindowPBLKzField` | Cubed-sphere local Kz from `pblh/ustar/pbl_hflux/t2m`. |
| `LocalHoltslagBovilleKzField` | GEOS/VDIFF local Kz from surface and column fields. |
| `PrecomputedCSDkgField` | Exact interface exchange read from a `:dkg` payload. |

The runtime recipe exposes `none`, `constant`, and three cubed-sphere modes:
`tm5_beljaars_viterbo_local_kz`, `geoschem_holtslag_boville_vdiff`, and
`tm5_dkg`. Each mode validates its required version-4 payload sections before
constructing the field. Other field types remain useful for direct Julia API
experiments.

## What's next

- [Operators](@ref Operator-concepts) — Advection, Convection, Diffusion, Sources behind
  abstract types with `No<Operator>` defaults.
- [Binary format](@ref) — the on-disk layout the runtime consumes.
