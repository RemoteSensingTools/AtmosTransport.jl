# Fields

Generic time-varying field containers consumed by runtime operators.
Boundary-layer calculations live in [Diffusion](../../Operators/Diffusion/README.md).

## Interface Contract

`field_value(f, idx)` is a pure, allocation-free, kernel-safe read.
`update_field!(f, t)` refreshes a field's time selection on the host.
`panel_field(f, p)` selects one cubed-sphere panel.

## File Map

- [`Fields.jl`](Fields.jl) — abstract contracts and `ConstantField`
- [`PreComputedKzField.jl`](PreComputedKzField.jl) — caller-owned rank-2/3 arrays
- [`CubedSphereField.jl`](CubedSphereField.jl) — six-panel field wrapper
- [`PrecomputedCSDkgField.jl`](PrecomputedCSDkgField.jl) — precomputed interface
  exchange storage and per-window payload copy
- [`GCHPNonlocalPBLField.jl`](GCHPNonlocalPBLField.jl) — non-local PBL storage,
  parameter container, constructor and Adapt. Kept here for the fenced
  SurfaceFlux dependency; all physics and refresh live in Diffusion.
- [`StepwiseField.jl`](StepwiseField.jl) — half-open time-window selection

`ProfileKzField`, `DerivedKzField` and `integral_between` had no production
callers and were removed. Spatial profiles use `PreComputedKzField`;
production PBL tests exercise `WindowPBLKzField` in Diffusion.

## Common Tasks

Add generic field storage here and implement the interface above. Add physics
closures next to their consuming operator. Stepwise fields reject times
outside the covered half-open interval; surface-flux temporal policies have
separate contracts.

## Depends on

Architectures (host staging), Adapt and Parameters (GEOS-Chem parameter container defaults). No operator
dependency.

## Used by

State re-exports the containers; Diffusion, Chemistry, SurfaceFlux, StepPlan,
Models and Adjoints consume them.

## Shared helpers owned here

`AbstractTimeVaryingField`, `AbstractCubedSphereField`, `AbstractCSDkgField`,
`field_value`, `update_field!`, `panel_field` and the containers in the file map.
`refresh_precomputed_cs_dkg_cache!` copies window forcing into storage.

## Forward<->adjoint mirrors

Storage and indexing are shared by forward and adjoint consumers. Numerical
PBL and diffusion transpose kernels live in Operators/Diffusion.

## Related Tests

`test/core/test_fields.jl`, `test/core/test_diffusion_flux_form.jl`,
`test/core/test_conservative_dkg.jl` and `test/core/test_gchp_nonlocal_pbl.jl`.

Architectures.host_array stages StepwiseField boundaries on the host; Array
inputs alias and views are copied, preserving the boundary-search policy.
