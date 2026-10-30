# Reconstruction

Scheme and limiter definitions, face reconstruction, and PPM sub-grid distributions.

This is a plain sub-folder of the Advection module; see the
[parent contract](../README.md) for the fenced-root dependency decision.

## File map

- `schemes.jl`: scheme, limiter and vertical-reconstruction types and traits.
- `limiters.jl`: slope and moment limiters.
- `reconstruction.jl`: face fluxes and the shared fourth-order PPM edge expression.
- `ppm_subgrid_distributions.jl`: PPM edge distributions used by Lin–Rood.

## Depends on

Parent operator types and Parameters; no topology driver dependency.

## Used by

Structured and CubedSphere sweeps, fenced Lin–Rood forward/adjoint code.

## Shared helpers owned here

Scheme/limiter types and traits; `_ppm_edge_value` and `_ppm_edge_weights` have one home here.

## Forward<->adjoint mirrors

`reconstruction.jl` ↔ `../CubedSphere/AdvectionAdjoint.jl`. The fenced D6 code still mirrors ORD5 edges.
