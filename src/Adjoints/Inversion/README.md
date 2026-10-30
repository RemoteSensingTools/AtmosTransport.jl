# Inversion

`AtmosTransport.Adjoints.Inversion` is the public facade for two modules:
`Observations` (records, binding, column sampling and IO) and `Solver`
(controls, covariance, preconditioning, Jacobian, cost/gradient and optimizers).
Its exported names remain available through Adjoints.

## File Map

- `Inversion.jl`: child includes and re-exports.
- `Observations/`: observation data flow through binding, replay sampling and departures.
- `Solver/`: the control-to-emissions map and its adjoint, optimization and results.

## Depends on

Adjoints core → Footprint → Inversion. Within Inversion, Solver depends on
Observations; Observations never depends on Solver. Child includes assemble
this facade and add no computational back edge.

## Used by

`scripts/inversions/cs_4dvar.jl`, synthetic experiments, the OCO-2 converter,
and inversion/identity/finite-difference tests.

## Shared helpers owned here

The facade re-exports the explicit observation and solver interfaces. Internal
helpers belong to their child module; consumers import from that owner.

## Forward<->adjoint mirrors

Observation functionals supply the forcing of one reverse pass.
Control-window rate assembly mirrors gradient accumulation.
`apply_B_half!` pairs with `apply_B_half_adjoint!`;
`apply_preconditioner!` is checked against its adjoint using finite differences.
