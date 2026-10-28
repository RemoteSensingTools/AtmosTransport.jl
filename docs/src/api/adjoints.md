# Adjoints and checkpointing API

The adjoint surface covers cubed-sphere objectives, observations, footprints,
preconditioning, and 4D-Var solvers. `Tape` owns checkpoint schedules and
device, pinned-host, and memory-mapped tape storage.

## Adjoints

Configure optimizer settings and logging on `CSGradientDescent` or `CSLBFGS`
and pass it as `optimizer` to `cs_surface_flux_4dvar_optimize`. Footprint
functions compute analytic adjoints; finite-difference perturbations belong
to the caller's validation code. `CSFootprintResult` contains `objective`,
`footprints`, `lag_steps`, `dt`, and `initial_adjoint`.

```@autodocs
Modules = [AtmosTransport.Adjoints]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```

## Tape and checkpointing

```@autodocs
Modules = [AtmosTransport.Tape]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```
