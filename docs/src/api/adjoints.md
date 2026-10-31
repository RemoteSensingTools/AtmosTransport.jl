# Adjoints and checkpointing API

The adjoint surface covers cubed-sphere objectives, observations, footprints,
preconditioning, and 4D-Var solvers. `Tape` owns checkpoint schedules and
device, pinned-host, and memory-mapped tape storage.
`Adjoints.Inversion.ObservationBinding` owns inversion records, column binding,
observation/departure IO and OCO-2 MIP conversion. `Observations` owns forward
placement, retrievals and sampling; inversion exports remain available through
the `Inversion` and `Adjoints` facades.

## Adjoints

Configure optimizer settings and logging on `CSGradientDescent` or `CSLBFGS`
and pass it as `optimizer` to `cs_surface_flux_4dvar_optimize`. One
`cs_surface_flux_4dvar` evaluation runs the forward model once: the run that
samples the observations records the tape of the reverse pass;
`evaluation = CSCostOnly()` runs it without a tape for the cost and simulated
values alone. Footprint
functions compute analytic adjoints; finite-difference perturbations belong
to the caller's validation code. `CSFootprintResult` contains `objective`,
`footprints`, `lag_steps`, `dt`, and `initial_adjoint`.

```@autodocs
Modules = [AtmosTransport.Adjoints, AtmosTransport.Adjoints.Footprint,
           AtmosTransport.Adjoints.Inversion.ObservationBinding, AtmosTransport.Adjoints.Inversion.Solver]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```

The solver exposes the forward preconditioner and its adjoint. The unused
inverse and tangent interfaces have been removed. The on-disk tape storage
`MmapCSTapeStorage(dir)` takes a required scratch directory; each call
reserves its tape there before recording and releases it when it returns.

## Tape and checkpointing

```@autodocs
Modules = [AtmosTransport.Tape]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```
