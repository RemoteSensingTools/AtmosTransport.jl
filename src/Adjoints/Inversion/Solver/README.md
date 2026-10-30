# Inversion solver

`AtmosTransport.Adjoints.Inversion.Solver` implements surface-flux 4D-Var.

## File Map

- `Solver.jl`: module imports, includes and exports.
- `Types.jl`: control windows, controls, Jacobian/4D-Var results and iteration logs.
- `Covariance.jl`: diagonal and panel-local Gaussian square roots and adjoints.
- `Preconditioning.jl`: Linear/LogNormal forward transforms and adjoint chain rule.
- `Jacobian.jl`: batched objective footprints and weighted window aggregation.
- `CostGradient.jl`: control-to-source rates, weighted departures, one forced
  reverse pass and background cost/gradient.
- `Optimizer.jl`: gradient descent, L-BFGS, cached evaluations and dispatch.
- `Departures.jl`: solver-result adapter to observation-owned departure IO.

## Depends on

Inversion.Observations, Footprint, Adjoints core, Tape, Grids, Parameters and
Operators (no-op defaults). External: KernelAbstractions, FFTW, Optim.

## Used by

Inversion facade, inversion scripts and numerical tests.

## Shared helpers owned here

Controls/results, covariance square roots/adjoints, preconditioner forward and
adjoint maps, `cs_surface_flux_jacobian`, `cs_surface_flux_4dvar`,
`cs_surface_flux_4dvar_optimize`, `cs_surface_flux_4dvar_solve`,
`CSGradientDescent`, `CSLBFGS`. Configure descent/logging through `optimizer`.
The inverse covariance/preconditioner and test-only tangent APIs were removed.

## Forward<->adjoint mirrors

Source-rate assembly and window-gradient accumulation keep their order.
Only active functionals enter forcing assembly; adding co-samples leaves the
cost and gradient bitwise unchanged. The objective is departure-squared/2
plus the background cost; preconditioned background cost is `0.5‖χ‖²`.
The Gaussian spectral floor and renormalization are unchanged, and there is
no correlation across panels. FFTW uses CPU scratch. LogNormal transforms
require positive backgrounds. Scratch is reused and is not thread-safe.
L-BFGS shares one cost/gradient evaluation per visited point, including its
initial point and returned result. GD uses L2 gradient tolerance, L-BFGS L∞.
