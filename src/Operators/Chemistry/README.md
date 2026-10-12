# Chemistry

Tracer source/sink operators composed with transport.

This folder owns the chemistry operator hierarchy, the current decay
operators, and sequential `CompositeChemistry` composition that
`TransportModel.step!` runs in the configured composition.

## Entry Points

- Type hierarchy and concrete operators:
  [`Chemistry.jl`](Chemistry.jl)
  defines `AbstractChemistryOperator`, `NoChemistry`,
  `ExponentialDecay`, and `CompositeChemistry`
- Kernel implementation:
  [`chemistry_kernels.jl`](chemistry_kernels.jl)
- Model-facing runtime entrypoint:
  [`Chemistry.jl`](Chemistry.jl)
  provides `apply!(state, meteo, grid, op, dt; workspace=nothing)`

## Current Scope

- `CellState` chemistry is live (LatLon and RG)
- `CubedSphereState` chemistry is live — per-panel apply over the
  `NTuple{6, Array{FT, 4}}` storage, using the same rank-agnostic
  decay kernel as the `CellState` path
- Time-varying scalar rate fields are supported through the
  `AbstractTimeVaryingField{FT, 0}` contract
- `ExponentialDecay` is applied as exact half-steps before and after each
  emissions window. If `Q = E T`, `a = exp(-λT)`, and `B_n` is the burden at
  a window boundary, the update and its fixed point are

  ```math
  B_{n+1} = a B_n + Q\sqrt{a}, \qquad
  B_* = \frac{Q\sqrt{a}}{1-a}
      = \frac{E T}{2\sinh(\lambda T/2)}.
  ```

  Relative to the continuous steady burden `E/λ`, its bias is
  `λT/[2sinh(λT/2)] - 1 = -(λT)²/24 + O((λT)⁴)`. For Rn-222 and
  one-hour windows this is `-0.000237736%`. The former emit-then-decay update
  had fixed point `E T/(exp(λT)-1)` and first-order bias `-λT/2`, or
  `-0.377204%` for the same window.
- See [`../TOPOLOGY_SUPPORT.md`](../TOPOLOGY_SUPPORT.md) for the
  canonical operator × topology matrix

## File Map

- [`Chemistry.jl`](Chemistry.jl) — submodule assembly, operator types,
  state-level `apply!`, and composition rules
- [`chemistry_kernels.jl`](chemistry_kernels.jl) — fused multi-tracer
  decay kernel

## Common Tasks

- Adding a new chemistry operator:
  define the type and `apply!` method in [`Chemistry.jl`](Chemistry.jl),
  then decide whether it belongs inside `CompositeChemistry`
- Adding a time-varying scalar rate:
  implement the field in `State/Fields`, then plug it into
  `ExponentialDecay`
- Debugging tracer selection:
  inspect `tracer_index` resolution in [`Chemistry.jl`](Chemistry.jl)
  before changing kernel code
- Tracing runtime behavior:
  start at [`../../Models/TransportModel.jl`](../../Models/TransportModel.jl)
  and then follow `apply!`

## Cross-Dependencies

- [`../../State/`](../../State/) provides tracer storage and the
  time-varying field interface
- [`../../MetDrivers/`](../../MetDrivers/) provides `current_time` for
  time-varying rate fields
- [`../../Models/TransportModel.jl`](../../Models/TransportModel.jl)
  composes the configured chemistry operator with transport, including the
  symmetric exponential-decay split
- [`../../../docs/20_RUNTIME_FLOW.md`](../../../docs/20_RUNTIME_FLOW.md)
  walks through the runtime block order

## Related Docs And Tests

- Topology coverage:
  [`../TOPOLOGY_SUPPORT.md`](../TOPOLOGY_SUPPORT.md)
- Tests:
  - [`../../../test/orphan/test_chemistry.jl`](../../../test/orphan/test_chemistry.jl)
  - [`../../../test/core/test_cs_chemistry.jl`](../../../test/core/test_cs_chemistry.jl)
  - [`../../../test/orphan/test_current_time.jl`](../../../test/orphan/test_current_time.jl)
