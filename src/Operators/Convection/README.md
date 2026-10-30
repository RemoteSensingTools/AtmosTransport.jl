# Convection

Convective transport, reusable storage, and checkpoint state. The facade keeps
`CMFMCConvection`, `CMFMCMatrixConvection`, `TM5Convection`, their workspaces,
cloud-base types, cache invalidation, and `apply_convection!` available at
`AtmosTransport.Operators.Convection` and through the existing package exports.
The concrete types are defined in the `CMFMC` and `TM5` child modules.

## Depends on

Architectures, Grids, State, MetDrivers (`ConvectionForcing`), and the parent
Operators interfaces (`apply!`, `AbstractConvection`, checkpoint state).
TM5 is independent of CMFMC. CMFMC's matrix form uses TM5's column solver.
Neither family depends on Models, Tape, StepPlans, or Adjoints.

## Used by

Models and StepPlans apply the operators; Adjoints dispatches their numerical
transposes. The CUDA extension adds the Float64 collaborative depth limit.

## Shared helpers owned here

`convection_workspace_for(op, state, grid)` owns runtime workspace construction.
The storage-only helpers below are shared by both child modules.

# shared: _cmfmc_metric_buffer, _empty_ledger_carry, _fitted_ledger_carry

## Forward<->adjoint mirrors

Each family's `adjoint_kernels.jl` sits beside its forward implementation.
`../../Adjoints/ConvectionAdjoint.jl` retains supported-variant checks, forcing
validation, and reverse dispatch using the recorded CMFMC substep count.
Checkpoint snapshot methods live here and extend Operators' shared interface.

## File map

- `Convection.jl`: facade, child-module assembly and exports.
- `operators.jl`: `NoConvection` and its identity `apply!` methods.
- `workspace_shared.jl`: topology-aware storage allocation helpers.
- `workspace_factory.jl`: runtime workspace factories and cell metrics.
- `workspace_state.jl`: convection checkpoint snapshot, restore and reset.
- [CMFMC/](CMFMC/README.md): direct CMFMC and CMFMC matrix convection, adjoint.
- [TM5/](TM5/README.md): TM5 column solve, topology kernels and transpose.

## Numerical contracts

Forcing is window-constant and refreshed by Models. Kernels preserve their
floating-point arithmetic, summation order, and precision. Column ledgers keep
the unrepresentable residual in each workspace between calls; adjoint replay
restores that persistent state at checkpoints.

Direct LL/RG CMFMC kernels remain separate from CS: they lack CS's clamp and
archived cloud-base handling. RG CMFMC without DTRAIN and the TM5 fallback's
dropped `lmax_conv` / `n_merge` remain deferred bugs. Unsupported collaborative
backends keep the legacy solve; scientific-option behavior is unchanged.
