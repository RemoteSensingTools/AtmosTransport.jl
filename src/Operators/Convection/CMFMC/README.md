# CMFMC

GEOS CMFMC and DTRAIN convection: direct interface transport and the
conservative matrix form that derives rates for the TM5 solve.

## Depends on

Architectures, Grids, State, MetDrivers and parent Convection storage/interfaces. TM5 supplies its matrix solve and column ledger.

## Used by

The Convection facade, Models, StepPlans, Adjoints and focused convection tests.

## Shared helpers owned here

Public operator and workspace names are re-exported through Convection.
The following numerical and validation helpers are shared with the reverse
dispatch or the sibling family; the forward implementation owns their arithmetic.

# shared: _cmfmc_cloud_base, _cmfmc_cs_panel_column_single_adjoint_kernel!, _cmfmc_dtrain_array, _cmfmc_matrix_synthetic_forcing, _cmfmc_tiny, _get_or_compute_n_sub!, _refresh_cmfmc_matrix_rates!, _require_edge_cloud_base

## Forward<->adjoint mirrors

`adjoint_kernels.jl` transposes the forward column in `cmfmc_kernels.jl`. The reverse
keeps the forward tiny-mass guards, cloud-base scan and Kahan sums. LL/RG
columns stay separate from CS because clamp and archived cloud-base support
differ. Those differences require a stated-delta follow-up.

## File map

- `CMFMC.jl`: module imports and exports.
- `workspace.jl`: CMFMCWorkspace, CFL cache and Adapt.
- `CMFMCConvection.jl`: direct operator, validation and topology dispatch.
- `cmfmc_kernels.jl`: CFL scans and direct LL/RG/CS transport.
- `CMFMCMatrixConvection.jl`: rate cache and delegation to TM5.
- `cmfmc_matrix_kernels.jl`: shared rate-derivation body and topology wrappers.
- `adjoint_kernels.jl`: transpose of the direct CS column.
