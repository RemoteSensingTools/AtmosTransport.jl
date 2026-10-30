# TM5

TM5 column construction, factorization, tracer solves, level merging and
collaborative GPU kernels.

## Depends on

Architectures, Grids, State, MetDrivers and parent Convection storage/interfaces.

## Used by

The Convection facade, Models, StepPlans, Adjoints and focused convection tests.
CMFMC also uses the TM5 matrix path; CUDA specializes the depth gate.

## Shared helpers owned here

Public operator and workspace names are re-exported through Convection.
The following numerical and validation helpers are shared with the reverse
dispatch or the sibling family; the forward implementation owns their arithmetic.

# shared: _TM5_COLLAB_MAX_DEPTH_FLOAT64_CUDA, _ensure_tm5_scratch!, _tm5_collab_max_depth, _tm5_column_sum, _tm5_cs_panel_column_adjoint_kernel!, _tm5_restore_column_mass!, _tm5_template

## Forward<->adjoint mirrors

`adjoint_kernels.jl` transposes the forward column in `tm5_column_solve.jl`; pivots are reversed in the transpose.
Collaborative wrappers keep their local-memory types (FT, Int32), topology
decoding and barrier phases explicit. Cloud scans and plume arithmetic use
shared inline bodies with the original topology readers. The legacy and
collaborative builds stay separate: signed-zero assembly, mass reads and
zero-pivot handling differ.

## File map

- `TM5.jl`: module imports and exports.
- `workspace.jl`: TM5Workspace, tiled scratch and Adapt.
- `tm5_column_solve.jl`: matrix build, LU and per-tracer solves.
- `tm5_kernels.jl`: LL/RG/CS kernel wrappers and shared barrier-free bodies.
- `TM5Convection.jl`: operator, topology dispatch and level merging.
- `adjoint_kernels.jl`: transpose solves and CS adjoint wrapper.
