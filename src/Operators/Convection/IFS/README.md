# IFS

Tracer transport of the ECMWF IFS convection scheme (OpenIFS CY48R1.1
`cuctracer.F90` with the implicit solver `cubidiag.F90`) driven by the four
TM5 convection fields `entu`, `detu`, `entd`, `detd`. Selected with
`[convection] kind = "ifs"`; forward model only.

## Depends on

Architectures (launch, ledger carry), Grids, State, MetDrivers
(`ConvectionForcing`), the parent Convection storage helpers and TM5's
compensated column sum (`_tm5_column_sum`).

## Used by

The Convection facade, Models (runtime spec, capability and window checks),
Adjoints (which refuse the operator) and the focused convection tests.

## Shared helpers owned here

None. Public names (`IFSConvection`, `IFSWorkspace`, the variants
`IFSCuctracer` and `IFSMonotoneUpstream` and their supertype) are re-exported
through Convection; the column helpers stay private to this module.

## Forward<->adjoint mirrors

None: the operator has no adjoint. `adjoint_support(::IFSConvection)` and the
cubed-sphere footprint guards reject it before anything is recorded.

## File map

- `IFS.jl`: module imports and exports.
- `workspace.jl`: `IFSWorkspace` (interface fluxes, plume scratch, column
  Courant numbers, ledger carry) and Adapt.
- `IFSConvection.jl`: operator and variant types, forcing checks, the
  Courant-number check and topology dispatch of `apply_convection!` / `apply!`.
- `ifs_column.jl`: column arithmetic shared by all topologies: interface
  fluxes, plume values, lowest-layer correction, the two compensating-flow
  solves and the sub-step loop with the column ledger.
- `ifs_kernels.jl`: lat-lon, reduced-Gaussian and cubed-sphere kernel wrappers.

## Common tasks

- Compare with TM5 on the same columns: run the same binary with
  `kind = "tm5"` and `kind = "ifs"`; both read the same sections and pass the
  same capability, forcing and mass-basis checks. `variant = "cuctracer"`
  gives the arithmetic of the IFS routine; the default `"monotone"` creates no
  new extrema.
- Change the plume or compensating-flow arithmetic: add or edit a method on
  the variant type in `ifs_column.jl`; `test/core/test_ifs_convection.jl`
  compares both variants with `test/helpers/ifs_convection_reference.jl` and
  the compiled OpenIFS results in `test/fixtures/ifs_cuctracer_columns.txt`
  (rebuilt by `test/fixtures/ifs_cuctracer_generator/`).

## Numerical contracts

- Layer `k = 1` is the model top; interface `k` is the top of layer `k`.
  `mu[k]`, `md[k]` hold the interface fluxes; the ground interface carries no
  flux and is not stored. The updraft is closed at the top of the highest
  layer with `detu > 0` and the downdraft at the bottom of the lowest layer
  with `detd > 0`, so a remainder of the cumulative sums of stored fields does
  not reach the model top or the ground.
- State stores tracer mass and air mass per cell; the mixing ratio is their
  ratio, and fluxes (kg m⁻² s⁻¹) meet layer mass through `m / area`.
- Both solves compute the increment `δ` of the mixing ratio and add `m δ` to
  the stored tracer mass, so an increment below half an ulp leaves a cell
  bit-identical.
- One work item per column; fluxes and the Courant number once per column and
  call, then per tracer `n` sub-steps (`n = ceil(courant / cfl_limit)`), each
  O(N_z) and followed by the column mass ledger over the whole column, in
  every precision. The transport changes layers from the one above the first
  interface with flux down to the ground; the ledger may add its residual to
  the column's largest cell wherever it is. Columns without flux are untouched
  bit for bit.
- The update is in flux form and conserves column tracer mass to rounding in
  any precision; the ledger only returns the rounding residual. It sums and
  deposits over the whole column but drops residuals above `16 n` ulps of the
  largest of the `n` cells the sub-step rewrote (`_ifs_restore_column_mass!`),
  so a large cell above the cloud cannot hide a loss inside it.
- The flux kernel returns status codes instead of a Courant number for a
  negative field value or a layer without air among the rewritten layers (the
  highest layer a flux touches and every layer below it); non-finite values
  give `Inf`. The host rejects them before any tracer moves. Layers without
  air above the rewritten ones are allowed.
- Scratch: four arrays the size of the air mass (`mu`, `md`, `cu`, `cd`), one
  Courant number per column and the ledger carry per column and tracer. No
  allocation in kernels.
