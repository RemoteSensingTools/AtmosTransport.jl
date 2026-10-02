# Chemistry

Tracer source/sink operators applied after transport.

This folder owns the chemistry operator hierarchy, the current decay
operators, and the `chemistry_block!` composition step that
`TransportModel.step!` runs after the transport block.

## Entry Points

- Type hierarchy and concrete operators:
  [`Chemistry.jl`](Chemistry.jl)
  defines `AbstractChemistryOperator`, `NoChemistry`,
  `ExponentialDecay`, and `CompositeChemistry`
- Kernel implementation:
  [`chemistry_kernels.jl`](chemistry_kernels.jl)
- Step-level block composition:
  [`chemistry_block.jl`](chemistry_block.jl)
  provides `chemistry_block!`
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
- The optional `AtmosChemistry.jl` extension couples stiff, batched gas
  chemistry to dry-mass transport states on CPU and CUDA. It maps only the
  active mechanism species, excludes cubed-sphere halos, and keeps chemistry
  state and scratch backend-resident.
- See [`../TOPOLOGY_SUPPORT.md`](../TOPOLOGY_SUPPORT.md) for the
  canonical operator × topology matrix

## AtmosChemistry Preview

`AtmosChemistryOperator` is loaded when `AtmosChemistry.jl` is available. Its
concrete model type determines the mechanism subset, floating-point type,
solver, reduction policy, and CPU or GPU architecture. The transport adapter
does not own chemistry science or grid topology.

Transport tracers use dry mass while chemistry integrates molecules per cubic
centimeter. Fused KernelAbstractions kernels convert one bounded cell tile in
and out, using supplied pressure or air number density and species molar
masses. A portable two-stage reduction checks dry-air mass and number density
before any state conversion. Failed cells are not stored back. `MoistBasis` is
rejected until water-vapor mass is available for an unambiguous conversion.

Forcing has two typed cadence policies:

- `ConstantChemistryForcing(forcing)` captures forcing at workspace creation.
- `CallUpdatedChemistryForcing(initial, update)` evaluates `update` once per
  chemistry call and refreshes preallocated forcing buffers in place.

The updated value must preserve field names, precision, and scalar/spatial
layout. Spatial cubed-sphere fields may be packed interior arrays or full
halo-padded panel arrays; full fields are projected into the existing interior
buffer. A call-updated GPU workspace owns one bounded host staging vector so
ordinary host meteorology can refresh packed cell or halo fields without
becoming a device-kernel argument. Backend-resident updates bypass that stage.
Constant forcing workspaces do not allocate it.

Workspace forcing is an owned packed snapshot. Updating it never mutates the
provider's `initial` forcing, even when that input was already resident on the
model backend. The adapter validates the complete forcing window once and then
uses AtmosChemistry's typed `PrevalidatedForcingCells` batch policy for each
tile, avoiding a repeated device-to-host cell-map transfer. GPU tiles also use
`BackendDiagnosticTransfer`: status and counters are scattered into one
backend-resident full-domain table, then synchronized and copied once at the
transport API boundary. The memory plan reports both this table and its host
diagnostic target.

Initial forcing is selected, packed, and, for cubed-sphere fields, projected
directly into its final owned buffers. Host, resident, and mixed-residency
sources are handled field by field, so construction does not create a second
device forcing snapshot. The public plan describes the same canonical fields
that allocation owns, including source aliases, dropped extra fields, and
halo-to-interior shape changes. Its device construction peak therefore equals
steady planned residency. Host construction staging is reported separately:
the plan conservatively sums temporary host allocations across fields and
panels because Julia does not guarantee reclamation between constructor calls.
`construction_peak_host_bytes` combines that transient total with persistent
host diagnostics and provider staging.

The concentration tile is fixed-capacity, but each `ChemistryBatch` carries
the logical number of active rows. A short terminal tile launches and solves
only those rows; unused capacity is neither filled with repeated cells nor
integrated. This keeps repeated calls allocation-free while avoiding thousands
of redundant solves for awkward domain sizes.

Active chemistry species are mapped to transport columns by name, once at
workspace construction, so the packed tracer order may differ from mechanism
order and the transport state may contain many additional tracers. Only mapped
columns are loaded and stored. For `SelectedSpecies(...;
closure=:fixed_background)`, omitted chemical species remain explicit forcing:
a same-named transported tracer is not implicitly sampled as background
chemistry. This keeps forcing cadence and scientific ownership unambiguous.

Call `chemistry_workspace_plan(operator, state, grid)` before allocation to
inspect the chemistry and adapter byte budgets. After allocation,
`chemistry_workspace_storage_bytes(workspace)` reports the storage actually
owned. The default transport concentration tile is at most 4096 cells and can
be reduced further by `WorkspacePolicy` memory limits.

CUDA is the current GPU integration gate. The kernels use
KernelAbstractions and avoid CUDA-specific code in the coupling extension, but
Metal remains a portability target rather than a validated chemistry backend.
Run the dedicated gate with:

```bash
julia --project=test/atmoschemistry test/atmoschemistry/runtests.jl
```

The reproducible adapter benchmark is:

```bash
julia --project=test/atmoschemistry \
  scripts/benchmarks/bench_atmoschemistry_adapter.jl 65536 float64
```

The record separates device-state staging, workspace construction, the first
chemistry result, cold reset-to-reset calls, and warm evolving trajectories. It
also records the GPU UUID and planned host forcing-stage bytes; constant forcing
should report no host forcing stage.

On an isolated development L40S, the schema-3 matched-state record measures
5.751 ms for a warm 65,536-cell adapter trajectory versus 4.897 ms for the
prevalidated raw core, a 1.174x adapter ratio. Cold reset-to-reset calls take
13.852 ms and 12.933 ms, respectively, a 1.071x ratio. A seven-row warm logical
tail takes 0.252 ms versus 0.376 ms when padded to 4,096 rows, a 1.49x speedup.
A fresh process takes 0.166 s to stage state, 11.847 s to construct the first
workspace, and 6.532 s for the first coupling call. Those startup timings
include order-dependent Julia and GPU compilation and are not steady execution
costs. See
[`atmoschemistry_adapter_l40s_gpu1_matched_warm_cold_65536_20260828.toml`](../../../scripts/benchmarks/results/atmoschemistry_adapter_l40s_gpu1_matched_warm_cold_65536_20260828.toml).
These figures characterize adapter and controller overhead only; they are not
full-chemistry performance claims.

## File Map

- [`Chemistry.jl`](Chemistry.jl) — submodule assembly, operator types,
  state-level `apply!`, and composition rules
- [`chemistry_kernels.jl`](chemistry_kernels.jl) — fused multi-tracer
  decay kernel
- [`chemistry_block.jl`](chemistry_block.jl) — post-transport chemistry
  block called from the model step

## Common Tasks

- Adding a new chemistry operator:
  define the type and `apply!` method in [`Chemistry.jl`](Chemistry.jl),
  then decide whether it belongs inside `CompositeChemistry`
- Changing AtmosChemistry coupling:
  keep mechanism and solver details in `AtmosChemistry.jl`; this package owns
  basis conversion, topology indexing, forcing cadence, and workspace lifetime
- Adding a time-varying scalar rate:
  implement the field in `State/Fields`, then plug it into
  `ExponentialDecay`
- Debugging tracer selection:
  inspect `tracer_index` resolution in [`Chemistry.jl`](Chemistry.jl)
  before changing kernel code
- Tracing runtime behavior:
  start at [`../../Models/TransportModel.jl`](../../Models/TransportModel.jl)
  and then follow `chemistry_block!`

## Cross-Dependencies

- [`../../State/`](../../State/) provides tracer storage and the
  time-varying field interface
- [`../../MetDrivers/`](../../MetDrivers/) provides `current_time` for
  time-varying rate fields
- [`../../Models/TransportModel.jl`](../../Models/TransportModel.jl)
  executes `chemistry_block!` after transport
- [`../../../docs/20_RUNTIME_FLOW.md`](../../../docs/20_RUNTIME_FLOW.md)
  walks through the runtime block order

## Related Docs And Tests

- Topology coverage:
  [`../TOPOLOGY_SUPPORT.md`](../TOPOLOGY_SUPPORT.md)
- Tests:
  - [`../../../test/test_chemistry.jl`](../../../test/test_chemistry.jl)
  - [`../../../test/test_current_time.jl`](../../../test/test_current_time.jl)
  - [`../../../test/core/test_atmoschemistry_extension.jl`](../../../test/core/test_atmoschemistry_extension.jl)
  - [`../../../test/diagnostic/test_atmoschemistry_gpu_extension.jl`](../../../test/diagnostic/test_atmoschemistry_gpu_extension.jl)
