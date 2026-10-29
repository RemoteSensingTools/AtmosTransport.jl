# src

Primary source tree for AtmosTransport.jl.

This is the runtime code, not just utilities. If you are trying to
understand how a transport step is assembled end to end, start here.

## Main Runtime Flow

At a high level, the live path is:

1. [`MetDrivers/README.md`](MetDrivers/README.md) loads or prepares a
   transport window and its grid
2. [`State/README.md`](State/README.md) holds air mass, conservative tracer storage, and
   face-flux storage in topology-appropriate containers
3. [`Models/README.md`](Models/README.md) composes state, grid, drivers,
   and operators into a runnable model/simulation
4. [`Operators/README.md`](Operators/README.md) applies transport and
   post-transport physics
5. [`Output/`](Output/) captures topology-native snapshots and writes
   diagnostic NetCDF products

Geometry and topology decisions come from
[`Grids/README.md`](Grids/README.md).

## Folder Map

- [`Grids/`](Grids/README.md) — mesh types, topology tags, vertical
  coordinates, geometry accessors
- [`State/`](State/README.md) — prognostic state, flux state, tracer
  accessors, time-varying operator fields
- [`MetDrivers/`](MetDrivers/README.md) — transport-window readers,
  runtime drivers, mass closure, convection forcing, ERA5-specific paths
- [`Operators/`](Operators/README.md) — advection, diffusion, surface
  flux, convection, chemistry
- [`Models/`](Models/README.md) — `TransportModel`, fixed-step runtime,
  window-driven runtime
- `Output/` — topology-dispatched snapshot capture, NetCDF schema, and
  derived diagnostics for LL, RG, and CS runs
- `Parameters/` — physical constants and planetary parameters
- `Architectures.jl` — unified CPU/vendor-specific GPU execution policies,
  kernel launching (`launch!`) and point operations (`AbstractPointOp`,
  `Fused`, `Sequence`: a kernel body written once, fused by dispatch;
  `copy_arrays!`: a set of device copies as one queued launch)
- `ConfigChecks.jl` — strict TOML Booleans, integers (`config_int`), dates
  (`config_date`, accepting TOML dates or strings), the `float_type` parser, and
  known-key checks with "did you mean" suggestions, shared by `Output`, `Preprocessing` and `Models`
- `AtomicFiles.jl` — publish a file only when it is complete: a sibling
  staging file created with `O_EXCL` and mode `0o666 & ~umask`, moved onto
  the target by one replacing rename on success, removed on error or when the
  rename fails (Base only; used by the transport-binary writers, the spectral
  day cache and the regridder cache)
- `Quantities/` — intensive/extensive/vector/flux traits used by regridding
- `Diagnostics/` — lightweight runtime instrumentation
- `Regridding/` — conservative regridding, weight application by quantity kind, nested cubed-sphere remapping and the CS block kernels
- [`GriddedIO/`](GriddedIO/README.md) — conventions of native gridded input
  files: coordinate discovery by name, fill policies, lat-lon orientation,
  source lat-lon meshes, time decoding by units class, staging of compressed
  inputs, flux-unit grammar; the `cs_native` and lat-lon surface-flux file
  schemas with their writers and the runtime's `cs_native` reader
- `Preprocessing/` — transport-binary and met-data preparation
- `Downloads/` — data acquisition helpers
- `Visualization/` — topology-aware snapshot views and optional Makie methods
- [`Analysis/`](Analysis/README.md) — post-run diagnostics numerics: snapshot
  time series, equal-area band means, seasonal/interannual decompositions,
  skill statistics and column profile tools (loaded after `Models`)
- `Tape/` — checkpoint schedules and device/pinned-host/mmap tape storage
- `Adjoints/` — reverse operators and objective seeding
- `Footprint/` — forward-record/reverse-loop footprint workflows included by
  `Adjoints`
- `Inversion/` — observations, covariance, costs, and optimizers included by
  `Adjoints`

## Read This First For Common Goals

- "I need to understand one model step":
  start with [`Models/README.md`](Models/README.md), then
  [`Operators/README.md`](Operators/README.md)
- "I need to understand why a topology dispatches differently":
  start with [`Grids/README.md`](Grids/README.md), then
  [`State/README.md`](State/README.md), then the relevant operator README
- "I need to trace data from disk to runtime":
  start with [`MetDrivers/README.md`](MetDrivers/README.md)
- "I need to understand runtime NetCDF output":
  start with
  [`../docs/src/getting_started/inspecting_output.md`](../docs/src/getting_started/inspecting_output.md)
- "I need to add a new operator input field":
  start with [`State/Fields/README.md`](State/Fields/README.md)

## Structure checks

Read the generated [code map](../docs/src/development/code_map.md) and the
folder README before editing. [`LAYERS.toml`](LAYERS.toml) records module and
folder dependency edges and a one-directional folder order. Existing reverse
folder edges have reasoned, shrink-only exceptions. The test-only JuliaSyntax
checker rejects new dependency edges, later-module references, cycles, new
duplicate helpers, body clones, private reach-ins, and detached docstrings.
Existing debt is snapshotted under
[`test/structure/`](../test/structure/README.md); allowances only shrink.
Each folder has a 65,000-token budget (immediate Julia source bytes / 3.5);
subfolders are budgeted separately. Existing oversized folders have exact,
shrink-only byte ceilings. Core modules cannot depend on Visualization,
Analysis, DataDownloads, the Makie extension, or future Extras modules;
existing package-facade re-exports are allowlisted with reasons.
The code map lists files, sizes, dependencies/users, public-name counts and
forward/adjoint mirrors. The gate rejects a stale map; regenerate it with
`scripts/checks/check_structure.jl --write-code-map` under `--project=test`.
README contract headings are reported only during the P1 rollout.

## Related Docs

- Runtime walkthrough:
  [`../docs/20_RUNTIME_FLOW.md`](../docs/20_RUNTIME_FLOW.md)
- Binary and driver overview:
  [`../docs/30_BINARY_AND_DRIVERS.md`](../docs/30_BINARY_AND_DRIVERS.md)
- Quality gates:
  [`../docs/40_QUALITY_GATES.md`](../docs/40_QUALITY_GATES.md)
- Operator block ordering:
  [`Operators/README.md`](Operators/README.md) and
  [`../docs/20_RUNTIME_FLOW.md`](../docs/20_RUNTIME_FLOW.md)

## Shared helpers owned here

`Architectures.compensated_total` is the public compensated Float64 reduction
for arrays and tuples of arrays, used by State mass diagnostics and Output
snapshot totals. Backend summation policies remain unchanged.
