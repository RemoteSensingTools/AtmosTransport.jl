# Advection

Finite-volume tracer advection with directional splitting or cubed-sphere
Lin–Rood horizontal cross terms.

This folder owns the transport core: reconstruction, limiter logic,
structured and face-indexed sweeps, cubed-sphere panel transport, and
the model-facing `apply!` entrypoints that the transport block calls.

## Entry Points

- Scheme hierarchy:
  [`schemes.jl`](schemes.jl)
  defines `AbstractAdvectionScheme`, `UpwindScheme`, `SlopesScheme`,
  `PPMScheme`, and `LinRoodPPMScheme`
- Structured and face-indexed runtime orchestrators:
  [`StrangSplitting.jl`](StrangSplitting.jl) provides `strang_split!`,
  [`multitracer_strang.jl`](multitracer_strang.jl) `strang_split_mt!`, and
  [`strang_apply.jl`](strang_apply.jl) the `apply!` entry points
- Cubed-sphere runtime orchestrator:
  [`CubedSphereStrang.jl`](CubedSphereStrang.jl) provides `strang_split_cs!`
  and `strang_split_cs_mt!`; [`cs_workspace.jl`](cs_workspace.jl)
  `CSAdvectionWorkspace`
- Lin–Rood horizontal transport:
  [`LinRood.jl`](LinRood.jl) provides `fv_tp_2d_cs!` and
  `CSLinRoodAdvectionWorkspace`; the runtime pairs it with vertical upwind
- Cubed-sphere halo support:
  [`HaloExchange.jl`](HaloExchange.jl)
  provides `fill_panel_halos!` and `copy_corners!`
- Vertical mass-flux diagnosis is provided by the meteorology contract as
  `diagnose_cm_from_continuity!`.

## Runtime Shape

`scheme="ppm"` selects standard split PPM with its default monotone limiter.
`scheme="linrood"` selects the CS cross-term path; `ppm_order=5` (default) or
`7` chooses its edge-value family. ORD=7 retains the order-5 interior and adds
special panel-edge treatment. It does not select a seventh-order transport
method. Configuration parsing lives in
[`../../Models/RuntimePhysicsSpecs.jl`](../../Models/RuntimePhysicsSpecs.jl).

- LatLon and reduced-Gaussian transport run through the `apply!` methods in
  [`strang_apply.jl`](strang_apply.jl) and the palindrome in
  [`StrangSplitting.jl`](StrangSplitting.jl)
- Cubed-sphere transport runs through
  [`CubedSphereStrang.jl`](CubedSphereStrang.jl)
- Diffusion and surface flux are not separate outer blocks here; they are
  threaded into the palindrome midpoint through imports from
  `../Diffusion` and `../SurfaceFlux`
- This folder is the main place where topology-specific transport
  execution diverges while the public operator API stays uniform

Packed CS sweep launch geometry is selected by
`_cs_packed_sweep_workgroupsize`. The CUDA extension uses a 32×2 tile for
Float32 PPM and a 32-thread row for Float64 PPM; other paths keep their
existing 256-thread default. These layouts reduce inactive threads on panel
rows while retaining the same kernels, tracer loop, air-mass update, and
copy-back/ping-pong behavior. GPU launch
regressions are checked by `test/diagnostic/test_cs_ppm_launch_gpu.jl`.

## File Map

- [`Advection.jl`](Advection.jl) — submodule assembly, imports, include order
- [`schemes.jl`](schemes.jl) — scheme and limiter type hierarchy
- [`limiters.jl`](limiters.jl) — limiter formulas and slope controls
- [`reconstruction.jl`](reconstruction.jl) — face-value reconstruction
  logic shared by the sweep kernels
- [`structured_kernels.jl`](structured_kernels.jl) — structured-grid
  sweep kernels and helpers
- [`multitracer_kernels.jl`](multitracer_kernels.jl) — fused multi-tracer
  transport kernels and `TracerView`
- [`workspace.jl`](workspace.jl) — `AdvectionWorkspace`, the double
  buffers of the structured and face-indexed sweeps
- [`sweeps.jl`](sweeps.jl) — directional sweeps: structured x/y/z
  (generated with `@eval`) and face-indexed horizontal/vertical
- [`subcycling.jl`](subcycling.jl) — CFL subcycling pass counts and
  subcycled sweeps
- [`StrangSplitting.jl`](StrangSplitting.jl) — structured transport
  palindrome `strang_split!` with the diffusion/surface-flux midpoint
- [`strang_apply.jl`](strang_apply.jl) — model-facing `apply!` for
  structured and face-indexed states (the RG `H → V → H` path)
- [`multitracer_strang.jl`](multitracer_strang.jl) — multi-tracer sweeps
  and `strang_split_mt!`
- [`HaloExchange.jl`](HaloExchange.jl) — cubed-sphere panel-edge halo
  exchange and corner fill
- [`cs_sweep_common.jl`](cs_sweep_common.jl), [`cs_sweep_x.jl`](cs_sweep_x.jl),
  [`cs_sweep_y.jl`](cs_sweep_y.jl), [`cs_sweep_z.jl`](cs_sweep_z.jl) —
  cubed-sphere panel sweeps (shared kernels and gamma-clamped upwind, then
  per direction, with paired seam transfers for X and Y)
- [`cs_workspace.jl`](cs_workspace.jl) — `CSAdvectionWorkspace`
- [`cs_subcycling.jl`](cs_subcycling.jl) — static palindrome CFL subcycle count
- [`CubedSphereStrang.jl`](CubedSphereStrang.jl) — panel-native
  cubed-sphere palindrome (`strang_split_cs!`, `strang_split_cs_mt!`)
- [`CubedSphereSeams.jl`](CubedSphereSeams.jl) — canonical physical seam
  transfers paired across panels within each directional group
- [`ppm_subgrid_distributions.jl`](ppm_subgrid_distributions.jl) — PPM
  subcell distributions shared by CS-specific code
- [`LinRoodSeams.jl`](LinRoodSeams.jl) — shared final seam estimates and
  their transpose for conservative Lin–Rood panel exchange
- [`LinRood.jl`](LinRood.jl) — Lin-Rood style cubed-sphere horizontal
  transport utilities
- [`linrood_adjoint_kernels.jl`](linrood_adjoint_kernels.jl) —
  Lin-Rood adjoint kernel helpers used by the CS reverse path

## Common Tasks

- Adding a new advection scheme:
  start in [`schemes.jl`](schemes.jl), then wire reconstruction behavior
  in [`reconstruction.jl`](reconstruction.jl)
- Debugging mass conservation:
  read the palindrome structure in [`StrangSplitting.jl`](StrangSplitting.jl)
  or [`CubedSphereStrang.jl`](CubedSphereStrang.jl) before changing kernels
- Extending cubed-sphere transport:
  start with [`CubedSphereStrang.jl`](CubedSphereStrang.jl) and
  [`HaloExchange.jl`](HaloExchange.jl); keep the panel-native boundary honest
- Performance tuning:
  compare [`multitracer_kernels.jl`](multitracer_kernels.jl) against
  [`structured_kernels.jl`](structured_kernels.jl) before adding another
  launch path
- Debugging topology dispatch:
  read the `apply!` methods in [`strang_apply.jl`](strang_apply.jl)

## Cross-Dependencies

- `../Diffusion` and `../SurfaceFlux` are imported into the palindrome
  midpoint; changes there can change advection runtime behavior
- [`../../State/`](../../State/) provides tracer slices, flux-state
  types, and the panel-native cubed-sphere containers
- [`../../Grids/`](../../Grids/) provides mesh geometry and panel
  connectivity
- [`../../MetDrivers/`](../../MetDrivers/) provides
  `diagnose_cm_from_continuity!`
- [`../../Models/TransportModel.jl`](../../Models/TransportModel.jl)
  chooses when this folder's `apply!` methods are executed

## Related Docs And Tests

- Topology coverage:
  [`../TOPOLOGY_SUPPORT.md`](../TOPOLOGY_SUPPORT.md)
- Runtime/block ordering:
  [`../../Models/TransportModel.jl`](../../Models/TransportModel.jl) and
  [`../../../docs/20_RUNTIME_FLOW.md`](../../../docs/20_RUNTIME_FLOW.md)
- Runtime walkthrough:
  [`../../../docs/20_RUNTIME_FLOW.md`](../../../docs/20_RUNTIME_FLOW.md)
- Tests:
  - [`../../../test/core/test_advection_kernels.jl`](../../../test/core/test_advection_kernels.jl)
  - [`../../../test/core/test_cubed_sphere_advection.jl`](../../../test/core/test_cubed_sphere_advection.jl)
  - [`../../../test/core/test_basis_explicit_core.jl`](../../../test/core/test_basis_explicit_core.jl)
  - [`../../../test/orphan/test_emissions_palindrome.jl`](../../../test/orphan/test_emissions_palindrome.jl)
  - [`../../../test/core/test_diffusion_palindrome_contract.jl`](../../../test/core/test_diffusion_palindrome_contract.jl)
