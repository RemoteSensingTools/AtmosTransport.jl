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
  [`linrood_horizontal.jl`](linrood_horizontal.jl) provides `fv_tp_2d_cs!`,
  [`LinRood.jl`](LinRood.jl) `CSLinRoodAdvectionWorkspace`; configured runs
  pair it by default with the FV3 `kord = 8` vertical profile
- Cubed-sphere halo support:
  [`HaloExchange.jl`](HaloExchange.jl)
  provides `fill_panel_halos!` and `copy_corners!`
- Vertical mass-flux diagnosis is provided by the meteorology contract as
  `diagnose_cm_from_continuity!`.

## Runtime Shape

`scheme="ppm"` selects standard split PPM with its default monotone limiter;
`limiter="cw84"` selects the complete Colella–Woodward PPM (`CW84Limiter`).
`scheme="linrood"` selects the CS cross-term path; `ppm_order=5` (default) or
`7` chooses its edge-value family. ORD=7 retains the order-5 interior and adds
special panel-edge treatment: one edge value per face, the Putman–Lin seam
average at the panel edge and FV3's one-sided values on the faces beside it
(`_ppm_edge_values_ord7`), in the forward and adjoint kernels alike. It does not select a seventh-order transport
method. Its configured vertical default is `fv3_kord8`; `vertical="upwind"`
selects first-order vertical transport. Both have an adjoint.
Configuration parsing lives in
[`../../Models/RuntimePhysicsSpecs.jl`](../../Models/RuntimePhysicsSpecs.jl).

- LatLon and reduced-Gaussian transport run through the `apply!` methods in
  [`strang_apply.jl`](strang_apply.jl) and the palindrome in
  [`StrangSplitting.jl`](StrangSplitting.jl)
- Cubed-sphere transport runs through
  [`CubedSphereStrang.jl`](CubedSphereStrang.jl)
- Diffusion and surface flux are not separate outer blocks here; they are
  threaded into the palindrome midpoint through imports from
  `../Diffusion` and `../SurfaceFlux`. Every topology builds the midpoint
  with `palindrome_center!` ([`StrangSplitting.jl`](StrangSplitting.jl)):
  one `V(dt)` without a surface-flux operator, otherwise
  `surface_flux_center!` with the diffusion operator's coupling,
  `V(dt/2) → S(dt) → V(dt/2)` or `S(dt) → V(dt)`
- A cubed-sphere split-sweep step starts with `fill_cs_step_start_halos!`
  ([`CubedSphereStrang.jl`](CubedSphereStrang.jl), all edge halos plus the
  X-orientation corners): the first X sweep reads halos, and convection or
  chemistry between steps changed the interiors
- CFL subcycling (every topology): one count `n` from the palindrome
  outflow budget against `min(m_start, m_end)` (no pass removes more than
  `cfl_limit / 2` of a cell's current mass; an invalid flux set throws), then `(X Y Z)^n → midpoint → (Z Y X)^n` with every flux
  divided by `n` (`H V` / `V H` on reduced Gaussian); `n = 1` is the plain
  palindrome
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
- [`subcycling.jl`](subcycling.jl) — palindrome CFL subcycle counts of the
  structured and face-indexed paths
- [`StrangSplitting.jl`](StrangSplitting.jl) — structured and cubed-sphere
  `strang_split!`, and `palindrome_center!`, the diffusion/surface-flux
  midpoint shared by every topology
- [`strang_apply.jl`](strang_apply.jl) — model-facing `apply!` for
  structured and face-indexed states (the RG `H → V → H` path)
- [`multitracer_strang.jl`](multitracer_strang.jl) — multi-tracer sweeps
  and `strang_split_mt!`
- [`HaloExchange.jl`](HaloExchange.jl) — cubed-sphere panel-edge halo
  exchange and corner fill, written as point operations (`EdgeHaloFill`,
  `CornerHaloFill`; see `Architectures.AbstractPointOp`): with the default
  fusion policy (CUDA) one launch fills all 24 panel edges and, for `dir = 1`
  or `2`, a second the corners; on Metal there are 24 edge launches and, for
  `dir = 1` or `2`, 6 corner launches, each with bound panel arrays; one
  synchronization follows the last launch under either policy. On the CPU the
  same bodies run as loops
- [`cs_sweep_common.jl`](cs_sweep_common.jl), [`cs_sweep_x.jl`](cs_sweep_x.jl),
  [`cs_sweep_y.jl`](cs_sweep_y.jl), [`cs_sweep_z.jl`](cs_sweep_z.jl) —
  cubed-sphere panel sweeps (shared kernels and gamma-clamped upwind, then
  per direction, with paired seam transfers for X and Y)
- [`cs_workspace.jl`](cs_workspace.jl) — `CSAdvectionWorkspace`
- [`cs_subcycling.jl`](cs_subcycling.jl) — the palindrome outflow budget
  shared by all split-sweep topologies and the cubed-sphere subcycle count
- [`CubedSphereStrang.jl`](CubedSphereStrang.jl) — panel-native
  cubed-sphere palindrome (`strang_split_cs!`, `strang_split_cs_mt!`) and
  the step-start halo refresh `fill_cs_step_start_halos!`
- [`vertical_fv3_profile.jl`](vertical_fv3_profile.jl) — vertical sweep with
  FV3's `scalar_profile` (kord 8) piecewise-parabolic tracer profile
- [`vertical_fv3_profile_adjoint.jl`](vertical_fv3_profile_adjoint.jl) —
  its reverse pass: the forward profile code replayed on `_LinearForm`
  numbers (value plus partials) to obtain the limiter branches and local
  Jacobians, and the transposed column kernel used by the Lin-Rood adjoint
- [`CubedSphereSeams.jl`](CubedSphereSeams.jl) — canonical physical seam
  transfers paired across panels within each directional group; a group's
  transfers are cached in one launch (`SeamTransferCache` point operations,
  see `Architectures.AbstractPointOp`) and applied seam by seam in queue order
- [`ppm_subgrid_distributions.jl`](ppm_subgrid_distributions.jl) — PPM
  subcell distributions shared by CS-specific code
- [`LinRoodSeams.jl`](LinRoodSeams.jl) — shared final seam estimates and
  their transpose for conservative Lin–Rood panel exchange
- [`LinRood.jl`](LinRood.jl) — Lin-Rood workspace, divergence damping, PPM
  face, pre-advection and update kernels
- [`linrood_horizontal.jl`](linrood_horizontal.jl) — Lin-Rood horizontal
  advection drivers (`fv_tp_2d_cs!`, q-space variant) and the Lin-Rood Strang split
- [`linrood_adjoint_kernels.jl`](linrood_adjoint_kernels.jl),
  [`linrood_adjoint_rm_faces.jl`](linrood_adjoint_rm_faces.jl),
  [`linrood_adjoint_panel.jl`](linrood_adjoint_panel.jl) — Lin-Rood adjoint
  kernels (update, pre-advection and q-input faces; rm-input faces; the
  one-panel horizontal adjoint) used by the CS reverse path

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
- The cubed-sphere adjoint (`../../Adjoints/`, `../../Footprint/`) composes
  its replayed and recorded steps from `fill_cs_step_start_halos!` and
  `surface_flux_center!`, so the adjoint's forward model is the production
  step. Change the cubed-sphere step composition in `strang_split!` together
  with `_replay_cs_step!` and the tape recorders;
  `test/core/test_cs_adjoint_production_parity.jl` compares them
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
  - [`../../../test/core/test_emissions_palindrome.jl`](../../../test/core/test_emissions_palindrome.jl)
  - [`../../../test/core/test_diffusion_palindrome_contract.jl`](../../../test/core/test_diffusion_palindrome_contract.jl)
