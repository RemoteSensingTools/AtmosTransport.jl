# Advection

Finite-volume tracer transport and its numerical transposes. `Advection.jl`
assembles one module, preserving `AtmosTransport.Operators.Advection` names.

## Layout and entry points

- [Reconstruction/](Reconstruction/README.md): scheme/limiter types, face
  reconstruction and PPM sub-grid distributions.
- [Structured/](Structured/README.md): latitude-longitude packed sweeps,
  reduced-Gaussian face sweeps, workspace, CFL budget and Strang splitting.
- [CubedSphere/](CubedSphere/README.md): panel halos, seams, packed sweeps,
  subcycling, CS state adapters, FV3 vertical profile and numerical adjoints.
- Root files: the fenced Lin–Rood implementation and `cs_workspace.jl`.

These are plain sub-folders of the same module. The fenced root workspaces
and Lin–Rood kernels use child helpers, and CS drivers need the root workspace
types. Separate submodules would introduce circular imports or require editing
the fenced implementation. Shared-module assembly keeps the existing forward kernel
and scheme identities; among the children dependencies point from CubedSphere
to Structured to Reconstruction. Root assembly wires all three. A future
LinRood folder can revisit this.

`apply!` / `strang_split!` dispatch on state topology. Latitude-longitude uses
`strang_split_mt!`; reduced Gaussian uses face sweeps; split CS schemes use
`strang_split_cs_mt!` / `strang_split_cs_mt_pingpong!`. The single-tracer CS
copy-back palindrome and structured single-tracer shells have been removed;
low-level callers pack one tracer in the fourth dimension. Lin–Rood retains
its production single-tracer `_sweep_z_panels!` and `_sweep_z!` helpers.

## Numerical contracts

Advection calls `apply_midpoint!(hook, active_rm, active_m[, tracer])` between
the two palindrome halves. `AbstractMidpointHook` is the interface and
`NoMidpoint()` is the default, with no floating-point work. StepPlan supplies
`TransportMidpoint` and owns diffusion/emission ordering and preflight.
Per-tracer RG and Lin–Rood drivers pass the tracer index. Packed LL/CS
drivers pass the active ping-pong buffers. Hooks are column-local and leave
air mass unchanged; they must never use the stale step-start tracer arrays.

CFL subcycling runs `(X Y Z)^n → midpoint → (Z Y X)^n`, or `(H V)^n →
midpoint → (V H)^n` on reduced Gaussian, with flux divided by `n` and a
budget against `min(m_start, m_end)`. Lin–Rood retains its own H/Z palindrome
and binary scheduling behavior. Every sweep reads separate input and output
buffers. The CS packed palindrome returns stale halos: callers refresh them
with `fill_cs_step_start_halos!` at the next step start. Z and midpoint
operators must remain column-local. Recorder hooks, halo fills, their order,
and the final host synchronization are part of the adjoint/performance
contract.

GPU kernels retain their original signatures, workgroups and array layouts.
`_cs_packed_sweep_workgroupsize` is the backend extension hook; packed upwind
still uses 256. X uses `_HaloPaddedAxis`; Y retains its bounds-check policy.

Mass-floor policies remain distinct: reconstruction divides by `max(m, eps)`;
CS upwind returns zero flux for non-positive mass; Lin–Rood uses its strict
`100eps` empty-cell threshold. The upwind adjoint's eps floor, the absent
LL/RG upwind Courant clamp and the three different dual-number implementations
remain follow-ups. No policy unification belongs in this structural change.

## File map

- `Advection.jl`: imports and include order.
- `midpoint.jl`: abstract midpoint, no-op default and the typed adapter for
  the fenced Lin–Rood zero-argument callback.
- `cs_dispatch.jl`: CS state dispatch and midpoint forwarding; the shared `apply!`
  and `strang_split!` facade methods call the CS drivers.
- `cs_workspace.jl`: fenced CS workspace, scratch and subcycle logging.
- `LinRood.jl`: fenced Lin–Rood workspaces and forward kernels.
- `LinRoodSeams.jl`: fenced shared face projection, also its own transpose.
- `linrood_horizontal.jl`: fenced forward horizontal drivers and palindrome.
- `linrood_adjoint_kernels.jl`: fenced update, predictor and q-face transposes.
- `linrood_adjoint_rm_faces.jl`: fenced tracer-mass face transposes.
- `linrood_adjoint_panel.jl`: fenced panel reverse composition.

The three sub-folder READMEs list every child file.

## Depends on

Grids (meshes, panel connectivity and seam indices), State (tracer and flux
containers), Parameters (defaults), Architectures (point operations),
MetDrivers (`uses_binary_substep_contract`), SectionTimer and the parent
Operators interfaces/recorder hooks. There are no Diffusion, SurfaceFlux,
StepPlan, Adjoints or Tape imports.

## Used by

Models, Operators/StepPlan, Adjoints, Footprint and the CUDA launch extension.
Adjoints keeps support policy, recording and reverse-walk wiring.

## Shared helpers owned here

- Scheme/limiter types, `required_halo_width`, `needs_column_scratch`,
  `AdvectionWorkspace`, `CSAdvectionWorkspace`, `CSLinRoodAdvectionWorkspace`.
- `CSAdjointWorkspace`, `CSAdjointLinearScheme`, `CSAdjointNonlinearScheme`,
  `CSAdjointFV3Vertical`: reverse scratch and kernel dispatch
  unions; the public `Adjoints.CSAdjointWorkspace` import remains available.
- `_adjoint_scheme_sweep!`, `_adjoint_fill_panel_halos!`: numerical reverse
  entrypoints consumed by the Adjoints tape walk and Lin–Rood tape wiring.
- `_cs_packed_sweep_workgroupsize`: backend launch-layout extension hook.
- `_share_lr_seam_faces!`, `_accumulate_linrood_halo_adjoint!`,
  `apply_linrood_horizontal_adjoint_single_panel!`, `apply_linrood_update_adjoint!`:
  the existing Lin–Rood reverse interface, implemented in fenced root files.
- `fill_panel_halos!`, `copy_corners!`, `fill_cs_step_start_halos!`,
  `AbstractMidpointHook`, `NoMidpoint`, `apply_midpoint!`, `strang_split!`, `strang_split_mt!`, `strang_split_cs_mt!`:
  forward execution and shared midpoint/halo contracts.

`_ppm_edge_value(..., FT)` is the common fourth-order edge expression for
reconstruction, ORD5/ORD7 scalar edges and the split-sweep adjoint value.
The fenced D6 Lin–Rood edge formula remains unchanged.

<!-- The structure checker reads these explicit owned interfaces. -->
# shared: _adjoint_scheme_sweep!, _adjoint_fill_panel_halos!, _cs_packed_sweep_workgroupsize, _share_lr_seam_faces!, _accumulate_linrood_halo_adjoint!

## Forward<->adjoint mirrors

CS sweep/reconstruction ↔ `CubedSphere/AdvectionAdjoint.jl`;
`CubedSphere/HaloExchange.jl` ↔ `CubedSphere/HaloAdjoint.jl`;
`CubedSphere/CubedSphereSeams.jl` ↔ `CubedSphere/CubedSphereSeamsAdjoint.jl`;
`CubedSphere/vertical_fv3_profile.jl` ↔ `vertical_fv3_profile_adjoint.jl`.
The FV3 transpose shares forward profile functions through `_LinearForm` and
is supported for Lin–Rood vertical sweeps. Split-sweep FV3 adjoint support is
unchanged. Lin–Rood forward ↔ fenced `linrood_adjoint_*.jl` plus
`Adjoints/LinRoodTape.jl` wiring. Exact file pairs are in
`test/structure/mirrors.toml`.

## Tests

`test_advection_kernels`, `test_cubed_sphere_advection` (including the independent
scalar Z-sweep reference), `test_subcycle_composition`, `test_cs_halo_fill`,
`test_cs_seam_exchange`, `test_fv3_vertical_profile`, `test_ppm_cw84_limiter`,
`test_cs_adjoint_production_parity` and `test_cs_adjoint_step_plan_parity` guard
the forward and reverse contracts. GPU goldens and a warm performance A/B are
required when reviewing host launch changes.
