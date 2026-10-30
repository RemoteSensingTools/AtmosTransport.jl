# CubedSphere

Panel halos, packed sweeps, seams, CS transport, FV3 vertical profiles and numerical transposes.

This is a plain sub-folder of the Advection module; see the
[parent contract](../README.md) for the fenced-root dependency decision.

## File map

- `HaloExchange.jl`: edge and corner halo point operations.
- `cs_sweep_common.jl`: packed generic kernels, common host launchers and seam-coupled x/y driver.
- `cs_sweep_x.jl`: packed upwind x kernel and directional entrypoints.
- `cs_sweep_y.jl`: packed upwind y kernel and directional entrypoints.
- `cs_sweep_z.jl`: packed upwind z kernel, production Lin–Rood single-tracer vertical sweep and 3D interior copy.
- `cs_subcycling.jl`: CS budget evaluation and binary CFL checks.
- `CubedSphereStrang.jl`: packed palindrome, step-start halos and vertical panel drivers.
- `CubedSphereSeams.jl`: canonical paired seam transfer and flux view.
- `vertical_fv3_profile.jl`: FV3 scalar_profile vertical sweep.
- `vertical_fv3_profile_adjoint.jl`: forward replay on _LinearForm and column transpose.
- `adjoint_workspace.jl`: reverse scratch and split-sweep scheme unions.
- `AdvectionAdjoint.jl`: split-sweep face derivatives, kernels and sweep driver.
- `HaloAdjoint.jl`: halo transpose kernels and launcher.
- `CubedSphereSeamsAdjoint.jl`: seam seeds and coupled seam transpose.

## Depends on

Reconstruction and Structured shared helpers, parent state dispatch and fenced workspace/Lin–Rood interfaces, Grids, State, Parameters, Architectures, SectionTimer and Operators recorder hooks. Diffusion/SurfaceFlux remain in the existing midpoint.

## Used by

Models, StepPlan and Adjoints/Footprint reverse wiring.

## Shared helpers owned here

CSAdjointWorkspace, CSAdjointLinearScheme, CSAdjointNonlinearScheme, CSAdjointFV3Vertical, `_adjoint_scheme_sweep!`, `_adjoint_fill_panel_halos!`, and `_cs_packed_sweep_workgroupsize`; exposed through the parent module contract.

## Forward<->adjoint mirrors

HaloExchange.jl ↔ HaloAdjoint.jl; CubedSphereSeams.jl ↔ CubedSphereSeamsAdjoint.jl; cs_sweep_*.jl ↔ AdvectionAdjoint.jl; vertical_fv3_profile.jl ↔ vertical_fv3_profile_adjoint.jl. Tape recording remains in Adjoints/Footprint.
