# Structured

Latitude-longitude packed transport, reduced-Gaussian face transport, and the shared Strang midpoint.

This is a plain sub-folder of the Advection module; see the
[parent contract](../README.md) for the fenced-root dependency decision.

## File map

- `multitracer_kernels.jl`: packed x/y/z kernels and TracerView.
- `workspace.jl`: structured/face-indexed buffers and Adapt support.
- `sweeps.jl`: reduced-Gaussian horizontal/vertical sweeps.
- `subcycling.jl`: shared palindrome budget primitives and LL/RG counts.
- `StrangSplitting.jl`: LL strang_split! and palindrome_center!.
- `strang_apply.jl`: LL/RG model-facing apply! methods.
- `multitracer_strang.jl`: packed directional launchers and LL palindrome.

## Depends on

Reconstruction, Grids, State, MetDrivers, Parameters, SectionTimer and parent operator interfaces; Diffusion/SurfaceFlux for the existing midpoint.

## Used by

Models, CubedSphere (TracerView, budget and midpoint), and StepPlan.

## Shared helpers owned here

AdvectionWorkspace, TracerView, strang_split_mt!, palindrome_center! and the palindrome budget primitives.

## Forward<->adjoint mirrors

No LL/RG numerical adjoint is shipped. Shared reconstruction is mirrored by `../CubedSphere/AdvectionAdjoint.jl`.
