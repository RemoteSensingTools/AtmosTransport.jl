# Runtime Physics

The `Models.Physics` submodule owns runtime specs, styles, recipes and metadata.
It does not depend on model or simulation types. Per-window forcing refresh
lives beside DrivenSimulation in Runner.

## File Map

- `Physics.jl`: module imports and ordered recipe includes.
- `RuntimeRecipeStyles.jl`: topology style tags and the unchanged reset-mode
  config default, defined before spec dispatch.
- `RuntimePhysicsSpecs.jl`: typed advection, diffusion, convection and chemistry
  config specs and materialization; shared CS host Kz-cache allocation.
- `RuntimePhysicsRecipe.jl`: context/style resolution, capability checks and
  operator composition. Each binary is validated independently.
- `RuntimePhysicsMetadata.jl`: physics and met/binary provenance attributes;
  imports the StepPlans cadence/reset normalizers it reads.

## Depends on

Operators (Diffusion, Convection, Chemistry, SurfaceFlux, StepPlans),
MetDrivers, State, Grids, ConfigChecks, Parameters, Architectures, SectionTimer;
No sibling or parent-facade dependency.

## Used by

DrivenRunner, driven construction/stepping, inversion setup and runtime tests.

## Shared helpers owned here

Public recipe/spec parsers and materializers retain their exports. The runtime
style tags and `runtime_physics_attributes` retain qualified access. The runner
uses these explicitly shared config assembly seams:

# shared: _ADVECTION_SPEC_KEYS, _advection_section, _diffusion_section, _convection_section, _chemistry_section

Other underscore helpers are private to Physics. The Models facade re-exports
only public names; tests reach private helpers as `Models.Physics._name`.

## Forward<->adjoint mirrors

Runner's `cs_adjoint_inputs` invokes its production forcing refresh. Physics
contains no separate adjoint arithmetic; forcing is captured for replay.

Scalar and choice parsing delegates to ConfigChecks, including the existing
forward/inversion limiter tables. No private numeric-parser interface remains.
