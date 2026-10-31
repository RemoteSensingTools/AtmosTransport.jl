# Runtime Physics

Plain subfolder of Models. Specs/recipes do not depend on Core types, while
refresh methods dispatch on Runner's `DrivenSimulation` and Core's
`TransportModel`. This requires the interleaved assembly in `Models.jl`.

## File Map

- `RuntimeRecipeStyles.jl`: topology style tags and the unchanged reset-mode
  config default, defined before spec dispatch.
- `RuntimePhysicsSpecs.jl`: typed advection, diffusion, convection and chemistry
  config specs and materialization; shared CS host Kz-cache allocation.
- `RuntimePhysicsRecipe.jl`: context/style resolution, capability checks and
  operator composition. Each binary is validated independently.
- `RuntimePhysicsMetadata.jl`: physics and met/binary provenance attributes.
- `driven_physics_refresh.jl`: per-window dz/Kz/convection refresh and midpoint
  forcing; the one-step-window rule remains unchanged. `_refreshes_fluxes`
  says at which steps `_refresh_forcing!` rewrites the face fluxes (a
  window's first step, or every step when interpolated).

## Depends on

Operators (Diffusion, Convection, Chemistry, SurfaceFlux, StepPlans),
MetDrivers, State, Grids, ConfigChecks, Parameters, Architectures, SectionTimer;
Core and Runner types through Models.

## Used by

DrivenRunner, driven construction/stepping, inversion setup and runtime tests.

## Shared helpers owned here

Public recipe/spec parsers and materializers retain their exports. The runtime
style tags and `runtime_physics_attributes` retain qualified access. The runner
uses these explicitly shared config assembly seams:

# shared: _spec_int, _ADVECTION_SPEC_KEYS, _advection_section, _diffusion_section, _convection_section, _chemistry_section

`_config_bool` remains a compatibility alias of ConfigChecks.config_bool;
InitialConditionIO and InputStaging now import directly from ConfigChecks.

## Forward<->adjoint mirrors

Core's `cs_adjoint_inputs` invokes production forcing refresh and copies the
fluxes exactly when `_refreshes_fluxes` says production rewrote them. This
folder contains no separate adjoint arithmetic; forcing is captured for replay.
