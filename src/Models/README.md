# Models

Minimal runtime composition layer for `src`.

This folder turns the lower-level state, grid, met-driver, and operator
pieces into runnable model objects. If you want to understand what
actually happens during a step, this is one of the first folders to
read.

## Entry Points

- Module assembly:
  [`Models.jl`](Models.jl)
- Core runtime object:
  [`TransportModel.jl`](TransportModel.jl)
  defines `TransportModel`, `step!`, and the `with_*` operator installers
- Fixed-Δt smoke harness:
  [`Simulation.jl`](Simulation.jl)
  defines `Simulation` and `run!`
- Window-driven production-style harness:
  [`DrivenSimulation.jl`](DrivenSimulation.jl)
  defines `DrivenSimulation`, window progression, forcing refresh, and
  runtime validation

## Runtime Composition Today

- `TransportModel.step!` runs:
  - transport block (advection, with diffusion and surface flux at
    the Strang midpoint)
  - convection block (CMFMC, matrix CMFMC, or TM5 on supported topologies)
  - chemistry block

## File Map

- [`Models.jl`](Models.jl) — submodule assembly
- [`TransportModel.jl`](TransportModel.jl) — main model struct,
  constructors, operator installers, runtime block order
- [`Simulation.jl`](Simulation.jl) — simple fixed-step loop for direct
  model runs
- [`DrivenSimulation.jl`](DrivenSimulation.jl) — met-window-driven loop,
  forcing interpolation, air-mass refresh, and runtime compatibility checks
- [`RuntimeRecipeStyles.jl`](RuntimeRecipeStyles.jl) — runtime-style traits
  (`AbstractRuntimeRecipeStyle` + LatLon/ReducedGaussian/CubedSphere) the
  physics-spec `materialize` methods dispatch on
- [`RuntimePhysicsSpecs.jl`](RuntimePhysicsSpecs.jl) — typed config specs parsed
  once from TOML + `materialize` (Oceananigans-style); convection family
  (`convection_spec`, the `lmax_conv`/`n_merge`-needs-`use_collab_lu` guard) +
  advection family (`advection_spec`, LinRood is cubed-sphere only) +
  chemistry family (`chemistry_spec`, `materialize` dispatches on run `FT`) +
  diffusion family (`diffusion_spec`; `materialize(spec, style, FT, context)`
  threads all three — topology/capability helpers stay in `RuntimePhysicsRecipe.jl`)
- [`RuntimePhysicsRecipe.jl`](RuntimePhysicsRecipe.jl) — topology-dispatched
  runtime recipe construction and capability validation for advection,
  diffusion, convection, chemistry, and surface forcing
- [`InitialConditionIO.jl`](InitialConditionIO.jl) — topology-dispatched
  VMR builder (`build_initial_mixing_ratio` on LL/RG/CS),
  basis-aware VMR → tracer-mass packer (`pack_initial_tracer_mass`),
  surface-flux loader + LL/RG/CS `build_surface_flux_source` builders
  with conservative regrid + cell-area integration,
  `FileInitialConditionSource` / `FileSurfaceFluxField` containers
- [`BinaryPathExpander.jl`](BinaryPathExpander.jl) —
  `expand_binary_paths(input_cfg)` resolves either an explicit
  `binary_paths = [...]` list or a `folder + start_date + end_date
  (+ file_pattern)` shape. Explicit lists retain their order; folder selection
  sorts by date and checks continuity on the closed date range.
- [`DrivenRunner.jl`](DrivenRunner.jl) — library-level
  `run_driven_simulation(cfg)` entry point for all driven runs. Owns the
  runtime flow behind `scripts/run_transport.jl`: first-driver
  construction, config and capability validation against TOML physics,
  tracer init via `build_initial_mixing_ratio` + basis-aware
  `pack_initial_tracer_mass`, surface-source wiring, GPU-residency
  assertion (`feedback_verify_gpu_runs_on_gpu`), per-window loop,
  and snapshot NetCDF output
- [`runner/`](runner/) — the runner's progress timer, configuration validation,
  runtime summary, owned input/output resources, observation-sampling glue
  (`runner/observations.jl`: builds the `[output.observations]` sampler from
  the model state, hands it to `RunSnapshotOutput` so it closes with the
  snapshot stream, resolves the run origin and run days, and holds the
  `validate_config` origin check; both runners call it at t = 0 and at every
  met-window end, forcing the per-window loop when sampling is on), and
  model setup. These files are included
  inside `DrivenRunner`; the top-level file retains the transport loops.
  Single-file NetCDF appends selected snapshots without retaining past frames;
  daily output owns and drains at most one background write. Input cleanup
  drains window prefetch before closing the driver and releasing mapped pages.
  Numerical state, flux arrays, and workspaces persist across input files;
  `DrivenSimulation` refreshes forcing, diffusion geometry, and caches.
  CS GPU setup transfers state first so workspace constructors allocate scratch
  directly on the device, avoiding temporary CPU workspaces. CS initialization
  converts one tracer at a time directly into its final packed state slot,
  using the shared VMR-to-storage conversion and preserving signed values,
  zero halos, and the existing tracer order. Analytic initializers reuse one
  private tuple of interior VMR panels across tracers. Native/file builders
  may replace that tuple; the public allocating builder always returns
  independent output arrays.
- [`initial_conditions/`](initial_conditions/) — cubed-sphere initialization,
  surface-inventory loading and storage-unit conversion, and conservative
  surface-flux remapping, included inside `InitialConditionIO`.
- [`InputStaging.jl`](InputStaging.jl) — opt-in rolling NVMe input staging
  (`InputStager`, `staged_path_for!`, `cleanup_staging!`) for the per-day
  binary loop: copies upcoming days NAS→local NVMe ahead of the GPU loop and
  evicts processed days, bounding local-disk use for multi-month/year runs.
  A directory has one active owner, and retained-copy reuse checks source
  identity metadata. Unavailable staging falls back to original source paths.
  Default off ⇒ bit-identical to a non-staged run

## Common Tasks

To follow a TOML physics option from input to execution:

| Stage | Read here | Responsibility |
|---|---|---|
| Parse the option | [`RuntimePhysicsSpecs.jl`](RuntimePhysicsSpecs.jl) | Convert section values into a typed specification; validate values and combinations. |
| Build an operator | `materialize` methods in the same file | Apply topology gates and construct the scheme; diffusion also needs driver context and tracer precision. |
| Check the forcing | [`RuntimePhysicsRecipe.jl`](RuntimePhysicsRecipe.jl) | Assemble operators and check required binary capabilities. |
| Allocate state and workspaces | [`runner/model_setup.jl`](runner/model_setup.jl) | Initialize tracer storage and build workspaces on its backend. |
| Advance the model | [`TransportModel.jl`](TransportModel.jl), [`DrivenSimulation.jl`](DrivenSimulation.jl) | Execute operator blocks and refresh forcing across meteorological windows. |

The public runner calls `validate_config` before the startup handoff into its
runtime implementation. Its checks live in
[`runner/configuration.jl`](runner/configuration.jl): table shapes first,
then path existence, precision/backend compatibility, and integer window
bounds. Tracer initialization and surface-flux values must be subtables.
The validator opens no binaries or model state; GPU auto-detection can still
probe optional runtimes. The CLI checks architecture shape before its separate
backend preload.

Matrix-convection solver eligibility is checked when the state backend,
precision, and vertical depth are known. Parsing `use_collab_lu=true` is a
request, not proof that a particular GPU kernel will run; see
[`../Operators/Convection/TM5Convection.jl`](../Operators/Convection/TM5Convection.jl)
for the support gates and fallback diagnostics.

- Changing operator block order:
  start in [`TransportModel.jl`](TransportModel.jl) and the runtime
  walkthrough in [`../../docs/20_RUNTIME_FLOW.md`](../../docs/20_RUNTIME_FLOW.md)
- Debugging "operator exists but never runs":
  check `TransportModel.step!` before editing operator code
- Debugging driver/model mismatch:
  start in [`DrivenSimulation.jl`](DrivenSimulation.jl), especially grid
  and basis compatibility checks
- Adding a new model-level runtime option:
  decide whether it belongs on `TransportModel`, `DrivenSimulation`, or
  both before threading it through the step loop

## Cross-Dependencies

- [`../State/README.md`](../State/README.md) provides the state and flux
  containers carried by the model
- [`../Operators/README.md`](../Operators/README.md) provides the actual
  physics blocks the model calls
- [`../MetDrivers/README.md`](../MetDrivers/README.md) provides the
  window-driven forcing and timing contracts
- [`../Grids/README.md`](../Grids/README.md) determines topology and
  therefore runtime dispatch

## Related Docs And Tests

- Runtime walkthrough:
  [`../../docs/20_RUNTIME_FLOW.md`](../../docs/20_RUNTIME_FLOW.md)
- Block-order design:
  [`TransportModel.jl`](TransportModel.jl) and
  [`../../docs/20_RUNTIME_FLOW.md`](../../docs/20_RUNTIME_FLOW.md)
- Tests:
  - [`../../test/core/test_driven_simulation.jl`](../../test/core/test_driven_simulation.jl)
  - [`../../test/core/test_no_advection.jl`](../../test/core/test_no_advection.jl)
  - [`../../test/orphan/test_transport_model_emissions.jl`](../../test/orphan/test_transport_model_emissions.jl)
  - [`../../test/orphan/test_current_time.jl`](../../test/orphan/test_current_time.jl)
