# AtmosTransport.jl - Project Guide

GPU-accelerated offline atmospheric transport model in Julia. Runs on lat-lon,
reduced-Gaussian and cubed-sphere grids driven by ERA5, GEOS (FP/IT) and
MERRA-2 meteorology, with mass-conserving advection, convection, and
boundary-layer diffusion. KernelAbstractions provides CPU/GPU portability.

## Working Rules

- Probe before building. For any transform or loader, print shape, dtype, and a
  physically meaningful value; compare against an independent reference when
  possible.
- Evidence, not speculation. If output looks unphysical, treat it as a bug
  until measurements prove otherwise.
- Persistent docs record facts, not hypotheses. The rendered manual under
  `docs/src/` is the current documentation; dated memos preserve history.
- For transport bugs: read the reference implementation first, find the exact
  semantic diff, and make one evidence-backed change at a time.
- Before committing, check whether the diff changes public behavior, scripts,
  file maps, workflows, or setup; if so, update the relevant `README.md` or
  reference docs.
- Before committing any non-trivial change, review the diff critically and get
  an independent review (Codex or a separate Claude agent, never the
  implementer itself); physics changes get a physics-focused review.
- Physics is documented first: equations, rationale, references and the
  differences from TM5/GCHP go into the manual (`docs/src/theory/`,
  `docs/src/for_tm5_gchp_users/`) before or with the code.
- Prefer production fixes. If a shortcut is unavoidable, mark it `# HACK:` with
  a TODO and the intended clean replacement.

## First Commands

```bash
julia --project=. scripts/run_transport.jl config/runs/<config>.toml
julia --project=. scripts/diagnostics/inspect_transport_binary.jl <path.bin>
julia --project=. -e 'using Pkg; Pkg.test()'
```

Use the binary inspector before debugging runtime behavior. It reports topology,
mass basis, supported operators, and load-time consistency gates.

## Code Map

- Entry point: `scripts/run_transport.jl` ->
  `run_driven_simulation(cfg)` in `src/Models/Runner/DrivenRunner/DrivenRunner.jl`
- Top-level include order follows `src/AtmosTransport.jl`:
  `ConfigChecks -> Architectures -> SectionTimer -> LogLevels -> AtomicFiles ->`
  `Quantities -> Parameters -> Grids -> State -> Output -> Observations ->`
  `MetDrivers -> Operators -> Tape -> Adjoints -> Regridding -> GriddedIO ->`
  `Preprocessing -> Models -> Diagnostics -> Extras`.
- Later modules may depend on earlier ones, never the reverse.
  `scripts/checks/check_structure.jl` (run with `--project=test`) enforces this
  against `src/LAYERS.toml`; a PR that moves, renames or splits a folder, module
  or source file declares it as rows in `test/structure/migrations.toml` and
  regenerates the code map (`--write-code-map`). See
  `docs/src/development/structure.md`.
- Submodules, in dependency order:
  - `Operators`: `Diffusion -> SurfaceFlux -> Convection -> Advection ->
    Chemistry -> StepPlans`. `StepPlans` composes a step and owns the
    emission/diffusion midpoint of the advection Strang palindrome; Advection
    only calls the abstract `apply_midpoint!` hook (`NoMidpoint` by default).
  - `Adjoints`: core -> `Footprint` -> `Inversion` (`Observations`, `Solver`).
  - `Preprocessing`, in pipeline order: `Common` (folder `Core/`) -> `Vertical`
    -> `Spectral` -> `MassClosure` -> `PhysicsInputs` -> `TransportBinary` ->
    `LatLonTarget` / `ReducedGaussianTarget` / `CubedSphereTarget` (folder
    `Targets/`) -> `GEOS`, `ERA5`, `MERRA2` (folder `Sources/`). The root owns
    only the config entry point, which dispatches to the `process_day` methods
    of the targets (ERA5 spectral) and the sources.
  - `Models`: `Physics -> Core -> InitialConditionIO -> Runner`.
- `src/Extras` (Visualization, Analysis, Downloads) is non-core: no tests,
  excluded from coverage.
- `GriddedIO` owns the conventions of native gridded input files (coordinate
  names, fill policy, lat-lon orientation, source meshes, time decoding, flux
  units, `cs_native` and lat-lon flux-file schemas); initial-condition and
  surface-flux loaders read files through it, never with private helpers.
- Each `src/` folder has a `README.md` (purpose, file map, Depends on, Used by,
  Shared helpers owned here, Forward<->adjoint mirrors; `Extras/Downloads/sources/`
  is covered by its parent's); read it before editing the folder.
- Physical constants live in `src/Parameters/PhysicalConstants.jl`; import
  them by name instead of writing literals.
- Active code lives in `src/`, tests in `test/`, tools in `scripts/`, configs
  in `config/`, and the maintained manual in `docs/src/`.
- File to focused test: `grep -rl <function> test/core`. Golden outputs of
  whole preprocessing and runtime cases are in `test/golden/` (see its README);
  check them before and after refactors.
- Physics should dispatch on types, not on ad-hoc grid or scheme conditionals.
- GPU extensions load when `using CUDA`, `using AMDGPU`, or `using Metal` happens before
  `using AtmosTransport`; `scripts/run_transport.jl` preloads the requested
  backend from `[architecture].backend`. Metal GPU runs require Float32.
- Transport binaries are dry-basis by default; preprocessing is responsible for
  carrier-mass conversion and continuity closure.

## High-Signal Invariants

- Cubed-sphere code uses the GEOS file panel convention. Panel connectivity and
  edge orientation live in `src/Grids/PanelConnectivity.jl`; the halo fills and
  seam-flux signs that use them are in `src/Operators/Advection/CubedSphere/`
  (`HaloExchange.jl`, `CubedSphereSeams.jl`).
- GEOS `MFXC` and `MFYC` are accumulated over the dynamics timestep, not the
  1-hour met interval. Use `mass_flux_dt_seconds = 450`.
- Wrong vertical level ordering causes extreme CFL and NaNs. Readers
  auto-detect GEOS-FP vs GEOS-IT ordering.
- Advection sweeps must ping-pong source and destination arrays. In-place sweep
  updates break mass conservation.
- FFTW `bfft` is the unnormalized inverse used for spectral synthesis. Do not
  divide by `N`.
- Julia is column-major: for `A[i, j, k]`, `i` must be the innermost loop.
- GEOS exports mixed moisture conventions: horizontal mass fluxes are dry,
  while `DELP`, `CMFMC`, and `DTRAIN` are moist.
- Preprocessing must Poisson-balance horizontal fluxes before diagnosing `cm`.
- Transport binaries must satisfy explicit-`dm` `cm` closure at write time. If
  replay or continuity checks fail, regenerate the binary.
- ERA5 spectral preprocessing pins global-mean `ps`; otherwise raw ERA5
  analysis drifts in total mass.
- Dry basis is the default runtime contract. Trace-gas VMRs are always dry
  VMRs, including column averages.
- If surface emissions disappear in column means, diffusion is probably
  disabled.
- If uniform tracers pump vertically under remap, check for hybrid-PE logic.
  The intended GCHP-matching path uses direct `cumsum` PE.

## Fast Failure Triage

| Symptom | First check |
|---|---|
| Transport is about 8x too slow | `mass_flux_dt_seconds = 450` |
| Extreme CFL or NaNs | vertical ordering, stale binary, or broken `cm` |
| About 10% mass loss per step | in-place sweep update |
| Panel-edge waves on cubed sphere | panel convention, flux rotation, or missing `linrood` |
| Surface emissions look invisible in column means | diffusion disabled |
| Uniform tracer jumps from 400 to about 535 ppm near the surface | hybrid PE in vertical remap |
| Day-boundary continuity or replay warnings | regenerate binary with current preprocessor |

## Extending The Code

- New advection scheme: add the type in `src/Operators/Advection/Reconstruction/schemes.jl`,
  implement its face fluxes in `src/Operators/Advection/Reconstruction/reconstruction.jl`
  (adjoint mirror: `src/Operators/Advection/CubedSphere/AdvectionAdjoint.jl`), wire it
  into the sweep path, and add uniform-field, mass-conservation, adjoint, and
  CPU/GPU tests.
- New met driver: subtype `AbstractMetDriver`, implement the required
  window/time methods in `src/MetDrivers/AbstractMetDriver.jl`, and add a
  source mapping in `config/met_sources/`.
- New operator: ship a `No<Operator>` default, keep the
  `apply!(state, meteo, grid, op, dt; workspace)` contract, wire it through
  `TransportModel` and `Operators.StepPlans` (operators applied at the
  palindrome midpoint go through its `TransportMidpoint`), and test that the
  default path is bit-exact to the explicit no-op path.

## Testing And Gotchas

- Tests should observe state through `get_tracer(state, name)` or
  `state.tracers.name`, not through input arrays cached before construction.
- Prefer focused test files around the touched module before running full
  `Pkg.test()`. Before pushing, run `scripts/dev/quick_gate.sh`: whitespace check,
  required-keyword call check, structure check, then the quick set plus the
  `--changed` topics (roughly 10 to 35 min, depending on the topics touched).
- Every PR runs the quick set (Julia 1; it includes the structure check) and the
  docs build. A PR with any path outside `docs/` that does not end in `.md` also
  runs the full suite on Julia 1.10 (LTS) and 1.13. A PR is mergeable only when
  its own `Full suite`, `Quick tests`, docs `build` and any reported Codecov
  statuses are green; a green `main` or nightly run does not count (see
  `test/README.md`, CI and merging).
- Refactors must reproduce the golden outputs bit for bit: `julia --project=. test/golden/run_goldens.jl check <ref> <new>`
  (see `test/golden/README.md`); a change that alters results states its deltas.
  A golden check writes about 25 GB; use a large local disk and point `TMPDIR`
  there as well.
- Adjoint and tape changes that are not meant to change gradients keep them
  bitwise identical on CPU (Float64 and Float32, one thread):
  `test/core/test_cs_adjoint_tape_bitwise.jl` runs
  `test/helpers/cs_adjoint_tape_harness.jl`; its committed digests are checked
  only on the reference host, so elsewhere A/B with `--record` on main and
  `--check` on the branch (see `test/README.md`). GPU gradients can differ in
  the last bits run to run (atomic adds), so bitwise checks run on CPU. mmap
  tape storage needs an explicit directory on a large local ext4/XFS disk.
- Imports between package modules are relative (`..X`, `...Preprocessing.Y`);
  never `using`/`import AtmosTransport.X` inside `src/` (tests that load a
  sandboxed copy of the package would bind the real one). Do not give a new
  module the name of a Base/Core/stdlib module or a package dependency: JET 0.9
  resolves a relative `..Core` to Julia's built-in `Core`. The one existing
  exception, `Models.Core`, is always imported by its full relative path
  (`...Models.Core`); inside `Models`, Julia's own module is `Base.Core`.
  `test/core/test_undefined_names.jl` catches names lost when code moves
  between modules.
- Do not add `Real -> FT` outer constructors unless there is a real ambiguity;
  Julia already converts through synthesized constructors.
- Parametric kwarg defaults like `Foo{FT}()` inside a `where FT` signature fail
  at module scope. Use `nothing` and construct inside the function body.
- For GPU-aware structs, keep array fields parametric and dispatch on
  `parent(arr)` or `KernelAbstractions.get_backend(arr)`, not `arr isa Array`.

## Detailed References

- Quick start: `docs/src/getting_started/quickstart.md`
- Architecture: `docs/src/concepts/architecture.md`
- Grid and data conventions: `docs/src/concepts/grids.md`,
  `docs/src/preprocessing/conventions.md`, `docs/src/config/data_sources.md`
- Advection details: `docs/src/theory/advection_schemes.md`
- Preprocessing and binary formats: `docs/src/preprocessing/overview.md`,
  `docs/src/concepts/binary_format.md`
- TM5/GCHP mappings: `docs/src/for_tm5_gchp_users/`
- Code structure, migration rows and the code map: `docs/src/development/structure.md`,
  `docs/src/development/code_map.md`
- Test tiers, the adjoint bitwise harness and CI policy: `test/README.md`
- Historical design and debugging records: `docs/memos/`, `docs/resources/`
