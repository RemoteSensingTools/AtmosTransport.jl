# AtmosTransport.jl - Project Guide

GPU-accelerated offline atmospheric transport model in Julia. Supports ERA5 /
MERRA latitude-longitude runs and GEOS cubed-sphere runs with mass-conserving
advection, convection, and boundary-layer diffusion. KernelAbstractions
provides CPU/GPU portability.

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
  a Codex review.
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
  `run_driven_simulation(cfg)` in `src/Models/DrivenRunner.jl`
- Top-level include order follows `src/AtmosTransport.jl`:
  `Architectures -> Quantities -> Parameters -> Grids -> State -> Output ->`
  `MetDrivers -> Operators -> Tape -> Adjoints -> Regridding -> Preprocessing ->`
  `Visualization -> Models -> Downloads`.
- Later modules may depend on earlier ones, never the reverse.
- Active code lives in `src/`, tests in `test/`, tools in `scripts/`, configs
  in `config/`, and the maintained manual in `docs/src/`.
- Physics should dispatch on types, not on ad-hoc grid or scheme conditionals.
- GPU extensions load when `using CUDA` or `using Metal` happens before
  `using AtmosTransport`; `scripts/run_transport.jl` preloads the requested
  backend from `[architecture].backend`. Metal GPU runs require Float32.
- Transport binaries are dry-basis by default; preprocessing is responsible for
  carrier-mass conversion and continuity closure.

## High-Signal Invariants

- Cubed-sphere code uses the GEOS file panel convention. Panel-edge flux
  rotation lives in `panel_connectivity.jl`.
- GEOS `MFXC` and `MFYC` are accumulated over the dynamics timestep, not the
  1-hour met interval. Use `mass_flux_dt = 450`.
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
| Transport is about 8x too slow | `mass_flux_dt = 450` |
| Extreme CFL or NaNs | vertical ordering, stale binary, or broken `cm` |
| About 10% mass loss per step | in-place sweep update |
| Panel-edge waves on cubed sphere | panel convention, flux rotation, or missing `linrood` |
| Surface emissions look invisible in column means | diffusion disabled |
| Uniform tracer jumps from 400 to about 535 ppm near the surface | hybrid PE in vertical remap |
| Day-boundary continuity or replay warnings | regenerate binary with current preprocessor |

## Extending The Code

- New advection scheme: add the type in `src/Operators/Advection/schemes.jl`,
  implement the interface in `src/Operators/Advection/Advection.jl`, wire it
  into the sweep path, and add uniform-field, mass-conservation, adjoint, and
  CPU/GPU tests.
- New met driver: subtype `AbstractMetDriver`, implement the required
  window/time methods in `src/MetDrivers/AbstractMetDriver.jl`, and add a
  source mapping in `config/met_sources/`.
- New operator: ship a `No<Operator>` default, keep the
  `apply!(state, meteo, grid, op, dt; workspace)` contract, wire it through
  `TransportModel`, and test that the default path is bit-exact to the explicit
  no-op path.

## Testing And Gotchas

- Tests should observe state through `get_tracer(state, name)` or
  `state.tracers.name`, not through input arrays cached before construction.
- Prefer focused test files around the touched module before running full
  `Pkg.test()`.
- Refactors must reproduce the golden outputs bit for bit: `julia --project=. test/golden/run_goldens.jl check <ref> <new>`
  (see `test/golden/README.md`); a change that alters results states its deltas.
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
- Historical design and debugging records: `docs/memos/`, `docs/resources/`
