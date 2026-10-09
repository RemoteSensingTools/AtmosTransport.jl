# Code structure, duplication and constants: audit and plan (2026-10-08)

Four read-only audits of `src/`, `scripts/` and `test/` at `cac6d436`
(Preprocessing; Operators and Models; the remaining modules and layering;
duplication, from a clone detector plus manual reading). Goals set by the
project owner: logical ordering, no oversized files, helpers in their own
files, dispatch on types, code readable and documented for scientists, little
duplication, structure that AI agents can navigate, and all physical
constants in `PlanetParameters`.

## Size and duplication

- `src/`: 82,477 lines in 222 files. 42 files exceed 600 lines and 22 exceed
  800; the largest are `Preprocessing/transport_binary/cubed_sphere_geos.jl`
  (2,298), `Operators/Advection/linrood_adjoint_kernels.jl` (1,761),
  `Preprocessing/reduced_transport_helpers.jl` (1,759),
  `Operators/Advection/StrangSplitting.jl` (1,753),
  `Preprocessing/sources/era5.jl` (1,725) and
  `Operators/Advection/CubedSphereStrang.jl` (1,624).
- Clone detector (scratchpad `clones/find_clones.py`): 5,465 duplicated lines
  in exact blocks of six or more code lines, 7,403 lines in structural blocks
  of ten or more (identifiers and numbers normalised). `scripts/`: 6,776 lines
  in exact blocks of eight or more.
- The dominant pattern is one algorithm written per grid (lat-lon, reduced
  Gaussian, cubed sphere), per axis (x, y, z), for single and packed tracers,
  and forward and adjoint. Estimated removable: about 7,000 `src/` lines and
  14,000 `scripts/` lines.

## A. Copies that have diverged (fix first, one commit each, with the measured effect)

1. Reduced-Gaussian Poisson balance target has the opposite sign to the
   cubed-sphere one (`ring_poisson_balance.jl:239` vs
   `cs_poisson_balance.jl:690`); a 4-cell probe gave a continuity error of
   0.17. To confirm on a real reduced-Gaussian binary; no test covers it.
2. The Float32 anomaly fix of the diffusion kernels (`cref`, 9ddf3d68) is only
   in the cubed-sphere kernels (`diffusion_kernels.jl:157,250,316,408`), not
   in the lat-lon (`:47`) or reduced-Gaussian (`:495,:573`) ones.
3. The CMFMC adjoint carries a forward copy that must "stay in lock-step"
   with `cmfmc_kernels.jl:301` but lacks its Kahan sums, `cloud_base` and
   clamp (`Adjoints/ConvectionAdjoint.jl:288-388`).
4. The Lin-Rood adjoint face value `_ppm_face_value_d6`
   (`linrood_adjoint_kernels.jl:380`) does not clamp the Courant fraction as
   the forward `LinRood.jl:237-246` does; the adjoint is not the exact
   transpose for CFL > 1.
5. Two cubed-sphere section tables; the writer's
   (`MetDrivers/transport_binary/cubed_sphere.jl:9`) lacks `:dam/:dbm/:dcm`
   that the reader's (`cubed_sphere_reader.jl:21`) has; nine script copies
   lack `:dkg`.
6. Two `_potential_temperature` methods in the same module:
   `GCHPNonlocalPBLField.jl:86` (R/cp from parameters) and
   `LocalHoltslagBovilleKzField.jl:110` (κ = 1/3.5 hard-coded).
7. Physical constants disagree (see section B).
8. The reduced-Gaussian Strang path ignores `DiffusiveSurfaceFluxBoundary`
   (`StrangSplitting.jl:1533`); only the config parser rejects it.
9. Inversion assigns observations to the nearest cell centre
   (`Inversion/ObservationBinding.jl:165-226`); forward sampling uses the
   exact containing cell (`Output/.../cell_locator.jl:158`). They can differ
   near cell edges and panel seams.
10. The structured `PPMScheme` flux is linear in the Courant number on PPM edge
    values, but its comments claim it matches Putman and Lin (2007)
    (`reconstruction.jl:520-535`); the cubed-sphere Lin-Rood path uses the
    full parabolic integral.
11. Preprocessing has seven adaptive-substep loops with different guards
    (column weights missing in three cubed-sphere paths; Poisson `max_iter`
    20,000 at `cubed_sphere_regrid.jl:426` but 5,000 at `:530`; the
    reduced-Gaussian path never uses the humidity-aware pin).

## B. Physical constants: all in `PlanetParameters`

Literals found outside `PlanetParameters`, with disagreeing values:

| constant | values in the code |
|---|---|
| dry-air molar mass | 28.9644e-3 (`initial_conditions/cubed_sphere.jl:314`, GEOS-Chem script), 28.96546e-3 (`initial_conditions/surface_flux.jl:25`) |
| gravity | 9.81 (`Diffusion/dz_helpers.jl:37`), 9.80665 (`PlanetParameters.jl:31`) |
| dry-air gas constant | 287.04, 287.058 |
| Earth radius | 6.371229e6 (`Preprocessing/constants.jl:6`, says it matches `PlanetParameters`), 6.371e6 (`PlanetParameters`) |
| κ | 1/3.5 literal (`LocalHoltslagBovilleKzField.jl:110`) |

Plan: (1) move every literal into `PlanetParameters` (named, documented, with
its source), keeping each call site's current value so results stay
bit-identical; (2) reconcile the disagreeing values in a separate commit and
report the change in a 1-day run and the true mass balance.

## C. Structure

- Layering follows the documented include order, but `Footprint/` and
  `Inversion/` are not modules (included into `Adjoints`), Adjoints imports 52
  private names from Operators and Tape, and Tape exports 17 underscore names.
- Mixed files to split by responsibility (target ≤ ~600 lines): the two Strang
  files, `LinRood.jl`, `linrood_adjoint_kernels.jl` (to `Adjoints/`),
  `DrivenSimulation.jl`, `DrivenRunner.jl`, `cubed_sphere_geos.jl`,
  `sources/era5.jl`, `sources/geos.jl`, `cs_poisson_balance.jl`,
  `cs_transport_helpers.jl`, `reduced_transport_helpers.jl`,
  `StrideCheckpoint.jl`, `CubedSphereMesh.jl`.
- Helpers in the wrong place: halo and seam code in Advection but used by
  Models, Adjoints, Footprint; generic cubed-sphere mass helpers in the GEOS
  driver; surface-flux loading under `initial_conditions/`; PBL physics in
  `State/Fields`; the transport-binary format inside MetDrivers; observation
  readers inside Output.
- Symbol, Bool and ENV switches that should be types: the GEOS `cm_closure`
  (six-way switch in the hot path), `mass_basis`, level order, surface-flux
  dataset `kind` (25 sites, already diverged unit conversions), air-mass reset
  mode, forcing sampling, tape storage, ENV reads per step.
- Per-source preprocessing drivers each run their own day loop (ERA5 N320,
  MERRA-2) instead of `run_unified_preprocessor_day!`; the CS window close
  (balance, mirror sync, cm) appears at eight sites.
- Dead code: about 1,000 lines (reduced-Gaussian CG chain, LL subcycled
  sweeps, q-space Lin-Rood helpers, `MetState`, unused cm diagnosers).

## D. Duplication to consolidate (largest first)

| step | abstraction | lines |
|---|---|---|
| column layouts for TM5, CMFMC and diffusion kernels | one kernel per algorithm, an isbits layout type (`LLColumns`, `RGColumns`, `CSPanelColumns`) for index decoding | −1,400 |
| axis-generic advection sweeps and subcycling | `Val{D}` methods instead of x/y/z copies and `@eval` | −700 |
| footprint checkpoint drivers | one recorder per scheme, one stride and one revolve driver | −1,000 |
| preprocessing window close, substep loop, graph CG, section table, readers/writers | shared functions with explicit options | −1,300 |
| face fluxes and the Lin-Rood adjoint | axis-generic core; forward helpers generic over the dual type | −750 |
| scripts | archive one-offs, merge plot families, `scripts/lib/`, use `TransportBinaryReader` | −14,000 |

## E. Agent and scientist navigation

- A README per folder: purpose, entry points, "to change X, edit Y",
  invariants (ping-pong buffers, dry basis, k = 1 at the top).
- `CLAUDE.md` code map including Footprint, Inversion, SectionTimer, and the
  rule "file → focused test".
- Tests mirroring `src/` (`runtests.jl` to `walkdir`); scripts grouped by
  purpose (`campaigns/`, `tools/`, `analysis/`, `archive/`), no hard-coded
  machine paths.
- Unique, grep-able names (four `operators.jl`, eight `process_day` methods
  including an unrelated CLI driver, `@eval`-generated sweep names).
- Equation-level docstrings for the undocumented physics (Poisson balance
  internals, cm closures, Lin-Rood seams, CMFMC), theory pages for convection
  and diffusion.

## F. Order and verification

0. Golden harness: 1-day lat-lon and C90 runs (CPU Float64, GPU Float32) and
   preprocessing goldens (GEOS synthetic per cm closure, unified LL/RG/CS
   tests, one real N320 and MERRA-2 day); compare snapshots and binaries with
   `==`, masking provenance header keys.
1. Section A items that are bugs, each with a test and its measured effect.
2. Constants into `PlanetParameters` (bit-identical), then reconcile values.
3. Dead code, stale references, READMEs, `CLAUDE.md` map (no numerical change).
4. Move-only file splits (`git mv`, new include lists).
5. Types for Symbol/ENV switches; operator-owned requirement traits.
6. Duplication consolidation (section D), each bit-identical against the goldens.
7. Unified wind-derived preprocessing pipeline; tests mirroring `src/`; scripts.

Each step is its own commit, reviewed (Julia style, bug check, Fable) before
committing; the production snapshots used by running jobs are unaffected.
