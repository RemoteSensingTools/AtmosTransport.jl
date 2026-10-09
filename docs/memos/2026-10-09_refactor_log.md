# Refactor log (from 2026-10-08)

Work log of the phased refactor planned in
`2026-10-08_code_structure_and_duplication_plan.md`, branch
`refactor/structure-2026-10` (off `3684b71a`). Every commit is reviewed by Codex
and checked against the golden outputs (`test/golden/`, reference
`/temp1/cfranken/goldens/ref_3684b71a` on wurst); results-changing commits state
their deltas here and in the commit message.

## Phase 0: golden harness

- `test/golden/`: 24 cases (5 preprocessing, 19 runtime; 2 reduced-Gaussian
  cases are known failures), compared bit for bit with provenance masked.
- Runtime surface-flux regridding weights can be redirected with
  `ATMOSTR_REGRID_CACHE_DIR` (default unchanged), so golden cases recompute them.
- Reference recorded at `3684b71a` (14 cases); negative control: a single
  changed value in a binary payload or a NetCDF variable is reported.

## Findings while building the harness

- Reduced-Gaussian preprocessing is broken: the write-time replay gate fails
  (continuity error 6e-3) because the Poisson balance target has the wrong sign
  (plan A1).
- Structured PPM is not positive-definite: after one day, 45 % of the cells of a
  fossil tracer are negative on a 5° lat-lon grid (−2 % of the positive mass);
  in the production C90 MERRA-2 run, 31–41 % of fossil-CO₂ cells and 2 % of
  Rn-222 cells are negative, −1e-5 of the mass (plan A10).

### A10 evidence: structured PPM is not positivity-preserving

Golden lat-lon 72 × 37 day (2021-12-01), fraction of negative cells after 24 h
and negative mass relative to positive mass:

| advection | fossil CO₂ | blob (uniform background 0) | Rn-222 |
|---|---|---|---|
| upwind (CPU F32) | 1.1 %, −3e-4 | 0 | 0 |
| slopes (GPU F32) | 0.2 %, −4e-4 | 0 | 0 |
| PPM (CPU F64) | 45 %, −2.0e-2 | 33 %, −3.7e-4 | 42 %, −4.5e-3 |

The small fossil negatives of upwind and slopes come from negative GridFED
cells. PPM's face mixing ratio `c + (1 − α)(q_R − c)` (the slopes flux with a
PPM edge offset) is bounded by the monotone edge values but drops the curvature
term of the swept-region mean, `c + (1 − α)(br − α·b0)` (FV3 `xppm`, our
`_ppm_face_value`), so a cell next to a steep gradient can export more tracer
than it holds. Production C90 MERRA-2 PPM run (2022-01-15): 31–41 % of fossil
cells and 2 % of Rn-222 cells negative, −1e-5 of the mass.

## Phase 1: diverged copies (bugs)

### A1 — reduced-Gaussian Poisson balance sign (commit after 5dde1005)

Both reduced-Gaussian balance implementations (`balance_compressed_horizontal_fluxes!`,
`balance_reduced_horizontal_fluxes!`) targeted an outflow divergence of
`(m_next − m_cur)/(2·steps)` instead of `(m_cur − m_next)/(2·steps)`, and their own
post-balance diagnostics reused the wrong sign, so only the independent write-time
replay gate caught it. One helper `_target_outflow` now serves all six sites; a new
test (`test/core/test_rg_poisson_balance.jl`) checks both implementations against the
continuity target (it failed with a residual of twice the target before the fix).
On the O24 golden day the replay error drops from 6.0e-3 to 2.0e-7; the remainder is
item A11 (no humidity-aware dry-mass pin in the reduced-Gaussian path), so the O24
goldens stay known failures. No other golden is affected (reduced-Gaussian only).

### Phase 0 accepted

Full repeat run of the 14 reference cases (`det_3684b71a`): every case identical,
including the runtime cases, which now recompute their regridding weights (the
reference used `~/.cache`). The 8 variant cases were recorded into the reference.

### A4 — Lin-Rood adjoint Courant-fraction clamp

The adjoint evaluated its own copy of the forward PPM face value without the
forward's clamp `|α| ≤ 1` and differentiated `α = F/m` through the donor mass where
the forward clamps it. `_courant_fraction(F, m)` (LinRood.jl) now returns the clamped
`α` (as `clamp`, so NaN stays NaN) and `∂α/∂m` (zero where clamped); the forward
`_ppm_face_value` is generic in the mixing-ratio type, so the adjoint evaluates it on
its dual numbers and the copy is gone. New tests: finite differences at Courant
numbers > 1 (ORD 5 and 7, Float64; the old adjoint gave −11.7 against 0.33) and
Float32 against Float64. Forward goldens (three Lin-Rood cases) identical.

### A2 — anomaly diffusion for lat-lon and reduced-Gaussian columns

The lat-lon packed and reduced-Gaussian packed and single diffusion kernels now
solve for the departure from each column's minimum, as the cubed-sphere kernels
do (the backward-Euler operator keeps a uniform column). Lat-lon and
reduced-Gaussian `DiffusionWorkspace`s carry per-tracer references; the packed
wrappers and the Strang preflight check them before touching the state.
Golden deltas: lat-lon Float32 runs, uniform-tracer mass drift +1.37e-6 → +5e-9
per day (blob +1.3e-6 → +7e-7, the level of the cubed-sphere runs); lat-lon
Float64 at rounding level (2e-14 relative); cubed-sphere runs identical.
Unit test: Float32 column-mass change over a day 8.0e-7 → 4.9e-8.

### A9 — inversion binds observations to the containing cell

`bind_to_mesh` and the alignment check of `build_departure_set` used the nearest
cell centre (a brute-force search), while the forward observation sampler uses
the containing cell (`Output.cell_locator`/`locate`); on C6 the two disagree for
905 of 20,000 random points. Both now use the locator (longitudes of any wrap
reduced to [0, 360)), and the nearest-centre cache is deleted. Tests: binding
equals `locate` for 2000 random points, wrapped longitudes, the departure check
accepts the bound observations and rejects a shifted cell. Open: records store
Float32 coordinates, so a point within Float32 rounding of an edge can bind to
the other side than the Float64 coordinate the forward run sampled.

### A3 — CMFMC adjoint replays the production forward

The cubed-sphere CMFMC adjoint replayed the forward with its own copy of the
production kernel, which lacked the Kahan-compensated sub-cloud sums (so the
replayed trajectory differed from the run's at rounding level), and kept its own
copies of the `tiny` threshold, the cloud-base search and the derived
detrainment. The replay now calls the production kernel on a one-tracer view;
the adjoint kernel uses the production helpers, the detrainment array of
`_cmfmc_dtrain_array` and compensated sums. (Without DTRAIN both sides derive
it from CMFMC, so there was no operator mismatch there.) Tests: replay equals
the production `apply_convection!` bit for bit on a deep uneven column (fails on
the old code), and the adjoint identity on that column with and without DTRAIN
and several substeps. Forward results unchanged.

### A5 — one cubed-sphere section table

The writer and the reader had separate section-size tables (the writer's lacked
the flux-delta sections `dam/dbm/dcm`); the reader's method now delegates to the
writer's function, which covers every section. Goldens identical (C24 and C90
preprocessing, three runtime cases); new table-driven test. The script copies of
the table (nine scripts, some without `dkg`) go with the scripts cleanup.

### A8 — reduced-Gaussian runs reject the diffusive surface-flux boundary

The reduced-Gaussian Strang palindrome couples emissions only as
V(dt/2) → S(dt) → V(dt/2); given a `DiffusiveSurfaceFluxBoundary` it silently
used that split (only the config validator rejected it). `apply!` now throws
before touching the state; test in `test_no_advection.jl`.

### A11 — reduced-Gaussian window ends: dry-mass pin, fluxes, next-day end point

The vertical-flux closure can absorb only a zero change of the global dry mass
between the two ends of a window. The lat-lon and cubed-sphere preprocessors
pin the global dry surface pressure with the native humidity; the
reduced-Gaussian path pinned the total surface pressure with the climatological
humidity (or, in its v2 configs, not at all), so the dry mass changed from
window to window and the write-time replay gate failed at 2e-7 (gate 1e-10)
after A1. Three more differences from the lat-lon path turned up on review:

- the pin moved the surface pressure but not the horizontal fluxes computed
  from it (lat-lon recomputes them); the fluxes, wind × Δp at the face, are now
  rescaled by Δp_new / Δp_old;
- the day's last window ended at its own mass instead of the next day's
  00 UTC state, so it carried zero mass tendency; the next day's state is now
  synthesized, pinned, converted and merged like every window (as in lat-lon);
- the synthesis indexed the hybrid coefficients by native level instead of
  selected level (only correct when `level_top = 1`, as in every current config).

The window synthesis now loads the humidity first; `pin_convert_merge_window!`
finishes every window and the next-day end point. `pin_global_mean_ps!` and
`pin_global_mean_ps_using_qv!` take any column layout (lat-lon `(Nx, Ny)`,
reduced-Gaussian `(ncell,)`), bit-identical for lat-lon; the reduced-Gaussian
copy of the climatological pin and the dead reduced-Gaussian `process_window!`
are gone. The three reduced-Gaussian v2 configs that disabled the mass fix ("not
yet wired through the RG preprocessor") enable it. On the O24 golden day the
replay error is 4.2e-15 with the pin alone; `pre_o24` and
`run_o24_upwind_cpu_f64` are recorded and lose the `known_failure` tag. Tests:
`test/core/test_global_ps_pin.jl` (both pins hit their targets, the two layouts
agree bit for bit, the flux rescale).

### A10 — structured PPM: analysis and proposed fix (not applied; decision for the owner)

The structured `PPMScheme` (lat-lon, and per panel on the cubed sphere, i.e. the
production CATRINE "ppm" runs) is not positivity-preserving, for two reasons:

1. **Edge values are not monotonized.** `_ppm_edge_value` is the plain
   fourth-order CW84 interpolation `7/12 (c_i + c_{i+1}) − 1/12 (c_{i−1} + c_{i+2})`,
   without CW84's monotonized slopes (Colella & Woodward 1984, eqs. 1.7–1.8), so
   next to a spike it returns −1/12 of the spike. `_ppm_limit_profile`
   (`MonotoneLimiter`) only removes overshoots of the parabola inside the cell;
   an edge outside the range of its two neighbours survives.
2. **The flux drops the curvature term** of the swept-region mean
   (`c + (1 − α)(q_R − c)` instead of FV3's `c + (1 − α)(br − α·b0)`).

Measured on a periodic 1-D ring (spike and box on a zero background, 100 steps,
Courant 0.3/0.7/1.0, `PPMScheme()`): minimum −0.013 / −0.038 now; with the
curvature term alone −0.010 / −0.026. Both fixes are needed for a monotone
scheme. The curvature term with matching forward and adjoint changes
(`_ppm_swept_moments`, one 6-point dual-number coefficient function for both
limiters replacing the 4-point NoLimiter coefficients) passes every existing
adjoint test (footprint 91/91, model-space 17/17, preconditioned 21/21) and is
saved as `/temp1/cfranken/goldens/patches/a10_ppm_curvature_partial.patch`
(+ `test_ppm_positivity.jl`). Still to do: monotonized edge slopes for
`MonotoneLimiter` (forward and dual-number adjoint), then golden deltas and a
CATRINE comparison against GCHP (which uses FV3's limited PPM, so the fix
should bring us closer). Every production PPM run changes, hence left for the
owner's decision.

## Phase 2: constants

### Step 1 — physical constants in one place (ab32a2bb)

`src/Parameters/PhysicalConstants.jl` holds the constants of nature with their
sources (Earth radii of the meshes and of the IFS, standard gravity and pressure,
dry-air heat capacity, Avogadro, virtual-temperature factor, species molar
masses) and the constant sets of the schemes that reproduce other models
(`TM5_CONSTANTS`, `GEOSCHEM_CONSTANTS`). `Preprocessing/constants.jl` is gone.
Every call site kept its value.

### Step 2 — one dry-air constant set (GEOS/MAPL)

`R_DRY_AIR = 287.04`, `CP_DRY_AIR = 1004.64` (= 3.5 × 287.04),
`DRY_AIR_MOLAR_MASS = 28.9644e-3`, `CP_OVER_R_DIATOMIC = 3.5`,
`THETA_REFERENCE_PRESSURE = 1e5` replace the module-local copies. Results
change where the copies disagreed:

| site | before | after | effect |
|---|---|---|---|
| diffusion layer thickness (`dz_helpers.jl`) | g = 9.81 | 9.80665 | dz +3.4e-4 (relative) in every Kz-diffusion run |
| surface-flux storage scale (`surface_flux.jl`) | M_air = 28.96546e-3 | 28.9644e-3 | emitted tracer −3.7e-5 (relative) |
| TM5 convection conversion (preprocessing) | R = 287.058, ε = 0.608 | 287.04, 0.61 | dz of future TM5 attachments −6e-5, +2e-3·q |

The cubed-sphere pressure-layer IC, the observation sampler, the Kz fields
(`cp_dry / 3.5`) and the GEOS-Chem θ reference pressure keep their values. The
two `_potential_temperature` methods with different semantics (plan A6) now
share one core `_potential_temperature(T, p, κ, p_ref)`; the GEOS-Chem and
local Holtslag-Boville fields call it through `_gchp_theta` and
`_local_hb_theta` (bit-identical). Tests: constant values and coherence, the
diffusion dz and TM5 dz against their formulas in both precisions.

Golden check of step 1 (`chk_p2a`, all 18 non-slow cases): identical.

### Step 2c — binaries record their mesh radius

Every preprocessor builds its target mesh with the IFS radius 6 371 229 m and
computes air masses `m = Δp A / g` on it, but the runtime rebuilt the mesh from
the binary with the default 6 371 000 m, so runtime cell areas were 7.2e-5 too
small for the air masses: emissions from flux densities (`density × area`), the
pressure thickness of the GEOS-Chem non-local PBL and local Holtslag-Boville
fields (`m g / A`), observation-sampler pressures, `cs_native` fluxes and the
per-area outputs were off by that factor. Binaries now record
`planet_radius_m`; `load_grid` builds the mesh and `AtmosGrid` with it, and the
runtime IC and surface-flux regridders put the source lat-lon mesh on the same
sphere. Binaries without the key read as 6 371 000 m, so existing binaries give
identical results; new binaries change the quantities above by 7.2e-5.

- Writers: the lat-lon and reduced-Gaussian writers take the radius from the
  grid; the cubed-sphere writer requires `planet_radius` (the six preprocessing
  paths pass their target mesh's, the coarsener and the TM5 attachment script
  keep the source binary's). The key is structural (`extra_header` cannot change
  it) and part of the output-reuse contract.
- The contract validator rejects a recorded radius that is not a positive length.
- Inventories given as per-cell totals (EDGAR tonnes without an area variable)
  are converted to densities on the destination mesh's sphere, so their global
  total does not depend on the radius. Inventories given as densities keep the
  density, so their totals scale with the sphere (+7.2e-5 on new binaries; the
  ECCO-Darwin regridding script bins on 6 371 000 m and is unchanged).
- The lat-lon → cubed-sphere binary regridder uses the source binary's radius
  (binaries without the key: the IFS radius they were all built with) and
  requires the target to share it.
- ATMSNAP snapshots record the radius; `binary_to_netcdf.jl`,
  `extract_cs_column_means.jl`, `extract_cs_xco2*.jl` and
  `ocean_xco2_monthly_range.jl` rebuild the mesh with it (older snapshots:
  6 371 000 m).
- Scripts: the TM5 benchmarks and the ERA5/GEOS-IT met comparison use the
  binary's radius. Other diagnostics that rebuild meshes are radius-invariant or
  assume the IFS radius; they go with the scripts cleanup. Leftover from step 1:
  one test and two diagnostics still used the removed `Preprocessing.GRAV`.
- Test `test/core/test_binary_planet_radius.jl`: lat-lon, reduced-Gaussian and
  cubed-sphere round trips in both precisions, headers rewritten without the
  key, the structural guard, ATMSNAP, the regridding source mesh, and an EDGAR
  total on two spheres.

## Phase 3: dead code, navigation

### Step 1 — dead code (no numerical change)

Definitions with no caller in `src/`, `test/`, `scripts/` or `ext/` (found by a
token scan that ignores comments and docstrings, repeated until nothing new
appeared), about 800 lines:

- reduced-Gaussian whole-day balance chain, superseded by the streaming
  two-window buffer: `ReducedWindowStorage`, `allocate_reduced_window_storage`,
  `store_reduced_window!`, `apply_reduced_poisson_balance!`,
  `verify_storage_continuity_rg!`, the face-indexed graph-Laplacian CG
  (`balance_reduced_horizontal_fluxes!`, `solve_graph_poisson_pcg!`,
  `_graph_laplacian_mul!`, `cell_face_degree` and the workspace's
  `face_degree`); the compressed-Laplacian solver is the one in use, and the
  A1 test keeps only it;
- lat-lon subcycled sweeps (`_sweep_{x,y,z}_subcycled!`,
  `_sweep_{x,y,z}_pp_subcycled!`);
- q-space Lin-Rood helpers (`rm_to_q_panels!`, `q_to_rm_panels!`,
  `compute_dp_from_m_panels!`, `set_m_from_dp_panels!` and their kernels);
- the per-direction CS CFL counter `_cs_static_subcycle_count` (the runtime
  uses `_cs_static_palindrome_subcycle_count`; its outflow formula moved into
  that docstring, and the comments and theory page that named the dead one
  now name the live one);
- small leftovers: `_CSTapeCounts`, `pack_panels_3d_to_flat!`,
  `copy_panel_tuple`, `_transport_is_cubed_sphere`, `_active_substep`,
  `_cs_coarsen_npanel`, `_date_range_from_str`, `SECONDS_PER_MONTH`, the
  unused ERA5 GRIB parameter-id constants.

Left for the owner (public API, used only by tests): `State.MetState`, and the
exported `diagnose_cm_from_continuity_vc!` / `diagnose_cm_from_continuity_ka!`
(no caller; `diagnose_cm_from_continuity!` is tested). Stale comments fixed:
the RG contract header (the positivity gate is wired into `process_day`).

### Fixes found while writing the READMEs

- `IdentityRegrid` (returned by `build_regridder` for equivalent meshes) had
  no `src_areas`/`dst_areas`, which the surface-flux and preprocessing
  density conversions read from every regridder: a surface-flux file on
  exactly the run's lat-lon grid (as the loader orients it, longitudes in
  [0, 360)) would have thrown. It now carries the mesh cell areas in the
  regridders' flat cell order. Test in `test_identity_regrid.jl`.
- The Makie extension converted the runtime's logged storage rate back to
  species mass with the old dry-air molar mass (step 2 changed the runtime's);
  it now uses `DRY_AIR_MOLAR_MASS`, `SPECIES_MOLAR_MASS`, `STANDARD_GRAVITY`.
- The `cs_surface_flux_jacobian` docstring was attached to the helper above it.

### Step 2 — folder READMEs, CLAUDE.md code map

Every `src/` folder now has a README (purpose, entry points, file map, common
tasks, invariants, focused tests): new for Adjoints, Diagnostics, Downloads,
Footprint, Inversion, MetDrivers/transport_binary, Models/initial_conditions,
Models/runner, Output/observations, Parameters, Preprocessing (and its
`sources/`, `transport_binary/`), Quantities, Regridding, Tape, Visualization.
They were written from the code and fact-checked by Codex. `CLAUDE.md` gains
`SectionTimer` in the include order, that `Footprint/` and `Inversion/` are
included into `Adjoints`, the READMEs, the constants rule, and the file → test
and golden-harness pointers.

### Findings from writing the READMEs (code reading, not yet acted on)

Behavior worth a decision:
- Visualization rebuilds cubed-sphere meshes from `Nc` and the panel
  convention only, ignoring the recorded definition and laws: snapshots on a
  non-default definition would be rasterized on the wrong geometry.
- ERA5 N320 source traits `has_surface`/`has_convection` return `false` even
  when the N320 writer emits those sections; its `surface_path` is checked
  when a day is opened but never read.
- `[mass_fix] mode = "initial_endpoint"` silently skips the pin for ERA5 N320
  and MERRA-2 (target NaN); GEOS initializes the target from the first window.
- The GEOS path ignores `ATMOSTR_NO_WRITE_REPLAY_CHECK`; the RG path writes
  to the final file name instead of a staged `.tmp`.
- The positivity gates differ: CS checks `2(out_x + out_y + out_z)` against
  `min(m, m_next)`, lat-lon and RG each direction against `m`, so one
  `positivity_cfl_limit` means different things.
- `cs_tape_byte_estimate` counts the split-sweep tape layout also for Lin-Rood,
  whose records are larger.
- Downloads: OPeNDAP is not implemented (the MERRA-2 recipe's default),
  `max_concurrent` is unused, HTTP/GCS ignore `retry_wait`, the default daily
  ERA5 file name repeats within a month when no template is set, and
  `data_root` does not expand `$ATMOSTRANSPORT_DATA_ROOT`.
- Lat-lon and RG surface fluxes default to bilinear sampling, CS to
  conservative regridding; streaming writers default to `mass_basis = :moist`,
  `write_transport_binary` to `:dry` (all in-tree callers pass it).

Public API with no production caller: `State.MetState`,
`diagnose_cm_from_continuity_vc!`/`_ka!`, `reset_workspace!` (exported, no
methods), `ERA5SpectralReader`/`end_of_day_seed` (tests only),
`SurfaceLapseTemperature` (tests only), the single-surface perturbation
helpers in `Adjoints`.

Duplication for Phase 6: the block-sum coarsening helpers exist in both
`cubed_sphere_geos.jl` and `cubed_sphere_coarsen.jl`; the ERA5 lat-lon header
is built in `core.jl` beside `_transport_common_header`; the static and
time-varying surface-flux loaders repeat reorientation and unit conversion;
the own-loop CS writers repeat balance → cm → verify → promote.

Stale documentation (module docstrings, history comments, citations of
removed files and "invariant N" numbers) is listed in the README reports and
goes with Phase 7.

Golden check at d50870d9 (all 20 non-slow cases): preprocessing and the
frozen-input ERA5 runs identical; the 11 runs that read the recorded
preprocessing binaries change only through the radius (step 2c): cell areas
+7.19e-5, emissions of density inventories +7.2e-5 (Rn-222; fossil CO₂ on
C90), unchanged where the total is renormalized (lat-lon GridFED) or given per
cell (EDGAR SF₆), mixing ratios at rounding level. Accepted into `ref_current`.

## Phase 4: move-only file splits

Each split cuts a file at its section boundaries into files included in the
original order; a check confirms that the code lines of the pieces, in include
order, equal the original's and that no comment is lost. Only the file-level
`using` line moves to the first piece.

### Step 1 — structured and cubed-sphere Strang files

- `StrangSplitting.jl` (1635 lines) → `workspace.jl`, `sweeps.jl`,
  `subcycling.jl`, `StrangSplitting.jl` (`strang_split!`),
  `strang_apply.jl` (`apply!` entry points), `multitracer_strang.jl`.
- `CubedSphereStrang.jl` (1559 lines) → `cs_sweep_common.jl`,
  `cs_sweep_x.jl`, `cs_sweep_y.jl`, `cs_sweep_z.jl`, `cs_workspace.jl`,
  `cs_subcycling.jl`, `CubedSphereStrang.jl` (`strang_split_cs!`,
  `strang_split_cs_mt!`).

References to the moved code (READMEs, theory pages, the RG boundary-stub
error message and its test) name the new files.

### Step 2 — GEOS native and ERA5 N320 preprocessing files

- `transport_binary/cubed_sphere_geos.jl` (2299 lines) →
  `geos_cs_mass_helpers.jl` (DELP ↔ mass, pressure fixer, smoothing),
  `geos_cs_omega.jl` (OMEGA-consistent `cm` target), `geos_cs_resolution.jl`
  (global pin, resolution strategies, payloads), `geos_cs_window.jl` (window
  workspace, preparation, `cm` closures, substeps), `cubed_sphere_geos.jl`
  (driver context, hooks, `process_day`).
- `sources/era5.jl` (1718 lines) → `era5.jl` (settings, paths, day handles),
  `era5_n320_window.jl`, `era5_n320_mass_convection.jl`, `era5_n320_to_cs.jl`.
