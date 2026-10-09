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
the table (twelve scripts) were replaced later (see "A5, script copies").

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

Plan item A11 also lists differing guards of the preprocessing adaptive-substep
loops. The final windows of the ERA5 spectral → cubed-sphere path and of the
lat-lon → cubed-sphere regrid used a CG limit of 5,000 where every other window
uses 20,000; they use 20,000 now (no change while the solve converges within
5,000 iterations). Column weights
missing from three cubed-sphere loops would change results and are left for
the owner.

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

### Step 3 — Lin-Rood adjoint kernels, reduced-Gaussian preprocessing

- `Advection/linrood_adjoint_kernels.jl` (1736 lines) →
  `linrood_adjoint_kernels.jl` (update, pre-advection, q-input faces),
  `linrood_adjoint_rm_faces.jl` (rm-input faces, ORD 5 and 7),
  `linrood_adjoint_panel.jl` (one-panel horizontal adjoint).
- `Preprocessing/reduced_transport_helpers.jl` (1321 lines) →
  `reduced_transport_helpers.jl` (workspaces, synthesis, fluxes, merge),
  `reduced_window_buffer.jl` (two-slot buffer, ingest/drain/flush),
  `reduced_spectral_day.jl` (window synthesis and pin, balance, driver hooks,
  `process_day`).

### Step 4 — DrivenSimulation

- `Models/DrivenSimulation.jl` (1050 lines) → `DrivenSimulation.jl` (type,
  compatibility checks, flux storage scaling), `driven_window_state.jl`,
  `driven_physics_refresh.jl`, `driven_stepping.jl`. The source-scanning
  guard in `test_architectures.jl` reads all four.

### Step 5 — GEOS reader, CS Poisson balance and helpers, checkpoint drivers

- `sources/geos.jl` (1320 lines) → `geos.jl` (settings, paths, handles,
  level orientation), `geos_panels.jl`, `geos_read_window.jl`.
- `cs_poisson_balance.jl` (1233 lines) → `cs_face_table.jl`,
  `cs_poisson_solver.jl`, `cs_poisson_balance.jl` (entry points, cm diagnosis;
  keeps the file header).
- `cs_transport_helpers.jl` (1059 lines) → `cs_transport_helpers.jl`,
  `cs_flux_reconstruction.jl`, `cs_wind_rotation.jl`, `cs_native_fluxes.jl`.
- `Footprint/StrideCheckpoint.jl` (1140 lines) → `StrideCheckpoint.jl`
  (linear tape, shared helpers), `stride_checkpoint_ppm.jl`,
  `stride_checkpoint_linrood.jl`, `revolve_checkpoint.jl`; its header no longer
  claims the PPM and Lin-Rood drivers are missing.
- A theory page credited a removed per-level mass correction with closing the
  regridded mass distribution; it now names the extensive-field regrid.

### Step 6 — cubed-sphere mesh, Lin-Rood forward

- `Grids/CubedSphereMesh.jl` (1014 lines) → `CubedSphereMesh.jl`,
  `cs_mesh_coordinates.jl` (forward geometry: projection, centers and corners,
  edge lengths, tangent bases), `cs_mesh_locate.jl` (inverse projection).
- `Advection/LinRood.jl` (839 lines) → `LinRood.jl` (workspace, damping,
  kernels), `linrood_horizontal.jl` (`fv_tp_2d_cs!`, q-space variant, Lin-Rood
  Strang split).

Phase 4 stops here: no `src/` file is above 1000 lines (the largest are now
`tm5_kernels.jl` 996, `era5_n320_regrid.jl` 990, `DrivenRunner.jl` 907). Some
800–1000-line files still mix responsibilities (`latlon_contracts.jl`,
`mass_support.jl`); they belong with the Phase 6 consolidation. Export lists
stayed where they were, so some now sit in a later piece than the code they
export.

## Phase 3, step 3 — stale comments and docstrings

The README findings that were documentation errors are fixed, each checked
against the code: module docstrings (Tape, Adjoints, Preprocessing,
Quantities, Regridding), file headers that described missing drivers or
history ("relocated unchanged", "follow-up", "subsequent breakpoints"),
citations of removed files and of numbered invariants that no longer exist,
misattached docstrings (`next_day_merged_fields`, `_pack_cs_window!`), the
SectionTimer column list and switch values, `docs/src/preprocessing/overview.md`
(which paths use the unified driver; ERA5 N320 added) and
`docs/src/config/toml_schema.md`. A script confirms that only comments and
docstrings changed, apart from seven error/warning messages that cited
removed invariants, named the wrong file, called non-C180 targets "C180", or
promised a "follow-up".

Golden check after Phase 4 step 2 (31a819f4, all 20 non-slow cases):
identical.

The full test suite (after Phase 4) stopped at `test_readme_current.jl`: the
Advection README had never listed `vertical_fv3_profile.jl` (added at
a2c99ed6), and a failing file ends the runner loop. The README lists it now;
the freshness test also covers every folder that gained a README (and the
Output README now lists `Output.jl`, `runtime_output.jl`, `binary_writer.jl`).

Golden check after Phase 4 (3ca56c55, all 20 non-slow cases): identical.
The full test suite (`Pkg.test()`, default tiers) passes at 903cba5b.

### A5, script copies

The twelve benchmark and diagnostic scripts that read cubed-sphere binaries
through their own copy of the section-size table now call the library's
`MetDrivers._cs_section_elements`. Evaluated against a mock header, every
copy agreed with the library for every section it knew, except sections that
cubed-sphere binaries cannot contain: `qv`, `qv_start`, `qv_end` (all copies)
and `hflux` (two copies). Most copies lacked `dkg`, `dam`/`dbm`/`dcm`, the VDIFF
sections, `pbl_eflux` and `cmfmc_cloud_base`.

## After the owner's review (2026-10-09, day)

The owner answered the open decisions: slight negatives of fossil CO₂ are
acceptable numerical noise, but a positivity-preserving PPM should exist as an
option; the public API without callers is removed or deprecated; the quick
fixes, the experiments and Phases 5–7 follow.

### Slow golden cases

The four `slow` cases were still at their pre-Phase-2 reference. At 602472fb:

- `run_c90_merra2_cpu_f64`, `run_c90_era5_tm5_cpu_f64`: the Phase 2 deltas
  (constants; for MERRA-2 also the radius key, cell areas +7.2e-5).
- `pre_era5_n320_c90_l117`: the header gains `planet_radius_m = 6371229`. Every
  payload section is bit-identical except the TM5 boundary-layer exchange
  `dkg`: median +3.1e-4, 99.9 % of values within +5.0e-4 (gravity and dry-air
  constants); 155 of 1.35e8 values change by more than 1 %, threshold cells of
  the diffusion diagnosis (the entrainment-fallback counts are identical in
  every window).
- `run_c90_era5_l117_gpu_f32`, rerun against the new N320 binary: cell areas
  +7.2e-5 from the key, tracers from the `dkg` change (column means within
  1.3e-5 relative).

All four are accepted into `ref_current`.

### A10 option — complete Colella–Woodward PPM (`limiter = "cw84"`)

`PPMScheme(CW84Limiter())`, TOML `[advection] limiter = "cw84"`, makes three
changes to the structured PPM face flux (lat-lon and cubed sphere, CPU and
GPU):

1. edges from van Leer-limited slopes (CW84 eqs. 1.7–1.8; the minmod of
   `_limited_slope`), so each edge lies between its neighbouring cell means;
2. the same monotone profile limiter as before;
3. the flux integrates the limited parabola over the swept fraction (CW84
   eq. 1.12), entering `_slopes_face_flux` as the moment
   `s_x = m (b_R − α b_0)` with the curvature `b_0 = (q_L − c) + (q_R − c)`.

The stencil is unchanged (six cells), so halos are unchanged. The default
`PPMScheme()` and the other limiters are bit-identical: the new code is
reached only through dispatch on `CW84Limiter`. The cubed-sphere adjoint
covers it: `_ppm_face_coeffs` (formerly `_ppm_monotone_face_coeffs`) takes the
limiter and differentiates the CW84 edge (minmod branch) and the curvature
term; `CSAdjointNonlinearScheme` includes it.

Tests (`test/core/test_ppm_cw84_limiter.jl`, plus CW84 in the lat-lon kernel,
cubed-sphere seam/offset, footprint and GPU-adjoint loops):
- 1-D ring, spike and box on zero, 100 steps, Courant 0.3/0.7/1.0/−0.45,
  Float32 and Float64: stays in [0, 1] and conserves mass; the default PPM
  reaches −0.16 (box, 0.7) and 1.015 (box, 0.3).
- One sweep with random divergent and convergent fluxes, outflow at most the
  cell mass: tracer mass stays non-negative (200 trials).
- Sine, one revolution at Courant 0.4: error ratio per doubling 4.05 (default
  3.46); 18 % lower error at 160 cells, 29 % higher at 20 cells (extremum
  clipping).
- Edge between neighbours (1000 random stencils); fourth-order edge where
  unlimited; uniform mixing ratio kept on non-uniform mass (x, y, z).
- Face adjoint against central differences of the forward, x/y/z, both
  limiters, 40 random stencils each (a mutant without the curvature
  derivative fails); footprint FD replay; GPU (L40S) footprint gradients.
- CPU/GPU agreement of the lat-lon kernels within 0.008 ulp.

Codex review findings addressed: the one-argument `PPMAdvectionSpec(vertical)`
constructor kept; positivity stated as a per-sweep condition (no cell exports
more than its mass) instead of "monotone for Courant ≤ 1"; CW84 in the GPU
adjoint test; the run log labels the option `PPM, CW84`; adjoint docs.

The golden harness gains an `add` field (a key the template lacks) and three
cases: `run_ll72_ppm_cw84_cpu_f64`, `run_c24_ppm_cw84_cpu_f64`,
`run_c90_merra2_cw84_gpu_f32`. Golden check of the change: all 16 existing
non-slow runtime cases identical; the three new cases are accepted. In them
the blob and Rn-222 (non-negative sources) have no negative cell, against
27–38 % with the default PPM. Fossil CO₂ keeps negative cells because GridFED
has negative cells: the negative mass is −3.9e-4 of the positive mass with
CW84, −4.0e-4 with slopes and −3.4e-4 with upwind on the lat-lon golden. At
its last snapshot the negative fossil cells are 23 % (upwind), 29 % (slopes),
42 % (default PPM) and 12 % (CW84); cells below −1e-6 of the maximum are
0.03 % for upwind, slopes and CW84 and 0.88 % for the default PPM. (The cell
percentages of the A10 table above are not reproduced from the current
references.)

C90 evaluation: the production MERRA-2 configuration
(`merra2_hm_gchp_ppm.toml`: GCHP-like flux construction, CMFMC convection,
VDIFF, Float32; GCHP initial state), 2021-12-01 to 2022-03-31, with
`limiter = "cw84"` (`/temp1/cfranken/jobs/cw84_eval/`; outputs in
`~/data/AtmosTransport/catrine_protocol_output_2026_10/merra2_hm_gchp_ppm_cw84_dec2021_mar2022`).
The default-PPM run used for comparison predates the Phase 2 constants
(emissions −3.7e-5, diffusion dz +3.4e-4 relative), which are small next to
the differences below.

| last snapshot of | default PPM: negative cells (fossil from Dec / Rn-222) | CW84 |
|---|---|---|
| 2021-12-01 | 22.5 % / 2.6 % | 0.1 % / 0.0 % |
| 2021-12-08 | 33.3 % / 13.0 % | 0.0 % / 0.0 % |
| 2022-01-15 | 26.5 % / 28.0 % | 0.0 % / 0.0 % |
| 2022-03-31 | 18.2 % / 4.0 % | 0.0 % / 0.0 % |

On the first day CW84 leaves 0.1 % slightly negative fossil cells (negative
mass −7e-6 of the positive mass) from the first emissions; from day 8 the
minima are positive.

Against GCHP (`catrine_compare_vs_geoschem.py`, period means, bias and RMSE
relative to the GCHP mean; output in
`/temp1/cfranken/catrine_protocol/compare_c90_cw84`), with the FV3 vertical
profile run (`hm_gchp_fv3`) for reference:

| tracer, band | default PPM | CW84 | FV3 vertical |
|---|---|---|---|
| fossil, column | −0.55 %, 1.49 % | −0.55 %, 1.51 % | −0.55 %, 1.50 % |
| fossil, surface–910 hPa | −0.43 %, 2.72 % | −0.56 %, 2.88 % | −0.31 %, 2.66 % |
| fossil, above 100 hPa | +0.41 %, 1.87 % | +4.44 %, 5.80 % | −1.53 %, 1.98 % |
| Rn-222, surface–910 hPa | +0.02 %, 4.21 % | −0.57 %, 4.39 % | −0.11 %, 3.94 % |
| Rn-222, 910–400 hPa | −1.77 %, 5.13 % | −1.41 %, 4.81 % | −1.67 %, 4.98 % |
| Rn-222, above 100 hPa | −10.7 %, 38.4 % | +18.1 %, 21.3 % | −4.8 %, 9.8 % |

CO₂ and SF₆ agree to 0.01–0.05 % in every band for all three. Columns are
unchanged. Near the surface CW84 is slightly further from GCHP; in the
stratosphere, where these tracers are small, its means are higher (the
default's stratospheric means include its negative cells). The FV3 vertical
profile, which GCHP uses, matches best there. CW84 horizontal sweeps with the
FV3 vertical profile (`limiter = "cw84"`, `vertical = "fv3_kord8"`) are
supported; a C90 run of that combination is the next evaluation.

Cost: three C90 days of the same configuration on one L40S, alternating the
two schemes (`/temp1/cfranken/jobs/cw84_eval/run_ab*.sh`), with other jobs
loading the machine. Transport time: CW84 22.5–24.7 s in all four runs;
default PPM 21.3 and 22.8 s in two runs, while its other two took 51 and
66 s (a cold start and a load spike). CW84 costs roughly 5–10 % of transport
time.

### Quick fixes (one commit each, Codex-reviewed in two rounds)

- `[mass_fix] mode = "initial_endpoint"` is refused outside the GEOS native
  path (MERRA-2 and ERA5 N320 silently wrote unpinned binaries); the
  `[mass_fix]` schema section was wrong (said spectral-only).
- The GEOS writer honors `ATMOSTR_NO_WRITE_REPLAY_CHECK`; with the gate off,
  the regrid, N320 and MERRA-2 writers now give the positivity gate the end
  mass as with the gate on.
- The reduced-Gaussian writer stages to `.tmp` (a failed day deleted an
  existing binary; the new test fails on the old writer).
- Visualization regrids cubed-sphere snapshots on their recorded definition
  (laws and longitude offset).
- OPeNDAP downloads (the MERRA-2 recipe) are refused before anything is
  created, unless dry-run or verify.

Golden check at 17bda3f1 (the quick fixes and the API cleanup): the four
non-slow preprocessing cases are identical, and the three CW84 cases
reproduce their accepted reference.

### Public API

`reset_workspace!` (exported, no methods) is removed; `MetState`,
`diagnose_cm_from_continuity_vc!` and `_ka!` are deprecated for removal in the
next minor release.

### Experiments (plan item A11 remainder and the positivity gates)

- Column weights in the three cubed-sphere loops: no defect. Only the ERA5
  N320 and MERRA-2 settings carry `column_balance_weights`, and both pass it
  to the balance and the `cm` closure. The ERA5 spectral, lat-lon → cubed
  sphere and GEOS paths have no such setting and use air-mass weights.
  Offering the key there is a feature decision.
- Positivity-gate denominators: on the lat-lon golden (72 × 37, 4 fixed
  substeps), the lat-lon gate (each direction's outflow over the start mass)
  peaks at 0.48 per substep; the cubed-sphere form (twice the summed outflow
  over min(m, m_next)) peaks at 0.97 (windows 6–7) and would fail the 0.95
  limit or need 5 substeps there. Adopting the cubed-sphere form for lat-lon
  changes lat-lon binaries; left for the owner. (Both ratios were computed
  per window from the stored substep fluxes and masses of the binary.)

### Phase 7 survey (scripts)

A read-only survey classified the scripts (maintained / one-off / unclear):
diagnostics 25 / 56 / 28, visualization 7 / 8 / 15, preprocessing 20 / 2 / 4,
benchmarks 9 / 8 / 2, validation 9 / 0 / 2, top level 1 / 3 / 7. Scripts
that walk transport-binary payloads by hand: 12 (the library has no public
single-section reader); five read ATMSNAP1 snapshots by hand (no library
reader). Constants are redefined in about 35 scripts; a lon/lat → xyz helper
exists in 11 Python files with two argument orders. Source and config comments
cite several scripts as provenance, and two deleted scripts are still cited,
ten times in eight files (`download_era5_physics.py` in
`src/Preprocessing/era5_physics_binary.jl`, its test,
`config/met_sources/era5.toml` and a legacy preprocessing config; a TRENDY
config generator in four batch configs).

### Phases 5–7 informed by Oceananigans.jl

At the owner's suggestion the structure of Oceananigans.jl (CliMA, read at
`bf47112f`) was compared with ours. Patterns to adopt, in order:

1. Requirement traits folded once at model construction (halo width, payload
   sections, capabilities) instead of per-call halo checks and per-operator
   capability errors. Oceananigans: `required_halo_size_x`,
   `closure_required_tracers`, automatic halo sizing.
2. ENV switches that change results or cadence become config keys recorded in
   binary headers or run metadata (Oceananigans has no `ENV` read in `src/`;
   we have about 40 read expressions outside comments). No header records
   which Poisson balance (column or per layer) built a binary. First: `ATMOSTR_ENABLE_HORIZONTAL_POISSON_BALANCE`,
   `ATMOSTR_FORCE_PER_SUBSTEP_PHYSICS`, `ATMOSTR_ASSERT_CS_BINARY_CFL`.
3. Types for the GEOS `cm_closure` (a Symbol tested at more than 15 sites) and
   for surface-flux datasets (unit conversions duplicated per `kind`).
4. One kernel-launch helper (Oceananigans `launch!`): 168 hand launches and
   124 `synchronize` calls today. Replace call sites one to one with their
   current workgroups (bit-identical), then remove synchronizations one at a
   time behind GPU timing gates (the halo synchronization paces prefetch).
5. Direction singletons and topology tags (`Periodic` x, `Bounded` y/z) for the
   nine x/y/z face-flux methods and about 25 per-direction kernels; real
   differences (the z mass floor) become named methods.
6. Single-tracer kernels folded into the packed ones (`TracerView`, Nt = 1).
7. A column-layout type for the TM5, CMFMC and diffusion column kernels.
8. One window loop with schedule callbacks instead of the structured and
   cubed-sphere loops in `DrivenRunner.jl`, which each poll snapshot times.
9. `show`/`summary` for schemes, operators, the model and the simulation,
   printed at run start.
10. Tests mirroring `src/` with one backend switch, ExplicitImports checks,
    and allocation budgets per window.
11. A bibliography (DocumenterCitations), tested doctests, a developer guide.

Not to copy: unsplit tendencies with Runge–Kutta stepping (the split
mass-flux sweeps are kept for parity with TM5 and GCHP and for our discrete
contracts: per-sweep positivity budgets and binary replay), `@muladd` and
global workgroup heuristics (they break bit-identical goldens), Unicode
operator names, adaptive time steps (fixed by the binary contract), global
mutable defaults, and the KernelAbstractions internals behind their mapped
kernels.

## Status (2026-10-09, afternoon)

`refactor/structure-2026-10` is pushed as PR #21 (25 commits on `3684b71a`,
stacked on `feature/catrine-c90-benchmark`). `refactor/wip` adds, not yet
pushed: the CW84 PPM option, five quick fixes, the public-API cleanup, the
script archive (Phase 7 step 1) and this log. Every commit was reviewed by Codex; every results-changing step is
in the golden reference `ref_current` (`ACCEPTED.txt`), now including the four
slow cases and three CW84 cases.

Done:
- Phases 0–4 (see above).
- A10 as an option (`limiter = "cw84"`), evaluated on C90 against GCHP; the
  default PPM is unchanged.
- The owner's decisions on the public API; the five quick fixes; the column
  weight question (no defect).
- Phase 7 step 1: 77 one-off scripts moved to `scripts/completed_experiments/`
  (index in its README; include and self-reference paths fixed, including
  20 earlier archived scripts whose include of `cs_regrid_utils.jl` was
  broken) and a `scripts/README.md`.

For the owner:
- The MERRA-2 default combination of advection, diffusion and convection that
  best matches GCHP (GCHP being a benchmark, the options remain for
  transport-uncertainty estimates). A C90 run of CW84 horizontal sweeps with
  the FV3 vertical profile is the next comparison.
- Whether lat-lon preprocessing should use the cubed-sphere positivity gate
  (more substeps on the lat-lon golden).
- Whether the ERA5 spectral, lat-lon → cubed sphere and GEOS paths should
  offer `column_balance_weights`.

Not started: Phases 5–7, in the order listed in "Phases 5–7 informed by
Oceananigans.jl", plus the library readers of Phase 7 (a public
single-section binary reader and an ATMSNAP1 reader; survey above) and the two block-coarsening helper sets (their area-weighted
versions differ in accumulation precision).
