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
