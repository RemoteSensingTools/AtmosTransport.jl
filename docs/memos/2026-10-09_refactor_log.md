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
