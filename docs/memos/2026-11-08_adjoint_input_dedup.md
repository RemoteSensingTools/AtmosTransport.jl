# 2026-11-08 — Exact de-duplication of the cubed-sphere 4D-Var inputs

Step 1 of the adjoint-memory plan: remove duplicated per-window adjoint inputs
without changing any result bit. The manual describes the mechanism
(`docs/src/theory/adjoint_status.md`, "Memory of the captured inputs"); this
memo records the measurements.

## What changed

- `cs_adjoint_inputs` keeps one layer-thickness tuple per met window
  (`diffusion_layer_thickness`, which replaces the `diffusion_workspace`
  keyword) instead of a full `DiffusionWorkspace` (factors, layer thickness,
  exchange, elimination). The replay owns one reverse-solve scratch.
- `CSStepSchedule(sim)` uses one array for window `w`'s end reset target and
  window `w + 1`'s start target when the two are bitwise equal, halos
  included.

## Setup of the measurements

- MERRA-2 cubed-sphere transport binaries for December 2021 (C24, C48, C90,
  L72), GEOS-Chem Holtslag–Boville diffusion with the surface flux as the
  solve's boundary condition, CMFMC convection (cloud base from CMFMC), binary
  window contract with `:preserve_tracer_mass` resets.
- One `cs_adjoint_inputs` capture per daily binary, concatenated into one
  schedule, as in the stage-0/1 identical-twin OSSE driver. C48 has seven
  binaries; the 31-day C48 runs cycle them (memory and step counts are
  representative, the meteorology across the wrap is not).
- Baseline: origin/main ad2e0b36c. After: the step-1 change on the same base.
  Same driver and inputs for both.

## Bitwise checks

- CPU, Float64, one thread, C24, two days, unlimited PPM and Lin–Rood ORD 7,
  601 synthetic observations (placed points, column means, averaging-kernel
  columns): cost, gradient and simulated values are bitwise identical before
  and after, for `FullCheckpoint`, `StrideCheckpoint` (one met window) and
  cost-only evaluations.
- GPU (NVIDIA L40S), Float32, C90, seven days, 50 000 observations,
  `StrideCheckpoint` of one met window: cost and simulated values are bitwise
  identical in every before/after pair of evaluations. Gradients are not
  reproducible between two evaluations of one unchanged build (atomic
  accumulation): before vs before differs by up to 2.6e-7 (unlimited PPM) and
  1.3e-7 (Lin–Rood) of the largest gradient; before vs after by up to 2.9e-7
  and 1.3e-7.

## Memory

Inputs after setup: CUDA pool memory in use after the capture (after garbage
collection), which is the live captured inputs. Peak: the process's device
memory sampled by `nvidia-smi` every 0.5 s through the evaluations. L40S
(46 GiB), Float32, 50 000 observations.

| Case | Inputs after setup, before → after | Peak, before → after |
| --- | --- | --- |
| C90, 7 days, unlimited PPM | 27.97 → 19.01 GiB | 35.7 → 27.6 GiB |
| C90, 7 days, Lin–Rood ORD 7 | 27.97 → 19.01 GiB | 42.0 → 37.5 GiB |
| C48, 31 days, unlimited PPM | 37.0 → 25.4 GiB | 44.5 (pool full) → 33.4 GiB |
| C48, 31 days, Lin–Rood ORD 7 | 37.0 → 25.4 GiB | out of memory in the first forward run → 37.6 GiB |
| C90, 11 days, Lin–Rood ORD 7 | — → 29.9 GiB | — → 44.5 GiB (pool full; fits) |

The C48 "before" rows come from the earlier OSSE sizing runs (same driver,
same inputs for both schemes).

Unique captured device bytes, C90 seven days: face fluxes 8.12 GB, convection
forcing 4.77 GB and Kz panels 2.35 GB are unchanged; the diffusion workspaces
(9.41 GB) became layer thickness (2.35 GB); the reset targets fell from 5.35
to 2.79 GB (175 unique arrays of 336). C90 inputs per day: 4.00 → 2.72 GiB.

## How often targets are shared

Within each daily binary all 23 interior window boundaries are bitwise
continuous: C24 two days 50 unique targets of 96, C90 seven days 175 of 336,
C48 31 days 775 of 1488. The day boundaries between separately captured days
keep two arrays; on the C24 binaries checked (December 1–3) they were bitwise
equal as well.
