# Float32 conservation attribution (2026-10-06)

Raw record behind `docs/src/theory/float32_conservation.md`. Branch
`feature/catrine-c90-benchmark`. `Float32` runs used the L40S on wurst and
`Float64` runs the A100 on curry, with the same code checkout.

## Runs

Workspace: `~/data/AtmosTransport/output/precision_attribution/`. It holds the
configs (`*.toml`), logs (`*.log`), outputs in `out/<tag>/precision_YYYYMMDD.nc`,
`run_set.sh` (sequential runner) and `budget.py`. The configs are equivalent
to those written by `scripts/diagnostics/float32_conservation_attribution.py`,
which names its outputs `attribution_YYYYMMDD.nc`; its `budget` command
reproduces the tables below from them.

- MERRA-2: C90 L72 hourly binaries
  (`met/merra2/c90/transport_binary_v4_l72_f32_physics`), 2021-12-01..07, PPM.
  Tracers: background 400 ppm (no source), CO₂ (CAMS 3-hourly, stepwise),
  SF₆ (LSCE initial state + EDGAR cs_native), fossil CO₂ (GridFED, from zero).
  Sets: `adv`, `adv_diff` (`geoschem_nonlocal_vdiff`), `adv_conv` (`cmfmc`,
  DQRCU cloud base), `full`.
- ERA5: C90 L66 binaries with TM5 convection
  (`met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour_v3`),
  2021-12-01..03, no sources. Tracers: background, SF₆ initial state. Sets:
  `era5_adv`, `era5_adv_dkg` (`tm5_dkg`), `era5_adv_tm5conv` (collaborative LU,
  `lmax_conv = 66`, `n_merge = 1`), `era5_full`.

Tags: `_f32` and `_f64` are runs before the fixes. `_fixed` adds the ledgers,
the profile-deposit TwoSum, `Float64` areas, decay and clock.
`era5_adv_tm5conv_ledger` has the TM5 ledger only. `_regridfix` and
`full_final` add `Float64` regridding geometry and source-extent snapping.
`era5_*_final` and `full_final2` are the committed code after review
(shared ledger helpers with the guard, fused dkg ledger, counter clock).

Values are relative changes of the `<tracer>_total_mass` series. Drift is
`(M_end − M_0)/M_0`. F32−F64 is
`(M32 − M64)/M64 − (M32 − M64)/M64 at t0`; this removes the constant
`Float32(400e-6)` offset of −2.53e-8. Fossil starts at zero; its F32−F64 is
relative to the emitted mass.

## ERA5, Float32 drift over 3 days (no sources)

| Set | Background | SF₆ |
|---|---|---|
| era5_adv | +8.65e-9 | −1.23e-9 |
| era5_adv_dkg | +9.02e-9 | −1.24e-8 |
| era5_adv_dkg_fixed | +8.35e-9 | +2.52e-9 |
| era5_adv_dkg_final | +8.35e-9 | +2.52e-9 |
| era5_adv_tm5conv | −9.71e-7 | −9.92e-7 |
| era5_adv_tm5conv_ledger | +7.04e-9 | −1.66e-9 |
| era5_adv_tm5conv_final | +7.04e-9 | −1.98e-9 |
| era5_full | −9.67e-7 | −1.00e-6 |
| era5_full_fixed | +9.17e-9 | +1.78e-9 |
| era5_full_final | +9.17e-9 | +2.09e-9 |

3-hourly increments (mean, std, t):

| Set | Background | SF₆ |
|---|---|---|
| era5_adv | +3.6e-10, 5.5e-10, +3.2 | −5.1e-11, 2.4e-10, −1.0 |
| era5_adv_dkg | +3.8e-10, 6.1e-10, +3.0 | −5.2e-10, 3.3e-10, −7.7 |
| era5_adv_dkg_fixed | +3.5e-10, 5.5e-10, +3.1 | +1.1e-10, 2.2e-10, +2.4 |
| era5_full | −4.0e-8, 1.5e-9, −134 | −4.2e-8, 1.3e-9, −153 |
| era5_full_fixed | +3.8e-10, 6.1e-10, +3.1 | +7.4e-11, 2.1e-10, +1.7 |

## MERRA-2, 7 days, F32−F64 (net of t0)

| Set | Background | CO₂ | SF₆ | Fossil (of emitted) |
|---|---|---|---|---|
| adv | +7.99e-9 | −5.54e-9 | −4.60e-9 | +8.23e-7 |
| adv_regridfix | +7.99e-9 | −4.20e-9 | −4.11e-9 | −3.99e-8 |
| adv_diff | +6.52e-9 | −2.63e-7 | +9.94e-8 | +8.21e-7 |
| adv_conv | +1.24e-8 | −2.71e-8 | −1.07e-8 | +8.36e-7 |
| full | −9.56e-9 | −2.44e-7 | +1.14e-7 | +8.45e-7 |
| full_fixed | −2.32e-9 | −1.88e-8 | −4.26e-9 | +8.43e-7 |
| full_final | −2.32e-9 | −2.00e-8 | −3.75e-9 | −1.81e-8 |
| full_final2 (F32 only, vs full_final F64) | −2.32e-9 | −2.01e-8 | −3.75e-9 | −1.87e-8 |

The F64 background total is constant to the last bit in every set. The
fossil F32−F64 is constant from the first 3-hour snapshot on (+8.3e-7), so it
is a fixed rate difference, not an accumulating leak. Its cause was the
regridder: see below. In `full_final` the background increments have no
detectable mean (−4.1e-11 per 3 h, std 6.1e-10, t = −0.5). The remaining CO₂
−2.0e-8 equals the CMFMC share before the fixes (adv_conv − adv = −2.2e-8);
CMFMC has no column ledger. CO₂ F32−F64 relative to the net flux stays between
−6.7e-5 and −8.1e-5 from day 1. Net 7-day burden change: CO₂ +2.98e-4, SF₆ +6.90e-4.

## Unit probes

- C90 areas, `Float32` mesh vs `Float64` mesh. Before: max |rel| 1.26e-4,
  mean −6.8e-8, global total −1.6e-8; Δx/Δy max 1.5e-5/1.8e-5. After: max
  5.7e-8, total −6.6e-10; Δx/Δy 4.4e-8. Areas are now exactly
  `Float32.(areas64)`.
- Rn-222 decayed fraction, `1 − exp(−λΔt)` in `Float32`, relative error:
  −3.4e-5 (300 s), +3.0e-7 (400 s), +2.9e-5 (450 s), −3.1e-6 (600 s),
  −4.3e-6 (900 s). `Float32(expm1(−λΔt))`: ≤ 3.2e-8.
- `Float32` clock `t += Δt`, two years. 300/400/900 s steps: exact. 7 steps
  per hour (514.29 s): 11306 s behind after 1 year, 85916 s ahead after 2.
- GridFED regridding. Centres are stored in `Float32` (−179.9499969…). The
  old `_build_source_latlon_mesh` took the spacing from the first difference
  (0.0999908°), so the extent missed 360° by 1.5e-5° and ±90° by 3.8e-6°.
  `_latlon_full_sphere` (atol 1e-6) failed, and the `Float64` regridder lost
  2.09e-6 of the flux. The `Float32` source mesh rounded onto the full sphere
  (2.2e-10) but produced different weights: a separate cache key, and
  `Spherical{Float32}`. After the fix both precisions load the same cache
  (`regridder_2dbfcfd0…`, `regridder_4a8cd892…`): F64 rel_err 0.0, F32
  6.2e-10 and 8.8e-10 (rate rounding).
- CUDA codegen (L40S): `q + x*f` compiles to `FFMA` (ptxas default
  `--fmad=true`). PTX had `mul.f32`/`add.f32` without `.rn`.
- `run_set.sh` printed `exit $?` after `$(date)` had reset it, so every run
  logged "exit 0". It is fixed now; failed runs show up as missing outputs.

## Agent sweep findings (CPU, 2026-10-06)

- TM5 LU in `Float32`: −1.29e-8 of the column per application. Delta form:
  −1.3e-11. Column ledger: ~1e-12.
- ProfileDeposit at 410 ppm: −9.8e-4 of the added emission.
- `dkg` diffusion: the ERA5 run gives SF₆ −3.7e-9 per day beyond advection
  (≈1e-7 per month).
- VMR Thomas path: 2–3e-6 per month. Clamped CMFMC: 5e-6 per month. Neither
  is used in production configurations; both are left open.
- CMFMC production path: ±8e-8 per month.
- Advection: no change of tracer mass at window resets. The MERRA-2 hourly
  binary's air mass jumps +3e-8 at window resets, which is harmless under
  `preserve_tracer_mass`.

## Ledger regime (independent review, pre-ledger code)

Residual of one solve, relative to the participating column, and in ulps of
the largest cell: TM5 1.2–2.1e-8, median 2–8 ulps, below ½ ulp in 2–10 % of
columns; `dkg` ≈5e-10, median ≈1 ulp, below ½ ulp in 21–28 % of columns. The
`dkg` ledger therefore removes the bias statistically. The guard
(`_ledger_residual`) returns at most 16 ulps of the largest cell per
participating cell; a deliberately leaking matrix (one column sum 1 + 1e-3)
stays visible (`test_float32_conservation.jl`).

## Cost

Transport-section time (the `transport` entry of the log's `Forward run wall`
line) over 3 ERA5 days, `Float32` L40S, before → after the ledgers:
advection + TM5 convection 22.9 → 25.2 s, advection + `dkg` 19.8 → 19.0 s, all
operators 26.8 → 23.3 s. These short runs vary by about 10 % between
repetitions, so no cost is resolved. The final code fuses the `dkg` ledger
into the existing column passes.

## Reviews (2026-10-06)

julia-style-reviewer, a bug checker and a Fable review found no must-fix
defects. All three confirmed zero `.f64` instructions in the touched kernels
on the L40S. Applied: shared compensated-sum helpers and a backend-aware
`Float64` total (also used by `total_mass`), the ledger guard, the fused `dkg`
ledger, the counter-based clock, Float64 time in diffusion paths, span-based
spacing for EDGAR cell areas, and corrected wording and numbers in the docs.
Noted, not changed: CUDA `ptxas` fuses `q + x·f` into FFMA (TwoSum bookkeeping
in the profile deposit is exact to ½ ulp of a share, and was bitwise equal on
CPU and GPU in a 2000-step probe); `Float32`-preprocessed binaries carry the
old `Float32` cell areas (≤1.3e-4 at C90, 6.8e-4 at C180, 9.7e-3 at C720).
