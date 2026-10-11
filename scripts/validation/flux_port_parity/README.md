# Flux-port parity harness

This harness checks that the Julia port of the Python surface-flux
preprocessors reproduces the Python outputs at the agreed level, and records
the measured numbers. Each comparison is a pair of `cs_native` flux files (a
reference and a candidate) with tolerances. `run_parity.jl` reads a TOML list
of comparisons and runs `scripts/validation/compare_cs_flux_files.jl` on each,
calling it as a function in the same process. The results go to one combined
JSON report and one markdown report.

The port's design is `/temp1/cfranken/flux_port_parity/design_final.md`
(validation plan in §4, phasing in §5). The P0 results are in
`/temp1/cfranken/flux_port_parity/p0_reports.md`.

| File | Purpose |
|---|---|
| `run_parity.jl` | Runner. The TOML layout is in its header; `--only NAME` selects comparisons; `--json` and `--markdown` write the reports; exit status 1 when any comparison fails or errors |
| `s0_bitwise_pairs.toml` | Example: the two S0 pairs that the P0 review found bitwise identical (hourly→daily C90 2021, refine C180 2021-12-31) |
| `budget_closure_parity.jl` | Diagnostics port (B2): `Analysis.budget_closure` against a report of the retired `check_trendy_npp_rh_run.py`, three ways (legacy arithmetic, runtime sources, runtime sources at the run's old constant); one JSON report; exit 1 unless the legacy run reproduces the report |
| `emulators/legacy_budget_emission.jl` | Harness-only expected emissions for `budget_closure`: `LegacyDailyFluxFileEmission` (the Python sums: file `cell_area`, whole days, M_air 0.02896546), `ScaledEmission`, `TabulatedEmission` |
| `README.md` | This file |

The comparator, `scripts/validation/compare_cs_flux_files.jl`, lives one
directory up and stays after the port. Its header defines every metric, the
reference-scale rule, the units rule, the n/a rule, the coastal-mask options,
and the rules that stop a comparison: variables or stored `lats`/`lons` outside
the cs_native layout of the mesh (or only one of `lats` and `lons`), a series
without units, and time bounds that are not start-stamped and contiguous. Time
slices are matched by start and end: equal stamps whose slices end at
different times are a time-axis difference. Tolerances, in the TOML as on the
command line, must be finite and non-negative, with `bitwise_fraction` at most
1 and `ulp_max` a whole number; the runner uses the comparator's own check. Without `variables`, every series of
either file is compared and a series present in only one file fails the
comparison; set `common_variables_only = true` for an intentional comparison of
products with different variable sets (for example SIF v1 against v3).
Bitwise equality is only counted between values of the same stored type. Both scripts are tested by `test/core/test_compare_cs_flux_files.jl`
and `test/core/test_flux_port_parity_harness.jl`, on synthetic C4 files.

## Validation levels (design §4.1)

| Level | What it checks | How the harness expresses it |
|---|---|---|
| S0 | Fresh Python references captured at `71a08395`, after the constant change, so 44/12 does not confound them | Rerun against the existing product, e.g. `s0_bitwise_pairs.toml`, with `bitwise_fraction = 1.0`. Compare data, never file hashes: a GFED5 rerun differs only in its `history` attribute |
| L1 | Algorithmic parity: the Julia library with harness-only legacy emulators plugged in by dispatch (`LegacyGeometryTarget`, `LegacyNearestCentreRemap`). Target ≤ 1 Float32 ulp and ≥ 99.9 % bitwise | `tolerances = { ulp_max = 1, bitwise_fraction = 0.999 }` |
| T | Temporal isolation: per-cell monthly means recovered from the Python daily file go through `MidMonthInterpolation` and are compared with that file; ≤ 2 Float32 ulp | `tolerances = { ulp_max = 2 }` |
| L2 | Production: exact remap and mesh areas. Totals per month and year, per-cell relative-difference quantiles (interior, coastal, polar panels), 60 sin-latitude bands. TransCom-11 region totals come in P7 | `month_total`, `year_total`, `cell_p99.interior`, `band`; a `mask` table for the coastal class; `reference_scale` for files written before `71a08395` |
| R | Span independence: a short build equals the same days of the long build bitwise | `common_times = true`, `tolerances = { bitwise_fraction = 1.0 }` |
| X | Runtime smoke: the logged source rate (`total_emission_rate`) equals Σ density × mesh area to 1e-6 | Not a file pair. It needs its own comparison kind in `run_parity.jl` (a `kind` key and a runner function) when P4 adds the first smoke |

L0 (synthetic unit tests) lives in `test/`, not here.

## S0 references

- Location: `/temp1/cfranken/flux_port_parity/python_ref_71a08395/`, one
  directory per driver: `fluxcom_x` (full 2021), `gfed5_c30_2014`,
  `hourly_to_daily_c90_2021`, `refine_c180_dec2021`, `sif_iav` (full
  2018-05-01 … 2025-12-31), `trendy_hourly_c90_2021`,
  `trendy_multiyear_c30_2015_2016`, `trendy_npp_rh_c30_2014_2024`.
- Manifest: `MANIFEST.json` in that directory. It lists 16 runs. Each run
  records the `driver` (the directory name), the exact command, exit code, wall
  time, peak memory, script and imported-module sha256, the Python and library
  versions, every input with size, mtime and sha256, and every output with
  sha256. To find the reference of a driver, look up its `driver` entries.
- Treat the output hashes as a fixity check only. Rerun checks compare data.
- The commands point at the scripts in the port worktree. After P4+ deletes
  them, reproducing a run needs a checkout at `71a08395`.
- Facts that constrain comparisons against S0 (P0 review, §6):
  - The multi-year TRENDY runs span 2015–2016 and hold values flat after the
    December 2016 midpoint. Feed L1/T the same month list (2014-12..2016-12),
    or leave out the second half of December 2016.
  - LPJ-GUESS has no multi-year reference, because the Python script raises
    `KeyError 'rh'` on its `arh` file. Use the NPP/Rh run or the 2021 hourly
    campaign instead.
  - Files written before `71a08395` use 44/12. Compare them with
    `reference_scale = { NAME = "carbon_44_12_to_iupac" }`, a factor computed
    from `Parameters`, as (SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS) / (44/12).
    The preset is exact for scripts that convert in Float64: GFED5, NPP/Rh,
    the 2021 hourly TRENDY driver and the SIF IAV fluxes `GPP_CO2_FLUX` and
    `RECO_FLAT_CO2_FLUX` are within 1 ulp after rescaling.
  - FLUXCOM-X is the exception: its script converts in float32 (NumPy 1.x keeps
    a float32 array times a Python float in float32), so the old file maps to
    the S0 file through Float32(1e-3·K_new)/Float32(1e-3·44/12) =
    0.9993248757811442, given as a number:
    `reference_scale = { GPP_CO2_FLUX = 0.9993248757811442, TER_CO2_FLUX = 0.9993248757811442 }`.
    Measured: 2 ulp (GPP) and 3 ulp (TER), year totals within 1.6e-11. With the
    preset the values stay within 3 ulp but the year totals are 3.137e-8 off.
    Commands and reports:
    `/temp1/cfranken/flux_port_parity/python_ref_71a08395/fluxcom_x/compare/`.
  - SIF IAV `GPP_CO2_FLUX_ANOM`, `_NET` and `_SEAS` are differences of
    near-equal values. Per-value ulp gates and SEAS year totals carry no
    information for them (ANOM up to 687,854,529 ulp, SEAS year relative
    difference 1.12 against v3 × K_new/K_old); gate their month totals, and
    check them against their operands (≤ 2.64 Float32 ulps of the largest
    operand, `python_ref_71a08395/_tools/probe_sif_derived_operand_ulp.jl`).

## Running

```bash
julia --project=. scripts/validation/flux_port_parity/run_parity.jl \
    scripts/validation/flux_port_parity/s0_bitwise_pairs.toml \
    --json /temp1/cfranken/flux_port_parity/harness_runs/s0_bitwise_pairs.json \
    --markdown /temp1/cfranken/flux_port_parity/harness_runs/s0_bitwise_pairs.md
```

Write reports under `/temp1/cfranken/flux_port_parity/`, never next to the
references. Comparisons run one after the other, and memory is that of the
largest single comparison (see the comparator header). Both reports record the
git commit, whether tracked files are modified, and the sha256 of
`run_parity.jl` and of `compare_cs_flux_files.jl` as loaded, so a report names
its code version even while the scripts are untracked or edited.

Measured on 2026-10-10 (wurst, Julia 1.12.6, port worktree at `71a08395` plus
the uncommitted P0 files). The command was the one above, with
`--project=<port>/test`:

- Result: PASS, 2 of 2. Wall time 29.1 s for the whole process, peak RSS
  1.42 GB.
- `hourly_to_daily_c90_2021`: 365 stamps, bitwise identical, year total
  difference 0. The C90 mask gave 5180 coastal, 14372 polar and 29048 interior
  cells, with a coastal cell max of 0.
- `refine_c180_20211231`: 24 common stamps (720 more in the December file),
  bitwise identical.
- Reports: `/temp1/cfranken/flux_port_parity/harness_runs/s0_bitwise_pairs.{json,md}`.
- Rerun after the script hashes were added (same command, reports
  `/temp1/cfranken/flux_port_parity/review_final/harness_s0_bitwise_pairs.{json,md}`):
  PASS 2 of 2, 27.8 s, peak RSS 1.37 GB; comparator sha256 `3d38ccc4…`.
- Reruns after the two rounds of Codex-review changes to the comparator and
  the runner (same command, reports in
  `/temp1/cfranken/flux_port_parity/review_codex_fix/`): PASS 2 of 2 with the
  round-1 version (28.5 s, peak RSS 1.39 GB, `harness_s0_bitwise_pairs.*`,
  comparator sha256 `71cc8073…`), the round-2 version (28.9 s, 1.42 GB,
  `round2_harness_s0_bitwise_pairs.*`, `304a8cc8…`) and the final version
  (28.6 s, 1.36 GB, `round3_harness_s0_bitwise_pairs.*`, `a55c1de1…`), same
  results.

A second run exercised the reference scale, a variable without a time
dimension and the C30 mask on real data. The TOML is
`/temp1/cfranken/flux_port_parity/taskA/s0_gfed5_rescaled.toml`; its reports
sit next to it. It compares the S0 GFED5 C30 2014 file with the 2014–2024
product written with 44/12, on 365 common days:

- With `reference_scale = { FIRE_CO2_FLUX = "carbon_44_12_to_iupac" }`:
  - every value within 1 Float32 ulp, 94.5 % bitwise;
  - month totals within 2.1e-9 and the 2014 total within 1.9e-10;
  - `cell_area` bitwise identical.
- As stored: max 11327 ulp and a year-total difference of 6.752e-4 (= 1 − factor).
- Wall time 30.0 s, peak RSS 1.21 GB.

## Budget closure (diagnostics port, B2)

`budget_closure_parity.jl` runs `Analysis.budget_closure` on one run three
times and compares the first with the Python report (design §4.3):

```bash
julia --project=. scripts/validation/flux_port_parity/budget_closure_parity.jl \
    --config /temp1/cfranken/trendy_allmodels_c30_20260908/run.toml \
    --reference /temp1/cfranken/trendy_allmodels_c30_20260908/validation.json \
    --json /temp1/cfranken/diag_port/budget_closure/full_parity.json
```

- `legacy`: `LegacyDailyFluxFileEmission`, the Python arithmetic (flux-file
  `cell_area`, whole-day sums, `M_air = 0.02896546`).
- `runtime`: the library default, the run's own sources with the current
  `DRY_AIR_MOLAR_MASS = 0.0289644`.
- `run_constant`: the runtime changes times 0.02896546 / `DRY_AIR_MOLAR_MASS`,
  the storage scale a run made before commit `d552d3e0` applied.

Measured on 2026-10-11 (wurst, Julia 1.12.6, `--project=<port>/test`), on the
23-model TRENDY C30 run of 2026-09-08 (46 tracers, 3775 daily files):

| Run | Variant | Pass (strict) | Max relative error | Against the Python report |
|---|---|---|---|---|
| full | legacy | 45/46 (TEM NPP fails) | 1.27141e-4 | expected ≤ 7.1e-16, actual identical, tolerance 1.9e-11, same flags |
| full | runtime | 43/46 (TEM, CLM-FATES, ELM NPP fail) | 1.63756e-4 | expected changes 3.66e-5 smaller (the constant) |
| full | run_constant | 45/46 (TEM NPP fails) | 1.27155e-4 | expected changes ≤ 5.48e-8 from the legacy ones |
| smoke (3 days) | legacy | 46/46 | 1.96004e-4 | expected ≤ 3.8e-16, actual identical, tolerance 2.4e-9, same flags |
| smoke | runtime | 46/46 | 1.59392e-4 | |
| smoke | run_constant | 46/46 | 1.95981e-4 | expected changes ≤ 6.2e-8 from the legacy ones |

The tolerance differs from the Python value only in the carrier term: the
Julia air mass is the binary's first window, the Python one
`total_mass(t₀) / carrier`. Reports and logs:
`/temp1/cfranken/diag_port/budget_closure/v2/{full,smoke}_parity.{json,log}`
(17:45 wall, 1.6 GB peak RSS for the full run). The run is Float32 with
3600 s windows of 3 to 8 steps; integrating its 7-step windows with the
Float32 step `Float32(3600)/Float32(7)` (3599.99994 s per window) instead of
3600 s moves the runtime expected changes by at most 1.3e-10 and none of the
numbers above. `scripts/diagnostics/tracer_budget_closure.jl series run.toml`
on the same run reports the runtime row (43/46 strict, all 46 operational) and
exits 1.

## Coastal masks per grid

The coastal class needs a land mask in the cs_native layout on the same grid.
The mask is read with `mask = { var, file, threshold }` (on the command line:
`--mask-var`, `--mask-file`, `--mask-threshold`). Only where the land class
changes matters, so a mask that is 1 on ocean gives the same coastal cells.
Measured with `/temp1/cfranken/flux_port_parity/taskA/probe_mask_sources.jl`
(command in its header; output `probe_mask_sources.log` next to it), using the
comparator's own `read_mask` and `cell_categories` on a GEOS-native GMAO mesh:

| Grid | Recommended source | Variable, threshold | Land cells | Coastal / polar / interior | Stored centres vs mesh |
|---|---|---|---|---|---|
| C90 | `/temp1/cfranken/sif_gpp_iav/fluxes/sif_gpp_daily_co2flux_c90_2018_2025_v3.nc` | `land_fraction`, 0.05 | 16,343 of 48,600 | 5180 / 14372 / 29048 | 2.32e-4° (limit 0.1°) |
| C30 | `/temp1/cfranken/trendy_transcom_c30_20260908/transcom_weights_c30.nc` | `nearest_land_fallback`, 0 | 1935 of 5400 (value 0) | 903 / 1560 / 2937 | 4.1e-5° (limit 0.3°) |
| C180 | none found | | | | |

- C90, `land_fraction` (Float32):
  - It is the vegetated land fraction of the FLUXCOM-X pilot, the field the SIF
    products use, and 0.05 is their land-cell rule.
  - 1006 land cells lie south of 60°S, so Antarctica counts as land.
  - No land cell has its centre north of 60°N between 75°W and 10°W
    (Greenland, Iceland, eastern Baffin Island).
  - All 484 cells of the box 15–30°N, 15°W–30°E (Sahara) are land.
  - With threshold 0 there are 17,446 land cells and 6020 coastal cells.
- C30, `nearest_land_fallback` (Int8):
  - No land-fraction file exists at C30.
  - The variable is 1 where no 0.25° sample of a TransCom-11 land region falls
    in the cell. It is an ocean indicator, which gives the same coastal cells
    as a land indicator.
  - No land cell lies south of 60°S or in the Greenland box above, so the
    Antarctic and Greenland coasts are not coastal under this mask.
  - All 53 cells of the Sahara box are land.
- C180: no land-fraction file was found.
  - The pilot TransCom tags file
    (`/kiwi-data/Data/satellite/FLUXCOM-X/X-BASE/2021/pilot/sif_gpp_hourly_co2flux_c180_dec2021_transcom_tags.nc`)
    stores `transcom_region_id`: box-based region ids, 0 for unassigned, ocean
    or excluded cells. Whether it works as a land mask was not checked.

## Adding a driver

1. Build the Julia candidate into a new directory under
   `/temp1/cfranken/flux_port_parity/` (for example `julia_p5/gfed5_c30_2014/`).
   Never write next to or over a reference.
2. Take the reference from `MANIFEST.json` (S0) or from the reference column of
   design §4.2. Check the facts above: span, LPJ-GUESS, 44/12.
3. Add one `[[comparison]]` per level and variable set to a TOML file in this
   directory, for example `p5_gfed5.toml`. Take the tolerances from design §4.2
   and name the level.
4. If an L1 comparison needs a legacy emulator, put it in this directory (for
   example `emulators/legacy_geometry_target.jl`). It plugs into the Julia
   library by dispatch on the library's target or remap type. Emulators never
   go into `src/`.
5. Run `run_parity.jl` with `--json` and `--markdown` into
   `/temp1/cfranken/flux_port_parity/`. Copy the measured numbers and the
   command into the facts memo (`docs/memos/2026-10-xx_python_flux_port_parity.md`)
   and into the PR.

## Lifetime

The legacy emulators live only in this directory and are deleted in step P11,
the production switch. The comparator `compare_cs_flux_files.jl` stays for the
long term.
