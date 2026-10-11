# Python surface-flux port, step P0: measured facts (2026-10-10)

Measured record of step P0 of the surface-flux port
(`2026-10-10_python_flux_port_design.md`): data and code probes, the fresh
Python references (S0) and their reproduction, the reference-numbers TOML, and
the validation runs of the file comparator and the parity harness. Every item names the command or
script that produced it and where its output lives. No plans are recorded
here.

## 0. Setting

- Code: worktree `AtmosTransportModel-port`, branch `port/p0` at `71a08395`.
  `git diff --quiet 71a08395 -- scripts/preprocessing` passes, so the Python
  scripts run are those of `71a08395`.
- Host: `wurst.gps.caltech.edu`. No GPU was used. During the daily-chunk
  timing probes the load average was 21–26 on 128 cores (a golden run and the
  test suite were running); during the hourly-chunk probes (§7.1) it was 6–10.
- Julia 1.12.6 with the worktree's Manifest: NCDatasets 0.14.15, CFTime 0.2.10,
  netCDF-C 4.10.0. `J` below stands for
  `$HOME/.juliaup/bin/julia --project=/home/cfranken/code/gitHub/AtmosTransportModel-port`.
- Python for the S0 runs: `/usr/bin/python3` 3.9.25, numpy 1.26.4, netCDF4
  1.5.8, scipy 1.9.1, netCDF-C 4.8.1, HDF5 1.12.1 (scipy warns that it expects
  numpy < 1.25). Python for the data probes: `/usr/bin/python3 -I`, netCDF4
  1.5.8, cftime 1.5.2, numpy 1.23.5.
- All inputs were opened read-only. All outputs are under
  `/temp1/cfranken/flux_port_parity/` (only there, not in the repository):

| Path (under `/temp1/cfranken/flux_port_parity/`) | Content |
|---|---|
| `probes/scripts/` | probe scripts |
| `probes/results/` | raw data-probe results (JSON, text) |
| `probes/p0_data_probes.md`, `probes/p0_julia_probes.md` | probe reports |
| `probes/p0_chunk_*.toml`, `probes/p0_coarsen_results.toml`, `*.log` | Julia probe results |
| `python_ref_71a08395/` | S0 references, `MANIFEST.json`, tools in `_tools/` |
| `probes/s0_reference_numbers/`, `probes/s0_compare_existing.txt` | reference-number JSONs; S0-vs-existing comparison |
| `probes/compare_cs_flux_files/` | comparator reports (JSON + text) of the first comparator version |
| `probes/reference_items/` | TOML fragments of the §4.2 items verified in Julia (§10) |
| `python_ref_71a08395/{fluxcom_x,sif_iav}/compare/` | comparisons of the two late S0 runs with the existing products (§9.3) |
| `taskA/` | comparator and harness development: coastal-mask probe, GFED5 rescale run, mutation copies (§11.3, §11.4) |
| `harness_runs/` | parity-harness run on the two bitwise S0 pairs (§12) |
| `review_p0/` | reruns and comparator runs of the P0 review |
| `review_final/` | test logs and comparator reruns of the final P0 review (§11.1, §11.3) |
| `review_codex_fix/` | test logs, comparator mutation copies and the harness rerun after the Codex-review fixes (§11.5) |

Decompressed TRENDY copies were written to `probes/staging/` one at a time and
deleted after each probe; the chunk-benchmark files in `probes/chunks/` were
deleted by the script.

---

## 1. TRENDY v14 S3 time axes

Commands, from `probes/scripts/`:
- `/usr/bin/python3 -I probe_trendy_meta.py --out ../results/trendy_meta_uncompressed.json`
  (metadata of every uncompressed gpp/ra/rh and LPJ-GUESS `arh` file);
- `bash probe_compressed.sh` (decompress → `ncdump -h` →
  `probe_trendy_meta.py --file` → `rm`, for SDGVM, VISIT and iMAPLE);
- `/usr/bin/python3 -I probe_decimal_year_residuals.py ../results/decimal_year_residuals.json`;
- `$J --threads=1 probe_julia_time_decode.jl ../results/julia_time_decode.json` and
  `$J --threads=1 probe_julia_months_since.jl > ../results/julia_months_since.txt`.

Python's month keys were taken from the pilot module's own functions
(`find_flux_file`, `month_keys`, `annual_keys`, `coordinate`, `flux_variable`,
`validate_units` in `scripts/preprocessing/prepare_trendy_s3_gpp_ter_c90.py`).

Of 25 model directories, 23 have gpp/ra/rh files; ISBA-CTRIP and OCN have none.
For all 23, Python's decoded month sequence is contiguous and ends in 2024-12.

| Model | time units | calendar | dtype, n | Stamp in the month | Python branch | NCDatasets 0.14.15 / CFTime 0.2.10 |
|---|---|---|---|---|---|---|
| CABLE-POP | `months since 1700-1-16 12:00:00` | 365_day | f64, 3900 | integer k | months-since regex | silently wrong: 3201/3900 month keys wrong, last stamp 2025-03-05 |
| CARDAMOM | none | none | int32, 264 (1 … 264) | index | auxiliary `Time` ("Monthly time step given in days since 01/01/2003") | raw Int32 |
| CLASSIC | `days since 1700-12-31` | 365_day | f64, 3888 | last day of the month | cftime | correct |
| CLM | `hours since 1700-01-16 12:00:00.000000` | noleap | int32, 3900 | day 15/16 | cftime | correct |
| CLM-FATES | `days since 1701-01-01` | 365_day | f32, 3888 | day 15/16 | cftime | correct |
| DLEM | `days since 1700-01-01 00:00:00` | gregorian | f64, 3900 | day 1 | cftime | correct |
| ED | `days since 1700-01-01 00:00:00` | missing | f32, 3900 | day 15 | cftime | correct |
| ELM | `days since 0001-01-01` | noleap | f64, 3900 | day 15/16 | cftime | correct |
| ELM-FATES | none | none | f64, 3888 (1701.0 … 2024.91667) | month start, k/12 | decimal years | raw Float64 |
| IBIS | `days since 1700-01-01 00:00:00` | noleap | f32, 3900 | day 16 | cftime | correct |
| iMAPLE | `months since 1700-1-1` | missing | **int16**, 3900 | month start | months-since regex | silently wrong: 296/3900 keys wrong |
| ISAM | `months since 1700-01-01 00:00:00` | proleptic_gregorian | f64, 3900 (0.5 … 3899.5) | k + ½ | months-since regex | 0 wrong keys (stamps off mid-month, e.g. 2024-12-16T11:54) |
| JSBACH | `hours since 1701-1-16 12:00:00` | proleptic_gregorian | f64, 3888 | day 15/16 12:00 | cftime | correct |
| JULES-ES | `days since 1700-01-01 00:00:00` | noleap | f64, 3900 | day 15/16 | cftime | correct |
| LPJ-GUESS gpp, ra | `days since 1700-01-01` | noleap | f64, 3900 | day 16 | cftime | correct |
| LPJ-GUESS arh | `days since 1700-01-01` | noleap | f64, 325 (181 … 118441) | annual, 1 July | cftime (`annual_keys`) | correct (1700 … 2024) |
| LPJml | `months since 1700-1-1` | 365_day | f64, 3900 | integer k | months-since regex | silently wrong: 2436/3900 keys wrong, last stamp 2025-02-18 |
| LPJwsl | `days since 1690-01-01` | gregorian | f32, 3900 | day 1 | cftime | correct |
| LPX-Bern | `days since 1700-01-01 00:00:00` | noleap | int64, 3900 | day 1 | cftime | correct |
| ORCHIDEE | `days since 1700-01-01 00:00:00` | noleap | f32, 3900 | day 16 | cftime | correct |
| SDGVM | `hours since 1900-01-01 00:00:00` | 360_day | f64, 1500 (360 … 1079640, step 720) | day 16 | cftime | correct |
| TEM | `days since 1700-01-01 00:00:00` | 365_day | f32, 3900 | day 15/16 | cftime | correct |
| VISIT | `months since 1860-01-01` | missing | f64, 1980 (0.5 … 1979.5) | k + ½ | months-since regex | 0 wrong keys (2024-12-15T17:51) |
| VISIT-UT | `years since AD 0-Jan-1st (double)` | none | f64, 3900 (1700.04167 … 2024.95833) | mid-month, (k + ½)/12 | decimal years | warning "cannot parse time units", raw Float64 |

Facts from these runs:
- **CFTime 0.2.10 decodes `months since` without error or warning, with a
  fixed month of 30.436849895833333 days**, whatever the calendar
  (`julia_months_since.txt`). With `365_day`, 3899 "months" after 1700-01-01
  decode to `DateTimeNoLeap(2025-02-18T06:39:57.069)`; the true month start is
  2024-12-01. The P0 review reproduced this with NCDatasets 0.14.15: with no
  calendar, 3899 months decode to 2024-12-01T06:39:57; with `365_day`, to
  2025-02-18T06:39:57.
- **Decimal years** (`decimal_year_residuals.json`): ELM-FATES (month starts)
  has 1296 of 3888 values just below the integer month; the largest residual
  is 9.094947017729282e-13 month. A plain floor of 12·(v − ⌊v⌋) gives 2592 bad
  steps; `floor(x + 1e-5)` (Python) is contiguous. VISIT-UT (mid-month) has the
  same maximum residual, 1300 negative residuals, and a plain floor has no bad
  step.
- Time `bounds` variables exist only for CLM (`time_bounds`), CLM-FATES, ED and
  ELM (`time_bnds`); none of the nine compressed files has one.
- The time variables of CLM-FATES, ED, LPJwsl, TEM and ELM-FATES carry
  `_FillValue = NaN`, so NCDatasets returns `Union{Missing, …}` element types;
  no missing value was observed.
- SDGVM's global attributes include `calendar = "no leap years, 360 day years"`.

## 2. TRENDY v14 S3 grids

Commands, from `probes/scripts/`:
`/usr/bin/python3 -I probe_trendy_grid_regularity.py ../results/trendy_grid_regularity.json`
(plus the metadata JSONs of Section 1).

| Model | Shape (t, lat, lon) | lat: dtype, order, range | Lat grid | lon: dtype, range | Lon class |
|---|---|---|---|---|---|
| CABLE-POP | (3900, 180, 360) | f64, asc, −89.5 … 89.5 | regular, cell-centred | f64, −179.5 … 179.5 | −180..180 |
| CARDAMOM | (264, 360, 720) | f64, asc, −89.75 … 89.75 | regular | f64, −179.75 … 179.75 | −180..180 |
| CLASSIC | (3888, 180, 360) | f64, asc, −89.5 … 89.5 | regular | f64, **−180 … 179** (whole degrees) | −180..180 |
| CLM | (3900, 192, 288) | f32, asc, −90 … 90 | **pole-centred**, −90 + k·180/191 (Float32 rounding, 3.8e-6°) | f32, 0 … 358.75 | 0..360, centre on 0° |
| CLM-FATES | (3888, 80, 144) | f32, asc, **−59.684 … 90** | f19 rows 16 … 95 (3.6e-6°); north pole only | f32, 0 … 177.5 then −180 … −2.5 | **wrapped**, centre on 0° |
| DLEM | (3900, 360, 720) | f64, asc, −89.75 … 89.75 | regular | f64, −179.75 … 179.75 | −180..180 |
| ED | (3900, 360, 720) | f64, asc, −89.75 … 89.75 | regular | **f32**, −179.75 … 179.75 | −180..180 |
| ELM | (3900, 192, 288) | f64, asc, −90 … 90 | **pole-centred** (f64 values that are Float32-rounded) | f64, −180 … 178.75 | −180..180, centre on −180° |
| ELM-FATES | (3888, 80, 144) | f32, asc, **−59.684 … 90** | f19 truncated (3.6e-6°); north pole only | f32, 0 … 357.5 | 0..360, centre on 0° |
| IBIS | (3900, 360, 720) | f32, asc, −89.75 … 89.75 | regular | f32, −179.75 … 179.75 | −180..180 |
| iMAPLE | (3900, **181**, 360) | f32, asc, −90 … 90 | **pole-centred**, step exactly 1° | f32, −179.5 … 179.5 | −180..180 |
| ISAM | (3900, 360, 720) | f64, asc, −89.75 … 89.75 | regular | f64, 0.25 … 359.75 | 0..360 |
| JSBACH | (3888, 96, 192) | f64, asc, −88.572 … 88.572 | **Gaussian** (T63; equals Gauss–Legendre to 1.4e-14°; spacing varies by 0.0156°) | f64, −180 … 178.125 | −180..180, centre on −180° |
| JULES-ES | (3900, 360, 720) | f64, asc, −89.75 … 89.75 | regular | f64, −179.75 … 179.75 | −180..180 |
| LPJ-GUESS | (3900, 360, 720); arh (325, 360, 720) | f64, asc, −89.75 … 89.75 | regular | f64, −179.75 … 179.75 | −180..180 |
| LPJml | (3900, 360, 720) | f64, asc, −89.75 … 89.75 | regular | f64, −179.75 … 179.75 | −180..180 |
| LPJwsl | (3900, 360, 720) | f32, asc, −89.75 … 89.75 | regular | f32, −179.75 … 179.75 | −180..180 |
| LPX-Bern | (3900, 360, 720) | f32, asc, −89.75 … 89.75 | regular | f32, −179.75 … 179.75 | −180..180 |
| ORCHIDEE | (3900, 360, 720) | f32, asc, −89.75 … 89.75 | regular | f32, −179.75 … 179.75 | −180..180 |
| SDGVM | (1500, 180, 360) | f64, asc, −89.5 … 89.5 | regular, step exactly 1° | f64, −179.5 … 179.5 | −180..180 |
| TEM | (3900, 360, 720) | f32, asc, −89.75 … 89.75 | regular | f32, −179.75 … 179.75 | −180..180 |
| VISIT | (1980, 360, 720) | f64, **desc**, 89.75 … −89.75 | regular, step exactly −0.5° | f64, −179.75 … 179.75 | −180..180 |
| VISIT-UT | (3900, 360, 720) | f32, **desc**, 89.75 … −89.75 | regular | f32, −179.75 … 179.75 | −180..180 |

- In every model, gpp, ra and rh (or `arh`) have bitwise-identical lat and lon
  arrays.
- For every model the stored lon equals lon₀ + k·360/n within 5.7e-14° (the
  limit of the probe's radian round trip).
- Where a centre sits on 0° or ±180°, the first longitude face is −180.5°
  (CLASSIC), −0.625° (CLM), −1.25° (CLM-FATES, ELM-FATES), −180.625° (ELM) or
  −180.9375° (JSBACH).
- ELM's `lat_bnds` for the pole rows extend to ±90.471°, and its `lon_bnds`
  row 0 is [179.375, −179.375].
- iMAPLE: a uniform 181-row mesh over −90 … 90 would put row centres up to
  0.5·180/181 = 0.497° from the file centres (arithmetic; not measured on the
  runtime loader).

**JSBACH Gaussian faces** (`probe_jsbach_gaussian_faces.py`,
`probe_jsbach_area_impact.py`; results `jsbach_gaussian_faces.json`,
`jsbach_area_impact.json`). Python's Voronoi-midpoint faces differ from the
Gauss-weight faces (sin φ_face = −1 + Σw) by up to 0.06526683053972704°; the
polar row is 5.787 % larger under the midpoint rule. 2021 totals with L = 1:

| Component | Midpoint faces (Pg C) | Gauss-weight faces (Pg C) | Relative difference |
|---|---|---|---|
| GPP | 140.82281997378115 | 140.82904213133148 | 4.418e-5 |
| RA | 74.7286556599189 | 74.73195713099534 | 4.418e-5 |
| RH | 59.158746330193964 | 59.161360557419286 | 4.419e-5 |

The midpoint numbers reproduce `annual_totals_PgC` of the Python 2021 JSBACH
audit JSON (`/kiwi-data/Data/groupMembers/cfranken/AtmosTransport/fluxes/TRENDYv14/S3/C90/2021/`;
140.82281997378118 / 74.72865565991887 / 59.158746330193964) to 2e-16.

**CLASSIC longitude registration** (`probe_classic_lon_registration.py`,
`probe_classic_lon_shift_scan.py`, `probe_classic_lon_shift_scan_025.py`,
`probe_lon_shift_control.py`, `probe_lonlat_shift_2d.py {CLASSIC,CABLE-POP}`;
results `classic_lon_*.json`, `cablepop_lon_shift_control.json`,
`lonlat_shift_2d_*.json`). CLASSIC's file has the global attribute
`node_offset = 1`. Land-mask correlation against the FLUXCOM-X 0.25° land
indicator (land fraction > 0), |lat| < 60°, with CLASSIC column k shifted to
[k − ½ + s, k + ½ + s]:

| Shift s (°) | −1.0 | −0.75 | −0.5 | −0.25 | 0 | +0.25 | +0.5 | +0.75 | +1.0 |
|---|---|---|---|---|---|---|---|---|---|
| CLASSIC `sftlf`, r | 0.96726 | 0.97693 | 0.98250 | **0.98298** | 0.97829 | 0.96910 | 0.95736 | 0.94423 | 0.93084 |
| CABLE-POP control, r | 0.92971 | 0.94328 | 0.95480 | 0.96297 | **0.96598** | 0.96318 | 0.95546 | 0.94410 | 0.93110 |

The 2-D lat × lon scan gives the best lat shift 0 and lon shift −0.25°. Against
the ORCHIDEE 0.5° land fraction: r = 0.9649 (−0.5), 0.9603 (0), 0.9400 (+0.5).

## 3. Units strings and fill attributes

From the metadata JSONs of Section 1.

| Model | Flux dtype | Units string | `_FillValue` | `missing_value` |
|---|---|---|---|---|
| CABLE-POP | float32 | `kg m-2 s-1` | −99999 | −99999 |
| CARDAMOM | float64 | `kg.m-2.s-1` | −99999 | — |
| CLASSIC | float64 | `kg m$^{-2}$ s$^{-1}$` | −99999 | −99999 |
| CLM | float32 | `kg m-2 s-1` | −99999 | — |
| CLM-FATES | float32 | `kg m-2 s-1` | −99999 | −99999 |
| DLEM | float64 | `kg m-2 s-1` | — | — |
| ED | float32 | `kg m-2 s-1` | −99999 | — |
| ELM | float32 | `kg m-2 s-1` | −99999 | — |
| ELM-FATES | float32 | `kg m-2 s-1` | NaN | — |
| IBIS | float32 | `kg m-2 s-1` | −99999 | — |
| iMAPLE | float32 | `kg m-2 s-1` | — | −99999 |
| ISAM | float64 | `kg  m-2 s-1` (two spaces) | −9999 | −9999 |
| JSBACH | float32 | `kg m-2 s-1` | −99999 | −99999 |
| JULES-ES | float32 | `kg m-2 s-1` | −999 | — |
| LPJ-GUESS (gpp, ra, arh) | float32 | `kg m-2 s-1` | −99999 | — |
| LPJml | float32 | `kg m-2 s-1` | −99999 | — |
| LPJwsl | float64 | `kg m-2 s-1` | NaN | — |
| LPX-Bern | float32 | `kg m-2 s-1` | −99999 | — |
| ORCHIDEE | float32 | `kg m-2 s-1` | −99999 | — |
| SDGVM | float32 | gpp, rh: `kg m-2 s-1`; ra: `kg m-2 s-1, per unit land area occupied by the PFT` | −99999 | — |
| TEM | float32 | `kg m-2 s-1` | −99999 | — |
| VISIT | float32 | `kg m-2 s-1` | — | −99999 |
| VISIT-UT | float32 | `kg m-2 s-1` | — | −99999 |

No flux variable has `scale_factor`, `add_offset`, `valid_range`, `valid_min`
or `valid_max`. Python's `validate_units` accepts the SDGVM `ra` string (it
looks only for the tokens `kg`, `m-2`, `s-1`).

## 4. Compressed TRENDY inputs

Command: `bash probes/scripts/probe_compressed.sh`; log
`probes/results/compressed_staging_log.txt`, headers `ncdump_h_*.txt`.

| File | Compressed bytes | Decompressed bytes | Decompression wall time (s) |
|---|---|---|---|
| `SDGVM_S3_{gpp,ra,rh}.nc.gz` | 52,139,806 / 50,697,596 / 44,287,212 | 388,817,220 / 388,817,268 / 388,817,224 | 1.54 / 1.59 / 1.52 |
| `VISIT_S3_{gpp,ra,rh}.nc.gz` | 330,385,362 / 445,903,489 / 443,937,834 | 2,052,889,100 each | 8.12 / 8.61 / 8.59 |
| `iMAPLE_S3_{gpp,ra,rh}.nc.tar.gz` | 235,580,465 / 264,365,328 / 264,730,529 | 1,016,514,760 / 1,016,514,772 / 1,016,514,760 | 4.74 / 4.87 / 4.86 |

All nine decompressed files are `NETCDF3_CLASSIC`. Each iMAPLE tar holds exactly
one member, `iMAPLE_S3_<kind>.nc` (`tar tzvf`). Within each model, gpp, ra and
rh share identical time and coordinate metadata.

## 5. GFED5 and FLUXCOM-X coordinates

Command: `/usr/bin/python3 -I probes/scripts/probe_coord_snapping.py probes/results/coord_snapping.json`,
over 62 GFED5 files (finalized `GFED5.1_monthly_{1997..2022}.nc`, NRT
`GFED5NRTeco_CMB_YYYY-MM.nc` for 2023–2025) and the 18 FLUXCOM-X X-BASE
`*_025_{daily,monthlycycle}.nc` files of 2014–2021. The exact centre is
origin + sign·(k + ½)·res with res = 180/n_lat or 360/n_lon.

| Product | lat/lon dtype | n_lat × n_lon | Order | max \|stored − exact\| lat / lon (°) | Bounds variables | Coordinate attributes |
|---|---|---|---|---|---|---|
| GFED5.1 finalized 1997–2000 | float32 | 180 × 360 | ascending | 0 / 0 | none | `degrees_north` / `degrees_east` |
| GFED5.1 finalized 2001–2022 | float32 | 720 × 1440 | ascending | 0 / 0 | none | same |
| GFED5 NRT 2023-01 … 2025-12 | float64 | 720 × 1440 | ascending | 0 / 0 | none | **no `units`, no `long_name`** |
| FLUXCOM-X X-BASE 2014–2021 | **float64** | 720 × 1440 | lat **descending** | 7.1e-15 / 2.84e-14 (≤ 1.1e-13 cells; 208/720 lat and 624/1440 lon values not bit-exact) | `lat_bnds`, `lon_bnds` (float64; `lon_bnds` from −179.99999999999997 to 180.00000000000003) | `lon` has **`units = degrees_north`, `long_name = latitude`** |

- The GFED5 Float32 values equal the exact centres bit for bit.
- The finalized 2001–2022 grid and the NRT grid have identical centres.
- Midpoints of the stored centres differ from the exact faces by 0 (GFED5)
  and ≤ 2.84e-14° (FLUXCOM-X).
- GFED5 finalized: `C(time = 12, lat, lon)`, float32, `units = "g C per month"`,
  no `_FillValue`, time `hours since 1800-01-01` without calendar. NRT:
  `EM(time = 1, lct = 16, lat, lon)`, float64, `_FillValue = NaN`, no units,
  time int64 `days since 2023-01-01 00:00:00`, proleptic_gregorian.

## 6. FATES southern row

Command: `/usr/bin/python3 -I probes/scripts/probe_fates_southern_row.py probes/results/fates_southern_row.json`
(full raw arrays, 3888 × 80 × 144 per file, no auto-masking; "valid" = not a
fill value and finite).

| File | Fill attributes | Row −59.684° (southernmost): valid / non-zero values / non-zero cells / max \|value\| | Row −57.789° | Row −55.895° |
|---|---|---|---|---|
| CLM-FATES gpp | `_FillValue = missing_value = -99999` | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 2 cells valid in all months; max 4.34e-9 kg m⁻² s⁻¹ |
| CLM-FATES ra | same | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 2 cells; max 1.51e-9 |
| CLM-FATES rh | same | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 2 cells; max 1.51e-9 |
| ELM-FATES gpp | `_FillValue = NaN` | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 2 cells; max 2.67e-8 |
| ELM-FATES ra | same | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 2 cells; max 1.14e-8 |
| ELM-FATES rh | same | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 2 cells; max 6.63e-9 |

All 144 cells of the southernmost row are fill in all 3888 months, and in the
window 2013-12 … 2024-12. The latitude centres are those of the CESM f19 grid,
rows 16 … 95: centre_k = −90 + (k + 16)·180/95, stored values within 3.6e-6°.
Python's Voronoi rule gives row 0 the faces [−90°, −58.7368°]; the physical
southern face is −90 + 15.5·180/95 = −60.631578947°. Python's row-0 band area
is 8.70× the physical one.

## 7. Chunk layout of daily `cs_native` files

Script `probes/scripts/p0_chunk_benchmark.jl`. Commands:

```bash
P=/temp1/cfranken/flux_port_parity/probes
$J $P/scripts/p0_chunk_benchmark.jl synthetic $P/p0_chunk_results_synthetic.toml   # run 1: same command
$J $P/scripts/p0_chunk_benchmark.jl clm_gpp   $P/p0_chunk_results_clm_gpp.toml
```

Method:
- One Float32 variable `(Xdim, Ydim, nf, time) = (90, 90, 6, 3776)` per file,
  deflate 1, shuffle, no `_FillValue`, written one daily slice per call.
  `ncdump -hs` confirmed the chunk sizes; slices 1, 1000 and 3776 read back
  bit-identical.
- Content: synthetic (smooth seasonal field on 32.9 % "land" cells with 10 %
  multiplicative noise, zeros elsewhere), and the real `GPP_CO2_FLUX` of
  `/temp1/cfranken/sif_gpp_iav/fluxes/trendy_v2/TRENDYv14_S3_CLM_gpp_ter_nee_daily_co2flux_c90_2014_2024.nc`
  (slices 1:3776; 68.0 % zeros).
- Access copied from `src/Models/initial_conditions/surface_flux_native.jl`:
  open, `raw_var[:, :, p, kr]` for p = 1:6, close. "Loader" = fresh open per
  call (open + close took 1.2–4.6 ms); "sequential" = file kept open,
  consecutive windows from slice 1000.
- Warm page cache (each file read once as raw bytes before timing). Chunk cache
  per variable: 67,108,864 B, 1000 slots, preemption 0.75.
- Peak RSS 2.0 GB (synthetic), 2.4 GB (CLM).

Medians in ms (CLM content; uncompressed variable 734,054,400 B):

| Chunks | File bytes | Write (s) | Loader k=1 | k=31 | k=366 | k=3776 | Sequential k=1 (mean) | Sequential k=31 (mean) |
|---|---|---|---|---|---|---|---|---|
| (90,90,1,1) | 246,271,592 | 8.84 | 0.97 | 22.8 | 265 | 2695 | 0.77 | 22.4 |
| (90,90,1,24) | 219,162,849 | 6.52 | 13.0 | 26.5 | 223 | 2422 | 0.99 | 22.8 |
| (90,90,1,32) | 218,610,301 | 6.49 | 16.9 | 34.5 | 232 | 2377 | 0.97 | 21.4 |
| (90,90,6,1) | 241,108,416 | 6.69 | 0.81 | 19.6 | 1287 | 13438 | 0.72 | 20.7 |

Synthetic content (run 2), same columns:

| Chunks | File bytes | Write (s) | Loader k=1 | k=31 | k=366 | k=3776 | Sequential k=1 | Sequential k=31 |
|---|---|---|---|---|---|---|---|---|
| (90,90,1,1) | 242,521,799 | 7.83 | 0.87 | 21.8 | 249 | 2730 | 0.77 | 28.6 |
| (90,90,1,24) | 233,446,864 | 6.53 | 14.2 | 29.5 | 242 | 2611 | 1.07 | 25.2 |
| (90,90,1,32) | 233,149,157 | 6.55 | 18.9 | 38.4 | 249 | 2555 | 1.06 | 22.9 |
| (90,90,6,1) | 237,304,260 | 6.26 | 0.81 | 20.0 | 1315 | 13667 | 0.74 | 22.9 |

- Repeatability: synthetic run 1 vs run 2 for (90,90,1,1): 0.89/0.87, 21.1/21.8,
  250/249, 2737/2730 ms; the other layouts agree within 4 %. The k = 31 numbers
  have outliers up to 3× the median (host load).
- (90,90,6,1) is 4.9–5.8× slower for k = 366 and 3776. Each of its chunks
  (194,400 B uncompressed) is needed six times by the panel-by-panel reads; 366
  slices need 71,150,400 B, more than the 64 MiB cache (345 chunks).
- (1,24) and (1,32) cost 13.0–18.9 ms per k = 1 loader read, because each panel
  read decompresses a 24- or 32-slice chunk. Kept open, their median is
  0.14 ms and their maximum 13–19 ms (first slice of each chunk).

### 7.1 Hourly files (8760 slices)

Script `probes/scripts/p0_chunk_benchmark_hourly.jl`; tables by
`probes/scripts/p0_chunk_hourly_tables.jl`; full report in
`probes/p0_julia_probes.md` §3. Commands (each under `/usr/bin/time -v`):

```bash
P=/temp1/cfranken/flux_port_parity/probes
$J $P/scripts/p0_chunk_benchmark_hourly.jl clm_gpp_hourly $P/p0_chunk_hourly_results_clm_gpp.toml
$J $P/scripts/p0_chunk_benchmark_hourly.jl clm_ter_hourly $P/p0_chunk_hourly_results_clm_ter.toml
$J $P/scripts/p0_chunk_benchmark_hourly.jl clm_gpp_hourly $P/p0_chunk_hourly_results_clm_gpp_run2.toml   # repeat
```

Method: one Float32 variable `(90, 90, 6, 8760)` per file, deflate 1, shuffle,
no `_FillValue`, written one output day (24 slices) per call; content read
read-only from
`/kiwi-data/Data/groupMembers/cfranken/AtmosTransport/fluxes/TRENDYv14/S3/C90/2021/TRENDYv14_S3_CLM_gpp_diurnal_ter_daily_hourly_co2flux_c90_2021.nc`
(`GPP_CO2_FLUX`, diurnal, 77.4 % zeros; `TER_CO2_FLUX`, constant within each
UTC day). Same loader pattern and warm cache as above; k = 25 and 745 are the
windows `_needed_slice_window` returns for one-day and one-month runs when the
end stamp is in the file. The existing Python file (chunks `(90,90,1,24)` in
Julia order, deflate 2) was timed read-only as a baseline. Wall 3:08–3:32 per
run, peak RSS 4.11 GB.

Medians in ms (fresh open per read):

| Variable, chunks | File MB | k=1 | k=24 aligned | k=25 aligned | k=744 aligned | k=8760 |
|---|---|---|---|---|---|---|
| GPP (90,90,1,1) | 451.7 | 0.81 | 14.9 | 15.7 | 470 | 5741 |
| GPP (90,90,1,24) | 430.8 | 12.3 | 13.2 | 25.7 | 417 | 5177 |
| GPP (90,90,6,1) | 446.3 | 0.76 | 14.1 | 15.0 | 2425 | 28890 |
| TER (90,90,1,1) | 574.9 | 0.87 | 12.7 | 13.7 | 424 | 5060 |
| TER (90,90,1,24) | 55.2 | 5.32 | 6.02 | 11.5 | 202 | 2563 |
| TER (90,90,6,1) | 565.3 | 0.78 | 14.5 | 15.2 | 2550 | 30316 |

- The GPP repeat agrees within 2 % and gives identical file sizes.
- `(90,90,1,24)` makes the day-constant TER 10.4× smaller and diurnal GPP 4.6 %
  smaller than `(90,90,1,1)`, and is faster for k ≥ 744 (GPP 9–11 %, TER about
  2×). Windows crossing 00 UTC (k = 25, unaligned k = 24) cost GPP 25.2–25.7 ms
  against 14.7–15.7 ms.
- `(90,90,6,1)` is 5.0–6.0× slower than `(90,90,1,1)` for k ≥ 744.
- Kept open, per-slice reads of `(90,90,1,24)` average 0.66 ms (GPP) and 0.36 ms
  (TER) against 0.74–0.75 ms for `(90,90,1,1)`.
- The existing Python files read within 2 % of `(90,90,1,24)` deflate 1 for
  k ≥ 744 and up to 9 % slower for 24- and 25-slice windows.

## 8. Equivalence of the two cubed-sphere coarsening kernel sets

Script `probes/scripts/p0_coarsen_equivalence.jl`; command
`$J --threads=4 $P/scripts/p0_coarsen_equivalence.jl $P/p0_coarsen_results.toml`
(log `p0_coarsen.log`). Kernels were called as `AtmosTransport.Preprocessing.<name>`.

- **Set A**, `src/Preprocessing/transport_binary/cubed_sphere_coarsen.jl`:
  `_coarsen_sum_cells3!`, `_coarsen_weighted_cells2!`, `_coarsen_xfaces!`,
  `_coarsen_yfaces!`; only caller `_coarsen_nested_cs_window!` in
  `coarsen_nested_cs_transport_binary`.
- **Set B**, `src/Preprocessing/transport_binary/geos_cs_resolution.jl`:
  `_coarsen_sum3!`, `_coarsen_area_weighted2!`, `_coarsen_area_weighted3!`,
  `_coarsen_xface_sum!`, `_coarsen_yface_sum!`; only callers the
  `GEOSCSBlockCoarsenStrategy{R}` methods.

Code reading: the sums and face sums use the same loop and accumulation order
in both sets. The area-weighted mean differs: Set A accumulates numerator and
denominator in Float64 and rounds once to FT; Set B accumulates in FT. Set A
accepts Float64 areas with Float32 data; for Set B, `hasmethod` with Float32
data and Float64 areas returns `false`. Neither set uses `@simd` or
`@fastmath`.

Test matrix: C8 → C4, C90 → C30, C180 → C90; Float32 and Float64; data uniform
[0, 1), signed wide-range (`randn·10^U(-6,6)`) and air-mass-like
(1e9 + 1e13·U); Nz = 7; areas from the GMAO mesh, the equiangular mesh and
random. Area-weighted results were compared with a correctly rounded 256-bit
BigFloat reference.

| Comparison | Result |
|---|---|
| Cell sums A vs B | bitwise identical, 18/18 cases |
| x- and y-face sums A vs B | bitwise identical, 18/18 cases |
| Area-weighted means, Float64 (2-D, and A per level vs B 3-D) | bitwise identical, 54/54 |
| Area-weighted means, Float32 | differ, 54/54. Set A correctly rounded in 100 % of cells; Set B in 37.5–81 % of cells, with at most 5 ulp error on positive data and up to 6172 ulp on cancelling signed data (e.g. C90 → C30 GMAO uniform: 509/900 cells differ, max 4 ulp; C180 → C90: 3914/8100, max 3 ulp) |

- Block-summed fine areas vs coarse mesh areas (Float64), max relative
  difference: GMAO C8 → C4 3.3e-16, C90 → C30 8.8e-14, C180 → C90 6.3e-13;
  equiangular C8 → C4 4.4e-16, C90 → C30 4.6e-14.
- Timing (Float32, Nz = 72, minimum of 20 calls): C180 → C90 cell sum 1.88 ms
  (A) / 2.74 ms (B), weighted 2-D 0.046 / 0.048 ms, weighted 3-D 3.18 / 3.88 ms;
  C90 → C30 0.43 / 0.47, 0.0085 / 0.0089, 0.59 / 0.59 ms.
- Reach: none of the golden cases in `test/golden/cases.toml` (`pre_ll72`,
  `pre_o24`, `pre_c24`, `pre_merra2_c90`, `pre_era5_n320_c90_l117` and the runs on
  their binaries) reaches either set; a grep finds no other callers. The
  configs that use Set B in Float32 are
  `config/preprocessing/geosit_c90_dec2021_catrine_f32_fullL72.toml`,
  `geosit_c45_dec2021_catrine_f32_fullL72.toml` and
  `geosfp_c720_native_to_cs180.toml`; Set B averages `pblh`, `ustar`, `hflux`,
  `t2m`, vdiff `u/v/t/qv`, `cmfmc` and `dtrain` there.

---

## 9. S0: fresh Python references at `71a08395`

Every run was wrapped as

```bash
/temp1/cfranken/flux_port_parity/python_ref_71a08395/_tools/run_one.sh <driver_dir> <name> -- /usr/bin/python3 -u <script> <args>
```

which records the command line, start and end time, exit code and
`/usr/bin/time -v` output in `<driver_dir>/logs/<name>.*`. The manifest
`python_ref_71a08395/MANIFEST.json` was assembled by
`$J _tools/build_manifest.jl`; for each run it records the command, the sha256
of the script and of the imported repository modules, Python and library
versions, every input (34 distinct files, 12.7 GB, with size, mtime and sha256;
resolved by `_tools/resolve_inputs.py` through the scripts' own discovery
functions), every output with sha256, wall time, peak RSS, exit code and notes.
All C30 runs used the geometry file
`/temp1/cfranken/sif_gpp_iav/output/trendy_oco2era_c30/trendy_oco2era_c30_20140901.nc`;
its lats, lons and corner-derived `cell_area` reproduce those of the existing
`trendy_c30`, `gfed5` C30 and NPP/Rh C30 files bitwise. Its stored `cell_area`
differs from the corner-derived areas by up to 2.69e-5.

Scripts are under `scripts/preprocessing/`; output directories under
`python_ref_71a08395/`:

| Run(s) | Script and arguments | Output directory | Exit | Wall (s) | Peak RSS |
|---|---|---|---|---|---|
| GFED5 C30 2014 | `prepare_gfed5_fire_flux.py --start-year 2014 --end-year 2014 --geometry <C30> --tag c30` | `gfed5_c30_2014/` | 0 | 1.22 | 0.18 GB |
| hourly → daily 2021 | `hourly_c90_flux_to_daily.py <pilot>/sif_gpp_hourly_co2flux_c90_2021.nc <out>` (no time options; full year) | `hourly_to_daily_c90_2021/` | 0 | 70.56 | 0.11 GB |
| refine 2021-12-01 | `refine_cs_surface_flux.py <pilot hourly C90> <out> --start 2021-12-01T00:00:00 --end 2021-12-02T00:00:00` | `refine_c180_dec2021/` | 0 | 0.24 | 0.07 GB |
| refine 2021-12-31 | same with `--start 2021-12-31T00:00:00 --end 2022-01-01T00:00:00` (end one step after the last stamp) | `refine_c180_dec2021/` | 0 | 0.23 | 0.07 GB |
| 2021 hourly, CLM | `prepare_trendy_s3_gpp_ter_c90.py --models CLM` (year and C90 pilot geometry are fixed in the script) | `trendy_hourly_c90_2021/` | 0 | 23.08 | 0.83 GB |
| multi-year C30, one run per member: CABLE-POP, CARDAMOM, CLM-FATES, CLM, IBIS, JSBACH, ORCHIDEE | `prepare_trendy_s3_multiyear_c90.py --models <M> --start-year 2015 --end-year 2016 --geometry <C30> --tag c30` | `trendy_multiyear_c30_2015_2016/` | 0 | 1.92–5.14 | 0.34–0.40 GB |
| multi-year C30, LPJ-GUESS | same | — | **1** | 1.78 | 0.39 GB |
| NPP/Rh C30 2014–2024 | `prepare_trendy_s3_npp_rh.py --geometry <C30> --models LPJ-GUESS CLM` (span fixed in the script) | `trendy_npp_rh_c30_2014_2024/` | 0 | 11.23 | 1.34 GB |
| FLUXCOM-X 2021 (added later, §9.3) | `prepare_fluxcom_x_gpp_ter_c90.py --input-dir /kiwi-data/Data/satellite/FLUXCOM-X/X-BASE/2021 --geometry <pilot>/sif_gpp_weekly_diurnal_c90_2021.nc` (2021, 365 days and 8760 records are fixed in the script) | `fluxcom_x/` | 0 | 68.35 | 1.71 GB |
| SIF IAV 2018-05-01 … 2025-12-31 (added later, §9.3) | `build_sif_gpp_multiyear_c90_flux.py --sif <TROPOMI C90 daily SIF> --npz <pilot>/fluxcom_x_c90_2021.npz --summary <pilot>/analysis_summary.json --clim-years 2019-2025 --clim-smooth 15` (every argument given, equal to the script defaults and the v3 attributes) | `sif_iav/` | 0 | 34.37 | 4.60 GB |

The table omits the `--outdir`/`--output-dir` and `--staging-dir` arguments;
the full command lines are in `MANIFEST.json` and in `<driver_dir>/logs/*.cmd`.
`<C30>` is the geometry file above and
`<pilot>` is `/kiwi-data/Data/satellite/FLUXCOM-X/X-BASE/2021/pilot`. The runs
with `--staging-dir` pointed it at `python_ref_71a08395/_staging`; no input was
compressed, so it stayed empty and was removed.

- **LPJ-GUESS cannot be built by the multi-year script.** It stops with
  `KeyError: "variable 'rh' not found"` on `LPJ-GUESS_S3_arh.nc`
  (`T.flux_variable(ds, 'rh')`), exit 1. The NPP/Rh script handles the annual
  `arh` file.
- The multi-year script accepts only whole years; with `2015 2016` it reads the
  months 2014-12 … 2016-12.
- The first 14 runs wrote 807 MB in total. The two later runs (FLUXCOM-X,
  521 MB; SIF IAV, 665 MB) were appended to `MANIFEST.json` by
  `$J _tools/append_manifest_taskb.jl` with the same schema, every input
  carrying a sha256; the pre-append copy is `MANIFEST.json.pre_taskb`, and an
  `amendments` record names the comparator version used for §9.3. The manifest
  now lists 16 runs; `not_captured` lists only the full TRENDY ensembles and the
  LPJ-GUESS multi-year failure. The full `<TROPOMI C90 daily SIF>` path is in
  `sif_iav/logs/*.cmd`.

### 9.1 S0 against the existing products

Commands: `/usr/bin/python3 python_ref_71a08395/_tools/probe_compare.py` and
`/usr/bin/python3 python_ref_71a08395/_tools/probe_multiyear.py`; output
`probes/s0_compare_existing.txt`. K_new/K_old = (44.0095/12.0107)/(44/12) =
0.999324844437952.

| S0 output | Compared with | Result |
|---|---|---|
| hourly → daily 2021 | `<pilot>/sif_gpp_daily_co2flux_c90_2021.nc` | (365, 6, 90, 90) bitwise equal; time equal, 0 … 8736 |
| refine 2021-12-01 | records 0–23 of `<pilot>/sif_gpp_hourly_co2flux_c180_dec2021.nc` | bitwise equal; time 8016 … 8039 equal |
| refine 2021-12-31 | records 720–743 of the same file | bitwise equal; time 8736 … 8759 equal |
| GFED5 C30 2014 | days 0 … 364 of `/temp1/cfranken/sif_gpp_iav/fluxes/gfed5/GFED5_fire_daily_co2flux_c30_2014_2024.nc` | ratio 0.9993247265 … 0.9993249625; max 1 ulp against Float32(old/K_old·K_new); same zero pattern |
| NPP/Rh LPJ-GUESS | 2026-09-08 campaign file (`/temp1/cfranken/trendy_allmodels_c30_20260908/fluxes/`) × K_new/K_old | NPP max 1 ulp, 91.43 % bitwise; RH max 1 ulp, 91.05 % bitwise; `annual_PgC` max relative difference 1.83e-9 |
| NPP/Rh CLM | same | NPP 90.46 %, RH 90.57 % bitwise, max 1 ulp; `annual_PgC` 1.69e-10 |
| 2021 hourly CLM | existing 2021 output (`/kiwi-data/Data/groupMembers/cfranken/AtmosTransport/fluxes/TRENDYv14/S3/C90/2021/`) | JSON annual totals identical (GPP 148.05793351340287, RA 83.66050893769412, RH 52.98549991618245, TER 136.64600885387657 Pg C); after × K_new/K_old: max 1 ulp, mean bitwise fraction 94.40 % (GPP) and 91.83 % (TER) |
| multi-year CLM, ORCHIDEE (2015–2016) | existing C30 2014–2024 files (`/temp1/cfranken/sif_gpp_iav/fluxes/trendy_c30/`) × K_new/K_old | January 2015 … November 2016: max 1 ulp, 89.7–90.6 % bitwise per variable. December 2016: per-cell relative differences up to 451 (CLM GPP), 3.8e16 (CLM RA), 3.4 (CLM RH), 71 (ORCHIDEE GPP), 848 (ORCHIDEE RA), 0.26 (ORCHIDEE RH): the short build holds values flat after the 2016-12 midpoint |

Every multi-year S0 file: (731, 6, 30, 30), time 0 … 17520
`hours since 2015-01-01 00:00:00 UTC`, TER == RA + RH and NEE == GPP + TER
bitwise, GPP ≤ 0, all finite. Annual totals (GPP, TER, NEE in Pg C, as printed
by `probe_multiyear.py`):

| Member | 2015 | 2016 |
|---|---|---|
| CABLE-POP | −123.307476, 119.876404, −3.431072 | −123.608267, 119.869363, −3.738904 |
| CARDAMOM | −114.358555, 107.098227, −7.260328 | −115.140078, 107.610837, −7.529241 |
| CLM-FATES | −129.994908, 124.287787, −5.707121 | −129.991251, 125.077108, −4.914143 |
| CLM | −145.85249, 134.935122, −10.917368 | −146.454344, 135.443985, −11.010359 |
| IBIS | −121.898574, 117.788371, −4.110203 | −123.158604, 118.279813, −4.878791 |
| JSBACH | −135.254873, 130.386651, −4.868222 | −136.116514, 130.192039, −5.924475 |
| ORCHIDEE | −116.38661, 110.505483, −5.881127 | −117.564221, 111.068381, −6.49584 |

### 9.2 Reproduction and fixity

- `cd python_ref_71a08395 && sha256sum -c _tools/sha256_outputs.txt`: all 19
  output hashes OK (re-run on 2026-10-10 while writing this memo).
- `sha256sum -c _tools/sha256_scripts.txt`: all 6 scripts OK, and each equals
  `git show 71a08395:scripts/preprocessing/<script> | sha256sum` (checked for all
  six).
- The P0 review reran two S0 commands into `review_p0/rerun/` (logs
  `review_p0/rerun/{gfed,refine}.log`):
  - refine 2021-12-01: byte-identical file (sha256 `b6142588e0adf7d1…` for both);
  - GFED5 C30 2014: different file sha256 (`f98074fe…` vs `f614b80d…`); all
    five variables (`time`, `lons`, `lats`, `cell_area`, `FIRE_CO2_FLUX`) are
    byte-identical and the only differing global attribute is `history`
    (creation timestamp). Checked with `ncdump` diff and with a netCDF4 script
    comparing `tobytes()` of every variable with auto-masking off.
- The manifest commands name the scripts by their path in the
  `AtmosTransportModel-port` worktree.
- Final P0 review (2026-10-10):
  `sha256sum -c _tools/sha256_outputs.txt _tools/sha256_outputs_taskb.txt _tools/sha256_scripts.txt _tools/sha256_scripts_taskb.txt`
  gives 30 of 30 OK (19 + 3 outputs, 6 + 2 scripts), and
  `git diff --quiet 71a08395 -- scripts/preprocessing` still passes.

### 9.3 FLUXCOM-X and SIF IAV S0 against the existing products

Comparator runs: `python_ref_71a08395/_tools/run_taskb_compare.sh` (commands
and reports in `{fluxcom_x,sif_iav}/compare/*.{cmd,json,txt}`; `--nc 90`, C90
coastal mask of §11.4); value-wise probes `$J _tools/probe_k_ratio.jl`,
`_tools/check_sif_identities.jl` and `_tools/probe_sif_derived_operand_ulp.jl`
(outputs in the same `compare/` directories). Each comparator run took about
50 s and at most 3.1 GB. They were run with comparator sha256 `de4d61c8…`; the
final review reran all five with the final comparator (§11.3) and got
identical reports.

FLUXCOM-X (S0 against
`/kiwi-data/Data/satellite/FLUXCOM-X/X-BASE/2021/derived/fluxcom_x_gpp_diurnal_ter_daily_hourly_co2flux_c90_2021.nc`,
written with 44/12):
- The JSON audit equals the old one except `output`: GPP 128.73807093105066 and
  TER 123.60947035600589 Pg C, the same negative fractions, C90 area closure
  4.175593204536199e-11. Time axes, zero patterns, `lons`, `lats` and
  `cell_area` are identical.
- As stored, every non-zero value differs and the 2021 totals differ by
  6.751e-4 (GPP and TER).
- The script multiplies float32 arrays by the Python float `KG_CO2_PER_G_C`,
  which NumPy 1.26 rounds to Float32. The old file therefore maps to the new one
  through r₃₂ = Float32(1e-3·44.0095/12.0107)/Float32(1e-3·44/12) =
  0.9993248757811442, not through K_new/K_old = 0.999324844437952. The
  area-weighted sum ratios are 0.999324875784254 (GPP) and 0.999324875784524
  (TER), equal to r₃₂ within 3.4e-12.

| Reference scale | Variable | Max ulp | ≤ 1 ulp | Bitwise | Month totals | Year total |
|---|---|---|---|---|---|---|
| r₃₂ = 0.9993248757811442 | GPP | 2 | 99.771 % | 89.21 % | 5.7e-11 | 9.8e-12 |
| r₃₂ | TER | 3 | 99.493 % | 86.54 % | 2.3e-10 | 1.56e-11 |
| `carbon_44_12_to_iupac` (K_new/K_old) | GPP | 3 | 99.160 % | 87.09 % | 3.14e-8 | 3.137e-8 |
| `carbon_44_12_to_iupac` | TER | 3 | 98.728 % | 84.02 % | 3.16e-8 | 3.137e-8 |

SIF IAV (S0 against
`/temp1/cfranken/sif_gpp_iav/fluxes/sif_gpp_daily_co2flux_c90_2018_2025_v3.nc`):
- Time axes identical (2802 days); `land_fraction`, `cell_area`, `lons` and
  `lats` bitwise equal. The script multiplies in Float64, so the raw totals
  differ by exactly 1 − K_new/K_old (6.752e-4).
- Identities in the S0 file: NET == FLUX ⊕ RECO (Float32 sum) for all
  136,177,200 values; the Float32 subtraction NET − FLUX − RECO is non-zero for
  2,628,121 values (v3: 2,625,150); RECO constant in time; no FLUX > 0, no
  RECO < 0; 16,343 cells with `land_fraction` > 0.05.
- With `--reference-scale <var>=carbon_44_12_to_iupac`:

| Variable | Max ulp | ≤ 1 ulp | Bitwise | Month totals | Year totals |
|---|---|---|---|---|---|
| GPP_CO2_FLUX | 1 | 100 % | 92.72 % | 2.6e-10 | 4.5e-11 |
| RECO_FLAT_CO2_FLUX | 1 | 100 % | 91.58 % | 4.1e-10 | 4.1e-10 |
| GPP_CO2_FLUX_ANOM | 687,854,529 | 82.92 % | 75.37 % | 1.7e-6 | 1.6e-7 |
| GPP_CO2_FLUX_NET | 8,399,935 | 89.83 % | 78.39 % | 3.8e-9 | 1.05e-7 |
| GPP_CO2_FLUX_SEAS | 138,302 | 98.58 % | 88.87 % | 7.3e-10 | 1.12 |

- ANOM, NET and SEAS are differences of near-equal values. Measured against
  the Float32 spacing of their largest operand instead of their own value,
  |b − a·K_new/K_old| stays within 2.58 (ANOM), 2.64 (NET) and 1.30 (SEAS).
  SEAS has a zero annual mean, so its year relative difference carries no
  information; NET's slice totals differ by up to 7.4e-4 for the same reason.

## 10. Reference-numbers TOML

File `test/real_data/surface_flux_reference_numbers.toml` (936 lines, untracked;
parses with `TOML.parse`). Produced by

```bash
/usr/bin/python3 python_ref_71a08395/_tools/probe_reference_numbers.py <section> [...]   # → probes/s0_reference_numbers/<section>.json
$J python_ref_71a08395/_tools/write_reference_toml.jl                                     # → the TOML
```

Every value table cites a `source`, `audit` or `manifest` (except `[meta]`,
which holds no values). File-derived annual totals use
PgC = Σ Float64(Float32 flux) · cell_area · 86400 / K / 10¹² with K the factor
the file was written with (44/12 before `71a08395`, 44.0095/12.0107 for S0).

Selected values:

| Item | Value | Source |
|---|---|---|
| NEE climatology, kg CO2 per year | −3.402094951918086e13 (every non-leap year); −3.4010306993331016e13 (2016, 2020, 2024) | `/temp1/cfranken/sif_gpp_iav/fluxes/trendy_c30/JULES-ES_NEEclim_daily_co2flux_c30_2014_2024.nc` |
| NEE climatology, Pg C (kg CO2 / (44/12)) | −9.278440777958417 (non-leap), −9.27553827090846 (leap) | same |
| hourly → daily 2021 | 365 records, time 0 … 8736 (no `calendar` attribute); max daily value 0.0; worst per-cell daily-vs-hourly integral mismatch 1.785e-8; global 2021 −4.6221927048330156e14 kg CO2 (daily) vs −4.622192704696102e14 (hourly) | `<pilot>/sif_gpp_{daily,hourly}_co2flux_c90_2021.nc` |
| refine Dec 2021 | 744 records, time 8016 … 8759; every C180 record is a 2×2 copy of C90 parent records 8016 … 8759, bitwise | `<pilot>/sif_gpp_hourly_co2flux_c180_dec2021.nc` |
| CLM 2021 GPP, file-derived | C30: −148.05793351930515 Pg C; C90 v2: −148.05793351200603 Pg C; both within 4e-11 of the 2021 hourly audit value 148.05793351340287 | `trendy_c30/`, `trendy_v2/` |
| GFED5 S0 C30 2014 | 3.2631743114658853 Pg C (within 5.8e-11 of the native 2014 sum) | S0 file |
| GFED5 existing C30, 2014 … 2024 | 3.2631743110237172 … 4.028272102062001 Pg C | `/temp1/cfranken/sif_gpp_iav/fluxes/gfed5/` |
| FLUXCOM-X 2021 JSON | GPP 128.73807093105066, TER 123.60947035600589 Pg C; negative fractions 0.0913147913102717 (GPP daily), 0.01563289665267889 (TER daily before clip), 0.2328722232422215 (GPP monthly diurnal); C90 area closure 4.18e-11 | `/kiwi-data/Data/satellite/FLUXCOM-X/X-BASE/2021/derived/…c90_2021.json` |
| SIF IAV v3 | 2802 records; `annual_gpp_PgC` attribute sum 995.778; 16,343 cells with land fraction > 0.05 (npz and file agree; the file's `land_fraction` equals the npz bitwise); RECO_FLAT constant in time; RECO 128.96182257359436 Pg C per 365.25-day year (the script's `reco_pg`), 128.87355301673358 per 365 days; NET == GPP + RECO as a Float32 sum bitwise in all records, while the Float32 difference NET − GPP − RECO is non-zero for 2,625,150 of 136,177,200 values | `/temp1/cfranken/sif_gpp_iav/fluxes/sif_gpp_daily_co2flux_c90_2018_2025_v3.nc` |
| ECCO-Darwin December | −410932.98505768104 kg CO2 s⁻¹; 168,566 non-zero cells | `~/data/AtmosTransport/catrine/Emissions/ECCO_Darwin/eccodarwin_ocean_co2_december_05deg.nc` |
| TransCom C30 | weights sum to 1 within 2.22e-16; largest partition error 5.92e-8 (47 checks, tolerance 2e-7); 3,465 nearest-land fallback cells | `/temp1/cfranken/trendy_transcom_c30_20260908/partition_validation.json`, `transcom_weights_c30.nc` |

Independent recomputation in Julia with NCDatasets
(`python_ref_71a08395/_tools/crosscheck_reference_toml.jl`): a subset matches
the TOML to at most 8.9e-16 relative. The P0 review recomputed three values
independently: GFED5 S0 2014 (relative difference 0), NEEclim 2014 kg CO2
(1.1e-16), CLM C30 2021 GPP (5.6e-16).

### 10.1 Items added after the first TOML

The TOML now has 1141 lines. The tables `refine.sum_rho_area`,
`nee_climatology.month_means`, `fluxcom_x.identity_24h`, `sif_iav.anomaly`,
`eccodarwin.perday_vs_monthly` and `transcom_c180`, and the `audit`/`output`
keys of every `trendy_hourly_2021` member, were computed in Julia by
`probes/scripts/p0_verify_reference_items.jl <section> <fragment.toml>` (run as
`$HOME/.juliaup/bin/julia --project=<port>/test`, constants from `Parameters`,
fragments in `probes/reference_items/`) and merged by
`probes/scripts/p0_merge_reference_items.jl`, which checks that all 794 earlier
values are unchanged. Each table records its source and command. Rerunning
`write_reference_toml.jl` would drop these additions (the TOML header says so);
rerun the merge script after any regeneration. Full report:
`probes/p0_julia_probes.md` §4.

| Item (TOML table) | Measured | Wall, peak RSS |
|---|---|---|
| 2021 hourly members (`trendy_hourly_2021.members.*`) | each member cites its JSON audit and NetCDF; all 23 JSON files equal the TOML values exactly, each manifest summary equals its JSON, every NetCDF has 8760 records | 11 s, 1.1 GB |
| refine (`refine.sum_rho_area`) | per-hour Σρ·A₁₈₀ = Σρ·A₉₀ to 2.45e-14 at most over 744 hours (median 3.9e-15); GMAO meshes, radius 6.371e6 m; both binaries' mesh areas equal the GMAO mesh bitwise | 51 s, 1.2 GB |
| TransCom C180 (`transcom_c180`) | Σ regions + UNASSIGNED == flux (Float32 sum in the script's order) in 744/744 hours; no cell in two regions; UNASSIGNED share 0.877 % (hourly 0.025–3.06 %) | 37 s, 3.0 GB |
| NEE climatology (`nee_climatology.month_means`) | climatology recomputed from its source; all 712,800 cell-month means within their Float32 rounding bound (at most 0.49 Float32 ulp, 2.8e-15 absolute); global month totals 6.2e-9, annual 2.8e-10 | 23 s, 1.2 GB |
| FLUXCOM-X (`fluxcom_x.identity_24h`) | annual totals of the hourly file vs the JSON: −2.63e-8 (GPP) and −2.66e-8 (TER) with the Float64 K, +2.4e-10 and +1.8e-11 with Float32(K); TER = Float32(d₃₂·Float32(K))/86400f0 bitwise in all 17,739,000 cell-days; the pilot npz `daily` is a different product | 133 s, 1.9 GB |
| ECCO-Darwin (`eccodarwin.perday_vs_monthly`) | per-day mean vs monthly at most 1.52e-15 absolute, identical non-zero mask (168,566 cells); a Julia re-evaluation from the LLC270 binaries reproduces both files bitwise | 15 s, 1.6 GB |
| SIF IAV (`sif_iav.anomaly`, v3 file) | anomaly record total −1.9270059441306637 Pg C; fit years 2019–2025 −0.009433700096752595; flux integral 995.77806 Pg C; sphere closure of the file `cell_area` 4.1756e-11 (GMAO mesh 2.2e-16) | 82 s, 1.3 GB |

Still not verified, with the reason recorded at the end of the TOML: the
TransCom C180 region ids (needs the P7 box partition); the FLUXCOM-X per-cell
24-hour GPP identity (the Float32 daily GPP intermediate is not stored); NEEclim
month means to 1e-12 (only possible on Float64 values before storage).

Final P0 review: `crosscheck_reference_toml.jl` rerun on the 1141-line TOML
matches to at most 8.9e-16 (log `review_final/crosscheck_reference_toml.log`,
6.7 s, 0.78 GB). The TOML has no entries for the FLUXCOM-X and SIF IAV S0 files
(§9.3); their values are in `MANIFEST.json` and the JSON reports.

## 11. File comparator `compare_cs_flux_files.jl`

Script `scripts/validation/compare_cs_flux_files.jl`, test
`test/core/test_compare_cs_flux_files.jl`, a row in `scripts/README.md` (all
uncommitted on `port/p0`).

Definitions fixed in the script header: per-slice totals are
Σ density × mesh cell area; slice k holds from its stamp to the next (the last
slice ends at its `time_bnds` upper bound, else one month later for
month-start stamps, else one spacing later), and month totals split slices that
cross a month boundary. Totals relative difference (b − a)/|a|; per-cell
relative difference |b − a|/max(|a|, |b|) in [0, 2], cells where both are zero
left out; quantiles of type 7 over Float32-rounded differences, exact max.
Cell partition in order: coastal (with a mask: land class differs from an
edge neighbour, across panel edges via `Preprocessing.build_cs_global_face_table`),
polar (panels 3 and 6 via `panel_labels`), interior. 60 bands uniform in
sin(latitude) of the mesh centres. Fill or non-finite values always fail. Cell
areas come from the mesh (`--binary`, or `--nc`/`--radius` with default
`Parameters.EARTH_RADIUS`); a stored `cell_area` is only compared with them.

Added after the P0 review (final version, sha256 `3d38ccc4…`):
- `--reference-scale NAME=FACTOR` or `NAME=carbon_44_12_to_iupac`: each stored
  reference value a becomes T(Float64(a)·factor), T its stored type, before any
  metric; fill values are counted on the stored values. The preset is
  (SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS)/(44/12) = 0.999324844437952; the
  only literal is the named legacy constant `LEGACY_CO2_PER_CARBON = 44/12`.
- (Xdim, Ydim, nf) variables such as `land_fraction` and `cell_area` are
  compared cell by cell when named with `--var`; with only such variables no
  time axis is read.
- A check whose sample is empty passes as n/a (a category without cells, a
  quantile where every value is zero in both files, a totals or band metric of
  a variable without time); a metric unavailable for another reason fails.
- Unequal `units` strings are an error unless `--allow-unit-mismatch`.
- The coastal mask options are `--mask-var`, `--mask-file`, `--mask-threshold`
  (they replace `--mask` and `--land-threshold`, used in §11.2). The mask must
  be in the cs_native layout; when its file stores `lats`/`lons`, they must lie
  within 0.1·90°/Nc of the mesh centres, else the mask is rejected.
- TransCom-11 region totals are not implemented (P7), and integer-stored
  fields (for example region ids) are rejected as non-floating-point.

### 11.1 Test

- `$HOME/.juliaup/bin/julia --project=/home/cfranken/code/gitHub/AtmosTransportModel-port/test test/core/test_compare_cs_flux_files.jl`:
  first version 185/185 pass; 58 s wall including one CLI subprocess (author
  run), 59 s and 1.49 GB peak RSS (review run). Final version 283/283 pass:
  2:13 and 1.82 GB (author run, log `taskA/test_compare_final.log`), 1:38 and
  2.0 GB (final review, `review_final/run1_test_compare_cs_flux_files.log`).
- Mutations of the final version (copies in `taskA/mutations/`), each caught:
  reference scale not rounded to the stored type (4 failures), n/a counted as
  failure (5), centre limit loosened (1), fill values counted after scaling (1).
- Its references are computed independently: neighbours from shared corners
  (not from the connectivity table), `Statistics.quantile`, its own ulp
  function, a cross-panel coast case.
- Mutations of the script (on copies), each caught: quantile off by one
  (1 error), one-sided coastal marking (24 failures), no month split
  (3 failures), latitude-uniform instead of sin-latitude bands (4 failures),
  coastal marking that skips cross-panel faces (24 failures).
- Review checks of the code: the face-table index `(p−1)Nc² + (j−1)Nc + i`
  matches `_cs_global_cell` in `src/Preprocessing/cs_face_table.jl`;
  `panel_cell_center_lonlat` returns degrees (C90 panel 3 latitudes
  35.66 … 89.22).
- `test_readme_current.jl` and `test_documentation_links.jl` pass with the new
  `scripts/README.md` row.

### 11.2 Real-data runs

These runs used the first comparator version. The JSON reports record the
reference, candidate, mesh, mask, threshold and tolerances of each run; the
commands below are written from those records, with the option names of that
version (`--mask`, `--land-threshold`; now `--mask-var`, `--mask-threshold`)
(`compare` stands for `julia --project=<port>/test scripts/validation/compare_cs_flux_files.jl`).
Meshes: C30 binary
`/kiwi-data/Data/groupMembers/spandey/atmos/met/c30_meteo/era5_c30_transport_20140101_float32.bin`;
C90 binary
`/kiwi-data/Data/groupMembers/cfranken/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_no_convection/era5_n320_transport_20140101_float32.bin`
(both radius 6.371e6 m, GEOS-native convention).

| Report | Arguments | Result | Wall, peak RSS |
|---|---|---|---|
| `probes/compare_cs_flux_files/gfed5_c30_self.{json,txt}` | GFED5 C30 2014–2024 file against itself; `--binary <C30> --tol bitwise_fraction=1 --tol year_total=0` | PASS; 4018 slices; year totals × 12/44: 2014 3.263174, 2024 4.028272 Pg C; monthly totals equal `prep_gfed5_c30.log` to its 4 digits (2014-01 0.2313, 2014-03 0.3444, 2024-12 0.2076); file `cell_area` vs C30 mesh: max element-wise relative deviation 2.668e-5 | 21.8 s, 1.27 GB |
| `…/sif_v2_vs_v3_gpp.{json,txt}` | `sif_gpp_daily_co2flux_c90_2018_2025_v2.nc` vs `_v3.nc`; `--binary <C90> --var GPP_CO2_FLUX --mask land_fraction --mask-file <v3> --land-threshold 0.05 --tol year_total=1e-3 --tol cell_p99.interior=1e-6` | PASS, bitwise identical over 2802 slices; cells: 5180 coastal, 14372 polar, 29048 interior; file `cell_area` vs C90 mesh 1.747e-4 | 28.8 s, 1.48 GB |
| `…/sif_v1_vs_v3.{json,txt}` | `sif_gpp_daily_co2flux_c90_2018_2025.nc` vs `_v3.nc`; same mask and tolerances, no `--var` (all common 4-D variables: `GPP_CO2_FLUX`, `GPP_CO2_FLUX_ANOM`) | PASS, both bitwise identical over 2802 slices (v3-only variables: `GPP_CO2_FLUX_NET`, `_SEAS`, `RECO_FLAT_CO2_FLUX`) | 34.6 s, 1.60 GB |
| `…/trendy_clm_2014_2024_vs_2018_2024.{json,txt}` | `trendy_v2/…CLM…c90_2014_2024.nc` vs `trendy/…CLM…c90_2018_2024.nc`; `--binary <C90> --common-times --tol month_total=1e-6 --tol cell_p99.interior=1e-6` | FAIL (exit 1) as expected for the cell gate; 2557 common days. GPP bitwise identical. TER: 68.3 % bitwise, cell p50 4.6e-4, p99 0.41, slice totals 1.44e-2, month totals 3.3e-10, year 5.0e-11. NEE: cell max 2.0, slice 130, month 1.6e-7, year 6.0e-10 | 46 s, 2.58 GB |
| `review_p0/compare/h2d.json` | pilot daily file vs the S0 hourly → daily output; `--nc 90 --tol bitwise_fraction=1 --tol year_total=0` (files without `calendar` attribute) | PASS, bitwise, 365 slices | 19.6 s, 1.39 GB |
| `review_p0/compare/refine31.json` | pilot C180 Dec-2021 file vs the S0 2021-12-31 day; `--nc 180 --common-times --tol bitwise_fraction=1` | PASS, bitwise, 24 common stamps | 18.6 s, 1.17 GB |

The P0 review split the TRENDY CLM report by year: the TER per-slice totals
differ on every day of every year 2018–2024 (maximum 1.2e-2 to 1.44e-2 per
year), while month totals agree to ≤ 3.3e-10.

Properties of the first version seen in these runs, all changed since except
the last: a gate on a category with no informative values failed as "n/a";
differing units produced only a warning; only 4-D `(Xdim, Ydim, nf, time)`
variables were compared; the reference could not be rescaled; TransCom-11
region totals are not computed.

### 11.3 Real-data runs of the final version

| Run | Command or TOML | Result | Wall, peak RSS |
|---|---|---|---|
| S0 GFED5 C30 2014 vs the 2014–2024 product (44/12), 365 common days, C30 mask | `taskA/s0_gfed5_rescaled.toml` through `run_parity.jl` (reports next to it) | with `reference_scale = { FIRE_CO2_FLUX = "carbon_44_12_to_iupac" }`: max 1 ulp, 94.51 % bitwise, month totals 2.1e-9, 2014 total 1.9e-10, `cell_area` bitwise; as stored: max 11327 ulp, totals 6.752e-4, FAIL as intended | 30.0 s, 1.21 GB |
| The five §9.3 comparisons, rerun | `review_final/rerun_taskb_compare.sh` (a copy of `_tools/run_taskb_compare.sh` writing to `review_final/rerun_taskb/`) | all exit 0; `review_final/diff_reports.jl` finds every leaf of every report (46,996–53,653 per report) identical to the reports of comparator `de4d61c8…`, apart from `created` | 50–54 s each, at most 3.0 GB |

### 11.4 Coastal masks per grid

Probe `taskA/probe_mask_sources.jl` (log `probe_mask_sources.log`, 8.9 s,
0.88 GB), using the comparator's `read_mask` and `cell_categories` on a GMAO
mesh:

| Grid | File, variable, threshold | Land cells | Coastal / polar / interior | Stored centres vs mesh |
|---|---|---|---|---|
| C90 | `/temp1/cfranken/sif_gpp_iav/fluxes/sif_gpp_daily_co2flux_c90_2018_2025_v3.nc`, `land_fraction`, 0.05 | 16,343 of 48,600 | 5180 / 14372 / 29048 | 2.32e-4° |
| C90 | same, threshold 0 | 17,446 | 6020 / 14378 / 28202 | 2.32e-4° |
| C30 | `/temp1/cfranken/trendy_transcom_c30_20260908/transcom_weights_c30.nc`, `nearest_land_fallback` (Int8, 1 = no TransCom-11 land sample), 0 | 1935 of 5400 (value 0) | 903 / 1560 / 2937 | 4.13e-5° |

- C90 (threshold 0.05): 1006 land cells south of 60°S; none north of 60°N
  between 75°W and 10°W (Greenland, Iceland); all 484 cells of the box
  15–30°N, 15°W–30°E are land.
- C30: no land south of 60°S and none in the Greenland box, so those coasts are
  not coastal under this mask; all 53 Sahara-box cells are land.
- C180: no land-fraction file was found; the pilot TransCom tags file's
  `transcom_region_id` was not checked as a mask.

### 11.5 Changes after the Codex review of the final version

Three rounds of Codex review of the final version (sha256 `3d38ccc4…`)
reported eight, three and one findings. The comparator, the runner and both
tests were changed (uncommitted on `port/p0`); the runs of §11.2–§11.4 used the
earlier versions. Rules of the changed comparator (its header has the full
text):

- Layout: a compared variable must have exactly the dimensions
  (Xdim, Ydim, nf, time) or (Xdim, Ydim, nf); any other names or order stop
  the comparison (before: a warning). Stored `lats`/`lons` of either file must
  be (Xdim, Ydim, nf) fields within 0.1·90°/Nc of the mesh centres, as for
  the mask file (before: a warning). A file with only one of `lats` and `lons`
  is an error, flux or mask file (round 2; before: silently not checked). The
  time coordinate is the variable
  `time` on dimension `time` (the `t` fallback was removed).
- Variable selection: without `--var`, every series of either file is
  compared, and a series present in only one file fails the comparison unless
  `--common-variables-only` (TOML key `common_variables_only`), which goes with
  the default selection only. Before, such series were only listed
  (`unmatched_variables`; now `variable_selection`). The §11.2 `sif_v1_vs_v3`
  comparison needs `--common-variables-only` under this rule.
- Time axes (round 2): slices are matched as (start, end) intervals, the end
  by the rule of the totals. A common stamp whose slices end at different times
  (unequal final bounds, or bounds in one file that disagree with the end
  inferred in the other) is a time-axis difference and fails by default; with
  `--common-times` only the matched intervals are compared. An unknown end (one
  slice without bounds) matches only an unknown end. Before, only the stamps
  were compared, so files with equal stamps and unequal final bounds were
  `identical`.
- Time bounds: slices are start-stamped. Lower bounds must equal the stamps
  and each upper bound the next lower bound; centre-stamped, gapped or
  overlapping bounds and a declared but absent bounds variable are errors
  (before: lower bounds off the stamps gave a warning; gaps and overlaps were
  accepted).
- Bitwise equality counts only values of the same stored type. With different
  stored types the fraction is not available and a `bitwise_fraction` check
  fails (before: values were promoted, so 1.0f0 and 1.0 counted as equal).
- Quantiles over all cells are selected from the three sorted category
  vectors (bisection on Float32 bit patterns), not from a merged copy, and each
  category vector is sorted in place (`QuickSort`, no scratch buffer). The
  memory statement now reads 4 bytes per differing value plus the growth slack
  of `push!` (before: it said 4 bytes, while the merged copy made it 8, plus
  the scratch buffer of the default sort).
- Units: a compared series without a non-empty `units` attribute in either
  file is an error (before: two missing attributes compared equal);
  (Xdim, Ydim, nf) fields may be unitless.
- Fill values are matched by the rule of NCDatasets
  (`CommonDataModel.isfillvalue`): a NaN fill value matches any NaN (before:
  `in`, so values under a NaN fill value were never counted).
- The tests define `LEGACY_CO2_PER_CARBON = 44/12` once instead of repeating
  the literal.
- Runner (round 2): a nested tolerance table must name a per-category metric
  and hold at least one category; the parent metric is checked before the table
  is flattened (before: `{ cell_p99 = {} }` or `{ nonsense = {} }` silently
  gave no check).
- Tolerances (round 3): one check, `validate_tolerance`, serves the command
  line, the `tolerances` argument and the runner: finite and non-negative,
  `bitwise_fraction` within [0, 1], `ulp_max` a whole number; a metric given
  twice is an error on all three paths (before: only NaN was rejected, so
  `slice_total = Inf` or `bitwise_fraction = -1` made their gates always pass,
  and the API accepted duplicates).

Measured (logs and reports in `review_codex_fix/`):

- Final version (comparator sha256 `a55c1de1…`, runner `2ee63a50…`):
  `test_compare_cs_flux_files.jl` 412/412, 96.8 s, 2.12 GB
  (`round3_test_compare_cs_flux_files.log`); `test_flux_port_parity_harness.jl`
  117/117, 44.5 s, 1.19 GB (`round3_test_flux_port_parity_harness.log`).
  Round-2 version (comparator `304a8cc8…`, runner `18d1440b…`): 372/372
  (94.9 s, 2.11 GB) and 108/108 (39.9 s test time). Round-1 version (comparator `71cc8073…`, runner `16c85844…`): 355/355
  (92.6 s, 2.02 GB) and 99/99 (43.4 s, 1.22 GB). `test_readme_current.jl`
  256/256 and `test_documentation_links.jl` 96/96 and 1/1 with the changed
  `scripts/README.md` row.
- Round-3 mutations of the final version (copies in `mutations_round3/`),
  each caught: finiteness and sign not checked (18 failures in the comparator
  test, 6 in the harness test), API bypassing the check (6), command line
  bypassing it (10), runner bypassing it (11), no upper bound on
  `bitwise_fraction` (4 and 2), `ulp_max` not required whole (4 and 1).
- Round-2 mutations of the round-2 version (copies in `mutations_round2/`),
  each caught: intervals matched by stamp only (9 failures, 5 errors), one-sided
  centres accepted (3), empty category table accepted (3, harness test), parent
  metric of a nested table not checked (3, harness test).
- Round-1 mutations of the round-1 comparator (copies in `mutations/`), each
  caught by `test_compare_cs_flux_files.jl`: layout accepted by dimension count (3
  failures), centre deviation not enforced (5), series of one file ignored (4),
  lower bounds not checked (1), contiguity not checked (2), stored types
  promoted for bit equality (1), bitwise fraction reported for mixed types (5
  failures, 1 error), merged copy for the quantiles (1), default sort with its
  scratch buffer (1), order statistic off by one (3), empty units accepted (4),
  NaN fill matched by == (3).
- SIF v1 against v3 with the round-1 version (`compare` with `--nc 90 --tol bitwise_fraction=1`;
  reports `sif_v1_vs_v3_{default,common}.{json,txt}`): exit 1 without the
  option (series only in v3: `GPP_CO2_FLUX_NET`, `GPP_CO2_FLUX_SEAS`,
  `RECO_FLAT_CO2_FLUX`; `GPP_CO2_FLUX` and `GPP_CO2_FLUX_ANOM` bitwise), exit 0
  with `--common-variables-only`; 31 s and 1.65 GB each.
- Parity harness example: see §12.

## 12. Parity harness `scripts/validation/flux_port_parity/`

- `run_parity.jl` reads a strictly checked TOML list of comparisons (unknown
  keys are errors; relative paths from the TOML's directory), calls
  `compare_cs_flux_files` as a function for each, records a comparison that
  throws as `error` and runs the rest, writes one JSON and one markdown report,
  takes `--only`, and exits 1 on any failure or error. The reports record the
  git commit, whether tracked files are modified, and the sha256 of the runner
  and the comparator as loaded (added in the final review). After the Codex
  review (§11.5) it takes `common_variables_only`, lists series present in
  one file only and common stamps whose slices end at different times,
  rejects empty nested tolerance tables, and checks tolerances with the
  comparator's `validate_tolerances`.
- `README.md`: purpose, levels S0/L1/T/L2/R/X, the S0 location and manifest,
  S0 comparison facts (§9.3, §11.4), how to add a driver, emulator lifetime
  (deleted in P11). The X level is not a file pair; the runner needs a
  comparison kind for it when P4 adds the first smoke.
- `s0_bitwise_pairs.toml`: hourly → daily C90 2021 (with the C90 mask) and
  refine C180 2021-12-31.
- Test `test/core/test_flux_port_parity_harness.jl` on synthetic C4 files:
  85/85 (author run, 42 s, 1.22 GB; three runner mutations each caught, copies
  in `taskA/mutations/h*`), 89/89 after the script hashes were added
  (`review_final/run2_test_flux_port_parity_harness.log`, 41 s, 1.20 GB).
- Real run of the example: `julia --project=<port>/test scripts/validation/flux_port_parity/run_parity.jl scripts/validation/flux_port_parity/s0_bitwise_pairs.toml --json <out>.json --markdown <out>.md`:
  PASS 2/2, 29.1 s, 1.42 GB (`harness_runs/s0_bitwise_pairs.*`); hourly →
  daily 365 stamps bitwise, coastal max 0; refine 24 common stamps bitwise.
  Rerun with the hash-recording runner: PASS 2/2, 27.8 s, 1.37 GB
  (`review_final/harness_s0_bitwise_pairs.*`). Reruns after the Codex-review
  changes, same results (C90 cells 5180 coastal, 14372 polar, 29048 interior;
  stored centres of both C90 files 2.32e-4° from the mesh; hourly → daily
  intervals identical; refine 24 matched intervals, 0 with other ends): PASS
  2/2 with the round-1 version (comparator `71cc8073…`, 28.5 s, 1.39 GB,
  `review_codex_fix/harness_s0_bitwise_pairs.*`), the round-2 version
  (comparator `304a8cc8…`, runner `18d1440b…`, 28.9 s, 1.42 GB,
  `review_codex_fix/round2_harness_s0_bitwise_pairs.*`) and the final version
  (comparator `a55c1de1…`, runner `2ee63a50…`, 28.6 s, 1.36 GB,
  `review_codex_fix/round3_harness_s0_bitwise_pairs.*`).

## 13. Not measured in P0

- Cold-disk read times (OS caches could not be dropped).
- The Julia CFTime check for the three compressed models used only the
  recorded first and last raw values plus synthetic full ranges (step 1, n as
  recorded); the decompressed files had already been deleted.
- TransCom C180 region ids (needs the P7 partition); the FLUXCOM-X per-cell
  24-hour GPP identity (the Float32 daily GPP intermediate is not stored);
  NEEclim month means to 1e-12 (only possible on Float64 values).
- A C180 coastal mask.
