# Brief ATBD: TRENDYv14 S3 GPP and TER C90 flux drivers for 2021

**Product family:** `TRENDYv14_S3_*_gpp_diurnal_ter_daily_hourly_co2flux_c90_2021.nc`  
**Version/status:** v0.1 experimental ensemble drivers, 2026-07-22  
**Grid/time:** GEOS-native C90 cubed sphere; hourly UTC; calendar year 2021  
**Output root:** `/kiwi-data/Data/groupMembers/cfranken/AtmosTransport/fluxes/TRENDYv14/S3/C90/2021`

## 1. Purpose and product definition

These files translate TRENDYv14 S3 monthly gross primary production (GPP),
autotrophic respiration (`ra`), and heterotrophic respiration (`rh`) into
hourly atmospheric CO2 surface-flux drivers for transport experiments. Each
model remains an independent pair of tracers:

- `GPP_CO2_FLUX` is negative atmospheric CO2 uptake.
- `TER_CO2_FLUX = ra + rh` is positive total ecosystem respiration.

After the documented nonnegative-component screening, the method conserves
every model's monthly carbon total through spatial and temporal processing. It
adds submonthly timing but does not add new monthly carbon information.

## 2. Source data and model coverage

The sources are the local TRENDYv14 S3 monthly files under
`/kiwi-data/Data/model/TRENDYv14/S3/<MODEL>/S3`. The 2021 product was generated
for 23 locally usable models:

`CABLE-POP`, `CARDAMOM`, `CLASSIC`, `CLM`, `CLM-FATES`, `DLEM`, `ED`, `ELM`,
`ELM-FATES`, `IBIS`, `ISAM`, `JSBACH`, `JULES-ES`, `LPJ-GUESS`, `LPJml`,
`LPJwsl`, `LPX-Bern`, `ORCHIDEE`, `SDGVM`, `TEM`, `VISIT`, `VISIT-UT`, and
`iMAPLE`.

`ISBA-CTRIP` and `OCN` were skipped because their local directories contain no
usable GPP, RA, or RH files. The compressed SDGVM, VISIT, and iMAPLE sources
are expanded one variable at a time into temporary storage and are not changed.

LPJ-GUESS does not provide monthly `rh`. Its documented annual heterotrophic
respiration variable, `arh`, is assigned as the RH monthly mean for all months
within its corresponding year. Monthly `ra` still varies. This product is
included but should be treated separately in analyses of TER seasonality.

## 3. Source normalization and land-area treatment

Input fluxes are interpreted as kg C m-2 s-1. Small negative GPP, RA, and RH
values are clipped to zero, with the affected fraction and original minimum
recorded in each model's JSON sidecar.

The following model-supplied fractional-area fields are applied before
remapping because these sources are treated as flux per unit land area:

| Model | Applied factor |
| --- | --- |
| CABLE-POP, CLM, ELM-FATES, IBIS, LPJml, ORCHIDEE, TEM | `1 - oceanCoverFrac` |
| CLASSIC | `land_fraction` |
| CARDAMOM | embedded `land_fraction` |

Other models are treated as reporting flux per total grid-cell area, without
additional area scaling. LPJ-GUESS is explicitly kept in this category per its
model README. The IBIS ocean-fraction file incorrectly declares the physically
valid value zero as a NetCDF `missing_value`; the generator ignores that
attribute while still honoring `_FillValue`.

This table is intentionally explicit and is the first item to review when a
model submission or README is revised.

## 4. Conservative spatial remapping

Source grids range from approximately 0.25 to 2.5 degrees. A common
0.25-degree spherical sampling grid estimates overlap between every regular
latitude-longitude source cell and the nearest GEOS-native C90 cell. The
sample weights belonging to each source cell are then renormalized to that
source cell's exact spherical area. Thus, for source cell `i` and target cell
`j`, the weights obey

`sum_j W[j,i] = A_source[i]`.

The C90 flux density is

`q_C90[j] = sum_i W[j,i] q_source[i] land_factor[i] / A_C90[j]`.

The overlap geometry is approximate at 0.25 degree, but the global carbon
integral is conserved to numerical precision. C90 areas are computed from
the archived cubed-sphere corners and checked against `4 pi R^2`, using
`R = 6,371,000 m`.

## 5. Month-conserving temporal interpolation

Fourteen monthly fields are read for each variable: December 2020, all months
of 2021, and January 2022. Linear interpolation between calendar-month
midpoints produces a smooth preliminary daily-mean series. For each grid cell
and calendar month, those daily values are then multiplied by

`s_m = q_monthly[m] / mean_days_in_month(q_interpolated)`.

Consequently, for the post-screening monthly field,

`mean_days_in_month(q_daily) = q_monthly[m]`

for every active C90 cell. Month length is respected, and the monthly carbon
total `q_monthly * seconds_in_month * area` is unchanged. The renormalization
can introduce a small discontinuity at a month boundary, but avoids the much
larger discontinuities of a fully stepwise monthly driver.

## 6. Hourly allocation

### GPP

Each model uses the same monthly C90 UTC-hour fractions previously prepared
from the FLUXCOM-X monthly diurnal GPP cycle. Fractions are nonnegative and
sum to one over 24 hours in each cell. Where FLUXCOM-X has no support, a
positive cosine-solar-zenith curve for the 15th day of the month is used.

For daily-mean carbon rate `q_GPP,d` and hourly fraction `f_h`,

`q_GPP(d,h) = 24 q_GPP,d f_h`.

The daily carbon total is therefore unchanged. This common timing isolates
differences in TRENDY monthly GPP magnitude and seasonality; it is not a
model-specific prediction of photosynthetic timing.

### TER

TER is calculated from independently processed nonnegative components:

`q_TER,d = q_RA,d + q_RH,d`.

It is held constant for all 24 UTC hours of each day. No artificial TER
diurnal cycle is imposed.

## 7. Atmospheric flux conversion and output

Carbon rates are converted using `44/12 kg CO2 per kg C`. Output fluxes are
per total C90 cell area in kg CO2 m-2 s-1:

- `GPP_CO2_FLUX = -(44/12) q_GPP(d,h)`;
- `TER_CO2_FLUX = +(44/12) q_TER,d`.

Each NetCDF contains 8,760 hourly records, C90 center coordinates, cell area,
the two flux variables, input provenance, remapping information, the applied
land-fraction rule, and sign conventions. Files are compressed and chunked by
day and cubed-sphere panel for efficient transport reads.

Each NetCDF has a JSON sidecar containing source paths, units, negative-value
statistics, annual GPP/RA/RH/TER totals, spatial conservation, and temporal
conservation. The ensemble manifest is
`TRENDYv14_S3_2021_manifest.json` in the output root.

## 8. Validation and limitations

Required validation checks are:

1. 8,760 records and C90 dimensions `(6,90,90)`.
2. GPP is never positive and TER is never negative.
3. TER has zero within-day range.
4. The 24 hourly GPP fractions conserve each daily mean.
5. Daily interpolation reconstructs every monthly mean cell by cell.
6. Native and C90 global monthly/annual integrals agree to numerical precision.
7. All values are finite and each annual integral is physically plausible.

For v0.1, all 23 files passed the structural and sampled sign/finite checks.
Native-to-C90 GPP conservation error was at most `7.8e-16` relative, and the
largest cell-wise monthly-mean reconstruction error was `2.6e-15` relative.
The Hovmoller reintegration agrees with the JSON annual totals within
`6.0e-8 Pg C`. Across the ensemble, annual GPP ranges from 71.67 to
182.14 Pg C and TER from 62.96 to 173.45 Pg C. The multi-model annual
mean/median are 133.38/133.48 Pg C for GPP and 126.32/128.52 Pg C for TER.
The 23 compressed NetCDF drivers occupy 9.62 GiB.

Important limitations are:

- Monthly TRENDY input cannot reproduce synoptic or event-scale variability.
- All models share FLUXCOM-X GPP timing, so the hourly products are not wholly
  independent of FLUXCOM-X.
- TER intentionally lacks a diurnal cycle.
- LPJ-GUESS RH is annual and has no monthly seasonality.
- The 0.25-degree overlap sampling is mass-conservative but not an exact
  polygon-intersection remap.
- Land-fraction conventions vary among TRENDY submissions and must remain
  under version-controlled model-specific review.
- Clipping negative component fluxes can change totals when a source contains
  meaningful negative predictions; the sidecars expose this fraction.

## 9. Ensemble Hovmoller diagnostics

Monthly zonal totals are binned uniformly in `sin(latitude)`, so equal vertical
increments have equal Earth area. Values are reported as
Pg C month-1 per unit `sin(latitude)`. For both GPP and TER the diagnostics
include:

- one absolute-value panel per model on a common scale;
- each model minus the multi-model mean;
- each model minus the multi-model median;
- ensemble mean and median reference panels;
- an NPZ archive of all plotted arrays and a CSV of annual totals.

These diagnostics are stored in the `hovmoller` subdirectory of the output
root. Mean and median are computed over the 23 available products only.

## 10. Reproducibility and maintenance

Primary generator:
`scripts/preprocessing/prepare_trendy_s3_gpp_ter_c90.py`.

Ensemble diagnostic generator:
`scripts/diagnostics/plot_trendy_s3_hovmoller_ensemble.py`.

Commands from the AtmosTransportModel repository root:

```bash
python scripts/preprocessing/prepare_trendy_s3_gpp_ter_c90.py --skip-existing
python scripts/diagnostics/plot_trendy_s3_hovmoller_ensemble.py
```

For a new TRENDY release or year:

1. Review source units, time axis, RA/RH availability, and land-area convention
   for every model.
2. Update the explicit land-fraction table and any annual-RH exception.
3. Change the requested year and required 14-month boundary window in the
   generator; do not silently reuse 2021 timing metadata.
4. Regenerate JSON sidecars and the ensemble manifest.
5. Run all seven validation checks before transport.
6. Regenerate mean and median diagnostics using only successfully validated
   model products.
