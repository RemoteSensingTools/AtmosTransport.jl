# Brief ATBD: 2021 SIF-derived GPP driver

**Product:** `sif_gpp_hourly_co2flux_c90_2021.nc`  
**Version/status:** 2021 experimental pilot; brief ATBD v0.1, 2026-07-22  
**Grid/time:** GEOS-native C90 cubed sphere; hourly UTC; calendar year 2021

## 1. Product definition

The driver is a gap-free, hourly gross-primary-production (GPP) estimate built
from observed satellite solar-induced chlorophyll fluorescence (SIF). It is a
**SIF-constrained FLUXCOM-X hybrid**, not a wholly independent SIF-only GPP
retrieval:

- TROPOMI and OCO-2 SIF determine weekly observed variability where sampling is
  adequate.
- FLUXCOM-X X-BASE GPP supplies the absolute SIF-to-GPP calibration, the prior
  where SIF is unavailable or downweighted, the dormant-season safeguard, and
  the within-week daily and hourly timing.

The product is intended as an alternative atmospheric-transport driver for
sensitivity experiments. It should not be interpreted as an independent
validation of FLUXCOM-X.

## 2. Input data

1. **TROPOMI S5P-PAL TROPOSIF:** baseline 743--758 nm retrieval reported at
   740 nm; product-provided length-of-day-corrected `SIF_Corr_743` is used.
2. **OCO-2 Lite SIF B11.2r:** SIF at 740 nm; product-provided
   length-of-day-corrected `Daily_SIF_740nm` is used. Quality flags 0 and 1 are
   gridded separately and also combined for gap filling.
3. **FLUXCOM-X X-BASE 1.0 GPP:** 2021 daily GPP and monthly mean hourly cycles
   at 0.25 degree, in g C m-2 d-1.

Instantaneous SIF is retained only as a diagnostic. The numerical TROPOMI units
of mW m-2 sr-1 nm-1 and OCO-2 units of W m-2 sr-1 um-1 are equivalent because
both numerator and wavelength denominator differ by 1000.

## 3. Spatial and temporal aggregation

Satellite Level-2 footprints are deposited from their footprint corners
directly onto the GEOS-native C90 cubed sphere. The analysis uses 52 complete
seven-day windows beginning 1 January through 24 December, followed by a
one-day 31 December interval. Observations from 2022 that could enter the
nominal final seven-day bin are explicitly excluded.

FLUXCOM-X is independently mapped from 0.25 degree to C90 using source-cell
area and FLUXCOM land-fraction weights. Small negative machine-learning GPP
predictions are set to zero before use as a physical prior.

## 4. Sensor harmonization

TROPOMI is placed on the OCO-2 QF0 scale using a global robust Huber regression
over collocated C90 cell-weeks:

`SIF_T,corrected = 0.040777 + 0.877893 SIF_T`.

The fit used 212,446 collocations; the corrected-variable correlation was
0.895 and residual RMSE was 0.094 W m-2 sr-1 um-1. Usable observations require:

- TROPOMI: at least 100 deposited observations and -1.5 < SIF < 4.0.
- OCO-2: at least 5 observations and -2.0 < SIF < 5.0.

## 5. Weekly SIF-to-GPP retrieval

For each calendar month and each of eight latitude bands with edges at
`[-90, -60, -40, -20, 0, 20, 40, 60, 90]` degrees, a robust regression is fit
against weekly FLUXCOM-X GPP:

`GPP_SIF = max(0, beta_0 + beta_1 SIF_corrected)`, with `beta_1 >= 0`.

Fits with fewer than 500 samples fall back to the global model. The global
reference fit was

`GPP = 0.01914 + 13.2561 SIF`,

with GPP in g C m-2 d-1, residual RMSE 1.066 g C m-2 d-1, and 698,444 training
cell-weeks. The month-by-latitude coefficients, rather than this global fit,
are used whenever adequately sampled.

The observation estimate is blended with the coincident FLUXCOM-X weekly
prior. For TROPOMI,

`GPP = w_T GPP_SIF + (1 - w_T) GPP_FLUXCOM`,

where `w_T` decreases from one toward zero for solar zenith angles between 45
and 70 degrees and for cloud fractions above 0.5. If TROPOMI is unavailable,
combined OCO-2 QF0+QF1 can supply the SIF estimate with a maximum weight of
0.5, also reduced at high solar zenith angle. Remaining gaps use FLUXCOM-X.
If weekly FLUXCOM-X GPP is below 0.05 g C m-2 d-1, the cell is treated as
dormant and forced to the FLUXCOM-X prior to suppress spurious high-latitude
SIF activity.

For 2021, source attribution comprised 698,852 TROPOMI cell-weeks, 5,436
OCO-2-only cell-weeks, and 220,350 FLUXCOM-prior cell-weeks.

## 6. Daily/hourly allocation and CO2 flux

The retrieved weekly mean is multiplied by the interval length to obtain a
weekly carbon total. This total is conserved during temporal disaggregation:

1. Positive FLUXCOM-X daily GPP within each weekly interval is normalized to
   daily fractions.
2. The nonnegative FLUXCOM-X monthly hourly cycle is normalized separately in
   every C90 cell to obtain 24 UTC-hour fractions. A cosine-solar-zenith pattern
   is used only where the monthly cycle has zero support.

For cell `c`, week `w`, day `d`, and hour `h`, the atmospheric flux is

`F_CO2 = -G_w f_daily(w,d) f_hour(month,h) L_c (44/12) 10^-3 / 3600`,

where `G_w` is g C m-2 per weekly interval over vegetated land, `L_c` is the
C90 vegetated-land fraction, and the result is kg CO2 m-2 s-1 over total cell
area. Negative values denote atmospheric uptake.

## 7. Validation and limitations

- Reintegrated annual SIF-hybrid GPP: **126.060 Pg C**; regridded FLUXCOM-X:
  **129.051 Pg C**.
- Maximum weekly conservation error after hourly allocation: 4.93e-7 relative
  for weekly totals above 0.1 g C m-2.
- The hourly file has 8,760 records, no NaNs, and no positive GPP fluxes.
- Peak UTC uptake tracks expected local solar noon with correlations of
  0.918--0.979 across five seasonal checks; median timing error is about
  0.7--1.0 hour.
- An every-fourth-week internal holdout gave correlation 0.941 and RMSE 0.961
  g C m-2 d-1. This is optimistic because the calibration target and gap prior
  are FLUXCOM-X and the test is neither independent-year nor independent-site.
- The uncertainty field is heuristic, combining regression residual and prior
  fractions; it is not a formal posterior uncertainty.

## 8. Reproducibility

Primary implementation:
`/home/cfranken/code/gitHub/OCO-dashboard/pilot/analyze_2021.py`.

Diagnostics and coefficients:
`/kiwi-data/Data/satellite/FLUXCOM-X/X-BASE/2021/pilot/analysis_summary.json`.

Weekly intermediate:
`/kiwi-data/Data/satellite/FLUXCOM-X/X-BASE/2021/pilot/sif_gpp_weekly_diurnal_c90_2021.nc`.

Hourly transport driver:
`/kiwi-data/Data/satellite/FLUXCOM-X/X-BASE/2021/pilot/sif_gpp_hourly_co2flux_c90_2021.nc`.
