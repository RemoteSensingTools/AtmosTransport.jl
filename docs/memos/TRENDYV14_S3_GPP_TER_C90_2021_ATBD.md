# Brief ATBD: TRENDYv14 S3 2021 C90 GPP and TER flux drivers

**Product family:** `TRENDYv14_S3_<MODEL>_gpp_diurnal_ter_daily_hourly_co2flux_c90_2021.nc`  
**Version/status:** experimental ensemble driver v0.1, 2026-07-22  
**Grid/time:** GEOS-native C90 cubed sphere; hourly UTC; calendar year 2021

## 1. Purpose and scope

These files provide separate gross-primary-production (GPP) uptake and total
ecosystem respiration (TER) emission drivers for atmospheric-transport
experiments using the TRENDYv14 S3 terrestrial-model ensemble. The processing
preserves every model's 2021 monthly carbon totals while supplying the daily
and hourly resolution required by the transport model.

The local archive contains enough information to prepare 23 model drivers.
Twenty-two models provide monthly `gpp`, `ra`, and `rh`. LPJ-GUESS provides
monthly `gpp` and `ra` but intentionally omits monthly heterotrophic
respiration; its documented annual `arh` is used as a constant monthly-mean
heterotrophic component. ISBA-CTRIP and OCN have no usable local GPP/respiration
inputs and are excluded. The output manifest is the authoritative record of
successful, skipped, and failed models.

## 2. Source quantities and sign conventions

Inputs are TRENDYv14 S3 monthly means in kg C m-2 s-1:

- `gpp`: positive photosynthetic uptake magnitude;
- `ra`: positive autotrophic respiration;
- `rh`: positive heterotrophic respiration; and
- `arh`: annual heterotrophic respiration for LPJ-GUESS only.

Negative input values are treated as nonphysical for these component fluxes
and clipped to zero before processing. TER is diagnosed independently of GPP:

`TER = max(0, ra) + max(0, rh)`.

GPP is written as a negative atmospheric flux and TER as a positive
atmospheric flux. Both output variables use kg CO2 m-2 s-1 over total C90 cell
area and use the carbon-to-CO2 mass conversion `44/12`.

## 3. Source-area interpretation

Source-cell areas are calculated from the latitude and longitude coordinates
using spherical Voronoi bounds. Static land fractions are applied before
remapping for the following land-normalized archives:

| Model | Applied factor |
|---|---|
| CABLE-POP | `1 - oceanCoverFrac` |
| CARDAMOM | embedded `land_fraction` |
| CLASSIC | `land_fraction` |
| CLM | `1 - oceanCoverFrac` |
| ELM-FATES | `1 - oceancoverfrac` |
| IBIS | `1 - oceanCoverFrac` |
| LPJml | `1 - oceanCoverFrac` |
| ORCHIDEE | `1 - oceanCoverFrac` |
| TEM/GDSTEM | `1 - oceanCoverFrac` |

Other model grid-box fluxes are interpreted as already normalized by total
grid-cell area. In particular, the LPJ-GUESS model README explicitly says not
to rescale its grid-box totals by land-cover fraction. Each output file and
JSON summary record the treatment actually applied.

## 4. Conservative spatial remapping

The regular source grids range from 0.25 to approximately 2.5 degrees. A
common 0.25-degree auxiliary grid samples the overlap between source cells and
C90 cells. Each auxiliary cell is assigned to its nearest C90 center on the
sphere. Within every source cell, the auxiliary weights are renormalized so
their sum equals the exact spherical source-cell area. For source flux density
`q_s`, target-cell density is

`q_t = sum_s(W_ts q_s L_s) / A_t`,

where `W_ts` is the normalized source-to-target area weight, `L_s` is the
applicable land fraction, and `A_t` is the spherical C90 area. Column
normalization of `W` makes the global carbon integral conservative to floating
point precision. The 0.25-degree sampling approximates regional cell overlap;
it is not an exact spherical polygon intersection.

## 5. Month-conservative temporal interpolation

December 2020 through January 2022 are read so the endpoints of 2021 can be
interpolated without persistence assumptions. For each C90 cell, monthly means
are placed at calendar-month midpoints and linearly interpolated to daily
midpoints. The provisional daily values are nonnegative and then rescaled
separately within every month:

`q_d = q_tilde_d M_m / mean_{d in m}(q_tilde_d)`,

where `M_m` is the original remapped monthly mean. Thus

`mean_{d in m}(q_d) = M_m`

for every active cell, preserving the source monthly total including the
actual number of days in each 2021 month. This procedure smooths month
boundaries without altering monthly or annual carbon budgets.

## 6. Hourly allocation

### GPP

The normalized 24-hour C90 fractions from the 2021 FLUXCOM-X monthly diurnal
product are reused to provide a common timing assumption across TRENDY models.
Where FLUXCOM-X has no support, a normalized positive cosine-solar-zenith curve
for the fifteenth day of the month is used. If `f(m,h,c)` sums to one over the
24 UTC hours, the atmospheric GPP flux is

`F_GPP(d,h,c) = -q_GPP(d,c) 24 f(m,h,c) (44/12)`.

The daily mean and all monthly totals are therefore unchanged.

### TER

No diurnal cycle is imposed. The interpolated daily mean is held constant for
all 24 UTC hours:

`F_TER(d,h,c) = q_TER(d,c) (44/12)`.

TER still varies smoothly from day to day, subject to exact monthly
renormalization.

## 7. Output and validation

Each model file contains 8,760 hourly records and two transport-ready fields:

- `GPP_CO2_FLUX(time,nf,Ydim,Xdim)`; and
- `TER_CO2_FLUX(time,nf,Ydim,Xdim)`.

Validation requirements are:

1. source and remapped annual integrals agree to floating point precision;
2. reconstructed daily means agree with each remapped monthly mean cell by
   cell;
3. GPP is nonpositive and TER is nonnegative in atmospheric sign convention;
4. TER has zero within-day range; and
5. each output has exactly 8,760 finite hourly records on C90.

Per-model annual GPP, RA, RH, TER, negative-input fractions, land treatment,
and conservation errors are recorded in adjacent JSON summaries. The ensemble
manifest supports restart and audit of partial processing.

## 8. Hovmöller diagnostics

Monthly output fluxes are integrated over time and C90 cell area, then binned
into 60 equal intervals of sin(latitude). Values are reported as Pg C month-1
per unit sin(latitude), so integration over the vertical coordinate recovers
the global monthly total. Separate GPP and TER products include:

- per-model absolute-value atlases on a common scale;
- per-model deviations from the multi-model mean;
- per-model deviations from the multi-model median; and
- multi-model mean and median reference panels.

The `.npz` diagnostic stores the underlying model-by-latitude-by-month arrays,
and the CSV stores annual model totals.

## 9. Reproducibility and known limitations

Primary generator:
`scripts/preprocessing/prepare_trendy_s3_gpp_ter_c90.py`.

Ensemble diagnostics:
`scripts/diagnostics/plot_trendy_s3_hovmoller_ensemble.py`.

Default output directory:
`/kiwi-data/Data/groupMembers/cfranken/AtmosTransport/fluxes/TRENDYv14/S3/C90/2021`.

Important limitations are that the hourly GPP timing is imposed from
FLUXCOM-X rather than predicted by each TRENDY model, TER has no subdaily
variability, the within-month evolution is interpolated rather than modeled,
and LPJ-GUESS heterotrophic respiration has annual rather than monthly source
resolution. These drivers preserve TRENDY monthly budgets but must not be
interpreted as native daily or hourly TRENDY output.
