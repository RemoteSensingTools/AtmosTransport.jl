# SIF-driven GPP interannual variability in XCO2, 2018-2025

**Status:** experiment built and validated 2026-08-28; transport run executed on
wurst. Facts below are measured, not estimated.

**Question.** How large an XCO2 signal, and with what spatial and temporal
structure, do we expect purely from interannual variability in satellite-observed
SIF interpreted as GPP?

## 1. Design

A GPP-only flux is a one-way sink: over this record it removes about 996 Pg C,
which is roughly 469 ppm of atmospheric drawdown. Detrending that after the fact
is possible but awkward, and a tracer that large also strains Float32 and risks
tripping the PPM positivity limiters.

Because advection and diffusion are **linear in the tracer**, the response to the
full flux decomposes exactly:

    XCO2(full flux) = XCO2(climatological seasonal flux) + XCO2(flux anomaly)

So the interannual signal is obtained *exactly*, with no statistical detrending,
by transporting the flux anomaly directly. The run therefore carries two tracers
on one met stream:

| tracer | forcing | carrier | purpose |
|---|---|---|---|
| `co2_gpp_anom` | flux minus smoothed day-of-year climatology | 100 ppm | **primary**: the IAV signal |
| `co2_gpp_full` | full GPP uptake flux | 900 ppm | raw drawdown, detrend afterwards |

Carriers keep both fields positive, which the PPM limiters assume. The anomaly
carrier is deliberately small (100 ppm, not the 400 ppm used by the 2021 pilot)
because Float32 transport rounding scales with the background, and the anomaly
signal is only order 1 ppm. Subtract the carrier from the output VMR.

## 2. Inputs

1. **TROPOMI S5P-PAL SIF, gap-filled daily, GEOS-native C90.**
   `/kiwi-data/Data/satellite/TROPOMI/TROPOMI_SIF_S5P-PAL/regridded/interpolated/`
   `TROPOMI_sif_20180501_20251231_C90_daily_gt-2SIF743lt5_SIF743ERRORlt10_filled.nc`
   Variable `sif_743_corr_filled` (length-of-day-corrected SIF at 740 nm),
   2802 contiguous daily steps, 2018-05-01 to 2025-12-31.
   The grid is **bit-identical** to the validated 2021 pilot driver
   (max |dlat| = max |dlon| = 0), so no horizontal regridding occurs anywhere.
2. **Frozen 2021 SIF-to-GPP calibration**, from
   `/kiwi-data/Data/satellite/FLUXCOM-X/X-BASE/2021/pilot/analysis_summary.json`
   (`gpp_models`: one global fit plus 96 month x latitude-band robust fits).
3. **Vegetated land fraction on C90**, from `fluxcom_x_c90_2021.npz`. This is
   FLUXCOM-X's land fraction, *not* the ETOPO1 `land_fraction` carried in the SIF
   file; only the former is consistent with the calibration. It yields 16,343
   cells with fraction > 0.05, matching the pilot ATBD exactly.
4. **Met:** ERA5-N320-to-C90, L66, Float32, format-4 transport binaries,
   `.../met/era5/n320_to_c90/transport_binary_v4_l66_f32_no_convection`.
   Complete for 2014-2025. Advection + pressure + PBL + exact TM5 DKG diffusion;
   **no convection mass fluxes**, so convection is off.

## 3. Method

    SIF_OCO2scale = 0.040777 + 0.877893 * SIF_TROPOMI
    GPP           = max(0, b0[month, lat_band] + b1[month, lat_band] * SIF_OCO2scale)
    GPP_CO2_FLUX  = -GPP * land_fraction * (44/12) * 1e-3 / 86400

GPP is in g C m-2 d-1 per vegetated area; the flux is kg CO2 m-2 s-1 per *total*
cell area, negative for uptake. `44/12` reproduces the 2021 pilot exactly.

**The calibration is held fixed in time on purpose.** Every year is converted
with identical coefficients, so all interannual variability in GPP comes from
observed SIF rather than from the FLUXCOM-X training target. This is also
forced by data availability: FLUXCOM-X X-BASE covers only 2014-2021, so any
re-fitted product would break at 2022.

**Dormancy.** The pilot suppressed spurious cold-season high-latitude
photosynthesis by snapping dormant cells to the FLUXCOM-X prior, which is
unavailable after 2021. Measurement shows no replacement gate is needed: boreal
(>60N) January-February raw SIF is reliably negative (mean -0.073, 98.9% below
zero, p99 = +0.001), the GPP zero-crossing sits at SIF_raw = -0.048, and the
`max(0, .)` clip alone leaves only **0.096 Pg C** of >60N winter GPP.

**Daily vs weekly.** The calibration was fit on weekly means, so applying it to
daily SIF rectifies slightly through the clip. Measured: daily gives
128.238 Pg C for 2021 against 127.807 Pg C for 7-day-smoothed SIF, i.e.
**+0.34%**. Daily resolution is kept.

**Anomaly.** A day-of-year climatology is built over the seven complete years
2019-2025 (February 29 folds onto February 28), smoothed with a 15-day circular
running mean, and subtracted. Over its own fit years the anomaly integrates to
-0.009 Pg C, i.e. mass-neutral by construction; the record total is -1.93 Pg C
(-0.91 ppm), essentially all of it the partial 2018 year, which is excluded from
the climatology.

## 4. Validation

| check | result |
|---|---|
| C90 cell areas close on the sphere | 4e-11 relative |
| land cells with fraction > 0.05 | 16,343 (pilot ATBD: 16,343) |
| independent flux-integral vs GPP integral | 995.78 vs 995.78 Pg C |
| 2021 total vs 2021 pilot driver | 128.238 vs 126.060 Pg C (+1.73%) |
| 2021 annual-mean spatial correlation vs pilot | **r = 0.9975** (17,073 cells) |
| 2021 global daily time-series correlation vs pilot | **r = 0.9968** |
| pilot driver reintegrated | 126.060 Pg C, matching its ATBD exactly |

The +1.73% excess over the pilot is expected and accounted for: no FLUXCOM
blending or dormancy snap (the pilot's blend pulls toward a slightly lower
prior) plus the +0.34% daily rectification.

Annual GPP and the resulting flux anomaly:

| year | GPP (Pg C) | anomalous C removed (Pg C) | as ppm of burden |
|---|---|---|---|
| 2018 (May-Dec) | 93.131 | -1.918 | -0.903 |
| 2019 | 124.998 | -3.876 | -1.825 |
| 2020 | 128.454 | -0.690 | -0.325 |
| 2021 | 128.238 | -0.635 | -0.299 |
| 2022 | 128.710 | -0.164 | -0.077 |
| 2023 | 128.837 | -0.036 | -0.017 |
| 2024 | 130.318 | +1.174 | +0.553 |
| 2025 | 133.092 | +4.219 | +1.986 |

Positive means carbon removed in **excess** of climatology, i.e. anomalously
strong uptake. **The XCO2 effect has the opposite sign**: 2019 removes 3.876 Pg C
*less* than climatology, so it should show a *positive* XCO2 anomaly, while 2025
removes 4.219 Pg C more and should show a negative one. In the output fields the
convention is stated directly: positive `xco2_anom` = anomalously weak uptake.

Runtime validation: `[gpu verified] backend=cuda backing=CuArray device=NVIDIA
L40S`, PPM, Float32, TM5 DKG diffusion, `NoConvection`, both tracers present. A
3-day smoke run reproduced the expected drawdown (0.45 ppm vs 0.49 ppm
predicted) and held the anomaly tracer's global mean at 100.00 ppm with no
drift.

## 4b. Expected magnitude, and the trend-versus-IAV split

Integrating the anomaly flux under a well-mixed assumption (an exact prediction
of the transported tracer's *global mean*, since mass is conserved) gives the
global-mean XCO2 anomaly:

| year | annual mean (ppm) | end of year (ppm) | implied growth-rate anomaly (ppm/yr) |
|---|---|---|---|
| 2018 (May-Dec) | +0.331 | +0.903 | - |
| 2019 | +1.971 | +2.727 | +1.825 |
| 2020 | +2.807 | +3.053 | +0.325 |
| 2021 | +3.243 | +3.352 | +0.299 |
| 2022 | +3.424 | +3.429 | +0.077 |
| 2023 | +3.466 | +3.446 | +0.017 |
| 2024 | +2.972 | +2.893 | -0.553 |
| 2025 | +1.879 | +0.907 | -1.986 |

Record range -0.015 to +3.650 ppm, i.e. **3.67 ppm peak-to-peak**. Positive
means anomalously weak uptake and therefore higher XCO2.

**But this is trend-dominated, and the trend is the least trustworthy part.**
Over the seven complete years the annual GPP rises at **+1.022 Pg C/yr** with
linear-fit R^2 = 0.824, and that trend accounts for **82% of the annual
variance**:

| | std of annual anomalies | as growth-rate anomaly |
|---|---|---|
| raw | 2.431 Pg C | 1.145 ppm/yr |
| detrended | 1.020 Pg C | **0.480 ppm/yr** |

Detrended residuals (Pg C): 2019 -0.886, 2020 +1.548, 2021 +0.310,
2022 -0.240, 2023 -1.134, 2024 -0.675, 2025 +1.077. Their cumulative XCO2
excursion peaks at only **0.507 ppm**, against 3.65 ppm when the trend is kept.

So the headline answer is bracketed: **~3.7 ppm** of cumulative global-mean XCO2
signal if the SIF trend is taken at face value, versus **~0.5 ppm** from
detrended year-to-year variability alone. Since TROPOMI degradation and
calibration drift alias directly into a fixed-calibration GPP trend, the
detrended number is the defensible one for genuine IAV, and the trend should be
treated as an upper bound contaminated by instrument drift.

For scale: the observed atmospheric CO2 growth-rate IAV is roughly 0.8-1.0 ppm/yr
(1 sigma). GPP IAV alone therefore implies growth-rate anomalies comparable to or
larger than the *total* observed IAV, which is the expected signature of GPP and
respiration anomalies largely cancelling in net flux.

## 4c. Is the GPP trend real greening or TROPOMI drift?

The trend carries 82% of the annual variance, so this matters more than anything
else in the experiment. Three tests, all on the SIF itself:

**Calibration structure is not the cause.** Recomputing annual GPP with the
single global fit instead of the 96 month x latitude-band fits changes the trend
only from +1.022 to +1.077 Pg C/yr and the detrended scatter from 1.020 to
1.089 Pg C. The trend is a property of the SIF record, not of the calibration.
Area-weighted land-mean raw SIF rises **+1.26%/yr**.

**The trend is not a uniform additive offset, but it is not proportional
either.** Binning land cells by decile of annual-mean SIF, the *absolute* trend
is far flatter than proportional (+0.0008 to +0.0025 per year across all
deciles) while the *relative* trend falls from +10.7%/yr in decile 3 to
+0.45%/yr in decile 10. Regionally the pattern is strongly structured, which a
detector artifact would not be: Amazon **+0.04%/yr** (essentially none) against
E China **+2.24%/yr**, India **+1.31%/yr**, US Midwest **+1.07%/yr** --
i.e. concentrated exactly where agricultural intensification and afforestation
are documented.

**Decisive test: the dormant season.** A positive SIF trend where there is no
photosynthesis can only be instrumental. At >55N land:

| season | mean SIF | trend/yr | t |
|---|---|---|---|
| DJF (dormant) | -0.0675 | +0.00025 | +2.4 |
| JJA (growing) | +0.1761 | +0.00345 | +5.3 |

The dormant-season trend is only **7% of the growing-season trend**, and the
austral-winter control (>40S land, JJA) gives +0.00010/yr. So the
signal-independent drift component is small, and the trend is overwhelmingly
growing-season and vegetation-phase-locked.

**Conclusion, with the residual caveat.** The trend behaves like real greening,
not like a detector offset, so the 3.7 ppm figure is more defensible than a
blanket "assume instrument drift" would suggest. What these tests *cannot*
exclude is a **multiplicative gain drift**, which would scale with signal, appear
only in the growing season, and therefore mimic proportional greening exactly.
The decile analysis argues against a pure gain drift (absolute trend is flatter
than proportional), but does not settle it. Report both bracketing numbers, and
prefer the detrended ~0.5 ppm when the claim depends on the trend being real.

## 4d. Transported result (run completed 2026-08-28)

2802 daily C90 snapshots, 2018-05-01 to 2025-12-31. Forward run 8065 s
(2.24 h) on one L40S after a ~13 min startup pass over the 2802 binary headers;
4913 GiB of met read; zero NaN, CFL or warning lines.

Global-mean XCO2 anomaly (ppm, carrier removed), and the full-flux tracer:

| year | `xco2_anom` | `xco2_full` |
|---|---|---|
| 2018 (May-Dec) | +0.331 | -24.64 |
| 2019 | +1.963 | -72.54 |
| 2020 | +2.796 | -132.25 |
| 2021 | +3.233 | -192.41 |
| 2022 | +3.413 | -252.72 |
| 2023 | +3.455 | -313.18 |
| 2024 | +2.963 | -374.21 |
| 2025 | +1.874 | -435.89 |

The transported values reproduce the well-mixed prediction of section 4b to
about 0.01 ppm in every year, as they must if mass is conserved.

**Validation gates on the completed run**

- *Mass conservation.* Transported global mean vs the well-mixed integral of the
  driving flux: `xco2_anom` rms residual 0.0093 ppm on a 3.665 ppm range
  (**0.25%**), final +0.9078 vs +0.9073 predicted; `xco2_full` rms 0.80 ppm on a
  468.7 ppm range (**0.17%**), final -467.42 vs -468.82 predicted. Residuals
  scale with signal, which is the Float32 rounding signature, not a leak.
- *Linearity.* `full - anom` is by construction the response to the
  climatological flux, which must contain no interannual variability. Its
  year-over-year steps are -60.54, -60.59, -60.49, -60.50, -60.53, -60.59 ppm:
  **std/mean = 0.0006**. The decomposition holds to six parts in ten thousand.
- *Sequence.* 2802 contiguous daily snapshots, no gaps; output grid latitudes
  bit-identical to the flux grid.

**Spatial structure.** Regional structure is modest next to the global mean:
the area-weighted standard deviation of the annual-mean anomaly across cells is
0.10-0.28 ppm and the p5-p95 spread 0.3-0.9 ppm, against a 2-3.5 ppm global
mean. Detectability therefore rests on the global and hemispheric mean, not on
regional gradients.

The informative spatial signal is the **interhemispheric gradient and its
reversal**. NH annual means run +2.05, +2.83, +3.39, +3.56, +3.48, +2.83, +1.66
for 2019-2025 while SH runs +1.87, +2.77, +3.08, +3.26, +3.43, +3.10, +2.08:
the NH leads on the way up and the SH peaks a year later (2023 vs 2022),
consistent with the roughly one-year interhemispheric exchange time. Through
2019-2022 the NH sits above the global mean and the SH below it; by 2024-2025
that flips as NH uptake strengthens. Meridional departures from the global mean
reach about +/-0.9 ppm.

Figures: `sif_gpp_iav_global_timeseries.png`, `sif_gpp_iav_annual_maps.png`,
`sif_gpp_iav_zonal_hovmoller.png`. The maps and the Hovmoller plot the
*departure from the contemporaneous global mean*; plotting the raw field is
uninformative because the global mean dwarfs the spatial structure.

## 4e. Hovmoller views, and why the anomaly tracer beats post-hoc detrending

`plot_sif_gpp_iav_xco2.py` writes equal-area **sin(latitude)** Hovmollers for
both tracers, each as two stacked panels: detrended (a least-squares line in
time removed per band), and additionally deseasonalised. The seasonal cycle is
removed by a **4-harmonic fit**, not a day-of-year climatology -- with only 7-8
samples per calendar day the latter injects its own sampling noise, which
appears as vertical striping.

The two tracers behave very differently, and the contrast is the point:

- `xco2_anom`: detrended +/-2.39 ppm, deseasonalised +/-2.34 ppm. Almost no
  change, because the driving flux was *already* defined against a day-of-year
  climatology, so the transported field carries essentially no mean seasonal
  cycle.
- `xco2_full`: detrended +/-18.65 ppm -- the classic NH growing-season drawdown
  wave propagating southward -- collapsing to +/-4.37 ppm once deseasonalised.

Comparing the two routes to the same answer:

| | anomaly tracer vs detrended+deseasonalised full tracer |
|---|---|
| global-mean series | **r = 0.994**, rms difference 0.098 ppm on ~0.89 ppm amplitude |
| per-latitude-band pattern | **r = 0.659**, rms difference 1.075 ppm |
| rms amplitude | 0.908 ppm (anomaly) vs 1.428 ppm (full route) |

So post-hoc detrending recovers the *global* interannual signal almost exactly,
but at the zonal level it leaves about 1 ppm of residual seasonal contamination
and inflates the apparent amplitude by ~57%. The reason is that the seasonal
cycle amplitude itself varies from year to year, and a fixed-amplitude harmonic
fit cannot remove that; a +/-18.7 ppm seasonal cycle with a few percent of
interannual variation leaves ~1 ppm behind. The anomaly tracer removes the
climatology exactly, at the flux level, before any transport happens. Use it for
anything spatially resolved; either route is fine for the global mean.

## 4f. Met-archive homogeneity: is there convection, and does it change?

Asked because ERA5 convection files were missing for 2024-2025, which could
have changed the run type mid-record and injected spurious interannual
variability straight into the signal. It did not. Reading the JSON header of
**all 2802 binaries** in the run window:

- `include_tm5conv = False`, `include_cmfmc = False`, `include_dtrain = False`,
  `tm5_convection_source = "none"`, and `n_entu = n_detu = n_entd = n_detd =
  n_cmfmc = n_dtrain = 0` in **every single file**. There is no convection
  anywhere in this archive, in any year, and no discontinuity at 2024.
- Of the 86 header fields, the **only** ones that ever differ are the adaptive
  CFL substep schedule (`steps_per_window`, `steps_per_window_by_window`) and
  the Poisson scale derived from it -- both are meant to vary daily. No other
  field differs anywhere in the record.
- 33 days of 2802 (1.2%) were written with `time_step_schedule = "constant"`
  rather than `"per_window"`. They are scattered roughly evenly across all
  eight years rather than batched, both contracts are self-consistent and
  supported by the reader, and the run logged zero CFL warnings.

So the transport configuration is homogeneous across the record: PPM advection,
pressure, PBL fields and exact TM5 DKG diffusion, no convection, throughout.
The absence of convection is a uniform limitation of the experiment, not a
source of spurious interannual variability.

## 4g. Transport-only control tracer

`config/runs/sif_gpp_clim_control_c90_2018_2025.toml` drives a single tracer
with `GPP_CO2_FLUX_SEAS`: the day-of-year GPP climatology with each cell's
annual mean removed. The forcing is **byte-identical in every year** (verified:
2021-07-15 and 2022-07-15 slices are exactly equal, while the corresponding
`ANOM` slices are not), so the tracer has no flux interannual variability by
construction and anything interannual it develops comes purely from transport.

Removing each cell's annual mean is what makes this affordable in Float32: the
raw climatology draws down ~468 ppm over the record and would need a large
carrier, whereas a zero-annual-mean seasonal cycle merely oscillates, so a
300 ppm carrier suffices.

*Known small artifact:* leap years apply the 28 February climatology slot twice
(29 February folds onto it), so a leap year's forcing integrates to -0.082 Pg C
rather than 0, about 0.039 ppm. Non-leap years integrate to 2.5e-9 Pg C. This
is ~4% of the signal discussed below and affects only 2020 and 2024.

**Preliminary result from the linearity shortcut.** Before running the
dedicated tracer, the same quantity is available for free: by linearity
`xco2_full - xco2_anom` is exactly the response to the identical-every-year
climatological flux. Taking equal-area zonal bands and removing the trend and
the mean seasonal cycle:

| quantity | rms (ppm) |
|---|---|
| transport-only control, zonal bands | **1.075** |
| flux-driven anomaly tracer, zonal bands | 0.908 |
| transport-only control, global mean | **0.098** |
| flux-driven anomaly tracer, global mean | 0.888 |

Two things follow. First, transport contributes essentially **nothing** to the
global-mean interannual signal (0.098 vs 0.888 ppm), exactly as mass
conservation requires. Second, in the **zonal** structure that a Hovmoller
shows, transport-driven variability is *comparable to and slightly larger than*
the flux-driven variability (1.075 vs 0.908 ppm). Meridional redistribution by
year-to-year differences in winds and boundary-layer mixing is therefore a
first-order term: zonal or regional XCO2 interannual variability cannot be read
as GPP interannual variability without this control.

**SETTLED by the dedicated control run (completed 2026-08-29).** The
difference-derived 1.075 ppm was wrong, and so was a first pass at the dedicated
run. The correct values, from the completed control tracer over 2019-2025 with
2018 dropped as spin-up:

| interannual variability (rms, ppm) | flux-driven | transport-only | ratio |
|---|---|---|---|
| zonal bands (equal-area sin lat) | 0.644 | **0.244** | **0.38** |
| global mean | 0.627 | **0.012** | 0.02 |

Correlation between the two zonal fields is r = -0.002, i.e. they are
independent, as they should be. Per-year the transport-only signal is
remarkably steady (0.226-0.261 ppm) while the flux-driven one varies
(0.163-0.256 ppm) and additionally carries between-year structure, which is why
its record-wide rms is much larger than any single year.

**Conclusions.** Transport contributes essentially nothing to the *global-mean*
interannual signal (2%), exactly as mass conservation requires. In *zonal*
structure it contributes about **38%** as much variance-by-rms as the flux does.
That is a real and material term -- zonal or regional XCO2 interannual
variability should not be read as GPP interannual variability without this
control -- but it is **not** the "comparable to or larger than" claim made
earlier from the difference method.

**The contamination is scale-dependent, and this is the number to quote.**
Transport variability is high-frequency and meridionally incoherent, whereas the
flux-driven signal is smooth and persists for a year at a time. Averaging
therefore suppresses transport far faster than it suppresses the flux signal:

| averaging scale | flux-driven | transport-only | ratio |
|---|---|---|---|
| daily | 0.644 | 0.244 | 0.38 |
| monthly | 0.626 | 0.169 | 0.27 |
| seasonal (90 d) | 0.589 | 0.130 | 0.22 |
| **annual mean** | 0.615 | **0.089** | **0.14** |

So for the annual-mean zonal analysis that this experiment is actually for,
transport contributes about **14%** as much rms as the flux -- worth subtracting,
but the signal is flux-dominated. Only at daily resolution does transport reach
38%.

**Two methodological errors produced the inflated earlier numbers; both are
worth remembering.**

1. *Differencing two large-carrier Float32 tracers.* `xco2_full - xco2_anom`
   conflates rounding noise with a genuine forcing difference (`clim` versus
   `clim` minus each cell's annual mean). Measured against the dedicated
   tracer, the two agree in structure (r = 0.968) but differ by 1.507 ppm rms.
2. *Detrending before deseasonalising.* These tracers carry a seasonal cycle far
   larger than their trend -- the control's seasonal std is 2.55 ppm against a
   true drift of +0.009 ppm/yr -- and a finite window makes the seasonal cycle
   not quite orthogonal to a linear basis. A global linear fit to the control
   picks up a **spurious -0.118 ppm/yr**, and removing it leaves a ~0.25 ppm
   sawtooth in the day-of-year residual that looks exactly like interannual
   variability. The fix is `interannual()` in
   `compare_sif_gpp_transport_control.py`: a **per-day-of-year linear detrend**,
   which removes the mean seasonal cycle and any slow evolution of it in one
   step, and never fits a trend through the seasonal cycle.

The second point also handles residual spin-up. The control starts from a
uniform field and its seasonal cycle is still equilibrating at the end of the
record -- at day-of-year 15 June the detrended global mean runs 1.86, 2.02,
2.14, 2.25, 2.37, 2.53, 2.63 ppm across 2019-2025, a monotonic drift as the
stratosphere fills. Dropping one year of spin-up is not enough; the per-day-of-
year detrend removes what remains.

## 4h. The GPP driver itself

`scripts/diagnostics/plot_sif_gpp_input_hovmoller.py` shows the model *input* on
the same equal-area sin(latitude) bands, as three stacked panels: raw, the mean
seasonal cycle, and raw minus that cycle.

**Units are total carbon per band (Pg C/yr), not flux density** (`--mode
density` gives the latter). The equal-area binning makes the two nearly
proportional -- band areas vary by only 2.77% rms -- but the absolute view is
what lets you read carbon off the plot, and land fraction per band ranges from
**0.2%** (Southern Ocean near 51 S) to **75%**, so the distinction matters
conceptually. The bands sum to **129.8 Pg C/yr**, consistent with section 4, and
the peak band reaches **19.41 Pg C/yr** near 50-60 N at midsummer.

**The seasonal cycle is empirical, not a harmonic fit.** It is a day-of-year
climatology averaged over complete years (7-9 samples per slot; 29 February
folds onto 28 February) and smoothed with a 31-day circular running mean. This
replaced an earlier 4-harmonic fit: four harmonics cannot represent the sharply
non-sinusoidal boreal growing season, and the residual aliased into the anomaly
as horizontal striping poleward of 60 N. With the empirical cycle that artifact
is gone, and the alternating dipoles that remain at high latitude are real --
they are growing-season onset shifting earlier or later than climatology.

**Daily SIF retrieval noise exceeds the interannual signal.** Unsmoothed, the
anomaly panel is noise. A 31-day running mean cuts the band rms from 0.2801 to
0.1801 Pg C/yr, so **36% of the unsmoothed variance was day-to-day retrieval
noise**. The long-term trend is retained by default because it is the greening
signal under investigation; `--detrend` removes it.

The legible structure: a progression from predominantly negative anomalies in
2018-2019 to predominantly positive by 2024-2025 across most latitudes (the
greening trend of section 4c), and a distinct negative excursion near 20-30 S in
2019, the weakest GPP year of the record.

## 4i. Why the detrended full tracer shows a large Southern-Hemisphere cycle

The detrended (not deseasonalised) full-uptake tracer shows a ~10 ppm seasonal
cycle in the SH extratropics. That is expected for this tracer, is not local
photosynthesis, and is larger than the real world -- all three points matter.

**It cannot be local.** Only **4.2% of global GPP** lies south of 30 S
(5.45 of 129.80 Pg C/yr) and its seasonal peak-to-peak is 6.45 Pg C/yr, against
**111.31 Pg C/yr north of 30 N**:

| region | mean GPP (Pg C/yr) | seasonal p2p | p2p/mean |
|---|---|---|---|
| north of 30 N | 40.82 | 111.31 | 2.73 |
| 0-30 N | 39.79 | 30.75 | 0.77 |
| 30 S-0 | 42.84 | 30.85 | 0.72 |
| south of 30 S | 5.52 | 6.45 | 1.17 |

**It is the oscillating interhemispheric gradient.** The NH and SH annual cycles
of the detrended tracer are **anti-correlated, r = -0.951**, with the NH maximum
on day 347 and the SH maximum on day 112 (130-day offset). A locally driven SH
cycle would peak in SH winter and correlate *positively* at a six-month lag.
Subtracting the global-mean cycle from an SH band also *increases* its amplitude,
which only happens if the band is anti-phased with the global mean. The
mechanism: this is an uncompensated sink, so the whole atmosphere drains at
about 60 ppm/yr and that drainage is strongly modulated by NH summer. Detrending
removes only the mean linear rate, leaving the deviation from linear, which every
latitude feels. The NH swings hard (34.73 ppm p2p) because it sits on the sink;
the SH drains more slowly and therefore reads high against a rapidly falling
global mean.

**The amplitude is inflated relative to observations.** SH/NH here is
10.11/34.73 = **29%**, whereas observed XCO2 gives roughly 1 ppm in the SH
against ~15 ppm at NH high latitudes, about **7%**. The difference is the
missing respiration: real NEE nearly cancels annually so the net forcing is
small and the SH barely responds, while GPP alone leaves a large uncompensated
sink whose seasonal modulation propagates globally.

This is a limitation of the full-uptake tracer, not a defect in the transport,
and it is a further argument for treating `xco2_anom` as the primary product:
its seasonal cycle accounts for only **0.3%** of its detrended variance
(versus **93.4%** for `xco2_full`), so it is immune to this artifact.

## 4j. How to visualise this without the artifact

Four options, in order of how well they work.

**1. Use `xco2_anom`.** For the interannual question it is immune: its seasonal
cycle is 0.3% of its detrended variance. Nothing further is needed.

**2. `--tracer balanced` (best for seasonal structure).** By linearity,
`xco2_anom` is forced by `F - C` and the control tracer by `C - M`, so their
**sum is forced by `F - M`** -- the full flux with each cell's annual mean
removed. That is annually balanced everywhere, so there is no secular drawdown
and no inflated interhemispheric seesaw, while the real seasonal cycle and the
real interannual variability are both retained. No extra run is needed beyond
the control. Measured on the control tracer alone (which is already annually
balanced), over 567 post-spin-up days:

| | SH/NH seasonal p2p | NH-SH correlation |
|---|---|---|
| full uptake tracer (uncompensated sink) | 29% | -0.951 |
| annually balanced tracer | **16.1%** | -0.956 |
| observed XCO2 (real NEE) | ~7% | -- |

Balancing roughly halves the inflation. The residual gap to ~7% is expected and
is *not* a model error: removing a constant annual mean is a crude respiration
proxy, equivalent to assuming respiration runs at the local annual-mean GPP rate
year-round. Real respiration peaks in summer, partly in phase with GPP, and so
cancels more of the seasonal signal. Closing that gap would require an actual
respiration seasonality (a Q10 temperature-driven term), which is outside the
scope of this experiment.

**3. `--minus-global`.** Subtracting the contemporaneous global mean removes the
drawdown and its seasonal modulation, leaving a clean, symmetric NH-SH seesaw
pivoting near 10 N. Good for showing *structure*; do not quote amplitudes from
it, because the seesaw itself is still inflated by the uncompensated sink.

**4. A per-latitude trend does not help, and is already applied.** `_detrend`
fits a separate line to every band. The per-band trends of the full tracer span
only **-58.74 to -61.01 ppm/yr**, a 3.75% spread, so a per-latitude trend is
almost identical to a single global one. Detrending removes the *mean* drawdown
rate; the residual is the seasonal *modulation* of that rate, which every
latitude feels. A per-band quadratic barely changes it either (SH rms 3.676 ->
3.644 ppm).

## 4k. Net-flux run: GPP minus a flat respiration (launched 2026-08-29)

`config/runs/sif_gpp_net_flatreco_c90_2018_2025.toml` carries two tracers on the
same met stream, driven from the v3 flux file:

- `co2_reco_flat`: each cell's climatological annual-mean GPP uptake,
  sign-flipped and constant in time -- a flat respiration proxy emitting
  **128.96 Pg C/yr** (equal to the climatology-mean GPP by construction).
  10 ppm carrier; it only gains mass (~469 ppm over the record).
- `co2_gpp_net`: `GPP_CO2_FLUX_NET = GPP + flat Reco`, the annually balanced
  net flux. Full seasonal cycle, full interannual variability, no secular
  drawdown. 150 ppm carrier (the analogous control tracer's column minimum was
  -59 ppm).

By linearity the net response must equal `xco2_anom + xco2_clim` (verified
already at the flux level: `NET - (ANOM + SEAS)` is at most 5.7e-14 kg m-2 s-1,
pure Float32 rounding, and `NET == FLUX + RECO` is bitwise). The direct run is
still worth having: a single ~150 ppm-carrier tracer is more precise than the
sum of two carriers, it provides the clean shareable product, and comparing it
against the tracer-field sum closes the linearity argument end-to-end through
the transport itself rather than only through the forcing.

Because the cs_native loader requires a `(time, nf, Ydim, Xdim)` variable,
`RECO_FLAT_CO2_FLUX` is stored as a full 2802-step series of identical slices
(zlib makes the constant data nearly free; the v3 file is 633 MB).

Smoke-validated 2026-08-29: after 3 days the reco tracer's global mean was
10.532 ppm against 10.499 expected from 128.96 Pg C/yr; the net tracer held its
carrier. Existing v1/v2 variables are bit-identical in v3.

**Completed 2026-08-29** (2802 snapshots, zero errors, 6.3 h wall). All closure
checks pass on the full record:

- `net` vs `anom + clim`: global-mean rms difference **0.0008 ppm** (max
  0.0027) on a 2.7 ppm signal; cell-level rms 0.0124 ppm, correlation
  0.999998. Linearity closes end-to-end through the transport.
- `net` vs `full + reco_flat`: agrees at 0.036 ppm rms -- 3x worse, the
  measured large-carrier Float32 penalty, confirming the direct small-carrier
  run as the definitive product.
- Mass conservation of `co2_gpp_net`: rms residual 0.0234 ppm against the
  well-mixed flux integral (10.6 ppm signal range); final -3.032 vs -3.038
  predicted. `co2_reco_flat` rises at 60.52 ppm/yr with r^2 = 1.000000 against
  60.7 expected from its 128.96 Pg C/yr source.

The direct net tracer reproduces the sum-route numbers exactly (SH/NH seasonal
ratio 16.4% both ways; interannual zonal rms 0.889 vs 0.888 ppm), so all
conclusions drawn from the sum stand. Definitive figure:
`plots/sif_gpp_iav/sif_gpp_net_definitive.png` (as-transported plus
interannual-component panels). The flat-Reco tracer also yields the
interhemispheric exchange time of section 4l and the stationary source
gradient (+8 ppm at the source latitudes to -24 ppm over Antarctica,
established over ~1.5 yr).

## 4l. Interhemispheric exchange time of this transport

The flat-Reco tracer is a constant source of known geography, so its
quasi-steady hemispheric gradient measures the interhemispheric exchange time
directly. The annual-mean NH-SH column-burden gradient converges by 2020 and
holds at ~11.0 ppm (2019-2024 means: 11.12, 12.02, 11.05, 11.08, 11.04,
11.61). With the source split 80.60/48.36 Pg C/yr NH/SH:

    tau_ex = Delta * 2.124 / (F_N - F_S) = 10.97 * 2.124 / 32.2 = **0.72 yr**

(identical under the export-flux definition). The SF6-based observational
benchmark is 1.3-1.4 yr, but that is **not like-for-like**: tau_ex depends
strongly on source geometry, SF6 is ~95% NH-midlatitude, and our source is
62.5% NH with a heavy tropical component -- sources near the ITCZ exchange
legitimately faster. How much of the 0.72 vs 1.3 difference is real model bias
is therefore not determinable from this tracer.

On the visual impression that NH seasonality "spills over too quickly": (1) the
NH amplitude itself is ~3x inflated by the flat-Reco approximation, so a correct
fractional leakage still looks large; (2) 32.6% of the GPP source lies in
0-30 S with a 30.8 Pg C/yr antiphase seasonal cycle, so much of the mid-SH
signal is locally forced rather than cross-equatorial; (3) Hovmoller diagonals
are phase propagation, faster than mass exchange. A pure-spillover lag model
brackets the measured SH/NH amplitude ratio (16.4%) between its tau = 0.72
prediction (20%) and tau = 1.3 prediction (12%).

The genuine suspicion is the missing convection (upper-troposphere
cross-equatorial pathways underrepresented; bias direction unclear). Decisive
test, not yet run: one constant tracer with SF6-like geometry (GridFED fossil
map, or the Reco map masked to >30 N), 2-3 years, same tau metric, compared
against 1.3-1.4 yr.

## 4m. TRENDY v14 S3 leg: three DGVMs through the same transport

Completed 2026-08-30: CLM6.0, JULES-ES and ORCHIDEE (gpp/ra/rh, monthly,
1700-2024), conservatively remapped to C90 by the 2021 pilot's machinery with a
new leap-aware continuous daily interpolation
(`scripts/preprocessing/prepare_trendy_s3_multiyear_c90.py`), transported
2018-05-01 to 2024-12-31 as 9 tracers (per model: GPP 900 ppm carrier, TER
10 ppm, **directly simulated NEE** 150 ppm). Validation: native-grid CLM 2021
GPP equals the C90 pipeline to 0.001%; renormalization exact to 3e-15; 2437
snapshots, zero errors; within-run `nee - (gpp + ter)` shows the expected
large-carrier penalty (0.03-0.06 ppm rms), so the direct NEE tracers are the
products. Annual GPP: CLM ~148, JULES-ES ~158, ORCHIDEE ~120 Pg C/yr; NEE
(no fire/land-use return flux) -6 to -12 Pg C/yr.

**Real respiration seasonality fixes the NH amplitude but not the SH/NH
ratio** (post-spin-up, 55 N vs 40 S, empirical DOY climatology):

| tracer | NH p2p (ppm) | SH p2p (ppm) | SH/NH | NH-SH corr |
|---|---|---|---|---|
| SIF net (flat Reco) | 29.5 | 5.1 | 17.1% | -0.965 |
| CLM6.0 NEE | 14.6 | 5.2 | 35.3% | +0.293 |
| JULES-ES NEE | 12.4 | 4.2 | 33.8% | +0.556 |
| ORCHIDEE NEE | 8.4 | 3.0 | 35.4% | +0.402 |
| observed XCO2 | ~10 | ~1 | ~7% | |

In-phase summer respiration collapses the NH amplitude from 29.5 to 8-15 ppm,
bracketing the observed ~10. But the SH amplitude barely changes (3-5 ppm) and
the NH-SH correlation flips positive: the DGVMs' SH signal is no longer the
flat-Reco drawdown seesaw but genuine (tropical/subtropical) SH land NEE
seasonality. Against the observed ~1 ppm this is several times too large; the
candidate explanations are (a) DGVM SH NEE seasonality genuinely too strong,
(b) no ocean flux in the tracer -- the real SH ocean cycle partially cancels the
land signal in observed XCO2, (c) missing fire seasonality, and (d) possibly
fast interhemispheric exchange (section 4l). This experiment cannot separate
them.

**GPP-Reco covariation cancels most interannual variability.** Interannual rms
(per-day-of-year detrend, post-spin-up):

| | global mean | zonal bands |
|---|---|---|
| SIF net (GPP IAV only) | 0.497 | 0.729 |
| CLM6.0 NEE | 0.125 | 0.199 |
| JULES-ES NEE | 0.251 | 0.308 |
| ORCHIDEE NEE | 0.159 | 0.200 |

DGVM NEE interannual variability in XCO2 is a factor 2-4 smaller than the
GPP-only signal -- Reco tracks GPP year to year and cancels most of it -- and
the three models disagree with each other by a further factor of two. Figure:
`plots/sif_gpp_iav/trendy_vs_sif_iav_hovmoller.png`; JULES-ES shows the largest
2023-24 (El Nino) excursion.

## 4n. Third method trap: isolating IAV of a drifting, strongly seasonal tracer

Found while making the TRENDY interannual Hovmollers, flagged by a sharp jump
at every 1 January. Measured Jan-1 day-step ratios (vs ordinary days) expose
both naive methods:

- **Per-day-of-year linear detrend** leaks interannual *curvature*: each year-
  boundary point is fitted as an endpoint on one side and an interior point on
  the other (Jan-1 steps 2.2-5.1x normal; worst for JULES-ES with its strong
  2023-24 El Nino swing). It also requires every calendar day to see the same
  set of years -- a window starting 2019-05-01 gives Jan-Apr days a different
  year set than May-Dec days and adds block edges at 1 January and 1 May.
- **Day-of-year mean, then a line in time** leaks *drift*: the climatology turns
  a ramp into a staircase and the residual is a sawtooth (Jan-1 steps ~100x for
  the NEE tracers, which drift -3 to -5 ppm/yr).

The correct tool is the **joint seasonal + trend decomposition**
(`x = a + b t + s(doy) + r`, s zero-mean, solved by backfitting; 5 iterations),
on complete calendar years only. Jan-1 step ratios drop to 1.7-2.5x, and the
remainder is real month-resolution structure in the monthly TRENDY forcing.
Clean-window (2020-2024) NEE interannual zonal rms: CLM 0.190, JULES-ES 0.324,
ORCHIDEE 0.213 ppm (supersedes the section 4m values computed on the
misaligned window).

**Reading an integrating tracer's IAV.** The interannual component of XCO2
appears as year-long *ramps* -- the tracer integrates the flux, so a persistent
seasonal-flux anomaly is a slope, not a block. For interpretation use the
**growth-rate anomaly** (time derivative of the interannual component, 61-day
mean, ppm/yr): `plots/sif_gpp_iav/trendy_nee_growthrate_hovmoller.png`.

## 4o. GPP comparison: SIF-derived vs TRENDY, in XCO2 and in flux space

XCO2 (GPP-only tracers, common window, detrended): seasonal patterns nearly
identical (r = 0.984-0.990 vs SIF); NH seasonal p2p SIF 34.9, CLM 44.4,
JULES-ES 34.1, ORCHIDEE 28.5 ppm -- SIF sits mid-ensemble. Interannual XCO2
fields correlate r = 0.88-0.93 with SIF.

Flux space is the stringent test (`gpp_flux_sif_vs_trendy_{absolute,iav}.png`,
band totals in Pg C/yr, joint decomposition, 31-day mean): anomaly amplitudes
agree well (rms 0.13-0.17 vs SIF 0.15 Pg C/yr per band) but pattern correlation
is only **r = 0.39-0.44**. The much higher XCO2-level correlation reflects what
transport does: it integrates fluxes in time and mixes them in space, so
low-frequency common signal dominates; flux-level disagreement in where/when is
largely smoothed away. Both statements are true and worth keeping side by side:
the atmosphere sees SIF-GPP and DGVM-GPP interannual variability as quite
similar (r ~ 0.9), while the underlying flux anomalies only agree at r ~ 0.4.

## 4p. Against OCO-2: observed XCO2 interannual variability

Observations: the OCO-dashboard daily 1-degree gridded OCO-2 XCO2
(`/kiwi-data/Data/satellite/OCO2/oco2_dashboard/oco2_daily.zarr`, count-weighted
to monthly sin-latitude band means, >=50 soundings per band-month, 91.6%
band-month coverage 2015-2024, polar-night gaps masked; note the `xco2_n`
fill value is negative and must be clipped). Same joint seasonal+trend
decomposition, common complete-year window 2020-2024, monthly resolution.
Figure: `plots/sif_gpp_iav/oco2_vs_tracers_iav_hovmoller.png`.

| tracer | r vs OCO-2 IAV | rms (ppm) |
|---|---|---|
| OCO-2 observed | 1 | 0.456 |
| TRENDY JULES-ES NEE | **+0.795** | 0.309 |
| TRENDY ORCHIDEE NEE | +0.732 | 0.203 |
| TRENDY CLM6.0 NEE | +0.517 | 0.167 |
| SIF-GPP net (flat Reco) | **-0.498** | 0.340 |

Two conclusions, one expected and one sharp:

1. **DGVM NEE tracers reproduce most of the observed zonal XCO2 interannual
   variability** despite carrying no fossil, ocean or fire fluxes -- JULES-ES
   reaches r = 0.80 and clearly tracks the observed sequence (positive 2020-21,
   the deep 2022-23 La Nina sink, the 2024-25 El Nino source spike), though all
   three are underdispersed (rms 0.17-0.31 vs 0.46 observed; fire, ocean and
   sampling noise are missing).
2. **The SIF-GPP net tracer is anti-correlated with the observations
   (r = -0.50).** GPP-only interannual variability has the *wrong sign* for net
   flux in this window: the climate anomalies that dominate observed growth-rate
   variability (La Nina cool/wet, El Nino hot/dry) move respiration and fire
   opposite to and more strongly than GPP alone, and in addition the SIF
   record's own upward trend places its strongest uptake exactly in the years
   when the real atmosphere saw the El Nino source spike. A GPP-only
   observation-driven flux, however good the GPP, cannot be used as a proxy for
   net-flux interannual variability.

## 4p. Against OCO-2: observed XCO2 interannual anomalies

The OCO-dashboard daily 1-degree gridded OCO-2 XCO2
(`/kiwi-data/Data/satellite/OCO2/oco2_dashboard/oco2_daily.zarr`, 2014-10 to
near-real-time) was binned to the same equal-area bands (sounding-count
weighted) and decomposed with a NaN-tolerant joint seasonal+trend backfit over
2020-2024 (no polar-night sampling; grey in the figure). Figure:
`plots/sif_gpp_iav/oco2_vs_tracers_iav_hovmoller.png`.

Pattern correlation of interannual anomalies with OCO-2 (valid cells,
31-day means):

| tracer | r vs OCO-2 | rms (obs 0.444 ppm) |
|---|---|---|
| TRENDY JULES-ES NEE | **+0.795** | 0.313 |
| TRENDY ORCHIDEE NEE | +0.742 | 0.205 |
| TRENDY CLM6.0 NEE | +0.507 | 0.170 |
| SIF-GPP net (flat Reco) | **-0.515** | 0.342 |

Findings:

1. **DGVM NEE reproduces observed XCO2 interannual structure remarkably well**
   -- JULES-ES reaches r = 0.80 with 70% of the observed amplitude, despite the
   observations also containing fossil, ocean and fire signals and the tracers
   containing none of them. The dominant observed pattern (positive 2020,
   deeply negative anomalies through the 2021-23 triple-dip La Nina, strongly
   positive from the 2023-24 El Nino onward) is present in all three DGVMs
   with model-dependent amplitude.
2. **The GPP-only SIF tracer is ANTI-correlated with observations (-0.52).**
   Its low-frequency component is roughly in antiphase: TROPOMI SIF has
   GPP strengthening into 2024-25 (uptake anomaly, negative XCO2) exactly when
   observed XCO2 anomalies turn strongly positive. Two readings, not mutually
   exclusive: (a) observed XCO2 IAV is set by respiration and fire, not by GPP
   -- consistent with the DGVM attribution (TER tracks GPP at r 0.84-0.89, and
   what survives into NEE is not GPP-shaped); (b) part of the late-record SIF
   rise is instrument drift (section 4c's unresolved gain-drift caveat), which
   would push the SIF tracer the wrong way exactly in 2024-25.
3. Model amplitudes are 2-3x weaker than observed (0.17-0.31 vs 0.44 ppm rms),
   as expected with fire, fossil-growth variations and ocean IAV absent.

## 4q. What fraction of XCO2 interannual variability is GPP vs Reco?

Computed exactly in XCO2 space by linearity (NEE' = GPP' + TER' tracer
anomalies; closure r = 0.999-1.000), 2020-2024, joint decomposition, 31-day
means. The dominant fact first: GPP-driven and Reco-driven XCO2 variability are
each 0.6-0.8 ppm rms and anti-correlated at **-0.92 to -0.98**, so NEE
variability is the ~25% residual of two large compensating signals.

Covariance attribution (share_i = cov(comp_i', NEE') / var(NEE'), sums to 1):

| | GPP share | Reco share |
|---|---|---|
| JULES-ES zonal | 97% | 3% |
| JULES-ES global mean | 47% | 52% |
| ORCHIDEE zonal | 71% | 29% |
| ORCHIDEE global mean | -21% | 121% |
| CLM6.0 | ill-conditioned (corr -0.98; shares +258/-157%) | |

JULES-ES: zonal structure GPP-shaped, global growth-rate anomaly an even split.
ORCHIDEE: zonally GPP-led but the global mean is Reco-driven with GPP damping.
CLM6.0: compensation too complete for a robust split. Caveat: as
corr(GPP',Reco') approaches -1 this decomposition becomes ill-conditioned, so
the JULES/ORCHIDEE numbers are robust splits, not precision values. Read
together with section 4p: observations anti-correlate with the GPP-only tracer,
i.e. reality's surviving residual looks more Reco/fire-shaped than the zonal
JULES/ORCHIDEE split suggests.

## 4r. Cross-instrument check: TROPOMI vs OCO-2 SIF anomalies

Motivated by the sign question of section 4p: is the SIF-GPP tracer
anti-correlated with observed XCO2 because TROPOMI SIF has the wrong sign
(instrument drift)? OCO-2 SIF (`sifdaily`, nadir+glint land, count-weighted)
from the OCO-dashboard zarr is fully independent -- different instrument,
calibration, overpass time and retrieval window. Monthly land anomalies on the
same equal-area bands, 2019-2025, TROPOMI on the harmonised OCO-2 scale.
Figure: `plots/sif_gpp_iav/sif_tropomi_vs_oco2_anomalies.png`.

- Seasonal fields agree at r = 0.925.
- Global-land anomaly series (trend retained) correlate at **r = +0.62**; both
  instruments show the same shape: low 2019, flat middle, strong rise from
  mid-2024 into 2025.
- **The late-record SIF rise is confirmed and is even STRONGER in OCO-2**:
  2019-2025 global-land trend +1.69 (TROPOMI harmonised) vs **+2.71**
  (OCO-2) mW m-2 sr-1 um-1 per year; 2025 annual anomaly +7.2 vs +15.7 (same
  units x1000).
- Detrended month-to-month anomalies agree only weakly (pattern r = 0.37,
  global r = 0.28) -- sub-annual SIF anomalies are dominated by sampling and
  retrieval noise in one or both records. A ~4 mW level divergence in 2021-2023
  (OCO-2 low, TROPOMI near zero) shows instrument-level uncertainty of that
  order remains.

**Conclusion -- SUPERSEDED same day, see below.** The paragraph above was based
on the *standard* B11 product.

**Correction with the tpk-fixed OCO-2 product (`oco2_daily_B11tpk_fix.zarr`).**
The throughput-peak degradation fix removes ~5 mW m-2 sr-1 um-1 per year of
spurious brightening -- larger than the signal -- and reverses the comparison:

| global-land SIF trend 2019-2025 (mW m-2 sr-1 um-1 /yr) | |
|---|---|
| TROPOMI (harmonised) | +1.69 |
| OCO-2 B11 standard | +2.71 |
| OCO-2 B11 tpk-fixed | **-2.25** |

Global anomaly correlation with TROPOMI flips from +0.62 (standard) to -0.30
(tpk-fixed); annual anomalies are high 2019-2021 and decline through 2023-24.
Detrended sub-annual patterns agree about equally either way (r ~ 0.4), so it
is specifically the multi-year trend whose SIGN is instrument-correction
dependent.

Three consequences:

1. The tpk-fixed OCO-2 SIF is *consistent* with observed XCO2: GPP high in
   2019-2021 (low XCO2 anomalies), dropping through 2023-24 (XCO2 surge) --
   the correct GPP-driven sign. TROPOMI runs the other way. The
   anti-correlation of the SIF-GPP tracer with observed XCO2 (section 4p) is
   therefore plausibly **substantially TROPOMI gain drift**, not purely
   Reco/fire physics. Section 4c's dormant-season test excluded an additive
   drift but explicitly not a multiplicative one; +1.5%/yr of gain on a ~0.2
   mean is exactly the required +2-3 mW/yr.
2. **The multi-year land-SIF trend is unresolved at the +/-3 mW/yr level**:
   instrument-correction uncertainty exceeds the signal. The trend-inclusive
   3.7 ppm cumulative XCO2 figure of section 4b should be treated as
   unreliable pending a TROPOMI degradation assessment.
3. The detrended ~0.5 ppm interannual figure stands: sub-annual anomalies
   agree across instruments and correction versions.

## 4s. GOSIF-GPP as a quasi-independent tie-breaker

GOSIF-GPP v2 (complete local archive 2000-2024, monthly 0.05-degree GeoTIFFs;
reader conventions from the pilot's `evaluate_gosif_gpp.py`: uint16, fill
>= 65534, scale 0.01/ndays) added to the GPP-anomaly comparison
(`sif_vs_jules_gpp_anomalies.png`, `sif_vs_jules_gpp_zonal_anomalies.png`).
Deseasonalised only, 2019-2024, totals per equal-area band:

| vs JULES-ES GPP | global r | zonal r |
|---|---|---|
| GOSIF-GPP v2 | +0.26 | **+0.43** |
| TROPOMI (harmonised) | +0.27 | +0.39 |
| OCO-2 standard B11 | +0.07 | +0.31 |
| OCO-2 tpk-fixed | -0.14 | +0.17 |

**GOSIF and TROPOMI agree with each other at r = +0.66 zonally** -- the
satellite products cluster, and both agree with each other better than either
does with the model. Because GOSIF's temporal variability is MODIS-EVI +
meteorology driven (OCO-2 SIF only anchors its static regression), it is
quasi-independent of the SIF detector-drift dispute of section 4r -- and its
global GPP rises 141.5 -> 149.0 Pg C/yr over 2019-2024 (+~1.3 Pg C/yr), the
same direction and magnitude as TROPOMI's +1.0. Two quasi-independent records
therefore support the greening trend against the tpk-fixed OCO-2 decline,
suggesting the tpk correction may overshoot. Caveats: GOSIF's absolute scale
traces to standard-B11 OCO-2 SIF; MODIS has its own degradation-correction
history; and all satellite lines reach only r ~ 0.3 with JULES at the
global-mean level, where sub-annual noise dominates.

## 4t. Common baseline for the OCO-2 pair: the correction is an early-mission ramp

Methodological fix: the two OCO-2 versions are the SAME soundings under
different degradation corrections, so their anomalies must be referenced to
ONE common climatology (here the mean of the two versions' climatologies).
Deseasonalising each against its own climatology silently re-centres each
version on itself, hiding the correction's level and end-anchoring and mildly
distorting correlations.

With the common baseline (figures `sif_vs_jules_gpp_anomalies.png`,
`sif_vs_jules_gpp_zonal_anomalies.png`, which also plot `tpk - standard`
directly): the tpk correction is a **backward-growing ramp anchored near zero
at the end of the record** -- +17 to +21 Pg C/yr-equivalent in 2019 decaying
monotonically at -2.9 Pg C/yr per year to ~0 by late 2024. The two versions
straddle the baseline at +/-6.7 Pg C/yr and **converge in 2024**, where all
five records (both OCO-2 versions, TROPOMI, GOSIF, JULES) cluster.

The trend dispute is therefore an **early-mission (2019-2021) level question**,
not a recent-surge question: tpk asserts the early mission read ~15-20% too
dim. Against that assertion, TROPOMI, GOSIF and JULES all show flat-to-rising
2019-2021 anomalies consistent with the standard version. Zonal r(JULES) under
the common baseline: standard +0.25 vs tpk +0.13 (TROPOMI +0.39, GOSIF +0.43).
The section 4r "trend unresolved" caveat accordingly narrows to: unresolved
only if the tpk early-mission correction is trusted against three
quasi-independent records that disagree with it.

## 4u. The 757 vs 771 nm fingerprint: both OCO-2 757 trends are instrumental

Real fluorescence preserves the physical spectral ratio SIF757/SIF771 ~ 1.5, so
a vegetation trend must appear in both bands with equal *relative* slopes; the
two microwindows sit in different detector regions, so drift generally does not
respect the ratio. Global-land, count-weighted, 2019-2025, joint
seasonal+trend fit:

| version | band | mean (W m-2 sr-1 um-1) | trend (%/yr) |
|---|---|---|---|
| standard | 757 | 0.2536 | **+2.93** |
| standard | 771 | 0.1662 | +0.63 |
| tpk-fixed | 757 | 0.3044 | **-2.27** |
| tpk-fixed | 771 | 0.1736 | -0.66 |

Verdict: **the bands diverge in both versions** -- the standard 757 trend is
4.6x its 771 counterpart, and the tpk fix drives 757 hard negative while
barely moving 771. Neither 757 record's trend is vegetation. Supporting
detail: the mean ratio is physical for the standard product (1.526 ~ 1.5) but
1.754 after the tpk fix, i.e. the correction pushes 757 above the physical
spectral ratio -- further evidence it overshoots.

**The 771 channel is the stable one**, bracketing zero (+0.63 / -0.66 %/yr
across versions). Convergent multi-record estimate of the true land-SIF/GPP
trend: modest positive, ~+0.5 to +1.3 %/yr (TROPOMI +1.26, GOSIF ~+0.9,
OCO-2 771 ~0 +/- 0.7), with standard-757 (+2.9) inflated and tpk-757 (-2.3)
overcorrected. Trend claims from OCO-2 SIF should use 771, not 757.

**GOSIF vs OCO-2 agreement** (GPP-equivalent anomalies, 2019-2024): vs
standard, zonal r = +0.57 (r^2 0.32), global deseasonalised r = +0.54 (0.29),
global detrended +0.38 (0.15). vs tpk the deseasonalised numbers are
trend-dominated (zonal +0.19, global -0.57), but detrended they recover to
zonal +0.55 -- identical to standard, as they must be (same soundings). The
satellite hierarchy on sub-trend structure: GOSIF-TROPOMI 0.66 > GOSIF-OCO-2
0.55 > any satellite vs JULES ~0.4.

## 4v. Inside JULES-ES: what drives its (XCO2-matching) NEE

Native monthly gpp/ra/rh, 2015-2024 (figure
`plots/sif_gpp_iav/jules_component_decomposition.png`).

**Ra is a fast, near-fixed fraction of GPP**: 47% in the mean (41% NH / 51%
tropics), corr(Ra,GPP) = 0.998 raw and 0.87-0.94 in monthly anomalies with best
lag 0; anomaly slope Ra' = 0.34-0.42 GPP'. Matches the documented formulation
(Clark et al. 2011 Part 2): growth respiration = (1-Yg)(GPP-Rm), Yg ~ 0.75,
plus N- and T-dependent maintenance. **Rh is lagged and damped**: seasonal peak
~1 month after GPP, seasonal p2p/mean 1.15 vs GPP's 1.87 in the NH, anomaly
corr with GPP only 0.58 (NH; 0.77 in the tropics via the shared moisture
driver). Formulation: 4-pool RothC with Q10-type T factor, wet/dry-optimum
moisture function and frozen-soil cutoff. (JULES source is MOSRS-licensed and
not local; formulation cited from the model description, not code.)

**2023-24 attribution** (NEE' = -GPP' + Ra' + Rh', Pg C/yr): global 2023
+0.34 (Ra+Rh +1.12 vs GPP-strengthening -0.78); global 2024 +1.09 (Rh' +1.04
the largest single term). Over the El Nino window Jul-2023..Jun-2024 the global
NEE' source of +1.75 decomposes into Rh' +1.23, Ra' +0.79, -GPP' -0.27 --
**respiration-driven globally** -- but the tropics-only budget is **entirely a
GPP collapse** (+1.18 from -GPP', respiration flat), while NH-2024 GPP gains
(+1.49, consistent with the TROPOMI/GOSIF surge) are fully cancelled by NH
Ra+Rh (+1.47). The JULES story: tropical drought-GPP decline plus extratropical
warm-respiration surge, extratropical GPP gains self-cancelling.

## 4w. Flux-space closure against observed growth rates (GRESO)

GRESO (spandey's OCO-2 growth-rate product,
`/home/spandey/OCO_growth_rates/greso_minimal_noaa_portable_260624/output_10sec_full_2025/`,
monthly + latitude-band + annual CSVs, 2015-01 to 2026-04, cross-validated
against NOAA MBL) overlaid on the per-model NEE' stacked-bar decomposition
(`trendy_nee_stackedbar_decomposition.png`; growth converted at 2.124 Pg C/ppm,
deseasonalised).

- **Monthly global NEE' variance is NPP-driven in all three models** (lumping
  Ra with GPP: shares 78% JULES / 85% CLM / 88% ORCHIDEE, Rh 13-23%), but the
  two big El Ninos differ in kind for JULES: 2015-16 is an NPP collapse,
  2023-24 is primarily an Rh surge (window attribution: Rh' +1.23 vs lumped
  NPP term +0.52 Pg C/yr).
- **corr(NEE', GRESO'): JULES +0.56 (peak +0.59 at 1-month obs lag), ORCHIDEE
  +0.49, CLM +0.36** -- the same ranking as the transported-XCO2 comparison
  (0.80/0.74/0.51), reproduced in pure flux space with an independent product.
  GRESO' rms 1.30 Pg C/yr vs model NEE' 1.63/1.01/0.86.
- GRESO endorses JULES' respiration-heavy reading of 2023-24: the observed
  double hump (late-2023, mid-2024) aligns with JULES' green (Rh) bars while
  CLM/ORCHIDEE undershoot.

## 4x. All 23 TRENDY models, with fire

`scripts/diagnostics/plot_trendy_allmodels_nee_decomposition.py` extends 4w to
the whole ensemble and adds the fire term. Figures
`plots/sif_gpp_iav/trendy_allmodels_summary_fire.png` (ranking + variance
shares) and `..._stackedbars_fire.png` (23 panels, sorted best-to-worst r,
legend once). Global monthly gpp/ra/rh come from
`scratch/trendy_all_models_global.json`, fFire from
`scratch/trendy_all_models_fire.json`; both deseasonalised by calendar-month
mean over the complete decade 2015-2024, no detrending.

- **fFire is available for 17 of 23 models** (missing: CABLE-POP, DLEM, IBIS,
  ISAM, OCN, TEM, iMAPLE; ORCHIDEE S3 ships fFire that is identically zero, so
  it is treated as fire-free). Two ship annual-resolution fire (LPJ-GUESS,
  LPJml); LPJwsl ships `units=''` but the magnitude (3.4 Pg C/yr) confirms
  kg m-2 s-1, so the strict unit gate is bypassed for fFire only. Mean fire
  spans 0.5 (ED) to 5.2 Pg C/yr (SDGVM) against GFED5's ~2.2.
- **Ranking with fire included** (corr NEE' vs GRESO', best of lag 0/1):
  JULES-ES +0.61, CLASSIC +0.58, ELM +0.58, VISIT-UT +0.55, DLEM +0.54,
  ISAM +0.53, IBIS/ORCHIDEE +0.51 ... SDGVM +0.19, LPJwsl +0.16,
  CARDAMOM +0.13, CLM-FATES +0.10, ELM-FATES -0.03. Adding each model's own
  fire term moves most models by <0.05 but promotes JULES-ES past CLASSIC
  (+0.59 -> +0.61) and lifts CLM (+0.36 -> +0.49) and ELM (+0.54 -> +0.58); it
  *hurts* ED (+0.36 -> +0.28) and CLASSIC (+0.61 -> +0.58).
- **The ensemble mean still beats every member: r = +0.64**, rms 0.84 vs
  GRESO' 1.30 Pg C/yr.
- **GFED5 fire alone correlates +0.56 with the observed growth-rate anomaly**
  (rms 0.63 Pg C/yr, i.e. half the observed anomaly amplitude) -- as skilful as
  all but the top model's *complete* NEE'. Substituting GFED5 for each model's
  own fire raises **every** model, to +0.36..+0.69, with JULES-ES at +0.69 the
  best combination found. Caveat: fire anomalies are drought-driven and so
  partly collinear with the GPP and Rh anomalies, so this is an upper bound on
  fire's independent contribution, not a clean attribution.
- **Fire variance share** is small for most models (0.5-10%) but dominates
  ELM-FATES (72%) and is large in CARDAMOM (24%), ELM (23%) and ED (22%) --
  the FATES pair's near-zero skill is a fire-driven NEE' that does not match
  observations.
- Models whose Rh carries no independent variance (LPX-Bern -23%, CARDAMOM
  -26%, LPJwsl -3%, i.e. Rh' anti-correlated with NEE') again populate the
  bottom of the ranking; the top of the ranking gives Rh a genuine 20-52%.

### Fire-swap test (`trendy_fire_swap_r2.png`)

Formal version of the swap: for each model, replace its own fFire' with GFED5'
in NEE' and rescore R^2 against GRESO', with the observation lag frozen at the
value the own-fire NEE' selected so all variants are scored identically.
Uncertainty from a 12-month moving-block bootstrap (4000 draws), which
preserves the near-annual autocorrelation.

- **R^2 rises for 23/23 models, 22 with a 90% CI clear of zero** (only CARDAMOM
  straddles). Median dR^2 **+0.121**. Restricting to the 16 models that
  actually carry a non-zero fFire: 16/16 improve. Best absolute result is
  JULES-ES + GFED5 at R^2 = 0.482 (own fire 0.367, no fire 0.351); largest gain
  is ED (+0.237, whose own fire is actively harmful: 0.130 -> 0.078).
- **The regression wants GFED5 at face value**: fitting GRESO' on
  [NEE'(no fire), GFED5'] gives a median GFED5 weight beta = 0.98 (range
  0.72-1.16). The observed fire flux enters at its physical magnitude, not as a
  rescaled index of opportunity.
- **GFED5 explains variance the models do not**: partial corr(GFED5', GRESO' |
  own NEE') = +0.40..+0.55 for every model.
- The coefficient on the fire-free NEE' does shrink when GFED5 joins (median
  0.47 -> 0.37), so part of the fire signal is already inside the models'
  drought-driven GPP/Rh response and the two terms are not independent. But the
  shrinkage is modest and beta stays at ~1, so this is not simple double
  counting.
- Reading: for global monthly growth-rate IAV, **observed fire is worth more
  than any TRENDY model's own fire parameterisation**, and roughly as much as
  the entire land-model NEE' of a mid-ranked model.

### Inter-model agreement per component

`trendy_intermodel_agreement.png` (correlation over covariance, shared
covariance scale) and the interactive `trendy_intermodel_agreement.html`
(Plotly, component + metric selectors, per-panel colour scale; also copied to
`~/www/trendy/`). Shared variance = var(ensemble mean) / mean member variance:
1.0 means identical members, 1/n means independent.

| component | n | mean pairwise r | shared var | member rms lo/med/hi | mean r at 12-mo |
|---|---|---|---|---|---|
| -(GPP-Ra)' | 23 | +0.53 | 0.51 | 0.67/1.26/2.17 | +0.69 |
| Rh' | 23 | +0.55 | 0.51 | 0.36/0.83/1.41 | +0.62 |
| NEE' | 23 | +0.35 | 0.38 | 0.68/1.25/2.12 | +0.50 |
| fFire' | 16 | +0.24 | 0.17 | 0.12/0.20/0.90 | +0.35 |

- **Rh agrees marginally better than NPP in correlation (+0.55 vs +0.53) and
  identically in shared variance (0.51 both)** -- i.e. the ensemble is no more
  coherent about photosynthesis-minus-autotrophic-respiration than about
  heterotrophic respiration, contrary to the usual assumption that Rh is the
  poorly constrained term. But Rh' amplitude is ~2/3 smaller (median rms 0.83
  vs 1.26 Pg C/yr), so in *covariance* the NPP term dominates: models agree
  equally well on Rh in shape while disagreeing far less about its size.
- **NEE' agreement (+0.35) is worse than either component**, because the
  disagreements are partly independent and partly compensating -- combining two
  terms that each correlate ~0.54 gives a worse-agreeing sum.
- **Fire is where the ensemble falls apart**: mean r +0.24, 22% of the 120
  pairs anti-correlated, shared variance 0.17 (near the 1/16 = 0.06 independent
  floor). Consistent with 4x's finding that observed fire beats modelled fire.
  ED and SDGVM are anti-correlated with nearly every other model's fire.
- Agreement improves at 12-month smoothing for every component (NPP +0.69,
  Rh +0.62, NEE +0.50, fire +0.35): the ensemble concurs on El Nino-scale
  swings and disagrees on month-to-month structure. Note the NPP term overtakes
  Rh at the annual timescale.

## 4y. C30 12-tracer run over the full OCO-2 era

`config/runs/trendy_v14_s3_gpp_ra_rh_c30_2014_2024.toml`, 2014-09-01 to
2024-12-31 (3775 daily files), 3 models x {GPP, Ra, Rh, NEE} on the spandey C30
L66 no-convection archive. 87 min wall on one L40S (transport 80%), about 20x
faster than the equivalent C90.

**Units trap.** `*_total_mass` is **not** kg of species. The prognostic state
stores dry VMR times dry-air mass (`_surface_flux_storage_scale` in
`src/Models/InitialConditionIO.jl` converts inbound kg species/s by
`M_air / M_species`), so `total_mass` is dry-air-equivalent kg. Convert with

    ppm  = total_mass / 5.135e18 * 1e6          (the pinned dry-air mass)
    kgCO2 = total_mass * M_CO2 / M_air,  M_air = 0.02896546, M_CO2 = 0.0440095

Comparing `total_mass` against a flux integral in kg CO2 without this factor
produces a spurious 34% deficit. The check that the conversion is right: the
implied GPP rates come out at 146.2 / 156.2 / 117.7 Pg C/yr against TRENDY's
native 147.7 / 157.4 / 118.8.

**Closure** (F64 `*_total_mass`, change since day 1, in global-mean ppm; the
tracer masses already carry the flux sign, so the identity is
`nee == gpp + ra + rh`, not `ra + rh - gpp`):

| model | GPP | Ra | Rh | sum | NEE | residual |
|---|---|---|---|---|---|---|
| CLM6.0 | -711.7 | +404.0 | +254.5 | -53.20 | -53.09 | +0.112 ppm |
| JULES-ES | -760.5 | +358.2 | +358.6 | -43.71 | -43.59 | +0.120 ppm |
| ORCHIDEE | -572.9 | +349.1 | +192.5 | -31.30 | -31.21 | +0.085 ppm |

Residual is 0.015-0.02% of the component spans over 10.3 years, the expected
F32 large-carrier penalty (cf. 0.03-0.06 ppm for the shorter C90 run).
Against the flux-file integral every tracer transports 0.99715 of what was
emitted (identical ratio for all 12 tracers and for fire, carrier-independent),
i.e. a **0.29% distributed deficit over 10.3 years**, not a missing-days
artifact: daily increments scatter 0.994-1.001 around it.

**Hovmoller** `trendy_c30_xco2_hovmoller.png`: NEE-tracer XCO2 anomaly in 40
equal-area bands with OCO-2 on one shared colour scale, real latitudes on the
axis, band-months with <2000 soundings greyed out (polar night sawtooth plus
the 2017 outage; 88% coverage survives).

- **R with OCO-2 over observed cells: ORCHIDEE +0.66, JULES-ES +0.57,
  CLM6.0 +0.32.** ORCHIDEE overtakes JULES here, unlike the flux-space GRESO
  ranking (4x) and the earlier C90 global comparison (JULES 0.80 / ORCHIDEE
  0.74 / CLM 0.51) -- JULES wins on global monthly timing, ORCHIDEE on the
  zonal pattern.
- Model anomaly rms 0.23-0.33 ppm against OCO-2's 0.41: land NEE alone explains
  most but not all of the observed spread (the rest is ocean, fire and fossil).
- Masking the sparse polar cells matters: it moved the correlations by +0.03 to
  +0.11 and cut the observed rms from 0.52 to 0.41 ppm.

**Zonal growth rates** `trendy_c30_zonal_growth.png`: monthly growth-rate
anomaly on GRESO's ten 10-degree bands, 3-month smoothed (raw monthly
differencing of the tracer is far noisier than GRESO, which is a smoothed
product).

- Pooled over all bands: ORCHIDEE +0.50, JULES-ES +0.49, CLM6.0 +0.31.
- **Skill is strongly latitude-dependent and decays northward**: at -15 deg
  CLM/JULES/ORCHIDEE reach +0.61/+0.65/+0.58, at +45 deg only
  +0.11/+0.17/+0.43. ORCHIDEE is the only model that keeps NH extratropical
  skill.
- Ceiling check: OCO-2's own differenced band XCO2 against GRESO gives
  +0.32..+0.82 (median ~0.75), so the models reach roughly two thirds of the
  achievable correlation in the tropics and much less in the NH.

## 4z. GFED5 fire as a superimposable tracer

The TRENDY drivers carry `Ra + Rh - GPP` and **no fire at all** (the flux files
contain only GPP/TER/RA/RH/NEE), so every transported NEE above is missing each
model's fire term. Rather than rebuild the TRENDY fluxes, fire is transported
once as a standalone tracer and superimposed, transport being linear.

- Driver `scripts/preprocessing/prepare_gfed5_fire_flux.py` (generic
  `--geometry`/`--tag`). Sources: GFED5.1 finalized `C` (g C/month/cell) to
  2022, GFED5NRT `EM` summed over the 16 `lct` classes after. `EM` reproduces
  the published monthly global total bitwise (2024-07: 0.55272904300404746
  Pg C both ways), so the products concatenate unmodified. Conservative remap
  with a per-month mass gate; daily by the same monthly-midpoint interpolation
  + exact renormalization as the TRENDY driver (error 1.6e-15). Every annual
  total matches the canonical GFED series to the printed precision.
- Run `config/runs/gfed5_fire_c30_2014_2024.toml`, identical met/grid/dates/
  physics to 4y. 70 min for one tracer (46.6 ms/it vs 58 for twelve -- met I/O
  dominates, so extra tracers are nearly free and fire should just ride along
  in the next run). Transported +16.75 ppm = 35.6 Pg C, 0.29% below the flux
  integral like every other tracer.
- Fire XCO2 anomaly rms is only 0.125 ppm, yet **adding it improves agreement
  with observed XCO2 for all three models**, by far more than that number
  suggests:

| | R (NEE) | R (NEE + GFED5 fire) | zonal growth vs GRESO |
|---|---|---|---|
| CLM6.0 | +0.32 | **+0.56** | +0.31 -> +0.50 |
| JULES-ES | +0.57 | **+0.70** | +0.49 -> +0.59 |
| ORCHIDEE | +0.66 | **+0.75** | +0.50 -> +0.62 |

- The gain is largest where the model was weakest (CLM +0.24) and the ranking
  is preserved. This is the 4x flux-space swap test confirmed in transported
  XCO2 space with an independent observation.
- Per band the improvement is everywhere but biggest in the NH extratropics,
  precisely where 4y found the models collapsing: at +45 deg CLM/JULES/ORCHIDEE
  go +0.11/+0.10/+0.28 -> +0.25/+0.30/+0.55 against a ceiling of +0.74.
- Figures `trendy_c30_xco2_hovmoller_fire.png`,
  `trendy_c30_zonal_growth_fire.png` (the fire-free versions keep their old
  names; `--no-fire` reproduces them).

## 4aa. Transport IAV vs flux IAV, settled with a control run

`config/runs/jules_nee_clim_control_c30_2014_2024.toml` transports JULES-ES NEE
replaced by its 2015-2024 mean seasonal cycle, repeated every year
(`scripts/preprocessing/prepare_climatological_nee_flux.py`; verified to repeat
at -9.285 Pg C/yr in every non-leap year), on exactly the met/grid/dates/physics
of 4y. Analysis `scripts/diagnostics/decompose_transport_vs_flux_iav.py`,
figure `transport_vs_flux_iav.png`.

**The decomposition is exact, not approximate.** Both tracers see the same
winds and transport is linear in the tracer, so

    R - C = T[f_real] - T[f_clim] = T[f_real - f_clim]

is precisely the XCO2 driven by flux IAV, and C is precisely what year-to-year
meteorology does to an unchanging flux. There is no cross-term to apportion.

**Global mean: transport IAV is ~1% and flux IAV is ~100%.**
Monthly global-mean anomaly rms: total 0.253, flux-driven 0.253,
transport-driven **0.002 ppm**. This is physically required -- winds
redistribute carbon, they do not create it -- and it doubles as the numerical
check: differencing two F32 tracers on 200 ppm carriers (memo 4n trap 1) has a
noise floor at or below 0.002 ppm here, three orders below the 0.25 ppm signal.

**Zonally it is a different story.** 40 equal-area bands, monthly,
detrended+deseasonalised, as fractions of total variance:

| | rms (ppm) | share of total variance |
|---|---|---|
| total (real NEE) | 0.331 | 100% |
| flux-driven | 0.279 | 71% |
| transport-driven | 0.148 | 20% |
| cross-term (corr +0.116) | | 9% |

- By zone, transport-driven rms as a fraction of total rms: **90S-30S 0.51,
  30S-30N 0.33, 30N-90N 0.55** (variance shares 26 / 11 / 30%). Transport IAV
  matters least where the flux signal is largest.
- At annual resolution the transport share stays substantial: transport/total
  rms 0.44 (19% of variance), i.e. it does not average away.
- Reading: for a *global* growth-rate or budget statement, meteorological IAV
  is negligible and the flux signal is the whole story. For *zonal or regional*
  XCO2 attribution, roughly a fifth to a third of the extratropical variance is
  wind-driven, and inversions that fix transport will alias it into fluxes.
- Caveat: the first ~6 months of 2015 in the transport panel are spin-up
  transient (both tracers start uniform on 2014-09-01).
- Not comparable to the earlier C90 SIF-GPP control number (transport
  0.14-0.38 of flux-driven): that ratio used the *GPP* tracer, whose flux
  signal is far larger, so its denominator differs.
- **Processing nuance (user caught this): R - C needs no deseasonalising and
  arguably no detrending.** The flux difference has zero calendar-month
  climatology by construction; the measured residual seasonal amplitude in
  R - C is 0.014 ppm against the 4.78 ppm cycle of the raw tracers. The
  backfit's detrend, meanwhile, removes *real* accumulated flux-IAV signal
  (there is no instrumental drift in a model difference). Standalone flux-IAV
  numbers, raw with only the time mean removed: **zonal rms 0.366, global-mean
  rms 0.343 ppm** (vs 0.279 / 0.253 after the backfit). The backfit remains
  necessary for the three-way table, because R and C individually carry the
  full seasonal cycle and trend; the 71/20/9 shares are conditional on that
  common processing. Daily vs monthly barely matters for this field: daily
  global-mean rms 0.338 vs monthly 0.343 (little sub-monthly variance in the
  difference, though local synoptic texture exists).

## 5. Reproducing

```bash
# 1. flux driver (about 20 s)
python3 scripts/preprocessing/build_sif_gpp_multiyear_c90_flux.py \
  --out $ATMOSTRANSPORT_DATA_ROOT/fluxes/sif_gpp_daily_co2flux_c90_2018_2025.nc

# 2. transport, format-4 snapshot required (about 3 h on one L40S)
julia --project=$MODEL_ROOT --threads=4 $MODEL_ROOT/scripts/run_transport.jl \
  config/runs/sif_gpp_iav_c90_2018_2025.toml

# 3. concatenate + diagnose, then plot
python3 scripts/diagnostics/analyze_sif_gpp_iav_xco2.py
python3 scripts/diagnostics/plot_sif_gpp_iav_xco2.py
```

This checkout's runtime requires binary `format_version=3` and the C90 archive
is `format_version=4`, so the run must use the frozen snapshot at
`.../runtime/AtmosTransportModel-format4-20260813`. `[input.staging]` is not
implemented there and is silently ignored; it is also unnecessary, since
kiwi-data delivers about 778 MB/s against the roughly 110 MB/s a C90 run
consumes.

## 6. Limitations

- **This is GPP only.** There is no respiration, no fire, no ocean, no fossil.
  The anomaly tracer is the XCO2 response to SIF-inferred *photosynthetic uptake*
  variability alone, not a net-flux prediction. Real NEE IAV is substantially
  smaller than GPP IAV because respiration partially compensates.
- **The 2019-to-2025 upward GPP trend (+8 Pg C) may be partly instrumental.**
  TROPOMI radiometric degradation and calibration drift alias into a SIF trend,
  and this product's fixed calibration passes any such drift straight into GPP.
  Treat the trend with far more caution than the year-to-year variability, and
  prefer detrended interpretation of the anomaly tracer.
- **No convection** in this met archive; vertical transport is advection plus
  TM5 DKG boundary-layer diffusion only. Column means are much less sensitive to
  this than vertical profiles would be.
- 1.13% of land cell-days are permanently missing SIF (`cell_status` 2 and 3)
  and are set to zero GPP. Being time-invariant, they contribute nothing to the
  anomaly.
- The SIF-to-GPP relation is linear per month and latitude band, fit against
  FLUXCOM-X, so the product inherits FLUXCOM-X's absolute scale and cannot be
  read as an independent validation of it.
- Float32 with a positive carrier: transport rounding scales with the carrier,
  which is why the anomaly tracer uses 100 ppm rather than 400 ppm.
