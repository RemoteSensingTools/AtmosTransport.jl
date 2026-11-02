# TM5-FLUX-2a: real-day spectral flux comparison

## Findings

Three full days at 1° and December O24 were compared over 24 windows each.
At 1°, horizontal flux relative RMS differences were 4.96–5.09% for am and 7.05–7.28% for bm.
Global cm differences were 49.97–50.53% at 1° and 39.09% on O24.
For December 1°, layer balance changed cm by 0.9233%; mean versus start sampling changed it by 25.85%, using separate reference denominators.
December 1° CO₂ flux effects in 24-hour increments were 0.3721/0.04408 ppm surface/column at K1 and 0.6331/0.04178 ppm at Calc.
Final cases passed production balance and replay gates; 1° binary storage and transport used Float32.
January/July Calc runs lacked complete surface inputs; these comparisons do not establish TM5 runtime parity.

2026-11-01. Execution source: `848271c0dcded7fd62e988c801e8999fcc6bc858`,
branch `tm5/flux2a-memo`. No `src/`, fence or git writes. All science outputs,
configs, logs and progress are retained in the script's output directory.
Execution used Julia 1.12.6 on CPU, with `TMPDIR` on a large disk.
Preprocessing/stages use 8 Julia / 8 OpenBLAS threads; transports use 8 / 4.
Preprocessing is serial.

Reproduction and metric definitions:
[`compare_flux_constructions.jl`](../../scripts/diagnostics/compare_flux_constructions.jl),
[diagnostics README](../../scripts/diagnostics/README.md). Retained commands:
`preprocess_queue.py`, `transport_queue.py`, `analysis_queue.py`,
`stages_queue.py`, `native137_full.py`, `pressure_queue.py`, and `configs/`
under the output root.
Case keys: `ll360` = December 1°; `20220115` / `20220715` = January / July 1°;
`o24` = December O24.

## Experiment

This memo compares flux constructions with each other; the comparison against the MERRA-2/GCHP runs is a separate experiment.

- R0: default point sampling, target truncation T179 at 1°, start-of-window
  rates. R1: point sampling retaining all source T639 modes through an
  oversampled FFT and gather. R2: production TM5 integrated PS/faces, dry
  endpoint mean and column balance. R3: full TM5, including moist per-layer
  balance against independent W. R4: balanced start endpoint instead of mean.
- R2−R1 includes quadrature, T+1 wind completion, pin order, moist-merge/dry-conversion order and time sampling;
  R3−R4 measures the sampling switch separately. Production functions are
  called directly; no methods or tolerances are overridden.
- 1° targets are 360×180, latitude centres −89.5°…89.5°. The common output
  vertical coordinate is `ml137_tropo34a`, 34 layers from native L137.
  Native L137 attribution uses all 24 December windows.
- The 2×2 control labelled K1 is the golden `run_spectral_grid.toml`
  diffusion: constant Kz = 1 m²/s. Calc is live `tm5_calc_kzz`, from
  `run_tm5pbl.toml`. Transport uses Upwind, implicit vertical diffusion,
  midpoint surface forcing, Rn decay and no convection, with replay enabled.
- Five golden tracers: uniform 400 ppm; CO₂ from CATRINE
  `startCO2_202112010000.nc` with static December CAMS/LMDZ `flux_apos` mean;
  zero-initialized fossil CO₂ with GridFED December 2021; zero-initialized
  Rn-222 with Zhang December and half-life 330350.4 s; Gaussian blob centred
  at (−60°,30°), amplitude 1e−6 mol/mol, widths 15° longitude / 10° latitude.
  These IC/source recipes are held fixed for the January and July weather.
- One-day runs save 0,3,…,24 h. Tracer differences are candidate minus
  reference on matching hybrid levels. Both absolute fields and changes from
  each run's own initial field are retained. Interaction is
  `(TM5flux,Calc − TM5flux,K1) − (defaultflux,Calc − defaultflux,K1)`.
- Cell statistics are area weighted. Face-rate RMS uses equal weights over
  unique faces; the periodic east mirror is excluded. Regions: global,
  |latitude|≤30°, 30°<|latitude|≤80°, north >80°, south <−80°. cm includes
  boundary interfaces. Relative RMS is RMS(difference)/RMS(reference).
- Stored flux amounts are divided by each window's own application seconds.
  Dry PS tendency is g Σdm/(area Δt). LL stores dm explicitly. For O24,
  windows 1–23 use adjacent stored masses; window 24 uses the encoded flux
  budget because its endpoint is not stored. The CSV labels each method.
  Moist tendency uses PS directly;
  the two writers' merged humidity has different mass weights. Saved stage
  next-day PS completes the last LL window; O24 uses 23 stored PS differences.
  Other binary metrics cover all 24 windows. Nominal pressures use ps=1000 hPa.

## Completed scope

| Day / grid | Full-day binaries | Transport | Attribution |
|---|---|---|---|
| 2021-12-01 / 1° | 24 windows | 2×2 | 24 L34 + 24 L137 windows |
| 2022-01-15 / 1° | 24 windows | constant-Kz flux pair; Calc inputs absent | 24 L34 windows |
| 2022-07-15 / 1° | 24 windows | constant-Kz flux pair; Calc inputs absent | 24 L34 windows |
| 2021-12-01 / O24 | 24 windows | 2×2 | — |

## Full-day binary metrics

All flux rates are kg/s; PS is Pa. Values are RMS(candidate−default)/RMS(default), in percent.

| Day / grid | am or hflux global | bm global | cm global | cm north cap | cm south cap | PS RMS difference, Pa | dry PS-tendency global |
|---|---:|---:|---:|---:|---:|---:|---:|
| ll360 | 5.019 | 7.276 | 50.31 | 98.94 | 93.69 | 471.9 | 5.66 |
| 20220115 | 4.959 | 7.05 | 50.53 | 95.25 | 81.53 | 476 | 5.441 |
| 20220715 | 5.092 | 7.259 | 49.97 | 88.72 | 89.54 | 458.6 | 6.538 |
| o24 | 8.751 | — | 39.09 | 98.43 | 97.61 | 872.8 | 11.19 |

## Mass and horizontal convergence

Relative RMS percent over all windows and levels. Dry-mass span is (maximum−minimum)/mean of the 24 stored global masses.

| Case | Mass global | Convergence global | Convergence north cap | Convergence south cap | Dry-mass span default / TM5 |
|---|---:|---:|---:|---:|---:|
| ll360 | 0.5551 | 57.42 | 96.35 | 96.79 | 1.687e-10 / 8.976e-11 |
| 20220115 | 0.5592 | 57.16 | 99.27 | 90.25 | 1.415e-10 / 7.694e-11 |
| 20220715 | 0.543 | 57.6 | 94.24 | 93.52 | 1.901e-10 / 1.048e-10 |
| o24 | 1.09 | 56.24 | 101.3 | 98.67 | 5.606e-15 / 5.406e-15 |

## Surface-pressure tendencies

Direct moist PS differences use 24 windows for 1° and 23 for O24. Each entry is RMS difference in Pa/s / relative RMS percent. Dry tendency uses the carrier-mass budget over all 24 windows.

| Case | Moist global | Moist north cap | Moist south cap | Dry global | Dry north cap | Dry south cap |
|---|---:|---:|---:|---:|---:|---:|
| ll360 | 0.0009975 / 5.768 | 0.0008173 / 6.059 | 0.0002534 / 4.184 | 0.0009945 / 5.66 | 0.0008171 / 6.046 | 0.0002533 / 4.187 |
| 20220115 | 0.00096 / 5.526 | 0.000644 / 6.285 | 0.0007109 / 4.533 | 0.000957 / 5.441 | 0.0006438 / 6.256 | 0.0007112 / 4.504 |
| 20220715 | 0.0009761 / 6.643 | 0.0008883 / 9.302 | 0.0007008 / 6.41 | 0.0009726 / 6.538 | 0.0008867 / 8.952 | 0.0007006 / 6.408 |
| o24 | 0.001909 / 11.41 | 0.002035 / 17.18 | 0.0009007 / 15.37 | 0.001904 / 11.19 | 0.002045 / 16.39 | 0.0008917 / 15.43 |

## ll360 global tracer factorial

Area-weighted RMS difference in each run’s change from its own initial field at 24 h. Each cell is surface / column. K1 = constant Kz 1 m²/s; Calc = live TM5 Calc_Kzz.

| Tracer / units | Flux at K1 | Flux at Calc | PBL at default flux | PBL at TM5 flux | Interaction |
|---|---:|---:|---:|---:|---:|
| co2 / ppm | 0.3721 / 0.04408 | 0.6331 / 0.04178 | 2.621 / 0.04334 | 2.456 / 0.04315 | 0.4446 / 0.01266 |
| fossil / ppb | 184.6 / 7.776 | 437.7 / 7.41 | 1660 / 9.397 | 1532 / 9.665 | 305 / 2.739 |
| rn222 / 10⁻²¹ mol/mol | 4.339 / 0.2729 | 9.928 / 0.2502 | 36.26 / 0.3386 | 33.66 / 0.3431 | 7.071 / 0.1296 |
| blob / ppb | 1.055 / 0.9852 | 1.1 / 0.987 | 4.118 / 0.1547 | 3.979 / 0.14 | 0.5006 / 0.03703 |
| uniform / ppm | 0.01053 / 0.002579 | 0.01015 / 0.002582 | 0.006481 / 0.0002845 | 0.006476 / 0.0002853 | 0.009152 / 0.0004007 |

CO₂ initial versus final flux contrasts (absolute fields, ppm):

| PBL | Initial surface | Final surface | Initial column | Final column |
|---|---:|---:|---:|---:|
| Flux at K1 | 0.01094 | 0.3715 | 0.009116 | 0.04548 |
| Flux at Calc | 0.01094 | 0.6325 | 0.009116 | 0.04332 |

## o24 global tracer factorial

Area-weighted RMS difference in each run’s change from its own initial field at 24 h. Each cell is surface / column. K1 = constant Kz 1 m²/s; Calc = live TM5 Calc_Kzz.

| Tracer / units | Flux at K1 | Flux at Calc | PBL at default flux | PBL at TM5 flux | Interaction |
|---|---:|---:|---:|---:|---:|
| co2 / ppm | 0.2546 / 0.06315 | 0.4103 / 0.06253 | 2.387 / 0.02208 | 2.348 / 0.02313 | 0.3082 / 0.007732 |
| fossil / ppb | 84.02 / 4.935 | 179.5 / 4.329 | 1346 / 3.785 | 1301 / 3.907 | 126.6 / 1.403 |
| rn222 / 10⁻²¹ mol/mol | 2.537 / 0.1885 | 7.185 / 0.1668 | 38.73 / 0.1765 | 38.67 / 0.178 | 6.082 / 0.06968 |
| blob / ppb | 1.817 / 0.8702 | 1.68 / 0.8707 | 2.771 / 0.03756 | 2.736 / 0.03811 | 0.5363 / 0.01287 |
| uniform / ppm | 2.393e-12 / 7.68e-13 | 2.339e-12 / 7.72e-13 | 1.199e-12 / 7.501e-14 | 1.215e-12 / 7.497e-14 | 1.714e-12 / 1.065e-13 |

CO₂ initial versus final flux contrasts (absolute fields, ppm):

| PBL | Initial surface | Final surface | Initial column | Final column |
|---|---:|---:|---:|---:|
| Flux at K1 | 0.01849 | 0.2537 | 0.01443 | 0.06521 |
| Flux at Calc | 0.01849 | 0.4094 | 0.01443 | 0.06468 |

## ll360 tracer regions and zonal means

RMS difference in the 24-h change; cap cells show surface / column. Zonal maximum is max absolute zonal-mean difference over all rings and levels.

| Tracer / units | PBL | Global surface / column | North cap surface / column | South cap surface / column | Zonal maximum |
|---|---|---:|---:|---:|---:|
| co2 / ppm | Flux at K1 | 0.3721 / 0.04408 | 0.3171 / 0.1679 | 0.01769 / 0.05032 | 2.009 |
| co2 / ppm | Flux at Calc | 0.6331 / 0.04178 | 0.4393 / 0.1697 | 0.01622 / 0.05046 | 2.036 |
| fossil / ppb | Flux at K1 | 184.6 / 7.776 | 0.2226 / 0.009862 | 2.301e-06 / 1.089e-07 | 114.7 |
| fossil / ppb | Flux at Calc | 437.7 / 7.41 | 0.5232 / 0.01252 | 2.518e-05 / 4.285e-07 | 204.9 |
| rn222 / 10⁻²¹ mol/mol | Flux at K1 | 4.339 / 0.2729 | 0.9117 / 0.02926 | 0.04261 / 0.003245 | 4.29 |
| rn222 / 10⁻²¹ mol/mol | Flux at Calc | 9.928 / 0.2502 | 2.981 / 0.04391 | 0.1235 / 0.003596 | 8.401 |
| blob / ppb | Flux at K1 | 1.055 / 0.9852 | 0.0317 / 0.09811 | 1.306e-20 / 1.196e-15 | 2.204 |
| blob / ppb | Flux at Calc | 1.1 / 0.987 | 0.04994 / 0.09649 | 1.027e-20 / 1.2e-15 | 2.196 |
| uniform / ppm | Flux at K1 | 0.01053 / 0.002579 | 0.008337 / 0.00173 | 0.007072 / 0.0019 | 0.003856 |
| uniform / ppm | Flux at Calc | 0.01015 / 0.002582 | 0.008878 / 0.001724 | 0.007425 / 0.001894 | 0.003856 |

## 20220115 tracer regions and zonal means

RMS difference in the 24-h change; cap cells show surface / column. Zonal maximum is max absolute zonal-mean difference over all rings and levels.

| Tracer / units | PBL | Global surface / column | North cap surface / column | South cap surface / column | Zonal maximum |
|---|---|---:|---:|---:|---:|
| co2 / ppm | Flux at K1 | 0.349 / 0.04121 | 0.2748 / 0.07742 | 0.01498 / 0.04077 | 1.533 |
| fossil / ppb | Flux at K1 | 181.2 / 8.209 | 1.105 / 0.1174 | 0.007678 / 0.0002569 | 123.9 |
| rn222 / 10⁻²¹ mol/mol | Flux at K1 | 4.294 / 0.2625 | 0.7228 / 0.02804 | 0.0343 / 0.002398 | 3.704 |
| blob / ppb | Flux at K1 | 1.758 / 1.414 | 0.0007511 / 8.262e-05 | 4.381e-21 / 1.704e-19 | 3.264 |
| uniform / ppm | Flux at K1 | 0.01538 / 0.003503 | 0.009432 / 0.002129 | 0.009061 / 0.002554 | 0.006521 |

## 20220715 tracer regions and zonal means

RMS difference in the 24-h change; cap cells show surface / column. Zonal maximum is max absolute zonal-mean difference over all rings and levels.

| Tracer / units | PBL | Global surface / column | North cap surface / column | South cap surface / column | Zonal maximum |
|---|---|---:|---:|---:|---:|
| co2 / ppm | Flux at K1 | 0.3411 / 0.04445 | 0.2672 / 0.0852 | 0.01262 / 0.05119 | 1.421 |
| fossil / ppb | Flux at K1 | 156.5 / 8.055 | 0.1413 / 0.008855 | 0.0003755 / 1.156e-05 | 148.7 |
| rn222 / 10⁻²¹ mol/mol | Flux at K1 | 3.793 / 0.2703 | 0.8706 / 0.0351 | 0.04027 / 0.002726 | 3.702 |
| blob / ppb | Flux at K1 | 0.8496 / 0.4977 | 0.0008588 / 0.001477 | 4.433e-19 / 3.544e-18 | 0.9236 |
| uniform / ppm | Flux at K1 | 0.007384 / 0.001744 | 0.005081 / 0.001048 | 0.003954 / 0.00108 | 0.004776 |

## o24 tracer regions and zonal means

RMS difference in the 24-h change; cap cells show surface / column. Zonal maximum is max absolute zonal-mean difference over all rings and levels.

| Tracer / units | PBL | Global surface / column | North cap surface / column | South cap surface / column | Zonal maximum |
|---|---|---:|---:|---:|---:|
| co2 / ppm | Flux at K1 | 0.2546 / 0.06315 | 0.9852 / 0.5251 | 0.03205 / 0.2189 | 1.319 |
| co2 / ppm | Flux at Calc | 0.4103 / 0.06253 | 1.093 / 0.5238 | 0.02578 / 0.2191 | 1.341 |
| fossil / ppb | Flux at K1 | 84.02 / 4.935 | 0.2861 / 0.01791 | 2.353e-05 / 1.373e-06 | 35.52 |
| fossil / ppb | Flux at Calc | 179.5 / 4.329 | 0.6646 / 0.01641 | 4.087e-05 / 1.42e-06 | 65.84 |
| rn222 / 10⁻²¹ mol/mol | Flux at K1 | 2.537 / 0.1885 | 0.1397 / 0.007154 | 0.05646 / 0.005609 | 1.047 |
| rn222 / 10⁻²¹ mol/mol | Flux at Calc | 7.185 / 0.1668 | 0.3974 / 0.007766 | 0.2095 / 0.005745 | 3.751 |
| blob / ppb | Flux at K1 | 1.817 / 0.8702 | 0.06764 / 0.09004 | 1.509e-15 / 1.545e-13 | 1.612 |
| blob / ppb | Flux at Calc | 1.68 / 0.8707 | 0.07216 / 0.08998 | 1.144e-15 / 1.545e-13 | 1.616 |
| uniform / ppm | Flux at K1 | 2.393e-12 / 7.68e-13 | 1.408e-12 / 3.373e-13 | 1.436e-12 / 4.324e-13 | 1.566e-12 |
| uniform / ppm | Flux at Calc | 2.339e-12 / 7.72e-13 | 1.563e-12 / 3.516e-13 | 1.568e-12 / 4.213e-13 | 1.566e-12 |

## Production-stage attribution

Relative RMS differences, percent. R1−R0 = retained-source-T point sampling; R2−R1 = integrated PS/faces and endpoint mean; R3−R2 = layer balance; R3−R4 = mean instead of start after layer balance. Each contrast has its own reference denominator; the RMS percentages are not additive.

| Sample | Contrast | am global | bm global | cm global | cm north cap | cm south cap |
|---|---|---:|---:|---:|---:|---:|
| dec34 | R1−R0 | 4.561 | 9.87 | 58.73 | 121.1 | 112.4 |
| dec34 | R2−R1 | 4.311 | 9.716 | 55.08 | 62.83 | 71.53 |
| dec34 | R3−R2 | 0.07896 | 0.1246 | 0.9233 | 1.786 | 2.288 |
| dec34 | R3−R4 | 3.515 | 6.134 | 25.85 | 13.49 | 12.33 |
| dec34 | R3−R0 | 5.019 | 7.276 | 50.31 | 98.94 | 93.69 |
| jan34 | R1−R0 | 4.456 | 9.973 | 59.03 | 112.8 | 93.72 |
| jan34 | R2−R1 | 4.358 | 9.794 | 56.11 | 71.41 | 62.9 |
| jan34 | R3−R2 | 0.07999 | 0.1662 | 1.044 | 1.735 | 2.1 |
| jan34 | R3−R4 | 3.518 | 5.861 | 25.97 | 14.98 | 19.49 |
| jan34 | R3−R0 | 4.959 | 7.05 | 50.53 | 95.25 | 81.53 |
| jul34 | R1−R0 | 4.78 | 10.29 | 60.56 | 103.3 | 107.6 |
| jul34 | R2−R1 | 4.361 | 10.8 | 58.87 | 61.53 | 61.67 |
| jul34 | R3−R2 | 0.07195 | 0.135 | 1.018 | 1.276 | 2.283 |
| jul34 | R3−R4 | 3.468 | 5.895 | 27.5 | 19.44 | 13.39 |
| jul34 | R3−R0 | 5.092 | 7.259 | 49.97 | 88.72 | 89.54 |
| dec137 | R1−R0 | 4.557 | 9.901 | 59.75 | 122.9 | 114.3 |
| dec137 | R2−R1 | 4.416 | 9.737 | 55.67 | 62.01 | 69.72 |
| dec137 | R3−R2 | 0.07496 | 0.1173 | 0.9131 | 1.637 | 2.03 |
| dec137 | R3−R4 | 3.652 | 6.292 | 26.08 | 13.79 | 11.99 |
| dec137 | R3−R0 | 5.119 | 7.434 | 51.53 | 99.6 | 93.88 |

## Uniform closure and conservative storage

| Grid | Run | Maximum saved relative uniform drift | Uniform storage fractional change |
|---|---|---:|---:|
| ll360 | default/K1 | 0.000199 | 1.773e-08 |
| ll360 | default/Calc | 0.0001988 | 1.382e-08 |
| ll360 | TM5/K1 | 0.000189 | 2.961e-08 |
| ll360 | TM5/Calc | 0.000189 | 2.951e-08 |
| 20220115 | default/K1 | 0.0003758 | -8.649e-08 |
| 20220115 | TM5/K1 | 0.0003755 | 7.968e-08 |
| 20220715 | default/K1 | 0.0002621 | 1.314e-08 |
| 20220715 | TM5/K1 | 0.0002083 | -3.018e-08 |
| o24 | default/K1 | 4.039e-14 | 0 |
| o24 | default/Calc | 4.161e-14 | 0 |
| o24 | TM5/K1 | 3.429e-14 | 0 |
| o24 | TM5/Calc | 3.429e-14 | 0 |

Conservative storage follows NetCDF `mixing_ratio_times_carrier_air_mass` in kg of air equivalent; the quoted fractions are invariant to a constant species-mass conversion.


## Representative level differences

All entries are relative RMS percent over 24 windows. Pressures are nominal at ps=1000 hPa; cm is on the selected layer’s lower interface. Full per-level/per-region values are in `comparisons/<case>/levels.csv`.

| Case | Layer | Mid / lower-interface hPa | am or hflux global | bm global | cm global | cm north cap | cm south cap |
|---|---:|---:|---:|---:|---:|---:|---:|
| ll360 | 9 | 96.28 / 106.4 | 2.4 | 5.968 | 47.35 | 72.5 | 84.68 |
| ll360 | 23 | 493.8 / 513 | 5.405 | 7.517 | 50.09 | 99.04 | 92.8 |
| ll360 | 29 | 858 / 883.4 | 8.167 | 9.606 | 58.69 | 106.8 | 93.47 |
| ll360 | 32 | 970.2 / 982.1 | 10.13 | 10.7 | 66.33 | 130.6 | 109.6 |
| 20220115 | 9 | 96.28 / 106.4 | 2.471 | 6 | 46.74 | 78 | 88.58 |
| 20220115 | 23 | 493.8 / 513 | 5.269 | 7.426 | 51.05 | 87.34 | 80.87 |
| 20220115 | 29 | 858 / 883.4 | 8.67 | 8.824 | 57.13 | 117.2 | 83.61 |
| 20220115 | 32 | 970.2 / 982.1 | 9.807 | 10.08 | 65.12 | 150 | 104.2 |
| 20220715 | 9 | 96.28 / 106.4 | 2.292 | 6.742 | 45.61 | 88.09 | 75.27 |
| 20220715 | 23 | 493.8 / 513 | 5.538 | 7.436 | 49.29 | 83.61 | 87.93 |
| 20220715 | 29 | 858 / 883.4 | 8.62 | 9.5 | 58.89 | 106.2 | 89.98 |
| 20220715 | 32 | 970.2 / 982.1 | 9.651 | 10.5 | 67.42 | 123.1 | 108.1 |
| o24 | 9 | 96.28 / 106.4 | 4.642 | — | 50.56 | 97.88 | 99.06 |
| o24 | 23 | 493.8 / 513 | 9.343 | — | 39.06 | 97.86 | 97.19 |
| o24 | 29 | 858 / 883.4 | 13.59 | — | 47.24 | 100.8 | 97.06 |
| o24 | 32 | 970.2 / 982.1 | 15.74 | — | 49.57 | 109.4 | 83.05 |

## Layer-balance correction by level and region

R3−R2 relative RMS percent; 24 windows. These are the changes in the final dry rates and diagnosed cm caused by enabling the production moist per-layer balance. Full native/merged profiles are in `stages/<case>/levels.csv` under `layer_balance/`.

| Sample | Mid / lower-interface hPa | am global | bm global | cm global | cm north cap | cm south cap |
|---|---:|---:|---:|---:|---:|---:|
| dec34 | 493.8 / 513 | 0.0482 | 0.07369 | 0.8832 | 1.572 | 2.292 |
| dec34 | 970.2 / 982.1 | 0.1159 | 0.1675 | 1.11 | 1.106 | 1.024 |
| jan34 | 493.8 / 513 | 0.05013 | 0.08983 | 0.9989 | 1.732 | 1.965 |
| jan34 | 970.2 / 982.1 | 0.1289 | 0.1896 | 1.119 | 1.234 | 0.8328 |
| jul34 | 493.8 / 513 | 0.04331 | 0.07817 | 0.9832 | 1.145 | 2.031 |
| jul34 | 970.2 / 982.1 | 0.1075 | 0.1709 | 1.078 | 1.157 | 0.9916 |
| dec137 | 503.3 / 513 | 0.05175 | 0.07917 | 0.8832 | 1.572 | 2.292 |
| dec137 | 975.9 / 978.1 | 0.1141 | 0.1713 | 1.086 | 1.103 | 1.012 |

## Balance, replay and resource measurements

| Preprocessing case | Max logged replay relative error | Positivity ratio | Steps/window min–max | Wall | Peak RSS GiB |
|---|---:|---:|---:|---:|---:|
| pre_ll360_default | 3.116e-06 | 0.950 | 107–150 | 5:30.01 | 11.19 |
| pre_ll360_tm5 | 3.235e-06 | 0.950 | 108–154 | 16:26.17 | 17.31 |
| pre_20220115_default | 5.159e-06 | 0.950 | 180–276 | 2:38.48 | 8.801 |
| pre_20220115_tm5 | 6.082e-06 | 0.950 | 180–271 | 13:17.81 | 16.6 |
| pre_20220715_default | 5.268e-06 | 0.950 | 90–154 | 2:34.17 | 8.744 |
| pre_20220715_tm5 | 4.671e-06 | 0.950 | 80–133 | 12:59.20 | 16.69 |
| pre_o24_default | 1.504e-14 | 0.235 | 4–4 | 3:15.13 | 4.663 |
| pre_o24_tm5 | 1.03e-14 | 0.251 | 4–4 | 15:10.47 | 15.38 |

| Attribution sample | Endpoint checks | Max pre rms / max | Max post rms / max | Pass counts | Max layer residual kg/s | Max column post residual kg/s |
|---|---:|---:|---:|---|---:|---:|
| dec34 | 48 | 2.564e-07 / 4.582e-06 | 7.666e-18 / 1.379e-17 | 1 | 2.168e-06 | 0.0006192 |
| jan34 | 48 | 3.598e-07 / 5.998e-06 | 5.369e-18 / 9.78e-18 | 1 | 1.58e-06 | 0.0004969 |
| jul34 | 48 | 2.881e-07 / 6.678e-06 | 7.247e-18 / 1.296e-17 | 1 | 1.52e-06 | 0.0009316 |
| dec137 | 48 | 2.564e-07 / 4.582e-06 | 7.665e-18 / 1.379e-17 | 1 | 5.867e-07 | 0.0006483 |

The pre/post guard is the production pressure-tendency mismatch divided by mean PS, in s⁻¹. Layer and column residuals are in kg/s. Timings below cover the whole attribution ladder, including point synthesis; they are not isolated rung timings.

| Attribution | Wall | Peak RSS GiB |
|---|---:|---:|
| dec34 | 16:23.10 | 16.29 |
| jan34 | 16:33.12 | 16.01 |
| jul34 | 16:45.18 | 16.27 |
| dec137 | 20:10.85 | 18.63 |

## Local inputs and retained failures

December 1/2 spectral, thermo and surface inputs originate in the local
ERA5 0.5° golden-input archive, as in the golden configs.
January 15/16 and July 15/16 core files come from the local ERA5 N320
hourly native model-level core archive.
The native adapter copies packed VO/D/LNSP messages unchanged and exports
production conservative N320 humidity remaps onto the common 1° grid.
Next-day files contain the required 00 UTC endpoint. Source paths, byte counts
and message counts are recorded in `inputs/n320/native_*.toml`.

All four January/July surface GRIBs contain 24 each of
sp/z/lsm/blh/zust/10u/10v/2t/2d/sshf/slhf. They lack fsr/ewss/nsss. Those dates have
flux-pair transports at K1; live-Calc runs have no complete local input group.
No missing PBL primitives were substituted. Counts: `surface_inventory.txt`.

December T/q were exported at the common 360×180 points with production
coordinate normalization and bilinear interpolation. Of 213,062,400
native-level/hour q samples, ten were negative (two at 18 UTC, eight at
19 UTC; minimum −1.36138601192215e−6 kg/kg). They were floored at zero in the
shared Float32 input export, used by both constructions. Per-hour counts,
extrema and adjustments: `inputs/dec/thermo/quality_20211201.csv`.
Original inputs are unchanged.

| Initial trial | Production outcome |
|---|---|
| 1° default, Float64, dt=300 s | Window 13 replay 1.0595267765565328e−12 > 1e−12 at (219,1,32); residual 0.042758941650390625 kg |
| 1° default, Float64, golden dt=900 s | Same window/cell; 1.0676558962624549e−12 > 1e−12, residual 0.043087005615234375 kg |
| 1° default, Float32, original thermo | Replay passed (3.116e−6); PBL writer rejected negative humidity: `tm5_pbl.q must be in [0,1)` |

No production tolerance was changed. 1° binary storage/runtime is Float32;
O24 and the rate-level attribution stages use Float64. Uniform-tracer rows
quantify the saved numerical closure baseline. CO₂ initial-field differences
are reported separately from differences in the one-day change.

Initial `free -g`: total 502 GiB, available 472 GiB; scratch free 272 GiB.
GPU probe ran only under an exclusive site GPU lock with `CUDA_VISIBLE_DEVICES=1`:
`CUDA.functional()` false and `CUDA_ERROR_NO_DEVICE`. All subsequent runs
use CPU. Timings include compilation and concurrent CPU jobs.


## Artifacts and validation

- Per-window budgets and schedules, all-window level/region RMSs and daily
  zonal sections: `comparisons/{ll360,20220115,20220715,o24}/`.
- Five factorial effects (or the flux-pair contrast) at every snapshot, 24-h level/zonal sections
  (including `/increment`) and conservative storage: `tracers/<case>/`.
- Production rung metrics, both endpoint guards, column corrections and
  next-day pressure endpoints: `stages/{dec34,jan34,jul34,dec137}/`.
- Full numerical tables: `RESULTS.md`, `RESULTS.json`, `MEMO_TABLES.md`;
  execution and final status: `PROGRESS.md` and `*_queue_status.json`.
- `FINAL_AUDIT.json`: eight final preprocessing cases, twelve one-day
  transports, four full-day ladders, 192 endpoint guard checks, matching
  snapshot/effect coverage and pressure-tendency coverage passed.
- Weighted regional/zonal probe passed. Existing Float64 LL72 golden binary
  self-comparison: all 55 field/region rows zero. Existing five-tracer golden
  factorial self-comparison: all 2250 rows zero, including initial and
  increment contrasts; endpoint 24 h. These goldens test the script and are
  not substituted for the new real-day results.
- Across 9 global am/bm/cm metrics, the maximum absolute discrepancy
  between the Float32 binary relative RMS and the Float64 R3−R0 stage
  relative RMS is 1.6612e-09.
- Keyword-call and structure checks passed; structure reports existing debt.
  Direct docs invocation lacked a resolved Documenter installation. A copied
  cached Julia docs environment passed doctests, exported-doc checks and
  cross-references, then npm install failed DNS. Copied cached Node modules
  rendered the full VitePress site in 33.76 s with dead-link checking enabled;
  home layout, logo and the manual pointer assertions passed. Logs:
  `docs_build_isolated.log`, `docs_vitepress_cached.log`.

This is a flux-construction comparison within AtmosTransport. It does not
establish TM5 runtime parity: CFL scheduling, advection slopes and splitting
remain those of AtmosTransport. Selected spectral/thermo inputs contain no
omega variable; no omega-regression result is reported.
