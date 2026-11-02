# TM5 default quadrature measurements, 2026-11-02

CPU, Float64, one Julia/BLAS thread per process. ERA5 input is T639,
2021-12-01, hourly analyses 00–23, native levels 1–137. These are raw
face rates before mass pins, balancing, drying, merging or window averaging.
Both alternatives use identical complete spectral wind coefficients and
surface pressure, isolating integration from wind-transform differences.

- `trapezoid_modes.csv`: direct composite trapezoids versus exact retained
  cosines and the analytic response x*cot(x)-1. All 32 cases agree to
  roundoff. nstep=5 reduces rather than eliminates the default error.
- `era5_day_levels.csv`: all-level daily profile, pooling squared RMS values
  over 24 analyses, rather than averaging relative errors. Every row covers
  all unique 1-degree meridian or parallel faces at its native level.

Default: eight-subdivision TM5 IntLat/IntLon, including the 361-longitude
meridian polar endpoint mean. Reference: 16-node Gauss latitude integration,
Fourier integration of longitude products sampled at 5760 points per circle.
At hour 00, levels 1/70/137, doubling both resolutions changed meridian and
parallel relative L2 by 8.2316e-13 and 3.9572e-16 respectively. The independent
review case at 80 E, 30–31 N, level 137, hour 01 reproduces
-501352.10886739806 versus -492967.10889356985 kg/s.

Rates use the IFS radius (6371229 m) and standard gravity on both sides,
as in the independent review. RMS is unweighted over faces and hours, in
kg/s. Relative L2 is
100*rms(default-reference)/rms(reference). Meridian face latitude is the
interval centre; parallel face latitude is its boundary. Parallel statistics
include zero polar rates. Caps use absolute latitude >85; nonpolar uses
absolute latitude <89; innercap uses 85 < absolute latitude <89. The outer
polar degree includes a different endpoint policy, so its discrepancy must
not all be called sinusoidal trapezoid error. The profile does not measure
balanced W or transport differences.

Reproduce from the repository root (see scripts/diagnostics/README.md):

```bash
export TMPDIR=/temp2/cfranken/tmp OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1
julia --startup-file=no --project=test scripts/diagnostics/tm5_default_quadrature.jl harmonics OUT
julia --startup-file=no --project=test scripts/diagnostics/tm5_default_quadrature.jl day SPECTRAL_DIR 2021-12-01 OUT
julia --startup-file=no --project=test scripts/diagnostics/tm5_default_quadrature.jl native-day SPECTRAL_DIR 2021-12-01 OUT
```

The raw native-day CSV has one row per hour/level/component. To form a daily
profile, take the square root of the mean squared hourly RMS for delta and
reference, then divide those pooled RMS values for relative L2. The same
procedure applies to cap and restricted-latitude columns. Raw CSVs, logs,
CPU timings, storage/bitwise probes and progress are retained locally under
`/temp2/cfranken/codex_scratch/flux2b/followup/`. No golden outputs were
recorded or changed.
