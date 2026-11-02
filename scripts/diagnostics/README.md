# Flux-construction comparison

`compare_flux_constructions.jl` compares existing binaries and runtime NetCDF
outputs and calls production stages for attribution. Run from the repository
root with `TMPDIR` on a large disk. Full-day preprocessing uses the canonical
`scripts/preprocessing/preprocess_transport_binary.jl`; transports use
`scripts/run_transport.jl`. Start from `test/golden/configs/pre_ll72_tm5pbl.toml`,
`pre_ll72_tm5_tm5pbl.toml`, their O24 counterparts, and
`run_spectral_grid.toml` / `run_tm5pbl.toml`. The golden default diffusion
control is constant Kz = 1 m²/s.

```bash
export TMPDIR=./scratch/tmp
export CUDA_VISIBLE_DEVICES=
export OPENBLAS_NUM_THREADS=8
export ATMOSTR_REGRID_CACHE_DIR=./scratch/regrid_cache
mkdir -p "$TMPDIR" "$ATMOSTR_REGRID_CACHE_DIR"
julia --project=. scripts/diagnostics/compare_flux_constructions.jl binary default.bin tm5.bin comparison
julia --project=. scripts/diagnostics/compare_flux_constructions.jl factorial default_constant.nc default_calc.nc tm5_constant.nc tm5_calc.nc tracers
julia --project=. scripts/diagnostics/compare_flux_constructions.jl pair default_constant.nc tm5_constant.nc tracer_pair
julia --project=. -t8 scripts/diagnostics/compare_flux_constructions.jl stages pre_tm5.toml 2021-12-01 stages
julia --project=. scripts/diagnostics/compare_flux_constructions.jl binary default.bin tm5.bin comparison stages/ps_end.nc
```

`stages` is currently lat-lon only. An optional final argument selects sorted,
unique window starts, e.g. `0,6,12,18`; otherwise all 24 hourly windows are
processed. It measures retained-spectrum point sampling (R1−R0), integrated
faces/PS with mean endpoint sampling (R2−R1), layer balance (R3−R2), and mean
versus start sampling (R3−R4). R0/R1 use production `vod2uv!`, staggering,
pinning, drying and merging. R1 retains all source modes by synthesizing an
oversampled ring and gathering at the target points. R2/R3/R4 use production
`construct_tm5_endpoint`; each alternative goes through the same production
column repair and explicit-dm cm diagnosis. No production methods are replaced.
The R2−R1 contrast includes the time sampling switch; R3−R4 measures sampling
separately. `balance.csv` records both endpoint guards and layer residuals.
The integrated-endpoint contrast also includes moist merging before dry
conversion, whereas the default point route dries native layers before merging.

Native N320 core GRIBs can supply the spectral files and humidity required by
the existing spectral preprocessor:

```bash
julia --project=. -t8 scripts/diagnostics/compare_flux_constructions.jl native N320_ROOT 2022-01-15 prepared
julia --project=. -t8 scripts/diagnostics/compare_flux_constructions.jl native N320_ROOT 2022-01-16 prepared 1
```

This copies the packed VO/D/LNSP messages unchanged into `prepared/spectral/`
and exports the production conservative N320 humidity remap to
`prepared/thermo/era5_thermo_ml_YYYYMMDD.nc` on 360×180 cell centres. The optional
record count `1` prepares only next-day 00 UTC. The export has **q only**;
live Calc_Kzz additionally requires real T and all its surface primitives.
No missing PBL field is synthesized. Use a fresh spectral output destination;
the adapter rejects existing spectral files.

The `thermo SOURCE_DIR DAY OUT [RECORDS]` command exports real T/q from the
existing thermo NetCDF schema at the common 360×180 target points, using
production bilinear interpolation. It records negative q counts, extrema and
adjustments in `quality_YYYYMMDD.csv`, floors negative q at zero, and writes
Float32 T/q. This was required by the December real-day experiment's PBL
humidity validation; the memo records the measured input adjustments. Both
constructions must use the same prepared input.

Binary outputs contain all-window `levels.csv`, `summary.csv`, `zonal.csv`,
`windows.csv` and nominal level pressures at ps = 1000 hPa. Flux amounts are
converted to kg/s with each binary's own per-window schedule and flux kind.
`ps_dry_tendency` is g Σdm/(area Δt), in Pa/s. PS itself is the stored moist
surface pressure. `ps_tendency` uses adjacent stored PS fields (23 windows).
LL reads explicit `dm` without requiring a flux-delta group. O24's layouts
omit `dm`: its first 23 mass changes use adjacent stored masses, and its last
uses the encoded horizontal/vertical flux budget. The latter is a tendency
diagnostic, not an independent replay check. `windows.csv` labels the method.
If the optional final `PS_END.nc` argument is supplied, its production-stage
next-day 00 UTC pressures complete window 24. The `stages` command writes
`ps_end.nc` when it includes window 23; comparison rounds the endpoint to each
binary's storage precision. Moist PS tendency is measured directly:
the two writers use different humidity merge weights, so moist endpoint
pressure is not reconstructed from merged humidity.
Column quantities use level 1 as a CSV placeholder. Cell statistics use cell-area weights; unique faces use
equal weights (the periodic east mirror is excluded). Regions are global,
|latitude| ≤ 30°, 30° < |latitude| ≤ 80°, north > 80°, and south < −80°.
cm includes both boundary interfaces. `relative_rms` is RMS(candidate−ref)
divided by RMS(ref); an exactly zero reference gives NaN.

Factorial outputs contain surface/column statistics at every saved snapshot,
final per-level and zonal profiles, tracer burdens and uniform-tracer drift.
The tracer commands expect the five golden tracer names and the uniform
reference 4e−4 mol/mol. Budgets retain NetCDF's conservative storage contract
`mixing_ratio_times_carrier_air_mass`, in kg of air equivalent; conversion to
species mass requires its molar-mass ratio.
The interaction is `(TM5flux,TM5pbl − TM5flux,defaultpbl) −
(defaultflux,TM5pbl − defaultflux,defaultpbl)`. VMRs are the native dry
mol/mol values. Initial differences and differences in each run's change from
its own initial field are reported separately; final level/zonal fields with
`/increment` contain the latter. The uniform drift covers saved snapshots
and levels.

Check `free -g` before T639 work; preprocess one case at a time. GPU commands
must run under the site lock, e.g.
`flock "${GPU_LOCK:?Set GPU_LOCK to the site lock file}" env CUDA_VISIBLE_DEVICES=1 ...`.
The [real-day memo](../../docs/memos/2026-11-01_tm5_flux_real_day.md) records
the measured scope, inputs, precision, failures and results.


## TM5 default quadrature error

`tm5_default_quadrature.jl` isolates raw face integration from spectral wind
transform changes, mass pins, drying, merging and balancing. CPU commands:

```bash
export TMPDIR=/temp2/cfranken/tmp OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1
julia --startup-file=no --project=test scripts/diagnostics/tm5_default_quadrature.jl harmonics OUT
julia --startup-file=no --project=test scripts/diagnostics/tm5_default_quadrature.jl day SPECTRAL_DIR 2021-12-01 OUT
```

The harmonic CSV compares composite trapezoids with exact cosine integrals and
`x*cot(x)-1` at four along-face frequencies and nstep=2/3/5/8, at T255/T639.
The day CSV covers every unique 1° LL face at levels 1/70/137 for hourly
analyses 00–23; the second CSV covers all 137 native levels at 00 UTC.
Rates use the IFS radius on both sides. Statistics are equal-face RMS in
kg/s, relative L2, and caps (absolute face
latitude >85°). Meridian face latitudes are interval centres; zonal faces use
their boundary latitude and include zero polar rates. Reference integration
uses 16-node Gauss latitude and Fourier primitives of longitude products
sampled at 5760 points per circle. The hour-00 reference check doubles both
resolutions. Both sides use identical complete wind coefficients, retaining
the integration-only contrast. Meridian default endpoints use TM5's
361-longitude polar mean, so polar-cell differences also include that policy.
Additional columns restrict all faces to absolute latitude <89° and caps to
85° < absolute latitude <89°, separating errors away from the polar interval.
The probe targets about one hour of single-thread CPU time; completed CSV
rows are flushed incrementally. A completed full day is not implied if the
budget stops the probe early. These are raw analysis face rates, rather than
balanced window fluxes or a full transport-day comparison.

The `native-day` command uses the same reference and measures every native
level at every hour, writing `ll_day_all_levels.csv`. Run `day` once for the
doubled-resolution convergence check; `native-day` reuses those quadrature
rules. Its 2600-second budget is checked after each complete hour; count CSV
hours before claiming a complete 24-analysis day. For example:

```bash
julia --startup-file=no --project=test scripts/diagnostics/tm5_default_quadrature.jl native-day SPECTRAL_DIR 2021-12-01 OUT
```
