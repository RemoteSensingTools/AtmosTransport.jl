# Observation sampling output (2026-10-02)

Branch `feature/observation-sampling` (from `main` 1a97ff50, v0.4.0).

## Purpose

Write model CO2 (or any tracer) profiles directly at satellite soundings
(OCO-2/3 Lite files) and NOAA ObsPack surface stations during a driven run,
instead of writing full hourly 3-D fields and sampling them offline.

## Decisions

- **Sampling instants.** Met-window ends only (t = 0 on the initial state,
  then after every window). In binary-scheduled runs convection and chemistry
  are applied once at window end, so mid-window states are incomplete.
- **Time interpolation.** Default `linear`: blend the two bracketing
  window-end states. Air and tracer *masses* are blended, then mixing ratios
  and pressures are formed, so the written profile is self-consistent.
  `nearest_window` is the alternative.
- **Horizontal.** Containing model cell, no interpolation. Lat-lon and
  reduced-Gaussian lookups are closed form on the face arrays; the cubed
  sphere uses the analytic inverse `lonlat_to_panel_xy`.
- **Vertical.** Interface pressures come from the state's layer air mass,
  `p_half[1] = A_ifc[1]`, `p_half[k+1] = p_half[k] + g·m[k]/area`; on a dry
  basis these are dry partial pressures and the variables are suffixed
  `_dry`. Station intake layers use hypsometric heights above ground with the
  binary's GCHP VDIFF layer temperature when present, else a 280 K constant
  (the diffusion helpers' 260 K is a mid-troposphere value). On GEOS L72 the
  surface layer is ~124 m, so 10–100 m intakes land in layer 72 and 500 m in
  layer 69.
- **Edge cases are counted, never silent.** Soundings before t = 0 or after
  the last window end, outside a regional mesh, or written one-sided after a
  change of window length are counted in file attributes and logged.
- **Concurrency.** netcdf-c is not thread-safe; every runtime NetCDF write
  (observation appends, single-file snapshot appends, the background daily
  snapshot task) takes `_NETCDF_IO_LOCK`.
- **Naming.** The reader entry point is `read_observation_requests` because
  `Adjoints` already exports `read_observations`; both modules are brought
  into `AtmosTransport` with `using`, so a second exported name would be
  ambiguous there.
- **Point events vs. station series.** Everything sampled once at its own
  time (satellite soundings, ObsPack records including aircraft, station time
  lists) is a point event in the `_soundings` file, with an intake-layer value.
  Stations sampled repeatedly are series in the `_sites` file. A station row
  picks its schedule by keys: none → `EveryWindow`, `start_time`/`end_time` →
  `TimeRange`, `times` → `TimeList` (expanded into point events). This covers
  the OCO-2 v11 MIP in-situ protocol (every ObsPack record, aircraft at
  altitude) and hourly TCCON-site profiles with one mechanism.
- **Quality filters are types.** `QualityFlagFilter(variable, max)` covers
  Lite `xco2_quality_flag` (0 = good). The MIP `assimilate_flag` is
  categorical (0 = not assimilated, 1 = assimilated, 2 = withheld; 742,558 /
  2,628,961 / 141,984 records in the OCO-2 file), so `<= max` cannot select
  the assimilated set; `QualityFlagValues(variable, values)` does.
  `NoQualityFilter` keeps every record, as the MIP co-sampling requires.
- **Times are never guessed.** NetCDF table time columns are decoded from
  their CF units; a bare number in CSV/TOML is an error.
- **Non-blocking writes.** Observation rows are queued and flushed when the
  shared NetCDF lock is free; retired daily files close when the lock is free.
  Measured before this change: the daily snapshot write held the lock for
  ~7 s at each C90 day boundary while the run waited.
- **Editor schema is checked.** Every config choice is a `oneOf` entry whose
  description names its Julia type; `test_observation_schema.jl` fails when
  the parser's choice tables and the schema disagree.

## Known limits

- All sources are read at startup, cut to the run span before request
  records are built. The whole-mission MIP file takes ~6 s and ~2 GB of
  transient memory to read.
- Two chained runs that share a boundary both emit an event exactly on it
  (the window end is inclusive).
- The transported span is estimated from the first binary's window count
  times the number of binaries.
- Dry surface pressure of the 2021 ERA5 C90 binaries equals OCO-2 retrieved
  total surface pressure over ocean (ratio 1.000; expected ~0.996): the
  binaries' global mean dry pressure (98,733 Pa) is the 5.135e18 kg dry-mass
  pin converted with R = 6.371e6 m and g = 9.80665. Mixing ratios are
  unaffected; map profiles to retrieval levels in normalised pressure (p/ps).

## Not in scope

Averaging-kernel application (offline, from `p_half_dry`, the profile, and
the Lite file's pressure levels), sub-step sampling, horizontal interpolation.

## Review process

Every phase was reviewed by two independent Claude agents (Julia style;
bugs and pitfalls) before commit. Codex review was suspended (no credits).
