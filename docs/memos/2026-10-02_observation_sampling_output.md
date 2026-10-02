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
  `Adjoints` already exports `read_observations` and both are re-exported at
  top level.

## Not in scope

Averaging-kernel application (offline, from `p_half_dry`, the profile, and
the Lite file's pressure levels), sub-step sampling, horizontal interpolation.

## Review process

Every phase was reviewed by two independent Claude agents (Julia style;
bugs and pitfalls) before commit. Codex review was suspended (no credits).
