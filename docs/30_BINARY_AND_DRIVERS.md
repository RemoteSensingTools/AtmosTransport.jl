# Binary And Driver Contract

This contributor note summarizes the timing boundary between preprocessing and
runtime. The rendered user reference is [Binary format](src/concepts/binary_format.md).

## Why semantics are explicit

Meteorological products may provide endpoint values, interval means, or
interval-integrated transport. Those quantities are not interchangeable. The
version-4 header therefore records both source provenance and the normalized
payload meaning; kernels never infer timing from filenames or source family.

The reader rejects older format versions and missing/unknown contract fields.
There is no compatibility fallback for an ambiguous transport product.

## Required header semantics

Every current binary declares:

- `source_flux_sampling`
- `air_mass_sampling`
- `flux_sampling`
- `flux_kind`
- `delta_semantics`
- `humidity_sampling`
- `poisson_balance_target_scale` and its semantics
- `steps_per_window` and `steps_per_window_by_window`
- `time_step_schedule`
- `poisson_balance_target_scale_by_window`

`air_mass_sampling` is currently `window_start_endpoint`. Lat-lon and reduced
Gaussian drivers accept normalized substep mass amounts and support
window-start, window-mean, or window-constant fluxes according to whether
endpoint deltas are present. Cubed-sphere forcing requires window-constant
flux sampling and accepts substep or full-window mass amounts; the simulation
normalizes full-window storage before operator application.

When deltas are present, `delta_semantics` must be
`forward_window_endpoint_difference`. Humidity is either absent or supplied as
the pair `qv_start` / `qv_end` with `window_endpoints` semantics.

## Stored schedule

The header owns the numerical advection schedule. Variable schedules carry one
positive step count and one Poisson target scale per window. Cubed-sphere
products with `runtime_substep_contract = "binary_schedule"` run transport at
that cadence and run convection/chemistry once at the meteorological-window
boundary.

Runtime may assert the stored schedule's CFL sufficiency, but it must not
silently replace a writer-verified schedule with a new one. A failed product
should be regenerated.

## Provenance versus runtime meaning

`source_flux_sampling` describes what the raw source supplied. The other
sampling fields describe what preprocessing wrote. For example, a source
interval integral can be normalized into per-substep mass amounts with
window-constant runtime sampling. Keeping both prevents source conventions
from leaking into the operator kernels.

## Driver responsibility

Drivers must:

- validate the version, geometry, basis, schedule, section shapes, and timing
  semantics before stepping;
- copy/convert required mmap sections into typed host windows;
- build topology-appropriate flux and forcing containers;
- interpolate only when the header and delta payload explicitly permit it;
- adapt or copy windows to the selected backend; and
- refuse a physics recipe whose required payload sections are absent.

Drivers must not diagnose missing vertical closure, guess wet/dry conversion,
or reinterpret unsupported source timing. Those transformations belong in
preprocessing.

## Dry-air output

When a moist-basis transport product supplies humidity endpoints, end-of-window
dry-VMR diagnostics must use the corresponding end humidity. Reusing
`qv_start` with the end air mass creates a sampling mismatch.

## Evidence

- Contract construction and validation:
  `src/MetDrivers/transport_binary/contract.jl`
- Typed header and accepted values:
  `src/MetDrivers/transport_binary/header.jl`
- Topology-specific runtime validation:
  `src/MetDrivers/transport_binary/driver.jl`
- Window/backend lifecycle:
  `src/Models/driven_window_state.jl`
