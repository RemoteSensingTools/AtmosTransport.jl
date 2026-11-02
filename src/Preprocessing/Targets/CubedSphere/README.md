# Preprocessing.CubedSphereTarget

Cubed-sphere spectral preprocessing, conservative LL-binary regridding, nested coarsening and contracts.

## File map

- [`CubedSphere.jl`](CubedSphere.jl) — module imports, exports and load order.
- [`cs_transport_helpers.jl`](cs_transport_helpers.jl) — CS scratch and conservative LL-to-panel regridding.
- [`cubed_sphere_coarsen.jl`](cubed_sphere_coarsen.jl) — experimental nested operator restriction.
- [`cubed_sphere_contracts.jl`](cubed_sphere_contracts.jl) — CS endpoint targets, replay and window contracts.
- [`cubed_sphere_regrid.jl`](cubed_sphere_regrid.jl) — LL-binary-to-CS streaming regrid.
- [`cubed_sphere_spectral.jl`](cubed_sphere_spectral.jl) — CS daily spectral workflow via LL staging.

- `mass_helpers.jl`: CS endpoint dry mass, surface pressure, global dry-mass pins.

## Depends on

Common, Spectral, MassClosure and TransportBinary; LatLonTarget spectral staging/logging; lower geometry, conservative regridding, binary IO and replay APIs. No parent or source implementation.

## Used by

CS spectral and LL-binary workflows; GEOS, ERA5 N320 and MERRA-2 day drivers use the shared contracts directly. Scripts and tests keep the facade names.

## Shared helpers owned here

`CubedSpherePreprocessWorkspace`, conservative panel regridding, `copy_cs_panels!`, CS replay/positivity wrappers and `CubedSphereContract`; endpoint/delta helpers; `regrid_ll_binary_to_cs`, `regrid_transport_binary`, experimental `coarsen_nested_cs_transport_binary` and spectral `process_day`.

Shared source helpers: `derive_c180_dry_mass!`, `ps_from_air_mass!`,
`cs_total_air_mass`, `pin_cs_global_air_mass!`,
`pin_cs_global_air_mass_from_sensitivity!`, `fill_cs_mass_delta_payload!`.
The in-place delta adapter delegates to the same out-of-place-capable loop.

## Forward<->adjoint mirrors

No adjoint. Window replay mirrors MetDrivers continuity. The shared palindrome scan lives in TransportBinary. Preserve mesh precision, face weights, balance order and endpoint-target arithmetic.

## Experimental coarsening

Nested coarsening restricts an existing operator with block sums; it still requires direct-target scientific validation. It retains its source schedule, replay and positivity gates.

ERA5 spectral `include_tm5_pbl=true` writes v5 with the point-sampled spectral
identity. It uses LatLonTarget's shared T/q sampler and moist layer merging,
panel-major cell centres, conservative surface means and a day-local input
cache. LL binary regridding passes the group through, with source moist-mass
weights for T/q and area means for surface fields. These means accumulate in
Float64, snap land fractions within 1e-12 to [0,1], then convert to the output
precision. The pass-through records a regridded
construction identity. N320→CS support is deferred pending archive coverage.

`fill_dry_mass_flux_thickness!` converts native-level pinned dry mass to `g*m/A`
for CS panel arrays or source-grid arrays. ERA5 and MERRA-2 share this rule.

## Direct TM5 spectral target

- `tm5_spectral_day.jl`: typed `TM5SpectralFluxes` day driver, refined-ring
  geometry setup, canonical-face/panel adapters, per-layer moist balance
  with independent W, adjacent-cell dry conversion, endpoint averaging,
  dry column repair, explicit-dm closure and streaming v5 write.
- The `PointSampledSpectralFluxes` dispatch retains the LL staging workflow.
  Direct TM5 never recovers winds from LL transport fluxes. Only PS and the
  independent spectral forcing use conservative refined-box overlap.
- The direct route requires next-day spectral/humidity endpoints and retained
  native interfaces. It uses the existing CS `m/am/bm/cm/ps/dm` sections and
  optional `tm5_pbl` group. `include_surface` is rejected; live Calc_Kzz inputs
  use `include_tm5_pbl`. Humidity is used internally for the dry carrier.
- Ordinary provenance keys record `great_circle_midpoint_bilinear_v1`,
  samples per face, refined shape, synthesis backend and actual precision;
  the v5 identity schema is unchanged. The writer also records the shared
  column-closure certificate for every window. The quadrature generalizes TM5's LL
  rule; the layer equation is TM5's equation on our CS graph, solved by a
  reused `DirectCSPoisson` factor. The serial run cache retains that factor,
  synthesis plans and geometry across days; its key includes geometry,
  truncation, refinement, backend, precision and batch size.
- `close_tm5_cs_window` restarts dry amount construction from pristine rates
  on every positivity retry and applies the shared replay/positivity gates.
  Two moist endpoints and one output window are retained. Files are promoted
  only after all windows and summary gates pass; failures abandon staging.

Focused tests: `test_tm5_cs_spectral_integrals.jl`, `test_tm5_cs_balance.jl`,
`test_tm5_cs_spectral_day.jl`. Golden definitions: `pre_c24_tm5`,
`pre_c24_tm5_tm5pbl`, `run_c24_tm5_cpu_f64` (references recorded separately).
