# CubedSphere

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

Core, Spectral, MassClosure and TransportBinary; LatLon spectral staging/logging; lower geometry, conservative regridding, binary IO and replay APIs. No parent or source implementation.

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
