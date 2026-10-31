# Deferred source day drivers

These files remain in the Preprocessing facade for PP4. Shared binary code is
in [TransportBinary](../TransportBinary/README.md); target workflows/contracts
are in [LatLon](../Targets/LatLon/README.md),
[ReducedGaussian](../Targets/ReducedGaussian/README.md) and
[CubedSphere](../Targets/CubedSphere/README.md).

## File map

- [`cubed_sphere_geos.jl`](cubed_sphere_geos.jl) — native GEOS-IT/FP day driver: passthrough or nested block coarsening to a CS binary.
- [`era5_n320_regrid.jl`](era5_n320_regrid.jl) — ERA5 N320 GRIB to CS daily writer.
- [`geos_cs_mass_helpers.jl`](geos_cs_mass_helpers.jl) — DELP ↔ air mass, surface pressure, pressure-fixer mass evolution and residual smoothing.
- [`geos_cs_omega.jl`](geos_cs_omega.jl) — OMEGA-consistent vertical-flux target (time interpolation, regularization, reconstruction).
- [`geos_cs_resolution.jl`](geos_cs_resolution.jl) — global dry-mass pin, panel-convention check and native → target resolution strategies.
- [`geos_cs_window.jl`](geos_cs_window.jl) — GEOS window workspace, per-window preparation, substep selection, ingest/drain/advance.
- [`merra2_latlon_regrid.jl`](merra2_latlon_regrid.jl) — MERRA-2 wind-derived to CS daily writer.

## Depends on

Core/Vertical/Spectral/MassClosure/PhysicsInputs, target contracts, TransportBinary,
and the deferred source readers. PP4 will move this source-specific orchestration.

## Used by

The Preprocessing entrypoint, source integration tests and preprocessing CLI.

## Shared helpers owned here

Source day drivers; GEOS omega, native-resolution and mass helpers. Their
cross-stage duplicates and numerical-policy differences are recorded in PP3.md.

## Forward<->adjoint mirrors

No adjoint. GEOS/N320/MERRA-2 share CS target replay and binary writer contracts.
