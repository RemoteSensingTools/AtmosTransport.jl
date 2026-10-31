# GEOS

Meteorological input readers and transport-binary workflows for GEOS.

## File map

- [`GEOS.jl`](GEOS.jl) — module imports and public reader/workflow API.
- [`configuration.jl`](configuration.jl) — GEOS-IT/FP TOML settings constructors.
- [`reader.jl`](reader.jl) — typed GEOSNativeReader and lifecycle extensions.
- [`geos.jl`](geos.jl) — settings, paths, daily NetCDF handles and orientation.
- [`geos_panels.jl`](geos_panels.jl) — panel slicing, coordinate-aware physics interpolation and endpoint dry mass.
- [`geos_read_window.jl`](geos_read_window.jl) — GEOS-IT/FP endpoints, fluxes and physics window readers.
- [`geos_cs_mass_helpers.jl`](geos_cs_mass_helpers.jl) — pressure-to-mass conversion, GEOS pressure-fixer evolution and smoothing.
- [`geos_cs_resolution.jl`](geos_cs_resolution.jl) — native passthrough and nested block coarsening strategies.
- [`geos_cs_omega.jl`](geos_cs_omega.jl) — OMEGA interpolation, regularization and reconstruction.
- [`geos_cs_window.jl`](geos_cs_window.jl) — GEOS window preparation, adaptation and ingest/drain.
- [`cubed_sphere_geos.jl`](cubed_sphere_geos.jl) — GEOS native-CS daily workflow.

## Depends on

Common source/target/factory protocols; Vertical transforms; Spectral synthesis
and mass support; MassClosure flux/balance primitives; PhysicsInputs conversions;
Targets/CubedSphere mass/pin/delta and window contracts; TransportBinary writers
and fidelity. Geometry, binary IO, conservative regridding and constants come
from the lower package modules. No other source or parent implementation.

## Used by

Preprocessing entrypoint and public facade; preprocessing CLI and focused tests.

## Shared helpers owned here

The exports in the module entry file are the source interface. New concrete
settings/readers extend the imported Common protocols; day workflows extend
TransportBinary.process_day. Private numerical helpers stay inside this source.

## Forward<->adjoint mirrors

No adjoint. Replay and positivity use the shared CS/binary contracts. Preserve
source level orientation, endpoint timing, conversions and arithmetic order.

GEOS dry endpoint arithmetic retains FT/SIMD summation and native dynamics-step flux scaling. OMEGA uses MassClosure divergence and solve scratch without changing solver policy.

ConfigChecks.env_flag preserves ATMOS_OMEGA_TIMING as the case-sensitive
strings "1", "true" and "yes".
