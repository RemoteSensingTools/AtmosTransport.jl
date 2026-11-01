# MERRA2

Meteorological input readers and transport-binary workflows for MERRA2.

## File map

- [`MERRA2.jl`](MERRA2.jl) — module imports and public reader/workflow API.
- [`configuration.jl`](configuration.jl) — MERRA-2 TOML constructor.
- [`merra2.jl`](merra2.jl) — NASA/GEOS-Chem archive paths, day handles, orientation and native window readers.
- [`merra2_latlon_regrid.jl`](merra2_latlon_regrid.jl) — lat-lon-to-CS regrid, physics pipeline and daily writer.

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

Endpoint blocks keep dry mass and stored pressure. The daily scratch owns the Float64 pressure accumulator; unused dry-DELP output is removed. OMEGA storage remains live when the input provides OMEGA.

Dry flux thickness uses the shared Targets/CubedSphere
`fill_dry_mass_flux_thickness!` rule (`g*m/A` from pinned mass); its arithmetic
and the historical moist default are unchanged.
