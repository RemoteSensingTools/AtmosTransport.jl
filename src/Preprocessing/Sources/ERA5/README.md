# ERA5

Meteorological input readers and transport-binary workflows for ERA5.

## File map

- [`ERA5.jl`](ERA5.jl) — module imports and public reader/workflow API.
- [`configuration.jl`](configuration.jl) — ERA5-N320 TOML constructor.
- [`era5_grib_geometry.jl`](era5_grib_geometry.jl) — native reduced-Gaussian GRIB geometry, extending Common protocols.
- [`era5.jl`](era5.jl) — settings, stream paths and GRIB day handles.
- [`era5_n320_window.jl`](era5_n320_window.jl) — N320 grid discovery, spectral synthesis, humidity and ARCO pressure.
- [`era5_n320_mass_convection.jl`](era5_n320_mass_convection.jl) — native dry mass and forecast convection reader.
- [`era5_n320_to_cs.jl`](era5_n320_to_cs.jl) — coverage-normalized CS regrid, humidity remap, TM5 conversion and window pipeline.
- [`era5_n320_regrid.jl`](era5_n320_regrid.jl) — N320-to-CS daily writer and native vertical-merge/pin policies.

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

The surface reader is initialized outside the daily `try` and closed in `finally`, including failed builds. Coverage division, Float64 dry mass and native merged pin sensitivity retain their existing policies.

GRIB levels are canonical top-down; the former single-value `level_orientation` setting is removed. State merges retain extensive mass and moist-dp-weighted thermodynamics; no merged dry-DELP or endpoint accumulator is stored.

ConfigChecks.env_flag preserves ERA5_N320_PROFILE as the exact string "1".
