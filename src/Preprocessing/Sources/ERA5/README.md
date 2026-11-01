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

With the global dry-mass pin enabled, dry flux thickness is `g*m/A` from
native-level mass with the stored endpoint pin applied. Each sliding-window pipeline retains its native CS mass. N320
line integrals use the same endpoint pressure offset on source-grid mass,
including next-day lookahead; they do not independently pin the source total.
Merged endpoint pin arithmetic and the moist thickness option are unchanged.

When pinning is disabled, source thickness keeps its original Float64
pressure/humidity calculation and the CS thickness keeps its original
`(1-q) Δp` calculation; neither is rebuilt through `g*m/A`, preserving both
Float32 and Float64 rounding bitwise.

A one-day ERA5 N320 → C90 L117 Float32 comparison (2022-01-15, line integrals)
measured `am`/`bm`/`cm` relative RMS changes of 0.016–0.018 % (maximum relative
change about 0.02 %). Stored `m`, `ps`, `dm`, and all included physics payloads
were bitwise identical. Flux-fidelity relative RMS improved from 2.5e-7 to
1.0e-7, and maximum relative error from 2.0e-4 to 6.8e-5. Closure and replay
residuals stayed at the same numerical scale, and global mass residuals were
unchanged. Existing ERA5 N320 dry-mass binaries built with the pin retain the
old, unpinned-thickness fluxes until regenerated, as do products that copy
their fluxes; both require regeneration to use the corrected thickness.
