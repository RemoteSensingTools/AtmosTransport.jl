# Preprocessing.Spectral

ERA5 spectral input, LL and batched reduced-Gaussian synthesis, humidity and
native mass support. Spectral files use one `era5_spectral_paths` constructor.
FFTW synthesis keeps the unnormalised inverse transform and all existing
floating-point order and conversions.

## File map

- `Spectral.jl`: module imports and shared interface.
- `spectral_io.jl`: GRIB coefficient decoding and atomic daily cache.
- `discovery.jl`: available dates and next-day hour-zero endpoint.
- `spectral_synthesis.jl`: Legendre recurrence, wind synthesis, LL mass/fluxes.
- `reduced_spectral_synthesis.jl`: batched reduced-Gaussian synthesis for N320.
- `mass_support.jl`: extensive merge adapter, pressure overlap, LL Poisson
  balance, thermo QV readers, drying and global mass pins.

## Depends on

Core geometry, humidity policy and column-correction protocols; Vertical's
extensive reduction; Grids, Parameters, GriddedIO, AtomicFiles; Dates, Logging,
Printf, FFTW, GRIB/ecCodes, SHA, JLD2, NCDatasets and LinearAlgebra.
Closure implementations register Core methods in the parent; Spectral does
not import the parent or another later stage.

## Used by

The root's RG helpers, topology workspaces, source readers and day workflows.

## Shared helpers owned here

The exports in `Spectral.jl` declare synthesis, spectral paths/cache,
pressure-overlap remap, dry-basis conversion, humidity readers, global mass
pinning, LL balance and mass-target support. Private implementation helpers
stay within this module.

`era5_thermo_path` formats daily thermodynamic filenames for binary and target workflows.

## Forward<->adjoint mirrors

None.
