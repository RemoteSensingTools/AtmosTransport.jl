# Preprocessing.Core

Configuration parsing, logging, met-source and reader contracts, target
geometry, and shared binary-pipeline interfaces. Core loads before Vertical
and Spectral and imports no later preprocessing stage. The parent registers
native GRIB geometry, source traits and column-balance implementations on its
generics. Target scratch has a concrete type parameter, preserving the same
scratch allocation while keeping the closure implementation out of Core.

## File map

- `Core.jl`: module imports, include order and shared interface.
- `binary_pipeline.jl`: staging paths, humidity clamp, column-weight and target
  balance protocols; workflow includes live in `../Preprocessing.jl`.
- `met_sources.jl`: settings types, full 14-argument `RawWindow`, source contracts.
- `met_readers.jl`: GEOS reader, seed policy and support/threading traits.
- `target_geometry.jl`: LL, RG and CS target geometry and metadata.
- `configuration.jl`: runtime settings and common flux/column-weight parsers.
- `logging.jl`: public `FlushingLogger` for preprocessing scripts.

## Depends on

Parameters, Grids, MetDrivers capability queries, ConfigChecks, AtomicFiles,
Regridding's cache path and the package data-path resolver; Dates, Logging,
Printf and FastGaussQuadrature.

## Used by

Vertical and Spectral; the parent registers source/target support methods,
source readers, closure weights and transport-binary workflows. Scripts use
these names through `Preprocessing`.

## Shared helpers owned here

The exports in `Core.jl` are the supported stage interface: source and reader
contracts, geometry, configuration parsers, `FlushingLogger`, `package_path`,
`open_preprocessing_staging`, and `MAX_SPECIFIC_HUMIDITY`. Native flux parsing
retains validation after keyword overrides; spectral parsing validates its
column-weight choice immediately.

## Forward<->adjoint mirrors

None.
