# Preprocessing.Common

Configuration parsing, logging, met-source and reader contracts, target
geometry, and shared binary-pipeline interfaces. Common loads before Vertical
and Spectral and imports no later preprocessing stage. Sources register native GRIB geometry and source traits on its generics;
MassClosure registers the column-balance implementation. Target scratch has a concrete type parameter, preserving the same
scratch allocation while keeping the closure implementation out of Common.

The module remains in `Core/Core.jl`; the folder and entry filename are kept
to avoid moving shared implementation files for a module-only rename.

## File map

- `Core.jl`: module imports, include order and shared interface.
- `binary_pipeline.jl`: staging paths, humidity clamp, column-weight and target
  balance protocols; workflow includes live in `../Preprocessing.jl`.
- `met_sources.jl`: settings types, full 14-argument `RawWindow`, source contracts.
- `met_readers.jl`: abstract reader contracts, seed policy and support/threading traits.
- `target_geometry.jl`: LL, RG and CS target geometry and metadata.
- `configuration.jl`: runtime settings and common flux/column-weight parsers.
- `logging.jl`: public `FlushingLogger` for preprocessing scripts.

- `source_factory.jl`: source settings factory, extension protocols and native output naming.

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

`spectral_construction.jl` owns `AbstractSpectralFluxConstruction`,
`PointSampledSpectralFluxes`, `TM5SpectralFluxes`, `AbstractSpectralSynthesis`,
`spectral_construction`, `spectral_resolution`,
`spectral_read_truncation`, `spectral_source_flux_sampling`, `balance_window_layers!`,
`validate_gpu_spectral_backend`, `resolve_spectral_construction` and
`validate_spectral_construction`. The strategy selector is
`[preprocessing].spectral_flux_construction`; existing face settings keep their meaning.

# shared: AbstractSpectralFluxConstruction, PointSampledSpectralFluxes, TM5SpectralFluxes, AbstractSpectralSynthesis, spectral_construction, spectral_resolution, spectral_read_truncation, spectral_source_flux_sampling, balance_window_layers!, validate_gpu_spectral_backend, resolve_spectral_construction, validate_spectral_construction

`validate_spectral_vertical` checks complete A/B columns and merge maps before
any meteorology read. ConfigChecks rejects unknown preprocessing keys and
native sources reject spectral-only options. Backend and automatic precision
choices are parsed into `CPUSynthesis`, `GPUSynthesis`, `AutoSynthesisPrecision`.

# shared: CPUSynthesis, GPUSynthesis, AutoSynthesisPrecision, validate_spectral_vertical

ConfigChecks owns common scalar parsing and the spectral mass-basis choice.
Deferred native flux-choice validation still occurs after keyword overrides.
