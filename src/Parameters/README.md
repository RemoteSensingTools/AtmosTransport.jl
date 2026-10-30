# Parameters

Physical constants, numerical defaults, met-source constants and planetary
parameters: every number that more than one place uses is defined here once.

[`PhysicalConstants.jl`](PhysicalConstants.jl) is the one place for the
constants of nature the model uses, in SI units, with their sources in the
docstrings. [`NumericalDefaults.jl`](NumericalDefaults.jl) holds the one
default of each numerical setting that several code paths offer (keyword
argument, config key, command-line option). [`MetSourceConstants.jl`](MetSourceConstants.jl)
holds the fixed properties of the external met datasets (level counts,
archive cadences, accumulation periods, grid offsets).
[`PlanetParameters.jl`](PlanetParameters.jl) defines the immutable planet
object (radius, gravity, reference pressure) that `AtmosGrid` carries.

`src/AtmosTransport.jl` loads this folder after `Architectures`,
`SectionTimer`, and `Quantities`, and before `Grids`. It imports nothing from
other AtmosTransport modules; every later module may import from it.

## Entry Points

- Constants of nature, in [`PhysicalConstants.jl`](PhysicalConstants.jl):
  - `EARTH_RADIUS` (6.371e6 m), the radius of every preprocessing target mesh,
    the default radius of other meshes, and the fallback for transport binaries
    that do not record `planet_radius_m`
  - `IFS_EARTH_RADIUS` (6.371229e6 m), used only for source-side calculations
    whose physical definition comes from IFS/TM5: ERA5 spectral wind synthesis,
    the native N320 source mesh, and the N320 humidity remap used by the TM5
    attachment path. It is not a transport target radius.
  - `STANDARD_GRAVITY`, `STANDARD_PRESSURE`, `THETA_REFERENCE_PRESSURE`
  - `R_DRY_AIR`, `CP_DRY_AIR`, `CP_OVER_R_DIATOMIC`, `DRY_AIR_MOLAR_MASS`,
    `VIRTUAL_TEMPERATURE_FACTOR`, `AVOGADRO`
  - `SPECIES_MOLAR_MASS`, the named tuple `(co2, sf6, rn222)` in kg mol⁻¹,
    `SPECIES_CARBON_ATOMS` and `SPECIES_LABELS` (the carbon counts and flux-unit
    spellings), and `CARBON_MOLAR_MASS` (kgC → kgCO₂ by `SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS`)
  - `RN222_HALF_LIFE` (3.8235 d = 330350.4 s)
  - `VON_KARMAN` (0.41) of the model's own boundary-layer schemes (the TM5
    and GEOS-Chem ports use their models' 0.4)
  - U.S. Standard Atmosphere 1976 values: `STANDARD_SEA_LEVEL_AIR_DENSITY`
    (1.225 kg m⁻³), `STANDARD_LAPSE_RATE` (0.0065 K m⁻¹),
    `STANDARD_TROPOPAUSE_TEMPERATURE` (216.65 K)
  - `HYDROSTATIC_REFERENCE_TEMPERATURE` (260 K), the isothermal temperature of
    hydrostatic layer thicknesses when no temperature is available
  - `SECONDS_PER_DAY` (86400 s), the day length used by gridded flux/time axes
  - `DAYS_PER_JULIAN_YEAR` (365.25), for annual totals given as rates
  - `GLOBAL_MEAN_DRY_SURFACE_PRESSURE` (98325 Pa), the shared modern-reanalysis
    dry-mass fix target,
    and `GLOBAL_MEAN_SPECIFIC_HUMIDITY` (0.00247 kg kg⁻¹), which turns it into
    a total-pressure target
- Numerical defaults, in [`NumericalDefaults.jl`](NumericalDefaults.jl):
  - `DEFAULT_CFL_LIMIT` (0.95), the runtime palindrome outflow budget of
    split-sweep advection on every topology (fraction of the smaller of a
    cell's masses at the start and end of the step that one palindrome's
    worth of subcycled passes may remove)
  - `DEFAULT_POSITIVITY_CFL_LIMIT` (0.95), the write-time substep positivity
    limit of transport binaries
  - adaptive write-time substep targets: `DEFAULT_SUBSTEP_CFL_TARGET` (0.95),
    `ERA5_N320_SUBSTEP_CFL_TARGET` (0.85, ERA5 N320 → cubed sphere),
    `COARSEN_SUBSTEP_CFL_TARGET` (0.90, cubed-sphere coarsener)
  - `DEFAULT_MET_INTERVAL_SECONDS` (3600 s), the met-window length when the
    source or config gives none
  - `DEFAULT_BACKGROUND_VMR` (4.0e-4, 400 ppm dry), the VMR of tracers whose
    initial condition sets none
  - `DEFAULT_GEOS_SMOOTHING_SWEEPS` (8, an `Int`) and
    `DEFAULT_GEOS_SMOOTHING_WEIGHT` (0.5), the sweeps of the GEOS moisture
    filter and column-drift correction and the weight of one sweep
  - `DEFAULT_MAX_SUBCYCLES` (4096, an `Int`), the most CFL subcycles one
    split-sweep advection step may need before the run stops
  - `DEFAULT_MAX_STEPS_PER_WINDOW` (4096, an `Int`), the most write-time
    substeps per window the cubed-sphere binary coarsener accepts
  - `ADAPTIVE_SUBSTEP_MAX_REFINEMENTS` (8, an `Int`), how many times the
    adaptive write-time substep schedule raises a window's substep count (to the
    count the measured Courant ratio implies) before accepting it
  - horizontal Poisson flux balance: `DEFAULT_POISSON_BALANCE_TOL` (1e-14,
    the conjugate-gradient residual tolerance), `DEFAULT_POISSON_BALANCE_MAX_ITER`
    (20000, an `Int`) and `DEFAULT_POISSON_PROJECT_EVERY` (50, an `Int`, the
    iterations between null-space projections of the cubed-sphere solve)
  - `HYBRID_LAYER_MIN_DB` (1e-6), the smallest `ΔB_k` at which a layer counts
    as hybrid in the `hybrid_mass` column-balance weighting (GEOS L72 layer 41,
    `ΔB = 8.2e-9`, is pure-pressure; ERA5 L137 layer 55, `7e-6`, is hybrid)
  - `COLUMN_CLOSURE_TOL_KG` (10 kg per flux application), the absolute floor
    of the per-cell closure tolerance `τ_c` of the cubed-sphere column balance
    and of the `cm` residual check (binding for Float64 payloads)
  - write-time OMEGA-fidelity gate of transport binaries:
    `OMEGA_FIDELITY_NRMSE_LIMIT` (1.0, the score of a zero `cm`),
    `OMEGA_FIDELITY_CORRELATION_MIN` (0.5), `OMEGA_FIDELITY_SLOPE_MIN` (0.5) and
    `OMEGA_FIDELITY_SLOPE_MAX` (1.5; the slope bounds reject a uniformly
    mis-scaled `cm`)
  - transport-binary replay check: `REPLAY_TOLERANCE_FLOAT64` (1e-12) and
    `REPLAY_TOLERANCE_FLOAT32` (1e-5), the largest accepted per-cell budget
    residual relative to the summed magnitudes of the budget's terms
    (`replay_tolerance(FT)` in `../MetDrivers/ReplayContinuity.jl` picks one)
  - `DEFAULT_GAUSSIAN_BLOB_SIGMA_DEG` (10°), the width of the Gaussian-blob
    initial condition when the config sets none
  - `DEFAULT_LINROOD_PPM_ORDER` (5, an `Int`), the Lin–Rood PPM edge
    reconstruction order when the config sets none (5 or 7)
- Met-source constants, in [`MetSourceConstants.jl`](MetSourceConstants.jl):
  - `ERA5_NATIVE_LEVEL_COUNT` (137, an `Int`), the IFS L137 levels of ERA5
  - immutable ERA5 L137 interface recipes `ECHLEVS_ML137_*`, including the
    verbatim TM5-MP r1112 `tropo34` and TM5 cy3-4dvar `tropo34a` lists, plus
    `ERA5_L137_ECHLEVS_PRESETS`, the named config library
  - `TM5_CONVECTION_MASS_BASIS_KEY`, the transport-header key that records
    whether TM5 convection sections carry dry- or moist-air fluxes
  - `TM5_CONVECTION_SOURCE_KEY`, `TM5_EC2TM_CONVENTION_KEY` and
    `TM5_EC2TM_CONVENTION` (`"tm5_halflevel_v2"`), the header record of where
    TM5 convection sections come from and of the ERA5 → TM5 index convention;
    the reader rejects ERA5-derived sections without the current convention
  - `ERA5_MET_INTERVAL_SECONDS` (3600 s), the ERA5 analysis interval and the
    accumulation period of its hourly accumulated fields
  - `GEOS_MASS_FLUX_DT_SECONDS` (450 s), the FV3 dynamics step over which
    GEOS-FP / GEOS-IT accumulate `MFXC`/`MFYC` (not the hourly met interval)
  - `GEOS_MET_INTERVAL_SECONDS` (3600 s) and `GEOS_3HOURLY_INTERVAL_SECONDS`
    (10800 s), the hourly CTM and three-hourly (A3, I3) GEOS / MERRA-2
    collections
  - `GMAO_CS_LONGITUDE_OFFSET_DEG` (−10°), the rotation of the GMAO cubed
    sphere
  - plausibility bounds of met input: `SURFACE_AIR_TEMPERATURE_RANGE`
    ((150, 350) K) for 2 m air temperature and `SURFACE_HEAT_FLUX_MAX_ABS`
    (5000 W m⁻²) for the surface sensible and latent heat fluxes; readers
    reject values outside them as a wrong variable, unit or fill value
- Constant sets for reproducing another model's scheme, in
  [`PhysicalConstants.jl`](PhysicalConstants.jl):
  - `TM5_CONSTANTS`: TM5 `binas.F90` values, used as the defaults of
    `BLDiffConstants` in `../Preprocessing/tm5_bldiff.jl`
  - `GEOSCHEM_CONSTANTS`: GEOS-Chem `physconstants.F90` / `vdiff_mod.F90`
    values, used as the defaults of `GCHPVdiffParameters` in
    `../State/Fields/GCHPNonlocalPBLField.jl`
  - `GEOSCHEM_SPECIES_MOLAR_MASS`: GEOS-Chem species-database molar masses
    `(co2, sf6, rn222)` = (44.01, 146.06, 222.0) × 10⁻³ kg mol⁻¹, used by
    `Analysis.GEOSChemCATRINEArchive` to convert GEOS-Chem emission fluxes to
    dry-air-equivalent mass as GEOS-Chem does
- Planet object, in [`PlanetParameters.jl`](PlanetParameters.jl):
  - `PlanetParameters{FT}(radius, gravity, reference_pressure)` throws
    `ArgumentError` unless all three are finite and positive
  - `PlanetParameters(; FT=Float64, radius, gravity, reference_pressure)`
    defaults to `EARTH_RADIUS`, `STANDARD_GRAVITY`, and `STANDARD_PRESSURE`,
    converted to `FT`
  - `earth_parameters(; FT=Float64)` returns the Earth defaults and is
    exported at the top level
- The grid accessors `radius(grid)`, `gravity(grid)`, and
  `reference_pressure(grid)` read `grid.planet`. They are defined in
  [`../Grids/AbstractMeshes.jl`](../Grids/AbstractMeshes.jl).

## File Map

- [`Parameters.jl`](Parameters.jl): module assembly
- [`PhysicalConstants.jl`](PhysicalConstants.jl): constants of nature and the
  scheme constant sets, with exports
- [`NumericalDefaults.jl`](NumericalDefaults.jl): shared defaults of numerical
  settings, with exports
- [`MetSourceConstants.jl`](MetSourceConstants.jl): fixed properties of the
  met datasets, with exports
- [`PlanetParameters.jl`](PlanetParameters.jl): `PlanetParameters` and
  `earth_parameters`

## Conventions

- Constants live here, not at call sites, and no module keeps a local copy of
  one. A consumer imports constants by name
  (`using ..Parameters: STANDARD_GRAVITY, R_DRY_AIR`; scripts use
  `using AtmosTransport.Parameters: ...` or the qualified name).
- A use site converts the constant to its own type: `FT(DEFAULT_CFL_LIMIT)` in
  `FT` code, `Float32(DEFAULT_BACKGROUND_VMR)` in Float32 code; `Int` constants
  stay `Int`. GPU kernels see only `FT`-typed values. A range constant is
  used through its ends, `first(SURFACE_AIR_TEMPERATURE_RANGE) < T < last(SURFACE_AIR_TEMPERATURE_RANGE)`.
- A keyword, config key or command-line option that offers one of these
  settings takes the constant as its default (`cfl_limit::Real = DEFAULT_CFL_LIMIT`),
  so the paths cannot drift apart.
- A scheme that reproduces another model's code takes that model's set
  (`TM5_CONSTANTS`, `GEOSCHEM_CONSTANTS`, `GEOSCHEM_SPECIES_MOLAR_MASS`) so it
  matches the reference model.
  Everything else uses the model's own values.
- Scheme tuning parameters, such as stability-function slopes, critical
  Richardson numbers and the floors of the PBL diagnostics, stay with their
  schemes, as documented fields of the scheme's parameter struct (for example
  `PBLPhysicsParameters`, `OmegaRegularization`). Struct defaults that are
  physical constants reference this module (`kappa_vk = FT(VON_KARMAN)`), and
  config fallbacks take the struct's own default
  (`get(cfg, "smoothing_steps", OmegaRegularization().smoothing_steps)`).
- Defaults are converted to the object's own precision. Tests require
  `PlanetParameters(; FT).radius === FT(EARTH_RADIUS)`, and the same holds for
  `CubedSphereMesh`, `BLDiffConstants`, `GCHPVdiffParameters`, and
  `PBLPhysicsParameters`.
- The dry-air set is coherent: `CP_DRY_AIR ≈ CP_OVER_R_DIATOMIC * R_DRY_AIR`
  (tested).

## Common Tasks

- Add a constant: add a documented `const` (units and source) and its
  `export` to the file it belongs in: a constant of nature to
  [`PhysicalConstants.jl`](PhysicalConstants.jl), a default that several code
  paths share to [`NumericalDefaults.jl`](NumericalDefaults.jl), a property of
  a met dataset to [`MetSourceConstants.jl`](MetSourceConstants.jl). Pin its
  value in the `PINNED_PARAMETERS` table of
  `test/core/test_physical_constants.jl` (the test fails for an exported
  constant that is not pinned) and import it by name where it is used. The API
  page picks up the docstring of every exported constant.
- Run on a non-default sphere: build meshes and `AtmosGrid` with an explicit
  `radius`. Transport binaries record `planet_radius_m`, and the runtime
  builds its mesh on that radius (see `test/core/test_binary_planet_radius.jl`).

## Tests And Docs

- [`../../test/core/test_physical_constants.jl`](../../test/core/test_physical_constants.jl):
  the value and type of every exported constant, coherence of the physical
  constants, precision of defaults (including the `PBLPhysicsParameters`
  floors and `BLDiffConstants.ustar_min`), one dry-air gas constant
- [`../../test/core/test_public_contracts.jl`](../../test/core/test_public_contracts.jl):
  `PlanetParameters` validation
- [`../../test/core/test_basis_explicit_core.jl`](../../test/core/test_basis_explicit_core.jl):
  `PlanetParameters` and `AtmosGrid`
- [`../../test/core/test_binary_planet_radius.jl`](../../test/core/test_binary_planet_radius.jl):
  radius round-trip through binaries and runtime meshes
- API page: [`../../docs/src/api/parameters.md`](../../docs/src/api/parameters.md)

`DEFAULT_OBSERVATION_LAYER_TEMPERATURE_KELVIN` in `NumericalDefaults.jl`
is the shared 280 K near-surface fallback for Observations and Inversion.

## Depends on

No other AtmosTransport module.

## Used by

All later modules that require physical constants or numerical defaults.

## Shared helpers owned here

The exported constants and planet parameters documented above.

## Forward<->adjoint mirrors

Forward and adjoint callers import the same constants.

`SURFACE_PRESSURE_BASIS_TOLERANCE` in `NumericalDefaults.jl` names the legacy
binary column-pressure classification threshold (2e-5).
