# Parameters

Physical constants and planetary parameters.

[`PhysicalConstants.jl`](PhysicalConstants.jl) is the one place for the
constants of nature the model uses, in SI units, with their sources in the
docstrings. [`PlanetParameters.jl`](PlanetParameters.jl) defines the immutable
planet object (radius, gravity, reference pressure) that `AtmosGrid` carries.

`src/AtmosTransport.jl` loads this folder after `Architectures`,
`SectionTimer`, and `Quantities`, and before `Grids`. It imports nothing from
other AtmosTransport modules; every later module may import from it.

## Entry Points

- Constants of nature, in [`PhysicalConstants.jl`](PhysicalConstants.jl):
  - `EARTH_RADIUS` (6.371e6 m), the default radius of meshes built without one
    (and of transport binaries that do not record `planet_radius_m`)
  - `IFS_EARTH_RADIUS` (6.371229e6 m), the radius of the ERA5 spectral transforms,
    TM5 and every preprocessing target mesh (recorded in binaries as `planet_radius_m`)
  - `STANDARD_GRAVITY`, `STANDARD_PRESSURE`, `THETA_REFERENCE_PRESSURE`
  - `R_DRY_AIR`, `CP_DRY_AIR`, `CP_OVER_R_DIATOMIC`, `DRY_AIR_MOLAR_MASS`,
    `VIRTUAL_TEMPERATURE_FACTOR`, `AVOGADRO`
  - `SPECIES_MOLAR_MASS`, the named tuple `(co2, sf6, rn222)` in kg mol⁻¹
- Constant sets for reproducing another model's scheme, in
  [`PhysicalConstants.jl`](PhysicalConstants.jl):
  - `TM5_CONSTANTS`: TM5 `binas.F90` values, used as the defaults of
    `BLDiffConstants` in `../Preprocessing/tm5_bldiff.jl`
  - `GEOSCHEM_CONSTANTS`: GEOS-Chem `physconstants.F90` / `vdiff_mod.F90`
    values, used as the defaults of `GCHPVdiffParameters` in
    `../State/Fields/GCHPNonlocalPBLField.jl`
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
- [`PhysicalConstants.jl`](PhysicalConstants.jl): all constants and the
  scheme constant sets, with exports
- [`PlanetParameters.jl`](PlanetParameters.jl): `PlanetParameters` and
  `earth_parameters`

## Conventions

- Constants live here, not at call sites. A consumer imports constants by name
  (`using ..Parameters: STANDARD_GRAVITY, R_DRY_AIR`). The current importers
  are `Grids`, `State/Fields`, `Output`, `MetDrivers` (and `MetDrivers/ERA5`),
  `Operators/Diffusion`, `Preprocessing`, `Models/InitialConditionIO`, and
  `Visualization`.
- A scheme that reproduces another model's code takes that model's set
  (`TM5_CONSTANTS`, `GEOSCHEM_CONSTANTS`) so it matches the reference model.
  Everything else uses the model's own values.
- Scheme tuning parameters, such as stability-function slopes and critical
  Richardson numbers, stay with their schemes, not here.
- Defaults are converted to the object's own precision. Tests require
  `PlanetParameters(; FT).radius === FT(EARTH_RADIUS)`, and the same holds for
  `CubedSphereMesh`, `BLDiffConstants`, `GCHPVdiffParameters`, and
  `PBLPhysicsParameters`.
- The dry-air set is coherent: `CP_DRY_AIR ≈ CP_OVER_R_DIATOMIC * R_DRY_AIR`
  (tested).

## Common Tasks

- Add a constant: add a documented `const` (units and source) and its
  `export` in [`PhysicalConstants.jl`](PhysicalConstants.jl). Pin its value in
  `test/core/test_physical_constants.jl` and import it by name where it is
  used.
- Run on a non-default sphere: build meshes and `AtmosGrid` with an explicit
  `radius`. Transport binaries record `planet_radius_m`, and the runtime
  builds its mesh on that radius (see `test/core/test_binary_planet_radius.jl`).

## Tests And Docs

- [`../../test/core/test_physical_constants.jl`](../../test/core/test_physical_constants.jl):
  pinned values, precision of defaults, one dry-air gas constant
- [`../../test/core/test_public_contracts.jl`](../../test/core/test_public_contracts.jl):
  `PlanetParameters` validation
- [`../../test/core/test_basis_explicit_core.jl`](../../test/core/test_basis_explicit_core.jl):
  `PlanetParameters` and `AtmosGrid`
- [`../../test/core/test_binary_planet_radius.jl`](../../test/core/test_binary_planet_radius.jl):
  radius round-trip through binaries and runtime meshes
- API page: [`../../docs/src/api/parameters.md`](../../docs/src/api/parameters.md)
