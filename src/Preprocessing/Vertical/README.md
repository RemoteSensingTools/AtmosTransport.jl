# Preprocessing.Vertical

Hybrid coordinates, native-level selection and vertical transforms. Extensive
sums visit native levels in ascending order and convert each value to the
output type before addition. Intensive means retain their existing weight
floor; pressure overlap remains the working spectral remap.

## File map

- `Vertical.jl`: module and interface.
- `vertical_coordinates.jl`: kept interfaces, merge maps, preset selection,
  coefficient path resolution and one hybrid-coefficient TOML loader.
- `vertical_transforms.jl`: production plans, field kinds and reductions.
- `configuration.jl`: spectral vertical setup, slicing and derived A/B metadata.

## Depends on

Core (`package_path`), Grids hybrid coordinates and Parameters presets/pressure;
TOML, Logging and the package data-path resolver.

## Used by

Spectral's extensive merge and deferred source/target workflows via the parent.

## Shared helpers owned here

The exports in `Vertical.jl`: `load_hybrid_coefficients`,
`load_era5_vertical_coordinate`, `build_vertical_setup`,
`merge_thin_levels`, `select_levels_echlevs`, `echlevs_preset`, production
`plan_vertical` / `apply_vertical!` and `sum_merged_layers!`.
The error-only PressureOverlap and test-only explicit-index policy are removed.
The former `load_ab_coefficients` file reader is removed; vertical setup derives
its coefficient differences from the already loaded native coordinate.

## Forward<->adjoint mirrors

None.
