# PhysicsInputs

Meteorological input conversion for convection and boundary-layer diffusion.

## File map

- [`PhysicsInputs.jl`](PhysicsInputs.jl) — module imports, public API and include order.
- [`convection_mass_basis.jl`](convection_mass_basis.jl) — interface humidity and moist-to-dry convection conversion.
- [`era5_surface_reader.jl`](era5_surface_reader.jl) — ERA5 single-level surface input IO, orientation, units and fill policy.
- [`tm5_bldiff.jl`](tm5_bldiff.jl) — TM5 boundary-layer Kvh, Kz and dkg column conversion.
- [`tm5_convection_conversion.jl`](tm5_convection_conversion.jl) — ECMWF rates to TM5 entrainment/detrainment and virtual-temperature dz.

## Depends on

Spectral supplies `normalize_lon_to_centered`; GriddedIO supplies surface source
meshes; Grids, State and Parameters supply geometry, mass-basis tags and constants.
Load order is Core → Vertical → Spectral → MassClosure → PhysicsInputs.
PhysicsInputs does not depend on MassClosure; it imports no target, source,
TransportBinary or parent-facade implementation.

## Used by

The Preprocessing facade preserves script, Models and test import sites.
Deferred target and source workflows call these stage APIs.

## Shared helpers owned here

- `BLDiffColumnScratch`
- `BLDiffConstants`
- `BLDiffDiag`
- `ERA5SurfaceReader`
- `ERA5_SURFACE_MAX_MASKED_FRACTION`
- `ERA5_SURFACE_TIME_DIMS`
- `ERA5_SURFACE_VAR_CANDIDATES`
- `ERA5_USTAR_DRAG_COEFFICIENT`
- `TM5CleanupStats`
- `close_era5_surface_reader`
- `dz_hydrostatic_virtual!`
- `ec2tm_from_rates!`
- `ecmwf_convection_to_basis!`
- `era5_surface_source_mesh`
- `era5_tm5_convection_header`
- `fill_masked_surface`
- `interface_specific_humidity`
- `load_era5_surface_window`
- `open_era5_surface_reader`
- `reset_bldiff_diag!`
- `resolve_era5_surface_path`
- `tm5_bldiff_center_kz_column!`
- `tm5_bldiff_dkg_column!`
- `tm5_bldiff_kvh_column!`
- `tm5_convection_to_basis!`

## Forward<->adjoint mirrors

None. These are offline CPU preprocessing routines.
