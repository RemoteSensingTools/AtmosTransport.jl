# Initial Conditions And Surface Fluxes

Cubed-sphere initial-condition builders and the file-based surface-flux sources
for every topology.

These files are included into the `InitialConditionIO` module at the end of
[`../InitialConditionIO.jl`](../InitialConditionIO.jl). The parent file owns the
lat-lon and reduced-Gaussian `build_initial_mixing_ratio` methods,
`pack_initial_tracer_mass`, the NetCDF IC loader
`_load_file_initial_condition_source`, and the log-pressure profile interpolation
these files reuse. The parent overview is [`../README.md`](../README.md).

## Entry Points

- [`cubed_sphere.jl`](cubed_sphere.jl):
  `build_initial_mixing_ratio(air_mass::NTuple{6}, grid::AtmosGrid{<:CubedSphereMesh},
  cfg; surface_pressure)` returns six interior `(Nc, Nc, Nz)` dry-VMR panels. The
  kinds are:
  - analytic: `uniform`, `latitude_step` (aliases `lat_step`, `hemisphere_step`),
    `gaussian_blob`
  - file-backed: `file`, `netcdf`, `file_field`, `catrine_co2`
  - CS only: `pressure_layer`, `cs_native`
- Cubed-sphere packing: `pack_initial_tracer_mass` (parent) dispatches to the CS
  `_pack_tracer_mass` methods here. They return halo-padded panels with a zeroed
  halo.
- Runner-private CS helpers: `_build_cs_initial_mixing_ratio(..., vmr; surface_pressure)`
  (reuses an interior buffer) and `_cs_pack_interior_into_halo!`. Both are called
  by `_initialize_cs_dry_state` in [`../runner/model_setup.jl`](../runner/model_setup.jl).
- [`surface_flux.jl`](surface_flux.jl): `build_surface_flux_source(grid, tracer_name,
  cfg, FT; reference_time)` (methods for lat-lon, reduced-Gaussian, and cubed-sphere
  grids) and `build_surface_flux_sources(grid, tracer_specs, FT; reference_time)`. They
  return `SurfaceFluxSource` or `TimeVaryingSurfaceFluxSource` from
  [`../../Operators/SurfaceFlux/`](../../Operators/SurfaceFlux/), or `nothing` for
  `kind = "none"`.
- Loaded-field containers: `FileSurfaceFluxField`, `TimeVaryingFileSurfaceFluxField`
- Shared source mesh: `_build_source_latlon_mesh(lon, lat; radius)` builds the
  Float64 `LatLonMesh` used as the conservative-regridding source for both CS file
  ICs and surface fluxes.

## File Map

- [`cubed_sphere.jl`](cubed_sphere.jl) — source lat-lon mesh construction, CS
  `build_initial_mixing_ratio` (analytic, file, `cs_native`, `pressure_layer`),
  CS dry/moist halo packing
- [`surface_flux.jl`](surface_flux.jl) — kind and default-file resolution, NetCDF
  load and unit conversion, GridFED calendar-year inference, time-varying series
  loader (with a run start, reads flux data only for the slices of the run
  period), topology builders; includes
  the two files below
- [`surface_flux_regridding.jl`](surface_flux_regridding.jl) — bilinear
  renormalisation, `_regridding_method`, cached conservative regridder build/apply
- [`surface_flux_native.jl`](surface_flux_native.jl) — `cs_native` time-varying
  loader, unit regex, `cell_area` agreement check

## Surface-Flux Kinds

| `kind` | Defaults in `_resolve_surface_flux_file` | Conversion to kg species m⁻² s⁻¹ | `time_varying` |
|---|---|---|---|
| `gridfed_fossil_co2` | GridFED `TOTAL`; needs `time_index` or `month` | kgCO2/month/m2 ÷ month length; year from `year`, time units, or file name | CS, 12 monthly slices |
| `edgar_sf6` | EDGAR v8 `emissions` | tonnes per cell per year ÷ cell area ÷ 365.25 d | no |
| `zhang_rn222` | Zhang `rnemis`; `month` (default 1) | per-second units | no |
| `lmdz_co2` | CAMS `flux_apos` | kgC ×44/12; static path averages all time slices | CS |
| `cs_native` | none; needs `file`, `variable` | kg species or kgC m⁻² s⁻¹ on the runtime cube | CS, required |
| any other | none; needs `file`, `variable` | units must be per second, or one of the forms above | no |

## Common Tasks

- To add a surface-flux dataset kind:
  1. Give it defaults in `_resolve_surface_flux_file` (optional). Without
     defaults, the kind requires `file` and `variable`.
  2. Add a unit branch to `_load_file_surface_flux_field`. For time-varying
     support, also add one to `_load_single_timevarying_surface_flux_field`, list
     the kind in `_surface_flux_supports_time_varying`, and set its default scheme
     in `_build_timevarying_cs_surface_flux_source`.
  3. Add the tracer molar mass to `_KNOWN_TRACER_MOLAR_MASS_KG_MOL`, or document
     that users must set `molar_mass_kg_mol`.
  4. Add the kind to the `surface_flux.kind` enum in
     [`../../../schemas/atmos_transport_run.schema.json`](../../../schemas/atmos_transport_run.schema.json)
     and to [`../../../docs/src/config/toml_schema.md`](../../../docs/src/config/toml_schema.md).
- To add a CS initial-condition kind, add a branch to
  `_build_cs_initial_mixing_ratio`:
  - Analytic kinds should fill `_cs_initial_vmr_storage(vmr, FT, Nc, Nz)` so the
    runner's buffer reuse works. File-backed kinds may return fresh panels.
  - Extend the `init.kind` enum in the run schema and the kinds table in
    `toml_schema.md`.
- To check emitted totals:
  - Conservative regridding logs source and destination totals and warns above
    `rel_err = 1e-6`.
  - The runner logs each source's total model-storage rate.
- To use a different regrid-weight cache, set `ATMOSTR_REGRID_CACHE_DIR`. Without
  it, `_regrid_cache_dir()` uses `~/.cache/AtmosTransport/cr_regridding`. CS file
  ICs call `build_regridder` without a cache directory.

## Invariants

- Builders return **dry VMR**. CS panels are interior-only. Halos are added at
  packing time and stay zero.
- Tracer storage is dry VMR × air mass on `DryBasis`, and dry VMR × air mass ×
  (1 − qv) on `MoistBasis`. The moist case requires `qv`, which canonical CS
  windows do not carry.
- Surface sources are model-storage rates per cell. They equal the physical
  kg species s⁻¹ times `DRY_AIR_MOLAR_MASS / M_species`
  (`_surface_flux_storage_scale`). An unknown tracer without `molar_mass_kg_mol` is
  an error.
- `k = 1` is the top of the atmosphere:
  - `cs_native` ICs default to `vertical_order = "surface_first"` and are flipped.
  - `pressure_layer` with `lowest_layer = true` selects `k = Nz`.
- CS file ICs and `pressure_layer` need the binary's `surface_pressure` panels.
  Target half levels come from the grid's hybrid coefficients and that pressure.
- `pressure_layer` indexes `air_mass` as halo-padded panels (`[Hp + i, Hp + j, k]`).
- The source lat-lon mesh and the EDGAR fallback cell areas use the destination
  mesh radius, so regridding weights compare areas on one sphere.
- CS surface fluxes are regridded conservatively, except `cs_native` sources,
  which are read on the run's panels. LL and RG default to
  `regridding = "bilinear"`, renormalised to the native total only when the file
  carries `cell_area` or `area`.
- Time-varying sources are cubed-sphere only (LL/RG builders throw):
  - Slice times are seconds since `reference_time`.
  - `lmdz_co2`, `gridfed_fossil_co2`, and `cs_native` default to
    `temporal_scheme = "stepwise"`.
  - `cs_native` checks a file `cell_area` against the runtime mesh (`rtol = 1e-4`).
- `cs_native` ICs keep negative values unless `clamp_negative = true`.

## Related Tests

Under [`../../../test/core/`](../../../test/core/): `test_initial_condition_io.jl`,
`test_cs_initial_state_packing.jl`, `test_cs_pressure_layer_initialization.jl`,
`test_native_cs_surface_flux.jl`, `test_cs_driven_builders.jl`,
`test_binary_planet_radius.jl` (source-mesh radius, inventory totals),
`test_regrid_cache_dir.jl`, `test_float32_conservation.jl` (source-mesh extents).
