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
- `surface_flux_species(tracer_name, cfg)`: the species of a tracer's flux
  (`surface_flux.species`, or the species its name implies).
- File conventions come from [`../../GriddedIO/`](../../GriddedIO/): coordinates
  by name (`find_coordinate`, with the field's dimension lengths), orientation
  (`latlon_orientation`, `orient`), fill values as 0 (`read_raw(FT, …)` with
  `MaskedAsZero`, straight into the run's float type), time axes (`decode_time`, classified by their units before any
  CFTime call), flux units (`parse_flux_units`), and the Float64 source mesh of
  conservative regridding for CS file ICs and surface fluxes
  (`source_latlon_mesh(lon, lat, UniformSpanFaces(); radius)` on the
  destination's sphere).

## File Map

- [`cubed_sphere.jl`](cubed_sphere.jl) — CS `build_initial_mixing_ratio`
  (analytic, file, `cs_native`, `pressure_layer`), CS dry/moist halo packing
- [`surface_flux.jl`](surface_flux.jl) — kind and default-file resolution,
  `surface_flux_species` and the storage factor, NetCDF load and unit
  conversion (`_to_flux_density!`, dispatched on the parsed units' area basis),
  GridFED calendar-year inference, EDGAR band areas, time-varying series loader
  (with a run start, reads flux data only for the slices of the run period),
  topology builders; includes the two files below
- [`surface_flux_regridding.jl`](surface_flux_regridding.jl) — bilinear
  renormalisation, `_regridding_method`, cached conservative regridder build/apply
- [`surface_flux_native.jl`](surface_flux_native.jl) — `cs_native` time-varying
  source: checks the file with `GriddedIO.cs_native_flux_info` (layout, units,
  `cell_area` gate, time checks live there) and reads it panel by panel with
  `GriddedIO.read_cs_native_flux!` straight into the source's `FT` arrays (fill
  and non-finite checks), then slice times on the run clock and, in place, cell
  areas and the units and storage factors, and the temporal scheme

## Surface-Flux Kinds

| `kind` | Defaults in `_resolve_surface_flux_file` | Units of the default file → kg species m⁻² s⁻¹ | `time_varying` |
|---|---|---|---|
| `gridfed_fossil_co2` | GridFED `TOTAL`; needs `time_index` or `month` | kgCO2/month/m2 ÷ month length; year from `year`, time units, or file name | CS, 12 monthly slices |
| `edgar_sf6` | EDGAR v8 `emissions` | Tonnes (per cell and year) × 1000 ÷ cell area ÷ 365.25 d | no |
| `zhang_rn222` | Zhang `rnemis`; `month` (default 1) | kg/m2/s | no |
| `lmdz_co2` | CAMS `flux_apos` | kgC m-2 s-1 × M(CO2)/M(C) ≈ 3.66419; static path averages all time slices | CS |
| `cs_native` | none; needs `file`, `variable` | kg species or kgC m⁻² s⁻¹ on the runtime cube (`GriddedIO.read_cs_native_flux!`) | CS, required |
| `file` | none; needs `file`, `variable` | any units `GriddedIO.parse_flux_units` reads (`GriddedIO.write_latlon_flux` writes static files) | no |

The conversion follows the units attribute of the variable, not the kind: the
loaders parse it with `GriddedIO.parse_flux_units` and divide by the cell area
for per-cell units (the file's `cell_area`/`area`, else the band areas of the
grid on the destination sphere), multiply by `GriddedIO.kg_per_unit` (kg of the
tracer's species per unit; kg of another species is an error) and divide by
`GriddedIO.period_seconds` (per-month units: the calendar month of the slice).
Time-varying lat-lon sources need densities (per m²). Any other `kind` is an
error (`_surface_flux_kind`); `none` (the default) emits nothing.

## Common Tasks

- To add a surface-flux dataset kind:
  1. Add it to `_SURFACE_FLUX_KINDS` in `surface_flux.jl`.
  2. Give it defaults in `_resolve_surface_flux_file` (optional). Without
     defaults, the kind requires `file` and `variable`.
  3. New units belong in the grammar of `GriddedIO.parse_flux_units`
     (`src/GriddedIO/flux_units.jl`), not in a kind branch. For time-varying
     support, list the kind in `_surface_flux_supports_time_varying` and set its
     default scheme in `_build_timevarying_cs_surface_flux_source`.
  4. A new species needs an entry in `Parameters.SPECIES_MOLAR_MASS` (and, to
     let tracers omit `species`, a tracer name in `_TRACER_NAME_SPECIES`); the
     run schema's `surface_flux.species` enum lists the species.
  5. Add the kind to the `surface_flux.kind` enum in
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
  it, `Regridding.default_regrid_cache_dir()` (the cache shared with the
  builders) uses `~/.cache/AtmosTransport/cr_regridding`. CS file ICs call
  `build_regridder` without a cache directory.

## Invariants

- Builders return **dry VMR**. CS panels are interior-only. Halos are added at
  packing time and stay zero.
- Tracer storage is dry VMR × air mass on `DryBasis`, and dry VMR × air mass ×
  (1 − qv) on `MoistBasis`. The moist case requires `qv`, which canonical CS
  windows do not carry.
- Surface sources are model-storage rates per cell. They equal the physical
  kg species s⁻¹ times `DRY_AIR_MOLAR_MASS / SPECIES_MOLAR_MASS[species]`
  (`_surface_flux_storage_scale`), with the species of
  `surface_flux_species`: `surface_flux.species` (`co2`, `sf6`, `rn222`), or
  the species of a tracer named `co2`, `co2_natural`, `co2_fossil`,
  `fossil_co2`, `sf6` or `rn222`. Any other tracer without `species` is an
  error.
- `k = 1` is the top of the atmosphere:
  - `cs_native` ICs default to `vertical_order = "surface_first"` and are flipped.
  - `pressure_layer` with `lowest_layer = true` selects `k = Nz`.
- CS file ICs and `pressure_layer` need the binary's `surface_pressure` panels.
  Target half levels come from the grid's hybrid coefficients and that pressure.
- `pressure_layer` indexes `air_mass` as halo-padded panels (`[Hp + i, Hp + j, k]`).
- The source lat-lon mesh and the EDGAR fallback cell areas use the destination
  mesh radius, so regridding weights compare areas on one sphere. Both are
  uniform in the span of the centres (`UniformSpanFaces`); pole-centred files
  (CATRINE SF6 initial state) get faces shifted from their cells.
- CS surface fluxes are regridded conservatively, except `cs_native` sources,
  which are read on the run's panels. LL and RG default to
  `regridding = "bilinear"`, renormalised to the native total only when the file
  carries `cell_area` or `area`.
- Time-varying sources are cubed-sphere only (LL/RG builders throw):
  - Slice times are seconds since `reference_time`, from the file's time axis
    as `GriddedIO.decode_time` decodes it (`days`/`hours`/`minutes`/`seconds
    since` through CFTime with any CF calendar, `months since` as month starts;
    other axes are refused), or, for GridFED files without one, the first day
    of each month of the source year.
  - `lmdz_co2`, `gridfed_fossil_co2`, and `cs_native` default to
    `temporal_scheme = "stepwise"`.
  - `cs_native` files follow the schema of `GriddedIO` (`CS_NATIVE_FLUX_DIMS`,
    kg species or kgC m⁻² s⁻¹, no fill values); `GriddedIO.check_cell_area`
    compares a file `cell_area` with the runtime mesh (norm-based, `rtol = 1e-4`
    per panel). New files come from `GriddedIO.open_cs_native_writer`.
- `cs_native` ICs keep negative values unless `clamp_negative = true`.

## Related Tests

Under [`../../../test/core/`](../../../test/core/): `test_initial_condition_io.jl`,
`test_cs_initial_state_packing.jl`, `test_cs_pressure_layer_initialization.jl`,
`test_native_cs_surface_flux.jl`, `test_cs_native_flux_schema.jl` (the `cs_native`
loader against a frozen copy of the pre-GriddedIO loader, bit for bit),
`test_cs_driven_builders.jl`,
`test_binary_planet_radius.jl` (source-mesh radius, inventory totals),
`test_regrid_cache_dir.jl`, `test_float32_conservation.jl` (source-mesh extents),
`test_config_matrix_contract.jl` (every shipped config's fluxes have a species).
The file conventions are tested under `test_gridded_io_*.jl` and
`test_flux_units.jl`.
