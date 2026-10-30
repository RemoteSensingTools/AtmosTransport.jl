# Initial Conditions And Surface Fluxes

Cubed-sphere initial-condition builders and the file-based surface-flux sources
for every topology.

These files are included into the `InitialConditionIO` module at the end of
[`InitialConditionIO.jl`](InitialConditionIO.jl). That file owns the
lat-lon and reduced-Gaussian `build_initial_mixing_ratio` methods,
`pack_initial_tracer_mass`, the NetCDF IC loader
`_load_file_initial_condition_source`, and the conservative hybrid-coordinate remap
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
  by `_initialize_cs_dry_state` in [`../Runner/model_setup.jl`](../Runner/model_setup.jl).
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
  (`source_latlon_mesh(lon, lat; radius)`, using `SnappedFaces` on the
  destination's sphere).

VMR builders accept `surface_pressure`, `pressure_basis`, and `qv` for
pressure remapping. Set `mass_basis` on `pack_initial_tracer_mass` when
converting the resulting dry VMR to model storage.

## File Initial Conditions

The file kinds read layer-mean dry VMR. A file with `ap` and `bp` interface
coefficients is remapped vertically; it does not need `Psurf`. For each model
column, both the source and target interfaces use the transport binary's own
surface pressure ``p_{s,t}``:

```math
p^s_{i+1/2}=a^s_{i+1/2}+b^s_{i+1/2}p_{s,t}, \qquad
p^t_{k+1/2}=A^t_{k+1/2}+B^t_{k+1/2}p_{s,t}.
```

This surface anchoring keeps boundary-layer parcels at the same position in
the model column despite topographic or dry-air-mass differences between the
file's model and the transport binary. Aloft, where ``b^s=0``, it reduces to
ordinary absolute-pressure placement.

`ap` is always interpreted in Pa. An explicit units attribute must say Pa; if
it is absent, only an all-zero pure-sigma coefficient or a magnitude clearly
in the pascal range is accepted. This prevents an hPa coefficient from moving
every pure-pressure interface by a factor of 100.

The remap treats each source value as a layer mean. Source humidity is opt-in
through `init.humidity_variable`; conventional names such as `q`, `QV`, and
`hus` are not auto-detected. With ``d^s=1-q_v^s``, the source-dry-air mean is

```math
C_q(p)=\int_{p^s_{top}}^p q^s(p')d^s(p')\,dp',\qquad
C_d(p)=\int_{p^s_{top}}^p d^s(p')\,dp',
\qquad q^t_k=\frac{\Delta C_q}{\Delta C_d},
```

each target value is source-dry-air weighted and a uniform source remains
uniform when target humidity differs. If a target
interface lies outside the source coverage, the nearest end source layer is
extended. This includes pure pressure-level sources (`bp = 0`) whose bottom
may lie above or below the model surface.

The previous point interpolation used the file's `Psurf` for source levels
and the binary's surface pressure for target levels. In the CATRINE C90 case,
that made the bottom-10%-of-column CO₂ depend on the dry-mass pin and produced
1.2–1.5 ppm RMS local errors (maxima 62–70 ppm). The surface-anchored
conservative remap reduces the RMS error to 0.007 ppm and the global mean error
to at most 0.001 ppm in the measured pin comparison. The real-file regression
in `test/core/test_initial_condition_io.jl` measures a −0.000153 ppm C90
near-surface difference for a uniform 400 Pa surface-pressure shift.

## File Map

- `InitialConditionIO.jl`: module, LL/RG builders, shared file-column remap and storage packing.

- [`cubed_sphere.jl`](cubed_sphere.jl) — CS `build_initial_mixing_ratio`
  (analytic, file, `cs_native`, `pressure_layer`), CS dry/moist halo packing
- [`surface_flux.jl`](surface_flux.jl) — kind and default-file resolution,
  `surface_flux_species` and the storage factor, NetCDF load and unit
  conversion (`_to_flux_density!`, dispatched on the parsed units' area basis),
  GridFED calendar-year inference, source cell areas (the file's, else the
  source mesh's band areas), time-varying series loader (with a run span,
  checks that the slices of all files together cover it,
  `_timevarying_source_intervals` and `_check_flux_time_coverage`, and
  reads flux data only for the slices of the run period), topology builders;
  includes the two files below
- [`surface_flux_regridding.jl`](surface_flux_regridding.jl) — the regridding
  methods `ConservativeFluxRegridding` (default; per-cell rate `F × A` over the
  source cells' covered areas) and `BilinearFluxRegridding` (explicit LL/RG;
  emission and uptake renormalised separately, factors logged and warned
  beyond 10 %, or an error), parsed by
  `_regridding_method`; cached conservative regridder build, the per-slice
  density regrid of time-varying series
- [`surface_flux_native.jl`](surface_flux_native.jl) — `cs_native` time-varying
  source: checks the file with `GriddedIO.cs_native_flux_info` (layout, units,
  `cell_area` gate, time checks live there), checks that its slices
  (`time_bnds`, else the stamps) cover the run span, and reads it panel by
  panel with `GriddedIO.read_cs_native_flux!` straight into the source's `FT`
  arrays (fill and non-finite checks), then slice times on the run clock and,
  in place, cell areas and the units and storage factors, and the temporal
  scheme

The `catrine_co2` default IC is
`$ATMOSTRANSPORT_DATA_ROOT/catrine/InitialConditions/startCO2_202112010000.nc`
(variable `CO2`), resolved by `expand_data_path`; the root defaults to
`~/data/AtmosTransport`. An explicit `file` overrides it.

## Surface-Flux Kinds

| `kind` | Defaults in `_resolve_surface_flux_file` | Units of the default file → kg species m⁻² s⁻¹ | `time_varying` |
|---|---|---|---|
| `gridfed_fossil_co2` | GridFED `TOTAL`; needs `time_index` or `month` | kgCO2/month/m2 ÷ the slice's CF time bounds/calendar month; coordinate-less legacy files alone use ordinal month plus `year`/file-name year | CS, 12 monthly slices for the coordinate-less format |
| `edgar_sf6` | EDGAR v8 `emissions` | Tonnes (per cell and year) × 1000 ÷ cell area ÷ 365.25 d | no |
| `zhang_rn222` | Zhang `rnemis`; `month` (default 1) | kg/m2/s | no |
| `lmdz_co2` | CAMS `flux_apos` | kgC m-2 s-1 × M(CO2)/M(C) ≈ 3.66419; static path averages all time slices; `month` raises `ArgumentError` | CS |
| `cs_native` | none; needs `file`, `variable` | kg species or kgC m⁻² s⁻¹ on the runtime cube (`GriddedIO.read_cs_native_flux!`) | CS, required |
| `file` | none; needs `file`, `variable` | any units `GriddedIO.parse_flux_units` reads (`GriddedIO.write_latlon_flux` writes static files) | no |

The conversion follows the units attribute of the variable, not the kind: the
loaders parse it with `GriddedIO.parse_flux_units` and divide by the cell area
for per-cell units (the file's `cell_area`/`area`, else the band areas of the
grid on the destination sphere), multiply by `GriddedIO.kg_per_unit` (kg of the
tracer's species per unit; kg of another species is an error) and divide by
`GriddedIO.period_seconds` (per-month units: CF time bounds when present, else
the coordinate's month in its own calendar). A static `lmdz_co2` source rejects `month`; select its input file or use a
time-varying source. Its `time_index` is checked for validity even though the
static source averages all slices. `time_index` is always a 1-based
positional selector. `month` selects the unique decoded calendar month when a
time coordinate exists, and is ordinal only for the documented legacy GridFED
format with twelve slices and no time coordinate. Stepwise time-varying fields
switch at CF bound starts when bounds exist, otherwise at coordinate stamps.
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
- The target hybrid coefficients define the model's level order:
  - equal-level `cs_native` ICs with `ap`/`bp` infer source order from interface
    pressure, orient values to the target order, and require coefficients that
    match the binary; without coefficients, `vertical_order` defaults to
    `"surface_first"`;
  - mismatched-level `cs_native` ICs require `ap`/`bp` and use the conservative
    hybrid remap, with optional `humidity_variable` dry-carrier weighting and
    the binary's typed total/dry surface-pressure basis.
  - `pressure_layer` with `lowest_layer = true` selects `k = Nz`.
- CS file ICs and `pressure_layer` need the binary's `surface_pressure` panels.
  Target half levels come from the grid's hybrid coefficients and that pressure.
- `pressure_layer` indexes `air_mass` as halo-padded panels (`[Hp + i, Hp + j, k]`).
- The source lat-lon mesh and the EDGAR fallback cell areas use the destination
  mesh radius, so regridding weights compare areas on one sphere. Both derive
  `SnappedFaces` from the oriented file coordinates, including midpoint faces
  and polar caps for pole-centred files.
- Static surface fluxes are regridded conservatively by default on every
  topology (`ConservativeFluxRegridding`): the per-cell rate `F × A`, with the
  file's `cell_area`/`area` or else the source mesh's band areas, is divided by
  each source cell's covered area (`Regridding.source_coverage`), regridded and
  multiplied by the target areas, so every model total equals the file's own
  total `native_total_mass_rate` to rounding (also on reduced-Gaussian targets,
  whose great-circle cells leave slivers uncovered). On a non-global target a
  partly covered source cell puts its whole rate into the covered part. A file
  area must be positive and finite wherever the flux is nonzero
  (`GriddedIO.check_latlon_cell_area(...; flux)`). LL and RG accept
  `regridding = "bilinear"`, renormalised per sign (emission to the file's
  emission, uptake to its uptake) or refused; CS warns and stays conservative.
  `cs_native` sources are read on the run's panels. Time-varying LL series are
  regridded per slice as densities (great-circle source areas, no file areas).
- Time-varying sources are cubed-sphere only (LL/RG builders throw):
  - Slice times are seconds since `reference_time`, from the file's time axis
    as `GriddedIO.decode_time` decodes it (`days`/`hours`/`minutes`/`seconds
    since` through CFTime with any CF calendar, `months since` as month starts;
    other axes are refused), or, for GridFED files without one, the first day
    of each month of the source year.
  - `lmdz_co2`, `gridfed_fossil_co2`, and `cs_native` default to
    `temporal_scheme = "stepwise"`.
  - With a run span (the runner always passes one), the slices' intervals
    over all files of the source (`_timevarying_source_intervals`: CF bounds
    and `months since` months per file, `GriddedIO.known_slice_intervals`;
    GridFED without a time axis: the months of its year; the bare stamps of
    all files merged into one series, `GriddedIO.stamp_intervals`, each
    holding for the series' regular step, gaps left uncovered; a single stamp
    or mid-month stamps are an error; `cs_native`: its one file,
    `GriddedIO.decode_slice_intervals`) must cover the run, up to
    `RUN_SPAN_CLOCK_SLACK_SECONDS` at its end; otherwise an `ArgumentError`
    names the uncovered dates. `hold_outside_coverage = true` skips the check,
    and the source holds its first or last slice outside its files.
  - `cs_native` files follow the schema of `GriddedIO` (`CS_NATIVE_FLUX_DIMS`,
    kg species or kgC m⁻² s⁻¹, no fill values); `GriddedIO.check_cell_area`
    compares a file `cell_area` with the runtime mesh (norm-based, `rtol = 1e-4`
    per panel). New files come from `GriddedIO.open_cs_native_writer`.
- `cs_native` ICs keep negative values unless `clamp_negative = true`.

## Related Tests

Under [`../../../test/core/`](../../../test/core/): `test_initial_condition_io.jl`,
`test_cs_initial_state_packing.jl`, `test_cs_pressure_layer_initialization.jl`,
`test_native_cs_surface_flux.jl` (units/storage factors and slice-selection matrix),
`test_cs_native_flux_schema.jl` (schema round trip and reader gates),
`test_cs_driven_builders.jl`,
`test_binary_planet_radius.jl` (source-mesh radius, inventory totals),
`test_float32_conservation.jl` (source-mesh extents),
`test_config_matrix_contract.jl` (every shipped config's fluxes have a species).
The file conventions are tested under `test_gridded_io_*.jl` and
`test_flux_units.jl`.

## Depends on

GriddedIO (native file metadata, vertical order, Pa units, year inference),
Grids, Regridding, State, ConfigChecks, Parameters and Operators.SurfaceFlux.
The `InitialConditionIO` module keeps its original qualified path.

## Used by

DrivenRunner, Diagnostics, surface-inventory scripts and IC/flux-ingestion tests.

## Shared helpers owned here

Public builders, `uniform_background`, `load_file_surface_flux_field` and
`conservative_surface_flux_rate` and `surface_flux_storage_scale`. The runner reuses these internal seams:

# shared: _build_cs_initial_mixing_ratio, _cs_pack_interior_into_halo!, _regridding_method, _surface_flux_kind, _hold_outside_coverage, _load_file_surface_flux_field, _conservative_surface_flux_rate, _surface_flux_storage_scale

`INITIAL_CONDITION_KINDS` is the shared alias-to-spec table for parsers, builders
and runner key checks; Gaussian parameter reads and scalar evaluation also have
one home here. `HybridRemapScratch` and `_file_ic_column!` own the shared file-IC column step;
all pressure/cumulative scratch stays Float64. Legacy qualified metadata helper
names import the same functions now owned by GriddedIO.

## Forward<->adjoint mirrors

None. Initial state and surface sources are built once; Core captures the
materialized emission rates for the adjoint.
