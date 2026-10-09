# sources

Native met-source readers and the TOML factory that builds their settings.

Each source file defines a settings type (`<: AbstractMetSettings`), a
day-handle type with open and close functions, per-window readers on the native
source grid, and the trait methods the entry point and writers query. The
writers that turn these fields into transport binaries live in
[`../transport_binary/`](../transport_binary/README.md). ERA5 spectral input
(VO/D/LNSP GRIB) does not go through this folder: `ERA5SpectralSettings` is in
[`../met_sources.jl`](../met_sources.jl) and the GRIB reader in
[`../spectral_io.jl`](../spectral_io.jl).

## Entry Points

- Factory: `load_met_settings(toml_path; root_dir, kwargs...)` in
  [`loader.jl`](loader.jl). It reads `[source].name` from a
  `config/met_sources/*.toml` file, maps it with `_settings_constructor`
  (`"GEOS-IT"`, `"GEOS-FP"`, `"ERA5-N320"`, `"MERRA-2"`), and builds the
  settings with `_build_met_settings` from `[grid]`, `[vertical]`, and
  `[preprocessing]`. Keyword arguments override TOML values; the entry point
  passes `root_dir`, `include_*` flags, `physics_dir`, `physics_layout`, and
  `coefficients_file` from the run config.
- GEOS ([`geos.jl`](geos.jl)): `GEOSITSettings`, `GEOSFPSettings`,
  `open_day` (`open_geos_day` / `open_geosfp_native_day`),
  `read_window!(raw::RawWindow, settings, handles, date, win)`,
  `endpoint_dry_mass!`, `detect_level_orientation`.
- ERA5 N320 ([`era5.jl`](era5.jl)): `ERA5N320Settings`, `open_era5_day`,
  `read_era5_n320_window_fields!`, `derive_n320_dry_mass!`,
  `derive_c180_dry_mass!`, `regrid_n320_to_c180!`,
  `read_era5_n320_convection_window!`, and the per-window bundle
  `ERA5N320ToC180Pipeline` / `process_era5_n320_window!`.
- MERRA-2 ([`merra2.jl`](merra2.jl)): `MERRA2Settings`, `NASAArchive`,
  `GEOSChemArchive`, `open_merra2_day`, `read_merra2_window_fields`,
  `read_merra2_physics_window`, `read_merra2_next_day_endpoint`,
  `detect_merra2_level_order`, `validate_merra2_settings`.

## Sources Compared

| | GEOS-IT / GEOS-FP | ERA5 N320 | MERRA-2 |
|---|---|---|---|
| Native grid | CS (C180 IT, C720 FP), L72 | reduced Gaussian N320, L137 | lat-lon 576 x 361, L72 |
| Files | IT: daily `GEOSIT.YYYYMMDD.<collection>.C<Nc>.nc`; FP: hourly `tavg_1hr_ctm` files | `<root>/ml_an_native_core/era5_core_YYYYMMDD.grib` (+ `ml_fc_convection/`, `sfc_an_native/`) | GES DISC `M2I3NVASM`/`M2T3NVASM`, or GEOS-Chem `MERRA2.YYYYMMDD.<tag>.05x0625.nc4` |
| `windows_per_day` | 24 | 24 | 8 (3-hour blocks) |
| Level order | detected from `DELP`, flipped to top-down | `level_orientation = :top_down` (default) | detected from `QV`, flipped to top-down |
| Horizontal flux | `MFXC`/`MFYC` divided by `mass_flux_dt` | built from winds by the writer | built from winds by the writer |
| Fills `RawWindow` | yes (`read_window!`) | no | no |
| `supports_day_threading` | `false` | `true` | `false` |

## File Map

- [`geos.jl`](geos.jl) — `GEOSSettings`, GEOS-IT and GEOS-FP path resolution,
  GEOS-FP physics fallback (`physics_dir`, `physics_layout`), day handles,
  `detect_level_orientation`, `endpoint_dry_mass!`, `read_window!` with
  optional surface, VDIFF, and convection fields, `_native_output_filename`
- [`era5.jl`](era5.jl) — `ERA5GRIBSettings`, stream paths (`era5_grib_path`,
  `era5_arco_sp_path`), `ERA5GRIBDayHandles`, N320 grid discovery, synthesis
  of U/V/T/PS plus reduced-Gaussian Q, dry mass on N320 and CS, convection
  forecast reader, N320-to-CS regrid, TM5 convection derivation, the per-window pipeline
- [`merra2.jl`](merra2.jl) — archive types, `MERRA2Settings`, path resolution,
  `MERRA2DayHandles`, level-order detection, window, physics, and next-day readers
- [`loader.jl`](loader.jl) — `load_met_settings`, `_settings_constructor`,
  `_build_met_settings` per source, rejection of unsupported `[preprocessing]` keys

## Common Tasks

- Adding a source:
  1. Define `NewSettings <: AbstractMetSettings` in a new file here and
     include it from [`../Preprocessing.jl`](../Preprocessing.jl) before
     `sources/loader.jl`.
  2. In [`loader.jl`](loader.jl), add `_settings_constructor(::Val{Symbol("NAME")})`
     and `_build_met_settings(::Type{NewSettings}, cfg, root_dir; kwargs...)`;
     extend `_supported_flux_keys` if the source honors flux-construction keys.
  3. Add `config/met_sources/<name>.toml` with `[source].name = "NAME"`.
  4. Implement `open_day`, `close_day!`, `windows_per_day(::NewSettings, ::Date)`,
     and the `has_surface` / `has_convection` / `has_vdiff_fields` traits.
  5. Add the writer and its `process_day`, `preprocessor_pair_supported`, and
     `_native_output_filename` methods (see [`../README.md`](../README.md)).
- Level order looks wrong: set GEOS `[preprocessing].level_orientation` to
  `"bottom_up"` or `"top_down"` to bypass detection. MERRA-2 detection errors
  unless one end of the column has more than 100 times the mean `QV` of the other.
- GEOS-FP surface or convection fields: set `physics_dir`; `physics_layout`
  is `auto`, `cubed_sphere`, or `latlon_025`.
- MERRA-2 surface, convection, VDIFF, and cloud-base fields require
  `layout = "geoschem"` (`GEOSChemArchive`).

## Invariants

- Readers return top-down arrays (`k = 1` at the model top).
- GEOS `MFXC`/`MFYC` are dry and pass through unchanged apart from the
  `1 / mass_flux_dt` scaling; `PS` and `DELP` are moist and are converted with
  `QV` (`endpoint_dry_mass!`); `CMFMC` and `DTRAIN` are converted to dry with
  the window-mean `QV`.
- GEOS `mass_flux_dt` defaults to `450` s; the loader accepts
  `[preprocessing].mass_flux_dt_seconds` in 100–3600 and warns outside 400–500.
- The last GEOS window needs the next day's `CTM_I1` endpoint and errors
  without it; ERA5 N320 and MERRA-2 fall back to a zero-tendency final window
  with a warning.
- The loader rejects options a source does not implement, for example
  `arco_surface_pressure` on GEOS, `include_vdiff_fields` on ERA5 N320, and
  `include_tm5_diffusion` on MERRA-2. ERA5 `include_tm5_diffusion` requires
  `include_surface`.

## Tests

In [`../../../test/core/`](../../../test/core/): `test_met_source_loader.jl`,
`test_met_sources_trait.jl`, `test_met_readers.jl`, `test_geos_reader.jl`,
`test_geosfp_native_physics_fallback.jl`, `test_geos_convection.jl`,
`test_era5_n320_reader.jl`, `test_era5_n320_window_reader.jl`,
`test_era5_n320_dry_mass.jl`, `test_era5_n320_convection.jl`,
`test_era5_n320_to_c180_regrid.jl`, `test_era5_n320_to_c180_pipeline.jl`,
`test_era5_c180_tm5_convection.jl`.
