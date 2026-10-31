# Sources

Native met-source readers and the TOML factory that builds their settings.

Each source file defines a settings type (`<: AbstractMetSettings`), a
day-handle type with open and close functions, per-window readers on the native
source grid, and the trait methods the entry point and writers query. The
writers that turn these fields into transport binaries live in
[source day drivers](GEOS/README.md). ERA5 spectral input
(VO/D/LNSP GRIB) does not go through this folder: `ERA5SpectralSettings` is in
[`../Core/met_sources.jl`](../Core/met_sources.jl) and the GRIB reader in
[`../Spectral/spectral_io.jl`](../Spectral/spectral_io.jl).

## Entry Points

- Factory: `load_met_settings(toml_path; root_dir, kwargs...)` in
  [`../Core/source_factory.jl`](../Core/source_factory.jl). It reads `[source].name` from a
  `config/met_sources/*.toml` file, maps it with `settings_constructor`
  (`"GEOS-IT"`, `"GEOS-FP"`, `"ERA5-N320"`, `"MERRA-2"`), and builds the
  settings with `build_met_settings` from `[grid]`, `[vertical]`, and
  `[preprocessing]`. Keyword arguments override TOML values; the entry point
  passes `root_dir`, `include_*` flags, `physics_dir`, `physics_layout`, and
  `coefficients_file` from the run config.
- GEOS ([`geos.jl`](GEOS/geos.jl), [`geos_panels.jl`](GEOS/geos_panels.jl), [`geos_read_window.jl`](GEOS/geos_read_window.jl)): `GEOSITSettings`, `GEOSFPSettings`,
  `open_day` (`open_geos_day` / `open_geosfp_native_day`),
  `read_window!(raw::RawWindow, settings, handles, date, win)`,
  `endpoint_dry_mass!`, `detect_level_orientation`.
- ERA5 N320 ([`era5.jl`](ERA5/era5.jl) and the `era5_n320_*.jl` files): `ERA5N320Settings`, `open_era5_day`,
  `read_era5_n320_window_fields!`, `derive_n320_dry_mass!`,
  `derive_c180_dry_mass!`, `regrid_n320_to_c180!`,
  `read_era5_n320_convection_window!`, and the per-window bundle
  `ERA5N320ToC180Pipeline` / `process_era5_n320_window!`.
- MERRA-2 ([`merra2.jl`](MERRA2/merra2.jl)): `MERRA2Settings`, `NASAArchive`,
  `GEOSChemArchive`, `open_merra2_day`, `read_merra2_window_fields`,
  `read_merra2_physics_window`, `read_merra2_next_day_endpoint`,
  `detect_merra2_level_order`, `validate_merra2_settings`.

## Sources Compared

| | GEOS-IT / GEOS-FP | ERA5 N320 | MERRA-2 |
|---|---|---|---|
| Native grid | CS (C180 IT, C720 FP), L72 | reduced Gaussian N320, L137 | lat-lon 576 x 361, L72 |
| Files | IT: daily `GEOSIT.YYYYMMDD.<collection>.C<Nc>.nc`; FP: hourly `tavg_1hr_ctm` (HH30) + `inst_1hr_ctm` (HH00) files | `<root>/ml_an_native_core/era5_core_YYYYMMDD.grib` (+ `ml_fc_convection/`, `sfc_an_native/`) | GES DISC `M2I3NVASM`/`M2T3NVASM`, or GEOS-Chem `MERRA2.YYYYMMDD.<tag>.05x0625.nc4` |
| `windows_per_day` | 24 | 24 | 8 (3-hour blocks) |
| Level order | detected from `DELP`, flipped to top-down | canonical GRIB order is top-down | detected from `QV`, flipped to top-down |
| Horizontal flux | `MFXC`/`MFYC` divided by `mass_flux_dt` | built from winds by the writer | built from winds by the writer |
| Fills `RawWindow` | yes (`read_window!`) | no | no |
| `supports_day_threading` | `false` | `false` | `false` |

## File Map

The children are direct `Preprocessing.GEOS`, `.ERA5` and `.MERRA2` modules,
like the target submodules. This container holds no Julia implementation.
See their file maps: [GEOS](GEOS/README.md), [ERA5](ERA5/README.md), [MERRA2](MERRA2/README.md).
The TOML factory lives in [Core/source_factory.jl](../Core/source_factory.jl);
source-specific constructors extend its protocols in each child's `configuration.jl`.

## Invariants

- Readers return top-down arrays (`k = 1` at the model top).
- GEOS `MFXC`/`MFYC` are dry and pass through unchanged apart from the
  `1 / mass_flux_dt` scaling; `PS` and `DELP` are moist and are converted with
  `QV` (`endpoint_dry_mass!`); `CMFMC` and `DTRAIN` are converted to dry with
  the window-mean `QV`.
- GEOS `mass_flux_dt` defaults to `450` s; the loader accepts
  `[preprocessing].mass_flux_dt_seconds` in 100–3600 that divides 3600 exactly,
  and warns outside 400–500. `[preprocessing].level_orientation` must be
  `auto` (default), `bottom_up` or `top_down`; anything else throws.
- GEOS-FP requires both `tavg_1hr_ctm_c0720_v72` (DELP for orientation,
  dry MFXC/MFYC) and `inst_1hr_ctm_c0720_v72` (endpoint PS/QV), in the flat
  root or `YYYYMMDD/` subdirectories. Window n is `(n-1):00`..`n:00` UTC;
  its endpoint dry DELP is `(1-QV_inst) * (ΔA + ΔB*PS_inst)`, using the
  same `endpoint_dry_mass!` as GEOS-IT. Tavg PS/DELP are window means and
  do not supply endpoint mass. There is no PS/QV fallback.
- The last GEOS window needs the next day's `CTM_I1` endpoint (GEOS-FP:
  `inst_1hr_ctm` at 00Z, required when opening the day by default) and errors
  without it; ERA5 N320 and MERRA-2 fall back with a warning to a zero-tendency
  final window only on the configured last day, and error if the endpoint is
  missing earlier.
- The loader rejects options a source does not implement, for example
  `arco_surface_pressure` on GEOS, `include_vdiff_fields` on ERA5 N320, and
  `include_tm5_diffusion` on MERRA-2. ERA5 `include_tm5_diffusion` requires
  `include_surface`. Of the flux-construction keys, GEOS takes only
  `column_balance_weights` (it keeps the native face fluxes).

## Tests

In [`../../../test/core/`](../../../test/core/): `test_met_source_loader.jl`,
 `test_geos_reader.jl`,
`test_geosfp_native_physics_fallback.jl`, `test_geos_convection.jl`,
`test_era5_n320_reader.jl`, `test_era5_n320_window_reader.jl`,
`test_era5_n320_dry_mass.jl`, `test_era5_n320_convection.jl`,
`test_era5_n320_to_c180_regrid.jl`, `test_era5_c180_tm5_convection.jl`. The non-ARCO decoder and pipeline allocation/coefficient guards need the ERA5
N320 GRIB archive (`ATMOS_ERA5_N320_ROOT`) and run in the optional block of
`test/core/test_era5_n320_to_c180_regrid.jl`. Full-day numerical output is guarded
by the preprocessing goldens.

- `era5_grib_geometry.jl`: native ERA5 reduced-Gaussian GRIB geometry and mesh readers (moved from MetDrivers.ERA5; preprocessing-only).

## Depends on

Sources depend on Core, Vertical, Spectral, MassClosure, PhysicsInputs, Targets
and TransportBinary. The three sources have no dependencies on one another.
Shared CS mass/pin/delta bodies moved to Targets/CubedSphere, dry CMFMC/DTRAIN
conversion and TM5 column scratch to PhysicsInputs, and source-factory/naming
protocols to Core. The lower stages never depend on sources.

## Used by

The Preprocessing root entrypoint and public facade; preprocessing scripts.

## Shared helpers owned here

None in this container. Each source declares its public reader/driver surface.

## Forward<->adjoint mirrors

No adjoint. All sources use the same target and binary replay contracts.
