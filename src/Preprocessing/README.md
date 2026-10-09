# Preprocessing

Offline generation of transport binaries from meteorological input.

ERA5 spectral or N320 GRIB, GEOS-IT/GEOS-FP cubed-sphere NetCDF, and MERRA-2
lat-lon NetCDF become one binary per day on a lat-lon, reduced-Gaussian, or
cubed-sphere target: a JSON header plus, per met window, `m`, `am`/`bm` (RG:
face-indexed `hflux`), `cm`, `ps`, forward deltas (CS `dm`; LL `dam`, `dbm`,
`dcm`, `dm`; none on RG), and optional physics sections. By default every
writer balances the fluxes, diagnoses `cm`, and runs the replay and positivity
gates before the file is kept (exceptions under Invariants). The runtime reads it through [`../MetDrivers/`](../MetDrivers/README.md).

## Entry Points

- CLI: [`../../scripts/preprocessing/preprocess_transport_binary.jl`](../../scripts/preprocessing/preprocess_transport_binary.jl)
  `<config.toml> [--day D | --start D --end D]` calls
  `process_day(cfg::Dict{String, Any}; day_override, start_date, end_date)` in
  [`transport_binary/entrypoint.jl`](transport_binary/entrypoint.jl). A config
  with `[source].toml` takes the native-source path (`load_met_settings`);
  otherwise it is ERA5 spectral (`[input].spectral_dir`, `ERA5SpectralSettings`).
- Per-day extension point: `process_day(date, grid::AbstractTargetGeometry,
  settings, vertical; ...)`; the fallback in
  [`transport_binary/topology_dispatch.jl`](transport_binary/topology_dispatch.jl)
  rejects pairs unless `preprocessor_pair_supported` returns `true`.
- Target grids: `build_target_geometry(cfg_grid, FT)` ([`target_geometry.jl`](target_geometry.jl));
  `[grid].type` is `latlon`, `era5_native_reduced_gaussian` (alias
  `reduced_gaussian`), `synthetic_reduced_gaussian`, or `cubed_sphere`.
- `load_met_settings` ([`sources/README.md`](sources/README.md)) and
  `run_unified_preprocessor_day!` ([`transport_binary/README.md`](transport_binary/README.md)).
- Other tools with CLIs under [`../../scripts/preprocessing/`](../../scripts/preprocessing/):
  `regrid_transport_binary` / `regrid_ll_binary_to_cs` (LL binary to CS),
  `coarsen_nested_cs_transport_binary` (experimental, not exported),
  `convert_era5_physics_nc_to_bin` (TM5 convection input).

## Supported Source and Target Pairs

| Settings type | Target | `process_day` lives in | Day loop |
|---|---|---|---|
| `ERA5SpectralSettings` | `LatLonTargetGeometry` | [`transport_binary/latlon_spectral.jl`](transport_binary/latlon_spectral.jl) | unified driver |
| `ERA5SpectralSettings` | `ReducedGaussianTargetGeometry` | [`reduced_spectral_day.jl`](reduced_spectral_day.jl) | unified driver |
| `ERA5SpectralSettings` | `CubedSphereTargetGeometry` | [`transport_binary/cubed_sphere_spectral.jl`](transport_binary/cubed_sphere_spectral.jl) | unified driver |
| `GEOSITSettings`, `GEOSFPSettings` | `CubedSphereTargetGeometry` | [`transport_binary/cubed_sphere_geos.jl`](transport_binary/cubed_sphere_geos.jl) | unified driver |
| `ERA5N320Settings` | `CubedSphereTargetGeometry` | [`transport_binary/era5_n320_regrid.jl`](transport_binary/era5_n320_regrid.jl) | own loop |
| `MERRA2Settings` | `CubedSphereTargetGeometry` | [`transport_binary/merra2_latlon_regrid.jl`](transport_binary/merra2_latlon_regrid.jl) | own loop |
| LL transport binary | `CubedSphereTargetGeometry` | [`transport_binary/cubed_sphere_regrid.jl`](transport_binary/cubed_sphere_regrid.jl) | own loop, own CLI |

GEOS targets need `panel_convention = "geos_native"` and a target `Nc` equal
to the source `Nc` or an integer divisor of it (nested block coarsening).

## Pipeline

The entry point resolves configuration once and calls `process_day` per date.
Native sources run days serially, passing `final_m` and `global_mass_target_kg`
forward; days run on threads only when `supports_day_threading(settings)` (ERA5
N320), `chain_mass = false`, and an enabled mass pin has a fixed target.

ERA5 spectral to LL, RG, or CS:

1. Read the day's VO/D/LNSP GRIB (`read_day_spectral`; missing files skip the
   day with a warning) and next-day 00 UTC when present (`next_day_hour0`).
2. Per hour, synthesize `ps` and winds and build native `m` and fluxes
   (`spectral_to_native_fields!`; CS uses an internal LL staging grid).
3. Pin global dry mass when `[mass_fix].enable` (default `true` here) with
   `pin_global_mean_ps_using_qv!` (hourly qv) or `pin_global_mean_ps!`; convert
   to dry basis; merge to output levels.
4. CS only: conservative regrid to panels, recover and rotate winds, rebuild
   face fluxes (`reconstruct_cs_fluxes!`).
5. Balance against the forward endpoint mass (LL `balance_column_mass_fluxes!`,
   CS `balance_cs_column_mass_fluxes!`, RG per-layer
   `balance_compressed_horizontal_fluxes!`), then diagnose `cm`
   (`recompute_cm_from_dm_target!`, `diagnose_cs_cm!`).
6. Choose the substep count, run the window contract, write. LL stores all
   windows and balances them at flush; RG and CS emit window `n` after
   ingesting window `n + 1`.

GEOS-IT / GEOS-FP to CS:

1. `read_window!` fills a `RawWindow`: dry endpoint mass from moist `PS`/`QV`
   (`endpoint_dry_mass!`), `MFXC`/`MFYC` divided by `mass_flux_dt`, top-down levels.
2. Stagger fluxes to faces (identity or block coarsening), apply the vertical
   plan, seed `m_cur` (previous endpoint when chaining) and the target from the
   raw dry endpoint, and pin both when `[mass_fix].enable`.
3. Default `geos_cm_closure = "endpoint_balanced"`: `balance_cs_column_mass_fluxes!`
   (per layer with `geos_balance_mode = "per_layer"`), then `diagnose_cs_cm!`.
4. Adaptive substeps, contract, write with `flux_kind = :full_window_mass_amount`.

ERA5 N320 and MERRA-2 to CS:

1. Read source fields, regrid PS, winds, and humidity conservatively to CS,
   re-derive dry layer mass there, and pin it when `[mass_fix].enable` and the
   target is finite (`mode = "target_ps_dry"`, the default).
2. Build face fluxes from winds ([`flux_construction.jl`](flux_construction.jl)
   options), merge levels (N320), column-balance, `diagnose_cs_cm!`.
3. Adaptive substeps, `verify_cs_window_contract!`, write, summarize, promote.

## File Map

Files directly in this folder; the subfolders have their own READMEs
([`sources/`](sources/README.md), [`transport_binary/`](transport_binary/README.md)).

- [`Preprocessing.jl`](Preprocessing.jl) — module, imports, include order, exports
- [`binary_pipeline.jl`](binary_pipeline.jl) — include list for most of `transport_binary/`
- [`logging.jl`](logging.jl) — `_FlushingLogger`, used by the CLI so logs reach redirected files
- [`met_sources.jl`](met_sources.jl) — `AbstractMetSettings`, `ERA5SpectralSettings`, `RawWindow`, source trait generics
- [`met_readers.jl`](met_readers.jl) — `AbstractMetReader`, `NoChain`/`ChainedMass`, `GEOSNativeReader`, `ERA5SpectralReader`, `preprocessor_pair_supported`, `supports_day_threading`
- [`target_geometry.jl`](target_geometry.jl) — `AbstractTargetGeometry`, LL/RG/CS target types, `build_target_geometry`, header metadata
- [`configuration.jl`](configuration.jl) — ERA5 spectral TOML parsing (`resolve_runtime_settings`, `build_vertical_setup`, dates, `next_day_hour0`)
- [`vertical_coordinates.jl`](vertical_coordinates.jl) — `merge_thin_levels`, `select_levels_echlevs`, echlevs presets, coefficient loaders
- [`vertical_transforms.jl`](vertical_transforms.jl) — `AbstractVerticalTransform` types, `VerticalPlan`, `plan_vertical`, `apply_vertical!`
- [`spectral_io.jl`](spectral_io.jl) — spectral GRIB reading, day cache, `read_day_spectral`, `read_hour0_spectral`
- [`spectral_synthesis.jl`](spectral_synthesis.jl) — Legendre/FFT synthesis to LL, `spectral_to_native_fields!`, native mass and fluxes
- [`reduced_spectral_synthesis.jl`](reduced_spectral_synthesis.jl) — batched spectral synthesis on reduced-Gaussian rings (`synthesize_reduced!`)
- [`reduced_transport_helpers.jl`](reduced_transport_helpers.jl) — RG workspaces, humidity, spectral synthesis, fluxes, level merge
- [`reduced_window_buffer.jl`](reduced_window_buffer.jl) — RG two-slot window buffer and workspace, ingest/drain/flush
- [`reduced_spectral_day.jl`](reduced_spectral_day.jl) — RG window synthesis, pin/dry/merge and next-day end point, `balance_window!`, driver hooks, RG `process_day`
- [`mass_support.jl`](mass_support.jl) — LL merge/remap, LL Poisson and column balance, qv readers, `apply_dry_basis_native!`, `pin_global_mean_ps!`, `pin_global_mean_ps_using_qv!`
- [`ring_poisson_balance.jl`](ring_poisson_balance.jl) — RG `CompressedLaplacian`, `balance_compressed_horizontal_fluxes!`
- [`cs_face_table.jl`](cs_face_table.jl) — CS global cell and face indexing (the face table)
- [`cs_poisson_solver.jl`](cs_poisson_solver.jl) — CS graph Laplacian, PCG solver, correction mapping, mirror synchronization
- [`cs_poisson_balance.jl`](cs_poisson_balance.jl) — `balance_cs_global_mass_fluxes!`, `balance_cs_column_mass_fluxes!`, column weights, `diagnose_cs_cm!`
- [`cs_transport_helpers.jl`](cs_transport_helpers.jl) — CS workspace, panel packing, LL → CS regrid helpers, wind recovery from LL fluxes
- [`cs_flux_reconstruction.jl`](cs_flux_reconstruction.jl) — CS face fluxes from cell-center winds (`cs_face_fluxes!`)
- [`cs_wind_rotation.jl`](cs_wind_rotation.jl) — east/north ↔ panel-local wind rotation, CS face fluxes → cell-center winds
- [`cs_native_fluxes.jl`](cs_native_fluxes.jl) — `geos_native_to_face_flux!` with panel halo sync, `compute_cs_cm_pressure_fixer!`
- [`face_line_integrals.jl`](face_line_integrals.jl) — `LineIntegralFaceFluxes` (cube-face fluxes integrated from N320 winds)
- [`flux_construction.jl`](flux_construction.jl) — `[preprocessing]` flux options (`face_fluxes`, `face_lengths`, `face_interpolation`, `wind_regrid`) for MERRA-2 and ERA5 N320
- [`era5_physics_binary.jl`](era5_physics_binary.jl) — ERA5 physics NetCDF to flat BIN converter and reader
- [`era5_surface_reader.jl`](era5_surface_reader.jl) — ERA5 single-level PBL fields (`open_era5_surface_reader`, `load_era5_surface_window`)
- [`tm5_convection_conversion.jl`](tm5_convection_conversion.jl) — `ec2tm!`, `ec2tm_from_rates!` (ECMWF to TM5 entu/detu/entd/detd)
- [`tm5_convection_pipeline.jl`](tm5_convection_pipeline.jl) — per-hour TM5 fields on the physics grid (`compute_tm5_merged_hour_on_source!`)
- [`tm5_bldiff.jl`](tm5_bldiff.jl) — TM5 `bldiff` column kernels; `tm5_bldiff_dkg_column!` feeds the N320 `dkg` payload

## Common Tasks

- Adding a met source: follow [`sources/README.md`](sources/README.md). Its
  `process_day` must accept the entry point's full keyword set (end with
  `kwargs...`) and return a NamedTuple; the serial loop reads `final_m` and
  `global_mass_target_kg` from it with `get`.
- Adding a target topology: subtype `AbstractTargetGeometry`, add
  `build_target_geometry(::Val{:kind}, cfg_grid, FT)` and
  `supports_spectral_massflux_preprocessing` (else `ensure_supported_target`
  rejects it), then a contract, writer, and hooks (see `transport_binary/`).
- Debugging a replay-gate failure ("Write-time replay gate FAILED for ...
  window N"): the tolerance is `replay_tolerance(FT)` (`1e-4` Float32, `1e-10`
  Float64). Check that `cm` was diagnosed after the last change to the
  horizontal fluxes, that balance and replay use the same `m_next`, and the
  final window's next-day endpoint. `ATMOSTR_NO_WRITE_REPLAY_CHECK=1` skips
  write-time replay on every path; positivity still runs. Inspect
  outputs with [`../../scripts/diagnostics/inspect_transport_binary.jl`](../../scripts/diagnostics/inspect_transport_binary.jl).
- Debugging a positivity failure ("Per-substep positivity contract violated"):
  the message recommends a substep count. Use `[numerics].substep_schedule =
  "adaptive_cfl"` (entry-point default) and `max_steps_per_window`, or
  `require_substep_positivity = false` to keep the binary with a warning.

## Invariants

- Dry basis is the default: `[output].mass_basis` defaults to `"dry"`; the
  GEOS, ERA5 N320, and MERRA-2 writers reject other values; spectral dry basis
  requires `[input].thermo_dir`.
- `k = 1` is the model top: GEOS files are flipped after
  `detect_level_orientation`, MERRA-2 readers return top-down arrays, ERA5
  N320 settings default to `level_orientation = :top_down`. `cm` is zero at
  the top (`k = 1`) and surface (`k = Nz + 1`) interfaces.
- Balance before `cm`: horizontal fluxes are balanced against the target
  `(m_next - m) / (2 * steps)` (each substep applies them twice; see
  `poisson_balance_target_scale`), then `cm` is diagnosed from that explicit
  `dm`. The GEOS diagnostic closures `pressure_fixer` and `pfix_corrected`
  skip the balance; every non-default closure logs a warning.
- A kept binary passed the write-time replay gate (unless
  `ATMOSTR_NO_WRITE_REPLAY_CHECK=1` skipped it) and the per-substep
  positivity gate against `positivity_cfl_limit` (default `0.95`, in
  `(0, 1]`). CS checks `2 * (out_x + out_y + out_z)` against
  `min(m, m_next)`; LL and RG check each direction's outflow against `m`. With
  `require_substep_positivity = true` (default) a violation errors and deletes
  the staged file; with `false` it is kept with a warning.
- Outputs are staged as `<out>.tmp` and renamed after the gates.
- GEOS `mass_flux_dt` defaults to `450` s (`[preprocessing].mass_flux_dt_seconds`
  must be 100–3600, warns outside 400–500); the GEOS writer uses
  `round(dt_met_seconds / mass_flux_dt)` source substeps per window.
- ERA5 spectral: `met_interval / dt` must be an integer (`exact_steps_per_window`).

## Related Docs And Tests

- Manual: [`overview.md`](../../docs/src/preprocessing/overview.md),
  [`spectral_era5.md`](../../docs/src/preprocessing/spectral_era5.md),
  [`geos_native_cs.md`](../../docs/src/preprocessing/geos_native_cs.md),
  [`conventions.md`](../../docs/src/preprocessing/conventions.md),
  [`binary_format.md`](../../docs/src/concepts/binary_format.md)
- Tests in [`../../test/core/`](../../test/core/): `test_preprocessor_unified_driver.jl`,
  `test_{ll,rg,cs}_preprocessor_contract.jl`, `test_preprocessor_substep_schedule.jl`,
  `test_replay_consistency.jl`, `test_global_ps_pin.jl`, `test_poisson_balance.jl`,
  `test_rg_poisson_balance.jl`, `test_vertical_transforms.jl`,
  `test_preprocessing_integrity.jl`, `test_preprocessing_cache_io.jl`; byte
  reproducibility in `test_ll_spectral_unified_driver.jl`,
  `test_rg_preprocessor_unified_driver.jl`, `test_cs_spectral_unified_driver.jl`.
  End-to-end cases against recorded references: [`../../test/golden/`](../../test/golden/).
