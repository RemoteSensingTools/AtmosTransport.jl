# Preprocessing

Offline generation of transport binaries from meteorological input.

ERA5 spectral or N320 GRIB, GEOS-IT/GEOS-FP cubed-sphere NetCDF, and MERRA-2
lat-lon NetCDF become one binary per day on a lat-lon, reduced-Gaussian, or
cubed-sphere target: a JSON header plus, per met window, `m`, `am`/`bm` (RG:
face-indexed `hflux`), `cm`, `ps`, forward deltas (CS `dm`; LL `dam`, `dbm`,
`dcm`, `dm`; none on RG), and optional physics sections. By default every
writer balances the fluxes, diagnoses `cm`, and runs the replay and positivity
gates before the file is kept (exceptions under Invariants). The runtime reads it through [`../MetDrivers/`](../MetDrivers/README.md).

Every output target is built on `Parameters.EARTH_RADIUS` (6 371 000 m), and
that value is written as `planet_radius_m` so preprocessing areas and runtime
areas are identical. `IFS_EARTH_RADIUS` (6 371 229 m) is source-side only:
ERA5 spectral transforms, the native N320 source sphere, and the TM5 humidity
remap retain it where reproducing IFS geometry requires it.

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
- Target grids: `build_target_geometry(cfg_grid, FT)` ([`Core/target_geometry.jl`](Core/target_geometry.jl));
  `[grid].type` is `latlon`, `era5_native_reduced_gaussian` (alias
  `reduced_gaussian`), `synthetic_reduced_gaussian`, or `cubed_sphere`.
- `load_met_settings` ([`sources/README.md`](sources/README.md)) and
  `run_unified_preprocessor_day!` ([`transport_binary/README.md`](transport_binary/README.md)).
- Other tools with CLIs under [`../../scripts/preprocessing/`](../../scripts/preprocessing/):
  `regrid_transport_binary` / `regrid_ll_binary_to_cs` (LL binary to CS),
  `coarsen_nested_cs_transport_binary` (experimental, not exported).
- TM5 convection sections come from the ERA5 N320 to CS path
  (`include_convection = true` in the met source) or from
  [`../../scripts/preprocessing/attach_catrine_tm5_convection_cs.jl`](../../scripts/preprocessing/attach_catrine_tm5_convection_cs.jl).
  The lat-lon spectral path writes none; a config with a `[tm5_convection]`
  table is rejected (`reject_removed_tm5_convection_route`).

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
forward. The driver can thread days only when `supports_day_threading(settings)`,
`chain_mass = false`, and an enabled mass pin has a fixed target; ERA5 N320
currently returns `false` because its inner kernels use static thread schedules.

ERA5 spectral to LL, RG, or CS:

1. Read the day's VO/D/LNSP GRIB (`read_day_spectral`; missing files skip the
   day with a warning) and next-day 00 UTC when present (`next_day_hour0`).
2. Per hour, synthesize `ps` and winds and build native `m` and fluxes
   (`spectral_to_native_fields!`; CS uses an internal LL staging grid).
3. Pin global dry mass when `[mass_fix].enable` (default `true` here) with
   `pin_global_mean_ps_using_qv!` (hourly qv) or `pin_global_mean_ps!`; convert
   to dry basis; merge to output levels.
4. CS only: conservative regrid to panels and build face fluxes through the
   shared typed construction (`VectorFaceFluxes`, `PanelAverageFluxes`, or
   `LineIntegralFaceFluxes`), averaging raw endpoint fluxes by default.
5. Column-balance against the forward endpoint mass (LL
   `balance_column_mass_fluxes!`, CS `balance_cs_column_mass_fluxes!`, RG
   `balance_compressed_column_fluxes!`; each spreads its face corrections over
   the layers with `spread_column_correction!`), then diagnose `cm`
   (`recompute_cm_from_dm_target!`, `diagnose_cs_cm!`).
6. Choose the substep count, run the window contract, write. LL stores all
   windows and balances them at flush; RG and CS emit window `n` after
   ingesting window `n + 1`.

GEOS-IT / GEOS-FP to CS:

1. `read_window!` fills a `RawWindow`: dry endpoint mass from moist `PS`/`QV`
   (`endpoint_dry_mass!`), `MFXC`/`MFYC` divided by `mass_flux_dt`, top-down levels.
   GEOS-FP fallback physics validates its own level order using physical
   `lev` metadata or matching physics I3 QV, then normalizes CMFMC/DTRAIN
   before coordinate-aware interpolation and drying. Cached daily stencils
   preserve the shortened polar intervals and periodic longitude seam.
   OF-119 requires regenerating binaries from `geosfp_c720_native_to_cs720.toml`
   and `geosfp_c720_native_to_cs180.toml` with `latlon_025`; the former reader
   displaced all fallback fields in latitude and inherited CTM order for
   convection (see [details](../../docs/src/preprocessing/geos_native_cs.md)).
2. Stagger fluxes to faces (identity or block coarsening), apply the vertical
   plan, seed `m_cur` (previous endpoint when chaining) and the target from the
   raw dry endpoint, and pin both when `[mass_fix].enable`.
3. Default `geos_cm_closure = "endpoint_balanced"`: `balance_cs_column_mass_fluxes!`,
   then `diagnose_cs_cm!`.
4. Adaptive substeps, contract, write with `flux_kind = :full_window_mass_amount`.

ERA5 N320 and MERRA-2 to CS:

1. Read source fields, regrid PS, winds, and humidity conservatively to CS,
   re-derive dry layer mass there, and pin it when `[mass_fix].enable` and the
   target is finite (`mode = "target_ps_dry"`, the default).
2. Build face fluxes from winds ([`MassClosure/flux_construction.jl`](MassClosure/flux_construction.jl)
   options), merge levels (N320), column-balance, `diagnose_cs_cm!`.
3. Adaptive substeps, `verify_cs_window_contract!`, write, summarize, promote.

## File Map

Stage submodules load in pipeline order: [Core](Core/README.md), [Vertical](Vertical/README.md),
[Spectral](Spectral/README.md), [MassClosure](MassClosure/README.md),
and [PhysicsInputs](PhysicsInputs/README.md). All retained public names remain reachable through
`Preprocessing`. Targets, TransportBinary and Sources remain in their existing folders for PP3/PP4.
The subfolders have their own READMEs
([`sources/`](sources/README.md), [`transport_binary/`](transport_binary/README.md)).

- [`Preprocessing.jl`](Preprocessing.jl) — module, imports, include order, exports
- [`Core/binary_pipeline.jl`](Core/binary_pipeline.jl) — pipeline protocols and staging paths; the root includes the deferred `transport_binary/` workflows
- [`Core/logging.jl`](Core/logging.jl) — `FlushingLogger`, used by the CLI so logs reach redirected files
- [`Core/met_sources.jl`](Core/met_sources.jl) — `AbstractMetSettings`, `ERA5SpectralSettings`, `RawWindow`, source trait generics
- [`Core/met_readers.jl`](Core/met_readers.jl) — `AbstractMetReader`, `NoChain`/`ChainedMass`, `GEOSNativeReader`, `preprocessor_pair_supported`, `supports_day_threading`
- [`Core/target_geometry.jl`](Core/target_geometry.jl) — `AbstractTargetGeometry`, LL/RG/CS target types, `build_target_geometry`, header metadata
- [`Core/configuration.jl`](Core/configuration.jl) — ERA5 spectral TOML parsing (`resolve_runtime_settings`, `build_vertical_setup`, dates, `next_day_hour0`)
- [`PhysicsInputs/convection_mass_basis.jl`](PhysicsInputs/convection_mass_basis.jl) — moist-to-dry conversion and TM5 convection flux basis
- [`Vertical/vertical_coordinates.jl`](Vertical/vertical_coordinates.jl) — `merge_thin_levels`, `select_levels_echlevs`, echlevs presets, coefficient loaders
- [`Vertical/vertical_transforms.jl`](Vertical/vertical_transforms.jl) — `AbstractVerticalTransform` types, `VerticalPlan`, `plan_vertical`, `apply_vertical!`
- [`Spectral/spectral_io.jl`](Spectral/spectral_io.jl) — spectral GRIB reading, day cache (published with `AtomicFiles.atomic_publish`), `read_day_spectral`, `read_hour0_spectral`
- [`Spectral/spectral_synthesis.jl`](Spectral/spectral_synthesis.jl) — Legendre/FFT synthesis to LL, `spectral_to_native_fields!`, native mass and fluxes
- [`Spectral/reduced_spectral_synthesis.jl`](Spectral/reduced_spectral_synthesis.jl) — batched spectral synthesis on reduced-Gaussian rings (`synthesize_reduced!`)
- [`reduced_transport_helpers.jl`](reduced_transport_helpers.jl) — RG workspaces, humidity, spectral synthesis, horizontal fluxes and level merge (cm diagnosis is in MassClosure/ring_closure.jl)
- [`reduced_window_buffer.jl`](reduced_window_buffer.jl) — RG two-slot window buffer and workspace, ingest/drain/flush
- [`reduced_spectral_day.jl`](reduced_spectral_day.jl) — RG window synthesis, pin/dry/merge and next-day end point, `balance_window!`, driver hooks, RG `process_day`
- [`Spectral/mass_support.jl`](Spectral/mass_support.jl) — LL merge/remap, LL Poisson and column balance, qv readers and their coordinates (`read_thermo_coordinates`, through `GriddedIO.find_coordinate`), `apply_dry_basis_native!`, `pin_global_mean_ps!`, `pin_global_mean_ps_using_qv!`
- [`MassClosure/ring_poisson_balance.jl`](MassClosure/ring_poisson_balance.jl) — RG `CompressedLaplacian`, `balance_compressed_column_fluxes!`
- [`../Grids/CSGlobalFaceTable.jl`](../Grids/CSGlobalFaceTable.jl) — CS global cell and face indexing (the face table)
- [`MassClosure/cs_poisson_solver.jl`](MassClosure/cs_poisson_solver.jl) — CS graph Laplacian, PCG solver, correction mapping, mirror synchronization
- [`MassClosure/cs_poisson_balance.jl`](MassClosure/cs_poisson_balance.jl) — `balance_cs_global_mass_fluxes!` (the CG solve on the column), `balance_cs_column_mass_fluxes!`, `diagnose_cs_cm!`; shared weights are in `MassClosure/column_weights.jl`
- [`cs_transport_helpers.jl`](cs_transport_helpers.jl) — CS workspace, LL → CS regrid helpers (`regrid_2d_to_cs_panels!` / `regrid_3d_to_cs_panels!` pass the quantity kind to `Regridding.apply_regridder!`) (wind recovery is in MassClosure)
- [`MassClosure/cs_flux_reconstruction.jl`](MassClosure/cs_flux_reconstruction.jl) — CS face fluxes from cell-center winds (`cs_face_fluxes!`)
- [`MassClosure/cs_wind_rotation.jl`](MassClosure/cs_wind_rotation.jl) — east/north → panel-local wind rotation and LL face fluxes → cell-center winds
- [`MassClosure/cs_native_fluxes.jl`](MassClosure/cs_native_fluxes.jl) — `geos_native_to_face_flux!` with panel halo sync, `compute_cs_cm_pressure_fixer!`
- [`MassClosure/face_line_integrals.jl`](MassClosure/face_line_integrals.jl) — `LineIntegralFaceFluxes` (cube-face fluxes integrated from N320 or spectral staging-grid winds)
- [`MassClosure/flux_construction.jl`](MassClosure/flux_construction.jl) — typed flux methods and wind regrids shared by MERRA-2, ERA5 N320, and spectral ERA5-to-CS
- [`PhysicsInputs/era5_surface_reader.jl`](PhysicsInputs/era5_surface_reader.jl) — ERA5 single-level PBL fields (`open_era5_surface_reader`, `load_era5_surface_window`)
- [`PhysicsInputs/tm5_convection_conversion.jl`](PhysicsInputs/tm5_convection_conversion.jl) — `ec2tm_from_rates!` (ECMWF to TM5 entu/detu/entd/detd), virtual-temperature hydrostatic `dz`
- [`PhysicsInputs/tm5_bldiff.jl`](PhysicsInputs/tm5_bldiff.jl) — TM5 `bldiff` column kernels; `tm5_bldiff_dkg_column!` feeds the N320 `dkg` payload

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
  window N"): every cell's residual must stay within `replay_tolerance(FT)`
  (`1e-5` Float32, `1e-12` Float64) of the summed magnitudes of its budget
  terms. Check that `cm` was diagnosed after the last change to the
  horizontal fluxes, that balance and replay use the same `m_next`, and the
  final window's next-day endpoint. `[numerics] write_replay_check = false`
  skips write-time replay on every path; positivity still runs. Inspect
  outputs with [`../../scripts/diagnostics/inspect_transport_binary.jl`](../../scripts/diagnostics/inspect_transport_binary.jl).
- Debugging a positivity failure ("Per-substep positivity contract violated"):
  the message recommends a substep count. Use `[numerics].substep_schedule =
  "adaptive_cfl"` (entry-point default) and `max_steps_per_window`, or
  `require_substep_positivity = false` to keep the binary with a warning.

## Invariants

- Dry basis is the default: `[output].mass_basis` defaults to `"dry"`; the
  GEOS, ERA5 N320, and MERRA-2 writers reject other values; spectral dry basis
  requires `[input].thermo_dir`.
- `k = 1` is the model top: native GEOS CTM fields normalize after
  `detect_level_orientation`; GEOS-FP fallback physics uses its own validated
  order. MERRA-2 readers return top-down arrays, and canonical ERA5 N320
  GRIB fields are top-down. `cm` is zero at
  the top (`k = 1`) and surface (`k = Nz + 1`) interfaces.
- The LL and RG staging divergence closures require `B_ifc` with exactly
  `Nz + 1` interfaces; missing or mismatched coefficients raise an error.
- Balance before `cm`: horizontal fluxes are balanced against the target
  `(m_next - m) / (2 * steps)` (each substep applies them twice; see
  `MetDrivers.poisson_balance_target`), then `cm` is diagnosed from that explicit
  `dm`. The GEOS diagnostic closures `pressure_fixer` and `pfix_corrected`
  skip the balance; every non-default closure logs a warning.
- A kept binary passed the write-time replay gate (unless
  `[numerics] write_replay_check = false` skipped it) and the per-substep
  positivity gate against `positivity_cfl_limit` (default `0.95`, in
  `(0, 1]`). CS and LL check the palindrome outflow `2 * (out_x + out_y +
  out_z)` against `min(m, m_next)` (one shared check); RG checks each
  direction's outflow against `m`. With
  `require_substep_positivity = true` (default) a violation errors and deletes
  the staged file; with `false` it is kept with a warning.
- Outputs use unique sibling staging files from `AtomicFiles` and are
  atomically published after the gates.
- GEOS `mass_flux_dt` defaults to `450` s (`[preprocessing].mass_flux_dt_seconds`
  must be 100–3600 and divide 3600 exactly, warns outside 400–500); the GEOS
  writer uses `exact_steps_per_window(dt_met_seconds, mass_flux_dt)` source
  substeps per window, which throws on a non-integer ratio.
- `[numerics] float_type` is parsed by `ConfigChecks.parse_float_type`
  (`"Float32"`/`"Float64"`, any letter case; anything else throws). The default
  is `Float32` on the spectral path and `Float64` on the native path.
- `MergeAbovePressure` requires `target_min_thickness_Pa > 0` (`Inf` allowed).
- ERA5 spectral: `met_interval / dt` must be an integer (`exact_steps_per_window`).
- ERA5 → TM5 convection (`ec2tm_from_rates!`): UDMF/DDMF level `k` is the
  flux through the top of layer `k`, so callers pass `[GRIB levels 1..Nz; 0]`
  (interface `Nz + 1`, the ground, must be zero), and the cloud top is the
  layer above the first interface with flux, as in TM5. Writers record
  `era5_tm5_convection_header` (`tm5_ec2tm_convention = "tm5_halflevel_v2"`);
  readers reject ERA5-derived TM5 sections without it.

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
  `test_spectral_pole_rows.jl`, `test_qv_longitude_normalization.jl`,
  `test_preprocessing_integrity.jl`, `test_preprocessing_cache_io.jl`; byte
  reproducibility in `test_ll_spectral_unified_driver.jl`,
  `test_rg_preprocessor_unified_driver.jl`, `test_cs_spectral_unified_driver.jl`.
  End-to-end cases against recorded references: [`../../test/golden/`](../../test/golden/).

Panel indexing, the global face table, panel packing, geometry-tag parsing and
flat area helpers are owned by Grids; basis-tag conversion is owned by State.

ERA5 native GRIB geometry is owned here: `sources/era5_grib_geometry.jl`
defines `ERA5ReducedGaussianGeometry`, `read_era5_reduced_gaussian_geometry`,
and `read_era5_reduced_gaussian_mesh`; no runtime dependency on GRIB geometry
is required. Binary metadata and payload sizes delegate to MetDrivers.BinaryFormat.

## Depends on

Core → Vertical → Spectral → MassClosure → PhysicsInputs supply the ordered
pipeline stages. PhysicsInputs needs Spectral and has no MassClosure dependency.
The facade assembles the deferred targets, transport binary and source implementations, using the lower package modules in `src/LAYERS.toml`.

## Used by

Preprocessing scripts and the package facade call the public names here.

## Shared helpers owned here

The root registers source and closure implementations on Core's pipeline
interfaces. Concrete source readers and topology writers remain here until
PP3 and PP4.

## Forward<->adjoint mirrors

None; preprocessing runs offline on the CPU.
