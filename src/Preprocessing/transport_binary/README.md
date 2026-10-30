# transport_binary

Per-topology day workflows, window contracts, writers, and the unified driver.

By default each workflow balances window fluxes against the forward endpoint
mass, diagnoses `cm`, verifies the window contract, and streams to a staged
file that is promoted after the gates (the exceptions, such as the GEOS
diagnostic closures and the RG direct write, are listed in
[`../README.md`](../README.md)). [`../binary_pipeline.jl`](../binary_pipeline.jl)
includes `core.jl` through `entrypoint.jl`; the GEOS, coarsen, N320, and MERRA-2
files are included from [`../Preprocessing.jl`](../Preprocessing.jl) after the source readers.

## Entry Points

- TOML entry: `process_day(cfg::Dict{String, Any}; day_override, start_date, end_date)`
  in [`entrypoint.jl`](entrypoint.jl); parses `[numerics]`, `[vertical]`,
  `[mass_fix]` and loops over dates.
- Unified driver: `run_unified_preprocessor_day!(UnifiedPreprocessorDay(reader,
  workspace, contract, writer; context); close_reader = true)` in [`driver.jl`](driver.jl).
- Contracts: `LatLonContract{FT}`, `ReducedGaussianContract{FT}`,
  `CubedSphereContract{FT}` with `verify_window!`, `update_accumulator!`,
  `summarize_status!`; direct gates `verify_cs_window_contract!`,
  `verify_write_replay_cs!`, `verify_substep_positivity_{ll,rg,cs}!`.
- Substep policy: `SubstepSchedulePolicy`, `initial_substeps`, `next_substeps`,
  `rescale_substep_amounts!`, `set_contract_steps_schedule!` ([`window_contracts.jl`](window_contracts.jl)).
- Writers: `LatLonBinaryWriter`, `ReducedGaussianBinaryWriter`,
  `CubedSphereBinaryWriter`; `write_window!`, `close_streaming_binary!`,
  `promote_streaming_binary!`, `quarantine_streaming_binary!` ([`writer_adapters.jl`](writer_adapters.jl)).
- Own-loop workflows: `process_era5_n320_to_cs_day`, `process_merra2_to_cs_day`,
  `regrid_ll_binary_to_cs`, `coarsen_nested_cs_transport_binary`.

## Unified Driver Lifecycle

1. For `win in 1:driver_windows_per_day(reader, context)`: call
   `driver_ingest_window!`, then handle every event from `driver_drain_ready_windows!`.
2. A `ReadyWindow` event gets `verify_window!` and `update_accumulator!`; a
   `PreverifiedWindow` is accumulated unless `accumulated = true`. Then
   `write_window!` verifies and accumulates the event's typed flux-fidelity
   evidence before serializing its unchanged payload, then
   `driver_after_write_window!`.
3. Events from `driver_flush_final_windows!` are handled the same way.
4. `driver_before_close_writer!`, `close_streaming_binary!`, `validate_staged_binary!`
   (file size), `summarize_status!(contract; quarantine_path)`, `promote_streaming_binary!`.
5. On an exception before promotion: close and quarantine the writer; close the
   reader when `close_reader = true` (the spectral workflows pass `false`).

| Workflow | Reader | Ingest | Drain | Flush |
|---|---|---|---|---|
| LL spectral ([`latlon_spectral.jl`](latlon_spectral.jl)) | `nothing` | synthesize, pin, dry, merge into `WindowStorage` | `()` | `apply_poisson_balance!` over all windows; one `PreverifiedWindow` each |
| RG spectral ([`../reduced_spectral_day.jl`](../reduced_spectral_day.jl)) | `nothing` | synthesize into a two-slot buffer | from window 2: balance and verify window `win - 1` | last window vs next-day 00 UTC, else its own mass |
| CS spectral ([`cubed_sphere_spectral.jl`](cubed_sphere_spectral.jl)) | `nothing` | synthesize on staging grid, regrid, build fluxes | from window 2: `_cs_spectral_contract_diag!` for `win - 1` | as RG |
| GEOS CS ([`cubed_sphere_geos.jl`](cubed_sphere_geos.jl)) | `GEOSNativeReader` | read, pin, balance, `cm`, pick substeps | `verify_window!`, scale fluxes by `2 * steps` | `()` |

`driver_before_close_writer!` patches `steps_per_window_by_window` into the
header (LL does it when `LatLonDeferredBinaryWriter` opens). GEOS uses
`driver_after_write_window!` to carry `m_cur` forward when chaining.

## Window Contracts

LL `apply_poisson_balance!` takes a per-window step vector, a `LatLonContract`,
and a `SubstepSchedulePolicy`, including for fixed schedules. Writers publish
through the typed writer adapters and atomic staging lifecycle. Contract policy
values live on the concrete contract; `verify_window!`, `update_accumulator!`,
and `summarize_status!` implement the verification interface.

- Replay: `verify_window_continuity` (from `MetDrivers`) integrates the
  stored fluxes from `m_cur` over the palindrome substeps and compares with
  `m_next` in every cell; it errors when a cell's residual exceeds
  `replay_tolerance(FT)` times the summed magnitudes of its budget terms.
  Every topology checks the arrays as stored (in the output float type).
- Positivity: CS and LL compare the palindrome outflow `2 * (out_x + out_y +
  out_z)` with `positivity_cfl_limit * min(m, m_next)` (one shared check); RG
  compares each direction's outflow with `positivity_cfl_limit * m`, skips pole
  stub faces and
  errors on non-zero flux through them (`verify_boundary_stub_flux_rg`).
  Non-positive or non-finite mass yields `ratio = Inf`.
- `summarize_status!` errors and deletes `quarantine_path` when the worst ratio
  exceeds the limit and `require_substep_positivity` is set; otherwise it warns.
- Contract constructors require a finite, positive `replay_tol` and
  `positivity_cfl_limit` in `(0, 1]`.

## Flux-fidelity certificate

`ReadyWindow.fidelity` (a `FluxFidelityEvidenceBundle`: flux evidence, OMEGA
evidence, column-closure record) carries typed evidence beside the payload.
Wind-derived paths independently evaluate a deterministic set of canonical
faces from `u`, `v`, and layer pressure thickness before Poisson balance, and
build only the sampled values (lat-lon: the sampled columns; reduced Gaussian:
the sampled latitude lines; N320 moist line integrals: thickness at the
stencil cells). Native GEOS fluxes have no independent source: their
certificate is `not_applicable/native_fluxes_not_rederivable` and records the
balance correction only, while binary-only regrid and coarsen paths state
`not_applicable/source_payload_only`. Where source OMEGA is open, the writer
also compares it with `g cm/(A_cell Δt)` and gates `NRMSE < 1`, correlation
`> 0.5`, `0.5 < slope < 1.5` (`OmegaFidelityBaseline`). Every cubed-sphere column
balance adds its `ColumnClosureRecord` (`column_closure_*` keys, describing
the stored fluxes), which the reader decodes into `ColumnClosureHeader` and
rejects when incomplete, not covering `nwindow` or above its tolerance. Algebraic
sample failures throw before the window write; an aggregate OMEGA failure
throws while the file is still staged, so it is quarantined (the RG/CS
adapters then also abandon the inner writer's staging file).
`close_streaming_binary!` patches the aggregate flat header keys without
touching payload bytes.

For a sampled face, the independently reconstructed mass amount is

`F = (Δt_amount/g) ∫face [(u eλ + v eφ) ⋅ n] Δp dl`.

The gate records `e_abs = |F_candidate - F_reference|` and
`e_rel = e_abs / max(|F_candidate|, |F_reference|, F_floor)`, with
`F_floor = max(floatmin(FT), eps(FT) max(F_scale, 1 kg))`. It passes a sample
when `e_abs ≤ 64 eps(FT) max(F_scale, 1 kg)` or
`e_rel ≤ 256 eps(FT)`. The header records both tolerances, RMS/max errors, ULP
distance, coverage, the worst face/level/window, and the separate raw-to-
balanced correction. See the binary-format guide for the full schema.

## Adaptive Substep Schedules

- `[numerics]`: `substep_schedule = "adaptive_cfl"` (default) or `"fixed"`,
  `substep_cfl_target` (default `positivity_cfl_limit`), `min_steps_per_window`,
  `max_steps_per_window`.
- Defaults come from `Parameters` (`src/Parameters/NumericalDefaults.jl`):
  `positivity_cfl_limit` is `DEFAULT_POSITIVITY_CFL_LIMIT` on every path. Called
  directly, the MERRA-2 workflow defaults `substep_cfl_target` to
  `DEFAULT_SUBSTEP_CFL_TARGET`, N320 to `ERA5_N320_SUBSTEP_CFL_TARGET` and the
  coarsener to `COARSEN_SUBSTEP_CFL_TARGET`; the spectral and GEOS `process_day`
  methods default it to `positivity_cfl_limit`, and so does
  `regrid_ll_binary_to_cs`, which is adaptive unless `steps_per_window` is given.
- `next_substeps` returns `ceil(steps * ratio / substep_cfl_target)`, never
  below the current count and clamped to the bounds; fixed policies keep the count.
  A NaN/Inf ratio throws (also with `require_substep_positivity = false`; a
  `fixed` schedule never reads the ratio), and so does a request above
  `typemax(Int) ÷ 2` when `max_steps_per_window` is left at its unbounded
  default (`2 * steps` would overflow). GEOS's per-window "hypothetical split"
  log statistic does not throw.
- On a change the workflow rescales (`rescale_substep_amounts!`, spectral
  paths and the LL → CS regrid), re-prepares from native fluxes (GEOS), or rebuilds fluxes from winds
  (N320, MERRA-2), then repeats balance, `cm`, and the positivity check.
  GEOS, N320, MERRA-2 and the LL → CS regrid stop after `ADAPTIVE_SUBSTEP_MAX_REFINEMENTS`
  refinements.
- Counts are written to `steps_per_window_by_window`
  (`set_streaming_steps_per_window_schedule!`,
  `set_transport_header_steps_per_window_schedule!`). CS binaries always carry
  `runtime_substep_contract = "binary_schedule"`.

## File Map

- [`core.jl`](core.jl) — `HEADER_SIZE`, `exact_steps_per_window`, provenance and fingerprints, LL `build_v4_header`, payload sizing, `existing_output_schema_matches`
- [`flux_fidelity.jl`](flux_fidelity.jl) — typed evidence/statuses, deterministic face maps, independent sampled evaluators, algebraic and OMEGA accumulators, final header certification
- [`window_contracts.jl`](window_contracts.jl) — abstract contract/workspace/writer types, mass-basis symbols, `SubstepSchedulePolicy`, `ReadyWindow`, `PreverifiedWindow`, `PreprocessorRunCache`, trait generics
- [`latlon_workspaces.jl`](latlon_workspaces.jl) — LL spectral workspaces and `WindowStorage`, qv loading (bilinear from the thermo points to the cell centres, `LatLonBilinearWeights`), mass fix, dry basis, merge, `close_pole_faces!`, `process_window!`
- [`latlon_contracts.jl`](latlon_contracts.jl) — `next_day_merged_fields`, `apply_poisson_balance!`, LL window serializer, LL positivity gate, `LatLonContract`
- [`cubed_sphere_contracts.jl`](cubed_sphere_contracts.jl) — CS mass-tendency and delta helpers, CS replay and positivity gates, `CubedSphereContract`
- [`reduced_gaussian_contracts.jl`](reduced_gaussian_contracts.jl) — RG positivity and stub-flux gates, `ReducedGaussianContract`
- [`writer_adapters.jl`](writer_adapters.jl) — typed LL/RG/CS writers, promote/quarantine, staged-size validation
- [`driver.jl`](driver.jl) — `UnifiedPreprocessorDay`, default hooks, `run_unified_preprocessor_day!`
- [`topology_dispatch.jl`](topology_dispatch.jl) — `preprocessor_pair_supported` fallback, `ensure_preprocessor_pair_supported`, fallback `process_day`
- [`latlon_spectral.jl`](latlon_spectral.jl) — ERA5 spectral to LL `process_day`, `LatLonDeferredBinaryWriter`, LL hooks
- [`cubed_sphere_spectral.jl`](cubed_sphere_spectral.jl) — ERA5 spectral to CS `process_day` through an LL staging grid, CS hooks
- [`cubed_sphere_regrid.jl`](cubed_sphere_regrid.jl) — `regrid_transport_binary`, `regrid_ll_binary_to_cs` (output substeps per window from the CS Courant number unless `steps_per_window` fixes them; winds recovered with each source window's own count; refuses ERA5 spectral LL sources without `pole_row_zonal_flux = "open"`; own loop)
- GEOS-IT/FP to CS, in include order:
  [`geos_cs_mass_helpers.jl`](geos_cs_mass_helpers.jl) — DELP ↔ air mass, surface pressure, pressure-fixer mass evolution, residual smoothing;
  [`geos_cs_omega.jl`](geos_cs_omega.jl) — OMEGA-consistent `cm` target (PCHIP time interpolation, regularization, reconstruction);
  [`geos_cs_resolution.jl`](geos_cs_resolution.jl) — global dry-mass pin, panel-convention check, identity and block-coarsening strategies, native → target payloads (cell sums and area-weighted means through `Regridding.coarsen_sum!` / `coarsen_area_weighted!`, face sums through `Regridding.coarsen_xface_sum!` / `coarsen_yface_sum!`);
  [`geos_cs_window.jl`](geos_cs_window.jl) — window workspace, per-window preparation with the `cm` closures and substep selection, ingest/drain/advance;
  [`cubed_sphere_geos.jl`](cubed_sphere_geos.jl) — driver context, GEOS hooks and `process_day`
- [`cubed_sphere_coarsen.jl`](cubed_sphere_coarsen.jl) — experimental nested CS binary coarsener (`coarsen_nested_cs_transport_binary`) using the nested face sums in `Regridding/cs_nesting.jl`, shared with the GEOS block coarsener
- [`era5_n320_regrid.jl`](era5_n320_regrid.jl) — `process_era5_n320_to_cs_day` and its `process_day` adapter; TM5 convection, surface, `dkg` payloads
- [`merra2_latlon_regrid.jl`](merra2_latlon_regrid.jl) — `process_merra2_to_cs_day` and adapter; 3-hour blocks split into hourly windows when `dt_met_seconds = 3600`
- [`entrypoint.jl`](entrypoint.jl) — TOML entry, `[numerics]`/`[vertical]`/`[mass_fix]` parsing, serial or threaded day loop

## Common Tasks

- Adding a topology: subtype `AbstractWindowWorkspace{G, FT}`,
  `AbstractWindowContract{G, FT}` (validate policy in the constructor), and
  `AbstractBinaryWriter{G, FT, Basis}` (`write_window!` for
  `ReadyWindow{G, FT}`, `close_streaming_binary!`, `validate_staged_binary!`;
  the default promote/quarantine use the `path`, `final_path`, `promoted`
  fields). Add a context type, the hooks, and `process_day`.
- Moving an own-loop writer onto the driver: follow `cubed_sphere_geos.jl`
  (reader-backed) or `cubed_sphere_spectral.jl` (one-window lag).
- Debugging a replay failure: the error names the window and cell
  (`(panel, i, j, k)` on CS). A failure only in the last window points at the
  next-day endpoint. `[numerics] write_replay_check = false` keeps a binary
  for inspection.
- `ATMOS_OMEGA_TIMING=1` logs GEOS OMEGA reconstruction timings.

## Invariants

- Stored fluxes are per-substep amounts (`flux_kind = :substep_mass_amount`)
  except GEOS (`:full_window_mass_amount`, divided by `2 * steps` at runtime);
  the coarsener accepts only per-substep inputs.
- `dm` is formed after the gates that need the absolute endpoint
  (`convert_cs_mass_target_to_delta!`, `_fill_cs_mass_delta_payload!`).
- GEOS `process_day` requires `mass_basis = :dry` and a `geos_native` panel
  convention; OMEGA closures require `[mass_fix].enable = true`.
- The spectral topologies share the file name from `output_binary_path`
  (`era5_transport_<date>_merged<min_dp>Pa_<float>.bin`); only the LL path
  skips an existing file whose size, sections, and generation keys match.

## Tests

In [`../../../test/core/`](../../../test/core/): `test_preprocessor_unified_driver.jl`,
`test_preprocessor_writer_adapters.jl`, `test_preprocessor_substep_schedule.jl`,
`test_{ll,rg,cs}_preprocessor_contract.jl`, `test_replay_consistency.jl`,
`test_{ll_spectral,rg_preprocessor,cs_spectral}_unified_driver.jl`,
`test_geos_cs_passthrough.jl`, `test_geos_omega_regularization.jl`,
`test_ll_to_cs_regrid_script.jl`, `test_era5_n320_vertical_merge.jl`,
`test_cs_binary_coarsener.jl`. The full N320 and
MERRA-2 day writers run only in [`../../../test/golden/`](../../../test/golden/).
