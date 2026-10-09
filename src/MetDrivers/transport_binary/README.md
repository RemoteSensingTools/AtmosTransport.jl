# Transport Binary

The version-4 transport-binary format: header schema, contract validation,
readers, writers, the runtime driver, and the inspector.

A transport binary holds `nwindow` met windows (typically one day) of air mass,
mass fluxes, and optional physics forcing for one horizontal topology. These
files are included into `MetDrivers` by [`../TransportBinary.jl`](../TransportBinary.jl)
(header through inspector) and [`../MetDrivers.jl`](../MetDrivers.jl) (window, driver,
cubed-sphere driver). The parent overview is [`../README.md`](../README.md).

## On-Disk Layout

- JSON header, null-terminated, zero-padded to `header_bytes`
- `n_geometry_elems` floats (current writers emit 0)
- `nwindow` blocks of `elems_per_window` floats of `float_type`. Within a block,
  sections follow `payload_sections` order. Cubed-sphere sections store the six
  panels back to back.

## Entry Points

- Header types: [`header.jl`](header.jl) defines `TransportBinaryHeader{G}`,
  `LatLonBinaryGeometry`, `ReducedGaussianBinaryGeometry`,
  `CubedSphereBinaryGeometry`, `TRANSPORT_BINARY_FORMAT_VERSION`, `grid_type`,
  `horizontal_topology`, and `binary_geometry`
- Contract: [`contract.jl`](contract.jl) defines `validate_transport_contract!`,
  `TransportBinaryContract`, `canonical_window_constant_contract`, and
  `validate_cs_writer_contract!`
- Reader: [`reader.jl`](reader.jl) defines `TransportBinaryReader(path; FT = Float32)`,
  which memory-maps the payload. Lat-lon and reduced-Gaussian methods
  include `load_window!` (returns `(m, ps, fluxes)`), `load_flux_delta_window!`,
  `load_qv_pair_window!`, `load_tm5_convection_window!`,
  `load_surface_window!`, `load_grid`, and the `has_*` predicates
- Cubed-sphere reader: [`cubed_sphere_reader.jl`](cubed_sphere_reader.jl) adds
  `load_window!`, which returns a NamedTuple of panel `NTuple`s. It also adds
  `load_grid(reader; Hp)`, `mesh_convention`, and `mesh_definition`
- Writers: [`writer.jl`](writer.jl) `write_transport_binary(path, grid, windows; ...)`
  (lat-lon and reduced Gaussian); [`streaming_writer.jl`](streaming_writer.jl)
  `open_streaming_transport_binary` (reduced Gaussian only),
  `write_streaming_window!`, `close_streaming_transport_binary!`, and
  `set_streaming_steps_per_window_schedule!`; [`cubed_sphere.jl`](cubed_sphere.jl)
  `open_streaming_cs_transport_binary` and `write_streaming_cs_window!`
- Runtime driver: [`driver.jl`](driver.jl) `TransportBinaryDriver(path; FT = Float64,
  arch, Hp, validate_windows, validate_replay, max_rel_cm)`, `load_transport_window`,
  `driver_grid`, `air_mass_basis`, `interpolate_fluxes!`, `expected_air_mass!`,
  `interpolate_qv!`, `uses_binary_substep_contract`, `release_payload!`
- Decoded window: [`window.jl`](window.jl) `TransportWindow` and the
  `StructuredFluxDeltas` / `FaceIndexedFluxDeltas` / `CubedSphereFluxDeltas` types
- Inspector: [`inspect.jl`](inspect.jl) `binary_capabilities(reader)` and
  `inspect_binary(path; io)`. The CLI wrapper is
  [`../../../scripts/diagnostics/inspect_transport_binary.jl`](../../../scripts/diagnostics/inspect_transport_binary.jl)

## File Map

- [`header.jl`](header.jl) — geometry types, `TransportBinaryHeader`, `_parse_transport_header`, `_transport_common_header`, section-name constants
- [`contract.jl`](contract.jl) — incremental JSON header read, contract struct, structural-key merge guard, `_validate_transport_layout!`, `validate_transport_contract!`
- [`reader.jl`](reader.jl) — mmap reader, accessors, lat-lon / reduced-Gaussian grid and window loaders
- [`cubed_sphere_reader.jl`](cubed_sphere_reader.jl) — panel-native window loading, mesh convention/definition, CS `load_grid`
- [`payload_sections.jl`](payload_sections.jl) — LL/RG section element counts, window-field accessors, writer-side window validation
- [`writer.jl`](writer.jl) — eager atomic `write_transport_binary` and payload packers
- [`streaming_writer.jl`](streaming_writer.jl) — `StreamingTransportBinaryWriter`, reduced-Gaussian opener, per-window schedule rewrite, finalisation
- [`cubed_sphere.jl`](cubed_sphere.jl) — `_cs_section_elements` (shared by reader and writer), CS panel packing, CS streaming opener and window writer
- [`inspect.jl`](inspect.jl) — capability NamedTuple and printed report
- [`window.jl`](window.jl) — `TransportWindow`, replay-delta types, `Adapt` rules
- [`driver.jl`](driver.jl) — `TransportBinaryDriver`, runtime-semantics checks, `cm` sanity and LL/RG replay gates, flux interpolation, LL/RG `load_transport_window`
- [`cubed_sphere_driver.jl`](cubed_sphere_driver.jl) — CS replay gate, halo padding, CS `load_transport_window`

## Common Tasks

- Inspect a binary before debugging a run:
  `julia --project=. scripts/diagnostics/inspect_transport_binary.jl <path.bin>`
  prints the header, capability rows, and whether `TransportBinaryDriver`
  construction succeeds.
- To add a metadata-only header key, pass it through the writer's
  `extra_header` and read it from `reader.header.raw_header`, as
  `uses_binary_substep_contract` does. Runtime-read CS keys get defaults in
  `open_streaming_cs_transport_binary` and are listed in `_CS_WRITER_CONTRACT_KEYS`.
- To add a typed header field, add it to `TransportBinaryHeader`, parse it in
  `_parse_transport_header`, emit it in `_transport_common_header`, and validate
  it in `validate_transport_contract!`. If it defines the layout, add it to
  `_TRANSPORT_STRUCTURAL_HEADER_KEYS`. The ERA5 lat-lon preprocessor builds its
  own header dict in
  [`../../Preprocessing/transport_binary/core.jl`](../../Preprocessing/transport_binary/core.jl),
  so update that too.
- To add a payload section:
  1. Add its element count to `_transport_structured_section_elements`,
     `_transport_faceindexed_section_elements`, or `_cs_section_elements` (CS
     also needs `_cs_section_panel_shape`).
  2. Add the writer accessor: `_transport_window_field` and
     `_transport_push_optional_sections!`, or, for CS, `_cs_window_section`, an
     `include_*` keyword, and `_validate_streaming_cs_window`.
  3. Add pairing rules to `_validate_transport_layout!`, plus a loader, a `has_*`
     predicate, and a `binary_capabilities` row.
- To convert stored fluxes to rates, use `flux_application_seconds` or
  `flux_storage_substep_scale` from [`../AbstractMetDriver.jl`](../AbstractMetDriver.jl).
  Do not hand-roll `dt / (2 * steps)`.

## Invariants

- Only `magic = "MFLX"` and `format_version = 4` load. Older files are
  regenerated, not defaulted (`validate_transport_contract!`).
- The contract fields must be present and known. `steps_per_window` must equal
  `maximum(steps_per_window_by_window)`, and `time_step_schedule` must say
  `"constant"` or `"per_window"` to match. The per-window Poisson target scale
  is `1 / (2 * steps)` for `:substep_mass_amount` and `1` for
  `:full_window_mass_amount`.
- `flux_kind = :full_window_mass_amount` and
  `runtime_substep_contract = "binary_schedule"` are accepted for cubed sphere
  only.
- Section sets are all-or-none: `qv_start`/`qv_end` (single-field `qv` is
  rejected), `entu`/`detu`/`entd`/`detd`, `pblh`/`ustar`/`pbl_hflux`/`t2m`, and
  `vdiff_u`/`vdiff_v`/`vdiff_t`/`vdiff_qv`. `pbl_eflux` needs the PBL set,
  `cmfmc_cloud_base` needs `cmfmc`, `:kz` is rejected, and `:dkg` requires
  `mass_basis = dry`.
- `elems_per_window` must match the section table. The file size must equal
  `header_bytes + (n_geometry_elems + nwindow * elems_per_window) * float_bytes`.
- `extra_header` cannot change keys in `_TRANSPORT_STRUCTURAL_HEADER_KEYS`,
  such as grid sizes, coordinates, `A_ifc`/`B_ifc`, `mass_basis`, sections, and
  `planet_radius_m`.
- If `planet_radius_m` is absent, it reads as `EARTH_RADIUS`. If present, it must
  be finite and positive, and `load_grid` builds the mesh on it.
- `p_half[k] = A_ifc[k] + B_ifc[k] * ps`, with `k = 1` at the top of the atmosphere
  and `nlevel + 1` interfaces.
- `write_transport_binary` defaults to `mass_basis = :dry`. Both streaming openers
  default to `:moist`, and the in-tree preprocessors pass the basis explicitly.
- Writers publish atomically. The eager writer renames `<path>.tmp`; streaming
  writers rename a temp file in the destination directory after rewriting the
  header at close, and refuse to publish an incomplete stream.
- The driver requires `:substep_mass_amount` for LL/RG. CS requires
  `flux_sampling = :window_constant` and `humidity_sampling = :none`
  (`_validate_runtime_semantics`).

## Related Docs And Tests

- Format reference: [`../../../docs/src/concepts/binary_format.md`](../../../docs/src/concepts/binary_format.md);
  producers: [`../../Preprocessing/`](../../Preprocessing/); continuity checks:
  [`../ReplayContinuity.jl`](../ReplayContinuity.jl)
- Tests under [`../../../test/core/`](../../../test/core/):
  `test_binary_inspector.jl`, `test_persistence_hardening.jl` (streaming fail-closed,
  structural keys, layout), `test_binary_planet_radius.jl`,
  `test_cs_optional_2d_sections.jl`, `test_cs_writer_contract_guard.jl`,
  `test_precomputed_dkg_binary_payload.jl`, `test_replay_consistency.jl`,
  `test_input_resource_lifetime.jl` (driver closes its reader on failure)
