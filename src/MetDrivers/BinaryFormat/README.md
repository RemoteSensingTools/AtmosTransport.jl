# BinaryFormat

Version-4 transport metadata, payload layout, decoded windows, writers and
inspection reports. Included as `MetDrivers.BinaryFormat`; application APIs
are re-exported by MetDrivers.

## Depends on

Architectures, ConfigChecks, AtomicFiles, Parameters, Grids, State, and parent capability
generics. Reader types and file ownership belong to BinaryReaders.

## Used by

BinaryReaders, Preprocessing, Models, Diagnostics, and the package facade.

## Shared helpers owned here

`section_shape(header, section)` returns an array shape (one panel for CS).
`section_elements(header, section)` includes all panels. The LL overload
`section_elements(LatLonBinaryGeometry, Nx, Ny, Nz, section)` supports header
construction. One section-to-shape-kind table and topology allow-lists serve
validation, allocation and serialization.

`poisson_balance_target(kind, schedule; steps_per_window=maximum(schedule))`
returns Float64 `(scale, semantics, by_window)` metadata for a validated
schedule. Physics tendency accumulation and rounding policies stay at callers.
`write_padded_header!` writes JSON plus NUL padding at the current position;
validation, seeking, flushing and publication remain caller responsibilities.
Its string overload preserves already serialized LL header bytes.
`dry_mass_pin_enabled` and `is_certificate_key` are public metadata helpers.
`fidelity_tag` and `FLUX_FIDELITY_SAMPLE_SEED` keep writer/reader certificate
spellings and deterministic sampling identical. Public `parse_*_fidelity_header`
and `parse_column_closure_header` methods decode certificate blocks.

The public submodule interface also supplies `parse_transport_header`,
`read_transport_header_json`, `transport_disk_float_type`,
`transport_is_structured`, `transport_is_faceindexed`,
`has_variable_step_schedule`, `steps_per_window_summary`,
`transport_geometry_summary`, `transport_qv_summary`,
`transport_semantics_summary`, `transport_surface_pressure_basis_label`,
`required_advection_sections`, `print_capability_rows`, and section-group constants.
BinaryReaders extends `binary_capabilities`, `binary_dry_mass_target` and
`inspect_binary` with concrete mmap-reader methods.

## Forward<->adjoint mirrors

No adjoint kernels. Writers and readers share section dimensions and ordering;
forward and adjoint forcing use the same `TransportWindow`.

## File map

- `BinaryFormat.jl`: submodule imports, assembly and interface.
- `header.jl`: geometry/header types, certificate parsing, sampling, common header.
- `contract.jl`: contract validation, Poisson metadata, padded-header writes.
- `payload_sections.jl`: section shape table, element counts, accessors and validation.
- `window.jl`: decoded `TransportWindow`, replay deltas and Adapt rules.
- `writer.jl`: eager LL/RG writer and flat payload packing.
- `streaming_writer.jl`: RG stream, scheduling, atomic completion and abandonment.
- `cubed_sphere.jl`: panel packing, CS stream and window validation.
- `inspect.jl`: inspection generics, dry-mass pin validation and report formatting.

## Invariants and tests

Header keys, values, section order, per-element conversion and NUL padding are
preserved. LL Preprocessing's `build_v4_header` remains a separate builder:
unifying its key set is a stated-delta follow-up. Writers publish only complete
atomic staging files. LL/RG reject CS-only sections; the current format is v4.
Tests: binary inspector, persistence hardening, atomic files, binary planet
radius, optional CS sections, and preprocessing writer/driver contracts.

ConfigChecks.parse_choice owns fidelity, surface-pressure basis and disk-type
choice parsing. Disk types remain case sensitive; other tags preserve their
previous case-insensitive String/Symbol policy. Header layout is unchanged.
