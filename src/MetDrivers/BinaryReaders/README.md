# BinaryReaders

Memory-mapped readers for format versions 4 and 5 and geometry-dispatched
LL/RG/CS drivers.
Included as `MetDrivers.BinaryReaders`, with application APIs re-exported by
MetDrivers. No on-disk format definitions or writers are duplicated here.

## Depends on

BinaryFormat; parent driver, forcing and replay interfaces; Architectures,
Parameters, Grids and State. GRIB reading belongs to Preprocessing.

## Used by

Models, Preprocessing, Diagnostics, Adjoints, Footprint and inspection tools.

## Shared helpers owned here

`TransportBinaryReader`, `TransportBinaryDriver`, `load_grid`, `load_window!`,
`load_transport_window`, `load_transport_window!`, capability predicates and
flux interpolation/copy functions retain their MetDrivers facade names.
`section_view(reader, window, section)` returns a zero-copy payload view.
`infer_surface_pressure_basis` classifies keyless binaries once, using
Float64 column sums and `Parameters.SURFACE_PRESSURE_BASIS_TOLERANCE`.
`raw_window_staging`, `read_window_payload!`, `unpack_window_payload!` and
`same_window_layout` provide device staging and host refill.

`has_dtrain(reader_or_capabilities)` and `has_dkg(reader)` own the DTRAIN/Dkg
payload-membership predicates shared by Models and the CS reader/driver. The
predicates retain the exact section-membership checks; Models retains its
existing topology fallback methods.

## Forward<->adjoint mirrors

No adjoint kernels. Forward and adjoint runners load the same forcing windows.
Shape/count mirrors with writers are owned by BinaryFormat.

## File map

- `BinaryReaders.jl`: submodule assembly, imports and interface.
- `reader.jl`: mmap ownership, accessors, legacy pressure inference, LL/RG loaders.
- `cubed_sphere_reader.jl`: panel allocation/loading and CS mesh reconstruction.
- `driver.jl`: common driver, LL/RG replay reduction, validation and interpolation.
- `cubed_sphere_driver.jl`: CS replay, halo padding, staging and refill.
- `inspect.jl`: mmap-backed capabilities, dry-mass target and inspector methods.

## Invariants and tests

LL/RG `load_window!` returns `(m, ps, fluxes)`; CS returns a NamedTuple of
panel tuples. Disk-to-load conversion remains elementwise `copyto!`.
`binary_mass_basis` returns a Symbol and `surface_pressure_basis` a typed tag.
Full-window fluxes use the parent's `flux_storage_substep_scale` at replay
and runtime. The load-time replay gate is opt-in (`validate_replay=true`).
The reader decodes the version-5 flux-construction identity into
`header.flux_construction`; `binary_capabilities` reports it with
`format_version`, and `inspect_binary` prints both.
Tests: reader round trips, binary inspector, format versions 4 and 5, CS
staging and optional sections, replay consistency, validate-replay config, and
cubed-sphere runtime.
