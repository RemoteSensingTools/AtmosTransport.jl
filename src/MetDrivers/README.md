# MetDrivers

Meteorological driver abstraction, forcing containers, and replay continuity.
The facade re-exports the binary API from two submodules. Existing runtime
entry points remain available through `AtmosTransport.MetDrivers`.

## Depends on

Architectures, Parameters, Grids, State, and AtomicFiles. BinaryReaders uses
BinaryFormat and the parent driver and replay interfaces. BinaryFormat uses
parent capability generics; neither child loads Operators, Models, or Preprocessing.

## Used by

Models, Preprocessing, Adjoints, Footprint, Diagnostics, and binary inspection tools.

## Shared helpers owned here

- `flux_application_seconds`, `flux_storage_substep_scale`: stored-flux normalization.
- `ReplayScratch`, `structured_replay_layout`, `faceindexed_replay_layout`,
  `recompute_cm_from_dm_target!`, `verify_window_continuity`,
  `verify_window_continuity_rg`, `verify_window_continuity_cs`: explicit-mass
  closure and replay diagnostics. `run_replay_gate` reduces per-window diagnostics;
  readers supply their historical initial diagnostic and failure policy.
- [BinaryFormat](BinaryFormat/README.md) owns headers, sections, windows, and writers.
- [BinaryReaders](BinaryReaders/README.md) owns readers and runtime adapters.

Surface availability is queried with `has_surface`; Preprocessing extends the
same generic for settings. Header coefficients are `reader.header.A_ifc` and
`reader.header.B_ifc`. `binary_mass_basis` returns a Symbol; State's `mass_basis`
returns a basis instance.

`has_dtrain(reader_or_capabilities)` and `has_dkg(reader)` own the DTRAIN/Dkg
payload-membership predicates shared by Models and the CS reader/driver. The
predicates retain the exact section-membership checks; Models retains its
existing topology fallback methods.

## Forward<->adjoint mirrors

No adjoint kernels live here. Forward and adjoint runners consume the same
window and flux-storage contract. Writer/reader payload order is mirrored
through BinaryFormat's section table.

## File map

- `MetDrivers.jl`: module assembly and binary API facade.
- `AbstractMetDriver.jl`: abstract driver, capabilities, time and normalization.
- `ConvectionForcing.jl`: convection forcing allocation, adaptation, and copying.
- `SurfaceForcing.jl`: raw PBL surface forcing and copy operations.
- `ReplayContinuity.jl`: divergence, explicit-dm closure, per-cell replay and gates.
- `BinaryFormat/`: metadata, contracts, section shapes, decoded windows, writers, reports.
- `BinaryReaders/`: mmap readers, mesh reconstruction, runtime drivers and staging.

ERA5 GRIB geometry is preprocessing-only and lives in
`../Preprocessing/Sources/ERA5/era5_grib_geometry.jl`. Its three public geometry
names are now owned by Preprocessing and remain bound on the package facade.
The test-only ERA5 pressure-fixer closure has been removed.

## Tests

`test_binary_inspector.jl`, `test_persistence_hardening.jl`,
`test_cs_optional_2d_sections.jl`, `test_cs_window_staging.jl`,
`test_replay_consistency.jl`, `test_validate_replay_config.jl`,
`test_precomputed_dkg_binary_payload.jl`, and `test_transport_binary_reader.jl`.
See `docs/src/concepts/binary_format.md` for the on-disk contract.
