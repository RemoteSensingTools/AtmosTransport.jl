# MetDrivers

Meteorological driver and transport-window infrastructure.

This folder is the bridge between external data and the runtime model.
It owns the abstract driver contract, binary readers, driver adapters,
replay diagnostics, convection-forcing containers, and the ERA5- and
cubed-sphere-specific data paths.

## Entry Points

- Driver contract:
  [`AbstractMetDriver.jl`](AbstractMetDriver.jl)
  defines `AbstractMetDriver`, capability traits, and `current_time`
- Module assembly and exports:
  [`MetDrivers.jl`](MetDrivers.jl)
- Version-4 transport-binary reader:
  [`TransportBinary.jl`](TransportBinary.jl)
- Geometry-dispatched runtime driver and forcing-window type:
  [`transport_binary/driver.jl`](transport_binary/driver.jl)
- Convection forcing container:
  [`ConvectionForcing.jl`](ConvectionForcing.jl)
- ERA5-specific stack:
  [`ERA5/README.md`](ERA5/README.md)

The public `has_surface` and `has_vdiff_fields` functions also accept
preprocessing `AbstractMetSettings`. `Preprocessing` imports these same
generics before defining its settings methods, so qualified and top-level
queries agree. Reader methods inspect declared binary payloads; settings
methods report configured source support without reading meteorological files.

## What This Folder Owns

- The abstract met-driver capability contract used by `DrivenSimulation`
- Binary/window readers and adapters that return runtime-friendly
  transport windows
- Replay-continuity diagnostics for loaded windows
- Driver-level capability traits such as `supports_diffusion` and
  `supports_convection`
- Simulation time plumbing via `current_time(meteo)`

## File Map

- [`MetDrivers.jl`](MetDrivers.jl) — submodule assembly and public exports
- [`AbstractMetDriver.jl`](AbstractMetDriver.jl) — required driver
  methods, traits, and the `current_time` contract
- [`ConvectionForcing.jl`](ConvectionForcing.jl) — convection forcing
  storage, adaptation, and copy helpers
- [`SurfaceForcing.jl`](SurfaceForcing.jl) — surface forcing payload
  containers and copy helpers for runtime physics fields
- [`ReplayContinuity.jl`](ReplayContinuity.jl) — runtime replay and
  continuity diagnostics for loaded transport windows
- [`TransportBinary.jl`](TransportBinary.jl) — thin includer for the
  transport-binary format; pulls in the concept files under
  [`transport_binary/`](transport_binary/) in dependency order and re-exports
  the public names. The implementation lives in:
  - `transport_binary/header.jl` — header schema, schedule/sampling parsers,
    typed write-time certificates (`FluxFidelityHeader`, `OmegaFidelityHeader`)
    and column-closure record (`ColumnClosureHeader`) with their load-time
    consistency checks, header `summary`/`show`, `_parse_transport_header`,
    `_transport_common_header`
  - `transport_binary/contract.jl` — `TransportBinaryContract`,
    `canonical_window_constant_contract`, `validate_cs_writer_contract!`,
    `validate_transport_contract!`
  - `transport_binary/reader.jl` — `TransportBinaryReader` struct, accessors,
    constructor, common accessors, and structured/face-indexed loaders
  - `transport_binary/cubed_sphere_reader.jl` — panel-native `load_grid` and
    window-loading specializations for the same reader type
  - `transport_binary/payload_sections.jl` — per-section element counts,
    window-field accessors, optional-section validators
  - `transport_binary/writer.jl` — eager `write_transport_binary` + packers;
    the file is published with `AtomicFiles.atomic_publish`
  - `transport_binary/streaming_writer.jl` — `StreamingTransportBinaryWriter`
    (structured / reduced-Gaussian); windows go to an
    `AtomicFiles.open_atomic_staging` file next to the target, which
    `close_streaming_transport_binary!` renames onto it only when complete
  - `transport_binary/cubed_sphere.jl` — cubed-sphere streaming writer (same
    staging file and close)
  - `transport_binary/inspect.jl` — `binary_capabilities`, `inspect_binary`,
    and capability reporting
- [`transport_binary/window.jl`](transport_binary/window.jl) — decoded forcing
  windows and topology-specific replay deltas.
- [`transport_binary/driver.jl`](transport_binary/driver.jl) — common driver,
  validation, replay checks, and structured/face-indexed loading
- [`transport_binary/cubed_sphere_driver.jl`](transport_binary/cubed_sphere_driver.jl)
  — panel-native validation and loading specializations for those common types;
  `load_transport_window!` (refill a host window in place), `same_window_layout`
  (whether another binary's windows fit the same storage), and the raw window
  staging of device runs (`raw_window_staging`, binaries whose on-disk element
  type is the run's: concurrent reads of a window's contiguous payload, then
  unpacking on the device. `RawWindowStaging` for discrete device memory reads
  into page-locked host memory and uploads section by section through a
  section-sized device buffer; `MappedWindowStaging` for unified memory
  (Metal, `Architectures.host_memory_model`) reads into a shared device buffer
  that the unpack kernels read in place, with no upload)
- [`ERA5/`](ERA5/README.md) — ERA5-specific readers, geometry, closure,
  and dry-flux building

## Common Tasks

- Adding a new driver:
  start with [`AbstractMetDriver.jl`](AbstractMetDriver.jl), then decide
  whether the path is reader-first, driver-first, or both
- Debugging time-dependent operator inputs:
  check whether `current_time(meteo)` is coming from the sim or from the
  fallback driver stub
- Debugging a window-driven runtime issue:
  read `load_transport_window`, `driver_grid`, and the driver capability
  traits before touching `DrivenSimulation`
- Extending topology support:
  keep binary-reader semantics separate from runtime-driver semantics
  rather than overloading one path to do both

## Cross-Dependencies

- [`../Models/README.md`](../Models/README.md) consumes these drivers in
  `DrivenSimulation`
- [`../State/README.md`](../State/README.md) provides the state and
  flux-storage types the windows populate
- [`../Grids/README.md`](../Grids/README.md) provides mesh construction
  for `load_grid`
- [`../Operators/Convection/README.md`](../Operators/Convection/README.md)
  consumes `ConvectionForcing`

## Related Docs And Tests

- Binary/runtime walkthrough:
  [`../../docs/30_BINARY_AND_DRIVERS.md`](../../docs/30_BINARY_AND_DRIVERS.md)
- Runtime flow:
  [`../../docs/20_RUNTIME_FLOW.md`](../../docs/20_RUNTIME_FLOW.md)
- Tests:
  - [`../../test/real_data/test_transport_binary_reader.jl`](../../test/real_data/test_transport_binary_reader.jl)
  - [`../../test/core/test_run_transport_binary_recipe.jl`](../../test/core/test_run_transport_binary_recipe.jl)
  - [`../../test/core/test_binary_path_expander.jl`](../../test/core/test_binary_path_expander.jl)
  - [`../../test/core/test_atomic_files.jl`](../../test/core/test_atomic_files.jl) and
    [`../../test/core/test_persistence_hardening.jl`](../../test/core/test_persistence_hardening.jl)
    (writers publish only complete files, with mode `0o666 & ~umask`)
  - [`../../test/real_data/test_dry_flux_interface.jl`](../../test/real_data/test_dry_flux_interface.jl)
  - [`../../test/core/test_multiday_source_clock.jl`](../../test/core/test_multiday_source_clock.jl)
  - [`../../test/core/test_cubed_sphere_runtime.jl`](../../test/core/test_cubed_sphere_runtime.jl)
