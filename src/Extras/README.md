# Extras

Non-core modules retained in AtmosTransport for a future package split. They
have no tests. Core modules must never import from Extras. Extras use only
public core names; each subfolder is meant to become its own package, with
its own `Project.toml`, in `lib/` or a separate repository.

## File Map

- [`Visualization/`](Visualization/README.md): snapshot views, rasterization
  and plotting entry points, module `AtmosTransport.Visualization`.
- [`Analysis/`](Analysis/README.md): bands, decompositions, skill statistics,
  column profiles and GEOS-Chem archive reading, module `AtmosTransport.Analysis`.
  It re-exports the conservation API from core
  [`Diagnostics/`](../Diagnostics/README.md).
- [`Downloads/`](Downloads/README.md): raw data acquisition, module
  `AtmosTransport.DataDownloads`.

The Makie extension stays in
[`ext/AtmosTransportMakieExt.jl`](../../ext/AtmosTransportMakieExt.jl), Julia's
extension location. It remains non-core and has no tests.

## Dependencies And Consumers

Depends on public core interfaces and external Julia packages. Used by scripts
and interactive workflows through the existing module names and top-level
plotting exports. Shared helpers belong to core, never the reverse.
There are no forward/adjoint mirrors here.
