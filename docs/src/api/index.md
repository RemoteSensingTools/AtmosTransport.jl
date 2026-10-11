# API reference

This section is the **auto-generated API reference** for the public
surface of every AtmosTransport.jl submodule. It is built directly
from the source docstrings; if a function is exported but the
docstring is sparse, that's both a quick-fix opportunity and a
signal that the function is internal-leaning.

Start with [Curated public API](@ref) if you want the short list of names a
scientist or new Julia user should learn first.

The submodules are listed in load order — earlier modules don't
depend on later ones:

| Module | What's in it |
|---|---|
| [Architectures](@ref Architectures-API) | `CPU`, `GPU`, backend selection. |
| [Parameters](@ref Parameters-API) | `PlanetParameters`, `earth_parameters`. |
| [Grids](@ref Grids-API) | `LatLonMesh`, `CubedSphereMesh`, `ReducedGaussianMesh`, `AtmosGrid`. |
| [State](@ref State-API) | `CellState`, `CubedSphereState`, basis tags, tracer accessors. |
| [MetDrivers](@ref MetDrivers-API) | `TransportBinaryDriver`, `inspect_binary`, the binary I/O. |
| [Operators](@ref Operators-API) | Advection / Convection / Diffusion / SurfaceFlux. |
| [Models](@ref Models-API) | `TransportModel`, `DrivenSimulation`, the runtime stepper. |
| [Analysis](@ref Analysis-API) | Post-run diagnostics numerics: snapshot series, equal-area band means, seasonal/interannual decompositions, skill statistics, column profiles, the budget closure of a run, the GEOS-Chem CATRINE archive reader. |
| [Preprocessing](@ref Preprocessing-API) | `process_day` and the source/target dispatch. |
| [DataDownloads](@ref DataDownloads-API) | Verified source/protocol download pipeline. |
| [Regridding](@ref Regridding-API) | Conservative offline regridding and persistence. |
| [GriddedIO](@ref GriddedIO-API) | Conventions of gridded input files: coordinates, fill values, source meshes, time axes, staging, flux units, and the `cs_native` and lat-lon flux-file schemas. |
| [Output and visualization](@ref Output-and-visualization-API) | Snapshot output, field views, and plotting. |
| [Adjoints and checkpointing](@ref Adjoints-and-checkpointing-API) | Footprints, 4D-Var, tape storage, and schedules. |
| [Infrastructure](@ref Infrastructure-API) | Quantity traits, opt-in section timing, config checks, log levels, atomic file publication. |

### A note on the public surface

The API reference shows **public** symbols (`Private = false`).
Internals — kernel implementations, workspace structs, dispatch
helpers — live in the source under `src/` and are not auto-rendered
here. If you find yourself reaching for a non-exported symbol, that
either means (a) you're doing something the public API supports
through a higher-level entry, or (b) the public API has a gap; in
the latter case, an issue / PR is welcome.

The generated pages below are the canonical symbol reference. The
**Concepts** chapter (start at [Grids](@ref)) and the **Theory &
Verification** chapter (start at [Mass conservation](@ref)) provide the
narrative explanation of how those types and functions fit together.
