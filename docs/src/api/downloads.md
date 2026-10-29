# DataDownloads API

The download module turns source/protocol TOML configuration into verified
ERA5, GEOS-FP, GEOS-IT, and MERRA-2 input files. See [Data sources](@ref) for
the configuration-oriented guide.

The module remains `AtmosTransport.DataDownloads`; its implementation now
lives in `src/Extras/Downloads/`. It is non-core, has no tests, and is intended
to become its own package. Core code never imports it.

```@autodocs
Modules = [AtmosTransport.DataDownloads]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```
