"""
    Downloads

TOML-driven download pipeline for meteorological and emissions data.

Provides a unified entry point for downloading ERA5, GEOS-FP, GEOS-IT,
and MERRA-2 data. All download parameters are specified in TOML config
files; the pipeline dispatches on source type and access protocol.

## Architecture

Two orthogonal dispatch axes:

    AbstractDownloadSource    — what data to download
    ├── ERA5Source            — ECMWF ERA5 reanalysis
    ├── GEOSFPSource          — NASA GEOS-FP forward processing
    ├── GEOSITSource          — NASA GEOS-IT retrospective
    └── MERRA2Source           — NASA MERRA-2 reanalysis

    AbstractDownloadProtocol  — how to download it
    ├── CDSProtocol           — CDS API (Python subprocess)
    ├── MARSProtocol          — MARS API (Python subprocess, CDS fallback)
    ├── OPeNDAPProtocol       — NCDatasets remote subset
    ├── HTTPProtocol          — Direct HTTP with Content-Length verification
    └── S3Protocol            — AWS S3 CLI

## Usage

```bash
julia --project=. scripts/downloads/download_data.jl config/downloads/era5_native_monthly.toml \\
    [--start 2021-12-01] [--end 2021-12-31] [--dry-run] [--verify]
```

Output paths follow the canonical Data Layout hierarchy:
    <data_root>/met/<source>/<grid>/<cadence>/<payload>/
"""
module DataDownloads

using Dates
using Logging
using Printf
using SHA
using TOML

# Type hierarchy (sources, protocols, tasks, configs)
include("types.jl")

# Python subprocess interop (CDS/MARS API calls)
include("python_interop.jl")

# Content-Length verified downloads (originally scripts/downloads/download_utils.jl, removed)
include("verification.jl")

# TOML configuration parsing
include("configuration.jl")

# Top-level download pipeline
include("pipeline.jl")

# Source-specific task builders
include("sources/era5.jl")
include("sources/era5_arco.jl")
include("sources/geosfp.jl")
include("sources/geosit.jl")
include("sources/merra2.jl")

# Exports
export download_data!
export parse_download_config, DownloadConfig
export ERA5Source, ERA5ARCOSource, GEOSFPSource, GEOSITSource, MERRA2Source
export CDSProtocol, MARSProtocol, OPeNDAPProtocol, HTTPProtocol, S3Protocol, GCSProtocol
export verified_download, verify_downloads
export detect_python_env, PythonEnvironment

end # module DataDownloads
