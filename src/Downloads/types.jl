# ===========================================================================
# Download type hierarchy
#
# Two orthogonal dispatch axes:
#   AbstractDownloadSource   — what data to download (ERA5, GEOS-FP, etc.)
#   AbstractDownloadProtocol — how to download it (CDS, MARS, OPeNDAP, HTTP, S3)
# ===========================================================================

# ---------------------------------------------------------------------------
# Download sources — define what data is available
# ---------------------------------------------------------------------------

abstract type AbstractDownloadSource end

"""
    ERA5Source

ERA5 reanalysis from ECMWF. Supports spectral GRIB, gridded model-level,
and surface fields. Monthly chunking is the production default.
"""
struct ERA5Source <: AbstractDownloadSource
    met_config::Dict{String, Any}
end

"""
    ERA5ARCOSource

ERA5 from Google's ARCO-ERA5 public GCS mirror (queue-free). Native spectral
model-level GRIB (core) + single_level netCDF (surface). Same underlying ERA5
data as `ERA5Source`, but served pre-staged over HTTPS instead of via CDS/MARS.
"""
struct ERA5ARCOSource <: AbstractDownloadSource
    met_config::Dict{String, Any}
end

"""
    GEOSFPSource

GEOS-FP forward processing from NASA GMAO. Native cubed-sphere (C720) or
regridded lat-lon (0.25° × 0.3125°).
"""
struct GEOSFPSource <: AbstractDownloadSource
    met_config::Dict{String, Any}
    product::String    # "geosfp_c720", "geosfp_025"
end

"""
    GEOSITSource

GEOS-IT retrospective from NASA GMAO. Native cubed-sphere (C180).
"""
struct GEOSITSource <: AbstractDownloadSource
    met_config::Dict{String, Any}
    product::String    # "geosit_c180"
end

"""
    MERRA2Source

MERRA-2 retrospective reanalysis (1980–present). 0.5° × 0.625° lat-lon.
"""
struct MERRA2Source <: AbstractDownloadSource
    met_config::Dict{String, Any}
end

source_name(::ERA5Source)    = "ERA5"
source_name(::ERA5ARCOSource) = "ERA5-ARCO"
source_name(::GEOSFPSource) = "GEOS-FP"
source_name(::GEOSITSource) = "GEOS-IT"
source_name(::MERRA2Source)  = "MERRA-2"

# ---------------------------------------------------------------------------
# Python environment (detected once at startup, used by CDS/MARS protocols)
# ---------------------------------------------------------------------------

"""
    PythonEnvironment

Cached Python interpreter configuration. Detected once, reused for all
CDS/MARS API calls in the session.
"""
struct PythonEnvironment
    python_path::String
    has_cdsapi::Bool
    has_ecmwfapi::Bool
    has_cfgrib::Bool
    has_xarray::Bool
    cds_credentials::Bool     # ~/.cdsapirc exists
    mars_credentials::Bool    # ~/.ecmwfapirc exists
end

# ---------------------------------------------------------------------------
# Download protocols — define how to download
# ---------------------------------------------------------------------------

abstract type AbstractDownloadProtocol end

"""
    CDSProtocol

Copernicus Climate Data Store API (Python cdsapi subprocess).
Requires ~/.cdsapirc.
"""
struct CDSProtocol <: AbstractDownloadProtocol
    python_env::PythonEnvironment
end

"""
    MARSProtocol

ECMWF MARS API (Python ecmwfapi subprocess). Falls back to CDS if MARS
credentials are unavailable. Requires ~/.ecmwfapirc.
"""
struct MARSProtocol <: AbstractDownloadProtocol
    python_env::PythonEnvironment
    fallback_to_cds::Bool
end

"""
    OPeNDAPProtocol

OPeNDAP remote subset via NCDatasets.jl. No Python dependency.
"""
struct OPeNDAPProtocol <: AbstractDownloadProtocol
    base_url::String
    auth_required::Bool
end

"""
    protocol_can_download(protocol) -> Bool

True for protocols with an `execute!` implementation. The others (OPeNDAP)
can dry-run and verify existing files, but not download.
"""
protocol_can_download(::AbstractDownloadProtocol) = false

"""
    HTTPProtocol

Direct HTTP file download via Downloads.jl with Content-Length verification.
"""
struct HTTPProtocol <: AbstractDownloadProtocol
    base_url::String
end

"""
    S3Protocol

AWS S3 download via `aws s3 cp` subprocess.
"""
struct S3Protocol <: AbstractDownloadProtocol
    bucket::String
    prefix::String
    no_sign_request::Bool
end

"""
    GCSProtocol

Google Cloud Storage public-bucket download over HTTPS (no auth, no queue).
`execute!` supports `assemble="single"` (one object) and `assemble="concat"`
(fetch component objects and concatenate — used to reassemble the ARCO core
GRIB from its per-variable-group files).
"""
struct GCSProtocol <: AbstractDownloadProtocol
    bucket_base::String    # e.g. "https://storage.googleapis.com/gcp-public-data-arco-era5"
end

protocol_can_download(::Union{CDSProtocol, MARSProtocol, HTTPProtocol, S3Protocol, GCSProtocol}) = true

# ---------------------------------------------------------------------------
# Download tasks — intermediate representation for dry-run and execution
# ---------------------------------------------------------------------------

"""
    DownloadTask

A single unit of work: download one file or one API request.
Built by `build_tasks`, printed by `--dry-run`, executed by `execute!`.
"""
struct DownloadTask
    name::String              # human-readable label
    source_url::String        # URL, S3 path, or API request ID
    dest_path::String         # local output path
    request::Dict{String,Any} # API request dict (for CDS/MARS) or empty
    estimated_size_mb::Float64
end

# ---------------------------------------------------------------------------
# Output configuration (derived from TOML [output] section)
# ---------------------------------------------------------------------------

"""
    OutputConfig

Canonical Data Layout output configuration. Constructs paths following:
    <data_root>/met/<met_source>/<grid_name>/<cadence>/<payload_type>/
"""
struct OutputConfig
    data_root::String
    met_source::String
    grid_name::String
    cadence::String
    payload_type::String
    subdirectory_by_date::Bool
    subdirectory_by_request::Bool
    filename_template::String
end

"""
    canonical_output_dir(oc::OutputConfig) -> String

Build the canonical output directory path.
"""
function canonical_output_dir(oc::OutputConfig)
    return joinpath(oc.data_root, "met", oc.met_source, oc.grid_name,
                    oc.cadence, oc.payload_type)
end

# ---------------------------------------------------------------------------
# Schedule configuration
# ---------------------------------------------------------------------------

"""
    ScheduleConfig

Controls date range and chunking strategy.
"""
struct ScheduleConfig
    start_date::Date
    end_date::Date
    chunk::Symbol             # :monthly, :daily, :per_file
    max_concurrent::Int
    max_retries::Int
    retry_wait_seconds::Int
    skip_existing::Bool
end

# ---------------------------------------------------------------------------
# Top-level download configuration
# ---------------------------------------------------------------------------

"""
    DownloadConfig{S, P}

Complete download configuration parsed from a recipe TOML.
Parameterized on source and protocol for concrete dispatch.
"""
struct DownloadConfig{S <: AbstractDownloadSource, P <: AbstractDownloadProtocol}
    source::S
    protocol::P
    output::OutputConfig
    schedule::ScheduleConfig
    requests::Vector{Dict{String, Any}}   # per-request configs from [[download.requests]]
end
