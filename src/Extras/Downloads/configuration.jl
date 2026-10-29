# ===========================================================================
# TOML configuration parsing → DownloadConfig
#
# Two-level TOML design:
#   1. Source definition (config/met_sources/*.toml) — what data exists
#   2. Download recipe (config/downloads/*.toml) — what to download
# ===========================================================================

"""
    parse_download_config(cfg::Dict{String,Any}) -> DownloadConfig

Parse a download recipe TOML into a fully resolved DownloadConfig.
"""
function parse_download_config(cfg::Dict{String, Any})
    source   = _build_source(cfg)
    protocol = _build_protocol(cfg, source)
    output   = _build_output(cfg)
    schedule = _build_schedule(cfg)
    requests = _build_requests(cfg)
    return DownloadConfig(source, protocol, output, schedule, requests)
end

"""
    download_logging(cfg) -> DownloadLogging

Parse the download recipe's `[logging]` table. `level` has the package-wide
`debug|info|warn|error` values; `progress` is `bar`, `lines` (the default), or
`none`, matching runtime configuration.
"""
function download_logging(cfg::AbstractDict)
    level = log_level(cfg)
    section = get(cfg, "logging", Dict{String, Any}())
    raw = get(section, "progress", "lines")
    kind = raw isa AbstractString ? lowercase(raw) : nothing
    progress = if kind == "bar"
        DownloadProgressBar()
    elseif kind == "lines"
        DownloadProgressLines()
    elseif kind == "none"
        NoDownloadProgress()
    else
        throw(ArgumentError(
            "[logging].progress must be \"bar\", \"lines\" or \"none\"; got $(repr(raw))"))
    end
    return DownloadLogging(level, progress)
end

# ---------------------------------------------------------------------------
# Source construction
# ---------------------------------------------------------------------------

function _build_source(cfg::Dict{String, Any})
    met_source_path = cfg["source"]["met_source"]
    met_source_path = expanduser(met_source_path)

    # Resolve relative paths against the project root
    if !isabspath(met_source_path)
        met_source_path = joinpath(_project_root(), met_source_path)
    end
    isfile(met_source_path) || error("Met source TOML not found: $met_source_path")

    met_cfg = TOML.parsefile(met_source_path)
    source_name = met_cfg["source"]["name"]

    return _build_source(Val(Symbol(replace(source_name, "-" => "", " " => ""))),
                         met_cfg, cfg)
end

function _build_source(::Val{:ERA5}, met_cfg, cfg)
    ERA5Source(met_cfg)
end

function _build_source(::Val{:ERA5ARCO}, met_cfg, cfg)
    ERA5ARCOSource(met_cfg)
end

function _build_source(::Val{:GEOSFP}, met_cfg, cfg)
    product = get(get(cfg, "download", Dict()), "product", "geosfp_c720")
    GEOSFPSource(met_cfg, product)
end

function _build_source(::Val{:GEOSIT}, met_cfg, cfg)
    product = get(get(cfg, "download", Dict()), "product", "geosit_c180")
    GEOSITSource(met_cfg, product)
end

function _build_source(::Val{:MERRA2}, met_cfg, cfg)
    MERRA2Source(met_cfg)
end

# ---------------------------------------------------------------------------
# Protocol construction
# ---------------------------------------------------------------------------

function _build_protocol(cfg::Dict{String, Any}, source::AbstractDownloadSource)
    dl = cfg["download"]
    proto = get(dl, "protocol", _default_protocol(source))
    return _build_protocol(Val(Symbol(proto)), dl, source)
end

_default_protocol(::ERA5Source)    = "cds"
_default_protocol(::ERA5ARCOSource) = "gcs"
_default_protocol(::GEOSFPSource) = "http"
_default_protocol(::GEOSITSource) = "s3"
_default_protocol(::MERRA2Source)  = "opendap"

function _build_protocol(::Val{:cds}, dl, ::AbstractDownloadSource)
    python_path = get(dl, "python", "python3")
    env = detect_python_env(python_path)
    env.has_cdsapi || error("Python package 'cdsapi' not found. Install: pip install cdsapi")
    env.cds_credentials || error("~/.cdsapirc not found. Configure CDS credentials first.")
    return CDSProtocol(env)
end

function _build_protocol(::Val{:mars}, dl, ::AbstractDownloadSource)
    python_path = get(dl, "python", "python3")
    env = detect_python_env(python_path)
    fallback = get(dl, "fallback_to_cds", true)

    if !env.mars_credentials || !env.has_ecmwfapi
        if fallback && env.has_cdsapi && env.cds_credentials
            @info "  MARS unavailable, falling back to CDS"
            return CDSProtocol(env)
        end
        error("MARS API not available and CDS fallback disabled. " *
              "Install ecmwf-api-client and configure ~/.ecmwfapirc")
    end
    return MARSProtocol(env, fallback)
end

function _build_protocol(::Val{:http}, dl, ::AbstractDownloadSource)
    base_url = dl["base_url"]
    return HTTPProtocol(base_url)
end

function _build_protocol(::Val{:s3}, dl, ::AbstractDownloadSource)
    bucket = dl["bucket"]
    prefix = get(dl, "prefix", "")
    no_sign = get(dl, "no_sign_request", true)
    return S3Protocol(bucket, prefix, no_sign)
end

function _build_protocol(::Val{:gcs}, dl, source::AbstractDownloadSource)
    # base URL comes from the recipe [download] or the met-source [access]
    base = get(dl, "bucket_base", "")
    if isempty(base)
        base = get(get(source.met_config, "access", Dict()), "bucket_base", "")
    end
    isempty(base) && error("gcs protocol: set [download].bucket_base or [access].bucket_base")
    return GCSProtocol(base)
end

function _build_protocol(::Val{:opendap}, dl, source::AbstractDownloadSource)
    met_cfg = source.met_config
    base_url = met_cfg["access"]["base_url"]
    auth_required = get(met_cfg["access"], "auth_required", false)
    return OPeNDAPProtocol(base_url, auth_required)
end

# ---------------------------------------------------------------------------
# Output configuration (follows canonical Data Layout)
# ---------------------------------------------------------------------------

function _build_output(cfg::Dict{String, Any})
    out = cfg["output"]

    data_root   = expanduser(get(out, "data_root", "~/data/AtmosTransport"))
    met_source  = out["met_source"]
    grid_name   = out["grid_name"]
    cadence     = out["cadence"]
    payload     = out["payload_type"]

    subdir_date    = get(out, "subdirectory_by_date", false)
    subdir_request = get(out, "subdirectory_by_request", false)
    filename_tmpl  = get(out, "filename_template", "")

    return OutputConfig(data_root, met_source, grid_name, cadence, payload,
                        subdir_date, subdir_request, filename_tmpl)
end

# ---------------------------------------------------------------------------
# Schedule configuration
# ---------------------------------------------------------------------------

function _build_schedule(cfg::Dict{String, Any})
    sched = cfg["schedule"]

    start_date = Date(sched["start_date"])
    end_date   = Date(sched["end_date"])

    chunk_str = get(sched, "chunk", "monthly")
    chunk = Symbol(chunk_str)
    chunk in (:monthly, :daily, :per_file) ||
        error("Unknown chunk strategy: $chunk_str. Use: monthly, daily, per_file")

    opts = get(cfg, "options", Dict())
    max_concurrent = get(opts, "max_concurrent", 4)
    max_retries    = get(opts, "max_retries", 3)
    retry_wait     = get(opts, "retry_wait_seconds", 30)
    skip_existing  = get(opts, "skip_existing", true)

    return ScheduleConfig(start_date, end_date, chunk,
                          max_concurrent, max_retries, retry_wait, skip_existing)
end

# ---------------------------------------------------------------------------
# Request list parsing
# ---------------------------------------------------------------------------

function _build_requests(cfg::Dict{String, Any})
    dl = cfg["download"]
    # [[download.requests]] is an array of tables in TOML — TOML.jl returns Vector{Any}
    raw = get(dl, "requests", Any[])
    return Dict{String,Any}[r for r in raw]
end

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

function _project_root()
    # Walk up from src/Extras/Downloads/ to find Project.toml
    dir = @__DIR__
    for _ in 1:5
        if isfile(joinpath(dir, "Project.toml"))
            return dir
        end
        dir = dirname(dir)
    end
    return pwd()
end
