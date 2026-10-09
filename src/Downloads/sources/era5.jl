# ===========================================================================
# ERA5 download task builder
#
# Handles monthly and daily chunking for CDS/MARS API requests.
# Monthly chunks use date ranges "YYYY-MM-01/to/YYYY-MM-NN" for efficiency.
#
# Supports request groups from [[download.requests]] in TOML:
#   - "core": model-level analyses (VO, D, T, Q, LNSP)
#   - "convection": forecast fields (UDMF, DDMF, UDRF, DDRF)
#   - "surface": single-level surface fields
# ===========================================================================

"""
    build_tasks(::ERA5Source, protocol, dates, output, requests)

Build CDS/MARS download tasks for ERA5 data.

For monthly chunking, each request group produces one GRIB file per month.
For daily chunking, each request group produces one GRIB file per day.
"""
function build_tasks(source::ERA5Source, protocol::Union{CDSProtocol, MARSProtocol},
                     dates::Vector{Date}, output::OutputConfig,
                     requests::Vector)
    tasks = DownloadTask[]
    out_base = canonical_output_dir(output)

    for req in requests
        name = req["name"]
        dataset = get(req, "dataset", "reanalysis-era5-complete")

        # Determine output subdirectory
        subdir = if output.subdirectory_by_request
            _era5_request_subdir(name)
        else
            ""
        end
        out_dir = isempty(subdir) ? out_base : joinpath(out_base, subdir)

        # Build date string for the API request
        date_str = _era5_date_string(dates)

        # Build filename from template or default
        filename = _era5_filename(output.filename_template, name, dates)

        dest = joinpath(out_dir, filename)

        # Build the API request dict
        api_request = _era5_api_request(req, date_str, dates)
        api_request["dataset"] = dataset

        # Estimate size
        est_mb = _era5_estimate_size(req, dates)

        push!(tasks, DownloadTask(
            "$name $(Dates.format(dates[1], "yyyy-mm"))",
            dataset, dest, api_request, est_mb
        ))
    end

    return tasks
end

# ---------------------------------------------------------------------------
# ERA5-specific helpers
# ---------------------------------------------------------------------------

function _era5_request_subdir(name::String)
    # Map request names to canonical subdirectory names
    subdirs = Dict(
        "core"       => "ml_an_native_core",
        "convection" => "ml_fc_convection",
        "surface"    => "sfc_an_native",
    )
    return get(subdirs, name, name)
end

function _era5_date_string(dates::Vector{Date})
    if length(dates) == 1
        return Dates.format(dates[1], "yyyy-mm-dd")
    else
        first = Dates.format(dates[1], "yyyy-mm-dd")
        last  = Dates.format(dates[end], "yyyy-mm-dd")
        return "$first/to/$last"
    end
end

function _era5_filename(template::String, name::String, dates::Vector{Date})
    if isempty(template)
        tag = Dates.format(dates[1], "yyyymm")
        return "era5_$(name)_$(tag).grib"
    end
    return replace(template,
        "{request_name}" => name,
        "{year_month}"   => Dates.format(dates[1], "yyyymm"),
        "{date}"         => Dates.format(dates[1], "yyyymmdd"),
    )
end

function _era5_api_request(req::Dict{String, Any}, date_str::String, dates::Vector{Date})
    request = Dict{String, Any}()

    # Standard ERA5 request fields
    request["class"]   = "ea"
    request["expver"]  = "1"
    request["stream"]  = "oper"
    request["date"]    = date_str

    # Field type: "an" (analysis) or "fc" (forecast)
    request["type"]    = get(req, "field_type", "an")
    request["levtype"] = "ml"
    request["format"]  = get(req, "format", "grib")

    # Levels
    if haskey(req, "levels")
        request["levelist"] = req["levels"]
    end

    # Parameters (MARS-style param IDs)
    if haskey(req, "params")
        request["param"] = req["params"]
    end

    # Variables (CDS-style names for single-levels)
    if haskey(req, "variables")
        request["variable"] = req["variables"]
        # Single-level datasets use different request format
        if get(req, "dataset", "") == "reanalysis-era5-single-levels"
            delete!(request, "class")
            delete!(request, "expver")
            delete!(request, "stream")
            delete!(request, "type")
            delete!(request, "levtype")
            request["product_type"] = ["reanalysis"]
            request["year"]  = unique(string.(year.(dates)))
            request["month"] = unique(@sprintf("%02d", month(d)) for d in dates)
            request["day"]   = [@sprintf("%02d", day(d)) for d in dates]
            delete!(request, "date")
            request["data_format"] = "grib"
            request["download_format"] = "unarchived"
            delete!(request, "format")
        end
    end

    # Times — MARS uses slash-separated strings, CDS v2 single-levels uses lists
    times_str = get(req, "times", "")
    is_single_level = get(req, "dataset", "") == "reanalysis-era5-single-levels"
    if times_str == "hourly"
        if is_single_level
            request["time"] = [@sprintf("%02d:00", h) for h in 0:23]
        else
            request["time"] = join([@sprintf("%02d:00:00", h) for h in 0:23], "/")
        end
    elseif !isempty(times_str)
        request["time"] = is_single_level ? split(times_str, "/") : times_str
    end

    # Forecast steps (for convection fields)
    if haskey(req, "steps")
        request["step"] = req["steps"]
    end

    return request
end

function _era5_estimate_size(req::Dict{String, Any}, dates::Vector{Date})
    # Rough estimates based on native T639/N320 sizes
    # T639 spectral field ≈ 1.6 MB, N320 gridpoint ≈ 0.8 MB
    n_days = length(dates)
    name = get(req, "name", "")

    if name == "core"
        # 5 params × 137 levels × 24 times × ~1.6 MB/field
        return n_days * 5.0 * 137 * 24 * 1.6
    elseif name == "convection"
        # 4 params × 137 levels × 24 steps (2 base × 12 steps) × ~1.6 MB
        return n_days * 4.0 * 137 * 24 * 1.6
    elseif name == "surface"
        # ~11 params × 1 level × 24 times × ~0.8 MB
        return n_days * 11.0 * 24 * 0.8
    else
        return 0.0
    end
end
