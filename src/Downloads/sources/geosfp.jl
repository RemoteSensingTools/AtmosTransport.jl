# ===========================================================================
# GEOS-FP download task builder
#
# Supports two products:
#   - C720 cubed-sphere mass fluxes from the GEOS-Chem public S3 archive (HTTP)
#   - 0.25° lat-lon surface/physics fields from AWS S3 (HTTP)
# ===========================================================================

"""
    build_tasks(::GEOSFPSource, ::HTTPProtocol, dates, output, requests)

Build HTTP download tasks for GEOS-FP data. One file per timestep per
collection, organized in YYYYMMDD/ subdirectories.
"""
function build_tasks(source::GEOSFPSource, protocol::HTTPProtocol,
                     dates::Vector{Date}, output::OutputConfig,
                     requests::Vector)
    tasks = DownloadTask[]
    out_base = canonical_output_dir(output)
    collections = _geosfp_collections(source, requests)

    for date in dates
        datestr = Dates.format(date, "yyyymmdd")
        out_dir = output.subdirectory_by_date ?
                  joinpath(out_base, datestr) : out_base

        file_list = _geosfp_file_urls(source, protocol, date, collections)

        for (url, fname, est_mb) in file_list
            dest = joinpath(out_dir, fname)
            push!(tasks, DownloadTask(
                "GEOS-FP $datestr $(splitext(fname)[1])",
                url, dest, Dict{String,Any}(), est_mb
            ))
        end
    end

    return tasks
end

function _geosfp_collections(source::GEOSFPSource, requests::Vector)
    for req in requests
        if haskey(req, "collections")
            return req["collections"]
        end
    end
    return _default_collections(source)
end

function _default_collections(source::GEOSFPSource)
    if source.product == "geosfp_c720"
        return ["tavg_1hr_ctm_c0720_v72"]
    elseif source.product == "geosfp_025"
        return ["A1", "A3mstE", "A3dyn"]
    else
        return String[]
    end
end

"""
    _geosfp_file_urls(source, protocol, date, collections)

Generate (url, filename, est_mb) tuples for a single day's files.
"""
function _geosfp_file_urls(source::GEOSFPSource, protocol::HTTPProtocol,
                           date::Date, collections::Vector)
    datestr = Dates.format(date, "yyyymmdd")
    results = Tuple{String, String, Float64}[]

    y = year(date)
    m = @sprintf("%02d", month(date))
    d = @sprintf("%02d", day(date))

    if source.product == "geosfp_c720"
        # C720 hourly CTM files from the GEOS-Chem public S3 archive — timestamp
        # is HHMM (30-min offset for time-averaged data), URL path is Y$y/M$m/D$d/
        for coll in collections
            for hour in 0:23
                timestamp = @sprintf("%02d30", hour)
                fname = "GEOS.fp.asm.$(coll).$(datestr)_$(timestamp).V01.nc4"
                url = "$(protocol.base_url)/Y$y/M$m/D$d/$fname"
                push!(results, (url, fname, 2700.0))  # ~2.7 GB per file
            end
        end
    elseif source.product == "geosfp_025"
        # 0.25° lat-lon surface/physics from S3
        for coll in collections
            fname = "GEOSFP.$(datestr).$(coll).025x03125.nc"
            url = "$(protocol.base_url)/$fname"
            est = coll == "A3mstE" ? 1400.0 :
                  coll == "A3dyn"  ? 2500.0 : 200.0
            push!(results, (url, fname, est))
        end
    end

    return results
end
