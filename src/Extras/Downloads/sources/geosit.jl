# ===========================================================================
# GEOS-IT download task builder
#
# GEOS-IT C180 cubed-sphere from AWS S3 (public, no auth).
# Daily files organized in YYYYMMDD/ subdirectories.
# ===========================================================================

"""
    build_tasks(::GEOSITSource, ::S3Protocol, dates, output, requests)

Build S3 download tasks for GEOS-IT C180 data. One file per collection
per day.
"""
function build_tasks(source::GEOSITSource, protocol::S3Protocol,
                     dates::Vector{Date}, output::OutputConfig,
                     requests::Vector)
    tasks = DownloadTask[]
    out_base = canonical_output_dir(output)

    collections = _geosit_collections(source, requests)

    for date in dates
        datestr = Dates.format(date, "yyyymmdd")
        y = string(year(date))
        m = @sprintf("%02d", month(date))
        out_dir = output.subdirectory_by_date ?
                  joinpath(out_base, datestr) : out_base

        for coll in collections
            fname = "GEOSIT.$(datestr).$(coll).C180.nc"
            # S3 path uses YYYY/MM/ subdirectories (not YYYYMMDD/)
            s3_key = "$(protocol.prefix)/$y/$m/$fname"
            est_mb = _geosit_collection_size(coll)

            push!(tasks, DownloadTask(
                "GEOS-IT $datestr $coll",
                s3_key, joinpath(out_dir, fname),
                Dict{String,Any}(), est_mb
            ))
        end
    end

    return tasks
end

function _geosit_collections(source::GEOSITSource, requests::Vector)
    # Check requests for explicit collection list
    for req in requests
        if haskey(req, "collections")
            return req["collections"]
        end
    end
    # Default: core transport + physics collections. CTM_I1 is required by
    # the dry-endpoint preprocessor even when a request omits explicit lists.
    return ["CTM_A1", "CTM_I1", "A3mstE", "A1", "A3dyn"]
end

function _geosit_collection_size(coll::String)
    sizes = Dict(
        "CTM_A1" => 4200.0,   # 4.2 GB
        "CTM_I1" => 851.0,
        "A1"     => 200.0,
        "A3mstE" => 1400.0,
        "A3dyn"  => 2500.0,
        "I3"     => 330.0,
    )
    return get(sizes, coll, 500.0)
end
