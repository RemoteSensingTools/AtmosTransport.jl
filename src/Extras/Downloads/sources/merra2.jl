# ===========================================================================
# MERRA-2 download task builder
#
# MERRA-2 via OPeNDAP (NASA GES DISC). Requires NASA Earthdata ~/.netrc.
# Stream codes: 100 (1980–91), 200 (92–00), 300 (01–10), 400 (11–99).
# ===========================================================================

"""
    build_tasks(::MERRA2Source, ::OPeNDAPProtocol, dates, output, requests)

Build OPeNDAP download tasks for MERRA-2 data.
"""
function build_tasks(source::MERRA2Source, protocol::OPeNDAPProtocol,
                     dates::Vector{Date}, output::OutputConfig,
                     requests::Vector)
    tasks = DownloadTask[]
    out_base = canonical_output_dir(output)

    collections = _merra2_collections(source, requests)

    for date in dates
        datestr = Dates.format(date, "yyyymmdd")

        for (coll_key, coll_info) in collections
            stream = _merra2_stream_code(source, date)
            coll_name = coll_info["collection_name"]
            dataset_id = coll_info["dataset"]

            fname = "MERRA2_$(stream).$(coll_name).$(datestr).nc4"
            url = "$(protocol.base_url)/$(dataset_id)/$(datestr[1:4])/$(datestr[5:6])/$fname"
            dest = joinpath(out_base, fname)

            push!(tasks, DownloadTask(
                "MERRA-2 $datestr $coll_key",
                url, dest, Dict{String,Any}(), 500.0
            ))
        end
    end

    return tasks
end

function _merra2_collections(source::MERRA2Source, requests::Vector)
    met_cfg = source.met_config
    all_colls = get(met_cfg, "collections", Dict())

    # Filter to requested collections if specified
    wanted = String[]
    for req in requests
        if haskey(req, "collections")
            wanted = req["collections"]
        end
    end

    if isempty(wanted)
        return all_colls
    else
        return Dict(k => v for (k, v) in all_colls if k in wanted)
    end
end

function _merra2_stream_code(source::MERRA2Source, date::Date)
    yr = year(date)
    streams = get(get(source.met_config, "access", Dict()), "streams", Dict())
    for (code, range_pair) in streams
        yr in range_pair[1]:range_pair[2] && return code
    end
    return "400"
end
