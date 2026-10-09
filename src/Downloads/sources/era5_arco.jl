# ===========================================================================
# ERA5-ARCO download task builder
#
# Google ARCO-ERA5 (gs://gcp-public-data-arco-era5) is a public, no-auth,
# queue-free mirror of the native ECMWF GRIB. Model-level dynamics are kept as
# spectral harmonic coefficients (T639) and moisture on the reduced-Gaussian
# N320 grid — exactly the form the N320 preprocessor (`sources/era5*.jl`)
# consumes. Unlike CDS (MARS-backed, hours-to-days queue), ARCO serves
# pre-staged objects over HTTPS at network speed (~78 MB/s parallel measured).
#
# The native "core" fields are split across per-variable-group GRIB2 files:
#   {YYYYMMDD}_hres_dve.grb2  → vorticity (138) + divergence (155), spectral
#   {YYYYMMDD}_hres_tw.grb2   → temperature (130), spectral + vertical velocity
#   {YYYYMMDD}_hres_o3q.grb2  → specific humidity (133), reduced-Gaussian + ozone
# GRIB is a concatenatable message stream and the reader filters by
# param+dataDate, so we simply `cat` the three into the single
# `era5_core_YYYYMMDD.grib` the preprocessor expects; extra o3/w/cloud messages
# are ignored. (Surface pressure `lnsp` is NOT in ARCO model-level GRIB — it is
# sourced from the single_level collection below and handled preprocessing-side.)
#
# Surface fields (surface pressure + PBL-diffusion inputs) come from ARCO's
# per-variable single_level netCDF collection.
# ===========================================================================

"""
    build_tasks(::ERA5ARCOSource, ::GCSProtocol, dates, output, requests)

Build ARCO-ERA5 GCS download tasks. Recognizes two request groups by `name`:
`"core"` (concatenated model-level GRIB) and `"surface"` (single_level netCDF).
"""
function build_tasks(source::ERA5ARCOSource, protocol::GCSProtocol,
                     dates::Vector{Date}, output::OutputConfig,
                     requests::Vector)
    tasks = DownloadTask[]
    out_base = canonical_output_dir(output)
    acc = source.met_config["access"]["arco"]

    for req in requests
        name = get(req, "name", "")
        if name == "core"
            append!(tasks, _arco_core_tasks(protocol, acc, dates, out_base))
        elseif name == "surface"
            append!(tasks, _arco_surface_tasks(protocol, acc, req, dates, out_base))
        else
            @warn "ERA5-ARCO: skipping unknown request '$name' (expected 'core' or 'surface')"
        end
    end
    return tasks
end

# One concatenated core GRIB per day: dve + tw + o3q -> era5_core_YYYYMMDD.grib
function _arco_core_tasks(protocol::GCSProtocol, acc, dates::Vector{Date}, out_base::String)
    tasks = DownloadTask[]
    out_dir = joinpath(out_base, "ml_an_native_core")
    stage   = joinpath(out_dir, "_arco_stage")
    components = acc["core_components"]              # e.g. ["dve", "tw", "o3q"]
    for date in dates
        ymd = Dates.format(date, "yyyymmdd")
        dir = replace(acc["core_dir_template"], "{YYYY}" => string(year(date)))
        # NamedTuples (not Dicts): the request dict is repr()'d into the download
        # manifest's task_identity, and NamedTuple field order is stable whereas
        # Dict iteration order is not — a Dict here risks a nondeterministic
        # identity → spurious :corrupt → 22 GB re-download.
        comp = [(name = "$(ymd)_hres_$(c).grb2",
                 url  = "$(protocol.bucket_base)/$dir/$(ymd)_hres_$(c).grb2")
                for c in components]
        dest = joinpath(out_dir, "era5_core_$(ymd).grib")
        request = Dict{String, Any}(
            "assemble"   => "concat",
            "components" => comp,
            "stage_dir"  => stage,
            "cleanup"    => true,
        )
        # ~11.4 GB (dve+tw spectral) + ~12 GB (o3q) ≈ 22.8 GB/day observed.
        push!(tasks, DownloadTask("ERA5-ARCO core $ymd", comp[1].url,
                                  dest, request, 22800.0))
    end
    return tasks
end

# One netCDF per (day, variable) from the single_level collection.
function _arco_surface_tasks(protocol::GCSProtocol, acc, req, dates::Vector{Date}, out_base::String)
    tasks = DownloadTask[]
    variables = get(req, "variables", String[])
    isempty(variables) && @warn "ERA5-ARCO surface: no [variables] listed"
    fname = get(acc, "single_level_filename", "surface.nc")
    for date in dates
        ymd = Dates.format(date, "yyyymmdd")
        y = string(year(date)); m = @sprintf("%02d", month(date)); d = @sprintf("%02d", day(date))
        for var in variables
            dir = replace(acc["single_level_dir_template"],
                          "{YYYY}" => y, "{MM}" => m, "{DD}" => d, "{variable}" => var)
            url  = "$(protocol.bucket_base)/$dir/$fname"
            dest = joinpath(out_base, "sfc_an_native", "arco", ymd, "$(var).nc")
            request = Dict{String, Any}("assemble" => "single")
            push!(tasks, DownloadTask("ERA5-ARCO surface $ymd $var", url, dest, request, 50.0))
        end
    end
    return tasks
end

# ---------------------------------------------------------------------------
# GCS execution — public bucket over HTTPS. `concat` assembles the per-group
# GRIB parts; `single` is a plain verified download.
# ---------------------------------------------------------------------------
function execute!(task::DownloadTask, proto::GCSProtocol;
                  max_retries::Int=3, retry_wait::Int=30)
    mkpath(dirname(task.dest_path))
    assemble = get(task.request, "assemble", "single")

    if assemble == "single"
        return verified_download(task.source_url, task.dest_path; max_retries=max_retries)

    elseif assemble == "concat"
        stage = task.request["stage_dir"]
        mkpath(stage)
        comps = task.request["components"]
        parts = [joinpath(stage, String(c.name)) for c in comps]
        oks = fill(false, length(comps))
        # Fetch the parts concurrently (libcurl multiplexes the transfers).
        @sync for (i, c) in enumerate(comps)
            @async oks[i] = verified_download(String(c.url), parts[i];
                                              max_retries=max_retries)
        end
        if !all(oks)
            @error "  ARCO concat: component download failed for $(basename(task.dest_path))"
            get(task.request, "cleanup", true) && foreach(p -> rm(p; force=true), parts)
            return false
        end
        tmp = task.dest_path * ".part"
        try
            @info "  Concatenating $(length(parts)) parts → $(basename(task.dest_path))"
            run(pipeline(Cmd(vcat("cat", parts)); stdout=tmp))   # streams; memory-safe
            mv(tmp, task.dest_path; force=true)
        catch e
            @error "  ARCO concat failed: $e"
            rm(tmp; force=true)
            return false
        finally
            get(task.request, "cleanup", true) && foreach(p -> rm(p; force=true), parts)
        end
        return isfile(task.dest_path) && filesize(task.dest_path) > 0

    else
        error("GCSProtocol: unknown assemble mode '$assemble'")
    end
end
