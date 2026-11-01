"""Cache one decoded hour, independently of the legacy whole-day cache."""
function spectral_hour_cache_path(cache_dir, paths, hour; T_target=0)
    day = spectral_day_cache_path(cache_dir, paths.vo_d, paths.lnsp; T_target)
    return replace(day, ".jld2" => "_hour$(lpad(hour, 2, '0')).jld2")
end

function validate_spectral_hour(spec)
    shape = (spec.T+1, spec.T+1, ERA5_NATIVE_LEVEL_COUNT)
    size(spec.vo) == size(spec.d) == shape || throw(DimensionMismatch("spectral hour VO/D"))
    size(spec.lnsp) == shape[1:2] || throw(DimensionMismatch("spectral hour LNSP"))
    return spec
end

"""
Call `f(read_hour)` with a bounded reader. ecCodes indexes locate each hour
without decoding other hours; only the caller's current coefficients survive.
Optional JLD2 caches contain one hour each. Index handles close even on failure.
"""
function with_spectral_hours(f, paths; T_target=0, cache_dir="")
    indices = nothing
    function read_hour(hour)
        0 <= hour <= 23 || throw(ArgumentError("ERA5 hour outside 0:23"))
        path = isempty(cache_dir) ? "" : spectral_hour_cache_path(cache_dir, paths, hour; T_target)
        if isfile(path)
            try
                return validate_spectral_hour(JLD2.load(path, "endpoint"))
            catch err
                @warn "Unreadable spectral hour cache; decoding GRIB" path exception=(err, catch_backtrace())
            end
        end
        if indices === nothing
            vo_index = GRIB.Index(paths.vo_d, "dataTime", "shortName")
            lnsp_index = try
                GRIB.Index(paths.lnsp, "dataTime", "shortName")
            catch
                destroy(vo_index)
                rethrow()
            end
            indices = (vo_index, lnsp_index)
        end
        vi, hi = indices
        GRIB.select!(hi, "dataTime", 100hour)
        GRIB.select!(hi, "shortName", "lnsp")
        H = nothing
        T_file = 0
        vals = Float64[]
        for msg in hi
            try
                H === nothing || throw(ArgumentError("duplicate LNSP hour $hour"))
                T_file = Int(msg["J"])
                H = zeros(ComplexF64, T_file+1, T_file+1)
                read_spectral_coeffs!(H, msg, vals)
            finally
                # Run and remove the registered GRIB finalizer exactly once.
                finalize(msg)
            end
        end
        H === nothing && throw(ArgumentError("missing LNSP hour $hour"))
        T = T_target > 0 ? min(T_target, T_file) : T_file
        vo = zeros(ComplexF64, T+1, T+1, ERA5_NATIVE_LEVEL_COUNT)
        d = similar(vo)
        flags = (vo=falses(ERA5_NATIVE_LEVEL_COUNT), d=falses(ERA5_NATIVE_LEVEL_COUNT))
        scratch = similar(H)
        GRIB.select!(vi, "dataTime", 100hour)
        for field in ("vo", "d")
            GRIB.select!(vi, "shortName", field)
            for msg in vi
                try
                    name = String(msg["shortName"])
                    name in ("vo", "d") || throw(ArgumentError("unexpected spectral field $name"))
                    level = Int(msg["level"])
                    1 <= level <= ERA5_NATIVE_LEVEL_COUNT || throw(ArgumentError("invalid native level $level"))
                    seen = name == "vo" ? flags.vo : flags.d
                    seen[level] && throw(ArgumentError("duplicate $name hour $hour level $level"))
                    read_spectral_coeffs!(scratch, msg, vals) == T_file || error("spectral truncation changed within hour")
                    target = name == "vo" ? vo : d
                    target[:,:,level] .= view(scratch, 1:T+1, 1:T+1)
                    seen[level] = true
                finally
                    # Run and remove the registered GRIB finalizer exactly once.
                    finalize(msg)
                end
            end
        end
        all(flags.vo) && all(flags.d) || throw(ArgumentError("missing VO/D levels at hour $hour"))
        endpoint = (; lnsp=copy(H[1:T+1,1:T+1]), vo, d, T)
        if !isempty(path)
            mkpath(dirname(path))
            try
                atomic_publish(path) do tmp
                    JLD2.jldsave(tmp; endpoint)
                end
            catch err
                @warn "Spectral hour cache write failed" path exception=(err, catch_backtrace())
            end
        end
        return endpoint
    end
    try
        return f(read_hour)
    finally
        if indices !== nothing
            foreach(destroy, indices)
        end
    end
end
