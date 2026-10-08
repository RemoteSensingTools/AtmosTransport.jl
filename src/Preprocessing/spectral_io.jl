"""
Read spectral coefficients from a GRIB message into a complex matrix.
Returns spec[n+1, m+1] for m=0..T, n=m..T (upper triangular).
"""
function read_spectral_coeffs!(spec::AbstractMatrix{ComplexF64}, msg)
    return read_spectral_coeffs!(spec, msg, Float64[])
end

function _check_eccodes_status(err::Integer, op::AbstractString)
    err == 0 && return nothing
    error("ecCodes $op failed with status code $err")
end

function _required_grib_key(msg, key::AbstractString)
    try
        return msg[key]
    catch err
        error("spectral GRIB message is missing required key '$key': $err")
    end
end

function _optional_grib_key(msg, key::AbstractString)
    try
        return msg[key]
    catch
        return nothing
    end
end

function _validate_spectral_header(msg, nvalues::Integer, spec::AbstractMatrix)
    grid_type = String(_required_grib_key(msg, "gridType"))
    grid_type == "sh" ||
        error("unsupported spectral GRIB gridType '$grid_type'; expected 'sh'")

    J = Int(_required_grib_key(msg, "J"))
    K = Int(_required_grib_key(msg, "K"))
    M = Int(_required_grib_key(msg, "M"))
    J >= 0 || error("invalid spectral truncation J=$J")
    (J == K == M) ||
        error("unsupported non-triangular spectral truncation J/K/M = $J/$K/$M")

    expected = (J + 1) * (J + 2)
    nvalues == expected ||
        error("spectral GRIB value count mismatch: got $nvalues, expected $expected for T$J")
    size(spec, 1) >= J + 1 && size(spec, 2) >= J + 1 ||
        error("spectral coefficient buffer $(size(spec)) too small for T$J")

    subset_keys = ("JS", "KS", "MS")
    subset = map(key -> _optional_grib_key(msg, key), subset_keys)
    if any(!isnothing, subset)
        all(!isnothing, subset) ||
            error("incomplete spectral sub-truncation metadata JS/KS/MS = $(subset)")
        JS, KS, MS = Int.(subset)
        (0 <= JS <= J && 0 <= KS <= J && 0 <= MS <= J) ||
            error("invalid spectral sub-truncation JS/KS/MS = $JS/$KS/$MS for T$J")
        (JS == KS == MS) ||
            error("unsupported non-triangular spectral sub-truncation JS/KS/MS = $JS/$KS/$MS")
    end

    laplacian = _optional_grib_key(msg, "laplacianOperator")
    if laplacian !== nothing
        isfinite(Float64(laplacian)) ||
            error("invalid spectral laplacianOperator=$laplacian")
    end

    return J
end

function read_spectral_coeffs!(spec::AbstractMatrix{ComplexF64}, msg, vals::Vector{Float64})
    handle = msg.ptr
    sz = Ref{Csize_t}(0)
    err = ccall((:codes_get_size, GRIB.eccodes), Cint,
                (Ptr{Cvoid}, Cstring, Ref{Csize_t}), handle, "values", sz)
    _check_eccodes_status(err, "codes_get_size(values)")

    T = _validate_spectral_header(msg, Int(sz[]), spec)

    resize!(vals, sz[])
    err = ccall((:codes_get_double_array, GRIB.eccodes), Cint,
                (Ptr{Cvoid}, Cstring, Ptr{Float64}, Ref{Csize_t}),
                handle, "values", vals, sz)
    _check_eccodes_status(err, "codes_get_double_array(values)")

    # One pass over `spec`: coefficients for m ≤ n ≤ T, zeros elsewhere.
    # `_validate_spectral_header` checked the value count and the buffer size.
    idx = 1
    @inbounds for m in axes(spec, 2) .- 1, n in axes(spec, 1) .- 1
        if n >= m && n <= T && m <= T
            spec[n + 1, m + 1] = complex(vals[idx], vals[idx + 1])
            idx += 2
        else
            spec[n + 1, m + 1] = zero(ComplexF64)
        end
    end
    return T
end

const SPECTRAL_DAY_CACHE_VERSION = 2
const _ERA5_SPECTRAL_DAY_HOURS = collect(0:23)
const _ERA5_SPECTRAL_LEVEL_COUNT = 137

function _validate_spectral_coverage(hours,
                                     vo_seen::AbstractDict,
                                     d_seen::AbstractDict;
                                     expected_hours = _ERA5_SPECTRAL_DAY_HOURS,
                                     nlevels::Int = _ERA5_SPECTRAL_LEVEL_COUNT)
    actual_hours = sort!(collect(hours))
    actual_hours == collect(expected_hours) || throw(ArgumentError(
        "incomplete ERA5 spectral day: LNSP hours=$(actual_hours), expected=$(collect(expected_hours))"))
    for (name, seen_by_hour) in (("vo", vo_seen), ("d", d_seen))
        sort!(collect(keys(seen_by_hour))) == actual_hours || throw(ArgumentError(
            "incomplete ERA5 spectral day: $name hours=$(sort!(collect(keys(seen_by_hour)))), expected=$actual_hours"))
        for hour in actual_hours
            seen = seen_by_hour[hour]
            length(seen) == nlevels || throw(ArgumentError(
                "$name hour $hour has $(length(seen)) level flags, expected $nlevels"))
            missing = findall(!, seen)
            isempty(missing) || throw(ArgumentError(
                "incomplete ERA5 spectral day: $name hour $hour is missing levels $(missing)"))
        end
    end
    return actual_hours
end

function _validate_cached_spectral_day(spec)
    spec.hours == _ERA5_SPECTRAL_DAY_HOURS || throw(ArgumentError(
        "cached spectral day has hours=$(spec.hours), expected=$(_ERA5_SPECTRAL_DAY_HOURS)"))
    spec.n_times == length(spec.hours) || throw(ArgumentError(
        "cached spectral day n_times=$(spec.n_times) does not match $(length(spec.hours)) hours"))
    expected_shape = (spec.T + 1, spec.T + 1, _ERA5_SPECTRAL_LEVEL_COUNT)
    for hour in spec.hours
        haskey(spec.lnsp_all, hour) || throw(ArgumentError("cached spectral day is missing LNSP hour $hour"))
        haskey(spec.vo_by_hour, hour) || throw(ArgumentError("cached spectral day is missing vo hour $hour"))
        haskey(spec.d_by_hour, hour) || throw(ArgumentError("cached spectral day is missing d hour $hour"))
        size(spec.lnsp_all[hour]) == (spec.T + 1, spec.T + 1) || throw(DimensionMismatch(
            "cached LNSP hour $hour has shape $(size(spec.lnsp_all[hour])); " *
            "expected $((spec.T + 1, spec.T + 1))"))
        size(spec.vo_by_hour[hour]) == expected_shape || throw(DimensionMismatch(
            "cached vo hour $hour has shape $(size(spec.vo_by_hour[hour])); expected $expected_shape"))
        size(spec.d_by_hour[hour]) == expected_shape || throw(DimensionMismatch(
            "cached d hour $hour has shape $(size(spec.d_by_hour[hour])); expected $expected_shape"))
    end
    return spec
end

"""
    spectral_day_cache_path(cache_dir, vo_d_path, lnsp_path; T_target=0)

Return the deterministic on-disk cache path for one decoded spectral day.

The key includes absolute input paths plus file sizes and mtimes, so replacing
either GRIB file automatically invalidates stale coefficient caches. The cache
stores only the truncated coefficient tensors consumed by the transport
preprocessor, not the original GRIB payload.
"""
function spectral_day_cache_path(cache_dir::AbstractString,
                                 vo_d_path::AbstractString,
                                 lnsp_path::AbstractString;
                                 T_target::Int=0)
    vo_stat = stat(vo_d_path)
    lnsp_stat = stat(lnsp_path)
    key = join((
        "spectral-day-v$SPECTRAL_DAY_CACHE_VERSION",
        abspath(vo_d_path), string(vo_stat.size), string(vo_stat.mtime),
        abspath(lnsp_path), string(lnsp_stat.size), string(lnsp_stat.mtime),
        "T_target=$T_target",
    ), "\0")
    return joinpath(cache_dir, "era5_spectral_" * bytes2hex(sha1(key)) * ".jld2")
end

function _load_spectral_day_cache(path::AbstractString)
    data = JLD2.load(path)
    Int(data["format_version"]) == SPECTRAL_DAY_CACHE_VERSION ||
        error("unsupported spectral cache version in $path")
    hours = Vector{Int}(data["hours"])
    lnsp_all = Dict{Int, Matrix{ComplexF64}}(data["lnsp_all"])
    vo_by_hour = Dict{Int, Array{ComplexF64, 3}}(data["vo_by_hour"])
    d_by_hour = Dict{Int, Array{ComplexF64, 3}}(data["d_by_hour"])
    T = Int(data["T"])
    n_times = Int(data["n_times"])
    return _validate_cached_spectral_day(
        (; hours, lnsp_all, vo_by_hour, d_by_hour, T, n_times))
end

function _write_spectral_day_cache(path::AbstractString, spec)
    mkpath(dirname(path))
    tmp = path * ".tmp-$(getpid())"
    isfile(tmp) && rm(tmp; force=true)
    try
        JLD2.jldsave(tmp;
            format_version = SPECTRAL_DAY_CACHE_VERSION,
            hours = spec.hours,
            lnsp_all = spec.lnsp_all,
            vo_by_hour = spec.vo_by_hour,
            d_by_hour = spec.d_by_hour,
            T = spec.T,
            n_times = spec.n_times)
        mv(tmp, path; force=true)
    catch
        isfile(tmp) && rm(tmp; force=true)
        rethrow()
    end
    return path
end

"""
    read_day_spectral(vo_d_path, lnsp_path; T_target=0, cache_dir="")

Read one ERA5 spectral day, optionally using a persistent decoded-coefficient
cache. Empty `cache_dir` disables disk caching and preserves the historical
direct-GRIB path. Cache writes are best-effort: a failed write logs a warning
but never invalidates the decoded in-memory result.
"""
function read_day_spectral(vo_d_path::String, lnsp_path::String;
                           T_target::Int=0,
                           cache_dir::AbstractString="")
    if !isempty(cache_dir)
        path = spectral_day_cache_path(cache_dir, vo_d_path, lnsp_path; T_target)
        if isfile(path)
            try
                spec = _load_spectral_day_cache(path)
                @info "  Spectral cache hit: $(path)"
                return spec
            catch err
                @warn "  Spectral cache unreadable; rebuilding from GRIB" path exception=(err, catch_backtrace())
            end
        end

        spec = read_day_spectral_streaming(vo_d_path, lnsp_path; T_target)
        try
            _write_spectral_day_cache(path, spec)
            @info "  Spectral cache wrote: $(path)"
        catch err
            @warn "  Spectral cache write failed; continuing without cache" path exception=(err, catch_backtrace())
        end
        return spec
    end

    return read_day_spectral_streaming(vo_d_path, lnsp_path; T_target)
end

"""
Streaming spectral reader: reads one day's GRIB data, grouped by hour.
Returns a NamedTuple with hours, lnsp_all, vo_by_hour, d_by_hour, T, n_times.
"""
function read_day_spectral_streaming(vo_d_path::String, lnsp_path::String;
                                     T_target::Int=0)
    # Read T from first LNSP message
    f = GribFile(lnsp_path)
    local T_file::Int
    try
        msg1 = first(f)
        T_file = msg1["J"]
    finally
        destroy(f)
    end
    T = T_target > 0 ? min(T_target, T_file) : T_file
    Nlevels = _ERA5_SPECTRAL_LEVEL_COUNT

    # Read all LNSP hours
    lnsp_all = Dict{Int, Matrix{ComplexF64}}()
    spec_buf = zeros(ComplexF64, T_file + 1, T_file + 1)
    vals_buf = Float64[]
    f = GribFile(lnsp_path)
    try
        for msg in f
            hour = div(msg["dataTime"], 100)
            haskey(lnsp_all, hour) && throw(ArgumentError(
                "duplicate ERA5 LNSP message for hour $hour"))
            read_spectral_coeffs!(spec_buf, msg, vals_buf)
            lnsp_all[hour] = copy(@view spec_buf[1:T + 1, 1:T + 1])
        end
    finally
        destroy(f)
    end

    # Read all VO/D hours
    vo_by_hour = Dict{Int, Array{ComplexF64, 3}}()
    d_by_hour  = Dict{Int, Array{ComplexF64, 3}}()
    vo_seen = Dict{Int, BitVector}()
    d_seen = Dict{Int, BitVector}()
    f = GribFile(vo_d_path)
    try
        for msg in f
            name = msg["shortName"]
            level = msg["level"]
            hour = div(msg["dataTime"], 100)
            1 <= level <= Nlevels || throw(ArgumentError(
                "ERA5 spectral $name message has level=$level outside 1:$Nlevels"))
            read_spectral_coeffs!(spec_buf, msg, vals_buf)
            if name == "vo"
                if !haskey(vo_by_hour, hour)
                    vo_by_hour[hour] = zeros(ComplexF64, T + 1, T + 1, Nlevels)
                end
                seen = get!(vo_seen, hour, falses(Nlevels))
                seen[level] && throw(ArgumentError(
                    "duplicate ERA5 spectral vo message for hour $hour level $level"))
                seen[level] = true
                vo_by_hour[hour][:, :, level] .= @view spec_buf[1:T + 1, 1:T + 1]
            elseif name == "d"
                if !haskey(d_by_hour, hour)
                    d_by_hour[hour] = zeros(ComplexF64, T + 1, T + 1, Nlevels)
                end
                seen = get!(d_seen, hour, falses(Nlevels))
                seen[level] && throw(ArgumentError(
                    "duplicate ERA5 spectral d message for hour $hour level $level"))
                seen[level] = true
                d_by_hour[hour][:, :, level] .= @view spec_buf[1:T + 1, 1:T + 1]
            end
        end
    finally
        destroy(f)
    end

    hours = _validate_spectral_coverage(keys(lnsp_all), vo_seen, d_seen)
    return (; hours, lnsp_all, vo_by_hour, d_by_hour, T, n_times=length(hours))
end

function read_hour0_spectral(spectral_dir::String, date::Date;
                             T_target::Int=0,
                             cache_dir::AbstractString="")
    date_str = Dates.format(date, "yyyymmdd")
    vo_d_path = joinpath(spectral_dir, "era5_spectral_$(date_str)_vo_d.gb")
    lnsp_path = joinpath(spectral_dir, "era5_spectral_$(date_str)_lnsp.gb")

    has_vo_d = isfile(vo_d_path)
    has_lnsp = isfile(lnsp_path)
    if !has_vo_d && !has_lnsp
        return nothing
    end
    vo_d_status = has_vo_d ? "present" : "missing"
    lnsp_status = has_lnsp ? "present" : "missing"
    has_vo_d == has_lnsp || throw(ArgumentError(
        "incomplete ERA5 spectral hour-0 input for $date: " *
        "vo_d=$vo_d_status, lnsp=$lnsp_status"))

    if !isempty(cache_dir)
        path = spectral_day_cache_path(cache_dir, vo_d_path, lnsp_path; T_target)
        if isfile(path)
            try
                spec = _load_spectral_day_cache(path)
                0 in spec.hours || throw(ArgumentError("spectral cache has no hour 0"))
                return (lnsp=spec.lnsp_all[0],
                        vo=spec.vo_by_hour[0],
                        d=spec.d_by_hour[0],
                        T=spec.T)
            catch err
                @warn "  Spectral hour-0 cache unreadable; falling back to GRIB" path exception=(err, catch_backtrace())
            end
        end
    end

    # Read T from first LNSP message
    f = GribFile(lnsp_path)
    local T_file::Int
    try
        msg1 = first(f)
        T_file = msg1["J"]
    finally
        destroy(f)
    end
    T = T_target > 0 ? min(T_target, T_file) : T_file
    Nlevels = _ERA5_SPECTRAL_LEVEL_COUNT

    # Read hour-0 LNSP
    spec_buf = zeros(ComplexF64, T_file + 1, T_file + 1)
    vals_buf = Float64[]
    lnsp_h0 = nothing
    f = GribFile(lnsp_path)
    try
        for msg in f
            hour = div(msg["dataTime"], 100)
            if hour == 0
                read_spectral_coeffs!(spec_buf, msg, vals_buf)
                lnsp_h0 = copy(@view spec_buf[1:T + 1, 1:T + 1])
                break
            end
        end
    finally
        destroy(f)
    end
    lnsp_h0 === nothing && throw(ArgumentError(
        "ERA5 spectral LNSP file for $date contains no hour-0 message: $lnsp_path"))

    # Read hour-0 VO/D
    vo_h0 = zeros(ComplexF64, T + 1, T + 1, Nlevels)
    d_h0 = zeros(ComplexF64, T + 1, T + 1, Nlevels)
    vo_seen = falses(Nlevels)
    d_seen = falses(Nlevels)
    f = GribFile(vo_d_path)
    try
        for msg in f
            hour = div(msg["dataTime"], 100)
            hour == 0 || continue
            name = msg["shortName"]
            level = msg["level"]
            1 <= level <= Nlevels || throw(ArgumentError(
                "ERA5 spectral $name hour-0 message has level=$level outside 1:$Nlevels"))
            read_spectral_coeffs!(spec_buf, msg, vals_buf)
            if name == "vo"
                vo_seen[level] && throw(ArgumentError(
                    "duplicate ERA5 spectral vo hour-0 level $level"))
                vo_seen[level] = true
                vo_h0[:, :, level] .= @view spec_buf[1:T + 1, 1:T + 1]
            elseif name == "d"
                d_seen[level] && throw(ArgumentError(
                    "duplicate ERA5 spectral d hour-0 level $level"))
                d_seen[level] = true
                d_h0[:, :, level] .= @view spec_buf[1:T + 1, 1:T + 1]
            end
        end
    finally
        destroy(f)
    end

    missing_vo = findall(!, vo_seen)
    missing_d = findall(!, d_seen)
    isempty(missing_vo) || throw(ArgumentError(
        "ERA5 spectral hour 0 is missing vo levels $missing_vo"))
    isempty(missing_d) || throw(ArgumentError(
        "ERA5 spectral hour 0 is missing d levels $missing_d"))
    return (lnsp=lnsp_h0, vo=vo_h0, d=d_h0, T=T)
end
