"""
    available_spectral_dates(spectral_dir) -> Vector{Date}

Discover all processing dates available in the spectral-input directory by
scanning for `era5_spectral_YYYYMMDD_lnsp.gb`.
"""
function available_spectral_dates(spectral_dir::String)
    dates = Date[]
    for f in readdir(spectral_dir)
        m = match(r"era5_spectral_(\d{8})_lnsp\.gb", f)
        m !== nothing && push!(dates, Date(m[1], dateformat"yyyymmdd"))
    end
    sort!(dates)
    isempty(dates) && error("No spectral GRIB files found in $spectral_dir")
    return dates
end

"""
    next_day_hour0(date, dates, spectral_dir, T_target)

Load the next day's hour-0 spectral fields when available so the final window of
the current day can form forward differences for `dam`, `dbm`, and `dm`.
Returns `nothing` when no next-day file is available.
"""
function next_day_hour0(date::Date, dates, spectral_dir::String, T_target::Int;
                        cache_dir::AbstractString="")
    next_day = date + Day(1)
    next_path = era5_spectral_paths(spectral_dir, next_day).lnsp
    if next_day in dates || isfile(next_path)
        return read_hour0_spectral(spectral_dir, next_day; T_target, cache_dir)
    end
    return nothing
end
