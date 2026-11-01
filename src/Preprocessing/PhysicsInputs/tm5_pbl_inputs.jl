# ERA5 hourly reanalysis accumulations are interval totals, not cumulative
# forecast-since-initialisation records. Divide only the selected interval.
function _tm5_surface_units(ds, name, key)
    units = lowercase(replace(String(get(ds[name].attrib, "units", "")), " "=>"", "**"=>"^"))
    valid = if key in (:sr,:fsr)
        units in ("m", "metre", "meter")
    elseif key === :lsm
        units in ("1", "(0-1)", "0-1")
    elseif key in (:u10,:v10)
        units in ("ms-1", "ms^-1", "m/s")
    elseif key === :oro
        units in ("m2s-2", "m^2s^-2", "m2/s2")
    elseif key in (:hflux,:lhflux)
        units in ("jm-2", "jm^-2", "j/m2", "wm-2", "wm^-2", "w/m2")
    else
        units in ("nm-2s", "nm^-2s", "nsm-2", "nsm^-2", "nm-2", "nm^-2", "pa", "pas")
    end
    valid || throw(ArgumentError("TM5 PBL $name has unsupported units $(repr(units))"))
    return units
end

function _tm5_surface_value(reader, key, record, ::Type{FT}) where FT
    binding = _surface_var_binding(reader, key; required = key !== :sr)
    # ERA5 exports fsr only in many archives: both roughness roles then use fsr.
    binding === nothing && (binding = _surface_var_binding(reader, :fsr))
    ds, name = binding
    units = _tm5_surface_units(ds, name, key)
    idx = _surface_time_index(reader, ds, record)
    value = _normalize_surface_ll!(_surface_slice_to_FT(ds, name, idx, FT), ds)
    all(isfinite, value) || throw(ArgumentError("TM5 PBL $name contains non-finite values"))
    return value, units
end

"""
    load_tm5_pbl_window(reader, endpoint_reader, record, FT=Float64;
                       endpoint_record=record+1, interval_seconds)

Ten surface primitives for live Calc_Kzz. Instantaneous winds and roughness
are means of hourly analysis endpoints; lsm and orography are constant.
Hourly reanalysis accumulated heat/stress fields are the record ending at
the right endpoint and are divided by the actual interval. Heat stays
ECMWF downward-positive. Already exported W/m² and N/m² pass unchanged.
The endpoint reader can be the next day's reader at midnight. `fsr` replaces
TM5's Olsson roughness; when `sr` is absent, fsr supplies both roles.
"""
function load_tm5_pbl_window(reader::ERA5SurfaceReader, endpoint_reader::ERA5SurfaceReader,
                              record::Int, ::Type{FT}=Float64;
                              endpoint_record::Int=record+1,
                              interval_seconds::Real) where FT
    isfinite(interval_seconds) && interval_seconds > 0 || throw(ArgumentError("TM5 PBL interval must be positive"))
    reader.lons == endpoint_reader.lons && reader.lats == endpoint_reader.lats ||
        throw(ArgumentError("TM5 PBL endpoint surface grids differ"))
    # Accumulations are labelled by their END time in ERA5 hourly reanalysis.
    accumulated = function(key)
        a, units = _tm5_surface_value(endpoint_reader,key,endpoint_record,FT)
        is_total = key in (:hflux,:lhflux) ? startswith(units,"j") :
                   units in ("nm-2s","nm^-2s","nsm-2","nsm^-2","pas")
        return is_total ? a ./ FT(interval_seconds) : a
    end
    averaged = function(key)
        a,_ = _tm5_surface_value(reader,key,record,FT)
        b,_ = _tm5_surface_value(endpoint_reader,key,endpoint_record,FT)
        return (a .+ b) ./ FT(2)
    end
    lsm,_ = _tm5_surface_value(reader,:lsm,record,FT)
    oro,_ = _tm5_surface_value(reader,:oro,record,FT)
    all(x->0<=x<=1,lsm) || throw(ArgumentError("ERA5 lsm must be a fraction in [0,1]"))
    return (;lsm,sr=averaged(:sr),fsr=averaged(:fsr),u10=averaged(:u10),v10=averaged(:v10),
        ewss=accumulated(:ewss),nsss=accumulated(:nsss),sshf=accumulated(:hflux),slhf=accumulated(:lhflux),oro)
end

"Open native-resolution TM5 surface inputs, supporting plain, split ZIP and ARCO NetCDF layouts."
function open_tm5_pbl_surface_reader(dir,date)
    arco = joinpath(expand_data_path(dir),"arco",Dates.format(date,"yyyymmdd"))
    if isdir(arco)
        paths = sort!(filter(f->endswith(lowercase(f),".nc"),readdir(arco;join=true)))
        isempty(paths) && throw(ArgumentError("TM5 PBL ARCO surface directory is empty: $arco"))
        path,tmp = arco,nothing
    else
        path = resolve_era5_surface_path(dir,date)
        paths,tmp = _surface_nc_paths(path)
    end
    dims = try
        NCDataset(first(paths)) do ds
            (length(ds["longitude"]),length(ds["latitude"]))
        end
    catch
        tmp === nothing || rm(tmp;recursive=true,force=true)
        rethrow()
    end
    return _open_era5_surface_reader_from(paths,tmp,path,date,dims...)
end
