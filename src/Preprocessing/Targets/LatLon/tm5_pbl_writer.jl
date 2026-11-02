# Shared LL/RG opt-in input assembly; transport mass/flux generation is unchanged.
"Zero/neutral sample with valid shapes for declaring the optional v5 writer layout."
function tm5_pbl_sample(::Type{FT}, shape) where FT
    surface_shape = shape[1:end-1]
    return (;t=fill(FT(280),shape),q=zeros(FT,shape),lsm=zeros(FT,surface_shape),
        sr=fill(FT(0.01),surface_shape),fsr=fill(FT(0.01),surface_shape),
        u10=zeros(FT,surface_shape),v10=zeros(FT,surface_shape),ewss=zeros(FT,surface_shape),
        nsss=zeros(FT,surface_shape),sshf=zeros(FT,surface_shape),slhf=zeros(FT,surface_shape),oro=zeros(FT,surface_shape))
end

"One open thermo file, its point weights and a bounded block of sampled endpoints."
mutable struct TM5PBLThermoReader
    dataset::NCDataset
    weights::LatLonBilinearWeights
    ny::Int
    batch::Int
    records::UnitRange{Int}
    values::Vector{Tuple{Matrix{Float64},Matrix{Float64}}}
end

"Day-local ownership of thermo/surface readers, shared endpoints and remap weights."
mutable struct TM5PBLInputCache
    readers::Dict{Date,ERA5SurfaceReader}
    endpoint::Any
    regridder::Any
    thermo_readers::Dict{String,TM5PBLThermoReader}
end
TM5PBLInputCache(readers,endpoint,regridder) =
    TM5PBLInputCache(readers,endpoint,regridder,Dict{String,TM5PBLThermoReader}())
TM5PBLInputCache() = TM5PBLInputCache(Dict{Date,ERA5SurfaceReader}(),nothing,nothing)
function Base.close(cache::TM5PBLInputCache)
    errors = Any[]
    try
        for reader in values(cache.readers)
            try
                close_era5_surface_reader(reader)
            catch err
                push!(errors,err)
            end
        end
    finally
        try
            for reader in values(cache.thermo_readers)
                try
                    close(reader.dataset)
                catch err
                    push!(errors,err)
                end
            end
        finally
            empty!(cache.readers)
            empty!(cache.thermo_readers)
            cache.endpoint = nothing
            cache.regridder = nothing
        end
    end
    isempty(errors) || throw(length(errors) == 1 ? only(errors) : CompositeException(errors))
    return nothing
end
function _tm5_reader!(cache,settings,date)
    return get!(cache.readers,date) do
        open_tm5_pbl_surface_reader(settings.surface_dir,date)
    end
end

function _tm5_thermo_reader!(cache,path,target_lons,target_lats)
    return get!(cache.thermo_readers,path) do
        ds = NCDataset(path)
        try
            units_t = lowercase(String(get(ds["t"].attrib,"units","")))
            units_q = lowercase(replace(String(get(ds["q"].attrib,"units",""))," "=>"", "**"=>"^"))
            units_t in ("k","kelvin") || throw(ArgumentError("TM5 thermo t requires Kelvin"))
            units_q in ("kgkg-1","kgkg^-1","kg/kg","1") || throw(ArgumentError("TM5 thermo q requires kg/kg total air"))
            lons,lats = read_thermo_coordinates(ds)
            weights = LatLonBilinearWeights(lons,lats,target_lons,target_lats)
            # Bound native scratch even when an archive chunks an entire day.
            batch = minimum(("t","q")) do name
                storage,chunks = chunking(ds[name])
                time_dim = first(dimnames(ds[name])) == "longitude" ? 4 : 1
                storage == :chunked ? min(3,chunks[time_dim]) : 1
            end
            return TM5PBLThermoReader(ds,weights,length(lats),batch,1:0,
                Tuple{Matrix{Float64},Matrix{Float64}}[])
        catch
            close(ds)
            rethrow()
        end
    end
end

function _tm5_thermo_endpoint(path,record,vertical,target_lons,target_lats,cache;
                              last_record=24)
    key = (path,record)
    cache.endpoint !== nothing && cache.endpoint.key == key && return cache.endpoint.value
    reader = _tm5_thermo_reader!(cache,path,target_lons,target_lats)
    if !(record in reader.records)
        first_record = record-mod(record-1,reader.batch)
        records = first_record:min(first_record+reader.batch-1,last_record)
        sampled = map(("t","q")) do variable
            fields = Matrix{Float64}[]
            foreach_thermo_record(reader.dataset,records,reader.ny;variable) do a, _
                dst = zeros(Float64,length(target_lons),length(vertical.level_range))
                interpolate_ll_qv!(dst,view(a,:,:,vertical.level_range),reader.weights)
                push!(fields,dst)
            end
            fields
        end
        reader.values = collect(zip(sampled...))
        reader.records = records
    end
    value = reader.values[record-first(reader.records)+1]
    cache.endpoint = (;key,value)
    return value
end

"""
    load_tm5_pbl_target_window(date, win, settings, vertical, ps, grid, shape; cache)

Assemble the optional LL/RG v5 group. T/q are interpolated at target cell
centres, mass-weighted onto merged layers at each hourly endpoint, then
averaged in time. Native moist Δp at the window surface pressure supplies
the merge weights. Surface primitives are intensive cell-area means through
Regridding.build_regridder/apply_regridder! on the actual LL/RG target mesh.
A day-local cache reuses the shared thermo endpoint, surface readers and weights.
"""
function load_tm5_pbl_target_window(date,win,settings,vertical,ps,grid,shape;
                                    cache::Union{Nothing,TM5PBLInputCache}=nothing)
    owned = cache === nothing
    owned && (cache = TM5PBLInputCache())
    try
        return _load_tm5_pbl_target_window(date,win,settings,vertical,ps,grid,shape,cache)
    finally
        owned && close(cache)
    end
end

function _load_tm5_pbl_target_window(date,win,settings,vertical,ps,grid,shape,cache)
    lons,lats = tm5_target_coordinates(grid)
    FT = settings.output_float_type
    hours_per_day = 24
    win <= hours_per_day || throw(ArgumentError("TM5 PBL expects hourly windows"))
    settings.met_interval == 3600 || throw(ArgumentError("TM5 PBL input assembly requires hourly analyses"))
    date2 = win==hours_per_day ? date+Day(1) : date
    record2 = win==hours_per_day ? 1 : win+1
    path = era5_thermo_path(settings.thermo_dir,date)
    path2 = era5_thermo_path(settings.thermo_dir,date2)
    t0,q0 = _tm5_thermo_endpoint(path,win,vertical,lons,lats,cache)
    t1,q1 = _tm5_thermo_endpoint(path2,record2,vertical,lons,lats,cache;
        last_record=date2 == date ? hours_per_day : 1)
    nc,nz = length(lons),shape[end]
    t,q,weights = zeros(Float64,nc,nz),zeros(Float64,nc,nz),zeros(Float64,nc,nz)
    # IntensiveCenterField semantics: sum(value * moist mass) / sum(moist mass).
    for (temperature,humidity) in ((t0,q0),(t1,q1))
        fill!(weights,0)
        tt,qq = zeros(Float64,nc,nz),zeros(Float64,nc,nz)
        for k in 1:vertical.Nz_native, c in 1:nc
            m = vertical.ab.dA[k] + vertical.ab.dB[k]*ps[c]
            target = vertical.merge_map[k]
            tt[c,target] += m*temperature[c,k]
            qq[c,target] += m*humidity[c,k]
            weights[c,target] += m
        end
        t .+= tt ./ weights ./ 2
        q .+= qq ./ weights ./ 2
    end
    reader = _tm5_reader!(cache,settings,date)
    endpoint = _tm5_reader!(cache,settings,date2)
    surface = load_tm5_pbl_window(reader,endpoint,win,Float64;endpoint_record=record2,
        interval_seconds=settings.met_interval)
    source = cache.regridder === nothing ? era5_surface_source_mesh(reader,grid.mesh.radius) : nothing
    means,cache.regridder = target_surface_means(source,grid,surface;regridder=cache.regridder)
    # Area means of a fraction leave [0,1] by rounding (≈3e-15 in all-land cells);
    # larger excursions are left for the binary validator to reject.
    snap(x) = -1e-12 <= x <= 1+1e-12 ? clamp(x,0,1) : x
    means = merge(means,(;lsm=snap.(means.lsm)))
    sampled = map(a->reshape(FT.(a),shape[1:end-1]),means)
    return (;t=reshape(FT.(t),shape),q=reshape(FT.(q),shape),sampled...)
end


"Coordinate lists in Julia cell order for a structured target."
function tm5_target_coordinates(grid::LatLonTargetGeometry)
    lons = [lon for lat in grid.lats, lon in grid.lons]
    lats = [lat for lat in grid.lats, lon in grid.lons]
    return vec(permutedims(lons)),vec(permutedims(lats))
end

function tm5_target_coordinates(grid::ReducedGaussianTargetGeometry)
    lons = reduce(vcat,grid.lons_by_ring)
    lats = reduce(vcat,[fill(grid.lats[j],grid.nlon_per_ring[j]) for j in eachindex(grid.lats)])
    return lons,lats
end
