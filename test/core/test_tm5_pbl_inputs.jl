using Test, Dates, NCDatasets
using AtmosTransport
const PI=AtmosTransport.Preprocessing.PhysicsInputs
const LL=AtmosTransport.Preprocessing.LatLonTarget
const Pre=AtmosTransport.Preprocessing

function write_pbl_surface(path;heat_units="J m-2",stress_units="N m-2 s",time_offset=nothing)
    NCDataset(path,"c") do ds
        ds.dim["longitude"]=4;ds.dim["latitude"]=3;ds.dim["time"]=24
        defVar(ds,"longitude",[0.,90.,180.,270.],("longitude",))
        defVar(ds,"latitude",[90.,0.,-90.],("latitude",))
        defVar(ds,"time",0:23,("time",))
        fields=(("lsm",.4,"1"),("fsr",.1,"m"),("u10",6.,"m s-1"),("v10",2.,"m s-1"),
            ("ewss",900.,stress_units),("nsss",-450.,stress_units),("sshf",-90000.,heat_units),
            ("slhf",-45000.,heat_units),("z",9806.65,"m2 s-2"))
        for (name,value,units) in fields
            a=fill(value,4,3,24)
            name=="u10" && (a[:,:,2].=10)
            if name=="u10" && time_offset !== nothing
                for h in 1:24
                    a[:,:,h] .= 6+time_offset+h-1
                end
            end
            defVar(ds,name,a,("longitude","latitude","time");attrib=Dict("units"=>units))
        end
    end
end
function write_pbl_thermo(path;exact=false,
        temperatures=exact ? [256.,256.] : [280.,300.],
        humidities=exact ? [2.0^-9,2.0^-7] : [.002,.01])
    nz=length(temperatures)
    @assert length(humidities)==nz
    NCDataset(path,"c") do ds
        ds.dim["longitude"]=4;ds.dim["latitude"]=3;ds.dim["level"]=nz;ds.dim["time"]=24
        defVar(ds,"longitude",[0.,90.,180.,270.],("longitude",))
        defVar(ds,"latitude",[90.,0.,-90.],("latitude",))
        t=[temperatures[k] for i in 1:4,j in 1:3,k in 1:nz,h in 1:24]
        q=[humidities[k] for i in 1:4,j in 1:3,k in 1:nz,h in 1:24]
        defVar(ds,"t",t,("longitude","latitude","level","time");attrib=Dict("units"=>"K"))
        defVar(ds,"q",q,("longitude","latitude","level","time");attrib=Dict("units"=>"kg kg-1"))
    end
end

@testset "TM5 chunked thermo preserves scalar reads and endpoint ownership" begin
    mktempdir() do dir
        date = Date(2021,12,1)
        vertical = (;Nz_native=4,Nz=2,level_range=1:4,
            ab=(;dA=[1000.,2000.,3000.,4000.],dB=[.1,.2,.3,.4]),merge_map=[1,1,2,2])
        grid = Pre.build_target_geometry(Dict("type"=>"latlon","nlon"=>4,"nlat"=>3),Float64)
        lons,lats = LL.tm5_target_coordinates(grid)
        # Both supported disk dimension orders, with both axis orientations.
        for longitude_first in (true,false), centered in (true,false)
            for day in (date,date+Day(1))
                path = Pre.Spectral.era5_thermo_path(dir,day)
                NCDataset(path,"c") do ds
                    for (name,n) in (("longitude",4),("latitude",3),("level",4),("time",24))
                        defDim(ds,name,n)
                    end
                    lon = centered ? [-180.,-90.,0.,90.] : [0.,90.,180.,270.]
                    lat = centered ? [-90.,0.,90.] : [90.,0.,-90.]
                    defVar(ds,"longitude",lon,("longitude",))
                    defVar(ds,"latitude",lat,("latitude",))
                    dims = longitude_first ? ("longitude","latitude","level","time") :
                        ("time","level","latitude","longitude")
                    chunks = longitude_first ? (2,2,2,3) : (3,2,2,2)
                    for (name,units) in (("t","K"),("q","kg kg-1"))
                        a = [Float32(name == "t" ? 250+sin(i+2j+3k+h)+h+Dates.day(day) :
                            .001+.00001*(i+2j+3k+h+Dates.day(day))) for i in 1:4,j in 1:3,k in 1:4,h in 1:24]
                        data = longitude_first ? a : permutedims(a,(4,3,2,1))
                        defVar(ds,name,data,dims;chunksizes=chunks,deflatelevel=1,attrib=Dict("units"=>units))
                    end
                end
                write_pbl_surface(joinpath(dir,"era5_surface_$(Dates.format(day,"yyyymmdd")).nc"))
                # The scalar reader remains the independent pre-batching path.
                NCDataset(path) do ds
                    for variable in ("t","q"), records in (1:3,22:24), FT in (Float32,Float64)
                        Pre.Spectral.foreach_thermo_record(ds,records,3;FT,variable) do field,record
                            reference = Pre.read_qv_from_thermo(path,record,4,3,4;FT,variable)
                            @test reinterpret(UInt8,vec(field)) == reinterpret(UInt8,vec(reference))
                        end
                    end
                end
            end
            settings = Pre.ERA5SpectralSettings((;output_float_type=Float64,met_interval=3600.,thermo_dir=dir,surface_dir=dir))
            cache = LL.TM5PBLInputCache()
            readers = LL.TM5PBLThermoReader[]
            try
                previous = nothing
                for win in (1,2,3,22,23,24,1)
                    ps = fill(1e5+win,4,3)
                    cached = LL.load_tm5_pbl_target_window(date,win,settings,vertical,ps,grid,(4,3,2);cache)
                    fresh = LL.load_tm5_pbl_target_window(date,win,settings,vertical,ps,grid,(4,3,2))
                    @test all(name->isequal(getproperty(cached,name),getproperty(fresh,name)),keys(cached))
                    reader = cache.thermo_readers[Pre.Spectral.era5_thermo_path(dir,date)]
                    @test reader.batch == 3
                    if win == 1
                        previous = reader.values
                    elseif win == 2
                        @test reader.values === previous
                        @test reader.records == 1:3
                    end
                end
                append!(readers,values(cache.thermo_readers))
                @test length(readers) == 2
                @test cache.thermo_readers[Pre.Spectral.era5_thermo_path(dir,date+Day(1))].records == 1:1
            finally
                close(cache)
            end
            @test isempty(cache.thermo_readers)
            @test cache.regridder === nothing
            @test all(reader->!isopen(reader.dataset),readers)
        end
    end
end

@testset "TM5 day cleanup closes all readers after surface cleanup failure" begin
    mktempdir() do dir
        date = Date(2021,12,1)
        cache = LL.TM5PBLInputCache()
        surface_readers = PI.ERA5SurfaceReader[]
        thermo_readers = LL.TM5PBLThermoReader[]
        extraction_dirs = String[]
        try
            for day in (date,date+Day(1))
                write_pbl_surface(joinpath(dir,"era5_surface_$(Dates.format(day,"yyyymmdd")).nc"))
                surface = PI.open_era5_surface_reader(dir,day,4,3)
                push!(surface_readers,surface)
                cache.readers[day] = surface
                surface.tmp_dir = mktempdir(dir)
                push!(extraction_dirs,surface.tmp_dir)
                path = Pre.Spectral.era5_thermo_path(dir,day)
                write_pbl_thermo(path)
                push!(thermo_readers,LL._tm5_thermo_reader!(cache,path,[0.],[0.]))
            end
            # Inject a removal failure independent of user privileges/platform.
            # Fail the first reader in actual cleanup order, so later closes matter.
            first(values(cache.readers)).tmp_dir = joinpath(dir,"invalid\0surface")
            cache.endpoint = (;key=("thermo",1),value=(ones(1,2),zeros(1,2)))
            cache.regridder = :retained_weights
            @test_throws ArgumentError close(cache)
            @test all(reader->all(ds->!isopen(ds),reader.datasets),surface_readers)
            @test all(reader->!isopen(reader.dataset),thermo_readers)
            @test isempty(cache.readers)
            @test isempty(cache.thermo_readers)
            @test cache.endpoint === nothing
            @test cache.regridder === nothing
            @test count(isdir,extraction_dirs) == 1
            @test close(cache) === nothing
        finally
            # Also release fixtures if setup or an assertion fails.
            for reader in surface_readers, ds in reader.datasets
                isopen(ds) && close(ds)
            end
            for reader in thermo_readers
                isopen(reader.dataset) && close(reader.dataset)
            end
        end
    end
end

@testset "TM5 surface intervals, units, endpoints and roughness" begin
    mktempdir() do dir
        path=joinpath(dir,"era5_surface_20211201.nc")
        write_pbl_surface(path)
        reader=PI.open_era5_surface_reader(dir,Date(2021,12,1),4,3)
        try
            for FT in (Float32,Float64)
                s=PI.load_tm5_pbl_window(reader,reader,1,FT;interval_seconds=900)
                @test all(==(FT(8)),s.u10)
                @test all(==(FT(-100)),s.sshf)
                @test all(==(FT(-50)),s.slhf)
                @test all(==(FT(1)),s.ewss)
                @test all(==(FT(-.5)),s.nsss)
                @test s.sr==s.fsr
                @test all(==(FT(.4)),s.lsm)
            end
            @test_throws ArgumentError PI.load_tm5_pbl_window(reader,reader,1;interval_seconds=0)
        finally
            PI.close_era5_surface_reader(reader)
        end
        write_pbl_surface(joinpath(dir,"era5_surface_20211202.nc");heat_units="W m-2",stress_units="N m-2")
        r=PI.open_era5_surface_reader(dir,Date(2021,12,2),4,3)
        try
            s=PI.load_tm5_pbl_window(r,r,1;interval_seconds=900)
            @test all(==(-90000),s.sshf)
            @test all(==(900),s.ewss)
        finally
            PI.close_era5_surface_reader(r)
        end
        write_pbl_surface(joinpath(dir,"era5_surface_20211203.nc");heat_units="bad")
        r=PI.open_era5_surface_reader(dir,Date(2021,12,3),4,3)
        try
            @test_throws ArgumentError PI.load_tm5_pbl_window(r,r,1;interval_seconds=900)
        finally
            PI.close_era5_surface_reader(r)
        end
        for day in (1,2)
            write_pbl_thermo(joinpath(dir,"era5_thermo_ml_2021120$(day).nc"))
        end
        settings=Pre.ERA5SpectralSettings((;output_float_type=Float64,met_interval=3600.,thermo_dir=dir,surface_dir=dir))
        vertical=(;Nz_native=2,Nz=1,level_range=1:2,ab=(;dA=[10000.,90000.],dB=[0.,0.]),merge_map=[1,1])
        grids=(Pre.build_target_geometry(Dict("type"=>"latlon","nlon"=>4,"nlat"=>3),Float64),
            Pre.build_target_geometry(Dict("type"=>"synthetic_reduced_gaussian","gaussian_number"=>2,"nlon_mode"=>"octahedral"),Float64))
        for grid in grids
            shape=grid isa Pre.LatLonTargetGeometry ? (4,3,1) : (ncells(grid.mesh),1)
            s=LL.load_tm5_pbl_target_window(Date(2021,12,1),1,settings,vertical,fill(1e5,prod(shape[1:end-1])),grid,shape)
            @test size(s.t)==shape
            @test all(==(298),s.t)
            @test all(x->isapprox(x,.0092),s.q)
            @test all(x->isapprox(x,-25;rtol=1e-13),s.sshf)
        end
    end
end

@testset "TM5 combined LL/RG PBL inputs: pressure_merge=$pressure_merge" for pressure_merge in (false,true)
    mktempdir() do dir
        date=Date(2021,12,1)
        spectral=joinpath(dir,"spectral");cache=joinpath(dir,"cache")
        mkpath(spectral);mkpath(cache)
        vo=joinpath(spectral,"era5_spectral_20211201_vo_d.gb")
        ln=joinpath(spectral,"era5_spectral_20211201_lnsp.gb")
        touch(vo);touch(ln)
        hours=collect(0:23);lnsp=zeros(ComplexF64,2,2);lnsp[1]=log(1e5)
        # A zonal harmonic makes cell-mean and centre-sampled PS differ well
        # beyond rounding, while remaining constant in time with zero winds.
        pressure_merge && (lnsp[2,2]=.15)
        z=zeros(ComplexF64,2,2,137)
        spec=(;hours,lnsp_all=Dict(h=>copy(lnsp) for h in hours),
            vo_by_hour=Dict(h=>copy(z) for h in hours),d_by_hour=Dict(h=>copy(z) for h in hours),T=1,n_times=24)
        Pre.write_spectral_day_cache(Pre.spectral_day_cache_path(cache,vo,ln;T_target=0),spec)
        next=(;lnsp,vo=z,d=z,T=1)
        paths=Pre.Spectral.era5_spectral_paths(spectral,date)
        for hour in hours
            Pre.Spectral.JLD2.jldsave(Pre.Spectral.spectral_hour_cache_path(cache,paths,hour);endpoint=next)
        end
        for day in (1,2)
            write_pbl_surface(joinpath(dir,"era5_surface_2021120$(day).nc");time_offset=24(day-1))
            if pressure_merge
                write_pbl_thermo(joinpath(dir,"era5_thermo_ml_2021120$(day).nc");
                    temperatures=[240.,260.,280.,310.],humidities=[.002,.006,.01,.02])
            else
                # Power-of-two primitives isolate serialization from pressure roundoff.
                write_pbl_thermo(joinpath(dir,"era5_thermo_ml_2021120$(day).nc");exact=true)
            end
        end
        vc=pressure_merge ? HybridSigmaPressure([0.,4000.,7000.,3000.,0.],[0.,.1,.3,.6,1.]) :
            HybridSigmaPressure([0.,0.,0.],[0.,.3,1.])
        retained=pressure_merge ? [1,3,5] : [1,2,3]
        merged_vc=HybridSigmaPressure(vc.A[retained],vc.B[retained])
        nz=length(vc.A)-1
        vertical=(;Nz_native=nz,Nz=2,level_range=1:nz,ab=(;dA=diff(vc.A),dB=diff(vc.B),b_ifc=vc.B),
            merge_map=pressure_merge ? [1,1,2,2] : [1,2],vc_native=vc,merged_vc)
        grids=(Pre.build_target_geometry(Dict("type"=>"latlon","nlon"=>4,"nlat"=>3),Float64),
            Pre.build_target_geometry(Dict("type"=>"synthetic_reduced_gaussian","gaussian_number"=>2,"nlon_mode"=>"octahedral"),Float64))
        BF=AtmosTransport.MetDrivers.BinaryFormat
        MD=AtmosTransport.MetDrivers
        # Exercise the actual CS spectral day writer, including its deferred
        # payload ownership and next-day endpoint. T/q vary by layer; surface
        # u10 varies by hour, so a stale or off-by-one group cannot pass.
        for FT in (Float64,Float32)
            cs=Pre.build_target_geometry(Dict("type"=>"cubed_sphere","Nc"=>2,
                "staging_nlon"=>4,"staging_nlat"=>3,"regridder_cache_dir"=>joinpath(dir,"cs_weights")),Float64)
            settings=Pre.ERA5SpectralSettings((;column_balance_weights=:mass,output_float_type=FT,
                spectral_dir=spectral,spectral_cache_dir=cache,T_target=0,min_dp=0.,include_qv=false,
                mass_basis=:dry,mass_fix_enable=false,target_ps_dry_pa=98325.,qv_global_climatology=0.,
                thermo_dir=dir,surface_dir=dir,half_dt=450.,met_interval=3600.,dt=900.,
                out_dir=joinpath(dir,"cs$(FT)"),include_surface=false,include_tm5_pbl=true,
                qv_preload=false,qv_preload_max_bytes=0,face_lengths=:edge,face_fluxes=:vector,
                face_interpolation=:fv3,wind_regrid=:cartesian,flux_time_sampling=:window_mean))
            path=Pre.process_day(date,cs,settings,vertical;next_day_hour0=next)
            driver=MD.TransportBinaryDriver(path;FT)
            try
                @test driver.reader.header.flux_construction isa BF.PointSampledSpectralFluxesHeader
                for win in (1,2,24)
                    window=MD.load_transport_window(driver,win)
                    @test all(a->all(x->isapprox(x,6+win-.5;rtol=1e-6),a),window.tm5_pbl.u10)
                    expected=LL.load_tm5_pbl_target_window(date,win,settings,vertical,
                        reduce(vcat,map(vec,window.surface_pressure)),cs,(24,2))
                    for name in BF.TM5_PBL_FIELD_NAMES
                        actual=getproperty(window.tm5_pbl,name)
                        flattened=name in (:t,:q) ? vcat(map(a->reshape(a,4,2),actual)...) : reduce(vcat,map(vec,actual))
                        @test flattened ≈ getproperty(expected,name) rtol=8eps(FT)
                    end
                end
            finally
                close(driver.reader)
            end
        end
        for (i,grid) in enumerate(grids), FT in (Float64,Float32)
            drivers=[]
            try
                for construction in (Pre.PointSampledSpectralFluxes(),Pre.TM5SpectralFluxes())
                    out=joinpath(dir,"out$(i)_$(FT)_$(typeof(construction))")
                    settings=Pre.ERA5SpectralSettings((;column_balance_weights=:mass,output_float_type=FT,
                        spectral_dir=spectral,spectral_cache_dir=cache,T_target=0,min_dp=0.,include_qv=false,
                        mass_basis=:dry,mass_fix_enable=false,target_ps_dry_pa=98325.,qv_global_climatology=0.,
                        thermo_dir=dir,surface_dir=dir,half_dt=450.,met_interval=3600.,dt=900.,out_dir=out,
                        include_surface=false,include_tm5_pbl=true,qv_preload=false,qv_preload_max_bytes=0,
                        spectral_flux_construction=construction,spectral_synthesis=Pre.Common.CPUSynthesis(),
                        flux_time_sampling=:window_mean))
                    path=Pre.process_day(date,grid,settings,vertical;next_day_hour0=next)
                    driver=MD.TransportBinaryDriver(path;FT)
                    push!(drivers,driver)
                    @test driver.reader.header.format_version==5
                    @test all(s->s in driver.reader.header.payload_sections,BF.TM5_PBL_PAYLOAD_SECTIONS)
                    @test driver.reader.header.flux_construction isa
                        (construction isa Pre.TM5SpectralFluxes ? BF.TM5SpectralFluxesHeader : BF.PointSampledSpectralFluxesHeader)
                end
                for win in 1:24
                    windows=map(d->MD.load_transport_window(d,win),drivers)
                    point,combined=map(w->w.tm5_pbl,windows)
                    for name in BF.TM5_PBL_FIELD_NAMES
                        pressure_merge && name in (:t,:q) && continue
                        @test isequal(getproperty(combined,name),getproperty(point,name))
                    end
                    if pressure_merge
                        for window in windows
                            ps=vec(window.surface_pressure)
                            @test maximum(ps)-minimum(ps)>1000
                            # Independent reference: difference hybrid interface
                            # pressures, then take moist-mass means over each pair.
                            # Area/g cancels; humidity must not dry these weights.
                            for (name,profile) in ((:t,[240.,260.,280.,310.]),(:q,[.002,.006,.01,.02]))
                                expected=[begin
                                    interfaces=vc.A .+ vc.B.*Float64(ps[c])
                                    dp=diff(interfaces)[levels]
                                    sum(profile[levels].*dp)/sum(dp)
                                end for c in eachindex(ps),levels in (1:2,3:4)]
                                actual=reshape(getproperty(window.tm5_pbl,name),:,2)
                                @test actual ≈ FT.(expected) rtol=8eps(FT)
                            end
                        end
                        changed=findall(abs.(vec(windows[1].surface_pressure).-vec(windows[2].surface_pressure)).>100)
                        @test !isempty(changed)
                        for name in (:t,:q)
                            a=reshape(getproperty(point,name),:,2)
                            b=reshape(getproperty(combined,name),:,2)
                            @test all(abs.(a[changed,:].-b[changed,:]).>8eps(FT).*abs.(a[changed,:]))
                        end
                        if win==1
                            @info "TM5 pressure-sensitive PBL merge" target=typeof(grid) FT ps_range=extrema(windows[2].surface_pressure) max_ps_difference=maximum(abs,windows[1].surface_pressure.-windows[2].surface_pressure) max_t_difference=maximum(abs,point.t.-combined.t) max_q_difference=maximum(abs,point.q.-combined.q)
                        end
                    else
                        @test all(==(FT(256)),combined.t)
                    end
                    # Every surface hour differs, including next-day 00 UTC.
                    @test all(x->isapprox(x,6+win-.5;rtol=1e-6),combined.u10)
                    @test all(x->isapprox(x,-25;rtol=1e-6),combined.sshf)
                end
                if grid isa Pre.LatLonTargetGeometry
                    cs=Pre.build_target_geometry(Dict("type"=>"cubed_sphere","Nc"=>2,
                        "regridder_cache_dir"=>joinpath(dir,"regrid_weights")),FT)
                    remapped=Pre.regrid_ll_binary_to_cs(drivers[1].reader.path,cs,joinpath(dir,"regrid$(FT).bin");FT)
                    reader=MD.TransportBinaryReader(remapped;FT)
                    try
                        @test reader.header.flux_construction isa BF.RegriddedTransportBinaryHeader
                        @test reader.header.flux_construction.source_flux_construction=="point_sampled_spectral"
                        raw=MD.load_window!(reader,2)
                        @test all(a->all(x->isapprox(x,7.5;rtol=1e-6),a),raw.tm5_pbl.u10)
                        @test all(a->all(isfinite,a),raw.tm5_pbl.t)
                        # Independent intensive-density formulation of the
                        # moist-mass mean (no production extensive adapter).
                        source=MD.load_transport_window(drivers[1],2)
                        R=AtmosTransport.Regridding
                        r=R.build_regridder(MD.driver_grid(drivers[1]).horizontal,cs.mesh)
                        q=reshape(Float64.(source.tm5_pbl.q),:,2)
                        density=reshape(Float64.(source.air_mass),:,2)./(1 .- q)./r.src_areas
                        denominator=zeros(length(r.dst_areas),2)
                        R.apply_regridder!(denominator,r,density)
                        for name in (:t,:q)
                            numerator=similar(denominator)
                            intensive=reshape(Float64.(getproperty(source.tm5_pbl,name)),:,2)
                            R.apply_regridder!(numerator,r,intensive.*density)
                            actual=vcat(map(a->reshape(a,4,2),getproperty(raw.tm5_pbl,name))...)
                            @test actual ≈ numerator./denominator rtol=8eps(FT)
                        end
                    finally
                        close(reader)
                    end
                end
            finally
                foreach(d->close(d.reader),drivers)
            end
        end
    end
end

@testset "TM5 coastal area means, orientation and shared endpoints" begin
    mktempdir() do dir
        for day in (1,2)
            surface=joinpath(dir,"era5_surface_2021120$(day).nc")
            thermo=joinpath(dir,"era5_thermo_ml_2021120$(day).nc")
            # 0–360 longitude, N→S latitude; a narrow coastal land strip.
            NCDataset(surface,"c") do ds
                defDim(ds,"longitude",36);defDim(ds,"latitude",19);defDim(ds,"time",24)
                defVar(ds,"longitude",collect(0.:10.:350.),("longitude",))
                defVar(ds,"latitude",collect(90.:-10.:-90.),("latitude",))
                for (name,base,scale,units) in (("lsm",0.,1.,"1"),("fsr",.01,.1,"m"),
                    ("u10",4.,2.,"m s-1"),("v10",1.,1.,"m s-1"),
                    ("ewss",.1,.2,"N m-2"),("nsss",.2,.3,"N m-2"),
                    ("sshf",-10.,-20.,"W m-2"),("slhf",-20.,-30.,"W m-2"),("z",0.,100.,"m2 s-2"))
                    a=[base+scale*(i==1) for i in 1:36,j in 1:19,h in 1:24]
                    defVar(ds,name,a,("longitude","latitude","time");attrib=Dict("units"=>units))
                end
            end
            NCDataset(thermo,"c") do ds
                defDim(ds,"longitude",36);defDim(ds,"latitude",19);defDim(ds,"level",2);defDim(ds,"time",24)
                lon=collect(0.:10.:350.);lat=collect(90.:-10.:-90.)
                defVar(ds,"longitude",lon,("longitude",));defVar(ds,"latitude",lat,("latitude",))
                t=[280+cosd(x)+y/90+2k+h for x in lon,y in lat,k in 1:2,h in 1:24]
                defVar(ds,"t",t,("longitude","latitude","level","time");attrib=Dict("units"=>"K"))
                defVar(ds,"q",fill(.005,36,19,2,24),("longitude","latitude","level","time");attrib=Dict("units"=>"kg kg-1"))
            end
        end
        settings=Pre.ERA5SpectralSettings((;output_float_type=Float64,met_interval=3600.,thermo_dir=dir,surface_dir=dir))
        vertical=(;Nz_native=2,Nz=2,level_range=1:2,ab=(;dA=[10000.,90000.],dB=[0.,0.]),merge_map=[1,2])
        for grid in (Pre.build_target_geometry(Dict("type"=>"latlon","nlon"=>12,"nlat"=>7),Float64),
            Pre.build_target_geometry(Dict("type"=>"synthetic_reduced_gaussian","gaussian_number"=>2,"nlon_mode"=>"octahedral"),Float64),
            Pre.build_target_geometry(Dict("type"=>"cubed_sphere","Nc"=>4,"regridder_cache_dir"=>joinpath(dir,"weights")),Float64))
            lons,lats=LL.tm5_target_coordinates(grid)
            shape=grid isa Pre.LatLonTargetGeometry ? (12,7,2) : (ncells(grid.mesh),2)
            cache=LL.TM5PBLInputCache()
            try
                a=LL.load_tm5_pbl_target_window(Date(2021,12,1),1,settings,vertical,fill(1e5,length(lons)),grid,shape;cache)
                shared=cache.endpoint
                weights=cache.regridder
                b=LL.load_tm5_pbl_target_window(Date(2021,12,1),2,settings,vertical,fill(1e5,length(lons)),grid,shape;cache)
                @test cache.regridder === weights
                @test length(cache.readers)==1
                @test LL._tm5_thermo_endpoint(shared.key...,vertical,lons,lats,LL.TM5PBLInputCache(Dict{Date,PI.ERA5SurfaceReader}(),shared,nothing)) === shared.value
                @test b.t ≈ a.t .+ 1
                # Point sampling at 0° gives 1; at ±15° gives 0. Both differ from a coastal fraction.
                c=findfirst(i->abs(lats[i])<30 && 0 < vec(a.lsm)[i] < .9,eachindex(lons))
                @test c !== nothing
                if c !== nothing
                    @test .01 < vec(a.lsm)[c] < .9
                    point_lsm = max(0.,1-abs(mod(lons[c]+180,360)-180)/10)
                    @test abs(vec(a.lsm)[c]-point_lsm) > .05
                    println("TM5 coastal ",typeof(grid)," lon=",lons[c]," lat=",lats[c],
                        " area_mean=",vec(a.lsm)[c]," centre_sample=",point_lsm)
                    for (key,base,scale) in ((:fsr,.01,.1),(:ewss,.1,.2),(:nsss,.2,.3),(:sshf,-10.,-20.),(:slhf,-20.,-30.))
                        @test vec(getproperty(a,key))[c] ≈ base+scale*vec(a.lsm)[c] rtol=1e-12
                    end
                end
                # Bilinear thermo has known values at these source-aligned longitudes/latitudes.
                t=reshape(a.t,:,2)
                for c in eachindex(lons)
                    x=mod(lons[c],360)/10;lo=floor(Int,x);w=x-lo
                    expected_cos=(1-w)*cosd(10lo)+w*cosd(10mod(lo+1,36))
                    @test t[c,1] ≈ 280+expected_cos+lats[c]/90+2+1.5
                end
            finally
                close(cache)
            end
            @test isempty(cache.readers) && cache.endpoint === nothing
            enabled=Pre.ERA5SpectralSettings((;include_tm5_pbl=true,thermo_dir=dir,met_interval=3600.,level_range=2:3))
            @test_throws r"level_top=1" Pre.validate_tm5_pbl_target(grid,enabled,vertical)
            enabled=Pre.ERA5SpectralSettings((;include_tm5_pbl=true,thermo_dir=dir,met_interval=3600.,level_range=1:2))
            @test_throws r"target_vertical" Pre.validate_tm5_pbl_target(grid,enabled,merge(vertical,(;merge_map=Int[])))
            @test_throws r"at least two" Pre.validate_tm5_pbl_target(grid,enabled,merge(vertical,(;Nz=1,merge_map=[1,1])))
        end
    end
end

@testset "Float32 LL to CS partial all-land surface means" begin
    mktempdir() do dir
        MD=AtmosTransport.MetDrivers
        R=AtmosTransport.Regridding
        CS=Pre.CubedSphereTarget
        nx,ny,nz,n=720,360,2,24
        mesh=LatLonMesh(;Nx=nx,Ny=ny)
        vc=HybridSigmaPressure([0.,50000.,100000.],zeros(3))
        grid=AtmosGrid(mesh,vc,CPU())
        input=LL.tm5_pbl_sample(Float32,(nx,ny,nz))
        # Both pure-land and pure-sea target cells, plus a fractional coastline.
        input.lsm[17:nx÷2+16,:] .= 1
        input.q .= .005
        input.t .= 280
        flux=allocate_face_fluxes(mesh,nz;FT=Float32,basis=DryBasis)
        for a in (flux.am,flux.bm,flux.cm)
            fill!(a,0)
        end
        w=(;m=ones(Float32,nx,ny,nz),ps=fill(1f5,nx,ny),
            am=flux.am,bm=flux.bm,cm=flux.cm,tm5_pbl=input)
        path=joinpath(dir,"source.bin")
        MD.write_transport_binary(path,grid,[w];mass_basis=:dry,source_flux_sampling=:window_start_endpoint,
            surface_pressure_basis=TotalSurfacePressure(),
            flux_construction=MD.BinaryFormat.PointSampledSpectralFluxesHeader())
        reader=MD.TransportBinaryReader(path;FT=Float32)
        try
            cs=Pre.build_target_geometry(Dict("type"=>"cubed_sphere","Nc"=>n),Float64)
            r=R.build_regridder(mesh,cs.mesh)
            ws=CS.allocate_cs_preprocess_workspace(n,nx,ny,nz,nx*ny,6n*n,Float32)
            out=CS._regrid_tm5_pbl_to_cs(reader,1,r,ws,n,Float32)
            lsm=reduce(vcat,map(vec,out.lsm))
            @test eltype(lsm)==Float32
            @test all(x->0<=x<=1,lsm)
            @test count(==(1),lsm)>length(lsm)÷4
            @test any(==(0),lsm) && any(x->0<x<1,lsm)
            @test all(p->MD.BinaryFormat.validate_tm5_pbl_inputs(map(a->a[p],out),(n,n,nz),(n,n)) === nothing,1:6)
        finally
            close(reader)
        end
    end
end

@testset "Spectral to CS panel-major nonuniform surface ordering" begin
    mktempdir() do dir
        # Smooth global field with independent longitude and latitude variation.
        # Cell-area means approach this analytic centre value at second order.
        f(l,p)=cosd(p)*(2+sind(l))+.5sind(p)
        g(l,p)=cosd(p)*cosd(l)+.3sind(p)
        mesh=LatLonMesh(;Nx=360,Ny=180)
        NCDataset(joinpath(dir,"era5_surface_20211201.nc"),"c") do ds
            defVar(ds,"longitude",mesh.λᶜ,("longitude",))
            defVar(ds,"latitude",mesh.φᶜ,("latitude",))
            defVar(ds,"time",0:23,("time",))
            for (name,value,units) in (("lsm",.5,"1"),("fsr",.1,"m"),
                    ("u10",0.,"m s-1"),("v10",1.,"m s-1"),("ewss",.1,"N m-2"),
                    ("nsss",.2,"N m-2"),("sshf",0.,"W m-2"),("slhf",-10.,"W m-2"),("z",0.,"m2 s-2"))
                a=[name=="u10" ? f(l,p) : name=="sshf" ? -20+3g(l,p) : value
                    for l in mesh.λᶜ,p in mesh.φᶜ,h in 1:24]
                defVar(ds,name,a,("longitude","latitude","time");attrib=Dict("units"=>units))
            end
        end
        write_pbl_thermo(joinpath(dir,"era5_thermo_ml_20211201.nc"))
        settings=Pre.ERA5SpectralSettings((;output_float_type=Float64,
            met_interval=3600.,thermo_dir=dir,surface_dir=dir))
        vertical=(;Nz_native=2,Nz=2,level_range=1:2,
            ab=(;dA=[10000.,90000.],dB=[0.,0.]),merge_map=[1,2])
        for convention in ("gnomonic","geos_native")
            n=8
            cs=Pre.build_target_geometry(Dict("type"=>"cubed_sphere","Nc"=>n,
                "panel_convention"=>convention,"regridder_cache_dir"=>dir),Float64)
            means=LL.load_tm5_pbl_target_window(Date(2021,12,1),1,settings,
                vertical,fill(1e5,6n*n),cs,(6n*n,2))
            # Pin the spectral writer's flattened panel order independently:
            # panel, then j, then i, with Julia i-fast storage on each panel.
            expected=reduce(vcat,[vec(f.(AtmosTransport.Grids.panel_cell_center_lonlat(cs.mesh,p)...)) for p in 1:6])
            @test maximum(abs.(means.u10.-expected)) < 2/n^2
            expected_g=reduce(vcat,[vec(g.(AtmosTransport.Grids.panel_cell_center_lonlat(cs.mesh,p)...)) for p in 1:6])
            @test maximum(abs.(means.sshf.-(-20 .+ 3expected_g))) < 6/n^2
            lons,lats=LL.tm5_target_coordinates(cs)
            @test f.(lons,lats)==expected
        end
    end
end
