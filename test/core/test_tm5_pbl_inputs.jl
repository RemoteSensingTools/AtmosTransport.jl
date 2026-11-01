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
            Pre.build_target_geometry(Dict("type"=>"synthetic_reduced_gaussian","gaussian_number"=>2,"nlon_mode"=>"octahedral"),Float64))
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
