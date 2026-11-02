using Test, AtmosTransport, Dates, NCDatasets, TOML, JSON3
const PP=AtmosTransport.Preprocessing
const SP=PP.Spectral
const CO=PP.Common
const CS=PP.CubedSphereTarget
const MD=AtmosTransport.MetDrivers

function cs_test_vertical()
    vc=AtmosTransport.Grids.HybridSigmaPressure([0.,2000.,0.],[0.,0.3,1.])
    return (;Nz_native=2,Nz=2,level_range=1:2,vc_native=vc,merged_vc=vc,
        merge_map=[1,2],ab=(dA=[2000.,-2000.],dB=[0.3,0.7]),vertical_mapping_method=:merge_map)
end

function cs_spectral_day_fixture(dir,date)
    spectral,cache=joinpath(dir,"spectral"),joinpath(dir,"cache")
    mkpath(spectral);mkpath(cache)
    paths=SP.era5_spectral_paths(spectral,date)
    touch(paths.vo_d);touch(paths.lnsp)
    h=zeros(ComplexF64,4,4);h[1]=log(100000.);h[3,2]=0.01
    vo=zeros(ComplexF64,4,4,137);d=copy(vo)
    vo[2,1,1:2].=1e-6;d[3,2,1:2].=1e-8
    for hour in 0:23
        endpoint=(;lnsp=h,vo=vo.*(1+0.02hour),d=d.*(1+0.02hour),T=3)
        SP.JLD2.jldsave(SP.spectral_hour_cache_path(cache,paths,hour);endpoint)
    end
    for day in (date,date+Day(1))
        NCDataset(SP.era5_thermo_path(dir,day),"c") do ds
            defDim(ds,"longitude",8);defDim(ds,"latitude",5)
            defDim(ds,"hybrid",2);defDim(ds,"time",24)
            defVar(ds,"longitude",collect(0.:45.:315.),("longitude",))
            defVar(ds,"latitude",collect(-90.:45.:90.),("latitude",))
            q=[0.001k+0.0001sinpi(i/4)*cospi(j/5) for i in 1:8,j in 1:5,k in 1:2,t in 1:24]
            defVar(ds,"q",q,("longitude","latitude","hybrid","time");attrib=Dict("units"=>"kg kg-1"))
            defVar(ds,"t",fill(280.,size(q)),("longitude","latitude","hybrid","time");attrib=Dict("units"=>"K"))
        end
        surface_path=joinpath(dir,"era5_surface_$(Dates.format(day,"yyyymmdd")).nc")
        NCDataset(surface_path,"c") do ds
            defDim(ds,"longitude",4);defDim(ds,"latitude",3);defDim(ds,"time",24)
            defVar(ds,"longitude",[0.,90.,180.,270.],("longitude",))
            defVar(ds,"latitude",[-90.,0.,90.],("latitude",))
            defVar(ds,"time",0:23,("time",))
            for (name,value,units) in (("lsm",0.4,"1"),("fsr",0.1,"m"),
                ("u10",6.,"m s-1"),("v10",2.,"m s-1"),("ewss",900.,"N m-2 s"),
                ("nsss",-450.,"N m-2 s"),("sshf",-90000.,"J m-2"),
                ("slhf",-45000.,"J m-2"),("z",9806.65,"m2 s-2"))
                defVar(ds,name,fill(value,4,3,24),("longitude","latitude","time");attrib=Dict("units"=>units))
            end
        end
    end
    return spectral,cache,(;lnsp=h,vo,d,T=3)
end

@testset "Direct TM5 CS day, provenance, dry closure and runtime replay" begin
    mktempdir() do dir
        date=Date(2021,12,1)
        spectral,cache,next=cs_spectral_day_fixture(dir,date)
        grid=PP.build_target_geometry(Dict("type"=>"cubed_sphere","Nc"=>2,
            "regridder_cache_dir"=>joinpath(dir,"weights")),Float64)
        run_cache=Dict{Symbol,Any}()
        previous_artifacts=nothing
        for (FT,pbl) in ((Float64,false),(Float32,false),(Float64,true))
            settings=PP.ERA5SpectralSettings((;spectral_flux_construction=PP.TM5SpectralFluxes(),
                spectral_synthesis=CO.CPUSynthesis(),spectral_synthesis_float=CO.AutoSynthesisPrecision(),
                spectral_dir=spectral,spectral_cache_dir=cache,T_target=0,min_dp=0.0,
                include_qv=false,include_surface=false,include_tm5_pbl=pbl,thermo_dir=dir,surface_dir=dir,half_dt=450.,
                met_interval=3600.,dt=900.,out_dir=joinpath(dir,string(FT)*string(pbl)),output_float_type=FT,
                flux_time_sampling=:window_mean,mass_basis=:dry,mass_fix_enable=true,
                target_ps_dry_pa=98325.,column_balance_weights=:hybrid_mass))
            @test_throws ErrorException PP.process_day(date,grid,settings,cs_test_vertical())
            path=PP.process_day(date,grid,settings,cs_test_vertical();next_day_hour0=next,run_cache)
            @test length(run_cache)==1
            artifacts=only(values(run_cache))
            previous_artifacts===nothing || @test artifacts===previous_artifacts
            previous_artifacts=artifacts
            @test isfile(path)
            info=MD.inspect_binary(path;io=devnull)
            @test info.grid_type===:cubed_sphere
            driver=MD.TransportBinaryDriver(path;FT,Hp=2)
            reader=driver.reader
            header=reader.header.raw_header
            @test header["format_version"]==5
            @test header["flux_construction"]=="tm5_spectral"
            @test header["cs_face_quadrature"]=="great_circle_midpoint_cardinal16_v2"
            @test header["samples_per_face"]>=16
            @test header["refined_nlon"]>=2*(3+1)+1
            @test header["source_flux_sampling"]=="window_mean"
            @test header["spectral_synthesis"]=="cpu"
            @test header["column_closure_windows"]==24
            @test header["column_closure_residual_to_tolerance_max"]<=1
            @test header["column_closure_cm_residual_to_tolerance_max"]<=1
            @test MD.binary_capabilities(reader).pbl_diffusion == pbl
            pbl && @test MD.load_transport_window(driver,1).tm5_pbl !== nothing
            close(reader)
            cfg=Dict{String,Any}("input"=>Dict("binary_paths"=>[path]),
                "run"=>Dict("start_window"=>1,"stop_window"=>1,"prefetch_windows"=>false),
                "numerics"=>Dict("float_type"=>string(FT)),"advection"=>Dict("scheme"=>"upwind"),
                "tracers"=>Dict("co2"=>Dict("init"=>Dict("kind"=>"uniform","background"=>4e-4))),
                "output"=>Dict("enabled"=>false))
            model=run_driven_simulation(cfg)
            @test maximum(mixing_ratio(model.state,:co2)) do panel
                interior=AtmosTransport.State.panel_interior(panel,model.state.halo_width)
                maximum(abs,interior.-FT(4e-4))
            end <= MD.replay_tolerance(FT)*FT(4e-4)
        end
    end
end

@testset "CS q=0 dry route equals hybrid-B closure" begin
    mktempdir() do dir
        grid=PP.build_target_geometry(Dict("type"=>"cubed_sphere","Nc"=>2),Float64)
        vertical=cs_test_vertical();c=PP.TM5SpectralFluxes(4)
        settings=PP.ERA5SpectralSettings((;spectral_flux_construction=c,output_float_type=Float64,
            mass_fix_enable=true,mass_basis=:dry,target_ps_dry_pa=98325.,
            column_balance_weights=:hybrid_mass,met_interval=3600.,dt=900.))
        ws=CS.tm5_cs_integral_workspace(grid,3,c;batch=2)
        h=zeros(ComplexF64,4,4);h[1]=log(100000.);h[3,2]=0.01
        vo=zeros(ComplexF64,4,4,2);d=copy(vo)
        vo[2,1,:].=1e-6;d[3,2,:].=1e-8
        q=zeros(grid.face_table.nc,2);plan=SP.TM5EndpointPlan(vertical)
        a=SP.construct_tm5_endpoint(ws,h,vo,d,3,q,plan,settings)
        h[3,2]+=0.0001
        b=SP.construct_tm5_endpoint(ws,h,vo,d,3,q,plan,settings)
        rates=CO.balance_window_layers!(c,a,b,grid,PP.MassClosure.DirectCSPoisson(grid.face_table),3600.,ws.quadrature.areas)
        contract=CS.CubedSphereContract{Float64}(;replay_tol=1e-12,positivity_cfl_limit=0.95,
            require_substep_positivity=true,steps_per_window=4)
        weights=CO.spectral_column_weights(grid,settings,vertical)
        policy=PP.TransportBinary.SubstepSchedulePolicy(adaptive_substeps=false,substep_cfl_target=0.95)
        closed=CS.close_tm5_cs_window(rates,a.ps,grid,settings,vertical,contract,policy,weights)
        payload=closed.payload
        flat_flux=zeros(grid.face_table.nf,2)
        ft=grid.face_table
        for k in 1:2,f in 1:ft.nf
            source=ft.face_dir[f]==1 ? payload.am[ft.face_panel[f]] : payload.bm[ft.face_panel[f]]
            flat_flux[f,k]=source[ft.face_idx_i[f],ft.face_idx_j[f],k]
        end
        convergence=PP.LatLonTarget.tm5_horizontal_convergence(flat_flux,zeros(0,2),grid)
        cm=reduce(vcat,map(p->reshape(p,:,3),payload.cm))
        expected=convergence[:,1].-vertical.merged_vc.B[2].*vec(sum(convergence;dims=2))
        @test maximum(abs,cm[:,2]-expected)/maximum(abs,flat_flux)<1e-12
        @test all(iszero,cm[:,1]) && all(iszero,cm[:,end])
        @test closed.diag.replay.max_rel_err<=1e-12
    end
end

@testset "Direct CS golden definitions pass preflight" begin
    for name in ("pre_c24_tm5","pre_c24_tm5_tm5pbl")
        cfg=TOML.parsefile(joinpath(pkgdir(AtmosTransport),"test/golden/configs",name*".toml"))
        cfg["input"]["coefficients"]=joinpath(pkgdir(AtmosTransport),"config/era5_L137_coefficients.toml")
        @test PP.validate_preprocessing_config(cfg)!==nothing
    end
end
