using Test, AtmosTransport, LinearAlgebra, Random, Dates, TOML, JSON3
import NCDatasets
const PP=AtmosTransport.Preprocessing
const SP=PP.Spectral
const MC=PP.MassClosure
const CO=PP.Common
const VT=PP.Vertical
const MD=AtmosTransport.MetDrivers
const Rv=AtmosTransport.Parameters.IFS_EARTH_RADIUS

function harmonic_samples(coeff,T,phi,lons)
    P=zeros(T+1,T+1);SP.compute_legendre_column!(P,T,sin(phi))
    return [sum((m==0 ? 1 : 2)*real(coeff[n+1,m+1]*P[n+1,m+1]*cis(m*l)) for m in 0:T for n in m:T) for l in lons]
end

@testset "TM5 retained spectral mathematics and quirk sizes" begin
    T=7;phi=0.31;H=zeros(ComplexF64,T+1,T+1)
    # Closed-form n=1 harmonics test normalization independently of the recurrence.
    lon=[0.,0.4,1.7,4.2]
    for phi0 in (-1.2,0.,0.7)
        mode=zeros(ComplexF64,3,3);mode[2,1]=0.2
        gx,gy=SP.lnsp_gradient(mode,2)
        @test harmonic_samples(gy,3,phi0,lon) ≈ fill(0.2sqrt(3)*cos(phi0)^2,4) rtol=1e-12
        mode[2,1]=0;mode[2,2]=0.2
        gx,gy=SP.lnsp_gradient(mode,2)
        @test harmonic_samples(gx,3,phi0,lon) ≈ -0.4sqrt(1.5)*cos(phi0).*sin.(lon) atol=1e-14 rtol=1e-12
        @test harmonic_samples(gy,3,phi0,lon) ≈ -0.4sqrt(1.5)*sin(phi0)*cos(phi0).*cos.(lon) atol=1e-14 rtol=1e-12
    end
    for (n,m) in ((T-1,2),(T,3),(T,T),(2,0))
        fill!(H,0);H[n+1,m+1]=0.03+0.02im
        gx,gy=SP.lnsp_gradient(H,T)
        lon=collect(range(0,2pi;length=66))[1:end-1]
        dl=1e-5
        exact_lon=(harmonic_samples(H,T,phi,lon.+dl).-harmonic_samples(H,T,phi,lon.-dl))./(2dl)
        exact_lat=cos(phi).*(harmonic_samples(H,T,phi+dl,lon).-harmonic_samples(H,T,phi-dl,lon))./(2dl)
        @test harmonic_samples(gx,T+1,phi,lon) ≈ exact_lon rtol=2e-8 atol=1e-12
        @test harmonic_samples(gy,T+1,phi,lon) ≈ exact_lat rtol=2e-8 atol=1e-12
    end
    # grid_type_sh.F90:1804: sign-only error is twice the retained lower term.
    fill!(H,0);H[T,3]=1
    _,gy=SP.lnsp_gradient(H,T)
    epsmn=-T*SP._delta(2,T)
    @test gy[T+1,3] ≈ -(T-1)*epsmn
    @test abs(-gy[T+1,3]-gy[T+1,3]) ≈ 2(T-1)*epsmn
    @test abs(gy[T+2,3]) == 0
    fill!(H,0);H[T+1,3]=1
    _,top_gradient=SP.lnsp_gradient(H,T)
    @test top_gradient[T+2,3] ≈ -T*sqrt(((T+1)^2-2^2)/(4(T+1)^2-1)) rtol=1e-12
    # P_T^T corner discarded by TM5 (grid_type_sh.F90:829–830).
    P=zeros(T+1,T+1);SP.compute_legendre_column!(P,T,0.0)
    @test P[T+1,T+1] ≈ prod(sqrt((2m+1)/(2m)) for m in 1:T)
    @test P[T+1,T+1] > 1
    # Each documented wind edge, plus the shared T+1 truncation that we fix.
    vo=zeros(ComplexF64,T+1,T+1,1);d=similar(vo);fill!(d,0)
    vo[T+1,2,1]=1
    u,v=SP.complete_wind_coefficients(vo,d,T)
    @test u[T,2,1] == -Rv*SP._delta(1,T) # omitted by TM5 n=T-1,m<=T-2
    @test u[T+2,2,1] == Rv*SP._delta(1,T+1)
    fill!(vo,0);d[T+1,3,1]=1
    u,v=SP.complete_wind_coefficients(vo,d,T)
    @test u[T+1,3,1] == im*Rv*SP._sigma(2,T) # dropped n=T iσ
    fill!(d,0);d[T+1,T+1,1]=1
    u,v=SP.complete_wind_coefficients(vo,d,T)
    @test u[T+1,T+1,1] != 0 # TM5 zeros (T,T)
    fill!(d,0);vo[T+1,T,1]=1
    u,v=SP.complete_wind_coefficients(vo,d,T)
    @test u[T,T,1] == -Rv*SP._delta(T-1,T) # special diagonal agrees
    @test abs(6371000/Rv-1) ≈ 229/Rv rtol=1e-9
    # Polar mean counts a periodic endpoint twice: this mode's true mean=0.
    nlon=16;adjacent=cos.(2pi.*(0:nlon)./nlon)
    @test sum(adjacent)/(nlon+1) ≈ 1/(nlon+1)
    nodes,w,cw=SP.latitude_quadrature(-pi/2,-pi/2+0.1,8)
    @test all(x->x>-pi/2,nodes)
    @test sum(w) ≈ 0.1
    @test sum(cw) ≈ 1-cos(0.1)
    # TM5's adjacent D*cos value at the pole is nonzero, physical value zero.
    @test cos(nodes[1]) > 0
    dphi=0.1/8
    @test (dphi/2)*cos(-pi/2+dphi) > 0 # spurious TM5 D*cos polar endpoint contribution
    @test abs(sum(w.*cos.(nodes))-(1-cos(0.1))) < (dphi/2)*sin(dphi)
    @info "TM5 single-mode quirk sizes" nabla_sign=2(T-1)*epsmn legendre_corner=P[T+1,T+1] polar_duplicate_mean=1/(nlon+1)
end

function spectral_example(T=4,nz=2;scale=1.0)
    h=zeros(ComplexF64,T+1,T+1);h[1]=log(100000.);h[3,2]=0.015scale
    vo=zeros(ComplexF64,T+1,T+1,nz);d=similar(vo);fill!(d,0)
    for k in 1:nz
        vo[2,1,k]=2e-6*k
        d[3,2,k]=1e-7*(3-k)
        d[T+1,2,k]=2e-8 # tests complete wind transform at truncation edge
    end
    return h,vo,d
end
function tm5_grid(rg=false)
    return rg ? PP.build_target_geometry(Val(:synthetic_reduced_gaussian),Dict("gaussian_number"=>2,"nlon_mode"=>"octahedral"),Float64) :
        PP.build_target_geometry(Dict("nlon"=>12,"nlat"=>6),Float64)
end

@testset "TM5 face integrals, independent divergence and polar convergence" begin
    h,vo,d=spectral_example()
    for rg in (false,true)
        grid=tm5_grid(rg)
        residuals=Float64[]
        for steps in (2,4,8,64)
            c=PP.TM5SpectralFluxes(steps)
            ws=SP.SpectralIntegralWorkspace(grid,4,c;batch=2)
            raw=SP.integrate_face_fluxes!(ws,h,vo,d,4,[2000.,-2000.],[0.3,0.7],c)
            convergence=PP.LatLonTarget.tm5_horizontal_convergence(raw.u,raw.v,grid)
            push!(residuals,norm(convergence+raw.C)/norm(raw.C))
            W=zeros(length(raw.ps),3)
            SP.integrate_vertical_flux!(W,raw.C,[0.,0.3,1.])
            @test all(iszero,W[:,1]) && all(iszero,W[:,end])
            @test all(isfinite,raw.u) && all(isfinite,raw.v)
            @test all(iszero,raw.v[1:ws.quadrature.boundary_counts[1],:])
            @test all(iszero,raw.v[end-ws.quadrature.boundary_counts[end]+1:end,:])
        end
        @info "TM5 Gauss convergence" rg residuals
        @test residuals[3] < residuals[2] < residuals[1]
        @test residuals[4] < (rg ? 2.5e-5 : 4.5e-5)
    end
    # Constant pressure/solid-body rotation has no C or W, exact face orientation.
    grid=tm5_grid();c=PP.TM5SpectralFluxes();ws=SP.SpectralIntegralWorkspace(grid,3,c;batch=1)
    h=zeros(ComplexF64,4,4);h[1]=log(100000.);vo=zeros(ComplexF64,4,4,1);d=similar(vo);fill!(d,0)
    vo[2,1,1]=1e-6
    raw=SP.integrate_face_fluxes!(ws,h,vo,d,3,[0.],[1.],c)
    @test raw.ps ≈ fill(100000.,length(raw.ps)) rtol=2e-15
    @test maximum(abs,raw.C)==0
    @test maximum(abs,raw.v)==0
    @test maximum(abs,PP.LatLonTarget.tm5_horizontal_convergence(raw.u,raw.v,grid))==0
    exact=[grid.mesh.radius*Rv*sqrt(3)*1e-6*100000/(2AtmosTransport.Parameters.STANDARD_GRAVITY)*
        (sind(grid.mesh.φᶠ[j+1])-sind(grid.mesh.φᶠ[j])) for j in 1:PP.nlat(grid) for i in 1:PP.nlon(grid)]
    @test maximum(abs,raw.u[:,1]-exact)/maximum(abs,exact) < 0.01
end

function tm5_vertical()
    vc=HybridSigmaPressure([0.,2000.,0.],[0.,0.3,1.])
    return (;Nz_native=2,Nz=2,level_range=1:2,vc_native=vc,merged_vc=vc,
        merge_map=[1,2],ab=(dA=[2000.,-2000.],dB=[0.3,0.7]),vertical_mapping_method=:merge_map)
end
function tm5_settings(;kwargs...)
    return PP.ERA5SpectralSettings(merge((;spectral_flux_construction=PP.TM5SpectralFluxes(),
        spectral_synthesis=CO.CPUSynthesis(),spectral_synthesis_float=CO.AutoSynthesisPrecision(),flux_time_sampling=:window_mean,
        mass_basis=:dry,mass_fix_enable=true,target_ps_dry_pa=98325.,column_balance_weights=:hybrid_mass),NamedTuple(kwargs)))
end
@testset "TM5 pin, level merge, layer balance and dry q=0 limit" begin
    h,vo,d=spectral_example();vertical=tm5_vertical()
    for rg in (false,true)
        grid=tm5_grid(rg);c=PP.TM5SpectralFluxes();ws=SP.SpectralIntegralWorkspace(grid,4,c;batch=2)
        nc=length(ws.quadrature.areas);solver=(rg ? MC.DirectRingPoisson(MC.build_compressed_laplacian(AtmosTransport.Grids.face_connectivity(grid.mesh)...,length(ws.quadrature.areas))) : SP.LLPoissonWorkspace(PP.nlon(grid),PP.nlat(grid)))
        @test_throws ArgumentError CO.validate_spectral_construction(c,grid,tm5_settings(mass_basis=:moist))
        @test_throws ArgumentError CO.validate_spectral_construction(c,grid,tm5_settings(flux_time_sampling=:window_start))
        @test_throws ArgumentError CO.validate_spectral_construction(c,grid,tm5_settings(met_interval=7200.0))
        @test_throws ArgumentError SP.integrate_vertical_flux!(zeros(nc,3),zeros(nc,2),[0.,0.3,0.8])
        q0=zeros(nc,2)
        a0=SP.construct_tm5_endpoint(ws,h,vo,d,4,q0,SP.TM5EndpointPlan(vertical),tm5_settings())
        baseline=CO.balance_window_layers!(c,a0,a0,grid,solver,3600.,ws.quadrature.areas)
        dq=1e-5
        wetter=SP.construct_tm5_endpoint(ws,h,2 .*vo,2 .*d,4,fill(dq,nc,2),SP.TM5EndpointPlan(vertical),tm5_settings())
        mixed=CO.balance_window_layers!(c,a0,wetter,grid,solver,3600.,ws.quadrature.areas)
        # The moist pin changes only the latitude-dependent tendency here,
        # hence its correction has no zonal component. Endpoint winds 1 and 2
        # with humidity 0 and dq give the analytic dry mean (1.5-dq)*F_u.
        @test norm(mixed.u-(1.5-dq).*baseline.u)/norm(baseline.u) < 1e-12
        forcing=copy(a0.W)
        forcing[:,2] .+= 1e8.*sin.(2pi.*collect(1:nc)./nc)
        forced=merge(a0,(;W=forcing))
        balanced=CO.balance_window_layers!(c,forced,forced,grid,solver,3600.,ws.quadrature.areas)
        for endpoint in balanced.diagnostics
            residual=PP.LatLonTarget.tm5_horizontal_convergence(endpoint.u,endpoint.v,grid).+diff(forcing;dims=2)
            projected=residual.-(ws.quadrature.areas./sum(ws.quadrature.areas))*sum(residual;dims=1)
            fluxscale=maximum(abs,forced.u)+maximum(abs,forced.v)+maximum(abs,forcing)
            @test maximum(abs,projected)/fluxscale < 1e-12
        end
        badflux=merge(a0,(;u=a0.u.*1e9,v=a0.v.*1e9))
        @test_throws ErrorException CO.balance_window_layers!(c,badflux,a0,grid,solver,3600.,ws.quadrature.areas)
        badmass=merge(a0,(;m=a0.m.*1.1,ps=a0.ps.*1.1))
        @test_throws ErrorException CO.balance_window_layers!(c,a0,badmass,grid,solver,3600.,ws.quadrature.areas)
        for q in (zeros(nc,2),fill(0.01,nc,2),[0.003+0.001sin(i+k) for i in 1:nc,k in 1:2])
            a=SP.construct_tm5_endpoint(ws,h,vo,d,4,q,SP.TM5EndpointPlan(vertical),tm5_settings())
            hh=copy(h);hh[3,2]+=0.0001
            b=SP.construct_tm5_endpoint(ws,hh,vo,d,4,q,SP.TM5EndpointPlan(vertical),tm5_settings())
            @test sum(a.m.*(1 .-a.q))*AtmosTransport.Parameters.STANDARD_GRAVITY/sum(ws.quadrature.areas) ≈ 98325. rtol=3e-15
            before=copy(a.u)
            win=CO.balance_window_layers!(c,a,b,grid,solver,3600.,ws.quadrature.areas)
            @test a.u == before # shared endpoint cannot be mutated by one window's tendency
            @test all(x->x.post.maximum<1e-6,win.diagnostics)
            if iszero(q)
                @test win.m == a.m
                @test win.m_next == b.m
            end
        end
    end
    vc=HybridSigmaPressure([0.,100.,200.,0.],[0.,0.1,0.3,1.])
    plan=VT.plan_vertical(VT.LevelSelection([0,2,3]),vc)
    x=reshape(collect(1.:16.),2,2,4);a=zeros(2,2,3);b=similar(a)
    VT.apply_vertical!(a,x,plan,VT.InterfaceMassFluxField())
    VT.apply_vertical!(b,x,plan,VT.ConvectionInterfaceFlux())
    @test a==b==x[:,:,[1,3,4]]
end

@testset "TM5 local pydyn balance and dynam0 fixtures" begin
    fixture_dir=get(ENV,"ATMOSTR_TM5_FIXTURES_DIR","")
    cases=[joinpath(fixture_dir,"balance",name*".toml") for name in
        ("ll6x4_l5_with_w","ll6x4_l5_without_w","ll36x18_l34_with_w")]
    if isempty(fixture_dir) || !all(isfile,cases)
        @info "Skipping local pydyn fixtures; set ATMOSTR_TM5_FIXTURES_DIR to the local tm5 fixture root"
        @test_skip false
    else
        for file in cases
            fixture=TOML.parsefile(file)
            function array(key)
                entry=fixture["arrays"][key]
                data=haskey(entry,"data") ? Float64.(entry["data"]) :
                    copy(reinterpret(Float64,read(joinpath(dirname(file),entry["file"]))))
                return reshape(data,entry["shape"]...)
            end
            # Exactly one adapter: pydyn bottom-up/upward W -> top-down/upward W;
            # am/bm stay east/north positive, cm becomes downward positive.
            u=reverse(array("pu_in");dims=3);v=reverse(array("pv_in");dims=3)
            W=reverse(array("pw");dims=3);target=reverse(array("dm_dt");dims=3).-diff(W;dims=3)
            nx,ny,nz=size(target)
            PP.balance_mass_fluxes!(u,v,target,PP.LLPoissonWorkspace(nx,ny);log_summary=false)
            expected_u=reverse(array("pu_out");dims=3);expected_v=reverse(array("pv_out");dims=3)
            scale=maximum(abs,expected_u)+maximum(abs,expected_v)
            @test maximum(abs,u-expected_u)/scale < 1e-12
            @test maximum(abs,v-expected_v)/scale < 1e-12
            h=fixture["case"]["half_step"]
            B=vec(reverse(array("bt")))
            convergence=u[1:end-1,:,:].-u[2:end,:,:].+v[:,1:end-1,:].-v[:,2:end,:]
            column=sum(convergence;dims=3)
            dm=zeros(nx,ny,nz)
            for k in 1:nz
                dm[:,:,k] .= h*(B[k+1]-B[k]).*column[:,:,1]
            end
            cm=zeros(nx,ny,nz+1)
            MD.recompute_cm_from_dm_target!(MD.structured_replay_layout(),zeros(nx,ny,nz),cm,
                ones(nx,ny,nz),dm,u.*h,v.*h)
            expected_cm=-reverse(array("cm");dims=3)
            @test maximum(abs,cm-expected_cm)/(h*scale) < 1e-12
        end
    end
end

function synthetic_tm5_day(dir,date)
    spectral=joinpath(dir,"spectral");cache=joinpath(dir,"cache")
    mkpath(spectral);mkpath(cache)
    paths=SP.era5_spectral_paths(spectral,date)
    touch(paths.vo_d);touch(paths.lnsp)
    h,vo2,d2=spectral_example(3)
    vo=zeros(ComplexF64,4,4,137);d=similar(vo);fill!(d,0)
    vo[:,:,1:2].=vo2;d[:,:,1:2].=d2
    spec=(;hours=collect(0:23),T=3,n_times=24,
        lnsp_all=Dict(i=>copy(h) for i in 0:23),vo_by_hour=Dict(i=>vo.*(1+0.02i) for i in 0:23),
        d_by_hour=Dict(i=>d.*(1+0.02i) for i in 0:23))
    for hour in spec.hours
        endpoint = (;lnsp=spec.lnsp_all[hour],vo=spec.vo_by_hour[hour],d=spec.d_by_hour[hour],T=spec.T)
        SP.JLD2.jldsave(SP.spectral_hour_cache_path(cache,paths,hour); endpoint)
    end
    for day in (date,date+Day(1))
        path=SP.era5_thermo_path(dir,day)
        NCDatasets.NCDataset(path,"c") do ds
            NCDatasets.defDim(ds,"longitude",8);NCDatasets.defDim(ds,"latitude",5)
            NCDatasets.defDim(ds,"hybrid",2);NCDatasets.defDim(ds,"time",24)
            NCDatasets.defVar(ds,"longitude",collect(0.:45.:315.),("longitude",))
            NCDatasets.defVar(ds,"latitude",collect(-90.:45.:90.),("latitude",))
            values=[0.001k+0.0001sinpi(i/4)*cospi(j/5) for i in 1:8,j in 1:5,k in 1:2,t in 1:24]
            NCDatasets.defVar(ds,"q",values,("longitude","latitude","hybrid","time"))
        end
    end
    return spectral,cache,(;lnsp=h,vo,d,T=3)
end

@testset "TM5 production day, v5 read/write and explicit-dm replay" begin
    mktempdir() do dir
        date=Date(2021,12,1);spectral,cache,next=synthetic_tm5_day(dir,date)
        for rg in (false,true),FT in (Float64,Float32)
            grid=tm5_grid(rg);out=joinpath(dir,"out_$(rg)_$(FT)")
            settings=tm5_settings(output_float_type=FT,spectral_dir=spectral,spectral_cache_dir=cache,
                T_target=0,min_dp=0.0,include_qv=!rg || FT===Float64,include_surface=false,qv_global_climatology=0.0,
                qv_preload=true,qv_preload_max_bytes=Int64(1000000),thermo_dir=dir,half_dt=450.0,
                met_interval=3600.0,dt=900.0,out_dir=out)
            path=PP.process_day(date,grid,settings,tm5_vertical();next_day_hour0=next)
            @test isfile(path)
            bytes=read(path,PP.HEADER_SIZE)
            stop=findfirst(==(0x00),bytes)
            header=JSON3.read(String(bytes[1:stop-1]))
            @test header.format_version==5
            @test header.flux_construction=="tm5_spectral"
            @test header.source_flux_sampling=="window_mean"
            @test header.spectral_truncation_used==3
            @test header.spectral_synthesis=="cpu"
            @test header.spectral_synthesis_float=="Float64"
            @test MD.inspect_binary(path;io=devnull).grid_type == (rg ? :reduced_gaussian : :latlon)
            reader=MD.TransportBinaryReader(path;FT)
            @test reader.header.nwindow==24
            first_flux=last(MD.load_window!(reader,1))
            second_flux=last(MD.load_window!(reader,2))
            first_values=Float64.(rg ? first_flux.horizontal_flux : first_flux.am)
            second_values=Float64.(rg ? second_flux.horizontal_flux : second_flux.am)
            # With fixed pressure/q, both balances are linear in these winds.
            # The endpoint-mean ratio differs from start-only sampling's 1.02.
            ratio=sum(first_values.*second_values)/sum(abs2,first_values)
            @test ratio ≈ 1.03/1.01 rtol=(FT===Float64 ? 1e-12 : 1e-6)
            close(reader)
            cfg=Dict{String,Any}(
                "input"=>Dict("binary_paths"=>[path]),
                "run"=>Dict("start_window"=>1,"stop_window"=>1,"prefetch_windows"=>false),
                "numerics"=>Dict("float_type"=>string(FT)),
                "advection"=>Dict("scheme"=>"upwind"),
                "tracers"=>Dict("co2"=>Dict("init"=>Dict("kind"=>"uniform","background"=>4e-4))),
                "output"=>Dict("enabled"=>false))
            model=run_driven_simulation(cfg)
            @test maximum(abs,mixing_ratio(model.state,:co2).-FT(4e-4)) <
                MD.replay_tolerance(FT)*FT(4e-4)
            @test_throws ErrorException PP.process_day(date,grid,settings,tm5_vertical())
        end
    end
end

@testset "TM5 q=0 final dry closure equals hybrid-B dynam0" begin
    vertical=tm5_vertical();h,vo,d=spectral_example()
    for rg in (false,true)
        grid=tm5_grid(rg);c=PP.TM5SpectralFluxes();ws=SP.SpectralIntegralWorkspace(grid,4,c;batch=2)
        nc=length(ws.quadrature.areas);q=zeros(nc,2);solver=(rg ? MC.DirectRingPoisson(MC.build_compressed_laplacian(AtmosTransport.Grids.face_connectivity(grid.mesh)...,length(ws.quadrature.areas))) : SP.LLPoissonWorkspace(PP.nlon(grid),PP.nlat(grid)))
        a=SP.construct_tm5_endpoint(ws,h,vo,d,4,q,SP.TM5EndpointPlan(vertical),tm5_settings())
        hh=copy(h);hh[3,2]+=0.0001
        b=SP.construct_tm5_endpoint(ws,hh,vo,d,4,q,SP.TM5EndpointPlan(vertical),tm5_settings())
        win=CO.balance_window_layers!(c,a,b,grid,solver,3600.,ws.quadrature.areas)
        seconds=450.;steps=4;dm=(win.m_next-win.m)./(2steps)
        u=win.u.*seconds;v=win.v.*seconds
        if rg
            left,right=AtmosTransport.Grids.face_connectivity(grid.mesh)
            flux=vcat(u,v);L=MC.build_compressed_laplacian(left,right,nc)
            scratch=(;psi=zeros(nc),rhs=zeros(nc),r=zeros(nc),p=zeros(nc),Ap=zeros(nc),z=zeros(nc))
            MC.balance_compressed_column_fluxes!(flux,win.m,win.m_next,left,right,L,steps,scratch;
                direct_solver=solver)
            cm=zeros(nc,3)
            MD.recompute_cm_from_dm_target!(MD.faceindexed_replay_layout(left,right),zeros(nc,2),cm,
                win.m,dm,flux)
            u .= flux[1:nc,:];v .= flux[nc+1:end,:]
        else
            am,bm=PP.LatLonTarget.directional_fluxes(u,v,grid);nx=PP.nlon(grid);ny=PP.nlat(grid)
            SP.balance_column_mass_fluxes!(am,bm,reshape(win.m,nx,ny,2),reshape(dm,nx,ny,2),solver)
            cc=zeros(nx,ny,3)
            MD.recompute_cm_from_dm_target!(MD.structured_replay_layout(),zeros(nx,ny,2),cc,
                reshape(win.m,nx,ny,2),reshape(dm,nx,ny,2),am,bm)
            cm=reshape(cc,nc,3)
            u .= reshape(am[1:nx,:,:],nc,2);v .= reshape(bm,size(v))
        end
        conv=PP.LatLonTarget.tm5_horizontal_convergence(u,v,grid)
        expected=zeros(nc,3)
        expected[:,2] .= conv[:,1].-vertical.merged_vc.B[2].*vec(sum(conv;dims=2))
        scale=maximum(abs,u)+maximum(abs,v)
        @test maximum(abs,cm-expected)/scale < 1e-12
        @test all(iszero,cm[:,1]) && maximum(abs,cm[:,end])/scale < 1e-12
    end
end

@testset "Dry pin global water mode cannot trip the polar guard" begin
    grid=PP.build_target_geometry(Dict("nlon"=>72,"nlat"=>37),Float64)
    nx=PP.nlon(grid); ny=PP.nlat(grid); nc=nx*ny
    areas=copy(SP.SpectralQuadrature(grid,0).areas)
    # Scaled caps reproduce the mean/min area ratio (~290) of a 0.25° mesh.
    areas[1:nx] .*= 0.04
    @test sum(areas)/nc/minimum(areas) > 290
    g=AtmosTransport.Parameters.STANDARD_GRAVITY
    dq=2e-5; ps=98325.0
    a=(;ps=fill(ps,nc),m=reshape(areas.*(ps/g),nc,1),q=zeros(nc,1),
        u=zeros(nc,1),v=zeros(nx*(ny+1),1),W=zeros(nc,2))
    b=merge(a,(;ps=fill(ps/(1-dq),nc),m=a.m./(1-dq),q=fill(dq,nc,1)))
    tendency=(b.m-a.m)./3600
    # Old solver discarded an index-space mean, then the guard divided by area.
    old_max=maximum(abs,vec(sum(tendency;dims=2)).*0 .+
        (sum(tendency)/nc).*g./areas./((a.ps+b.ps)./2))
    @test old_max > 1e-6
    win=CO.balance_window_layers!(PP.TM5SpectralFluxes(),a,b,grid,
        SP.LLPoissonWorkspace(nx,ny),3600.0,areas)
    @test all(x->x.passes==1 && x.post.maximum<1e-12,win.diagnostics)
    @test win.m ≈ win.m_next rtol=3e-16
    @info "Scaled 0.25-degree water-mode regression" area_ratio=sum(areas)/nc/minimum(areas) old_max new_post=win.diagnostics[1].post
end

@testset "Complete vertical contract before spectral IO" begin
    vertical=tm5_vertical(); c=PP.TM5SpectralFluxes()
    truncated=merge(vertical,(;level_range=20:21,
        vc_native=HybridSigmaPressure([100.,2000.,0.],[0.,0.3,1.])))
    for bad in (truncated,merge(vertical,(;vertical_mapping_method=:pressure_overlap)),
        merge(vertical,(;vc_native=HybridSigmaPressure([0.,2000.,0.],[0.,0.3,0.9]))))
        @test_throws ArgumentError CO.validate_spectral_vertical(c,bad)
        # No source paths are supplied: validation must fail before reading them.
        for rg in (false,true)
            @test_throws ArgumentError PP.process_day(Date(2022,1,1),tm5_grid(rg),tm5_settings(),bad)
        end
    end
end

@testset "Nonconstant cell pressure cos-linear quadrature" begin
    # Exact primitive of (a+bφ)cosφ; nonzero b detects a reversed moment sign.
    south,north=-0.7,0.2
    nodes,_,weights=SP.latitude_quadrature(south,north,13)
    primitive(phi)=1.7sin(phi)+0.4*(phi*sin(phi)+cos(phi))
    @test dot(weights,1.7 .+0.4.*nodes) ≈ primitive(north)-primitive(south) atol=2e-15 rtol=2e-15
    # Production PS integration of an independent n=1 zonal harmonic.
    grid=tm5_grid(); c=PP.TM5SpectralFluxes(64)
    h=zeros(ComplexF64,3,3); h[1]=log(100000.); h[2,1]=0.02
    vo=zeros(ComplexF64,3,3,1); d=copy(vo)
    ws=SP.SpectralIntegralWorkspace(grid,2,c;batch=1)
    raw=SP.integrate_face_fluxes!(ws,h,vo,d,2,[0.],[1.],c)
    for j in 2:PP.nlat(grid)-1
        lo,hi=sin.(deg2rad.(grid.mesh.φᶠ[j:j+1]))
        # Analytic area average of exp(log(p0)+a sinφ) in μ=sinφ.
        alpha=0.02sqrt(3)
        expected=100000*exp(alpha*lo)*expm1(alpha*(hi-lo))/(alpha*(hi-lo))
        @test raw.ps[(j-1)*PP.nlon(grid)+1] ≈ expected rtol=1e-6
    end
end

@testset "TM5 public preflight rejects partial columns" begin
    cfg=TOML.parsefile(joinpath(pkgdir(AtmosTransport),"test/golden/configs/pre_ll72_tm5.toml"))
    cfg["input"]["coefficients"]=joinpath(pkgdir(AtmosTransport),"config/era5_L137_coefficients.toml")
    @test PP.validate_preprocessing_config(cfg) !== nothing
    cfg["grid"]["level_top"]=20
    delete!(cfg["grid"],"echlevs")
    @test_throws ArgumentError PP.validate_preprocessing_config(cfg)
end
