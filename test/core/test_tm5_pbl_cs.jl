using Test, Adapt, Random, LinearAlgebra
include("../helpers/tm5_pbl_cs.jl")

@testset "TM5 CS refresh, column consistency and chunking" begin
    for FT in (Float64,Float32),convention in (AtmosTransport.Grids.GnomonicPanelConvention(),AtmosTransport.Grids.GEOSNativePanelConvention())
        f=tm5_cs_fixture(FT;convention)
        g=tm5_cs_fixture(FT;convention,chunk_columns=7)
        for x in (f,g)
            refresh_tm5_calc_kzz!(x.field,x.input,x.mass,x.fluxes,x.grid.vertical,900.,DryBasis())
        end
        @test isequal(f.field.dkg,g.field.dkg)
        @test all(d->all(isfinite,d)&&any(>(0),d),f.field.dkg)
        @test !applicable(AtmosTransport.State.panel_field,f.field,1)
        @test adapt(Array,f.field).params == f.field.params
        s=f.field.scratch
        # Column arrays assembled by the production accessors vs pure column physics.
        for c in (1,8,20,54)
            out=map(a->copy(vec(a[c,:])),s.out)
            p=(c-1)÷9+1; local_c=(c-1)%9+1; i=mod1(local_c,3);j=(local_c-1)÷3+1
            surface=map(a->a[p][i,j],Base.structdiff(f.input,(;t=nothing,q=nothing)))
            col=(;p=vec(s.centres.p[c,:]),p_geometry=vec(s.centres.p[c,:]),
                t=reverse(vec(f.input.t[p][i,j,:])),q=reverse(vec(f.input.q[p][i,j,:])),
                mass=reverse(vec(f.mass[p][i+2,j+2,:])),u=zeros(FT,6),v=zeros(FT,6),surface)
            d=zeros(FT,6)
            CSTD.tm5_calc_kzz_column!(d,out,col,f.field.params)
            @test isequal(vec(f.field.dkg[p][i,j,:]),vcat(reverse(d[1:5]),FT(0)))
        end
        # Geometry-independent D per unit area for identical resting columns.
        ratios=[f.field.dkg[p][i,j,k]/f.field.metrics.area[i,j] for i in 1:3,j in 1:3,p in 1:6,k in 1:6]
        tol=FT==Float64 ? 5e-12 : 5e-4
        @test all(k->all(x->isapprox(x,ratios[1,1,1,k];rtol=tol),ratios[:,:,:,k]),1:6)
    end
end

@testset "TM5 CS refresh rejects halo and input shape mismatches" begin
    f=tm5_cs_fixture()
    refresh(input,mass,fluxes)=refresh_tm5_calc_kzz!(f.field,input,mass,fluxes,f.grid.vertical,900.,DryBasis())
    @test_throws DimensionMismatch refresh(f.input,tm5_cs_fixture(;Hp=1).mass,f.fluxes)
    for key in (:am,:bm)
        bad=ntuple(p->p==6 ? zeros(7,7,6) : getproperty(f.fluxes,key)[p],6)
        fluxes=key===:am ? (;am=bad,bm=f.fluxes.bm) : (;am=f.fluxes.am,bm=bad)
        @test_throws DimensionMismatch refresh(f.input,f.mass,fluxes)
    end
    for key in CSBF.TM5_PBL_FIELD_NAMES
        # A bad last panel must also be rejected before any kernel runs.
        bad=ntuple(p->p==6 ? zeros(2,3,6) : getproperty(f.input,key)[p],6)
        @test_throws DimensionMismatch refresh(merge(f.input,NamedTuple{(key,)}((bad,))),f.mass,f.fluxes)
    end
end

@testset "TM5 CS v5 group, recipe and in-place reload" begin
    mktempdir() do dir
        f=tm5_cs_fixture()
        w1=tm5_cs_window(f)
        # Every section varies across cells, panels and windows, with valid units.
        input2=map(f.input) do panels
            ntuple(6) do p
                a=panels[p]
                a .+ 0.0001 .*(reshape(1:length(a),size(a)).+100p)
            end
        end
        w2=tm5_cs_window(f;input=input2)
        path=joinpath(dir,"cs.bin")
        kwargs=(;planet_radius=f.grid.horizontal.radius,mass_basis=:dry,surface_pressure_basis=TotalSurfacePressure(),include_tm5_pbl=true,steps_per_window=2)
        open_writer(;kw...)=CSMD.open_streaming_cs_transport_binary(path,3,6,6,2,f.grid.vertical;kwargs...,kw...)
        @test_throws ArgumentError open_writer()
        writer=open_writer(;flux_construction=CSBF.PointSampledSpectralFluxesHeader())
        CSMD.write_streaming_cs_window!(writer,w1,3,6)
        @test_throws ArgumentError CSMD.write_streaming_cs_window!(writer,merge(w2,(;tm5_pbl=merge(input2,(;q=ntuple(_->fill(1.,3,3,6),6))))),3,6)
        CSMD.write_streaming_cs_window!(writer,w2,3,6)
        CSMD.close_streaming_transport_binary!(writer)
        for FT in (Float64,Float32)
            driver=CSMD.TransportBinaryDriver(path;FT,Hp=2)
            try
                partial=copy(driver.reader.header.raw_header)
                partial["payload_sections"]=filter(!=("tm5_t"),String.(driver.reader.header.payload_sections))
                @test_throws ArgumentError CSBF.validate_transport_contract!(partial)
                @test CSMD.binary_capabilities(driver.reader).pbl_diffusion
                report=IOBuffer()
                CSMD.inspect_binary(path;io=report)
                @test occursin("tm5_calc_kzz_inputs_v1",String(take!(report)))
                win=CSMD.load_transport_window(driver,1)
                @test win.tm5_pbl==map(panels->map(a->FT.(a),panels),f.input)
                first_arrays=win.tm5_pbl
                CSMD.load_transport_window!(win,driver,2)
                fresh=CSMD.load_transport_window(driver,2)
                @test isequal(win.tm5_pbl,fresh.tm5_pbl)
                @test win.tm5_pbl === first_arrays
                @test all(name->size(getproperty(win.tm5_pbl,name)[1])==(name in (:t,:q) ? (3,3,6) : (3,3)),CSBF.TM5_PBL_FIELD_NAMES)
                for boundary in (false,true)
                    op=CSPH.build_runtime_diffusion(Dict("diffusion"=>Dict("kind"=>"tm5_calc_kzz","surface_flux_boundary"=>boundary)),driver,FT)
                    @test op.kz_field isa TM5CalcKzzCSField
                    grid=CSMD.driver_grid(driver)
                    initial=CSMD.load_transport_window(driver,1)
                    state=CubedSphereState(DryBasis,initial.air_mass;tracer=map(a->a.*FT(4e-4),initial.air_mass),halo_width=2)
                    model=TransportModel(state,initial.fluxes,grid,NoAdvection();diffusion=op)
                    sim=DrivenSimulation(model,driver;stop_window=2)
                    d1=map(copy,op.kz_field.dkg)
                    run_window!(sim);run_window!(sim)
                    @test !isequal(d1,op.kz_field.dkg)
                end
            finally
                close(driver.reader)
            end
        end
    end
end

@testset "TM5 CS direct D solve, conservation and transpose" begin
    rng=MersenneTwister(241)
    for FT in (Float64,Float32)
        f=tm5_cs_fixture(FT)
        refresh_tm5_calc_kzz!(f.field,f.input,f.mass,f.fluxes,f.grid.vertical,900.,DryBasis())
        op=ImplicitVerticalDiffusion(;kz_field=f.field)
        ws=DiffusionWorkspace(f.mass,2,0)
        foreach(a->fill!(a,FT(NaN)),ws.layer_thickness)
        rm=map(m->m.*FT(4e-4),f.mass)
        before=map(copy,rm)
        apply_vertical_diffusion_vmr!(rm,f.mass,op,ws,FT(900);halo_width=2)
        @test isequal(rm,before)
        x=map(m->m.*FT.(randn(rng,size(m))),f.mass)
        before=map(copy,x)
        apply_vertical_diffusion_vmr!(x,f.mass,op,ws,FT(900);halo_width=2)
        for p in 1:6,j in 3:5,i in 3:5
            @test sum(Float64,x[p][i,j,:]) ≈ sum(Float64,before[p][i,j,:]) rtol=(FT==Float64 ? 2e-13 : 3e-6)
        end
        seed=map(a->FT.(randn(rng,size(a))),x)
        back=map(copy,seed)
        AtmosTransport.Adjoints._apply_cs_diffusion_adjoint!(back,f.mass,op,ws,900.,nothing,f.grid.horizontal)
        inner(a,b)=sum(sum(view(a[p],3:5,3:5,:).*view(b[p],3:5,3:5,:)) for p in 1:6)
        @test inner(seed,x) ≈ inner(back,before) rtol=(FT==Float64 ? 1e-13 : 3e-6)
        # Capturing omits the bounded scratch, and later refresh cannot mutate D.
        frozen=AtmosTransport.Models.Runner._adjoint_window_operator(op)
        @test frozen.kz_field.scratch === nothing && frozen.kz_field.metrics === nothing
        # Object overhead differs between Julia versions (1.10 adds ~0.6 KB here).
        @test Base.summarysize(frozen) < 1.1*sum(sizeof,frozen.kz_field.dkg)+4096
        @test isequal(frozen.kz_field.dkg,f.field.dkg)
        fill!(f.field.dkg[1],0)
        @test !isequal(frozen.kz_field.dkg,f.field.dkg)
    end
end

@testset "CS line-integral wind recovery converges at second order" begin
    MC=AtmosTransport.Preprocessing.MassClosure
    G=AtmosTransport.Grids
    for convention in (G.GnomonicPanelConvention(),G.GEOSNativePanelConvention()), flow in (:solid,:deformation)
        errors=Float64[]; shear_errors=Float64[]
        for n in (12,24,48)
            f=tm5_cs_fixture(;Nc=n,Hp=1,convention)
            source=LatLonMesh(;Nx=4n,Ny=2n)
            lon=vec([l for l in source.λᶜ,lat in source.φᶜ])
            lat=vec([lat for l in source.λᶜ,lat in source.φᶜ])
            wind(l,phi)=flow===:solid ? (15cosd(phi),0.) :
                (15sind(l)^2*sind(2phi),15sind(2l)*cosd(phi))
            rotate(w,k)=(cosd(30k)*w[1]-sind(30k)*w[2],sind(30k)*w[1]+cosd(30k)*w[2])
            u=hcat([first.(rotate.(wind.(lon,lat),k)) for k in 1:6]...)
            v=hcat([last.(rotate.(wind.(lon,lat),k)) for k in 1:6]...)
            dp=repeat(reshape(diff(f.grid.vertical.A).*(1-.005),1,6),length(lon),1)
            ft=G.build_cs_global_face_table(n,G.panel_connectivity_for(f.grid.horizontal.convention))
            method=MC.LineIntegralFaceFluxes(source,f.grid.horizontal,ft;samples_per_face=8)
            am=ntuple(_->zeros(n+1,n,6),6);bm=ntuple(_->zeros(n,n+1,6),6)
            MC.line_integral_face_fluxes!(am,bm,method,u,v,dp,9.80665,900.,6)
            for p in 1:6
                f.fluxes.am[p][2:n+2,2:n+1,:] .= am[p]
                f.fluxes.bm[p][2:n+1,2:n+2,:] .= bm[p]
            end
            refresh_tm5_calc_kzz!(f.field,f.input,f.mass,f.fluxes,f.grid.vertical,900.,DryBasis())
            squares=Float64[]; shear_squares=Float64[]
            for p in 1:6
                ll=G.panel_cell_center_lonlat(f.grid.horizontal,p)
                for j in 1:n,i in 1:n
                    c=i+(j-1)*n+(p-1)*n*n
                    w6=rotate(wind(ll[1][i,j],ll[2][i,j]),6)
                    w5=rotate(wind(ll[1][i,j],ll[2][i,j]),5)
                    U,V=f.field.scratch.centres.u,f.field.scratch.centres.v
                    push!(squares,(hypot(U[c,1],V[c,1])-hypot(w6...))^2)
                    push!(shear_squares,(hypot(U[c,2]-U[c,1],V[c,2]-V[c,1])-hypot((w5.-w6)...))^2)
                end
            end
            push!(errors,sqrt(sum(squares)/length(squares)))
            push!(shear_errors,sqrt(sum(shear_squares)/length(shear_squares)))
        end
        @info "TM5 CS wind recovery RMS" convention flow errors shear_errors
        @test all(errors[2:3].<errors[1:2]./3)
        @test all(shear_errors[2:3].<shear_errors[1:2]./3)
    end
end

@testset "CS random D, empty layers, packed solve and adjoint validation" begin
    f=tm5_cs_fixture()
    rng=MersenneTwister(82)
    for d in f.field.dkg
        d[:,:,3:5] .= rand(rng,3,3,3).*1e13
    end
    op=ImplicitVerticalDiffusion(;kz_field=f.field)
    AtmosTransport.Adjoints._validate_cs_diffusion_kz_for_adjoint(op)
    ws=DiffusionWorkspace(f.mass,2,2)
    x=map(m->m.*randn(rng,size(m)),f.mass)
    second=map(m->m.*randn(rng,size(m)),f.mass)
    packed=map((a,b)->cat(a,b;dims=4),x,second)
    apply_vertical_diffusion_vmr!(packed,f.mass,op,ws,900.;halo_width=2)
    for (t,a) in enumerate((x,second))
        apply_vertical_diffusion_vmr!(a,f.mass,op,ws,900.;halo_width=2)
        @test all(isequal(view(packed[p],:,:,:,t),a[p]) for p in 1:6)
    end
    f.mass[2][3,3,4]=0
    before=map(m->m.*randn(rng,size(m)),f.mass)
    x=map(copy,before)
    seed=map(m->randn(rng,size(m)),f.mass)
    back=map(copy,seed)
    apply_vertical_diffusion_vmr!(x,f.mass,op,ws,900.;halo_width=2)
    AtmosTransport.Adjoints._apply_cs_diffusion_adjoint!(back,f.mass,op,ws,900.,nothing,f.grid.horizontal)
    inner(a,b)=sum(sum(view(a[p],3:5,3:5,:).*view(b[p],3:5,3:5,:)) for p in 1:6)
    @test inner(seed,x) ≈ inner(back,before) rtol=1e-13
    @test all(p->isequal(x[p][3:5,3:5,1:2],before[p][3:5,3:5,1:2]),1:6)
    @test back[2][3,3,4]==0
    for (k,value) in ((1,1.),(6,1.),(4,-1.),(4,NaN))
        old=f.field.dkg[1][1,1,k]
        f.field.dkg[1][1,1,k]=value
        @test_throws ArgumentError AtmosTransport.Adjoints._validate_cs_diffusion_kz_for_adjoint(op)
        f.field.dkg[1][1,1,k]=old
    end
end

@testset "CS shear frame invariance and LL column physics" begin
    # Same physical column in a 2.5-degree LL cell and in each CS frame.
    llmesh=LatLonMesh(;Nx=144,Ny=72)
    metrics=CSPH._tm5_metrics(llmesh,Float64)
    c=144*36+72
    llmetrics=CSTD.TM5LLMetrics([metrics.area[c]],[metrics.dx[c]],[metrics.dy[c]],1)
    f=tm5_cs_fixture(;Nc=48,Hp=1)
    vc=f.grid.vertical
    llmass=reshape([llmetrics.area[1]*diff(vc.A)[k]/9.80665*(1-.005) for k in 1:6],1,1,6)
    llinput=map(f.input) do panels
        a=first(panels)
        ndims(a)==3 ? reshape(copy(a[1,1,:]),1,1,6) : fill(a[1,1],1,1)
    end
    llfield=CSTD.TM5CalcKzzField(zeros(1,1,6),llmetrics,vc)
    u=[20.,15.,12.,8.,5.,2.];v=[-3.,-2.,1.,3.,4.,5.]
    llflux=(;am=zeros(2,1,6),bm=zeros(1,2,6))
    for k in 1:6
        llflux.am[:,:,k] .= llmass[1,1,k]*900*u[k]/llmetrics.dx[1]
        llflux.bm[:,:,k] .= llmass[1,1,k]*900*v[k]/llmetrics.dy[1]
    end
    refresh_tm5_calc_kzz!(llfield,llinput,llmass,llflux,vc,900.,DryBasis())
    for convention in (AtmosTransport.Grids.GnomonicPanelConvention(),AtmosTransport.Grids.GEOSNativePanelConvention())
        f=tm5_cs_fixture(;Nc=48,Hp=1,convention)
        results=[]
        for theta in (0.,.73)
            ur=cos(theta).*u.-sin(theta).*v
            vr=sin(theta).*u.+cos(theta).*v
            for p in 1:6,k in 1:6
                i,j,h=1,2,1
                cosine,sine=f.field.metrics.basis[p]
                normal1=sine[i,j]*ur[k]-cosine[i,j]*vr[k]
                normal2=vr[k]
                mass=f.mass[p][i+h,j+h,k];area=f.field.metrics.area[i,j]
                for face in i:i+1
                    f.fluxes.am[p][face+h,j+h,k]=mass/area*900*normal1*f.field.metrics.lx[face,j]
                end
                for face in j:j+1
                    f.fluxes.bm[p][i+h,face+h,k]=mass/area*900*normal2*f.field.metrics.ly[i,face]
                end
            end
            refresh_tm5_calc_kzz!(f.field,f.input,f.mass,f.fluxes,vc,900.,DryBasis())
            push!(results,[f.field.dkg[p][1,2,k]/f.field.metrics.area[1,2] for p in 1:6,k in 1:6])
        end
        @test results[1] ≈ results[2] rtol=1e-12
        @test all(p->isapprox(vec(results[1][p,:]),vec(llfield.dkg)./llmetrics.area[1];rtol=1e-12),1:6)
    end
end

@testset "Frozen CS D retains at most 10% object overhead at C8" begin
    f=tm5_cs_fixture(;Nc=8)
    op=ImplicitVerticalDiffusion(;kz_field=f.field)
    frozen=AtmosTransport.Models.Runner._adjoint_window_operator(op)
    @test Base.summarysize(frozen) <= 1.1sum(sizeof,frozen.kz_field.dkg)
end
