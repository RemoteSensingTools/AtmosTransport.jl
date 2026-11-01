using Test, LinearAlgebra, Adapt
using AtmosTransport
using AtmosTransport.Operators.Diffusion: TM5CalcKzzField, refresh_tm5_calc_kzz!,
    ImplicitVerticalDiffusion, DiffusionWorkspace, apply_vertical_diffusion_vmr!
const MD = AtmosTransport.MetDrivers
const BF = MD.BinaryFormat
const TD = AtmosTransport.Operators.Diffusion
const PH = AtmosTransport.Models.Physics

function pbl_runtime_fixture(mesh,::Type{FT}) where FT
    vc=HybridSigmaPressure(FT[0,10000,30000,60000,80000,95000,100000],zeros(FT,7))
    grid=AtmosGrid(mesh,vc,CPU();FT)
    shape=mesh isa LatLonMesh ? (mesh.Nx,mesh.Ny,6) : (ncells(mesh),6)
    spatial=shape[1:end-1]
    nc=prod(spatial)
    mass=reshape([FT(PH._tm5_metrics(mesh,FT).area[c]*(vc.A[k+1]-vc.A[k])/9.80665*(1-0.005)) for c in 1:nc,k in 1:6],shape)
    t=reshape([FT(300*(0.5*(vc.A[k]+vc.A[k+1])/1e5)^(287.2978576364893/1004)) for c in 1:nc,k in 1:6],shape)
    input=(;t,q=fill(FT(.005),shape),lsm=fill(FT(.4),spatial),sr=fill(FT(.03),spatial),
        fsr=fill(FT(.1),spatial),u10=fill(FT(6),spatial),v10=fill(FT(2),spatial),
        ewss=fill(FT(.1),spatial),nsss=fill(FT(.05),spatial),sshf=fill(FT(-100),spatial),
        slhf=fill(FT(-30),spatial),oro=zeros(FT,spatial))
    fluxes=allocate_face_fluxes(mesh,6;FT,basis=DryBasis)
    fill!(fluxes.cm,0)
    if mesh isa LatLonMesh
        fill!(fluxes.am,0);fill!(fluxes.bm,0)
    else
        fill!(fluxes.horizontal_flux,0)
    end
    return (;grid,mass,input,fluxes,shape)
end

@testset "TM5 KA CPU refresh, direct-D solve and isolation" begin
    for FT in (Float64,Float32),mesh in (LatLonMesh(;Nx=4,Ny=3),ReducedGaussianMesh([-60.,0.,60.],[4,8,4]))
        f=pbl_runtime_fixture(mesh,FT)
        if mesh isa LatLonMesh
            for k in 1:6,j in 1:mesh.Ny,i in 1:mesh.Nx+1
                f.fluxes.am[i,j,k]=FT(100i+10j+k)
            end
            for k in 1:6,j in 1:mesh.Ny+1,i in 1:mesh.Nx
                f.fluxes.bm[i,j,k]=FT(50i+20j+k)
            end
        else
            for k in 1:6,face in 1:nfaces(mesh)
                f.fluxes.horizontal_flux[face,k]=FT(10face+k)
            end
        end
        field=TM5CalcKzzField(zeros(FT,f.shape),PH._tm5_metrics(mesh,FT),f.grid.vertical)
        @test adapt(Array,field).params.lmax==field.params.lmax
        refresh_tm5_calc_kzz!(field,f.input,f.mass,f.fluxes,f.grid.vertical,FT(900),DryBasis())
        @test all(isfinite,field.dkg) && all(>=(0),field.dkg)
        # Wind/mass/pressure adapters are in runtime state order; column physics is bottom-up.
        nc=prod(f.shape[1:end-1]);mm=reshape(f.mass,nc,6)
        for c in 1:nc,k in 1:6
            if mesh isa LatLonMesh
                i=mod1(c,mesh.Nx);j=(c-1)÷mesh.Nx+1
                fw,fe=f.fluxes.am[i,j,k],f.fluxes.am[i+1,j,k]
                fs,fn=f.fluxes.bm[i,j,k],f.fluxes.bm[i,j+1,k]
            else
                edge=AtmosTransport.Grids.cell_faces(mesh,c)
                # Derive meridional sides from face neighbours, independently of cached CSR offsets.
                fw,fe=f.fluxes.horizontal_flux[edge[1],k],f.fluxes.horizontal_flux[edge[2],k]
                south=[face for face in edge[3:end] if last(AtmosTransport.Grids.face_cells(mesh,face))==c]
                north=[face for face in edge[3:end] if first(AtmosTransport.Grids.face_cells(mesh,face))==c]
                fs=sum(face->f.fluxes.horizontal_flux[face,k],south)
                fn=sum(face->f.fluxes.horizontal_flux[face,k],north)
            end
            ue=field.metrics.dx[c]*(fw+fe)/2/mm[c,k]/FT(900)
            ve=field.metrics.dy[c]*(fs+fn)/2/mm[c,k]/FT(900)
            @test field.scratch.centres.u[c,7-k] ≈ ue
            @test field.scratch.centres.v[c,7-k] ≈ ve
        end
        @test vec(field.scratch.centres.p[:,1]) ≈ fill(FT(1e5),prod(f.shape[1:end-1]))
        s=field.scratch
        out=map(a->view(a,1,:),s.out)
        surface=map(a->first(a),Base.structdiff(f.input,(;t=nothing,q=nothing)))
        col=(;p=view(s.centres.p,1,:),p_geometry=view(s.centres.p,1,:),t=view(s.centres.t,1,:),q=view(s.centres.q,1,:),mass=view(s.centres.mass,1,:),u=view(s.centres.u,1,:),v=view(s.centres.v,1,:),surface)
        d=zeros(FT,6);r=TD.tm5_calc_kzz_column!(d,out,col,field.params)
        @test vec(reshape(field.dkg,:,6)[1,:]) == vcat(reverse(d[1:5]),FT(0))
        op=ImplicitVerticalDiffusion(;kz_field=field)
        ws=DiffusionWorkspace(f.mass,3)
        fill!(ws.layer_thickness,FT(NaN)) # direct exchange must never read dz
        nc=prod(f.shape[1:end-1]);m=reshape(f.mass,nc,6)
        # Exercise weak and stiff exchange, signed tracer, surface plume and constant VMR.
        for strength in (FT(1e-6),FT(1),FT(1e3))
            D=reshape(field.dkg,nc,6)
            D[:,1:2].=0;D[:,3:5].=FT(1e13)*strength
            rm=zeros(FT,f.shape...,3)
            rr=reshape(rm,nc,6,3)
            for c in 1:nc,k in 1:6
                rr[c,k,1]=m[c,k]*FT(4e-4)
                rr[c,k,2]=m[c,k]*(k==6 ? FT(1e-5) : FT(0))
                rr[c,k,3]=m[c,k]*FT((k-3)*1e-6)
            end
            before=copy(rr)
            apply_vertical_diffusion_vmr!(rm,f.mass,op,ws,FT(900))
            @test rr[:,1:2,:]==before[:,1:2,:]
            @test rr[:,:,1]==before[:,:,1]
            for c in 1:nc,tr in 2:3
                A=Matrix(Diagonal(Float64.(m[c,:])))
                for k in 1:5
                    a=900*Float64(D[c,k]); A[k,k]+=a;A[k+1,k+1]+=a;A[k,k+1]-=a;A[k+1,k]-=a
                end
                reference=Float64.(m[c,:]).*(A\Float64.(before[c,:,tr]))
                @test rr[c,:,tr] ≈ reference rtol=(FT==Float64 ? 1e-11 : 2e-4) atol=(FT==Float64 ? 1e-4 : 2e3)
                @test sum(Float64,rr[c,:,tr]) ≈ sum(Float64,before[c,:,tr]) rtol=(FT==Float64 ? 1e-13 : 2e-7)
            end
        end
    end
end

@testset "TM5 v5 LL/RG input round trip and runtime recipe" begin
    mktempdir() do dir
        for mesh in (LatLonMesh(;Nx=4,Ny=3),ReducedGaussianMesh([-60.,0.,60.],[4,8,4]))
            f=pbl_runtime_fixture(mesh,Float64)
            w=mesh isa LatLonMesh ? (;m=f.mass,ps=fill(1e5,f.shape[1:end-1]),am=f.fluxes.am,bm=f.fluxes.bm,cm=f.fluxes.cm,tm5_pbl=f.input) :
                (;m=f.mass,ps=fill(1e5,f.shape[1:end-1]),hflux=f.fluxes.horizontal_flux,cm=f.fluxes.cm,tm5_pbl=f.input)
            path=joinpath(dir,"test.bin")
            kwargs=(;mass_basis=:dry,surface_pressure_basis=TotalSurfacePressure(),source_flux_sampling=:window_start_endpoint,steps_per_window=2)
            @test_throws ArgumentError MD.write_transport_binary(path,f.grid,[w];kwargs...)
            MD.write_transport_binary(path,f.grid,[w,w];kwargs...,flux_construction=BF.PointSampledSpectralFluxesHeader())
            reader=MD.TransportBinaryReader(path)
            @test reader.header.format_version==5
            partial=copy(reader.header.raw_header)
            partial["payload_sections"]=filter(!=("tm5_t"),String.(reader.header.payload_sections))
            @test_throws ArgumentError BF.validate_transport_contract!(partial)
            invalid=merge(w,(;tm5_pbl=merge(f.input,(;q=fill(1.,f.shape)))))
            @test_throws ArgumentError MD.write_transport_binary(joinpath(dir,"bad.bin"),f.grid,[invalid];kwargs...,flux_construction=BF.PointSampledSpectralFluxesHeader())
            @test all(s->s in reader.header.payload_sections,BF.TM5_PBL_PAYLOAD_SECTIONS)
            driver=MD.TransportBinaryDriver(path;FT=Float64)
            window=MD.load_transport_window(driver,1)
            @test window.tm5_pbl==f.input
            @test adapt(Array,window).tm5_pbl==f.input
            cfg=Dict("diffusion"=>Dict("kind"=>"tm5_calc_kzz"))
            op=PH.build_runtime_diffusion(cfg,driver,Float64)
            @test op.kz_field isa TM5CalcKzzField
            @test_throws ArgumentError PH.diffusion_spec(Dict("kind"=>"tm5_calc_kzz","surface_flux_boundary"=>1))
            if mesh isa ReducedGaussianMesh
                @test_throws ArgumentError PH.build_runtime_diffusion(Dict("diffusion"=>Dict("kind"=>"tm5_calc_kzz","surface_flux_boundary"=>true)),driver,Float64)
            end
            # Use real DrivenSimulation refresh (after reset) at construction and handoff.
            state=CellState(DryBasis,copy(f.mass);tracer=f.mass .* 4e-4)
            model=TransportModel(state,f.fluxes,f.grid,NoAdvection();diffusion=op)
            sim=DrivenSimulation(model,driver;stop_window=2)
            @test all(isfinite,op.kz_field.dkg) && any(>(0),op.kz_field.dkg)
            run_window!(sim)
            run_window!(sim)
            @test get_tracer(state,:tracer) ≈ f.mass .* 4e-4
            close(driver.reader);close(reader)
        end
    end
end

@testset "TM5 active entrainment warning occurs once per run" begin
    f=pbl_runtime_fixture(LatLonMesh(;Nx=4,Ny=3),Float64)
    # All layer midpressures exceed the mixing cutoff. A neutral theta profile
    # with positive buoyancy places h at the highest centre and makes the last
    # active entrainment gradient zero; unused singular interfaces must not count.
    vc=HybridSigmaPressure([10000.,20000.,40000.,70000.,90000.,105000.,110000.],zeros(7))
    input=merge(f.input,(;q=zeros(size(f.mass))))
    metrics=PH._tm5_metrics(f.grid.horizontal,Float64)
    prime=TM5CalcKzzField(zeros(f.shape),metrics,vc)
    refresh_tm5_calc_kzz!(prime,input,f.mass,f.fluxes,vc,900.,DryBasis())
    p=prime.scratch.centres.p;nc=prod(f.shape[1:end-1])
    temperature=reshape(input.t,nc,6)
    for c in 1:nc,k in 1:6
        l=7-k;pf=(p[c,l]+p[c,l+1])/2
        temperature[c,k]=256/(1e5/pf)^(prime.params.constants.r_air/prime.params.constants.cp_air)
    end
    for run in 1:2
        field=TM5CalcKzzField(zeros(f.shape),metrics,vc)
        @test_logs (:warn,r"non-finite active entrainment interfaces") refresh_tm5_calc_kzz!(field,input,f.mass,f.fluxes,vc,900.,DryBasis())
        @test all(==(1),field.scratch.guarded)
        @test field.guard_warned[]
        @test_logs refresh_tm5_calc_kzz!(field,input,f.mass,f.fluxes,vc,900.,DryBasis())
        @test TD.update_field!(field,0.) === field
    end
end
