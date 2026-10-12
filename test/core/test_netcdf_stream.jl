using Test, AtmosTransport, NCDatasets
using AtmosTransport.Output
using AtmosTransport.Grids: ncells
const O = AtmosTransport.Output

@testset "Incremental NetCDF matches batch output and retains no frames" begin
    for FT in (Float32, Float64), topology in (:ll, :rg, :cs), mode in ("full", "selected", "none")
        Nz = 4
        mesh = topology === :ll ? LatLonMesh(;Nx=6,Ny=4,FT) :
               topology === :rg ? ReducedGaussianMesh(FT[-45,45],[4,8];FT) :
               CubedSphereMesh(;Nc=4,Hp=3,FT)
        vc = HybridSigmaPressure(zeros(FT,Nz+1), FT.(range(0,1;length=Nz+1)))
        grid = AtmosGrid(mesh,vc,CPU();FT)
        shape = topology === :ll ? (6,4,Nz) : topology === :rg ? (ncells(mesh),Nz) : (10,10,Nz)
        a = reshape(FT.(1:prod(shape)),shape)
        air = topology === :cs ? ntuple(p -> a .* FT(p),6) : a
        rm = topology === :cs ? map(m -> m .* FT(0.0004),air) : air .* FT(0.0004)
        state = topology === :cs ? CubedSphereState(DryBasis,air;co2=rm,other=rm,halo_width=3) :
                CellState(DryBasis,air;co2=rm,other=rm)
        model = (;state,grid)
        fields = output_field_spec(Dict{String,Any}("layers"=>mode,"levels"=>[1,3],
            "air_mass_layers"=>mode,"tracers"=>["co2"],
            "column_mean"=>true,"column_mass_per_area"=>true))
        options = SnapshotWriteOptions(;float_type=FT,deflate_level=1)
        mktempdir() do dir
            path = joinpath(dir,"stream.nc")
            stream = O.NetCDFSnapshotStream(path,grid;fields,options)
            @test !isfile(path)
            reference = AbstractSnapshotFrame[]
            retained = 0
            last_frame = nothing
            for time in 0:3
                # Changing data and times catches an append that overwrites record 1.
                topology === :cs ? foreach(m -> (m .*= FT(1.01)),state.air_mass) :
                                   (state.air_mass .*= FT(1.01))
                halo_width = topology === :cs ? 3 : 0
                full = capture_snapshot(model;time_hours=time,halo_width)
                push!(reference,full)
                selected = capture_snapshot(model;time_hours=time,halo_width,fields)
                O.append_snapshot!(stream,selected)
                @test stream.count == time + 1
                time == 0 && (retained = Base.summarysize(stream))
                @test Base.summarysize(stream) == retained
                NCDataset(path) do ds
                    @test ds["time"][:] == Float64.(0:time)
                    @test ds.attrib["completed_snapshots"] == time + 1
                end
                last_frame = selected
            end
            batch = write_snapshot_netcdf(joinpath(dir,"batch.nc"),reference,grid;fields,options)
            NCDataset(batch) do x
                NCDataset(path) do y
                    @test Set(keys(x)) == Set(keys(y))
                    @test Dict(x.dim) == Dict(y.dim)
                    for name in keys(x)
                        @test isequal(x[name][:],y[name][:])
                        @test Dict(x[name].attrib) == Dict(y[name].attrib)
                    end
                    @test y.attrib["output_fields"] == x.attrib["output_fields"]
                    @test endswith(y.attrib["history"],"with 4 frame(s)")
                end
            end
            before = read(path)
            @test_throws ArgumentError O.append_snapshot!(stream,last_frame)
            @test read(path) == before
            @test !stream.failed
            @test stream.count == 4
            close(stream)
            @test stream.closed
            @test stream.dataset === nothing
            @test_throws ArgumentError O.append_snapshot!(stream,last_frame)
            snap = AtmosTransport.Visualization.open_snapshot(path)
            ref = AtmosTransport.Visualization.open_snapshot(batch)
            @test AtmosTransport.Visualization.snapshot_times(snap) == Float64.(0:3)
            @test :co2 in AtmosTransport.Visualization.available_variables(snap)
            for ti in 1:4
                got = AtmosTransport.Visualization.fieldview(snap,:co2;time=ti)
                expected = AtmosTransport.Visualization.fieldview(ref,:co2;time=ti)
                @test isequal(got.values,expected.values)
            end
            if topology === :cs
                @test snap.topology.nlevel == Nz
                if mode == "selected"
                    layer = AtmosTransport.Visualization.fieldview(
                        snap, :co2;
                        transform=AtmosTransport.Visualization.LevelSlice(), level=3)
                    NCDataset(path) do ds
                        @test layer.values == ds["co2"][:,:,:,2,1]
                    end
                    @test_throws ArgumentError AtmosTransport.Visualization.fieldview(
                        snap, :co2;
                        transform=AtmosTransport.Visualization.SurfaceSlice())
                end
            end
        end
    end
end

@testset "NetCDF stream rejects incompatible frames before touching output" begin
    mesh = LatLonMesh(;Nx=3,Ny=2)
    vc = HybridSigmaPressure(zeros(3),[0.0,0.5,1.0])
    grid = AtmosGrid(mesh,vc,CPU())
    air = ones(3,2,2)
    frame = SnapshotFrame(0.0,air,Dict(:co2=>copy(air)),:dry)
    mktempdir() do dir
        path = joinpath(dir,"out.nc")
        write(path,"existing output")
        stream = O.NetCDFSnapshotStream(path,grid)
        @test_throws ArgumentError O.append_snapshot!(stream,
            SnapshotFrame(NaN,air,frame.tracers,:dry))
        @test read(path,String) == "existing output"
        O.append_snapshot!(stream,frame)
        before = read(path)
        for bad in (SnapshotFrame(1.0,air,Dict(:other=>copy(air)),:dry),
                    SnapshotFrame(1.0,air,frame.tracers,:moist),
                    SnapshotFrame(1.0,ones(3,2,3),Dict(:co2=>ones(3,2,3)),:dry))
            @test_throws Exception O.append_snapshot!(stream,bad)
            @test read(path) == before
        end
        O.append_snapshot!(stream,SnapshotFrame(1.0,air,frame.tracers,:dry))
        @test stream.count == 2
        close(stream)
        close(stream) # Idempotent cleanup.
        directory = joinpath(dir,"directory")
        mkdir(directory)
        failed = O.NetCDFSnapshotStream(directory,grid)
        @test_throws Exception O.append_snapshot!(failed,frame)
        @test failed.failed
        @test failed.closed
        @test failed.dataset === nothing
        rm(directory)
        @test_throws ArgumentError O.append_snapshot!(failed,frame)
        collision = O.NetCDFSnapshotStream(joinpath(dir,"collision.nc"),grid)
        @test_throws Exception O.append_snapshot!(collision,
            SnapshotFrame(0.0,air,Dict(:air_mass=>copy(air)),:dry))
        @test collision.failed
        @test collision.closed
    end
end
