using Test, AtmosTransport, NCDatasets
using AtmosTransport.Output
using AtmosTransport.Grids: ncells

@testset "Column fallback retains cancellation across slab boundaries" begin
    O = AtmosTransport.Output
    for shape in ((7, 33), (4, 3, 33)), levels in ((1, 2, 3), (15, 16, 17), (16, 17, 33))
        values = zeros(Float32, shape)
        axis = ndims(values)
        selectdim(values, axis, levels[1]) .= 2f0^24
        selectdim(values, axis, levels[2]) .= 1f0
        selectdim(values, axis, levels[3]) .= -2f0^24
        actual = O._backend_column_sum(values, Float32)
        @test eltype(actual) === Float64
        @test actual == ones(Float64, shape[1:end-1])
        @test actual == O._column_sum(values)
    end
    @test O._backend_column_sum(zeros(Float32, 4, 0), Float32) == zeros(4)
end

@testset "Selected capture preserves full-capture NetCDF values" begin
    for FT in (Float32, Float64), topology in (:ll, :rg, :cs)
        Nz = 4
        mesh = topology === :ll ? LatLonMesh(; Nx=6, Ny=4, FT) :
               topology === :rg ? ReducedGaussianMesh(FT[-45,45], [4,8]; FT) :
               CubedSphereMesh(; Nc=4, Hp=3, FT)
        vc = HybridSigmaPressure(zeros(FT,Nz+1), collect(range(zero(FT),one(FT);length=Nz+1)))
        grid = AtmosGrid(mesh, vc, CPU(); FT)
        shape = topology === :ll ? (6,4,Nz) : topology === :rg ? (ncells(mesh),Nz) : (10,10,Nz)
        a = reshape(FT.(1:prod(shape)), shape)
        q = reshape(FT.(range(0.0003,0.0006;length=prod(shape))), shape)
        air = topology === :cs ? ntuple(p -> a .* FT(p),6) : a
        rm = topology === :cs ? map(m -> m .* q,air) : a .* q
        state = topology === :cs ? CubedSphereState(DryBasis,air; co2=rm,unused=rm,halo_width=3) :
                CellState(DryBasis,air; co2=rm,unused=rm)
        model = (; state, grid)
        halo_width = topology === :cs ? 3 : 0
        full = capture_snapshot(model; halo_width)
        for mode in ("none","selected","full")
            fields = output_field_spec(Dict{String,Any}(
                "tracers"=>["co2"], "layers"=>mode, "levels"=>[1,3],
                "air_mass_layers"=>mode, "air_mass"=>true,
                "air_mass_per_area"=>true, "column_air_mass_per_area"=>true,
                "column_mean"=>true, "column_mass_per_area"=>true))
            captured = capture_snapshot(model; halo_width, fields)
            @test Set(keys(captured.tracers)) == Set([:co2])
            @test captured.nlevel == Nz
            mode == "none" && @test isempty(captured.levels)
            mode == "selected" && @test captured.levels == [1,3]
            mktempdir() do dir
                options = SnapshotWriteOptions(;float_type=Float64)
                p = write_snapshot_netcdf(joinpath(dir,"full.nc"),[full],grid;fields,options)
                q = write_snapshot_netcdf(joinpath(dir,"selected.nc"),[captured],grid;fields,options)
                NCDataset(p) do x
                    NCDataset(q) do y
                        @test Set(keys(x)) == Set(keys(y))
                        for name in keys(x)
                            @test isequal(x[name][:], y[name][:])
                        end
                    end
                end
            end
        end
    end
end

@testset "Cubed-sphere writer ignores air layers absent from selected capture" begin
    # With both air-mass outputs disabled, selected capture stores only the
    # tracer layers, so the configured air-layer selection is not captured.
    for FT in (Float32, Float64)
        Nz, Nc, Hp = 4, 4, 3
        mesh = CubedSphereMesh(; Nc, Hp, FT)
        vc = HybridSigmaPressure(zeros(FT,Nz+1), collect(range(zero(FT),one(FT);length=Nz+1)))
        grid = AtmosGrid(mesh, vc, CPU(); FT)
        shape = (Nc+2Hp, Nc+2Hp, Nz)
        a = reshape(FT.(1:prod(shape)), shape)
        q = reshape(FT.(range(0.0003,0.0006;length=prod(shape))), shape)
        air = ntuple(p -> a .* FT(p),6)
        state = CubedSphereState(DryBasis,air; co2=map(m -> m .* q,air),halo_width=Hp)
        model = (; state, grid)
        full = capture_snapshot(model; halo_width=Hp)
        for (air_layers, tracer_layers) in (("full","selected"), ("selected","none"))
            fields = output_field_spec(Dict{String,Any}(
                "tracers"=>["co2"], "layers"=>tracer_layers, "levels"=>[1,3],
                "air_mass_layers"=>air_layers, "air_mass"=>false,
                "air_mass_per_area"=>false, "column_air_mass_per_area"=>true,
                "column_mean"=>true, "column_mass_per_area"=>true))
            captured = capture_snapshot(model; halo_width=Hp, fields)
            @test captured.levels == (tracer_layers == "selected" ? [1,3] : Int[])
            mktempdir() do dir
                options = SnapshotWriteOptions(;float_type=Float64)
                pfull = write_snapshot_netcdf(joinpath(dir,"full.nc"),[full],grid;fields,options)
                psel = write_snapshot_netcdf(joinpath(dir,"selected.nc"),[captured],grid;fields,options)
                NCDataset(pfull) do x
                    NCDataset(psel) do y
                        @test !haskey(y,"air_mass") && !haskey(y,"air_mass_per_area")
                        @test Set(keys(x)) == Set(keys(y))
                        for name in keys(x)
                            @test isequal(x[name][:], y[name][:])
                        end
                        if tracer_layers == "selected"
                            # Tracer mass is air × q, so the written VMR is q on
                            # the panel interior at the selected levels.
                            r = Hp+1:Hp+Nc
                            expected = Array{Float64}(undef, Nc, Nc, 6, 2)
                            for panel in 1:6
                                expected[:, :, panel, :] = Float64.(q[r, r, [1,3]])
                            end
                            @test y["co2"].var[:, :, :, :, 1] ≈ expected rtol=4eps(FT)
                        else
                            @test !haskey(y,"co2")
                        end
                    end
                end
            end
        end
    end
end

@testset "Selected signed totals survive omitted layers and diagnostics" begin
    mesh = LatLonMesh(;Nx=4,Ny=2,FT=Float64)
    grid = AtmosGrid(mesh,HybridSigmaPressure(zeros(3),[0.0,0.5,1.0]),CPU())
    air = ones(4,2,2)
    signed = zeros(4,2,2)
    signed[1]=1e30; signed[2]=1; signed[3]=-1e30; signed[1,1,2]=7
    state = CellState(DryBasis,air;signed)
    model = (;state,grid)
    fields = output_field_spec(Dict("layers"=>"none","air_mass"=>false,
        "air_mass_per_area"=>false,"column_air_mass_per_area"=>false,
        "column_mean"=>false,"column_mass_per_area"=>false))
    frame = capture_snapshot(model;fields)
    @test isempty(frame.air_mass)
    @test isempty(frame.tracers[:signed])
    @test frame.column_air === nothing
    @test isempty(frame.column_tracers)
    @test frame.tracer_total_mass[:signed] == 8.0
    @test frame.tracer_total_mass == capture_snapshot(model).tracer_total_mass
    mktempdir() do dir
        stream = AtmosTransport.Output.NetCDFSnapshotStream(joinpath(dir,"out.nc"),grid;fields)
        try
            AtmosTransport.Output.append_snapshot!(stream,frame)
            get_tracer(state,:signed)[2]=2
            AtmosTransport.Output.append_snapshot!(stream,capture_snapshot(model;fields,time_hours=1))
        finally
            close(stream)
        end
        NCDataset(stream.path) do ds
            @test ds["signed_total_mass"][:] == [8.0,9.0]
            @test !haskey(ds,"signed")
            @test !haskey(ds,"signed_column_mean")
        end
        # A malformed diagnostic must not remove a previous output file.
        bad = capture_snapshot(model;fields)
        bad.tracer_total_mass[:signed]=NaN
        path=joinpath(dir,"previous.nc")
        write(path,"previous result")
        @test_throws ArgumentError write_snapshot_netcdf(path,[bad],grid;fields)
        @test read(path,String)=="previous result"
    end
end
