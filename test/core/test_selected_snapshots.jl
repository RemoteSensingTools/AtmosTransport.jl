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
