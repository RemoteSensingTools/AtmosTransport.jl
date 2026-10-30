using Test, AtmosTransport, NCDatasets
const R = AtmosTransport.Models.DrivenRunner
const O = AtmosTransport.Output

@testset "Run output drains background writes on exceptional exit" begin
    output = R.RunSnapshotOutput()
    started, release, finished = Channel{Nothing}(1), Channel{Nothing}(1), Channel{Nothing}(1)
    writer = output.pending_write = Threads.@spawn begin
        put!(started,nothing)
        take!(release)
        put!(finished,nothing)
    end
    take!(started)
    runner = @async try
        R._with_run_resource(output) do
            error("transport failed after a daily write started")
        end
    catch e
        e
    end
    yield()
    @test !istaskdone(runner)
    @test !istaskdone(writer)
    put!(release,nothing)
    failure = fetch(runner)
    @test failure isa ErrorException
    @test istaskdone(writer)
    @test isready(finished)
    @test output.pending_write === nothing
    @test close(output) === nothing
end


@testset "Daily output owns frames and closes other resources on write failure" begin
    mesh = LatLonMesh(;FT=Float64,Nx=2,Ny=2)
    grid = AtmosGrid(mesh,HybridSigmaPressure([0.0,1.0],[0.0,1.0]),CPU();FT=Float64)
    air = fill(2.0,2,2,1)
    frame = O.SnapshotFrame(0.0,air,Dict(:co2=>air.*400e-6),:dry)
    mktempdir() do dir
        output = R.RunSnapshotOutput()
        spec = O.runtime_output_spec(Dict("path"=>joinpath(dir,"daily.nc"),"split"=>"daily"),Float64)
        frames = O.AbstractSnapshotFrame[frame]
        R._start_daily_output!(output,spec,joinpath(dir,"one.nc"),frames,grid,:dry)
        @test isempty(frames)
        push!(frames,O.SnapshotFrame(1.0,air,Dict(:co2=>air.*500e-6),:dry))
        R._start_daily_output!(output,spec,joinpath(dir,"two.nc"),frames,grid,:dry)
        close(output)
        @test isempty(frames)
        @test output.pending_write === nothing
        for (name,time,q) in (("one.nc",0.0,400e-6),("two.nc",1.0,500e-6))
            NCDataset(joinpath(dir,name)) do ds
                @test ds["time"][:] == [time]
                @test all(ds["co2"][:,:,:,:] .≈ q)
            end
        end
    end
end
