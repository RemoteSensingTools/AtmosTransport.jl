using Test, AtmosTransport
const R = AtmosTransport.Models.DrivenRunner
const M = AtmosTransport.Models
const MD = AtmosTransport.MetDrivers

function input_lifetime_fixture(path)
    FT = Float64
    mesh = LatLonMesh(; Nx=4, Ny=3, FT)
    grid = AtmosGrid(mesh, HybridSigmaPressure(FT[0,0,0],FT[0,0.5,1]), CPU(); FT)
    window = (; m=ones(FT,4,3,2), am=zeros(FT,5,3,2), bm=zeros(FT,4,4,2),
                cm=zeros(FT,4,3,3), ps=fill(FT(100000),4,3))
    write_transport_binary(path, grid, [window,window]; FT,
        dt_met_seconds=3600.0, half_dt_seconds=900.0, steps_per_window=2,
        mass_basis=:dry, source_flux_sampling=:window_start_endpoint,
        flux_sampling=:window_constant,
        extra_header=Dict("poisson_balance_target_scale"=>0.25,
            "poisson_balance_target_semantics"=>"forward_window_mass_difference / (2 * steps_per_window)"))
    return grid
end

function input_lifetime_sim(path, grid)
    driver = TransportBinaryDriver(path; FT=Float64, arch=CPU())
    state = CellState(DryBasis, ones(4,3,2); co2=fill(400e-6,4,3,2))
    fluxes = M.allocate_face_fluxes(grid.horizontal,2; FT=Float64,basis=DryBasis)
    model = TransportModel(state,fluxes,grid,UpwindScheme())
    return driver, DrivenSimulation(model,driver)
end

@testset "Runner drains prefetch before closing input" begin
    mktempdir() do dir
        path = joinpath(dir,"transport.bin")
        grid = input_lifetime_fixture(path)
        driver, sim = input_lifetime_sim(path,grid)
        input = R.RunInputResources(driver,sim)
        started, release = Channel{Nothing}(1), Channel{Nothing}(1)
        sim.prefetch_window_index = 2
        prefetch = sim.prefetch_task = Threads.@spawn begin
            put!(started,nothing)
            take!(release)
            @test isopen(driver.reader.io)
            # Exercise a real read while the runner is already exiting.
            MD.load_transport_window(driver,2)
        end
        take!(started)
        runner = @async try
            R._with_run_resource(input) do
                error("stepping failed while prefetch was reading")
            end
        catch err
            err
        end
        yield()
        @test !istaskdone(runner)
        @test isopen(driver.reader.io)
        put!(release,nothing)
        failure = fetch(runner)
        @test failure isa ErrorException
        @test occursin("stepping failed",sprint(showerror,failure))
        @test istaskdone(prefetch)
        @test !isopen(driver.reader.io)
        @test sim.prefetch_window_index == 0
        @test !istaskstarted(sim.prefetch_task)
        @test input.driver === input.simulation === nothing
        @test close(input) === nothing

        driver, sim = input_lifetime_sim(path,grid)
        input = R.RunInputResources(driver,sim)
        sim.prefetch_window_index = 2
        sim.prefetch_task = Threads.@spawn error("prefetch read failed")
        @test_throws TaskFailedException close(input)
        @test !isopen(driver.reader.io)
        @test sim.prefetch_window_index == 0
        @test close(input) === nothing

        # No scheduled prefetch: closing must not wait on the placeholder Task.
        driver, sim = input_lifetime_sim(path,grid)
        input = R.RunInputResources(driver,sim)
        @test R._with_run_resource(() -> :done,input) === :done
        @test !isopen(driver.reader.io)
    end
end

@testset "Runner closes a first driver when setup rejects the run" begin
    mktempdir() do dir
        path = joinpath(dir,"transport.bin")
        input_lifetime_fixture(path)
        cfg = Dict{String,Any}("input"=>Dict("binary_paths"=>[path]),
            "advection"=>Dict("scheme"=>"upwind"),
            "convection"=>Dict("kind"=>"tm5"),
            "tracers"=>Dict("co2"=>Dict("init"=>Dict("kind"=>"uniform","background"=>400e-6))))
        input = R.RunInputResources()
        output = R.RunSnapshotOutput()
        stager = M.InputStager([path],Dict("enabled"=>false))
        observed_driver = Ref{Any}(nothing)
        @test_throws ArgumentError R._with_run_resource(input) do
            try
                R._run_driven_simulation_structured([path],cfg,stager,CPU(),output,input)
            finally
                observed_driver[] = input.driver
            end
        end
        @test observed_driver[] isa TransportBinaryDriver
        @test !isopen(observed_driver[].reader.io)
        @test input.driver === nothing
        close(output)
    end
end

# Driver construction opens its own reader before validating physics/geometry.
# On Linux, inspect the actual file descriptors without relying on GC finalizers.
function input_lifetime_open_handles(path)
    count(readdir("/proc/self/fd"; join=true)) do descriptor
        try
            readlink(descriptor) == path
        catch
            false
        end
    end
end

@testset "Rejected driver construction closes its owned reader" begin
    if Sys.islinux()
        mktempdir() do dir
            path = joinpath(dir,"transport.bin")
            input_lifetime_fixture(path)
            before = input_lifetime_open_handles(path)
            @test_throws ArgumentError TransportBinaryDriver(path; max_rel_cm=-1.0)
            @test input_lifetime_open_handles(path) == before

            cs_path = joinpath(dir,"cs.bin")
            vc = HybridSigmaPressure([0.0,0.0,0.0],[0.0,0.5,1.0])
            writer = MD.open_streaming_cs_transport_binary(cs_path,2,6,2,1,vc; planet_radius = AtmosTransport.Parameters.EARTH_RADIUS,
                FT=Float64,dt_met_seconds=3600.0,steps_per_window=2,mass_basis=:dry)
            window = (;m=ntuple(_->ones(2,2,2),6),am=ntuple(_->zeros(3,2,2),6),
                       bm=ntuple(_->zeros(2,3,2),6),cm=ntuple(_->zeros(2,2,3),6),
                       ps=ntuple(_->fill(100000.0,2,2),6))
            MD.write_streaming_cs_window!(writer,window,2,6)
            MD.close_streaming_transport_binary!(writer)
            before = input_lifetime_open_handles(cs_path)
            @test_throws ArgumentError TransportBinaryDriver(cs_path; Hp=-1)
            @test input_lifetime_open_handles(cs_path) == before
        end
    end
end

@testset "Multi-file partial ranges reject skipped forcing" begin
    mktempdir() do dir
        path = joinpath(dir,"transport.bin")
        input_lifetime_fixture(path)
        cfg = Dict{String,Any}("input"=>Dict("binary_paths"=>[path,path]),
            "tracers"=>Dict("co2"=>Dict("init"=>Dict("kind"=>"uniform","background"=>400e-6))))
        for bounds in (Dict("start_window"=>2),Dict("stop_window"=>1))
            failure = try
                M.run_driven_simulation(merge(cfg,Dict("run"=>bounds)))
            catch err
                err
            end
            @test failure isa ArgumentError
            @test occursin("skip forcing",sprint(showerror,failure))
        end
        # Exercise the public runner's supported single-file partial path.
        partial_cfg = merge(cfg,Dict("input"=>Dict("binary_paths"=>[path]),
                                     "run"=>Dict("start_window"=>2,"stop_window"=>2)))
        model = M.run_driven_simulation(partial_cfg)
        @test all(model.state.tracers_raw .≈ 400e-6)
        @test all(model.state.air_mass .== 1)
    end
end

@testset "Runner disables its instrumentation after failure" begin
    ST = AtmosTransport.SectionTimer
    mktempdir() do dir
        path = joinpath(dir,"transport.bin")
        input_lifetime_fixture(path)
        cfg = Dict{String,Any}("input"=>Dict("binary_paths"=>[path]),
            "tracers"=>Dict("co2"=>Dict("init"=>Dict("kind"=>"uniform","background"=>400e-6))))
        broken = joinpath(dir,"broken.bin")
        write(broken,"invalid transport header")
        failures = (merge(cfg,Dict("input"=>Dict("binary_paths"=>[broken]))),
                    merge(cfg,Dict("convection"=>Dict("kind"=>"tm5"))))
        try
            for (timing,nvtx) in (("1","0"),("0","1"))
                withenv("ATMOSTR_TIMERS"=>timing,"ATMOSTR_NVTX"=>nvtx,
                        "ATMOSTR_ALLOC_TIMERS"=>"1") do
                    for bad_cfg in failures
                        failure = try
                            M.run_driven_simulation(bad_cfg)
                        catch err
                            err
                        end
                        @test failure isa Exception
                        @test !(failure isa CompositeException)
                        @test !ST.is_enabled()
                        @test !ST._ALLOC_ENABLED[]
                        @test !ST._NVTX_ENABLED[]
                        @test ST._timer_snapshot().wall_total > 0
                    end
                end
            end
            before = ST._timer_snapshot()
            withenv("ATMOSTR_TIMERS"=>"0","ATMOSTR_NVTX"=>"0") do
                model = M.run_driven_simulation(cfg)
                @test all(model.state.tracers_raw .≈ 400e-6)
            end
            @test ST._timer_snapshot().timings == before.timings
            withenv("ATMOSTR_TIMERS"=>"1","ATMOSTR_NVTX"=>"0") do
                M.run_driven_simulation(cfg)
            end
            @test !ST.is_enabled()
            @test !isempty(ST._timer_snapshot().timings)
        finally
            ST.disable!()
        end
    end
end
