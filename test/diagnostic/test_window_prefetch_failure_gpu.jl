# Shares the opt-in device guard with test_window_prefetch_gpu.jl.
using Test
if get(ENV,"ATMOSTR_RUN_PREFETCH_GPU_TESTS","0") != "1"
    @info "Skipping opt-in window prefetch failure GPU test"
else
    using CUDA, Adapt
    expected_device = get(ENV,"ATMOSTR_PREFETCH_GPU_NAME","A100")
    isempty(expected_device) && error("ATMOSTR_PREFETCH_GPU_NAME must name the authorized GPU")
    @assert occursin(expected_device,CUDA.name(CUDA.device())) "Wrong device for prefetch GPU test"
    @assert Threads.nthreads() > 1 "Prefetch requires multiple Julia threads"
    CUDA.allowscalar(false)
    include(joinpath(@__DIR__,"..","fixtures","window_prefetch.jl"))
    const M = WindowPrefetchFixtures.M
    const R = M.DrivenRunner
    Base.close(::WindowPrefetchFixtures.CountedWindowDriver) = nothing
    @testset "Failed prefetch is consumed once before resource cleanup" begin
        let
            model,driver = WindowPrefetchFixtures.prefetch_fixture()
            sim = M.DrivenSimulation(Adapt.adapt(CuArray,model),driver;prefetch_windows=true)
            M._finish_window_prefetch!(sim)
            active = sim.window
            sim.prefetch_window_index = 2
            failed_task = sim.prefetch_task = Threads.@spawn error("prefetch input failed")
            try
                input = R.RunInputResources(driver,sim)
                failure = try
                    R._with_run_resource(input) do
                        M._take_prefetched_window!(sim,2)
                    end
                catch err
                    err
                end
                @test failure isa TaskFailedException
                @test failure isa TaskFailedException && failure.task === failed_task
                @test input.driver === input.simulation === nothing
                @test sim.window === active
                @test sim.prefetch_window_index == 0
                @test !istaskstarted(sim.prefetch_task)
                @test M._finish_window_prefetch!(sim) === nothing
                @test istaskfailed(failed_task)
            finally
                # Baseline failures must not leave an unobserved task behind.
                try
                    M._finish_window_prefetch!(sim)
                catch err
                    err isa TaskFailedException || rethrow()
                end
            end
        end
    end
end
