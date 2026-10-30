# Runner callback/schedule smoke, including a NaN introduced during transport.
using Test, AtmosTransport
using AtmosTransport.State: get_tracer
include(joinpath(@__DIR__, "..", "fixtures", "window_prefetch.jl"))

@testset "a driven simulation runs its callbacks on their schedules" begin
    model, driver = WindowPrefetchFixtures.prefetch_fixture()
    for w in driver.windows, a in (w.fluxes.am, w.fluxes.bm, w.fluxes.cm)
        fill!(a, 0)
    end
    window_ends = Ref(0)
    steps = Ref(0)
    sim = DrivenSimulation(model, driver;
                           callbacks = (ends = Callback(sim -> (window_ends[] += 1), WindowInterval(1)),
                                        every = sim -> (steps[] += 1)))
    run!(sim)
    @test window_ends[] == 2                     # two windows
    @test steps[] == 4                           # two steps each

    # A NaN that appears mid-window stops the run at the window end.
    model, driver = WindowPrefetchFixtures.prefetch_fixture()
    poison = Callback(sim -> (get_tracer(sim.model.state, :co2)[1] = NaN), IterationInterval(3))
    sim = DrivenSimulation(model, driver;
                           callbacks = (poison = poison,
                                        nan_checker = Callback(NaNChecker(), WindowInterval(1))))
    err = try run!(sim); nothing catch e; e end
    @test err isa ErrorException

    # One callback shared by the simulations of two input files (as the runner
    # does) counts windows over both: interval 3 actuates once, at window 3.
    actuated = Ref(0)
    shared = (every3 = Callback(sim -> (actuated[] += 1), WindowInterval(3)),)
    for _ in 1:2
        model, driver = WindowPrefetchFixtures.prefetch_fixture()
        for w in driver.windows, a in (w.fluxes.am, w.fluxes.bm, w.fluxes.cm)
            fill!(a, 0)
        end
        run!(DrivenSimulation(model, driver; callbacks = shared))
    end
    @test actuated[] == 1
end
