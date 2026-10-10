module WindowPrefetchFixtures

using Test, AtmosTransport, Adapt
const MD = AtmosTransport.MetDrivers
const M = AtmosTransport.Models

struct CountedWindowDriver{G,W} <: MD.AbstractMetDriver
    grid::G
    windows::W
    reads::Vector{Int}
end
MD.driver_grid(d::CountedWindowDriver) = d.grid
MD.total_windows(d::CountedWindowDriver) = length(d.windows)
MD.window_dt(::CountedWindowDriver) = 3600.0
MD.steps_per_window(::CountedWindowDriver) = 2
MD.air_mass_basis(::CountedWindowDriver) = :dry
MD.supports_native_vertical_flux(::CountedWindowDriver) = true
function MD.load_transport_window(d::CountedWindowDriver, win::Int)
    push!(d.reads,win)
    return deepcopy(d.windows[win])
end

# Include humidity, flux deltas and all TM5 forcing arrays in the alias check.
function prefetch_fixture()
    FT = Float32
    mesh = LatLonMesh(;Nx=4,Ny=3,FT)
    grid = AtmosGrid(mesh,HybridSigmaPressure(FT[0,0,0],FT[0,0.5,1]),CPU();FT)
    windows = map(1:2) do win
        m = fill(FT(win),4,3,2)
        fluxes = allocate_face_fluxes(mesh,2;FT,basis=DryBasis)
        foreach(a -> fill!(a,FT(win)),(fluxes.am,fluxes.bm,fluxes.cm))
        deltas = MD.StructuredFluxDeltas(copy(fluxes.am),copy(fluxes.bm),copy(fluxes.cm),copy(m))
        tm5 = (;entu=copy(m),detu=copy(m),entd=copy(m),detd=copy(m))
        forcing = AtmosTransport.Operators.ConvectionForcing(nothing,nothing,tm5)
        MD.TransportWindow(m,fill(FT(100000+win),4,3),fluxes;
            qv_start=fill(FT(0.01win),4,3,2),qv_end=fill(FT(0.02win),4,3,2),
            deltas,convection=forcing)
    end
    state = CellState(DryBasis,copy(windows[1].air_mass);co2=fill(FT(400e-6),4,3,2))
    model = TransportModel(state,deepcopy(windows[1].fluxes),grid,UpwindScheme())
    return model,CountedWindowDriver(grid,windows,Int[])
end

payload_arrays(a::AbstractArray) = [a]
payload_arrays(::Nothing) = []
payload_arrays(x) = reduce(vcat,(payload_arrays(getfield(x,i)) for i in 1:fieldcount(typeof(x)));init=[])

function check_prefetch_startup(adapter; enabled=true, stop_window=2, device_windows=false)
    model,driver = prefetch_fixture()
    model = Adapt.adapt(adapter,model)
    if device_windows
        driver = CountedWindowDriver(driver.grid,Adapt.adapt.(Ref(adapter),driver.windows),Int[])
    end
    sim = DrivenSimulation(model,driver;stop_window,prefetch_windows=enabled)
    # `prefetch_windows = false` always loads synchronously; CPU runs never prefetch.
    @test sim.prefetch_enabled == (enabled && M._window_backend_adapter(model.state.air_mass) !== Array &&
                                   Threads.nthreads() > 1)
    prefetching = sim.prefetch_enabled && stop_window > 1
    try
        prefetching && wait(sim.prefetch_task)
        @test driver.reads == (prefetching ? [1,2] : [1])
        active = payload_arrays(sim.window)
        expected = payload_arrays(driver.windows[1])
        @test !isempty(active)
        @test length(active) == length(expected)
        for (a,b) in zip(active,expected)
            @test Array(a) == Array(b)
        end
        if prefetching
            pending = payload_arrays(sim.prefetch_window)
            for (a,b) in zip(pending,payload_arrays(driver.windows[2]))
                @test Array(a) == Array(b)
            end
            # Mutating every prefetched payload must leave active forcing intact.
            foreach(a -> fill!(a,zero(eltype(a))),pending)
            for (a,b) in zip(active,expected)
                @test Array(a) == Array(b)
            end
            old_current,old_prefetch = sim.window,sim.prefetch_window
            M._take_prefetched_window!(sim,2)
            @test sim.window === old_prefetch
            @test sim.prefetch_window === old_current
            @test sim.prefetch_window_index == 0
        else
            @test sim.prefetch_window === sim.window
        end
    finally
        M._finish_window_prefetch!(sim)
    end
end

export check_prefetch_startup
end
