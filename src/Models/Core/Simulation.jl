"""
    Simulation

Low-level fixed-step harness for custom in-memory experiments.

Most users should call `run_driven_simulation` with a run config. Use
`Simulation` only when you already constructed a `TransportModel` and want to
drive it with a custom time loop.
"""
mutable struct Simulation{ModelT, FT, CB}
    model      :: ModelT
    Δt         :: FT
    stop_time  :: FT
    time       :: FT
    iteration  :: Int
    callbacks  :: CB
end

function Simulation(model::TransportModel;
                    Δt::Real,
                    stop_time::Real,
                    callbacks = NamedTuple())
    FT = promote_type(typeof(float(Δt)), typeof(float(stop_time)))
    isfinite(Δt) && Δt > 0 ||
        throw(ArgumentError("Simulation Δt must be finite and positive; got $(Δt)"))
    isfinite(stop_time) && stop_time >= 0 ||
        throw(ArgumentError("Simulation stop_time must be finite and nonnegative; got $(stop_time)"))
    ratio = FT(stop_time) / FT(Δt)
    isapprox(ratio, round(ratio); rtol=zero(FT),
             atol=FT(8) * eps(FT) * max(abs(ratio), one(FT))) ||
        throw(ArgumentError(
            "Simulation stop_time must be an integer multiple of Δt because face fluxes are stored as mass per substep; " *
            "got stop_time=$(stop_time), Δt=$(Δt)"))
    return Simulation{typeof(model), FT, typeof(callbacks)}(
        model, FT(Δt), FT(stop_time), zero(FT), 0, callbacks)
end

function step!(sim::Simulation)
    sim.iteration < round(Int, sim.stop_time / sim.Δt) ||
        throw(ArgumentError("Simulation has already reached stop_time=$(sim.stop_time)"))
    step!(sim.model, sim.Δt)
    sim.time += sim.Δt
    sim.iteration += 1
    if sim.iteration == round(Int, sim.stop_time / sim.Δt)
        sim.time = sim.stop_time
    end
    for callback in values(sim.callbacks)
        callback(sim)
    end
    return nothing
end

"""
    run!(simulation)

Advance a Simulation or DrivenSimulation until its configured stop condition.
Returns the mutated simulation.
"""
function run!(sim::Simulation)
    target_iteration = round(Int, sim.stop_time / sim.Δt)
    while sim.iteration < target_iteration
        step!(sim)
    end
    sim.time = sim.stop_time
    return sim
end

export Simulation, run!
