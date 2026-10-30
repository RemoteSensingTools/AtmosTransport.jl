# Convenience adapter; observation IO does not depend on solver result types.
build_departure_set(set::CSObservationSet, result::CS4DVarResult, mesh::CubedSphereMesh,
                    t_start::AbstractString, dt::Real, nsteps::Integer; kwargs...) =
    build_departure_set(set, result.observations, result.simulated, result.stencils, mesh,
                        t_start, dt, nsteps; kwargs...)
