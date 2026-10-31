# Resource ownership for the driven file/window loop. Numerical stepping does
# not own external driver handles; this runner closes what it opens.

function _with_run_resource(f, resource)
    result = try
        f()
    catch run_error
        try
            close(resource)
        catch cleanup_error
            # A cleanup failure must not hide the original run failure.
            throw(CompositeException(Any[run_error, cleanup_error]))
        end
        rethrow()
    end
    close(resource)
    return result
end

mutable struct RunInputResources
    driver::Union{Nothing,AbstractMetDriver}
    simulation::Union{Nothing,DrivenSimulation}
end
RunInputResources() = RunInputResources(nothing, nothing)

function Base.close(input::RunInputResources)
    driver, sim = input.driver, input.simulation
    input.driver = nothing
    input.simulation = nothing
    driver === nothing && return nothing
    try
        _with_run_resource(driver) do
            # A GPU prefetch task can still be reading the driver's mapping.
            # Finish it before closing the reader or releasing mapped pages.
            sim === nothing || _finish_window_prefetch!(sim)
        end
    finally
        driver isa TransportBinaryDriver && release_payload!(driver)
    end
    return nothing
end
