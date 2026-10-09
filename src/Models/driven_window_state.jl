# DrivenSimulation: state halos, air-mass reset at window ends, window payload copies and prefetch.
# Split from DrivenSimulation.jl (refactor phase 4); included by Models.jl in this order.

@inline _storage_eltype(reference) = eltype(reference)
@inline _storage_eltype(reference::NTuple{6}) = eltype(reference[1])

@inline _empty_prefetch_task() = Task(() -> nothing)

_refresh_state_halos!(state, _mesh) = state

function _refresh_state_halos!(state::CubedSphereState, mesh::CubedSphereMesh)
    fill_panel_halos!(state.air_mass, mesh; dir = 1)
    fill_panel_halos!(state.tracers_raw, mesh; dir = 1)
    return state
end

function _reset_air_mass_preserve_vmr!(state::CellState, new_air_mass, _mesh)
    old_air_mass = state.air_mass
    FT = eltype(old_air_mass)
    floor_m = eps(FT)
    for (_name, rm) in eachtracer(state)
        @. rm = ifelse(old_air_mass > floor_m,
                       rm / old_air_mass * new_air_mass,
                       zero(FT))
    end
    copyto!(old_air_mass, new_air_mass)
    return state
end

function _reset_air_mass_preserve_vmr!(state::CubedSphereState,
                                       new_air_mass::NTuple{6},
                                       mesh::CubedSphereMesh)
    fill_panel_halos!(new_air_mass, mesh; dir = 1)
    FT = eltype(state.air_mass[1])
    floor_m = eps(FT)
    for p in 1:6
        old_p = state.air_mass[p]
        new_p = new_air_mass[p]
        raw_p = state.tracers_raw[p]
        for idx in 1:length(tracer_names(state))
            rm = selectdim(raw_p, ndims(raw_p), idx)
            @. rm = ifelse(old_p > floor_m,
                           rm / old_p * new_p,
                           zero(FT))
        end
        copyto!(old_p, new_p)
    end
    return _refresh_state_halos!(state, mesh)
end

function _reset_air_mass_preserve_tracer_mass!(state::CellState, new_air_mass, _mesh)
    copyto!(state.air_mass, new_air_mass)
    return state
end

function _reset_air_mass_preserve_tracer_mass!(state::CubedSphereState,
                                               new_air_mass::NTuple{6},
                                               mesh::CubedSphereMesh)
    fill_panel_halos!(new_air_mass, mesh; dir = 1)
    for p in 1:6
        copyto!(state.air_mass[p], new_air_mass[p])
    end
    return _refresh_state_halos!(state, mesh)
end

function _normalize_air_mass_reset_mode(air_mass_reset_mode)
    mode = air_mass_reset_mode === nothing ? :none : Symbol(air_mass_reset_mode)
    mode in (:none, :preserve_vmr, :preserve_tracer_mass) ||
        throw(ArgumentError("air_mass_reset_mode must be one of :none, " *
                            ":preserve_vmr, or :preserve_tracer_mass; got $(repr(mode))"))
    return mode
end

function _reset_air_mass!(state, new_air_mass, mesh, mode::Symbol)
    mode === :none && return state
    mode === :preserve_vmr &&
        return _reset_air_mass_preserve_vmr!(state, new_air_mass, mesh)
    mode === :preserve_tracer_mass &&
        return _reset_air_mass_preserve_tracer_mass!(state, new_air_mass, mesh)
    throw(ArgumentError("unknown air_mass_reset_mode $(repr(mode))"))
end

@inline function _allocate_qv_buffer(window)
    has_humidity_endpoints(window) || return nothing
    return _allocate_storage_like(window.qv_start)
end

@inline function _window_backend_adapter(reference_array)
    return array_adapter_for(reference_array)
end

@inline _window_backend_adapter(reference_array::NTuple{6}) = _window_backend_adapter(reference_array[1])

@inline function _adapt_window_to_model_backend(window, model_air_mass)
    adaptor = _window_backend_adapter(model_air_mass)
    return adaptor === Array ? window : Base.invokelatest(Adapt.adapt, adaptor, window)
end

@inline function _copy_optional_storage!(dest, src, field::Symbol)
    if dest === nothing || src === nothing
        dest === src ||
            throw(ArgumentError("transport window capability for `$(field)` changed between windows"))
        return dest
    end
    return _copy_storage!(dest, src)
end

@inline function _copy_optional_convection!(dest, src)
    if dest === nothing || src === nothing
        dest === src ||
            throw(ArgumentError("transport window convection capability changed between windows"))
        return dest
    end
    return copy_convection_forcing!(dest, src)
end

@inline function _copy_optional_surface!(dest, src)
    if dest === nothing || src === nothing
        dest === src ||
            throw(ArgumentError("transport window surface-forcing capability changed between windows"))
        return dest
    end
    return _copy_surface_forcing!(dest, src)
end

@inline function _copy_optional_vdiff!(dest, src)
    if dest === nothing || src === nothing
        dest === src ||
            throw(ArgumentError("transport window VDIFF capability changed between windows"))
        return dest
    end
    propertynames(dest) == propertynames(src) ||
        throw(ArgumentError("transport window VDIFF fields changed between windows"))
    for name in propertynames(dest)
        _copy_storage!(getproperty(dest, name), getproperty(src, name))
    end
    return dest
end

@inline _copy_surface_forcing!(dest, src) =
    throw(ArgumentError("unsupported surface-forcing refresh from $(typeof(src)) to $(typeof(dest))"))

@inline function _copy_surface_forcing!(dest::PBLSurfaceForcing, src::PBLSurfaceForcing)
    _copy_storage!(dest.pblh, src.pblh)
    _copy_storage!(dest.ustar, src.ustar)
    _copy_storage!(dest.hflux, src.hflux)
    _copy_storage!(dest.t2m, src.t2m)
    _copy_optional_storage!(dest.eflux, src.eflux, :eflux)
    return dest
end

@inline function _copy_flux_deltas!(dest::StructuredFluxDeltas, src::StructuredFluxDeltas)
    _copy_storage!(dest.dam, src.dam)
    _copy_storage!(dest.dbm, src.dbm)
    _copy_storage!(dest.dcm, src.dcm)
    _copy_storage!(dest.dm, src.dm)
    return dest
end

@inline function _copy_flux_deltas!(dest::FaceIndexedFluxDeltas, src::FaceIndexedFluxDeltas)
    _copy_storage!(dest.dhflux, src.dhflux)
    _copy_storage!(dest.dcm, src.dcm)
    _copy_storage!(dest.dm, src.dm)
    return dest
end

@inline function _copy_flux_deltas!(dest::CubedSphereFluxDeltas, src::CubedSphereFluxDeltas)
    _copy_storage!(dest.dm, src.dm)
    return dest
end

@inline function _copy_optional_deltas!(dest, src)
    if dest === nothing || src === nothing
        dest === src ||
            throw(ArgumentError("transport window flux-delta capability changed between windows"))
        return dest
    end
    return _copy_flux_deltas!(dest, src)
end

function _copy_common_window_payload!(dest, src)
    _copy_storage!(dest.air_mass, src.air_mass)
    _copy_storage!(dest.surface_pressure, src.surface_pressure)
    copy_fluxes!(dest.fluxes, src.fluxes)
    _copy_optional_storage!(dest.qv_start, src.qv_start, :qv_start)
    _copy_optional_storage!(dest.qv_end, src.qv_end, :qv_end)
    _copy_optional_deltas!(dest.deltas, src.deltas)
    _copy_optional_convection!(dest.convection, src.convection)
    return dest
end

function _copy_window_payload!(dest::TransportWindow{B},
                               src::TransportWindow{B}) where {B <: AbstractMassBasis}
    _copy_common_window_payload!(dest, src)
    _copy_optional_surface!(dest.surface, src.surface)
    _copy_optional_vdiff!(dest.vdiff, src.vdiff)
    _copy_optional_storage!(dest.dkg, src.dkg, :dkg)
    return dest
end

function _load_window_into_existing_backend!(existing_window,
                                             driver::AbstractMetDriver,
                                             win::Int,
                                             model_air_mass)
    loaded = SectionTimer.time_section(:window_load_host) do
        _load_window(driver, win)
    end
    adaptor = _window_backend_adapter(model_air_mass)
    if adaptor === Array
        return loaded
    end
    SectionTimer.time_section(:window_backend_copy) do
        _copy_window_payload!(existing_window, loaded)
    end
    return existing_window
end

@inline _prefetch_enabled(model_air_mass) =
    get(ENV, "ATMOSTR_DISABLE_PREFETCH", "0") != "1" &&
    _window_backend_adapter(model_air_mass) !== Array && Threads.nthreads() > 1

function _start_window_prefetch!(sim::DrivenSimulation, target_window::Int)
    if target_window > sim.stop_window || !_prefetch_enabled(sim.model.state.air_mass)
        sim.prefetch_window_index = 0
        sim.prefetch_task = _empty_prefetch_task()
        return nothing
    end
    target_slot = sim.prefetch_window
    driver = sim.driver
    model_air_mass = sim.model.state.air_mass
    sim.prefetch_window_index = target_window
    sim.prefetch_task = Threads.@spawn SectionTimer.time_section(:prefetch_task_total) do
        _load_window_into_existing_backend!(
            target_slot, driver, target_window, model_air_mass)
    end
    return nothing
end

# The runner calls this on normal and exceptional file exits. An unscheduled
# placeholder Task must not be waited on; a nonzero index owns a real prefetch.
function _finish_window_prefetch!(sim::DrivenSimulation)
    sim.prefetch_window_index == 0 && return nothing
    try
        wait(sim.prefetch_task)
    finally
        sim.prefetch_task = _empty_prefetch_task()
        sim.prefetch_window_index = 0
    end
    return nothing
end

function _take_prefetched_window!(sim::DrivenSimulation, next_window::Int)
    if _prefetch_enabled(sim.model.state.air_mass) &&
       sim.prefetch_window_index == next_window
        task = sim.prefetch_task
        fetched = try
            SectionTimer.time_section(:prefetch_fetch_wait) do
                fetch(task)
            end
        finally
            # A completed task's failure has now been observed. Keep ownership
            # if the wait was interrupted while the task was still reading.
            if istaskdone(task)
                sim.prefetch_task = _empty_prefetch_task()
                sim.prefetch_window_index = 0
            end
        end
        fetched === sim.prefetch_window ||
            throw(ArgumentError("prefetched transport window identity changed unexpectedly"))
        old_current = sim.window
        sim.window = sim.prefetch_window
        sim.prefetch_window = old_current
        return nothing
    end
    sim.window = SectionTimer.time_section(:window_sync_load_total) do
        _load_window_into_existing_backend!(sim.window, sim.driver,
                                           next_window,
                                           sim.model.state.air_mass)
    end
    return nothing
end

function _reclaim_backend_pool_after_startup!(model_air_mass)
    _window_backend_adapter(model_air_mass) === Array && return nothing
    # Startup can allocate large transient CuArrays while adapting initial
    # conditions and first-window forcing. They are dead before the run loop,
    # but CUDA.jl's pool keeps them reserved unless we explicitly trim it.
    GC.gc(false)
    reclaim_backend_pool!(model_air_mass)
    return nothing
end

@inline function _adapt_sources_to_model_backend(surface_sources, model_air_mass)
    adaptor = _window_backend_adapter(model_air_mass)
    return adaptor === Array ? surface_sources :
           map(source -> Base.invokelatest(Adapt.adapt, adaptor, source), surface_sources)
end
