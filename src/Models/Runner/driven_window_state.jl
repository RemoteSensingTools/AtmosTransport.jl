# DrivenSimulation: state halos, air-mass reset at window ends, window payload copies and prefetch.
# Split from DrivenSimulation.jl (refactor phase 4); included by Runner.jl in this order.

@inline _storage_eltype(reference) = eltype(reference)
@inline _storage_eltype(reference::NTuple{6}) = eltype(reference[1])

@inline _empty_prefetch_task() = Task(() -> nothing)

# State halos and the air-mass reset (`_refresh_state_halos!`,
# `_reset_air_mass!`, `_normalize_air_mass_reset_mode`) live with the step plan
# (`Operators.StepPlans`), whose `AirMassResetStage` applies them.

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

@inline function _copy_optional!(copier, dest, src, capability)
    if dest === nothing || src === nothing
        dest === src || throw(ArgumentError(
            capability isa Symbol ?
            "transport window capability for `$(capability)` changed between windows" :
            "transport window $(capability) capability changed between windows"))
        return dest
    end
    return copier(dest, src)
end

@inline _copy_optional_storage!(dest, src, field::Symbol) =
    _copy_optional!(_copy_storage!, dest, src, field)

@inline _copy_optional_convection!(dest, src) =
    _copy_optional!(copy_convection_forcing!, dest, src, "convection")

@inline _copy_optional_surface!(dest, src) =
    _copy_optional!(_copy_surface_forcing!, dest, src, "surface-forcing")

@inline _copy_optional_vdiff!(dest, src) =
    _copy_optional!(_copy_vdiff!, dest, src, "VDIFF")

@inline _copy_vdiff!(dest, src) = _copy_window_fields!(dest, src, "VDIFF")

@inline function _copy_window_fields!(dest, src, label)
    propertynames(dest) == propertynames(src) ||
        throw(ArgumentError("transport window $label fields changed between windows"))
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

@inline _copy_optional_deltas!(dest, src) =
    _copy_optional!(_copy_flux_deltas!, dest, src, "flux-delta")

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
    _copy_optional!((d,s)->_copy_window_fields!(d,s,"tm5_pbl"),dest.tm5_pbl,src.tm5_pbl,"tm5_pbl")
    return dest
end

# Staging of the window loads of a device-resident run: the driver's raw window
# buffers when it has them (`raw_window_staging`: one read into page-locked
# host memory and section uploads, or, with unified memory, into a device
# buffer; unpacking on the device), else the first host window
# when the driver can refill it in place, page-locked where the backend
# supports it. CPU runs use each loaded window directly (it becomes the
# simulation's window), so they keep loading fresh ones.
function _host_staging_window(loaded_window, driver, model_air_mass)
    _window_backend_adapter(model_air_mass) === Array && return nothing
    _window_backend_adapter(loaded_window.air_mass) === Array || return nothing
    raw = raw_window_staging(driver, model_air_mass)
    raw === nothing || return raw
    hasmethod(load_transport_window!, Tuple{typeof(loaded_window), typeof(driver), Int}) ||
        return nothing
    return Architectures.pin_host_memory!(loaded_window, model_air_mass)
end

# The window storage of a finished simulation `previous` that a new simulation
# of the same model (state and convection forcing storage) on `driver` can
# refill (see `DrivenSimulation`'s `reuse_buffers_of`), or `nothing`. A device
# run keeps staging only with raw window buffers or an in-place loader;
# `separate_prefetch` says whether the new simulation needs a second device
# window.
_reusable_window_buffers(::Nothing, _model, _driver, _separate_prefetch) = nothing

function _reusable_window_buffers(previous::DrivenSimulation, model, driver, separate_prefetch::Bool)
    _finish_window_prefetch!(previous)          # its buffers must be idle
    previous.host_staging === nothing && return nothing
    previous.model.state.air_mass === model.state.air_mass &&
        previous.model.convection_forcing === model.convection_forcing || return nothing
    separate_prefetch && previous.prefetch_window === previous.window && return nothing
    same_window_layout(previous.driver, driver) || return nothing
    return (; previous.window, previous.prefetch_window, previous.host_staging,
              previous.expected_air_mass, previous.qv_buffer)
end

function _load_window_into_existing_backend!(existing_window,
                                             driver::AbstractMetDriver,
                                             win::Int,
                                             model_air_mass,
                                             host_staging = nothing)
    _window_backend_adapter(model_air_mass) === Array &&
        return SectionTimer.time_section(() -> _load_window(driver, win), :window_load_host)
    _staged_window_load!(existing_window, driver, win, host_staging)
    # The uploads (asynchronous from page-locked memory) and unpack kernels run
    # on this task's queue: finish them before the staging buffers are refilled
    # or the device window is handed to the stepping task.
    SectionTimer.time_section(:window_backend_copy) do
        Architectures.synchronize_storage!(model_air_mass)
    end
    return existing_window
end

# One device window load by staging kind: none (a fresh host window), a host
# window refilled in place, or raw window buffers.
function _staged_window_load!(existing_window, driver, win, ::Nothing)
    loaded = SectionTimer.time_section(() -> _load_window(driver, win), :window_load_host)
    SectionTimer.time_section(() -> _copy_window_payload!(existing_window, loaded),
                              :window_backend_copy)
    return existing_window
end

function _staged_window_load!(existing_window, driver, win, staging::TransportWindow)
    loaded = SectionTimer.time_section(() -> load_transport_window!(staging, driver, win),
                                       :window_load_host)
    SectionTimer.time_section(() -> _copy_window_payload!(existing_window, loaded),
                              :window_backend_copy)
    return existing_window
end

function _staged_window_load!(existing_window, driver, win, staging::AbstractRawWindowStaging)
    SectionTimer.time_section(() -> read_window_payload!(staging, driver, win), :window_load_host)
    SectionTimer.time_section(() -> unpack_window_payload!(existing_window, driver, staging),
                              :window_backend_copy)
    return existing_window
end

# Prefetch needs the request, a device backend and a second Julia thread.
@inline _prefetch_enabled(requested::Bool, model_air_mass) =
    requested && _window_backend_adapter(model_air_mass) !== Array && Threads.nthreads() > 1

function _start_window_prefetch!(sim::DrivenSimulation, target_window::Int)
    if target_window > sim.stop_window || !sim.prefetch_enabled
        sim.prefetch_window_index = 0
        sim.prefetch_task = _empty_prefetch_task()
        return nothing
    end
    target_slot = sim.prefetch_window
    driver = sim.driver
    model_air_mass = sim.model.state.air_mass
    host_staging = sim.host_staging
    # The prefetch overwrites the window the stepping task used before the last
    # swap: its queued reads of that window must be finished first.
    Architectures.synchronize_storage!(model_air_mass)
    sim.prefetch_window_index = target_window
    sim.prefetch_task = Threads.@spawn SectionTimer.time_section(:prefetch_task_total) do
        _load_window_into_existing_backend!(
            target_slot, driver, target_window, model_air_mass, host_staging)
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
    # A nonzero index means a prefetch was started.
    if sim.prefetch_window_index == next_window
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
    # A prefetch of another window may still be reading the shared host staging
    # window; it stays available for a later take.
    sim.host_staging === nothing || sim.prefetch_window_index == 0 || wait(sim.prefetch_task)
    sim.window = SectionTimer.time_section(:window_sync_load_total) do
        _load_window_into_existing_backend!(sim.window, sim.driver,
                                           next_window,
                                           sim.model.state.air_mass,
                                           sim.host_staging)
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
