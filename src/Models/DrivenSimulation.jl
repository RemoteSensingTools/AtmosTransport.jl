"""
    DrivenSimulation

Low-level window-driven runtime for custom driver/model wiring.

Most TOML-based runs should go through `run_driven_simulation`, which
builds and validates this object for you. Construct `DrivenSimulation` directly
only when you need a custom met driver, callback loop, or model assembly.

A `DrivenSimulation` keeps transport-window timing and forcing in the driver,
while the model retains ownership of prognostic tracer and air-mass state.
The runtime interpolates forcing within each met window and advances the model
with the same `step!(model, Δt)` entry point used by the fixed-flux smoke
harness.

For experienced users, the constructor is meant to be assembled from the same
pieces used by `run_driven_simulation`: a met driver, a basis-compatible state,
empty flux storage owned by the model, and a runtime physics recipe. A minimal
LL/RG manual setup looks like this:

```julia
using TOML, AtmosTransport
using AtmosTransport.MetDrivers: air_mass_basis, driver_grid, flux_kind

# First run examples/generate_synthetic_quickstart.jl from the terminal.
cfg = TOML.parsefile("config/examples/minimal_template.toml")
paths = expand_binary_paths(cfg["input"])
FT = Float64

driver = TransportBinaryDriver(first(paths); FT = FT, arch = CPU())
recipe = build_runtime_physics_recipe(cfg, driver, FT)
validate_runtime_physics_recipe(recipe, driver)

grid = driver_grid(driver)
window1 = load_transport_window(driver, 1)
Basis = air_mass_basis(driver) === :dry ? DryBasis : MoistBasis
air = copy(window1.air_mass)

vmr = build_initial_mixing_ratio(
    air, grid, Dict("kind" => "uniform", "background" => 400e-6);
    surface_pressure = window1.surface_pressure,
)
co2 = pack_initial_tracer_mass(grid, air, vmr; mass_basis = Basis())

state = CellState(Basis, air; CO2 = co2)
fluxes = allocate_face_fluxes(grid.horizontal, nlevels(grid);
                              FT = FT, basis = Basis)
model = TransportModel(state, fluxes, grid, recipe.advection;
                       diffusion = recipe.diffusion,
                       convection = recipe.convection)

sim = DrivenSimulation(model, driver;
                       stop_window = min(total_windows(driver), 24),
                       chemistry = recipe.chemistry)
run_window!(sim)
```

The full runner adds the practical edges around this skeleton: resolving many
daily binaries, GPU adaptation, surface-flux source construction, snapshot
output, progress reporting, and capability checks against every file.

`SurfaceFluxSource` lives with the surface-flux operator in
`src/Operators/SurfaceFlux/`.
"""
mutable struct DrivenSimulation{ModelT, DriverT, WindowT, AT, QT, FT, CB, PT}
    model                 :: ModelT
    driver                :: DriverT
    window                :: WindowT
    prefetch_window       :: WindowT
    prefetch_task         :: PT
    prefetch_window_index :: Int
    expected_air_mass     :: AT
    qv_buffer             :: QT
    Δt                    :: FT
    window_dt             :: FT
    steps_per_window      :: Int
    steps_per_window_schedule :: Vector{Int}
    time                  :: Float64    # model clock [s]; Float64 for any FT
    start_time            :: Float64    # clock at the start of `start_window`
    iteration             :: Int
    start_window          :: Int
    current_window_index  :: Int
    current_window_start_iteration :: Int
    current_window_end_iteration   :: Int
    stop_window           :: Int
    final_iteration       :: Int
    callbacks                   :: CB
    initialize_air_mass         :: Bool
    use_midpoint_forcing        :: Bool
    interpolate_fluxes_within_window :: Bool
    air_mass_reset_mode         :: Symbol
end

@inline _basis_symbol(::DryBasis) = :dry
@inline _basis_symbol(::MoistBasis) = :moist

_same_values(a, b) = a == b
_same_horizontal_geometry(a::LatLonMesh, b::LatLonMesh) =
    a.Nx == b.Nx && a.Ny == b.Ny && a.radius == b.radius &&
    _same_values(a.λᶠ, b.λᶠ) && _same_values(a.φᶠ, b.φᶠ)
_same_horizontal_geometry(a::ReducedGaussianMesh, b::ReducedGaussianMesh) =
    a.radius == b.radius && a.nlon_per_ring == b.nlon_per_ring &&
    _same_values(a.latitudes, b.latitudes) && _same_values(a.lat_faces, b.lat_faces)
_same_horizontal_geometry(a::CubedSphereMesh, b::CubedSphereMesh) =
    a.Nc == b.Nc && a.Hp == b.Hp && a.radius == b.radius &&
    repr(a.definition) == repr(b.definition) && repr(a.convention) == repr(b.convention)
_same_horizontal_geometry(a, b) =
    typeof(a) === typeof(b) && ncells(a) == ncells(b) && nfaces(a) == nfaces(b)

_same_vertical_geometry(a::HybridSigmaPressure, b::HybridSigmaPressure) =
    a.A == b.A && a.B == b.B
_same_vertical_geometry(a, b) = typeof(a) === typeof(b) && a == b

function _check_grid_compatibility(model_grid::AtmosGrid, driver_grid_ref::AtmosGrid)
    typeof(model_grid.horizontal) === typeof(driver_grid_ref.horizontal) ||
        throw(ArgumentError("model grid $(typeof(model_grid.horizontal)) does not match driver grid $(typeof(driver_grid_ref.horizontal))"))
    nlevels(model_grid) == nlevels(driver_grid_ref) ||
        throw(ArgumentError("model and driver vertical levels do not match"))
    ncells(model_grid.horizontal) == ncells(driver_grid_ref.horizontal) ||
        throw(ArgumentError("model and driver horizontal cell counts do not match"))
    nfaces(model_grid.horizontal) == nfaces(driver_grid_ref.horizontal) ||
        throw(ArgumentError("model and driver horizontal face counts do not match"))
    _same_horizontal_geometry(model_grid.horizontal, driver_grid_ref.horizontal) ||
        throw(ArgumentError("model and driver horizontal geometry differs despite matching topology/counts"))
    typeof(model_grid.vertical) === typeof(driver_grid_ref.vertical) ||
        throw(ArgumentError("model and driver vertical-coordinate types do not match"))
    _same_vertical_geometry(model_grid.vertical, driver_grid_ref.vertical) ||
        throw(ArgumentError("model and driver vertical-coordinate coefficients do not match"))
    model_grid.planet == driver_grid_ref.planet ||
        throw(ArgumentError("model and driver planetary parameters do not match"))
    return nothing
end

function _check_basis_compatibility(model::TransportModel, driver::D) where {D <: AbstractMetDriver}
    basis_sym = air_mass_basis(driver)
    _basis_symbol(mass_basis(model.state)) == basis_sym ||
        throw(ArgumentError("model state basis $(_basis_symbol(mass_basis(model.state))) does not match driver basis $(basis_sym)"))
    _basis_symbol(mass_basis(model.fluxes)) == basis_sym ||
        throw(ArgumentError("model flux basis $(_basis_symbol(mass_basis(model.fluxes))) does not match driver basis $(basis_sym)"))
    return nothing
end

@inline function _substep_fraction(substep::Int, steps_per_window::Int, ::Type{FT}, use_midpoint::Bool) where FT
    if steps_per_window == 1
        return zero(FT)
    elseif use_midpoint
        return (FT(substep) - FT(0.5)) / FT(steps_per_window)
    else
        return (FT(substep) - one(FT)) / FT(steps_per_window)
    end
end

@inline _active_substep(iteration::Int, steps_per_window::Int) = mod(iteration, steps_per_window) + 1

function _driver_step_schedule(driver::AbstractMetDriver)
    schedule = Int.(steps_per_window_schedule(driver))
    length(schedule) == total_windows(driver) ||
        throw(ArgumentError("driver steps_per_window_schedule length $(length(schedule)) " *
                            "does not match total_windows=$(total_windows(driver))"))
    all(>=(1), schedule) ||
        throw(ArgumentError("driver steps_per_window_schedule must contain only positive integers"))
    return schedule
end

@inline _allocate_storage_like(reference) = Base.invokelatest(similar, reference)
@inline _allocate_storage_like(reference::NTuple{6}) =
    ntuple(p -> Base.invokelatest(similar, reference[p]), 6)

@inline _copy_storage!(dest, src) = copyto!(dest, src)
@inline function _copy_storage!(dest::NTuple{6}, src::NTuple{6})
    @inbounds for p in 1:6
        copyto!(dest[p], src[p])
    end
    return dest
end

@inline _scale_storage!(dest, scale) = (dest .*= scale; dest)
@inline function _scale_storage!(dest::NTuple{6}, scale)
    @inbounds for p in 1:6
        dest[p] .*= scale
    end
    return dest
end

@inline function _scale_runtime_fluxes!(fluxes, _scale)
    throw(ArgumentError("flux_kind=:full_window_mass_amount is only implemented " *
                        "for CubedSphereFaceFluxState runtime fluxes; got $(typeof(fluxes))."))
end
@inline function _scale_runtime_fluxes!(fluxes::CubedSphereFaceFluxState, scale)
    _scale_storage!(fluxes.am, scale)
    _scale_storage!(fluxes.bm, scale)
    _scale_storage!(fluxes.cm, scale)
    return fluxes
end

@inline function _apply_runtime_flux_storage_scale!(sim::DrivenSimulation)
    flux_kind(sim.driver) === :full_window_mass_amount || return nothing
    scale = inv(typeof(sim.Δt)(2 * sim.steps_per_window))
    _scale_runtime_fluxes!(sim.model.fluxes, scale)
    return nothing
end

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

# Surface-source shape and compatibility helpers live with the operator.
using ..Operators.SurfaceFlux: _surface_shape,
                                _check_surface_source_compatibility
using ..Operators.Diffusion: NoDiffusion, ImplicitVerticalDiffusion,
                              fill_dz_hydrostatic_constT!,
                              fill_dz_hydrostatic_virtualT!

# ---------------------------------------------------------------------------
# Diffusion layer-thickness refresh.
#
# `apply_vertical_diffusion_vmr!` uses `dz` per cell. The workspace allocator
# intentionally leaves `layer_thickness` undefined, so it must be refreshed
# from the just-loaded window's
# surface pressure + the grid's hybrid-σp coefficients each time the
# simulation advances to a new met window.
#
# `dz` only depends on (ps, ak, bk) (constant-T_ref hydrostatic); within a
# window ps is fixed, so one fill per window is correct and cheap.
# ---------------------------------------------------------------------------
@inline function _refresh_dz_for_window!(sim::DrivenSimulation)
    sim.model.diffusion isa NoDiffusion && return nothing
    diffusion_workspace = sim.model.workspace.diffusion_ws
    diffusion_workspace === nothing && return nothing
    layer_thickness = diffusion_workspace.layer_thickness
    vertical = sim.model.grid.vertical
    ps = sim.window.surface_pressure
    _fill_dz_for_diffusion!(layer_thickness, ps, vertical.A, vertical.B,
                             sim.model.diffusion, sim.window)
    return nothing
end

# When the diffusion operator uses LocalHoltslagBovilleKzField (which
# itself derives column geometry from VDIFF virtual-T), populate
# layer thickness from the SAME virtual-T-per-layer the Kz cache uses. Closes
# the previous inconsistency where the kernel divided by a 260 K-constant
# `dz` while Kz had been computed on layer-varying `dz`.
#
# All other diffusion configurations stay on the constant-T_ref path.
@inline _fill_dz_for_diffusion!(layer_thickness, ps, ak, bk, _diffop, _window) =
    fill_dz_hydrostatic_constT!(layer_thickness, ps, ak, bk)

@inline function _fill_dz_for_diffusion!(
        layer_thickness, ps, ak, bk,
        op::ImplicitVerticalDiffusion{FT, <:LocalHoltslagBovilleKzField},
        window) where FT
    vdiff = window.vdiff
    # Defensive fallback: if VDIFF isn't actually present on the window
    # (shouldn't happen — the diffusion runtime validator rejects this
    # case at config-load time), drop back to the constant-T_ref path and
    # warn loudly so a silently degraded config doesn't go unnoticed.
    if vdiff === nothing || !hasproperty(vdiff, :t) || !hasproperty(vdiff, :qv)
        @warn """
        _fill_dz_for_diffusion!: LocalHoltslagBovilleKzField was selected
        but the active window lacks `vdiff.t` / `vdiff.qv`. Falling back
        to constant-T_ref hydrostatic dz, which is INCONSISTENT with the
        Kz cache's virtual-T column geometry. Check the binary's VDIFF
        payload and the [diffusion] runtime config.
        """
        return fill_dz_hydrostatic_constT!(layer_thickness, ps, ak, bk)
    end
    fill_dz_hydrostatic_virtualT!(layer_thickness, vdiff.t, vdiff.qv, ps, ak, bk)
    return layer_thickness
end

@inline _refresh_pbl_kz_for_window!(_field, _sim::DrivenSimulation) = nothing

function _refresh_pbl_kz_for_window!(field::WindowPBLKzField,
                                     sim::DrivenSimulation)
    mesh = sim.model.grid.horizontal
    refresh_pbl_kz_cache!(field, sim.window.surface, sim.window.air_mass,
                           mesh.cell_areas; halo_width = mesh.Hp)
    return nothing
end

function _refresh_pbl_kz_for_window!(field::LocalHoltslagBovilleKzField,
                                     sim::DrivenSimulation)
    mesh = sim.model.grid.horizontal
    refresh_local_holtslag_boville_kz_cache!(
        field, sim.window.surface, sim.window.vdiff, sim.window.air_mass,
        mesh.cell_areas; halo_width = mesh.Hp)
    return nothing
end

function _refresh_pbl_kz_for_window!(field::GCHPNonlocalPBLField,
                                     sim::DrivenSimulation)
    mesh = sim.model.grid.horizontal
    refresh_gchp_nonlocal_pbl!(field, sim.window.surface, sim.window.vdiff, sim.window.air_mass,
                               mesh.cell_areas, sim.model.grid.vertical;
                               halo_width = mesh.Hp)
    return nothing
end

function _refresh_pbl_kz_for_window!(field::PrecomputedCSDkgField,
                                     sim::DrivenSimulation)
    refresh_precomputed_cs_dkg_cache!(field, sim.window.dkg)
    return nothing
end

@inline function _fill_dz_for_diffusion!(layer_thickness, _ps, _ak, _bk,
        ::ImplicitVerticalDiffusion{FT, <:AbstractCSDkgField}, _window) where FT
    return layer_thickness
end

@inline _refresh_pbl_kz_for_window!(::NoDiffusion, _sim::DrivenSimulation) = nothing

function _refresh_pbl_kz_for_window!(op::ImplicitVerticalDiffusion,
                                     sim::DrivenSimulation)
    _refresh_pbl_kz_for_window!(op.kz_field, sim)
    return nothing
end

"""
    _validate_convection_window!(op, window, driver) -> nothing

Per-operator validation of a loaded transport window. Operator
authors add a method for their concrete type; the fallback method
throws `ArgumentError` naming the operator and pointing at this
function as the place to add a method.

Validation dispatches on the operator type rather than an
`if/elseif op isa …` chain, so adding `TM5Convection` (or any future
operator) only requires adding a method here.
"""
_validate_convection_window!(::NoConvection,
                              ::TransportWindow,
                              ::AbstractMetDriver) = nothing

function _validate_convection_window!(::CMFMCConvection,
                                       window::TransportWindow,
                                       driver::AbstractMetDriver)
    window.convection.cmfmc === nothing &&
        throw(ArgumentError(
            "CMFMCConvection requires `window.convection.cmfmc` to be populated; " *
            "driver $(typeof(driver)) provided convection forcing without CMFMC."))
    return nothing
end

function _validate_convection_window!(::TM5Convection,
                                       window::TransportWindow,
                                       driver::AbstractMetDriver)
    window.convection.tm5_fields === nothing &&
        throw(ArgumentError(
            "TM5Convection requires `window.convection.tm5_fields` " *
            "(NamedTuple with :entu, :detu, :entd, :detd) to be populated; " *
            "driver $(typeof(driver)) provided convection forcing without TM5 fields. " *
            "Preprocess the binary with `scripts/preprocessing/preprocess_transport_binary.jl` " *
            "and `[tm5_convection] enable = true` in the preprocessing config, or fall back to " *
            "`CMFMCConvection()` if you have GEOS-FP CMFMC data instead."))
    return nothing
end

# CMFMCMatrixConvection reads the SAME binary sections as CMFMCConvection
# (cmfmc + dtrain) — only the runtime numerics differ (GCHP two-pass vs
# the conservative TM5 LU on derived rates).
function _validate_convection_window!(::CMFMCMatrixConvection,
                                       window::TransportWindow,
                                       driver::AbstractMetDriver)
    window.convection.cmfmc === nothing &&
        throw(ArgumentError(
            "CMFMCMatrixConvection requires `window.convection.cmfmc` to be populated; " *
            "driver $(typeof(driver)) provided convection forcing without CMFMC."))
    window.convection.dtrain === nothing &&
        throw(ArgumentError(
            "CMFMCMatrixConvection requires `window.convection.dtrain` to be populated " *
            "(the matrix variant uses dtrain as the explicit detrainment rate); " *
            "driver $(typeof(driver)) provided CMFMC but no DTRAIN."))
    return nothing
end

function _validate_convection_window!(op::AbstractConvection,
                                       ::TransportWindow,
                                       ::AbstractMetDriver)
    throw(ArgumentError(
        "DrivenSimulation does not support convection operator $(typeof(op)) yet. " *
        "Add a `_validate_convection_window!(::$(typeof(op)), window, driver)` " *
        "method in `src/Models/DrivenSimulation.jl` that checks its forcing " *
        "requirements."))
end

_validate_convection_runtime(model::TransportModel,
                             driver::AbstractMetDriver,
                             window::TransportWindow) =
    _validate_convection_runtime(model.convection, model, driver, window)

@inline _validate_convection_runtime(::NoConvection, ::TransportModel,
                                     ::AbstractMetDriver,
                                     ::TransportWindow) = nothing

function _validate_convection_runtime(op::AbstractConvection,
                                      model::TransportModel,
                                      driver::AbstractMetDriver,
                                      window::TransportWindow)
    window.convection === nothing &&
        throw(ArgumentError(
            "DrivenSimulation loaded a transport window without convection forcing, " *
            "but model.convection = $(typeof(op)) is active. " *
            "Install a driver/window path that populates `window.convection` " *
            "for this operator."))

    _validate_convection_window!(op, window, driver)
    return nothing
end

function _install_convection_forcing(model::TransportModel,
                                     driver::AbstractMetDriver,
                                     window::TransportWindow)
    _validate_convection_runtime(model, driver, window)
    return _install_convection_forcing(model.convection, model, window)
end

@inline _install_convection_forcing(::NoConvection, model::TransportModel,
                                    ::TransportWindow) = model

function _install_convection_forcing(::AbstractConvection, model::TransportModel,
                                     window::TransportWindow)
    forcing = allocate_convection_forcing_like(window.convection, model.state.air_mass)
    copy_convection_forcing!(forcing, window.convection)
    # A retained workspace must not reuse the previous driver's derived forcing.
    invalidate_cmfmc_cache!(model.workspace.convection_ws)
    return with_convection_forcing(model, forcing)
end

@inline _refresh_convection_forcing!(::NoConvection, ::TransportModel,
                                     ::TransportWindow) = nothing

function _refresh_convection_forcing!(::AbstractConvection, model::TransportModel,
                                      window::TransportWindow)
    copy_convection_forcing!(model.convection_forcing, window.convection)
    return nothing
end

function _refresh_forcing!(sim::DrivenSimulation, substep::Int)
    λ = _substep_fraction(substep, sim.steps_per_window, typeof(sim.Δt), sim.use_midpoint_forcing)
    if sim.interpolate_fluxes_within_window
        interpolate_fluxes!(sim.model.fluxes, sim.window, λ)
    else
        copy_fluxes!(sim.model.fluxes, sim.window.fluxes)
    end
    _apply_runtime_flux_storage_scale!(sim)
    expected_air_mass!(sim.expected_air_mass, sim.window, λ)
    if sim.qv_buffer !== nothing
        interpolate_qv!(sim.qv_buffer, sim.window, λ)
    end
    _refresh_convection_forcing!(sim.model.convection, sim.model, sim.window)
    return λ
end

function _load_window(driver::D, win::Int) where {D <: AbstractMetDriver}
    return load_transport_window(driver, win)
end

function _maybe_advance_window!(sim::DrivenSimulation)
    if sim.iteration > 0 && sim.iteration == sim.current_window_end_iteration
        next_window = sim.current_window_index + 1
        next_window <= sim.stop_window ||
            throw(ArgumentError("DrivenSimulation attempted to step past stop_window=$(sim.stop_window)"))
        sim.current_window_start_iteration = sim.iteration
        sim.current_window_index = next_window
        sim.steps_per_window = sim.steps_per_window_schedule[next_window]
        sim.current_window_end_iteration = sim.iteration + sim.steps_per_window
        sim.Δt = sim.window_dt / typeof(sim.Δt)(sim.steps_per_window)
        _take_prefetched_window!(sim, next_window)
        if sim.air_mass_reset_mode !== :none
            _reset_air_mass!(sim.model.state, sim.window.air_mass,
                             sim.model.grid.horizontal,
                             sim.air_mass_reset_mode)
        end
        if sim.qv_buffer !== nothing && !has_humidity_endpoints(sim.window)
            throw(ArgumentError("driver humidity endpoint support changed between windows"))
        end
        _validate_convection_runtime(sim.model, sim.driver, sim.window)
        _refresh_dz_for_window!(sim)
        _refresh_pbl_kz_for_window!(sim.model.diffusion, sim)
        # Under the canonical `:window_constant` contract, the runtime's own
        # flux divergence should integrate to `(m_next - m)` over each window.
        # `air_mass_reset_mode` controls whether the binary endpoint is still
        # treated as authoritative at window boundaries.
        invalidate_cmfmc_cache!(sim.model.workspace.convection_ws)
        _start_window_prefetch!(sim, next_window + 1)
    end
    return nothing
end

function _maybe_reset_to_window_endpoint!(sim::DrivenSimulation)
    (sim.air_mass_reset_mode !== :none && _uses_binary_transport_schedule(sim)) ||
        return nothing
    expected_air_mass!(sim.expected_air_mass, sim.window, one(typeof(sim.Δt)))
    _reset_air_mass!(sim.model.state, sim.expected_air_mass,
                     sim.model.grid.horizontal,
                     sim.air_mass_reset_mode)
    return nothing
end

"""
    DrivenSimulation(model, driver; kwargs...)

Construct a window-driven `src` runtime.

Keyword arguments:
- `start_window=1`
- `stop_window=total_windows(driver)`
- `initialize_air_mass=true`
- `use_midpoint_forcing=true`
- `interpolate_fluxes_within_window=nothing` (derive from driver)
- `air_mass_reset_mode=:preserve_tracer_mass` — one of `:none`, `:preserve_vmr`, or
  `:preserve_tracer_mass`. When non-`:none`, each newly loaded window
  replaces prognostic air mass using the selected tracer invariant. For
  binary-scheduled runs, the same endpoint reset is applied before the
  once-per-window convection/chemistry block so physics sees the binary's
  authoritative window-end mass.
- `surface_sources=()`
- `chemistry=NoChemistry()` — applied after advection + surface sources each step
- `callbacks=NamedTuple()`
- `start_time=0` — simulation clock origin [s]. Multi-binary runners MUST pass
  the accumulated run time here when rebuilding the sim per binary: `sim.time`
  feeds `current_time(meteo)`, which time-varying surface-flux sources use to
  select their emission slice (seconds since the RUN start, not the binary
  start). Restarting the clock at 0 each day silently replays day-1 fluxes —
  the December-2021 co2_natural +1 Pg/month surplus (plan 45 Stage-4 A/B
  experiment attributed the leak to exactly this).
"""
function DrivenSimulation(model::TransportModel,
                          driver::D;
                          start_window::Integer = 1,
                          stop_window::Integer = total_windows(driver),
                          initialize_air_mass::Bool = true,
                          use_midpoint_forcing::Bool = true,
                          interpolate_fluxes_within_window = nothing,
                          air_mass_reset_mode = :preserve_tracer_mass,
                          surface_sources = (),
                          chemistry::AbstractChemistryOperator = NoChemistry(),
                          callbacks = NamedTuple(),
                          start_time::Real = 0) where {D <: AbstractMetDriver}
    1 <= start_window <= stop_window <= total_windows(driver) ||
        throw(ArgumentError("invalid window range: start_window=$(start_window), stop_window=$(stop_window), total_windows=$(total_windows(driver))"))
    supports_native_vertical_flux(driver) ||
        throw(ArgumentError("DrivenSimulation requires native vertical mass fluxes in the met-driver contract"))
    isfinite(start_time) ||
        throw(ArgumentError("DrivenSimulation start_time must be finite; got $(start_time)"))

    _check_grid_compatibility(model.grid, driver_grid(driver))
    _check_basis_compatibility(model, driver)

    # Read/decode the first forcing window once. Adapt its host payload twice
    # when prefetching so the two device buffers remain independently writable.
    loaded_window = _load_window(driver, start_window)
    window = _adapt_window_to_model_backend(loaded_window, model.state.air_mass)
    prefetch_window = if _prefetch_enabled(model.state.air_mass) && start_window < stop_window
        # A custom driver may already supply device arrays; adapting those to
        # the same backend can alias, so copy that window explicitly instead.
        _window_backend_adapter(loaded_window.air_mass) === Array ?
            _adapt_window_to_model_backend(loaded_window, model.state.air_mass) : deepcopy(window)
    else
        window
    end
    prefetch_task = _empty_prefetch_task()
    expected_air_mass = _allocate_storage_like(model.state.air_mass)
    qv_buffer = _allocate_qv_buffer(window)
    surface_sources_adapted = _adapt_sources_to_model_backend(Tuple(surface_sources), model.state.air_mass)
    foreach(source -> _check_surface_source_compatibility(model.state, source), surface_sources_adapted)

    # Chemistry + emissions are applied inside the model's transport block,
    # not as a sim-level post-step. `with_emissions` installs the
    # user-supplied surface sources as a `SurfaceFluxOperator` inside the
    # wrapped model so the palindrome's S slot runs at the correct
    # center-of-transport position. `with_chemistry` installs the user's
    # chemistry in the model; `step!(model)` runs
    # `advection → emissions → diffusion → chemistry` as ONE composed
    # call. The step loop delegates entirely to the model-level operator
    # composition.
    #
    # The palindrome integration preserves TM5's
    # `advection → emissions → chemistry` order with emissions inside the
    # palindrome, so the sim delegates entirely to `step!(model)`.
    model = with_chemistry(model, chemistry)
    if !isempty(surface_sources_adapted)
        emissions_op = SurfaceFluxOperator(PerTracerFluxMap(surface_sources_adapted))
        model = with_emissions(model, emissions_op)
    end
    model = _install_convection_forcing(model, driver, window)
    FT = _storage_eltype(model.state.air_mass)
    step_schedule = _driver_step_schedule(driver)
    all(>(0), step_schedule) || throw(ArgumentError(
        "DrivenSimulation driver step schedule must contain only positive integers"))
    isfinite(window_dt(driver)) && window_dt(driver) > 0 || throw(ArgumentError(
        "DrivenSimulation driver window_dt must be finite and positive"))
    steps_current = step_schedule[Int(start_window)]
    Δt = FT(window_dt(driver)) / FT(steps_current)
    nsteps_total = sum(@view step_schedule[Int(start_window):Int(stop_window)])

    flux_interp = interpolate_fluxes_within_window === nothing ?
                  (flux_interpolation_mode(driver) === :interpolate) : Bool(interpolate_fluxes_within_window)
    reset_mode = _normalize_air_mass_reset_mode(air_mass_reset_mode)

    sim = DrivenSimulation{typeof(model), typeof(driver), typeof(window),
                           typeof(expected_air_mass), typeof(qv_buffer), FT,
                           typeof(callbacks), typeof(prefetch_task)}(
        model,
        driver,
        window,
        prefetch_window,
        prefetch_task,
        0,
        expected_air_mass,
        qv_buffer,
        Δt,
        FT(window_dt(driver)),
        steps_current,
        step_schedule,
        Float64(start_time),
        Float64(start_time),
        0,
        Int(start_window),
        Int(start_window),
        0,
        steps_current,
        Int(stop_window),
        Int(nsteps_total),
        callbacks,
        initialize_air_mass,
        use_midpoint_forcing,
        flux_interp,
        reset_mode,
    )

    if initialize_air_mass
        _copy_storage!(sim.model.state.air_mass, sim.window.air_mass)
        _refresh_state_halos!(sim.model.state, sim.model.grid.horizontal)
    elseif sim.air_mass_reset_mode !== :none
        _reset_air_mass!(sim.model.state, sim.window.air_mass,
                         sim.model.grid.horizontal,
                         sim.air_mass_reset_mode)
    else
        _refresh_state_halos!(sim.model.state, sim.model.grid.horizontal)
    end
    copy_fluxes!(sim.model.fluxes, sim.window.fluxes)
    _copy_storage!(sim.expected_air_mass, sim.window.air_mass)
    if sim.qv_buffer !== nothing
        _copy_storage!(sim.qv_buffer, sim.window.qv_start)
    end
    _refresh_dz_for_window!(sim)
    _refresh_pbl_kz_for_window!(sim.model.diffusion, sim)
    _reclaim_backend_pool_after_startup!(sim.model.state.air_mass)
    _start_window_prefetch!(sim, Int(start_window) + 1)
    return sim
end

window_index(sim::DrivenSimulation) = sim.current_window_index
function substep_index(sim::DrivenSimulation)
    if sim.iteration == sim.current_window_end_iteration &&
       sim.current_window_index < sim.stop_window
        return 1
    end
    return min(sim.steps_per_window,
               sim.iteration - sim.current_window_start_iteration + 1)
end
current_qv(sim::DrivenSimulation) = sim.qv_buffer

"""
    current_time(sim::DrivenSimulation) -> Float64

Simulation time [s] at the start of the next step. Returns
`sim.time` (Float64), which is initialized to `start_time` at sim construction
(seconds since the RUN start for multi-binary runs) and recomputed from the
window and step counters at the end of each `step!(sim)`, so window ends fall
exactly on multiples of `window_dt`.

`sim` is threaded through operators via the `meteo` kwarg:

    step!(sim.model, sim.Δt; meteo = sim)   # not sim.driver

so operators that need time (`StepwiseField` emission rates,
time-varying Kz, future convection DerivedConvMassFluxField) read
`current_time(meteo)` and get `sim.time`. `meteo.driver` remains
accessible for operator code that needs driver-level capabilities
(e.g. `supports_cmfmc(meteo.driver)`).

Meteorological drivers are stateless and deliberately do not implement
`current_time`; operators receive the simulation clock, not `sim.driver`.
"""
MetDrivers.current_time(sim::DrivenSimulation) = sim.time

# The clock follows the counters instead of accumulating steps: a Float32
# clock adding 3600/7 s steps is 3 h off after a year, and even a Float64 sum
# misses window ends by an ulp.
_clock_time(sim::DrivenSimulation) = sim.start_time + Float64(sim.window_dt) *
    ((sim.current_window_index - sim.start_window) +
     (sim.iteration - sim.current_window_start_iteration) / sim.steps_per_window)

# Diagnostic override for convection-cadence sensitivity studies. Setting
# ATMOSTR_FORCE_PER_SUBSTEP_PHYSICS=1 forces convection + chemistry to run every
# advection substep (the pre-2026-05-31 behaviour) even on a binary that declares
# the per-window contract, so the two cadences can be A/B-compared on the SAME
# binary. Default off — never affects production runs.
@inline _uses_binary_transport_schedule(sim::DrivenSimulation) =
    uses_binary_substep_contract(sim.driver) &&
    get(ENV, "ATMOSTR_FORCE_PER_SUBSTEP_PHYSICS", "0") != "1"

function step!(sim::DrivenSimulation)
    sim.iteration < sim.final_iteration ||
        throw(ArgumentError("DrivenSimulation has already completed all scheduled steps"))

    SectionTimer.@section :window_advance _maybe_advance_window!(sim)
    substep = substep_index(sim)
    SectionTimer.@section :forcing_refresh _refresh_forcing!(sim, substep)

    # The default path keeps the live operator suite in one call.
    # Transport binaries carry an advection substep contract, not a
    # physics cadence contract, so driven
    # binary-scheduled runs apply only the transport block at each stored
    # substep and defer convection + chemistry to the end of the met window.
    if _uses_binary_transport_schedule(sim)
        transport_step!(sim.model, sim.Δt; meteo = sim)
    else
        step!(sim.model, sim.Δt; meteo = sim)
    end
    sim.iteration += 1
    sim.time = _clock_time(sim)
    if _uses_binary_transport_schedule(sim) &&
       sim.iteration == sim.current_window_end_iteration
        _maybe_reset_to_window_endpoint!(sim)
        convection_chemistry_step!(sim.model, sim.window_dt; meteo = sim)
    end
    for callback in values(sim.callbacks)
        callback(sim)
    end
    return nothing
end

"""
    run_window!(sim::DrivenSimulation)

Advance exactly the current meteorological window and return sim. If sim is
positioned at a completed non-final window, the next window is loaded first.
"""
function run_window!(sim::DrivenSimulation)
    if sim.iteration == sim.current_window_end_iteration &&
       sim.current_window_index < sim.stop_window
        SectionTimer.@section :window_advance _maybe_advance_window!(sim)
    end
    target_iteration = min(sim.final_iteration, sim.current_window_end_iteration)
    while sim.iteration < target_iteration
        step!(sim)
    end
    return sim
end

function run!(sim::DrivenSimulation)
    while sim.iteration < sim.final_iteration
        step!(sim)
    end
    return sim
end

export DrivenSimulation, run_window!, window_index, substep_index, current_qv
