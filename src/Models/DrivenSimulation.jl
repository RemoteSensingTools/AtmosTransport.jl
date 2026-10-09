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
    physics_every_substep       :: Bool
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
