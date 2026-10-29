#!/usr/bin/env julia
"""
Plan 23 Commit 6: DrivenSimulation end-to-end with TM5Convection.

Builds a synthetic in-memory window-driver (paralleling the
`_ConvectionWindowDriver` pattern in test_transport_model_convection.jl)
whose windows carry `tm5_fields` forcing. Installs `TM5Convection`
on the model and runs the sim through two windows. Verifies:

  - `DrivenSimulation` allocates the expected `TM5Workspace`.
  - Per-substep forcing refresh populates
    `sim.model.convection_forcing.tm5_fields` from the window
    (capability-invariant between windows per Decision 27).
  - TM5 convection block runs inside `step!` between transport
    and chemistry; tracer state changes.
  - Total tracer mass conserved across the full sim run.

The plan's CATRINE-style 1-day ERA5 real-data test and the
operator-ordering study (plan-17-parallel A/B/C/D positions)
require preprocessed ERA5 binaries with TM5 sections. The ec2tm!
preprocessor integration is deferred (Commit 3 §scope decision).
This synthetic end-to-end test unblocks DrivenSimulation + TM5
regression protection until the ECMWF convective download path
ships.
"""

using Test

using AtmosTransport

using .AtmosTransport.State: CellState, DryBasis, allocate_face_fluxes
using .AtmosTransport.Architectures: CPU as GridsCPU
using .AtmosTransport.Grids: AtmosGrid, LatLonMesh, HybridSigmaPressure,
                              cell_areas_by_latitude
using .AtmosTransport.Operators: TM5Convection, TM5Workspace,
                                 UpwindScheme, AbstractConvection
using .AtmosTransport.MetDrivers: ConvectionForcing, TransportWindow,
                                   AbstractMetDriver
using .AtmosTransport.Models: TransportModel, DrivenSimulation,
                               with_convection, step!, window_index

const _BMASS_PER_LAYER = 5e3
const FT = Float64

# -------------------------------------------------------------------------
# Synthetic in-memory driver with TM5 forcing per window.
# -------------------------------------------------------------------------
struct _TM5WindowDriver{FT, GridT, WindowT} <: AbstractMetDriver
    grid    :: GridT
    windows :: Vector{WindowT}
    dt      :: FT
    steps   :: Int
end

AtmosTransport.total_windows(driver::_TM5WindowDriver) = length(driver.windows)
AtmosTransport.window_dt(driver::_TM5WindowDriver) = driver.dt
AtmosTransport.steps_per_window(driver::_TM5WindowDriver) = driver.steps
AtmosTransport.load_transport_window(driver::_TM5WindowDriver, win::Int) = driver.windows[win]
AtmosTransport.driver_grid(driver::_TM5WindowDriver) = driver.grid
AtmosTransport.air_mass_basis(::_TM5WindowDriver) = :dry
AtmosTransport.MetDrivers.flux_interpolation_mode(::_TM5WindowDriver) = :constant
AtmosTransport.MetDrivers.supports_native_vertical_flux(::_TM5WindowDriver) = true
AtmosTransport.supports_convection(::_TM5WindowDriver) = true

function _make_grid(; FT = Float64, Nx = 4, Ny = 3, Nz = 5)
    mesh = LatLonMesh(; FT = FT, Nx = Nx, Ny = Ny)
    vertical = HybridSigmaPressure(
        FT[0, 100, 300, 600, 1000, 2000],
        FT[0, 0, 0.1, 0.3, 0.7, 1],
    )
    return AtmosGrid(mesh, vertical, GridsCPU(); FT = FT)
end

function _make_air_mass(grid, FT, Nx, Ny, Nz)
    areas = cell_areas_by_latitude(grid.horizontal)
    air_mass = zeros(FT, Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        air_mass[i, j, k] = FT(_BMASS_PER_LAYER) * FT(areas[j])
    end
    return air_mass
end

function _make_tm5_forcing(FT, Nx, Ny, Nz;
                            peak_entu = FT(0.02),
                            peak_detu = FT(0.01))
    # TM5 forcings are kg/m²/s. The runtime workspace carries cell
    # areas so the matrix can divide by air_mass / cell_area.
    # Entrainment profile spans surface (k=Nz) through mid-cloud so
    # the updraft actually picks up surface-layer air. TM5 has no
    # well-mixed sub-cloud treatment — without surface entrainment
    # the surface tracer would stay put.
    entu = zeros(FT, Nx, Ny, Nz)
    entu[:, :, Nz] .= peak_entu * FT(0.3)   # surface-layer entrainment
    entu[:, :, 4]  .= peak_entu * FT(0.5)
    entu[:, :, 3]  .= peak_entu
    detu = zeros(FT, Nx, Ny, Nz)
    detu[:, :, 2]  .= peak_detu
    entd = zeros(FT, Nx, Ny, Nz)
    detd = zeros(FT, Nx, Ny, Nz)
    return ConvectionForcing(nothing, nothing,
                              (entu = entu, detu = detu,
                               entd = entd, detd = detd))
end

function _make_tm5_window_driver(; FT = Float64)
    grid = _make_grid(FT = FT)
    Nx, Ny, Nz = 4, 3, 5
    air_mass = _make_air_mass(grid, FT, Nx, Ny, Nz)
    ps = fill(FT(95_000), Nx, Ny)
    fluxes = allocate_face_fluxes(grid.horizontal, Nz; FT = FT, basis = DryBasis)

    forcing_a = _make_tm5_forcing(FT, Nx, Ny, Nz;
                                   peak_entu = FT(0.01))
    forcing_b = _make_tm5_forcing(FT, Nx, Ny, Nz;
                                   peak_entu = FT(0.02))

    window_a = TransportWindow(air_mass, ps, fluxes; convection = forcing_a)
    window_b = TransportWindow(air_mass, ps, fluxes; convection = forcing_b)
    driver = _TM5WindowDriver{FT, typeof(grid), typeof(window_a)}(
        grid, [window_a, window_b], FT(1800), 1)
    return driver, forcing_a, forcing_b
end

@testset "plan 23 Commit 6: DrivenSimulation with TM5Convection" begin
    driver, forcing_a, forcing_b = _make_tm5_window_driver(FT = FT)

    Nx, Ny, Nz = 4, 3, 5
    air_mass = _make_air_mass(driver.grid, FT, Nx, Ny, Nz)
    tracer = zeros(FT, Nx, Ny, Nz)
    tracer[:, :, Nz] .= FT(1e-6) .* air_mass[:, :, Nz]
    state = CellState(air_mass; CO2 = tracer)
    fluxes = allocate_face_fluxes(driver.grid.horizontal, Nz; FT = FT, basis = DryBasis)

    model = TransportModel(state, fluxes, driver.grid, UpwindScheme();
                            convection = TM5Convection())

    sim = DrivenSimulation(model, driver; start_window = 1, stop_window = 2)

    # (A) Workspace allocated by DrivenSimulation construction.
    @test sim.model.workspace.convection_ws isa TM5Workspace{FT}

    # (B) Forcing is a COPY, not an alias — capability preserved
    #     and buffers preallocated for per-substep refresh.
    @test sim.model.convection_forcing !== forcing_a
    @test sim.model.convection_forcing.tm5_fields !== nothing
    @test sim.model.convection_forcing.tm5_fields.entu !== forcing_a.tm5_fields.entu
    @test sim.model.convection_forcing.tm5_fields.entu == forcing_a.tm5_fields.entu

    # (C) Step 1: refresh + TM5 apply + chemistry (noop).
    mass_before_step1 = sum(sim.model.state.tracers_raw)
    step!(sim)
    mass_after_step1 = sum(sim.model.state.tracers_raw)
    # Mass conservation to F64 machine precision across transport + TM5.
    @test isapprox(mass_after_step1, mass_before_step1; rtol = 1e-10)

    # (D) Step 2: advances to window 2, refresh picks up forcing_b.
    step!(sim)
    @test window_index(sim) == 2
    @test sim.model.convection_forcing.tm5_fields.entu == forcing_b.tm5_fields.entu
    mass_after_step2 = sum(sim.model.state.tracers_raw)
    @test isapprox(mass_after_step2, mass_before_step1; rtol = 1e-10)

    # (E) Tracer redistributed vertically (not silent identity).
    @test sim.model.state.tracers_raw != reshape(tracer, Nx, Ny, Nz, 1)
    # Surface layer lost mass to cloud layers.
    surface_mass = sum(sim.model.state.tracers_raw[:, :, Nz, 1])
    initial_surface = sum(tracer[:, :, Nz])
    @test surface_mass < initial_surface
end

@testset "plan 23 Commit 6: DrivenSimulation preserves FT for TM5" begin
    # Float32 variant — the sim's Δt, window_dt, and forcing arrays
    # should all stay FT = Float32 through the DrivenSimulation
    # construction (matches the CMFMC FT-preservation test in
    # test_transport_model_convection.jl).
    FT32 = Float32
    driver, forcing_a, _ = _make_tm5_window_driver(FT = FT32)

    Nx, Ny, Nz = 4, 3, 5
    air_mass = _make_air_mass(driver.grid, FT32, Nx, Ny, Nz)
    tracer   = FT32(1e-6) .* air_mass
    state = CellState(air_mass; CO2 = tracer)
    fluxes = allocate_face_fluxes(driver.grid.horizontal, Nz;
                                    FT = FT32, basis = DryBasis)
    model = TransportModel(state, fluxes, driver.grid, UpwindScheme();
                            convection = TM5Convection())

    sim = DrivenSimulation(model, driver; start_window = 1, stop_window = 1)
    @test typeof(sim.Δt) === FT32
    @test typeof(sim.window_dt) === FT32
    @test eltype(sim.model.convection_forcing.tm5_fields.entu) === FT32
    @test sim.model.convection_forcing.tm5_fields.entu == forcing_a.tm5_fields.entu
end
