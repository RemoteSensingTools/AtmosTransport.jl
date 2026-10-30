# Synthetic window driver shared by convection and chemistry runtime checks.

const _REALISTIC_AIR_MASS_KG = 1e16

function _make_convection_grid(; FT = Float64, Nx = 4, Ny = 3, Nz = 5)
    mesh = LatLonMesh(; FT = FT, Nx = Nx, Ny = Ny)
    vertical = HybridSigmaPressure(
        FT[0, 100, 300, 600, 1000, 2000],
        FT[0, 0, 0.1, 0.3, 0.7, 1],
    )
    return AtmosGrid(mesh, vertical, CPU(); FT = FT)
end

function _make_cmfmc_forcing(FT, Nx, Ny, Nz; peak = FT(0.02), top_detrain = FT(0.01))
    cmfmc = zeros(FT, Nx, Ny, Nz + 1)
    cmfmc[:, :, 4] .= peak * FT(0.5)
    cmfmc[:, :, 3] .= peak
    cmfmc[:, :, 2] .= peak * FT(0.5)

    dtrain = zeros(FT, Nx, Ny, Nz)
    dtrain[:, :, 1] .= top_detrain
    return ConvectionForcing(cmfmc, dtrain, nothing)
end

struct _ConvectionWindowDriver{FT, GridT, WindowT} <: AbstractMetDriver
    grid    :: GridT
    windows :: Vector{WindowT}
    dt      :: FT
    steps   :: Int
    binary_contract :: Bool
end

AtmosTransport.total_windows(driver::_ConvectionWindowDriver) = length(driver.windows)
AtmosTransport.window_dt(driver::_ConvectionWindowDriver) = driver.dt
AtmosTransport.steps_per_window(driver::_ConvectionWindowDriver) = driver.steps
AtmosTransport.load_transport_window(driver::_ConvectionWindowDriver, win::Int) = driver.windows[win]
AtmosTransport.driver_grid(driver::_ConvectionWindowDriver) = driver.grid
AtmosTransport.air_mass_basis(::_ConvectionWindowDriver) = :dry
AtmosTransport.MetDrivers.flux_interpolation_mode(::_ConvectionWindowDriver) = :constant
AtmosTransport.MetDrivers.supports_native_vertical_flux(::_ConvectionWindowDriver) = true
AtmosTransport.MetDrivers.uses_binary_substep_contract(driver::_ConvectionWindowDriver) =
    driver.binary_contract
AtmosTransport.supports_convection(::_ConvectionWindowDriver) = true

struct _CountingChemistry <: AbstractChemistryOperator
    calls    :: Base.RefValue{Int}
    total_dt :: Base.RefValue{Float64}
end

function AtmosTransport.Operators.apply!(state, meteo, grid,
                                         op::_CountingChemistry, dt;
                                         workspace = nothing)
    op.calls[] += 1
    op.total_dt[] += Float64(dt)
    return state
end

function _make_convection_window_driver(; FT = Float64, steps = 1,
                                        binary_contract = false, tm5 = false)
    grid = _make_convection_grid(FT = FT)
    Nx, Ny, Nz = 4, 3, 5
    air_mass = fill(FT(_REALISTIC_AIR_MASS_KG), Nx, Ny, Nz)
    ps = fill(FT(95_000), Nx, Ny)
    fluxes = allocate_face_fluxes(grid.horizontal, Nz; FT = FT, basis = DryBasis)

    if tm5
        areas = AtmosTransport.Grids.cell_areas_by_latitude(grid.horizontal)
        for j in 1:Ny
            air_mass[:,j,:] .= FT(5e3) * FT(areas[j])
        end
        make_forcing(peak) = begin
            entu, detu, entd, detd = ntuple(_ -> zeros(FT,Nx,Ny,Nz),4)
            entu[:,:,Nz] .= FT(0.3)*peak
            entu[:,:,4] .= FT(0.5)*peak
            entu[:,:,3] .= peak
            detu[:,:,2] .= FT(0.01)
            ConvectionForcing(nothing,nothing,(;entu,detu,entd,detd))
        end
        forcing_a, forcing_b = make_forcing(FT(0.01)), make_forcing(FT(0.02))
    else
        forcing_a = _make_cmfmc_forcing(FT,Nx,Ny,Nz; peak=FT(0.02),top_detrain=FT(0.01))
        forcing_b = _make_cmfmc_forcing(FT,Nx,Ny,Nz; peak=FT(0.5),top_detrain=FT(0.1))
    end

    window_a = TransportWindow(air_mass, ps, fluxes; convection = forcing_a)
    window_b = TransportWindow(air_mass, ps, fluxes; convection = forcing_b)
    driver = _ConvectionWindowDriver{FT, typeof(grid), typeof(window_a)}(
        grid, [window_a, window_b], FT(1800), Int(steps), Bool(binary_contract))
    return driver, forcing_a, forcing_b
end
