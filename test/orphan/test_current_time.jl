#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan 18 A3 regression: sim-level `current_time` override
#
# Plan 17 threaded `current_time(meteo)` through operator `apply!` methods,
# but the meteo kwarg was passed as `meteo = sim.driver` — and the driver
# is stateless (struct holds only reader + grid). Any operator reading
# `current_time(meteo)` silently got `0.0` throughout the run. Plan 18 A3
# fixes this by passing `meteo = sim` instead, and adding a
# `current_time(sim::DrivenSimulation) = sim.time` override.
#
# Tests:
# 1. `current_time(nothing) == 0.0` fallback (no meteo / unit test context).
# 2. `current_time(sim)` at construction returns 0.0.
# 3. `current_time(sim)` advances by `sim.Δt` after each `step!(sim)`.
# 4. `step!(sim.model, sim.Δt; meteo = sim)` gives operators access to
#    the sim-level clock — verified by a custom chemistry operator that
#    captures the meteo it sees.
# ---------------------------------------------------------------------------

using Test

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport: AbstractChemistryOperator

# Local copy of the minimal driven-sim binary-writer (same shape as the
# one at the top of test_driven_simulation.jl). Inlined here so we don't
# `include` that file and pick up its testsets.
function _write_sim_latlon_binary(path::AbstractString;
                                   FT::Type{<:AbstractFloat} = Float64,
                                   window_mass_scales::Tuple{Vararg{Real}} = (1, 1))
    Nx, Ny, Nz = 4, 3, 2
    mesh = LatLonMesh(; FT = FT, Nx = Nx, Ny = Ny)
    vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)

    windows = [
        begin
            m   = fill(FT(window_mass_scales[win]), Nx, Ny, Nz)
            am  = zeros(FT, Nx + 1, Ny, Nz)
            bm  = zeros(FT, Nx, Ny + 1, Nz)
            cm  = zeros(FT, Nx, Ny, Nz + 1)
            ps  = fill(FT(95000 + 100win), Nx, Ny)
            qv_start = fill(FT(0.01win), Nx, Ny, Nz)
            qv_end   = fill(FT(0.01win + 0.01), Nx, Ny, Nz)
            dam = fill(FT(0.2win), Nx + 1, Ny, Nz)
            dbm = fill(FT(0.4win), Nx, Ny + 1, Nz)
            dcm = fill(FT(0.6win), Nx, Ny, Nz + 1)
            dm  = fill(FT(0.8win), Nx, Ny, Nz)
            (; m, am, bm, cm, ps, qv_start, qv_end, dam, dbm, dcm, dm)
        end for win in 1:length(window_mass_scales)
    ]

    write_transport_binary(path, grid, windows;
                           FT = FT,
                           dt_met_seconds = 3600.0,
                           half_dt_seconds = 1800.0,
                           steps_per_window = 2,
                           mass_basis = :moist,
                           surface_pressure_basis = TotalSurfacePressure(),
                           source_flux_sampling = :window_start_endpoint,
                           flux_sampling = :window_constant,
                           extra_header = Dict(
                               "poisson_balance_target_scale" => 0.25,
                               "poisson_balance_target_semantics" => "forward_window_mass_difference / (2 * steps_per_window)",
                           ))
    return grid
end

@testset "current_time(nothing) == 0.0" begin
    @test current_time(nothing) === 0.0
end

@testset "current_time(sim) starts at 0.0 and advances per step" begin
    mktemp() do path, io
        close(io)
        _write_sim_latlon_binary(path; FT = Float64, window_mass_scales = (1, 1))

        driver = TransportBinaryDriver(path; FT = Float64, arch = CPU())
        grid = driver_grid(driver)
        state = CellState(MoistBasis, ones(Float64, 4, 3, 2); CO2 = fill(400e-6, 4, 3, 2))
        fluxes = allocate_face_fluxes(grid.horizontal, 2; FT = Float64, basis = MoistBasis)
        model = TransportModel(state, fluxes, grid, UpwindScheme())
        sim = DrivenSimulation(model, driver;
                               start_window = 1, stop_window = 2)

        # At construction
        @test current_time(sim) ≈ 0.0  rtol = 1e-14

        # After one step
        step!(sim)
        @test current_time(sim) ≈ sim.Δt  rtol = 1e-14

        # After several steps (still within window 1)
        n = sim.steps_per_window - 1   # fills the first window
        for _ in 1:n
            step!(sim)
        end
        @test current_time(sim) ≈ (1 + n) * sim.Δt  rtol = 1e-14

        close(driver)
    end
end

# ---------------------------------------------------------------------------
# Custom chemistry operator that captures the meteo it receives. Used to
# verify that `step!` threads `meteo = sim` (plan 18 A3), not `sim.driver`.
# Stores the captured meteo in a 1-elem Ref so the test can inspect it.
# ---------------------------------------------------------------------------
struct _MeteoCaptureChemistry{R} <: AbstractChemistryOperator
    captured :: R   # Ref{Any} — the last meteo seen by `apply!`
end
_MeteoCaptureChemistry() = _MeteoCaptureChemistry(Ref{Any}(nothing))

# Minimal apply! for our capture op — matches plan 15 chemistry signature.
function AtmosTransport.apply!(state,
                                meteo,
                                grid,
                                op::_MeteoCaptureChemistry,
                                dt::Real;
                                workspace = nothing)
    op.captured[] = meteo
    return state
end

@testset "step!(sim) threads meteo = sim (not sim.driver) — plan 18 A3" begin
    mktemp() do path, io
        close(io)
        _write_sim_latlon_binary(path; FT = Float64, window_mass_scales = (1, 1))

        driver = TransportBinaryDriver(path; FT = Float64, arch = CPU())
        grid = driver_grid(driver)
        state = CellState(MoistBasis, ones(Float64, 4, 3, 2); CO2 = fill(400e-6, 4, 3, 2))
        fluxes = allocate_face_fluxes(grid.horizontal, 2; FT = Float64, basis = MoistBasis)
        model = TransportModel(state, fluxes, grid, UpwindScheme())

        capture_op = _MeteoCaptureChemistry()
        sim = DrivenSimulation(model, driver;
                               start_window = 1, stop_window = 2,
                               chemistry = capture_op)

        # Before any step — capture ref is still empty
        @test capture_op.captured[] === nothing

        # One step — the chemistry block should have received `sim` as meteo
        step!(sim)
        captured = capture_op.captured[]
        @test captured === sim                                   # identity match
        @test current_time(captured) ≈ sim.Δt  rtol = 1e-14      # sim.time was advanced after step!
        @test captured isa DrivenSimulation                      # type check
        # And the driver is still reachable via meteo.driver
        @test captured.driver === driver

        close(driver)
    end
end
