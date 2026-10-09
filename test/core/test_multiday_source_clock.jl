#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Multi-day simulation clock — time-varying surface-flux slice selection
#
# ROOT-CAUSE REGRESSION TEST for the December-2021 co2_natural +1 Pg/month
# surplus: multi-binary runners rebuild `DrivenSimulation` per daily binary,
# and `sim.time` restarted at 0 each day — so `current_time(meteo)`-driven
# time-varying surface-flux sources replayed DAY-1's emission slices every
# day (constant-rate tracers were unaffected, which is why sf6/co2_fossil
# budgets closed while co2_natural gained 31×day1 − Σmonth ≈ +1 Pg). The
# plan-45 Stage-4 A/B experiment (anomaly-reference transport left the
# surplus unchanged) refuted the F32 hypothesis and exposed this.
#
# The fix: `DrivenSimulation(...; start_time)` sets the clock origin, and
# both runner loops pass the accumulated run time.
# ---------------------------------------------------------------------------

using Test

using AtmosTransport
using .AtmosTransport.MetDrivers: driver_grid, current_time
using .AtmosTransport.Operators.SurfaceFlux: StepwiseFlux, _flux_temporal_weights

function _write_clock_test_binary(path; FT = Float64, steps = 2)
    Nx, Ny, Nz = 4, 3, 2
    mesh = LatLonMesh(; FT = FT, Nx = Nx, Ny = Ny)
    vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
    windows = [
        begin
            m = fill(FT(1), Nx, Ny, Nz)
            am = zeros(FT, Nx + 1, Ny, Nz)
            bm = zeros(FT, Nx, Ny + 1, Nz)
            cm = zeros(FT, Nx, Ny, Nz + 1)
            ps = fill(FT(95000 + 100win), Nx, Ny)
            qv_start = fill(FT(0.01win), Nx, Ny, Nz)
            qv_end = fill(FT(0.01win + 0.01), Nx, Ny, Nz)
            (; m, am, bm, cm, ps, qv_start, qv_end)
        end for win in 1:2
    ]
    write_transport_binary(path, grid, windows;
                           FT = FT, dt_met_seconds = 3600.0,
                           half_dt_seconds = 1800.0, steps_per_window = steps,
                           mass_basis = :moist,
                           source_flux_sampling = :window_start_endpoint)
    return grid
end

@testset "DrivenSimulation start_time sets the clock origin" begin
    mktemp() do path, io
        close(io)
        _write_clock_test_binary(path)
        driver = TransportBinaryDriver(path; FT = Float64, arch = CPU())
        grid = driver_grid(driver)
        state = CellState(MoistBasis, ones(Float64, 4, 3, 2);
                          CO2 = fill(400e-6, 4, 3, 2))
        fluxes = allocate_face_fluxes(grid.horizontal, 2; FT = Float64,
                                      basis = MoistBasis)
        model = TransportModel(state, fluxes, grid, UpwindScheme())

        # default: clock starts at zero (back-compat)
        sim0 = DrivenSimulation(model, driver)
        @test current_time(sim0) == 0.0

        # day-2 rebuild: the clock must carry the accumulated run time
        day2 = 86400.0
        sim = DrivenSimulation(model, driver; start_time = day2)
        @test current_time(sim) == day2
        step!(sim)
        @test current_time(sim) == day2 + sim.Δt   # advances FROM the origin
    end
end

@testset "Float64 clock lands on window ends for any step count" begin
    # Seven steps per hour: a Float32 clock adding 3600/7 s is 3 h off after a
    # year, and a Float64 sum misses window ends by an ulp.
    mktemp() do path, io
        close(io)
        _write_clock_test_binary(path; FT = Float32, steps = 7)
        driver = TransportBinaryDriver(path; FT = Float32, arch = CPU())
        grid = driver_grid(driver)
        state = CellState(MoistBasis, ones(Float32, 4, 3, 2); CO2 = fill(400f-6, 4, 3, 2))
        fluxes = allocate_face_fluxes(grid.horizontal, 2; FT = Float32, basis = MoistBasis)
        sim = DrivenSimulation(TransportModel(state, fluxes, grid, UpwindScheme()), driver;
                               start_time = 86400.0)
        times = map(1:14) do _
            step!(sim)
            current_time(sim)
        end
        @test eltype(times) === Float64
        @test times[7] == 86400.0 + 3600 && times[14] == 86400.0 + 7200
        @test times ≈ 86400.0 .+ (1:14) .* (3600 / 7) rtol = 4eps()
    end
end

@testset "stepwise slice selection uses absolute run time" begin
    # 3-hourly slices over two days (the lmdz/CAMS layout)
    times = collect(0.0:10800.0:(2 * 86400.0 - 10800.0))   # 16 slices

    # day 1, hour 4.5 -> slice block 2 (t in [3h, 6h))
    i0, i1, w0, w1 = _flux_temporal_weights(StepwiseFlux(), times, 4.5 * 3600, 450.0)
    @test i0 == 2 && w0 == 1

    # day 2, hour 4.5 ABSOLUTE (t = 28.5h) -> slice block 10, NOT 2.
    # A per-day clock reset would re-select block 2 here: that is exactly
    # the day-1 replay that produced the +1 Pg co2_natural surplus.
    i0, i1, w0, w1 = _flux_temporal_weights(StepwiseFlux(), times,
                                            (24 + 4.5) * 3600, 450.0)
    @test i0 == 10 && w0 == 1
end

@testset "runner wiring tripwire: start_time is passed at both call sites" begin
    # Codex should-fix: the unit tests above cannot catch deletion of the
    # runner wiring. Until a 2-binary end-to-end fixture exists, pin the two
    # call sites the 31-day verification campaign validated empirically.
    src = read(normpath(joinpath(@__DIR__, "..", "..", "src", "Models",
                                 "DrivenRunner.jl")), String)
    @test occursin("start_time = run_time_seconds", src)        # LL/RG loop
    @test occursin("start_time = total_hour * 3600.0", src)     # CS day loop
    @test occursin("run_time_seconds += (stop_window - start_window + 1)", src)
end

println("test_multiday_source_clock.jl OK")
