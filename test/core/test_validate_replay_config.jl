# `[input] validate_replay = true` replays every binary's continuity when the
# run opens it (the removed environment variables no longer act).
using Test
using AtmosTransport
using Logging
include(joinpath(@__DIR__, "..", "fixtures", "cs_multifile.jl"))
using .CSDriverHandoffFixtures

const ReplayMD = AtmosTransport.MetDrivers

# A two-window cubed-sphere binary whose air mass grows by 10 % with zero
# fluxes: the stored fluxes cannot carry window 1's mass to window 2.
function write_inconsistent_cs_binary(path; Nc = 3, Nz = 2)
    vc = HybridSigmaPressure([0.0, 5000.0, 0.0], [0.0, 0.5, 1.0])
    writer = ReplayMD.open_streaming_cs_transport_binary(
        path, Nc, 6, Nz, 2, vc; planet_radius = AtmosTransport.Parameters.EARTH_RADIUS,
        FT = Float64, mass_basis = :dry,
        surface_pressure_basis = TotalSurfacePressure(),
        dt_met_seconds = 3600.0, steps_per_window = 2)
    try
        for scale in (1.0, 1.1)
            window = (m  = ntuple(_ -> fill(1e16 * scale, Nc, Nc, Nz), 6),
                      am = ntuple(_ -> zeros(Nc + 1, Nc, Nz), 6),
                      bm = ntuple(_ -> zeros(Nc, Nc + 1, Nz), 6),
                      cm = ntuple(_ -> zeros(Nc, Nc, Nz + 1), 6),
                      ps = ntuple(_ -> fill(1e5, Nc, Nc), 6))
            ReplayMD.write_streaming_cs_window!(writer, window, Nc, 6)
        end
    finally
        ReplayMD.close_streaming_transport_binary!(writer)
    end
    return path
end

# The same for a lat-lon binary (structured runner).
function write_inconsistent_ll_binary(path; Nx = 4, Ny = 3, Nz = 2)
    grid = AtmosGrid(LatLonMesh(; FT = Float64, Nx, Ny),
                     HybridSigmaPressure([0.0, 5000.0, 0.0], [0.0, 0.5, 1.0]), CPU(); FT = Float64)
    windows = [(m = fill(1e16 * scale, Nx, Ny, Nz), am = zeros(Nx + 1, Ny, Nz),
                bm = zeros(Nx, Ny + 1, Nz), cm = zeros(Nx, Ny, Nz + 1), ps = fill(1e5, Nx, Ny))
               for scale in (1.0, 1.1)]
    write_transport_binary(path, grid, windows; FT = Float64, dt_met_seconds = 3600.0,
                           half_dt_seconds = 1800.0, steps_per_window = 2, mass_basis = :dry,
                           surface_pressure_basis = TotalSurfacePressure(),
                           source_flux_sampling = :window_start_endpoint,
                           flux_sampling = :window_constant)
    return path
end

# A two-window lat-lon binary whose window-1 fluxes carry window 1's air mass
# to window 2 exactly in Float64 (cm from the explicit-dm closure), stored as
# `FT`: only storage rounding is left in the replay residual.
function write_consistent_ll_binary(path; FT = Float32, Nx = 4, Ny = 3, Nz = 2, steps = 2)
    grid = AtmosGrid(LatLonMesh(; FT = Float64, Nx, Ny),
                     HybridSigmaPressure([0.0, 5000.0, 0.0], [0.0, 0.5, 1.0]), CPU(); FT = Float64)
    m1 = [1e16 * (1 + 0.1 * sinpi(i / Nx) + 0.05 * j + 0.02 * k) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    am = [3e12 * cospi(0.7i + 0.3j + 0.5k) for i in 1:Nx + 1, j in 1:Ny, k in 1:Nz]
    am[Nx + 1, :, :] .= am[1, :, :]                      # periodic in longitude
    bm = [2e12 * sinpi(0.4i + 0.6j + 0.2k) for i in 1:Nx, j in 1:Ny + 1, k in 1:Nz]
    bm[:, 1, :] .= 0; bm[:, Ny + 1, :] .= 0               # no flux through the poles
    dm = [1e11 * cospi(0.9i + 0.4j) * (k == 1 ? 1 : -1) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    cm = zeros(Nx, Ny, Nz + 1)
    ReplayMD.recompute_cm_from_dm_target!(cm, am, bm, m1, dm)
    m2 = similar(m1)                                         # the endpoint the fluxes reach
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        m2[i, j, k] = m1[i, j, k] - 2steps * ((am[i + 1, j, k] - am[i, j, k]) +
                      (bm[i, j + 1, k] - bm[i, j, k]) + (cm[i, j, k + 1] - cm[i, j, k]))
    end
    windows = [(m = m1, am = am, bm = bm, cm = cm, ps = fill(1e5, Nx, Ny)),
               (m = m2, am = zeros(Nx + 1, Ny, Nz), bm = zeros(Nx, Ny + 1, Nz),
                cm = zeros(Nx, Ny, Nz + 1), ps = fill(1e5, Nx, Ny))]
    write_transport_binary(path, grid, windows; FT, dt_met_seconds = 3600.0,
                           half_dt_seconds = 1800.0, steps_per_window = steps, mass_basis = :dry,
                           surface_pressure_basis = TotalSurfacePressure(),
                           source_flux_sampling = :window_start_endpoint,
                           flux_sampling = :window_constant)
    return path
end

run_quietly(cfg) = with_logger(NullLogger()) do
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            AtmosTransport.Models.run_driven_simulation(cfg)
        end
    end
end

@testset "[input] validate_replay" begin
    # The defaults are tested without the caller's replay variables.
    withenv("ATMOSTR_REPLAY_CHECK" => nothing, "ATMOSTR_NO_REPLAY_CHECK" => nothing) do
    mktempdir() do dir
        path = write_inconsistent_cs_binary(joinpath(dir, "inconsistent.bin"))
        config(input) = Dict{String, Any}(
            "input" => merge(Dict{String, Any}("binary_paths" => [path]), input),
            "architecture" => Dict("use_gpu" => false),
            "numerics" => Dict("float_type" => "Float64"),
            "advection" => Dict("scheme" => "upwind"),
            "tracers" => Dict("co2" => Dict("init" => Dict("kind" => "uniform",
                                                           "background" => 4e-4))))
        # Off by default: the run does not replay the binary.
        @test run_quietly(config(Dict{String, Any}())) !== nothing
        err = try
            run_quietly(config(Dict{String, Any}("validate_replay" => true))); nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("replay-consistency gate FAILED", err.msg)
        @test occursin("validate_replay", err.msg)
        # The key must be a Boolean.
        ok, errors = with_logger(NullLogger()) do
            validate_config(config(Dict{String, Any}("validate_replay" => "yes")))
        end
        @test !ok && any(contains("[input].validate_replay"), errors)
        # The removed environment variable no longer enables the check.
        withenv("ATMOSTR_REPLAY_CHECK" => "1") do
            driver = TransportBinaryDriver(path; FT = Float64, arch = CPU(), Hp = 1)
            try
                @test driver isa TransportBinaryDriver
            finally
                close(driver)
            end
        end
        # The removed ATMOSTR_NO_REPLAY_CHECK no longer bypasses a requested check.
        withenv("ATMOSTR_NO_REPLAY_CHECK" => "1") do
            @test_throws ArgumentError TransportBinaryDriver(path; FT = Float64, arch = CPU(), Hp = 1,
                                                             validate_replay = true)
        end
        # Every binary of a multi-file run is checked, not only the first.
        good = joinpath(dir, "consistent.bin")
        cs_handoff_fixture(good, [1.0, 1.0]; FT = Float64)
        two = config(Dict{String, Any}("validate_replay" => true))
        two["input"]["binary_paths"] = [good, path]
        err = try run_quietly(two); nothing catch e; e end
        @test err isa ArgumentError && occursin("inconsistent.bin", err.msg)

        # The structured (lat-lon) runner passes the key too.
        ll = write_inconsistent_ll_binary(joinpath(dir, "inconsistent_ll.bin"))
        ll_cfg = config(Dict{String, Any}("validate_replay" => true))
        ll_cfg["input"]["binary_paths"] = [ll]
        err = try run_quietly(ll_cfg); nothing catch e; e end
        @test err isa ArgumentError && occursin("replay-consistency gate FAILED", err.msg)
        ll_cfg["input"]["validate_replay"] = false
        @test run_quietly(ll_cfg) !== nothing

        # A Float32 binary opened in Float64 is checked at the Float32 tolerance:
        # the loaded values carry Float32 rounding whatever the run's float type.
        f32 = write_consistent_ll_binary(joinpath(dir, "consistent_f32.bin"); FT = Float32)
        reader = ReplayMD.TransportBinaryReader(f32; FT = Float64)
        m1, _, fl = ReplayMD.load_window!(reader, 1)
        m2, _, _ = ReplayMD.load_window!(reader, 2)
        rounding = ReplayMD.verify_window_continuity_ll(m1, fl.am, fl.bm, fl.cm, m2, 2).max_rel_err
        close(reader)
        @test ReplayMD.replay_tolerance(Float64) < rounding < ReplayMD.replay_tolerance(Float32)
        for FT in (Float64, Float32)
            driver = with_logger(NullLogger()) do
                TransportBinaryDriver(f32; FT, arch = CPU(), validate_replay = true)
            end
            @test driver isa TransportBinaryDriver
            close(driver)
        end
    end
end
end
