# `[input] validate_replay = true` replays every binary's continuity when the
# run opens it; it replaces the environment variable ATMOSTR_REPLAY_CHECK.
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
        FT = Float64, mass_basis = :dry, dt_met_seconds = 3600.0, steps_per_window = 2)
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
        # The deprecated environment variable still enables the check, with a warning.
        withenv("ATMOSTR_REPLAY_CHECK" => "1") do
            @test_logs (:warn, r"ATMOSTR_REPLAY_CHECK=1 is deprecated") match_mode = :any begin
                @test_throws ArgumentError TransportBinaryDriver(path; FT = Float64, arch = CPU(), Hp = 1)
            end
        end
        driver = TransportBinaryDriver(path; FT = Float64, arch = CPU(), Hp = 1)
        try
            @test driver isa TransportBinaryDriver
        finally
            close(driver)
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
    end
end
end
