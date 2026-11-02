# Run-config checks before any binary is opened: unknown keys (errors with
# "did you mean" suggestions) and unread keys (warnings), physics/run/output
# settings parsed by `validate_config`, and the snapshot-schedule checks at setup.
using Test
using AtmosTransport
using Logging

const Runner = AtmosTransport.Models.DrivenRunner

@testset "float_type: one strict parser on the runtime and both preprocessing paths" begin
    numerics(raw) = Dict{String, Any}("numerics" => Dict{String, Any}("float_type" => raw))
    resolvers = ((Runner._cfg_float_type, Float64),
                 (AtmosTransport.Preprocessing.resolve_output_float_type, Float32),
                 (AtmosTransport.Preprocessing._resolve_float_type, Float64))
    for (resolve, default) in resolvers
        @test resolve(Dict{String, Any}("numerics" => Dict{String, Any}())) === default
        @test resolve(numerics("Float32")) === Float32
        @test resolve(numerics("Float64")) === Float64
        @test resolve(numerics("float64")) === Float64
        for bad in ("Float16", "Float32 ", "double", "F64", 64)
            err = try resolve(numerics(bad)); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("\"Float32\" or \"Float64\"", err.msg)
        end
    end
end

@testset "unknown and unread keys" begin
    f = Runner._config_key_findings(Dict(
        "diffussion" => Dict("kind" => "constant"),
        "advection" => Dict("scheme" => "upwind", "ppm_order" => 7)))
    @test !isempty(f.errors)
    @test isempty(Runner._config_key_findings(Dict("advection" => Dict("scheme" => "upwind"))).errors)
    @test !isempty(f.warnings)
end

@testset "validate_config rejects invalid config classes before opening binaries" begin
    mktemp() do path, io
        base = Dict{String,Any}("input" => Dict("binary_paths" => [path]))
        @test first(validate_config(base))
        for patch in (Dict("diffussion" => Dict("kind" => "constant")),
                      Dict("run" => Dict("scheme" => "linrood")),
                      Dict("diffusion" => Dict("kind" => "unknown")),
                      Dict("run" => 7))
            cfg = merge(base, patch)
            ok, errors = with_logger(() -> validate_config(cfg), NullLogger())
            @test !ok && !isempty(errors)
            @test_throws ArgumentError run_driven_simulation(cfg)
            @test filesize(path) == 0
        end
        cfg = merge(base, Dict("diffusion" => Dict("kind" => "none", "value" => 1.0)))
        logs, (ok, errors) = Test.collect_test_logs() do
            validate_config(cfg)
        end
        @test ok && isempty(errors)
        @test any(l -> l.level == Logging.Warn, logs)
    end
end

@testset "snapshot schedule at setup" begin
    # Matching: within half a window and at most half an hour of a window end.
    @test Runner._snapshot_due(3.0, 3.0, 1.0) && !Runner._snapshot_due(2.0, 3.0, 1.0)
    @test Runner._snapshot_due(10 / 60, 10 / 60, 10 / 60)
    @test !Runner._snapshot_due(10 / 60, 20 / 60, 10 / 60)   # 10-minute windows
    @test !Runner._snapshot_due(0.0, 10 / 60, 10 / 60)
    @test Runner._check_window_range([(3600.0, 4), (1800.0, 6)];
                                     stop_window_override = 4) === nothing
    @test_throws ArgumentError Runner._check_window_range(
        [(3600.0, 4), (1800.0, 3)]; stop_window_override = 4)
    @test_throws ArgumentError Runner._check_window_range(
        [(3600.0, 2)]; start_window = 3)
end
