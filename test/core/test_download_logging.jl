#!/usr/bin/env julia

using Dates
using Logging
using Test
using AtmosTransport

const DD = AtmosTransport.DataDownloads
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

function download_fixture(dir; logging=nothing)
    cfg = Dict{String,Any}(
        "source" => Dict("met_source" => joinpath(REPO_ROOT, "config", "met_sources",
                                                   "geosit.toml")),
        "download" => Dict{String,Any}(
            "protocol" => "s3",
            "bucket" => "fixture-bucket",
            "prefix" => "GEOS_C180/GEOS_IT",
            "no_sign_request" => true,
            "requests" => Any[Dict(
                "name" => "transport",
                "description" => "logging fixture",
                "collections" => ["CTM_A1", "A1", "A3dyn"],
            )],
        ),
        "output" => Dict(
            "data_root" => dir,
            "met_source" => "geosit",
            "grid_name" => "C180",
            "cadence" => "daily",
            "payload_type" => "raw",
            "subdirectory_by_date" => true,
        ),
        "schedule" => Dict(
            "start_date" => "2021-12-01",
            "end_date" => "2021-12-01",
            "chunk" => "per_file",
        ),
        "options" => Dict("skip_existing" => true),
    )
    logging === nothing || (cfg["logging"] = logging)
    return cfg
end

function http_verify_fixture(dir, level)
    return Dict{String,Any}(
        "source" => Dict("met_source" => joinpath(REPO_ROOT, "config", "met_sources",
                                                   "geosfp.toml")),
        "download" => Dict{String,Any}(
            "protocol" => "http",
            "base_url" => "not a URL",
            "product" => "geosfp_025",
            "requests" => Any[Dict("name" => "surface", "collections" => ["A1"])],
        ),
        "output" => Dict(
            "data_root" => dir,
            "met_source" => "geosfp",
            "grid_name" => "0.25x0.3125",
            "cadence" => "daily",
            "payload_type" => "raw",
            "subdirectory_by_date" => true,
        ),
        "schedule" => Dict(
            "start_date" => "2021-12-01",
            "end_date" => "2021-12-01",
            "chunk" => "per_file",
        ),
        "logging" => Dict("level" => level, "progress" => "lines"),
    )
end

function captured_download(cfg; kwargs...)
    logs = nothing
    value = nothing
    stdout = mktemp() do _, io
        redirect_stdout(io) do
            logs, value = Test.collect_test_logs(min_level=Logging.Debug) do
                DD.download_data!(cfg; kwargs...)
            end
        end
        flush(io)
        seekstart(io)
        read(io, String)
    end
    return stdout, logs, value
end

@testset "download logging configuration and progress" begin
    @test DD.download_logging(Dict{String,Any}()).progress isa DD.DownloadProgressLines
    @test DD.download_logging(Dict{String,Any}(
        "logging" => Dict("progress" => "none"))).progress isa DD.NoDownloadProgress
    @test DD.download_logging(Dict{String,Any}(
        "logging" => Dict("progress" => "bar"))).progress isa DD.DownloadProgressBar
    @test_throws ArgumentError DD.download_logging(Dict{String,Any}(
        "logging" => Dict("progress" => "dots")))
    @test_throws ArgumentError DD.download_logging(Dict{String,Any}(
        "logging" => Dict("level" => "verbose")))

    mktempdir() do dir
        stdout, logs, value = captured_download(download_fixture(dir); dry_run=true)
        @test isempty(stdout)
        @test value === nothing
        messages = string.(getproperty.(logs, :message))
        task_messages = filter(m -> startswith(strip(m), "→"), messages)
        @test basename.(first.(split.(task_messages, " (~"))) == [
            "GEOSIT.20211201.CTM_A1.C180.nc",
            "GEOSIT.20211201.A1.C180.nc",
            "GEOSIT.20211201.A3dyn.C180.nc",
        ]
    end

    mktempdir() do dir
        cfg = download_fixture(dir; logging=Dict("level" => "info", "progress" => "none"))
        stdout, logs, _ = captured_download(cfg; dry_run=true)
        messages = string.(getproperty.(logs, :message))
        @test isempty(stdout)
        @test messages == ["AtmosTransport download started",
                           "AtmosTransport download finished"]

        cfg["logging"]["progress"] = "lines"
        _, line_logs, _ = captured_download(cfg; dry_run=true)
        line_messages = string.(getproperty.(line_logs, :message))
        @test any(startswith("[1/1]"), line_messages)
        @test count(m -> startswith(strip(m), "→"), line_messages) == 3
    end
end

@testset "download info, warning, and debug filtering" begin
    mktempdir() do dir
        cfg = http_verify_fixture(dir, "debug")
        parsed = DD.parse_download_config(cfg)
        task = only(DD.build_tasks(parsed.source, parsed.protocol, [Date(2021, 12, 1)],
                                   parsed.output, parsed.requests))
        mkpath(dirname(task.dest_path))
        write(task.dest_path, "existing fixture")

        stdout, debug_logs, _ = captured_download(cfg; verify_only=true)
        @test isempty(stdout)
        @test Logging.Debug in getproperty.(debug_logs, :level)
        @test Logging.Info in getproperty.(debug_logs, :level)
        @test Logging.Warn in getproperty.(debug_logs, :level)
        @test any(contains("HEAD request failed"), string.(getproperty.(debug_logs, :message)))

        cfg["logging"]["level"] = "info"
        _, info_logs, _ = captured_download(cfg; verify_only=true)
        @test Logging.Debug ∉ getproperty.(info_logs, :level)
        @test Logging.Info in getproperty.(info_logs, :level)
        @test Logging.Warn in getproperty.(info_logs, :level)

        cfg["logging"]["level"] = "warn"
        _, warn_logs, _ = captured_download(cfg; verify_only=true)
        @test !isempty(warn_logs)
        @test all(log -> log.level == Logging.Warn, warn_logs)

        cfg["logging"]["level"] = "error"
        _, error_logs, _ = captured_download(cfg; verify_only=true)
        @test isempty(error_logs)
    end
end

@testset "Python subprocess output is reported, not printed" begin
    python = something(Sys.which("python3"), "python3")
    env = DD.PythonEnvironment(python, false, false, false, false, false, false)
    reporter = DD.DownloadProgressLines()
    logs = nothing
    stdout, stderr = mktemp() do _, stdout_io
        mktemp() do _, stderr_io
            redirect_stdout(stdout_io) do
                redirect_stderr(stderr_io) do
                    logs, _ = Test.collect_test_logs() do
                        DD.run_python("import sys\nprint('python stdout')\nprint('python stderr', file=sys.stderr)\n",
                                      env; reporter)
                    end
                end
            end
            flush(stdout_io)
            flush(stderr_io)
            seekstart(stdout_io)
            seekstart(stderr_io)
            (read(stdout_io, String), read(stderr_io, String))
        end
    end
    @test isempty(stdout)
    @test isempty(stderr)
    @test string.(getproperty.(logs, :message)) == ["python stdout", "python stderr"]
end
