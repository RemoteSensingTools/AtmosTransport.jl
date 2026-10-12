#!/usr/bin/env julia

using Test
using JSON3
using TOML
using Dates
using NCDatasets
using AtmosTransport

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

function _script_module(name::Symbol, relative_path::AbstractString)
    mod = Module(name)
    Core.eval(mod, :(include(path::AbstractString) = Base.include($mod, path)))
    Core.eval(mod, :(eval(expr) = Core.eval($mod, expr)))
    Base.include(mod, joinpath(REPO_ROOT, relative_path))
    return mod
end

function _write_mixed_window_binary(path, window_seconds)
    FT = Float64
    Nx, Ny, Nz = 4, 3, 2
    mesh = LatLonMesh(; FT, Nx, Ny)
    vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vertical, CPU(); FT)
    windows = [(
        m = ones(FT, Nx, Ny, Nz),
        am = zeros(FT, Nx + 1, Ny, Nz),
        bm = zeros(FT, Nx, Ny + 1, Nz),
        cm = zeros(FT, Nx, Ny, Nz + 1),
        ps = fill(FT(95_000), Nx, Ny),
    ) for _ in 1:2]
    write_transport_binary(path, grid, windows;
                           FT, dt_met_seconds = window_seconds,
                           half_dt_seconds = window_seconds / 2,
                           steps_per_window = 1, mass_basis = :dry,
                           surface_pressure_basis = TotalSurfacePressure(),
                           source_flux_sampling = :window_start_endpoint,
                           flux_sampling = :window_constant)
    return path
end

function _write_full_window_cs_binary(path)
    MD = AtmosTransport.MetDrivers
    FT = Float64
    Nc, Nz = 2, 2
    vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    writer = MD.open_streaming_cs_transport_binary(
        path, Nc, 6, Nz, 2, vertical;
        FT, dt_met_seconds = 3600.0, steps_per_window = 2,
        flux_kind = :full_window_mass_amount, mass_basis = :dry,
        surface_pressure_basis = TotalSurfacePressure(),
        planet_radius = AtmosTransport.Parameters.EARTH_RADIUS)
    try
        for win in 1:2
            m = ntuple(_ -> begin
                values = ones(FT, Nc, Nc, Nz)
                win == 2 && (values[:, :, Nz] .= 0.9)
                values
            end, 6)
            cm = ntuple(_ -> begin
                values = zeros(FT, Nc, Nc, Nz + 1)
                win == 1 && (values[:, :, Nz + 1] .= 0.1)
                values
            end, 6)
            window = (
                m,
                am = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), 6),
                bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), 6),
                cm,
                ps = ntuple(_ -> fill(FT(95_000), Nc, Nc), 6),
            )
            MD.write_streaming_cs_window!(writer, window, Nc, 6)
        end
    finally
        MD.close_streaming_transport_binary!(writer)
    end
    return path
end

@testset "runtime preflight reports malformed tables before reading binaries" begin
    mktemp() do path, io
        # An existing empty file passes path checks, but is not a transport
        # binary. Preflight must inspect configuration without opening it.
        base = Dict{String,Any}("input" => Dict("binary_paths" => [path]))
        cases = (
            ("run = 7", "[run]"),
            ("numerics = false", "[numerics]"),
            ("architecture = \"cpu\"", "[architecture]"),
            ("advection = \"ppm\"", "[advection]"),
            ("[tracers]\nco2 = 0.0004", "[tracers.co2]"),
            ("[tracers.co2]\ninit = \"uniform\"", "[tracers.co2.init]"),
            ("[tracers.co2]\nsurface_flux = false", "[tracers.co2.surface_flux]"),
        )
        for (text, label) in cases
            cfg = merge(base, TOML.parse(text))
            before = deepcopy(cfg)
            ok, errors = validate_config(cfg)
            @test !ok
            @test any(error -> occursin(label, error) && occursin("TOML table", error), errors)
            @test cfg == before
        end

        cfg = merge(base, TOML.parse("run = 7\n[tracers]\nco2 = 0.0004"))
        ok, errors = validate_config(cfg)
        @test !ok
        @test length(errors) == 2
        @test any(error -> occursin("[run]", error), errors)
        @test any(error -> occursin("[tracers.co2]", error), errors)

        # The public CPU runner must report the configuration error rather
        # than attempting to read the invalid binary header.
        error = try
            run_driven_simulation(cfg)
            nothing
        catch err
            err
        end
        @test error isa ArgumentError
        @test occursin("Invalid AtmosTransport run config", sprint(showerror, error))
        @test occursin("[tracers.co2]", sprint(showerror, error))
        @test filesize(path) == 0
        @test_throws "[architecture] must be a TOML table" run_driven_simulation(
            merge(base, Dict("architecture" => "cpu")))

        for removed in ("host", "nvidia", "apple", "apple_metal", "gpu")
            cfg = merge(base, Dict("architecture" => Dict("backend" => removed)))
            ok, errors = validate_config(cfg)
            @test !ok
            @test any(error -> occursin("spelling \"$(removed)\" was removed", error),
                      errors)
        end

        # Nested input validation must also work when no tracers are supplied.
        for staging in (false, "nvme", 7, ["cache"])
            cfg = Dict("input" => Dict("binary_paths" => [path], "staging" => staging))
            before = deepcopy(cfg)
            ok, errors = validate_config(cfg)
            @test !ok
            @test length(errors) == 1
            @test occursin("[input.staging] must be a TOML table", only(errors))
            @test_throws "[input.staging] must be a TOML table" run_driven_simulation(cfg)
            @test cfg == before
            @test filesize(path) == 0
        end
        @test validate_config(Dict("input" => Dict("binary_paths" => [path],
            "staging" => Dict("enabled" => false)))) == (true, String[])
    end
end

@testset "runtime window indices follow the integer TOML contract" begin
    mktemp() do path, io
        base = Dict{String,Any}("input" => Dict("binary_paths" => [path]))
        for key in ("start_window", "stop_window"), value in (true, false, 1.0, 1.5, "1")
            cfg = merge(base, Dict("run" => Dict(key => value)))
            ok, errors = validate_config(cfg)
            @test !ok
            @test any(error -> occursin(key, error) && occursin("must be an integer", error), errors)
        end
        for run in (Dict("start_window" => 0),
                    Dict("start_window" => 3, "stop_window" => 2))
            ok, errors = validate_config(merge(base, Dict("run" => run)))
            @test !ok
            @test any(error -> occursin("must be >=", error), errors)
        end
        for run in (Dict{String,Any}(), Dict("start_window" => 1, "stop_window" => 2),
                    Dict("start_window" => Int32(2), "stop_window" => nothing))
            @test validate_config(merge(base, Dict("run" => run))) == (true, String[])
        end
        @test validate_config(merge(base, TOML.parse("""
            [tracers.anomaly.init]
            kind = "uniform"
            background = -0.000001
            """))) == (true, String[])
    end
end

@testset "documented synthetic quickstart runs end to end" begin
    generator_module = _script_module(
        :SyntheticQuickstartTest,
        "examples/generate_synthetic_quickstart.jl",
    )
    generate = getfield(generator_module, :generate_synthetic_quickstart)

    mktempdir() do dir
        binary_path = generate(joinpath(dir, "synthetic_latlon_v4.bin"))
        @test isfile(binary_path)

        reader = TransportBinaryReader(binary_path; FT = Float32)
        try
            @test grid_type(reader) == :latlon
            @test reader.header.nwindow == 4
        finally
            close(reader)
        end

        cfg = TOML.parsefile(joinpath(
            REPO_ROOT,
            "config",
            "examples",
            "minimal_template.toml",
        ))
        output_path = joinpath(dir, "synthetic_output.nc")
        cfg["input"]["binary_paths"] = [binary_path]
        cfg["output"]["path"] = output_path

        ok, errors = validate_config(cfg)
        @test ok
        @test isempty(errors)
        run_driven_simulation(cfg)
        @test isfile(output_path)

        NCDataset(output_path) do ds
            column_mean = ds["co2_bl_column_mean"][:, :, :]
            @test size(column_mean) == (36, 18, 5)
            @test all(isfinite, column_mean)

            # The documented output schedule is part of the public example.
            @test ds["time"][:] == collect(0.0:4.0)

            # Constant periodic mass flux should move the Gaussian east while
            # conserving both air mass and the model's mass-like tracer
            # storage. Multiplying column storage per area by cell area
            # recovers the domain-integrated conservative storage quantity.
            @test maximum(abs.(
                @view(column_mean[:, :, end]) .-
                @view(column_mean[:, :, 1]))) > 1f-6

            air_mass = ds["air_mass"][:, :, :, :]
            air_totals = [sum(@view air_mass[:, :, :, t])
                          for t in axes(air_mass, 4)]
            @test all(x -> isapprox(x, air_totals[1]; rtol = 2f-6), air_totals)

            cell_area = ds["cell_area"][:, :]
            storage_per_area = ds["co2_bl_column_mass_per_area"][:, :, :]
            storage_totals = [sum(@view(storage_per_area[:, :, t]) .* cell_area)
                              for t in axes(storage_per_area, 3)]
            @test all(x -> isapprox(x, storage_totals[1]; rtol = 2f-6), storage_totals)
        end
    end
end

@testset "runner handles mixed binary window lengths end to end" begin
    mktempdir() do dir
        half_hour = _write_mixed_window_binary(joinpath(dir, "half_hour.bin"), 1800.0)
        hourly = _write_mixed_window_binary(joinpath(dir, "hourly.bin"), 3600.0)
        output_path = joinpath(dir, "mixed_windows.nc")
        cfg = Dict{String, Any}(
            "input" => Dict("binary_paths" => [half_hour, hourly]),
            "advection" => Dict("scheme" => "upwind"),
            "tracers" => Dict("co2" => Dict(
                "init" => Dict("kind" => "uniform", "background" => 4e-4))),
            "output" => Dict("path" => output_path,
                             "hours" => [0.0, 0.5, 1.0, 2.0, 3.0]),
        )
        model = run_driven_simulation(cfg)
        @test all(mixing_ratio(model.state, :co2) .== 4e-4)
        NCDataset(output_path) do ds
            @test ds["time"][:] == [0.0, 0.5, 1.0, 2.0, 3.0]
        end
    end
end

@testset "CS continuity verifier scales full-window flux amounts" begin
    verifier = _script_module(
        :VerifyCSBinaryContinuityTest,
        "scripts/validation/verify_cs_binary_continuity.jl",
    )
    mktempdir() do dir
        binary = _write_full_window_cs_binary(joinpath(dir, "full_window.bin"))
        @test getfield(verifier, :main)(["--binary", binary, "--threshold", "1e-14"]) == 0
    end
end

tryparse_toml(path) = try TOML.parsefile(path) catch; nothing end

@testset "runtime editor schema follows public config contract" begin
    schema_path = joinpath(REPO_ROOT, "schemas", "atmos_transport_run.schema.json")
    schema = JSON3.read(read(schema_path, String))
    @test schema[Symbol("\$schema")] == "http://json-schema.org/draft-07/schema#"

    properties = schema.properties
    @test hasproperty(properties, :advection)
    # The advection keys moved from [run] to [advection]; [run] lists them only
    # to reject any value, with a pointer to [advection].
    @test haskey(properties.run.properties.scheme, :not)

    # Every choice and key is compared with the parsers in test_config_schema_agreement.jl.

    output = schema.definitions.output.properties
    for key in (:enabled, :format, :path, :hours, :cadence_hours, :split,
                :deflate_level, :shuffle, :fields, :observations)
        @test hasproperty(output, key)
    end
    @test schema.definitions.observation_output.additionalProperties == false
    @test schema.definitions.observation_source.additionalProperties == false
    # Choices are documented `oneOf` consts; test_observation_schema.jl checks them in full.
    @test Set(String(c.const) for c in schema.definitions.observation_source.properties.kind.oneOf) ==
          Set(["oco2_lite", "obspack", "table"])
    for retired in (:mode, :path_template, :frequency, :provenance)
        @test !hasproperty(output, retired)
    end

    # The advection keys moved from [run]: the schema rejects any value there,
    # and no TOML in the repository still sets them under [run].
    for key in (:scheme, :ppm_order, :vertical, :limiter)
        @test isempty(properties.run.properties[key].not)
    end
    legacy = String[]
    for (dir, _, files) in walkdir(REPO_ROOT), f in files
        endswith(f, ".toml") && !occursin("/.git", dir) || continue
        run = get(something(tryparse_toml(joinpath(dir, f)), Dict()), "run", nothing)
        run isa AbstractDict && any(k -> haskey(run, k), ("scheme", "ppm_order", "vertical", "limiter")) &&
            push!(legacy, relpath(joinpath(dir, f), REPO_ROOT))
    end
    @test isempty(legacy)

    for example in ("atmos_transport_schema_demo.toml", "minimal_template.toml")
        cfg = TOML.parsefile(joinpath(REPO_ROOT, "config", "examples", example))
        @test haskey(cfg, "advection")
        @test !haskey(get(cfg, "run", Dict{String, Any}()), "scheme")
        @test haskey(cfg["output"], "path")
        @test haskey(cfg["output"], "hours")
    end
end

@testset "canonical command-line parsers fail closed" begin
    cfg = joinpath(REPO_ROOT, "config", "examples", "minimal_template.toml")

    run_cli = _script_module(:RunTransportCLITest, "scripts/run_transport.jl")
    run_parse = getfield(run_cli, :_parse_cli)
    @test run_parse(["--help"]) === nothing
    @test run_parse([cfg]) == cfg
    @test_throws ArgumentError run_parse(String[])
    @test_throws ArgumentError run_parse([cfg, "unexpected"])
    @test_throws ArgumentError run_parse(["--unknown"])

    # ATMOSTR_PROFILE_MODE follows the config's [architecture], not whether
    # CUDA is loaded yet: the first GPU run of a session that included the
    # script (no preload) is profiled like any later one.
    configured = getfield(run_cli, :_configured_backend)
    action = (mode, backend) -> Base.invokelatest(getfield(run_cli, :_profile_action), mode, backend)
    gpu_cfg = Dict{String, Any}("architecture" => Dict{String, Any}("use_gpu" => true, "backend" => "cuda"))
    @test Base.invokelatest(configured, gpu_cfg) === :cuda          # resolved without loading CUDA
    @test Base.invokelatest(configured, Dict{String, Any}()) === :cpu
    @test Base.invokelatest(configured, Dict{String, Any}("architecture" => Dict{String, Any}("backend" => "metal"))) === :metal
    @test action("full", :cuda) === :full
    @test action("Window", :cuda) === :window
    @test action("", :cuda) === :none
    @test action("full", :cpu) === :needs_cuda
    @test action("window", :metal) === :needs_cuda
    @test action("sampling", :cuda) === :unknown

    preprocess_cli = _script_module(
        :PreprocessTransportCLITest,
        "scripts/preprocessing/preprocess_transport_binary.jl",
    )
    preprocess_parse = getfield(preprocess_cli, :_parse_cli)
    @test preprocess_parse(["--help"]) === nothing
    @test preprocess_parse(["-h"]) === nothing
    @test preprocess_parse([cfg, "--day", "2021-12-01"])[2] == "2021-12-01"
    @test preprocess_parse([
        cfg, "--start", "2021-12-01", "--end", "2021-12-03",
    ])[3:4] == ("2021-12-01", "2021-12-03")
    @test_throws ArgumentError preprocess_parse([cfg, "--unknown"])
    @test_throws ArgumentError preprocess_parse([cfg, "--day"])
    @test_throws ArgumentError preprocess_parse([
        cfg, "--day", "2021-12-01", "--start", "2021-12-01", "--end", "2021-12-03",
    ])
    @test_throws ArgumentError preprocess_parse([cfg, "--start", "2021-12-03"])
    @test_throws ArgumentError preprocess_parse([
        cfg, "--start", "2021-12-03", "--end", "2021-12-01",
    ])

    download_cli = _script_module(:DownloadDataCLITest, "scripts/downloads/download_data.jl")
    download_parse = getfield(download_cli, :_parse_cli)
    opts = download_parse([
        cfg, "--start", "2021-12-01", "--end", "2021-12-03", "--dry-run",
    ])
    @test opts.start_date == Date(2021, 12, 1)
    @test opts.end_date == Date(2021, 12, 3)
    @test opts.dry_run
    @test !opts.verify
    @test_throws ArgumentError download_parse([cfg, "--unknown"])
    @test_throws ArgumentError download_parse([cfg, "--start"])
    @test_throws ArgumentError download_parse([cfg, "--dry-run", "--dry-run"])
    @test_throws ArgumentError download_parse([
        cfg, "--start", "2021-12-03", "--end", "2021-12-01",
    ])
end

@testset "OPeNDAP recipes refuse to download before any task runs" begin
    DL = AtmosTransport.DataDownloads
    cfg = TOML.parsefile(joinpath(REPO_ROOT, "config", "downloads", "merra2.toml"))
    mktempdir() do dir
        cfg["output"]["data_root"] = dir
        err = try
            DL.download_data!(cfg; start_date = Date(2021, 12, 1), end_date = Date(2021, 12, 1))
        catch e
            e
        end
        @test err isa ArgumentError && occursin("--dry-run", err.msg)
        @test isempty(readdir(dir))                       # nothing created
        @test DL.download_data!(cfg; start_date = Date(2021, 12, 1), end_date = Date(2021, 12, 1),
                                dry_run = true) === nothing
    end
end

@testset "runtime preload reports malformed architecture settings" begin
    mktempdir() do dir
        for (name, contents, expected) in (
            ("invalid-gpu-flag", "[architecture]\nuse_gpu = 1\nbackend = \"cpu\"\n",
             "[architecture].use_gpu must be true or false; got 1"),
            ("invalid-architecture-table", "architecture = \"cpu\"\n",
             "[architecture] must be a TOML table"),
        )
            cfg = joinpath(dir, name * ".toml")
            write(cfg, contents)

            script = joinpath(REPO_ROOT, "scripts", "run_transport.jl")
            cmd = addenv(
                `$(Base.julia_cmd()) --project=$(joinpath(REPO_ROOT, "test")) $script $cfg`,
                "ATMOSTR_NO_AUTO_THREADS" => "1",
            )
            stdout_buffer = IOBuffer()
            stderr_buffer = IOBuffer()
            process = run(pipeline(ignorestatus(cmd);
                                   stdout=stdout_buffer, stderr=stderr_buffer))

            @test !success(process)
            @test occursin(expected, String(take!(stderr_buffer)))
            @test isempty(take!(stdout_buffer))
        end
    end
end
