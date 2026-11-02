#!/usr/bin/env julia

using Test
using JSON3
using TOML
using Dates
using NCDatasets
using AtmosTransport

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

@testset "runtime preflight validates dates and halo values" begin
    mktemp() do path, io
        for date in ("2021-12-01", Date(2021, 12, 1))
            cfg = Dict("input" => Dict("binary_paths" => [path], "start_date" => date))
            @test first(validate_config(cfg))
            @test AtmosTransport.Models.DrivenRunner._run_reference_time(cfg) == DateTime(2021, 12, 1)
        end
        for date in ("2021-13-01", "invalid", true, 20211201)
            cfg = Dict("input" => Dict("binary_paths" => [path], "start_date" => date))
            @test !first(validate_config(cfg))
            @test_throws ArgumentError AtmosTransport.Models.DrivenRunner._run_reference_time(cfg)
        end
        for key in ("Hp", "halo_padding"), value in (true, "3", 2.5, 0, -1)
            cfg = Dict("input" => Dict("binary_paths" => [path]), "run" => Dict(key => value))
            @test !first(validate_config(cfg))
        end
        sf = Dict("kind" => "lmdz_co2", "month" => 12)
        cfg = Dict("input" => Dict("binary_paths" => [path]),
                   "tracers" => Dict("co2" => Dict("surface_flux" => sf)))
        @test !first(validate_config(cfg))
    end
end

function _script_module(name::Symbol, relative_path::AbstractString)
    mod = Module(name)
    Core.eval(mod, :(include(path::AbstractString) = Base.include($mod, path)))
    Core.eval(mod, :(eval(expr) = Core.eval($mod, expr)))
    Base.include(mod, joinpath(REPO_ROOT, relative_path))
    return mod
end

function _write_mixed_window_binary(path, window_seconds; extra_header = Dict{String, Any}())
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
                           flux_sampling = :window_constant, extra_header)
    return path
end

@testset "runtime output origin uses original path through staging" begin
    mktempdir() do dir
        original = _write_mixed_window_binary(
            joinpath(dir, "transport_20211201.bin"), 3600.0)
        stager = AtmosTransport.Models.InputStager([original], Dict{String, Any}(
            "enabled" => true, "dir" => joinpath(dir, "stage"),
            "lookahead_days" => 0, "cleanup_on_exit" => true))
        try
            staged = AtmosTransport.Models.staged_path_for!(stager, 1)
            @test basename(staged) != basename(original)
            driver = AtmosTransport.MetDrivers.TransportBinaryDriver(staged; FT = Float64)
            try
                DR = AtmosTransport.Models.DrivenRunner
                origin = DR._run_time_origin(Dict{String, Any}(), driver, original)
                @test origin == DateTime(2021, 12, 1)
                @test DR._runtime_output_origin(origin, driver) == DateTime(2021, 12, 1)
                @test DR._runtime_output_origin(origin, driver; start_window = 2) ==
                      DateTime(2021, 12, 1, 1)
            finally
                close(driver)
            end
        finally
            AtmosTransport.Models.cleanup_staging!(stager)
        end
    end
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

@testset "run origin: start_date, header date and file-name date must agree" begin
    DR = AtmosTransport.Models.DrivenRunner
    dated(day) = Dict{String, Any}("input" => Dict{String, Any}("start_date" => day))
    origin_error(cfg, driver, path) =
        try
            DR._run_time_origin(cfg, driver, path)
            ""
        catch err
            err isa ArgumentError || rethrow()
            sprint(showerror, err)
        end
    with_driver(f, path) = begin
        driver = AtmosTransport.MetDrivers.TransportBinaryDriver(path; FT = Float64)
        try
            f(driver)
        finally
            close(driver)
        end
    end
    mktempdir() do dir
        # Legacy binary: no `date` header, the day only in the file name. A
        # start_date one day later must not silently win over the file name.
        legacy = _write_mixed_window_binary(joinpath(dir, "transport_20211201.bin"), 3600.0)
        with_driver(legacy) do driver
            @test DR._run_time_origin(dated("2021-12-01"), driver, legacy) == DateTime(2021, 12, 1)
            msg = origin_error(dated("2021-12-02"), driver, legacy)
            @test occursin("[input].start_date = 2021-12-02 disagrees with the date in the " *
                           "first transport binary's file name transport_20211201.bin = 2021-12-01",
                           msg)
        end

        # Header and file name are compared even without start_date.
        headed = _write_mixed_window_binary(joinpath(dir, "transport_2021-12-01.bin"), 3600.0;
                                            extra_header = Dict{String, Any}("date" => "2021-12-02"))
        with_driver(headed) do driver
            msg = origin_error(Dict{String, Any}(), driver, headed)
            @test occursin("`date` header = 2021-12-02 disagrees with the date in the first " *
                           "transport binary's file name transport_2021-12-01.bin = 2021-12-01", msg)
            @test occursin("disagrees with the first transport binary's `date` header",
                           origin_error(dated("2021-12-01"), driver, headed))
        end

        # All three present and equal.
        agreeing = _write_mixed_window_binary(joinpath(dir, "transport_20211202.bin"), 3600.0;
                                              extra_header = Dict{String, Any}("date" => "2021-12-02"))
        with_driver(agreeing) do driver
            @test DR._run_time_origin(dated("2021-12-02"), driver, agreeing) == DateTime(2021, 12, 2)
        end

        # Undated: no origin; output then needs one, a run without output does not.
        undated = _write_mixed_window_binary(joinpath(dir, "transport.bin"), 3600.0)
        with_driver(undated) do driver
            @test DR._run_time_origin(Dict{String, Any}(), driver, undated) === nothing
            @test DR._run_time_origin(dated("2021-12-03"), driver, undated) == DateTime(2021, 12, 3)
            @test_throws ArgumentError DR._runtime_output_origin(nothing, driver)
        end

        # Only whole `YYYYMMDD` or `YYYY-MM-DD` tokens, delimited by non-digits, date a file.
        @test DR._filename_date("/x/era5_transport_20211201_merged1000Pa_float64.bin") ==
              Date(2021, 12, 1)
        @test DR._filename_date("transport_2021-12-01_v4.bin") == Date(2021, 12, 1)
        for undated_name in ("transport_2021-1201.bin", "transport_202112-01.bin",
                             "run_120211201.bin", "c90_2021.bin")
            @test DR._filename_date(undated_name) === nothing
        end

        # A delimited 8-digit token that is not a calendar date does not date the file.
        not_a_date = _write_mixed_window_binary(joinpath(dir, "transport_20211301.bin"), 3600.0)
        with_driver(not_a_date) do driver
            @test (@test_logs (:warn, r"not a calendar date") DR._run_time_origin(
                dated("2021-12-01"), driver, not_a_date)) == DateTime(2021, 12, 1)
        end

        # The runners check the origin whether or not they write output: a
        # mismatched start_date would otherwise shift time-varying fluxes.
        no_output = Dict{String, Any}(
            "input" => Dict{String, Any}("binary_paths" => [legacy], "start_date" => "2021-12-02"),
            "advection" => Dict{String, Any}("scheme" => "upwind"),
            "tracers" => Dict{String, Any}("co2" => Dict{String, Any}(
                "init" => Dict{String, Any}("kind" => "uniform", "background" => 4e-4))))
        err = try run_driven_simulation(no_output); nothing catch e; e end
        @test err isa ArgumentError && occursin("disagrees with the date in the first transport " *
                                                "binary's file name", sprint(showerror, err))
        cs_legacy = _write_full_window_cs_binary(joinpath(dir, "cs_20211201.bin"))
        cs_no_output = deepcopy(no_output)
        cs_no_output["input"]["binary_paths"] = [cs_legacy]
        err = try run_driven_simulation(cs_no_output); nothing catch e; e end
        @test err isa ArgumentError && occursin("disagrees with the date in the first transport " *
                                                "binary's file name cs_20211201.bin", sprint(showerror, err))
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
            @test ds["time"].var[:] == collect(0.0:4.0)
            @test ds["time"][:] == DateTime(2021, 12, 1) .+ Hour.(0:4)

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
    mktempdir() do dir
        half_hour = _write_mixed_window_binary(joinpath(dir, "half_hour.bin"), 1800.0)
        hourly = _write_mixed_window_binary(joinpath(dir, "hourly.bin"), 3600.0)
        output_path = joinpath(dir, "mixed_windows.nc")
        cfg = Dict{String, Any}(
            "input" => Dict("binary_paths" => [half_hour, hourly],
                            "start_date" => "2021-12-01"),
            "advection" => Dict("scheme" => "upwind"),
            "tracers" => Dict("co2" => Dict(
                "init" => Dict("kind" => "uniform", "background" => 4e-4))),
            "output" => Dict("path" => output_path,
                             "hours" => [0.0, 0.5, 1.0, 2.0, 3.0]),
        )
        model = run_driven_simulation(cfg)
        @test all(mixing_ratio(model.state, :co2) .== 4e-4)
        NCDataset(output_path) do ds
            @test ds["time"].var[:] == [0.0, 0.5, 1.0, 2.0, 3.0]
            @test ds["time"][:] == DateTime(2021, 12, 1) .+
                                   Millisecond.([0, 1_800_000, 3_600_000,
                                                7_200_000, 10_800_000])
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



@testset "runtime preload reports malformed architecture settings" begin
    mktempdir() do dir
        for (name, contents, expected) in (
            ("invalid-gpu-flag", "[architecture]\nuse_gpu = 1\nbackend = \"cpu\"\n",
             "[architecture].use_gpu must be true or false; got 1"),

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
