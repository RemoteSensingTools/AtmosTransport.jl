#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# test_binary_path_expander.jl
#
# Verifies the two accepted `[input]` TOML shapes for resolving transport
# binary paths: the existing explicit `binary_paths = [...]` list (Shape A)
# and the new `folder + start_date + end_date (+ file_pattern)` shape
# (Shape B). Shape B must verify continuity, sort by parsed date, reject
# mutual-exclusion, and produce precise error messages.
# ---------------------------------------------------------------------------

using Test

using AtmosTransport

using Dates

@testset "input dates accept TOML dates in both input modes" begin
    mktempdir() do dir
        path = joinpath(dir, "transport_20211201.bin")
        touch(path)
        explicit = Dict{String, Any}("binary_paths" => [path], "start_date" => Date(2021, 12, 1))
        @test expand_binary_paths(explicit) == [path]
        folder = Dict{String, Any}("folder" => dir, "start_date" => Date(2021, 12, 1),
                                   "end_date" => Date(2021, 12, 1))
        @test expand_binary_paths(folder) == [path]
        for invalid in (true, 20211201, "2021-13-01")
            @test_throws ArgumentError expand_binary_paths(merge(explicit, Dict("start_date" => invalid)))
            @test_throws ArgumentError expand_binary_paths(merge(folder, Dict("end_date" => invalid)))
        end
    end
end

"""Create a set of empty files in `dir` named per the supplied dates and
the `YYYYMMDD` prefix/suffix. Returns the absolute paths."""
function _touch_binary_set(dir::AbstractString, dates; prefix = "era5_", suffix = ".bin")
    paths = String[]
    for d in dates
        token = Dates.format(d, dateformat"yyyymmdd")
        p = joinpath(dir, prefix * token * suffix)
        open(p, "w") do io
            write(io, "")   # empty placeholder; expander never opens the file
        end
        push!(paths, p)
    end
    return paths
end

@testset "expand_binary_paths" begin

    @testset "Shape A: explicit list is returned verbatim" begin
        cfg = Dict("binary_paths" => [
            "/tmp/foo/a.bin",
            "/tmp/foo/b.bin",
        ])
        paths = expand_binary_paths(cfg)
        @test paths == ["/tmp/foo/a.bin", "/tmp/foo/b.bin"]
    end

    @testset "Shape A: tilde expansion" begin
        cfg = Dict("binary_paths" => ["~/a.bin"])
        paths = expand_binary_paths(cfg)
        @test startswith(paths[1], expanduser("~"))
    end

    @testset "Shape A: data-root expansion" begin
        withenv("ATMOSTRANSPORT_DATA_ROOT" => "/tmp/atmos_root") do
            @test expand_data_path(raw"$ATMOSTRANSPORT_DATA_ROOT/met/a.bin") ==
                  "/tmp/atmos_root/met/a.bin"
            @test expand_data_path(raw"${ATMOSTRANSPORT_DATA_ROOT}/met/a.bin") ==
                  "/tmp/atmos_root/met/a.bin"
            cfg = Dict("binary_paths" => [raw"$ATMOSTRANSPORT_DATA_ROOT/met/a.bin"])
            @test expand_binary_paths(cfg) == ["/tmp/atmos_root/met/a.bin"]
        end
    end

    @testset "Shape A: non-list errors" begin
        cfg = Dict("binary_paths" => "not-a-list.bin")
        @test_throws ArgumentError expand_binary_paths(cfg)
    end

    @testset "Shape B: complete folder → sorted paths" begin
        mktempdir() do dir
            dates = [Date(2021, 12, d) for d in 1:5]
            _touch_binary_set(dir, dates;
                              prefix = "era5_transport_",
                              suffix = "_merged1000Pa_float32.bin")
            cfg = Dict(
                "folder"     => dir,
                "start_date" => "2021-12-01",
                "end_date"   => "2021-12-05",
            )
            paths = expand_binary_paths(cfg)
            @test length(paths) == 5
            # Sorted by parsed date
            @test paths == sort(paths)
            # First and last match expected dates
            @test occursin("20211201", paths[1])
            @test occursin("20211205", paths[end])
        end
    end

    @testset "Shape B: subrange of present files" begin
        mktempdir() do dir
            dates = [Date(2021, 12, d) for d in 1:10]
            _touch_binary_set(dir, dates)
            cfg = Dict(
                "folder"     => dir,
                "start_date" => "2021-12-03",
                "end_date"   => "2021-12-05",
            )
            paths = expand_binary_paths(cfg)
            @test length(paths) == 3
            @test occursin("20211203", paths[1])
            @test occursin("20211205", paths[end])
        end
    end

    @testset "Shape B: missing day in range errors with listing" begin
        mktempdir() do dir
            dates = [Date(2021, 12, 1), Date(2021, 12, 3)]  # missing Dec 2
            _touch_binary_set(dir, dates)
            cfg = Dict(
                "folder"     => dir,
                "start_date" => "2021-12-01",
                "end_date"   => "2021-12-03",
            )
            err = try
                expand_binary_paths(cfg); nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("2021-12-02", sprint(showerror, err))
        end
    end

    @testset "Shape B: start_date > end_date errors" begin
        mktempdir() do dir
            _touch_binary_set(dir, [Date(2021, 12, 1)])
            cfg = Dict(
                "folder"     => dir,
                "start_date" => "2021-12-05",
                "end_date"   => "2021-12-01",
            )
            @test_throws ArgumentError expand_binary_paths(cfg)
        end
    end

    @testset "Shape B: folder missing errors" begin
        cfg = Dict(
            "folder"     => "/nonexistent/dir/xyz",
            "start_date" => "2021-12-01",
            "end_date"   => "2021-12-03",
        )
        @test_throws ArgumentError expand_binary_paths(cfg)
    end

    @testset "Shape B: file_pattern placeholder required" begin
        mktempdir() do dir
            _touch_binary_set(dir, [Date(2021, 12, 1)])
            cfg = Dict(
                "folder"       => dir,
                "start_date"   => "2021-12-01",
                "end_date"     => "2021-12-01",
                "file_pattern" => "no-placeholder.bin",
            )
            @test_throws ArgumentError expand_binary_paths(cfg)
        end
    end

    @testset "Shape B: file_pattern with placeholder matches strictly" begin
        mktempdir() do dir
            # One matching file, one non-matching extra (wrong prefix)
            _touch_binary_set(dir, [Date(2021, 12, 1)];
                              prefix = "era5_transport_",
                              suffix = "_v4.bin")
            open(joinpath(dir, "other_20211201.bin"), "w") do io
                write(io, "")
            end
            cfg = Dict(
                "folder"       => dir,
                "start_date"   => "2021-12-01",
                "end_date"     => "2021-12-01",
                "file_pattern" => "era5_transport_{YYYYMMDD}_v4.bin",
            )
            paths = expand_binary_paths(cfg)
            @test length(paths) == 1
            @test occursin("era5_transport_20211201_v4.bin", paths[1])
        end
    end

    @testset "Mutual exclusion: binary_paths + folder errors" begin
        cfg = Dict(
            "binary_paths" => ["/tmp/a.bin"],
            "folder"       => "/tmp",
            "start_date"   => "2021-12-01",
            "end_date"     => "2021-12-01",
        )
        @test_throws ArgumentError expand_binary_paths(cfg)
    end

    @testset "Empty [input] errors" begin
        cfg = Dict{String, Any}()
        @test_throws ArgumentError expand_binary_paths(cfg)
    end

    @testset "validate_config reports missing input paths and bad float types" begin
        cfg = Dict{String, Any}(
            "input" => Dict("binary_paths" => ["/definitely/missing/transport.bin"]),
            "numerics" => Dict("float_type" => "Float128"),
        )
        ok, errors = validate_config(cfg)
        @test !ok
        @test any(err -> occursin("resolved path does not exist", err), errors)
        @test any(err -> occursin("float_type", err), errors)
    end

end
