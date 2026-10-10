#!/usr/bin/env julia

using Test

import AtmosTransport
const AT = AtmosTransport
const MD = AT.MetDrivers
const TapeMod = AT.Tape
const DD = AT.DataDownloads
const PP = AT.Preprocessing

using .AtmosTransport.Architectures: CPU
using .AtmosTransport.Grids: AtmosGrid, HybridSigmaPressure, ncells, nfaces

function _rg_fixture(::Type{FT}=Float32) where FT
    mesh = AT.ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT=FT)
    vc = HybridSigmaPressure(FT[0, 1000], FT[0, 1])
    grid = AtmosGrid(mesh, vc, CPU(); FT=FT)
    window = (
        m = ones(FT, ncells(mesh), 1),
        hflux = zeros(FT, nfaces(mesh), 1),
        cm = zeros(FT, ncells(mesh), 2),
        ps = fill(FT(90_000), ncells(mesh)),
    )
    return grid, window
end

function _open_rg(path, nwindow=1; extra_header=Dict{String,Any}())
    grid, window = _rg_fixture()
    writer = MD.open_streaming_transport_binary(
        path, grid, nwindow, window;
        FT=Float32, header_bytes=4096, steps_per_window=1,
        source_flux_sampling=:window_start_endpoint,
        humidity_sampling=:none, delta_semantics=:none, mass_basis=:dry,
        extra_header=extra_header,
    )
    return writer, window
end

function _ll_fixture(::Type{FT}=Float32) where FT
    mesh = AT.LatLonMesh(; Nx=4, Ny=3, FT)
    vc = HybridSigmaPressure(FT[0, 1000], FT[0, 1])
    grid = AtmosGrid(mesh, vc, CPU(); FT)
    window = (
        m=ones(FT, 4, 3, 1),
        am=zeros(FT, 5, 3, 1),
        bm=zeros(FT, 4, 4, 1),
        cm=zeros(FT, 4, 3, 2),
        ps=fill(FT(90_000), 4, 3),
    )
    return grid, window
end

@testset "download verification sidecars preserve safe resume semantics" begin
    mktempdir() do dir
        path = joinpath(dir, "download.dat")
        write(path, Vector{UInt8}(codeunits("complete-payload")))
        env = DD.PythonEnvironment("python3", false, false, false, false,
                                   false, false)
        protocol = DD.CDSProtocol(env)
        task = DD.DownloadTask("fixture", "dataset/request", path,
                               Dict{String,Any}("variable" => "co2"), 1.0)

        @test DD._existing_task_status(task, protocol) == (:unverifiable, 0)
        manifest_path = DD._write_download_manifest(task, protocol)
        @test isfile(manifest_path)
        @test DD._existing_task_status(task, protocol) ==
              (:verified, filesize(path))
        different_environment = DD.CDSProtocol(DD.PythonEnvironment(
            "/different/python", true, true, true, true, true, true))
        @test DD._existing_task_status(task, different_environment) ==
              (:verified, filesize(path))

        changed_request = DD.DownloadTask(
            "fixture", "dataset/request", path,
            Dict{String,Any}("variable" => "temperature"), 1.0)
        @test first(DD._existing_task_status(changed_request, protocol)) == :corrupt

        bytes = read(path)
        bytes[1] = bytes[1] == 0x00 ? 0x01 : bytes[1] - 0x01
        write(path, bytes)
        @test first(DD._existing_task_status(task, protocol)) == :corrupt
    end
end

@testset "generation fingerprints cover source and preprocessing settings" begin
    provenance = (script_path="preprocess.jl", script_mtime=1.0,
                  git_commit="abc123", git_dirty=false,
                  creation_time="ignored")
    kwargs = (spectral_resolution=127, source_paths=String[],
              next_day_hour0=nothing, provenance=provenance)
    base = PP.generation_fingerprint(
        ; settings=(T_target=127, source_dir="a"), kwargs...)
    changed_resolution = PP.generation_fingerprint(
        ; settings=(T_target=255, source_dir="a"), kwargs...)
    changed_source = PP.generation_fingerprint(
        ; settings=(T_target=127, source_dir="b"), kwargs...)
    @test base != changed_resolution
    @test base != changed_source
end

@testset "streaming binary writes fail closed" begin
    mktempdir() do dir
        @testset "generic windows retain the sample shape contract" begin
            path = joinpath(dir, "rg-shape.bin")
            sentinel = Vector{UInt8}(codeunits("previous-valid-stream"))
            write(path, sentinel)
            writer, window = _open_rg(path)
            bad = merge(window, (; m=zeros(Float32, length(window.m))))
            @test_throws DimensionMismatch MD.write_streaming_window!(writer, bad)
            @test_throws ArgumentError MD.close_streaming_transport_binary!(writer)
            @test read(path) == sentinel
            @test !ispath(writer.staging_path)
        end

        @testset "generic windows retain the sample payload contract" begin
            path = joinpath(dir, "rg-payload.bin")
            writer, window = _open_rg(path)
            qv = zeros(Float32, size(window.m))
            @test_throws ArgumentError MD.write_streaming_window!(
                writer, merge(window, (; qv)))
            @test_throws ArgumentError MD.write_streaming_window!(
                writer, merge(window, (; qv_start=qv, qv_end=qv)))
            @test_throws ArgumentError MD.close_streaming_transport_binary!(writer)
            @test !ispath(path)
            @test !ispath(writer.staging_path)

            empty_path = joinpath(dir, "rg-empty-optionals.bin")
            empty_writer, empty_window = _open_rg(empty_path)
            MD.write_streaming_window!(
                empty_writer,
                merge(empty_window, (; qv_start=nothing, qv_end=nothing, dm=nothing)),
            )
            MD.close_streaming_transport_binary!(empty_writer)
            @test isfile(empty_path)
        end

        @testset "published binaries honour the umask" begin
            path = joinpath(dir, "rg-mode.bin")
            # A known umask (restored afterwards): 0o666 & ~0o022 = 0o644, while
            # a `mktemp`-staged file would be published as 0o600.
            old_umask = Sys.iswindows() ? nothing : ccall(:umask, Cuint, (Cuint,), 0o022)
            try
                writer, window = _open_rg(path)
                MD.write_streaming_window!(writer, window)
                MD.close_streaming_transport_binary!(writer)
            finally
                old_umask === nothing || ccall(:umask, Cuint, (Cuint,), old_umask)
            end
            @test isfile(path)
            Sys.iswindows() || @test filemode(path) & 0o777 == 0o644
            @test isempty(filter(f -> occursin("jl_", f), readdir(dir)))   # no staging leftovers
        end

        @testset "CS validates arguments, panel count, and every panel shape" begin
            path = joinpath(dir, "cs-shape.bin")
            Nc, npanel, Nz = 2, 6, 1
            vc = HybridSigmaPressure(Float32[0, 1000], Float32[0, 1])
            writer = MD.open_streaming_cs_transport_binary(
                path, Nc, npanel, Nz, 1, vc; planet_radius = AtmosTransport.Parameters.EARTH_RADIUS,
                FT=Float32, header_bytes=4096, steps_per_window=1,
                mass_basis=:dry,
            )
            window = (
                m=ntuple(_ -> ones(Float32, Nc, Nc, Nz), npanel),
                am=ntuple(_ -> zeros(Float32, Nc + 1, Nc, Nz), npanel),
                bm=ntuple(_ -> zeros(Float32, Nc, Nc + 1, Nz), npanel),
                cm=ntuple(_ -> zeros(Float32, Nc, Nc, Nz + 1), npanel),
                ps=ntuple(_ -> fill(90_000f0, Nc, Nc), npanel),
            )
            @test_throws DimensionMismatch MD.write_streaming_cs_window!(writer, window, Nc + 1, npanel)
            bad_panels = Base.setindex(window.m, zeros(Float32, Nc + 1, Nc, Nz), 3)
            @test_throws DimensionMismatch MD.write_streaming_cs_window!(
                writer, merge(window, (; m=bad_panels)), Nc, npanel)
            @test_throws ArgumentError MD.close_streaming_transport_binary!(writer)
            @test !ispath(path)
            @test !ispath(writer.staging_path)
        end

        @testset "CS rejects undeclared humidity payloads" begin
            path = joinpath(dir, "cs-payload.bin")
            Nc, npanel, Nz = 2, 6, 1
            vc = HybridSigmaPressure(Float32[0, 1000], Float32[0, 1])
            writer = MD.open_streaming_cs_transport_binary(
                path, Nc, npanel, Nz, 1, vc; planet_radius = AtmosTransport.Parameters.EARTH_RADIUS,
                FT=Float32, header_bytes=4096, steps_per_window=1,
                mass_basis=:dry,
            )
            window = (
                m=ntuple(_ -> ones(Float32, Nc, Nc, Nz), npanel),
                am=ntuple(_ -> zeros(Float32, Nc + 1, Nc, Nz), npanel),
                bm=ntuple(_ -> zeros(Float32, Nc, Nc + 1, Nz), npanel),
                cm=ntuple(_ -> zeros(Float32, Nc, Nc, Nz + 1), npanel),
                ps=ntuple(_ -> fill(90_000f0, Nc, Nc), npanel),
            )
            qv = ntuple(_ -> zeros(Float32, Nc, Nc, Nz), npanel)
            @test_throws ArgumentError MD.write_streaming_cs_window!(
                writer, merge(window, (; qv)), Nc, npanel)
            @test_throws ArgumentError MD.write_streaming_cs_window!(
                writer, merge(window, (; qv_start=qv, qv_end=qv)), Nc, npanel)
            @test_throws ArgumentError MD.close_streaming_transport_binary!(writer)
            @test !ispath(path)
            @test !ispath(writer.staging_path)

            empty_path = joinpath(dir, "cs-empty-optionals.bin")
            empty_writer = MD.open_streaming_cs_transport_binary(
                empty_path, Nc, npanel, Nz, 1, vc; planet_radius = AtmosTransport.Parameters.EARTH_RADIUS,
                FT=Float32, header_bytes=4096, steps_per_window=1,
                mass_basis=:dry,
            )
            empty_optionals = (
                cmfmc=nothing, dtrain=nothing, dkg=nothing,
                surface=nothing, vdiff=nothing,
                qv_start=nothing, qv_end=nothing,
                dam=nothing, dbm=nothing, dhflux=nothing, dcm=nothing,
            )
            MD.write_streaming_cs_window!(
                empty_writer, merge(window, empty_optionals), Nc, npanel)
            MD.close_streaming_transport_binary!(empty_writer)
            @test isfile(empty_path)
        end

        @testset "extra metadata cannot rewrite structural fields" begin
            path = joinpath(dir, "override.bin")
            @test_throws ArgumentError _open_rg(path, 1; extra_header=Dict("ncell" => 999))
            @test_throws ArgumentError _open_rg(
                path, 1; extra_header=Dict("latitudes" => [-30.0, 30.0]))
            @test !ispath(path)
        end

        @testset "header layout metadata is internally consistent" begin
            path = joinpath(dir, "header-contract.bin")
            writer, window = _open_rg(path)
            header = deepcopy(writer.header)
            for (key, value) in (("float_type", "Float128"),
                                 ("mass_basis", "unknown"),
                                 ("elems_per_window", header["elems_per_window"] + 1),
                                 ("dt_met_seconds", Inf),
                                 ("nface_h", header["nface_h"] + 1))
                bad = deepcopy(header)
                bad[key] = value
                @test_throws ArgumentError MD.validate_transport_contract!(bad)
            end
            bad = deepcopy(header)
            push!(bad["payload_sections"], "qv_start")
            @test_throws ArgumentError MD.validate_transport_contract!(bad)
            for key in ("nlat", "latitudes", "nlon_per_ring")
                bad = deepcopy(header)
                delete!(bad, key)
                @test_throws ArgumentError MD.validate_transport_contract!(bad)
            end
            bad = deepcopy(header)
            bad["ring_latitudes"] = pop!(bad, "latitudes")
            @test_throws ArgumentError MD.validate_transport_contract!(bad)
            bad = deepcopy(header)
            bad["latitudes"] .= first(bad["latitudes"])
            @test_throws ArgumentError MD.validate_transport_contract!(bad)
            bad = deepcopy(header)
            bad["nface_h"] += 1
            @test_throws ArgumentError MD.validate_transport_contract!(bad)
            MD.write_streaming_window!(writer, window)
            MD.close_streaming_transport_binary!(writer)
        end

        @testset "lat-lon coordinates are canonical and nondegenerate" begin
            path = joinpath(dir, "ll-header-contract.bin")
            grid, window = _ll_fixture()
            MD.write_transport_binary(
                path, grid, [window]; FT=Float32, header_bytes=4096,
                steps_per_window=1,
                source_flux_sampling=:window_start_endpoint,
                humidity_sampling=:none, delta_semantics=:none,
                mass_basis=:dry,
            )
            header = open(path, "r") do io
                raw = MD._read_transport_header_json(io; source=path)
                Dict{String,Any}(String(k) => v for (k, v) in
                                 pairs(MD.JSON3.read(String(raw))))
            end
            for key in ("lons", "lats")
                bad = deepcopy(header)
                bad[key] = fill(first(bad[key]), length(bad[key]))
                @test_throws ArgumentError MD.validate_transport_contract!(bad)
            end
        end

        @testset "eager writers ignore empty optional slots" begin
            cases = (
                ("ll", _ll_fixture,
                 (; qv_start=nothing, qv_end=nothing,
                    dam=nothing, dbm=nothing, dcm=nothing, dm=nothing)),
                ("rg", _rg_fixture,
                 (; qv_start=nothing, qv_end=nothing,
                    dhflux=nothing, dcm=nothing, dm=nothing)),
            )
            for (name, fixture, empty_optionals) in cases
                path = joinpath(dir, "$(name)-eager-empty-optionals.bin")
                grid, window = fixture()
                MD.write_transport_binary(
                    path, grid, [merge(window, empty_optionals)];
                    FT=Float32, header_bytes=4096, steps_per_window=1,
                    source_flux_sampling=:window_start_endpoint,
                    humidity_sampling=:none, delta_semantics=:none,
                    mass_basis=:dry,
                )
                @test isfile(path)
            end
        end

        @testset "failed final header publication preserves the destination" begin
            path = joinpath(dir, "header-publication.bin")
            sentinel = Vector{UInt8}(codeunits("previous-valid-stream"))
            write(path, sentinel)
            writer, window = _open_rg(path)
            MD.write_streaming_window!(writer, window)
            writer.header["oversized_metadata"] = "x"^writer.header_bytes
            @test_throws ArgumentError MD.close_streaming_transport_binary!(writer)
            @test read(path) == sentinel
            @test !ispath(writer.staging_path)
        end

        @testset "eager writes replace the destination only after success" begin
            path = joinpath(dir, "atomic-eager.bin")
            sentinel = Vector{UInt8}(codeunits("previous-valid-artifact"))
            write(path, sentinel)
            grid, window = _rg_fixture()
            bad = merge(window, (; m=fill("not-a-number", size(window.m))))
            @test_throws Exception MD.write_transport_binary(
                path, grid, [bad];
                FT=Float32, header_bytes=4096, steps_per_window=1,
                source_flux_sampling=:window_start_endpoint,
                humidity_sampling=:none, delta_semantics=:none,
                mass_basis=:dry,
            )
            @test read(path) == sentinel
            @test !ispath(path * ".tmp")
        end
    end
end

@testset "binary readers reject size mismatches and close failed opens" begin
    mktempdir() do dir
        valid = joinpath(dir, "valid.bin")
        writer, window = _open_rg(valid)
        MD.write_streaming_window!(writer, window)
        MD.close_streaming_transport_binary!(writer)
        reader = MD.TransportBinaryReader(valid)
        close(reader)

        open(valid, "a") do io
            write(io, UInt8(0xff))
        end
        @test_throws ArgumentError MD.TransportBinaryReader(valid)

        malformed = joinpath(dir, "malformed.bin")
        open(malformed, "w") do io
            write(io, "{not-json")
        end
        if isdir("/proc/self/fd")
            before = length(readdir("/proc/self/fd"))
            for _ in 1:100
                @test_throws Exception MD.TransportBinaryReader(malformed)
                @test_throws Exception MD.TransportBinaryReader(malformed)
            end
            after = length(readdir("/proc/self/fd"))
            @test after <= before + 2
        end
    end
end

@testset "binary readers support headers larger than the legacy probe" begin
    mktempdir() do dir
        metadata = Dict{String,Any}("oversized_metadata" => "x"^300_000)

        rg_path = joinpath(dir, "large-header-rg.bin")
        grid, window = _rg_fixture()
        rg_writer = MD.open_streaming_transport_binary(
            rg_path, grid, 1, window;
            FT=Float32, header_bytes=400_000, steps_per_window=1,
            source_flux_sampling=:window_start_endpoint,
            humidity_sampling=:none, delta_semantics=:none, mass_basis=:dry,
            extra_header=metadata,
        )
        MD.write_streaming_window!(rg_writer, window)
        MD.close_streaming_transport_binary!(rg_writer)
        rg_reader = MD.TransportBinaryReader(rg_path)
        close(rg_reader)
        @test MD.inspect_binary(rg_path; io=IOBuffer()).grid_type == :reduced_gaussian

        cs_path = joinpath(dir, "large-header-cs.bin")
        Nc, npanel, Nz = 1, 6, 1
        vc = HybridSigmaPressure(Float32[0, 1000], Float32[0, 1])
        cs_writer = MD.open_streaming_cs_transport_binary(
            cs_path, Nc, npanel, Nz, 1, vc; planet_radius = AtmosTransport.Parameters.EARTH_RADIUS,
            FT=Float32, header_bytes=400_000, steps_per_window=1,
            mass_basis=:dry, extra_header=metadata,
        )
        cs_window = (
            m=ntuple(_ -> ones(Float32, Nc, Nc, Nz), npanel),
            am=ntuple(_ -> zeros(Float32, Nc + 1, Nc, Nz), npanel),
            bm=ntuple(_ -> zeros(Float32, Nc, Nc + 1, Nz), npanel),
            cm=ntuple(_ -> zeros(Float32, Nc, Nc, Nz + 1), npanel),
            ps=ntuple(_ -> fill(90_000f0, Nc, Nc), npanel),
        )
        MD.write_streaming_cs_window!(cs_writer, cs_window, Nc, npanel)
        MD.close_streaming_transport_binary!(cs_writer)
        cs_reader = MD.TransportBinaryReader(cs_path)
        close(cs_reader)
        @test MD.inspect_binary(cs_path; io=IOBuffer()).grid_type == :cubed_sphere

        # JSON permits trailing whitespace, but its NUL terminator must still
        # occur inside the declared header region rather than in the payload.
        open(cs_path, "r+") do io
            prefix = read(io, 400_001)
            null_index = something(findfirst(==(0x00), prefix))
            seek(io, null_index - 1)
            write(io, fill(UInt8(' '), 400_001 - null_index))
            write(io, UInt8(0))
        end
        @test_throws ArgumentError MD.TransportBinaryReader(cs_path)
    end

    @test_throws ArgumentError MD._read_transport_header_json(
        IOBuffer(Vector{UInt8}(codeunits("{\"unterminated\":true}")));
        source="test binary",
    )
end

@testset "mmap tape manifest is atomic and cannot outlive its records" begin
    mktempdir() do dir
        panels = ntuple(_ -> ones(Float32, 2, 2, 1), 6)
        first_storage = TapeMod.MmapCSTapeStorage(
            dir=dir, cleanup_on_finalize=false)
        TapeMod._stage_panels(first_storage, panels)
        TapeMod.finalize_tape!(first_storage)
        @test isfile(joinpath(dir, "manifest.toml"))

        second_storage = TapeMod.MmapCSTapeStorage(
            dir=dir, cleanup_on_finalize=false)
        @test !ispath(joinpath(dir, "manifest.toml"))
        @test filesize(joinpath(dir, "records.bin")) == 0
        @test_throws ArgumentError TapeMod.load_mmap_tape(dir)
        TapeMod.finalize_tape!(second_storage)

        open(joinpath(dir, "records.bin"), "a") do io
            write(io, UInt8(0x01))
        end
        @test_throws ArgumentError TapeMod.load_mmap_tape(dir)
    end
end
