#!/usr/bin/env julia
# Transport-binary format versions 4 and 5 (docs/src/concepts/binary_format.md).
# Version 5 is version 4 plus the flux-construction identity: writers record it
# only when given the typed `flux_construction` keyword, so every default
# output stays version 4 with unchanged bytes. Readers accept exactly {4, 5};
# version-5 keys in a version-4 header, a version-5 header without a valid
# identity, and every other version are rejected.

using Test
using JSON3
using AtmosTransport
using AtmosTransport.Parameters: IFS_EARTH_RADIUS, EARTH_RADIUS
using AtmosTransport.MetDrivers: TransportBinaryReader, TransportBinaryDriver,
                                 write_transport_binary, total_windows,
                                 TRANSPORT_BINARY_FORMAT_VERSION, TRANSPORT_BINARY_FORMAT_VERSIONS,
                                 FLUX_CONSTRUCTION_FORMAT_VERSION, NoFluxConstructionHeader,
                                 PointSampledSpectralFluxesHeader, TM5SpectralFluxesHeader
using AtmosTransport.MetDrivers.BinaryFormat: flux_construction_tag
const MD = AtmosTransport.MetDrivers
const BF = AtmosTransport.MetDrivers.BinaryFormat

const V5_TM5 = TM5SpectralFluxesHeader(1, 639, 2, IFS_EARTH_RADIUS)
const V5_POINT = PointSampledSpectralFluxesHeader()
const V5_KEYS = ("flux_construction", "flux_construction_revision", "spectral_truncation_used",
                 "quadrature_refinement_steps", "vod2uv_radius_m")

function ll_binary(path; kwargs...)
    Nx, Ny, Nz = 4, 3, 5
    mesh = LatLonMesh(; Nx, Ny)
    grid = AtmosGrid(mesh, HybridSigmaPressure([0, 100, 300, 600, 1000, 2000.0],
                                               [0, 0, 0.1, 0.3, 0.7, 1.0]), CPU())
    window = (m = [1e16 + i + 10j + 100k for i in 1:Nx, j in 1:Ny, k in 1:Nz],
              am = [0.5i - j + 0.25k for i in 1:Nx+1, j in 1:Ny, k in 1:Nz],
              bm = [0.1i + j - 0.5k for i in 1:Nx, j in 1:Ny+1, k in 1:Nz],
              cm = zeros(Nx, Ny, Nz + 1), ps = fill(95_000.0, Nx, Ny))
    write_transport_binary(path, grid, [window, window]; dt_met_seconds = 3600.0,
                           half_dt_seconds = 1800.0, steps_per_window = 2, mass_basis = :dry,
                           surface_pressure_basis = TotalSurfacePressure(),
                           source_flux_sampling = :window_start_endpoint,
                           flux_sampling = :window_constant, kwargs...)
    return path
end

function rg_grid_window()
    mesh = ReducedGaussianMesh([-45.0, 45.0], [4, 4])
    grid = AtmosGrid(mesh, HybridSigmaPressure([0, 100, 300.0], [0, 0, 1.0]), CPU())
    nc, nf, nl = ncells(mesh), nfaces(mesh), nlevels(grid)
    window = (m = [1.0 + c + 0.5k for c in 1:nc, k in 1:nl],
              hflux = [0.01f - 0.02k for f in 1:nf, k in 1:nl],
              cm = zeros(nc, nl + 1), ps = fill(90_000.0, nc))
    return grid, window
end

const RG_KW = (; dt_met_seconds = 3600.0, half_dt_seconds = 1800.0, steps_per_window = 2,
               source_flux_sampling = :window_start_endpoint, mass_basis = :moist,
               surface_pressure_basis = TotalSurfacePressure())

function rg_binary(path; kwargs...)
    grid, window = rg_grid_window()
    write_transport_binary(path, grid, [window, window]; RG_KW..., kwargs...)
    return path
end

function rg_stream_binary(path; schedule = nothing, kwargs...)
    grid, window = rg_grid_window()
    writer = MD.open_streaming_transport_binary(path, grid, 2, window; RG_KW..., kwargs...)
    schedule === nothing || MD.set_streaming_steps_per_window_schedule!(writer, schedule)
    MD.write_streaming_window!(writer, window)
    MD.write_streaming_window!(writer, window)
    MD.close_streaming_transport_binary!(writer)
    return path
end

function cs_binary(path; kwargs...)
    Nc, Nz = 2, 2
    writer = MD.open_streaming_cs_transport_binary(
        path, Nc, 6, Nz, 1, HybridSigmaPressure([0, 100, 300.0], [0, 0, 1.0]);
        planet_radius = EARTH_RADIUS, dt_met_seconds = 3600.0, steps_per_window = 2,
        mass_basis = :dry, surface_pressure_basis = TotalSurfacePressure(), kwargs...)
    window = (m = ntuple(p -> fill(1e15 + p, Nc, Nc, Nz), 6),
              am = ntuple(_ -> zeros(Nc + 1, Nc, Nz), 6),
              bm = ntuple(_ -> zeros(Nc, Nc + 1, Nz), 6),
              cm = ntuple(_ -> zeros(Nc, Nc, Nz + 1), 6),
              ps = ntuple(_ -> fill(90_000.0, Nc, Nc), 6))
    MD.write_streaming_cs_window!(writer, window, Nc, 6)
    MD.close_streaming_transport_binary!(writer)
    return path
end

const WRITERS = (ll = ll_binary, rg = rg_binary, rg_stream = rg_stream_binary, cs = cs_binary)

header_json(path) = open(io -> String(BF.read_transport_header_json(io)), path)
header_dict(path) = Dict{String, Any}(String(k) => v for (k, v) in pairs(JSON3.read(header_json(path))))
header_bytes(path) = Int(header_dict(path)["header_bytes"])
payload(path) = read(path)[header_bytes(path) + 1:end]

# Replace the JSON header in place through the production padded-header writer.
function rewrite_header!(path; updates = Dict{String, Any}(), delete_keys = String[])
    header = header_dict(path)
    foreach(key -> delete!(header, key), delete_keys)
    merge!(header, updates)
    open(path, "r+") do io
        BF.write_padded_header!(io, header, Int(header["header_bytes"]))
    end
    return path
end

# The ArgumentError a call throws, or `nothing` (closing a reader it returned).
rejection(f) = try
    result = f()
    result isa TransportBinaryReader && close(result)
    nothing
catch err
    err isa ArgumentError ? err : rethrow()
end

rejects(f, pattern) = (err = rejection(f); err !== nothing && occursin(pattern, err.msg))

@testset "format version 4: default writers record no construction" begin
    mktempdir() do dir
        for (name, write_binary) in pairs(WRITERS)
            default = write_binary(joinpath(dir, "$(name)_default.bin"))
            explicit = write_binary(joinpath(dir, "$(name)_explicit.bin");
                                    flux_construction = NoFluxConstructionHeader())
            # The explicit no-op keyword writes the default file bit for bit.
            @test read(default) == read(explicit)
            header = header_dict(default)
            @test header["format_version"] == TRANSPORT_BINARY_FORMAT_VERSION == 4
            @test !any(key -> haskey(header, key), V5_KEYS)
            reader = TransportBinaryReader(default; FT = Float64)
            @test reader.header.format_version == 4
            @test reader.header.flux_construction === NoFluxConstructionHeader()
            close(reader)
            report = sprint(io -> inspect_binary(default; io))
            @test occursin("format_version   = 4", report)
            @test occursin("flux_construction = not recorded (format v4)", report)
        end
    end
end

@testset "format version 5: construction identity round trip" begin
    @test TRANSPORT_BINARY_FORMAT_VERSIONS == (4, 5)
    @test FLUX_CONSTRUCTION_FORMAT_VERSION == 5
    mktempdir() do dir
        for (name, write_binary) in pairs(WRITERS), construction in (V5_TM5, V5_POINT)
            tag = flux_construction_tag(construction)
            v4 = write_binary(joinpath(dir, "$(name)_v4.bin"))
            v5 = write_binary(joinpath(dir, "$(name)_$(tag).bin"); flux_construction = construction)
            header = header_dict(v5)
            @test header["format_version"] == 5
            @test header["flux_construction"] == tag
            # The writer records exactly the identity and the keys of its construction.
            @test filter(key -> haskey(header, key), BF._FLUX_CONSTRUCTION_HEADER_KEYS) ==
                  (BF._FLUX_CONSTRUCTION_KEY, BF._flux_construction_keys(typeof(construction))...)
            if construction isa TM5SpectralFluxesHeader
                @test header["flux_construction_revision"] === 1
                @test header["spectral_truncation_used"] === 639
                @test header["quadrature_refinement_steps"] === 2
                @test header["vod2uv_radius_m"] == IFS_EARTH_RADIUS   # JSON3 may read it as an Int
            else
                @test !any(key -> haskey(header, key), V5_KEYS[2:end])
            end
            # Version 5 changes the header only: the payload is the version-4 payload.
            @test payload(v5) == payload(v4)
            @test BF.parse_transport_header(Vector{UInt8}(header_json(v5))).flux_construction ==
                  construction
            reader = TransportBinaryReader(v5; FT = Float64)
            @test reader.header.format_version == 5
            @test reader.header.flux_construction == construction
            reader4 = TransportBinaryReader(v4; FT = Float64)
            @test reader.data == reader4.data
            close(reader4)
            close(reader)
            driver = TransportBinaryDriver(v5; FT = Float64, arch = CPU())
            @test total_windows(driver) == (name === :cs ? 1 : 2)
            close(driver)
            caps = inspect_binary(v5; io = devnull)
            @test caps.format_version == 5
            @test caps.flux_construction == construction
            report = sprint(io -> inspect_binary(v5; io))
            @test occursin("format_version   = 5", report)
            @test occursin("flux_construction = $(BF.flux_construction_summary(construction))",
                           report)
        end
        report = sprint(io -> inspect_binary(joinpath(dir, "ll_tm5_spectral.bin"); io))
        @test occursin("tm5_spectral (revision 1, spectral_truncation_used=639, " *
                       "quadrature_refinement_steps=2, vod2uv_radius_m=6.371229e6)", report)

        # The streaming header rewrite (substep schedule) keeps the identity.
        streamed = rg_stream_binary(joinpath(dir, "rg_schedule.bin"); schedule = [1, 2],
                                    flux_construction = V5_TM5)
        reader = TransportBinaryReader(streamed; FT = Float64)
        @test reader.header.steps_per_window_by_window == [1, 2]
        @test reader.header.flux_construction == V5_TM5
        close(reader)
    end
end

@testset "unsupported versions and inconsistent identities are rejected" begin
    mktempdir() do dir
        v4 = ll_binary(joinpath(dir, "v4.bin"))
        v5 = ll_binary(joinpath(dir, "v5.bin"); flux_construction = V5_TM5)
        open_reader(path) = () -> TransportBinaryReader(path; FT = Float64)
        function patched(source, name; kwargs...)
            path = joinpath(dir, name)
            cp(source, path; force = true)
            return rewrite_header!(path; kwargs...)
        end

        # Only versions 4 and 5 are read, by the contract and by the typed parser.
        for (version, pattern) in ((3, "Obsolete transport binary format_version=3"),
                                   (6, "Unsupported transport binary format_version=6"))
            path = patched(v4, "v$(version).bin"; updates = Dict("format_version" => version))
            @test rejects(open_reader(path), pattern)
            @test rejects(() -> BF.parse_transport_header(Vector{UInt8}(header_json(path))),
                          pattern)
        end

        # Version-5 keys in a version-4 header, by the contract and by the typed parser.
        for key in ("flux_construction", "vod2uv_radius_m")
            value = key == "flux_construction" ? "tm5_spectral" : IFS_EARTH_RADIUS
            path = patched(v4, "v4_$(key).bin"; updates = Dict(key => value))
            pattern = "carries the version-5 key(s) `$(key)`"
            @test rejects(open_reader(path), pattern)
            @test rejects(() -> BF.parse_transport_header(Vector{UInt8}(header_json(path))),
                          pattern)
        end
        path = patched(v5, "v5_unknown.bin"; updates = Dict("flux_construction" => "tm5"))
        @test rejects(() -> BF.parse_transport_header(Vector{UInt8}(header_json(path))),
                      "unsupported flux_construction=\"tm5\"")
        # A complete identity under format_version 4 is rejected as well.
        path = patched(v5, "v5_as_v4.bin"; updates = Dict("format_version" => 4))
        @test rejects(open_reader(path), "must declare format_version=5")

        # A version-5 header must record a valid identity.
        cases = (
            (Dict{String, Any}(), ["flux_construction"], "requires `flux_construction`"),
            (Dict{String, Any}("flux_construction" => "tm5"), String[],
             "unsupported flux_construction=\"tm5\""),
            (Dict{String, Any}("flux_construction" => 5), String[], "must be a string"),
            (Dict{String, Any}(), ["vod2uv_radius_m"], "requires `vod2uv_radius_m`"),
            (Dict{String, Any}("spectral_truncation_used" => 0), String[], "integer ≥ 1"),
            # JSON3 reads an integral number such as 639.0 as an Int; 639.5 stays a Float64.
            (Dict{String, Any}("spectral_truncation_used" => 639.5), String[],
             "must be an integer"),
            (Dict{String, Any}("quadrature_refinement_steps" => true), String[],
             "must be an integer"),
            (Dict{String, Any}("vod2uv_radius_m" => -1.0), String[], "finite and positive"),
            (Dict{String, Any}("vod2uv_radius_m" => "6371229"), String[], "must be a number"),
            (Dict{String, Any}("flux_construction" => "point_sampled_spectral"), String[],
             "does not use `flux_construction_revision`"),
        )
        for (index, (updates, delete_keys, pattern)) in enumerate(cases)
            path = patched(v5, "v5_bad_$(index).bin"; updates, delete_keys)
            @test rejects(open_reader(path), pattern)
        end

        # Writers take the identity only through the typed keyword.
        for extra in (Dict{String, Any}("flux_construction" => "tm5_spectral"),
                      Dict{String, Any}("vod2uv_radius_m" => IFS_EARTH_RADIUS))
            @test rejects(() -> ll_binary(joinpath(dir, "extra.bin"); extra_header = extra),
                          "use the typed flux_construction writer keyword")
        end
        @test rejects(() -> ll_binary(joinpath(dir, "extra.bin");
                                      extra_header = Dict{String, Any}("format_version" => 5)),
                      "cannot override structural field \"format_version\"")
        @test !isfile(joinpath(dir, "extra.bin"))
    end

    @test rejects(() -> TM5SpectralFluxesHeader(0, 639, 2, IFS_EARTH_RADIUS), "integer ≥ 1")
    @test rejects(() -> TM5SpectralFluxesHeader(true, 639, 2, IFS_EARTH_RADIUS), "integer ≥ 1")
    @test rejects(() -> TM5SpectralFluxesHeader(1, 639, 2, Inf), "finite and positive")
end
