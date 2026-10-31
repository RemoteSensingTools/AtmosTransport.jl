#!/usr/bin/env julia
# Plan 41 P3a - focused tests for the typed binary-writer adapters.

using Test

using AtmosTransport
include(joinpath(@__DIR__, "..", "helpers", "preprocessing_evidence.jl"))
const Pre = AtmosTransport.Preprocessing
using .AtmosTransport.Architectures: CPU
using .AtmosTransport.Grids: AtmosGrid, HybridSigmaPressure, ncells, nfaces
using .AtmosTransport.MetDrivers: open_streaming_transport_binary,
                                  open_streaming_cs_transport_binary
using .AtmosTransport.Preprocessing: AbstractBinaryWriter,
                                       LatLonBinaryWriter,
                                       ReducedGaussianBinaryWriter,
                                       CubedSphereBinaryWriter,
                                       ReadyWindow,
                                       LatLonTargetGeometry,
                                       ReducedGaussianTargetGeometry,
                                       CubedSphereTargetGeometry,
                                       DryBasis,
                                       build_target_geometry,
                                       allocate_window_storage,
                                       allocate_merge_workspace,
                                       write_window!,
                                       close_streaming_binary!,
                                       promote_streaming_binary!,
                                       quarantine_streaming_binary!,
                                       validate_staged_binary!,
                                       writer_staging_path,
                                       writer_final_path,
                                       HEADER_SIZE

function _fill_ll_storage!(storage, ::Type{FT}, Nx, Ny, Nz) where FT
    storage.all_m[1]  = fill(FT(1), Nx, Ny, Nz)
    storage.all_am[1] = fill(FT(0.1), Nx + 1, Ny, Nz)
    storage.all_bm[1] = fill(FT(0.2), Nx, Ny + 1, Nz)
    storage.all_cm[1] = fill(FT(0.3), Nx, Ny, Nz + 1)
    storage.all_ps[1] = fill(FT(90000), Nx, Ny)
    return storage
end

@testset "LatLonBinaryWriter delegates to v4 window writer" begin
    mktempdir() do tmp
        FT = Float32
        Nx, Ny, Nz = 3, 2, 2
        grid = build_target_geometry(Val(:latlon),
                                      Dict{String, Any}("nlon" => Nx, "nlat" => Ny),
                                      FT)
        storage = _fill_ll_storage!(allocate_window_storage(1, FT), FT, Nx, Ny, Nz)
        merged = allocate_merge_workspace(grid, Nz, Nz, FT)
        settings = (include_qv = false,
                    include_surface = false)
        final = joinpath(tmp, "ll.bin")
        staging = Pre.open_preprocessing_staging(final)
        @test startswith(staging, abspath(final) * ".tmp-")

        writer = LatLonBinaryWriter(staging, "{}", settings, merged, nothing,
                                    FT, DryBasis(); final_path = final)
        @test writer isa AbstractBinaryWriter{LatLonTargetGeometry, FT, DryBasis}
        @test writer_staging_path(writer) == staging
        @test writer_final_path(writer) == final

        ready = ReadyWindow{LatLonTargetGeometry, FT}(1, (; storage), payload_only_evidence(LatLonTargetGeometry, FT))
        bytes = write_window!(writer, ready)

        @test close_streaming_binary!(writer) == staging
        @test close_streaming_binary!(writer) == staging
        @test promote_streaming_binary!(writer) == final
        @test writer_staging_path(writer) == staging
        @test isfile(final)
        @test !isfile(staging)
        @test filesize(final) == HEADER_SIZE + bytes
        @test quarantine_streaming_binary!(writer) == staging
        @test isfile(final)
    end
end

@testset "staged-size validation runs before final promotion" begin
    mktempdir() do tmp
        FT = Float64
        Nc, npanel, Nz = 2, 6, 1
        vc = HybridSigmaPressure(FT[0, 1000], FT[0, 1])
        final = joinpath(tmp, "cs.bin")
        staging = Pre.open_preprocessing_staging(final)
        inner = open_streaming_cs_transport_binary(
            staging, Nc, npanel, Nz, 1, vc; planet_radius = AtmosTransport.Parameters.EARTH_RADIUS,
            FT = FT,
            header_bytes = 4096,
            steps_per_window = 1,
            include_flux_delta = true,
            mass_basis = :dry,
            surface_pressure_basis = TotalSurfacePressure(),
        )
        window = (m = ntuple(_ -> fill(FT(1), Nc, Nc, Nz), npanel),
                  am = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), npanel),
                  bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), npanel),
                  cm = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), npanel),
                  ps = ntuple(_ -> fill(FT(90000), Nc, Nc), npanel),
                  dm = ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel))
        writer = CubedSphereBinaryWriter(inner, DryBasis();
                                         Nc = Nc, npanel = npanel,
                                         final_path = final)

        sentinel = Vector{UInt8}(codeunits("previous-valid-output"))
        write(final, sentinel)
        write_window!(writer, ReadyWindow{CubedSphereTargetGeometry, FT}(1, window, payload_only_evidence(CubedSphereTargetGeometry, FT)))
        close_streaming_binary!(writer)
        open(writer.path, "a") do io
            write(io, UInt8(0xff))
        end
        @test_throws ArgumentError validate_staged_binary!(writer)
        quarantine_streaming_binary!(writer)
        @test read(final) == sentinel
    end
end

@testset "ReducedGaussianBinaryWriter wraps streaming writer" begin
    mktempdir() do tmp
        FT = Float64
        Nz = 1
        target = build_target_geometry(Val(:synthetic_reduced_gaussian),
                                       Dict{String, Any}("gaussian_number" => 1),
                                       FT)
        vc = HybridSigmaPressure(FT[0, 1000], FT[0, 1])
        grid = AtmosGrid(target.mesh, vc, CPU(); FT = FT,
                         radius = target.mesh.radius)
        ncell = ncells(target.mesh)
        nface = nfaces(target.mesh)
        window = (m = fill(FT(1), ncell, Nz),
                  hflux = zeros(FT, nface, Nz),
                  cm = zeros(FT, ncell, Nz + 1),
                  ps = fill(FT(90000), ncell))
        final = joinpath(tmp, "rg.bin")
        staging = Pre.open_preprocessing_staging(final)
        inner = open_streaming_transport_binary(
            staging, grid, 1, window;
            FT = FT,
            header_bytes = 4096,
            steps_per_window = 1,
            source_flux_sampling = :window_start_endpoint,
            mass_basis = :dry,
            surface_pressure_basis = TotalSurfacePressure(),
            humidity_sampling = :none,
            delta_semantics = :none,
        )
        writer = ReducedGaussianBinaryWriter(inner, DryBasis(); final_path = final)
        @test writer isa AbstractBinaryWriter{ReducedGaussianTargetGeometry, FT, DryBasis}
        @test writer_staging_path(writer) == staging

        write_window!(writer, ReadyWindow{ReducedGaussianTargetGeometry, FT}(1, window, payload_only_evidence(ReducedGaussianTargetGeometry, FT)))
        @test_throws MethodError write_window!(writer, ReadyWindow{LatLonTargetGeometry, FT}(1, (; storage = nothing), payload_only_evidence(LatLonTargetGeometry, FT)))
        @test close_streaming_binary!(writer) == staging
        @test promote_streaming_binary!(writer) == final
        @test isfile(final)
        @test !isfile(staging)
        @test filesize(final) > 4096
    end
end

@testset "CubedSphereBinaryWriter wraps streaming writer" begin
    mktempdir() do tmp
        FT = Float64
        Nc, npanel, Nz = 2, 6, 1
        vc = HybridSigmaPressure(FT[0, 1000], FT[0, 1])
        final = joinpath(tmp, "cs.bin")
        staging = Pre.open_preprocessing_staging(final)
        inner = open_streaming_cs_transport_binary(
            staging, Nc, npanel, Nz, 1, vc; planet_radius = AtmosTransport.Parameters.EARTH_RADIUS,
            FT = FT,
            header_bytes = 4096,
            steps_per_window = 1,
            include_flux_delta = true,
            mass_basis = :dry,
            surface_pressure_basis = TotalSurfacePressure(),
        )
        window = (m = ntuple(_ -> fill(FT(1), Nc, Nc, Nz), npanel),
                  am = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), npanel),
                  bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), npanel),
                  cm = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), npanel),
                  ps = ntuple(_ -> fill(FT(90000), Nc, Nc), npanel),
                  dm = ntuple(_ -> zeros(FT, Nc, Nc, Nz), npanel))
        writer = CubedSphereBinaryWriter(inner, DryBasis();
                                         Nc = Nc, npanel = npanel,
                                         final_path = final)
        @test writer isa AbstractBinaryWriter{CubedSphereTargetGeometry, FT, DryBasis}
        @test writer_final_path(writer) == final

        write_window!(writer, ReadyWindow{CubedSphereTargetGeometry, FT}(1, window, payload_only_evidence(CubedSphereTargetGeometry, FT)))
        @test_throws MethodError write_window!(writer, ReadyWindow{ReducedGaussianTargetGeometry, FT}(1, window, payload_only_evidence(ReducedGaussianTargetGeometry, FT)))
        @test close_streaming_binary!(writer) == staging
        @test promote_streaming_binary!(writer) == final
        @test isfile(final)
        @test !isfile(staging)
        @test filesize(final) > 4096
    end
end
