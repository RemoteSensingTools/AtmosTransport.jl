#!/usr/bin/env julia
# Binary capability and inspector contracts against a tiny lat-lon fixture.

using Test

using AtmosTransport

const _INSPECTOR_AIR_MASS = 1e16

function _inspector_fixture_binary(path::AbstractString;
                                   FT::Type{<:AbstractFloat} = Float64,
                                   with_tm5::Bool = false,
                                   with_surface::Bool = false,
                                   with_single_qv::Bool = false,
                                   flux_kind::Symbol = :substep_mass_amount)
    Nx, Ny, Nz = 4, 3, 5
    mesh = LatLonMesh(; FT = FT, Nx = Nx, Ny = Ny)
    vertical = HybridSigmaPressure(
        FT[0, 100, 300, 600, 1000, 2000],
        FT[0, 0, 0.1, 0.3, 0.7, 1],
    )
    grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)

    m  = fill(FT(_INSPECTOR_AIR_MASS), Nx, Ny, Nz)
    am = zeros(FT, Nx + 1, Ny, Nz)
    bm = zeros(FT, Nx, Ny + 1, Nz)
    cm = zeros(FT, Nx, Ny, Nz + 1)
    ps = fill(FT(95_000), Nx, Ny)

    optional = if with_tm5
        tm5 = (entu = zeros(FT, Nx, Ny, Nz),
               detu = zeros(FT, Nx, Ny, Nz),
               entd = zeros(FT, Nx, Ny, Nz),
               detd = zeros(FT, Nx, Ny, Nz))
        (; tm5_fields = tm5)
    else
        NamedTuple()
    end
    if with_surface
        surface = AtmosTransport.MetDrivers.PBLSurfaceForcing(
            fill(FT(900), Nx, Ny), fill(FT(0.3), Nx, Ny),
            fill(FT(80), Nx, Ny), fill(FT(290), Nx, Ny))
        optional = merge(optional, (; surface))
    end
    with_single_qv &&
        (optional = merge(optional, (; qv = zeros(FT, Nx, Ny, Nz))))
    window = merge((; m, am, bm, cm, ps), optional)

    write_transport_binary(path, grid, [window];
                           FT = FT,
                           dt_met_seconds = 3600.0,
                           half_dt_seconds = 1800.0,
                           steps_per_window = 1,
                           mass_basis = :dry,
                           source_flux_sampling = :window_start_endpoint,
                           flux_sampling = :window_constant,
                           flux_kind)
    return nothing
end

@testset "binary_capabilities + inspect_binary" begin

    @testset "one version-4 reader API" begin
        @test isdefined(AtmosTransport, :TransportBinaryReader)
        @test !isdefined(AtmosTransport, :CubedSphereBinaryReader)
        @test !isdefined(AtmosTransport, :CubedSphereBinaryHeader)
        @test !isdefined(AtmosTransport, :load_cs_window)
        @test !isdefined(AtmosTransport, :CubedSphereTransportDriver)
        @test !isdefined(AtmosTransport, :StructuredTransportWindow)
        @test !isdefined(AtmosTransport, :FaceIndexedTransportWindow)
        @test !isdefined(AtmosTransport, :CubedSphereTransportWindow)
        @test isdefined(AtmosTransport, :TransportWindow)
    end

    @testset "LL writer rejects unsupported full-window flux storage" begin
        @test_throws ArgumentError _inspector_fixture_binary(
            tempname(); flux_kind = :full_window_mass_amount)
    end

    @testset "LL writer rejects obsolete single-field humidity" begin
        @test_throws ArgumentError _inspector_fixture_binary(
            tempname(); with_single_qv = true)
    end

    @testset "basic LL fixture (no TM5)" begin
        mktempdir() do dir
            path = joinpath(dir, "fixture_base.bin")
            _inspector_fixture_binary(path; with_tm5 = false)

            reader = TransportBinaryReader(path; FT = Float64)
            caps = binary_capabilities(reader)
            @test caps.advection === true
            @test caps.tm5_convection === false
            @test caps.cmfmc_convection === false
            @test caps.surface_pressure === true
            @test caps.mass_basis === :dry
            @test caps.grid_type === :latlon
            @test caps.nwindow == 1 && caps.window_seconds == 3600.0
            @test binary_geometry(reader) isa LatLonBinaryGeometry
            @test caps.flux_kind === :substep_mass_amount
            @test :m in caps.payload_sections
            @test :am in caps.payload_sections
            close(reader)
        end
    end

    @testset "LL fixture with TM5 convection sections" begin
        mktempdir() do dir
            path = joinpath(dir, "fixture_tm5.bin")
            _inspector_fixture_binary(path; with_tm5 = true)

            reader = TransportBinaryReader(path; FT = Float64)
            caps = binary_capabilities(reader)
            @test caps.tm5_convection === true
            @test :entu in caps.payload_sections
            @test :detu in caps.payload_sections
            @test :entd in caps.payload_sections
            @test :detd in caps.payload_sections
            close(reader)
        end
    end

    @testset "LL surface payload does not advertise unsupported diffusion" begin
        mktempdir() do dir
            path = joinpath(dir, "fixture_surface.bin")
            _inspector_fixture_binary(path; with_surface = true)

            reader = TransportBinaryReader(path; FT = Float64)
            @test has_surface(reader)
            caps = binary_capabilities(reader)
            @test caps.pbl_diffusion === false
            @test caps.gchp_vdiff === false
            driver = TransportBinaryDriver(reader)
            @test !supports_diffusion(driver)
            @test AtmosTransport.Models._runtime_has_surface(driver) === false
            close(driver)
        end
    end

    @testset "inspect_binary returns capability NamedTuple + prints report" begin
        mktempdir() do dir
            path = joinpath(dir, "fixture_inspect.bin")
            _inspector_fixture_binary(path; with_tm5 = true)

            io = IOBuffer()
            caps = inspect_binary(path; io = io)
            report = String(take!(io))

            # Programmatic summary
            @test caps.advection === true
            @test caps.tm5_convection === true
            @test caps.surface_pressure === true
            @test caps.grid_type === :latlon

            # Human report contains capability rows with check marks
            @test occursin("Capabilities:", report)
            @test occursin("✓ advection", report)
            @test occursin("✓ TM5 convection", report)
            @test occursin("✓ surface pressure", report)
            @test occursin("mass_basis       = dry", report)
            @test occursin("grid_type        = latlon", report)
        end
    end

    @testset "inspect_binary errors on missing file" begin
        @test_throws ArgumentError inspect_binary("/nonexistent/xyz.bin")
    end

end
