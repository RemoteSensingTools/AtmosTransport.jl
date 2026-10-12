#!/usr/bin/env julia
# Optional cubed-sphere sections round-trip through the streaming writer and
# reader: the 2-D `pbl_eflux` (GCHP non-local PBL input) and
# `cmfmc_cloud_base` (GCHP convective cloud base), whose prerequisites are
# enforced, and the 3-D GCHP VDIFF `u`/`v`/`t`/`qv` sections
# (`include_gchp_vdiff`).

using Test
using AtmosTransport
const MD = AtmosTransport.MetDrivers

@testset "CS optional 2-D sections" begin
    FT = Float64
    Nc, Nz, np = 2, 4, 6
    vc = HybridSigmaPressure(FT[0, 20, 100, 400, 1000], FT[0, 0.05, 0.3, 0.7, 1])
    panels3(x, nz = Nz) = ntuple(p -> fill(FT(x + p), Nc, Nc, nz), np)
    panels2(x) = ntuple(p -> fill(FT(x + p), Nc, Nc), np)
    core = (m = panels3(1e6),
            am = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), np),
            bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), np),
            cm = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), np),
            ps = panels2(9e4))
    surface = (pblh = panels2(500), ustar = panels2(0.3), hflux = panels2(20), t2m = panels2(280))
    eflux = ntuple(p -> FT[100p + 10i + j for i in 1:Nc, j in 1:Nc], np)
    cloud_base = ntuple(p -> FT[Nz - mod(p + i + j, 3) for i in 1:Nc, j in 1:Nc], np)
    open_writer(path; kw...) = MD.open_streaming_cs_transport_binary(
        path, Nc, np, Nz, 1, vc; planet_radius = AtmosTransport.Parameters.EARTH_RADIUS,
        FT, mass_basis = :dry, surface_pressure_basis = TotalSurfacePressure(), kw...)

    mktemp() do path, io
        close(io)
        writer = open_writer(path; include_surface = true, include_cmfmc = true,
                             include_pbl_eflux = true, include_cmfmc_cloud_base = true)
        window = merge(core, (; surface = merge(surface, (; eflux)), cmfmc = panels3(0.01, Nz + 1),
                                cmfmc_cloud_base = cloud_base))
        MD.write_streaming_cs_window!(writer, window, Nc, np)
        MD.close_streaming_transport_binary!(writer)

        reader = MD.TransportBinaryReader(path; FT)
        try
            @test MD.has_pbl_eflux(reader) && MD.has_cmfmc_cloud_base(reader)
            @test reader.header.payload_sections[end-1:end] == [:pbl_eflux, :cmfmc_cloud_base]
            loaded = MD.load_window!(reader, 1)
            @test loaded.surface.eflux == eflux
            @test loaded.cmfmc_cloud_base == cloud_base
            @test loaded.surface.pblh == surface.pblh                 # neighbours unaffected
        finally
            close(reader.io)
        end
    end

    # A binary without the sections reports them absent.
    mktemp() do path, io
        close(io)
        writer = open_writer(path)
        MD.write_streaming_cs_window!(writer, core, Nc, np)
        MD.close_streaming_transport_binary!(writer)
        reader = MD.TransportBinaryReader(path; FT)
        try
            @test !MD.has_pbl_eflux(reader) && !MD.has_cmfmc_cloud_base(reader)
            loaded = MD.load_window!(reader, 1)
            @test loaded.surface === nothing && loaded.cmfmc_cloud_base === nothing
        finally
            close(reader.io)
        end
    end

    # Prerequisites: the cloud base needs CMFMC, EFLUX needs the surface sections,
    # and a window must carry exactly the declared sections.
    @test_throws ArgumentError open_writer(tempname(); include_cmfmc_cloud_base = true)
    @test_throws ArgumentError open_writer(tempname(); include_pbl_eflux = true)
    mktemp() do path, io
        close(io)
        writer = open_writer(path; include_surface = true, include_pbl_eflux = true)
        @test_throws ArgumentError MD.write_streaming_cs_window!(writer, merge(core, (; surface)), Nc, np)
    end
end

@testset "CS GCHP VDIFF sections" begin
    FT = Float32
    Nc, Nz, np = 2, 3, 6
    vc = HybridSigmaPressure(FT[0, 10, 50, 100], FT[0, 0.1, 0.5, 1])
    panels3(x) = ntuple(p -> fill(FT(x + p), Nc, Nc, Nz), np)
    panels2(x) = ntuple(p -> fill(FT(x + p), Nc, Nc), np)
    surface = MD.PBLSurfaceForcing(panels2(1000), panels2(0.3), panels2(40), panels2(280))
    vdiff = (u = panels3(1), v = panels3(11), t = panels3(221),
             qv = ntuple(p -> fill(FT(0.001 * p), Nc, Nc, Nz), np))
    window = (m = panels3(10),
              am = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), np),
              bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), np),
              cm = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), np),
              ps = panels2(90000),
              dm = ntuple(_ -> zeros(FT, Nc, Nc, Nz), np),
              surface, vdiff)

    mktemp() do path, io
        close(io)
        writer = MD.open_streaming_cs_transport_binary(
            path, Nc, np, Nz, 1, vc; planet_radius = AtmosTransport.Parameters.EARTH_RADIUS,
            FT, dt_met_seconds = 3600, steps_per_window = 4, include_flux_delta = true,
            include_surface = true, include_gchp_vdiff = true, mass_basis = :dry,
            surface_pressure_basis = TotalSurfacePressure())
        MD.write_streaming_cs_window!(writer, window, Nc, np)
        MD.close_streaming_transport_binary!(writer)

        reader = MD.TransportBinaryReader(path; FT)
        try
            @test MD.has_surface(reader) && MD.has_vdiff_fields(reader)
            @test MD.binary_capabilities(reader).gchp_vdiff === true
            @test :vdiff_u in reader.header.payload_sections
            @test reader.header.raw_header["include_gchp_vdiff"] == true
            loaded = MD.load_window!(reader, 1)
            @test loaded.vdiff.u == vdiff.u && loaded.vdiff.v == vdiff.v
            @test loaded.vdiff.t == vdiff.t && loaded.vdiff.qv == vdiff.qv
            @test loaded.surface.pblh == surface.pblh                 # neighbours unaffected
        finally
            close(reader.io)
        end
    end
end
