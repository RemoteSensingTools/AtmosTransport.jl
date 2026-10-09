#!/usr/bin/env julia

using Test

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

@testset "GCHP VDIFF CS binary payload roundtrips" begin
    FT = Float32
    Nc = 2
    Nz = 3
    np = 6
    vc = AtmosTransport.HybridSigmaPressure(FT[0, 10, 50, 100],
                                            FT[0, 0.1, 0.5, 1])

    mktemp() do path, io
        close(io)
        writer = AtmosTransport.MetDrivers.open_streaming_cs_transport_binary(
            path, Nc, np, Nz, 1, vc; planet_radius = AtmosTransport.Parameters.EARTH_RADIUS,
            FT = FT,
            dt_met_seconds = 3600,
            steps_per_window = 4,
            include_flux_delta = true,
            include_surface = true,
            include_gchp_vdiff = true,
            mass_basis = :dry,
        )

        panels3(x) = ntuple(p -> fill(FT(x + p), Nc, Nc, Nz), np)
        qv_panels() = ntuple(p -> fill(FT(0.001 * p), Nc, Nc, Nz), np)
        panels2(x) = ntuple(p -> fill(FT(x + p), Nc, Nc), np)
        window = (
            m = panels3(10),
            am = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), np),
            bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), np),
            cm = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), np),
            ps = panels2(90000),
            dm = ntuple(_ -> zeros(FT, Nc, Nc, Nz), np),
            surface = AtmosTransport.MetDrivers.PBLSurfaceForcing(
                panels2(1000), panels2(0.3), panels2(40), panels2(280)),
            vdiff = (
                u = panels3(1),
                v = panels3(11),
                t = panels3(221),
                qv = qv_panels(),
            ),
        )
        AtmosTransport.MetDrivers.write_streaming_cs_window!(writer, window, Nc, np)
        AtmosTransport.MetDrivers.close_streaming_transport_binary!(writer)

        reader = AtmosTransport.MetDrivers.TransportBinaryReader(path; FT = FT)
        try
            @test AtmosTransport.MetDrivers.has_surface(reader)
            @test AtmosTransport.MetDrivers.has_vdiff_fields(reader)
            caps = AtmosTransport.MetDrivers.binary_capabilities(reader)
            @test caps.gchp_vdiff === true
            @test :vdiff_u in reader.header.payload_sections
            @test reader.header.raw_header["include_gchp_vdiff"] == true

            loaded = AtmosTransport.MetDrivers.load_window!(reader, 1)
            @test loaded.vdiff !== nothing
            @test loaded.vdiff.u[3] == window.vdiff.u[3]
            @test loaded.vdiff.v[4] == window.vdiff.v[4]
            @test loaded.vdiff.t[5] == window.vdiff.t[5]
            @test loaded.vdiff.qv[6] == window.vdiff.qv[6]
            @test loaded.surface.pblh[2] == window.surface.pblh[2]
        finally
            close(reader.io)
        end
    end
end
