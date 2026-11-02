#!/usr/bin/env julia

using Test
using AtmosTransport
using .AtmosTransport.State: PrecomputedCSDkgField, panel_field
using .AtmosTransport.State.Fields: refresh_precomputed_cs_dkg_cache!, field_value

@testset "Exact TM5 dkg binary and runtime path" begin
    FT = Float64
    Nc, Nz, np = 2, 4, 6
    vc = HybridSigmaPressure(FT[0, 20, 100, 400, 1000], FT[0, 0.05, 0.3, 0.7, 1])
    panels3(x) = ntuple(p -> fill(FT(x + p), Nc, Nc, Nz), np)
    panels2(x) = ntuple(p -> fill(FT(x + p), Nc, Nc), np)

    mktemp() do path, io
        close(io)
        writer = AtmosTransport.MetDrivers.open_streaming_cs_transport_binary(
            path, Nc, np, Nz, 1, vc; planet_radius = AtmosTransport.Parameters.EARTH_RADIUS, FT, include_flux_delta = true,
            include_precomputed_dkg = true, mass_basis = :dry,
            surface_pressure_basis = TotalSurfacePressure())
        dkg = ntuple(p -> FT[(k == Nz ? 0 : 100p + 10k + i + j)
                             for i in 1:Nc, j in 1:Nc, k in 1:Nz], np)
        window = (m = panels3(1e6),
                  am = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), np),
                  bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), np),
                  cm = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), np),
                  ps = panels2(9e4), dm = panels3(0), dkg)
        AtmosTransport.MetDrivers.write_streaming_cs_window!(writer, window, Nc, np)
        AtmosTransport.MetDrivers.close_streaming_transport_binary!(writer)

        reader = AtmosTransport.MetDrivers.TransportBinaryReader(path; FT)
        try
            @test :dkg in reader.header.payload_sections
            @test reader.header.raw_header["precomputed_dkg_payload"] ==
                  "tm5_bldiff_interface_dkg_dry_v1"
            loaded = AtmosTransport.MetDrivers.load_window!(reader, 1)
            @test loaded.dkg == dkg

            cache = ntuple(_ -> zeros(FT, Nc, Nc, Nz), np)
            field = PrecomputedCSDkgField(cache)
            refresh_precomputed_cs_dkg_cache!(field, loaded.dkg)
            @test field_value(panel_field(field, 4), (2, 1, 3)) == dkg[4][2, 1, 3]
            @test_throws ArgumentError refresh_precomputed_cs_dkg_cache!(field, nothing)

            spec = AtmosTransport.Models.diffusion_spec(Dict("kind" => "tm5_dkg"))
            op = AtmosTransport.Models.materialize(
                spec, AtmosTransport.Models.CubedSphereRuntimeRecipeStyle(), FT, reader)
            @test op.kz_field isa PrecomputedCSDkgField
            driver = AtmosTransport.MetDrivers.TransportBinaryDriver(reader; Hp = 1)
            @test AtmosTransport.MetDrivers.supports_diffusion(driver)

            # The direct-Dkg runtime does not consume geometric dz. Its
            # specialized window refresh must leave layer_thickness untouched
            # (and therefore must not reconstruct dry/total pressure).
            runtime_window = load_transport_window(driver, 1)
            mesh = driver_grid(driver).horizontal
            tracer = ntuple(p -> runtime_window.air_mass[p] .* FT(1e-6), np)
            state = CubedSphereState(DryBasis, mesh, runtime_window.air_mass;
                                     CO2 = tracer)
            fluxes = allocate_face_fluxes(mesh, Nz; FT, basis = DryBasis)
            model = TransportModel(state, fluxes, driver_grid(driver), UpwindScheme();
                                   diffusion = op)
            foreach(x -> fill!(x, FT(NaN)), model.workspace.diffusion_ws.layer_thickness)
            sim = DrivenSimulation(model, driver; start_window = 1, stop_window = 1)
            @test all(panel -> all(isnan, panel),
                      sim.model.workspace.diffusion_ws.layer_thickness)
        finally
            close(reader.io)
        end
    end

    @test_throws ArgumentError AtmosTransport.MetDrivers.open_streaming_cs_transport_binary(
        tempname(), Nc, np, Nz, 1, vc; planet_radius = AtmosTransport.Parameters.EARTH_RADIUS, FT,
        include_precomputed_dkg = true, mass_basis = :moist,
        surface_pressure_basis = TotalSurfacePressure())
end
