#!/usr/bin/env julia

using Test

using AtmosTransport
using .AtmosTransport.Preprocessing: coarsen_nested_cs_transport_binary

@testset "Experimental nested CS binary coarsener" begin
    FT = Float64
    source_Nc, target_Nc, Nz, np = 6, 2, 2, 6
    vc = HybridSigmaPressure(FT[0, 500, 2000], FT[0, 0.4, 1.0])
    panels3(value) = ntuple(_ -> fill(FT(value), source_Nc, source_Nc, Nz), np)
    panels2(value) = ntuple(_ -> fill(FT(value), source_Nc, source_Nc), np)
    am = ntuple(_ -> fill(FT(40), source_Nc + 1, source_Nc, Nz), np)
    bm = ntuple(_ -> zeros(FT, source_Nc, source_Nc + 1, Nz), np)
    cm = ntuple(_ -> zeros(FT, source_Nc, source_Nc, Nz + 1), np)
    dkg = ntuple(p -> FT[p + k + i / 10 + j / 100
                          for i in 1:source_Nc, j in 1:source_Nc, k in 1:Nz], np)
    surface = (
        pblh = ntuple(p -> FT[100p + i + 2j for i in 1:source_Nc, j in 1:source_Nc], np),
        ustar = panels2(0.4), hflux = panels2(50), t2m = panels2(280),
    )

    mktempdir() do dir
        source_path = joinpath(dir, "source_c6.bin")
        output_path = joinpath(dir, "output_c2.bin")
        writer = AtmosTransport.MetDrivers.open_streaming_cs_transport_binary(
            source_path, source_Nc, np, Nz, 1, vc;
            FT, steps_per_window = 3,
            include_flux_delta = true,
            include_surface = true,
            include_precomputed_dkg = true,
            mass_basis = :dry,
            extra_header = Dict("date" => "2021-01-01",
                                "adaptive_substeps" => true))
        window = (m = panels3(100), am, bm, cm, ps = panels2(90_000),
                  dm = panels3(0), surface, dkg)
        AtmosTransport.MetDrivers.write_streaming_cs_window!(writer, window, source_Nc, np)
        AtmosTransport.MetDrivers.set_streaming_steps_per_window_schedule!(writer, [3])
        AtmosTransport.MetDrivers.close_streaming_transport_binary!(writer)

        result = coarsen_nested_cs_transport_binary(
            source_path, output_path; target_Nc, mark_gated = true)
        @test result.experimental
        @test result.schedule == [1]
        @test result.output_bytes < result.input_bytes
        @test isfile(output_path * ".coarsen-gated")

        reader = AtmosTransport.MetDrivers.CubedSphereBinaryReader(output_path; FT)
        try
            @test reader.header.Nc == target_Nc
            @test reader.header.steps_per_window_by_window == [1]
            @test reader.header.raw_header["experimental"] == true
            @test reader.header.raw_header["validation_status"] ==
                  "testing_only_not_yet_scientifically_validated"
            loaded = AtmosTransport.MetDrivers.load_cs_window(reader, 1)
            @test all(panel -> all(==(FT(900)), panel), loaded.m)
            @test all(panel -> all(==(FT(360)), panel), loaded.am)
            @test all(panel -> all(iszero, panel), loaded.bm)
            @test all(panel -> all(iszero, panel), loaded.cm)
            @test all(panel -> all(x -> isapprox(x, FT(90_000); rtol = 10eps(FT)), panel),
                      loaded.ps)
            for p in 1:np, k in 1:Nz, j in 1:target_Nc, i in 1:target_Nc
                expected = zero(FT)
                for jj in ((j - 1) * 3 + 1):(j * 3), ii in ((i - 1) * 3 + 1):(i * 3)
                    expected += dkg[p][ii, jj, k]
                end
                @test loaded.dkg[p][i, j, k] == expected
            end
            driver = AtmosTransport.MetDrivers.CubedSphereTransportDriver(reader; Hp = 1)
            close(driver)
        finally
            isopen(reader.io) && close(reader)
        end

        @test_throws ArgumentError coarsen_nested_cs_transport_binary(
            source_path, joinpath(dir, "bad.bin"); target_Nc = 4)
    end
end
