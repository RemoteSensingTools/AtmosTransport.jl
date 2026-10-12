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
        window = (m = panels3(100), am, bm, cm, ps = panels2(90_000),
                  dm = panels3(0), surface, dkg)
        function write_source(path; horizontal_balance = nothing)
            extra_header = Dict{String, Any}(
                "date" => "2021-01-01", "adaptive_substeps" => true,
                "geos_global_mass_pin_enabled" => true,
                "mass_fix_target_ps_dry_pa" => 98_325.0)
            horizontal_balance === nothing ||
                (extra_header["horizontal_balance"] = horizontal_balance)
            writer = AtmosTransport.MetDrivers.open_streaming_cs_transport_binary(
                path, source_Nc, np, Nz, 1, vc;
                planet_radius = AtmosTransport.Parameters.EARTH_RADIUS,
                FT, steps_per_window = 3,
                include_flux_delta = true,
                include_surface = true,
                include_precomputed_dkg = true,
                mass_basis = :dry,
                extra_header)
            AtmosTransport.MetDrivers.write_streaming_cs_window!(writer, window, source_Nc, np)
            AtmosTransport.MetDrivers.set_streaming_steps_per_window_schedule!(writer, [3])
            AtmosTransport.MetDrivers.close_streaming_transport_binary!(writer)
            return path
        end

        source_path = write_source(joinpath(dir, "source_c6.bin"))
        output_path = joinpath(dir, "output_c2.bin")

        result = coarsen_nested_cs_transport_binary(
            source_path, output_path; target_Nc, mark_gated = true)
        @test result.experimental
        @test result.schedule == [1]
        @test result.output_bytes < result.input_bytes
        @test isfile(output_path * ".coarsen-gated")

        reader = AtmosTransport.MetDrivers.TransportBinaryReader(output_path; FT)
        try
            @test AtmosTransport.MetDrivers.binary_geometry(reader).Nc == target_Nc
            @test reader.header.steps_per_window_by_window == [1]
            @test reader.header.raw_header["experimental"] == true
            @test reader.header.raw_header["mass_fix_enabled"] === true
            @test reader.header.raw_header["mass_fix_target_ps_dry_pa"] == 98_325.0
            @test reader.header.raw_header["surface_pressure_basis"] == "dry"
            @test AtmosTransport.MetDrivers.surface_pressure_basis(reader) isa
                  DrySurfacePressure
            @test !haskey(reader.header.raw_header, "geos_global_mass_pin_enabled")
            @test reader.header.raw_header["validation_status"] ==
                  "testing_only_not_yet_scientifically_validated"
            # Legacy binaries predate this provenance field. The coarsener
            # accepts them and does not guess which balance produced them.
            @test !haskey(reader.header.raw_header, "horizontal_balance")
            loaded = AtmosTransport.MetDrivers.load_window!(reader, 1)
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
            driver = AtmosTransport.MetDrivers.TransportBinaryDriver(reader; Hp = 1)
            close(driver)
        finally
            isopen(reader.io) && close(reader)
        end

        @test_throws ArgumentError coarsen_nested_cs_transport_binary(
            source_path, joinpath(dir, "bad.bin"); target_Nc = 4)

        for balance in ("column", "none")
            tagged_source = write_source(joinpath(dir, "source_$(balance).bin");
                                         horizontal_balance = balance)
            tagged_output = joinpath(dir, "output_$(balance).bin")
            coarsen_nested_cs_transport_binary(tagged_source, tagged_output; target_Nc)
            tagged_reader = AtmosTransport.MetDrivers.TransportBinaryReader(tagged_output; FT)
            try
                @test tagged_reader.header.raw_header["horizontal_balance"] == balance
            finally
                close(tagged_reader)
            end
        end

        per_layer_source = write_source(joinpath(dir, "source_per_layer.bin");
                                        horizontal_balance = "per_layer")
        per_layer_output = joinpath(dir, "output_per_layer.bin")
        err = try
            coarsen_nested_cs_transport_binary(per_layer_source, per_layer_output; target_Nc)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test contains(err.msg, "regenerate the source binary with the current preprocessor")
        @test !isfile(per_layer_output)
    end
end
