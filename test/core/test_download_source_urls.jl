#!/usr/bin/env julia

using Dates
using Test
using TOML

using AtmosTransport

const DD = AtmosTransport.DataDownloads
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

@testset "GEOS-FP C720 public S3 URL construction" begin
    recipe_path = joinpath(REPO_ROOT, "config", "downloads", "geosfp_c720.toml")
    recipe = TOML.parsefile(recipe_path)
    config = DD.parse_download_config(recipe)

    @test config.source isa DD.GEOSFPSource
    @test config.protocol isa DD.HTTPProtocol
    @test config.protocol.base_url ==
          "https://geos-chem.s3.amazonaws.com/GEOS_C720/GEOS_FP_Raw"
    @test config.source.met_config["access"]["base_url"] ==
          config.protocol.base_url

    tasks = DD.build_tasks(config.source, config.protocol, [Date(2024, 6, 1)],
                           config.output, config.requests)
    @test length(tasks) == 24
    @test first(tasks).source_url ==
          "https://geos-chem.s3.amazonaws.com/GEOS_C720/GEOS_FP_Raw/" *
          "Y2024/M06/D01/" *
          "GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.20240601_0030.V01.nc4"
    @test last(tasks).source_url ==
          "https://geos-chem.s3.amazonaws.com/GEOS_C720/GEOS_FP_Raw/" *
          "Y2024/M06/D01/" *
          "GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.20240601_2330.V01.nc4"
    @test all(!occursin("/tavg_1hr_ctm_c0720_v72/Y", task.source_url)
              for task in tasks)
end
