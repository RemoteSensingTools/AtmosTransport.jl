using Test, AtmosTransport, NCDatasets, Dates
using AtmosTransport.Output: SingleOutputFile, DailyOutputFiles
using AtmosTransport.Observations: NoObservationOutput, NoObservationSampler, ObservationOutputSpec,
                                   LinearWindowInterpolation, NearestWindowSampling, TableSource,
                                   OCO2LiteSource, ObsPackSource, SoundingMode, SiteMode,
                                   SiteCodeGrouping, LocationGrouping, AutoTableFormat,
                                   TOMLTableFormat
const O = AtmosTransport.Observations
const R = AtmosTransport.Models.DrivenRunner

function _obs_cfg(; kwargs...)
    obs = Dict{String, Any}(
        "path" => "/tmp/obs.nc",
        "sources" => Any[Dict{String, Any}("kind" => "table", "mode" => "soundings",
                                           "path" => "/tmp/points.csv")],
    )
    for (k, v) in kwargs
        obs[String(k)] = v
    end
    return Dict{String, Any}("path" => "/tmp/snap.nc", "hours" => [0.0], "observations" => obs)
end

@testset "observation UTC origin" begin
    for (value, expected) in (("2021-12-02", DateTime(2021, 12, 2)),
                              ("2021-12-02 12:00:00", DateTime(2021, 12, 2, 12)),
                              ("2021-12-02T12:00", DateTime(2021, 12, 2, 12)),
                              ("2021-12-02T06:30:00.250Z", DateTime(2021, 12, 2, 6, 30, 0, 250)),
                              (Date(2021, 12, 2), DateTime(2021, 12, 2)),
                              (DateTime(2021, 12, 2, 1), DateTime(2021, 12, 2, 1)))
        @test O.observation_output_spec(_obs_cfg(; start_time = value)).start_time == expected
    end
end

@testset "unknown observation keys are rejected" begin
    @test_throws ArgumentError O.observation_output_spec(_obs_cfg(; bogus = true))
    @test_throws ArgumentError O.observation_output_spec(_obs_cfg(; start_time = "20211202"))
    @test_throws ArgumentError O.observation_output_spec(_obs_cfg(; sources = Any[Dict("kind" => "unknown")]))
end

@testset "default quality filter and coordinate order (D28)" begin
    source = Dict{String, Any}("kind" => "oco2_lite", "path" => "/tmp/x.nc4")
    filter = only(O.observation_output_spec(_obs_cfg(; sources = Any[source])).sources).quality_filter
    @test filter == O.QualityFlagFilter("xco2_quality_flag", 0)
    H, P = O.HeightAboveGroundCoordinate(), O.PressureCoordinate()
    placement(; kwargs...) = O.observation_output_spec(_obs_cfg(; kwargs...)).placement
    # Default: height above the true ground first, then pressure (D28); total pressures.
    @test placement().coordinates == (H, P) == O.DEFAULT_VERTICAL_COORDINATES
    @test placement().pressure_basis isa O.TotalFromHumidity
    for (names, expected) in ((["height_above_ground", "pressure"], (H, P)),
                              (["pressure", "height_above_ground"], (P, H)),
                              (["pressure"], (P,)), (["height_above_ground"], (H,)))
        @test placement(; vertical_coordinates = names).coordinates == expected
    end
    @test placement(; pressure_basis = "dry").pressure_basis isa O.DryInterfaces
    @test placement(; pressure_basis = "total").pressure_basis isa O.TotalFromHumidity
end

@testset "explicit binaries need an observation origin" begin
    input = Dict{String, Any}("binary_paths" => ["/nonexistent/transport_20211202.bin"])
    # Enabled: an explicit binary list needs start_time.
    ok, errors = validate_config(Dict{String, Any}("input" => input, "output" => _obs_cfg()))
    @test !ok
    @test any(e -> occursin("absolute run origin", e), errors)
    with_origin = _obs_cfg(; start_time = "2021-12-02T00:00:00")
    ok, errors = validate_config(Dict{String, Any}("input" => input, "output" => with_origin))
    @test !any(e -> occursin("absolute run origin", e), errors)
    dated = Dict{String, Any}("folder" => "/nonexistent", "start_date" => "2021-12-02",
                              "end_date" => "2021-12-02")
    ok, errors = validate_config(Dict{String, Any}("input" => dated, "output" => _obs_cfg()))
    @test !any(e -> occursin("absolute run origin", e), errors)
end
