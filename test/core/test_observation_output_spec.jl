using Test, AtmosTransport, NCDatasets, Dates
using AtmosTransport.Output: NoObservationOutput, NoObservationSampler, ObservationOutputSpec,
                             SingleOutputFile, DailyOutputFiles, LinearWindowInterpolation,
                             NearestWindowSampling, TableSource, OCO2LiteSource, ObsPackSource,
                             SoundingMode, SiteMode, SiteCodeGrouping, LocationGrouping,
                             AutoTableFormat, TOMLTableFormat
const O = AtmosTransport.Output
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

function _rejects(cfg, needle)
    result = @test_throws ArgumentError O.observation_output_spec(cfg)
    result isa Test.Pass && @test occursin(needle, sprint(showerror, result.value))
    return nothing
end

@testset "[output.observations] absent or disabled is a no-op" begin
    @test O.observation_output_spec(Dict{String, Any}()) isa NoObservationOutput
    @test O.observation_output_spec(Dict{String, Any}("path" => "x.nc")) isa NoObservationOutput
    spec = O.observation_output_spec(_obs_cfg(; enabled = false))
    @test spec isa NoObservationOutput
    @test !O.observations_enabled(spec)
    @test O.build_observation_sampler(spec) isa NoObservationSampler
    sampler = NoObservationSampler()
    @test O.observe_window_boundary!(sampler, nothing, 0.0) === nothing
    @test O.begin_observation_day!(sampler, "20211202", 1) === nothing
    @test O.finish_observations!(sampler) === nothing
    @test close(sampler) === nothing
    out = R.RunSnapshotOutput()
    @test out.observations isa NoObservationSampler
    @test close(out) === nothing
    @test R._install_observation_sampler!(out, _obs_cfg(; enabled = false), SingleOutputFile(), nothing,
                                          nothing; cfg = Dict{String, Any}(), binary_paths = String[],
                                          halo_width = 0) isa NoObservationSampler
    @test out.observations isa NoObservationSampler
end

@testset "defaults" begin
    spec = O.observation_output_spec(_obs_cfg())
    @test spec isa ObservationOutputSpec
    @test O.observations_enabled(spec)
    @test spec.partition isa SingleOutputFile
    @test spec.time_interpolation isa LinearWindowInterpolation
    @test O.time_interpolation_label(spec.time_interpolation) === :linear
    @test spec.tracers === nothing
    @test spec.write_profile_for_sites == false
    @test spec.layer_height_temperature_kelvin == 280.0
    @test spec.start_time === nothing
    @test spec.deflate_level == 0
    @test length(spec.sources) == 1
    source = spec.sources[1]
    @test source isa TableSource
    @test O.source_kind(source) === :table
    @test O.source_mode(source) isa SoundingMode
    @test O.mode_label(O.source_mode(source)) === :soundings
    @test source.format isa AutoTableFormat
    @test O.source_path_template(source) == "/tmp/points.csv"
end

@testset "explicit values" begin
    cfg = _obs_cfg(; time_interpolation = "nearest_window",
                   tracers = ["co2_natural", "co2_fossil"],
                   write_profile_for_sites = true, layer_height_temperature_kelvin = 265,
                   start_time = "2021-12-02T06:30:00Z", deflate_level = 4)
    cfg["split"] = "daily"
    spec = O.observation_output_spec(cfg)
    @test spec.partition isa DailyOutputFiles
    @test spec.time_interpolation isa NearestWindowSampling
    @test O.time_interpolation_label(spec.time_interpolation) === :nearest_window
    @test spec.tracers == [:co2_natural, :co2_fossil]
    @test spec.write_profile_for_sites
    @test spec.layer_height_temperature_kelvin == 265.0
    @test spec.start_time == DateTime(2021, 12, 2, 6, 30)
    @test spec.deflate_level == 4
    @test O.observation_output_spec(cfg; partition = SingleOutputFile()).partition isa SingleOutputFile
    for (value, expected) in (("2021-12-02", DateTime(2021, 12, 2)),
                              ("2021-12-02 12:00:00", DateTime(2021, 12, 2, 12)),
                              ("2021-12-02T12:00", DateTime(2021, 12, 2, 12)),
                              ("2021-12-02T06:30:00.250Z", DateTime(2021, 12, 2, 6, 30, 0, 250)),
                              (Date(2021, 12, 2), DateTime(2021, 12, 2)),
                              (DateTime(2021, 12, 2, 1), DateTime(2021, 12, 2, 1)))
        @test O.observation_output_spec(_obs_cfg(; start_time = value)).start_time == expected
    end
    @test O.observation_output_spec(_obs_cfg(; tracers = "co2")).tracers == [:co2]
    @test O.observation_output_spec(_obs_cfg(; tracers = "all")).tracers === nothing
    # The snapshot field parser shares the scalar-string convention.
    @test O.output_field_spec(Dict{String, Any}("tracers" => "co2")).tracers == [:co2]
end

@testset "sources" begin
    oco = Dict{String, Any}("kind" => "oco2_lite", "path" => "/tmp/oco2_LtCO2_{YYMMDD}_c.nc4",
                            "quality_flag_max" => 1)
    obspack = Dict{String, Any}("kind" => "obspack", "mode" => "sites",
                                "path" => "/tmp/obspack/*.nc", "site_grouping" => "location")
    table = Dict{String, Any}("kind" => "table", "mode" => "sites", "path" => "/tmp/sites.toml",
                              "format" => "toml")
    spec = O.observation_output_spec(_obs_cfg(; sources = Any[oco, obspack, table]))
    @test spec.sources[1] isa OCO2LiteSource
    @test spec.sources[1].quality_flag_max == 1
    @test O.source_kind(spec.sources[1]) === :oco2_lite
    @test O.source_mode(spec.sources[1]) isa SoundingMode
    @test spec.sources[2] isa ObsPackSource
    @test O.source_mode(spec.sources[2]) isa SiteMode
    @test spec.sources[2].site_grouping isa LocationGrouping
    @test O.source_kind(spec.sources[2]) === :obspack
    @test spec.sources[3] isa TableSource
    @test O.source_mode(spec.sources[3]) isa SiteMode
    @test spec.sources[3].format isa TOMLTableFormat
    default_grouping = Dict{String, Any}("kind" => "obspack", "mode" => "soundings", "path" => "/tmp/o.nc")
    @test O.observation_output_spec(_obs_cfg(; sources = Any[default_grouping])).sources[1].site_grouping isa
          SiteCodeGrouping
    oco_explicit = Dict{String, Any}("kind" => "oco2_lite", "mode" => "soundings", "path" => "/tmp/x.nc4")
    @test O.observation_output_spec(_obs_cfg(; sources = Any[oco_explicit])).sources[1].quality_flag_max == 0
    home = Dict{String, Any}("kind" => "table", "mode" => "sites", "path" => "~/points.csv")
    @test O.observation_output_spec(_obs_cfg(; sources = Any[home])).sources[1].path_template ==
          joinpath(homedir(), "points.csv")
end

@testset "rejections name the offending key" begin
    _rejects(_obs_cfg(; bogus = 1), "Unknown [output.observations] option(s): bogus")
    _rejects(Dict{String, Any}("observations" => 1), "[output.observations] must be a TOML table")
    missing_path = _obs_cfg(); delete!(missing_path["observations"], "path")
    _rejects(missing_path, "non-empty `path`")
    _rejects(_obs_cfg(; path = ""), "non-empty `path`")
    no_sources = _obs_cfg(); delete!(no_sources["observations"], "sources")
    _rejects(no_sources, "requires at least one")
    _rejects(_obs_cfg(; sources = Any[]), "non-empty array")
    _rejects(_obs_cfg(; sources = Any[1]), "entry 1 must be a table")
    _rejects(_obs_cfg(; sources = Any[Dict{String, Any}("path" => "/tmp/a")]), "requires a `kind`")
    _rejects(_obs_cfg(; sources = Any[Dict{String, Any}("kind" => "tccon", "path" => "/tmp/a")]),
             "kind must be one of")
    _rejects(_obs_cfg(; sources = Any[Dict{String, Any}("kind" => "table", "mode" => "sites",
                                                        "path" => "/tmp/a", "foo" => 1)]),
             "Unknown [[output.observations.sources]] entry 1 option(s): foo")
    _rejects(_obs_cfg(; sources = Any[Dict{String, Any}("kind" => "obspack", "path" => "/tmp/a")]),
             "requires `mode")
    _rejects(_obs_cfg(; sources = Any[Dict{String, Any}("kind" => "table", "mode" => "profiles",
                                                        "path" => "/tmp/a")]),
             "mode must be one of")
    _rejects(_obs_cfg(; sources = Any[Dict{String, Any}("kind" => "oco2_lite", "mode" => "sites",
                                                        "path" => "/tmp/a")]),
             "only support mode")
    _rejects(_obs_cfg(; sources = Any[Dict{String, Any}("kind" => "oco2_lite", "path" => "/tmp/a",
                                                        "quality_flag_max" => -1)]),
             "non-negative integer")
    _rejects(_obs_cfg(; sources = Any[Dict{String, Any}("kind" => "obspack", "mode" => "sites",
                                                        "path" => "/tmp/a", "site_grouping" => "x")]),
             "site_grouping must be one of")
    _rejects(_obs_cfg(; sources = Any[Dict{String, Any}("kind" => "table", "mode" => "sites",
                                                        "path" => "/tmp/a", "format" => "xlsx")]),
             "format must be one of")
    _rejects(_obs_cfg(; sources = Any[Dict{String, Any}("kind" => "table", "mode" => "sites",
                                                        "path" => "")]),
             "path must be a non-empty string")
    _rejects(_obs_cfg(; time_interpolation = "cubic"), "time_interpolation must be")
    _rejects(_obs_cfg(; tracers = String[]), "at least one tracer")
    _rejects(_obs_cfg(; tracers = "none"), "at least one tracer")
    _rejects(_obs_cfg(; tracers = 5), "tracer name")
    _rejects(_obs_cfg(; tracers = Any["co2", 3]), "tracer name")
    _rejects(_obs_cfg(; tracers = ["co2", "co2"]), "must not repeat")
    _rejects(_obs_cfg(; tracers = ""), "empty names")
    _rejects(_obs_cfg(; layer_height_temperature_kelvin = -1), "positive number")
    _rejects(_obs_cfg(; deflate_level = 10), "0..9")
    _rejects(_obs_cfg(; start_time = "yesterday"), "ISO-8601")
    _rejects(_obs_cfg(; start_time = "20211202"), "ISO-8601")   # compact date must not parse as year 20211202
    _rejects(_obs_cfg(; start_time = "2021-12"), "ISO-8601")
    _rejects(_obs_cfg(; start_time = 42), "date-time or string")
    _rejects(_obs_cfg(; enabled = "yes"), "must be true or false")
end

@testset "output paths follow the snapshot day template" begin
    single = O.observation_output_spec(_obs_cfg(; path = "/tmp/run/obs.nc"))
    @test O.observation_output_path(single, SoundingMode(), "20211202", 1) ==
          expand_data_path("/tmp/run/obs_soundings.nc")
    @test O.observation_output_path(single, SiteMode(), "", 3) == expand_data_path("/tmp/run/obs_sites.nc")
    # A single-file run with a day token names the pair after the first day.
    single_token = O.observation_output_spec(_obs_cfg(; path = "/tmp/run/obs_{YYYYMMDD}.nc"))
    @test O.observation_output_path(single_token, SoundingMode(), "20211202", 1) ==
          expand_data_path("/tmp/run/obs_20211202_soundings.nc")
    @test_throws MethodError O.observation_output_path(single, :profiles, "", 1)
    templated = _obs_cfg(; path = "/tmp/run/obs_{YYYYMMDD}.nc"); templated["split"] = "daily"
    daily = O.observation_output_spec(templated)
    @test O.observation_output_path(daily, SoundingMode(), "20211202", 1) ==
          expand_data_path("/tmp/run/obs_20211202_soundings.nc")
    plain = _obs_cfg(; path = "/tmp/run/obs.nc"); plain["split"] = "daily"
    daily_plain = O.observation_output_spec(plain)
    @test O.observation_output_path(daily_plain, SiteMode(), "20211202", 1) ==
          expand_data_path("/tmp/run/obs_20211202_sites.nc")
    @test O.observation_output_path(daily_plain, SiteMode(), "", 2) ==
          expand_data_path("/tmp/run/obs_002_sites.nc")
    snap = runtime_output_spec(Dict{String, Any}("path" => "/tmp/run/snap_{date}.nc",
                                                 "hours" => [0.0], "split" => "daily"), Float64)
    @test O.output_path_for_day(snap, "20211202", 1) == expand_data_path("/tmp/run/snap_20211202.nc")
    @test O.output_path_for_day(runtime_output_spec(Dict{String, Any}("path" => "/tmp/run/snap.nc",
                                                                      "hours" => [0.0],
                                                                      "split" => "daily"), Float64),
                                "", 7) == expand_data_path("/tmp/run/snap_007.nc")
end

@testset "validate_config preflights the observation table" begin
    input = Dict{String, Any}("binary_paths" => ["/nonexistent/transport_20211202.bin"])
    ok, errors = validate_config(Dict{String, Any}("input" => input, "output" => _obs_cfg(; bogus = true)))
    @test !ok
    @test any(e -> occursin("Unknown [output.observations] option(s): bogus", e), errors)

    shape = Dict{String, Any}("observations" => Dict{String, Any}("path" => "/tmp/o.nc", "sources" => "nope"))
    ok, errors = validate_config(Dict{String, Any}("input" => input, "output" => shape))
    @test any(e -> occursin("sources must be an array of tables", e), errors)

    ok, errors = validate_config(Dict{String, Any}("input" => input,
                                                   "output" => Dict{String, Any}("observations" => 42)))
    @test any(e -> occursin("[output.observations] must be a TOML table", e), errors)

    entry = Dict{String, Any}("observations" => Dict{String, Any}("path" => "/tmp/o.nc", "sources" => Any[3]))
    ok, errors = validate_config(Dict{String, Any}("input" => input, "output" => entry))
    @test any(e -> occursin("[[output.observations.sources]] entry 1 must be a TOML table", e), errors)

    # Disabled observations add no errors, and `[output].split` is not parsed on their behalf.
    off = _obs_cfg(; enabled = false); off["split"] = "bogus"
    ok, errors = validate_config(Dict{String, Any}("input" => input, "output" => off))
    @test !any(e -> occursin("observations", e) || occursin("split", e), errors)

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

function _synthetic_ll_binary(dir)
    mesh = LatLonMesh(; FT = Float64, Nx = 2, Ny = 2)
    grid = AtmosGrid(mesh, HybridSigmaPressure(Float64[0, 1], Float64[0, 1]), CPU(); FT = Float64)
    m = fill(2.0, 2, 2, 1)
    windows = [(m = m, am = zeros(Float64, 3, 2, 1), bm = zeros(Float64, 2, 3, 1),
                cm = zeros(Float64, 2, 2, 2), ps = fill(95000.0, 2, 2))]
    bin = joinpath(dir, "transport_20211202.bin")
    write_transport_binary(bin, grid, windows; FT = Float64, dt_met_seconds = 3600.0,
                           half_dt_seconds = 1800.0, steps_per_window = 1, mass_basis = :dry,
                           source_flux_sampling = :window_start_endpoint,
                           flux_sampling = :window_constant)
    return bin
end

@testset "runner: disabled observations leave gridded output unchanged" begin
    mktempdir() do dir
        bin = _synthetic_ll_binary(dir)
        function run_with(out, observations)
            cfg = Dict{String, Any}(
                "input" => Dict("binary_paths" => [bin]),
                "numerics" => Dict("float_type" => "Float64"),
                "run" => Dict("tracer_name" => "co2"),
                "init" => Dict("kind" => "uniform", "background" => 400e-6),
                "output" => Dict{String, Any}("path" => out, "hours" => [0.0, 1.0],
                                              "split" => "single"),
            )
            observations === nothing || (cfg["output"]["observations"] = observations)
            run_driven_simulation(cfg)
            return NCDataset(out, "r") do ds
                (Array(ds["co2"][:, :, :, :]), Array(ds["co2_column_mean"][:, :, :]),
                 Array(ds["time"][:]))
            end
        end
        reference = run_with(joinpath(dir, "reference.nc"), nothing)
        disabled = run_with(joinpath(dir, "disabled.nc"), Dict{String, Any}("enabled" => false))
        @test reference[1] == disabled[1]
        @test reference[2] == disabled[2]
        @test reference[3] == disabled[3]

        # A missing literal source file fails the run before any output is written.
        missing_source = Dict{String, Any}(
            "path" => joinpath(dir, "obs.nc"),
            "start_time" => "2021-12-02T00:00:00",
            "sources" => Any[Dict{String, Any}("kind" => "table", "mode" => "soundings",
                                               "path" => joinpath(dir, "points.csv"))])
        @test_throws ArgumentError run_with(joinpath(dir, "enabled.nc"), missing_source)
        @test !isfile(joinpath(dir, "obs_soundings.nc"))
    end
end
