using Test, AtmosTransport, NCDatasets, Dates
using AtmosTransport.Output: with_netcdf_lock
using AtmosTransport.Observations: build_observation_sampler, begin_observation_day!,
                                   observe_window_boundary!, finish_observations!,
                                   read_observation_requests, build_observation_set, TableSource,
                                   OCO2LiteSource, ObsPackSource, SoundingMode, SiteMode,
                                   AutoTableFormat, SiteCodeGrouping, EveryWindow, TimeRange,
                                   TimeList, QualityFlagFilter, QualityFlagValues, NoQualityFilter,
                                   SoundingNetCDFStream, append_soundings!, CellLocation,
                                   SoundingBatch
const O = AtmosTransport.Observations
include(joinpath(@__DIR__, "..", "fixtures", "observation_fixtures.jl"))
using .ObservationTestFixtures

const DAY = [Date(2021, 12, 2)]

@testset "site tables choose a schedule per row" begin
    mktempdir() do dir
        csv = joinpath(dir, "sites.csv")
        write(csv, """
            id,lat,lon,elevation,intake_height,altitude,start_time,end_time,times
            every,10,-100,100,40,,,,
            ranged,10,-100,100,,140,2021-12-02T01:00:00,2021-12-02T02:00:00,
            listed,10,-100,,10,,,,2021-12-02T00:30:00;2021-12-02T01:30:00
            """)
        _, sites = read_observation_requests(TableSource(csv, SiteMode(), AutoTableFormat()), 1, ORIGIN, DAY)
        by_id = Dict(s.id => s for s in sites)
        @test by_id["every"].schedule isa EveryWindow
        @test by_id["every"].intake_height_m == 40.0
        @test by_id["ranged"].schedule == TimeRange(3600.0, 7200.0)
        @test by_id["ranged"].intake_height_m == 40.0        # altitude 140 m asl − elevation 100 m
        @test by_id["listed"].schedule isa TimeList
        @test by_id["listed"].schedule.times_seconds == [1800.0, 5400.0]

        toml = joinpath(dir, "sites.toml")
        write(toml, """
            [[sites]]
            id = "flask"
            lat = 19.5
            lon = -155.6
            elevation = 3397.0
            intake_height = 40.0
            times = [2021-12-02T19:00:00, 2021-12-02T21:00:00]
            """)
        _, tsites = read_observation_requests(TableSource(toml, SiteMode(), AutoTableFormat()), 2, ORIGIN, DAY)
        @test tsites[1].schedule.times_seconds == [19 * 3600.0, 21 * 3600.0]

        # A time list becomes point events carrying the site's heights; ranges stay series.
        set = build_observation_set(Any[TableSource(csv, SiteMode(), AutoTableFormat()),
                                        TableSource(toml, SiteMode(), AutoTableFormat())] |>
                                    x -> AtmosTransport.Observations.AbstractObservationSource[x...], ORIGIN, DAY)
        @test sort([s.id for s in set.sites]) == ["every", "ranged"]
        @test [r.id for r in set.soundings] == ["listed", "listed", "flask", "flask"]
        @test set.soundings[3].intake_height_m == 40.0 && set.soundings[3].elevation_m == 3397.0

        bad_both = joinpath(dir, "both.csv")
        write(bad_both, "id,lat,lon,start_time,end_time,times\nx,1,2,2021-12-02T00:00:00,2021-12-02T01:00:00,2021-12-02T00:30:00\n")
        @test_throws ArgumentError read_observation_requests(TableSource(bad_both, SiteMode(), AutoTableFormat()), 1, ORIGIN, DAY)
        bad_half = joinpath(dir, "half.csv")
        write(bad_half, "id,lat,lon,start_time\nx,1,2,2021-12-02T00:00:00\n")
        @test_throws ArgumentError read_observation_requests(TableSource(bad_half, SiteMode(), AutoTableFormat()), 1, ORIGIN, DAY)
        # An altitude without elevation is read as it is; the sampler places it
        # through a ground elevation or refuses it (test_observation_placement.jl).
        alt_only = joinpath(dir, "alt.csv")
        write(alt_only, "id,lat,lon,altitude\nx,1,2,500\n")
        _, alt_sites = read_observation_requests(TableSource(alt_only, SiteMode(), AutoTableFormat()), 1, ORIGIN, DAY)
        @test isnan(alt_sites[1].intake_height_m) && alt_sites[1].altitude_m == 500.0 && isnan(alt_sites[1].elevation_m)
        @test_throws ArgumentError TimeRange(10.0, 5.0)
        @test_throws ArgumentError TimeList(Float64[])
    end
end

@testset "sampler honours schedules and writes intake values for point events" begin
    mktempdir() do dir
        csv = joinpath(dir, "sites.csv")
        write(csv, """
            id,lat,lon,intake_height,start_time,end_time,times
            always,10,-100,10,,,
            later,10,-100,10,2021-12-02T01:00:00,2021-12-02T02:00:00,
            flask,10,-100,100000,,,2021-12-02T00:30:00;2021-12-02T01:30:00
            """)
        spec = sampler_spec(dir; sources = Any[table_source(csv; mode = "sites")], tracers = ["co2"])
        model, _ = fake_ll_model()
        s = build_observation_sampler(spec, model.state, model.grid; origin = ORIGIN,
                                      window_seconds = (0.0, 7200.0), halo_width = 0)
        @test length(s.sites) == 2 && length(s.soundings) == 2
        begin_observation_day!(s, "20211202", 1)
        for t in (0.0, 3600.0, 7200.0)
            observe_window_boundary!(s, model.state, t; next_window_seconds = 3600.0)
        end
        finish_observations!(s); close(s)
        NCDataset(joinpath(dir, "obs_sites.nc"), "r") do ds
            @test ds["site_id"][:] == ["always", "later"]
            v = ds["co2_intake"][:, :]
            @test all(v[1, :] .≈ 400e-6)
            @test isnan(v[2, 1]) && v[2, 2] ≈ 400e-6 && v[2, 3] ≈ 400e-6      # outside / inside the range
            @test ds["intake_level"][2, 1] == 0
            @test ds["schedule_start"][2] == ORIGIN + Hour(1)
            @test isnan(ds["schedule_start"].var[1])
        end
        NCDataset(joinpath(dir, "obs_soundings.nc"), "r") do ds
            @test ds["id"][:] == ["flask", "flask"]
            @test ds["intake_height"][:] == [100000.0, 100000.0]
            @test all(ds["intake_level"][:] .== 1)                  # far above the column top
            @test all(ds["co2_intake"][:] .≈ 400e-6)
            @test all(ds["interp_flag"][:] .== 0)
        end
    end
end

@testset "events exactly at the final window end are kept" begin
    mktempdir() do dir
        csv = joinpath(dir, "s.csv")
        write(csv, "id,time,lat,lon\nend,2021-12-02T02:00:00,10,-100\nafter,2021-12-02T02:00:01,10,-100\n")
        spec = sampler_spec(dir; sources = Any[table_source(csv)])
        model, _ = fake_ll_model()
        s = build_observation_sampler(spec, model.state, model.grid; origin = ORIGIN,
                                      window_seconds = (0.0, 7200.0), halo_width = 0)
        @test [r.id for r in s.soundings] == ["end"]                 # inclusive end, `after` is outside
        begin_observation_day!(s, "20211202", 1)
        temperature = fill(250.0, 4, 3, 3)
        for t in (0.0, 3600.0, 7200.0)
            observe_window_boundary!(s, model.state, t; next_window_seconds = 3600.0, temperature)
        end
        finish_observations!(s); close(s)
        NCDataset(joinpath(dir, "obs_soundings.nc"), "r") do ds
            @test ds["id"][:] == ["end"]
            @test ds["sample_time_prev"][1] == ds["sample_time_next"][1] == ORIGIN + Hour(2)
            @test ds.attrib["n_after_end"] == 0
            # The held row keeps the layer temperature of its sample.
            @test ds["height_method"][1] == O.height_method_code(O.ProfileLayerTemperature([250.0]))
        end
    end
end

@testset "quality filters for Lite and MIP 10-second files" begin
    mktempdir() do dir
        path = joinpath(dir, "OCO2_b11.2_10sec_GOOD_r2.nc4")
        NCDataset(path, "c") do ds
            defDim(ds, "sounding_id", 4)
            defVar(ds, "sounding_id", Int64, ("sounding_id",))[:] = [20211202010000 + k for k in 1:4]
            defVar(ds, "time", Float64, ("sounding_id",); attrib = Dict("units" => "seconds since 1970-01-01 00:00:00"))[:] =
                [unix(ORIGIN + Hour(k)) for k in 1:4]
            defVar(ds, "latitude", Float32, ("sounding_id",))[:] = Float32[1, 2, 3, 4]
            defVar(ds, "longitude", Float32, ("sounding_id",))[:] = Float32[5, 6, 7, 8]
            defVar(ds, "assimilate_flag", Int8, ("sounding_id",))[:] = Int8[0, 1, 2, 1]
        end
        all_records, _ = read_observation_requests(OCO2LiteSource(path, NoQualityFilter()), 1, ORIGIN, DAY)
        @test length(all_records) == 4
        @test all_records[1].id == "20211202010001"
        # assimilate_flag is categorical (0 not assimilated, 1 assimilated, 2 withheld): select by value.
        assimilated, _ = read_observation_requests(OCO2LiteSource(path, QualityFlagValues("assimilate_flag", [1])), 1, ORIGIN, DAY)
        @test [r.id for r in assimilated] == ["20211202010002", "20211202010004"]
        not_withheld, _ = read_observation_requests(OCO2LiteSource(path, QualityFlagValues("assimilate_flag", [0, 1])), 1, ORIGIN, DAY)
        @test length(not_withheld) == 3
        @test_throws ArgumentError QualityFlagValues("assimilate_flag", Int[])
        @test_throws ArgumentError QualityFlagFilter("xco2_quality_flag", true)
        @test OCO2LiteSource(SubString(path, 1), NoQualityFilter()).path_template == path
        # The reader's window prefilter counts what it drops.
        wstats = O.Sources.ReadStats()
        inside, _ = read_observation_requests(OCO2LiteSource(path, NoQualityFilter()), 1, ORIGIN, DAY;
                                              stats = wstats, window = (0.0, 2.5 * 3600))
        @test length(inside) == 2 && wstats.outside_window == 2
        # The Lite default variable is absent from MIP files: a clear error, not silent acceptance.
        @test_throws ArgumentError read_observation_requests(OCO2LiteSource(path, QualityFlagFilter("xco2_quality_flag", 0)), 1, ORIGIN, DAY)
        stats = O.Sources.ReadStats()
        read_observation_requests(OCO2LiteSource(path, QualityFlagFilter("assimilate_flag", 0)), 1, ORIGIN, DAY; stats)
        @test stats.skipped_quality == 3
    end
end


@testset "daily files never collide and names are checked" begin
    mktempdir() do dir
        csv = joinpath(dir, "s.csv")
        write(csv, "id,time,lat,lon\na,2021-12-02T00:30:00,10,-100\n")
        model, _ = fake_ll_model()
        # A daily partition whose path has no day token and dateless binaries would reuse one name.
        cfg = Dict{String, Any}("path" => joinpath(dir, "snap.nc"), "hours" => [0.0], "split" => "daily",
                                "observations" => Dict{String, Any}("path" => joinpath(dir, "obs.nc"),
                                                                    "sources" => Any[table_source(csv)],
                                                                    "pressure_basis" => "dry"))
        s = build_observation_sampler(O.observation_output_spec(cfg), model.state, model.grid; origin = ORIGIN,
                                      window_seconds = (0.0, 86400.0), halo_width = 0)
        begin_observation_day!(s, "20211202", 1)
        @test_throws ArgumentError begin_observation_day!(s, "20211202", 2)
        close(s)
    end

end


@testset "repeated site ids merge time lists; other conflicts fail" begin
    mktempdir() do dir
        day1, day2 = joinpath(dir, "flask_20211202.csv"), joinpath(dir, "flask_20211203.csv")
        write(day1, "id,lat,lon,times\nbrw,71.3,-156.6,2021-12-02T05:00:00\n")
        write(day2, "id,lat,lon,times\nbrw,71.3,-156.6,2021-12-03T07:00:00\n")
        src = O.AbstractObservationSource[TableSource(joinpath(dir, "flask_{YYYYMMDD}.csv"), SiteMode(), AutoTableFormat())]
        set = quiet(() -> build_observation_set(src, ORIGIN, [Date(2021, 12, 2), Date(2021, 12, 3)]))
        @test [(r.id, r.time_seconds) for r in set.soundings] == [("brw", 5 * 3600.0), ("brw", 31 * 3600.0)]

        mixed = joinpath(dir, "mixed.csv")
        write(mixed, "id,lat,lon,start_time,end_time,times\n" *
                     "x,1,2,,,2021-12-02T05:00:00\nx,1,2,2021-12-02T00:00:00,2021-12-02T06:00:00,\n")
        msrc = O.AbstractObservationSource[TableSource(mixed, SiteMode(), AutoTableFormat())]
        @test_throws ArgumentError quiet(() -> build_observation_set(msrc, ORIGIN, DAY))
        same = joinpath(dir, "same.csv")
        write(same, "id,lat,lon\ny,1,2\ny,1,2\n")
        ssrc = O.AbstractObservationSource[TableSource(same, SiteMode(), AutoTableFormat())]
        @test length(quiet(() -> build_observation_set(ssrc, ORIGIN, DAY)).sites) == 1

        # A series never carries a time list.
        listed = O.SiteRequest("z", 1.0, 2.0, NaN, NaN, 1, TimeList([0.0]))
        @test_throws ArgumentError O.ObservationSet(ORIGIN, O.SoundingRequest[], [listed])
    end
end

@testset "table time cells are decoded, never guessed" begin
    mktempdir() do dir
        nc = joinpath(dir, "sites.nc")
        NCDataset(nc, "c") do ds
            defDim(ds, "site", 1)
            defVar(ds, "id", String, ("site",))[1] = "ranged"
            defVar(ds, "lat", Float64, ("site",))[:] = [10.0]
            defVar(ds, "lon", Float64, ("site",))[:] = [20.0]
            units = Dict("units" => "hours since 2021-12-02 00:00:00")
            defVar(ds, "start_time", Float64, ("site",); attrib = units)[:] = [1.0]
            defVar(ds, "end_time", Float64, ("site",); attrib = units)[:] = [2.0]
            defVar(ds, "extra_metadata", Float64, ("site",))[:] = [0.0]      # ignored in NetCDF
        end
        _, sites = read_observation_requests(TableSource(nc, SiteMode(), AutoTableFormat()), 1, ORIGIN, DAY)
        @test sites[1].schedule == TimeRange(3600.0, 7200.0)

        numeric = joinpath(dir, "numeric.csv")
        write(numeric, "id,time,lat,lon\na,1638406800,10,20\n")
        @test_throws ArgumentError read_observation_requests(TableSource(numeric, SoundingMode(), AutoTableFormat()), 1, ORIGIN, DAY)
        typo = joinpath(dir, "typo.csv")
        write(typo, "id,lat,lon,intake_hieght\na,10,20,30\n")
        @test_throws ArgumentError read_observation_requests(TableSource(typo, SiteMode(), AutoTableFormat()), 1, ORIGIN, DAY)
        schedule_on_event = joinpath(dir, "event.csv")
        write(schedule_on_event, "id,time,lat,lon,start_time\na,2021-12-02T01:00:00,10,20,2021-12-02T00:00:00\n")
        @test_throws ArgumentError read_observation_requests(TableSource(schedule_on_event, SoundingMode(), AutoTableFormat()), 1, ORIGIN, DAY)
    end
end


@testset "daily output paths are checked before the run" begin
    check(path_for, binaries) = AtmosTransport.Models.DrivenRunner._check_unique_day_paths(
        AtmosTransport.Output.DailyOutputFiles(), path_for, binaries, "[output].path")
    templated = (label, idx) -> AtmosTransport.Output._substitute_day_template("/o/obs_{date}.nc", label, idx)
    @test check(templated, ["/b/transport_20211202.bin", "/b/transport_20211203.bin"]) === nothing
    @test_throws ArgumentError check(templated, ["/b/a_20211202_00z.bin", "/b/b_20211202_12z.bin"])
    @test check((label, idx) -> AtmosTransport.Output._substitute_day_template("/o/obs_{date}_{day}.nc", label, idx),
                ["/b/a_20211202_00z.bin", "/b/b_20211202_12z.bin"]) === nothing
end
