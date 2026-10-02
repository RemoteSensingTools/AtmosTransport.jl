using Test, AtmosTransport, NCDatasets, Dates
using AtmosTransport.Output: OCO2LiteSource, ObsPackSource, TableSource, SoundingMode, SiteMode,
                             SiteCodeGrouping, LocationGrouping, AutoTableFormat, CSVTableFormat,
                             TOMLTableFormat, NetCDFTableFormat, SoundingRequest, SiteRequest,
                             ObservationSet, read_observation_requests, build_observation_set,
                             expand_observation_paths
const O = AtmosTransport.Output

const ORIGIN = DateTime(2021, 12, 2)
unix(dt) = datetime2unix(dt)

# -- synthetic fixtures -----------------------------------------------------

function write_oco2_lite(path; n = 5, bad_flag_at = 3, fill_lat_at = 4, with_flag = true)
    NCDataset(path, "c") do ds
        defDim(ds, "sounding_id", n)
        ids = defVar(ds, "sounding_id", UInt64, ("sounding_id",))
        ids[:] = UInt64[2021120206301500 + k for k in 1:n]
        t = defVar(ds, "time", Float64, ("sounding_id",);
                   attrib = Dict("units" => "seconds since 1970-01-01 00:00:00", "missing_value" => -999999.0))
        t[:] = [unix(ORIGIN + Hour(6) + Minute(30) + Second(k)) for k in 1:n]
        lat = defVar(ds, "latitude", Float32, ("sounding_id",); attrib = Dict("missing_value" => -999999.0f0))
        lon = defVar(ds, "longitude", Float32, ("sounding_id",); attrib = Dict("missing_value" => -999999.0f0))
        lat[:] = Float32[10 + k for k in 1:n]
        lon[:] = Float32[-100 + k for k in 1:n]
        lat.var[fill_lat_at] = -999999.0f0
        if with_flag
            flag = defVar(ds, "xco2_quality_flag", Int8, ("sounding_id",))
            flag[:] = zeros(Int8, n)
            flag[bad_flag_at] = Int8(1)
        end
        xco2 = defVar(ds, "xco2", Float32, ("sounding_id",))
        xco2[:] = fill(410.0f0, n)
    end
    return path
end

function write_obspack(path; site_code = "mlo", intakes = [40.0, 40.0, 40.0], lat = 19.536, lon = -155.576,
                       elevation = 3397.0, with_attrs = true, char_ids = true, with_intake = true,
                       times = nothing, time_units = "seconds since 1970-01-01T00:00:00Z")
    n = length(intakes)
    NCDataset(path, "c") do ds
        defDim(ds, "obs", n)
        t = defVar(ds, "time", Float64, ("obs",); attrib = Dict("units" => time_units, "_FillValue" => -1.0e34))
        t[:] = times === nothing ? [unix(ORIGIN + Hour(k)) for k in 1:n] : times
        # NetCDF forbids a mistyped _FillValue, but `missing_value` is a plain attribute:
        # a Float64 fill on Float32 data must still be recognised.
        latv = defVar(ds, "latitude", Float32, ("obs",); attrib = Dict("missing_value" => -1.0e34))
        latv[:] = fill(Float32(lat), n)
        defVar(ds, "longitude", Float64, ("obs",))[:] = fill(lon, n)
        defVar(ds, "elevation", Float64, ("obs",))[:] = fill(elevation, n)
        with_intake && (defVar(ds, "intake_height", Float64, ("obs",))[:] = intakes)
        defVar(ds, "altitude", Float64, ("obs",))[:] = elevation .+ intakes
        defVar(ds, "value", Float64, ("obs",))[:] = fill(415e-6, n)
        if char_ids
            defDim(ds, "string_length", 24)
            idv = defVar(ds, "obspack_id", Char, ("string_length", "obs"))
            for k in 1:n
                s = "$(site_code)~$(k)"
                idv[:, k] = collect(s * "\0"^(24 - length(s)))
            end
        else
            defVar(ds, "obspack_id", String, ("obs",))[:] = ["$(site_code)~$(k)" for k in 1:n]
        end
        if with_attrs
            ds.attrib["site_code"] = site_code
            ds.attrib["site_latitude"] = lat
            ds.attrib["site_longitude"] = lon
            ds.attrib["site_elevation"] = elevation
        end
    end
    return path
end

# -- tests ------------------------------------------------------------------

@testset "path templates expand per day with basename wildcards" begin
    mktempdir() do dir
        for name in ("oco2_LtCO2_211202_B11.nc4", "oco2_LtCO2_211203_B11.nc4", "oco2_LtCO2_211203_B10.nc4", "other.txt")
            touch(joinpath(dir, name))
        end
        dates = [Date(2021, 12, 2), Date(2021, 12, 3), Date(2021, 12, 4)]
        found = expand_observation_paths(joinpath(dir, "oco2_LtCO2_{YYMMDD}_B11.nc4"), dates)
        @test basename.(found) == ["oco2_LtCO2_211202_B11.nc4", "oco2_LtCO2_211203_B11.nc4"]
        found = expand_observation_paths(joinpath(dir, "oco2_LtCO2_{YYMMDD}_*.nc4"), dates)
        @test basename.(found) == ["oco2_LtCO2_211202_B11.nc4", "oco2_LtCO2_211203_B10.nc4",
                                   "oco2_LtCO2_211203_B11.nc4"]
        found = expand_observation_paths(joinpath(dir, "oco2_LtCO2_*_B1?.nc4"), dates)
        @test length(found) == 3
        @test expand_observation_paths(joinpath(dir, "*.txt"), dates) == [joinpath(dir, "other.txt")]
        @test expand_observation_paths(joinpath(dir, "{YYYY}/{MM}/{DD}/x_{date}.nc"), dates) == String[]
        @test expand_observation_paths(joinpath(dir, "other.txt"), dates) == [joinpath(dir, "other.txt")]
        @test_throws ArgumentError expand_observation_paths(joinpath(dir, "missing.txt"), dates)
        @test_throws ArgumentError expand_observation_paths(joinpath(dir, "*", "x.nc"), dates)
        # A wildcard without date tokens that matches nothing is a user error ...
        @test_throws ArgumentError expand_observation_paths(joinpath(dir, "nope", "*.nc"), dates)
        @test_throws ArgumentError expand_observation_paths(joinpath(dir, "typo_*.nc4"), dates)
        # ... while a templated pattern may legitimately have no files on some days.
        @test expand_observation_paths(joinpath(dir, "nope", "{YYYYMMDD}_*.nc"), dates) == String[]
        @test O._substitute_date_tokens("{YYYY}/{YYYYMMDD}/{YYMMDD}/{MM}-{DD}/{date}", Date(2021, 12, 3)) ==
              "2021/20211203/211203/12-03/20211203"
    end
end

@testset "OCO-2 Lite soundings" begin
    mktempdir() do dir
        write_oco2_lite(joinpath(dir, "oco2_LtCO2_211202_c.nc4"))
        source = OCO2LiteSource(joinpath(dir, "oco2_LtCO2_{YYMMDD}_c.nc4"), 0)
        soundings, sites = read_observation_requests(source, 1, ORIGIN, [Date(2021, 12, 2), Date(2021, 12, 3)])
        @test isempty(sites)
        # 5 soundings minus the flagged one minus the fill-latitude one
        @test length(soundings) == 3
        @test [s.id for s in soundings] == ["2021120206301501", "2021120206301502", "2021120206301505"]
        @test soundings[1].time_seconds == 6 * 3600 + 30 * 60 + 1
        @test soundings[1].lon == -99.0 && soundings[1].lat == 11.0
        @test all(s.source == 1 for s in soundings)
        lenient = OCO2LiteSource(joinpath(dir, "oco2_LtCO2_{YYMMDD}_c.nc4"), 1)
        @test length(first(read_observation_requests(lenient, 2, ORIGIN, [Date(2021, 12, 2)]))) == 4
        # A different origin shifts times by whole days.
        shifted, _ = read_observation_requests(source, 1, ORIGIN - Day(1), [Date(2021, 12, 2)])
        @test shifted[1].time_seconds == soundings[1].time_seconds + 86400
        # No file for the requested day: empty (with a warning), not an error.
        @test (@test_logs (:warn, r"matched no files") isempty(first(read_observation_requests(source, 1, ORIGIN, [Date(2021, 12, 9)]))))
        # The quality flag is required: silently passing everything is not an option.
        write_oco2_lite(joinpath(dir, "noflag_211202.nc4"); with_flag = false)
        @test_throws ArgumentError read_observation_requests(OCO2LiteSource(joinpath(dir, "noflag_{YYMMDD}.nc4"), 0), 1, ORIGIN, [Date(2021, 12, 2)])
    end
end

@testset "NetCDF fill, packing, and CF origin handling" begin
    mktempdir() do dir
        path = joinpath(dir, "packed.nc")
        NCDataset(path, "c") do ds
            defDim(ds, "obs", 3)
            lat = defVar(ds, "lat", Int16, ("obs",); attrib = Dict("scale_factor" => 0.01, "add_offset" => 0.0, "_FillValue" => Int16(-9999)))
            lat.var[:] = Int16[1050, -9999, 4525]
            t = defVar(ds, "time", Float32, ("obs",); attrib = Dict("units" => "hours since 2021-12-01 00:00:00 +00:00", "missing_value" => -999999.0))
            t.var[:] = Float32[24.0, 25.0, -999999.0]
        end
        NCDataset(path, "r") do ds
            lat = O._raw_numeric(ds["lat"], "packed")
            @test lat[1] == 10.5 && isnan(lat[2]) && lat[3] == 45.25
            t = O._time_unix_seconds(ds["time"], "packed")
            @test t[1] == unix(DateTime(2021, 12, 2))
            @test t[2] == unix(DateTime(2021, 12, 2, 1))
            @test isnan(t[3])
        end
        for units in ("seconds since 1970-01-01 00:00:00", "seconds since 1970-01-01T00:00:00Z",
                      "seconds since 1970-01-01 00:00:00 UTC", "seconds since 1970-01-01 00:00:00 +0:00",
                      "seconds since 1970-01-01 00:00:00 -00:00")
            @test O._cf_unix_seconds([0.0], units, "u") == [0.0]
        end
        @test O._cf_unix_seconds([1.0], "days since 1900-1-1", "u") == [unix(DateTime(1900, 1, 2))]
        @test_throws ArgumentError O._cf_unix_seconds([0.0], "seconds since 1970-01-01 00:00:00 +01:00", "u")
        @test_throws ArgumentError O._cf_unix_seconds([0.0], "fortnights since 1970-01-01", "u")
        @test_throws ArgumentError O._cf_unix_seconds([0.0], "seconds", "u")
    end
end

@testset "ObsPack sites and soundings" begin
    mktempdir() do dir
        write_obspack(joinpath(dir, "co2_mlo_surface-insitu_1_allvalid.nc"))
        write_obspack(joinpath(dir, "co2_lef_tower-insitu_1_allvalid.nc"); site_code = "lef",
                      intakes = [30.0, 122.0, 396.0, 396.0], lat = 45.9459, lon = -90.2731, elevation = 472.0)
        write_obspack(joinpath(dir, "co2_brw_surface-flask_1_representative.nc"); site_code = "brw",
                      intakes = [NaN, NaN], lat = 71.323, lon = -156.611, elevation = 11.0,
                      with_attrs = false, char_ids = false)
        pattern = joinpath(dir, "co2_*_*.nc")
        dates = [Date(2021, 12, 2)]

        by_code = ObsPackSource(pattern, SiteMode(), SiteCodeGrouping())
        soundings, sites = read_observation_requests(by_code, 3, ORIGIN, dates)
        @test isempty(soundings)
        @test sort([s.id for s in sites]) == ["co2_brw_surface-flask_1_representative",
                                               "co2_lef_tower-insitu_1_allvalid_122magl",
                                               "co2_lef_tower-insitu_1_allvalid_30magl",
                                               "co2_lef_tower-insitu_1_allvalid_396magl",
                                               "co2_mlo_surface-insitu_1_allvalid"]
        by_id = Dict(s.id => s for s in sites)
        mlo = by_id["co2_mlo_surface-insitu_1_allvalid"]
        @test (mlo.lon, mlo.lat, mlo.elevation_m, mlo.intake_height_m) == (-155.576, 19.536, 3397.0, 40.0)
        @test mlo.source == 3
        brw = by_id["co2_brw_surface-flask_1_representative"]   # no site attributes: records, intake unknown
        @test brw.lon == -156.611
        @test brw.lat ≈ 71.323 atol = 1e-4     # Float32 latitude storage
        @test brw.elevation_m == 11.0
        @test isnan(brw.intake_height_m)
        lef = by_id["co2_lef_tower-insitu_1_allvalid_396magl"]
        @test lef.lat == 45.9459 && lef.lon == -90.2731
        @test lef.intake_height_m == 396.0

        by_location = ObsPackSource(pattern, SiteMode(), LocationGrouping())
        _, located = read_observation_requests(by_location, 1, ORIGIN, dates)
        @test length(located) == 5
        @test any(s -> s.id == "lef_45.95N_-90.27E_396magl", located)
        # Without a site_code attribute the dataset name stands in for the code.
        @test any(s -> s.id == "co2_brw_surface-flask_1_representative_71.32N_-156.61E_surface", located)

        events = ObsPackSource(joinpath(dir, "co2_mlo_*.nc"), SoundingMode(), SiteCodeGrouping())
        records, none = read_observation_requests(events, 2, ORIGIN, dates)
        @test isempty(none)
        @test [r.id for r in records] == ["mlo~1", "mlo~2", "mlo~3"]
        @test [r.time_seconds for r in records] == [3600.0, 7200.0, 10800.0]
        @test records[1].lon == -155.576

        # Fill time (Float64 _FillValue on Float64 time) drops the record; without an
        # intake_height variable the intake falls back to altitude - elevation.
        write_obspack(joinpath(dir, "co2_alt_surface-insitu_1_allvalid.nc"); site_code = "alt",
                      intakes = [10.0, 10.0], with_intake = false,
                      times = [unix(ORIGIN + Hour(1)), -1.0e34])
        alt_events = ObsPackSource(joinpath(dir, "co2_alt_*.nc"), SoundingMode(), SiteCodeGrouping())
        alt_records, _ = read_observation_requests(alt_events, 1, ORIGIN, dates)
        @test length(alt_records) == 1
        _, alt_sites = read_observation_requests(ObsPackSource(joinpath(dir, "co2_alt_*.nc"), SiteMode(), SiteCodeGrouping()), 1, ORIGIN, dates)
        @test alt_sites[1].intake_height_m == 10.0
    end
end

@testset "generic tables: CSV, TOML, NetCDF" begin
    mktempdir() do dir
        csv = joinpath(dir, "points.csv")
        write(csv, """
            # id, time (UTC), lat, lon
            id,time,lat,lon
            a,2021-12-02T01:30:00,10.5,-20.25
            b,2021-12-02 02:00:00Z,-33,150
            """)
        source = TableSource(csv, SoundingMode(), AutoTableFormat())
        soundings, _ = read_observation_requests(source, 1, ORIGIN, [Date(2021, 12, 2)])
        @test [s.id for s in soundings] == ["a", "b"]
        @test soundings[1].time_seconds == 5400.0
        @test soundings[2].time_seconds == 7200.0
        @test (soundings[1].lon, soundings[1].lat) == (-20.25, 10.5)
        @test (soundings[2].lon, soundings[2].lat) == (150.0, -33.0)

        sites_csv = joinpath(dir, "sites.csv")
        write(sites_csv, "id,latitude,longitude,intake_height,elevation\nmlo,19.536,-155.576,40,3397\nspo,-89.98,-24.8,,2810\n")
        _, sites = read_observation_requests(TableSource(sites_csv, SiteMode(), CSVTableFormat()), 2, ORIGIN, [Date(2021, 12, 2)])
        @test [s.id for s in sites] == ["mlo", "spo"]
        @test sites[1].intake_height_m == 40.0 && sites[1].elevation_m == 3397.0
        @test isnan(sites[2].intake_height_m) && sites[2].elevation_m == 2810.0
        @test all(s.source == 2 for s in sites)

        toml = joinpath(dir, "points.toml")
        write(toml, """
            [[soundings]]
            id = "t1"
            time = 2021-12-02T03:00:00
            lat = 1.0
            lon = 2.0

            [[soundings]]
            id = "t2"
            time = "2021-12-02T04:00:00"
            lat = 3.0
            lon = 4.0

            [[sites]]
            id = "tower"
            lat = 50.0
            lon = 10.0
            intake_height = 100.0
            """)
        soundings, _ = read_observation_requests(TableSource(toml, SoundingMode(), AutoTableFormat()), 1, ORIGIN, [Date(2021, 12, 2)])
        @test [s.time_seconds for s in soundings] == [10800.0, 14400.0]
        _, sites = read_observation_requests(TableSource(toml, SiteMode(), TOMLTableFormat()), 1, ORIGIN, [Date(2021, 12, 2)])
        @test sites[1].id == "tower" && sites[1].intake_height_m == 100.0 && isnan(sites[1].elevation_m)

        nc = joinpath(dir, "points.nc")
        NCDataset(nc, "c") do ds
            defDim(ds, "obs", 2)
            defVar(ds, "id", String, ("obs",))[:] = ["n1", "n2"]
            t = defVar(ds, "time", Float64, ("obs",); attrib = Dict("units" => "hours since 2021-12-01 00:00:00"))
            t[:] = [25.0, 26.5]
            defVar(ds, "lat", Float64, ("obs",))[:] = [5.0, 6.0]
            defVar(ds, "lon", Float64, ("obs",))[:] = [7.0, 8.0]
        end
        soundings, _ = read_observation_requests(TableSource(nc, SoundingMode(), NetCDFTableFormat()), 1, ORIGIN, [Date(2021, 12, 2)])
        @test [s.id for s in soundings] == ["n1", "n2"]
        @test [s.time_seconds for s in soundings] == [3600.0, 2.5 * 3600]

        # Integer ids stay exact decimal strings; NC_CHAR ids are accepted; fill times are errors.
        nc2 = joinpath(dir, "points2.nc")
        NCDataset(nc2, "c") do ds
            defDim(ds, "obs", 2)
            defVar(ds, "sounding_id", UInt64, ("obs",))[:] = UInt64[2021120206301501, 7]
            t = defVar(ds, "time", Float64, ("obs",); attrib = Dict("units" => "seconds since 1970-01-01 00:00:00", "_FillValue" => -999999.0))
            t[:] = [unix(ORIGIN + Hour(1)), unix(ORIGIN + Hour(2))]
            defVar(ds, "latitude", Float64, ("obs",))[:] = [5.0, 6.0]
            defVar(ds, "longitude", Float64, ("obs",))[:] = [7.0, 8.0]
        end
        soundings, _ = read_observation_requests(TableSource(nc2, SoundingMode(), AutoTableFormat()), 1, ORIGIN, [Date(2021, 12, 2)])
        @test [s.id for s in soundings] == ["2021120206301501", "7"]
        nc3 = joinpath(dir, "sites3.nc")
        NCDataset(nc3, "c") do ds
            defDim(ds, "obs", 2); defDim(ds, "slen", 4)
            idv = defVar(ds, "site_id", Char, ("slen", "obs"))
            idv[:, 1] = collect("mlo\0"); idv[:, 2] = collect("spo\0")
            defVar(ds, "lat", Float64, ("obs",))[:] = [19.5, -89.9]
            defVar(ds, "lon", Float64, ("obs",))[:] = [-155.5, -24.8]
            defVar(ds, "intake_height", Float64, ("obs",); attrib = Dict("_FillValue" => -999.0))[:] = [40.0, -999.0]
        end
        _, nsites = read_observation_requests(TableSource(nc3, SiteMode(), AutoTableFormat()), 1, ORIGIN, [Date(2021, 12, 2)])
        @test [s.id for s in nsites] == ["mlo", "spo"]
        @test nsites[1].intake_height_m == 40.0 && isnan(nsites[2].intake_height_m)
        nc4 = joinpath(dir, "filltime.nc")
        NCDataset(nc4, "c") do ds
            defDim(ds, "obs", 1)
            defVar(ds, "id", String, ("obs",))[:] = ["x"]
            t = defVar(ds, "time", Float64, ("obs",); attrib = Dict("units" => "seconds since 1970-01-01 00:00:00", "_FillValue" => -999999.0))
            t.var[:] = [-999999.0]
            defVar(ds, "lat", Float64, ("obs",))[:] = [5.0]
            defVar(ds, "lon", Float64, ("obs",))[:] = [7.0]
        end
        @test_throws ArgumentError read_observation_requests(TableSource(nc4, SoundingMode(), AutoTableFormat()), 1, ORIGIN, [Date(2021, 12, 2)])

        # UTF-8 BOM and CRLF are tolerated; optional TOML columns may be absent per row.
        bom = joinpath(dir, "bom.csv")
        write(bom, "\ufeffid,time,lat,lon\r\nq,2021-12-02T00:00:00,1,2\r\n")
        @test length(first(read_observation_requests(TableSource(bom, SoundingMode(), AutoTableFormat()), 1, ORIGIN, [Date(2021, 12, 2)]))) == 1
        toml_opt = joinpath(dir, "opt.toml")
        write(toml_opt, "[[sites]]\nid = \"a\"\nlat = 1.0\nlon = 2.0\nintake_height = 10.0\n\n[[sites]]\nid = \"b\"\nlat = 3.0\nlon = 4.0\n")
        _, osites = read_observation_requests(TableSource(toml_opt, SiteMode(), AutoTableFormat()), 1, ORIGIN, [Date(2021, 12, 2)])
        @test osites[1].intake_height_m == 10.0 && isnan(osites[2].intake_height_m)
        toml_case = joinpath(dir, "case.toml")
        write(toml_case, "[[sites]]\nid = \"a\"\nlat = 1.0\nLat = 2.0\nlon = 2.0\n")
        @test_throws ArgumentError read_observation_requests(TableSource(toml_case, SiteMode(), AutoTableFormat()), 1, ORIGIN, [Date(2021, 12, 2)])

        bad = joinpath(dir, "bad.csv")
        write(bad, "id,time,lat\nx,2021-12-02T00:00:00,1\n")
        @test_throws ArgumentError read_observation_requests(TableSource(bad, SoundingMode(), AutoTableFormat()), 1, ORIGIN, [Date(2021, 12, 2)])
        badtime = joinpath(dir, "badtime.csv")
        write(badtime, "id,time,lat,lon\nx,20211202,1,2\n")
        @test_throws ArgumentError read_observation_requests(TableSource(badtime, SoundingMode(), AutoTableFormat()), 1, ORIGIN, [Date(2021, 12, 2)])
        badloc = joinpath(dir, "badloc.csv")
        write(badloc, "id,time,lat,lon\nx,2021-12-02T00:00:00,95,2\n")
        @test_throws ArgumentError read_observation_requests(TableSource(badloc, SoundingMode(), AutoTableFormat()), 1, ORIGIN, [Date(2021, 12, 2)])
        @test_throws ArgumentError read_observation_requests(TableSource(joinpath(dir, "points.xlsx"), SoundingMode(), AutoTableFormat()), 1, ORIGIN, [Date(2021, 12, 2)])
    end
end

@testset "build_observation_set sorts soundings and merges sites" begin
    mktempdir() do dir
        write_oco2_lite(joinpath(dir, "oco2_LtCO2_211202_c.nc4"))
        csv = joinpath(dir, "early.csv")
        write(csv, "id,time,lat,lon\nearly,2021-12-02T00:10:00,0,0\nlate,2021-12-02T23:00:00,0,0\n")
        sites_a = joinpath(dir, "a.csv"); write(sites_a, "id,lat,lon,intake_height\nmlo,19.536,-155.576,40\n")
        sites_b = joinpath(dir, "b.csv"); write(sites_b, "id,lat,lon,intake_height\nmlo,19.536,-155.576,40\nspo,-89.98,-24.8,10\n")
        sources = AtmosTransport.Output.AbstractObservationSource[
            TableSource(csv, SoundingMode(), AutoTableFormat()),
            OCO2LiteSource(joinpath(dir, "oco2_LtCO2_{YYMMDD}_c.nc4"), 0),
            TableSource(sites_a, SiteMode(), AutoTableFormat()),
            TableSource(sites_b, SiteMode(), AutoTableFormat()),
        ]
        set = build_observation_set(sources, ORIGIN, [Date(2021, 12, 2)])
        @test set isa ObservationSet
        @test set.origin == ORIGIN
        @test set.dropped_outside_window == 0
        @test issorted([s.time_seconds for s in set.soundings])
        @test [s.id for s in set.soundings][[1, end]] == ["early", "late"]
        @test [s.source for s in set.soundings] == [1, 2, 2, 2, 1]
        # Requests outside the run days are dropped and counted.
        # The Lite template has no Dec-3 file, so only the two CSV soundings are read, and both fall outside.
        narrow = build_observation_set(sources[1:2], ORIGIN + Day(1), [Date(2021, 12, 3)])
        @test isempty(narrow.soundings)
        @test narrow.dropped_outside_window == 2
        shifted = build_observation_set(sources[1:1], ORIGIN - Day(1), [Date(2021, 12, 1), Date(2021, 12, 2)])
        @test length(shifted.soundings) == 2
        @test shifted.soundings[1].time_seconds == 86400.0 + 600.0
        @test [s.id for s in set.sites] == ["mlo", "spo"]
        @test set.sites[1].source == 3 && set.sites[2].source == 4
        @test !isempty(set)
        @test isempty(ObservationSet(ORIGIN, SoundingRequest[], SiteRequest[]))

        conflict = joinpath(dir, "c.csv"); write(conflict, "id,lat,lon,intake_height\nmlo,20.0,-155.576,40\n")
        push!(sources, TableSource(conflict, SiteMode(), AutoTableFormat()))
        @test_throws ArgumentError build_observation_set(sources, ORIGIN, [Date(2021, 12, 2)])
    end
end
