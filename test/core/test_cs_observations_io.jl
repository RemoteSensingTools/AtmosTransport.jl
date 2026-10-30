#!/usr/bin/env julia

using Test
using NCDatasets: NCDataset, defDim, defVar

import AtmosTransport
const AT = AtmosTransport

# ---------------------------------------------------------------------------
# Observation IO, schema v2 (`schemas/cs_observations_v2.toml`).
#
# Verifies:
#   * `write_observations` emits the v2 layout (dimensions, variable dtypes
#     with Float64 coordinates, root attributes).
#   * `read_observations(write_observations(...))` recovers every field bit
#     for bit, NaN "absent" vertical fields included.
#   * The loader rejects schema v1 files with an explanation, other versions,
#     missing variables / attributes, Float32 coordinates and a wrong
#     `date_component` length.
#   * `CSObservationRecord` coerces dtypes, defaults the optional v2 fields,
#     and rejects invalid values (positional and keyword constructors).
# ---------------------------------------------------------------------------

function _sample_records()
    return [
        AT.CSObservationRecord(id = 1, date_components = (2024, 6, 15, 12, 0, 0),
                               lat = 34.05, lon = -118.25, kind = "column",
                               elevation_m = 100.0, time_window_s = 3600,
                               value = 420.1e-6, value_sigma = 0.5e-6,
                               instrument_type = "TCCON", tracer = "CO2"),
        AT.CSObservationRecord(id = 2, date_components = (2024, 6, 15, 13, 30, 0),
                               lat = -33.87, lon = 151.21, intake_height_m = 50.0, elevation_m = 12.5,
                               value = 1925.3e-9, value_sigma = 1.2e-9, model_sigma = 3e-9,
                               instrument_type = "ICOS", tracer = "CH4"),
        AT.CSObservationRecord(id = 99, date_components = (2024, 12, 31, 23, 59, 59),
                               lat = 0.123456789012345, lon = 359.987654321098, altitude_m = 5321.5,
                               pressure_pa = 52000.0, flag = 2, value = 410e-6, value_sigma = 1e-6,
                               instrument_type = "aircraft", tracer = "CO2"),
    ]
end

@testset "CSObservationRecord v2 constructors" begin
    r = AT.CSObservationRecord(id = 7, date_components = (2024, 1, 1, 0, 0, 0), lat = 1, lon = 2,
                               value = 10, value_sigma = 0.1, instrument_type = "X", tracer = "Y")
    @test r.id === Int64(7)
    @test r.date_components === (Int16(2024), Int16(1), Int16(1), Int16(0), Int16(0), Int16(0))
    @test r.lat === 1.0 && r.lon === 2.0
    @test r.kind === AT.PointObservationKind()
    @test isnan(r.elevation_m) && isnan(r.intake_height_m) && isnan(r.altitude_m) && isnan(r.pressure_pa)
    @test r.time_window_s === 0.0 && r.flag === Int8(0) && r.model_sigma === 0.0
    @test r.value === 10.0 && r.value_sigma === 0.1
    # Float64 coordinates keep digits a Float32 would round away (OF-84).
    x = 12.345678901234567
    @test AT.CSObservationRecord(id = 1, date_components = (2024, 1, 1, 0, 0, 0), lat = x, lon = x,
                                 value = 1, value_sigma = 1, instrument_type = "I", tracer = "T").lat === x
    @test AT.CSObservationRecord(id = 1, date_components = (2024, 1, 1, 0, 0, 0), lat = 0, lon = 0,
                                 kind = AT.ColumnObservationKind(), value = 1, value_sigma = 1,
                                 instrument_type = "I", tracer = "T").kind === AT.ColumnObservationKind()

    ctor(; kwargs...) = AT.CSObservationRecord(; id = 1, date_components = (2024, 1, 1, 0, 0, 0),
                                               lat = 0.0, lon = 0.0, value = 1.0, value_sigma = 1.0,
                                               instrument_type = "I", tracer = "T", kwargs...)
    @test ctor() isa AT.CSObservationRecord
    @test_throws ArgumentError AT.CSObservationRecord(id = 1, date_components = (1, 2, 3, 4, 5),
                                                      lat = 0, lon = 0, value = 0, value_sigma = 1.0,
                                                      instrument_type = "I", tracer = "T")
    for bad in ((lat = NaN,), (lat = Inf,), (lat = 90.5,), (lat = -91.0,), (lon = NaN,), (lon = Inf,),
                (lon = 721.0,), (value = NaN,), (value = Inf,), (value_sigma = NaN,), (value_sigma = Inf,),
                (value_sigma = -1.0,), (value_sigma = 0.0,), (model_sigma = -1.0,), (model_sigma = NaN,),
                (time_window_s = -1.0,), (time_window_s = Inf,), (elevation_m = Inf,),
                (intake_height_m = -Inf,), (altitude_m = Inf,), (pressure_pa = 0.0,), (pressure_pa = -5.0,),
                (pressure_pa = Inf,), (kind = "satellite",), (kind = :point,))
        @test_throws ArgumentError ctor(; bad...)
    end
    # Column records state no height or pressure.
    for field in (:intake_height_m, :altitude_m, :pressure_pa)
        @test_throws ArgumentError ctor(; kind = "column", NamedTuple{(field,)}((100.0,))...)
    end
    @test ctor(; kind = "column", elevation_m = 300.0).elevation_m == 300.0
    # The positional constructor validates too.
    @test_throws ArgumentError AT.CSObservationRecord(
        Int64(1), (Int16(2024), Int16(1), Int16(1), Int16(0), Int16(0), Int16(0)),
        NaN, 0.0, AT.PointObservationKind(), NaN, NaN, NaN, NaN, 0.0, Int8(0),
        420.0, 1.0, 0.0, "TCCON", "CO2", nothing)
end

@testset "CSObservationSet v2 NetCDF round trip, bit for bit" begin
    set = AT.CSObservationSet(_sample_records(); time_origin = "1900-01-01 00:00:00")
    mktempdir() do dir
        path = joinpath(dir, "obs_v2.nc")
        @test AT.write_observations(path, set) == path
        NCDataset(path, "r") do ds
            @test all(name -> eltype(ds[name].var) === Float64,
                      ("lat", "lon", "time_window_s"))
        end
        back = AT.read_observations(path)
        @test back isa AT.CSObservationSet
        @test back.time_origin == set.time_origin
        @test length(back) == length(set)
        for (orig, got) in zip(set, back)
            for field in fieldnames(AT.CSObservationRecord)
                a, b = getfield(orig, field), getfield(got, field)
                @test a === b || (a isa Float64 && isnan(a) && isnan(b))
            end
        end
        # An empty set round-trips too.
        empty_path = joinpath(dir, "empty.nc")
        AT.write_observations(empty_path, AT.CSObservationSet(AT.CSObservationRecord[]))
        @test isempty(AT.read_observations(empty_path))
        # Ids are rechecked on write (the records vector is mutable), before
        # any file is created.
        dup = AT.CSObservationSet(_sample_records(); time_origin = "2024-01-01 00:00:00")
        push!(dup.records, dup.records[1])
        dup_path = joinpath(dir, "dup.nc")
        @test_throws ArgumentError AT.write_observations(dup_path, dup)
        @test !isfile(dup_path)
    end
end

@testset "read_observations rejects v1 files and schema violations" begin
    set = AT.CSObservationSet(_sample_records(); time_origin = "1900-01-01 00:00:00")
    mktempdir() do dir
        good = joinpath(dir, "good.nc")
        AT.write_observations(good, set)
        @test AT.read_observations(good) isa AT.CSObservationSet

        # A schema-v1 file (Float32 lat/lon/alt) is rejected with the reason.
        v1 = joinpath(dir, "v1.nc")
        NCDataset(v1, "c") do ds
            defDim(ds, "obs", 1)
            defDim(ds, "date_component", 6)
            ds.attrib["cs_observations_schema"] = "v1"
            ds.attrib["time_origin"] = "1900-01-01 00:00:00"
            defVar(ds, "id", Int64, ("obs",))[:] = [1]
            defVar(ds, "date_components", Int16, ("date_component", "obs"))[:, :] =
                reshape(Int16[2024, 1, 1, 0, 0, 0], 6, 1)
            for v in ("lat", "lon", "alt")
                defVar(ds, v, Float32, ("obs",))[:] = Float32[0]
            end
            defVar(ds, "value", Float64, ("obs",))[:] = [400.0]
            defVar(ds, "value_sigma", Float64, ("obs",))[:] = [1.0]
            defVar(ds, "instrument_type", String, ("obs",))[:] = ["ICOS"]
            defVar(ds, "tracer", String, ("obs",))[:] = ["CO2"]
        end
        @test_throws ArgumentError AT.read_observations(v1)


        # Another version, a missing schema attribute or time origin.
        for (name, mutate!) in (("v3", ds -> ds.attrib["cs_observations_schema"] = "v3"),
                                ("no_schema", ds -> delete!(ds.attrib, "cs_observations_schema")),
                                ("no_origin", ds -> delete!(ds.attrib, "time_origin")))
            path = joinpath(dir, "$(name).nc")
            cp(good, path)
            NCDataset(mutate!, path, "a")
            @test_throws ArgumentError AT.read_observations(path)
        end
        @test_throws ArgumentError AT.read_observations(joinpath(dir, "missing.nc"))

        # A v2 file with Float32 coordinates, or without a v2 variable.
        function write_v2(path; lat_type = Float64, omit = nothing, dc = 6)
            NCDataset(path, "c") do ds
                defDim(ds, "obs", 1)
                defDim(ds, "date_component", dc)
                ds.attrib["cs_observations_schema"] = "v2"
                ds.attrib["time_origin"] = "1900-01-01 00:00:00"
                vars = (("id", Int64, [1]), ("lat", lat_type, lat_type[10]), ("lon", Float64, [20.0]),
                        ("kind", String, ["point"]), ("elevation_m", Float64, [NaN]),
                        ("intake_height_m", Float64, [10.0]), ("altitude_m", Float64, [NaN]),
                        ("pressure_pa", Float64, [NaN]), ("time_window_s", Float64, [0.0]),
                        ("flag", Int8, Int8[0]), ("value", Float64, [400.0]), ("value_sigma", Float64, [1.0]),
                        ("model_sigma", Float64, [0.0]), ("instrument_type", String, ["ICOS"]),
                        ("tracer", String, ["CO2"]))
                for (name, T, data) in vars
                    name == omit && continue
                    defVar(ds, name, T, ("obs",))[:] = data
                end
                omit == "date_components" ||
                    (defVar(ds, "date_components", Int16, ("date_component", "obs"))[:, :] =
                         reshape(Int16[2024, 1, 1, 0, 0, 0][1:dc], dc, 1))
            end
            return path
        end
        @test AT.read_observations(write_v2(joinpath(dir, "ok.nc"))) isa AT.CSObservationSet
        @test_throws ArgumentError AT.read_observations(write_v2(joinpath(dir, "f32.nc"); lat_type = Float32))
        @test_throws ArgumentError AT.read_observations(write_v2(joinpath(dir, "dc5.nc"); dc = 5))
    end
end

@testset "read_observations: fill values and repeated ids" begin
    mktempdir() do dir
        # An xarray-style file: float variables carry `_FillValue = NaN`, so
        # NCDatasets reads NaN entries as missing; elevation uses -9999.
        path = joinpath(dir, "xarray.nc")
        NCDataset(path, "c") do ds
            defDim(ds, "obs", 2)
            defDim(ds, "date_component", 6)
            ds.attrib["cs_observations_schema"] = "v2"
            ds.attrib["time_origin"] = "1900-01-01 00:00:00"
            nanfill = Dict("_FillValue" => NaN)
            defVar(ds, "id", Int64, ("obs",))[:] = [1, 2]
            defVar(ds, "date_components", Int16, ("date_component", "obs"))[:, :] =
                repeat(Int16[2024, 1, 1, 0, 0, 0], 1, 2)
            defVar(ds, "lat", Float64, ("obs",); attrib = nanfill)[:] = [10.0, 20.0]
            defVar(ds, "lon", Float64, ("obs",); attrib = nanfill)[:] = [30.0, 40.0]
            defVar(ds, "kind", String, ("obs",))[:] = ["point", "column"]
            defVar(ds, "elevation_m", Float64, ("obs",); attrib = Dict("_FillValue" => -9999.0))[:] =
                [120.0, -9999.0]
            defVar(ds, "intake_height_m", Float64, ("obs",); attrib = nanfill)[:] = [10.0, NaN]
            defVar(ds, "altitude_m", Float64, ("obs",); attrib = nanfill)[:] = [NaN, NaN]
            defVar(ds, "pressure_pa", Float64, ("obs",); attrib = nanfill)[:] = [NaN, NaN]
            for (v, data) in (("time_window_s", [0.0, 3600.0]), ("value", [400.0, 401.0]),
                              ("value_sigma", [1.0, 2.0]), ("model_sigma", [0.0, 0.5]))
                defVar(ds, v, Float64, ("obs",); attrib = nanfill)[:] = data
            end
            defVar(ds, "flag", Int8, ("obs",))[:] = Int8[0, 0]
            defVar(ds, "instrument_type", String, ("obs",))[:] = ["ICOS", "TCCON"]
            defVar(ds, "tracer", String, ("obs",))[:] = ["CO2", "CO2"]
        end
        set = AT.read_observations(path)
        @test set[1].intake_height_m == 10.0 && set[1].elevation_m == 120.0
        @test isnan(set[1].altitude_m) && isnan(set[1].pressure_pa)
        @test isnan(set[2].elevation_m) && isnan(set[2].intake_height_m)
        @test set[2].kind === AT.ColumnObservationKind() && set[2].model_sigma == 0.5
        # A fill value in a required variable is an error.
        NCDataset(ds -> ds["value"][2] = NaN, path, "a")
        err = try AT.read_observations(path) catch e e end
        @test err isa ArgumentError
        # Repeated ids are rejected.
        dup = joinpath(dir, "dup.nc")
        AT.write_observations(dup, AT.CSObservationSet(_sample_records()))
        NCDataset(ds -> ds["id"][3] = 1, dup, "a")
        err = try AT.read_observations(dup) catch e e end
        @test err isa ArgumentError
    end
end

@testset "write_observations rejects empty time_origin" begin
    @test_throws ArgumentError AT.write_observations(
        tempname(), AT.CSObservationSet(_sample_records(); time_origin = ""))
end
