"""
`AtmosTransport.Diagnostics.daily_snapshot_series` (`src/Diagnostics/snapshot_series.jl`)
on snapshot-like NetCDF fixtures written into `mktempdir()`.

Asserts:
- frames of several files are concatenated in file order and stamped
  `origin + hours`; the hour-0 initial frame of the first file is kept;
- a frame repeating an hour already read is dropped and the first occurrence
  wins (its values, not the repeat's, are returned);
- the stored element type is kept (Float32 stays Float32); `_FillValue` and
  `missing_value` values become NaN; 5-D variables keep their level axis and
  time-only variables (`<tracer>_total_mass`) come back as vectors;
- time units other than the writer's bare `"hours"`, a frame earlier than the
  last one read, a missing variable, a variable without a trailing `time`
  dimension, an integer variable and a shape change between files are errors;
- a real runtime snapshot written by `write_snapshot_netcdf` reads back with
  the same values and times;
- `keep_frame` reads only the frames it accepts (one per day from several
  files, none from some files, none at all), still drops repeated hours and
  still rejects a frame out of time order when that frame is not kept; a
  predicate that does not return a `Bool` is an error.
"""

using Test
using Dates
using NCDatasets
using AtmosTransport: CubedSphereMesh, AtmosGrid, HybridSigmaPressure, CPU, SnapshotFrame,
                      write_snapshot_netcdf
using AtmosTransport.Diagnostics: daily_snapshot_series

const NX, NY, NF, NL = 3, 2, 6, 4

# A cubed-sphere snapshot-like file: column means (Xdim, Ydim, nf, time) and a
# level field (Xdim, Ydim, nf, lev, time), both Float32.
function write_fixture(path, hours, value_of;
                       units = "hours since 2021-12-01 00:00:00",
                       calendar = "proleptic_gregorian", fill = nothing,
                       missing_value = nothing)
    NCDataset(path, "c") do ds
        defDim(ds, "Xdim", NX); defDim(ds, "Ydim", NY); defDim(ds, "nf", NF)
        defDim(ds, "lev", NL); defDim(ds, "time", length(hours))
        tattrib = Dict("units" => units)
        calendar === nothing || (tattrib["calendar"] = calendar)
        t = defVar(ds, "time", Float64, ("time",), attrib = tattrib)
        t[:] = hours
        attrib = Dict{String, Any}("units" => "mol mol-1 dry")
        missing_value === nothing || (attrib["missing_value"] = missing_value)
        cm = fill === nothing ?
             defVar(ds, "co2_column_mean", Float32, ("Xdim", "Ydim", "nf", "time"); attrib) :
             defVar(ds, "co2_column_mean", Float32, ("Xdim", "Ydim", "nf", "time"); attrib, fillvalue = fill)
        lv = defVar(ds, "co2", Float32, ("Xdim", "Ydim", "nf", "lev", "time"))
        area = defVar(ds, "cell_area", Float64, ("Xdim", "Ydim", "nf"))
        area[:, :, :] = ones(NX, NY, NF)
        for (k, h) in enumerate(hours)
            cm.var[:, :, :, k] = Float32[value_of(h) + i + 10j + 100p for i in 1:NX, j in 1:NY, p in 1:NF]
            lv.var[:, :, :, :, k] = Float32[value_of(h) + l for i in 1:NX, j in 1:NY, p in 1:NF, l in 1:NL]
        end
    end
    return path
end

@testset "daily snapshot series" begin
    dir = mktempdir()
    origin = DateTime(2021, 12, 1)
    f1 = write_fixture(joinpath(dir, "run_20211201.nc"), [0.0, 24.0], h -> h)
    f2 = write_fixture(joinpath(dir, "run_20211202.nc"), [24.0, 48.0], h -> h == 24 ? -1.0f6 : h)  # repeats hour 24
    f3 = write_fixture(joinpath(dir, "run_20211203.nc"), [60.0, 72.0], h -> h)
    s = daily_snapshot_series([f1, f2, f3], ["co2_column_mean", :co2])
    @test s.times == origin .+ Hour.([0, 24, 48, 60, 72])
    cm = s.values["co2_column_mean"]
    @test cm isa Array{Float32, 4}
    @test size(cm) == (NX, NY, NF, 5)
    @test cm[2, 1, 3, :] == Float32[0, 24, 48, 60, 72] .+ (2 + 10 + 300)
    @test minimum(cm) > 0                                   # the repeat (-1e6) was dropped
    lev = s.values["co2"]
    @test size(lev) == (NX, NY, NF, NL, 5)
    @test lev[1, 1, 1, :, 3] == Float32[49, 50, 51, 52]

    # Fill values become NaN: _FillValue, and missing_value alone.
    g1 = write_fixture(joinpath(dir, "fill.nc"), [0.0, 3.0], h -> h == 3 ? 1.0f15 : h; fill = 1.0f15)
    g2 = write_fixture(joinpath(dir, "missing.nc"), [6.0], h -> -999.0f0; missing_value = -999.0f0)
    g = daily_snapshot_series([g1, g2], ["co2_column_mean"]; origin)
    @test g.times == origin .+ Hour.([0, 3, 6])
    @test !any(isnan, g.values["co2_column_mean"][:, :, :, 1])
    # value_of(h) + offsets: 1e15 + small offsets rounds back to 1e15 in Float32.
    @test all(isnan, g.values["co2_column_mean"][:, :, :, 2])
    @test count(isnan, g.values["co2_column_mean"][:, :, :, 3]) == 0   # -999 + offsets ≠ -999
    g3 = write_fixture(joinpath(dir, "missing_exact.nc"), [9.0], h -> 0.0f0; missing_value = 111.0f0)
    @test isnan(daily_snapshot_series([g3], ["co2_column_mean"]; origin).values["co2_column_mean"][1, 1, 1, 1])

    # Errors.
    legacy = write_fixture(joinpath(dir, "legacy.nc"), [0.0], h -> h;
                           units = "hours", calendar = nothing)
    @test_throws ArgumentError daily_snapshot_series([legacy], ["co2"])
    @test daily_snapshot_series([legacy], ["co2"]; origin).times == [origin]
    @test_throws ArgumentError daily_snapshot_series([f3, f1], ["co2"]; origin)          # out of order
    cf = write_fixture(joinpath(dir, "cf.nc"), [0.0], h -> h)
    @test daily_snapshot_series([cf], ["co2"]).times == [origin]
    @test_throws ArgumentError daily_snapshot_series([cf], ["co2"];
                                                     origin = DateTime(2020, 1, 1))
    @test_throws ArgumentError daily_snapshot_series([f1], ["nope"]; origin)
    @test_throws ArgumentError daily_snapshot_series([f1], ["cell_area"]; origin)
    @test_throws ArgumentError daily_snapshot_series([f1], ["co2", "co2"]; origin)
    @test_throws ArgumentError daily_snapshot_series(String[], ["co2"]; origin)
    other = joinpath(dir, "other.nc")
    NCDataset(other, "c") do ds
        defDim(ds, "Xdim", NX + 1); defDim(ds, "time", 1)
        defVar(ds, "time", Float64, ("time",), attrib = Dict(
            "units" => "hours since 2021-12-01 00:00:00",
            "calendar" => "proleptic_gregorian"))[:] = [96.0]
        defVar(ds, "co2_column_mean", Float32, ("Xdim", "time"))[:, :] = zeros(Float32, NX + 1, 1)
        defVar(ds, "count", Int32, ("Xdim", "time"))[:, :] = zeros(Int32, NX + 1, 1)
    end
    @test_throws DimensionMismatch daily_snapshot_series([f1, other], ["co2_column_mean"]; origin)
    @test_throws ArgumentError daily_snapshot_series([other], ["count"]; origin)
end

@testset "runtime snapshot files" begin
    dir = mktempdir()
    mesh = CubedSphereMesh(; FT = Float64, Nc = 4)
    grid = AtmosGrid(mesh, HybridSigmaPressure(Float64[0, 1, 2], Float64[0, 0.5, 1]), CPU(); FT = Float64)
    air = ntuple(_ -> fill(2.0, 4, 4, 2), 6)
    frame(h, vmr) = SnapshotFrame(h, air, Dict(:co2 => ntuple(_ -> fill(vmr * 2.0, 4, 4, 2), 6)), :dry)
    origin = DateTime(2014, 9, 1)
    a = write_snapshot_netcdf(joinpath(dir, "run_20211201.nc"), [frame(0.0, 400e-6), frame(24.0, 401e-6)], grid; time_origin=origin)
    b = write_snapshot_netcdf(joinpath(dir, "run_20211202.nc"), [frame(24.0, 999e-6), frame(48.0, 402e-6)], grid; time_origin=origin)
    s = daily_snapshot_series([a, b], ["co2_column_mean", "air_mass_per_area", "co2_total_mass"])
    @test s.times == origin .+ Hour.([0, 24, 48])
    cm = s.values["co2_column_mean"]
    @test size(cm) == (4, 4, 6, 3)
    @test cm[:, :, :, 1] ≈ fill(400e-6, 4, 4, 6) rtol = 1e-6
    @test cm[:, :, :, 2] ≈ fill(401e-6, 4, 4, 6) rtol = 1e-6         # first occurrence of hour 24
    @test cm[:, :, :, 3] ≈ fill(402e-6, 4, 4, 6) rtol = 1e-6
    @test size(s.values["air_mass_per_area"]) == (4, 4, 6, 2, 3)
    # Time-only variables (the compensated tracer mass) come back as vectors.
    total = s.values["co2_total_mass"]
    @test total isa Vector{Float64} && length(total) == 3
    @test total[2] / total[1] ≈ 401 / 400 rtol = 1e-12
    @test total[3] / total[1] ≈ 402 / 400 rtol = 1e-12
end

@testset "keep_frame" begin
    dir = mktempdir()
    origin = DateTime(2021, 12, 1)
    f1 = write_fixture(joinpath(dir, "run_20211201.nc"), [0.0, 12.0, 24.0], h -> h)
    f2 = write_fixture(joinpath(dir, "run_20211202.nc"), [24.0, 36.0, 48.0], h -> h == 24 ? -1.0f6 : h)
    f3 = write_fixture(joinpath(dir, "run_20211203.nc"), [60.0], h -> h)
    at_00 = t -> Dates.hour(t) == 0
    s = daily_snapshot_series([f1, f2, f3], ["co2_column_mean", "co2"]; origin, keep_frame = at_00)
    @test s.times == origin .+ Hour.([0, 24, 48])                     # f3 holds no 00 UTC frame
    cm = s.values["co2_column_mean"]
    @test size(cm) == (NX, NY, NF, 3)
    @test cm[1, 1, 1, :] == Float32[0, 24, 48] .+ (1 + 10 + 100)      # hour 24 from f1, not the repeat
    @test size(s.values["co2"]) == (NX, NY, NF, NL, 3)
    early = daily_snapshot_series([f1, f2, f3], ["co2_column_mean"]; origin, keep_frame = t -> t < origin + Hour(20))
    @test early.times == origin .+ Hour.([0, 12])
    nothing_kept = daily_snapshot_series([f1, f2], ["co2_column_mean"]; origin, keep_frame = t -> false)
    @test isempty(nothing_kept.times) && size(nothing_kept.values["co2_column_mean"]) == (NX, NY, NF, 0)
    # The order check covers frames that are not kept.
    back = write_fixture(joinpath(dir, "back.nc"), [18.0], h -> h)
    @test_throws ArgumentError daily_snapshot_series([f1, back], ["co2"]; origin, keep_frame = at_00)
    @test_throws TypeError daily_snapshot_series([f1], ["co2"]; origin, keep_frame = t -> 1)
end
