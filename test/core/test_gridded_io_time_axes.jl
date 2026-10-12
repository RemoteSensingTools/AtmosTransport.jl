#!/usr/bin/env julia
#
# GriddedIO time axes: `decode_time` classifies an axis by its raw `units`
# string before any CFTime call. The cases are the TRENDY v14 S3 time axes of
# step P0 of the surface-flux port (docs/memos/2026-10-10_python_flux_port_p0_facts.md,
# Section 1): CF `days/hours since` axes in the gregorian, proleptic_gregorian,
# noleap, 365_day and 360_day calendars; `months since` axes (CABLE-POP 365_day,
# LPJml, ISAM, VISIT, iMAPLE Int16 without calendar), which NCDatasets/CFTime
# decode silently with a fixed 30.44-day month; decimal years (ELM-FATES,
# VISIT-UT); an index axis with an auxiliary time variable (CARDAMOM).
# Malformed origins (month 13, day 40, hour 25, no date) are errors, never
# normalised; `decode_month_keys` takes months from the calendar dates, so a
# 360_day 30 February, which `decode_time` cannot return as a DateTime, is
# February.
#
# For every units form the runtime decodes today, the new seconds are compared
# bit for bit with the frozen runtime parser (`_surface_flux_times_seconds`).

using Test
using Dates
using Logging
using NCDatasets

using AtmosTransport
using AtmosTransport.GriddedIO: decode_time, decode_month_keys, CFDecoder, MonthsSince,
                                DecimalYears, MonthStart, MidMonth, MonthEnd, AuxiliaryTimeVariable,
                                DEFAULT_TIME_DECODERS, MonthKey, TimeInterval, interval, month_keys,
                                seconds_since, check_last_month
const GIO = AtmosTransport.GriddedIO

# ---- Frozen runtime parser (src/Models/initial_conditions/surface_flux.jl) ----

function frozen_parse_cf_time_origin(s::AbstractString)
    ss = replace(strip(String(s)), "T" => " ")
    for fmt in (dateformat"y-m-d H:M:S", dateformat"y-m-d H:M", dateformat"y-m-d")
        try
            return DateTime(ss, fmt)
        catch
        end
    end
    throw(ArgumentError("time-varying surface flux: cannot parse time origin '$(s)'"))
end

function frozen_surface_flux_times_seconds(time_vals, units::AbstractString, reference_time::DateTime)
    n = length(time_vals)
    times_sec = Vector{Float64}(undef, n)
    sample = first(time_vals)
    if sample isa Real
        m = match(r"(\w+)\s+since\s+(.+)", strip(String(units)))
        m === nothing && throw(ArgumentError("cannot parse numeric time units '$(units)'"))
        unit_word = lowercase(m.captures[1])
        per_unit_sec = unit_word in ("second", "seconds", "sec", "s") ? 1.0 :
                       unit_word in ("minute", "minutes", "min")       ? 60.0 :
                       unit_word in ("hour", "hours", "hr", "h")        ? 3600.0 :
                       unit_word in ("day", "days", "d")                ? 86400.0 :
                       throw(ArgumentError("unsupported time unit '$(unit_word)'"))
        origin = frozen_parse_cf_time_origin(strip(m.captures[2]))
        ref_offset_sec = Dates.value(origin - reference_time) / 1000.0
        for k in 1:n
            times_sec[k] = ref_offset_sec + Float64(time_vals[k]) * per_unit_sec
        end
    else
        for k in 1:n
            times_sec[k] = Dates.value(DateTime(time_vals[k]) - reference_time) / 1000.0
        end
    end
    return times_sec
end

# ---- Fixtures ----

# The attributes go on the raw variable after it is defined, so writing the
# fixture never parses its time units.
function time_file(dir, values; units = nothing, calendar = nothing, T = eltype(values),
                   fill = nothing, extra = ds -> nothing)
    path = joinpath(dir, "t$(rand(UInt32)).nc")
    NCDataset(path, "c") do ds
        defDim(ds, "time", length(values))
        fill === nothing ? defVar(ds, "time", T, ("time",)) :
                           defVar(ds, "time", T, ("time",); fillvalue = fill)
        raw = NCDatasets.variable(ds, "time")
        units === nothing || (raw.attrib["units"] = units)
        calendar === nothing || (raw.attrib["calendar"] = calendar)
        raw[:] = T.(values)
        extra(ds)
    end
    return path
end

# The runtime replica reads through NCDatasets' CF layer, which warns about
# units it cannot parse; those warnings are the replica's, not under test.
quietly(f) = Logging.with_logger(f, Logging.NullLogger())
frozen_seconds(path, units, ref) =
    quietly(() -> NCDataset(ds -> frozen_surface_flux_times_seconds(ds["time"][:], units, ref), path))

decode_file(path, decoders = DEFAULT_TIME_DECODERS) = NCDataset(ds -> decode_time(ds, "time", decoders), path)
month_keys_file(path, decoders = DEFAULT_TIME_DECODERS) =
    NCDataset(ds -> decode_month_keys(ds, "time", decoders), path)
error_text(f) = try f(); "" catch err; err isa ArgumentError || rethrow(); sprint(showerror, err) end

# Day of year (0-based) of the first of each month in a 365-day year.
const NOLEAP_MONTH_START = cumsum([0; [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]])

@testset "CF axes: bit-identical to the runtime parser" begin
    dir = mktempdir()
    ref = DateTime(2021, 12, 1)
    cases = [
        ("hours since 2021-12-01 00:00:00", nothing, [0.0, 1.5, 3.0, 744.0], Float64),
        ("hours since 2021-12-01 00:00:00", "standard", [0.0, 3.0, 6.0], Float64),
        ("hours since 2021-12-01 00:00:00", "gregorian", [0.0, 3.0, 6.0], Float32),
        ("hours since 2021-12-01 00:00:00", "proleptic_gregorian", Int32[0, 24, 48], Int32),
        ("days since 1700-01-01 00:00:00", "gregorian", [117260.0, 117291.0], Float64),     # DLEM
        ("days since 1690-01-01", "gregorian", Float32[120910.0, 120941.0], Float32),       # LPJwsl
        ("days since 2021-11-01", "julian", [0.0, 30.0], Float64),
        ("hours since 1701-1-16 12:00:00", "proleptic_gregorian", [0.0, 744.0], Float64),  # JSBACH
        ("minutes since 2021-12-01 00:00", nothing, [0.0, 90.0], Float64),
        ("seconds since 2021-12-01", nothing, [0.0, 0.25, 86400.0], Float64),
        ("hours since 2015-01-01 00:00:00 UTC", nothing, [0.0, 24.0, 8736.0], Float64),  # Python outputs
        ("hours since 2021-12-01T00:00:00Z", nothing, [0.0, 3.0], Float64),
        ("Hours since 2021-12-01", nothing, [0.0, 3.0], Float64),
        ("hours since 2021-12-01 00:00:00.000000", nothing, [0.0, 3.0], Float64),
    ]
    for (units, calendar, values, T) in cases
        path = time_file(dir, values; units, calendar, T)
        new = seconds_since(decode_file(path), ref)
        old = frozen_seconds(path, units, ref)
        @test new == old
        @test new isa Vector{Float64}
    end
    # A time variable with `_FillValue = NaN` and no missing value (CLM-FATES, ED,
    # LPJwsl, TEM): decoded like the others.
    path = time_file(dir, Float32[15.5, 45.0]; units = "days since 1701-01-01", calendar = "standard",
                     fill = NaN32)
    @test seconds_since(decode_file(path), ref) == frozen_seconds(path, "days since 1701-01-01", ref)
    # The abbreviations the runtime parsed itself (CFTime does not know them):
    # same seconds for whole and half units.
    for (units, values) in (("h since 2021-12-01", [0.0, 1.5, 3.0]), ("hr since 2021-12-01 00:00", [0.0, 2.0]),
                            ("d since 2021-12-01", [0.0, 0.5, 31.0]), ("s since 2021-12-01", [0.0, 10.0]),
                            ("min since 2021-12-01", [0.0, 30.0]), ("sec since 2021-12-01", [0.0, 7.5]))
        path = time_file(dir, values; units)
        @test seconds_since(decode_file(path), ref) == frozen_seconds(path, units, ref)
    end
    # The stamps themselves.
    @test decode_file(time_file(dir, [0.0, 1.5]; units = "hours since 2021-12-01")) ==
          [DateTime(2021, 12, 1), DateTime(2021, 12, 1, 1, 30)]
    @test decode_file(time_file(dir, [0.0, 2.0]; units = "Days Since 2021-12-01")) ==
          [DateTime(2021, 12, 1), DateTime(2021, 12, 3)]
end

@testset "noleap, 365_day and 360_day calendars" begin
    dir = mktempdir()
    years = 1700:1702
    # CLM: hours since 1700-01-16 12:00 (noleap, Int32), stamps on day 16, 00:00.
    mid = [Int32(((y - 1700) * 365 + NOLEAP_MONTH_START[m] + 15) * 24 - (15 * 24 + 12)) for y in years for m in 1:12]
    t = decode_file(time_file(dir, mid; units = "hours since 1700-01-16 12:00:00.000000", calendar = "noleap"))
    @test t == [DateTime(y, m, 16) for y in years for m in 1:12]
    @test month_keys(t) == [MonthKey(y, m) for y in years for m in 1:12]
    # CLASSIC: days since 1700-12-31 (365_day), stamps on the last day of the month.
    last_day = [Float64((y - 1701) * 365 + NOLEAP_MONTH_START[m] + [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m])
                for y in 1701:1703 for m in 1:12]
    t = decode_file(time_file(dir, last_day; units = "days since 1700-12-31", calendar = "365_day"))
    @test month_keys(t) == [MonthKey(y, m) for y in 1701:1703 for m in 1:12]
    @test Dates.day.(t) == [Dates.daysinmonth(1701, m) for _ in 1:3 for m in 1:12]   # 28 Feb, never 29
    # Leap years of the Gregorian calendar keep 28 February as the last day.
    t = decode_file(time_file(dir, [Float64(4 * 365 + 58)]; units = "days since 1700-01-01", calendar = "noleap"))
    @test t == [DateTime(1704, 2, 28)]
    # Fractional days are rounded to the millisecond first.
    t = decode_file(time_file(dir, [1.5, 1.0 / 3]; units = "days since 2021-12-01", calendar = "noleap"))
    @test t == [DateTime(2021, 12, 2, 12), DateTime(2021, 12, 1, 8)]
    # SDGVM: hours since 1900-01-01 (360_day), 360 … step 720: day 16 of each month.
    t = decode_file(time_file(dir, collect(360.0:720.0:(360.0 + 720 * 35)); units = "hours since 1900-01-01 00:00:00",
                              calendar = "360_day"))
    @test t == [DateTime(y, m, 16) for y in 1900:1902 for m in 1:12]
    # " UTC" on a noleap origin.
    t = decode_file(time_file(dir, [0.0, 24.0]; units = "hours since 2015-01-01 00:00:00 UTC", calendar = "noleap"))
    @test t == [DateTime(2015, 1, 1), DateTime(2015, 1, 2)]
    # Dates that the proleptic Gregorian calendar does not have: decode_time
    # names the date and points to decode_month_keys.
    feb30 = time_file(dir, [29.0]; units = "days since 1900-02-01", calendar = "360_day")
    msg = error_text(() -> decode_file(feb30))
    @test occursin("360_day date 1900-02-30", msg) && occursin("DateTime", msg) &&
          occursin("decode_month_keys", msg)
    msg = error_text(() -> decode_file(time_file(dir, [59.0]; units = "days since 1901-01-01", calendar = "all_leap")))
    @test occursin("all_leap date 1901-02-29", msg) && occursin("decode_month_keys", msg)
    @test_throws ArgumentError decode_file(time_file(dir, [0.0]; units = "days since 2000-01-01", calendar = "no_such"))
end

@testset "decode_month_keys: months in the axis's own calendar" begin
    dir = mktempdir()
    # A 360_day axis with month-end stamps (day 30 of every month, 30 February
    # included): decode_time cannot hold 30 February, the month keys can.
    ends = time_file(dir, [Float64(30m - 1) for m in 1:24]; units = "days since 2001-01-01", calendar = "360_day")
    @test_throws ArgumentError decode_file(ends)
    @test month_keys_file(ends) == [MonthKey(2001, 1) + k for k in 0:23]
    # 29 February of a common year in all_leap.
    @test month_keys_file(time_file(dir, [59.0, 60.0]; units = "days since 1901-01-01", calendar = "all_leap")) ==
          [MonthKey(1901, 2), MonthKey(1901, 3)]
    # Where decode_time succeeds, the keys are month_keys(decode_time(...)).
    sdgvm = time_file(dir, collect(360.0:720.0:(360.0 + 720 * 35)); units = "hours since 1900-01-01 00:00:00",
                      calendar = "360_day")
    @test month_keys_file(sdgvm) == month_keys(decode_file(sdgvm)) == [MonthKey(1900, 1) + k for k in 0:35]
    noleap = time_file(dir, [0.0, 31.0, 59.0, 365.0]; units = "days since 1700-01-01", calendar = "noleap")
    @test month_keys_file(noleap) == month_keys(decode_file(noleap))
    gregorian = time_file(dir, [0.0, 744.0]; units = "hours since 2021-12-01 00:00:00")
    @test month_keys_file(gregorian) == [MonthKey(2021, 12), MonthKey(2022, 1)]
    cable = time_file(dir, Float64.(0:25); units = "months since 1700-1-16 12:00:00", calendar = "365_day")
    @test month_keys_file(cable) == [MonthKey(1700, 1) + k for k in 0:25]
    elm = time_file(dir, [1701 + k / 12 for k in 0:13])
    @test month_keys_file(elm, (DecimalYears(MonthStart()),)) == [MonthKey(1701, 1) + k for k in 0:13]
    # The same decoder choice as decode_time.
    @test_throws ArgumentError month_keys_file(elm)
    @test_throws ArgumentError month_keys_file(cable, (CFDecoder(),))
end

@testset "months since: never through CFTime" begin
    dir = mktempdir()
    n = 3900
    expected = [MonthKey(1700, 1) + k for k in 0:(n - 1)]
    # CABLE-POP: months since 1700-1-16 12:00:00, 365_day, integer stamps.
    cable = time_file(dir, Float64.(0:(n - 1)); units = "months since 1700-1-16 12:00:00", calendar = "365_day")
    t = decode_file(cable)
    @test month_keys(t) == expected
    @test t[1] == DateTime(1700, 1, 1) && t[end] == DateTime(2024, 12, 1)
    check_last_month(month_keys(t), MonthKey(2024, 12))
    # What CFTime would have made of it (the P0 failure this decoder avoids).
    cftime_keys = NCDataset(ds -> [MonthKey(Dates.year(x), Dates.month(x)) for x in ds["time"][:]], cable)
    # decode_time itself never builds a CF view of a `months since` axis: no
    # CFTime parse, so no warning even with a calendar CFTime rejects.
    bogus = time_file(dir, [0.0, 1.0]; units = "months since 2000-11-01", calendar = "not_a_calendar")
    @test_logs min_level = Logging.Warn decode_file(bogus)
    @test count(cftime_keys .!= expected) > 3000
    # iMAPLE: Int16, months since 1700-1-1, no calendar.
    imaple = time_file(dir, Int16.(0:(n - 1)); units = "months since 1700-1-1")
    @test month_keys(decode_file(imaple)) == expected
    # ISAM and VISIT: mid-month stamps k + ½.
    isam = time_file(dir, collect(0.5:1.0:(n - 0.5)); units = "months since 1700-01-01 00:00:00",
                     calendar = "proleptic_gregorian")
    @test month_keys(decode_file(isam)) == expected
    visit = time_file(dir, collect(0.5:1.0:1979.5); units = "months since 1860-01-01")
    @test month_keys(decode_file(visit)) == [MonthKey(1860, 1) + k for k in 0:1979]
    # Stamps a hair below the integer still belong to their month.
    @test month_keys(decode_file(time_file(dir, [2.0 - 1e-9, 3.0]; units = "months since 2000-01-01"))) ==
          [MonthKey(2000, 3), MonthKey(2000, 4)]
    # The calendar is not read: CFTime would reject this calendar, MonthsSince does not.
    @test month_keys(decode_file(time_file(dir, [0.0, 1.0]; units = "months since 2000-11-01",
                                           calendar = "not_a_calendar"))) == [MonthKey(2000, 11), MonthKey(2000, 12)]
    # " UTC" on the origin.
    @test decode_file(time_file(dir, [0.0]; units = "months since 2000-05-01 00:00:00 UTC")) == [DateTime(2000, 5, 1)]
    # Without MonthsSince among the decoders, the axis is refused (never CFTime).
    @test_throws ArgumentError decode_file(cable, (CFDecoder(),))
end

@testset "decimal years" begin
    dir = mktempdir()
    n = 3888
    # ELM-FATES: month starts k/12 in Float64, many of them a hair (~1e-12
    # month) below the integer month.
    v = [1701 + k / 12 for k in 0:(n - 1)]
    elm = time_file(dir, v)                                   # no units, no calendar
    t = decode_file(elm, (DecimalYears(MonthStart()),))
    @test month_keys(t) == [MonthKey(1701, 1) + k for k in 0:(n - 1)]
    @test t[1] == DateTime(1701, 1, 1)
    # A plain floor of 12·(v − ⌊v⌋) would misplace those stamps.
    @test count(k -> floor(Int, 12 * (v[k] - floor(v[k]))) != mod(k - 1, 12), 1:n) > 0
    # VISIT-UT: mid-month, unparseable units.
    w = [1700 + (k + 0.5) / 12 for k in 0:3899]
    vut = time_file(dir, w; units = "years since AD 0-Jan-1st (double)")
    @test month_keys(decode_file(vut, (CFDecoder(), MonthsSince(), DecimalYears(MidMonth())))) ==
          [MonthKey(1700, 1) + k for k in 0:3899]
    # Residual check: mid-month stamps are not month starts, and the reverse.
    @test_throws ArgumentError decode_file(vut, (DecimalYears(MonthStart()),))
    @test_throws ArgumentError decode_file(elm, (DecimalYears(MidMonth()),))
    @test_throws ArgumentError decode_file(time_file(dir, [2000 + 0.3 / 12]), (DecimalYears(MonthStart()),))
    # Without a named decoder, an axis without usable units is refused.
    err = try decode_file(elm); nothing catch e; e end
    @test err isa ArgumentError && occursin("DecimalYears", sprint(showerror, err))
    @test_throws ArgumentError decode_file(vut)
    # `years since` with a proper origin is not decoded silently either.
    @test_throws ArgumentError decode_file(time_file(dir, [0.0, 1.0]; units = "years since 2000-01-01"))
end

@testset "auxiliary time variable (CARDAMOM)" begin
    dir = mktempdir()
    # CARDAMOM's `Time`: days since 2003-01-01 at the end of each month.
    month_ends = Float64[Dates.value(Date(2003, 1, 1) + Month(k) - Date(2003, 1, 1)) for k in 1:264]
    @test month_ends[1:4] == [31.0, 59.0, 90.0, 120.0] && month_ends[end] == 8036.0
    aux(values; dim = "time") = ds -> begin
        dim == "time" || defDim(ds, dim, length(values))
        defVar(ds, "Time", Float32, (dim,))
        v = NCDatasets.variable(ds, "Time")
        v.attrib["units"] = "d"
        v.attrib["long_name"] = "Monthly time step given in days since 01/01/2003"
        v[:] = Float32.(values)
    end
    path = time_file(dir, Int32.(1:264); extra = aux(month_ends))
    d = AuxiliaryTimeVariable("Time", Date(2003, 1, 1), Day(1), MonthEnd())
    t = decode_file(path, (CFDecoder(), MonthsSince(), d))
    @test month_keys(t) == [MonthKey(2003, 1) + k for k in 0:263]
    @test t[1] == DateTime(2003, 1, 1)
    check_last_month(month_keys(t), MonthKey(2024, 12))
    # Declared as month starts, the axis runs one month late; `last_month` catches it.
    late = month_keys(decode_file(path, (AuxiliaryTimeVariable("Time", Date(2003, 1, 1), Day(1), MonthStart()),)))
    @test late[1] == MonthKey(2003, 2)
    @test_throws ArgumentError check_last_month(late, MonthKey(2024, 12))
    # Mid-month stamps, fractional days.
    mid = time_file(dir, Int32[1, 2]; extra = aux([15.5, 45.0]))
    @test decode_file(mid, (AuxiliaryTimeVariable("Time", Date(2003, 1, 1), Day(1), MidMonth()),)) ==
          [DateTime(2003, 1, 1), DateTime(2003, 2, 1)]
    @test_throws ArgumentError decode_file(path, (AuxiliaryTimeVariable("Nope", Date(2003, 1, 1), Day(1), MonthEnd()),))
    short = time_file(dir, Int32[1, 2, 3, 4]; extra = aux(month_ends[1:3]; dim = "other"))
    @test_throws DimensionMismatch decode_file(short, (d,))
    # Decimal years at month ends.
    ends = time_file(dir, [2003 + k / 12 for k in 1:3])
    @test month_keys(decode_file(ends, (DecimalYears(MonthEnd()),))) == [MonthKey(2003, 1), MonthKey(2003, 2), MonthKey(2003, 3)]
end

@testset "malformed origins are errors, never normalised" begin
    dir = mktempdir()
    # `months since`: the origin must be a valid date Y-M[-D] [h:m[:s]].
    for (units, reason) in (("months since 2000-13-01", "month 13 is not in 1:12"),
                            ("months since 2000-0-01", "month 0 is not in 1:12"),
                            ("months since 2000-01-40", "day 40 is not in 1:31"),
                            ("months since 2000-01-00", "day 0 is not in 1:31"),
                            ("months since 2000-01-01 25:00:00", "not a valid time"),
                            ("months since 2000-01-01 12:61", "not a valid time"),
                            ("months since the start of the run", "is not a date"),
                            ("months since 2000", "is not a date"),
                            ("months since 2000-01-01 noon", "is not a date"))
        path = time_file(dir, [0.0, 1.0]; units)
        msg = error_text(() -> decode_file(path))
        @test occursin(units, msg) && occursin(reason, msg) && occursin("'time'", msg)
        # Not even a decimal-year decoder takes such an axis.
        @test_throws ArgumentError decode_file(path, (DecimalYears(MonthStart()),))
        @test_throws ArgumentError month_keys_file(path)
    end
    # Valid origin forms still decode (month only, fractional seconds, `T`, `Z`).
    @test decode_file(time_file(dir, [0.0]; units = "months since 2000-03")) == [DateTime(2000, 3, 1)]
    @test decode_file(time_file(dir, [1.0]; units = "months since 1700-1-16 12:00:00.000")) == [DateTime(1700, 2, 1)]
    @test decode_file(time_file(dir, [0.0]; units = "months since 2000-12-31T23:59:59Z")) == [DateTime(2000, 12, 1)]
    # CF units: CFTime would normalise these origins silently in the standard
    # calendar (2000-13-01 as 2001-01-01, 2000-02-30 as 2000-03-01).
    for (units, calendar, reason) in (("days since 2000-13-01", nothing, "month 13 is not in 1:12"),
                                      ("hours since 2000-01-40", "proleptic_gregorian", "day 40 is not in 1:31"),
                                      ("days since 2000-01-01 25:00:00", "noleap", "not a valid time"),
                                      ("days since 2001-02-29", "standard", "2001-02-29 does not exist in the standard calendar"),
                                      ("days since 2000-02-30", "proleptic_gregorian", "2000-02-30 does not exist"),
                                      ("days since 2001-02-29", "noleap", "2001-02-29 does not exist in the noleap calendar"))
        path = time_file(dir, [0.0, 1.0]; units, calendar)
        msg = error_text(() -> decode_file(path))
        @test occursin("origin is not a valid instant", msg) && occursin(reason, msg)
    end
    # Dates valid in their own calendar are accepted: 30 February of 360_day
    # (decode_month_keys gives its month), 29 February of a Julian 1900.
    @test month_keys_file(time_file(dir, [0.0, 1.0]; units = "days since 2000-02-30", calendar = "360_day")) ==
          [MonthKey(2000, 2), MonthKey(2000, 3)]
    @test decode_file(time_file(dir, [1.0]; units = "days since 2000-02-29", calendar = "julian")) ==
          [DateTime(2000, 3, 14)]
end

@testset "decode_time_bounds: the instants decode_time gives" begin
    dir = mktempdir()
    function bounded_file(values, bounds; units, calendar)
        time_file(dir, values; units, calendar, extra = ds -> begin
            NCDatasets.variable(ds, "time").attrib["bounds"] = "time_bnds"
            defDim(ds, "nv", 2)
            defVar(ds, "time_bnds", Float64, ("nv", "time"))[:, :] = bounds
        end)
    end
    decode_bounds(path) = NCDataset(ds -> GIO.decode_time_bounds(ds, "time"), path)
    # Coordinates at the bound starts, so row 1 must equal decode_time.
    cases = [
        # Julian 2000-01-01 is proleptic Gregorian 2000-01-14 (13 days apart);
        # Julian 2000-03-01 is 2000-03-14. Copying the Julian fields would
        # switch stepwise slices 13 days early.
        ("days since 2000-01-01", "julian", [0.0 31.0; 31.0 60.0],
         [DateTime(2000, 1, 14) DateTime(2000, 2, 14); DateTime(2000, 2, 14) DateTime(2000, 3, 14)]),
        # `standard` before 1582-10-15 is Julian: 1500-01-01 is 1500-01-10.
        ("days since 1500-01-01", "standard", [0.0 31.0; 31.0 59.0],
         [DateTime(1500, 1, 10) DateTime(1500, 2, 10); DateTime(1500, 2, 10) DateTime(1500, 3, 10)]),
        ("hours since 2021-01-01", "proleptic_gregorian", [0.0 744.0; 744.0 1416.0],
         [DateTime(2021, 1, 1) DateTime(2021, 2, 1); DateTime(2021, 2, 1) DateTime(2021, 3, 1)]),
        # Non-Gregorian calendars keep their calendar fields, as decode_time does.
        ("days since 2001-01-01", "noleap", [0.0 31.0; 31.0 59.0],
         [DateTime(2001, 1, 1) DateTime(2001, 2, 1); DateTime(2001, 2, 1) DateTime(2001, 3, 1)]),
    ]
    for (units, calendar, bounds, expected) in cases
        path = bounded_file(vec(bounds[1, :]), bounds; units, calendar)
        decoded = decode_bounds(path)
        @test decoded == expected
        @test decoded[1, :] == decode_file(path)
    end
    # Coordinates at mid-interval: bounds and coordinate agree on the Julian shift.
    julian = bounded_file([15.5, 45.5], [0.0 31.0; 31.0 60.0]; units = "days since 2000-01-01",
                          calendar = "julian")
    @test decode_file(julian) == [DateTime(2000, 1, 29, 12), DateTime(2000, 2, 28, 12)]
    @test decode_bounds(julian)[1, :] == [DateTime(2000, 1, 14), DateTime(2000, 2, 14)]
    @test decode_bounds(time_file(dir, [0.0]; units = "days since 2000-01-01")) === nothing
end

@testset "fill values and malformed axes" begin
    dir = mktempdir()
    path = time_file(dir, [0.0, -1.0]; units = "hours since 2021-12-01", fill = -1.0)
    @test_throws ArgumentError decode_file(path)
    path = time_file(dir, [0.0, -1.0]; units = "months since 2021-12-01", fill = -1.0)
    @test_throws ArgumentError decode_file(path)
    NCDataset(time_file(dir, [0.0]; units = "hours since 2021-12-01")) do ds
        @test_throws ArgumentError decode_time(ds, "no_time")
    end
end

@testset "month keys, intervals and seconds" begin
    k = MonthKey(2024, 12)
    @test k + 1 == MonthKey(2025, 1)
    @test k - 12 == MonthKey(2023, 12)
    @test MonthKey(2024, 1) - 1 == MonthKey(2023, 12)
    @test MonthKey("2024-12") == k
    @test MonthKey(DateTime(2024, 12, 31, 23)) == k
    @test isless(MonthKey(2023, 12), k)
    @test sort([k, MonthKey(2020, 3), MonthKey(2024, 1)]) == [MonthKey(2020, 3), MonthKey(2024, 1), k]
    @test_throws ArgumentError MonthKey(2024, 13)
    @test_throws ArgumentError MonthKey("2024/12")
    @test interval(MonthKey(2024, 2)) == TimeInterval(DateTime(2024, 2, 1), DateTime(2024, 3, 1))
    @test interval(k) == TimeInterval(DateTime(2024, 12, 1), DateTime(2025, 1, 1))
    @test Dates.daysinmonth(MonthKey(2024, 2)) == 29 && Dates.daysinmonth(MonthKey(2023, 2)) == 28
    @test_throws ArgumentError TimeInterval(DateTime(2024, 1, 2), DateTime(2024, 1, 1))
    @test DateTime(2024, 2, 29) in interval(MonthKey(2024, 2))
    @test !(DateTime(2024, 3, 1) in interval(MonthKey(2024, 2)))
    ref = DateTime(2021, 12, 1)
    @test seconds_since([ref, ref + Hour(3), ref - Millisecond(500)], ref) == [0.0, 10800.0, -0.5]
    @test seconds_since(ref + Day(1), ref) === 86400.0
    # The catalog's last month.
    keys = [MonthKey(2024, 11), k]
    @test check_last_month(keys, k) === keys
    err = try check_last_month(keys[1:1], k); nothing catch e; e end
    @test err isa ArgumentError && occursin("2024-11", sprint(showerror, err)) &&
          occursin("2024-12", sprint(showerror, err))
    @test_throws ArgumentError check_last_month(MonthKey[], k)
end
