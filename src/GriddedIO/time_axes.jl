# ---------------------------------------------------------------------------
# Time axes of gridded input files
#
# NCDatasets 0.14 / CFTime 0.2 decode `months since` (and `years since`)
# silently with a fixed month of 30.436849895833 days whatever the calendar:
# on the TRENDY files this gives 3201 of 3900 wrong month keys for CABLE-POP
# (365_day), 2436 for LPJml and 296 for iMAPLE. Axes without units come back
# as raw numbers. `decode_time` therefore reads the raw `units` string first,
# classifies the axis, and hands only `<unit> since <origin>` axes with a
# fixed-length unit (day and shorter) to CFTime. P0 facts memo, Section 1.
# ---------------------------------------------------------------------------

"""
    TimeInterval(start::DateTime, stop::DateTime)

The half-open time interval `[start, stop)`; `start < stop` is required.
"""
struct TimeInterval
    start :: DateTime
    stop  :: DateTime
    function TimeInterval(start::DateTime, stop::DateTime)
        start < stop || throw(ArgumentError(
            "a time interval needs start < stop; got [$(start), $(stop))"))
        return new(start, stop)
    end
end

Base.in(t::DateTime, iv::TimeInterval) = iv.start <= t < iv.stop

"""
    MonthKey(year, month)
    MonthKey(t::Dates.TimeType)
    MonthKey("YYYY-MM")

A calendar month of the proleptic Gregorian calendar. Keys order
chronologically; `k + n` is the month `n` months later.
"""
struct MonthKey
    year  :: Int
    month :: Int
    function MonthKey(year::Integer, month::Integer)
        1 <= month <= 12 || throw(ArgumentError("month must be in 1:12; got $(month)"))
        return new(Int(year), Int(month))
    end
end

MonthKey(t::Dates.TimeType) = MonthKey(Dates.year(t), Dates.month(t))

function MonthKey(s::AbstractString)
    m = match(r"^\s*(-?\d+)-(\d{1,2})\s*$", s)
    m === nothing && throw(ArgumentError("a month key is written \"YYYY-MM\"; got $(repr(s))"))
    return MonthKey(parse(Int, m.captures[1]), parse(Int, m.captures[2]))
end

# Months since year 0, January: an integer index that orders and adds.
_month_index(k::MonthKey) = 12 * k.year + (k.month - 1)
_month_key(index::Integer) = MonthKey(fld(index, 12), mod(index, 12) + 1)

Base.isless(a::MonthKey, b::MonthKey) = isless(_month_index(a), _month_index(b))
Base.:+(k::MonthKey, n::Integer) = _month_key(_month_index(k) + n)
Base.:-(k::MonthKey, n::Integer) = _month_key(_month_index(k) - n)
Base.show(io::IO, k::MonthKey) = print(io, "MonthKey(", k.year, ", ", k.month, ")")
Dates.daysinmonth(k::MonthKey) = Dates.daysinmonth(k.year, k.month)

"""
    interval(k::MonthKey) -> TimeInterval

The month `k` as `[first instant of k, first instant of the next month)`.
"""
function interval(k::MonthKey)
    next = k + 1
    return TimeInterval(DateTime(k.year, k.month, 1), DateTime(next.year, next.month, 1))
end

"""
    month_keys(times) -> Vector{MonthKey}

The calendar month of each time stamp.
"""
month_keys(times::AbstractVector{<:Dates.TimeType}) = MonthKey[MonthKey(t) for t in times]

"""
    seconds_since(t, reference::DateTime) -> Float64
    seconds_since(times, reference::DateTime) -> Vector{Float64}

Seconds from `reference` to each time stamp, `Dates.value(t - reference) / 1000`
(millisecond resolution, the resolution of `DateTime`); negative before
`reference`. This is the runtime's slice-time convention.
"""
seconds_since(t::DateTime, reference::DateTime) = Dates.value(t - reference) / 1000.0
seconds_since(times::AbstractVector{DateTime}, reference::DateTime) =
    Float64[seconds_since(t, reference) for t in times]

"""
    check_last_month(keys, last_month::MonthKey; context = "time axis") -> keys

Throw an `ArgumentError` unless the last of the month `keys` is `last_month`
(a catalog's `last_month`, e.g. `MonthKey(2024, 12)` for TRENDY v14); return
`keys` otherwise.
"""
function check_last_month(keys::AbstractVector{MonthKey}, last_month::MonthKey;
                          context::AbstractString = "time axis")
    isempty(keys) && throw(ArgumentError("$(context) is empty; expected it to end in $(last_month)"))
    last(keys) == last_month || throw(ArgumentError(
        "$(context) ends in $(last(keys).year)-$(lpad(last(keys).month, 2, '0')); " *
        "expected $(last_month.year)-$(lpad(last_month.month, 2, '0'))"))
    return keys
end

# ---------------------------------------------------------------------------
# Classification of the raw units string
# ---------------------------------------------------------------------------

# Units of a time axis, read from the raw `units` attribute before any CFTime
# call. `CFTimeUnits.units` is the normalised "<unit> since <origin>" string
# (unit word in CF spelling, a trailing " UTC" dropped); `period` is one unit;
# `date` is the origin's (year, month, day) when it has the `Y-M-D` form.
abstract type AbstractTimeUnits end
struct CFTimeUnits <: AbstractTimeUnits
    units  :: String
    origin :: String
    period :: Dates.Period
    date   :: Union{Nothing, NTuple{3, Int}}
end
struct MonthsSinceUnits <: AbstractTimeUnits
    year  :: Int
    month :: Int
end
struct UnlabelledUnits <: AbstractTimeUnits     # no units, `years since`, unparseable
    text :: String
end
struct MalformedTimeUnits <: AbstractTimeUnits  # `<unit> since <origin>` with an invalid origin
    text   :: String
    reason :: String
end

# Fixed-length unit words: CF spellings (any case, singular or plural) and the
# abbreviations the runtime has always accepted.
const _FIXED_TIME_UNITS = Dict{String, Tuple{String, Dates.Period}}(
    "day" => ("days", Dates.Day(1)), "days" => ("days", Dates.Day(1)), "d" => ("days", Dates.Day(1)),
    "hour" => ("hours", Dates.Hour(1)), "hours" => ("hours", Dates.Hour(1)),
    "hr" => ("hours", Dates.Hour(1)), "hrs" => ("hours", Dates.Hour(1)), "h" => ("hours", Dates.Hour(1)),
    "minute" => ("minutes", Dates.Minute(1)), "minutes" => ("minutes", Dates.Minute(1)),
    "min" => ("minutes", Dates.Minute(1)), "mins" => ("minutes", Dates.Minute(1)),
    "second" => ("seconds", Dates.Second(1)), "seconds" => ("seconds", Dates.Second(1)),
    "sec" => ("seconds", Dates.Second(1)), "secs" => ("seconds", Dates.Second(1)),
    "s" => ("seconds", Dates.Second(1)),
    "millisecond" => ("milliseconds", Dates.Millisecond(1)),
    "milliseconds" => ("milliseconds", Dates.Millisecond(1)),
)

const _SINCE_UNITS = r"^\s*([A-Za-z]+)\s+since\s+(\S.*?)\s*$"i

# Origin with a trailing " UTC" (any case) removed: CF times are UTC.
_strip_utc(origin::AbstractString) = String(replace(strip(origin), r"\s+[Uu][Tt][Cc]$" => ""))

# An origin `Y-M[-D]` (day 1 when omitted), optionally followed (after spaces
# or `T`) by a time of day `h:m` or `h:m:s[.fff]` and a `Z`.
const _ORIGIN = r"^(-?\d+)-(\d{1,2})(?:-(\d{1,2}))?(?:(?:\s+|T)(\d{1,2}):(\d{1,2})(?::(\d{1,2}(?:\.\d*)?))?Z?)?$"

# (year, month, day) of an origin of the `_ORIGIN` form, `nothing` for another
# form, or the reason it is not a valid instant (month outside 1:12, day outside
# 1:31, hour > 23, minute > 59, second ≥ 60). CFTime would normalise such an
# origin silently in the standard calendar (`2000-13-01` as 2001-01-01).
function _parse_origin(origin::AbstractString)
    m = match(_ORIGIN, origin)
    m === nothing && return nothing
    year, month = parse(Int, m[1]), parse(Int, m[2])
    day = m[3] === nothing ? 1 : parse(Int, m[3])
    1 <= month <= 12 || return "the origin month $(month) is not in 1:12"
    1 <= day <= 31 || return "the origin day $(day) is not in 1:31"
    if m[4] !== nothing
        hour, minute = parse(Int, m[4]), parse(Int, m[5])
        second = m[6] === nothing ? 0.0 : parse(Float64, m[6])
        hour <= 23 && minute <= 59 && second < 60 ||
            return "the origin time of day $(m[4]):$(m[5])$(m[6] === nothing ? "" : ":" * m[6]) is not a valid time"
    end
    return (year, month, day)
end

function _classify_time_units(units::AbstractString)
    m = match(_SINCE_UNITS, units)
    m === nothing && return UnlabelledUnits(String(units))
    word = lowercase(m.captures[1])
    origin = _strip_utc(m.captures[2])
    if haskey(_FIXED_TIME_UNITS, word)
        cf_word, period = _FIXED_TIME_UNITS[word]
        date = _parse_origin(origin)
        date isa String && return MalformedTimeUnits(String(units), date)
        # Another origin form goes to CFTime, which parses or rejects it.
        return CFTimeUnits("$(cf_word) since $(origin)", origin, period, date)
    elseif word in ("month", "months")
        date = _parse_origin(origin)
        date === nothing && return MalformedTimeUnits(String(units),
            "the origin $(repr(origin)) is not a date Y-M[-D] with an optional time of day h:m[:s]")
        date isa String && return MalformedTimeUnits(String(units), date)
        return MonthsSinceUnits(date[1], date[2])
    end
    return UnlabelledUnits(String(units))                 # `years since …` and the like
end

# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------

"""
    AbstractTimeDecoder

A rule that turns a time axis into `DateTime` stamps. [`decode_time`](@ref)
classifies the axis by its raw `units` string and uses the first decoder of
its list that accepts that class:

| Raw units | Decoder |
|---|---|
| `days`/`hours`/`minutes`/`seconds since <origin>` | [`CFDecoder`](@ref) |
| `months since <origin>` | [`MonthsSince`](@ref) |
| none, `years since …`, or unparseable | [`DecimalYears`](@ref) or [`AuxiliaryTimeVariable`](@ref), named per product |
"""
abstract type AbstractTimeDecoder end

"""
    CFDecoder()

Decoder of `<unit> since <origin>` axes whose unit has a fixed length (day,
hour, minute, second, millisecond; also the abbreviations `d`, `h`, `hr`,
`min`, `s`, `sec`), through CFTime, honouring the `calendar` attribute:

- `standard`, `gregorian`, `proleptic_gregorian`, `julian` or no calendar:
  NCDatasets' own decoding, rounded to the millisecond (the runtime's stamps,
  bit for bit; a `julian` axis keeps its instants, so its dates shift by the
  Julian–Gregorian difference);
- `noleap`/`365_day`, `all_leap`/`366_day`, `360_day`: the calendar date and
  time of day of each stamp in its own calendar (values rounded to the
  millisecond first), carried over to a `DateTime`. Every `noleap`/`365_day`
  date exists in the proleptic Gregorian calendar; an `all_leap`/`366_day` or
  `360_day` date that does not (29 February of a common year, 30 February) is
  an `ArgumentError` naming it. [`decode_month_keys`](@ref) takes the months
  of such axes from the calendar dates themselves.

A trailing `" UTC"` on the origin is dropped. Fill values in the axis are an
error.
"""
struct CFDecoder <: AbstractTimeDecoder end

"""
    MonthsSince()

Decoder of `months since <origin>` axes (CABLE-POP, ISAM, LPJml, VISIT,
iMAPLE). The month of stamp `v` is the origin month plus `⌊v + 10⁻⁶⌋`. The
origin must be a valid date `Y-M[-D]` with an optional time of day (an
`ArgumentError` otherwise, see [`decode_time`](@ref)); only its year and month
are read (CABLE-POP's origin is `1700-1-16 12:00:00`). Integer stamps (month starts) and stamps `k + ½`
(mid-month) both give month `k`. Each stamp decodes to the first instant of
its month; the calendar attribute does not matter.
"""
struct MonthsSince <: AbstractTimeDecoder end

"Offset added before taking the floor in [`MonthsSince`](@ref)."
const MONTHS_SINCE_FLOOR_OFFSET = 1e-6

"""
    AbstractMonthStamp

Where a monthly time stamp sits in its month, for [`DecimalYears`](@ref) and
[`AuxiliaryTimeVariable`](@ref): [`MonthStart`](@ref), [`MidMonth`](@ref) or
[`MonthEnd`](@ref). Which one a product uses is a fact about the product,
named in its configuration; the catalog's `last_month` check
([`check_last_month`](@ref)) catches a wrong choice.
"""
abstract type AbstractMonthStamp end

"Stamps at the start of the month: decimal year `y + k/12`; the stamp lies in its month."
struct MonthStart <: AbstractMonthStamp end

"Stamps in the middle of the month: decimal year `y + (k + ½)/12`; the stamp lies in its month."
struct MidMonth <: AbstractMonthStamp end

"""
Stamps at the end of the month: the first instant of the next month (CARDAMOM's
`Time` = 31, 59, 90, … days since 2003-01-01 for January, February, March, …);
decimal year `y + (k + 1)/12`.
"""
struct MonthEnd <: AbstractMonthStamp end

_stamp_offset(::MonthStart) = 0.0
_stamp_offset(::MidMonth) = 0.5
_stamp_offset(::MonthEnd) = 1.0

# The month a stamp belongs to.
_month_of_stamp(t::DateTime, ::Union{MonthStart, MidMonth}) = MonthKey(t)
_month_of_stamp(t::DateTime, ::MonthEnd) = MonthKey(t - Dates.Millisecond(1))

"""
    DecimalYears(stamp::AbstractMonthStamp)

Decoder of monthly axes stored as decimal years without usable units
(ELM-FATES, month starts: `DecimalYears(MonthStart())`; VISIT-UT, mid-month,
units `years since AD 0-Jan-1st (double)`: `DecimalYears(MidMonth())`).
With `m = 12 v` months since year 0, the month index is
`k̂ = round(m − o)` (`o = 0` for month starts, `½` for mid-month stamps, `1`
for month ends) and
the residual `|m − o − k̂|` must not exceed `10⁻⁶` month, else an
`ArgumentError`. Rounding to the nearest month is required: ELM-FATES has 1296
values 9.1e-13 month below the integer, where a plain floor gives 2592 bad
steps. Each stamp decodes to the first instant of its month.
"""
struct DecimalYears{P <: AbstractMonthStamp} <: AbstractTimeDecoder
    stamp :: P
end

"Largest residual, in months, of a decimal-year stamp from its month position ([`DecimalYears`](@ref))."
const DECIMAL_YEAR_RESIDUAL_TOLERANCE = 1e-6

"""
    AuxiliaryTimeVariable(variable, origin::Date, unit::Dates.Period, stamp::AbstractMonthStamp)

Decoder of a monthly time axis that is only an index (no units), whose stamps
are in another variable of the file. CARDAMOM: `time` = 1 … 264; the `Time`
variable (units `"d"`, long name "Monthly time step given in days since
01/01/2003") holds 31, 59, 90, …, 8036, the end of each month, so
`AuxiliaryTimeVariable("Time", Date(2003, 1, 1), Day(1), MonthEnd())` gives
2003-01 … 2024-12. Stamp `k` is `DateTime(origin) + v_k · unit`, rounded to
the millisecond, where `v_k` is value `k` of `variable`, which must have as
many values as the axis; it decodes to the first instant of the month it
belongs to under `stamp`.
"""
struct AuxiliaryTimeVariable{P <: Dates.Period, S <: AbstractMonthStamp} <: AbstractTimeDecoder
    variable :: String
    origin   :: Date
    unit     :: P
    stamp    :: S
end
AuxiliaryTimeVariable(variable::AbstractString, origin::Date, unit::Dates.Period,
                      stamp::AbstractMonthStamp) =
    AuxiliaryTimeVariable{typeof(unit), typeof(stamp)}(String(variable), origin, unit, stamp)

"""
The decoders [`decode_time`](@ref) uses unless given others: `(CFDecoder(),
MonthsSince())`. Axes without usable units need a decoder named per product.
"""
const DEFAULT_TIME_DECODERS = (CFDecoder(), MonthsSince())

_accepts(::CFDecoder, ::CFTimeUnits) = true
_accepts(::MonthsSince, ::MonthsSinceUnits) = true
_accepts(::DecimalYears, ::UnlabelledUnits) = true
_accepts(::AuxiliaryTimeVariable, ::UnlabelledUnits) = true
_accepts(::AbstractTimeDecoder, ::AbstractTimeUnits) = false

_needed_decoder(::CFTimeUnits) = "CFDecoder()"
_needed_decoder(::MonthsSinceUnits) = "MonthsSince()"
_needed_decoder(::UnlabelledUnits) = "DecimalYears(...) or AuxiliaryTimeVariable(...)"

_check_time_units(::AbstractTimeUnits, name, ds) = nothing
_check_time_units(u::MalformedTimeUnits, name, ds) = throw(ArgumentError(
    "time axis '$(name)' of $(_dataset_path(ds)) has the units $(repr(u.text)), whose origin " *
    "is not a valid instant: $(u.reason)"))

_string_attribute(v, name::AbstractString, default::AbstractString = "") =
    String(string(get(v.attrib, name, default)))

"""
    decode_time(ds, name, decoders::Tuple = DEFAULT_TIME_DECODERS) -> Vector{DateTime}

Time stamps of the time axis `name` of the NetCDF dataset `ds`. The axis is
classified by the `units` attribute of the raw variable **before any CFTime
call** (a `months since` axis never reaches CFTime), and decoded by the first
decoder in `decoders` that accepts its class (see
[`AbstractTimeDecoder`](@ref)); an `ArgumentError` names the decoder that is
needed when none does. The length of the axis is never used to guess its
meaning, and time bounds are not read.

The origin of `<unit> since <origin>` units must be a valid instant: an origin
`Y-M-D[ h:m[:s]]` with a month outside 1:12, a day outside 1:31 (or beyond the
month's length in the axis's calendar, for [`CFDecoder`](@ref)) or an invalid
time of day is an `ArgumentError`, never normalised (`months since 2000-13-01`
is not January 2001). A `months since` origin must have that form.

The month-resolution decoders ([`MonthsSince`](@ref), [`DecimalYears`](@ref),
[`AuxiliaryTimeVariable`](@ref)) return the first instant of each month;
[`CFDecoder`](@ref) returns the stamps themselves. `DateTime` is proleptic
Gregorian: a stamp of a non-Gregorian calendar whose date that calendar lacks
(30 February of `360_day`, 29 February of a common year in `all_leap` or
`360_day`) is an `ArgumentError` naming the date; [`decode_month_keys`](@ref)
gives the months of such an axis.
"""
function decode_time(ds, name::AbstractString, decoders::Tuple = DEFAULT_TIME_DECODERS)
    decoder, class = _time_decoder(ds, name, decoders)
    return _decode(decoder, class, ds, String(name))
end

"""
    decode_month_keys(ds, name, decoders::Tuple = DEFAULT_TIME_DECODERS) -> Vector{MonthKey}

The month of each stamp of the time axis `name` of `ds`, in the axis's own
calendar: the flux builders' monthly keys. Classification, decoder choice and
origin checks are those of [`decode_time`](@ref). A [`CFDecoder`](@ref) axis
in `noleap`/`365_day`, `all_leap`/`366_day` or `360_day` gives the year and
month of each calendar date without converting it to `DateTime`, so 30
February of a `360_day` axis is February; other axes give
`month_keys(decode_time(ds, name, decoders))`.
"""
function decode_month_keys(ds, name::AbstractString, decoders::Tuple = DEFAULT_TIME_DECODERS)
    decoder, class = _time_decoder(ds, name, decoders)
    return _decode_month_keys(decoder, class, ds, String(name))
end

"""
    decode_month_seconds(ds, name) -> Vector{Float64}

Length in seconds of the calendar month represented by each slice of time axis
`name`. This is the conversion basis for flux variables whose units are per
calendar month.

When the coordinate names a CF bounds variable, each slice is assigned from
the midpoint of its bounds. For fixed-length CF units the bounds' actual
duration is returned; for `months since` bounds, one calendar month is
required. Without bounds, the coordinate stamp assigns the month. Month
lengths use the coordinate's `calendar` (`standard` by default), including
`360_day`, `365_day`/`noleap`, and `366_day`/`all_leap`. Thus neither a slice's
ordinal position nor the proleptic-Gregorian month length is guessed.
"""
function decode_month_seconds(ds, name::AbstractString)
    axis = String(name)
    decoder, class = _time_decoder(ds, axis, DEFAULT_TIME_DECODERS)
    bounds = _time_bounds_values(ds, axis)
    values = bounds === nothing ? _raw_time_values(ds, axis) :
             vec((bounds[1, :] .+ bounds[2, :]) ./ 2)
    keys = _decode_month_keys_values(decoder, class, ds, axis, values)
    DT = _time_calendar_type(ds, axis)
    calendar_seconds = Float64[CFTime.daysinmonth(DT, k.year, k.month) * 86400 for k in keys]
    bounds === nothing && return calendar_seconds

    if class isa CFTimeUnits
        seconds_per_unit = Dates.value(Dates.Millisecond(class.period)) / 1000.0
        return Float64[(bounds[2, i] - bounds[1, i]) * seconds_per_unit
                       for i in axes(bounds, 2)]
    elseif class isa MonthsSinceUnits
        for i in axes(bounds, 2)
            width = bounds[2, i] - bounds[1, i]
            isapprox(width, 1.0; atol = 1e-9, rtol = 0.0) || throw(ArgumentError(
                "time bounds for per-month slice $i of $(_dataset_path(ds)) span " *
                "$(width) months; expected exactly one calendar month"))
        end
        return calendar_seconds
    end
    throw(ArgumentError(
        "time axis '$axis' of $(_dataset_path(ds)) cannot define calendar-month durations"))
end

# The decoder of `decoders` for axis `name`, and the class of its units.
function _time_decoder(ds, name::AbstractString, decoders::Tuple)
    haskey(ds, name) || throw(ArgumentError(
        "time variable '$(name)' not found in $(_dataset_path(ds))"))
    raw = NCDatasets.variable(ds, name)
    ndims(raw) == 1 || throw(ArgumentError(
        "time variable '$(name)' in $(_dataset_path(ds)) has $(ndims(raw)) dimensions"))
    units = _string_attribute(raw, "units")
    class = _classify_time_units(units)
    _check_time_units(class, name, ds)
    i = findfirst(d -> _accepts(d, class), decoders)
    i === nothing && throw(ArgumentError(
        "time axis '$(name)' of $(_dataset_path(ds)) has units $(repr(units)), which need " *
        "the decoder $(_needed_decoder(class)); the decoders given are $(decoders)"))
    return decoders[i], class
end

_decode_month_keys(d::AbstractTimeDecoder, class::AbstractTimeUnits, ds, name::String) =
    month_keys(_decode(d, class, ds, name))
_decode_month_keys(::CFDecoder, class::CFTimeUnits, ds, name::String) =
    month_keys(last(_cf_calendar_stamps(class, ds, name)))

_decode_month_keys_values(::MonthsSince, class::MonthsSinceUnits, ds, name::String,
                          values) = begin
    base = 12 * class.year + (class.month - 1)
    MonthKey[_month_key(base + floor(Int, v + MONTHS_SINCE_FLOOR_OFFSET)) for v in values]
end

function _decode_month_keys_values(::CFDecoder, class::CFTimeUnits, ds, name::String,
                                   values)
    DT = _time_calendar_type(ds, name)
    _check_cf_origin(class, DT, ds, name)
    ms_per_unit = Dates.value(Dates.Millisecond(class.period))
    ms = Int64[round(Int64, v * ms_per_unit) for v in values]
    stamps = CFTime.timedecode(DT, ms, "milliseconds since $(class.origin)")
    return MonthKey[MonthKey(Dates.year(t), Dates.month(t)) for t in stamps]
end

# The optional CF bounds of a one-dimensional time coordinate, normalised to
# `(2, ntime)`. Bounds inherit the coordinate's units and calendar under CF, so
# their raw numeric values use the coordinate decoder selected above.
function _time_bounds_values(ds, name::String)
    axis = NCDatasets.variable(ds, name)
    bounds_name = strip(_string_attribute(axis, "bounds"))
    isempty(bounds_name) && return nothing
    haskey(ds, bounds_name) || throw(ArgumentError(
        "time axis '$name' of $(_dataset_path(ds)) names missing bounds variable " *
        repr(bounds_name)))
    bounds_var = NCDatasets.variable(ds, bounds_name)
    ndims(bounds_var) == 2 || throw(ArgumentError(
        "time bounds '$bounds_name' of $(_dataset_path(ds)) must be two-dimensional"))
    vals = Array(NCDatasets.cfvariable(ds, bounds_name; units = nothing,
                                      calendar = nothing, maskingvalue = missing))
    any(ismissing, vals) && throw(ArgumentError(
        "time bounds '$bounds_name' of $(_dataset_path(ds)) have fill values"))
    ntime = length(axis)
    bounds = if size(vals) == (2, ntime)
        Float64.(vals)
    elseif size(vals) == (ntime, 2)
        permutedims(Float64.(vals))
    else
        throw(DimensionMismatch(
            "time bounds '$bounds_name' of $(_dataset_path(ds)) have size $(size(vals)); " *
            "expected (2, $ntime) or ($ntime, 2)"))
    end
    for i in axes(bounds, 2)
        isfinite(bounds[1, i]) && isfinite(bounds[2, i]) && bounds[1, i] < bounds[2, i] ||
            throw(ArgumentError(
                "time bounds for slice $i of $(_dataset_path(ds)) must be finite and ordered"))
    end
    return bounds
end

# Raw numeric values of a variable, with fill values as `missing` (an error)
# and `scale_factor`/`add_offset` applied, never time-decoded.
function _raw_time_values(ds, name::String)
    v = NCDatasets.cfvariable(ds, name; units = nothing, calendar = nothing, maskingvalue = missing)
    vals = Array(v)
    any(ismissing, vals) && throw(ArgumentError(
        "time variable '$(name)' of $(_dataset_path(ds)) has fill values"))
    return Float64[Float64(x) for x in vals]
end

const _GREGORIAN_LIKE = (CFTime.DateTimeStandard, CFTime.DateTimeProlepticGregorian,
                         CFTime.DateTimeJulian)

# (calendar, stamps) of a CF axis, the stamps in the axis's own calendar:
# `DateTime` for the Gregorian-like calendars (NCDatasets' decoding, rounded to
# the millisecond), CFTime stamps otherwise (values rounded to the millisecond
# first). An origin day beyond its month in that calendar is an error.
function _time_calendar_type(ds, name::String)
    raw = NCDatasets.variable(ds, name)
    calendar = lowercase(strip(_string_attribute(raw, "calendar", "standard")))
    DT = try
        CFTime.timetype(calendar)
    catch
        throw(ArgumentError(
            "time axis '$(name)' of $(_dataset_path(ds)) has the unsupported calendar $(repr(calendar))"))
    end
    return DT
end

function _check_cf_origin(class::CFTimeUnits, DT, ds, name::String)
    calendar = lowercase(strip(_string_attribute(NCDatasets.variable(ds, name),
                                                 "calendar", "standard")))
    if class.date !== nothing
        y, m, d = class.date
        d <= CFTime.daysinmonth(DT, y, m) || throw(ArgumentError(
            "time axis '$(name)' of $(_dataset_path(ds)) has the units $(repr(class.units)), " *
            "whose origin is not a valid instant: $(y)-$(lpad(m, 2, '0'))-$(lpad(d, 2, '0')) " *
            "does not exist in the $(calendar) calendar"))
    end
    return nothing
end

function _cf_calendar_stamps(class::CFTimeUnits, ds, name::String)
    raw = NCDatasets.variable(ds, name)
    calendar = lowercase(strip(_string_attribute(raw, "calendar", "standard")))
    DT = _time_calendar_type(ds, name)
    _check_cf_origin(class, DT, ds, name)
    if DT in _GREGORIAN_LIKE
        vals = Array(NCDatasets.cfvariable(ds, name; units = class.units, calendar,
                                           maskingvalue = missing))
        any(ismissing, vals) && throw(ArgumentError(
            "time variable '$(name)' of $(_dataset_path(ds)) has fill values"))
        return calendar, DateTime[DateTime(x) for x in vals]
    end
    ms_per_unit = Dates.value(Dates.Millisecond(class.period))
    ms = Int64[round(Int64, x * ms_per_unit) for x in _raw_time_values(ds, name)]
    return calendar, CFTime.timedecode(DT, ms, "milliseconds since $(class.origin)")
end

function _decode(::CFDecoder, class::CFTimeUnits, ds, name::String)
    calendar, stamps = _cf_calendar_stamps(class, ds, name)
    return DateTime[_gregorian_datetime(t, calendar, name, ds) for t in stamps]
end

# A stamp as a DateTime: itself, or the calendar date and time of day of a
# non-Gregorian CFTime stamp, which must exist in the proleptic Gregorian
# calendar.
_gregorian_datetime(t::DateTime, calendar, name, ds) = t
function _gregorian_datetime(t, calendar, name, ds)
    parts = (Dates.year(t), Dates.month(t), Dates.day(t), Dates.hour(t), Dates.minute(t),
             Dates.second(t), Dates.millisecond(t))
    parts[3] <= Dates.daysinmonth(parts[1], parts[2]) || throw(ArgumentError(
        "time axis '$(name)' of $(_dataset_path(ds)) has the $(calendar) date " *
        "$(parts[1])-$(lpad(parts[2], 2, '0'))-$(lpad(parts[3], 2, '0')), which a DateTime " *
        "(proleptic Gregorian) cannot hold, so decode_time cannot return it; " *
        "decode_month_keys gives the months of such an axis"))
    return DateTime(parts...)
end

_month_start(index::Integer) = (k = _month_key(index); DateTime(k.year, k.month, 1))

function _decode(::MonthsSince, class::MonthsSinceUnits, ds, name::String)
    base = 12 * class.year + (class.month - 1)
    return DateTime[_month_start(base + floor(Int, v + MONTHS_SINCE_FLOOR_OFFSET))
                    for v in _raw_time_values(ds, name)]
end

_stamp_label(::MonthStart) = "month-start"
_stamp_label(::MidMonth) = "mid-month"
_stamp_label(::MonthEnd) = "month-end"

# First instant of the month of decimal year `v` (months since year 0 = 12 v).
function _decimal_year_month_start(v::Float64, stamp::AbstractMonthStamp, name, ds)
    m = 12 * v - _stamp_offset(stamp)
    k̂ = round(Int, m)
    abs(m - k̂) <= DECIMAL_YEAR_RESIDUAL_TOLERANCE || throw(ArgumentError(
        "time axis '$(name)' of $(_dataset_path(ds)): decimal year $(v) is $(m - k̂) " *
        "months from a $(_stamp_label(stamp)) stamp (tolerance $(DECIMAL_YEAR_RESIDUAL_TOLERANCE))"))
    return _month_start(k̂)
end

_decode(d::DecimalYears, ::UnlabelledUnits, ds, name::String) =
    DateTime[_decimal_year_month_start(v, d.stamp, name, ds) for v in _raw_time_values(ds, name)]

function _decode(d::AuxiliaryTimeVariable, ::UnlabelledUnits, ds, name::String)
    haskey(ds, d.variable) || throw(ArgumentError(
        "auxiliary time variable '$(d.variable)' not found in $(_dataset_path(ds))"))
    n = length(NCDatasets.variable(ds, name))
    vals = _raw_time_values(ds, d.variable)
    length(vals) == n || throw(DimensionMismatch(
        "auxiliary time variable '$(d.variable)' has $(length(vals)) values; the axis " *
        "'$(name)' has $(n)"))
    ms_per_unit = Dates.value(Dates.Millisecond(d.unit))
    t0 = DateTime(d.origin)
    return DateTime[_month_start(_month_index(_month_of_stamp(
                        t0 + Dates.Millisecond(round(Int64, v * ms_per_unit)), d.stamp)))
                    for v in vals]
end

# ---------------------------------------------------------------------------
# CF encoding of contiguous intervals (the flux writers)
# ---------------------------------------------------------------------------

# Calendar of every time axis the flux writers produce: the runtime's clock is
# the proleptic Gregorian calendar of `DateTime`.
const _WRITER_CALENDAR = "proleptic_gregorian"

# Milliseconds per hour, from `Dates`.
const _MS_PER_HOUR = Dates.value(Dates.Millisecond(Dates.Hour(1)))

# The intervals of a written series must be non-empty, start-ordered and
# contiguous: slice k holds on [start_k, stop_k) and stop_k = start_{k+1}, so a
# start-stamped stepwise reading (the runtime's `StepwiseFlux`) holds each slice
# for exactly its interval.
function _check_contiguous(intervals::AbstractVector{TimeInterval})
    isempty(intervals) && throw(ArgumentError("a flux series needs at least one time interval"))
    for k in 1:(length(intervals) - 1)
        a, b = intervals[k], intervals[k + 1]
        a.stop == b.start || throw(ArgumentError(
            "flux intervals must be contiguous: interval $(k) ends at $(a.stop), interval " *
            "$(k + 1) starts at $(b.start) ($(a.stop < b.start ? "a gap" : "an overlap"))"))
    end
    return intervals
end

# "hours since <first day of the first interval> 00:00:00".
_interval_time_units(intervals::AbstractVector{TimeInterval}) =
    "hours since " * Dates.format(Date(first(intervals).start), dateformat"yyyy-mm-dd") * " 00:00:00"

_hours_since(t::DateTime, origin::DateTime) = Dates.value(t - origin) / _MS_PER_HOUR

"""
    _define_interval_time_axis!(ds, intervals)

Define the time axis of a flux file written from contiguous `intervals`:
dimensions `time` (one per interval) and `nv` (2); `time` (Float64) holds each
interval's start, `time_bnds` `(nv, time)` its start and stop, both in
`"hours since <first day> 00:00:00"` (the first interval's day at midnight) with
`calendar = "proleptic_gregorian"`; `time` names `time_bnds` as its `bounds`.
Whole-hour stamps are stored exactly. The stored values are decoded back
through CFTime, and a stamp that does not round-trip to its `DateTime` is an
error.
"""
function _define_interval_time_axis!(ds, intervals::AbstractVector{TimeInterval})
    _check_contiguous(intervals)
    units = _interval_time_units(intervals)
    origin = DateTime(Date(first(intervals).start))
    starts = Float64[_hours_since(iv.start, origin) for iv in intervals]
    stops = Float64[_hours_since(iv.stop, origin) for iv in intervals]
    decoded = CFTime.timedecode(vcat(starts, stops), units, _WRITER_CALENDAR)
    expected = vcat([iv.start for iv in intervals], [iv.stop for iv in intervals])
    all(DateTime.(decoded) .== expected) || throw(ArgumentError(
        "time stamps of the intervals do not round-trip through $(repr(units))"))
    n = length(intervals)
    defDim(ds, "time", n)
    haskey(ds.dim, "nv") || defDim(ds, "nv", 2)
    defVar(ds, "time", Float64, ("time",);
           attrib = Dict("units" => units, "calendar" => _WRITER_CALENDAR,
                         "standard_name" => "time", "axis" => "T",
                         "long_name" => "start of the interval each slice holds for",
                         "bounds" => "time_bnds"))[:] = starts
    defVar(ds, "time_bnds", Float64, ("nv", "time");
           attrib = Dict("units" => units, "calendar" => _WRITER_CALENDAR,
                         "long_name" => "start and end of the interval each slice holds for"))[:, :] =
        permutedims(hcat(starts, stops))
    return nothing
end
