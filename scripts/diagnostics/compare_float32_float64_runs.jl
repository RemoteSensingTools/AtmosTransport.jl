#!/usr/bin/env julia
# ===========================================================================
# compare_float32_float64_runs.jl: Float32 against Float64 twin runs
#
# Two runs with the same configuration apart from `[numerics] float_type`,
# each writing daily snapshot files with `<tracer>_total_mass` (the global
# total, a compensated Float64 sum at every snapshot) and
# `<tracer>_column_mean` (the dry column-mean mixing ratio). Per tracer:
#
#   * the relative difference of the global totals at every common snapshot,
#     (M32 - M64)/M64, raw and net of the first snapshot
#     (`Analysis.twin_relative_difference`; the first snapshot carries the
#     Float32 rounding of the initial mixing ratios, a constant offset);
#   * once per day, at `--hour` UTC, the area-weighted mean and RMS of the
#     column-mean difference F32 - F64 over the cells (weights: the Float64
#     file's `cell_area`; `Analysis.weighted_difference_stats`), and the
#     difference field of the last such snapshot.
#
# Files are matched by the date in their names (the 8 digits after the last
# `_`, `<prefix>_YYYYMMDD.nc`); only days present in both runs are read.
# Snapshot `time` counts hours since the run start; `--origin` gives that
# start (default: the first common day at 00 UTC). The first and last file of
# each run must hold frames of their own day, which catches a wrong origin.
# Totals are compared at the snapshot times both runs hold, column means at
# the snapshots stamped exactly `--hour`:00 UTC (for a run started at 00 UTC
# the first one is the initial state). Both runs must start at the same
# snapshot: the net difference is taken against it, which is the offset of the
# initial state when the files start at the run start (for a later first day,
# the difference accumulated after that day's first snapshot).
#
# Outputs, in --out:
#   float32_vs_float64.json  {"days": [first, last, n], "tracers": {name:
#       {final_total_rel_net_of_start, final_total_rel_raw,
#        final_column_mean_rms, final_column_mean_mean, unit, ...}}}; the
#       column-mean values are in the display unit `unit` (ppm, ppt, ...);
#   float32_vs_float64.nc    the series behind it: totals of both runs and
#       their relative differences on `time`, the daily column-mean mean and
#       RMS on `sample_time` (mol mol-1, with `display_scale`/`display_units`
#       attributes) and the number of cells each sample leaves out for
#       non-finite values, the last-day difference fields on (Xdim, Ydim, nf),
#       and `lons`, `lats`, `cell_area`.
# scripts/visualization/plot_float32_vs_float64.py draws the figures from
# these two files.
#
# Usage:
#   julia --project=. scripts/diagnostics/compare_float32_float64_runs.jl \
#       --f32 '/temp1/.../merra2_full_f32/catrine_c90_merra2_*.nc' \
#       --f64 '~/data/.../catrine_merra2_f64/catrine_c90_merra2_*.nc' --out DIR \
#       [--hour 0] [--origin 2021-12-01] [--tracers co2_natural,sf6] [--label 'MERRA-2 C90']
# The file arguments are a directory plus a file-name pattern with `*`/`?`
# (quote it). Without --tracers: co2_natural, co2_fossil_from_dec2021, sf6,
# rn222 (the CATRINE benchmark tracers).
#
# Replaces the numerics of scripts/diagnostics/compare_float32_float64_runs.py
# (last in commit a5df02ae).
# ===========================================================================

using Dates
using Printf
using JSON3
using NCDatasets
using AtmosTransport.AtomicFiles: atomic_publish
using AtmosTransport.Output: define_provenance_attributes!
using AtmosTransport.Analysis: daily_snapshot_series, twin_relative_difference, weighted_difference_stats

const USAGE = """
usage: compare_float32_float64_runs.jl --f32 PATTERN --f64 PATTERN --out DIR
                                       [--hour H] [--origin YYYY-MM-DD[THH:MM]] [--tracers a,b,...] [--label TEXT]"""

"""Display label, scale and unit of a tracer's column-mean difference."""
const DISPLAY = Dict(
    "co2_natural"             => (label = "CO₂", scale = 1e6, unit = "ppm"),
    "co2_fossil_from_dec2021" => (label = "Fossil CO₂ (from 2021-12-01)", scale = 1e6, unit = "ppm"),
    "sf6"                     => (label = "SF₆", scale = 1e12, unit = "ppt"),
    "rn222"                   => (label = "Rn-222", scale = 1e21, unit = "10⁻²¹ mol/mol"),
)
const DEFAULT_TRACERS = ["co2_natural", "co2_fossil_from_dec2021", "sf6", "rn222"]
display_of(tracer) = get(DISPLAY, tracer, (label = tracer, scale = 1.0, unit = "mol mol-1"))

const JSON_NAME = "float32_vs_float64.json"
const NETCDF_NAME = "float32_vs_float64.nc"

# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

"""
    glob_files(pattern) -> Vector{String}

Files matching `pattern`, a directory plus a file name with `*` (any run of
characters) and `?` (one character), sorted.
"""
function glob_files(pattern::AbstractString)
    path = expanduser(pattern)
    dir, name = dirname(path), basename(path)
    occursin(r"[*?\[]", dir) && throw(ArgumentError("$(repr(pattern)): wildcards are allowed in the file name only"))
    isdir(isempty(dir) ? "." : dir) || throw(ArgumentError("$(repr(pattern)): no directory $(repr(dir))"))
    escaped = replace(name, r"[\\^$.|+()\[\]{}]" => s -> "\\" * s)
    regex = Regex("^" * replace(escaped, "*" => "[^/]*", "?" => ".") * "\$")
    return sort!([joinpath(dir, f) for f in readdir(isempty(dir) ? "." : dir) if occursin(regex, f)])
end

"""
    files_by_day(files) -> Dict{Date, String}

Daily snapshot files keyed by the date in their names: the 8 digits that
start the last `_`-separated part (`catrine_c90_merra2_20211201.nc`).
"""
function files_by_day(files)
    days = Dict{Date, String}()
    for f in files
        m = match(r"_(\d{8})[^_]*$", basename(f))
        m === nothing && throw(ArgumentError("$(f): no `_YYYYMMDD` date stamp in the file name"))
        day = Date(m[1], dateformat"yyyymmdd")
        haskey(days, day) && throw(ArgumentError("two files for $(day): $(days[day]) and $(f)"))
        days[day] = f
    end
    return days
end

# The frames of the daily file for `day` must lie in [day 00 UTC, day + 1 00 UTC].
function check_file_day(path, day::Date, origin::DateTime)
    hours = NCDataset(ds -> Float64.(collect(ds["time"].var[:])), path, "r")
    stamps = [origin + Millisecond(round(Int64, h * 3_600_000)) for h in hours]
    lo, hi = DateTime(day), DateTime(day) + Day(1)
    all(t -> lo <= t <= hi, stamps) || throw(ArgumentError(
        "$(path): frames $(first(stamps)) .. $(last(stamps)) are not on $(day) with origin $(origin); " *
        "pass the run start with --origin"))
    return nothing
end

# Indices of the times both series hold, in time order.
function common_frames(t32::Vector{DateTime}, t64::Vector{DateTime})
    common = sort!(intersect(t32, t64))
    isempty(common) && throw(ArgumentError("the two runs share no snapshot time"))
    return common, indexin(common, t32), indexin(common, t64)
end

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

"""
    compare_twin_runs(files32, files64; tracers, hour = 0, origin = nothing)

Compare two runs given as lists of daily snapshot files (matched by the dates
in their names). Returns a NamedTuple with the common `days`, the `origin`,
the total frames `times`, the column-mean `sample_times`, the F64 geometry
(`lons`, `lats`, `cell_area`) and, per tracer, the totals, their relative
differences (`raw`, `net`), the daily column-mean difference statistics
(`mean`, `rms` in mol mol-1, and `excluded`, the number of cells left out for
non-finite values at each sample) and the last `field` difference.
"""
function compare_twin_runs(files32::AbstractVector{<:AbstractString}, files64::AbstractVector{<:AbstractString};
                           tracers::AbstractVector{<:AbstractString} = DEFAULT_TRACERS,
                           hour::Integer = 0, origin::Union{Nothing, DateTime} = nothing)
    0 <= hour <= 23 || throw(ArgumentError("hour must be in 0:23; got $(hour)"))
    isempty(tracers) && throw(ArgumentError("no tracers given"))
    by32, by64 = files_by_day(files32), files_by_day(files64)
    days = sort!(collect(intersect(keys(by32), keys(by64))))
    isempty(days) && throw(ArgumentError("the two runs have no day in common"))
    paths32, paths64 = [by32[d] for d in days], [by64[d] for d in days]
    origin = something(origin, DateTime(first(days)))
    for (paths, i) in ((paths32, 1), (paths32, length(days)), (paths64, 1), (paths64, length(days)))
        check_file_day(paths[i], days[i], origin)
    end

    total_vars = ["$(v)_total_mass" for v in tracers]
    field_vars = ["$(v)_column_mean" for v in tracers]
    at_hour = t -> Time(t) == Time(hour)
    tot32 = daily_snapshot_series(paths32, total_vars; origin)
    tot64 = daily_snapshot_series(paths64, total_vars; origin)
    fld32 = daily_snapshot_series(paths32, field_vars; origin, keep_frame = at_hour)
    fld64 = daily_snapshot_series(paths64, field_vars; origin, keep_frame = at_hour)
    first(tot32.times) == first(tot64.times) || throw(ArgumentError(
        "the runs start at different snapshots ($(first(tot32.times)) in Float32, $(first(tot64.times)) " *
        "in Float64); the net difference needs their common first snapshot"))
    times, i32, i64 = common_frames(tot32.times, tot64.times)
    sample_times, j32, j64 = common_frames(fld32.times, fld64.times)

    lons, lats, cell_area = NCDataset(first(paths64), "r") do ds
        Tuple(Array{Float64}(nomissing(ds[name][:, :, :])) for name in ("lons", "lats", "cell_area"))
    end

    results = Dict{String, Any}()
    for v in tracers
        m32 = Float64.(tot32.values["$(v)_total_mass"][i32])
        m64 = Float64.(tot64.values["$(v)_total_mass"][i64])
        rel = twin_relative_difference(m32, m64)
        a = fld32.values["$(v)_column_mean"]
        b = fld64.values["$(v)_column_mean"]
        size(a)[1:3] == size(b)[1:3] == size(cell_area) || throw(DimensionMismatch(
            "$(v)_column_mean is $(size(a)[1:3]) (F32) and $(size(b)[1:3]) (F64); cell_area $(size(cell_area))"))
        stats = [weighted_difference_stats(view(a, :, :, :, j), view(b, :, :, :, k), cell_area)
                 for (j, k) in zip(j32, j64)]
        field = Float64.(view(a, :, :, :, last(j32))) .- Float64.(view(b, :, :, :, last(j64)))
        results[v] = (; m32, m64, raw = rel.raw, net = rel.net,
                      mean = [s.mean for s in stats], rms = [s.rms for s in stats],
                      excluded = [length(cell_area) - s.n for s in stats], field)
    end
    return (; days, origin, hour, times, sample_times, lons, lats, cell_area,
            tracers = collect(String, tracers), results)
end

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

_day(d::Date) = Dates.format(d, dateformat"yyyymmdd")

"""
    summary_dict(cmp) -> NamedTuple

The JSON summary: the keys of the Python script it replaces, plus the origin,
the times the final values refer to and the number of cells the final
column-mean statistics leave out for non-finite values. Tracers keep their
order.
"""
function summary_dict(cmp)
    entry(v) = (r = cmp.results[v]; d = display_of(v);
                (; final_total_rel_net_of_start = r.net[end],
                   final_total_rel_raw = r.raw[end],
                   final_column_mean_rms = r.rms[end] * d.scale,
                   final_column_mean_mean = r.mean[end] * d.scale,
                   unit = d.unit,
                   final_total_time = string(cmp.times[end]),
                   final_column_mean_time = string(cmp.sample_times[end]),
                   column_mean_cells_excluded = r.excluded[end]))
    tracers = NamedTuple{Tuple(Symbol.(cmp.tracers))}(Tuple(entry(v) for v in cmp.tracers))
    return (; days = Any[_day(first(cmp.days)), _day(last(cmp.days)), length(cmp.days)],
            origin = string(cmp.origin), column_mean_hour_utc = cmp.hour, tracers)
end

_jsonable(x::AbstractFloat) = isfinite(x) ? Float64(x) : string(x)
_jsonable(x::NamedTuple) = NamedTuple{keys(x)}(map(_jsonable, values(x)))
_jsonable(x::AbstractVector) = [_jsonable(v) for v in x]
_jsonable(x) = x

function write_summary_json(path, summary)
    atomic_publish(path) do tmp
        open(io -> (JSON3.pretty(io, JSON3.write(_jsonable(summary))); println(io)), tmp, "w")
    end
end

_hours_since(times, origin) = [Dates.value(t - origin) / 3_600_000 for t in times]

"""
    write_comparison_netcdf(path, cmp; f32 = "", f64 = "", label = "")

The series behind the summary (see the script header); `f32`, `f64` and
`label` are recorded as global attributes.
"""
function write_comparison_netcdf(path, cmp; f32::AbstractString = "", f64::AbstractString = "",
                                 label::AbstractString = "")
    units = "hours since " * Dates.format(cmp.origin, dateformat"yyyy-mm-dd HH:MM:SS")
    atomic_publish(path) do tmp
        NCDataset(tmp, "c") do ds
            ds.attrib["title"] = "Float32 - Float64 twin-run comparison"
            ds.attrib["label"] = label
            ds.attrib["float32_files"] = f32
            ds.attrib["float64_files"] = f64
            ds.attrib["first_day"] = _day(first(cmp.days))
            ds.attrib["last_day"] = _day(last(cmp.days))
            ds.attrib["n_days"] = length(cmp.days)
            ds.attrib["column_mean_hour_utc"] = cmp.hour
            ds.attrib["source"] = "scripts/diagnostics/compare_float32_float64_runs.jl"
            define_provenance_attributes!(ds)
            ncv = (name, data, dims, attrib) -> defVar(ds, name, data, dims; attrib)
            ncv("time", _hours_since(cmp.times, cmp.origin), ("time",),
                ["units" => units, "long_name" => "snapshot time held by both runs"])
            ncv("sample_time", _hours_since(cmp.sample_times, cmp.origin), ("sample_time",),
                ["units" => units, "long_name" => "daily column-mean snapshot held by both runs"])
            ncv("lons", cmp.lons, ("Xdim", "Ydim", "nf"), ["units" => "degrees_east", "long_name" => "cell-center longitude"])
            ncv("lats", cmp.lats, ("Xdim", "Ydim", "nf"), ["units" => "degrees_north", "long_name" => "cell-center latitude"])
            ncv("cell_area", cmp.cell_area, ("Xdim", "Ydim", "nf"),
                ["units" => "m2", "long_name" => "horizontal cell area (Float64 run)"])
            for v in cmp.tracers
                r = cmp.results[v]
                d = display_of(v)
                display = ["display_label" => d.label, "display_scale" => d.scale, "display_units" => d.unit]
                ncv("$(v)_total_mass_float32", r.m32, ("time",), ["units" => "kg", "long_name" => "$(v) global total, Float32 run"])
                ncv("$(v)_total_mass_float64", r.m64, ("time",), ["units" => "kg", "long_name" => "$(v) global total, Float64 run"])
                ncv("$(v)_total_rel_diff", r.raw, ("time",),
                    ["units" => "1", "long_name" => "(M32 - M64) / M64 of the $(v) global total"])
                ncv("$(v)_total_rel_diff_net", r.net, ("time",),
                    ["units" => "1", "long_name" => "(M32 - M64) / M64 of the $(v) global total, net of the first snapshot",
                     "comment" => "equal to $(v)_total_rel_diff when the Float64 total starts at zero", display[1]])
                ncv("$(v)_column_mean_diff_mean", r.mean, ("sample_time",),
                    ["units" => "mol mol-1", "long_name" => "area-weighted mean of the $(v) column-mean difference F32 - F64",
                     display...])
                ncv("$(v)_column_mean_diff_rms", r.rms, ("sample_time",),
                    ["units" => "mol mol-1", "long_name" => "area-weighted RMS of the $(v) column-mean difference F32 - F64",
                     display...])
                ncv("$(v)_column_mean_cells_excluded", r.excluded, ("sample_time",),
                    ["units" => "1", "long_name" => "cells left out of the $(v) column-mean statistics for non-finite values"])
                ncv("$(v)_column_mean_diff", r.field, ("Xdim", "Ydim", "nf"),
                    ["units" => "mol mol-1", "long_name" => "$(v) column-mean difference F32 - F64 at the last sample_time",
                     "coordinates" => "lons lats", display...])
            end
        end
    end
end

"""
    summary_lines(summary) -> String

One stdout line per tracer, as the Python script printed them.
"""
summary_lines(summary) = join((@sprintf("%-26s total (net) %+.2e  column-mean RMS %.2e %s  mean %+.2e",
                                        v, s.final_total_rel_net_of_start, s.final_column_mean_rms,
                                        s.unit, s.final_column_mean_mean)
                               for (v, s) in pairs(summary.tracers)), "\n") * "\n"

# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

function parse_options(argv)
    f32 = f64 = out = origin = nothing
    hour = 0
    tracers = DEFAULT_TRACERS
    label = ""
    i = 1
    value() = i + 1 <= length(argv) ? argv[i + 1] : throw(ArgumentError("missing value for $(argv[i])\n$(USAGE)"))
    while i <= length(argv)
        arg = argv[i]
        if arg == "--f32"
            f32 = value(); i += 2
        elseif arg == "--f64"
            f64 = value(); i += 2
        elseif arg == "--out"
            out = value(); i += 2
        elseif arg == "--hour"
            hour = something(tryparse(Int, value()), -1)
            0 <= hour <= 23 || throw(ArgumentError("--hour expects an integer in 0:23, got `$(value())`"))
            i += 2
        elseif arg == "--origin"
            origin = tryparse(DateTime, value())
            origin === nothing && throw(ArgumentError("--origin expects YYYY-MM-DD[THH:MM[:SS]], got `$(value())`"))
            i += 2
        elseif arg == "--tracers"
            tracers = String.(filter(!isempty, strip.(split(value(), ",")))); i += 2
        elseif arg == "--label"
            label = value(); i += 2
        else
            throw(ArgumentError("unknown argument `$(arg)`\n$(USAGE)"))
        end
    end
    (f32 === nothing || f64 === nothing || out === nothing) &&
        throw(ArgumentError("--f32, --f64 and --out are required\n$(USAGE)"))
    return (; f32, f64, out = abspath(expanduser(out)), origin, hour, tracers, label)
end

function main(argv = ARGS)
    if any(in(("-h", "--help")), argv)
        println(USAGE)
        return 0
    end
    opts = parse_options(argv)
    files32, files64 = glob_files(opts.f32), glob_files(opts.f64)
    isempty(files32) && throw(ArgumentError("no file matches --f32 $(opts.f32)"))
    isempty(files64) && throw(ArgumentError("no file matches --f64 $(opts.f64)"))
    cmp = compare_twin_runs(files32, files64; tracers = opts.tracers, hour = opts.hour, origin = opts.origin)
    println("$(length(cmp.days)) common days $(_day(first(cmp.days))) .. $(_day(last(cmp.days)))")
    mkpath(opts.out)
    summary = summary_dict(cmp)
    write_comparison_netcdf(joinpath(opts.out, NETCDF_NAME), cmp; f32 = opts.f32, f64 = opts.f64, label = opts.label)
    write_summary_json(joinpath(opts.out, JSON_NAME), summary)
    for v in cmp.tracers
        n = cmp.results[v].excluded
        any(>(0), n) && @warn "$(v): cells with non-finite values were left out of the daily column-mean " *
                               "statistics at $(count(>(0), n)) of $(length(n)) samples (at most $(maximum(n)) cells)"
    end
    print(summary_lines(summary))
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
