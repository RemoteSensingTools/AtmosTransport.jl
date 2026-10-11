"""
The Float32 diagnostics command lines on synthetic twin outputs written into
`mktempdir()`:

`scripts/diagnostics/float32_conservation_attribution.jl`
- tag templates: the default, general, per-archive and per-archive-and-precision
  templates (most specific wins), unknown keys and placeholders, invalid and
  colliding tags;
- the 16 run configs: built as Dicts equal to the configs the Python script
  wrote (two pinned texts), molar masses and the background from
  `Parameters`, written with `TOML.print` and read back equal, every one
  passing `validate_config` (on a data root with stand-in binaries); nothing
  is written when `validate_config` fails;
- `budget`: days, drift, F32 - F64 (net of the first frame; relative to the
  emitted mass for a tracer from zero), increment statistics of source-free
  tracers, blanks and skipped sets, the table text in the Python layout, the
  file prefix and tag templates, the command line, and the errors for a
  missing frame, a run without its initial state and a non-finite total.

`scripts/diagnostics/compare_float32_float64_runs.jl`
- file patterns and day matching (an extra day in one run is ignored;
  duplicate or missing date stamps are errors); the origin check;
- totals compared at the common snapshot times (a frame missing in one run is
  skipped; runs that start at different snapshots are an error), column means
  at the snapshots of `hour` UTC, cells with fill values left out and counted
  per sample;
- the JSON summary (the Python keys, display units) and the NetCDF (series,
  last difference field, geometry, display attributes) against direct
  computations, the stdout lines and the command line (help anywhere in the
  arguments, for both scripts).
"""

using Test
using Dates
using Printf
using Statistics: mean, std
using TOML
using JSON3
using NCDatasets
using AtmosTransport: validate_config
using AtmosTransport.Analysis: daily_snapshot_series, weighted_difference_stats
using AtmosTransport.Parameters: DEFAULT_BACKGROUND_VMR

const SCRIPTS = joinpath(@__DIR__, "..", "..", "scripts", "diagnostics")
const ATT = Module()
Base.include(ATT, joinpath(SCRIPTS, "float32_conservation_attribution.jl"))
const TWIN = Module()
Base.include(TWIN, joinpath(SCRIPTS, "compare_float32_float64_runs.jl"))
att(f::Symbol, args...; kw...) = Base.invokelatest(getproperty(ATT, f), args...; kw...)
twin(f::Symbol, args...; kw...) = Base.invokelatest(getproperty(TWIN, f), args...; kw...)

function error_message(f)
    try
        f()
        return "no error"
    catch err
        return sprint(showerror, err)
    end
end

const NC = 2                       # cells per panel edge of the fixtures

# A snapshot-like daily file: `time` in bare hours, `<tracer>_total_mass` and
# (optionally) `<tracer>_column_mean` of type `T` with a 1e15 fill value, and
# the geometry variables.
function write_snapshot(path, hours; totals = Dict{String, Vector{Float64}}(),
                        columns = Dict{String, Array{<:Real, 4}}(), T = Float32)
    NCDataset(path, "c") do ds
        defDim(ds, "Xdim", NC); defDim(ds, "Ydim", NC); defDim(ds, "nf", 6); defDim(ds, "time", length(hours))
        defVar(ds, "time", Float64, ("time",), attrib = Dict("units" => "hours"))[:] = hours
        defVar(ds, "lons", Float64, ("Xdim", "Ydim", "nf"))[:, :, :] = [10.0i + 2j + 60p for i in 1:NC, j in 1:NC, p in 1:6]
        defVar(ds, "lats", Float64, ("Xdim", "Ydim", "nf"))[:, :, :] = [-80.0 + 5i + 3j + 20p for i in 1:NC, j in 1:NC, p in 1:6]
        defVar(ds, "cell_area", Float64, ("Xdim", "Ydim", "nf"))[:, :, :] = AREA
        for (name, v) in totals
            defVar(ds, "$(name)_total_mass", Float64, ("time",), attrib = Dict("units" => "kg"))[:] = v
        end
        for (name, v) in columns
            var = defVar(ds, "$(name)_column_mean", T, ("Xdim", "Ydim", "nf", "time");
                         fillvalue = T(1e15), attrib = Dict("units" => "mol mol-1 dry"))
            var.var[:, :, :, :] = T.(v)
        end
    end
    return path
end

const AREA = [1.0e10 * (1 + 0.1i + 0.05j + 0.01p) for i in 1:NC, j in 1:NC, p in 1:6]
day_hours(d) = d == 1 ? collect(0.0:3:24) : collect(24.0 * (d - 1) + 3 : 3 : 24.0 * d)

# ---------------------------------------------------------------------------
# float32_conservation_attribution.jl: tags and configs
# ---------------------------------------------------------------------------

@testset "attribution tag templates" begin
    T(specs...) = att(:TagTemplates, collect(String, specs))
    merra2, era5 = ATT.MERRA2Attribution(), ATT.ERA5Attribution()
    @test att(:run_tag, T(), merra2, "adv_diff", Float32) == "merra2_adv_diff_float32"
    @test att(:run_tag, T("{set}-{ft}"), era5, "full", Float64) == "full-f64"
    t = T("merra2={set}_final_{ft}", "era5=era5_{set}_{ft}", "merra2:float64={set}_final_f64_ref")
    @test att(:run_tag, t, merra2, "full", Float32) == "full_final_f32"
    @test att(:run_tag, t, merra2, "full", Float64) == "full_final_f64_ref"
    @test att(:run_tag, t, era5, "adv_dkg", Float64) == "era5_adv_dkg_f64"
    @test att(:run_tag, T("x_{archive}_{set}_{float_type}"), era5, "adv", Float32) == "x_era5_adv_float32"
    @test occursin("unknown key", error_message(() -> T("geos=a_{set}")))
    @test occursin("two general", error_message(() -> T("a_{set}", "b_{set}")))
    @test occursin("two templates", error_message(() -> T("era5=a_{set}", "era5=b_{set}")))
    @test occursin("empty template", error_message(() -> T("era5=")))
    @test occursin("unknown placeholder", error_message(() -> att(:run_tag, T("{met}_{set}"), era5, "adv", Float32)))
    @test occursin("invalid run tag", error_message(() -> att(:run_tag, T("a/{set}"), era5, "adv", Float32)))
    runs = att(:attribution_runs)
    @test length(runs) == 16
    @test [r.tag for r in runs[1:3]] == ["merra2_adv_float32", "merra2_adv_float64", "merra2_adv_diff_float32"]
    @test runs[end].tag == "era5_full_float64"
    @test occursin("one tag to several runs", error_message(() -> att(:attribution_runs, T("{archive}_{float_type}"))))
end

# Two configs as the Python script wrote them (ROOT = --data-root, OUT = --out).
const PYTHON_MERRA2_FULL_F32 = raw"""
[input]
folder       = "ROOT/met/merra2/c90/transport_binary_v4_l72_f32_physics"
start_date   = "2021-12-01"
end_date     = "2021-12-07"
file_pattern = "merra2_transport_{YYYYMMDD}_float32.bin"
[architecture]
use_gpu = true
backend = "cuda"
[numerics]
float_type = "Float32"
[advection]
scheme = "ppm"
[tracers.background]
  [tracers.background.init]
  kind = "uniform"
  background = 400e-6               # no source: total mass is invariant
[tracers.co2_natural]
  [tracers.co2_natural.init]
  kind = "catrine_co2"
  [tracers.co2_natural.surface_flux]
  kind = "lmdz_co2"
  files = ["ROOT/catrine/Emissions/LMDZ_fluxes/z_cams_l_cams55_202112_FT24r2_ra_sfc_3h_co2_flux.nc"]
  time_varying = true
  temporal_scheme = "stepwise"
[tracers.sf6]
  [tracers.sf6.init]
  kind = "file"
  file = "ROOT/catrine/InitialConditions/startSF6_202112010000.nc"
  variable = "SF6"
  [tracers.sf6.surface_flux]
  kind = "cs_native"
  file = "ROOT/catrine/fluxes_c90/catrine_sf6_c90.nc"
  variable = "sf6"
  species = "sf6"
  time_varying = true
  temporal_scheme = "stepwise"
[tracers.fossil]
  [tracers.fossil.init]
  kind = "uniform"
  background = 0.0
  [tracers.fossil.surface_flux]
  kind = "gridfed_fossil_co2"
  files = ["ROOT/catrine/Emissions/gridfed/GCP-GridFEDv2024.0_2021.short.nc"]
  species = "co2"
  time_varying = true
  temporal_scheme = "stepwise"
[diffusion]
kind = "geoschem_nonlocal_vdiff"
[convection]
kind = "cmfmc"
cloud_base = "dqrcu"
[output]
format = "netcdf"
split = "daily"
snapshot_interval_hours = 3
path = "OUT/out/merra2_full_float32/attribution_{YYYYMMDD}.nc"
  [output.fields]
  layers = "none"
  column_mean = false
  column_mass_per_area = false
  air_mass = false
"""

const PYTHON_ERA5_TM5CONV_F64 = raw"""
[input]
folder       = "ROOT/met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour_v3"
start_date   = "2021-12-01"
end_date     = "2021-12-03"
file_pattern = "era5_n320_transport_{YYYYMMDD}_float32.bin"
[architecture]
use_gpu = true
backend = "cuda"
[numerics]
float_type = "Float64"
[advection]
scheme = "ppm"
[tracers.background]
  [tracers.background.init]
  kind = "uniform"
  background = 400e-6               # no source: total mass is invariant
[tracers.sf6]
  [tracers.sf6.init]
  kind = "file"
  file = "ROOT/catrine/InitialConditions/startSF6_202112010000.nc"
  variable = "SF6"
[convection]
kind = "tm5"
tile_workspace_gib = 0.25
use_collab_lu = true
lmax_conv = 66
n_merge = 1
[output]
format = "netcdf"
split = "daily"
snapshot_interval_hours = 3
path = "OUT/out/era5_adv_tm5conv_float64/attribution_{YYYYMMDD}.nc"
  [output.fields]
  layers = "none"
  column_mean = false
  column_mass_per_area = false
  air_mass = false
"""

# A data root with stand-in binaries: `validate_config` checks that they exist.
function stand_in_data_root()
    root = mktempdir()
    for (archive, days) in ((ATT.MERRA2Attribution(), 1:7), (ATT.ERA5Attribution(), 1:3))
        b = att(:input_binaries, archive, root)
        mkpath(b.folder)
        for d in days
            touch(joinpath(b.folder, replace(b.file_pattern, "{YYYYMMDD}" => @sprintf("202112%02d", d))))
        end
    end
    return root
end

@testset "attribution configs" begin
    root = stand_in_data_root()
    out = mktempdir()
    python(text) = TOML.parse(replace(text, "ROOT" => root, "OUT" => out))
    cfg = att(:attribution_config, ATT.MERRA2Attribution(), "full", Float32;
              data_root = root, out, tag = "merra2_full_float32", prefix = "attribution")
    @test cfg == python(PYTHON_MERRA2_FULL_F32)
    @test att(:attribution_config, ATT.ERA5Attribution(), "adv_tm5conv", Float64;
              data_root = root, out, tag = "era5_adv_tm5conv_float64", prefix = "attribution") ==
          python(PYTHON_ERA5_TM5CONV_F64)
    @test cfg["tracers"]["sf6"]["surface_flux"]["species"] == "sf6"
    @test cfg["tracers"]["fossil"]["surface_flux"]["species"] == "co2"
    @test cfg["tracers"]["background"]["init"]["background"] === DEFAULT_BACKGROUND_VMR
    adv = att(:attribution_config, ATT.MERRA2Attribution(), "adv", Float64; data_root = root, out,
              tag = "t", prefix = "p")
    @test !haskey(adv, "diffusion") && !haskey(adv, "convection")
    @test adv["output"]["path"] == "$(out)/out/t/p_{YYYYMMDD}.nc"
    @test_throws ArgumentError att(:attribution_config, ATT.ERA5Attribution(), "adv_diff", Float32;
                                   data_root = root, out, tag = "t", prefix = "p")

    # write-configs: 16 validated configs, read back equal to the Dicts.
    io = IOBuffer()
    paths = att(:write_configs, out; data_root = root, io)
    @test length(paths) == 16
    @test occursin("16 configs written", String(take!(io)))
    for (r, path) in zip(att(:attribution_runs), paths)
        @test path == joinpath(out, "$(r.tag).toml")
        @test isdir(joinpath(out, "out", r.tag))
        parsed = TOML.parsefile(path)
        @test parsed == att(:attribution_config, r.archive, r.set, r.float_type; data_root = root, out,
                            tag = r.tag, prefix = "attribution")
        @test first(validate_config(parsed))
    end
    @test startswith(read(paths[end], String), "# Float32 conservation attribution: era5, full, Float64.")

    # Custom tag and prefix.
    out2 = mktempdir()
    paths2 = att(:write_configs, out2; data_root = root, tags = att(:TagTemplates, ["merra2={set}_{ft}"]),
                 prefix = "precision", io = devnull)
    @test joinpath(out2, "full_f32.toml") in paths2 && joinpath(out2, "era5_adv_float64.toml") in paths2
    @test TOML.parsefile(joinpath(out2, "full_f32.toml"))["output"]["path"] ==
          "$(out2)/out/full_f32/precision_{YYYYMMDD}.nc"
    @test_throws ArgumentError att(:write_configs, out2; data_root = root, prefix = "a/b", io = devnull)

    # A config that fails validate_config: nothing is written.
    rm(joinpath(root, "met", "era5"); recursive = true)
    out3 = mktempdir()
    msg = error_message(() -> att(:write_configs, out3; data_root = root, io = devnull))
    @test occursin("rejected 8 of 16", msg) && occursin("era5_adv_float32", msg)
    @test isempty(readdir(out3))
end

# ---------------------------------------------------------------------------
# float32_conservation_attribution.jl: budget
# ---------------------------------------------------------------------------

# Totals of a run over `days` daily files (3-hourly frames, the runtime layout).
function write_run(dir, tag, prefix, days, series::Dict{String, <:Function}; skip_hour = nothing)
    mkpath(joinpath(dir, "out", tag))
    k0 = 0
    for d in days
        hours = filter(h -> h != skip_hour, day_hours(d))
        totals = Dict(name => [f(h) for h in hours] for (name, f) in series)
        write_snapshot(joinpath(dir, "out", tag, @sprintf("%s_202112%02d.nc", prefix, d)), hours; totals)
    end
end

const B = 2.0e15
merra2_f64() = Dict{String, Function}(
    "background" => h -> B, "co2_natural" => h -> 1.0e15 * (1 + 1e-5h), "sf6" => h -> 5.0e9 * (1 + 2e-5h),
    "fossil" => h -> 1.0e9h)
merra2_f32() = Dict{String, Function}(
    "background" => h -> B * (1 - 2.5e-8) * (1 + 1e-9 * sin(h)), "co2_natural" => h -> 1.0e15 * (1 + 1e-5h) * (1 + 3e-9h),
    "sf6" => h -> 5.0e9 * (1 + 2e-5h) * (1 - 1e-9h), "fossil" => h -> 1.0e9h * (1 + 1e-7))

@testset "attribution budget" begin
    dir = mktempdir()
    write_run(dir, "merra2_full_float32", "attribution", 1:2, merra2_f32())
    write_run(dir, "merra2_full_float64", "attribution", 1:2, merra2_f64())
    write_run(dir, "merra2_adv_float32", "other", 1:2, merra2_f32())          # another prefix: not read
    era5 = Dict{String, Function}("background" => h -> B * (1 + 1e-9 * cos(h)), "sf6" => h -> 5.0e9 * (1 - 1e-10h))
    write_run(dir, "era5_adv_float32", "attribution", 1:1, era5)
    rows = att(:budget_rows, dir)
    @test [(r.archive, r.set) for r in rows] == [fill(("merra2", "full"), 4); fill(("era5", "adv"), 2)]
    @test [r.tracer for r in rows] == ["background", "co2_natural", "sf6", "fossil", "background", "sf6"]

    hours = 0.0:3:48
    series(f) = [f(h) for h in hours]
    for r in rows[1:4]
        a, b = series(merra2_f32()[r.tracer]), series(merra2_f64()[r.tracer])
        @test r.days == 2.0
        if r.tracer == "fossil"
            @test r.drift === nothing
            @test r.f32_minus_f64 == (a[end] - b[end]) / (b[end] - b[1])         # relative to the emitted mass
        else
            @test r.drift == (a[end] - a[1]) / a[1]
            @test r.f32_minus_f64 == (a[end] - b[end]) / b[end] - (a[1] - b[1]) / b[1]
        end
        if r.tracer == "background"
            d = diff(a) ./ a[1]
            @test r.increments.increment_mean ≈ mean(d) rtol = 1e-14
            @test r.increments.increment_std ≈ std(d) rtol = 1e-14
            @test r.increments.t ≈ mean(d) / (std(d) / sqrt(16)) rtol = 1e-13
        else
            @test r.increments === nothing
        end
    end
    @test rows[5].days == 1.0 && rows[5].f32_minus_f64 === nothing && rows[6].increments !== nothing

    # The table in the layout of the Python script.
    lines = split(att(:budget_table, rows), '\n'; keepempty = false)
    @test lines[1] == "archive operators  tracer        days  F32 drift    F32-F64    mean/3h     std/3h      t"
    bg = rows[1]
    @test lines[2] == @sprintf("%-7s %-10s %-12s %5.1f %10.2e %10.2e %10.1e %10.1e %6.1f", "merra2", "full",
                               "background", 2.0, bg.drift, bg.f32_minus_f64, bg.increments.increment_mean,
                               bg.increments.increment_std, bg.increments.t)
    @test lines[5] == @sprintf("%-7s %-10s %-12s %5.1f %10s %10.2e %10s %10s %6s", "merra2", "full", "fossil",
                               2.0, "-", rows[4].f32_minus_f64, "-", "-", "-")
    @test lines[6] == @sprintf("%-7s %-10s %-12s %5.1f %10.2e %10s %10.1e %10.1e %6.1f", "era5", "adv",
                               "background", 1.0, rows[5].drift, "-", rows[5].increments.increment_mean,
                               rows[5].increments.increment_std, rows[5].increments.t)

    # Tag templates and prefix select another layout.
    dir2 = mktempdir()
    write_run(dir2, "full_final_f32", "precision", 1:2, merra2_f32())
    write_run(dir2, "full_final_f64", "precision", 1:2, merra2_f64())
    rows2 = att(:budget_rows, dir2; tags = att(:TagTemplates, ["merra2={set}_final_{ft}"]), prefix = "precision")
    @test [r.f32_minus_f64 for r in rows2] == [r.f32_minus_f64 for r in rows[1:4]]
    @test isempty(att(:budget_rows, dir2))

    # A missing frame in the Float32 run breaks the 3-hour spacing; a run must
    # start with the initial state (the net difference is taken against it) and
    # hold finite totals (a NaN start must not pass as a tracer from zero).
    dir3 = mktempdir()
    write_run(dir3, "merra2_full_float32", "attribution", 1:2, merra2_f32(); skip_hour = 30.0)
    @test occursin("snapshot spacing", error_message(() -> att(:budget_rows, dir3)))
    dir4 = mktempdir()
    write_run(dir4, "merra2_full_float32", "attribution", 1:2, merra2_f32())
    write_run(dir4, "merra2_full_float64", "attribution", 2:2, merra2_f64())
    @test occursin("not the initial state", error_message(() -> att(:budget_rows, dir4)))
    dir5 = mktempdir()
    nan_start = merra2_f32()
    nan_start["background"] = h -> h == 0 ? NaN : B
    write_run(dir5, "merra2_full_float32", "attribution", 1:1, nan_start)
    @test occursin("non-finite", error_message(() -> att(:budget_rows, dir5)))

    # Command line.
    text = mktemp() do path, io
        redirect_stdout(io) do
            @test att(:main, ["budget", "--out", dir]) == 0
        end
        close(io)
        read(path, String)
    end
    @test text == att(:budget_table, rows)
    @test occursin("unknown command", error_message(() -> att(:main, ["plot", "--out", dir])))
    @test occursin("--out is required", error_message(() -> att(:main, ["budget"])))
    @test occursin("unknown argument", error_message(() -> att(:main, ["budget", "--out", dir, "--days", "3"])))
    help = mktemp() do path, io                                    # help after the command too
        redirect_stdout(() -> (@test att(:main, ["budget", "--help"]) == 0), io)
        close(io)
        read(path, String)
    end
    @test startswith(help, "usage: float32_conservation_attribution.jl")
end

# ---------------------------------------------------------------------------
# compare_float32_float64_runs.jl
# ---------------------------------------------------------------------------

v64(name, h) = [(name == "co2_natural" ? 400e-6 : 1e-12) * (1 + 1e-3 * sin(i + 2j + 3p) + 1e-5 * h / 24)
                for i in 1:NC, j in 1:NC, p in 1:6]
v32(name, h) = Float32.(v64(name, h) .* (1 .+ 1e-6 .* cos.(h .+ [i + j + p for i in 1:NC, j in 1:NC, p in 1:6])))
m64(name, h) = (name == "co2_natural" ? 2.0e15 : 0.0) + 1.0e9h
m32(name, h) = m64(name, h) * (1 - 2.5e-8) * (1 + 1e-9h / 24)

# A twin run of `days`; `fill_at` = (hour, cell) stores the fill value there;
# `skip_hour` drops a frame.
function write_twin_run(dir, prefix, days, T; fill_at = nothing, skip_hour = nothing)
    mkpath(dir)
    names = ("co2_natural", "my_tracer")
    for d in days
        hours = filter(h -> h != skip_hour, day_hours(d))
        totals = Dict(n => [T === Float32 ? m32(n, h) : m64(n, h) for h in hours] for n in names)
        columns = Dict{String, Array{<:Real, 4}}()
        for n in names
            c = cat((T === Float32 ? Float64.(v32(n, h)) : v64(n, h) for h in hours)...; dims = 4)
            if fill_at !== nothing && fill_at[1] in hours
                view(c, :, :, :, findfirst(==(fill_at[1]), hours))[fill_at[2]] = 1e15
            end
            columns[n] = c
        end
        write_snapshot(joinpath(dir, @sprintf("%s_202112%02d.nc", prefix, d)), hours; totals, columns, T)
    end
    return dir
end

@testset "twin-run comparison" begin
    root = mktempdir()
    d32 = write_twin_run(joinpath(root, "f32"), "run_c2", 1:4, Float32; fill_at = (72.0, 5))
    d64 = write_twin_run(joinpath(root, "f64"), "run_c2", 1:3, Float64; skip_hour = 51.0)
    files32 = twin(:glob_files, joinpath(d32, "run_c2_*.nc"))
    files64 = twin(:glob_files, joinpath(d64, "run_c2_2021120?.nc"))
    @test length(files32) == 4 && length(files64) == 3
    tracers = ["co2_natural", "my_tracer"]
    cmp = twin(:compare_twin_runs, files32, files64; tracers)
    origin = DateTime(2021, 12, 1)
    @test cmp.days == Date(2021, 12, 1):Day(1):Date(2021, 12, 3)
    @test cmp.origin == origin
    hours = filter(!=(51.0), collect(0.0:3:72))                   # 51 h is missing in the F64 run
    @test cmp.times == origin .+ Hour.(Int.(hours))
    @test cmp.sample_times == origin .+ Hour.([0, 24, 48, 72])
    @test cmp.cell_area == AREA

    # Regression against the former all-days-in-memory calculation. The
    # production helper reads one common 3-D snapshot at a time, but its
    # statistics and retained last field are bitwise identical on this fixture.
    field_vars = ["$(n)_column_mean" for n in tracers]
    at_midnight = t -> Time(t) == Time(0)
    old32 = daily_snapshot_series(files32, field_vars; origin, keep_frame = at_midnight)
    old64 = daily_snapshot_series(files64, field_vars; origin, keep_frame = at_midnight)
    old_times, old_i32, old_i64 = twin(:common_frames, old32.times, old64.times)
    streamed = twin(:_column_difference_series, files32, files64, field_vars, AREA;
                    origin, keep_frame = at_midnight)
    @test streamed.times == old_times
    for name in field_vars
        a, b = old32.values[name], old64.values[name]
        stats = [weighted_difference_stats(view(a, :, :, :, j), view(b, :, :, :, k), AREA)
                 for (j, k) in zip(old_i32, old_i64)]
        expected_field = Float64.(view(a, :, :, :, last(old_i32))) .-
                         Float64.(view(b, :, :, :, last(old_i64)))
        got = streamed.values[name]
        @test got.mean == [s.mean for s in stats]
        @test got.rms == [s.rms for s in stats]
        @test got.excluded == [length(AREA) - s.n for s in stats]
        @test isequal(got.field, expected_field)
    end

    w = AREA ./ sum(AREA)
    for n in tracers
        r = cmp.results[n]
        a, b = [m32(n, h) for h in hours], [m64(n, h) for h in hours]
        raw = [y == 0 ? NaN : (x - y) / y for (x, y) in zip(a, b)]
        @test isequal(r.raw, raw)
        @test isequal(r.net, n == "co2_natural" ? raw .- raw[1] : raw)
        for (k, h) in enumerate((0.0, 24.0, 48.0, 72.0))
            d = Float64.(v32(n, h)) .- v64(n, h)
            keep = trues(size(d))
            h == 72.0 && (keep[5] = false)                           # the fill value
            wk = w[keep] ./ sum(w[keep])
            @test r.mean[k] ≈ sum(wk .* d[keep]) rtol = 1e-12
            @test r.rms[k] ≈ sqrt(sum(wk .* d[keep] .^ 2)) rtol = 1e-12
        end
        @test r.excluded == [0, 0, 0, 1]                              # per sample
        last = Float64.(v32(n, 72.0)) .- v64(n, 72.0)
        @test isnan(r.field[5]) && r.field[[1:4; 6:end]] == last[[1:4; 6:end]]
    end

    # Summary, NetCDF and stdout lines.
    s = twin(:summary_dict, cmp)
    @test s.days == ["20211201", "20211203", 3]
    c = s.tracers.co2_natural
    @test c.final_total_rel_net_of_start == cmp.results["co2_natural"].net[end]
    @test c.final_total_rel_raw == cmp.results["co2_natural"].raw[end]
    @test c.final_column_mean_rms == cmp.results["co2_natural"].rms[end] * 1e6
    @test c.final_column_mean_mean == cmp.results["co2_natural"].mean[end] * 1e6
    @test c.unit == "ppm" && c.column_mean_cells_excluded == 1
    @test s.tracers.my_tracer.unit == "mol mol-1"
    @test s.tracers.my_tracer.final_column_mean_rms == cmp.results["my_tracer"].rms[end]
    @test collect(keys(s.tracers)) == [:co2_natural, :my_tracer]
    @test twin(:summary_lines, s) == join([@sprintf("%-26s total (net) %+.2e  column-mean RMS %.2e %s  mean %+.2e",
                                                    n, t.final_total_rel_net_of_start, t.final_column_mean_rms,
                                                    t.unit, t.final_column_mean_mean)
                                           for (n, t) in pairs(s.tracers)], "\n") * "\n"

    out = mktempdir()
    text = mktemp() do path, io
        redirect_stdout(io) do
            @test twin(:main, ["--f32", joinpath(d32, "run_c2_*.nc"), "--f64", joinpath(d64, "*.nc"),
                               "--out", out, "--tracers", "co2_natural,my_tracer", "--label", "C2 test"]) == 0
        end
        close(io)
        read(path, String)
    end
    @test occursin("3 common days 20211201 .. 20211203", text)
    @test endswith(text, twin(:summary_lines, s))
    json = JSON3.read(read(joinpath(out, "float32_vs_float64.json"), String))
    @test json.days == ["20211201", "20211203", 3]
    @test json.tracers.co2_natural.final_total_rel_net_of_start == c.final_total_rel_net_of_start
    @test json.tracers.co2_natural.final_column_mean_mean == c.final_column_mean_mean
    NCDataset(joinpath(out, "float32_vs_float64.nc")) do ds
        @test ds.attrib["label"] == "C2 test" && ds.attrib["n_days"] == 3
        @test ds["time"].attrib["units"] == "hours since 2021-12-01 00:00:00"
        @test ds["time"].var[:] == hours
        @test ds["sample_time"].var[:] == [0.0, 24.0, 48.0, 72.0]
        @test ds["cell_area"][:, :, :] == AREA
        r = cmp.results["co2_natural"]
        @test ds["co2_natural_total_mass_float32"][:] == r.m32
        @test ds["co2_natural_total_rel_diff_net"][:] == r.net
        @test ds["co2_natural_column_mean_diff_rms"][:] == r.rms
        @test ds["co2_natural_column_mean_cells_excluded"][:] == [0, 0, 0, 1]
        @test ds["co2_natural_column_mean_diff_mean"].attrib["display_scale"] == 1e6
        @test ds["co2_natural_column_mean_diff_mean"].attrib["display_units"] == "ppm"
        f = ds["co2_natural_column_mean_diff"].var[:, :, :]
        @test isequal(f, r.field)
        @test haskey(ds.attrib, "framework_commit")
    end

    # Another hour, an explicit origin, and the errors.
    noon = twin(:compare_twin_runs, files32, files64; tracers = ["my_tracer"], hour = 12)
    @test noon.sample_times == origin .+ Hour.([12, 36, 60])
    @test occursin("--origin", error_message(() -> twin(:compare_twin_runs, files32, files64; tracers,
                                                        origin = DateTime(2021, 11, 30))))
    @test twin(:compare_twin_runs, files32, files64; tracers, origin).times == cmp.times
    @test occursin("hour", error_message(() -> twin(:compare_twin_runs, files32, files64; hour = 24)))
    late = write_twin_run(joinpath(root, "f64_late"), "run_c2", 1:3, Float64; skip_hour = 0.0)
    @test occursin("different snapshots",
                   error_message(() -> twin(:compare_twin_runs, files32, twin(:glob_files, joinpath(late, "*.nc")); tracers)))
    @test occursin("no day in common", error_message(() -> twin(:compare_twin_runs, files32[4:4], files64)))
    @test occursin("file name only", error_message(() -> twin(:glob_files, joinpath(root, "f*", "x.nc"))))
    @test occursin("no directory", error_message(() -> twin(:glob_files, joinpath(root, "nope", "*.nc"))))
    @test occursin("two files for", error_message(() -> twin(:files_by_day, ["a_20211201.nc", "b/c_20211201.nc"])))
    @test occursin("date stamp", error_message(() -> twin(:files_by_day, ["run.nc"])))
    @test occursin("required", error_message(() -> twin(:main, ["--f32", "x"])))
    help = mktemp() do path, io
        redirect_stdout(() -> (@test twin(:main, ["--f32", "x", "--help"]) == 0), io)
        close(io)
        read(path, String)
    end
    @test startswith(help, "usage: compare_float32_float64_runs.jl")
end
