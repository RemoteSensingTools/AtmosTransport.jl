#!/usr/bin/env julia
# ===========================================================================
# float32_conservation_attribution.jl: run configs and budget tables of the
# Float32 tracer-conservation attribution
#
# The experiment behind `docs/src/theory/float32_conservation.md`: short runs
# with operator subsets in Float32 and Float64, compared through the global
# tracer totals. Two met archives are covered:
#
#   merra2  MERRA-2 C90 L72 hourly windows (GEOS-Chem archive), 2021-12-01..07;
#           advection / + GEOS-Chem non-local VDIFF / + CMFMC (DQRCU cloud
#           base) / all; tracers: a source-free 400 ppm background, CO2 (CAMS
#           3-hourly flux), SF6 (LSCE initial state, EDGAR cs_native flux) and
#           fossil CO2 (GridFED, from zero).
#   era5    ERA5 C90 L66 with TM5 convection, 2021-12-01..03, no sources;
#           advection / + TM5 dkg diffusion / + TM5 convection / all; tracers:
#           the 400 ppm background and SF6 from its initial state.
#
# Without sources the global total is invariant, so any drift is arithmetic
# error. With sources, a Float32 run is compared with the Float64 run of the
# same configuration. Totals are the `<tracer>_total_mass` snapshot variables,
# accumulated during the run in Float64 with compensated summation.
#
# Commands:
#
#   write-configs  Builds the 16 run configs (2 archives x 4 operator sets x 2
#                  precisions) as Dicts, checks each with `validate_config`
#                  (nothing is written unless all pass) and writes
#                  DIR/<tag>.toml with `TOML.print`, plus the empty output
#                  folders DIR/out/<tag>/. Snapshots go to
#                  DIR/out/<tag>/<prefix>_YYYYMMDD.nc every 3 hours.
#   budget         Reads the snapshot totals of every run found and prints, per
#                  archive, operator set and tracer:
#                    days       length of the Float32 series;
#                    F32 drift  (M_end - M_0)/M_0 of the Float32 run, for a
#                               tracer that starts with mass;
#                    F32-F64    Analysis.twin_relative_difference at the last
#                               common frame: net of the first frame (the
#                               Float32 rounding of the initial state,
#                               Float32(400e-6) is 2.53e-8 low), or relative to
#                               the emitted mass for a tracer that starts at
#                               zero (fossil);
#                    mean/3h, std/3h, t
#                               for source-free tracers, the mean, sample
#                               standard deviation and t-value of the 3-hourly
#                               relative increments (Analysis.drift_statistics).
#                  A set whose Float32 run has no files is skipped; without its
#                  Float64 run the F32-F64 column is blank.
#
# Run tags: `--tag TEMPLATE` names the runs (config file, output folder) with
# the placeholders {archive} (merra2, era5), {set} (adv, adv_diff, adv_conv,
# adv_dkg, adv_tm5conv, full), {float_type} (float32, float64) and {ft} (f32,
# f64). The default is {archive}_{set}_{float_type}. A template may be given
# for one archive (`--tag merra2=TEMPLATE`) or one archive and precision
# (`--tag merra2:float64=TEMPLATE`); the most specific one applies. `--prefix`
# is the snapshot file prefix (default `attribution`). The 2026-10-06 workspace
# (`~/data/AtmosTransport/output/precision_attribution`, memo
# docs/memos/2026-10-06_float32_conservation_attribution.md) names its runs
# <set>_<variant>_<ft> (MERRA-2) and era5_<set>_<variant>_<ft>, with prefix
# `precision`. The "after" rows of the documentation page are the Float32 run
# full_final2 against the Float64 run full_final, and era5_*_final:
#
#   julia --project=. scripts/diagnostics/float32_conservation_attribution.jl budget \
#       --out ~/data/AtmosTransport/output/precision_attribution --prefix precision \
#       --tag 'merra2:float32={set}_final2_f32' --tag 'merra2:float64={set}_final_f64' \
#       --tag 'era5=era5_{set}_final_{ft}'
#
# Usage:
#   julia --project=. scripts/diagnostics/float32_conservation_attribution.jl write-configs \
#       --out DIR [--data-root ~/data/AtmosTransport] [--tag [KEY=]TEMPLATE ...] [--prefix NAME]
#   julia --project=. scripts/run_transport.jl DIR/merra2_full_float32.toml     # etc., on a GPU
#   julia --project=. scripts/diagnostics/float32_conservation_attribution.jl budget \
#       --out DIR [--tag [KEY=]TEMPLATE ...] [--prefix NAME]
#
# Replaces scripts/diagnostics/float32_conservation_attribution.py (last in
# commit a5df02ae).
# ===========================================================================

using Dates
using Printf
using TOML
using AtmosTransport: validate_config
using AtmosTransport.AtomicFiles: atomic_publish
using AtmosTransport.Analysis: daily_snapshot_series, drift_statistics, twin_relative_difference
using AtmosTransport.Parameters: SPECIES_MOLAR_MASS, DEFAULT_BACKGROUND_VMR

const START_DATE = Date(2021, 12, 1)
const SNAPSHOT_INTERVAL_HOURS = 3
const DEFAULT_TAG_TEMPLATE = "{archive}_{set}_{float_type}"
const DEFAULT_PREFIX = "attribution"
const FLOAT_TYPES = (Float32, Float64)

const USAGE = """
usage: float32_conservation_attribution.jl write-configs --out DIR [--data-root DIR]
                                           [--tag [KEY=]TEMPLATE ...] [--prefix NAME]
       float32_conservation_attribution.jl budget --out DIR [--tag [KEY=]TEMPLATE ...] [--prefix NAME]
  TEMPLATE placeholders: {archive} {set} {float_type} {ft}; default $(DEFAULT_TAG_TEMPLATE)
  KEY: an archive (merra2, era5) or archive:float_type (e.g. era5:float64)
  --prefix: snapshot file prefix, default $(DEFAULT_PREFIX); --data-root default ~/data/AtmosTransport"""

# ---------------------------------------------------------------------------
# The experiment: one type per met archive
# ---------------------------------------------------------------------------

abstract type AttributionArchive end

"""MERRA-2 C90 L72 (GEOS-Chem archive), 7 days, with surface sources."""
struct MERRA2Attribution <: AttributionArchive end

"""ERA5 C90 L66 with TM5 convection, 3 days, no sources."""
struct ERA5Attribution <: AttributionArchive end

const ARCHIVES = (MERRA2Attribution(), ERA5Attribution())

archive_name(::MERRA2Attribution) = "merra2"
archive_name(::ERA5Attribution) = "era5"

end_date(::MERRA2Attribution) = Date(2021, 12, 7)
end_date(::ERA5Attribution) = Date(2021, 12, 3)

input_binaries(::MERRA2Attribution, root) =
    (folder = "$(root)/met/merra2/c90/transport_binary_v4_l72_f32_physics",
     file_pattern = "merra2_transport_{YYYYMMDD}_float32.bin")
input_binaries(::ERA5Attribution, root) =
    (folder = "$(root)/met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour_v3",
     file_pattern = "era5_n320_transport_{YYYYMMDD}_float32.bin")

tracer_names(::MERRA2Attribution) = ("background", "co2_natural", "sf6", "fossil")
tracer_names(::ERA5Attribution) = ("background", "sf6")

"Tracers without sources or sinks: their total is invariant."
source_free_tracers(::MERRA2Attribution) = ("background",)
source_free_tracers(::ERA5Attribution) = ("background", "sf6")

geoschem_diffusion() = Dict{String, Any}("kind" => "geoschem_nonlocal_vdiff")
cmfmc_convection() = Dict{String, Any}("kind" => "cmfmc", "cloud_base" => "dqrcu")
tm5_dkg_diffusion() = Dict{String, Any}("kind" => "tm5_dkg", "surface_flux_boundary" => true)
tm5_convection() = Dict{String, Any}("kind" => "tm5", "tile_workspace_gib" => 0.25, "use_collab_lu" => true,
                                     "lmax_conv" => 66, "n_merge" => 1)

"Operator sets in table order: name => the physics sections added to advection."
operator_sets(::MERRA2Attribution) = [
    "adv"      => Dict{String, Any}(),
    "adv_diff" => Dict{String, Any}("diffusion" => geoschem_diffusion()),
    "adv_conv" => Dict{String, Any}("convection" => cmfmc_convection()),
    "full"     => Dict{String, Any}("diffusion" => geoschem_diffusion(), "convection" => cmfmc_convection()),
]
operator_sets(::ERA5Attribution) = [
    "adv"         => Dict{String, Any}(),
    "adv_dkg"     => Dict{String, Any}("diffusion" => tm5_dkg_diffusion()),
    "adv_tm5conv" => Dict{String, Any}("convection" => tm5_convection()),
    "full"        => Dict{String, Any}("diffusion" => tm5_dkg_diffusion(), "convection" => tm5_convection()),
]

background_tracer() = Dict{String, Any}("init" => Dict{String, Any}("kind" => "uniform",
                                                                    "background" => DEFAULT_BACKGROUND_VMR))
sf6_initial_state(root) = Dict{String, Any}(
    "kind" => "file", "file" => "$(root)/catrine/InitialConditions/startSF6_202112010000.nc",
    "variable" => "SF6")

# `molar_mass_kg_mol` is the current surface-flux key; step P2c of the port
# replaces it with `species` (design memo 2026-10-10, Section 0).
tracers_table(::ERA5Attribution, root) = Dict{String, Any}(
    "background" => background_tracer(),
    "sf6" => Dict{String, Any}("init" => sf6_initial_state(root)))
tracers_table(::MERRA2Attribution, root) = Dict{String, Any}(
    "background" => background_tracer(),
    "co2_natural" => Dict{String, Any}(
        "init" => Dict{String, Any}("kind" => "catrine_co2"),
        "surface_flux" => Dict{String, Any}(
            "kind" => "lmdz_co2",
            "files" => ["$(root)/catrine/Emissions/LMDZ_fluxes/z_cams_l_cams55_202112_FT24r2_ra_sfc_3h_co2_flux.nc"],
            "time_varying" => true, "temporal_scheme" => "stepwise")),
    "sf6" => Dict{String, Any}(
        "init" => sf6_initial_state(root),
        "surface_flux" => Dict{String, Any}(
            "kind" => "cs_native", "file" => "$(root)/catrine/fluxes_c90/catrine_sf6_c90.nc",
            "variable" => "sf6", "molar_mass_kg_mol" => SPECIES_MOLAR_MASS.sf6,
            "time_varying" => true, "temporal_scheme" => "stepwise")),
    "fossil" => Dict{String, Any}(
        "init" => Dict{String, Any}("kind" => "uniform", "background" => 0.0),
        "surface_flux" => Dict{String, Any}(
            "kind" => "gridfed_fossil_co2",
            "files" => ["$(root)/catrine/Emissions/gridfed/GCP-GridFEDv2024.0_2021.short.nc"],
            "molar_mass_kg_mol" => SPECIES_MOLAR_MASS.co2,
            "time_varying" => true, "temporal_scheme" => "stepwise")))

# ---------------------------------------------------------------------------
# Run tags
# ---------------------------------------------------------------------------

"""
    TagTemplates(specs = String[])

Run-tag templates from `--tag` values `[KEY=]TEMPLATE`, KEY an archive name or
`archive:float32|float64`. Without a general template the default
`{archive}_{set}_{float_type}` applies.
"""
struct TagTemplates
    general::String
    specific::Dict{String, String}
end

function TagTemplates(specs::AbstractVector{<:AbstractString} = String[])
    general = nothing
    specific = Dict{String, String}()
    keys_allowed = [k for a in ARCHIVES for k in (archive_name(a),
                    ("$(archive_name(a)):$(lowercase(string(FT)))" for FT in FLOAT_TYPES)...)]
    for spec in specs
        m = match(r"^([A-Za-z0-9_]+(?::[A-Za-z0-9_]+)?)=(.*)$", spec)
        key, template = m === nothing ? (nothing, String(spec)) : (String(m[1]), String(m[2]))
        isempty(template) && throw(ArgumentError("--tag $(repr(spec)): empty template"))
        if key === nothing
            general === nothing || throw(ArgumentError("--tag: two general templates given"))
            general = template
        else
            key in keys_allowed || throw(ArgumentError(
                "--tag $(repr(spec)): unknown key `$(key)`; use one of $(join(keys_allowed, ", "))"))
            haskey(specific, key) && throw(ArgumentError("--tag: two templates for `$(key)`"))
            specific[key] = template
        end
    end
    return TagTemplates(something(general, DEFAULT_TAG_TEMPLATE), specific)
end

"""
    run_tag(templates, archive, set, FT) -> String

The tag of one run: the most specific template filled in.
"""
function run_tag(t::TagTemplates, archive::AttributionArchive, set::AbstractString, ::Type{FT}) where {FT}
    name = archive_name(archive)
    float_type = lowercase(string(FT))
    template = get(t.specific, "$(name):$(float_type)", get(t.specific, name, t.general))
    tag = replace(template, "{archive}" => name, "{set}" => set, "{float_type}" => float_type,
                  "{ft}" => replace(float_type, "float" => "f"))
    occursin(r"[{}]", tag) && throw(ArgumentError(
        "tag template $(repr(template)) has an unknown placeholder; use {archive}, {set}, {float_type}, {ft}"))
    (isempty(tag) || occursin('/', tag) || tag in (".", "..")) &&
        throw(ArgumentError("tag template $(repr(template)) gives the invalid run tag $(repr(tag))"))
    return tag
end

# ---------------------------------------------------------------------------
# write-configs
# ---------------------------------------------------------------------------

"""
    attribution_config(archive, set, FT; data_root, out, tag, prefix) -> Dict{String, Any}

The run config of one archive, operator set and float type.
"""
function attribution_config(archive::AttributionArchive, set::AbstractString, ::Type{FT};
                            data_root::AbstractString, out::AbstractString,
                            tag::AbstractString, prefix::AbstractString) where {FT}
    physics = Dict(operator_sets(archive))
    haskey(physics, set) || throw(ArgumentError("$(archive_name(archive)) has no operator set `$(set)`"))
    binaries = input_binaries(archive, data_root)
    cfg = Dict{String, Any}(
        "input" => Dict{String, Any}("folder" => binaries.folder,
                                     "start_date" => string(START_DATE),
                                     "end_date" => string(end_date(archive)),
                                     "file_pattern" => binaries.file_pattern),
        "architecture" => Dict{String, Any}("use_gpu" => true, "backend" => "cuda"),
        "numerics" => Dict{String, Any}("float_type" => string(FT)),
        "advection" => Dict{String, Any}("scheme" => "ppm"),
        "tracers" => tracers_table(archive, data_root),
        "output" => Dict{String, Any}(
            "format" => "netcdf", "split" => "daily",
            "snapshot_interval_hours" => SNAPSHOT_INTERVAL_HOURS,
            "path" => "$(out)/out/$(tag)/$(prefix)_{YYYYMMDD}.nc",
            "fields" => Dict{String, Any}("layers" => "none", "column_mean" => false,
                                          "column_mass_per_area" => false, "air_mass" => false)))
    merge!(cfg, deepcopy(physics[set]))
    return cfg
end

"""
    attribution_runs(tags = TagTemplates()) -> Vector{NamedTuple}

Every run of the experiment in table order, as `(; archive, set, float_type,
tag)`: 2 archives x 4 operator sets x (Float32, Float64). Two runs with one
tag are an error.
"""
function attribution_runs(tags::TagTemplates = TagTemplates())
    runs = [(; archive, set, float_type = FT, tag = run_tag(tags, archive, set, FT))
            for archive in ARCHIVES for (set, _) in operator_sets(archive) for FT in FLOAT_TYPES]
    all_tags = [r.tag for r in runs]
    dup = [t for t in unique(all_tags) if count(==(t), all_tags) > 1]
    isempty(dup) || throw(ArgumentError("the tag templates give one tag to several runs: $(join(dup, ", "))"))
    return runs
end

_check_prefix(prefix) = (isempty(prefix) || occursin('/', prefix)) &&
    throw(ArgumentError("invalid snapshot file prefix $(repr(prefix))"))

"""
    write_configs(out; data_root, tags, prefix, io = stdout) -> Vector{String}

Build, validate and write the 16 run configs; returns their paths. Every
config must pass `validate_config` (which also checks that the binaries
exist) before any file is written.
"""
function write_configs(out::AbstractString; data_root::AbstractString,
                       tags::TagTemplates = TagTemplates(), prefix::AbstractString = DEFAULT_PREFIX,
                       io::IO = stdout)
    _check_prefix(prefix)
    out = abspath(expanduser(out))
    data_root = abspath(expanduser(data_root))
    runs = attribution_runs(tags)
    configs = [attribution_config(r.archive, r.set, r.float_type; data_root, out, r.tag, prefix) for r in runs]
    failures = String[]
    for (r, cfg) in zip(runs, configs)
        ok, errors = validate_config(cfg)
        ok || push!(failures, "$(r.tag):\n  " * join(errors, "\n  "))
    end
    isempty(failures) || throw(ArgumentError(
        "validate_config rejected $(length(failures)) of $(length(configs)) configs; none written:\n" *
        join(failures, "\n")))
    paths = String[]
    for (r, cfg) in zip(runs, configs)
        mkpath(joinpath(out, "out", r.tag))
        path = joinpath(out, "$(r.tag).toml")
        atomic_publish(path) do tmp
            open(tmp, "w") do fh
                println(fh, "# Float32 conservation attribution: $(archive_name(r.archive)), $(r.set), $(r.float_type).")
                println(fh, "# Written by scripts/diagnostics/float32_conservation_attribution.jl write-configs.")
                TOML.print(fh, cfg; sorted = true)
            end
        end
        push!(paths, path)
    end
    println(io, "$(length(paths)) configs written to $(out) (validate_config: all pass)")
    return paths
end

# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------

"""
    snapshot_files(out, tag, prefix) -> Vector{String}

The daily snapshot files `out/out/<tag>/<prefix>_YYYYMMDD.nc`, in date order.
"""
function snapshot_files(out::AbstractString, tag::AbstractString, prefix::AbstractString)
    dir = joinpath(out, "out", tag)
    isdir(dir) || return String[]
    pattern = Regex("^" * replace(prefix, r"[\\^$.|?*+()\[\]{}]" => s -> "\\" * s) * raw"_\d{8}\.nc$")
    return sort!([joinpath(dir, f) for f in readdir(dir) if occursin(pattern, f)])
end

"""
    run_totals(out, tag, prefix, tracers) -> (; times, values) or nothing

The `<tracer>_total_mass` series of one run (nothing when it has no files),
stamped from the experiment start. The series must start with the initial
state, be spaced by the experiment's snapshot interval and be finite.
"""
function run_totals(out, tag, prefix, tracers)
    files = snapshot_files(out, tag, prefix)
    isempty(files) && return nothing
    series = daily_snapshot_series(files, ["$(n)_total_mass" for n in tracers];
                                   origin = DateTime(START_DATE))
    dir = joinpath(out, "out", tag)
    steps = unique(diff(series.times))
    (isempty(steps) || steps == [Hour(SNAPSHOT_INTERVAL_HOURS)]) || throw(ArgumentError(
        "$(dir): snapshot spacing $(sort(steps)) is not the experiment's " *
        "$(SNAPSHOT_INTERVAL_HOURS) h; a frame is missing or the run used another interval"))
    first(series.times) == DateTime(START_DATE) || throw(ArgumentError(
        "$(dir): the first snapshot is $(first(series.times)), not the initial state at $(DateTime(START_DATE))"))
    for (name, v) in series.values
        all(isfinite, v) || throw(ArgumentError("$(dir): `$(name)` holds non-finite values"))
    end
    return series
end

"""
    budget_rows(out; tags, prefix) -> Vector{NamedTuple}

One row per archive, operator set (with a Float32 run) and tracer:
`(; archive, set, tracer, days, drift, f32_minus_f64, increments)`, where
`drift` is `nothing` for a tracer that starts at zero, `f32_minus_f64` is
`nothing` without the Float64 run, and `increments` is the `drift_statistics`
result for source-free tracers (else `nothing`).
"""
function budget_rows(out::AbstractString; tags::TagTemplates = TagTemplates(),
                     prefix::AbstractString = DEFAULT_PREFIX)
    _check_prefix(prefix)
    out = abspath(expanduser(out))
    rows = NamedTuple[]
    runs = attribution_runs(tags)
    for archive in ARCHIVES
        names = tracer_names(archive)
        for (set, _) in operator_sets(archive)
            tag32, tag64 = (only(r.tag for r in runs if r.archive == archive && r.set == set && r.float_type == FT)
                            for FT in FLOAT_TYPES)
            s32 = run_totals(out, tag32, prefix, names)
            s32 === nothing && continue
            s64 = run_totals(out, tag64, prefix, names)
            common = s64 === nothing ? DateTime[] : intersect(s32.times, s64.times)
            s64 === nothing || !isempty(common) || throw(ArgumentError(
                "runs $(tag32) and $(tag64) share no snapshot time"))
            for n in names
                a = s32.values["$(n)_total_mass"]
                stats = a[1] > 0 ? drift_statistics(a) : nothing
                diff = if s64 === nothing
                    nothing
                else
                    b = s64.values["$(n)_total_mass"]
                    twin_relative_difference(a[indexin(common, s32.times)],
                                             b[indexin(common, s64.times)]).net[end]
                end
                push!(rows, (; archive = archive_name(archive), set, tracer = n,
                             days = Dates.value(s32.times[end] - s32.times[1]) / Dates.value(Millisecond(Day(1))),
                             drift = stats === nothing ? nothing : stats.drift,
                             f32_minus_f64 = diff,
                             increments = n in source_free_tracers(archive) ? stats : nothing))
            end
        end
    end
    return rows
end

const _BLANK = lpad("-", 10)
_e(x, digits) = x === nothing ? _BLANK : Printf.format(Printf.Format("%10.$(digits)e"), x)

"""
    budget_table(rows) -> String

The budget table, in the layout (and digits) of the Python script it replaces.
"""
function budget_table(rows)
    h = SNAPSHOT_INTERVAL_HOURS
    io = IOBuffer()
    println(io, rpad("archive", 7), " ", rpad("operators", 10), " ", rpad("tracer", 12), " ", lpad("days", 5), " ",
            lpad("F32 drift", 10), " ", lpad("F32-F64", 10), " ", lpad("mean/$(h)h", 10), " ",
            lpad("std/$(h)h", 10), " ", lpad("t", 6))
    for r in rows
        inc = r.increments === nothing ? "$(_BLANK) $(_BLANK) $(lpad("-", 6))" :
              @sprintf("%10.1e %10.1e %6.1f", r.increments.increment_mean, r.increments.increment_std, r.increments.t)
        println(io, rpad(r.archive, 7), " ", rpad(r.set, 10), " ", rpad(r.tracer, 12), " ",
                @sprintf("%5.1f", r.days), " ", _e(r.drift, 2), " ", _e(r.f32_minus_f64, 2), " ", inc)
    end
    return String(take!(io))
end

# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

function parse_options(argv)
    out = nothing
    data_root = joinpath(homedir(), "data", "AtmosTransport")
    tag_specs = String[]
    prefix = DEFAULT_PREFIX
    i = 1
    value() = i + 1 <= length(argv) ? argv[i + 1] : throw(ArgumentError("missing value for $(argv[i])\n$(USAGE)"))
    while i <= length(argv)
        arg = argv[i]
        if arg == "--out"
            out = value(); i += 2
        elseif arg == "--data-root"
            data_root = value(); i += 2
        elseif arg == "--tag"
            push!(tag_specs, value()); i += 2
        elseif arg == "--prefix"
            prefix = value(); i += 2
        else
            throw(ArgumentError("unknown argument `$(arg)`\n$(USAGE)"))
        end
    end
    out === nothing && throw(ArgumentError("--out is required\n$(USAGE)"))
    return (; out, data_root, tags = TagTemplates(tag_specs), prefix)
end

function main(argv = ARGS)
    if isempty(argv) || any(in(("-h", "--help")), argv)
        println(USAGE)
        return isempty(argv) ? 2 : 0
    end
    command = argv[1]
    opts = parse_options(argv[2:end])
    if command == "write-configs"
        write_configs(opts.out; data_root = opts.data_root, tags = opts.tags, prefix = opts.prefix)
    elseif command == "budget"
        print(stdout, budget_table(budget_rows(opts.out; tags = opts.tags, prefix = opts.prefix)))
    else
        throw(ArgumentError("unknown command `$(command)`; use write-configs or budget\n$(USAGE)"))
    end
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
