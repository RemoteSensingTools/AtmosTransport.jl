#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# test_flux_port_parity_harness.jl
#
# The parity harness runner `scripts/validation/flux_port_parity/run_parity.jl`
# of the surface-flux port. Synthetic C4 `cs_native` files and a TOML list of
# comparisons are written into a temporary directory. Checked: parsing and
# validation of the TOML (relative paths, canonical tolerance keys, masks,
# reference scales, `common_variables_only`, nested tolerance tables, the shared
# tolerance validation (infinite, negative, out-of-range values), every
# configuration error (including empty category tables) raised
# before anything runs), the status of each comparison (pass, fail, including a
# dropped series, error for a units mismatch and for a missing file), that each
# report equals a direct call of the comparator, the
# combined JSON and markdown reports (including the sha256 of the runner and the
# comparator), `--only`, the exit status of the command line, and that the
# example TOML of the directory parses.
# The script is included into a module, as in `test_compare_cs_flux_files.jl`.
# ---------------------------------------------------------------------------

using Test
using Dates
using JSON3
using Logging
using NCDatasets
using SHA: sha256
using AtmosTransport
using AtmosTransport.Grids: CubedSphereMesh, GEOSNativePanelConvention
using AtmosTransport.Parameters: EARTH_RADIUS, SPECIES_MOLAR_MASS, CARBON_MOLAR_MASS

const HARNESS_DIR = joinpath(@__DIR__, "..", "..", "scripts", "validation", "flux_port_parity")
const RUNNER = joinpath(HARNESS_DIR, "run_parity.jl")
const COMPARATOR = joinpath(HARNESS_DIR, "..", "compare_cs_flux_files.jl")
file_sha256(path) = bytes2hex(open(sha256, path))
const H = Module()
Base.include(H, RUNNER)
h_call(f::Symbol, args...; kw...) = Base.invokelatest(getproperty(H, f), args...; kw...)
quiet(f) = with_logger(f, NullLogger())

# The C -> CO2 factor of the Python flux files written before commit 71a08395:
# a legacy encoding the reference-scale preset undoes, not a physical constant.
const LEGACY_CO2_PER_CARBON = 44 / 12

const NC = 4
const MESH = CubedSphereMesh(; Nc = NC, FT = Float64, Hp = 0,
                             convention = GEOSNativePanelConvention(), radius = EARTH_RADIUS)

# Daily cs_native file: `fields` are name => (Xdim, Ydim, nf, time) Float32
# arrays, `static` name => (Xdim, Ydim, nf) arrays.
function write_flux(path, times, fields; static = Pair{String, Any}[], units = "kg CO2 m-2 s-1")
    NCDataset(path, "c") do ds
        defDim(ds, "Xdim", NC); defDim(ds, "Ydim", NC); defDim(ds, "nf", 6)
        defDim(ds, "time", length(times))
        origin = DateTime(2021, 1, 1)
        defVar(ds, "time", Float64, ("time",);
               attrib = Dict("units" => "hours since 2021-01-01 00:00:00", "calendar" => "proleptic_gregorian"))[:] =
            [Millisecond(t - origin) / Millisecond(Hour(1)) for t in times]
        for (name, data) in fields
            defVar(ds, name, Float32, ("Xdim", "Ydim", "nf", "time"); attrib = Dict("units" => units))[:, :, :, :] = data
        end
        for (name, data) in static
            defVar(ds, name, eltype(data), ("Xdim", "Ydim", "nf"))[:, :, :] = data
        end
    end
    return path
end

function smooth_field(nt)
    A = Array{Float32}(undef, NC, NC, 6, nt)
    for k in 1:nt, p in 1:6, j in 1:NC, i in 1:NC
        A[i, j, p, k] = Float32(1e-9 * (1 + 0.1i + 0.01j + 0.2p) * (1 + 0.05k))
    end
    return A
end

function land_mask()
    m = zeros(Float32, NC, NC, 6)
    m[1:2, :, 1] .= 1
    return m
end

const CONFIG_TEXT = """
title = "synthetic C4 parity"

[[comparison]]
name = "identical"
level = "S0"
description = "a file against a copy of itself"
reference = "ref.nc"
candidate = "same.nc"
nc = 4
variables = ["FLUX", "land_fraction"]
mask = { var = "land_fraction", threshold = 0.5 }
tolerances = { bitwise_fraction = 1.0, year_total = 0.0, cell_p99.coastal = 0.0 }

[[comparison]]
name = "scaled_reference"
level = "L2"
reference = "ref.nc"
candidate = "iupac.nc"
nc = 4
reference_scale = { FLUX = "carbon_44_12_to_iupac" }
tolerances = { bitwise_fraction = 1.0 }

[[comparison]]
name = "perturbed"
level = "L1"
reference = "ref.nc"
candidate = "perturbed.nc"
nc = 4
tolerances = { ulp_max = 1, bitwise_fraction = 0.99 }

[[comparison]]
name = "short_span"
level = "R"
reference = "ref.nc"
candidate = "short.nc"
nc = 4
common_times = true
tolerances = { bitwise_fraction = 1.0 }

[[comparison]]
name = "units_differ"
level = "L2"
reference = "ref.nc"
candidate = "units.nc"
nc = 4

[[comparison]]
name = "missing_candidate"
level = "T"
reference = "ref.nc"
candidate = "does_not_exist.nc"
nc = 4

[[comparison]]
name = "dropped_series"
level = "L2"
description = "the candidate lacks the reference's TER series"
reference = "two_series.nc"
candidate = "same.nc"
nc = 4

[[comparison]]
name = "dropped_series_common"
level = "L2"
reference = "two_series.nc"
candidate = "same.nc"
nc = 4
common_variables_only = true
tolerances = { bitwise_fraction = 1.0 }
"""

# One valid comparison followed by `extra` lines, for the configuration errors.
one_comparison(extra; head = "") = """
$(head)
[[comparison]]
name = "a"
level = "L2"
reference = "ref.nc"
candidate = "same.nc"
$(extra)
"""

@testset "flux-port parity harness" begin
    mktempdir() do dir
        times = [DateTime(2021, 1, 1) + Day(k) for k in 0:4]
        A = smooth_field(length(times))
        mask = land_mask()
        K = (SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS) / LEGACY_CO2_PER_CARBON
        ref = write_flux(joinpath(dir, "ref.nc"), times, ["FLUX" => A]; static = ["land_fraction" => mask])
        same = write_flux(joinpath(dir, "same.nc"), times, ["FLUX" => copy(A)]; static = ["land_fraction" => mask])
        write_flux(joinpath(dir, "iupac.nc"), times, ["FLUX" => Float32.(Float64.(A) .* K)])
        B = copy(A); B[2, 2, 2, 3] = nextfloat(B[2, 2, 2, 3], 3)
        write_flux(joinpath(dir, "perturbed.nc"), times, ["FLUX" => B])
        write_flux(joinpath(dir, "short.nc"), times[2:3], ["FLUX" => A[:, :, :, 2:3]])
        write_flux(joinpath(dir, "units.nc"), times, ["FLUX" => A]; units = "kg C m-2 s-1")
        write_flux(joinpath(dir, "two_series.nc"), times, ["FLUX" => A, "TER" => A])
        config = joinpath(dir, "parity.toml")
        write(config, CONFIG_TEXT)
        names = ["identical", "scaled_reference", "perturbed", "short_span", "units_differ", "missing_candidate",
                 "dropped_series", "dropped_series_common"]

        @testset "configuration" begin
            cfg = h_call(:load_parity_config, config)
            @test cfg.title == "synthetic C4 parity" && cfg.description === nothing
            @test [c.name for c in cfg.comparisons] == names
            @test [c.level for c in cfg.comparisons] == ["S0", "L2", "L1", "R", "L2", "T", "L2", "L2"]
            c1 = cfg.comparisons[1]
            @test c1.reference == normpath(joinpath(dir, "ref.nc"))       # relative to the TOML file
            @test c1.nc == 4 && c1.binary === nothing && c1.radius === nothing
            @test c1.variables == ["FLUX" => "FLUX", "land_fraction" => "land_fraction"]
            @test c1.mask == (; var = "land_fraction", file = nothing, threshold = 0.5)
            @test c1.tolerances == ["bitwise_fraction.all" => 1.0, "cell_p99.coastal" => 0.0, "year_total" => 0.0]
            @test c1.description == "a file against a copy of itself" && !c1.common_times
            scale = only(cfg.comparisons[2].reference_scale)
            @test scale.first == "FLUX" && scale.second.factor == K && scale.second.preset == "carbon_44_12_to_iupac"
            @test cfg.comparisons[3].tolerances == ["bitwise_fraction.all" => 0.99, "ulp_max.all" => 1.0]
            nested = joinpath(dir, "nested_tolerances.toml")
            write(nested, one_comparison("nc = 4\ntolerances = { cell_p99 = { interior = 1e-6, polar = 1e-5 } }"))
            @test only(h_call(:load_parity_config, nested).comparisons).tolerances ==
                  ["cell_p99.interior" => 1e-6, "cell_p99.polar" => 1e-5]
            @test cfg.comparisons[4].common_times
            @test [c.common_variables_only for c in cfg.comparisons] == [fill(false, 7); true]

            # Every configuration error is raised when the file is read.
            bad = [
                one_comparison("nc = 4\ntolerance = { cell_max = 1 }"),             # unknown key
                replace(one_comparison("nc = 4"), "level = \"L2\"\n" => ""),        # no level
                replace(one_comparison("nc = 4"), "\"L2\"" => "\"L9\""),            # unknown level
                replace(one_comparison("nc = 4"), "name = \"a\"" => "name = \"a b\""),
                one_comparison("nc = 4\nbinary = \"c4.bin\""),                       # two meshes
                one_comparison(""),                                                 # no mesh
                one_comparison("binary = \"c4.bin\"\nradius = 6.4e6"),
                one_comparison("nc = 0"),
                one_comparison("nc = 4\ntolerances = { nonsense = 1 }"),
                one_comparison("nc = 4\ntolerances = { cell_max = 1, \"cell_max.all\" = 2 }"),
                one_comparison("nc = 4\ntolerances = { cell_max = \"tight\" }"),
                one_comparison("nc = 4\ntolerances = { slice_total.interior = 1 }"),
                one_comparison("nc = 4\ntolerances = { cell_p99.coastal = 1 }"),     # coastal without a mask
                one_comparison("nc = 4\ntolerances = { nonsense = {} }"),            # empty table, unknown metric
                one_comparison("nc = 4\ntolerances = { cell_p99 = {} }"),            # empty category table
                one_comparison("nc = 4\ntolerances = { slice_total = {} }"),         # no categories for totals
                one_comparison("nc = 4\ntolerances = { bitwise_fraction = 1.0, cell_max = {} }"),
                one_comparison("nc = 4\ntolerances = { nonsense = { interior = 1 } }"),
                one_comparison("nc = 4\ntolerances = { slice_total = inf }"),          # would always pass
                one_comparison("nc = 4\ntolerances = { year_total = nan }"),
                one_comparison("nc = 4\ntolerances = { month_total = -1e-9 }"),
                one_comparison("nc = 4\ntolerances = { bitwise_fraction = -1 }"),      # would always pass
                one_comparison("nc = 4\ntolerances = { bitwise_fraction = 1.5 }"),
                one_comparison("nc = 4\ntolerances = { ulp_max = 0.5 }"),              # a count
                one_comparison("nc = 4\ntolerances = { cell_p99 = { interior = -1e-6 } }"),
                one_comparison("nc = 4\nreference_scale = { FLUX = \"nonsense\" }"),
                one_comparison("nc = 4\nreference_scale = { FLUX = 0 }"),
                one_comparison("nc = 4\nmask = { variable = \"land_fraction\" }"),
                one_comparison("nc = 4\nmask = { threshold = 0.5 }"),
                one_comparison("nc = 4\nvariables = \"FLUX\""),
                one_comparison("nc = 4\ncommon_times = \"yes\""),
                one_comparison("nc = 4\ncommon_variables_only = 1"),
                one_comparison("nc = 4\nvariables = [\"FLUX\"]\ncommon_variables_only = true"),
                one_comparison("nc = 4") * one_comparison("nc = 4"),                # repeated name
                one_comparison("nc = 4"; head = "author = \"x\""),                  # unknown top-level key
                "title = \"no comparisons\"\n",
            ]
            for (k, text) in enumerate(bad)
                path = joinpath(dir, "bad_$(k).toml")
                write(path, text)
                @test_throws ArgumentError h_call(:load_parity_config, path)
            end
            @test_throws ArgumentError h_call(:load_parity_config, joinpath(dir, "absent.toml"))
            # Nested tolerance tables: the parent metric is checked before the table is flattened.
            nested_error(text) = (path = joinpath(dir, "nested_error.toml"); write(path, one_comparison(text));
                                  try h_call(:load_parity_config, path); "no error" catch err sprint(showerror, err) end)
            @test occursin("unknown tolerance metric `nonsense`", nested_error("nc = 4\ntolerances = { nonsense = {} }"))
            @test occursin("`slice_total`", nested_error("nc = 4\ntolerances = { slice_total = {} }")) &&
                  occursin("takes no category table", nested_error("nc = 4\ntolerances = { slice_total = {} }"))
            @test occursin("finite and non-negative", nested_error("nc = 4\ntolerances = { slice_total = inf }")) &&
                  occursin("comparison `a`", nested_error("nc = 4\ntolerances = { slice_total = inf }"))
            @test occursin("must lie in [0, 1]", nested_error("nc = 4\ntolerances = { bitwise_fraction = 1.5 }"))
            @test occursin("category table of tolerance `cell_p99`", nested_error("nc = 4\ntolerances = { cell_p99 = {} }")) &&
                  occursin("is empty", nested_error("nc = 4\ntolerances = { cell_p99 = {} }"))
        end

        json = joinpath(dir, "out", "parity.json")
        md = joinpath(dir, "out", "parity.md")
        combined = quiet(() -> h_call(:run_parity, config; json, markdown = md))
        results = Dict(r.name => r for r in combined.comparisons)

        @testset "statuses" begin
            @test [r.name for r in combined.comparisons] == names
            @test Dict(n => r.status for (n, r) in results) ==
                  Dict("identical" => "pass", "scaled_reference" => "pass", "perturbed" => "fail",
                       "short_span" => "pass", "units_differ" => "error", "missing_candidate" => "error",
                       "dropped_series" => "fail", "dropped_series_common" => "pass")
            @test !combined.pass && combined.n_comparisons == 8
            @test combined.n_pass == 4 && combined.n_fail == 2 && combined.n_error == 2
            @test occursin("units differ", results["units_differ"].error) && results["units_differ"].report === nothing
            @test occursin("candidate file not found", results["missing_candidate"].error)
            @test all(r -> r.wall_seconds >= 0, combined.comparisons)
        end

        @testset "reports equal direct comparator calls" begin
            r = results["identical"].report
            flux, lf = r.variables
            @test flux.kind == "series" && lf.kind == "static"
            @test r.categories.n_cells.coastal > 0 && r.categories.mask_threshold == 0.5
            @test r.categories.mask_file == abspath(ref)
            @test [c.metric for c in lf.checks] == ["finite", "bitwise_fraction.all", "cell_p99.coastal", "year_total"]
            @test [c.status for c in lf.checks] == ["pass", "pass", "pass", "n/a"]      # year_total: no time
            direct = quiet() do
                Base.invokelatest(H.FluxFileComparator.compare_cs_flux_files, ref, same, MESH;
                                  variables = ["FLUX", "land_fraction"], mask_var = "land_fraction",
                                  mask_threshold = 0.5,
                                  tolerances = ["bitwise_fraction" => 1.0, "cell_p99.coastal" => 0.0, "year_total" => 0.0])
            end
            @test direct.variables[1].cells == flux.cells && direct.variables[2].cells == lf.cells
            @test direct.variables[1].slice_totals.reference == flux.slice_totals.reference
            @test [c.status for c in direct.variables[2].checks] == [c.status for c in lf.checks]
            s = only(results["scaled_reference"].report.variables)
            @test s.reference_scale.factor == K && s.cells.all.bitwise_equal_fraction == 1.0
            p = only(results["perturbed"].report.variables)
            @test p.cells.all.ulp_max == 3
            @test [(c.metric, c.status) for c in p.checks] ==
                  [("finite", "pass"), ("bitwise_fraction.all", "pass"), ("ulp_max.all", "fail")]
            sr = results["short_span"].report
            @test sr.time.n_compared == 2 && !sr.time.identical && sr.time.pass
            # A dropped series fails the comparison although the common series agrees.
            d = results["dropped_series"].report
            @test !d.pass && only(d.variables).pass && d.variable_selection.series_only_in_reference == ["TER"]
            dc = results["dropped_series_common"].report
            @test dc.pass && dc.variable_selection.common_variables_only
            @test dc.variable_selection.series_only_in_reference == ["TER"]
        end

        @testset "combined JSON and markdown" begin
            parsed = JSON3.read(read(json, String))
            @test parsed.pass === false && parsed.n_pass == 4 && length(parsed.comparisons) == 8
            @test parsed.config == abspath(config) && parsed.title == "synthetic C4 parity"
            @test parsed.comparisons[1].report.variables[1].cells.all.bitwise_equal_fraction == 1.0
            @test parsed.comparisons[1].report.variables[2].checks[4].status == "n/a"
            @test parsed.comparisons[2].spec.reference_scale[1].preset == "carbon_44_12_to_iupac"
            @test parsed.comparisons[5].report === nothing && occursin("units differ", parsed.comparisons[5].error)
            @test parsed.comparisons[1].spec.mask.var == "land_fraction"
            @test parsed.comparisons[8].spec.common_variables_only === true
            # The report names the runner and comparator versions by their sha256.
            @test parsed.scripts.runner.sha256 == file_sha256(RUNNER)
            @test parsed.scripts.comparator.sha256 == file_sha256(COMPARATOR)
            @test parsed.scripts.comparator.path == normpath(abspath(COMPARATOR))
            @test !any(f -> occursin(".tmp-", f), readdir(dirname(json)))
            text = read(md, String)
            @test startswith(text, "# Flux-port parity report")
            @test occursin("- Scripts: `run_parity.jl` sha256 $(file_sha256(RUNNER)), " *
                           "`compare_cs_flux_files.jl` sha256 $(file_sha256(COMPARATOR))", text)
            @test occursin("- Result: **FAIL**, 4 of 8 comparisons pass (2 fail, 2 error)", text)
            @test occursin("| identical | S0 | PASS | 2 | 5 |", text)
            @test occursin("| units_differ | L2 | ERROR | - | - |", text)
            @test occursin("## perturbed (L1): FAIL", text) && occursin("## missing_candidate (T): ERROR", text)
            @test occursin("n/a: not applicable", text)
            @test occursin("| FLUX | series | 0.99932484443795", text)      # the scaled comparison's row
            @test occursin("- Series in one file only: reference `TER`; candidate none **FAIL**", text)
            @test occursin("- Series in one file only: reference `TER`; candidate none " *
                           "(not compared: common_variables_only)", text)
        end

        @testset "selection and command line" begin
            picked = quiet(() -> h_call(:run_parity, config; only = ["identical", "short_span"]))
            @test picked.pass && [r.name for r in picked.comparisons] == ["identical", "short_span"]
            @test_throws ArgumentError h_call(:run_parity, config; only = ["absent"])
            parse = getproperty(H, :_parse_parity_args)
            opts = Base.invokelatest(parse, [config, "--only", "a", "--only", "b", "--json", "x.json"])
            @test opts.config == config && opts.only == ["a", "b"] && opts.json == "x.json" && opts.markdown === nothing
            for bad in (String[], [config, config], [config, "--bogus"], [config, "--json"])
                @test_throws ArgumentError Base.invokelatest(parse, bad)
            end
            main = getproperty(H, :main)
            j2 = joinpath(dir, "out", "only.json")
            @test quiet(() -> redirect_stdout(devnull) do
                Base.invokelatest(main, [config, "--only", "identical", "--json", j2])
            end) == 0
            @test JSON3.read(read(j2, String)).pass === true
            @test quiet(() -> redirect_stdout(devnull) do
                Base.invokelatest(main, [config, "--markdown", joinpath(dir, "out", "all.md")])
            end) == 1
            # Without output files the markdown report goes to stdout.
            captured = joinpath(dir, "stdout.txt")
            open(captured, "w") do io
                quiet(() -> redirect_stdout(io) do
                    Base.invokelatest(main, [config, "--only", "short_span"])
                end)
            end
            @test occursin("## short_span (R): PASS", read(captured, String))
        end

        @testset "example TOML of the harness" begin
            cfg = h_call(:load_parity_config, joinpath(HARNESS_DIR, "s0_bitwise_pairs.toml"))
            @test [c.name for c in cfg.comparisons] == ["hourly_to_daily_c90_2021", "refine_c180_20211231"]
            @test all(c -> c.level == "S0" && c.tolerances[1] == ("bitwise_fraction.all" => 1.0), cfg.comparisons)
            @test cfg.comparisons[1].nc == 90 && !cfg.comparisons[1].common_times
            @test cfg.comparisons[2].nc == 180 && cfg.comparisons[2].common_times
            @test !any(c -> c.common_variables_only, cfg.comparisons)
            @test all(c -> isabspath(c.reference) && isabspath(c.candidate), cfg.comparisons)
        end
    end
end
