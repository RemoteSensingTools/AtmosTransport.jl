#!/usr/bin/env julia
# ===========================================================================
# run_parity.jl: run the flux-port parity comparisons listed in a TOML file
#
# Runner of the parity harness of the Python -> Julia surface-flux port (see
# README.md in this directory). Every `[[comparison]]` of the TOML file is run
# by calling `compare_cs_flux_files` of `scripts/validation/compare_cs_flux_files.jl`
# in this process (a function call, not a subprocess), one comparison after the
# other; target meshes are built once per (nc, radius) or binary. The results go
# to one combined JSON report (every comparator report in full) and one markdown
# report. A comparison that throws (missing file, unequal units, wrong layout)
# is recorded with status `error`, and the remaining comparisons still run.
# Both reports name the code that produced them: the git commit, whether
# tracked files are modified, and the sha256 of this runner and of the
# comparator, taken when they are loaded (the scripts may be untracked or
# modified, which the commit alone does not show).
#
# TOML layout (unknown keys are errors; relative paths are taken from the TOML
# file's directory, `~` is expanded):
#
#   title = "..."                          # optional
#   description = "..."                    # optional
#
#   [[comparison]]
#   name = "hourly_to_daily_c90_2021"      # required, unique; letters, digits, _ . -
#   level = "S0"                           # required: S0, L1, T, L2, R or X (design §4.1)
#   description = "..."                    # optional
#   reference = "/path/reference.nc"       # required
#   candidate = "/path/candidate.nc"       # required
#   nc = 90                                # target mesh: exactly one of `nc` or `binary`
#   radius = 6.371e6                       # optional, with `nc` only (default EARTH_RADIUS)
#   binary = "/path/transport.bin"
#   variables = ["GPP_CO2_FLUX", "NAME_A:NAME_B", "land_fraction"]   # optional; default: every
#                                          # series of either file, one in only one file fails
#   reference_scale = { GPP_CO2_FLUX = "carbon_44_12_to_iupac" }     # number or preset
#   allow_unit_mismatch = false            # optional
#   mask = { var = "land_fraction", file = "/path/mask.nc", threshold = 0.05 }
#                                          # optional; `file` defaults to the reference
#   common_times = false                   # optional
#   common_variables_only = false          # optional, without `variables` only: compare the
#                                          # series of both files, list the others
#   block_slices = 64                      # optional
#   tolerances = { bitwise_fraction = 1.0, year_total = 1e-9, cell_p99.interior = 1e-6 }
#                                          # a nested table (cell_p99 = { interior = 1e-6 })
#                                          # needs a per-category metric and >= 1 category;
#                                          # values finite and >= 0, bitwise_fraction <= 1,
#                                          # ulp_max a whole number
#
# The keys map one to one onto the options of `compare_cs_flux_files.jl`
# (`--var`, `--reference-scale`, `--allow-unit-mismatch`, `--mask-var`,
# `--mask-file`, `--mask-threshold`, `--common-times`, `--common-variables-only`,
# `--block-slices`, `--tol`); see its header for every metric and rule.
#
# Usage:
#   julia --project=. scripts/validation/flux_port_parity/run_parity.jl <comparisons.toml> \
#       [--only NAME]... [--json report.json] [--markdown report.md]
#
# Without --json and --markdown the markdown report goes to stdout. The exit
# status is 0 when every selected comparison passes and 1 otherwise.
# ===========================================================================

using Dates
using Printf
using SHA: sha256
using TOML

const COMPARATOR_FILE = normpath(joinpath(@__DIR__, "..", "compare_cs_flux_files.jl"))

module FluxFileComparator
include(joinpath(@__DIR__, "..", "compare_cs_flux_files.jl"))
end

_script_info(path) = (; path = normpath(abspath(path)), sha256 = bytes2hex(open(sha256, path)))

# Versions of the code that writes the reports, taken at load time.
const SCRIPT_VERSIONS = (; runner = _script_info(@__FILE__), comparator = _script_info(COMPARATOR_FILE))

const PARITY_LEVELS = ("S0", "L1", "T", "L2", "R", "X")
const TOP_LEVEL_KEYS = ("title", "description", "comparison")
const COMPARISON_KEYS = ("name", "level", "description", "reference", "candidate", "nc", "radius",
                         "binary", "variables", "reference_scale", "allow_unit_mismatch", "mask",
                         "common_times", "common_variables_only", "block_slices", "tolerances")
const MASK_KEYS = ("var", "file", "threshold")
const NAME_PATTERN = r"^[A-Za-z0-9_.-]+$"

const PARITY_USAGE = """
Usage: julia --project=. scripts/validation/flux_port_parity/run_parity.jl <comparisons.toml>
           [--only NAME]... [--json <report.json>] [--markdown <report.md>]

Runs every [[comparison]] of the TOML file (or the ones named with --only) with
compare_cs_flux_files and writes one combined JSON and markdown report. Exit
status: 0 when every comparison passes, 1 otherwise.
"""

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

function _check_keys(table::AbstractDict, allowed, where)
    unknown = sort!([k for k in keys(table) if !(k in allowed)])
    isempty(unknown) || throw(ArgumentError(
        "unknown key(s) $(join(unknown, ", ")) in $(where); allowed: $(join(allowed, ", "))"))
    return table
end

function _string(table, key, where; required::Bool = false)
    haskey(table, key) || (required ? throw(ArgumentError("`$(key)` is required in $(where)")) : return nothing)
    value = table[key]
    value isa AbstractString && !isempty(value) ||
        throw(ArgumentError("`$(key)` in $(where) must be a non-empty string, got $(repr(value))"))
    return String(value)
end

function _bool(table, key, where, default::Bool)
    value = get(table, key, default)
    value isa Bool || throw(ArgumentError("`$(key)` in $(where) must be true or false, got $(repr(value))"))
    return value
end

_path(p::AbstractString, base) = (q = expanduser(p); isabspath(q) ? normpath(q) : normpath(joinpath(base, q)))

# metric[.category] => tolerance; nested TOML tables (`cell_p99.interior = 1e-6`)
# are flattened, then keys and values are checked by the comparator's
# `validate_tolerances` (the same check as its command line and API).
function _tolerances(raw, where)
    raw isa AbstractDict || throw(ArgumentError("`tolerances` in $(where) must be a table"))
    flat = Pair{String, Any}[]
    for (key, value) in raw
        if value isa AbstractDict
            # A nested table `metric = { category = value, ... }`: the parent must be a
            # per-category metric and the table must name at least one category.
            metric = Symbol(key)
            haskey(FluxFileComparator.TOLERANCE_METRICS, metric) || throw(ArgumentError(
                "unknown tolerance metric `$(key)` in $(where); known: " *
                "$(join(keys(FluxFileComparator.TOLERANCE_METRICS), ", "))"))
            FluxFileComparator.TOLERANCE_METRICS[metric][2] || throw(ArgumentError(
                "tolerance metric `$(key)` in $(where) takes no category table"))
            isempty(value) && throw(ArgumentError(
                "the category table of tolerance `$(key)` in $(where) is empty"))
            append!(flat, ["$(key).$(category)" => x for (category, x) in value])
        else
            push!(flat, String(key) => value)
        end
    end
    return sort!(FluxFileComparator.validate_tolerances(flat; context = " in $(where)"); by = first)
end

function _mask(raw, where, base)
    raw === nothing && return nothing
    raw isa AbstractDict || throw(ArgumentError("`mask` in $(where) must be a table {var, file, threshold}"))
    _check_keys(raw, MASK_KEYS, "the mask of $(where)")
    var = _string(raw, "var", "the mask of $(where)"; required = true)
    file = _string(raw, "file", "the mask of $(where)")
    threshold = get(raw, "threshold", 0.0)
    threshold isa Real && !(threshold isa Bool) && isfinite(threshold) || throw(ArgumentError(
        "the mask threshold in $(where) must be a finite number, got $(repr(threshold))"))
    return (; var, file = file === nothing ? nothing : _path(file, base), threshold = Float64(threshold))
end

function _parse_comparison(entry, k, base, path)
    entry isa AbstractDict || throw(ArgumentError("comparison $(k) of $(path) is not a table"))
    where = "comparison $(k) of $(path)"
    _check_keys(entry, COMPARISON_KEYS, where)
    name = _string(entry, "name", where; required = true)
    occursin(NAME_PATTERN, name) || throw(ArgumentError(
        "comparison name `$(name)` in $(path) may only hold letters, digits, `_`, `.` and `-`"))
    where = "comparison `$(name)` of $(path)"
    level = _string(entry, "level", where; required = true)
    level in PARITY_LEVELS || throw(ArgumentError(
        "level `$(level)` of $(where) is not one of $(join(PARITY_LEVELS, ", "))"))
    reference = _path(_string(entry, "reference", where; required = true), base)
    candidate = _path(_string(entry, "candidate", where; required = true), base)

    nc, radius = get(entry, "nc", nothing), get(entry, "radius", nothing)
    binary = _string(entry, "binary", where)
    (nc === nothing) == (binary === nothing) &&
        throw(ArgumentError("give the target mesh of $(where) by exactly one of `nc` or `binary`"))
    nc === nothing || (nc isa Integer && !(nc isa Bool) && nc >= 1) ||
        throw(ArgumentError("`nc` of $(where) must be a positive integer, got $(repr(nc))"))
    binary !== nothing && radius !== nothing &&
        throw(ArgumentError("`radius` of $(where) goes with `nc`; a binary carries its own radius"))
    radius === nothing || (radius isa Real && !(radius isa Bool) && isfinite(radius) && radius > 0) ||
        throw(ArgumentError("`radius` of $(where) must be a positive length in m, got $(repr(radius))"))

    raw_variables = get(entry, "variables", String[])
    raw_variables isa AbstractVector && all(v -> v isa AbstractString, raw_variables) ||
        throw(ArgumentError("`variables` of $(where) must be a list of strings"))
    variables = [FluxFileComparator._variable_pair(String(v)) for v in raw_variables]
    common_variables_only = _bool(entry, "common_variables_only", where, false)
    common_variables_only && !isempty(variables) && throw(ArgumentError(
        "`common_variables_only` of $(where) goes with the default variable selection; " *
        "with `variables` the selection is explicit"))

    raw_scales = get(entry, "reference_scale", Dict{String, Any}())
    raw_scales isa AbstractDict ||
        throw(ArgumentError("`reference_scale` of $(where) must be a table NAME = factor or preset"))
    reference_scale = sort!([String(n) => FluxFileComparator.resolve_reference_scale(v) for (n, v) in raw_scales];
                            by = first)

    mask = _mask(get(entry, "mask", nothing), where, base)
    tolerances = _tolerances(get(entry, "tolerances", Dict{String, Any}()), where)
    mask === nothing && any(t -> endswith(t.first, ".coastal"), tolerances) &&
        throw(ArgumentError("coastal tolerances of $(where) need a `mask`"))
    block_slices = get(entry, "block_slices", nothing)
    block_slices === nothing || (block_slices isa Integer && !(block_slices isa Bool) && block_slices >= 1) ||
        throw(ArgumentError("`block_slices` of $(where) must be a positive integer, got $(repr(block_slices))"))

    return (; name, level, description = _string(entry, "description", where), reference, candidate,
              nc, radius = radius === nothing ? nothing : Float64(radius),
              binary = binary === nothing ? nothing : _path(binary, base),
              variables, reference_scale, allow_unit_mismatch = _bool(entry, "allow_unit_mismatch", where, false),
              mask, common_times = _bool(entry, "common_times", where, false), common_variables_only,
              block_slices, tolerances)
end

"""
    load_parity_config(path) -> (; path, title, description, comparisons)

Read and check a parity TOML file (layout in the file header). Every error in
the file is raised here, before any comparison runs; only missing data files
are found when the comparison runs.
"""
function load_parity_config(path::AbstractString)
    isfile(path) || throw(ArgumentError("parity config not found: $(path)"))
    raw = TOML.parsefile(path)
    _check_keys(raw, TOP_LEVEL_KEYS, "the top level of $(path)")
    entries = get(raw, "comparison", nothing)
    entries isa AbstractVector && !isempty(entries) ||
        throw(ArgumentError("$(path) has no [[comparison]] entries"))
    base = dirname(abspath(path))
    comparisons = [_parse_comparison(e, k, base, path) for (k, e) in enumerate(entries)]
    names = [c.name for c in comparisons]
    repeated = unique(n for n in names if count(==(n), names) > 1)
    isempty(repeated) || throw(ArgumentError("comparison name(s) $(join(repeated, ", ")) repeated in $(path)"))
    return (; path = abspath(path), title = _string(raw, "title", path),
              description = _string(raw, "description", path), comparisons)
end

# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

# What was asked, for the report (independent of whether the comparison ran).
_spec_summary(c) = (; reference = c.reference, candidate = c.candidate,
                      mesh = (; nc = c.nc, radius_m = c.radius, binary = c.binary),
                      variables = [p.first == p.second ? p.first : "$(p.first):$(p.second)" for p in c.variables],
                      reference_scale = [(; variable = n, s.factor, s.preset) for (n, s) in c.reference_scale],
                      allow_unit_mismatch = c.allow_unit_mismatch, mask = c.mask,
                      common_times = c.common_times, common_variables_only = c.common_variables_only,
                      block_slices = c.block_slices,
                      tolerances = [(; metric = k, tolerance = v) for (k, v) in c.tolerances])

"""
    run_comparison(comparison, mesh_cache = Dict()) -> NamedTuple

Run one parsed comparison with `compare_cs_flux_files`. `status` is `pass`,
`fail` or `error` (the comparison threw; `error` holds the message and `report`
is `nothing`).
"""
function run_comparison(c, mesh_cache::AbstractDict = Dict{Any, Any}())
    started = time()
    report = nothing
    message = nothing
    try
        isfile(c.reference) || throw(ArgumentError("reference file not found: $(c.reference)"))
        isfile(c.candidate) || throw(ArgumentError("candidate file not found: $(c.candidate)"))
        c.binary === nothing || isfile(c.binary) || throw(ArgumentError("transport binary not found: $(c.binary)"))
        key = c.binary === nothing ? (:nc, c.nc, c.radius) : (:binary, c.binary)
        mesh, mesh_info = get!(mesh_cache, key) do
            FluxFileComparator.load_target_mesh(; binary = c.binary, Nc = c.nc, radius = c.radius)
        end
        mask = c.mask
        report = FluxFileComparator.compare_cs_flux_files(
            c.reference, c.candidate, mesh; variables = c.variables, reference_scale = c.reference_scale,
            allow_unit_mismatch = c.allow_unit_mismatch,
            mask_var = mask === nothing ? nothing : mask.var,
            mask_file = mask === nothing ? nothing : mask.file,
            mask_threshold = mask === nothing ? 0.0 : mask.threshold,
            common_times = c.common_times, common_variables_only = c.common_variables_only,
            tolerances = c.tolerances, block_slices = c.block_slices, mesh_info)
    catch err
        err isa InterruptException && rethrow()
        message = sprint(showerror, err)
        @error "comparison $(c.name) raised an error" exception = (err, catch_backtrace())
    end
    status = message !== nothing ? "error" : report.pass ? "pass" : "fail"
    return (; name = c.name, level = c.level, description = c.description, status,
              pass = status == "pass", error = message, wall_seconds = time() - started,
              spec = _spec_summary(c), report)
end

function _git_state(dir)
    try
        commit = readchomp(`git -C $(dir) rev-parse HEAD`)
        dirty = !isempty(readchomp(`git -C $(dir) status --porcelain --untracked-files=no`))
        return (; commit, tracked_changes = dirty)
    catch
        return nothing
    end
end

"""
    run_parity(config_path; only = String[], json = nothing, markdown = nothing) -> NamedTuple

Run the comparisons of a parity TOML file (all, or the ones named in `only`),
write the combined JSON report to `json` and the markdown report to `markdown`
when given, and return the combined report. Its `pass` is `true` when every
selected comparison passes.
"""
function run_parity(config_path::AbstractString; only = String[], json = nothing, markdown = nothing)
    config = load_parity_config(config_path)
    names = [c.name for c in config.comparisons]
    unknown = setdiff(only, names)
    isempty(unknown) || throw(ArgumentError(
        "--only names not in $(config.path): $(join(unknown, ", ")); known: $(join(names, ", "))"))
    selected = isempty(only) ? config.comparisons : filter(c -> c.name in only, config.comparisons)
    mesh_cache = Dict{Any, Any}()
    results = Any[]
    for (k, c) in enumerate(selected)
        @info "Parity comparison $(k)/$(length(selected)): $(c.name) ($(c.level))"
        push!(results, run_comparison(c, mesh_cache))
    end
    n = length(results)
    n_pass = count(r -> r.status == "pass", results)
    n_error = count(r -> r.status == "error", results)
    combined = (; tool = "scripts/validation/flux_port_parity/run_parity.jl",
                  created = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS"),
                  config = config.path, title = config.title, description = config.description,
                  host = gethostname(), julia = string(VERSION),
                  code = _git_state(joinpath(@__DIR__, "..", "..", "..")),
                  scripts = SCRIPT_VERSIONS,
                  pass = n > 0 && n_pass == n, n_comparisons = n, n_pass, n_fail = n - n_pass - n_error,
                  n_error, comparisons = results)
    json === nothing || FluxFileComparator.write_report_json(json, combined)
    markdown === nothing ||
        FluxFileComparator._write_atomically(io -> print(io, markdown_report(combined)), markdown)
    return combined
end

# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

_md(x) = replace(string(x), "|" => "\\|", "\n" => " ")
_fmt(x) = FluxFileComparator._fmt(x)
_frac(x) = x === nothing ? "n/a" : @sprintf("%.6f", x)

function _variable_row(v)
    a = v.cells.all
    scale = v.reference_scale === nothing ? "-" :
            @sprintf("%.17g", v.reference_scale.factor) *
            (v.reference_scale.preset === nothing ? "" : " ($(v.reference_scale.preset))")
    name = v.reference_variable == v.candidate_variable ? v.reference_variable :
           "$(v.reference_variable) / $(v.candidate_variable)"
    month = v.month_totals === nothing ? "-" : _fmt(v.month_totals.max_abs_rel_diff)
    year = v.year_totals === nothing ? "-" : _fmt(v.year_totals.max_abs_rel_diff)
    return "| " * join(_md.((name, v.kind, scale, _frac(a.bitwise_equal_fraction), _frac(a.within_1ulp_fraction),
                             _fmt(a.ulp_max), _fmt(a.p99), _fmt(a.max), month, year,
                             v.pass ? "PASS" : "FAIL")), " | ") * " |"
end

function _comparison_section(io, r)
    println(io, "## ", r.name, " (", r.level, "): ", uppercase(r.status))
    println(io)
    r.description === nothing || (println(io, r.description); println(io))
    println(io, "- Reference: `", r.spec.reference, "`")
    println(io, "- Candidate: `", r.spec.candidate, "`")
    println(io, @sprintf("- Wall time: %.1f s", r.wall_seconds))
    if r.report === nothing
        println(io)
        println(io, "Error:")
        println(io)
        println(io, "```")
        println(io, r.error)
        println(io, "```")
        println(io)
        return
    end
    rep = r.report
    m = rep.mesh
    println(io, "- Mesh: C", m.Nc, " ", m.convention, ", ", m.source,
            m.path === nothing ? "" : " `$(m.path)`", ", radius ", m.radius_m, " m")
    t = rep.time
    if t === nothing
        println(io, "- Times: none read (only variables without a time dimension)")
    else
        println(io, "- Times: ", t.n_compared, " compared, ",
                t.identical ? "identical" : "$(t.n_only_in_reference) only in the reference, $(t.n_only_in_candidate) only in the candidate" *
                                            (t.n_end_differs == 0 ? "" : ", $(t.n_end_differs) common stamps with other slice ends"),
                t.n_compared > 0 ? " ($(t.first) .. $(t.last))" : "", t.pass ? "" : " **FAIL**")
    end
    sel = rep.variable_selection
    if !isempty(sel.series_only_in_reference) || !isempty(sel.series_only_in_candidate)
        names(v) = isempty(v) ? "none" : join(("`$(n)`" for n in v), ", ")
        println(io, "- Series in one file only: reference ", names(sel.series_only_in_reference),
                "; candidate ", names(sel.series_only_in_candidate),
                sel.pass ? " (not compared: common_variables_only)" : " **FAIL**")
    end
    c = rep.categories
    println(io, "- Cells: interior ", c.n_cells.interior, ", coastal ", c.n_cells.coastal, ", polar ", c.n_cells.polar,
            c.mask_var === nothing ? " (no mask)" : " (mask `$(c.mask_var)` > $(c.mask_threshold) from `$(c.mask_file)`)")
    rep.reference_scale_rule === nothing || println(io, "- Reference scale: ", rep.reference_scale_rule)
    println(io)
    if isempty(rep.variables)
        println(io, "No variable was compared.")
        println(io)
        return
    end
    println(io, "| Variable | Kind | Reference scale | Bitwise | ≤ 1 ulp | Max ulp | Cell p99 | Cell max | Month total | Year total | Result |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|---|")
    foreach(v -> println(io, _variable_row(v)), rep.variables)
    println(io)
    println(io, "| Variable | Check | Value | Bound | Status |")
    println(io, "|---|---|---|---|---|")
    for v in rep.variables, chk in v.checks
        status = chk.status == "n/a" ? "n/a: $(chk.note)" :
                 chk.note === nothing ? chk.status : "$(chk.status): $(chk.note)"
        println(io, "| ", join(_md.((v.reference_variable, chk.metric, _fmt(chk.value),
                                     "$(chk.bound) $(_fmt(chk.tolerance))", status)), " | "), " |")
    end
    println(io)
end

"""
    markdown_report(combined) -> String

The markdown report of a `run_parity` result: an overview table, then one
section per comparison with its variables and checks.
"""
function markdown_report(combined)
    io = IOBuffer()
    println(io, "# Flux-port parity report")
    println(io)
    combined.title === nothing || (println(io, combined.title); println(io))
    combined.description === nothing || (println(io, combined.description); println(io))
    println(io, "- Config: `", combined.config, "`")
    println(io, "- Created: ", combined.created, " UTC on ", combined.host, ", Julia ", combined.julia)
    code = combined.code
    code === nothing ||
        println(io, "- Code: ", code.commit, code.tracked_changes ? " with uncommitted changes to tracked files" : "")
    s = combined.scripts
    println(io, "- Scripts: `", basename(s.runner.path), "` sha256 ", s.runner.sha256, ", `",
            basename(s.comparator.path), "` sha256 ", s.comparator.sha256)
    println(io, "- Result: **", combined.pass ? "PASS" : "FAIL", "**, ", combined.n_pass, " of ",
            combined.n_comparisons, " comparisons pass (", combined.n_fail, " fail, ", combined.n_error, " error)")
    println(io)
    println(io, "| Comparison | Level | Result | Variables | Compared slices | Wall time (s) |")
    println(io, "|---|---|---|---|---|---|")
    for r in combined.comparisons
        nvar = r.report === nothing ? "-" : string(length(r.report.variables))
        nslices = r.report === nothing || r.report.time === nothing ? "-" : string(r.report.time.n_compared)
        println(io, "| ", join(_md.((r.name, r.level, uppercase(r.status), nvar, nslices,
                                     @sprintf("%.1f", r.wall_seconds))), " | "), " |")
    end
    println(io)
    foreach(r -> _comparison_section(io, r), combined.comparisons)
    return String(take!(io))
end

# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

function _parse_parity_args(argv)
    config = json = markdown = nothing
    only = String[]
    i = 1
    function value()
        i + 1 <= length(argv) || throw(ArgumentError("missing value for $(argv[i])\n$(PARITY_USAGE)"))
        return argv[i + 1]
    end
    while i <= length(argv)
        arg = argv[i]
        if arg == "--only"
            push!(only, value()); i += 2
        elseif arg == "--json"
            json = expanduser(value()); i += 2
        elseif arg == "--markdown"
            markdown = expanduser(value()); i += 2
        elseif arg in ("-h", "--help")
            println(PARITY_USAGE); exit(0)
        elseif startswith(arg, "--")
            throw(ArgumentError("unknown argument `$(arg)`\n$(PARITY_USAGE)"))
        else
            config === nothing || throw(ArgumentError("one comparisons TOML file, got `$(config)` and `$(arg)`"))
            config = expanduser(arg); i += 1
        end
    end
    config === nothing && throw(ArgumentError("the comparisons TOML file is required\n$(PARITY_USAGE)"))
    return (; config, only, json, markdown)
end

function main(argv = ARGS)
    opts = _parse_parity_args(argv)
    combined = run_parity(opts.config; only = opts.only, json = opts.json, markdown = opts.markdown)
    if opts.json === nothing && opts.markdown === nothing
        print(stdout, markdown_report(combined))
    else
        for r in combined.comparisons
            println(stdout, rpad(r.name, 40), " ", rpad(r.level, 3), " ", uppercase(r.status),
                    r.error === nothing ? "" : "  ($(first(r.error, 200)))")
        end
        println(stdout, "run_parity: ", combined.pass ? "PASS" : "FAIL", " (", combined.n_pass, "/",
                combined.n_comparisons, ")")
    end
    return combined.pass ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
